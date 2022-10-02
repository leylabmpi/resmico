import _pickle as pickle
import os
import sys
import csv
import gzip
from pathlib import Path
import logging
from toolz import itertoolz
from functools import partial
from collections import defaultdict
import multiprocessing as mp
import tables
import pathos
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv1D, ReLU, BatchNormalization, Add, Cropping1D, AveragePooling1D
from tensorflow.keras.layers import Masking, LayerNormalization, MultiHeadAttention, Dropout

from resmico import contig_reader
from resmico import models_fl as Models

def class_recall_0(y_true, y_pred):
    label = 0
    class_id_preds = K.cast(K.greater(y_pred, 0.5), 'int32')
    y_true = K.cast(y_true, 'int32')
    accuracy_mask = K.cast(K.equal(y_true, label), 'int32')
    class_acc_tensor = K.cast(K.equal(y_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc


def class_recall_1(y_true, y_pred):
    label = 1
    class_id_preds = K.cast(K.greater(y_pred, 0.5), 'int32')
    y_true = K.cast(y_true, 'int32')
    accuracy_mask = K.cast(K.equal(y_true, label), 'int32')
    class_acc_tensor = K.cast(K.equal(y_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc


'''
Constructs the essential building unit of Resmico: the residual block. 
'''
def relu_bn(inputs):
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def bn_relu(inputs):
    bn = BatchNormalization()(inputs)
    relu = ReLU()(bn)
    return relu

# ### 1st version with deterministic predictions
def old_residual_block(x, downsample: bool, filters, kernel_size):
    y = Conv1D(kernel_size=kernel_size,
               strides=(1 if not downsample else 2),
               filters=filters,
               padding='valid' if downsample else 'same')(x)
    y = relu_bn(y)
    y = Conv1D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding='valid')(y)

    if downsample:
        x = Conv1D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding='valid')(x)
        # the additional cropping is needed in order to match the size of the y=Conv1D() output, since here we
        # user kernel_size=1
        x = Cropping1D((0,kernel_size//2))(x)
    x = Cropping1D((0, kernel_size-1))(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

'''
Residual block with all valid pading
'''
def residual_block(x, downsample: bool, filters, kernel_size):
    y = Conv1D(kernel_size=kernel_size,
               strides=(1 if not downsample else 2),
               filters=filters,
               padding='valid')(x)
    y = relu_bn(y)
    y = Conv1D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding='valid')(y)

    if downsample:
        x = Conv1D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding='valid')(x)
        # the cropping is needed to make up for the size difference caused by the convolutions on y
        x = Cropping1D((0,kernel_size-1 + kernel_size//2))(x)

    else:
        x = Cropping1D((0, 2*kernel_size-2))(x)

    out = Add()([x, y])
    out = relu_bn(out)
    return out


def dilated_residual_block(x, dilate: bool, filters, kernel_size):
    ### if dilate==True: apply dilation rate 3

    if filters>64:
        #bottleneck layer
        y = Conv1D(kernel_size=1,
               filters=64,
               padding='valid')(x)
        y = bn_relu(y)
        y = Conv1D(kernel_size=kernel_size,
               dilation_rate=(1 if not dilate else 3),
               filters=filters,
               padding='valid')(y)
        y = bn_relu(y)
        y = Conv1D(kernel_size=1,
               filters=filters,
               padding='valid')(y)
        y = BatchNormalization()(y)
        
        if dilate: 
            x = Conv1D(kernel_size=1,
                       filters=filters,
                       padding='valid')(x)
            x = Cropping1D((0,((kernel_size-1)*3)))(x)
        else:
            x = Cropping1D((0, (kernel_size-1)))(x)
    
    else:
        #residual block with two weighted layers
        y = Conv1D(kernel_size=kernel_size,
                   dilation_rate=(1 if not dilate else 3),
                   filters=filters,
                   padding='valid')(x)
        y = bn_relu(y)
        y = Conv1D(kernel_size=kernel_size,
                   dilation_rate=1,
                   filters=filters,
                   padding='valid')(y)
        y = BatchNormalization()(y)
        if dilate: 
            #also number of filters changed, so conv layer needed
            x = Conv1D(kernel_size=1,
                       filters=filters,
                       padding='valid')(x)
            x = Cropping1D((0,(kernel_size-1)*3 + (kernel_size-1)))(x)
        else:
            x = Cropping1D((0, 2*kernel_size-2))(x)
    out = Add()([x, y])
    out = ReLU()(out)
    return out


def transformer_encoder(inputs, head_size, num_heads, dropout=0):
    # Normalization and Attention
    x = LayerNormalization()(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization()(res)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
#     x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def n_moves_window(cont_len, window, step):
    if cont_len < window:
        return 0
    else:
        return np.ceil((cont_len - window) / step)

# look at predictions
def add_stats(df, column_name='chunk_scores'):
    chunk_scores = np.array(df[column_name])
    df['min'] = [np.min(list_scores) for list_scores in chunk_scores]
    df['mean'] = [np.mean(list_scores) for list_scores in chunk_scores]
    df['std'] = [np.std(list_scores) for list_scores in chunk_scores]
    df['max'] = [np.max(list_scores) for list_scores in chunk_scores]

    percent_names = ['p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90']
    pers = np.arange(10, 100, 10)
    result_shape = (df.shape[0], len(pers))
    compute_pers = [np.percentile(list_scores, pers) for list_scores in chunk_scores]
    df[percent_names] = np.array(compute_pers).reshape(result_shape)

    return df


def aggregate_chunks(batches_list, all_lens, all_labels, all_names, all_preds, window, step):
    if all_labels != []:
        dic_predictions = {'cont_name': [], 'length': [], 'label': [], 'chunk_scores': []}
    else:
        dic_predictions = {'cont_index': [], 'length': [], 'chunk_scores': []}

    start_pos = 0

    for cont_inds in batches_list:
        for cont_ind in cont_inds:
            cont_len = all_lens[cont_ind]
            dic_predictions['length'].append(cont_len)

            end_pos = start_pos + int(1 + n_moves_window(all_lens[cont_ind], window, step))
            cont_preds = all_preds[start_pos:end_pos].reshape(-1)

            # contigs of length of length 5k-6k treated the same as 5k
            if (cont_len >= 5000) & (cont_len <= 6000):
                cont_preds = cont_preds[0]
                # the second window is not informative and can be harmful, because has the same weight as the first

            dic_predictions['chunk_scores'].append(cont_preds)
            start_pos = end_pos
            if all_labels != []:
                dic_predictions['cont_name'].append(all_names[cont_ind])
                dic_predictions['label'].append(all_labels[cont_ind])

            else:
                dic_predictions['cont_index'].append(cont_ind)

    return dic_predictions


def update_progress(current: int, total: int, prefix: str, tail: str):
    """
    Displays or updates a console progress bar.
    Accepts a float between 0 and 1. Any int will be converted to a float.
    A value under 0 represents a 'halt'. A value at 1 or bigger represents 100%.
    """
    barLength = 100
    status = tail
    progress = current / total
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    current_str = str(current).rjust(len(str(total)))
    text = f'\r{prefix}[{"#" * block + "-" * (barLength - block)}] {current_str}/{total} {status}'
    sys.stdout.write(text)
    sys.stdout.flush()
    
def sma(arr, window_size=2):
    if len(arr)<=window_size:
        return [np.mean(arr)]
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []
    while i < len(arr) - window_size + 1:
        window_average = np.sum(arr[
          i:i+window_size]) / window_size
        moving_averages.append(window_average)
        i += 1
    return moving_averages

def prob_score_aggr(arr):
    arr = np.array(arr)
    arr = 1-arr
    return 1 - np.prod(arr)

def calib_prob_score_aggr(scores, model):  
    return prob_score_aggr(model.predict(scores))

def test_calibprob_vc_max(model: tf.keras.Model, num_gpus: int, args):
    start = time.time()

    is_fixed_length = model.layers[0].input_shape[0][1] is not None  
    convoluted_size = Models.construct_convolution_lambda(model)
#     else:  # when not padding, the convoluted size is unused
#         convoluted_size = lambda len, pad: 0

    logging.info('Loading contig data...')
    reader = contig_reader.ContigReader(args.feature_files_path, args.features, args.n_procs,
                                        args.no_cython, args.stats_file, args.min_contig_len,
                                        min_avg_coverage=args.min_avg_coverage)

    if args.val_ind_f:
        eval_idx = list(pd.read_csv(args.val_ind_f)['val_ind'])
        logging.info(f'Using {len(eval_idx)} indices in {args.val_ind_f} for prediction')
    else:
        logging.info('Should be done on the validation set')

    #split into validation and calibration
    calib_idx, eval_idx = train_test_split(eval_idx, test_size=0.5, random_state=args.seed)
    #for calibration part filter < 20k length
    short_calib = []
    for ind in calib_idx:
        if reader.contigs[ind].length < 20000:
             short_calib.append(ind)
    calib_idx = short_calib

    
    #create predictions
    calib_data = Models.BinaryDatasetEval(reader, calib_idx, args.features, args.max_len, max(250, args.max_len), 
                                        int(args.gpu_eval_mem_gb * 1e9 * 0.8), cache_results=False,
                                        show_progress=True, convoluted_size=convoluted_size,
                                        pad_to_max_len=is_fixed_length, batch_size=args.batch_size)
    calib_data_y = np.array([0 if reader.contigs[idx].misassembly == 0 else 1 for idx in calib_data.indices])
    data_iter = lambda: (s for s in calib_data)
    calib_data_tf = tf.data.Dataset.from_generator(
        data_iter,
        output_signature=(
            (tf.TensorSpec(shape=(None, None, len(calib_data.expanded_feature_names)), dtype=tf.float32),
             tf.TensorSpec(shape=(None, None), dtype=tf.bool)),
            tf.TensorSpec(shape=(None), dtype=tf.bool)
        ))

    calib_data_tf = calib_data_tf.prefetch(4 * num_gpus)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    calib_data_tf = calib_data_tf.with_options(options)  # avoids Tensorflow ugly console barf
    calib_data_flat_y = model.predict(x=calib_data_tf,
                                     workers=args.n_procs,
                                     use_multiprocessing=True,
                                     max_queue_size=max(args.n_procs, 10),
                                     verbose=0)
    calib_data_predicted_score = calib_data.group(calib_data_flat_y, max) #it is anyway only one score
    auc_calib = average_precision_score(calib_data_y, calib_data_predicted_score)
    recall1_calib = recall_score(calib_data_y, calib_data_predicted_score > 0.5, pos_label=1)
    recall0_calib = recall_score(calib_data_y, calib_data_predicted_score > 0.5, pos_label=0)
    logging.info(f'Prediction scores for calib: aucPR: {auc_calib} - recall1: {recall1_calib} - recall0: {recall0_calib}')
    
    #fit isotonic regression
    iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(calib_data_predicted_score, calib_data_y)
    
    #for valid part predict with normal model
    predict_data = Models.BinaryDatasetEval(reader, eval_idx, args.features, args.max_len, max(250, args.max_len), 
                                            int(args.gpu_eval_mem_gb * 1e9 * 0.8), cache_results=False,
                                            show_progress=True, convoluted_size=convoluted_size,
                                            pad_to_max_len=is_fixed_length, batch_size=args.batch_size)

    eval_data_y = np.array([0 if reader.contigs[idx].misassembly == 0 else 1 for idx in predict_data.indices])
    # convert the slow Keras predict_data of type Sequence to a tf.data object
    data_iter = lambda: (s for s in predict_data)
    predict_data_tf = tf.data.Dataset.from_generator(
        data_iter,
        output_signature=(
            # first dimension is batch size, second is contig length, third is number of features
            (tf.TensorSpec(shape=(None, None, len(predict_data.expanded_feature_names)), dtype=tf.float32),
             # first dimension is batch size, second is contig length (no third dimension,
             # as all features are masked the same way)
             tf.TensorSpec(shape=(None, None), dtype=tf.bool)),
            tf.TensorSpec(shape=(None), dtype=tf.bool)
        ))
    predict_data_tf = predict_data_tf.prefetch(4 * num_gpus)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    predict_data_tf = predict_data_tf.with_options(options)  # avoids Tensorflow ugly console barf
    eval_data_flat_y = model.predict(x=predict_data_tf,
                                     workers=args.n_procs,
                                     use_multiprocessing=True,
                                     max_queue_size=max(args.n_procs, 10),
                                     verbose=0)
    #aggregate with max
    eval_data_predicted_max = predict_data.group(eval_data_flat_y, max)
    auc_val = average_precision_score(eval_data_y, eval_data_predicted_max)
    recall1_val = recall_score(eval_data_y, eval_data_predicted_max > 0.5, pos_label=1)
    recall0_val = recall_score(eval_data_y, eval_data_predicted_max > 0.5, pos_label=0)
    logging.info(f'Eval data with max: aucPR: {auc_val} - recall1: {recall1_val} - recall0: {recall0_val}')
    
    #aggregate with isotonic + prob
    eval_data_calib_prob = predict_data.group(eval_data_flat_y, 'calib_prob', model=iso_reg)
    auc_calib_prob = average_precision_score(eval_data_y, eval_data_calib_prob)
    recall1_calib_prob = recall_score(eval_data_y, eval_data_calib_prob > 0.5, pos_label=1)
    recall0_calib_prob = recall_score(eval_data_y, eval_data_calib_prob > 0.5, pos_label=0)
    logging.info(f'Eval data with calib_prob: aucPR: {auc_calib_prob} - recall1: {recall1_calib_prob} - recall0: {recall0_calib_prob}')
    
    duration = time.time() - start
    logging.info(f'Prediction done in {duration:.0f}s.')

    # saving output
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    out_file = os.path.join(args.save_path, args.save_name + '.csv')    
   

    with open(out_file, 'w') as outF:
        outF.write('cont_name,length,label,calib_prob,max\n')
        for idx in range(len(eval_idx)):
            contig = reader.contigs[predict_data.indices[idx]]
            outF.write(f'{os.path.join(os.path.dirname(contig.file), contig.name)},{contig.length},'
                       f'{contig.misassembly},{eval_data_calib_prob[idx]},'
                       f'{eval_data_predicted_max[idx]}\n')

    logging.info(f'Predictions saved to: {out_file}')