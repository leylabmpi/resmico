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

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv1D, ReLU, BatchNormalization, Add, Cropping1D, AveragePooling1D
from tensorflow.keras.layers import Masking, LayerNormalization, MultiHeadAttention, Dropout

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