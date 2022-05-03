import logging
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from sklearn.metrics import recall_score, average_precision_score

from resmico import contig_reader
from resmico import models_fl as Models  # to use contigs of variable length
from resmico import utils


def predict_bin_data(model: tf.keras.Model, num_gpus: int, args):
    start = time.time()

    is_fixed_length = model.layers[0].input_shape[0][1] is not None

    if args.mask_padding:
        convoluted_size = Models.construct_convolution_lambda(model)
    else:  # when not padding, the convoluted size is unused
        convoluted_size = lambda len, pad: 0

    logging.info('Loading contig data...')
    reader = contig_reader.ContigReader(args.feature_files_path, args.features, args.n_procs, args.chunks,
                                        args.no_cython, args.stats_file, args.min_len,
                                        min_avg_coverage=args.min_avg_coverage)

    if args.val_ind_f:
        eval_idx = list(pd.read_csv(args.val_ind_f)['val_ind'])
        logging.info(f'Using {len(eval_idx)} indices in {args.val_ind_f} for prediction')
    elif args.embeddings:
        all_idx = np.arange(len(reader))
        logging.info(f'Using 10k samples for embeddings')
        np.random.shuffle(all_idx)
        eval_idx = all_idx[:10000]
    else:
        logging.info(f'Using all indices for prediction')
        eval_idx = np.arange(len(reader))

    predict_data = Models.BinaryDatasetEval(reader, eval_idx, args.features, args.max_len, max(250, args.max_len - 500),
                                            int(args.gpu_eval_mem_gb * 1e9 * 0.8), cache_results=False,
                                            show_progress=True, convoluted_size=convoluted_size,
                                            pad_to_max_len=is_fixed_length)

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
    eval_data_predicted_min = predict_data.group(eval_data_flat_y, min)
    eval_data_predicted_mean = predict_data.group(eval_data_flat_y, np.mean)
    eval_data_predicted_std = predict_data.group(eval_data_flat_y, np.std)
    eval_data_predicted_max = predict_data.group(eval_data_flat_y, max)
    eval_data_predicted_score = predict_data.group(eval_data_flat_y, max) #'sma'

    auc_val = average_precision_score(eval_data_y, eval_data_predicted_score)
    recall1_val = recall_score(eval_data_y, eval_data_predicted_score > 0.5, pos_label=1)
    recall0_val = recall_score(eval_data_y, eval_data_predicted_score > 0.5, pos_label=0)
    logging.info(f'Prediction scores: aucPR: {auc_val} - recall1: {recall1_val} - recall0: {recall0_val}')
    duration = time.time() - start
    logging.info(f'Prediction done in {duration:.0f}s.')
    
    if args.embeddings:
        middle_output = Model(inputs=model.input, outputs=model.layers[args.emb_ind].output)
        eval_data_emb = middle_output.predict(x=predict_data_tf,
                                                 workers=args.n_procs,
                                                 use_multiprocessing=True,
                                                 max_queue_size=max(args.n_procs, 10),
                                                 verbose=1)

        eval_data_emb = predict_data.group_emb(eval_data_emb, np.mean)
        
        out_file = open(args.save_path + '/' + args.save_name + '.csv', 'w')
        out_file.write('cont_name,length,label,embedding,score,min,mean,std,max\n')
        for idx in range(len(eval_idx)):
            contig = reader.contigs[predict_data.indices[idx]]
            out_file.write(f'{os.path.join(os.path.dirname(contig.file), contig.name)},{contig.length},'
                           f'{contig.misassembly},{eval_data_emb[idx]},{eval_data_predicted_score[idx]},'
                           f'{eval_data_predicted_min[idx]},{eval_data_predicted_mean[idx]},'
                           f'{eval_data_predicted_std[idx]},{eval_data_predicted_max[idx]}\n')        
    else:
        out_file = open(args.save_path + '/' + args.save_name + '.csv', 'w')
        out_file.write('cont_name,length,label,score,min,mean,std,max\n')
        for idx in range(len(eval_idx)):
            contig = reader.contigs[predict_data.indices[idx]]
            out_file.write(f'{os.path.join(os.path.dirname(contig.file), contig.name)},{contig.length},'
                           f'{contig.misassembly},{eval_data_predicted_score[idx]},'
                           f'{eval_data_predicted_min[idx]},{eval_data_predicted_mean[idx]},'
                           f'{eval_data_predicted_std[idx]},{eval_data_predicted_max[idx]}\n')

    logging.info(f'Predictions saved to: {out_file}')


def predict_with_method(model, args):
    if args.filter10:
        data_dict = utils.build_sample_index(Path(args.feature_files_path), args.n_procs, filter10=True,
                                             sdepth=args.sdepth, rich=args.rich)
    elif args.rep10:
        data_dict = utils.build_sample_index(Path(args.feature_files_path), args.n_procs, rep10=True,
                                             sdepth=args.sdepth, rich=args.rich)
    else:
        data_dict = utils.build_sample_index(Path(args.feature_files_path), args.n_procs,
                                             sdepth=args.sdepth, rich=args.rich, longdir=args.longdir)
    logging.info('Data dictionary created. Number of samples: {}'.format(len(data_dict)))
    all_contigs = list(data_dict.items())
    all_names = list(data_dict.keys())
    all_lens = utils.read_all_lens(data_dict)
    all_labels = utils.read_all_labels(data_dict)

    inds_sel = np.arange(len(all_lens))[np.array(all_lens) >= args.min_len]
    inds_toolong = np.arange(len(all_lens))[np.array(all_lens) > args.mem_lim]  # change if want 3k
    inds_sel = np.array([el for el in inds_sel if el not in inds_toolong])

    logging.info('{} -- too long contigs'.format(len(inds_toolong)))
    logging.info('{} -- selected contigs'.format(len(inds_sel)))

    if args.method_pred == 'random':
        np.random.seed(args.seed)
        long_test_data_dict = dict(np.array(all_contigs)[inds_sel])
        data_dict = long_test_data_dict
        max_len = args.window
        batch_size = 100

        num_runs = 5
        aupr_scores = []
        labels_val = utils.read_all_labels(data_dict)
        for i in range(num_runs):
            data_gen = Models.GeneratorBigD(data_dict, max_len, batch_size,  # only contigs to pred for should be given
                                            shuffle=False, nprocs=args.n_procs, rnd_seed=i)
            logging.info("Number of batches: {}".format(len(data_gen)))
            start = time.time()
            score_val = model.predict(data_gen, use_multiprocessing=args.n_procs > 1, workers=args.n_procs)
            duration = time.time() - start
            logging.info("measured time {}".format(duration))
            aupr = average_precision_score(labels_val, score_val)
            logging.info('AUC_PR: {:3f}, mean_label: {:3f}, mean_pred: {:3f}'.format(
                aupr, np.mean(labels_val), np.mean(score_val)))
            aupr_scores.append(aupr)

            logging.info('Mean AUC_PR: {}, std: {}'.format(np.mean(aupr_scores), np.std(aupr_scores)))
            # save last dictionary
            dic_predictions = {'cont_name': all_names[inds_sel], 'length': all_lens[inds_sel],
                               'label': labels_val, 'score': np.array(score_val).reshape(-1)}
            logging.info('Dictionary created')

            if num_runs > 1:
                df_preds = pd.DataFrame.from_dict(dic_predictions)
                df_name = args.save_path + '/' + 'r' + str(i) + args.save_name + '.csv'
                df_preds.to_csv(df_name, index=False)
                logging.info("csv table saved: {}".format(df_name))

    elif args.method_pred == 'fulllength':
        batches_list = utils.create_batch_inds(all_lens, inds_sel, args.mem_lim, fulllen=True)
        logging.info("Number of batches: {}".format(len(batches_list)))
        data_gen = Models.GeneratorFullLen(data_dict, batches_list, nprocs=args.n_procs)  # contigs filtered by indexing
        start = time.time()
        score_val = model.predict(data_gen, use_multiprocessing=args.n_procs > 1, workers=args.n_procs)
        duration = time.time() - start
        logging.info("measured time {}".format(duration))
        aupr = average_precision_score(all_labels[inds_sel], score_val)
        logging.info('AUC_PR: {:3f}'.format(aupr))
        dic_predictions = {'cont_name': all_names[inds_sel], 'length': all_lens[inds_sel],
                           'label': all_labels[inds_sel], 'score': np.array(score_val).reshape(-1)}
        logging.info('Dictionary created')

    elif args.v1:
        logging.info('predict with deepmased v1')
        window = 10000
        batches_list = utils.create_batch_inds(all_lens, inds_sel, args.mem_lim)
        logging.info("Number of batches: {}".format(len(batches_list)))
        data_gen = Models.Generator_v1(data_dict, batches_list, window=window, step=window, nprocs=args.n_procs)

        score_val = model.predict(data_gen, use_multiprocessing=args.n_procs > 1, workers=args.n_procs)

        dic_predictions = utils.aggregate_chunks(batches_list, all_lens, all_labels, all_names,
                                                 all_preds=score_val, window=window, step=window)
        logging.info('Dictionary created')

    elif args.method_pred == 'chunks':
        batches_list = utils.create_batch_inds(all_lens, inds_sel, args.mem_lim)
        logging.info("Number of batches: {}".format(len(batches_list)))
        data_gen = Models.GeneratorPredLong(data_dict, batches_list, window=args.window,
                                            step=args.window / 2., nprocs=args.n_procs)
        # contigs filtered by indexing
        start = time.time()
        score_val = model.predict(data_gen, use_multiprocessing=args.n_procs > 1, workers=args.n_procs)
        duration = time.time() - start
        logging.info("measured time {}".format(duration))
        dic_predictions = utils.aggregate_chunks(batches_list, all_lens, all_labels, all_names,
                                                 all_preds=score_val, window=args.window, step=args.window / 2)
        logging.info('Dictionary created')

    else:
        logging.info('Pred method is not supported')

    df_preds = pd.DataFrame.from_dict(dic_predictions)

    if args.method_pred == 'chunks' or args.v1:
        df_preds = utils.add_stats(df_preds)
    df_name = args.save_path + '/' + args.save_name + '.csv'
    df_preds.to_csv(df_name, index=False)
    logging.info("csv table saved: {}".format(df_name))

def verify_insert_size(args):
    
    logging.info('Loading contig data...')
    reader = contig_reader.ContigReader(args.feature_files_path, ['mean_insert_size_Match'], args.n_procs, args.chunks,
                                        args.no_cython, args.stats_file, args.min_len,
                                        min_avg_coverage=args.min_avg_coverage)

    contig_data = [reader.contigs[i] for i in range(0, len(reader), 10)] ##10% of the data
    
    features_data = reader.read_contigs(contig_data, return_raw=True)

    insert_size_data = []
    for cont in features_data:
        insert_size_data.extend(cont['mean_insert_size_Match'])

        
    low_train, high_train = 178, 372 #0.05 and 0.95 quantiles of the n9k-train dataset
     
    lowq = np.nanquantile(insert_size_data, 0.06)
    hiq = np.nanquantile(insert_size_data, 0.94)
    if lowq >= low_train and  hiq <= high_train:
        if np.nanquantile(insert_size_data, 0.05) >= low_train and np.nanquantile(insert_size_data, 0.95) <= high_train:
            logging.info('The insert size distribution lies inside the training one. It is safe to apply ResMiCo.')
        else:
            logging.info('The insert size distribution lies close to the border of the training one. ResMiCo can be applied.')
    else:
        logging.info('The insert size distribution is dissimilar to the training data. ResMiCo predictions are not reliable.')
        
def main(args):
    """Main interface
    """
    if args.verify_insert_size:
        verify_insert_size(args)
        exit()
    if args.v1:
        custom_obj = {'metr': utils.class_recall_0}
    else:
        # TOOD: remove GlobalMaskedMaxPooling1D, once annotation kicks in
        custom_obj = {'class_recall_0': utils.class_recall_0, 'class_recall_1': utils.class_recall_1,
                      'GlobalMaskedMaxPooling1D': Models.GlobalMaskedMaxPooling1D}

    if not os.path.exists(args.model):
        raise IOError(f'Cannot find {args.model}')
    strategy = tf.distribute.MirroredStrategy()
    logging.info(f'Number of devices: {strategy.num_replicas_in_sync}')
    logging.info(f'Loading model: {args.model}')
    with strategy.scope():
        model = load_model(args.model, custom_objects=custom_obj)
    logging.info('Model loaded')

    if args.binary_data:
        predict_bin_data(model, strategy.num_replicas_in_sync, args)
    else:
        predict_with_method(model, args)


if __name__ == '__main__':
    pass
