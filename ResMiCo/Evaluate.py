import os
import logging
from pathlib import Path

# 3rd party
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import average_precision_score

from ResMiCo import Models_FL as Models  # to use contigs of variable length
from ResMiCo import Utils


def predict_with_method(model, args):
    if args.filter10:
        data_dict = Utils.build_sample_index(Path(args.feature_files_path), args.n_procs, filter10=True,
                                             sdepth=args.sdepth, rich=args.rich)
    elif args.rep10:
        data_dict = Utils.build_sample_index(Path(args.feature_files_path), args.n_procs, rep10=True,
                                             sdepth=args.sdepth, rich=args.rich)
    else:
        data_dict = Utils.build_sample_index(Path(args.feature_files_path), args.n_procs,
                                             sdepth=args.sdepth, rich=args.rich)
    logging.info('Data dictionary created. Number of samples: {}'.format(len(data_dict)))
    all_contigs = list(data_dict.items())
    all_lens = Utils.read_all_lens(data_dict)
    all_labels = Utils.read_all_labels(data_dict)

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
        labels_val = Utils.read_all_labels(data_dict)
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
            dic_predictions = {'cont_glob_index': inds_sel, 'length': all_lens[inds_sel],
                               'label': labels_val, 'score': np.array(score_val).reshape(-1)}
            logging.info('Dictionary created')

            if num_runs > 1:
                df_preds = pd.DataFrame.from_dict(dic_predictions)
                df_name = args.save_path + '/' + 'r'+str(i) + args.save_name + '.csv'
                df_preds.to_csv(df_name, index=False)
                logging.info("csv table saved: {}".format(df_name))

    elif args.method_pred == 'fulllength':
        batches_list = Utils.create_batch_inds(all_lens, inds_sel, args.mem_lim, fulllen=True)
        logging.info("Number of batches: {}".format(len(batches_list)))
        data_gen = Models.GeneratorFullLen(data_dict, batches_list, nprocs=args.n_procs) #contigs filtered by indexing
        start = time.time()
        score_val = model.predict(data_gen, use_multiprocessing=args.n_procs > 1, workers=args.n_procs)
        duration = time.time() - start
        logging.info("measured time {}".format(duration))
        aupr = average_precision_score(all_labels[inds_sel], score_val)
        logging.info('AUC_PR: {:3f}'.format(aupr))
        dic_predictions = {'cont_glob_index': inds_sel, 'length': all_lens[inds_sel],
                           'label': all_labels[inds_sel], 'score': np.array(score_val).reshape(-1)}
        logging.info('Dictionary created')

    elif args.v1:
        logging.info('predict with deepmased v1')
        window = 10000
        batches_list = Utils.create_batch_inds(all_lens, inds_sel, args.mem_lim)
        logging.info("Number of batches: {}".format(len(batches_list)))
        data_gen = Models.Generator_v1(data_dict, batches_list, window=window, step=window, nprocs=args.n_procs)

        score_val = model.predict(data_gen, use_multiprocessing=args.n_procs > 1, workers=args.n_procs)

        dic_predictions = Utils.aggregate_chunks(batches_list, all_lens, all_labels,
                                                 all_preds=score_val, window=window, step=window)
        logging.info('Dictionary created')

    elif args.method_pred == 'chunks':
        batches_list = Utils.create_batch_inds(all_lens, inds_sel, args.mem_lim)
        logging.info("Number of batches: {}".format(len(batches_list)))
        data_gen = Models.GeneratorPredLong(data_dict, batches_list, window=args.window, step=args.window/2.,
                                           nprocs=args.n_procs)
        # contigs filtered by indexing
        start = time.time()
        score_val = model.predict(data_gen, use_multiprocessing=args.n_procs > 1, workers=args.n_procs)
        duration = time.time() - start
        logging.info("measured time {}".format(duration))
        dic_predictions = Utils.aggregate_chunks(batches_list, all_lens, all_labels,
                                          all_preds=score_val, window=args.window, step=args.window/2)
        logging.info('Dictionary created')

    else:
        logging.info('Pred method is not supported')

    df_preds = pd.DataFrame.from_dict(dic_predictions)

    if args.method_pred == 'chunks' or args.v1:
        df_preds = Utils.add_stats(df_preds)
    df_name = args.save_path + '/' + args.save_name + '.csv'
    df_preds.to_csv(df_name, index=False)
    logging.info("csv table saved: {}".format(df_name))


def main(args):
    """Main interface
    """
    if args.v1:
        path = '/cluster/home/omineeva/global_projects/projects/projects2019-contig_quality/'
        h5_file = path + 'gitlab/deepmased/DeepMAsED/Model/deepmased_model.h5'
        custom_obj = {'metr' : Utils.class_recall_0}
    else:
        custom_obj = {'class_recall_0': Utils.class_recall_0, 'class_recall_1': Utils.class_recall_1}
        h5_file = os.path.join(args.model_path, args.model_name)

    if not os.path.exists(h5_file):
        msg = 'Cannot find {} file in {}'
        raise IOError(msg.format(args.model_name, args.model_path))
    strategy = tf.distribute.MirroredStrategy()
    logging.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    logging.info('Loading model: {}'.format(h5_file))
    with strategy.scope():
        model = load_model(h5_file, custom_objects=custom_obj)
    logging.info('Model loaded')

    predict_with_method(model, args)

    
if __name__ == '__main__':
    pass
