# import
## batteries
import os
import sys
import logging
import _pickle as pickle
from pathlib import Path
## 3rd party
import time
import numpy as np
import pandas as pd
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
from tensorflow.keras.models import load_model
# import IPython
from sklearn.metrics import average_precision_score
## application
from DeepMAsED import Models_FL as Models  #to use contigs of variable length
from DeepMAsED import Utils


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

    inds_long = np.arange(len(all_lens))[np.array(all_lens) >= args.long_def]
    inds_toolong = np.arange(len(all_lens))[np.array(all_lens) > args.mem_lim]  # change if want 3k
    inds_long = np.array([el for el in inds_long if el not in inds_toolong])

    logging.info('{} -- too long contigs'.format(len(inds_toolong)))
    logging.info('{} -- selected contigs'.format(len(inds_long)))

    if args.method_pred == 'random':
        np.random.seed(args.seed)
        long_test_data_dict = dict(np.array(all_contigs)[inds_long])
        data_dict = long_test_data_dict
        max_len = args.window
        batch_size = 100

        aupr_scores = []
        labels_val = Utils.read_all_labels(data_dict)
        for i in range(1):  # set to 3 if interested in std estimation
            dataGen = Models.GeneratorBigD(data_dict, max_len, batch_size,  # only long contigs should be given?
                                           shuffle=False, nprocs=args.n_procs, rnd_seed=i)
            logging.info("Number of batches: {}".format(len(dataGen)))
            start = time.time()
            score_val = model.predict(dataGen, use_multiprocessing=args.n_procs > 1, workers=args.n_procs)
            duration = time.time() - start
            logging.info("measured time {}".format(duration))
            aupr = average_precision_score(labels_val, score_val)
            logging.info('AUC_PR: {:3f}, mean_label: {:3f}, mean_pred: {:3f}'.format(
                aupr, np.mean(labels_val), np.mean(score_val)))
            aupr_scores.append(aupr)
        logging.info('Mean AUC_PR: {}, std: {}'.format(np.mean(aupr_scores), np.std(aupr_scores)))
        # save last dictionary
        dic_predictions = {'cont_glob_index': inds_long, 'length': all_lens[inds_long],
                           'label': labels_val, 'score': np.array(score_val).reshape(-1)}
        logging.info('Dictionary created')


    elif args.method_pred == 'fulllength':
        batches_list = Utils.create_batch_inds(all_lens, inds_long, args.mem_lim, fulllen=True)
        logging.info("Number of batches: {}".format(len(batches_list)))
        dataGen = Models.GeneratorFullLen(data_dict, batches_list, nprocs=args.n_procs)
        start = time.time()
        score_val = model.predict(dataGen, use_multiprocessing=args.n_procs > 1, workers=args.n_procs)
        duration = time.time() - start
        logging.info("measured time {}".format(duration))
        aupr = average_precision_score(all_labels[inds_long], score_val)
        logging.info('AUC_PR: {:3f}'.format(aupr))
        dic_predictions = {'cont_glob_index': inds_long, 'length': all_lens[inds_long],
                           'label': all_labels[inds_long], 'score': np.array(score_val).reshape(-1)}
        logging.info('Dictionary created')


    elif args.method_pred == 'chunks':
        batches_list = Utils.create_batch_inds(all_lens, inds_long, args.mem_lim)
        logging.info("Number of batches: {}".format(len(batches_list)))
        dataGen = Models.GeneratorPredLong(data_dict, batches_list, window=args.window, nprocs=args.n_procs)
        start = time.time()
        score_val = model.predict(dataGen, use_multiprocessing=args.n_procs > 1, workers=args.n_procs)
        duration = time.time() - start
        logging.info("measured time {}".format(duration))
        dic_predictions = Utils.agregate_chunks(batches_list, all_lens, all_labels,
                                          all_preds=score_val, window=args.window)
        logging.info('Dictionary created')


    else:
        logging.info('Pred method is not supported')

    df_preds = pd.DataFrame.from_dict(dic_predictions)
    if args.method_pred == 'chunks':
        df_preds = Utils.add_stats(df_preds)
    df_name = args.save_path + '/' + args.save_name + '.csv'
    df_preds.to_csv(df_name, index=False)
    logging.info("csv table saved: {}".format(df_name))

def main(args):
    """Main interface
    """
    #load model
    custom_obj = {'class_recall_0':Utils.class_recall_0, 'class_recall_1': Utils.class_recall_1}
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

    if args.big_data:
        predict_with_method(model, args)


    else: #old version of code and data required
    # loading features
        logging.info('Loading synthetic features')
        x, y, i2n = Utils.load_features(args.feature_file_table,
                                        max_len = args.max_len,
                                        technology = args.technology,
                                        chunks=False)  #False to use contigs of variable length

        logging.info('Loaded {} contigs'.format(len(set(i2n.values()))))
        n2i = Utils.reverse_dict(i2n)
        x = [xi for xmeta in x for xi in xmeta]
        y = np.concatenate(y)
        logging.info('Running model generator...')
        dataGen = Models.Generator(x, y, args.max_len, batch_size=args.batch_size,  shuffle=False)

        logging.info('Computing predictions for {}...'.format(args.technology))
        scores = Utils.compute_predictions_y_known(y, n2i, model, dataGen, args.n_procs, x=x) #give x if chunks=False

        outfile = os.path.join(args.save_path, '_'.join([args.save_name, args.technology + '.pkl']))
        with open(outfile, 'wb') as spred:
            pickle.dump(scores, spred)
        logging.info('File written: {}'.format(outfile))

    
if __name__ == '__main__':
    pass
