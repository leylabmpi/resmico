# import
## Batteries
import os
import sys
import logging
import time
import _pickle as pickle
from pathlib import Path
## 3rd party
import numpy as np
import math
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import recall_score, roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import train_test_split

import IPython
## Application
from DeepMAsED import Models_FL as Models
from DeepMAsED import Utils

class Config(object):
    def __init__(self, args):
        self.max_len = args.max_len
        self.filters = args.filters
        self.n_conv = args.n_conv
        self.n_fc = args.n_fc
        self.n_hid = args.n_hid
        self.n_features = 21
        # self.pool_window = args.pool_window
        self.dropout = args.dropout
        self.lr_init = args.lr_init


def main(args):
    #flags
    TRAINFULL = False #if we want at the very end to use all data
    FILTERLONG = True

    # init
    np.random.seed(args.seed)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    save_path = args.save_path
    
    config = Config(args)
    # Load data
    # do not distinguish megahit and metaspades
    train_data_dict = Utils.build_sample_index(Path(args.feature_files_path), args.n_procs)
    logging.info('Train data dictionary created. number of samples: {}'.format(len(train_data_dict)))

    if FILTERLONG:
        all_lens = Utils.read_all_lens(train_data_dict)
        inds_long = np.arange(len(all_lens))[np.array(all_lens) > args.max_len] #to fit model on top
        inds_short = np.arange(len(all_lens))[np.array(all_lens) <= args.max_len]
        all_contigs = list(train_data_dict.items())
        long_train_data_dict = dict(np.array(all_contigs)[inds_long])
        train_data_dict = dict(np.array(all_contigs)[inds_short])
        logging.info('{} long contigs are filtered out, {} contigs left'.format(len(inds_long), len(inds_short)))

    # kfold cross validation
    if args.n_folds >= 0:
        logging.info('Running kfold cross validation. n-folds: {}'.format(args.n_folds))
    #     outfile_h5 = os.path.join(save_path, str(args.n_folds - 1) + '_model.h5')
    #     if os.path.exists(outfile_h5) and args.force_overwrite is False:
    #         msg = 'Output already exists ({}). Use --force-overwrite to overwrite the file'
    #         raise IOError(msg.format(outfile_h5))
    #
    #     # iter over folds
    #     ap_scores = []
    #     for val_idx in range(args.n_folds):
    #         logging.info('Fold {}: Constructing model...'.format(val_idx))
    #         x_tr, x_val, y_tr, y_val = Utils.kfold(x, y, val_idx, k=args.n_folds)
    #         deepmased = Models.deepmased(config)
    #         deepmased.print_summary()
    #
    #         #Construct generator
    #         dataGen = Models.Generator(x_tr, y_tr, args.max_len,
    #                                    batch_size=args.batch_size)
    #
    #         # Init validation generator and
    #         dataGen_val = Models.Generator(x_val, y_val, args.max_len,
    #                                        batch_size=args.batch_size,
    #                                        shuffle=False)
    #
    #         #Train model
    #         tb_logs = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(save_path, 'logs'),
    #                                               histogram_freq=0,
    #                                               write_graph=True, write_images=True)
    #         logging.info('Fold {}: Training network...'.format(val_idx))
    #         ## binary classification (extensive misassembly)
    #
    #         deepmased.net.fit(x=dataGen,
    #                                     validation_data=dataGen_val,
    #                                     epochs=args.n_epochs,
    #                                     use_multiprocessing=args.n_procs > 1,
    #                                     workers=args.n_procs,
    #                                     verbose=2,
    #                                     callbacks=[tb_logs, deepmased.reduce_lr])
    #         # AUC scores
    #         logging.info('Fold {}: Computing AUC scores...'.format(val_idx))
    #         scores_val = deepmased.predict_generator(dataGen_val)
    #         ap_scores.append(average_precision_score(y_val[0 : scores_val.size], scores_val))
    #
    #         # Saving data
    #         outfile_h5_fold = os.path.join(save_path, str(val_idx) + '_model.h5')
    #         deepmased.save(outfile_h5_fold)
    #         logging.info('Fold {}: File written: {}'.format(val_idx, outfile_h5_fold))
    #         outfile_pkl_fold = os.path.join(save_path, 'scores.pkl')
    #         with open(outfile_pkl_fold, 'wb') as f:
    #             pickle.dump(ap_scores, f)
    #         logging.info('Fold {}: File written: {}'.format(val_idx, outfile_pkl_fold))

    else:
        # Skip kfold and simply pool all the data for training
        ## all elements in x and y are combined
        logging.info('NOTE: Training on all pooled data!')

        logging.info('Constructing model...')
        strategy = tf.distribute.MirroredStrategy()
        logging.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        with strategy.scope():
            deepmased = Models.deepmased(config)
        deepmased.print_summary()

        tb_logs = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(save_path, 'logs_final'),
                                              histogram_freq=0,
                                              write_graph=True, write_images=True)
        # save model every epoch
        mc_file = os.path.join(save_path, '_'.join(['mc_epoch', "{epoch}", args.save_name, 'model.h5']))
        logging.info('mc_file : {}'.format(mc_file))
        mc = ModelCheckpoint(mc_file, save_freq="epoch", verbose=1)

        if args.val_path:
            logging.info('Training network with validation...')
            val_data_dict = Utils.build_sample_index(Path(args.val_path), args.n_procs)
            logging.info('Validation data dictionary created. number of samples: {}'.format(len(val_data_dict)))
            dataGen_val = Models.GeneratorBigD(val_data_dict, args.max_len, args.batch_size,
                                           shuffle=False,
                                           rnd_seed=args.seed, nprocs=args.n_procs)
            list_callbacks = [tb_logs, mc,
                              deepmased.reduce_lr,
                              Utils.auc_callback(dataGen_val)]
            # if args.early_stop:
            #     es = EarlyStopping(monitor='val_loss', verbose=1, patience=9)
            #     list_callbacks.append(es)
            #     mc_file = os.path.join(save_path, '_'.join(['mc', args.save_name, args.technology, 'model.h5']))
            #     mc = ModelCheckpoint(mc_file, monitor='val_loss', verbose=1, save_best_only=True)
            #     list_callbacks.append(mc)

            deepmased.net.fit(x=dataGen,validation_data=dataGen_val,
                                        epochs=args.n_epochs,
                                        use_multiprocessing=args.n_procs > 1,
                                        workers=args.n_procs,
                                        verbose=2,
                                        callbacks=list_callbacks)
        elif TRAINFULL==False:
            logging.info('split data')
            contigs = list(train_data_dict.items())
            y_all = Utils.read_all_labels(train_data_dict)
            items_train, items_test = train_test_split(contigs, test_size=0.2, random_state=args.seed,
                                                       shuffle=True, stratify=y_all)
            split_train_dict = dict(items_train)
            split_val_dict = dict(items_test)
            logging.info('Train dataset:')
            dataGen_split_train = Models.GeneratorBigD(split_train_dict, args.max_len, args.batch_size,
                                           shuffle=True, fraq_neg=args.fraq_neg,
                                           rnd_seed=args.seed, nprocs=args.n_procs)
            logging.info('Validation dataset:')
            dataGen_split_val = Models.GeneratorBigD(split_val_dict, args.max_len, args.batch_size,
                                           shuffle=False, nprocs=args.n_procs)
            y_true_val = Utils.read_all_labels(split_val_dict)


            def scheduler(epoch, lr):
                if epoch < 5:
                    return lr
                else:
                    return lr * tf.math.exp(-0.1)
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
            list_callbacks = [lr_scheduler], #tb_logs, mc
                              # deepmased.reduce_lr,
                              # Utils.auc_callback(dataGen_split_val)]

            logging.info('Training network...')
            # idea to check model on validation set every n epochs -> save model if it is better, reduce lr?
            num_epochs = 2
            auc_val_best = 0.
            for iter in range(math.ceil(args.n_epochs/num_epochs)):
                start = time.time()
                deepmased.net.fit(x=dataGen_split_train,
                            epochs=num_epochs,
                            workers=args.n_procs,
                            use_multiprocessing=args.n_procs > 1,
                            max_queue_size=max(args.n_procs, 10),
                            verbose=2,
                            callbacks=list_callbacks)
                duration = time.time() - start
                logging.info("time to fit {} epochs: {}".format(num_epochs, duration))

                logging.info('validation')
                start = time.time()
                y_pred_val = deepmased.predict(dataGen_split_val)
                auc_val = average_precision_score(y_true_val, y_pred_val)
                loss_val = log_loss(y_true_val, y_pred_val)
                recall1_val = recall_score(y_true_val, y_pred_val>0.5, pos_label=1)
                # recall0_val = recall_score(y_true_val, y_pred_val>0.5, pos_label=0)
                logging.info('Validation scores after {} epochs: aucPR: {} - loss: {} - recall1: {} - mean: {}'.format(
                    (iter+1)*num_epochs, auc_val, loss_val, recall1_val, np.mean(y_pred_val)))
                duration = time.time() - start
                logging.info("time validation {}".format(duration))

                if auc_val > auc_val_best:
                    auc_val_best = auc_val
                    best_file = os.path.join(save_path, '_'.join(
                        ['mc_epoch', str(iter*num_epochs), 'aucPR', str(auc_val_best)[:4], args.save_name, 'model.h5']))
                    deepmased.save(best_file)
                    logging.info('  File written: {}'.format(best_file))

        else:
            dataGen = Models.GeneratorBigD(train_data_dict, args.max_len, args.batch_size,
                                           shuffle=True, fraq_neg=args.fraq_neg,
                                           rnd_seed=args.seed, nprocs=args.n_procs)
            logging.info('Training network...')

            deepmased.net.fit(x=dataGen,
                        epochs=args.n_epochs,
                        workers=args.n_procs,
                        use_multiprocessing=args.n_procs > 1,
                        verbose=2,
                        callbacks=[mc]) #tb_logs
            
        logging.info('Saving trained model...')
        x = [args.save_name, args.technology, 'model.h5']
        outfile = os.path.join(save_path, '_'.join(x))
        deepmased.save(outfile)
        logging.info('  File written: {}'.format(outfile))

        #predict for long and fit classifier on top
        # long_train_data_dict

if __name__ == '__main__':
    pass
        
