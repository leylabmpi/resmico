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
from tensorflow.keras import backend as K
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
        self.n_feat = args.n_feat
        # self.pool_window = args.pool_window
        self.dropout = args.dropout
        self.lr_init = args.lr_init
        self.net_type = args.net_type
        self.num_blocks = args.num_blocks
        self.ker_size = args.ker_size
        self.seed = args.seed


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

    logging.info('Constructing model...')
    strategy = tf.distribute.MirroredStrategy()
    logging.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        deepmased = Models.deepmased(config)
    deepmased.print_summary()

    #check if seed works, print weights for the first 5 layers -> seed works
    # for n, layer in enumerate(deepmased.net.layers):
    #     if n<5:
    #         logging.info('layer: {}, config: {}, weights: {}'.format(
    #             n, layer.get_config(), layer.get_weights()))
    #     else:
    #         break

    tb_logs = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(save_path, 'logs_final'),
                                          histogram_freq=0,
                                          write_graph=True, write_images=True)
    # save model every epoch
    mc_file = os.path.join(save_path, '_'.join(['mc_epoch', "{epoch}", args.save_name, 'model.h5']))
    logging.info('mc_file : {}'.format(mc_file))
    mc = ModelCheckpoint(mc_file, save_freq="epoch", verbose=1)

    if args.val_path:
        logging.info('Code needs update...')
        # val_data_dict = Utils.build_sample_index(Path(args.val_path), args.n_procs)
        # logging.info('Validation data dictionary created. number of samples: {}'.format(len(val_data_dict)))
        # dataGen_val = Models.GeneratorBigD(val_data_dict, args.max_len, args.batch_size,
        #                                shuffle=False,
        #                                rnd_seed=args.seed, nprocs=args.n_procs)
        # list_callbacks = [tb_logs, mc,
        #                   deepmased.reduce_lr,
        #                   Utils.auc_callback(dataGen_val)]
        # # if args.early_stop:
        # #     es = EarlyStopping(monitor='val_loss', verbose=1, patience=9)
        # #     list_callbacks.append(es)
        # #     mc_file = os.path.join(save_path, '_'.join(['mc', args.save_name, args.technology, 'model.h5']))
        # #     mc = ModelCheckpoint(mc_file, monitor='val_loss', verbose=1, save_best_only=True)
        # #     list_callbacks.append(mc)
        #
        # deepmased.net.fit(x=dataGen,validation_data=dataGen_val,
        #                             epochs=args.n_epochs,
        #                             use_multiprocessing=args.n_procs > 1,
        #                             workers=args.n_procs,
        #                             verbose=2,
        #                             callbacks=list_callbacks)

    elif TRAINFULL==False:
        #main working area
        logging.info('Split data: reps 1-9 for training, 10 for validation')
        train_data_dict = Utils.build_sample_index(Path(args.feature_files_path), args.n_procs, filter10=True)
        logging.info('Train data dictionary created. number of samples: {}'.format(len(train_data_dict)))
        val_data_dict = Utils.build_sample_index(Path(args.feature_files_path), args.n_procs, rep10=True)
        logging.info('Validation data dictionary created. number of samples: {}'.format(len(val_data_dict)))

        if FILTERLONG:
            #train
            logging.info('max_len for train: {}'.format(args.max_len))
            all_lens = Utils.read_all_lens(train_data_dict)
            #TODO: try to keep a bit longer contigs, to learn that chunck does not always start in the beginning
            inds_short = np.arange(len(all_lens))[np.array(all_lens) <= args.max_len] # + 1000
            all_contigs = list(train_data_dict.items())
            train_data_dict = dict(np.array(all_contigs)[inds_short])
            logging.info('from {}, {} short contigs left'.format(len(all_lens), len(inds_short)))
            # validation is not downsampled and always the same max len 5k -- NO, now args.max_len
            logging.info('max_len for validation: {}'.format(args.max_len))
            all_lens = Utils.read_all_lens(val_data_dict)
            inds_short = np.arange(len(all_lens))[np.array(all_lens) <= args.max_len]
            all_contigs = list(val_data_dict.items())
            val_data_dict = dict(np.array(all_contigs)[inds_short])
            logging.info('from {}, {} short contigs left'.format(len(all_lens), len(inds_short)))

        # random split
        # logging.info('split data randomly')
        # contigs = list(train_data_dict.items())
        # y_all = Utils.read_all_labels(train_data_dict)
        # items_train, items_test = train_test_split(contigs, test_size=0.2, random_state=args.seed,
        #                                            shuffle=True, stratify=y_all)
        # split_train_dict = dict(items_train)
        # split_val_dict = dict(items_test)

        logging.info('Train dataset:')
        dataGen_split_train = Models.GeneratorBigD(train_data_dict, args.max_len, args.batch_size,
                                       shuffle_data=True, fraq_neg=args.fraq_neg,
                                       rnd_seed=args.seed, nprocs=args.n_procs)
        logging.info('Validation dataset:')
        dataGen_split_val = Models.GeneratorBigD(val_data_dict, args.max_len, args.batch_size,
                                       shuffle_data=False, nprocs=args.n_procs)
        y_true_val = Utils.read_all_labels(val_data_dict)


        # def scheduler(epoch, lr):
        #     if epoch < 5:
        #         logging.info('scheduler returns: {}'.format(lr))
        #         return lr
        #     else:
        #         lr = lr * 0.9
        #         logging.info('scheduler returns reduced lr: {}'.format(lr))
        #         return lr
        # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        # list_callbacks = [lr_scheduler], #tb_logs, mc
        #                   # deepmased.reduce_lr,
        #                   # Utils.auc_callback(dataGen_split_val)]

        logging.info('Training network...')
        num_epochs = 2 #todo: last run monitor more often
        auc_val_best = 0.84
        for iter in range(math.ceil(args.n_epochs/num_epochs)):
            start = time.time()
            deepmased.net.fit(x=dataGen_split_train,
                        epochs=num_epochs,
                        workers=args.n_procs,
                        use_multiprocessing=args.n_procs > 1,
                        max_queue_size=max(args.n_procs, 10),
                        verbose=2)
                        # callbacks=list_callbacks)
            duration = time.time() - start
            logging.info("time to fit {} epochs: {}".format(num_epochs, duration))

            logging.info('validation')
            start = time.time()
            y_pred_val = deepmased.predict(dataGen_split_val,
                                           workers=args.n_procs,
                                           use_multiprocessing=args.n_procs > 1,
                                           max_queue_size=max(args.n_procs, 10),
                                           verbose=2)
            auc_val = average_precision_score(y_true_val, y_pred_val)
            # loss_val = log_loss(y_true_val, y_pred_val)
            recall1_val = recall_score(y_true_val, y_pred_val>0.5, pos_label=1)
            recall0_val = recall_score(y_true_val, y_pred_val>0.5, pos_label=0)
            logging.info('Validation scores after {} epochs: aucPR: {} - - recall1: {} - recall0: {} - mean: {}'.format(
                (iter+1)*num_epochs, auc_val, recall1_val, recall0_val, np.mean(y_pred_val)))
            duration = time.time() - start
            logging.info("time validation {}".format(duration))

            # update the learning rate
            if ((iter+1)*num_epochs>10) and (auc_val < auc_val_best):
                lr_old = K.get_value(deepmased.net.optimizer.lr)
                print("[INFO] old learning rate: {}".format(lr_old))
                K.set_value(deepmased.net.optimizer.lr, lr_old*0.8) #changed for 120h jobs
                print("[INFO] new learning rate: {}".format(K.get_value(deepmased.net.optimizer.lr)))

            if auc_val > auc_val_best:
                auc_val_best = auc_val
                best_file = os.path.join(save_path, '_'.join(
                    ['mc_epoch', str((iter+1)*num_epochs), 'aucPR', str(auc_val_best)[:5], args.save_name, 'model.h5']))
                deepmased.save(best_file)
                logging.info('  File written: {}'.format(best_file))


    else:
        train_data_dict = Utils.build_sample_index(Path(args.feature_files_path), args.n_procs)
        logging.info('Train data dictionary created. number of samples: {}'.format(len(train_data_dict)))

        if FILTERLONG:
            all_lens = Utils.read_all_lens(train_data_dict)
            inds_long = np.arange(len(all_lens))[np.array(all_lens) > args.max_len]  # to fit model on top
            inds_short = np.arange(len(all_lens))[np.array(all_lens) <= args.max_len]
            all_contigs = list(train_data_dict.items())
            long_train_data_dict = dict(np.array(all_contigs)[inds_long])
            train_data_dict = dict(np.array(all_contigs)[inds_short])
            logging.info('{} long contigs are filtered out, {} contigs left'.format(len(inds_long), len(inds_short)))

        dataGen = Models.GeneratorBigD(train_data_dict, args.max_len, args.batch_size,
                                       shuffle_data=True, fraq_neg=args.fraq_neg,
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
        
