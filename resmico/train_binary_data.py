import os
import logging
import math
import time
import atexit

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from sklearn.metrics import recall_score, average_precision_score

from resmico import contig_reader
from resmico import models_fl as Models
from resmico import utils


def main(args):
    """
    Trains ResMiCo on binary data files produced by ResMiCo-SM.
    """
    np.random.seed(args.seed)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # disable tf allocating all memory on the device at the very beginning, as this seems to be causing
    # LSTM's to crash at prediction time.
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    logging.info('Building Tensorflow model...')
    logging.info(args)
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        resmico = Models.Resmico(args)
        if os.path.exists(args.model_checkpoint):
            logging.info(f'Loading model: {args.model_checkpoint}')
#             custom_obj = {'class_recall_0': utils.class_recall_0, 'class_recall_1': utils.class_recall_1,
#                   'GlobalMaskedMaxPooling1D': Models.GlobalMaskedMaxPooling1D}
            resmico.net.load_weights(args.model_checkpoint)
            logging.info('Model loaded')
            
    resmico.print_summary()

    # tensorboard logs
    tb_logs = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.save_path, args.save_name),
                                             histogram_freq=0,
                                             write_graph=True,
                                             write_images=True)

    logging.info('Loading contig data...')
    reader = contig_reader.ContigReader(args.feature_files_path, args.features, args.n_procs,
                                        args.no_cython, args.stats_file, min_len=args.min_contig_len,
                                        min_avg_coverage=args.min_avg_coverage,
                                        feature_file_match=args.feature_file_match)

    # separate data into 90% for training and 10% for evaluation
    all_idx = np.arange(len(reader))
    if args.val_ind_f:
        logging.info(f'Split data: using {args.val_ind_f} for validation, for training everything else')
        eval_idx = list(pd.read_csv(args.val_ind_f)['val_ind'])
        train_idx = list(set(all_idx) - set(eval_idx))
    else:
        np.random.shuffle(all_idx)
        train_idx = all_idx[:(9 * len(reader)) // 10]
        eval_idx = all_idx[(9 * len(reader)) // 10:]
        df = pd.DataFrame(eval_idx, columns=['val_ind'])
        if os.path.isfile(args.feature_files_path):
            fname = os.path.join(os.path.split(args.feature_files_path)[0], "evaluation_indices.csv")
        else:
            fname = os.path.join(args.feature_files_path.split(",")[0], "evaluation_indices.csv")
        df.to_csv(fname)
        logging.info(f'Evaluation indices saved to: {fname}')
    logging.info(f'Using {len(train_idx)} contigs for training, {len(eval_idx)} contigs for evaluation')

    # create data generators for training data and evaluation data
    train_data = Models.BinaryDatasetTrain(reader, train_idx, args.batch_size, args.features, args.max_len,
                                           args.num_translations, args.max_translation_bases, args.fraq_neg,
                                           args.cache_train or args.cache, args.log_progress, resmico.convoluted_size,
                                           resmico.fixed_length, args.weight_factor)
    # convert the slow Keras train_data of type Sequence to a tf.data object
    # first, we convert the keras sequence into a generator-like object
    data_iter = lambda: (s for s in train_data)

    # second, we use tf.data.Dataset.from_generator to create a tf.data.Dataset object and use this for training
    train_data_tf = tf.data.Dataset.from_generator(
        data_iter,
        output_signature=(
            (tf.TensorSpec(shape=(args.batch_size, None, len(train_data.expanded_feature_names)), dtype=tf.float32),
             tf.TensorSpec(shape=(args.batch_size, None), dtype=tf.bool)),
            tf.TensorSpec(shape=(args.batch_size), dtype=tf.uint8),
            tf.TensorSpec(shape=(args.batch_size), dtype=tf.float32)))

    # add a prefetch option that builds the next batch ready for consumption by the GPU as it is working on
    # the current batch.
    train_data_tf = train_data_tf.prefetch(2 * strategy.num_replicas_in_sync)

    # set the sharding policy to DATA in order to avoid Tensorflow ugly console barf
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_data_tf = train_data_tf.with_options(options)

    np.seterr(all='raise')
    eval_data = Models.BinaryDatasetEval(reader, eval_idx, args.features, args.max_len, args.max_len-500,
                                         int(args.gpu_eval_mem_gb * 1e9 * 0.8), args.cache_validation or args.cache,
                                         args.log_progress, resmico.convoluted_size, 
                                         resmico.fixed_length, batch_size=args.batch_size)

    eval_data_y = np.array([0 if reader.contigs[idx].misassembly == 0 else 1 for idx in eval_data.indices])

    # convert the slow Keras eval_data of type Sequence to a tf.data object
    data_iter = lambda: (s for s in eval_data)
    eval_data_tf = tf.data.Dataset.from_generator(
        data_iter,
        output_signature=(
            # first dimension is batch size, second is contig length, third is number of features
            (tf.TensorSpec(shape=(None, None, len(eval_data.expanded_feature_names)), dtype=tf.float32),
             # first dimension is batch size, second is contig length (no third dimension,
             # as all features are masked the same way)
             tf.TensorSpec(shape=(None, None), dtype=tf.bool)),
            tf.TensorSpec(shape=(None), dtype=tf.bool)
        ))
    eval_data_tf = eval_data_tf.prefetch(4 * strategy.num_replicas_in_sync)
    eval_data_tf = eval_data_tf.with_options(options)  # avoids Tensorflow ugly console barf

    logging.info('Training network...')
    num_epochs = 2
    train_data_tf = train_data_tf.repeat(num_epochs)
    auc_val_best = 0
    auc_val_prev = 0
    best_file = None
    for epoch in range(math.ceil(args.n_epochs / num_epochs)):
        start = time.time()
        resmico.net.fit(x=train_data_tf,
                        epochs=num_epochs,
                        steps_per_epoch=len(train_data),
                        workers=args.n_procs,
                        use_multiprocessing=True,
                        max_queue_size=max(args.n_procs, 10),
#                         callbacks=[tb_logs],
                        verbose=2)

        duration = time.time() - start
        logging.info(f'Fitted {num_epochs} epochs in {duration:.0f}s')
        train_data.on_epoch_end()

        logging.info('Starting validation')
        start = time.time()
        eval_data_flat_y = resmico.predict(x=eval_data_tf,
                                           workers=args.n_procs,
                                           use_multiprocessing=True,
                                           max_queue_size=max(args.n_procs, 10),
                                           verbose=2)
        eval_data_predicted_y = eval_data.group(eval_data_flat_y)

        auc_val = average_precision_score(eval_data_y, eval_data_predicted_y)
        recall1_val = recall_score(eval_data_y, eval_data_predicted_y > 0.5, pos_label=1)
        recall0_val = recall_score(eval_data_y, eval_data_predicted_y > 0.5, pos_label=0)
        logging.info(f'Validation scores after {(epoch + 1) * num_epochs} epochs: aucPR: {auc_val} - '
                     f'recall1: {recall1_val} - recall0: {recall0_val} - mean: {np.mean(eval_data_predicted_y)}')
        duration = time.time() - start
        logging.info(f'Validation done in {duration:.0f}s')

        # update the learning rate
        cur_epoch = (epoch + 1) * num_epochs
        if cur_epoch > 10 and auc_val < auc_val_prev:
            lr_old = K.get_value(resmico.net.optimizer.lr)
            K.set_value(resmico.net.optimizer.lr, lr_old * 0.8)  # changed for 120h jobs
            logging.info(f'Updated learning rate from: {lr_old} to {K.get_value(resmico.net.optimizer.lr)}')

        auc_val_prev = auc_val
        if auc_val > auc_val_best:
            if best_file:  # delete old best model
                try:
                    os.remove(best_file)
                except OSError:
                    logging.warning('Unable to remove: ' + best_file)
            auc_val_best = auc_val
            best_file = os.path.join(args.save_path, '_'.join(
                ['mc_epoch', str(cur_epoch), 'aucPR', str(auc_val_best)[:5], args.save_name,
                 'model.h5']))
            resmico.save(best_file)
            logging.info(f'New best model written to: {best_file}')

    # saving
    logging.info('Saving trained model...')
    outfile = os.path.join(args.save_path, args.save_name + '_' + str(auc_val)[:5] + '_e' + str(args.n_epochs) + '.h5')
    resmico.save(outfile)
    logging.info(f'Latest model written to: {outfile}')
    
    # exit
    atexit.register(strategy._extended._collective_ops._pool.close)


if __name__ == '__main__':
    pass
