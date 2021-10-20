import os
import logging
import math
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.metrics import recall_score, average_precision_score

from ResMiCo import ContigReader
from ResMiCo import Models_FL as Models


def main(args):
    """
    Trains ResMiCo on binary data files produced by ResMiCo-SM.
    """
    np.random.seed(args.seed)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    logging.info('Building Tensorflow model...')
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        resmico = Models.Resmico(args)
    resmico.print_summary()

    # save model every epoch
    model_file = os.path.join(args.save_path, '_'.join(['mc_epoch', "{epoch}", args.save_name, 'model.h5']))
    logging.info(f'Model will be saved to: {model_file}')
    mc = ModelCheckpoint(model_file, save_freq="epoch", verbose=1)

    # tensorboard logs
    tb_logs = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.save_path, 'logs_final'),
                                             histogram_freq=0,
                                             write_graph=True, write_images=True)

    logging.info('Loading contig data...')
    reader = ContigReader.ContigReader(args.feature_files_path, args.features, args.n_procs, args.chunks,
                                       args.no_cython)

    # separate data into 90% for training and 10% for evaluation
    all_idx = np.arange(len(reader))
    np.random.shuffle(all_idx)
    train_idx = all_idx[:(9 * len(reader)) // 10]
    eval_idx = all_idx[(9 * len(reader)) // 10:]
    logging.info(f'Using {len(train_idx)} contigs for training, {len(eval_idx)} contigs for evaluation')

    # create data generators for training data and evaluation data
    train_data = Models.BinaryData(reader, train_idx, args.batch_size, args.features, args.max_len, args.fraq_neg,
                                   args.cache, args.log_progress)

    # convert the slow Keras train_data of type Sequence to a tf.data object
    # first, we convert the keras sequence into a generator-like object
    data_iter = lambda: (s for s in train_data)

    # second, we use tf.data.Dataset.from_generator to create a tf.data.Dataset object and use this for training
    train_data_tf = tf.data.Dataset.from_generator(
        data_iter,
        output_signature=(
            tf.TensorSpec(shape=(args.batch_size, None, len(train_data.expanded_feature_names)), dtype=tf.float32),
            tf.TensorSpec(shape=(args.batch_size), dtype=tf.uint8)))

    # add a prefetch option that builds the next batch ready for consumption by the GPU as it is working on
    # the current batch.
    train_data_tf = train_data_tf.prefetch(2 * strategy.num_replicas_in_sync)

    # set the sharding policy to DATA in order to avoid Tensorflow ugly console barf
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_data_tf = train_data_tf.with_options(options)

    np.seterr(all='raise')
    eval_data = Models.BinaryDataEval(reader, eval_idx, args.features, args.max_len, args.max_len // 2,
                                      int(args.gpu_eval_mem_gb * 1e9 * 0.8), args.cache_validation or args.cache,
                                      args.log_progress)
    eval_data_y = np.array([0 if reader.contigs[idx].misassembly == 0 else 1 for idx in eval_data.all_indices])

    # convert the slow Keras eval_data of type Sequence to a tf.data object
    data_iter = lambda: (s for s in eval_data)
    eval_data_tf = tf.data.Dataset.from_generator(
        data_iter,
        output_signature=(tf.TensorSpec(shape=(None, None, len(eval_data.expanded_feature_names)), dtype=tf.float32)))
    eval_data_tf = eval_data_tf.prefetch(4 * strategy.num_replicas_in_sync)
    eval_data_tf = eval_data_tf.with_options(options)  # avoids Tensorflow ugly console barf

    logging.info('Training network...')
    num_epochs = 2  # todo: last run monitor more often
    auc_val_best = 0.5
    for epoch in range(math.ceil(args.n_epochs / num_epochs)):
        start = time.time()
        resmico.net.fit(x=train_data_tf,
                        epochs=num_epochs,
                        workers=args.n_procs,
                        use_multiprocessing=True,
                        max_queue_size=max(args.n_procs, 10),
                        callbacks=[mc, tb_logs],
                        verbose=2)
        duration = time.time() - start
        logging.info(f'Fitted {num_epochs} epochs in {duration:.0f}s')
        train_data.on_epoch_end()

        logging.info('Starting validation')
        start = time.time()
        eval_data_flat_y = resmico.predict(x=eval_data_tf,
                                           workers=args.n_procs,
                                           use_multiprocessing=True,
                                           max_queue_size=max(args.n_procs, 10))
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
        if cur_epoch > 10 and auc_val < auc_val_best:
            lr_old = K.get_value(resmico.net.optimizer.lr)
            K.set_value(resmico.net.optimizer.lr, lr_old * 0.8)  # changed for 120h jobs
            logging.info(f'Updated learning rate from: {lr_old} to {K.get_value(resmico.net.optimizer.lr)}')

        if auc_val > auc_val_best:
            auc_val_best = auc_val
            best_file = os.path.join(args.save_path, '_'.join(
                ['mc_epoch', str(cur_epoch), 'aucPR', str(auc_val_best)[:5], args.save_name,
                 'model.h5']))
            resmico.save(best_file)
            logging.info('New best model written to: {}'.format(best_file))

    logging.info('Saving trained model...')
    x = [args.save_name, args.technology, 'model.h5']
    outfile = os.path.join(args.save_path, '_'.join(x))
    resmico.save(outfile)
    logging.info(f'Latest model written to: {outfile}')


if __name__ == '__main__':
    pass
