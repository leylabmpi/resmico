# import
## batteries
import os
import sys
import logging
import _pickle as pickle
from pathlib import Path
## 3rd party
import numpy as np
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
from tensorflow.keras.models import load_model
# import IPython
from sklearn.metrics import average_precision_score
## application
from DeepMAsED import Models_FL as Models  #to use contigs of variable length
from DeepMAsED import Utils



def main(args):
    """Main interface
    """
    # init
    np.random.seed(args.seed)
    ## where to save the plot
    save_plot = args.save_plot
    if save_plot is None:
        save_plot = args.save_path
                
    # Load and process data
    # Provide objective to load
    custom_obj = {'class_recall_0':Utils.class_recall_0, 'class_recall_1': Utils.class_recall_1}    
    h5_file = os.path.join(args.model_path, args.model_name)
    if not os.path.exists(h5_file):
        msg = 'Cannot find {} file in {}'
        raise IOError(msg.format(args.model_name, args.model_path))

    if args.big_data:
        # do not distinguish megahit and metaspades
        data_dict = Utils.build_sample_index(Path(args.feature_files_path), args.n_procs,
                                             sdepth=args.sdepth, rich=args.rich)
        logging.info('Data dictionary created. Number of samples: {}'.format(len(data_dict)))
        dataGen = Models.GeneratorBigD(data_dict, args.max_len, args.batch_size,
                                      shuffle=False, rnd_seed=args.seed, nprocs=args.n_procs)
        labels_val = Utils.read_all_labels(data_dict)

        strategy = tf.distribute.MirroredStrategy()
        logging.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        logging.info('Loading model: {}'.format(h5_file))
        with strategy.scope():
            model = load_model(h5_file, custom_objects=custom_obj)
            optimizer = tf.keras.optimizers.Adam()
            model.compile(loss='binary_crossentropy',
                             optimizer=optimizer,
                             metrics=[Utils.class_recall_0, Utils.class_recall_1])
            score_val = model.predict(dataGen, use_multiprocessing=args.n_procs > 1, workers=args.n_procs)
        aupr = average_precision_score(labels_val, score_val)
        logging.info('average_precision_score: {}'.format(aupr))
        scores = {'y': labels_val, 'pred': score_val}

    else:
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
