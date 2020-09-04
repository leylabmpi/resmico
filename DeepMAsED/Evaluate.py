# import
## batteries
import os
import sys
import logging
import _pickle as pickle
## 3rd party
import numpy as np
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
from tensorflow.keras.models import load_model
import IPython
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
    logging.info('Loading model: {}'.format(h5_file))
    model = load_model(h5_file, custom_objects=custom_obj)
   
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
