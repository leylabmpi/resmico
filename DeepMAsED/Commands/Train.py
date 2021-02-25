from __future__ import print_function
# import
## batteries
import os
import sys
import argparse
import logging
## application
from DeepMAsED import Train
from DeepMAsED import Train_BigD

# functions
def parse_args(test_args=None, subparsers=None):
    desc = 'Train model'
    epi = """DESCRIPTION:
    #-- Recommended training flow --#
    * Partition your data into train & test, and just use
      the train data for the following 
        * see feature file table description below
    * Select a grid search of hyper-parameters to consider
      (learning rate, number of layers, etc).
    * Train with kfold = 5 (for example) for each combination of 
      hyper-parameters.
    * For each combination of hyper-parameters, check scores.pkl, 
      which contains the cross validation scores, and select the 
      hyper-parameters leading to the highest average CV
    * Re-launch the whole training with `--n-folds -1` and the best 
      hyper-parameters (this is now one single run). 

    #-- Feature File Table format --#
    * DeepMAsED-SM will generate a feature file table that lists all
      feature files and their associated metadata (eg., assembler & sim-rep).
    * The table must contain the following columns:
      * `feature_file` = the path to the feature file (created by DeepMAsED-SM, see README)
        * The files can be (gzip'ed) tab-delim or pickled (see below on `--pickle-only`)
      * `rep` = the metagenome simulation replicate 
        * Set to 1 if real data
      * `assembler` = the metadata assembler

    #-- Pickled feature files --#
    DeepMAsED-SM will generate tab-delim feature tables; however,
    DeepMAsED uses formatted & pickled versions of the tab-delim feature tables as input.
    Pickled preprocessed files should be created beforehand with Preprocess command.
    """
    if subparsers:
        parser = subparsers.add_parser('train', description=desc, epilog=epi,
                                       formatter_class=argparse.RawTextHelpFormatter)
    else:
        parser = argparse.ArgumentParser(description=desc, epilog=epi,
                                         formatter_class=argparse.RawTextHelpFormatter)

    # args
    parser.add_argument('--big-data', action='store_true', default=False,
                        help='Use True if work with large dataset in h5 format')
    parser.add_argument('--feature-files-path',  default='', type=str,
                        help='Path to h5 feature files')
    parser.add_argument('--feature-file-table',  default='', type=str,
                        help='Table listing feature table files (see DESCRIPTION)')
    parser.add_argument('--technology', default='all-asmbl', type=str, 
                        help='Assembler name in the data_path. "all-asmbl" will use all assemblers (default: %(default)s)')    
    parser.add_argument('--save-path', default='model', type=str, 
                        help='Where to save training weights and logs (default: %(default)s)')
    parser.add_argument('--save-name', default='deepmased', type=str, 
                        help='Prefix for name in the save-path (default: %(default)s)') 
    parser.add_argument('--val-path', default=None, type=str, 
                        help='Path to validation data (default: %(default)s)')
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help='Early stopping. Can be used only if val-path provided (default: %(default)s)')
    parser.add_argument('--net-type', default='cnn_globpool', type=str,
                        help='Type of NN: lstm, cnn_globpool, cnn_resnet, cnn_lstm (default: %(default)s)')
    parser.add_argument('--num-blocks', default=5, type=int,
                        help='Number of residual blocks (3 or 4, 5, 6) (default: %(default)s)')
    parser.add_argument('--filters', default=16, type=int,
                        help='N of filters for first conv layer. Then x2 (default: %(default)s)')
    parser.add_argument('--ker-size', default=5, type=int,
                        help='CNN kernel size (default: %(default)s)')
    parser.add_argument('--n-hid', default=50, type=int, 
                        help='N of units in fully connected layers (default: %(default)s)')
    parser.add_argument('--n-conv', default=5, type=int, 
                        help='N of conv layers (default: %(default)s)')
    parser.add_argument('--n-fc', default=2, type=int,
                        help='N of fully connected layers (default: %(default)s)')
    parser.add_argument('--n-epochs', default=10, type=int, 
                        help='N of training epochs (default: %(default)s)')
    parser.add_argument('--batch-size', default=6, type=int, 
                        help='Batch size (default: %(default)s)')
    parser.add_argument('--max-len', default=10000, type=int,
                        help='Max contig len, fixed input for CNN (default: %(default)s)')
    parser.add_argument('--dropout', default=0.2, type=float,
                        help='Rate of dropout (default: %(default)s)')
    parser.add_argument('--n-folds', default=-1, type=int, 
                        help='How many folds for CV. Use "-1" to skip & pool all data for training (default: %(default)s)')
    parser.add_argument('--lr-init', default=0.001, type=float,
                        help='Size of test set (default: %(default)s)')
    parser.add_argument('--seed', default=12, type=int, 
                        help='Seed used for numpy.random and tf (default: %(default)s)')
    parser.add_argument('--n-procs', default=1, type=int, 
                        help='Number of parallel processes (default: %(default)s)')
    parser.add_argument('--fraq_neg', default=1., type=float,
                        help='Portion of samples to keep in overrepresented class (default: %(default)s)')
    # running test args
    if test_args:
        args = parser.parse_args(test_args)
        return args

    return parser


def main(args=None):
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
    # Input
    if args is None:
        args = parse_args()
    # Main interface
    if args.big_data:
        Train_BigD.main(args)
    else:   
        Train.main(args)
    
# main
if __name__ == '__main__':
    pass


