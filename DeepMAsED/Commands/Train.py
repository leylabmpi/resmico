from __future__ import print_function
# import
## batteries
import os
import sys
import argparse
## application
from DeepMAsED import Train

# functions
def get_desc():
    desc = 'Train model'
    return desc

def parse_args(test_args=None, subparsers=None):
    desc = get_desc()
    epi = """DESCRIPTION:
    The features table files should either be tab-delim & 
    gzip'ed (output from DeepMAsED-SM) and labeled features.tsv.gz'. 
    You can also provide a pickled version: 'features_new.pkl'.
    """
    if subparsers:
        parser = subparsers.add_parser('train', description=desc, epilog=epi,
                                       formatter_class=argparse.RawTextHelpFormatter)
    else:
        parser = argparse.ArgumentParser(description=desc, epilog=epi,
                                         formatter_class=argparse.RawTextHelpFormatter)

    # args
    parser.add_argument('--data_path', default='data', type=str, 
                        help='Where to find feature table (default: %(default)s)')
    parser.add_argument('--save_path', default='model', type=str, 
                        help='Where to save training weights and logs (default: %(default)s)')
    parser.add_argument('--filters', default=8, type=int, 
                        help='N of filters for first conv layer. Then x2 (default: %(default)s)')
    parser.add_argument('--n_hid', default=20, type=int, 
                        help='N of units in fully connected layers (default: %(default)s)')
    parser.add_argument('--n_conv', default=2, type=int, 
                        help='N of conv layers (default: %(default)s)')
    parser.add_argument('--n_fc', default=1, type=int, 
                        help='N of fully connected layers (default: %(default)s)')
    parser.add_argument('--n_epochs', default=50, type=int, 
                        help='N of training epochs (default: %(default)s)')
    parser.add_argument('--standard', default=1, type=int, 
                        help='Binary, whether or not to standardize the features (default: %(default)s)')
    parser.add_argument('--max_len', default=10000, type=int, 
                        help='Max contig len, fixed input for CNN (default: %(default)s)')
    parser.add_argument('--dropout', default=0.1, type=float, 
                        help='Rate of dropout (default: %(default)s)')
    parser.add_argument('--pool_window', default=40, type=int, 
                        help='Window size for average pooling (default: %(default)s)')
    parser.add_argument('--n_folds', default=5, type=int, 
                        help='How many folds for CV (default: %(default)s)')
    parser.add_argument('--lr_init', default=0.001, type=float, 
                        help='Size of test set (default: %(default)s)')
    parser.add_argument('--mode', default='chimera', type=str, 
                        help='Chimera or edit distance (default: %(default)s)')
    parser.add_argument('--pickle_only', default=False, type=bool, 
                        help='Only pickle files (default: %(default)s)')
    parser.add_argument('--norm_raw', default=1, type=int, 
                        help='Whether to normalize the four one-hot feature of raw (default: %(default)s)')
    
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
    Train.main(args)
    
# main
if __name__ == '__main__':
    pass

