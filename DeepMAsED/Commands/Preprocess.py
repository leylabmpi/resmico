from __future__ import print_function
# import
## batteries
import os
import sys
import argparse
import logging
## application
from DeepMAsED import Preprocess

# functions
def parse_args(test_args=None, subparsers=None):
    desc = 'Preprocess data'
    epi = """DESCRIPTION:
    #-- Recommended flow --#

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
    DeepMAsED-SM generates tab-delim feature tables; however,
    DeepMAsED uses formatted & pickled versions of the tab-delim feature tables.
    `DeepMAsED preprocess` need to be applied to the data before training and testing.
    It works in 3 steps. 
    Firstly, use `--pickle-tsv` argument to create pickle version of tsv datatables 
    with selected features. Apply it to both train and test datasets.
    Secondly, `--compute-mean-std` using pickled version of train dataset.
    Lastly,run '--standard-data' with precomputed mean and std, this will standartize data
    and rewrite pickled version of datatables. Apply it to both train and test set.
    The resulting pickle files could be used as input to the model.
    """
    if subparsers:
        parser = subparsers.add_parser('preprocess', description=desc, epilog=epi,
                                       formatter_class=argparse.RawTextHelpFormatter)
    else:
        parser = argparse.ArgumentParser(description=desc, epilog=epi,
                                         formatter_class=argparse.RawTextHelpFormatter)

    # actions
    parser.add_argument('--pickle-tsv', action='store_true', default=False,
                        help='Create pickle files from tsv (default: %(default)s)')
    parser.add_argument('--compute-mean-std', action='store_true', default=False,
                        help='Compute mean and std from pickle (default: %(default)s)')
    parser.add_argument('--standard-data', action='store_true', default=False,
                        help='Standardize data with precomputed mean and std (default: %(default)s)')
    parser.add_argument('--downsample', action='store_true', default=False,
                        help='NOT SUPPORTED YET') #to reduce number of good contigs
    
    parser.add_argument('--feature-file-table',  default='feature_file_table', type=str, 
                        help='Table listing feature table files (see DESCRIPTION)')
    parser.add_argument('--mean-std-file',  default='', type=str, 
                        help='File with precomputed sum, sum of squares and number of elements')    
    parser.add_argument('--technology', default='all-asmbl', type=str, 
                        help='Assembler name in the data_path. "all-asmbl" will use all assemblers (default: %(default)s)')    
    parser.add_argument('--set-target', default=True, type=str, 
                        help='True if label is known')
    parser.add_argument('--force-overwrite', action='store_true', default=False,
                        help='Force re-creation of pickle files (default: %(default)s)')
    parser.add_argument('--n-procs', default=1, type=int, 
                        help='Number of parallel processes (default: %(default)s)')
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
    Preprocess.main(args)
    
# main
if __name__ == '__main__':
    pass


