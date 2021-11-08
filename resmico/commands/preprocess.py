from __future__ import print_function
import argparse
import logging
from resmico import preprocess

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
    
    The new data loader relies on .h5 format, so in addition run convert_pickles_to_hdf.sh
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

    # not needed for the final version
    # parser.add_argument('--add_pos_feat', action='store_true', default=False,
    #                     help='Add positional features (default: %(default)s)')
    # parser.add_argument('--name_input_folder',  default='features', type=str,
    #                     help='Name in a path that need to be changed -> + _pos (see DESCRIPTION)')


    parser.add_argument('--rch',  default='', type=str,
                        help='Richness to process parts of data in parallel (see DESCRIPTION)')

    #  not needed for the final version
    # parser.add_argument('--add_feat_h5', action='store_true', default=False,
    #                     help='Add features to h5 files (default: %(default)s)')
    # parser.add_argument('--input_folder',  default='', type=str,
    #                     help='path with oiginal h5 files')
    # parser.add_argument('--output_folder',  default='', type=str,
    #                     help='path for h5 files with new feature')

    parser.add_argument('--real_data', action='store_true', default=False,
                        help='Preprocessing is different for real data (default: %(default)s)')

    parser.add_argument('--deepmased_v1', action='store_true', default=False,
                        help='Preprocessing is different for the deepmased v1 (default: %(default)s)')
#     parser.add_argument('--longdir', action='store_true', default=False,
#                         help='Six variable parameters in simulation (default: %(default)s)')
    
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
    preprocess.main(args)
    
# main
if __name__ == '__main__':
    pass


