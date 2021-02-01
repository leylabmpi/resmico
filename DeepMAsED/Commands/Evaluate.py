from __future__ import print_function
from pkg_resources import resource_filename
# import
## batteries
import os
import sys
import argparse
import logging
## application
from DeepMAsED import Evaluate

# functions
def get_desc():
    desc = 'Evaluate model'
    return desc

def parse_args(test_args=None, subparsers=None):
    desc = get_desc()
    epi = """DESCRIPTION:
    Evaluate a trained model generated by `DeepMAsED train`.

    All feature tables must be labeled either "features.tsv" or "features.tsv.gz"
    (or "features.pkl" if already processed).
    """
    if subparsers:
        parser = subparsers.add_parser('evaluate', description=desc, epilog=epi,
                                       formatter_class=argparse.RawTextHelpFormatter)
    else:
        parser = argparse.ArgumentParser(description=desc, epilog=epi,
                                         formatter_class=argparse.RawTextHelpFormatter)

    #default trained model
    pkg_model = resource_filename('DeepMAsED','Model/fl_all_model.h5')
    pkg_mstd = resource_filename('DeepMAsED','Model/fl_all_mean_std.pkl')
    pkg_path, pkg_model = os.path.split(pkg_model)
    _, pkg_mstd  = os.path.split(pkg_mstd)
    # args
    parser.add_argument('--big-data', action='store_true', default=False,
                        help='Use True if work with large dataset in h5 format')
    parser.add_argument('--feature-files-path',  default='', type=str,
                        help='Path to h5 feature files')
    parser.add_argument('--feature-file-table',  default='', type=str,
                        help='Table listing feature table files (see DESCRIPTION)')
    parser.add_argument('--model-path',  default=pkg_path, type=str, 
                        help='Directory containing the model (default: %(default)s)')
    parser.add_argument('--model-name', default=pkg_model, type=str, 
                        help='Model name in the model_path (default: %(default)s)')      
    parser.add_argument('--save-path', default='.', type=str, 
                        help='Directory where to save output (default: %(default)s)')
    parser.add_argument('--save-name', default='deepmased', type=str, 
                        help='Prefix for name in the save_path (default: %(default)s)')
    parser.add_argument('--save-plot', default=None, type=str,                          #not implemented
                        help='Where to save plots (default: %(default)s)')
    parser.add_argument('--batch-size', default=4, type=int, 
                        help='Batch size (default: %(default)s)')
    parser.add_argument('--max-len', default=10000, type=int, 
                        help='Max contig len, fixed input for CNN (default: %(default)s)')
    parser.add_argument('--technology', default='all-asmbl', type=str, 
                        help='Assembler name in the data_path. "all-asmbl" will use all assemblers (default: %(default)s)')
    parser.add_argument('--sdepth', default=None, type=str,
                        help='Use only data with this sequencing depth (default: %(default)s)')
    parser.add_argument('--rich', default=None, type=str,
                        help='Use only data with this comunity richness (default: %(default)s)')
    parser.add_argument('--seed', default=12, type=int, 
                        help='Seed used for numpy.random (default: %(default)s)')
    parser.add_argument('--n-procs', default=1, type=int, 
                        help='Number of parallel processes (default: %(default)s)')
    parser.add_argument('--filter10', action='store_true', default=False,
                        help='Use True if want to train on 1-9 reps')
    parser.add_argument('--rep10', action='store_true', default=False,
                        help='use only rep 10 (validation set)')
    parser.add_argument('--method-pred', default='random', type=str,
                        help='How to predict: random, fulllength, chunks')
    parser.add_argument('--long-def', default=1000, type=int,
                        help='Definition of -long- contig. If want predict for all use(default: %(default)s)')
    parser.add_argument('--mem-lim', default=500000, type=int,
                        help='Max contig that fits in one batch (default: %(default)s)')
    parser.add_argument('--window', default=5000, type=int,
                        help='Window size for chunks method, size of piece for random method (default: %(default)s)')

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
    print()
    print (args)
    print()
    Evaluate.main(args)
    
# main
if __name__ == '__main__':
    pass


