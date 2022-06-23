from __future__ import print_function
from pkg_resources import resource_filename
import os
import argparse
import logging
from resmico import filter_contigs

# functions
def get_desc():
    desc = 'Filter out misassembled contigs'
    return desc

def parse_args(test_args=None, subparsers=None):
    desc = get_desc()
    epi = """DESCRIPTION:
    Filter out contigs predicted to be misassembled. 
    Contigs must be fasta-formatted.
    Filtered contig fasta files will be written to the designed output directory.
    Make sure to adjust the --score-cutoff as needed!
    """
    if subparsers:
        parser = subparsers.add_parser('filter', description=desc, epilog=epi,
                                       formatter_class=argparse.RawTextHelpFormatter)
    else:
        parser = argparse.ArgumentParser(description=desc, epilog=epi,
                                         formatter_class=argparse.RawTextHelpFormatter)
    # args
    parser.add_argument('prediction_table',  metavar='prediction_table', type=str, 
                        help='A table that includes per-contig misassembly predictions. '
                             'The table should be formatted as the output from "resmico predict"')
    parser.add_argument('fasta',  metavar='fasta', type=str, nargs='+',
                        help='>=1 fasta file containing the associated contigs.'
                        ' Alternatively, a list of fasta files (1 per line) can be provided.')
    parser.add_argument('--outdir', default='resmico-filter', type=str, 
                        help='Output directory (default: %(default)s)')
    parser.add_argument('--score-cutoff', default=0.8, type=float, 
                        help='Prediction score cutoff for filtering: >=[score] will be filtered (default: %(default)s)')
    parser.add_argument('--score-delim', default=',', type=str, 
                        help='Delimiter for predictions table (default: %(default)s)')
    parser.add_argument('--add-score', action='store_true', default=False,
                        help='Add prediction score to sequence header "score=<score>"? (default: %(default)s)')
    parser.add_argument('--error-on-missing', action='store_true', default=False,
                        help='Error if a contig does not have a prediction? (default: %(default)s)')
    parser.add_argument('--max-length', default=0, type=int, 
                        help='Only apply filtering to contigs < max-length.\n'
                             'Longer contigs are retained, regardless of the prediction score.\n'
                             'If <1, no cutoff applied. (default: %(default)s)')
    parser.add_argument('--n-proc', default=1, type=int, 
                        help='Number of parallel processes (default: %(default)s)')
    # test args
    if test_args:
        args = parser.parse_args(test_args)
        return args
    # return
    return parser

def main(args=None):
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
    # Input
    if args is None:
        args = parse_args()
        print()
        print (args)
        print()
    # Main interface
    filter_contigs.main(args)
    
# main
if __name__ == '__main__':
    pass


