import argparse
from pkg_resources import resource_filename

def add_common_args(parser: argparse.ArgumentParser):
    """
    Adds arguments common to both training and evaluation to parser.
    """
    # default stats for n9k-train training dataset
    pkg_stats = resource_filename('resmico', 'model/stats_cov.json')
    parser_g1 = parser.add_argument_group('General arguments')
    parser_g1.add_argument('--feature-files-path', default='.', type=str,
                           help='Path to the feature files produced by ResMiCo-SM (default: %(default)s).\n'
                           '2 options are available:\n'
                           '  1) Provide the base path, and subdirectories will be searched.\n'
                           '  2) Provide a file that lists all stats files.\n'
                           '     Note: associated files (eg., toc files) must be in the same directories.\n')
    parser_g1.add_argument('--feature-file-match', default='', type=str,
                           help='String that paths to feature files must match.\n'
                           'Example: use "0.005" to select file paths containing "0.005".\n'
                           'Use "" to match all paths (default: %(default)s)')
    parser_g1.add_argument('--stats-file', default=pkg_stats,
                           help='File containing the feature means/stdevs of the training set.\n'
                           'Set to an empty string when training on a new data set. \n(default: %(default)s)')
    parser_g1.add_argument('--save-path', default='.', type=str,
                           help='Directory where to save output (default: %(default)s)')
    parser_g1.add_argument('--save-name', default='resmico', type=str,
                           help='Prefix for name in the save_path (default: %(default)s)')
    parser_g1.add_argument('--n-procs', default=1, type=int,
                           help='Number of parallel processes (default: %(default)s)')
    parser_g1.add_argument('--gpu-eval-mem-gb', default=1.0, type=float,
                           help='Amount of GPU memory used for validation data.\n'
                           'The amount will be divided per GPU (default: %(default)s)')
    parser_g1.add_argument('--max-len', default=20000, type=int,
                           help='Max contig length; larger contigs are split (default: %(default)s)')
    parser_g1.add_argument('--min-contig-len', default=1000, type=int,
                           help='Ignore contigs with length smaller than this value (default: %(default)s)')
    parser_g1.add_argument('--min-avg-coverage', default=1.0, type=float,
                           help='Minimum average coverage for a contig to be considered during evaluation\n'
                           'or training (default: %(default)s)')
    parser_g1.add_argument('--features', nargs='+', help='Features used by resmico (default: %(default)s)',
                           default=[
                               'num_query_A',
                               'num_query_C', 
                               'num_query_G', 
                               'num_query_T',
                               'mean_mapq_Match',
                               'stdev_al_score_Match',
                               'mean_al_score_Match',
                               'mean_insert_size_Match',
                               'coverage',
                               'min_al_score_Match',
                               'num_SNPs',
                               'min_insert_size_Match',
                               'num_proper_Match',
                               'num_orphans_Match'])
    parser_g1.add_argument('--val-ind-f', default=None, type=str,
                           help='Validation data indices (default: %(default)s)')
    parser_g1.add_argument('--seed', default=12, type=int,
                           help='Seed used for numpy.random and tf (default: %(default)s)')
    parser_g1.add_argument('--log-level', default='INFO',
                           choices = ['CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                           help='Logging level (default: %(default)s)')
    parser_g1.add_argument('--no-cython', dest='no_cython', action='store_true',
                           help='If set, data is read using pure Python rather than using the Cython bindings\n'
                           '(about 2x slower; only useful for debugging)')
