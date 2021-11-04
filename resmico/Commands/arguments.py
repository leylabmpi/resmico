import argparse


def add_common_args(parser: argparse.ArgumentParser):
    """
    Adds arguments common to both training and evaluation to parser.
    """
    parser.add_argument('--binary-data', dest='binary_data', action='store_true',
                        help='If present, read features in binary format rather than the (deprecated) h5 '
                             'files transformed from zipped tsv feature files')
    parser.add_argument('--gpu-eval-mem-gb', default=3.0, type=float,
                        help='Amount of GPU memory used for validation data (amount will be divided per GPU)')
    parser.add_argument('--val-ind-f', default=None, type=str,
                        help='Validation data indices (default: %(default)s)')
    parser.add_argument('--chunks', dest='chunks', action='store_true',
                        help='If set, use the toc_chunked/binary_features_chunked data instead of toc/binary_features')
    parser.add_argument('--log-level', default='INFO',
                        help='Logging level, one of [CRITICAL, FATAL, ERROR, WARNING, INFO, DEBUG]')
    parser.add_argument('--no-cython', dest='no_cython', action='store_true',
                        help='If set, data is read using pure Python rather than using the Cython bindings '
                             '(about 2x slower, only useful for debugging')
    parser.add_argument('--seed', default=12, type=int,
                        help='Seed used for numpy.random and tf (default: %(default)s)')
    parser.add_argument('--n-procs', default=1, type=int,
                        help='Number of parallel processes (default: %(default)s)')
    parser.add_argument('--max-len', default=10000, type=int,
                        help='Max contig len, fixed input for CNN (default: %(default)s)')
    parser.add_argument('--save-path', default='.', type=str,
                        help='Directory where to save output (default: %(default)s)')
    parser.add_argument('--save-name', default='resmico', type=str,
                        help='Prefix for name in the save_path (default: %(default)s)')
    parser.add_argument('--feature-files-path', default='', type=str,
                        help='Path to the feature files produced by Resmico-SM')
    parser.add_argument('--feature-file-table', default='', type=str,
                        help='Table listing feature table files (see DESCRIPTION)')
    parser.add_argument('--technology', default='all-asmbl', type=str,
                        help='Assembler name in the data_path. "all-asmbl" will use all assemblers (default: %(default)s)')
    parser.add_argument('--big-data', action='store_true', default=False,
                        help='Use True if work with large dataset in h5 format')
    parser.add_argument('--features', nargs='+', help='Features to use for training', default=[
        'ref_base',
        'num_query_A',
        'num_query_C',
        'num_query_G',
        'num_query_T',
        'coverage',
        'num_proper_Match',
        'num_orphans_Match',
        'max_insert_size_Match',
        'mean_insert_size_Match',
        'min_insert_size_Match',
        'stdev_insert_size_Match',
        'mean_mapq_Match',
        'min_mapq_Match',
        'stdev_mapq_Match',
        'mean_al_score_Match',
        'min_al_score_Match',
        'stdev_al_score_Match',
        'seq_window_perc_gc',
        'num_proper_SNP',
    ])
