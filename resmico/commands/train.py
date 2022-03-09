import argparse
import logging
import sys
from resmico import train_big_data
from resmico import train_binary_data
from resmico.commands import arguments


def parse_args(curr_args=None, subparsers=None):
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

    parser.add_argument('--val-path', default=None, type=str,
                        help='Path to validation data (default: %(default)s)')
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help='Early stopping. Can be used only if val-path provided (default: %(default)s)')
    parser.add_argument('--net-type', default='cnn_resnet', type=str,
                        help='Type of NN: lstm, cnn_globpool, cnn_resnet, cnn_lstm, fixlen_cnn_resnet'
                             ' (default: %(default)s)')
    parser.add_argument('--num-blocks', default=4, type=int,
                        help='Number of residual blocks (3 or 4, 5, 6) (default: %(default)s)')
    parser.add_argument('--filters', default=16, type=int,
                        help='N of filters for first conv layer. Then x2 (default: %(default)s)')
    parser.add_argument('--ker-size', default=5, type=int,
                        help='CNN kernel size (default: %(default)s)')
    parser.add_argument('--n-hid', default=50, type=int,
                        help='N of units in fully connected layers (default: %(default)s)')
    parser.add_argument('--n-conv', default=5, type=int,
                        help='N of conv layers (default: %(default)s)')
    parser.add_argument('--n-fc', default=1, type=int,
                        help='N of fully connected layers (default: %(default)s)')
    parser.add_argument('--n-epochs', default=10, type=int,
                        help='N of training epochs (default: %(default)s)')
    parser.add_argument('--batch-size', default=6, type=int,
                        help='Batch size (default: %(default)s)')
    parser.add_argument('--dropout', default=0, type=float,
                        help='Rate of dropout (default: %(default)s)')
    parser.add_argument('--n-folds', default=-1, type=int,
                        help='How many folds for CV. Use "-1" to skip & pool all data for training (default: %(default)s)')
    parser.add_argument('--lr-init', default=0.001, type=float,
                        help='Initial learning rate')
    parser.add_argument('--fraq-neg', default=1., type=float,
                        help='Portion of samples to keep in overrepresented class (default: %(default)s)')
    parser.add_argument('--cache', dest='cache', action='store_true',
                        help='If set, train+validation data be cached in memory')
    parser.add_argument('--cache-validation', dest='cache_validation', action='store_true',
                        help='If set, the validation data will be cached in memory for quicker access time')
    parser.add_argument('--cache-train', dest='cache_train', action='store_true',
                        help='If set, the train data will be cached in memory for quicker access time')
    parser.add_argument('--log-progress', default=False,
                        help='If enabled, a progressbar will be shown for training/evaluation progress',
                        dest='log_progress', action='store_true')
    parser.add_argument('--num-translations', default=1, type=int,
                        help='How many variations to select for each sample (for positive samples, '
                             'the variation will be around the breaking point)')
    parser.add_argument('--max-translation-bases', default=0, type=int,
                        help='Maximum number of bases to translate around the breaking point'
                             '(i.e. misassembled contigs) that are longer than max-len')
    parser.add_argument('--weight-factor', default=0, type=int,
                        help='Factor by which contigs are weighted based on '
                             'their length. w = min(1, contig_len/weight-factor). 0==no weighting')
    parser.add_argument('--min-contig-len', default=0, type=int,
                        help='Ignore contigs with length smaller than this value')

    arguments.add_common_args(parser)

    # running test args
    if curr_args:
        args = parser.parse_args(curr_args)
        return args

    return parser


def main(args=None):
    if args is None:
        args = parse_args(sys.argv[1:])

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging._nameToLevel[args.log_level.upper()])
    # Main interface
    if args.binary_data:
        train_binary_data.main(args)
    else:
        train_big_data.main(args)


# main
if __name__ == '__main__':
    main()
