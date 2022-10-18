import argparse
import logging
import sys
from resmico import train_binary_data
from resmico.commands import arguments


def parse_args(curr_args=None, subparsers=None):
    desc = "Train a new model using resmico"
    if subparsers:
        parser = subparsers.add_parser('train', description=desc,
                                       formatter_class=argparse.RawTextHelpFormatter)
    else:
        parser = argparse.ArgumentParser(description=desc,
                                         formatter_class=argparse.RawTextHelpFormatter)
    parser_g1 = parser.add_argument_group('Training-specific arguments')
    parser_g1.add_argument('--early-stop', action='store_true', default=False,
                           help='Use early stopping (default: %(default)s)')
    parser_g1.add_argument('--net-type', default='cnn_resnet_avg', type=str,
                           help='ResMiCo NN type (default: %(default)s)')
    parser_g1.add_argument('--num-blocks', default=4, type=int,
                           help='Number of residual blocks (3 or 4, 5, 6) (default: %(default)s)')
    parser_g1.add_argument('--filters', default=16, type=int,
                           help='N of filters for first conv layer. Then x2 (default: %(default)s)')
    parser_g1.add_argument('--ker-size', default=5, type=int,
                           help='CNN kernel size (default: %(default)s)')
    parser_g1.add_argument('--n-hid', default=50, type=int,
                           help='N of units in fully connected layers (default: %(default)s)')
    parser_g1.add_argument('--n-conv', default=5, type=int,
                           help='N of conv layers (default: %(default)s)')
    parser_g1.add_argument('--n-fc', default=1, type=int,
                           help='N of fully connected layers (default: %(default)s)')
    parser_g1.add_argument('--n-epochs', default=50, type=int,
                           help='N of training epochs (default: %(default)s)')
    parser_g1.add_argument('--batch-size', default=16, type=int,
                           help='Batch size (default: %(default)s)')
    parser_g1.add_argument('--dropout', default=0, type=float,
                           help='Rate of dropout (default: %(default)s)')
    parser_g1.add_argument('--lr-init', default=0.0001, type=float,
                           help='Initial learning rate')
    parser_g1.add_argument('--fraq-neg', default=1., type=float,
                           help='Portion of samples to keep in overrepresented class (default: %(default)s)')
    parser_g1.add_argument('--cache', dest='cache', action='store_true',
                           help='Cache the train & validation data in memory for quicker access (default: %(default)s)')
    parser_g1.add_argument('--cache-validation', dest='cache_validation', action='store_true',
                           help='Cache the validation data in memory for quicker access (default: %(default)s)')
    parser_g1.add_argument('--cache-train', dest='cache_train', action='store_true',
                           help='Cache the train data in memory for quicker access (default: %(default)s)')
    parser_g1.add_argument('--log-progress', default=False,
                           help='Show a progressbar for training/evaluation progress (default: %(default)s)',
                           dest='log_progress', action='store_true')
    parser_g1.add_argument('--num-translations', default=1, type=int,
                           help='How many variations to select for each sample (default: %(default)s)\n'
                           'For positive samples, the variation will be around the breaking point)')
    parser_g1.add_argument('--max-translation-bases', default=0, type=int,
                           help='Maximum number of bases to translate around the breaking point\n'
                           '(i.e. misassembled contigs) that are longer than max-len (default: %(default)s)')
    parser_g1.add_argument('--weight-factor', default=0, type=int,
                           help='Factor by which contigs are weighted based on their length (default: %(default)s).\n'
                           'w = min(1, contig_len/weight-factor). 0==no weighting')
    parser_g1.add_argument('--model-checkpoint', default='', type=str,
                           help='Path to the trained model to continue the training')

    arguments.add_common_args(parser)

    # running test args
    if curr_args:
        args = parser.parse_args(curr_args)
        return args

    return parser


def main(args=None):
    """
    Main interface
    """
    if args is None:
        args = parse_args(sys.argv[1:])
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging._nameToLevel[args.log_level.upper()])
    train_binary_data.main(args)


# main
if __name__ == '__main__':
    main()
