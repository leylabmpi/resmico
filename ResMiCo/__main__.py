#!/usr/bin/env python
from __future__ import print_function

__version__ = '0.3.1'

import sys
import argparse

from ResMiCo.Commands import Train
from ResMiCo.Commands import Predict
from ResMiCo.Commands import Evaluate
from ResMiCo.Commands import Preprocess


def main(args=None):
    """Main entry point for application
    """
    if args is None:
        args = sys.argv[1:]

    desc = 'ResMiCo: increasing the quality of metagenome-assembledgenomes with deep learning'
    epi = """DESCRIPTION:
    Usage: ResMiCo <subcommand> <subcommand_params>
    Example: ResMiCo train -h

    For general info, see https://github.com/leylabmpi/ResMiCo/
    """
    parser = argparse.ArgumentParser(description=desc,
                                     epilog=epi,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--version', action='version', version=__version__)

    # subparsers
    subparsers = parser.add_subparsers()
    # preprocess
    preprocess = Preprocess.parse_args(subparsers=subparsers)
    preprocess.set_defaults(func=Preprocess.main)
    # train
    train = Train.parse_args(subparsers=subparsers)
    train.set_defaults(func=Train.main)
    # predict
    predict = Predict.parse_args(subparsers=subparsers)
    predict.set_defaults(func=Predict.main)
    # evaluate
    evaluate = Evaluate.parse_args(subparsers=subparsers)
    evaluate.set_defaults(func=Evaluate.main)

    # parsing args
    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()

    # running subcommands
    if len(vars(args)) > 0:
        args.func(args)
    else:
        parser.parse_args(['--help'])


if __name__ == '__main__':
    main()
