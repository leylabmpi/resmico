#!/usr/bin/env python
import sys
import argparse

from resmico.commands import train
from resmico.commands import predict
from resmico.commands import evaluate
from resmico.commands import preprocess


def main(args=None):
    """Main entry point for application
    """
    if args is None:
        args = sys.argv[1:]

    desc = 'ResMiCo: mis-assembly detection with deep learning'
    epi = """DESCRIPTION:
    Usage: resmico <subcommand> <subcommand_params>
    subcommand is one of: train, predict, evaluate
    Example: resmico train -h

    For general info, see https://github.com/leylabmpi/resmico/
    """
    parser = argparse.ArgumentParser(description=desc,
                                     epilog=epi,
                                     formatter_class=argparse.RawTextHelpFormatter)

    # subparsers
    subparsers = parser.add_subparsers()
    # preprocess
    preprocessor = preprocess.parse_args(subparsers=subparsers)
    preprocessor.set_defaults(func=preprocess.main)
    # train
    trainer = train.parse_args(subparsers=subparsers)
    trainer.set_defaults(func=train.main)
    # predict
    predictor = predict.parse_args(subparsers=subparsers)
    predictor.set_defaults(func=predict.main)
    # evaluate
    evaluator = evaluate.parse_args(subparsers=subparsers)
    evaluator.set_defaults(func=evaluate.main)

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
