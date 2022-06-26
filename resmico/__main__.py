#!/usr/bin/env python
import sys
import argparse

from resmico.commands import bam2feat
from resmico.commands import train
from resmico.commands import evaluate
from resmico.commands import filter_contigs


def main(args=None):
    """Main entry point for application
    """
    if args is None:
        args = sys.argv[1:]

    desc = 'ResMiCo: mis-assembly detection with deep learning'
    epi = """DESCRIPTION:
    Usage: resmico <subcommand> <subcommand_params>
    subcommand is one of: bam2feat, train, evaluate, filter
    Example: resmico train -h

    For general info, see https://github.com/leylabmpi/resmico/
    """
    parser = argparse.ArgumentParser(description=desc,
                                     epilog=epi,
                                     formatter_class=argparse.RawTextHelpFormatter)

    # subparsers
    subparsers = parser.add_subparsers()
    # bam2feat
    b2f = bam2feat.parse_args(subparsers=subparsers)
    b2f.set_defaults(func=bam2feat.main)
    # train
    trainer = train.parse_args(subparsers=subparsers)
    trainer.set_defaults(func=train.main)
    # evaluate
    evaluator = evaluate.parse_args(subparsers=subparsers)
    evaluator.set_defaults(func=evaluate.main)
    # filter
    filter_cntgs = filter_contigs.parse_args(subparsers=subparsers)
    filter_cntgs.set_defaults(func=filter_contigs.main)    

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
