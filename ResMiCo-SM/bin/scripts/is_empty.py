#!/usr/bin/env python
from __future__ import print_function
import sys,os
import argparse
import gzip

desc = 'Determine if file is empty'
epi = """DESCRIPTION:
Returns '1' if empty; else returns '0'.
Can handle gzip'ed and uncompressed files.
"""
parser = argparse.ArgumentParser(description=desc,
                                 epilog=epi,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('input_file', metavar='input_file', type=str,
                    help='Input file')
parser.add_argument('-m', '--min-size-bytes', type=int, default=50,
                    help='Minimum number of bytes (default: $(default)s)')
parser.add_argument('--version', action='version', version='0.0.1')


def main(args):
    # simple check of file size
    if os.stat(args.input_file).st_size > args.min_size_bytes:
        return 0
    else:
        return 1
    
if __name__ == '__main__':
    args = parser.parse_args()
    ret = main(args)
    print(ret)
