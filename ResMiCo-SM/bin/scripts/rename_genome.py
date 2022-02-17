#!/usr/bin/env python
from __future__ import print_function
import sys,os
import re
import argparse
import logging
import gzip
from functools import partial


desc = 'Renaming single genome fasta'
epi = """DESCRIPTION:
Renaming genome fasta file based on taxon name.
"""
parser = argparse.ArgumentParser(description=desc,
                                 epilog=epi,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('fasta_file', metavar='fasta_file', type=str,
                    help='Genome fasta file')
parser.add_argument('-l', '--length', type=int, default=1000,
                    help='Contig length cutoff (>=X bp)')
parser.add_argument('--version', action='version', version='0.0.1')

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


def write_seq(inF, seq_header, seq, length_cutoff):
    if seq_header is None or len(seq) < args.length:
        return 0
    print(seq_header)
    print(seq)
    return 1

def format_fasta(old_fasta, length_cutoff):
    if old_fasta.endswith('.gz'):
        _open = partial(gzip.open, mode='rt')
    else:
        _open = open        
    logging.info('Reading fasta file: {}'.format(old_fasta))

    regex = re.compile(r'[^A-Za-z0-9_]+')
    seq_cnt = 0
    seq_header = None
    seq = ''
    with _open(old_fasta) as inF:
        for line in inF:
            line = line.rstrip()
            if line == '':
                continue
            if line.startswith('>'):
                # existing seq
                seq_cnt += write_seq(inF, seq_header, seq, length_cutoff)
                # new seq
                line = regex.sub('_', line.lstrip('>'))
                line = '>{}_CONTIG{}'.format(line, seq_cnt)
                seq_header = line
                seq = ''
            else:
                seq += line
    # last sequence
    seq_cnt += write_seq(inF, seq_header, seq, args.length)
    logging.info(f'No. of sequences written: {seq_cnt}')
            
def main(args):
    format_fasta(args.fasta_file, args.length)

    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
