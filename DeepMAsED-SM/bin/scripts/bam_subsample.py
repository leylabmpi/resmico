#!/usr/bin/env python
# import
## batteries
import sys,os
import copy
import argparse
import logging
import itertools
import statistics
from math import log
from random import shuffle
from functools import partial
from collections import deque, defaultdict
from multiprocessing import Pool
## 3rd party
import pysam

# logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass

# argparse
desc = 'Per-contig read subsampling from a BAM'
epi = """DESCRIPTION:
Subsample reads so that each contig (reference)
does not exceed a coverage threashold.
"""
parser = argparse.ArgumentParser(description=desc,
                                 epilog=epi,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('bam_file', metavar='bam_file', type=str,
                    help='bam (or sam) file')
parser.add_argument('fasta_file', metavar='fasta_file', type=str,
                    help='Reference sequences for the bam (sam) file')
parser.add_argument('-m', '--max-coverage', type=float, default=20.0,
                    help='Max per-contig coverage')
parser.add_argument('-o', '--output', type=str, default='subsampled.bam',
                    help='Output BAM file name')
parser.add_argument('--version', action='version', version='0.0.1')

# functions
def main(args):
    bam = pysam.AlignmentFile(args.bam_file)
    fasta = pysam.FastaFile(args.fasta_file) 
    output = pysam.AlignmentFile(args.output, 'wb', template=bam)
    # getting contig lengths
    logging.info('Getting contig lengths...')
    contig_lens = {}
    for contig in bam.references:
        contig_lens[contig] = len(fasta.fetch(contig))
    # per contig: subsample down to max coverage
    logging.info('Subsampling reads...')
    contig_cov = {contig : 0 for contig in bam.references}
    for i,read in enumerate(bam.fetch()):
        # if cov hit, skipping read
        if contig_cov[read.reference_name] >= args.max_coverage:
            continue
        # contig length
        try:
            ref_len = float(contig_lens[read.reference_name])
        except KeyError:
            msg = 'Cannot find contig: {}'
            raise KeyError(msg.format(read.reference_name))
        # keeping read
        contig_cov[read.reference_name] = contig_cov[read.reference_name] + \
                                          read.query_length / ref_len
        output.write(read)
        # status
        if (i+1) % 100000 == 0:
            logging.info('  Reads processed: {}'.format(i+1))
    # finish up
    bam.close()
    fasta.close()
    output.close()
    # status
    print('\t'.join(['contig', 'contig_length', 'coverage']))
    for contig,cov in sorted(contig_cov.items(), key=lambda x: -x[1]):
        print('\t'.join([contig, str(contig_lens[contig]), str(cov)]))
        
                        
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

