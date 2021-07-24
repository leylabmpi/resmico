#!/usr/bin/env python
# import
## batteries
import sys,os
import copy
import argparse
import logging
import random
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
parser.add_argument('-M', '--max-insert-size', type=int, default=30000,
                    help='Max insert size')
parser.add_argument('-o', '--output', type=str, default='subsampled.bam',
                    help='Output BAM file name')
parser.add_argument('-S', '--seed', type=str, default=928,
                    help='Randomization seed')
parser.add_argument('--version', action='version', version='0.0.1')

# functions
def main(args):
    random.seed(args.seed)
    bam = pysam.AlignmentFile(args.bam_file)
    fasta = pysam.FastaFile(args.fasta_file) 
    output = pysam.AlignmentFile(args.output, 'wb', template=bam)
    # getting contig lengths
    logging.info('Subsampling reads...')
    contig_cov = {contig : 0 for contig in bam.references}
    for contig in bam.references:
        logging.info('Processing contig: {}'.format(contig))
        contig_len = len(fasta.fetch(contig))
        for read in sorted(bam.fetch(contig), key=lambda k: random.random()):
            # new cov
            added_cov = read.query_length / contig_len
            # if max cov hit, skipping read
            if contig_cov[read.reference_name] + added_cov >= args.max_coverage:
                continue
            # if very large insert size, skipping
            if abs(read.template_length) > args.max_insert_size:
                continue
            # keeping read & tracking added coverage
            contig_cov[read.reference_name] = contig_cov[read.reference_name] + added_cov
            output.write(read)
        logging.info('  Final coverage: {}'.format(round(contig_cov[contig],2)))
                    
    # finish up
    bam.close()
    fasta.close()
    output.close()
    # status
    print('\t'.join(['contig', 'coverage']))
    for contig,cov in sorted(contig_cov.items(), key=lambda x: -x[1]):
        print('\t'.join([contig, str(cov)]))
        
                        
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

