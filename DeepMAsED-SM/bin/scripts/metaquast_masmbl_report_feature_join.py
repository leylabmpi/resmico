#!/usr/bin/env python
from __future__ import print_function
import sys,os
import re
import argparse
import logging
from collections import OrderedDict
# 3rd party
from intervaltree import Interval, IntervalTree

desc = 'Adding "contigs_report_contigs_filtered.mis_contigs.info" to feature table'
epi = """DESCRIPTION:
Adding per-position mis-assembly info to the bam2feat.py feature table.

Output is written to STDOUT
"""
parser = argparse.ArgumentParser(description=desc,
                                 epilog=epi,
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('fasta_file', metavar='fasta_file', type=str,
                    help='fasta file of all misassembly contigs')
parser.add_argument('report_file', metavar='report_file', type=str,
                    help='"contigs_report_contigs_filtered.mis_contigs.info" file created by metaQUAST')
parser.add_argument('feature_file', metavar='feature_file', type=str,
                    help='features table created by bam2feat.py')
parser.add_argument('--version', action='version', version='0.0.1')

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

def contig_lengths(fasta_file):
    """ Getting lengths of all miassembly contigs
    """ 
    contig_lens = {}
    with open(fasta_file) as inF:
        seq_len = 0
        contig_ID = None
        for line in inF:
            line = line.rstrip()
            if line.startswith('>'):
                # seqID
                line = line.lstrip('>')
                if contig_ID is not None:
                    contig_lens[contig_ID] = seq_len
                    seq_len = 0
                contig_ID = line
            else:
                # sequence
                seq_len += len(line)
        # last sequence
        contig_lens[contig_ID] = seq_len

    return contig_lens

def masmbl_itrees(report_file, contig_lens):
    """ Converting metaQUAST extensive misassembly report file to
    an interval tree.

    Returns:
      {contigID : itree}   # itree[start:end] : [misassembly_type, inverted_positions?, pos1 or pos2?]

    Notes:
      MetaQUAST position info is 1-indexed
    """
    contig_itrees = {}
    with open(report_file) as inF:
        contigID = None
        itree = IntervalTree()
        for line in inF:
            line = line.rstrip()
            if line.startswith('Extensive misassembly'):
                # getting contig len
                try:
                    contig_len = contig_lens[contigID]
                except KeyError:
                    msg = 'The fasta does include the contig: {}'
                    raise KeyError(msg.format(contigID))
                # parsing line
                line = line.split(' between ')
                masmbl_type = line[0].split('(')[1].rstrip(')').strip()
                y = line[1].split(' ')
                ## range 1
                start1 = int(y[0])
                end1 = int(y[1])
                ### inversion? if yes, find forward strand position
                invert1 = False
                if start1 > end1:
                    end1f = contig_len - (start1 - 1)
                    start1f = contig_len - (end1 - 1)
                    start1,end1 = end1f,start1f
                    invert1 = True
                ## range 2
                start2 = int(y[3])
                end2 = int(y[4])
                ### inversion? if yes, find forward strand position
                invert2 = False
                if start2 > end2:
                    end2f = contig_len - (start2 - 1)
                    start2f = contig_len - (end2 - 1)
                    start2,end2 = end2f,start2f
                    invert2 = True
                # adding to interval tree
                itree[start1:(end1+1)] = [masmbl_type, invert1, 1]
                if masmbl_type != 'interspecies translocation':
                    itree[start2:(end2+1)] = [masmbl_type, invert2, 2]
            else:
                # getting the contigID
                if contigID is not None and itree is not None:
                    contig_itrees[contigID] = itree
                contigID = line
                itree = IntervalTree()

    return contig_itrees            

def add_masmbl_info(feature_file, contig_itrees):
    """
    New columns:
      'Extensive_misassembly' : 1 if true, else 0
      'Extensive_misassembly_by_pos' : comma-delim list of misassembies by position
    """
    contigs_joined = 0
    msgK = 'Cannot find mis-assembly contig in the feature table: {}'
    msgV = 'Cannot find mis-assembly contig position in the feature table: {} => {}'    
    with open(feature_file) as inF:
        header = None
        curr_contig = ''
        itree = None
        for line in inF:
            line = line.rstrip().split('\t')
            if header is None:
                header = OrderedDict((x,i) for i,x in enumerate(line))
                header['Extensive_misassembly'] = len(header.keys()) + 1
                header['Extensive_misassembly_by_pos'] = len(header.keys()) + 1
                print('\t'.join(header.keys()))
            else:
                contigID = line[header['contig']]
                contig_pos = int(line[header['position']])
                # new contig in table? itree exists for contig?
                if contigID != curr_contig:                    
                    try:
                        itree = contig_itrees[contigID]
                        ext_masmbl = '1'
                        contigs_joined += 1
                        logging.info('Joining misassembly contig: {}'.format(contigID))
                    except KeyError:
                        itree = None
                        ext_masmbl = '0'
                # contig position in itree?
                if itree is not None:
                    x = list(itree[contig_pos+1])                  
                    try:
                        masmbl = ','.join([y[2][0] for y in x])
                    except IndexError:
                        masmbl = 'None'
                    if masmbl == '':
                        masmbl = 'None'
                else:
                    masmbl = 'None'
                # adding to feature table
                line += [ext_masmbl, masmbl]
                print('\t'.join(line))
                # new 'current' contig
                curr_contig = contigID                
    
def main(args):
    # getting contig lengths
    logging.info('Parsing contig lengths...')
    contig_lens = contig_lengths(args.fasta_file)
    # parsing metaQUAST misassemblies
    logging.info('Parsing metaQUAST misassembly report...')
    contig_itrees = masmbl_itrees(args.report_file, contig_lens)
    # adding info to feature table
    logging.info('Adding per-position misassembly info to the feature table...')
    add_masmbl_info(args.feature_file, contig_itrees)
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
