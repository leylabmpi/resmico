#!/usr/bin/env python
from __future__ import print_function
import sys,os
import argparse
import logging
import shutil

desc = 'symlinking genomes fasta files'
epi = """DESCRIPTION:

"""
parser = argparse.ArgumentParser(description=desc,
                                 epilog=epi,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('genomes_tbl', metavar='genomes_tbl', type=str,
                    help='genomes.tsv file created by MGSIM_genome_rename')
parser.add_argument('abund_tbl', metavar='abund_tbl', type=str,
                    help='abundance table created by MGSIM communities')
parser.add_argument('out_dir', metavar='out_dir', type=str,
                    help='output directory for symlinked genomes')
parser.add_argument('-c', '--copy', action='store_true', default=False,
                    help='Copy instead of symlink? (default: %(default)s)')
parser.add_argument('--version', action='version', version='0.0.1')

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

def read_ref_genome_tbl(genomes_tbl):
    ref_genomes = {}
    with open(genomes_tbl) as inF:
        header = {}
        msgK = 'Cannot find column "{}"'        
        msgV = 'Cannot find value in column "{}"'
        for i,line in enumerate(inF):
            line = line.rstrip().split('\t')
            if i == 0:
                header = {x:i for i,x in enumerate(line)}
            else:
                try:
                    taxon = line[header['Taxon']]
                except KeyError:
                    raise KeyError(msgK.format('Taxon'))
                except IndexError:
                    raise IndexError(msgK.format('Taxon'))
                try:
                    fasta = line[header['Fasta']]
                except KeyError:
                    raise KeyError(msgK.format('Fasta'))
                except IndexError:
                    raise IndexError(msgK.format('Fasta'))
                ref_genomes[taxon] = fasta
    return ref_genomes

def read_abund_tbl(abund_tbl, ref_genomes):
    comm_genomes = {}
    with open(abund_tbl) as inF:
        header = {}
        msgK = 'Cannot find column "{}"'        
        msgV = 'Cannot find value in column "{}"'
        for i,line in enumerate(inF):
            line = line.rstrip().split('\t')
            if i == 0:
                header = {x:i for i,x in enumerate(line)}
            else:
                try:
                    taxon = line[header['Taxon']]
                except KeyError:
                    raise KeyError(msgK.format('Taxon'))
                except IndexError:
                    raise IndexError(msgK.format('Taxon'))
                try:
                    abund = float(line[header['Perc_rel_abund']])
                except KeyError:
                    raise KeyError(msgK.format('Perc_rel_abund'))
                except IndexError:
                    raise IndexError(msgK.format('Perc_rel_abund'))
                if abund <= 0.0:
                    continue
                try:
                    comm_genomes[taxon] = ref_genomes[taxon]
                except KeyError:
                    msg = '"{}" in community but not in all references'
                    raise KeyError(msg.format(taxon))
    return comm_genomes

def symlink_genomes(comm_genomes, out_dir, copy=False):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    print('\t'.join(['Taxon', 'Fasta']))
    for genome,fasta in comm_genomes.items():
        out_file = os.path.join(out_dir, os.path.split(fasta)[1])
        if fasta == out_file:
            raise ValueError('input & output files match: {}'.format(out_file))
        if os.path.islink(out_file) or os.path.isfile(out_file):
            os.unlink(out_file)
        if copy is True:
            shutil.copyfile(fasta, out_file)
        else:
            os.symlink(fasta, out_file)
        logging.info('Symlink created: {} => {}'.format(fasta, out_file))
        print('\t'.join([genome, out_file]))

def main(args):
    # load table of all ref genomes
    ref_genomes = read_ref_genome_tbl(args.genomes_tbl)
    # loading MGSIM genome abundanc table
    comm_genomes = read_abund_tbl(args.abund_tbl, ref_genomes)
    # symlinking comm genomes    
    symlink_genomes(comm_genomes, args.out_dir, args.copy)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
