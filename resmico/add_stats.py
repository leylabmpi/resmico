import argparse
import json
import os
import logging
import sys

import numpy as np

from resmico import contig_reader
from resmico import models_fl as Models
from resmico.commands import arguments


def get_toc_name(contig_info: Models.ContigInfo, is_chunked: bool, base_dir: str):
    return os.path.join(os.path.dirname(os.path.join(base_dir, contig_info.file)), 'toc' + (
        '_chunked' if is_chunked else '') + '_new')


def get_stats_name(fname: Models.ContigInfo, base_dir: str):
    return os.path.join(os.path.dirname(os.path.join(base_dir, fname)), 'stats')


def main():
    parser = argparse.ArgumentParser(description='Add stats')
    arguments.add_common_args(parser)

    args = parser.parse_args(sys.argv[1:])

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging._nameToLevel[args.log_level.upper()])

    args.features = ['coverage', 'seq_window_perc_gc', 'seq_window_entropy']

    logging.info('Loading contig metadata data...')
    reader = contig_reader.ContigReader(args.feature_files_path, args.features, args.n_procs, args.chunks,
                                        args.no_cython)

    logging.info(f'Loaded metadata for {len(reader.contigs)} contigs. Loading actual contig data')
    if len(reader.contigs) == 0:
        logging.info('Nothing to do')
        exit(0)

    contig_data = reader.read_contigs(reader.contigs, return_raw=True)
    logging.info('All contigs loaded in memory')

    prev_file = ''
    toc_new = None
    toc_header = 'Contig\tLengthBases\tMisassemblyCnt\tSizeBytes\tBreakingPoints\tAvgCoverage\n'
    sum_entropy: float = 0
    sum_gc_percent: float = 0
    sum2_entropy: float = 0
    sum2_gc_percent: float = 0

    for contig_info, contig_data in zip(reader.contigs, contig_data):
        if prev_file != contig_info.file:  # new file starting
            if toc_new is not None:
                toc_new.close()
            toc_new = open(get_toc_name(contig_info, args.chunks, args.feature_files_path), 'w')
            toc_new.write(toc_header)
            # write previous data, if any
            if prev_file != '' and not args.chunks:  # stats are computed on the full data (not the contig chunks)
                stats_file = open(get_stats_name(prev_file, args.feature_files_path))
                stats = json.load(stats_file)
                stats['seq_window_entropy'] = {'sum': sum_entropy, 'sum2': sum2_entropy}
                stats['seq_window_perc_gc'] = {'sum': sum_gc_percent, 'sum2': sum2_gc_percent}
                stats_file = open(get_stats_name(prev_file, args.feature_files_path), 'w')
                json.dump(stats, stats_file, indent=2)
                sum_entropy: float = 0
                sum_gc_percent: float = 0
                sum2_entropy: float = 0
                sum2_gc_percent: float = 0
        prev_file = contig_info.file

        breakpoints = '-' if len(contig_info.breakpoints) == 0 else ",".join(
            str(b[0]) + "-" + str(b[1]) for b in contig_info.breakpoints)
        toc_new.write(f'{contig_info.name}\t{contig_info.length}\t{contig_info.misassembly}\t'
                      f'{contig_info.size_bytes}\t{breakpoints}\t'
                      f'{round(np.average(contig_data["coverage"]), 5)}\n')

        sum_entropy += np.sum(contig_data['seq_window_entropy'])
        sum_gc_percent += np.sum(contig_data['seq_window_perc_gc'])
        sum2_entropy += np.sum(contig_data['seq_window_entropy'] ** 2)
        sum2_gc_percent += np.sum(contig_data['seq_window_perc_gc'] ** 2)

        # write leftover data
        stats_file = open(get_stats_name(prev_file, args.feature_files_path))
        stats = json.load(stats_file)
        stats['seq_window_entropy'] = {'sum': sum_entropy, 'sum2': sum2_entropy}
        stats['seq_window_perc_gc'] = {'sum': sum_gc_percent, 'sum2': sum2_gc_percent}
        stats_file = open(get_stats_name(prev_file, args.feature_files_path), 'w')
        json.dump(stats, stats_file, indent=2)


if __name__ == '__main__':
    main()
