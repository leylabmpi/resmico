import argparse
import csv
import gzip
import json
import logging
import math
from multiprocessing import Pool
from pathlib import Path
from timeit import default_timer as timer
import struct

import numpy as np

float_feature_names = ['min_insert_size_Match',
                       'mean_insert_size_Match',
                       'stdev_insert_size_Match',
                       'max_insert_size_Match',
                       'min_mapq_Match',
                       'mean_mapq_Match',
                       'stdev_mapq_Match',
                       'max_mapq_Match',
                       'min_al_score_Match',
                       'mean_al_score_Match',
                       'stdev_al_score_Match',
                       'max_al_score_Match', ]
feature_names = ['ref_base_A', 'ref_base_C', 'ref_base_G', 'ref_base_T', 'coverage', 'num_query_A', 'num_query_C',
                 'num_query_G', 'num_query_T', 'num_SNPs', 'num_discordant'] \
                + float_feature_names \
                + ['num_proper_SNP', 'seq_window_perc_gc', 'Extensive_misassembly_by_pos']


def _replace_with_nan(arr, v):
    """Replaces all elements in arr that are equal to v with np.nan"""
    arr[arr == v] = np.nan


def _read_contig_data(feature_file_name, feature_names):
    """
    Read a binary gzipped file containing the features for a single contig, as written by bam2feat. Features that don't
    exist are silently ignored.
    Parameters:
         - feature_file_name: the file to read
         - feature_names list of feature names to return (e.g. ['coverage', 'num_discordant', 'min_mapq_Match'])
    Returns:
         - a map from feature name to feature data
    """
    logging.debug(f'Reading {feature_file_name}')
    data = {}
    with gzip.open(feature_file_name, mode='rb') as f:
        contig_size = struct.unpack('I', f.read(4))[0]
        ref_base = np.frombuffer(f.read(contig_size), dtype=np.uint8)
        # create the one-hot encoding for the reference base
        data['ref_base_A'] = np.where(ref_base == 65, 1, 0)
        data['ref_base_C'] = np.where(ref_base == 67, 1, 0)
        data['ref_base_G'] = np.where(ref_base == 71, 1, 0)
        data['ref_base_T'] = np.where(ref_base == 84, 1, 0)
        data['coverage'] = np.frombuffer(f.read(2 * contig_size), dtype=np.uint16)
        # everything is converted to float32, because frombuffer creates an immutable array, so the int values need to
        # be made mutable (in order to convert from fixed point back to float) and the float values need to be copied
        # (in order to make them writeable for normalization)
        data['num_query_A'] = np.frombuffer(f.read(2 * contig_size), dtype=np.uint16).astype(np.float32) / 10000
        data['num_query_C'] = np.frombuffer(f.read(2 * contig_size), dtype=np.uint16).astype(np.float32) / 10000
        data['num_query_G'] = np.frombuffer(f.read(2 * contig_size), dtype=np.uint16).astype(np.float32) / 10000
        data['num_query_T'] = np.frombuffer(f.read(2 * contig_size), dtype=np.uint16).astype(np.float32) / 10000
        data['num_SNPs'] = np.frombuffer(f.read(2 * contig_size), dtype=np.uint16).astype(np.float32) / 10000
        data['num_discordant'] = np.frombuffer(f.read(2 * contig_size), dtype=np.uint16).astype(np.float32) / 10000

        data['min_insert_size_Match'] = np.frombuffer(f.read(2 * contig_size), dtype=np.uint16).astype(np.float32)
        data['mean_insert_size_Match'] = np.frombuffer(f.read(4 * contig_size), dtype=np.float32).astype(np.float32)
        data['stdev_insert_size_Match'] = np.frombuffer(f.read(4 * contig_size), dtype=np.float32).astype(np.float32)
        data['max_insert_size_Match'] = np.frombuffer(f.read(2 * contig_size), dtype=np.uint16).astype(np.float32)
        _replace_with_nan(data['min_insert_size_Match'], 65535)
        _replace_with_nan(data['max_insert_size_Match'], 65535)

        data['min_mapq_Match'] = np.frombuffer(f.read(contig_size), dtype=np.uint8).astype(np.float32)
        data['mean_mapq_Match'] = np.frombuffer(f.read(4 * contig_size), dtype=np.float32).astype(np.float32)
        data['stdev_mapq_Match'] = np.frombuffer(f.read(4 * contig_size), dtype=np.float32).astype(np.float32)
        data['max_mapq_Match'] = np.frombuffer(f.read(contig_size), dtype=np.uint8).astype(np.float32)
        _replace_with_nan(data['min_mapq_Match'], 255)
        _replace_with_nan(data['max_mapq_Match'], 255)

        data['min_al_score_Match'] = np.frombuffer(f.read(contig_size), dtype=np.int8).astype(np.float32)
        data['mean_al_score_Match'] = np.frombuffer(f.read(4 * contig_size), dtype=np.float32).astype(np.float32)
        data['stdev_al_score_Match'] = np.frombuffer(f.read(4 * contig_size), dtype=np.float32).astype(np.float32)
        data['max_al_score_Match'] = np.frombuffer(f.read(contig_size), dtype=np.int8).astype(np.float32)
        _replace_with_nan(data['min_al_score_Match'], 127)
        _replace_with_nan(data['max_al_score_Match'], 127)

        data['num_proper_Match'] = np.frombuffer(f.read(2 * contig_size), dtype=np.uint16).astype(np.float32) / 10000
        data['num_orphans_Match'] = np.frombuffer(f.read(2 * contig_size), dtype=np.uint16).astype(np.float32) / 10000
        data['num_proper_SNP'] = np.frombuffer(f.read(2 * contig_size), dtype=np.uint16).astype(np.float32) / 10000
        data['seq_window_perc_gc'] = np.frombuffer(f.read(4 * contig_size), dtype=np.float32).astype(np.float32)
        data['Extensive_misassembly_by_pos'] = np.frombuffer(f.read(contig_size), dtype=np.uint8).astype(np.float32)
    # keep desired features only
    result = {f: data[f] for f in feature_names}
    return result


class ContigInfo:
    def __init__(self, filename, size: int, misassembly: int):
        self.filename = filename
        self.size = size
        self.misassembly = misassembly


class ContigReader:
    """
    Reads contig data from binary files written by ResMiCo-SM.
    """

    def __init__(self, input_dir, feature_names, process_count):
        # means and stdevs are a map from feature name to the mean and standard deviation precomputed for that
        # feature across *all* contigs, stored as a tuple
        self.means = {}
        self.stdevs = {}
        # a list of ContigInfo objects with metadata about all contigs found in the given directory
        self.contigs = []

        self._compute_global_mean_stdev(input_dir)

        # names of the features we are using for training
        self.feature_names = feature_names

        # number of processes used for parallel loading of feature data
        self.process_count = process_count

    def __len__(self):
        return len(self.contigs)

    def read_contigs(self, contig_files):
        """
        Reads the features from the given contig feature files and returns the result in a list of len(contig_files)
        array of shape (contig_len, num_features).
        """
        start = timer()
        # pool = Pool(self.process_count)
        result = []
        # TODO: try using pool.map() instead for larger datasets; for the small test set, not using a pool
        #  is 100x faster
        for contig_data in map(self._read_and_normalize, contig_files):
            result.append(contig_data)
        logging.debug(f'Contigs read in {(timer() - start):5.2f}s')
        return result

    def _compute_global_mean_stdev(self, input_dir):
        logging.info('Computing global means and standard deviations. Looking for stats/toc files...')
        file_list = [str(f) for f in list(Path(input_dir).rglob("**/stats"))]
        logging.info(f'Processing {len(file_list)} stats/toc files found in {input_dir} ...');
        if not file_list:
            logging.info('Noting to do.')
            exit(0)

        mean_count = 0
        stddev_count = 0
        metrics = ['insert_size', 'mapq', 'al_score']  # insert size, mapping quality, alignment quality
        metric_types = ['min', 'mean', 'stdev', 'max']

        for metric in metrics:
            for mtype in metric_types:
                feature_name = f'{mtype}_{metric}_Match'
                self.means[feature_name] = 0
                self.stdevs[feature_name] = 0

        # TODO: check if parallelization is needed and helps
        for fname in file_list:
            with open(fname) as f:
                stats = json.load(f)
                mean_count += stats['mean_cnt']
                stddev_count += stats['stdev_cnt']
                for metric in metrics:
                    for mtype in metric_types:
                        feature_name = f'{mtype}_{metric}_Match'
                        self.means[feature_name] += stats[metric]['sum'][mtype]
                        self.stdevs[feature_name] += stats[metric]['sum2'][mtype]

        for metric in metrics:
            for mtype in metric_types:
                feature_name = f'{mtype}_{metric}_Match'
                # the count of non-nan position is different (lower) for the stdev_* features (because computing
                # the standard deviation requires at least coverage 2, while for max/mean/min coverage=1 is sufficient)
                cnt = stddev_count if mtype == 'stdev' else mean_count
                var = cnt * self.stdevs[feature_name] - self.means[feature_name] ** 2
                if -0.1 < var < 0:  # fix small rounding errors that may lead to negative values
                    var = 0
                self.stdevs[feature_name] = math.sqrt(var) / (cnt ** 2)
                self.means[feature_name] /= cnt
        logging.info('Computed global means and stdevs:')
        # print the computed values
        header = ''
        values = ''
        for metric in metrics:
            for mtype in metric_types:
                feature_name = f'{mtype}_{metric}_Match'
                header += f'{mtype}_{metric}_Match\t\t\t'
                values += f'{self.means[feature_name]:.2f}/{self.stdevs[feature_name]}\t\t\t'

        print('_' * 300 + '\n', header, '\n', values, '\n', '_' * 300)

        contig_count = 0
        for fname in file_list:
            toc_file = fname[:-len('stats')] + 'toc'
            contig_info = []
            contig_prefix = fname[:-len('stats')] + 'contig_stats/'
            with open(toc_file) as f:
                rd = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
                next(rd)  # skip CSV header: Assembler, Contig_name, MissassembleCount, ContigLen
                for row in rd:
                    self.contigs.append(ContigInfo(contig_prefix + row[1] + '.gz', int(row[2]), int(row[3])))
                    contig_count += 1

        logging.info(f'Found {contig_count} contigs')

    def _read_and_normalize(self, file_name):
        """
        Reads and normalizes the float features present in features using the precomputed means and standard deviations
        in #mean_stdev
        Parameters:
            file_name: the file name to read and normalize
        """

        # features is a map from feature name (e.g. 'coverage') to a numpy array containing the feature
        features = _read_contig_data(file_name, self.feature_names)
        for feature_name in float_feature_names:
            if feature_name not in features:
                continue
            if feature_name not in self.means or feature_name not in self.stdevs:
                logging.warning('Could not find mean/standard deviation for feature: {fname}. Skipping normalization')
                continue
            features[feature_name] -= self.means[feature_name]
            if features[feature_name].dtype == np.float32:  # we can do the division in place
                features[feature_name] /= self.stdevs[feature_name]
            else:  # need to create a new floating point numpy array
                features[feature_name] = features[feature_name] / self.stdevs[feature_name]
            # replace NANs with 0 (the new mean)
            nan_pos = np.isnan(features[feature_name])
            features[feature_name][nan_pos] = 0
        return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--i', help='Input directory. All .tsv.gz files will be read recursively.', default='./')
    parser.add_argument('--p', help='Number of processes to use for parallelization', default=1, type=int)
    parser.add_argument('--features', help='List of features to pre-process',
                        default=[
                            'ref_base',
                            'num_query_A',
                            'num_query_C',
                            'num_query_G',
                            'num_query_T',
                            'coverage',
                            'num_proper_Match',
                            'num_orphans_Match',
                            'max_insert_size_Match',
                            'mean_insert_size_Match',
                            'min_insert_size_Match',
                            'stdev_insert_size_Match',
                            'mean_mapq_Match',
                            'min_mapq_Match',
                            'stdev_mapq_Match',
                            'seq_window_perc_gc',
                            'num_proper_SNP',
                            'Extensive_misassembly_by_pos',
                        ])

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

    reader = ContigReader(args.i, args.features, args.p)
    contigs_data = reader.read_contigs(reader.contigs)

if __name__ == '__main__':
    pass
