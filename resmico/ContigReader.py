from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import Future
import csv
import gzip
import json
import logging
import math
import mmap
import os
from pathlib import Path
from timeit import default_timer as timer
import struct

import numpy as np

from resmico import Reader


def _replace_with_nan(data, feature_name, v):
    """Replaces all elements in arr that are equal to v with np.nan"""
    if feature_name not in data:
        return
    arr = data[feature_name]
    arr[arr == v] = np.nan


def _read_feature(file: gzip.GzipFile, data, feature_name: str, bytes: int, dtype, feature_names: list[str],
                  normalize_by: int = 1):
    if feature_name not in feature_names:
        file.seek(bytes, os.SEEK_CUR)
        return
    # the astype is needed even if dtype already is np.float32, otherwise the array is read-only
    data[feature_name] = np.frombuffer(file.read(bytes), dtype=dtype).astype(np.float32)
    if normalize_by != 1:
        data[feature_name] /= normalize_by


def _to_float_and_normalize(features, result, feature_name, normalize_by):
    if feature_name in features:
        result[feature_name] = features[feature_name].astype(np.float32)
        result[feature_name] /= normalize_by


def _to_float_and_nan(features, result, feature_name, nan_value):
    if feature_name in features:
        result[feature_name] = features[feature_name].astype(np.float32)
        result[feature_name][result[feature_name] == nan_value] = np.nan


def _assign(result, orig, feature_names: list[str]):
    for feature_name in feature_names:
        if feature_name in orig:
            result[feature_name] = orig[feature_name]


def _post_process_features(features):
    result = {}
    if 'ref_base' in features:
        ref_base = features['ref_base']
        result['ref_base_A'] = np.where(ref_base == 65, 1, 0)
        result['ref_base_C'] = np.where(ref_base == 67, 1, 0)
        result['ref_base_G'] = np.where(ref_base == 71, 1, 0)
        result['ref_base_T'] = np.where(ref_base == 84, 1, 0)

    _to_float_and_normalize(features, result, 'num_query_A', 10000)
    _to_float_and_normalize(features, result, 'num_query_C', 10000)
    _to_float_and_normalize(features, result, 'num_query_G', 10000)
    _to_float_and_normalize(features, result, 'num_query_T', 10000)
    _to_float_and_normalize(features, result, 'num_SNPs', 10000)
    _to_float_and_normalize(features, result, 'num_discordant', 10000)
    _to_float_and_normalize(features, result, 'num_proper_Match', 10000)
    _to_float_and_normalize(features, result, 'num_orphans_Match', 10000)
    _to_float_and_normalize(features, result, 'num_proper_SNP', 10000)

    _to_float_and_nan(features, result, 'min_insert_size_Match', 65535)
    _to_float_and_nan(features, result, 'max_insert_size_Match', 65535)
    _to_float_and_nan(features, result, 'min_mapq_Match', 255)
    _to_float_and_nan(features, result, 'max_mapq_Match', 255)
    _to_float_and_nan(features, result, 'min_al_score_Match', 127)
    _to_float_and_nan(features, result, 'max_al_score_Match', 127)

    _assign(result, features, [
        'coverage',
        'mean_insert_size_Match',
        'stdev_insert_size_Match',
        'mean_mapq_Match',
        'stdev_mapq_Match',
        'mean_al_score_Match',
        'stdev_al_score_Match',
        'seq_window_perc_gc',
        'entropy'])

    return result


def _read_contig_data(input_file, feature_names: list[str]):
    """
    Read a binary gzipped file containing the features for a single contig, as written by bam2feat. Features that don't
    exist are silently ignored.
    Parameters:
         - input_file: the file to read, opened at the correct offset
         - feature_names list of feature names to return (e.g. ['coverage', 'num_discordant', 'min_mapq_Match'])
    Returns:
         - a map from feature name to feature data
    """
    data = {}
    with gzip.open(input_file) as f:
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
        _read_feature(f, data, 'num_query_A', 2 * contig_size, np.uint16, feature_names, 10000)
        _read_feature(f, data, 'num_query_C', 2 * contig_size, np.uint16, feature_names, 10000)
        _read_feature(f, data, 'num_query_G', 2 * contig_size, np.uint16, feature_names, 10000)
        _read_feature(f, data, 'num_query_T', 2 * contig_size, np.uint16, feature_names, 10000)
        _read_feature(f, data, 'num_SNPs', 2 * contig_size, np.uint16, feature_names, 10000)
        _read_feature(f, data, 'num_discordant', 2 * contig_size, np.uint16, feature_names, 10000)

        _read_feature(f, data, 'min_insert_size_Match', 2 * contig_size, np.uint16, feature_names)
        _read_feature(f, data, 'mean_insert_size_Match', 4 * contig_size, np.float32, feature_names)
        _read_feature(f, data, 'stdev_insert_size_Match', 4 * contig_size, np.float32, feature_names)
        _read_feature(f, data, 'max_insert_size_Match', 2 * contig_size, np.uint16, feature_names)
        _replace_with_nan(data, 'min_insert_size_Match', 65535)
        _replace_with_nan(data, 'max_insert_size_Match', 65535)

        _read_feature(f, data, 'min_mapq_Match', contig_size, np.uint8, feature_names)
        _read_feature(f, data, 'mean_mapq_Match', 4 * contig_size, np.float32, feature_names)
        _read_feature(f, data, 'stdev_mapq_Match', 4 * contig_size, np.float32, feature_names)
        _read_feature(f, data, 'max_mapq_Match', contig_size, np.uint8, feature_names)
        _replace_with_nan(data, 'min_mapq_Match', 255)
        _replace_with_nan(data, 'max_mapq_Match', 255)

        _read_feature(f, data, 'min_al_score_Match', contig_size, np.int8, feature_names)
        _read_feature(f, data, 'mean_al_score_Match', 4 * contig_size, np.float32, feature_names)
        _read_feature(f, data, 'stdev_al_score_Match', 4 * contig_size, np.float32, feature_names)
        _read_feature(f, data, 'max_al_score_Match', contig_size, np.int8, feature_names)
        _replace_with_nan(data, 'min_al_score_Match', 127)
        _replace_with_nan(data, 'max_al_score_Match', 127)

        _read_feature(f, data, 'num_proper_Match', 2 * contig_size, np.uint16, feature_names, 10000)
        _read_feature(f, data, 'num_orphans_Match', 2 * contig_size, np.uint16, feature_names, 10000)
        _read_feature(f, data, 'num_proper_SNP', 2 * contig_size, np.uint16, feature_names, 10000)
        _read_feature(f, data, 'seq_window_perc_gc', 4 * contig_size, np.float32, feature_names)
        _read_feature(f, data, 'entropy', 4 * contig_size, np.float32, feature_names)
    return data


class ContigInfo:
    """
    Contains metadata about a single contig.
    """

    def __init__(self, name: str, file_name: str, length: int, offset: int, size_bytes: int, misassembly_count: int,
                 breakpoints: list[(int, int)]):
        self.name: str = name
        self.file: str = file_name
        self.length: int = length
        self.offset: int = offset
        self.size_bytes: int = size_bytes
        self.misassembly: int = misassembly_count
        self.features: dict[str:np.array] = {}
        self.breakpoints = breakpoints


class ContigReader:
    """
    Reads contig data from binary files written by ResMiCo-SM.
    """

    def __init__(self, input_dir: str, feature_names: list[str], process_count: int, is_chunked: bool,
                 no_cython: bool = False, stats_file: str = ''):
        """
        Arguments:
            - input_dir: location on disk where the feature data is stored
            - feature_names: feature names to use in training
            - process_count: number of processes to use for loading data in parallel
            - is_chunked: if True, we are loading data from contig chunks (toc_chunked rather than toc)
            - no_cython: whether to read data from disk using pure Python or using Cython bindings
            - stats_file: if present, specifies a stats file to read the statistics for each feature from
        """
        # means and stdevs are a map from feature name to the mean and standard deviation precomputed for that
        # feature across *all* contigs, stored as a tuple
        self.means: dict[str, float] = {}
        self.stdevs: dict[str, float] = {}
        # a list of ContigInfo objects with metadata about all contigs found in #input_dir
        self.contigs: list[ContigInfo] = []

        self.feature_names = feature_names
        self.process_count = process_count
        self.is_chunked = is_chunked
        self.no_cython = no_cython

        # just temp attributes for measuring performance
        self.normalize_time = 0
        self.read_time = 0

        self.feature_mask: list[int] = [1 if feature in feature_names else 0 for feature in Reader.feature_names]

        logging.info('Looking for stats/toc files...')
        file_list = [str(f) for f in list(Path(input_dir).rglob("**/stats"))]
        logging.info(f'Processing {len(file_list)} stats/toc files found in {input_dir} ...')
        if not file_list:
            logging.info('Nothing to do.')
            exit(0)

        if stats_file == '':
            logging.info('Computing global means and standard deviations...')
            self._compute_mean_stdev(file_list)
        else:
            logging.info(f'Loading feature means and standard deviations from {stats_file}')
            means_stdevs = json.load(open(stats_file))
            self.means, self.stdevs = means_stdevs['means'], means_stdevs['stdevs']

        self._load_contigs_metadata(file_list)
        out_file = os.path.join(input_dir, 'stats.json')
        json.dump({'means': self.means, 'stdevs': self.stdevs}, open(out_file, 'w'))
        logging.info(f'Means and stdevs saved to: {out_file}')

    def __len__(self):
        return len(self.contigs)

    def read_contigs(self, contig_infos: list[ContigInfo]):
        """
        Reads the features for the given contig_infos from file and returns the result in a list of len(contig_infos)
        dictionaries of {'feature_name', feature_data}
        """
        start = timer()
        self.normalize_time = 0
        self.read_time = 0
        result = []
        if self.no_cython:
            for contig_data in map(self._read_and_normalize, contig_infos):
                result.append(contig_data)
        else:
            file_names: list[bytes] = []
            lengths: list[int] = []
            offsets: list[int] = []
            sizes: list[int] = []
            for c in contig_infos:
                file_names.append(c.file.encode('utf-8'))
                lengths.append(c.length)
                offsets.append(c.offset)
                sizes.append(c.size_bytes)

            features_raw = Reader.read_contigs_py(file_names, lengths, offsets, sizes, self.feature_mask,
                                                  self.process_count)
            for f in features_raw:
                features = _post_process_features(f)
                self._normalize(features)
                result.append(features)
        logging.debug(f'Contigs read in {(timer() - start):5.2f}s; read: {self.read_time:5.2f}s '
                      f'normalize: {self.normalize_time:5.2f}s')
        return result

    def _compute_mean_stdev(self, file_list):
        mean_count = 0
        stddev_count = 0
        metrics = ['insert_size', 'mapq', 'al_score']  # insert size, mapping quality, alignment quality
        metric_types = ['min', 'mean', 'stdev', 'max']

        for metric in metrics:
            for mtype in metric_types:
                feature_name = f'{mtype}_{metric}_Match'
                self.means[feature_name] = 0
                self.stdevs[feature_name] = 0

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
                # the count of non-NaN position is different (lower) for the stdev_* features (because computing
                # the standard deviation requires at least coverage 2, while for max/mean/min coverage=1 is sufficient)
                cnt = stddev_count if mtype == 'stdev' else mean_count
                var = cnt * self.stdevs[feature_name] - self.means[feature_name] ** 2
                if -0.1 < var < 0:  # fix small rounding errors that may lead to negative values
                    var = 0
                self.stdevs[feature_name] = math.sqrt(var) / cnt
                self.means[feature_name] /= cnt
        # print the computed values
        header = ''
        values = ''
        for metric in metrics:
            for mtype in metric_types:
                feature_name = f'{mtype}_{metric}_Match'
                header += f'{mtype}_{metric}_Match\t\t\t'
                values += f'{self.means[feature_name]:.2f}/{self.stdevs[feature_name]}\t\t\t'

        separator = '_' * 300
        logging.info(
            'Computed global means and stdevs:\n' + separator + '\n' + header + '\n' + values + '\n' + separator)

    def _load_contigs_metadata(self, file_list):
        contig_count = 0
        total_len = 0
        breakpoint_hist = np.zeros(50)
        for fname in file_list:
            toc_file = fname[:-len('stats')] + 'toc'
            contig_fname = fname[:-len('stats')] + 'features_binary'
            if self.is_chunked:
                contig_fname += '_chunked'
                toc_file += '_chunked'
            offset = 0
            with open(toc_file) as f:
                rd = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
                next(rd)  # skip CSV header: Assembler, Contig_name, MissassembleCount, ContigLen
                for row in rd:
                    size_bytes = int(row[3])
                    # the fields in row are: name, length (bases), misassembly_count, size_bytes, breakpoints
                    breakpoints = []
                    if len(row) == 5:  # breakpoints is present; TODO: remove this if once all datasets have it
                        if row[4] != '-':
                            all_breakpoints = row[4].split(',')
                            for break_point in all_breakpoints:
                                start_stop = break_point.split('-')
                                breakpoints.append((int(start_stop[0]), int(start_stop[1])))
                                breakpoint_hist[min(49, (int(start_stop[0]) + int(start_stop[1])) // 200)] += 1

                    contig_info = ContigInfo(row[0], contig_fname, int(row[1]), offset, size_bytes, int(row[2]),
                                             breakpoints)
                    total_len += contig_info.length
                    self.contigs.append(contig_info)
                    offset += size_bytes
                    contig_count += 1

        logging.info(
            f'Found {contig_count} contigs, {total_len} total length, '
            f'memory needed (assuming fraq-neg=1) {total_len * Reader.bytes_per_base / 1e9:6.2f}GB')
        logging.info(f'Breakpoint location histogram: {breakpoint_hist}')

    def read_file(self, fname):
        toc_file = fname[:-len('stats')] + 'toc'
        contig_fname = fname[:-len('stats')] + 'features_binary'
        if self.is_chunked:
            contig_fname += '_chunked'
            toc_file += '_chunked'
        offset = 0
        result = []
        with open(toc_file) as f, open(contig_fname) as binary_file:
            binary_file = open(contig_fname, 'rb')
            # memory-map the file, size 0 means whole file
            mm = mmap.mmap(binary_file.fileno(), 0, access=mmap.ACCESS_READ)

            rd = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            next(rd)  # skip CSV header: Assembler, Contig_name, MissassembleCount, ContigLen
            for row in rd:
                size_bytes = int(row[3])
                # the gzip reader reads ahead and messes up the current position, so we need to re-seek
                mm.seek(offset)
                features = _read_contig_data(mm, self.feature_names)
                self._normalize(features)
                result.append(features)
                offset += size_bytes
        return result

    def _read_and_normalize(self, contig_info: ContigInfo):
        """
        Reads and normalizes the float features present in features using the precomputed means and standard deviations
        in #mean_stdev
        Parameters:
            contig_info: the metadata of the contig to be loaded
        """
        if contig_info.features:  # data is cached in memory, simply return cached value
            return contig_info.features
        start = timer()

        # features is a map from feature name (e.g. 'coverage') to a numpy array containing the feature

        input_file = open(contig_info.file, mode='rb')
        input_file.seek(contig_info.offset)
        features = _read_contig_data(input_file, self.feature_names)

        self.read_time += (timer() - start)
        self._normalize(features)
        return features

    def _normalize(self, features):
        start = timer()
        for feature_name in Reader.float_feature_names:
            if feature_name not in features:
                continue
            if feature_name not in self.means or feature_name not in self.stdevs:
                logging.warning('Could not find mean/standard deviation for feature: {fname}. Skipping normalization')
                continue
            # if np.isnan(sum(features[feature_name])):
            #     logging.warning(f'Exception for feature {feature_name}')
            #     print(features[feature_name])
            features[feature_name] -= self.means[feature_name]

            if features[feature_name].dtype != np.float32:  # we can do the division in place
                features[feature_name] = features[feature_name].astype(np.float32)
            features[feature_name] /= self.stdevs[feature_name]
            # replace NANs with 0 (the new mean)
            nan_pos = np.isnan(features[feature_name])
            features[feature_name][nan_pos] = 0
        self.normalize_time += (timer() - start)


if __name__ == '__main__':
    pass
