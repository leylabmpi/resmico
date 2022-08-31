import csv
import gzip
import json
import logging
import math
import mmap
import os
import statistics
import struct

from glob import glob
from timeit import default_timer as timer
from typing import Dict, List, Tuple

import numpy as np
from resmico import reader


def _replace_with_nan(data, feature_name, v):
    """Replaces all elements in arr that are equal to v with np.nan"""
    if feature_name not in data:
        return
    arr = data[feature_name]
    arr[arr == v] = np.nan


def _read_feature(file: gzip.GzipFile, data, feature_name: str, bytes: int, dtype, feature_names: List[str],
                  normalize_by: int = 1):
    if feature_name not in feature_names:
        file.seek(bytes, os.SEEK_CUR)
        return
    # `.astype()` is needed even if dtype already is np.float32, otherwise the array is read-only
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


def _assign(result, orig, feature_names: List[str]):
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

    # coverage is uint16, but it needs to be cast to flaot32 because it's going to be normalized by mean/stdev later
    if 'coverage' in features:
        result['coverage'] = features['coverage'].astype(np.float32)
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
        'mean_insert_size_Match',
        'stdev_insert_size_Match',
        'mean_mapq_Match',
        'stdev_mapq_Match',
        'mean_al_score_Match',
        'stdev_al_score_Match',
        'seq_window_perc_gc',
        'seq_window_entropy'])

    return result


def _read_contig_data(input_file, feature_names: List[str]):
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
        data['coverage'] = np.frombuffer(f.read(2 * contig_size), dtype=np.uint16).astype(np.float32)
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
        _read_feature(f, data, 'seq_window_entropy', 4 * contig_size, np.float32, feature_names)
    return data


class ContigInfo:
    """
    Contains metadata about a single contig.
    """

    def __init__(self, name: str, file_name: str, length: int, offset: int, size_bytes: int, misassembly_count: int,
                 breakpoints: List[Tuple[int, int]], avg_coverage: float):
        self.name: str = name
        self.file: str = file_name
        self.length: int = length
        self.offset: int = offset
        self.size_bytes: int = size_bytes
        self.misassembly: int = misassembly_count
        self.features: Dict[str:np.array] = {}
        self.breakpoints = breakpoints
        self.avg_coverage = avg_coverage


class ContigReader:
    """
    Reads contig data from binary files written by ResMiCo-SM.
    """

    def __init__(self, input_dirs: str, feature_names: List[str], process_count: int,
                 no_cython: bool = False, stats_file: str = '', min_len: int = 0, min_avg_coverage: int = 0,
                 feature_file_match: str = ''):
        """
        Arguments:
            - input_dirs: location on disk where the feature data is stored
            - feature_names: feature names to use in training
            - process_count: number of processes to use for loading data in parallel
            - no_cython: whether to read data from disk using pure Python or using Cython bindings
            - stats_file: if present, specifies a stats file to read the statistics for each feature from
            - min_len: exclude all contigs shorter than min_len
            - input_dir_match string that the directories for the feature files must match; useful for filtering
              contigs with certain properties, such as sequencing depth, abundance, etc.
        """
        # means and stdevs are a map from feature name to the mean and standard deviation precomputed for that
        # feature across *all* contigs, stored as a tuple
        self.means: Dict[str, float] = {}
        self.stdevs: Dict[str, float] = {}
        # a list of ContigInfo objects with metadata about all contigs found in #input_dir
        self.contigs: List[ContigInfo] = []

        self.feature_names = feature_names
        self.process_count = process_count
        self.no_cython = no_cython
        
        # just temp attributes for measuring performance
        self.normalize_time = 0
        self.read_time = 0
        self.min_len = min_len
        self.min_avg_coverage = min_avg_coverage
        self.feature_file_match = feature_file_match

        self.feature_mask: List[int] = [1 if feature in feature_names else 0 for feature in reader.feature_names]

        # getting feature file paths
        file_list = []
        if os.path.isfile(input_dirs):
            msg = '  Assuming that the feature files are provided in a file as a list'
            logging.info(msg)
            base_dir = os.path.split(input_dirs)[0]
            with open(input_dirs) as inF:
                for line in inF:
                    line = line.rstrip().split(' ')[0]
                    if os.path.isfile(line):
                        file_list.append(line)
                    else:
                        # appending on the path of the file list
                        line = os.path.join(base_dir, line)
                        if os.path.isfile(line):
                            file_list.append(line)
        else:
            logging.info('Looking for stats/toc files...')
            for input_dir in input_dirs.split(','):
                fl = list(glob(os.path.join(input_dir, '**/stats'), recursive=True))
                fx = os.path.join(input_dir, 'stats')
                if os.path.isfile(fx) and not fx in fl:
                    fl.append(fx)
                if feature_file_match:
                    count = len(fl)
                    fl = [f for f in fl if feature_file_match in f]
                    logging.info(
                        f'Filtered for directories matching: {feature_file_match}. {len(file_list)} out of {count} kept')
                file_list.extend(fl)        
                logging.info(f'Processing {len(file_list)} stats/toc files found in {input_dir} ...')        
        if not file_list:
            logging.info('Nothing to do.')
            exit(0)
        elif ',' in input_dirs:
            logging.info(f'A total of {len(file_list)} stats/toc files found')

        # stats.json file
        if stats_file == '':
            logging.info('Computing global means and standard deviations...')
            self._compute_mean_stdev(file_list)
            if ',' not in input_dirs:
                if os.path.isfile(input_dirs):
                    out_file = os.path.join(os.path.split(input_dirs)[0], 'stats.json')
                else:
                    out_file = os.path.join(input_dirs, 'stats.json')
                json.dump({'means': self.means, 'stdevs': self.stdevs}, open(out_file, 'w'), indent=2)
                logging.info(f'Means and stdevs saved to: {out_file}')
            else:
                logging.info('Composed input dir: not writing stats.json')
        else:
            logging.info(f'Loading feature means and standard deviations from {stats_file}')
            means_stdevs = json.load(open(stats_file))
            self.means, self.stdevs = means_stdevs['means'], means_stdevs['stdevs']

        self._load_contigs_metadata(file_list)    
        
    def __len__(self):
        return len(self.contigs)

    def read_contigs(self, contig_infos: List[ContigInfo], return_raw=False):
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
            file_names: List[bytes] = []
            lengths: List[int] = []
            offsets: List[int] = []
            sizes: List[int] = []
            for c in contig_infos:
                file_names.append(c.file.encode('utf-8'))
                lengths.append(c.length)
                offsets.append(c.offset)
                sizes.append(c.size_bytes)

            features_raw = reader.read_contigs_py(file_names, lengths, offsets, sizes, self.feature_mask,
                                                  self.process_count)
            # traverse features for each contig, convert to proper data type and normalize by mean/stdev if needed
            for f in features_raw:
                features = _post_process_features(f)
                if not return_raw:
                    self._normalize(features)
                result.append(features)
        logging.debug(f'Contigs read in {(timer() - start):5.2f}s; read: {self.read_time:5.2f}s '
                      f'normalize: {self.normalize_time:5.2f}s')
        return result

    def _compute_mean_stdev(self, file_list):
        mean_count = 0
        stddev_count = 0
        all_count = 0
        metrics = ['insert_size', 'mapq', 'al_score']  # insert size, mapping quality, alignment quality
        metric_types = ['min', 'mean', 'stdev', 'max']

        # test and see if the new metrics (coverage, gc, entropy) are already in the stats (either because
        # the new bam2feat version was used or because they were added via add_stats.py
        # TODO: remove when all datasets have the new metrics
        has_new_metrics = False
        with open(file_list[0]) as f:
            stats = json.load(f)
            if 'all_count' in stats and 'seq_window_perc_gc' in stats and 'seq_window_entropy' in stats and 'coverage' in stats:
                has_new_metrics = True

        has_new_metrics = False  # TODO: remove
        new_metrics = ['coverage', 'seq_window_entropy', 'seq_window_perc_gc'] if has_new_metrics else []

        for feature_name in reader.float_feature_names + new_metrics:
            self.means[feature_name] = 0
            self.stdevs[feature_name] = 0

        for fname in file_list:
            with open(fname) as f:
                stats = json.load(f)
                if 'all_count' in stats:
                    all_count += stats['all_count']
                mean_count += stats['mean_cnt']
                stddev_count += stats['stdev_cnt']
                for metric in metrics:
                    for mtype in metric_types:
                        feature_name = f'{mtype}_{metric}_Match'
                        self.means[feature_name] += stats[metric]['sum'][mtype]
                        self.stdevs[feature_name] += stats[metric]['sum2'][mtype]
                for metric in new_metrics:
                    self.means[metric] += stats[metric]['sum']
                    self.stdevs[metric] += stats[metric]['sum2']

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
        for feature_name in new_metrics:
            cnt = all_count
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
                header += f'{feature_name}\t\t\t'
                values += f'{self.means[feature_name]:.2f}/{self.stdevs[feature_name]}\t\t\t'
        for feature_name in new_metrics:
            header += f'{feature_name}\t\t\t'
            values += f'{self.means[feature_name]:.2f}/{self.stdevs[feature_name]}\t\t\t'
        separator = '_' * 300
        logging.info(
            'Computed global means and stdevs:\n' + separator + '\n' + header + '\n' + values + '\n' + separator)

    def _load_contigs_metadata(self, file_list):
        contig_count = 0
        contig_count_misassembled = 0
        total_len = 0
        breakpoint_hist = np.zeros(50, np.int32)
        breakpoint_relpos_hist = np.zeros(20, np.int32)
        excluded_count = 0
        contig_lengths = []
        for fname in file_list:
            toc_file = fname[:-len('stats')] + 'toc'
            contig_fname = fname[:-len('stats')] + 'features_binary'
            offset = 0
            with open(toc_file) as f:
#                 logging.info(f'FILE: {toc_file}')
                rd = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
                try: #file is not empty
                    next(rd)  # skip CSV header: Assembler, Contig_name, MissassembleCount, ContigLen
                except:
                    continue
                for row in rd:
                    # the fields in row are: name, length (bases), misassembly_count, size_bytes, breakpoints, coverage
                    contig_name = row[0]
                    contig_len = int(row[1])
                    size_bytes = int(row[3])
                    breakpoints = []
                    if len(row) >= 5:  # breakpoints is present; TODO: remove this once all datasets have it
                        if row[4] != '-':
                            all_breakpoints = row[4].split(',')
                            for break_point in all_breakpoints:
                                start_stop = break_point.split('-')
                                breakpoints.append((int(start_stop[0]), int(start_stop[1])))
                                mid = (int(start_stop[0]) + int(start_stop[1])) // 2
                                # since contigs can be reversed, chose the breakpoint closer to the edge
                                # (or simply duplicate the breakpoint in case of relative position histogram)
                                breakpoint_relpos_hist[int(mid * 20 / contig_len)] += 1
                                breakpoint_relpos_hist[int((contig_len - mid) *20 / contig_len)] += 1
                                if contig_len - mid < mid:
                                    mid = contig_len - mid
                                breakpoint_hist[min(49, mid // 200)] += 1
                    avg_coverage = float(row[5]) if len(row) >= 6 else 100

                    contig_info = ContigInfo(contig_name, contig_fname, contig_len, offset, size_bytes, int(row[2]),
                                             breakpoints, avg_coverage)
                    if contig_info.length >= self.min_len and contig_info.avg_coverage >= self.min_avg_coverage:
                        contig_lengths.append(contig_info.length)
                        total_len += contig_info.length
                        self.contigs.append(contig_info)
                    else:
                        excluded_count += 1
                    offset += size_bytes
                    if contig_info.misassembly:
                        contig_count_misassembled += 1
                    contig_count += 1
        logging.info(
            f'Found {contig_count} contigs, {contig_count_misassembled} misassembled, {excluded_count} excluded, '
            f'{total_len} total length, {statistics.median(contig_lengths)} median length, '
            f'memory needed (assuming fraq-neg=1) {total_len * reader.bytes_per_base / 1e9:6.2f}GB')
        logging.info(f'Breakpoint location histogram: {",".join([str(x) for x in breakpoint_hist])}')
        logging.info(f'Breakpoint relative position histogram: {",".join([str(x) for x in breakpoint_relpos_hist])}')

    def read_file(self, fname):
        toc_file = fname[:-len('stats')] + 'toc'
        contig_fname = fname[:-len('stats')] + 'features_binary'
        offset = 0
        result = []
        with open(toc_file) as f, open(contig_fname) as binary_file:
            binary_file = open(contig_fname, 'rb')
            # memory-map the file, size 0 means whole file
            mm = mmap.mmap(binary_file.fileno(), 0, access=mmap.ACCESS_READ)

            rd = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            try:
                next(rd)  # skip CSV header: Assembler, Contig_name, MissassembleCount, ContigLen
            except:
                return
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
        has_new_metric = False
        if 'coverage' in self.means and 'coverage' in self.stdevs:
            has_new_metric = True
        new_metric = ['coverage'] if has_new_metric else []
        for feature_name in reader.float_feature_names + new_metric:
            if feature_name not in features:
                continue
            if feature_name not in self.means or feature_name not in self.stdevs:
                logging.warning(
                    f'Could not find mean/standard deviation for feature: {feature_name}. Skipping normalization')
                continue
            features[feature_name] -= self.means[feature_name]

            if features[feature_name].dtype != np.float32:  # we can do the division in place
                features[feature_name] = features[feature_name].astype(np.float32)
            if self.stdevs[feature_name] > 1e-3:
                features[feature_name] /= self.stdevs[feature_name]
            else:
                continue
                logging.warning(f'Stdev for {feature_name} is too low ({self.stdevs[feature_name]}). Not normalizing')
            #             replace NANs with 0 (the new mean)
            nan_pos = np.isnan(features[feature_name])
            features[feature_name][nan_pos] = 0
        self.normalize_time += (timer() - start)


if __name__ == '__main__':
    pass

