import csv
import gzip
import json
import logging
import math
import os
# from multiprocessing import Pool
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
    data[feature_name] = np.frombuffer(file.read(bytes), dtype=dtype).astype(np.float32)
    if normalize_by != 1:
        data[feature_name] /= 10000


def _read_contig_data(feature_file_name: str, offset: int, feature_names: list[str]):
    """
    Read a binary gzipped file containing the features for a single contig, as written by bam2feat. Features that don't
    exist are silently ignored.
    Parameters:
         - feature_file_name: the file to read
         - offset: where in the feature file the data for the contig starts
         - feature_names list of feature names to return (e.g. ['coverage', 'num_discordant', 'min_mapq_Match'])
    Returns:
         - a map from feature name to feature data
    """
    data = {}
    input_file = open(feature_file_name, mode='rb')
    input_file.seek(offset)
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
        _read_feature(f, data, 'Extensive_misassembly_by_pos', contig_size, np.uint8, feature_names)
    # keep desired features only
    result = {f: data[f] for f in feature_names}
    return result


class ContigInfo:
    """
    Contains metadata about a single contig.
    """

    def __init__(self, name: str, file: str, length: int, offset: int, size_bytes: int, misassembly: int):
        self.name = name
        self.file = file
        self.length = length
        self.offset = offset
        self.size_bytes = size_bytes
        self.misassembly = misassembly


class ContigReader:
    """
    Reads contig data from binary files written by ResMiCo-SM.
    """

    def __init__(self, input_dir: str, feature_names: list[str], process_count: int, is_chunked: bool,
                 normalize_stdev: bool = True):
        """
        Arguments:
            - input_dir: location on disk where the feature data is stored
            - feature_names: feature names to use in training
            - process_count: number of processes to use for loading data in parallel
            - is_chunked: if True, we are loading data from contig chunks (toc_chunked rather than toc)
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
        self.normalize_stdev = normalize_stdev

        self._load_contigs_metadata(input_dir)

        # just temp attributes for measuring performance
        self.normalize_time = 0
        self.read_time = 0

    def __len__(self):
        return len(self.contigs)

    def read_contigs(self, contig_infos: list[ContigInfo]):
        """
        Reads the features from the given contig feature files and returns the result in a list of len(contig_files)
        arrays of shape (contig_len, num_features).
        """
        start = timer()
        self.normalize_time = 0
        self.read_time = 0
        # pool = Pool(self.process_count)
        result = []
        # TODO: try using pool.map() instead for larger batch-sizes; for the small test set, not using a pool
        #  is 100x faster (probably bc. mini-batch size is only 6)
        for contig_data in map(self._read_and_normalize, contig_infos):
            result.append(contig_data)
        logging.debug(f'Contigs read in {(timer() - start):5.2f}s; read: {self.read_time:5.2f}s '
                     f'normalize: {self.normalize_time:5.2f}s')
        return result

    def _load_contigs_metadata(self, input_dir):
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

        contig_count = 0
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
                    # the fields in row are: name, length (bases), misassembly_count, size_bytes
                    self.contigs.append(ContigInfo(row[0], contig_fname, int(row[1]), offset, size_bytes, int(row[2])))
                    offset += size_bytes
                    contig_count += 1

        logging.info(f'Found {contig_count} contigs')

    def _read_and_normalize(self, contig_info: ContigInfo):
        """
        Reads and normalizes the float features present in features using the precomputed means and standard deviations
        in #mean_stdev
        Parameters:
            contig_info: the metadata of the contig to be loaded
        """
        start = timer()
        # features is a map from feature name (e.g. 'coverage') to a numpy array containing the feature
        # logging.debug(f'Reading contig {contig_info.name} from {contig_info.file} at offset {contig_info.offset}')
        features = _read_contig_data(contig_info.file, contig_info.offset, self.feature_names)
        self.read_time += (timer() - start)
        start = timer()
        for feature_name in float_feature_names:
            if feature_name not in features:
                continue
            if feature_name not in self.means or feature_name not in self.stdevs:
                logging.warning('Could not find mean/standard deviation for feature: {fname}. Skipping normalization')
                continue
            features[feature_name] -= self.means[feature_name]
            if self.normalize_stdev:
                if features[feature_name].dtype == np.float32:  # we can do the division in place
                    features[feature_name] /= self.stdevs[feature_name]
                else:  # need to create a new floating point numpy array
                    features[feature_name] = features[feature_name] / self.stdevs[feature_name]
            # replace NANs with 0 (the new mean)
            nan_pos = np.isnan(features[feature_name])
            features[feature_name][nan_pos] = 0
        self.normalize_time += (timer() - start)
        return features


if __name__ == '__main__':
    pass
