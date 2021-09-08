import argparse
import csv
import gzip
import itertools
import logging
import math
from multiprocessing import Pool
import os
import psutil
from pathlib import Path

import numpy as np


def get_numpy_type(str_type):
    """Returns the numpy type corresponding to str_type"""
    if str_type == 'b':
        return np.bool_
    if str_type == 'i':
        return np.single  # because int types are normalized by coverage, they become float
    if str_type == 'f':
        return np.single


def get_feature_info(fname, features, features_types):
    """
    Returns a list of tuples containing:
     1. column number (position) of each feature in features
     2. the type of each feature
    """
    assert (len(features) == len(features_types))
    f = gzip.open(fname, mode='rt')

    keys = f.readline().split('\t')
    assert keys[0] == 'assembler' and keys[1] == 'contig', 'The first two columns must be "assembler" and "contig"'
    feature_info = [(keys.index(f), tp) for f, tp in zip(features, features_types)]
    return feature_info, keys.index('coverage')


def read_file(fname, feature_info, coverage_pos):
    """
    Reads a zipped tsv file containing tab separated features and places the result for the desired features into
    numpy arrays. For float arrays the sum and sum2 of the values is also computed (to be used for normalization)
    Parameters:
     - fname: the file to read
     - feature_info: list of (position, type) tuples indicating the column number and type of the features that we are
     interested in
     - coverage_pos the position of the 'coverage' column in the file (used to normalize the count features). If
       negative, no normalization by coverage takes place
    """
    logging.info(f'Reading {fname}')
    result = [None] * len(feature_info)

    # read the first row and set the constant fields (e.g. contig name, assembler)
    f = gzip.open(fname, mode='rt')
    f.readline()  # skip header
    rd = csv.reader(f, delimiter='\t')
    row = next(rd)

    for i, (pos, tp) in enumerate(feature_info):
        if tp == 'f' or tp == 'i':
            result[i] = []
        else:  # a constant string value, such as contig name or assembler
            result[i] = row[pos]

    # unread the first row
    f.seek(0)
    f.readline()

    rd = csv.reader(f, delimiter='\t')
    for row in rd:
        coverage = max(1, int(row[coverage_pos])) if coverage_pos > 0 else 1
        for i, (pos, tp) in enumerate(feature_info):
            if tp == 'f':
                # None will be converted to np.nan when we create the Numpy array
                result[i].append(float(row[pos]) if row[pos] != 'NA' else None)
            elif tp == 'i':
                if pos != coverage_pos:
                    result[i].append(int(row[pos]) / coverage)
                else:
                    result[i].append(coverage)

    logging.info(f'Finished reading data into memory')
    for i, (pos, tp) in enumerate(feature_info):
        if tp == 'f':
            np_array = np.array(result[i], dtype=np.single)
            result[i] = (np_array, np.nansum(np_array), np.nansum(np_array ** 2))
        if tp == 'i':
            result[i] = np.array(result[i], dtype=np.single)

    logging.info(f'Finished converting to numpy array')
    return result


def read_file_np(fname, feature_info, coverage_pos):
    """
    Reads a zipped tsv file containing tab separated features and places the result for the desired features into
    numpy arrays. For float arrays the sum and sum2 of the values is also computed (to be used for normalization)
    Parameters:
     - fname: the file to read
     - feature_info: list of (position, type) tuples indicating the column number and type of the features that we are
     interested in
     - coverage_pos the position of the 'coverage' column in the file (used to normalize the count features). If
       negative, no normalization by coverage takes place
    """
    logging.info(f'Reading {fname}')
    result = [None] * len(feature_info)

    # read the first row and set the constant fields (e.g. contig name, assembler)
    f = gzip.open(fname, mode='rt')
    f.readline()  # skip header
    rd = csv.reader(f, delimiter='\t')
    row = next(rd)

    cols_to_read = []
    types_to_read = []
    for i, (pos, tp) in enumerate(feature_info):
        if pos == coverage_pos:
            cov_j = len(cols_to_read)
        if tp == 'f':
            cols_to_read.append(pos)
            types_to_read.append('f4')
        elif tp == 'i':
            cols_to_read.append(pos)
            types_to_read.append('f4')

        else:  # a constant string value, such as contig name or assembler
            result[i] = row[pos]

    # read the entire file module the 's' columns
    if cols_to_read:
        data = np.genfromtxt(fname, delimiter='\t', usecols=cols_to_read, dtype=types_to_read, names=True, unpack=True,
                             loose=True)

    logging.info(f'Finished reading data into memory')
    j = 0
    for i, (pos, tp) in enumerate(feature_info):
        if tp == 'i':
            np.divide(data[j], data[cov_j], data[j], where=data[cov_j] != 0)
            result[i] = data[j]
            j += 1
        if tp == 'f':
            result[i] = (data[j], np.nansum(data[j]), np.nansum(data[j] ** 2))
            j += 1

    logging.info(f'Finished reading {fname}')
    return result


def preprocess(process_count, input_dir, features, feature_types):
    file_list = [str(f) for f in list(Path(input_dir).rglob("*.tsv.gz"))]
    logging.info(f'Processing {len(file_list)} *.tsv.gz files found in {input_dir} ...');
    if not file_list:
        logging.info('Noting to do.')
        exit(0)

    result = {}
    pool = Pool(process_count)
    feature_info, coverage_pos = get_feature_info(file_list[0], features, feature_types)
    params = zip(file_list, itertools.repeat(feature_info), itertools.repeat(coverage_pos))
    for file_data in pool.starmap(read_file, params):
        result[(file_data[0], file_data[1])] = file_data
        process = psutil.Process(os.getpid())
        logging.info(f'Memory used: {process.memory_info().rss // 1e6}MB')

    means = [0.0] * len(feature_info)
    std_devs = [0.0] * len(feature_info)
    counts = [0] * len(feature_info)

    # compute first the sums and the sums of squares for each float (normalizable) feature in *all* files
    for (assembler, contig), contig_data in result.items():
        for i, (pos, tp) in enumerate(feature_info):
            if tp == 'f':
                counts[i] += len(contig_data[i][0]) - np.count_nonzero(np.isnan(contig_data[i][0]))
                means[i] += contig_data[i][1]
                std_devs[i] += contig_data[i][2]

    # compute the mean and std deviation for each feature
    for i, (pos, tp) in enumerate(feature_info):
        if tp == 'f':
            cnt = counts[i]  # number of non-nan positions
            if cnt != 0:
                sum = means[i]
                sum2 = std_devs[i]
                means[i] /= cnt
                std_devs[i] = math.sqrt((cnt * sum2 - sum ** 2) / (cnt ** 2))
            else:
                means[i] = 0
                std_devs[i] = 0

    # normalize each float feature using the computed mean and stddev
    for (assembler, contig), contig_data in result.items():
        for i, (pos, tp) in enumerate(feature_info):
            if tp == 'f':
                contig_data[i] = (contig_data[i][0] - means[i]) / std_devs[i]

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--i', help='Input directory. All .tsv.gz files will be read recursively.', default='./')
    parser.add_argument('--p', help='Number of processes to use for parallelization', default=1, type=int)
    parser.add_argument('--features', help='List of features to pre-process',
                        default=['assembler',
                                 'contig',
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
                                 'num_proper_SNP'
                                 ])
    parser.add_argument('--feature_types',
                        help='Type of each feature in features. s=static,i=int (normalized by coverage),'
                             'f=float (normalized to mean 0 and stddev 1',
                        default=['s', 's', 'i', 'i', 'i', 'i', 'f', 'i', 'i', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f',
                                 'i'])
    args = parser.parse_args()
    assert 'coverage' in args.features, '"overage" must be in --features, because it\'s used for normalization'
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

    all_data = preprocess(args.p, args.i, args.features, args.feature_types)

if __name__ == '__main__':
    pass
