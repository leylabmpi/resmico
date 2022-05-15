# distutils: language = c++
cimport cython
from libc.stdlib cimport malloc, free

from array import array
from libc.stdint cimport uint32_t
from libc.stdint cimport uint64_t
from libc.stdint cimport uint16_t
from libc.stdint cimport uint8_t

import numpy as np
from cython.parallel import parallel, prange
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from typing import List

cdef extern from "Python.h":
    char *PyBytes_AS_STRING(object)

# define the available feature, their size in the feature file, their size as numpy arrays, and if they need to
# be normalized
DEF N_FEATURES = 25
feature_tuples = [('ref_base', np.uint8, np.float32, False),  # because we use one-hot encoding, so 4 bytes
                  ('coverage', np.uint16, np.float32, False),  # converting to float32 bc it's going to be normalized
                  ('num_query_A', np.uint16, np.float32, False),
                  ('num_query_C', np.uint16, np.float32, False),
                  ('num_query_G', np.uint16, np.float32, False),
                  ('num_query_T', np.uint16, np.float32, False),
                  ('num_SNPs', np.uint16, np.float32, False),
                  ('num_discordant', np.uint16, np.float32, False),
                  ('min_insert_size_Match', np.uint16, np.float32, True),
                  ('mean_insert_size_Match', np.float32, np.float32, True),
                  ('stdev_insert_size_Match', np.float32, np.float32, True),
                  ('max_insert_size_Match', np.uint16, np.float32, True),
                  ('min_mapq_Match', np.uint8, np.float32, True),
                  ('mean_mapq_Match', np.float32, np.float32, True),
                  ('stdev_mapq_Match', np.float32, np.float32, True),
                  ('max_mapq_Match', np.uint8, np.float32, True),
                  ('min_al_score_Match', np.int8, np.float32, True),
                  ('mean_al_score_Match', np.float32, np.float32, True),
                  ('stdev_al_score_Match', np.float32, np.float32, True),
                  ('max_al_score_Match', np.int8, np.float32, True),
                  ('num_proper_Match', np.uint16, np.float32, False),
                  ('num_orphans_Match', np.uint16, np.float32, False),
                  ('num_proper_SNP', np.uint16, np.float32, False),
                  ('seq_window_perc_gc', np.float32, np.float32, True),
                  ('seq_window_entropy', np.float32, np.float32, True),
                   ]

float_feature_names = [f[0] for f in feature_tuples if f[3]]

feature_names = [f[0] for f in feature_tuples]
feature_types = [f[1] for f in feature_tuples]
feature_np_types = [f[2] for f in feature_tuples]
feature_sizes = [np.dtype(feature_types[i]).itemsize for i in range(N_FEATURES)]
feature_np_sizes = [np.dtype(feature_types[i]).itemsize for i in range(N_FEATURES)]

bytes_per_base = sum(feature_sizes)

assert len(feature_names) == N_FEATURES
assert len(feature_sizes) == N_FEATURES
assert len(feature_types) == N_FEATURES

cdef extern from 'contig_reader.hpp':
    cdef void read_contig_features(const char *fname, uint64_t offset, uint32_t size_bytes,
                          uint32_t length_bases, uint32_t num_features,
                          uint16_t b_per_base, uint8_t *feature_mask,
                          uint8_t *feature_sizes_bytes, char **features) nogil
    cdef void read_contig_features_buf(const char *fname, uint64_t offset, uint32_t size_bytes,
                                   uint32_t length_bases, uint32_t num_features,
                                   uint16_t b_per_base, uint8_t *feature_mask,
                                   uint8_t *feature_sizes_bytes, char *buf, char ** features, int thread) nogil


@cython.boundscheck(False)
cdef read_contig_cpp(const char* file_name, uint32_t length, uint64_t offset, uint32_t size, uint8_t[:] feature_mask):
    cdef uint32_t[2] lengths = {1, length}
    cdef char[:] view
    np_data = [None] * N_FEATURES
    cdef char *all_data[25]
    for i in range(N_FEATURES):
        np_data[i] = np.empty([lengths[feature_mask[i]]], dtype = feature_types[i])
        view = np_data[i].view(np.int8)
        all_data[i] = &view[0]

    cdef uint8_t feature_sizes_bytes[N_FEATURES]
    feature_sizes_bytes[:] = feature_sizes

    read_contig_features(file_name, offset, size, length, N_FEATURES, bytes_per_base, &feature_mask[0],
                         &feature_sizes_bytes[0], &all_data[0])

    result = {feature_name: data for feature_name, data in zip(feature_names, np_data)}
    return result


def read_contig_py(str file_name, int length, int offset, int size, py_feature_names):
    py_feature_mask = [1 if feature in py_feature_names else 0 for feature in feature_names]
    cdef uint8_t[:] feature_mask = np.array(py_feature_mask, dtype=np.uint8)
    result = read_contig_cpp(file_name.encode('utf-8'), length, offset, size, feature_mask)
    return {key: result[key] for key in py_feature_names}

# Reads contig features from #file_names and returns a list of {'feature_name', 'feature_data'} dictionaries, for each
# contig. Data is read in parallel (using Cython bindings).
# Parameters:
#   file_names: names of the binary files to read data from, one per contig; file names are not necessarily distinct,
#           since each file contains data for many (hundreds) of contigs
#   py_lengths: the length of each contig
#   py_offsets: the position in the file where the contig data begins
#   py_sizes: the size of data in bytes, for each contig (used to allocate memory in the C code)
#   py_feature_mask: 0/1 mask denoting the features that need to be read
#   num_threads: how many threads to use to read the data
@cython.boundscheck(False)
@cython.wraparound(False)
def read_contigs_py(file_names:List[bytes], py_lengths: List[int],  py_offsets: List[int],  py_sizes: List[int],
                    py_feature_mask: List[int], int num_threads):
    assert len(file_names) == len(py_lengths) == len(py_offsets) == len(py_sizes)
    cdef uint32_t contig_count = len(file_names)
    cdef int max_len = max(py_lengths)
    cdef int[:] lengths = array('i', py_lengths)
    cdef uint64_t[:] offsets = array('L', py_offsets)
    cdef int[:] sizes = array('i', py_sizes)
    cdef uint8_t[:] feature_mask = array('B', py_feature_mask)
    # the buffer used by the C++ code to write the features into; one buffer per contig; for each contig one buffer
    # per feature
    cdef char ***all_data = <char ***> PyMem_Malloc(sizeof(char **) * contig_count)

    py_all_data = [None] * contig_count
    cdef uint32_t[2] arr_len
    cdef char[:] view
    views = [] # keep all views in a list to avoid garbage collection
    for ctg_idx in range(contig_count):
        np_data = [None] * N_FEATURES
        np_data_int8 = [None] * N_FEATURES
        all_data[ctg_idx] = <char **> PyMem_Malloc(sizeof(char **) * N_FEATURES)
        for feat_idx in range(N_FEATURES):
            if feature_mask[feat_idx]:
                np_data[feat_idx] = np.empty(lengths[ctg_idx], dtype=feature_types[feat_idx])
                view = np_data[feat_idx].view(np.int8)
                views.append(view)
                all_data[ctg_idx][feat_idx] = &view[0]
            else:
                all_data[ctg_idx][feat_idx] = NULL
        py_all_data[ctg_idx] = np_data

    cdef uint8_t feature_sizes_bytes[N_FEATURES]
    feature_sizes_bytes[:] = feature_sizes


    cdef char ** c_file_names = <char **>PyMem_Malloc(sizeof(char*) * contig_count)
    for i in range(contig_count):
        c_file_names[i] = PyBytes_AS_STRING(file_names[i])

    # This is the code that is actually parallelized
    cdef Py_ssize_t ctg_idx_c
    cdef uint32_t bytes_per_base_c = bytes_per_base
    cdef char * buf
    with nogil, parallel(num_threads = num_threads):
        # the buffer used by the C++ code to unzip the data (one buffer for each thread)
        buf = <char *> malloc(sizeof(char) * max_len * bytes_per_base_c + 4)
        for ctg_idx_c in prange(contig_count, schedule='guided'):
            read_contig_features_buf(c_file_names[ctg_idx_c], offsets[ctg_idx_c], sizes[ctg_idx_c], lengths[ctg_idx_c],
                                 N_FEATURES, bytes_per_base_c, &feature_mask[0], &feature_sizes_bytes[0],
                                 buf, all_data[ctg_idx_c], cython.parallel.threadid())
        free(buf)

    results = []
    for ctg_idx in range(contig_count):
        results.append({feature_name: data for feature_name, data in zip(feature_names, py_all_data[ctg_idx])
                        if data is not None })

    for feat_idx in range(contig_count):
        PyMem_Free(all_data[feat_idx])
    PyMem_Free(all_data)
    PyMem_Free(c_file_names)
    return results
