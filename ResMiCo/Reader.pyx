# distutils: language = c++
cimport cython
from libc.stdio cimport printf

from array import array
from libc.stdint cimport uint32_t
from libc.stdint cimport uint16_t
from libc.stdint cimport uint8_t

import numpy as np
from cython.parallel import prange
from cpython.mem cimport PyMem_Malloc, PyMem_Free

DEF N_FEATURES = 25
float_feature_tuples = [('min_insert_size_Match', np.uint16),
                        ('mean_insert_size_Match', np.float32),
                        ('stdev_insert_size_Match', np.float32),
                        ('max_insert_size_Match', np.uint16),
                        ('min_mapq_Match', np.uint8),
                        ('mean_mapq_Match', np.float32),
                        ('stdev_mapq_Match', np.float32),
                        ('max_mapq_Match', np.uint8),
                        ('min_al_score_Match', np.int8),
                        ('mean_al_score_Match', np.float32),
                        ('stdev_al_score_Match', np.float32),
                        ('max_al_score_Match', np.int8), ]
feature_tuples = [('ref_base', np.uint8),
                  ('coverage', np.uint16),
                  ('num_query_A', np.uint16),
                  ('num_query_C', np.uint16),
                  ('num_query_G', np.uint16),
                  ('num_query_T', np.uint16),
                  ('num_SNPs', np.uint16),
                  ('num_discordant', np.uint16)] \
                + float_feature_tuples \
                + [('num_proper_Match', np.uint16),
                   ('num_orphans_Match', np.uint16),
                   ('num_proper_SNP', np.uint16),
                   ('seq_window_perc_gc', np.float32),
                   ('Extensive_misassembly_by_pos', np.uint8)]

float_feature_names = [f[0] for f in float_feature_tuples]

feature_names = [f[0] for f in feature_tuples]
feature_types = [f[1] for f in feature_tuples]
feature_sizes = [np.dtype(feature_types[i]).itemsize for i in range(N_FEATURES)]

bytes_per_base = sum(feature_sizes)

assert len(feature_names) == N_FEATURES
assert len(feature_sizes) == N_FEATURES
assert len(feature_types) == N_FEATURES

cdef extern from 'contig_reader.hpp':
    cdef void read_contig_features(const char *fname, uint32_t offset, uint32_t size_bytes,
                          uint32_t length_bases, uint32_t num_features,
                          uint16_t b_per_base, uint8_t *feature_mask,
                          uint8_t *feature_sizes_bytes, char **features) nogil
    cdef void read_contig_features_buf(const char *fname, uint32_t offset, uint32_t size_bytes,
                                   uint32_t length_bases, uint32_t num_features,
                                   uint16_t b_per_base, uint8_t *feature_mask,
                                   uint8_t *feature_sizes_bytes, char *buf, char ** features, int thread) nogil


@cython.boundscheck(False)
cdef read_contig_cpp(const char* file_name, uint32_t length, uint32_t offset, uint32_t size, uint8_t[:] feature_mask):
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
    cdef uint8_t bytes_per_base = 58 # this must be equal to the sum of feature_size_bytes

    read_contig_features(file_name, offset, size, length, N_FEATURES, bytes_per_base, &feature_mask[0],
                         &feature_sizes_bytes[0], &all_data[0])

    result = {feature_name: data for feature_name, data in zip(feature_names, np_data)}
    return result


def read_contig_py(str file_name, int length, int offset, int size, py_feature_names):
    py_feature_mask = [1 if feature in py_feature_names else 0 for feature in feature_names]
    cdef uint8_t[:] feature_mask = np.array(py_feature_mask, dtype=np.uint8)
    result = read_contig_cpp(file_name.encode('utf-8'), length, offset, size, feature_mask)
    return {key: result[key] for key in py_feature_names}

@cython.boundscheck(False)
@cython.wraparound(False)
def read_contigs_py(file_names:list[bytes], py_lengths: list[int],  py_offsets: list[int],  py_sizes: list[int],
                    py_feature_mask: list[int], int num_threads):
    assert len(file_names) == len(py_lengths) == len(py_offsets) == len(py_sizes)
    cdef uint32_t contig_count = len(file_names)
    cdef int max_len = max(py_lengths)
    cdef int[:] lengths = array('i', py_lengths)
    cdef int[:] offsets = array('i', py_offsets)
    cdef int[:] sizes = array('i', py_sizes)
    cdef uint8_t[:] feature_mask = array('B', py_feature_mask)
    # the buffer used by the C++ code to write the features into; one buffer per contig; for each contig one buffer
    # per feature
    cdef char ***all_data = <char ***> PyMem_Malloc(sizeof(char **) * contig_count)
    # the buffer used by the C++ code to unzip the data; one buffer for each thread
    cdef char **buf = <char **>PyMem_Malloc(sizeof(char *) * num_threads);
    for i in range(num_threads):
        buf[i] = <char *>PyMem_Malloc(sizeof(char) * max_len * bytes_per_base + 4)
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
                np_data[feat_idx] = np.empty(lengths[ctg_idx], dtype = feature_types[feat_idx])
                view = np_data[feat_idx].view(np.int8)
                views.append(view)
                all_data[ctg_idx][feat_idx] = &view[0]
            else:
                all_data[ctg_idx][feat_idx] = NULL
        py_all_data[ctg_idx] = np_data

    cdef uint8_t feature_sizes_bytes[N_FEATURES]
    feature_sizes_bytes[:] = feature_sizes


    cdef char ** c_file_names = <char **>PyMem_Malloc(sizeof(char**) * contig_count)
    for i in range(contig_count):
        c_file_names[i] = file_names[i]

    # This is the code that is actually parallelized
    cdef Py_ssize_t ctg_idx_c
    cdef uint32_t bytes_per_base_c = bytes_per_base
    cdef int thread_id = -1
    for ctg_idx_c in prange(contig_count, nogil=True, num_threads = num_threads):
        thread_id = cython.parallel.threadid()
        printf("Thread ID: %d\n", thread_id)
        read_contig_features_buf(c_file_names[ctg_idx], offsets[ctg_idx_c], sizes[ctg_idx_c], lengths[ctg_idx_c],
                             N_FEATURES, bytes_per_base_c, &feature_mask[0], &feature_sizes_bytes[0],
                             buf[cython.parallel.threadid()], all_data[ctg_idx_c], cython.parallel.threadid())

    results = []
    for ctg_idx in range(contig_count):
        results.append({feature_name: data for feature_name, data in zip(feature_names, py_all_data[ctg_idx])
                        if data is not None })

    for i in range(num_threads):
        PyMem_Free(buf[i])
    for feat_idx in range(contig_count):
        PyMem_Free(all_data[feat_idx])
    PyMem_Free(all_data)
    PyMem_Free(buf)
    return results
