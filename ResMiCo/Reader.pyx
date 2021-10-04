# distutils: language = c++
cimport cython
from libc.stdint cimport uint32_t
from libc.stdint cimport uint16_t
from libc.stdint cimport uint8_t

import numpy as np
from cython.parallel import prange
from cpython.mem cimport PyMem_Malloc, PyMem_Free

DEF N_FEATURES = 25
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
feature_names = ['ref_base', 'coverage', 'num_query_A', 'num_query_C',
                 'num_query_G', 'num_query_T', 'num_SNPs', 'num_discordant'] \
                + float_feature_names \
                + ['num_proper_Match', 'num_orphans_Match', 'num_proper_SNP',
                   'seq_window_perc_gc', 'Extensive_misassembly_by_pos']

feature_types = [np.uint8, np.uint16,np.uint16,np.uint16,np.uint16,np.uint16,np.uint16,np.uint16,
                 np.uint16, np.float32, np.float32, np.uint16,
                 np.uint8, np.float32, np.float32, np.uint8,
                 np.int8, np.float32, np.float32, np.int8,
                 np.uint16,np.uint16,np.uint16, np.float32, np.uint8]

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


@cython.boundscheck(False)
cdef read_contig_cpp(const char* file_name, uint32_t length, uint32_t offset, uint32_t size, uint8_t[:] feature_mask):
    cdef uint32_t[2] lengths = {1, length}
    np_ref_base = np.empty([lengths[feature_mask[0]]], dtype = np.uint8)
    cdef char[::1] ref_base = np_ref_base.view(np.int8)

    np_coverage = np.empty([lengths[feature_mask[1]]], dtype = np.uint16)
    cdef char[::1] coverage = np_coverage.view(np.int8)

    np_num_query_A = np.empty([lengths[feature_mask[2]]], dtype=np.uint16)
    cdef char[::1] num_query_A = np_num_query_A.view(np.int8)
    np_num_query_C = np.empty([lengths[feature_mask[3]]], dtype=np.uint16)
    cdef char[::1] num_query_C = np_num_query_C.view(np.int8)
    np_num_query_G = np.empty([lengths[feature_mask[4]]], dtype=np.uint16)
    cdef char[::1] num_query_G = np_num_query_G.view(np.int8)
    np_num_query_T = np.empty([lengths[feature_mask[5]]], dtype=np.uint16)
    cdef char[::1] num_query_T = np_num_query_T.view(np.int8)
    np_num_SNPs = np.empty([lengths[feature_mask[6]]], dtype=np.uint16)
    cdef char[::1] num_SNPs = np_num_SNPs.view(np.int8)
    np_num_discordant = np.empty([lengths[feature_mask[7]]], dtype=np.uint16)
    cdef char[::1] num_discordant = np_num_discordant.view(np.int8)

    np_min_insert_size_Match = np.empty([lengths[feature_mask[8]]], dtype=np.uint16)
    cdef char[::1] min_insert_size_Match = np_min_insert_size_Match.view(np.int8)
    np_mean_insert_size_Match = np.empty([lengths[feature_mask[9]]], dtype=np.float32)
    cdef char[::1] mean_insert_size_Match = np_mean_insert_size_Match.view(np.int8)
    np_stdev_insert_size_Match = np.empty([lengths[feature_mask[10]]], dtype=np.float32)
    cdef char[::1] stdev_insert_size_Match = np_stdev_insert_size_Match.view(np.int8)
    np_max_insert_size_Match = np.empty([lengths[feature_mask[11]]], dtype=np.uint16)
    cdef char[::1] max_insert_size_Match = np_max_insert_size_Match.view(np.int8)

    np_min_mapq_Match = np.empty([lengths[feature_mask[12]]], dtype=np.uint8)
    cdef char[::1] min_mapq_Match = np_min_mapq_Match.view(np.int8)
    np_mean_mapq_Match = np.empty([lengths[feature_mask[13]]], dtype=np.float32)
    cdef char[::1] mean_mapq_Match = np_mean_mapq_Match.view(np.int8)
    np_stdev_mapq_Match = np.empty([lengths[feature_mask[14]]], dtype=np.float32)
    cdef char[::1] stdev_mapq_Match = np_stdev_mapq_Match.view(np.int8)
    np_max_mapq_Match = np.empty([lengths[feature_mask[15]]], dtype=np.uint8)
    cdef char[::1] max_mapq_Match = np_max_mapq_Match.view(np.int8)

    np_min_al_score_Match = np.empty([lengths[feature_mask[16]]], dtype=np.int8)
    cdef char[::1] min_al_score_Match = np_min_al_score_Match.view(np.int8)
    np_mean_al_score_Match = np.empty([lengths[feature_mask[17]]], dtype=np.float32)
    cdef char[::1] mean_al_score_Match = np_mean_al_score_Match.view(np.int8)
    np_stdev_al_score_Match = np.empty([lengths[feature_mask[18]]], dtype=np.float32)
    cdef char[::1] stdev_al_score_Match = np_stdev_al_score_Match.view(np.int8)
    np_max_al_score_Match = np.empty([lengths[feature_mask[19]]], dtype=np.int8)
    cdef char[::1] max_al_score_Match = np_max_al_score_Match.view(np.int8)

    np_num_proper_Match = np.empty([lengths[feature_mask[20]]], dtype=np.uint16)
    cdef char[::1] num_proper_Match = np_num_proper_Match.view(np.int8)
    np_num_orphans_Match = np.empty([lengths[feature_mask[21]]], dtype=np.uint16)
    cdef char[::1] num_orphans_Match = np_num_orphans_Match.view(np.int8)
    np_num_proper_SNP = np.empty([lengths[feature_mask[22]]], dtype=np.uint16)
    cdef char[::1] num_proper_SNP = np_num_proper_SNP.view(np.int8)

    np_seq_window_perc_gc = np.empty([lengths[feature_mask[23]]], dtype=np.float32)
    cdef char[::1] seq_window_perc_gc = np_seq_window_perc_gc.view(np.int8)
    np_extensive_misassembly_by_pos = np.empty([lengths[feature_mask[24]]], dtype=np.uint8)
    cdef char[::1] extensive_misassembly_by_pos = np_extensive_misassembly_by_pos.view(np.int8)

    cdef char* feature_data[25]
    feature_data[:] = [
        &ref_base[0],
        &coverage[0],
        &num_query_A[0],
        &num_query_C[0],
        &num_query_G[0],
        &num_query_T[0],
        &num_SNPs[0],
        &num_discordant[0],
        &min_insert_size_Match[0],
        &mean_insert_size_Match[0],
        &stdev_insert_size_Match[0],
        &max_insert_size_Match[0],
        &min_mapq_Match[0],
        &mean_mapq_Match[0],
        &stdev_mapq_Match[0],
        &max_mapq_Match[0],
        &min_al_score_Match[0],
        &mean_al_score_Match[0],
        &stdev_al_score_Match[0],
        &max_al_score_Match[0],
        &num_proper_Match[0],
        &num_orphans_Match[0],
        &num_proper_SNP[0],
        &seq_window_perc_gc[0],
        &extensive_misassembly_by_pos[0]
    ]
    cdef uint8_t feature_sizes_bytes[25]
    feature_sizes_bytes = [1, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 2, 1, 4, 4, 1, 1, 4, 4, 1, 2, 2, 2, 4, 1]

    read_contig_features(file_name, offset, size, length, 25, bytes_per_base, &feature_mask[0],
                         &feature_sizes_bytes[0], &feature_data[0])

    result = {'ref_base': np_ref_base,
              'coverage': np_coverage,
              'num_query_A': np_num_query_A,
              'num_query_C': np_num_query_C,
              'num_query_G': np_num_query_G,
              'num_query_T': np_num_query_T,
              'num_SNPs': np_num_SNPs,
              'num_discordant': np_num_discordant,
              'min_insert_size_Match': np_min_insert_size_Match,
              'mean_insert_size_Match': np_mean_insert_size_Match,
              'stdev_insert_size_Match': np_stdev_insert_size_Match,
              'max_insert_size_Match': np_max_insert_size_Match,
              'min_mapq_Match': np_min_mapq_Match,
              'mean_mapq_Match': np_mean_mapq_Match,
              'stdev_mapq_Match': np_stdev_mapq_Match,
              'max_mapq_Match': np_max_mapq_Match,
              'min_al_score_Match': np_min_al_score_Match,
              'mean_al_score_Match': np_mean_al_score_Match,
              'stdev_al_score_Match': np_stdev_al_score_Match,
              'max_al_score_Match': np_max_al_score_Match,
              'num_proper_Match': np_num_proper_Match,
              'num_orphans_Match': np_num_orphans_Match,
              'num_proper_SNP': np_num_proper_SNP,
              'seq_window_perc_gc': np_seq_window_perc_gc,
              'Extensive_misassembly_by_pos': np_extensive_misassembly_by_pos
            }
    return result

@cython.boundscheck(False)
cdef read_contig_cpp2(const char* file_name, uint32_t length, uint32_t offset, uint32_t size, uint8_t[:] feature_mask):
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
    result = read_contig_cpp2(file_name.encode('utf-8'), length, offset, size, feature_mask)
    return {key: result[key] for key in py_feature_names}

@cython.boundscheck(False)
def read_contigs_py(list[bytes] file_names, list[int] lengths, list[int] offsets, list[int] sizes, py_feature_names, int num_threads):
    assert len(file_names) == len(lengths) == len(offsets) == len(sizes)
    cdef uint32_t contig_count = len(file_names)

    py_feature_mask = [1 if feature in py_feature_names else 0 for feature in feature_names]
    cdef uint8_t[:] feature_mask = np.array(py_feature_mask, dtype=np.uint8)
    cdef char ***all_data = <char ***> PyMem_Malloc(sizeof(char ***) * contig_count)
    cdef uint32_t[2] arr_len
    cdef char[:] view
    for ctg_idx in range(contig_count):
        arr_len = {1, lengths[ctg_idx]}
        np_data = [None] * N_FEATURES
        all_data[ctg_idx] = <char **> PyMem_Malloc(sizeof(char **) * N_FEATURES)
        for feat_idx in range(N_FEATURES):
           np_data[feat_idx] = np.empty([arr_len[feature_mask[feat_idx]]], dtype = feature_types[feat_idx])
           view = np_data[feat_idx].view(np.int8)
           all_data[ctg_idx][feat_idx] = &view[0]

    cdef uint8_t feature_sizes_bytes[N_FEATURES]
    feature_sizes_bytes[:] = feature_sizes


    cdef char ** c_file_names = <char **>PyMem_Malloc(sizeof(char**) * contig_count)
    for i in range(contig_count):
        c_file_names[i] = file_names[i]

    # This is the code that is actually parallelized
    cdef Py_ssize_t ctg_idx_c
    cdef uint32_t bytes_per_base_c = bytes_per_base
    for ctg_idx_c in prange(contig_count, nogil=True, num_threads = num_threads):
        read_contig_features(c_file_names[ctg_idx], offsets[ctg_idx_c], sizes[ctg_idx_c], lengths[ctg_idx_c],
                             N_FEATURES, bytes_per_base_c, &feature_mask[0], &feature_sizes_bytes[0],
                             all_data[ctg_idx_c])

    results = []
    for ctg_idx in range(contig_count):
        results.append({feature_name: data for feature_name, data in zip(feature_names, np_data)})

    for feat_idx in range(contig_count):
        PyMem_Free(all_data[feat_idx])
    PyMem_Free(all_data)
    return results
