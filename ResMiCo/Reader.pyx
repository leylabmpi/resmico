# distutils: language = c++
from libc.stdint cimport uint32_t
from libc.stdint cimport uint16_t
from libc.stdint cimport uint8_t

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
feature_names = ['ref_base', 'coverage', 'num_query_A', 'num_query_C',
                 'num_query_G', 'num_query_T', 'num_SNPs', 'num_discordant'] \
                + float_feature_names \
                + ['num_proper_Match', 'num_orphans_Match', 'num_proper_SNP',
                   'seq_window_perc_gc', 'Extensive_misassembly_by_pos']

cdef extern from 'contig_reader.hpp':
    cdef void read_contig_features(const char *fname, uint32_t offset, uint32_t size_bytes,
                          uint32_t length_bases, uint32_t num_features,
                          uint16_t bytes_per_base, uint8_t *feature_mask,
                          uint8_t *feature_sizes_bytes, char **features)


cdef read_contig_cpp(str file_name, uint32_t length, uint32_t offset, uint32_t size, uint8_t[:] feature_mask):
    cdef uint32_t[2] lengths = {1, length}
    np_ref_base = np.empty([lengths[feature_mask[0]]], dtype = np.uint8)
    cdef char[:] ref_base = np_ref_base.view(np.int8)

    np_coverage = np.empty([lengths[feature_mask[0]]], dtype = np.uint16)
    cdef char[:] coverage = np_coverage.view(np.int8)

    np_num_query_A = np.empty([lengths[feature_mask[1]]], dtype=np.uint16)
    cdef char[:] num_query_A = np_num_query_A.view(np.int8)
    np_num_query_C = np.empty([lengths[feature_mask[2]]], dtype=np.uint16)
    cdef char[:] num_query_C = np_num_query_C.view(np.int8)
    np_num_query_G = np.empty([lengths[feature_mask[3]]], dtype=np.uint16)
    cdef char[:] num_query_G = np_num_query_G.view(np.int8)
    np_num_query_T = np.empty([lengths[feature_mask[4]]], dtype=np.uint16)
    cdef char[:] num_query_T = np_num_query_T.view(np.int8)
    np_num_SNPs = np.empty([lengths[feature_mask[6]]], dtype=np.uint16)
    cdef char[:] num_SNPs = np_num_SNPs.view(np.int8)
    np_num_discordant = np.empty([lengths[feature_mask[7]]], dtype=np.uint16)
    cdef char[:] num_discordant = np_num_discordant.view(np.int8)

    np_min_insert_size_Match = np.empty([lengths[feature_mask[8]]], dtype=np.uint16)
    cdef char[:] min_insert_size_Match = np_min_insert_size_Match.view(np.int8)
    np_mean_insert_size_Match = np.empty([lengths[feature_mask[9]]], dtype=np.float32)
    cdef char[:] mean_insert_size_Match = np_mean_insert_size_Match.view(np.int8)
    np_stdev_insert_size_Match = np.empty([lengths[feature_mask[10]]], dtype=np.float32)
    cdef char[:] stdev_insert_size_Match = np_stdev_insert_size_Match.view(np.int8)
    np_max_insert_size_Match = np.empty([lengths[feature_mask[11]]], dtype=np.uint16)
    cdef char[:] max_insert_size_Match = np_max_insert_size_Match.view(np.int8)

    np_min_mapq_Match = np.empty([lengths[feature_mask[12]]], dtype=np.uint8)
    cdef char[:] min_mapq_Match = np_min_mapq_Match.view(np.int8)
    np_mean_mapq_Match = np.empty([lengths[feature_mask[13]]], dtype=np.float32)
    cdef char[:] mean_mapq_Match = np_mean_mapq_Match.view(np.int8)
    np_stdev_mapq_Match = np.empty([lengths[feature_mask[14]]], dtype=np.float32)
    cdef char[:] stdev_mapq_Match = np_stdev_mapq_Match.view(np.int8)
    np_max_mapq_Match = np.empty([lengths[feature_mask[15]]], dtype=np.uint8)
    cdef char[:] max_mapq_Match = np_max_mapq_Match.view(np.int8)

    np_min_al_score_Match = np.empty([lengths[feature_mask[16]]], dtype=np.int8)
    cdef char[:] min_al_score_Match = np_min_al_score_Match.view(np.int8)
    np_mean_al_score_Match = np.empty([lengths[feature_mask[17]]], dtype=np.float32)
    cdef char[:] mean_al_score_Match = np_mean_al_score_Match.view(np.int8)
    np_stdev_al_score_Match = np.empty([lengths[feature_mask[18]]], dtype=np.float32)
    cdef char[:] stdev_al_score_Match = np_stdev_al_score_Match.view(np.int8)
    np_max_al_score_Match = np.empty([lengths[feature_mask[19]]], dtype=np.int8)
    cdef char[:] max_al_score_Match = np_max_al_score_Match.view(np.int8)

    np_num_proper_Match = np.empty([lengths[feature_mask[20]]], dtype=np.uint16)
    cdef char[:] num_proper_Match = np_num_proper_Match.view(np.int8)
    np_num_orphans_Match = np.empty([lengths[feature_mask[21]]], dtype=np.uint16)
    cdef char[:] num_orphans_Match = np_num_orphans_Match.view(np.int8)
    np_num_proper_SNP = np.empty([lengths[feature_mask[22]]], dtype=np.uint16)
    cdef char[:] num_proper_SNP = np_num_proper_SNP.view(np.int8)

    np_seq_window_perc_gc = np.empty([lengths[feature_mask[23]]], dtype=np.float32)
    cdef char[:] seq_window_perc_gc = np_seq_window_perc_gc.view(np.int8)
    np_extensive_misassembly_by_pos = np.empty([lengths[feature_mask[24]]], dtype=np.uint8)
    cdef char[:] extensive_misassembly_by_pos = np_extensive_misassembly_by_pos.view(np.int8)

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
    cdef uint8_t bytes_per_base = 58 # this must be equal to the sum of feature_size_bytes
    read_contig_features(file_name.encode('utf-8'), offset, size, length, 25, bytes_per_base, &feature_mask[0],
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


def read_contig_py(str file_name, int length, int offset, int size, py_feature_names):
    py_feature_mask = [1 if feature in py_feature_names else 0 for feature in feature_names]
    cdef uint8_t[:] feature_mask = np.array(py_feature_mask, dtype=np.uint8)
    result = read_contig_cpp(file_name, length, offset, size, feature_mask)
    return {key: result[key] for key in py_feature_names}
