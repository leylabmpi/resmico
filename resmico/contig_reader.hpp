#pragma once
#include <cstdint>

void read_contig_features(const char *fname, uint64_t offset, uint32_t size_bytes,
                          uint32_t length_bases, uint32_t num_features,
                          uint16_t bytes_per_base, uint8_t *feature_mask,
                          uint8_t *feature_sizes_bytes, char **features);

void read_contig_features_buf(const char *fname, uint64_t offset,
                              uint32_t size_bytes, uint32_t length_bases,
                              uint32_t num_features, uint16_t bytes_per_base,
                              uint8_t *feature_mask,
                              uint8_t *feature_sizes_bytes,
                              char *buf,
                              char **features, int thread);
