#include "contig_reader.hpp"

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>

#include "zlib.h" // declare the external fns -- uses zconf.h, too

int uncompress_data(const char *abSrc, int nLenSrc, uint8_t *abDst,
                    int nLenDst) {
  z_stream zInfo;
  zInfo.total_in = zInfo.avail_in = nLenSrc;
  zInfo.total_out = zInfo.avail_out = nLenDst;
  zInfo.next_in = (uint8_t *)abSrc;
  zInfo.next_out = abDst;
  zInfo.state = 0;
  zInfo.opaque = 0;
  zInfo.adler = 0;
  zInfo.reserved = 0;
  zInfo.data_type = 0;
  zInfo.zalloc = 0;
  zInfo.zfree = 0;

  int nErr, nRet = -1;
  nErr = inflateInit(&zInfo);
  int err = inflateReset2(&zInfo, 31); // zlib function
  if (err == Z_OK && nErr == Z_OK) {
    nErr = inflate(&zInfo, Z_FINISH); // zlib function
    if (nErr == Z_STREAM_END) {
      nRet = zInfo.total_out;
    }
  }
  inflateEnd(&zInfo); // zlib function
  return (nRet);      // -1 or len of output
}

void read_feature(std::ifstream &f, int is_read, char *dest, uint32_t size) {
  if (!is_read) {
    f.seekg(size, std::ios::cur);
    return;
  }
  f.read(dest, size);
}

void read_contig_features(const char *fname, uint64_t offset,
                          uint32_t size_bytes, uint32_t length_bases,
                          uint32_t num_features, uint16_t bytes_per_base,
                          uint8_t *feature_mask, uint8_t *feature_sizes_bytes,
                          char **features) {
  std::ifstream f(fname);
  f.seekg(offset);
  // buffer for compressed data
  std::unique_ptr<char[]> cbuf(new char[size_bytes]);
  f.read(cbuf.get(), size_bytes);
  std::unique_ptr<char[]> buf(new char[length_bases * bytes_per_base + 4]);
  uint32_t bytes_uncompressed = uncompress_data(
      cbuf.get(), size_bytes, reinterpret_cast<uint8_t *>(buf.get()),
      length_bases * bytes_per_base + 4);
  std::ignore = bytes_uncompressed;
  // sanity check: make sure we uncompressed exactly as many bytes as needed to
  // store the contig features
  assert(bytes_uncompressed == length_bases * bytes_per_base + 4);
  uint32_t contig_size;
  std::memcpy(&contig_size, buf.get(), 4);
  //  std::cout << "Read contig of size: " << contig_size << " from " << fname
  //            << " at offset " << offset << std::endl;
  assert(contig_size == length_bases);

  char *ptr = buf.get() + 4;

  for (uint32_t i = 0; i < num_features; ++i) {
    if (feature_mask[i]) {
      std::memcpy(features[i], ptr, length_bases * feature_sizes_bytes[i]);
    }
    ptr += length_bases * feature_sizes_bytes[i];
  }
}

void read_contig_features_buf(const char *fname, uint64_t offset,
                              uint32_t size_bytes, uint32_t length_bases,
                              uint32_t num_features, uint16_t bytes_per_base,
                              uint8_t *feature_mask,
                              uint8_t *feature_sizes_bytes,
                              char *buf,
                              char **features, int thread) {
  std::ifstream f(fname);
  f.seekg(offset);
  // buffer for compressed data
  std::unique_ptr<char[]> cbuf(new char[size_bytes]);
  f.read(cbuf.get(), size_bytes);
  uint32_t bytes_uncompressed = uncompress_data(
      cbuf.get(), size_bytes, reinterpret_cast<uint8_t *>(buf),
      length_bases * bytes_per_base + 4);
  std::ignore = bytes_uncompressed;
  // sanity check: make sure we uncompressed exactly as many bytes as needed to
  // store the contig features
  assert(bytes_uncompressed == length_bases * bytes_per_base + 4);
  uint32_t contig_size;
  std::memcpy(&contig_size, buf, 4);
  //  std::cout << "Read contig of size: " << contig_size << " from " << fname
  //            << " at offset " << offset << std::endl;
  assert(contig_size == length_bases);

  char *ptr = buf + 4;

  for (uint32_t i = 0; i < num_features; ++i) {
    if (feature_mask[i]) {
      std::memcpy(features[i], ptr, length_bases * feature_sizes_bytes[i]);
    }
    ptr += length_bases * feature_sizes_bytes[i];
  }
}

int main() {
  char *data = new char[1071];
  uint8_t feature_mask[] = {1};
  uint8_t feature_sizes[] = {1, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 2, 1,
                             4, 4, 1, 1, 4, 4, 1, 2, 2, 2, 4, 1};
  const char *fname = "/tmp/small/features_binary";
  read_contig_features(fname, 0, 2752, 1071, 1, 58, feature_mask, feature_sizes,
                       &data);
  read_contig_features(fname, 2752, 2994, 1012, 1, 58, feature_mask,
                       feature_sizes, &data);

  return 0;
}
