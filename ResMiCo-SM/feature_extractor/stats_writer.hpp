#pragma once

#include "contig_stats.hpp"
#include "metaquast_parser.hpp"
#include "util/gzstream.hpp"

#include <iostream>
#include <unordered_map>

// the columns that are written to binary files
extern std::vector<std::string> bin_headers;
// all columns (including those written to TSV only)
extern std::vector<std::string> headers;

/**
 * Returns a stream corresponding to each header in #headers.
 * @param out_prefix this prefix will be used for the stream names (the suffix will be the
 * header/column name)
 * @return the map of column-name to opened gzipped stream, one for each header, plus one for the
 * "toc" (table of contents) stream.
 */
std::unordered_map<std::string, std::unique_ptr<ogzstream>>
get_streams(const std::string &out_prefix);

/**
 * An item to placed on the wait-queue: it contains the statistics for one contig.
 */
struct QueueItem {
    std::vector<Stats> stats;
    std::string reference_name; // name of reference contig
    std::string reference; // the actual reference contig
};

/**
 * Write the given #QueueItem to a gziped TSV file and to individually gzipped binary columns.
 */
void write_stats(QueueItem &&item,
                 const std::string &assembler,
                 const std::vector<MisassemblyInfo>& mis,
                 std::ofstream *o,
                 std::ofstream *toc,
                 std::unordered_map<std::string, std::unique_ptr<ogzstream>> *bin_streams,
                 uint32_t *count_mean,
                 uint32_t *count_std_dev,
                 std::vector<double> *sums,
                 std::vector<double> *sums2);
