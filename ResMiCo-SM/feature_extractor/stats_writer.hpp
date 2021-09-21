#pragma once

#include "contig_stats.hpp"
#include "metaquast_parser.hpp"
#include "util/gzstream.hpp"

#include <filesystem>
#include <iostream>
#include <random>
#include <unordered_map>

// the columns that are written to binary files
extern std::vector<std::string> bin_headers;
// all columns (including those written to TSV only)
extern std::vector<std::string> headers;

/** Stores all the stats for a contig, in order to avoid reallocating memory */
struct ContigStats {
    static const uint32_t LEN = 5000;
    ContigStats() { resize(LEN); }
    void resize(uint32_t size);

    std::vector<uint16_t> coverage;
    std::array<std::vector<uint16_t>, 4> n_bases;

    std::vector<uint16_t> num_snps;
    std::vector<uint16_t> num_discordant;

    std::vector<uint16_t> min_insert_size;
    std::vector<uint16_t> max_insert_size;
    std::vector<float> mean_insert_size;
    std::vector<float> std_dev_insert_size;

    std::vector<uint8_t> min_map_qual;
    std::vector<uint8_t> max_map_qual;
    std::vector<float> mean_map_qual;
    std::vector<float> std_dev_map_qual;

    std::vector<int8_t> min_al_score;
    std::vector<int8_t> max_al_score;
    std::vector<float> mean_al_score;
    std::vector<float> std_dev_al_score;

    std::vector<uint16_t> num_proper_match;
    std::vector<uint16_t> num_orphans_match;

    std::vector<uint16_t> num_proper_snp;
    std::vector<float> gc_percent;

    std::vector<uint8_t> misassembly_by_pos;

    uint32_t size() { return num_snps.size(); }
};

/**
 * An item to placed on the wait-queue: it contains the statistics for one contig.
 */
struct QueueItem {
    std::vector<Stats> stats;
    std::string reference_name; // name of reference contig
    std::string reference; // the actual reference contig
};

/**
 * Writes statistics for all the contigs in a BAM alignment file.
 */
class StatsWriter {
  public:
    /**
     * @param out_dir directory where output files (features.tsv.gz, stats, toc, etc. will be written
     * @param chunk_size size of contig chunks created around breakpoints
     * @param breakpoint_offset maximum (random) offset around the breakpoint for the middle of the
     * chunk
     */
    StatsWriter(const std::filesystem::path &out_dir,
                uint32_t chunk_size,
                uint32_t breakpoint_offset);

    ~StatsWriter() {
        tsv_stream.close();
        toc.close();
    }

    /**
     * Write the given #QueueItem to a gziped TSV file and to individually gzipped binary columns.
     * @param contig_stats the contig stats to be written to disk
     * @param assembler the name of the assembler used to create the contig
     * @param mis possibly empty mis-assembly information as detected by metaQUAST for the contig in
     * contig_stats
     * @param binary_stats_file name of the file where the binary stats are going to be written
     */
    void write_stats(QueueItem &&contig_stats,
                     const std::string &assembler,
                     const std::vector<MisassemblyInfo> &mis);

    void write_summary();

  public:
    // exposing some information for testing
    std::vector<int32_t> offsets;

  private:
    ContigStats contig_stats;

    std::filesystem::path out_dir;

    /**
     * The number of bases around a breaking point (a mis-assembly point, as
     * detected by metaQUAST) that are going to be written to train the network on representative
     * data; for contigs that are not misassembled, a random sequence of size chunk_size is picked
     */
    uint32_t chunk_size;

    /**
     * Number of bases around the actual breakpoint where the center of the misassembled contig
     * stats will be
     */
    int32_t breakpoint_offset;

    /** the stream where the tab separated textual data is written (soon deprecated) */
    ogzstream tsv_stream;

    /** "Table of contents" stream, where we write short stats (name, length, is misassembly) about
    each contig that was written using #write_stats.*/
    std::ofstream toc;

    /** Table of contents for the contig chunks. If a contig has multiple breakpoints (rare) one
     * line for each breakpoint will be written */
    std::ofstream toc_chunk;

    /** File containing the features for all the contigs in #toc */
    std::string binary_features;

    /** File containing the features for all the contig chunks in #toc */
    std::string binary_chunk_features;

    std::mt19937 random_engine;

    /** Number of positions where a mean value could be computed (coverage > 0) */
    uint32_t count_mean = 0;

    /** Number of positions where a stddev value could be computed (coverage > 1) */
    uint32_t count_std_dev = 0;

    /** The sums of all the non-nan position for each of the 12 float metrics */
    std::vector<double> sums;

    /** The sums of squares of all the non-nan position for each of the 12 float metrics */
    std::vector<double> sums2;
};
