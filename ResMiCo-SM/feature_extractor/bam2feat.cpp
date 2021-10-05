#include "contig_stats.hpp"
#include "metaquast_parser.hpp"
#include "stats_writer.hpp"
#include "util/filesystem.hpp"
#include "util/logger.hpp"
#include "util/util.hpp"
#include "util/wait_queue.hpp"

#include <api/BamReader.h>
#include <gflags/gflags.h>
#include <util/gzstream.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <future>
#include <string>

DEFINE_string(bam_file, "", "bam (or sam) file");
DEFINE_string(fasta_file, "", "Reference sequences for the bam (sam) file");
DEFINE_string(misassembly_file, "", "metaQUAST file containing misassembly info");
DEFINE_string(o, "", "Output file");
DEFINE_string(assembler, "unknown", "Name of metagenome assembler used to create the contigs");
DEFINE_int32(procs, 1, "Number of parallel processes");
DEFINE_int32(window, 4, "Sliding window size for sequence entropy & GC content");
DEFINE_bool(short, false, "Short feature list instead of all features?");
DEFINE_bool(debug, false, "Debug mode; just for troubleshooting");
DEFINE_uint32(
        queue_size,
        32,
        "Maximum size of the queue for stats waiting to be written to disk, before blocking.");

DEFINE_uint32(chunk_size, 500, "Contig length used when training on small bad/good chunks");
DEFINE_uint32(breakpoint_max_offset,
              200,
              "Maximum offset (to left or right) around the breaking point used when creating "
              "a chunk");

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_bam_file.empty()) {
        logger()->error("Please specify a BAM file to process via --bam_file");
        std::exit(1);
    }
    if (!std::filesystem::exists(FLAGS_bam_file)) {
        logger()->error("BAM file does not seem to exist (or I can't see it): {}", FLAGS_bam_file);
        std::exit(1);
    }
    if (FLAGS_window > 255) {
        logger()->error("Window size too large {}. Maximum is 255", FLAGS_window);
        std::exit(1);
    }

    if (FLAGS_fasta_file.empty()) {
        logger()->error("Please specify a FASTA reference genome via --fasta_file");
        std::exit(1);
    }
    if (!std::filesystem::exists(FLAGS_fasta_file)) {
        logger()->error("FASTA file does not seem to exist (or I can't see it): {}",
                        FLAGS_fasta_file);
        std::exit(1);
    }

    if (FLAGS_misassembly_file.empty()) {
        logger()->error("Please specify a metaQUAST misassembly file via --misassembly_file");
        std::exit(1);
    }
    if (!std::filesystem::exists(FLAGS_misassembly_file)) {
        logger()->error("metaQUAST misassembly file does not seem to exist (or I can't see it): {}",
                        fLS::FLAGS_misassembly_file);
        std::exit(1);
    }

    if (FLAGS_o.empty()) {
        logger()->error("Please specify an output directory via --o output_directory.");
        std::exit(1);
    }

    if (!std::filesystem::exists(FLAGS_o)) {
        std::error_code ec;
        logger()->info("Creating dirctory: {}", FLAGS_o);
        std::filesystem::create_directories(FLAGS_o);
        if (ec) {
            logger()->error("Could not create output directory '{}'. Bailing out.", FLAGS_o);
            std::exit(1);
        }
    }

    if (!std::filesystem::is_directory(FLAGS_o)) {
        logger()->error("--o must be a directory, not a file {}", FLAGS_o);
        std::exit(1);
    }

    logger()->info("Using {} threads, {} assembler, window of size {}", FLAGS_procs,
                   FLAGS_assembler, FLAGS_window);

    logger()->info("Parsing mis-assembly info...");
    std::unordered_map<std::string, std::vector<MisassemblyInfo>> mi_info
            = parse_misassembly_info(FLAGS_misassembly_file);

    // Getting contig list
    if (!ends_with(FLAGS_bam_file, ".bam")) {
        logger()->error("Only BAM files supported, given: {}", FLAGS_bam_file);
    }

    BamTools::BamReader reader;
    if (!reader.Open(FLAGS_bam_file)) {
        logger()->error("Could not open BAM file (invalid BAM?): {}", FLAGS_bam_file);
        std::exit(1);
    }
    std::vector<std::string> ref_names(reader.GetReferenceCount());
    for (uint32_t i = 0; i < ref_names.size(); ++i) {
        ref_names[i] = reader.GetReferenceData()[i].RefName;
    }
    logger()->info("Number of contigs in the bam file: {}", ref_names.size());

    // debug (just smallest 10 contigs)
    if (FLAGS_debug) {
        ref_names = std::vector(ref_names.end() - 10, ref_names.end());
    }

    std::vector<std::future<void>> futures;
    std::mutex mutex;

    util::WaitQueue<QueueItem> wq(32);

    StatsWriter stats_writer(FLAGS_o, FLAGS_chunk_size, FLAGS_breakpoint_max_offset);

    std::thread t([&] {
        for (;;) {
            QueueItem stats;
            if (!wq.pop_back(&stats)) {
                break;
            }
            std::vector<MisassemblyInfo> mis = mi_info[stats.reference_name];
            stats_writer.write_stats(std::move(stats), FLAGS_assembler, mis);
        }
    });

#pragma omp parallel for num_threads(FLAGS_procs)
    for (uint32_t c = 0; c < ref_names.size(); ++c) {
        const std::string reference_seq = get_sequence(FLAGS_fasta_file, ref_names[c]);
        std::vector<Stats> stats = contig_stats(ref_names[c], reference_seq, FLAGS_bam_file,
                                                FLAGS_window, FLAGS_short);
        wq.push_front({ std::move(stats), ref_names[c], reference_seq });
    }

    logger()->info("Waiting for pending data to be written to disk...");
    wq.shutdown();
    t.join();
    stats_writer.write_summary();
    logger()->info("All done.");
}
