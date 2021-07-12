#include "contig_stats.hpp"
#include "util/fasta_reader.hpp"
#include "util/logger.hpp"
#include "util/util.hpp"

#include <api/BamReader.h>
#include <gflags/gflags.h>
#include <utils/bamtools_fasta.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <future>
#include <string>

DEFINE_string(bam_file, "", "bam (or sam) file");
DEFINE_string(fasta_file, "", "Reference sequences for the bam (sam) file");
DEFINE_string(assembler, "unknown", "Name of metagenome assembler used to create the contigs");
DEFINE_int32(batches, 100, "Number of contigs batches for parallel processing");
DEFINE_int32(chunks, 50, "No. of bins to process before writing; lower values = lower memory");
DEFINE_int32(procs, 1, "Number of parallel processes");
DEFINE_int32(window, 4, "Sliding window size for sequence entropy & GC content");
DEFINE_bool(short, false, "Short feature list instead of all features?");
DEFINE_bool(debug, false, "Debug mode; just for troubleshooting");

/** Pretty print of the results */
void write_stats(const std::vector<Stats> &stats,
                 const std::string &assembler,
                 const std::string &contig_name,
                 std::mutex *mutex) {
    std::unique_lock<std::mutex> lock(*mutex);

    logger()->info("Writing features...");
    for (uint32_t pos = 0; pos < stats.size(); ++pos) {
        std::cout << assembler << '\t' << contig_name << '\t' << pos << '\t';
        const Stats &s = stats[pos];
        std::cout << s.ref_base << '\t' << s.n_bases[0] << '\t' << s.n_bases[1] << '\t'
                  << s.n_bases[2] << '\t' << s.n_bases[3] << '\t' << s.num_snps() << '\t'
                  << s.coverage() << '\t' << s.n_discord << '\t';
        for (bool match : { false, true }) {
            std::cout << s.s[match].min_i_size << '\t' << s.s[match].mean_i_size << '\t'
                      << s.s[match].std_dev_i_size << '\t' << s.s[match].max_i_size << '\t';
            std::cout << s.s[match].n_proper << '\t' << s.s[match].n_diff_strand << '\t'
                      << s.s[match].n_orphan << '\t' << s.s[match].n_sup << '\t' << s.s[match].n_sec
                      << '\t' << s.s[match].n_discord << '\t';
            std::cout << s.entropy << '\t' << s.gc_percent << std::endl;
        }

        std::cout << std::endl;
    }
}

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

    logger()->info("Using {} threads, {} as assembler, window of size {}", FLAGS_procs,
                   FLAGS_assembler, FLAGS_window);


    // output table header
    std::vector<std::string> H = { "assembler",   "contig",      "position",      "ref_base",
                                   "num_query_A", "num_query_C", "num_query_G",   "num_query_T",
                                   "num_SNPs",    "coverage",    "num_discordant" };

    if (!FLAGS_short) {
        H.insert(H.end(),
                 { "min_insert_size_Match", "mean_insert_size_Match", "stdev_insert_size_Match",
                   "max_insert_size_Match", "min_mapq_Match",         "mean_mapq_Match",
                   "stdev_mapq_Match",      "max_mapq_Match",         "num_proper_Match",
                   "num_diff_strand_Match", "num_orphans_Match",      "num_supplementary_Match",
                   "num_secondary_Match",   "num_discordant_Match",   "min_insert_size_SNP",
                   "mean_insert_size_SNP",  "stdev_insert_size_SNP",  "max_insert_size_SNP",
                   "min_mapq_SNP",          "mean_mapq_SNP",          "stdev_mapq_SNP",
                   "max_mapq_SNP",          "num_proper_SNP",         "num_diff_strand_SNP",
                   "num_orphans_SNP",       "num_supplementary_SNP",  "num_secondary_SNP",
                   "num_discordant_SNP" });
    }
    H.insert(H.end(), { "seq_window_entropy", "seq_window_perc_gc" });

    std::cout << join_vec(H, '\t');

    // Getting contig list
    if (!ends_with(FLAGS_bam_file, ".bam")) {
        logger()->error("Only BAM files supported, given: {}", FLAGS_bam_file);
    }

    BamTools::BamReader reader;
    reader.Open(FLAGS_bam_file);
    std::vector<std::string> contigs(reader.GetReferenceCount());
    for (uint32_t i = 0; i < contigs.size(); ++i) {
        contigs[i] = reader.GetReferenceData()[i].RefName;
    }
    logger()->info("Number of contigs in the bam file: {}", contigs.size());

    // debug (just smallest 10 contigs)
    if (FLAGS_debug) {
        contigs = std::vector(contigs.end() - 10, contigs.end());
    }

    std::vector<std::future<void>> futures;
    std::mutex mutex;
#pragma omp parallel for num_threads(FLAGS_procs)
    for (uint32_t c = 0; c < contigs.size(); ++c) {
        std::vector<Stats> stats = contig_stats(contigs[c], FLAGS_bam_file, FLAGS_fasta_file,
                                                FLAGS_window, FLAGS_short);
        futures.push_back(std::async(std::launch::async, write_stats, stats, FLAGS_assembler,
                                     contigs[c], &mutex));
    }

    // make sure all futures are done, although in theory the destructor of future should block
    for (auto &f : futures) {
        f.get();
    }
}
