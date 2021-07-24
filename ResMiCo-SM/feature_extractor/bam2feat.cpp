#include "contig_stats.hpp"
#include "util/filesystem.hpp"
#include "util/logger.hpp"
#include "util/util.hpp"
#include "util/wait_queue.hpp"

#include <api/BamReader.h>
#include <gflags/gflags.h>

#include <array>
#include <chrono>
#include <cmath>
#include <future>
#include <string>

DEFINE_string(bam_file, "", "bam (or sam) file");
DEFINE_string(fasta_file, "", "Reference sequences for the bam (sam) file");
DEFINE_string(o, "", "Output file");
DEFINE_string(assembler, "unknown", "Name of metagenome assembler used to create the contigs");
DEFINE_int32(batches, 100, "Number of contigs batches for parallel processing");
DEFINE_int32(chunks, 50, "No. of bins to process before writing; lower values = lower memory");
DEFINE_int32(procs, 1, "Number of parallel processes");
DEFINE_int32(window, 4, "Sliding window size for sequence entropy & GC content");
DEFINE_bool(short, false, "Short feature list instead of all features?");
DEFINE_bool(debug, false, "Debug mode; just for troubleshooting");
DEFINE_uint32(
        queue_size,
        32,
        "Maximum size of the queue for stats waiting to be written to disk, before blocking.");

/** Truncate to 3 decimals */
std::string r3(float v) {
    return std::to_string(static_cast<int>(v * 1000) / 1000) + '.'
            + std::to_string(static_cast<int>(v * 1000) % 1000);
}

template <typename T>
std::string stri(T v) {
    if (v == std::numeric_limits<T>::max()) {
        return "NA";
    }
    return std::to_string(v);
}

struct QueueItem {
    std::vector<Stats> stats;
    std::string reference_name;
    std::string reference;
};

/** Pretty print of the results */
void write_stats(QueueItem &&item, const std::string &assembler, std::ofstream *o) {
    std::ofstream &out = *o;
    logger()->info("Writing features for contig {}...", item.reference_name);
    out.precision(3);
    for (uint32_t pos = 0; pos < item.stats.size(); ++pos) {
        out << assembler << '\t' << item.reference_name << '\t' << pos << '\t';
        const Stats &s = item.stats[pos];
        assert(s.ref_base == 0 || s.ref_base == item.reference[pos]);
        out << item.reference[pos] << '\t' << s.n_bases[0] << '\t' << s.n_bases[1] << '\t'
            << s.n_bases[2] << '\t' << s.n_bases[3] << '\t' << s.num_snps() << '\t' << s.coverage()
            << '\t' << s.n_discord << '\t';
        if (std::isnan(s.mean_i_size)) { // zero coverage, no i_size, no mapping quality
            out << "NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t";
        } else {
            out << stri(s.min_i_size) << '\t' << round2(s.mean_i_size) << '\t'
                << round2(s.std_dev_i_size) << '\t' << stri(s.max_i_size) << '\t';
            out << (int)s.min_map_qual << '\t' << round2(s.mean_map_qual) << '\t'
                << round2(s.std_dev_map_qual) << '\t' << (int)s.max_map_qual << '\t';
        }
        out << s.n_proper_match << '\t' << s.n_orphan << '\t' << s.n_discord << '\t';

        out << s.n_proper_snp << '\t' << s.entropy << '\t' << s.gc_percent << '\n';
        // uncomment and use of py compatibility
        //        if (s.entropy == 0) {
        //            out << "0.0\t";
        //        } else if (s.entropy == 1) {
        //            out << "1.0\t";
        //        } else if (s.entropy == 2) {
        //            out << "2.0\t";
        //        } else {
        //            out << s.entropy << '\t';
        //        }
        //        if (s.gc_percent == 0) {
        //            out << "0.0\n";
        //        } else if (s.gc_percent == 1) {
        //            out << "1.0\n";
        //        } else {
        //            out << s.gc_percent << '\n';
        //        }
    }
    logger()->info("Writing features for contig {} done.", item.reference_name);
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

    if (FLAGS_o.empty()) {
        logger()->error(
                "Please specify an output file via --o output_file (writing to the console is "
                "SLOW)");
        std::exit(1);
    }


    logger()->info("Using {} threads, {} assembler, window of size {}", FLAGS_procs,
                   FLAGS_assembler, FLAGS_window);


    // output table header
    std::vector<std::string> H = { "assembler",   "contig",      "position",      "ref_base",
                                   "num_query_A", "num_query_C", "num_query_G",   "num_query_T",
                                   "num_SNPs",    "coverage",    "num_discordant" };

    if (!FLAGS_short) {
        H.insert(H.end(),
                 {
                         "min_insert_size_Match",
                         "mean_insert_size_Match",
                         "stdev_insert_size_Match",
                         "max_insert_size_Match",
                         "min_mapq_Match",
                         "mean_mapq_Match",
                         "stdev_mapq_Match",
                         "max_mapq_Match",
                         "num_proper_Match",
                         "num_orphans_Match",
                         "num_discordant_Match",
                         "num_proper_SNP",
                 });
    }
    H.insert(H.end(), { "seq_window_entropy", "seq_window_perc_gc" });

    std::ofstream out(FLAGS_o.c_str());
    out << join_vec(H, '\t');

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

    std::thread t([&] {
        for (;;) {
            QueueItem stats;
            if (!wq.pop_back(&stats)) {
                break;
            }
            write_stats(std::move(stats), FLAGS_assembler, &out);
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
    logger()->info("All done.");
}
