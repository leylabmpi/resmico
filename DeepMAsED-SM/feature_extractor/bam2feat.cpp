#include "contig_stats.hpp"
#include "util/fasta_reader.hpp"
#include "util/gzstream.hpp"
#include "util/logger.hpp"
#include "util/util.hpp"

#include <api/BamReader.h>
#include <gflags/gflags.h>
#include <utils/bamtools_fasta.h>

#include <array>
#include <cmath>
#include "util/filesystem.hpp"
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

/** Truncate to 2 decimals */
std::string r2(float v) {
    if (std::isnan(v)) {
        return "NA";
    }
    return std::to_string(static_cast<int>(std::round(v * 100)) / 100) + '.'
            + std::to_string(static_cast<int>(v * 100) % 100);
}

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

/** Pretty print of the results */
void write_stats(std::vector<Stats> &&stats,
                 const std::string &assembler,
                 const std::string &contig_name,
                 const std::string &contig,
                 ogzstream *o,
                 std::mutex *mutex) {
    std::unique_lock<std::mutex> lock(*mutex);
    ogzstream &out = *o;
    logger()->info("Writing features for contig {}...", contig_name);
    // out.setf(std::ios::fixed,std::ios::floatfield);
    out.precision(3);
    for (uint32_t pos = 0; pos < stats.size(); ++pos) {
        out << assembler << '\t' << contig_name << '\t' << pos << '\t';
        const Stats &s = stats[pos];
        assert(s.ref_base == 0 || s.ref_base == contig[pos]);
        out << contig[pos] << '\t' << s.n_bases[0] << '\t' << s.n_bases[1] << '\t' << s.n_bases[2]
            << '\t' << s.n_bases[3] << '\t' << s.num_snps() << '\t' << s.coverage() << '\t'
            << s.n_discord << '\t';
        for (bool match : { false, true }) {
            if (isnan(s.s[match].mean_i_size)) { // zero coverage, no i_size, no mapping quality
                out << "NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t";
            } else {
                out << stri(s.s[match].min_i_size) << '\t' << r2(s.s[match].mean_i_size) << '\t'
                    << r2(s.s[match].std_dev_i_size) << '\t' << stri(s.s[match].max_i_size) << '\t';
                out << (int)s.s[match].min_map_qual << '\t' << r2(s.s[match].mean_map_qual) << '\t'
                    << r2(s.s[match].std_dev_map_qual) << '\t' << (int)s.s[match].max_map_qual
                    << '\t';
            }
            out << s.s[match].n_proper << '\t' << s.s[match].n_diff_strand << '\t'
                << s.s[match].n_orphan << '\t' << s.s[match].n_sup << '\t' << s.s[match].n_sec
                << '\t' << s.s[match].n_discord << '\t';
        }

        out << s.entropy << '\t' << s.gc_percent << '\n';
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
    logger()->info("Writing features for contig {} done.", contig_name);
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

    ogzstream out(FLAGS_o.c_str());
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

#pragma omp parallel for num_threads(FLAGS_procs)
    for (uint32_t c = 0; c < ref_names.size(); ++c) {
        const std::string reference_seq = get_sequence(FLAGS_fasta_file, ref_names[c]);
        std::vector<Stats> stats = contig_stats(ref_names[c], reference_seq, FLAGS_bam_file,
                                                FLAGS_window, FLAGS_short);
        futures.push_back(std::async(std::launch::async, write_stats, std::move(stats),
                                     FLAGS_assembler, ref_names[c], reference_seq, &out, &mutex));
    }

    // make sure all futures are done, although in theory the destructor of future should block
    for (auto &f : futures) {
        f.get();
    }
}
