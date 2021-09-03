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
#include <cstddef>
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

/**
 * An item to placed on the waitqueue: it contains the statistics for one contig.
 */
struct QueueItem {
    std::vector<Stats> stats;
    std::string reference_name;
    std::string reference;
};

inline void write_value(uint16_t v, uint16_t normalize_by, std::ofstream &out) {
    static uint16_t MAX_16 = std::numeric_limits<uint16_t>::max();
    assert(v <= normalize_by || v == MAX_16);
    uint16_t to_write = v == MAX_16 ? MAX_16 : static_cast<uint16_t>((v * 10000.) / normalize_by);
    out.write(reinterpret_cast<const char *>(&to_write), 2);
}

inline void write_float_value(float v, std::ofstream &out) {
    assert(std::isnan(v) || v <= std::numeric_limits<int16_t>::max());
    int16_t to_write
            = isnan(v) ? std::numeric_limits<int16_t>::max() : static_cast<int16_t>(v * 100);
    out.write(reinterpret_cast<const char *>(&to_write), 2);
}

/** Pretty print of the results */
void write_stats(QueueItem &&item,
                 const std::string &assembler,
                 std::ofstream *o,
                 std::unordered_map<std::string, std::unique_ptr<std::ofstream>> &streams,
                 uint32_t *count_mean,
                 uint32_t *count_std_dev,
                 std::vector<double> *sums,
                 std::vector<double> *sums2) {
    std::ofstream &out = *o;
    logger()->info("Writing features for contig {}...", item.reference_name);
    out.precision(3);

    *streams["toc"] << assembler << '\t' << item.reference_name << '\t' << item.stats.size()
                    << std::endl;
    for (uint32_t pos = 0; pos < item.stats.size(); ++pos) {
        const Stats &s = item.stats[pos];

        std::cout << pos << '\t' << std::endl;
        //        uint16_t v = s.coverage();
        //        streams["coverage"]->write(reinterpret_cast<char *>(&v), 2);
        //        write_value(s.n_bases[0], s.coverage(), *streams["num_query_A"]);
        //        write_value(s.n_bases[1], s.coverage(), *streams["num_query_C"]);
        //        write_value(s.n_bases[2], s.coverage(), *streams["num_query_G"]);
        //        write_value(s.n_bases[3], s.coverage(), *streams["num_query_T"]);
        //        write_value(s.num_snps(), s.coverage(), *streams["num_SNPs"]);
        //        write_value(s.n_discord, s.coverage(), *streams["num_discordant"]);
        //
        //        streams["min_insert_size_Match"]->write(reinterpret_cast<const char
        //        *>(&s.min_i_size), 2); write_float_value(s.mean_i_size,
        //        *streams["mean_insert_size_Match"]); write_float_value(s.std_dev_i_size,
        //        *streams["stdev_insert_size_Match"]);
        //        streams["max_insert_size_Match"]->write(reinterpret_cast<const char
        //        *>(&s.max_i_size), 2);
        //
        //        streams["min_mapq_Match"]->put(s.min_map_qual);
        //        write_float_value(s.mean_map_qual, *streams["mean_mapq_Match"]);
        //        write_float_value(s.std_dev_map_qual, *streams["stdev_mapq_Match"]);
        //        streams["max_mapq_Match"]->put(s.max_map_qual);
        //
        //        streams["min_al_score_Match"]->put(s.min_al_score);
        //        write_float_value(s.mean_al_score, *streams["mean_al_score_Match"]);
        //        write_float_value(s.std_dev_al_score, *streams["stdev_al_score_Match"]);
        //        streams["max_al_score_Match"]->put(s.max_al_score);
        //
        //        write_value(s.n_proper_snp, s.n_proper_snp, *streams["num_proper_SNP"]);
        //        write_float_value(s.gc_percent, *streams["seq_window_perc_gc"]);
        //        streams["ref_base"]->put(s.ref_base);

        if (!std::isnan(s.mean_i_size)) { // coverage > 0
            (*count_mean)++;
            sums->at(0) += s.min_i_size;
            sums2->at(0) += s.min_i_size * s.min_i_size;
            sums->at(1) += s.mean_i_size;
            sums2->at(1) += s.mean_i_size * s.mean_i_size;
            sums->at(3) += s.max_i_size;
            sums2->at(3) += s.max_i_size * s.max_i_size;

            sums->at(4) += s.min_map_qual;
            sums2->at(4) += s.min_map_qual * s.min_map_qual;
            sums->at(5) += s.mean_map_qual;
            sums2->at(5) += s.mean_map_qual * s.mean_map_qual;
            sums->at(7) += s.max_map_qual;
            sums2->at(7) += s.max_map_qual * s.max_map_qual;

            sums->at(8) += s.min_al_score;
            sums2->at(8) += s.min_al_score * s.min_al_score;
            sums->at(9) += s.mean_al_score;
            sums2->at(9) += s.mean_al_score * s.mean_al_score;
            sums->at(11) += s.max_al_score;
            sums2->at(11) += s.max_al_score * s.max_al_score;

            if (!std::isnan(s.std_dev_i_size)) {
                assert(!std::isnan(s.std_dev_al_score && !std::isnan(s.std_dev_map_qual)));
                (*count_std_dev)++;
                sums->at(2) += s.std_dev_i_size;
                sums2->at(2) += s.std_dev_i_size * s.std_dev_i_size;

                sums->at(6) += s.std_dev_map_qual;
                sums2->at(6) += s.std_dev_map_qual * s.std_dev_map_qual;

                sums->at(10) += s.std_dev_al_score;
                sums2->at(10) += s.std_dev_al_score * s.std_dev_al_score;
            }
        }

        out << assembler << '\t' << item.reference_name << '\t' << pos << '\t';

        assert(s.ref_base == 0 || s.ref_base == item.reference[pos]);
        out << item.reference[pos] << '\t' << s.n_bases[0] << '\t' << s.n_bases[1] << '\t'
            << s.n_bases[2] << '\t' << s.n_bases[3] << '\t' << s.num_snps() << '\t' << s.coverage()
            << '\t' << s.n_discord << '\t';
        if (std::isnan(s.mean_i_size)) { // zero coverage, no i_size, no mapping quality
            out << "NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t";
        } else {
            count_mean++;
            if (!std::isnan(s.std_dev_i_size)) {
                assert(!std::isnan(s.std_dev_al_score && !std::isnan(s.std_dev_map_qual)));
                count_std_dev++;
            }
            out << stri(s.min_i_size) << '\t' << round2(s.mean_i_size) << '\t'
                << round2(s.std_dev_i_size) << '\t' << stri(s.max_i_size) << '\t';
            out << (int)s.min_map_qual << '\t' << round2(s.mean_map_qual) << '\t'
                << round2(s.std_dev_map_qual) << '\t' << (int)s.max_map_qual << '\t';
            out << (int)s.min_al_score << '\t' << round2(s.mean_al_score) << '\t'
                << round2(s.std_dev_al_score) << '\t' << (int)s.max_al_score << '\t';
        }
        out << s.n_proper_match << '\t' << s.n_orphan << '\t' << s.n_discord_match << '\t';

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
                         "min_al_score_Match",
                         "mean_al_score_Match",
                         "stdev_al_score_Match",
                         "max_al_score_Match",
                         "num_proper_Match",
                         "num_orphans_Match",
                         "num_discordant_Match",
                         "num_proper_SNP",
                 });
    }
    H.insert(H.end(), { "seq_window_entropy", "seq_window_perc_gc" });

    std::unordered_map<std::string, std::unique_ptr<std::ofstream>> binary_streams;
    for (const std::string &header : H) {
        auto fname = std::filesystem::path(FLAGS_o).replace_extension("_" + header);
        binary_streams[header] = std::make_unique<std::ofstream>(fname, std::ios::binary);
    }
    // the "Table of Contents" stream
    binary_streams["toc"] = std::make_unique<std::ofstream>(
            std::filesystem::path(FLAGS_o).replace_extension("_toc"));

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

    // the sum, sum of squares and non-NAN counts for each of the 12 float fields
    uint32_t count_mean = 0, count_std_dev = 0;
    std::vector<double> sums(12, 0);
    std::vector<double> sums2(12, 0);

    //============
    for (uint32_t c = 0; c < ref_names.size(); ++c) {
        const std::string reference_seq = get_sequence(FLAGS_fasta_file, ref_names[c]);
        std::vector<Stats> stats = contig_stats(ref_names[c], reference_seq, FLAGS_bam_file,
                                                FLAGS_window, FLAGS_short);
        write_stats({ std::move(stats), ref_names[c], reference_seq }, FLAGS_assembler, &out,
                    binary_streams, &count_mean, &count_std_dev, &sums, &sums2);
    }
    //============
    util::WaitQueue<QueueItem> wq(32);

    std::thread t([&] {
        for (;;) {
            QueueItem stats;
            if (!wq.pop_back(&stats)) {
                break;
            }
            write_stats(std::move(stats), FLAGS_assembler, &out, binary_streams, &count_mean,
                        &count_std_dev, &sums, &sums2);
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
    std::ofstream stats(std::filesystem::path(FLAGS_o).replace_extension("_stats"));
    stats << count_mean << std::endl << count_std_dev << std::endl;
    for (uint32_t i = 0; i < sums.size(); ++i) {
        stats << sums[i] << '\t' << sums2[i] << std::endl;
    }
    logger()->info("All done.");
}
