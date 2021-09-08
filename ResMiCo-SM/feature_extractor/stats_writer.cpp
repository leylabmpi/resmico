#include "stats_writer.hpp"

#include "util/logger.hpp"
#include "util/util.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

std::vector<std::string> bin_headers = { "ref_base",
                                         "num_query_A",
                                         "num_query_C",
                                         "num_query_G",
                                         "num_query_T",
                                         "num_SNPs",
                                         "coverage",
                                         "num_discordant",
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
                                         "num_proper_SNP",
                                         "seq_window_perc_gc",
                                         "misassembly_by_pos" };


std::vector<std::string> headers = { "assembler",
                                     "contig",
                                     "position",
                                     "ref_base",
                                     "num_query_A",
                                     "num_query_C",
                                     "num_query_G",
                                     "num_query_T",
                                     "num_SNPs",
                                     "coverage",
                                     "num_discordant",
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
                                     "seq_window_entropy",
                                     "seq_window_perc_gc",
                                     "Extensive_misassembly",
                                     "Extensive_misassembly_by_pos" };

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

inline void write_value(uint16_t v, uint16_t normalize_by, ogzstream &out) {
    static uint16_t MAX_16 = std::numeric_limits<uint16_t>::max();
    assert(v <= normalize_by || v == MAX_16);
    if (normalize_by == 0) {
        normalize_by = 1;
    }
    uint16_t to_write = v == MAX_16 ? MAX_16 : static_cast<uint16_t>((v * 10000.) / normalize_by);
    out.write(reinterpret_cast<const char *>(&to_write), 2);
}

inline void write_float_value(float v, ogzstream &out) {
    assert(std::isnan(v) || v <= std::numeric_limits<int16_t>::max());
    int16_t to_write
            = isnan(v) ? std::numeric_limits<int16_t>::max() : static_cast<int16_t>(v * 100);
    out.write(reinterpret_cast<const char *>(&to_write), 2);
}

std::unordered_map<std::string, std::unique_ptr<ogzstream>>
get_streams(const std::string &out_prefix) {
    std::unordered_map<std::string, std::unique_ptr<ogzstream>> binary_streams;
    for (const std::string &header : bin_headers) {
        auto fname = std::filesystem::path(out_prefix).replace_extension(header + ".gz");
        binary_streams[header] = std::make_unique<ogzstream>(fname.string().c_str());
    }
    return binary_streams;
}

/** Pretty print of the results */
void write_stats(QueueItem &&item,
                 const std::string &assembler,
                 const std::vector<MisassemblyInfo> &mis,
                 std::ofstream *o,
                 std::ofstream *toc,
                 std::unordered_map<std::string, std::unique_ptr<ogzstream>> *bin_streams,
                 uint32_t *count_mean,
                 uint32_t *count_std_dev,
                 std::vector<double> *sums,
                 std::vector<double> *sums2) {
    assert(sums->size() == 12 && sums2->size() == 12);

    std::ofstream &out = *o;
    auto &streams = *bin_streams;
    logger()->info("Writing features for contig {}...", item.reference_name);
    out.precision(3);
    *toc << assembler << '\t' << item.reference_name << '\t' << item.stats.size() << '\t'
         << mis.size() << std::endl;

    // get the misassembly information for each position
    const std::vector<uint8_t> mi_per_pos = expand(item.reference.size(), mis);
    streams["misassembly_by_pos"]->write(reinterpret_cast<const char *>(mi_per_pos.data()),
                                         mi_per_pos.size());

    for (uint32_t pos = 0; pos < item.stats.size(); ++pos) {
        const Stats &s = item.stats[pos];

        uint16_t v = s.coverage;
        streams["coverage"]->write(reinterpret_cast<char *>(&v), 2);
        write_value(s.n_bases[0], s.coverage, *streams["num_query_A"]);
        write_value(s.n_bases[1], s.coverage, *streams["num_query_C"]);
        write_value(s.n_bases[2], s.coverage, *streams["num_query_G"]);
        write_value(s.n_bases[3], s.coverage, *streams["num_query_T"]);
        write_value(s.num_snps(), s.coverage, *streams["num_SNPs"]);
        write_value(s.n_discord, s.coverage, *streams["num_discordant"]);

        streams["min_insert_size_Match"]->write(reinterpret_cast<const char *>(&s.min_i_size), 2);
        write_float_value(s.mean_i_size, *streams["mean_insert_size_Match"]);
        write_float_value(s.std_dev_i_size, *streams["stdev_insert_size_Match"]);
        streams["max_insert_size_Match"]->write(reinterpret_cast<const char *>(&s.max_i_size), 2);

        streams["min_mapq_Match"]->put(s.min_map_qual);
        write_float_value(s.mean_map_qual, *streams["mean_mapq_Match"]);
        write_float_value(s.std_dev_map_qual, *streams["stdev_mapq_Match"]);
        streams["max_mapq_Match"]->put(s.max_map_qual);

        streams["min_al_score_Match"]->put(s.min_al_score);
        write_float_value(s.mean_al_score, *streams["mean_al_score_Match"]);
        write_float_value(s.std_dev_al_score, *streams["stdev_al_score_Match"]);
        streams["max_al_score_Match"]->put(s.max_al_score);

        write_value(s.n_proper_snp, s.coverage, *streams["num_proper_SNP"]);
        write_float_value(s.gc_percent * 100, *streams["seq_window_perc_gc"]);
        streams["ref_base"]->put(item.reference[pos]);

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
            << s.n_bases[2] << '\t' << s.n_bases[3] << '\t' << s.num_snps() << '\t' << s.coverage
            << '\t' << s.n_discord << '\t';
        if (std::isnan(s.mean_i_size)) { // zero coverage, no i_size, no mapping quality
            out << "NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t";
        } else {
            out << stri(s.min_i_size) << '\t' << round2(s.mean_i_size) << '\t'
                << round2(s.std_dev_i_size) << '\t' << stri(s.max_i_size) << '\t';
            out << (int)s.min_map_qual << '\t' << round2(s.mean_map_qual) << '\t'
                << round2(s.std_dev_map_qual) << '\t' << (int)s.max_map_qual << '\t';
            out << (int)s.min_al_score << '\t' << round2(s.mean_al_score) << '\t'
                << round2(s.std_dev_al_score) << '\t' << (int)s.max_al_score << '\t';
        }
        out << s.n_proper_match << '\t' << s.n_orphan << '\t' << s.n_discord_match << '\t';

        out << s.n_proper_snp << '\t' << s.entropy << '\t' << s.gc_percent << '\t';
        out << (mis.empty() ? 0 : 1) << '\t' << type_to_string(mi_per_pos[pos]) << '\n';
    }
    logger()->info("Writing features for contig {} done.", item.reference_name);
}
