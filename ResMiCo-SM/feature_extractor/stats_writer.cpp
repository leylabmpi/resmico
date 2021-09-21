#include "stats_writer.hpp"

#include "util/logger.hpp"
#include "util/util.hpp"

#include <cstddef>
#include <filesystem>
#include <json/json.hpp>
#include <limits>
#include <string>
#include <vector>

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

inline uint16_t normalize(uint16_t v, uint16_t normalize_by) {
    static uint16_t MAX_16 = std::numeric_limits<uint16_t>::max();
    assert(v <= normalize_by);
    if (normalize_by == 0) {
        normalize_by = 1;
    }
    return v == MAX_16 ? MAX_16 : static_cast<uint16_t>((v * 10000.) / normalize_by);
}

void append_file(const std::string &dest, const std::string &source) {
    std::string cat_command = "cat " + source + " >> " + dest;
    if (std::system(cat_command.c_str())) {
        throw std::runtime_error("Error while cat-ing files: " + cat_command);
    }
    std::filesystem::remove(source);
}

void ContigStats::resize(uint32_t size) {
    coverage.resize(size);
    num_snps.resize(size);
    num_discordant.resize(size);

    min_insert_size.resize(size);
    max_insert_size.resize(size);
    mean_insert_size.resize(size);
    std_dev_insert_size.resize(size);

    min_map_qual.resize(size);
    max_map_qual.resize(size);
    mean_map_qual.resize(size);
    std_dev_map_qual.resize(size);

    min_al_score.resize(size);
    max_al_score.resize(size);
    mean_al_score.resize(size);
    std_dev_al_score.resize(size);

    num_proper_match.resize(size);
    num_orphans_match.resize(size);
    num_proper_snp.resize(size);
    gc_percent.resize(size);

    misassembly_by_pos.resize(size);

    for (uint32_t i : { 0, 1, 2, 3 }) {
        n_bases[i].resize(size);
    }
}

void write_data(const std::string &reference,
                const std::string &binary_stats_file,
                ContigStats &cs,
                uint32_t start,
                uint32_t end) {
    assert(start < end && end <= reference.size());
    uint32_t len = end - start;
    assert(len <= cs.size());
    ogzstream bin_stream(binary_stats_file.c_str());
    bin_stream.write(reinterpret_cast<char *>(&len), sizeof(len));
    bin_stream.write(reinterpret_cast<const char *>(reference.data() + start), len);

    bin_stream.write(reinterpret_cast<char *>(cs.coverage.data() + start),
                     len * sizeof(cs.coverage[0]));
    for (uint32_t i : { 0, 1, 2, 3 }) {
        bin_stream.write(reinterpret_cast<char *>(cs.n_bases[i].data() + start),
                         len * sizeof(cs.n_bases[0][0]));
    }
    bin_stream.write(reinterpret_cast<char *>(cs.num_snps.data() + start),
                     len * sizeof(cs.num_snps[0]));
    bin_stream.write(reinterpret_cast<char *>(cs.num_discordant.data() + start),
                     len * sizeof(cs.num_discordant[0]));

    bin_stream.write(reinterpret_cast<char *>(cs.min_insert_size.data() + start),
                     len * sizeof(cs.min_insert_size[0]));
    bin_stream.write(reinterpret_cast<char *>(cs.mean_insert_size.data() + start),
                     len * sizeof(cs.mean_insert_size[0]));
    bin_stream.write(reinterpret_cast<char *>(cs.std_dev_insert_size.data() + start),
                     len * sizeof(cs.std_dev_insert_size[0]));
    bin_stream.write(reinterpret_cast<char *>(cs.max_insert_size.data() + start),
                     len * sizeof(cs.max_insert_size[0]));

    bin_stream.write(reinterpret_cast<char *>(cs.min_map_qual.data() + start),
                     len * sizeof(cs.min_map_qual[0]));
    bin_stream.write(reinterpret_cast<char *>(cs.mean_map_qual.data() + start),
                     len * sizeof(cs.mean_map_qual[0]));
    bin_stream.write(reinterpret_cast<char *>(cs.std_dev_map_qual.data() + start),
                     len * sizeof(cs.std_dev_map_qual[0]));
    bin_stream.write(reinterpret_cast<char *>(cs.max_map_qual.data() + start),
                     len * sizeof(cs.max_map_qual[0]));

    bin_stream.write(reinterpret_cast<char *>(cs.min_al_score.data() + start),
                     len * sizeof(cs.min_al_score[0]));
    bin_stream.write(reinterpret_cast<char *>(cs.mean_al_score.data() + start),
                     len * sizeof(cs.mean_al_score[0]));
    bin_stream.write(reinterpret_cast<char *>(cs.std_dev_al_score.data() + start),
                     len * sizeof(cs.std_dev_al_score[0]));
    bin_stream.write(reinterpret_cast<char *>(cs.max_al_score.data() + start),
                     len * sizeof(cs.max_al_score[0]));

    bin_stream.write(reinterpret_cast<char *>(cs.num_proper_match.data() + start),
                     len * sizeof(cs.num_proper_match[0]));
    bin_stream.write(reinterpret_cast<char *>(cs.num_orphans_match.data() + start),
                     len * sizeof(cs.num_orphans_match[0]));
    bin_stream.write(reinterpret_cast<char *>(cs.num_proper_snp.data() + start),
                     len * sizeof(cs.num_proper_snp[0]));
    bin_stream.write(reinterpret_cast<char *>(cs.gc_percent.data() + start),
                     len * sizeof(cs.gc_percent[0]));

    bin_stream.write(reinterpret_cast<const char *>(cs.misassembly_by_pos.data() + start), len);
    bin_stream.close();
}

void write_data(const std::string &reference,
                const std::string &binary_stats_file,
                ContigStats &cs) {
    write_data(reference, binary_stats_file, cs, 0, reference.size());
}

StatsWriter::StatsWriter(const std::filesystem::path &out_dir,
                         uint32_t chunk_size,
                         uint32_t breakpoint_offset)
    : out_dir(out_dir),
      chunk_size(chunk_size),
      breakpoint_offset(breakpoint_offset),
      random_engine(std::mt19937(12345)) {
    std::error_code ec1;
    std::filesystem::create_directories(out_dir, ec1);
    if (ec1) {
        logger()->error("Could not create {}, bailing out", out_dir);
        std::exit(1);
    }

    // open the output stream (directory is now created)
    tsv_stream.open((out_dir / "features.tsv.gz").c_str());
    toc.open(out_dir / "toc");
    toc_chunk.open(out_dir / "toc_chunked");
    binary_features = out_dir / "features_binary";
    binary_chunk_features = out_dir / "features_binary_chunked";
    // make sure the feature files are empty (we don't inadvertently append to existing data)
    std::filesystem::remove(binary_features);
    std::filesystem::remove(binary_chunk_features);

    sums.resize(12, 0);
    sums2.resize(12, 0);
    tsv_stream.precision(3);

    // write tsv header
    tsv_stream << join_vec(headers, '\t');

    // write toc header for binary features (entire contig)
    toc << "Contig" << '\t' << "Size" << '\t' << "MisassemblCnt" << '\t' << "Offset" << std::endl;
    // write toc header for binary features (contig chunk)
    toc_chunk << "Contig" << '\t' << "Misassembly" << '\t' << "Offset" << std::endl;
}

void StatsWriter::write_stats(QueueItem &&item,
                              const std::string &assembler,
                              const std::vector<MisassemblyInfo> &mis) {
    if (item.reference.empty()) {
        return;
    }

    logger()->info("Writing features for contig {}...", item.reference_name);

    ContigStats &cs = contig_stats;
    const uint32_t contig_len = item.reference.size();
    cs.resize(contig_len);

    // get the misassembly information for each position
    cs.misassembly_by_pos = expand(contig_len, mis);

    for (uint32_t pos = 0; pos < item.stats.size(); ++pos) {
        const Stats &s = item.stats[pos];

        cs.coverage[pos] = s.coverage;
        for (uint32_t i : { 0, 1, 2, 3 }) {
            cs.n_bases[i][pos] = normalize(s.n_bases[i], s.coverage);
        }
        cs.num_snps[pos] = normalize(s.num_snps(), s.coverage);
        cs.num_discordant[pos] = normalize(s.n_discord, s.coverage);

        cs.min_insert_size[pos] = s.min_i_size;
        cs.max_insert_size[pos] = s.max_i_size;
        cs.mean_insert_size[pos] = s.mean_i_size;
        cs.std_dev_insert_size[pos] = s.std_dev_i_size;

        cs.min_map_qual[pos] = s.min_map_qual;
        cs.max_map_qual[pos] = s.max_map_qual;
        cs.mean_map_qual[pos] = s.mean_map_qual;
        cs.std_dev_map_qual[pos] = s.std_dev_map_qual;

        cs.min_al_score[pos] = s.min_al_score;
        cs.max_al_score[pos] = s.max_al_score;
        cs.mean_al_score[pos] = s.mean_al_score;
        cs.std_dev_al_score[pos] = s.std_dev_al_score;

        cs.num_proper_match[pos] = normalize(s.n_proper_match, s.coverage);
        cs.num_orphans_match[pos] = normalize(s.n_orphan_match, s.coverage);
        cs.num_proper_snp[pos] = normalize(s.n_proper_snp, s.coverage);
        cs.gc_percent[pos] = s.gc_percent * 100;

        if (!std::isnan(s.mean_i_size)) { // coverage > 0
            count_mean++;
            sums[0] += s.min_i_size;
            sums2[0] += s.min_i_size * s.min_i_size;
            sums[1] += s.mean_i_size;
            sums2[1] += s.mean_i_size * s.mean_i_size;
            sums[3] += s.max_i_size;
            sums2[3] += s.max_i_size * s.max_i_size;

            sums[4] += s.min_map_qual;
            sums2[4] += s.min_map_qual * s.min_map_qual;
            sums[5] += s.mean_map_qual;
            sums2[5] += s.mean_map_qual * s.mean_map_qual;
            sums[7] += s.max_map_qual;
            sums2[7] += s.max_map_qual * s.max_map_qual;

            sums[8] += s.min_al_score;
            sums2[8] += s.min_al_score * s.min_al_score;
            sums[9] += s.mean_al_score;
            sums2[9] += s.mean_al_score * s.mean_al_score;
            sums[11] += s.max_al_score;
            sums2[11] += s.max_al_score * s.max_al_score;

            if (!std::isnan(s.std_dev_i_size)) {
                assert(!std::isnan(s.std_dev_al_score && !std::isnan(s.std_dev_map_qual)));
                count_std_dev++;
                sums[2] += s.std_dev_i_size;
                sums2[2] += s.std_dev_i_size * s.std_dev_i_size;

                sums[6] += s.std_dev_map_qual;
                sums2[6] += s.std_dev_map_qual * s.std_dev_map_qual;

                sums[10] += s.std_dev_al_score;
                sums2[10] += s.std_dev_al_score * s.std_dev_al_score;
            }
        }

        tsv_stream << assembler << '\t' << item.reference_name << '\t' << pos << '\t';

        assert(s.ref_base == 0 || s.ref_base == item.reference[pos]);
        tsv_stream << item.reference[pos] << '\t' << s.n_bases[0] << '\t' << s.n_bases[1] << '\t'
                   << s.n_bases[2] << '\t' << s.n_bases[3] << '\t' << s.num_snps() << '\t'
                   << s.coverage << '\t' << s.n_discord << '\t';
        if (std::isnan(s.mean_i_size)) { // zero coverage, no i_size, no mapping quality
            tsv_stream << "NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t";
        } else {
            tsv_stream << stri(s.min_i_size) << '\t' << round2(s.mean_i_size) << '\t'
                       << round2(s.std_dev_i_size) << '\t' << stri(s.max_i_size) << '\t';
            tsv_stream << (int)s.min_map_qual << '\t' << round2(s.mean_map_qual) << '\t'
                       << round2(s.std_dev_map_qual) << '\t' << (int)s.max_map_qual << '\t';
            tsv_stream << (int)s.min_al_score << '\t' << round2(s.mean_al_score) << '\t'
                       << round2(s.std_dev_al_score) << '\t' << (int)s.max_al_score << '\t';
        }
        tsv_stream << s.n_proper_match << '\t' << s.n_orphan_match << '\t' << s.n_discord_match
                   << '\t';

        tsv_stream << s.n_proper_snp << '\t' << s.entropy << '\t' << s.gc_percent << '\t';
        tsv_stream << (mis.empty() ? 0 : 1) << '\t' << type_to_string(cs.misassembly_by_pos[pos])
                   << '\n';
    }

    std::string binary_stats_file = out_dir / (item.reference_name + ".gz");
    write_data(item.reference, binary_stats_file, cs);
    toc << item.reference_name << '\t' << item.stats.size() << '\t' << mis.size() << '\t'
        << std::filesystem::file_size(binary_stats_file) << std::endl;
    append_file(binary_features, binary_stats_file);

    // ----- start selecting a chunk and writing its stats to disk ----
    offsets.clear();
    uint32_t start, stop;
    if (mis.empty()) {
        // select a chunk of length chunk_size randomly from the string
        if (contig_len <= chunk_size) {
            start = 0;
            stop = contig_len;
        } else {
            std::uniform_int_distribution<uint32_t> rnd_start(0, contig_len - chunk_size);
            start = rnd_start(random_engine);
            stop = start + chunk_size;
        }
        std::string fname = out_dir / (item.reference_name + ".ok.gz");
        write_data(item.reference, fname, cs, start, stop);
        toc_chunk << item.reference_name << "\t0\t" << std::filesystem::file_size(fname)
                  << std::endl;
        append_file(binary_chunk_features, fname);
    } else {
        // select a chunk of size chunk_size around the breaking point
        std::uniform_int_distribution<int32_t> offset_gen(-breakpoint_offset, breakpoint_offset);
        // create one stats file for each mis-assembly breakpoint
        for (uint32_t i = 0; i < mis.size(); ++i) {
            uint32_t mid = (mis[i].break_start + mis[i].break_end) / 2;
            if (mis[i].break_end < chunk_size / 2
                || mis[i].break_start + chunk_size / 2 > contig_len) {
                logger()->info("Dismissed breaking point {}/{} for contig {}. Too close to edge.",
                               mis[i].break_start, mis[i].break_end, item.reference_name);
                continue;
            }
            int32_t offset = offset_gen(random_engine);
            offsets.push_back(offset);
            start = chunk_size / 2 < mid + offset ? mid + offset - chunk_size / 2 : chunk_size / 2;
            stop = start + chunk_size;
            // move the chunk to left if it exceeds contig bounds
            if (stop > contig_len) {
                start = contig_len - chunk_size;
                stop = contig_len;
            }
            std::string fname
                    = out_dir / (item.reference_name + ".mis" + std::to_string(i) + ".gz");
            write_data(item.reference, fname, cs, start, stop);
            toc_chunk << item.reference_name + "_" + std::to_string(i) << "\t1\t"
                      << std::filesystem::file_size(fname) << std::endl;
            append_file(binary_chunk_features, fname);
        }
    }

    assert(cs.misassembly_by_pos.size() == contig_len);
    logger()->info("Writing features for contig {} done.", item.reference_name);
}

void StatsWriter::write_summary() {
    nlohmann::json j;

    j["mean_cnt"] = count_mean;
    j["stdev_cnt"] = count_std_dev;

    j["insert_size"]["sum"]
            = { { "min", sums[0] }, { "mean", sums[1] }, { "stdev", sums[2] }, { "max", sums[3] } };
    j["insert_size"]["sum2"] = {
        { "min", sums2[0] }, { "mean", sums2[1] }, { "stdev", sums2[2] }, { "max", sums2[3] }
    };

    j["mapq"]["sum"]
            = { { "min", sums[4] }, { "mean", sums[5] }, { "stdev", sums[6] }, { "max", sums[7] } };
    j["mapq"]["sum2"] = {
        { "min", sums2[4] }, { "mean", sums2[5] }, { "stdev", sums2[6] }, { "max", sums2[7] }
    };

    j["al_score"]["sum"] = {
        { "min", sums[8] }, { "mean", sums[9] }, { "stdev", sums[10] }, { "max", sums[11] }
    };
    j["al_score"]["sum2"] = {
        { "min", sums2[8] }, { "mean", sums2[9] }, { "stdev", sums2[10] }, { "max", sums2[11] }
    };

    std::ofstream stats(out_dir / "stats");
    stats << j.dump(2);
}
