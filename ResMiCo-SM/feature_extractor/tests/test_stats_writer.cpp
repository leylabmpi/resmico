#include "stats_writer.hpp"

#include "util/gzstream.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <json/json.hpp>

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
using namespace ::testing;

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

TEST(WriteStats, Empty) {
    QueueItem item;

    StatsWriter stats_writer("/tmp/stats", 500, 1);
    stats_writer.write_stats(std::move(item), "metaQuast", {});
}

TEST(WriteStats, TwoReads) {
    std::unordered_map<std::string, std::vector<MisassemblyInfo>> mi_info
            = parse_misassembly_info("data/test.mis_contigs.info");

    StatsWriter stats_writer("/tmp/stats/", 5, 1);

    std::string contig_names[] = { "Contig2", "Contig1" };
    std::string fasta_files[] = { "data/test2.fa.gz", "data/test.fa" };
    std::string bam_files[] = { "data/test2.bam", "data/test1.bam" };
    for (uint32_t i : { 0, 1 }) {
        std::string reference_seq = get_sequence(fasta_files[i], contig_names[i]);
        std::vector<Stats> stats
                = contig_stats(contig_names[i], reference_seq, bam_files[i], 4, false);
        QueueItem item = { std::move(stats), contig_names[i], reference_seq };
        stats_writer.write_stats(std::move(item), "metaSpades", mi_info[contig_names[i]]);
    }
    stats_writer.write_summary();

    std::string out_files[] = { "/tmp/stats/features.tsv.gz",
                                "/tmp/stats/toc",
                                "/tmp/stats/stats",
                                "/tmp/stats/contig_stats/Contig1.gz",
                                "/tmp/stats/contig_stats/Contig2.gz",
                                "/tmp/stats/contig_chunk_stats/Contig1.ok.gz",
                                "/tmp/stats/contig_chunk_stats/Contig2.mis0.gz" };
    for (const std::string &fname : out_files) {
        ASSERT_TRUE(std::filesystem::exists(fname));
    }

    uint32_t len;

    igzstream stats2("/tmp/stats/contig_stats/Contig2.gz");
    stats2.read(reinterpret_cast<char *>(&len), 4);

    std::string contig(len, 'N');
    std::vector<uint16_t> coverage(len);
    std::array<std::vector<uint16_t>, 4> n_bases;
    for (uint32_t i : { 0, 1, 2, 3 }) {
        n_bases[i].resize(len);
    }

    std::vector<uint16_t> num_snps(len);
    std::vector<uint16_t> num_discordant(len);

    std::vector<uint16_t> min_insert_size(len);
    std::vector<uint16_t> max_insert_size(len);
    std::vector<float> mean_insert_size(len);
    std::vector<float> std_dev_insert_size(len);

    std::vector<uint8_t> min_map_qual(len);
    std::vector<uint8_t> max_map_qual(len);
    std::vector<float> mean_map_qual(len);
    std::vector<float> std_dev_map_qual(len);

    std::vector<int8_t> min_al_score(len);
    std::vector<int8_t> max_al_score(len);
    std::vector<float> mean_al_score(len);
    std::vector<float> std_dev_al_score(len);

    std::vector<uint16_t> num_proper_snp(len);
    std::vector<float> gc_percent(len);

    std::vector<uint8_t> misassembly_by_pos(len);

    stats2.read(reinterpret_cast<char *>(contig.data()), len);
    stats2.read(reinterpret_cast<char *>(coverage.data()), len * sizeof(coverage[0]));
    for (uint32_t i : { 0, 1, 2, 3 }) {
        stats2.read(reinterpret_cast<char *>(n_bases[i].data()), len * sizeof(n_bases[i][0]));
    }
    stats2.read(reinterpret_cast<char *>(num_snps.data()), len * sizeof(num_snps[0]));
    stats2.read(reinterpret_cast<char *>(num_discordant.data()), len * sizeof(num_discordant[0]));

    stats2.read(reinterpret_cast<char *>(min_insert_size.data()), len * sizeof(min_insert_size[0]));
    stats2.read(reinterpret_cast<char *>(mean_insert_size.data()),
                len * sizeof(mean_insert_size[0]));
    stats2.read(reinterpret_cast<char *>(std_dev_insert_size.data()),
                len * sizeof(std_dev_insert_size[0]));
    stats2.read(reinterpret_cast<char *>(max_insert_size.data()), len * sizeof(max_insert_size[0]));

    stats2.read(reinterpret_cast<char *>(min_map_qual.data()), len * sizeof(min_map_qual[0]));
    stats2.read(reinterpret_cast<char *>(mean_map_qual.data()), len * sizeof(mean_map_qual[0]));
    stats2.read(reinterpret_cast<char *>(std_dev_map_qual.data()),
                len * sizeof(std_dev_map_qual[0]));
    stats2.read(reinterpret_cast<char *>(max_map_qual.data()), len * sizeof(max_map_qual[0]));

    stats2.read(reinterpret_cast<char *>(min_al_score.data()), len * sizeof(min_al_score[0]));
    stats2.read(reinterpret_cast<char *>(mean_al_score.data()), len * sizeof(mean_al_score[0]));
    stats2.read(reinterpret_cast<char *>(std_dev_al_score.data()),
                len * sizeof(std_dev_al_score[0]));
    stats2.read(reinterpret_cast<char *>(max_al_score.data()), len * sizeof(max_al_score[0]));

    stats2.read(reinterpret_cast<char *>(num_proper_snp.data()), len * sizeof(num_proper_snp[0]));
    stats2.read(reinterpret_cast<char *>(gc_percent.data()), len * sizeof(gc_percent[0]));
    stats2.read(reinterpret_cast<char *>(misassembly_by_pos.data()),
                len * sizeof(misassembly_by_pos[0]));

    for (uint32_t i = 0; i < 500; ++i) {
        ASSERT_EQ(i < 498 ? 'A' : 'C', contig[i]) << "Position: " << i;
        ASSERT_EQ(i >= 420 && i < 425 ? 10'000 : (i > 0 && i < 5 ? 5000 : 0),
                  num_proper_snp[i])
                << "at position: " << i; // 5000 is 0.5
        ASSERT_EQ((i >= 420 && i < 425) || i < 5 ? 2 : 0, coverage[i]) << "at position: " << i;

        ASSERT_EQ(num_discordant[i], 0);

        uint16_t base_counts[] = { n_bases[0][i], n_bases[1][i], n_bases[2][i], n_bases[3][i] };
        ASSERT_THAT(base_counts,
                    i == 0 ? ElementsAre(10'000, 0, 0, 0)
                           : (i < 5 ? ElementsAre(5'000, 0, 5'000, 0)
                                    : (i >= 420 && i < 425 ? ElementsAre(0, 5'000, 0, 5'000)
                                                           : ElementsAre(0, 0, 0, 0))));
        ASSERT_EQ(min_al_score[i], i == 0 ? -28 : i < 5 ? 0 : 127);
        ASSERT_EQ(max_al_score[i], i < 5 ? 0 : 127);
        ASSERT_TRUE((i >= 5 && std::isnan(mean_al_score[i]))
                    || mean_al_score[i] == (i == 0 ? -14 : 0));

        ASSERT_EQ(gc_percent[i], i < 498 ? 0 : i == 498 ? 25. : 50.) << "Position: " << i;
        ASSERT_EQ(num_snps[i], i > 0 && i < 5 ? 5'000 : i >= 420 && i < 425 ? 10'000 : 0);
        ASSERT_EQ(min_map_qual[i], i < 5 ? 6 : std::numeric_limits<uint8_t>::max());
        ASSERT_EQ(max_map_qual[i], i == 0 ? 7 : i < 5 ? 6 : std::numeric_limits<uint8_t>::max());
        ASSERT_TRUE((i >= 5 && std::isnan(mean_map_qual[i]))
                    || mean_map_qual[i] == (i == 0 ? 6.5 : 6));
        ASSERT_EQ(misassembly_by_pos[i], i >= 20 ? 0 : 1) << "Position " << i;
    }

    // check the table of contents file, and make sure that each contig has length 500, the first
    // contig is labeled as misassembled, and the 2nd contig is not
    std::ifstream toc_read("/tmp/stats/toc");
    std::string line;
    std::getline(toc_read, line); // skip header
    for (uint32_t i : { 0, 1 }) {
        std::string assembler, contig_name;
        uint32_t length, is_missasembly;
        toc_read >> assembler >> contig_name >> length >> is_missasembly;
        ASSERT_EQ("metaSpades", assembler);
        ASSERT_EQ("Contig" + std::to_string(i == 0 ? 2 : 1), contig_name);
        ASSERT_EQ(500, length);
        ASSERT_EQ(i == 0 ? 1 : 0, is_missasembly);
    }
}

// Make sure that the summary stats (sums, sums of squares) generated for each float metric are
// correct
TEST(WriteStats, TwoReadsStats) {
    std::unordered_map<std::string, std::vector<MisassemblyInfo>> mi_info
            = parse_misassembly_info("data/test.mis_contigs.info");

    StatsWriter stats_writer("/tmp/stats/", 5, 1);

    std::string contig_names[] = { "Contig1", "Contig2" };
    std::string fasta_files[] = { "data/test.fa", "data/test2.fa.gz" };
    std::string bam_files[] = { "data/test1.bam", "data/test2.bam" };
    for (uint32_t i : { 0, 1 }) {
        std::string reference_seq = get_sequence(fasta_files[i], contig_names[i]);
        std::vector<Stats> stats
                = contig_stats(contig_names[i], reference_seq, bam_files[i], 4, false);
        QueueItem item = { std::move(stats), contig_names[i], reference_seq };
        stats_writer.write_stats(std::move(item), "metaSpades", mi_info[contig_names[i]]);
    }
    stats_writer.write_summary();

    std::string stats_files[]
            = { "/tmp/stats/contig_stats/Contig1.gz", "/tmp/stats/contig_stats/Contig2.gz" };

    uint32_t mean_count = 0;
    uint32_t std_dev_count = 0;

    uint32_t min_insert_sum = 0;
    uint32_t min_insert_sum2 = 0;
    uint32_t max_insert_sum = 0;
    uint32_t max_insert_sum2 = 0;
    double mean_insert_sum = 0;
    double mean_insert_sum2 = 0;
    double std_dev_insert_sum = 0;
    double std_dev_insert_sum2 = 0;

    uint32_t min_map_qual_sum = 0;
    uint32_t min_map_qual_sum2 = 0;
    uint32_t max_map_qual_sum = 0;
    uint32_t max_map_qual_sum2 = 0;
    double mean_map_qual_sum = 0;
    double mean_map_qual_sum2 = 0;
    double std_dev_map_qual_sum = 0;
    double std_dev_map_qual_sum2 = 0;

    int32_t min_al_score_sum = 0;
    int32_t min_al_score_sum2 = 0;
    int32_t max_al_score_sum = 0;
    int32_t max_al_score_sum2 = 0;
    double mean_al_score_sum = 0;
    double mean_al_score_sum2 = 0;
    double std_dev_al_score_sum = 0;
    double std_dev_al_score_sum2 = 0;

    for (const std::string &stats_file : stats_files) {
        uint32_t len;
        igzstream stats(stats_file.c_str());
        stats.read(reinterpret_cast<char *>(&len), 4);

        // skip contig, coverage (2 bytes), bases (2*4 bytes), num_snps (2 bytes), num_discoraant (2
        // bytes)
        std::vector<char> buf(len * (1 + 2 + 2 * 4 + 2 + 2));
        stats.read(buf.data(), buf.size());

        std::vector<uint16_t> min_insert_size(len);
        std::vector<uint16_t> max_insert_size(len);
        std::vector<float> mean_insert_size(len);
        std::vector<float> std_dev_insert_size(len);

        std::vector<uint8_t> min_map_qual(len);
        std::vector<uint8_t> max_map_qual(len);
        std::vector<float> mean_map_qual(len);
        std::vector<float> std_dev_map_qual(len);

        std::vector<int8_t> min_al_score(len);
        std::vector<int8_t> max_al_score(len);
        std::vector<float> mean_al_score(len);
        std::vector<float> std_dev_al_score(len);

        stats.read(reinterpret_cast<char *>(min_insert_size.data()),
                   len * sizeof(min_insert_size[0]));
        stats.read(reinterpret_cast<char *>(mean_insert_size.data()),
                   len * sizeof(mean_insert_size[0]));
        stats.read(reinterpret_cast<char *>(std_dev_insert_size.data()),
                   len * sizeof(std_dev_insert_size[0]));
        stats.read(reinterpret_cast<char *>(max_insert_size.data()),
                   len * sizeof(max_insert_size[0]));

        stats.read(reinterpret_cast<char *>(min_map_qual.data()), len * sizeof(min_map_qual[0]));
        stats.read(reinterpret_cast<char *>(mean_map_qual.data()), len * sizeof(mean_map_qual[0]));
        stats.read(reinterpret_cast<char *>(std_dev_map_qual.data()),
                   len * sizeof(std_dev_map_qual[0]));
        stats.read(reinterpret_cast<char *>(max_map_qual.data()), len * sizeof(max_map_qual[0]));

        stats.read(reinterpret_cast<char *>(min_al_score.data()), len * sizeof(min_al_score[0]));
        stats.read(reinterpret_cast<char *>(mean_al_score.data()), len * sizeof(mean_al_score[0]));
        stats.read(reinterpret_cast<char *>(std_dev_al_score.data()),
                   len * sizeof(std_dev_al_score[0]));
        stats.read(reinterpret_cast<char *>(max_al_score.data()), len * sizeof(max_al_score[0]));

        // test mean counts
        size_t cnt = std::count_if(min_insert_size.begin(), min_insert_size.end(), [](uint16_t v) {
            return v != std::numeric_limits<uint16_t>::max();
        });
        ASSERT_EQ(cnt,
                  std::count_if(max_insert_size.begin(), max_insert_size.end(), [](uint16_t v) {
                      return v != std::numeric_limits<uint16_t>::max();
                  }));
        ASSERT_EQ(cnt, std::count_if(mean_insert_size.begin(), mean_insert_size.end(), [](float v) {
                      return !std::isnan(v);
                  }));

        ASSERT_EQ(cnt, std::count_if(min_map_qual.begin(), min_map_qual.end(), [](uint8_t v) {
                      return v != std::numeric_limits<uint8_t>::max();
                  }));
        ASSERT_EQ(cnt, std::count_if(max_map_qual.begin(), max_map_qual.end(), [](uint8_t v) {
                      return v != std::numeric_limits<uint8_t>::max();
                  }));
        ASSERT_EQ(cnt, std::count_if(mean_map_qual.begin(), mean_map_qual.end(), [](float v) {
                      return !std::isnan(v);
                  }));

        ASSERT_EQ(cnt, std::count_if(min_al_score.begin(), min_al_score.end(), [](int8_t v) {
                      return v != std::numeric_limits<int8_t>::max();
                  }));
        ASSERT_EQ(cnt, std::count_if(max_al_score.begin(), max_al_score.end(), [](int8_t v) {
                      return v != std::numeric_limits<int8_t>::max();
                  }));
        ASSERT_EQ(cnt, std::count_if(mean_al_score.begin(), mean_al_score.end(), [](float v) {
                      return !std::isnan(v);
                  }));

        mean_count += cnt;

        // test standard deviation count
        cnt = std::count_if(std_dev_insert_size.begin(), std_dev_insert_size.end(),
                            [](float v) { return !std::isnan(v); });
        ASSERT_EQ(cnt, std::count_if(std_dev_map_qual.begin(), std_dev_map_qual.end(), [](float v) {
                      return !std::isnan(v);
                  }));
        ASSERT_EQ(cnt, std::count_if(std_dev_al_score.begin(), std_dev_al_score.end(), [](float v) {
                      return !std::isnan(v);
                  }));
        std_dev_count += cnt;

        // tests sums and sums2
        // insert size
        std::for_each(min_insert_size.begin(), min_insert_size.end(), [&](uint16_t v) {
            min_insert_sum += v != std::numeric_limits<uint16_t>::max() ? v : 0;
            min_insert_sum2 += v != std::numeric_limits<uint16_t>::max() ? v * v : 0;
        });
        std::for_each(max_insert_size.begin(), max_insert_size.end(), [&](uint16_t v) {
            max_insert_sum += v != std::numeric_limits<uint16_t>::max() ? v : 0;
            max_insert_sum2 += v != std::numeric_limits<uint16_t>::max() ? v * v : 0;
        });
        std::for_each(mean_insert_size.begin(), mean_insert_size.end(), [&](float v) {
            mean_insert_sum += std::isnan(v) ? 0 : v;
            mean_insert_sum2 += std::isnan(v) ? 0 : v * v;
        });
        std::for_each(std_dev_insert_size.begin(), std_dev_insert_size.end(), [&](float v) {
            std_dev_insert_sum += std::isnan(v) ? 0 : v;
            std_dev_insert_sum2 += std::isnan(v) ? 0 : v * v;
        });

        // mapping quality
        std::for_each(min_map_qual.begin(), min_map_qual.end(), [&](uint8_t v) {
            min_map_qual_sum += v != std::numeric_limits<uint8_t>::max() ? v : 0;
            min_map_qual_sum2 += v != std::numeric_limits<uint8_t>::max() ? v * v : 0;
        });
        std::for_each(max_map_qual.begin(), max_map_qual.end(), [&](uint8_t v) {
            max_map_qual_sum += v != std::numeric_limits<uint8_t>::max() ? v : 0;
            max_map_qual_sum2 += v != std::numeric_limits<uint8_t>::max() ? v * v : 0;
        });
        std::for_each(mean_map_qual.begin(), mean_map_qual.end(), [&](float v) {
            mean_map_qual_sum += std::isnan(v) ? 0 : v;
            mean_map_qual_sum2 += std::isnan(v) ? 0 : v * v;
        });
        std::for_each(std_dev_map_qual.begin(), std_dev_map_qual.end(), [&](float v) {
            std_dev_map_qual_sum += std::isnan(v) ? 0 : v;
            std_dev_map_qual_sum2 += std::isnan(v) ? 0 : v * v;
        });

        // alignment quality
        std::for_each(min_al_score.begin(), min_al_score.end(), [&](int8_t v) {
            min_al_score_sum += v != std::numeric_limits<int8_t>::max() ? v : 0;
            min_al_score_sum2 += v != std::numeric_limits<int8_t>::max() ? v * v : 0;
        });
        std::for_each(max_al_score.begin(), max_al_score.end(), [&](int8_t v) {
            max_al_score_sum += v != std::numeric_limits<int8_t>::max() ? v : 0;
            max_al_score_sum2 += v != std::numeric_limits<int8_t>::max() ? v * v : 0;
        });
        std::for_each(mean_al_score.begin(), mean_al_score.end(), [&](float v) {
            mean_al_score_sum += std::isnan(v) ? 0 : v;
            mean_al_score_sum2 += std::isnan(v) ? 0 : v * v;
        });
        std::for_each(std_dev_al_score.begin(), std_dev_al_score.end(), [&](float v) {
            std_dev_al_score_sum += std::isnan(v) ? 0 : v;
            std_dev_al_score_sum2 += std::isnan(v) ? 0 : v * v;
        });
    }

    std::ifstream js_stats("/tmp/stats/stats");
    nlohmann::json j;
    js_stats >> j;
    ASSERT_EQ(j["mean_cnt"], mean_count);
    ASSERT_EQ(j["stdev_cnt"], std_dev_count);

    ASSERT_EQ(j["insert_size"]["sum"]["min"], min_insert_sum);
    ASSERT_EQ(j["insert_size"]["sum2"]["min"], min_insert_sum2);
    ASSERT_EQ(j["insert_size"]["sum"]["max"], max_insert_sum);
    ASSERT_EQ(j["insert_size"]["sum2"]["max"], max_insert_sum2);
    ASSERT_EQ(j["insert_size"]["sum"]["mean"], mean_insert_sum);
    ASSERT_EQ(j["insert_size"]["sum2"]["mean"], mean_insert_sum2);
    ASSERT_EQ(j["insert_size"]["sum"]["stdev"], std_dev_insert_sum);
    ASSERT_EQ(j["insert_size"]["sum2"]["stdev"], std_dev_insert_sum2);

    ASSERT_EQ(j["mapq"]["sum"]["min"], min_map_qual_sum);
    ASSERT_EQ(j["mapq"]["sum2"]["min"], min_map_qual_sum2);
    ASSERT_EQ(j["mapq"]["sum"]["max"], max_map_qual_sum);
    ASSERT_EQ(j["mapq"]["sum2"]["max"], max_map_qual_sum2);
    ASSERT_NEAR(j["mapq"]["sum"]["mean"], mean_map_qual_sum, 1e-5);
    ASSERT_NEAR(j["mapq"]["sum2"]["mean"], mean_map_qual_sum2, 1e-5);
    ASSERT_NEAR(j["mapq"]["sum"]["stdev"], std_dev_map_qual_sum, 1e-5);
    ASSERT_NEAR(j["mapq"]["sum2"]["stdev"], std_dev_map_qual_sum2, 1e-5);

    ASSERT_EQ(j["al_score"]["sum"]["min"], min_al_score_sum);
    ASSERT_EQ(j["al_score"]["sum2"]["min"], min_al_score_sum2);
    ASSERT_EQ(j["al_score"]["sum"]["max"], max_al_score_sum);
    ASSERT_EQ(j["al_score"]["sum2"]["max"], max_al_score_sum2);
    ASSERT_NEAR(j["al_score"]["sum"]["mean"], mean_al_score_sum, 1e-5);
    ASSERT_NEAR(j["al_score"]["sum2"]["mean"], mean_al_score_sum2, 1e-5);
    ASSERT_NEAR(j["al_score"]["sum"]["stdev"], std_dev_al_score_sum, 1e-5);
    ASSERT_NEAR(j["al_score"]["sum2"]["stdev"], std_dev_al_score_sum2, 1e-5);
}

// Make sure that the chunks stats (data around breaking points) are correct
TEST(WriteStats, TwoReadsChunkStats) {
    std::unordered_map<std::string, std::vector<MisassemblyInfo>> mi_info
            = parse_misassembly_info("data/test.mis_contigs.info");

    // chunk_size=5, breakpoint_offset=0 (so it's deterministic)
    StatsWriter stats_writer("/tmp/stats/", 5, 0);

    std::string contig_names[] = { "Contig2", "Contig1" };
    std::string fasta_files[] = { "data/test2.fa.gz", "data/test.fa" };
    std::string bam_files[] = { "data/test2.bam", "data/test1.bam" };
    for (uint32_t i : { 0, 1 }) {
        std::string reference_seq = get_sequence(fasta_files[i], contig_names[i]);
        std::vector<Stats> stats
                = contig_stats(contig_names[i], reference_seq, bam_files[i], 4, false);
        QueueItem item = { std::move(stats), contig_names[i], reference_seq };
        stats_writer.write_stats(std::move(item), "metaSpades", mi_info[contig_names[i]]);
    }
    stats_writer.write_summary();

    std::string out_files[] = { "/tmp/stats/contig_chunk_stats/Contig1.ok.gz",
                                "/tmp/stats/contig_chunk_stats/Contig2.mis0.gz" };
    for (const std::string &fname : out_files) {
        ASSERT_TRUE(std::filesystem::exists(fname));
    }

    uint32_t len;

    igzstream stats2("/tmp/stats/contig_chunk_stats/Contig2.mis0.gz");
    stats2.read(reinterpret_cast<char *>(&len), 4);
    ASSERT_EQ(5, len);

    std::string contig(len, 'N');
    std::vector<uint16_t> coverage(len);
    std::array<std::vector<uint16_t>, 4> n_bases;
    for (uint32_t i : { 0, 1, 2, 3 }) {
        n_bases[i].resize(len);
    }

    std::vector<uint16_t> num_snps(len);
    std::vector<uint16_t> num_discordant(len);

    std::vector<uint16_t> min_insert_size(len);
    std::vector<uint16_t> max_insert_size(len);
    std::vector<float> mean_insert_size(len);
    std::vector<float> std_dev_insert_size(len);

    std::vector<uint8_t> min_map_qual(len);
    std::vector<uint8_t> max_map_qual(len);
    std::vector<float> mean_map_qual(len);
    std::vector<float> std_dev_map_qual(len);

    std::vector<int8_t> min_al_score(len);
    std::vector<int8_t> max_al_score(len);
    std::vector<float> mean_al_score(len);
    std::vector<float> std_dev_al_score(len);

    std::vector<uint16_t> num_proper_snp(len);
    std::vector<float> gc_percent(len);

    std::vector<uint8_t> misassembly_by_pos(len);

    stats2.read(reinterpret_cast<char *>(contig.data()), len);
    stats2.read(reinterpret_cast<char *>(coverage.data()), len * sizeof(coverage[0]));
    for (uint32_t i : { 0, 1, 2, 3 }) {
        stats2.read(reinterpret_cast<char *>(n_bases[i].data()), len * sizeof(n_bases[i][0]));
    }
    stats2.read(reinterpret_cast<char *>(num_snps.data()), len * sizeof(num_snps[0]));
    stats2.read(reinterpret_cast<char *>(num_discordant.data()), len * sizeof(num_discordant[0]));

    stats2.read(reinterpret_cast<char *>(min_insert_size.data()), len * sizeof(min_insert_size[0]));
    stats2.read(reinterpret_cast<char *>(mean_insert_size.data()),
                len * sizeof(mean_insert_size[0]));
    stats2.read(reinterpret_cast<char *>(std_dev_insert_size.data()),
                len * sizeof(std_dev_insert_size[0]));
    stats2.read(reinterpret_cast<char *>(max_insert_size.data()), len * sizeof(max_insert_size[0]));

    stats2.read(reinterpret_cast<char *>(min_map_qual.data()), len * sizeof(min_map_qual[0]));
    stats2.read(reinterpret_cast<char *>(mean_map_qual.data()), len * sizeof(mean_map_qual[0]));
    stats2.read(reinterpret_cast<char *>(std_dev_map_qual.data()),
                len * sizeof(std_dev_map_qual[0]));
    stats2.read(reinterpret_cast<char *>(max_map_qual.data()), len * sizeof(max_map_qual[0]));

    stats2.read(reinterpret_cast<char *>(min_al_score.data()), len * sizeof(min_al_score[0]));
    stats2.read(reinterpret_cast<char *>(mean_al_score.data()), len * sizeof(mean_al_score[0]));
    stats2.read(reinterpret_cast<char *>(std_dev_al_score.data()),
                len * sizeof(std_dev_al_score[0]));
    stats2.read(reinterpret_cast<char *>(max_al_score.data()), len * sizeof(max_al_score[0]));

    stats2.read(reinterpret_cast<char *>(num_proper_snp.data()), len * sizeof(num_proper_snp[0]));
    stats2.read(reinterpret_cast<char *>(gc_percent.data()), len * sizeof(gc_percent[0]));
    stats2.read(reinterpret_cast<char *>(misassembly_by_pos.data()),
                len * sizeof(misassembly_by_pos[0]));

    for (uint32_t i = 0; i < len; ++i) {
        ASSERT_EQ('A', contig[i]) << "Position: " << i;
        ASSERT_EQ(0, num_proper_snp[i]);
        ASSERT_EQ(0, coverage[i]);
        ASSERT_EQ(num_discordant[i], 0);

        uint16_t base_counts[] = { n_bases[0][i], n_bases[1][i], n_bases[2][i], n_bases[3][i] };
        ASSERT_THAT(base_counts, ElementsAre(0, 0, 0, 0));
        ASSERT_EQ(127, min_al_score[i]);
        ASSERT_EQ(127, max_al_score[i]);
        ASSERT_TRUE(std::isnan(mean_al_score[i]));

        ASSERT_EQ(0, gc_percent[i]);
        ASSERT_EQ(0, num_snps[i]);
        ASSERT_EQ(std::numeric_limits<uint8_t>::max(), min_map_qual[i]);
        ASSERT_EQ(std::numeric_limits<uint8_t>::max(), max_map_qual[i]);
        ASSERT_TRUE(std::isnan(mean_map_qual[i]));
        ASSERT_EQ(1, misassembly_by_pos[i]);
    }
}

// Make sure that the chunks stats (data around breaking points are correct
TEST(WriteStats, TwoReadsChunkStatsWithOffset) {
    std::unordered_map<std::string, std::vector<MisassemblyInfo>> mi_info
            = parse_misassembly_info("data/test.mis_contigs.info");
    const uint32_t breakpoint_pos = mi_info["Contig2"][0].break_start;
    ASSERT_EQ(10, breakpoint_pos);
    ASSERT_EQ(10, mi_info["Contig2"][0].break_end);

    constexpr uint32_t chunk_size = 5;
    constexpr uint32_t breakpoint_offset = 5;
    StatsWriter stats_writer("/tmp/stats/", chunk_size, breakpoint_offset);

    std::vector<uint32_t> expected_coverage(breakpoint_pos + breakpoint_offset + chunk_size / 2 + 1,
                                            0);
    for (uint32_t i = 0; i < 5; ++i) {
        expected_coverage[i] = 2;
    }

    std::string contig_names[] = { "Contig2", "Contig1" };
    std::string fasta_files[] = { "data/test2.fa.gz", "data/test.fa" };
    std::string bam_files[] = { "data/test2.bam", "data/test1.bam" };
    int32_t offset = 0;
    for (uint32_t rep = 0; rep < 10; ++rep) {
        for (uint32_t i : { 0, 1 }) {
            std::string reference_seq = get_sequence(fasta_files[i], contig_names[i]);
            std::vector<Stats> stats
                    = contig_stats(contig_names[i], reference_seq, bam_files[i], 4, false);
            QueueItem item = { std::move(stats), contig_names[i], reference_seq };
            stats_writer.write_stats(std::move(item), "metaSpades", mi_info[contig_names[i]]);
            if (i == 0) {
                offset = stats_writer.offsets[0];
                ASSERT_EQ(1, stats_writer.offsets.size());
                ASSERT_TRUE(offset >= -5 && offset <= 5);
            } else {
                ASSERT_EQ(0, stats_writer.offsets.size());
            }
        }
        stats_writer.write_summary();

        std::string out_files[] = { "/tmp/stats/contig_chunk_stats/Contig1.ok.gz",
                                    "/tmp/stats/contig_chunk_stats/Contig2.mis0.gz" };
        for (const std::string &fname : out_files) {
            ASSERT_TRUE(std::filesystem::exists(fname));
        }


        uint32_t len;

        igzstream stats2("/tmp/stats/contig_chunk_stats/Contig2.mis0.gz");
        stats2.read(reinterpret_cast<char *>(&len), 4);
        ASSERT_EQ(5, len);

        std::string contig(len, 'N');
        std::vector<uint16_t> coverage(len);
        std::array<std::vector<uint16_t>, 4> n_bases;
        for (uint32_t i : { 0, 1, 2, 3 }) {
            n_bases[i].resize(len);
        }
        std::vector<uint16_t> num_snps(len);

        stats2.read(reinterpret_cast<char *>(contig.data()), len);
        stats2.read(reinterpret_cast<char *>(coverage.data()), len * sizeof(coverage[0]));
        for (uint32_t i : { 0, 1, 2, 3 }) {
            stats2.read(reinterpret_cast<char *>(n_bases[i].data()), len * sizeof(n_bases[i][0]));
        }
        stats2.read(reinterpret_cast<char *>(num_snps.data()), len * sizeof(num_snps[0]));


        for (uint32_t i = 0; i < len; ++i) {
            ASSERT_EQ('A', contig[i]) << "Position: " << i;
            assert(i + breakpoint_pos + offset > chunk_size / 2);
            ASSERT_EQ(expected_coverage[i + breakpoint_pos + offset - chunk_size / 2], coverage[i]);
            // uint16_t base_counts[] = { n_bases[0][i], n_bases[1][i], n_bases[2][i], n_bases[3][i]
            // }; ASSERT_THAT(base_counts, ElementsAre(0, 0, 0, 0));
        }
    }
}
} // namespace
