#include "stats_writer.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
using namespace ::testing;

TEST(WriteStats, Empty) {
    QueueItem item;
    uint32_t count_mean = 0, count_std_dev = 0;
    std::vector<double> sums(12, 0), sums2(12, 0);
    std::ofstream out;
    std::unordered_map<std::string, std::unique_ptr<ogzstream>> bin_streams
            = get_streams("/tmp/out");
    std::ofstream toc("/tmp/out.toc");

    write_stats(std::move(item), "metaQuast", {}, &out, &toc, &bin_streams, &count_mean,
                &count_std_dev, &sums, &sums2);

    for (const auto &stream : bin_streams) {
        std::string fname = "/tmp/out." + stream.first + ".gz";
        ASSERT_TRUE(std::filesystem::exists(fname));
        igzstream in(fname.c_str());
        char c;
        in.get(c);
        ASSERT_FALSE(in.good());
    }
}

TEST(WriteStats, TwoReads) {
    uint32_t count_mean = 0, count_std_dev = 0;
    std::vector<double> sums(12, 0), sums2(12, 0);
    std::ofstream out;
    std::unordered_map<std::string, std::unique_ptr<ogzstream>> bin_streams
            = get_streams("/tmp/out");
    std::ofstream toc("/tmp/out.toc");

    std::unordered_map<std::string, std::vector<MisassemblyInfo>> mi_info
            = parse_misassembly_info("data/test.mis_contigs.info");
    std::string contig_names[] = { "Contig2", "Contig1" };
    std::string fasta_files[] = { "data/test2.fa.gz", "data/test.fa" };
    std::string bam_files[] = { "data/test2.bam", "data/test1.bam" };
    for (uint32_t i : { 0, 1 }) {
        std::string reference_seq = get_sequence(fasta_files[i], contig_names[i]);
        std::vector<Stats> stats
                = contig_stats(contig_names[i], reference_seq, bam_files[i], 4, false);
        QueueItem item = { std::move(stats), contig_names[i], reference_seq };
        write_stats(std::move(item), "metaSpades", mi_info[contig_names[i]], &out, &toc,
                    &bin_streams, &count_mean, &count_std_dev, &sums, &sums2);
    }

    std::unordered_map<std::string, std::unique_ptr<igzstream>> in_streams;
    for (const auto &stream : bin_streams) {
        bin_streams[stream.first]->close();
        std::string fname = "/tmp/out." + stream.first + ".gz";
        ASSERT_TRUE(std::filesystem::exists(fname));
        in_streams[stream.first] = std::make_unique<igzstream>(fname.c_str());
        ASSERT_TRUE(in_streams[stream.first]->good());
    }

    for (uint32_t i = 0; i < 500; ++i) {
        char base;
        in_streams["ref_base"]->get(base);
        uint16_t n_proper_snp;
        uint16_t coverage;
        ASSERT_EQ(i < 498 ? 'A' : 'C', base) << "Position: " << i;
        in_streams["num_proper_SNP"]->read(reinterpret_cast<char *>(&n_proper_snp), 2);
        in_streams["coverage"]->read(reinterpret_cast<char *>(&coverage), 2);
        uint16_t num_query_A, num_query_C, num_query_G, num_query_T;
        in_streams["num_query_A"]->read(reinterpret_cast<char *>(&num_query_A), 2);
        in_streams["num_query_C"]->read(reinterpret_cast<char *>(&num_query_C), 2);
        in_streams["num_query_G"]->read(reinterpret_cast<char *>(&num_query_G), 2);
        in_streams["num_query_T"]->read(reinterpret_cast<char *>(&num_query_T), 2);
        int8_t min_al_score, max_al_score;
        int16_t mean_al_score;
        in_streams["min_al_score_Match"]->get((char &)min_al_score);
        in_streams["mean_al_score_Match"]->read(reinterpret_cast<char *>(&mean_al_score), 2);
        in_streams["max_al_score_Match"]->get((char &)max_al_score);
        char mi;
        in_streams["misassembly_by_pos"]->get(mi);
        ASSERT_EQ(i >= 420 && i < 425 ? 10'000 : (i > 0 && i < 5 ? 5000 : 0),
                  n_proper_snp)
                << "at position: " << i; // 5000 is 0.5
        ASSERT_EQ(i >= 420 && i < 425 || i >= 0 && i < 5 ? 2 : 0, coverage) << "at position: " << i;

        uint16_t num_discordant;
        in_streams["num_discordant"]->read(reinterpret_cast<char *>(&num_discordant), 2);
        ASSERT_EQ(num_discordant, 0);

        ASSERT_THAT(std::vector<uint16_t>({ num_query_A, num_query_C, num_query_G, num_query_T }),
                    i == 0                ? ElementsAre(10'000, 0, 0, 0)
                    : i < 5               ? ElementsAre(5'000, 0, 5'000, 0)
                    : i >= 420 && i < 425 ? ElementsAre(0, 5'000, 0, 5'000)
                                          : ElementsAre(0, 0, 0, 0));
        ASSERT_EQ(min_al_score, i == 0 ? -28 : i < 5 ? 0 : 127);
        ASSERT_EQ(max_al_score, i < 5 ? 0 : 127);
        ASSERT_EQ(mean_al_score, i == 0 ? -1400 : i < 5 ? 0 : std::numeric_limits<int16_t>::max());

        uint16_t gc_percent;
        in_streams["seq_window_perc_gc"]->read(reinterpret_cast<char *>(&gc_percent), 2);
        ASSERT_EQ(gc_percent, i < 498 ? 0 : i == 498 ? 2500 : 5000) << "Position: " << i;
        uint16_t num_snps;
        in_streams["num_SNPs"]->read(reinterpret_cast<char *>(&num_snps), 2);
        ASSERT_EQ(num_snps, i > 0 && i < 5 ? 5'000 : i >= 420 && i < 425 ? 10'000 : 0);
        uint8_t min_mapq, max_mapq;
        int16_t mean_mapq;
        in_streams["min_mapq_Match"]->get((char &)min_mapq);
        in_streams["mean_mapq_Match"]->read(reinterpret_cast<char *>(&mean_mapq), 2);
        in_streams["max_mapq_Match"]->get((char &)max_mapq);
        ASSERT_EQ(min_mapq, i < 5 ? 6 : std::numeric_limits<uint8_t>::max());
        ASSERT_EQ(max_mapq, i == 0 ? 7 : i < 5 ? 6 : std::numeric_limits<uint8_t>::max());
        ASSERT_EQ(mean_mapq, i == 0 ? 650 : i < 5 ? 600 : std::numeric_limits<int16_t>::max());
        ASSERT_EQ(mi, i < 9 || i >= 30 ? 0 : 1) << "Position " << i;
    }

    // check the table of contents file, and make sure that each contig has length 500, the first
    // contig is labeled as misassembled, and the 2nd contig is not
    std::ifstream toc_read("/tmp/out.toc");
    for (uint32_t i : { 0, 1 }) {
        std::string assembler, contig;
        uint32_t length, is_missasembly;
        toc_read >> assembler >> contig >> length >> is_missasembly;
        ASSERT_EQ("metaSpades", assembler);
        ASSERT_EQ("Contig" + std::to_string(i == 0 ? 2 : 1), contig);
        ASSERT_EQ(500, length);
        ASSERT_EQ(i == 0 ? 1 : 0, is_missasembly);
    }
}
} // namespace
