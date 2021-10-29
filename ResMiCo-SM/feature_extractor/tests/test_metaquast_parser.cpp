#include "metaquast_parser.hpp"

#include <gtest/gtest.h>

namespace {
TEST(ParseLine, Inversion2ndReverse) {
    std::vector<MisassemblyInfo> mis
            = parse_line("Extensive misassembly ( inversion ) between 1 467 and 551 458");
    ASSERT_EQ(1, mis.size());
    MisassemblyInfo mi = mis[0];
    ASSERT_EQ(458, mi.start);
    ASSERT_EQ(551, mi.end);

    ASSERT_EQ(458, mi.break_start);
    ASSERT_EQ(467, mi.break_end);

    ASSERT_EQ(MisassemblyInfo::INVERSION, mi.type);
}

TEST(ParseLine, InversionFirstReverse) {
    std::vector<MisassemblyInfo> mis
            = parse_line("Extensive misassembly ( inversion ) between 12494 8289 and 12400 15339");
    ASSERT_EQ(1, mis.size());
    MisassemblyInfo mi = mis[0];
    ASSERT_EQ(8289, mi.start);
    ASSERT_EQ(12494, mi.end);

    ASSERT_EQ(12400, mi.break_start);
    ASSERT_EQ(12494, mi.break_end);

    ASSERT_EQ(MisassemblyInfo::INVERSION, mi.type);
}

TEST(ParseLine, Inversion2ndReverseNoOverlap) {
    std::vector<MisassemblyInfo> mis
            = parse_line("Extensive misassembly ( inversion ) between 1 467 and 551 498");
    ASSERT_EQ(1, mis.size());
    MisassemblyInfo mi = mis[0];
    ASSERT_EQ(498, mi.start);
    ASSERT_EQ(551, mi.end);

    ASSERT_EQ(467, mi.break_start);
    ASSERT_EQ(498, mi.break_end);

    ASSERT_EQ(MisassemblyInfo::INVERSION, mi.type);
}

TEST(ParseLine, InversionFirstReverseNoOverlap) {
    std::vector<MisassemblyInfo> mis
            = parse_line("Extensive misassembly ( inversion ) between 12494 8289 and 12550 15339");
    ASSERT_EQ(1, mis.size());
    MisassemblyInfo mi = mis[0];
    ASSERT_EQ(8289, mi.start);
    ASSERT_EQ(12494, mi.end);

    ASSERT_EQ(12494, mi.break_start);
    ASSERT_EQ(12494, mi.break_end);

    ASSERT_EQ(MisassemblyInfo::INVERSION, mi.type);
}

TEST(ParseLine, RelocationOverlap) {
    std::vector<MisassemblyInfo> mis = parse_line(
            "Extensive misassembly ( relocation, inconsistency = -1933 ) between 568 808 and 787 "
            "1924");
    ASSERT_EQ(1, mis.size());
    MisassemblyInfo mi = mis[0];
    ASSERT_EQ(568, mi.start);
    ASSERT_EQ(1924, mi.end);

    ASSERT_EQ(787, mi.break_start);
    ASSERT_EQ(808, mi.break_end);

    ASSERT_EQ(MisassemblyInfo::RELOCATION, mi.type);
}

TEST(ParseLine, RelocationNoOverlap) {
    std::vector<MisassemblyInfo> mis = parse_line(
            "Extensive misassembly ( relocation, inconsistency = 9143 ) between 607 6 and 4714 "
            "1516");
    ASSERT_EQ(2, mis.size());
    MisassemblyInfo mi = mis[0];
    ASSERT_EQ(6, mi.start);
    ASSERT_EQ(607, mi.end);

    ASSERT_EQ(607, mi.break_start);
    ASSERT_EQ(607, mi.break_end);

    ASSERT_EQ(MisassemblyInfo::RELOCATION, mi.type);

    MisassemblyInfo mi2 = mis[1];
    ASSERT_EQ(1516, mi2.start);
    ASSERT_EQ(4714, mi2.end);

    ASSERT_EQ(1516, mi2.break_start);
    ASSERT_EQ(1516, mi2.break_end);

    ASSERT_EQ(MisassemblyInfo::RELOCATION, mi2.type);
}

TEST(ParseLine, TranslocationOverlap) {
    std::vector<MisassemblyInfo> mis
            = parse_line("Extensive misassembly ( translocation ) between 1 35306 and 35286 35922");
    ASSERT_EQ(1, mis.size());
    MisassemblyInfo mi = mis[0];
    ASSERT_EQ(1, mi.start);
    ASSERT_EQ(35922, mi.end);

    ASSERT_EQ(35286, mi.break_start);
    ASSERT_EQ(35306, mi.break_end);

    ASSERT_EQ(MisassemblyInfo::TRANSLOCATION, mi.type);
}

TEST(ParseLine, InterspeciesTranslocationOverlap) {
    std::vector<MisassemblyInfo> mis = parse_line(
            "Extensive misassembly ( interspecies translocation ) between 1 30794 and 31583 30771");
    ASSERT_EQ(1, mis.size());
    MisassemblyInfo mi = mis[0];
    ASSERT_EQ(1, mi.start);
    ASSERT_EQ(31583, mi.end);

    ASSERT_EQ(30771, mi.break_start);
    ASSERT_EQ(30794, mi.break_end);

    ASSERT_EQ(MisassemblyInfo::INTERSPECIES_TRANSLOCATION, mi.type);
}

TEST(ParseMisassemblyInfo, File) {
    std::unordered_map<std::string, std::vector<MisassemblyInfo>> mi_info
            = parse_misassembly_info("data/metaQUAST.mis_contigs.info");
    ASSERT_EQ(14, mi_info.size());

    auto scaffold_1 = mi_info.find("scaffold_1");
    ASSERT_EQ("scaffold_1", scaffold_1->first);
    std::vector<MisassemblyInfo> mis = scaffold_1->second;
    ASSERT_EQ(1, mis.size());

    MisassemblyInfo mi = mis[0];
    ASSERT_EQ(1, mi.start);
    ASSERT_EQ(690424, mi.end);

    ASSERT_EQ(396296, mi.break_start);
    ASSERT_EQ(396778, mi.break_end);

    ASSERT_EQ(MisassemblyInfo::TRANSLOCATION, mi.type);

    auto scaffold_3 = mi_info.find("scaffold_3");
    ASSERT_EQ("scaffold_3", scaffold_3->first);
    std::vector<MisassemblyInfo> mis3 = scaffold_3->second;
    ASSERT_EQ(4, mis3.size());
    std::vector<MisassemblyInfo::Type> expected_types
            = { MisassemblyInfo::RELOCATION, MisassemblyInfo::RELOCATION,
                MisassemblyInfo::TRANSLOCATION, MisassemblyInfo::TRANSLOCATION };
    for (uint32_t i = 0; i < 4; ++i) {
        ASSERT_EQ(expected_types[i], mis3[i].type);
    }
}

/**
 * MetaQUAST changed the output format at some point. This tests that the other output format is
 * also correctly parsed.
 */
TEST(ParseMisassemblyInfo, FileDifferentFormat) {
    std::unordered_map<std::string, std::vector<MisassemblyInfo>> mi_info
            = parse_misassembly_info("data/metaQUAST.mis_contigs2.info");
    ASSERT_EQ(4, mi_info.size());

    auto contig1 = mi_info.find("NODE_3_length_359076_cov_40.230120");
    ASSERT_EQ("NODE_3_length_359076_cov_40.230120", contig1->first);
    std::vector<MisassemblyInfo> mis = contig1->second;
    ASSERT_EQ(1, mis.size());

    MisassemblyInfo mi = mis[0];
    ASSERT_EQ(1, mi.start);
    ASSERT_EQ(359076, mi.end);

    ASSERT_EQ(166932, mi.break_start);
    ASSERT_EQ(166933, mi.break_end);

    ASSERT_EQ(MisassemblyInfo::RELOCATION, mi.type);

    auto contig2 = mi_info.find("NODE_1461_length_3859_cov_3.129075");
    ASSERT_EQ("NODE_1461_length_3859_cov_3.129075", contig2->first);
    std::vector<MisassemblyInfo> mis2 = contig2->second;
    ASSERT_EQ(1, mis2.size());

    MisassemblyInfo mi2 = mis2[0];
    ASSERT_EQ(1, mi2.start);
    ASSERT_EQ(3859, mi2.end);

    ASSERT_EQ(3134, mi2.break_start);
    ASSERT_EQ(3135, mi2.break_end);

    ASSERT_EQ(MisassemblyInfo::TRANSLOCATION, mi2.type);


    auto contig3 = mi_info.find("NODE_2583_length_2796_cov_2.680044");
    ASSERT_EQ("NODE_2583_length_2796_cov_2.680044", contig3->first);
    std::vector<MisassemblyInfo> mis3 = contig3->second;
    ASSERT_EQ(1, mis3.size());

    MisassemblyInfo mi3 = mis3[0];
    ASSERT_EQ(1, mi3.start);
    ASSERT_EQ(2796, mi3.end);

    ASSERT_EQ(1762, mi3.break_start);
    ASSERT_EQ(1763, mi3.break_end);

    ASSERT_EQ(MisassemblyInfo::INTERSPECIES_TRANSLOCATION, mi3.type);

    auto contig4 = mi_info.find("k141_92042");
    ASSERT_EQ("k141_92042", contig4->first);
    std::vector<MisassemblyInfo> mis4 = contig4->second;
    ASSERT_EQ(1, mis4.size());

    MisassemblyInfo mi4 = mis4[0];
    ASSERT_EQ(5, mi4.start);
    ASSERT_EQ(1074, mi4.end);

    ASSERT_EQ(241, mi4.break_start);
    ASSERT_EQ(255, mi4.break_end);

    ASSERT_EQ(MisassemblyInfo::INTERSPECIES_TRANSLOCATION, mi4.type);
}

} // namespace
