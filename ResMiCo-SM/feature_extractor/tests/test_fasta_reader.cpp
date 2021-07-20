#include "util/fasta_reader.hpp"

#include <gtest/gtest.h>

TEST(FastaReader, Empty) {
  FastaReader reader("data/empty.fa.gz");
  ASSERT_EQ("", reader.read("foo"));
}

TEST(FastaReader, OneContig) {
    FastaReader reader("data/onecontig.fa.gz");
    ASSERT_EQ("AAAACCCC", reader.read("1"));
    ASSERT_EQ("", reader.read("foo"));
}

TEST(FastaReader, OneContigWrongOrder) {
    FastaReader reader("data/onecontig.fa.gz");
    ASSERT_EQ("", reader.read("foo"));
    ASSERT_EQ("", reader.read("1"));
}


TEST(FastaReader, TwoContigs) {
    FastaReader reader("data/twocontigs.fa.gz");
    ASSERT_EQ("GGGGGTTTTT", reader.read("2"));
    ASSERT_EQ("", reader.read("foo"));
}
