#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

constexpr uint8_t IDX[128]
        = { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 5, 1, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 5, 1, 5, 5, 5, 2,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };

/** Calculate Shannon entropy of sequence. */
std::pair<double, double> entropy_gc_percent(const std::array<uint8_t, 4> &counts);

/**
 * The statistics computed for a position in the contig sequence.
 */
struct Stats {
    uint8_t ref_base; // the base (ACGT) in the reference contig

    std::array<uint16_t, 4> n_bases; // total number of bases (ACGT) aligned to this position

    // number of SNPs (relative to the reference contig
    uint16_t num_snps() const;

    uint16_t coverage() const { return n_bases[0] + n_bases[1] + n_bases[2] + n_bases[3]; }

    uint16_t n_discord = 0;

    uint16_t min_i_size = std::numeric_limits<uint16_t>::max();
    float mean_i_size = NAN;
    float std_dev_i_size = NAN;
    uint16_t max_i_size = std::numeric_limits<uint16_t>::max();

    uint8_t min_map_qual = std::numeric_limits<uint8_t>::max();
    float mean_map_qual = NAN;
    float std_dev_map_qual = NAN;
    uint8_t max_map_qual = std::numeric_limits<uint8_t>::max();

    uint8_t min_al_score = std::numeric_limits<uint8_t>::max();
    float mean_al_score = NAN;
    float std_dev_al_score = NAN;
    uint8_t max_al_score = std::numeric_limits<uint8_t>::max();

    uint16_t n_proper_match = 0;
    uint16_t n_proper_snp = 0;
    uint16_t n_diff_strand = 0;
    uint16_t n_orphan = 0;
    uint16_t n_sup = 0;
    uint16_t n_sec = 0;
    uint16_t n_discord_match = 0;


    std::vector<int32_t> i_sizes;
    std::vector<uint8_t> map_quals;
    std::vector<uint8_t> al_scores; // alignment scores as computed by BowTie2

    float gc_percent;
    float entropy;
};

/**
 * Calculate the sequence entropy across a sliding window. In order to obtain as many entropies and
 * gcs as there are positions in the sequence, we arbitrarily decide to divide the sequence in the
 * middle. For the first half of the sequence, the GC percent and entropy are computed by looking
 * "forward" #window_size elements. For the second half, we are looking backward "window_size"
 * elements.
 * @param seq py.FastaFile.fetch object
 * @param window_size the window over which the entropy and gc percent are computed
 * @return pair of sequence_entropy, gc-percent, one for each position in #seq
 */
void fill_seq_entropy(const std::string &seq, uint32_t window_size, std::vector<Stats> *stats);

/**
 * Reads data from the given BAM file and counts the number of A/C/G/T bases at each position.
 */
std::vector<Stats> pileup_bam(const std::string &reference,
                              const std::string &reference_name,
                              const std::string &bam_file);

/**
 * Extracting contig-specific info from the contig named #contig_name.
 * @param contigs pysam.AlignmentFile.references object
 * @param bam_file bam file path
 * @param fasta_file bam file path
 * @param assembler which assembler used?
 * @param window_size window size for calculating window-based stats
 * @param is_short  just short feature list?
 * @return vector of #Stats, one for each position in the contig
 */
std::vector<Stats> contig_stats(const std::string &reference_name,
                                const std::string &bam_file,
                                const std::string &fasta_file,
                                uint32_t window_size,
                                bool is_short);

std::string get_sequence(const std::string &fasta_file, const std::string &seq_name);
