#include "contig_stats.hpp"
#include "util/fasta_reader.hpp"
#include "util/logger.hpp"
#include "util/util.hpp"

#include <api/BamReader.h>
#include <utils/bamtools_fasta.h>

#include <cmath>
#include "util/filesystem.hpp"

std::pair<double, double> entropy_gc_percent(const std::array<uint8_t, 4> &counts) {
    uint32_t sum = counts[0] + counts[1] + counts[2] + counts[3];
    if (sum == 0) {
        return { 0, 0 };
    }
    double ent = 0;
    for (uint32_t j : { 0, 1, 2, 3 }) {
        double prob = static_cast<double>(counts[j]) / sum;
        // round corner case that would cause math domain error
        if (prob != 0) {
            ent += prob * std::log2(prob);
        }
    }
    double gc_percent = (counts[1] + counts[2]) / static_cast<double>(sum);
    return { -1 * ent, gc_percent };
}

void fill_seq_entropy(const std::string &seq, uint32_t window_size, std::vector<Stats> *stats) {
    assert(seq.size() == stats->size());

    if (seq.empty()) {
        return;
    }

    uint32_t midpoint = seq.size() / 2;
    if (window_size > midpoint) {
        window_size = midpoint;
    }
    // 1st half (forward)
    std::array<uint8_t, 4> counts = { 0, 0, 0, 0 };

    for (int32_t i = 0; i < static_cast<int32_t>(window_size - 1); ++i) {
        counts[IDX[(int)seq[i]]]++;
    }
    for (uint32_t i = window_size - 1; i < midpoint + window_size - 1; ++i) {
        counts[IDX[(int)seq[i]]]++;
        uint32_t cur_idx = i + 1 - window_size;
        std::tie(stats->at(cur_idx).entropy, stats->at(cur_idx).gc_percent)
                = entropy_gc_percent(counts);
        assert(counts[IDX[(int)seq[cur_idx]]] > 0);
        counts[IDX[(int)seq[cur_idx]]]--;
    }

    counts = { 0, 0, 0, 0 };

    for (uint32_t i = midpoint - window_size + 1; i < midpoint; ++i) {
        counts[IDX[(int)seq[i]]]++;
    }
    // 2nd half (reverse)
    for (uint32_t i = midpoint; i < seq.size(); ++i) {
        counts[IDX[(int)seq[i]]]++;
        std::tie(stats->at(i).entropy, stats->at(i).gc_percent) = entropy_gc_percent(counts);
        assert(window_size == 0 || counts[IDX[(int)seq[i - window_size + 1]]] > 0);
        counts[IDX[(int)seq[i - window_size + 1]]]--;
    }
}

uint32_t Stats::num_snps() const {
    uint32_t result = 0;
    for (uint32_t i : { 0, 1, 2, 3 }) {
        if (i != IDX[ref_base]) {
            result += n_bases[i];
        }
    }
    return result;
}

std::vector<Stats> pileup_bam(const std::string &reference,
                              const std::string &reference_name,
                              const std::string &bam_file) {
    BamTools::BamReader reader;
    reader.Open(bam_file);
    if (!std::filesystem::exists(bam_file + ".bai")) {
        logger()->error(
                "Bam file {} has no index. Please run samtools index to create an index, otherwise "
                "I can't be fast",
                bam_file);
        std::exit(1);
    }
    int32_t ref_id = reader.GetReferenceID(reference_name);
    if (ref_id == -1) {
        logger()->error("Reference with name {} not found in {}", reference_name, bam_file);
        std::exit(1);
    }
    reader.OpenIndex(bam_file + ".bai");
    if (!reader.Jump(ref_id, 0)) {
        logger()->warn("Could not jump to contig {} in file {}", ref_id, bam_file);
    }

    uint32_t contig_len = reader.GetReferenceData()[ref_id].RefLength;
    assert(contig_len == reference.size());
    std::vector<Stats> result(contig_len);

    // iterate through all alignments, only keeping ones with high map quality
    BamTools::BamAlignment al;
    while (reader.GetNextAlignmentCore(al)) {
        if (al.RefID != ref_id) {
            return result; // all data for the given contig was processed
        }

        al.BuildCharData();

        uint32_t offset = 0; // if the CIGAR string contains inserts, we need to adjust the offset
        uint32_t del_offset = 0; // TODO: unused, remove it
        uint32_t cigar_idx = 0;
        // skip soft/hard clips
        while (al.CigarData[cigar_idx].Type == 'H' || al.CigarData[cigar_idx].Type == 'S') {
            cigar_idx++;
        }
        // the last position in the current cigar chunk
        uint32_t cigar_end = al.CigarData[cigar_idx].Length;
        for (uint32_t i = 0; i + offset < al.AlignedBases.size(); ++i) {
            // check if we stepped outside the current CIGAR chunk
            while (i >= cigar_end) {
                cigar_idx++;
                assert(cigar_idx < al.CigarData.size());

                // if the aligned string has inserted bases relative to the reference, we simply
                // skip those bases by increasing the offset and move on to the next CIGAR chunk,
                // if there is one
                if (al.CigarData.at(cigar_idx).Type == 'I') {
                    offset += al.CigarData[cigar_idx].Length;
                    // on rare occasions, the last chunk in a cigar string is an I; if that's the
                    // case we are done with the current alignment
                    if (i + offset >= al.AlignedBases.size()) {
                        assert(cigar_idx == al.CigarData.size() - 1);
                        break;
                    }
                    continue;
                } else if (al.CigarData[cigar_idx].Type == 'D'
                           || al.CigarData[cigar_idx].Type == 'N') {
                    // deleted bases don't show up in AlignedBases, but they do show up in Qualities
                    del_offset += al.CigarData[cigar_idx].Length;
                }
                cigar_end += al.CigarData[cigar_idx].Length;
            }

            if (i + offset >= al.AlignedBases.size()) {
                assert(cigar_idx == al.CigarData.size() - 1);
                break;
            }

            uint8_t base = IDX[(uint8_t)al.AlignedBases[i + offset]];

            Stats &stat = result.at(al.Position + i);
            stat.ref_base = reference[al.Position + i];
            bool is_snp = base != IDX[stat.ref_base];
            if (al.IsPaired() && al.IsMapped()) {
                if (!al.IsProperPair() && al.IsMateMapped()) {
                    stat.s[is_snp].n_discord++;
                    stat.n_discord++;
                } else if (al.IsProperPair() && al.IsMateMapped()) {
                    stat.s[is_snp].n_proper++;
                } else if (al.IsMateMapped() && al.IsReverseStrand() != al.IsMateReverseStrand()) {
                    stat.s[is_snp].n_diff_strand++;
                } else if (!al.IsMateMapped()) {
                    stat.s[is_snp].n_orphan++;
                }
            }
            // insert size
            stat.s[is_snp].i_sizes.push_back(std::abs(al.InsertSize));

            constexpr uint32_t BAM_FSUPPLEMENTARY = 2048;
            // sup/sec reads
            if (al.AlignmentFlag & BAM_FSUPPLEMENTARY) {
                stat.s[is_snp].n_sup++;
            }

            if (!al.IsPrimaryAlignment()) {
                stat.s[is_snp].n_sec++;
            }
            stat.s[is_snp].map_quals.push_back(al.MapQuality);

            // make sure we have a '-' on a deleted position
            assert(al.CigarData[cigar_idx].Type != 'D' || al.AlignedBases[i + offset] == '-');
            // make sure we have a 'N' on an alignment gap position
            assert(al.CigarData[cigar_idx].Type != 'N' || al.AlignedBases[i + offset] == 'N');

            if (base == 5) { // probably an 'N'
                continue;
            }

            result.at(al.Position + i).n_bases[base]++;
        }
    }

    return result;
}

std::vector<Stats> contig_stats(const std::string &reference_name,
                                const std::string &reference_seq,
                                const std::string &bam_file,
                                uint32_t window_size,
                                bool is_short) {
    logger()->info("Processing contig: {}", reference_name);

    logger()->info("Getting per-read characteristics");
    std::vector<Stats> stats = pileup_bam(reference_seq, reference_name, bam_file);

    // aggregate data
    if (!is_short) {
        for (uint32_t pos = 0; pos < reference_seq.size(); ++pos) {
            Stats &stat = stats[pos];

            for (bool snp_match : { true, false }) {
                // insert sizes
                const std::vector<uint16_t> &i_sizes = stat.s[snp_match].i_sizes;
                if (!i_sizes.empty()) {
                    std::tie(stat.s[snp_match].min_i_size, stat.s[snp_match].mean_i_size,
                             stat.s[snp_match].max_i_size)
                            = min_mean_max(i_sizes);
                    stat.s[snp_match].std_dev_i_size
                            = std_dev(i_sizes, stat.s[snp_match].mean_i_size);
                }

                //  Mapping Quality
                const std::vector<uint8_t> &map_quals = stat.s[snp_match].map_quals;
                if (!map_quals.empty()) {
                    std::tie(stat.s[snp_match].min_map_qual, stat.s[snp_match].mean_map_qual,
                             stat.s[snp_match].max_map_qual)
                            = min_mean_max(map_quals);
                    stat.s[snp_match].std_dev_map_qual
                            = std_dev(map_quals, stat.s[snp_match].mean_map_qual);
                }
            }
        }
    }

    logger()->info("Computing entropy and GC percent");
    fill_seq_entropy(reference_seq, window_size, &stats);
    logger()->info("Done");
    return stats;
}

std::string get_sequence(const std::string &fasta_file, const std::string &seq_name) {
    if (!std::filesystem::exists(fasta_file)) {
        logger()->error("File {} does not exist", seq_name);
        std::exit(1);
    }
    std::string reference_seq;
    if (ends_with(fasta_file, "gz")) {
        logger()->info("Gzipped fasta file detected. Using kseq");
        reference_seq = FastaReader(fasta_file).read(seq_name);
    } else {
        logger()->info("Uncompressed fasta file detected. Using BamTools");
        BamTools::Fasta fasta;
        fasta.Open(fasta_file);
        if (!fasta.GetSequence(seq_name, reference_seq)) {
            logger()->error("Sequence not found: {}", seq_name);
            std::exit(1);
        }
    }
    return reference_seq;
}
