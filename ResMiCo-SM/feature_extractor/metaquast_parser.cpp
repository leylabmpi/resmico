#include "metaquast_parser.hpp"

#include "util/logger.hpp"
#include "util/util.hpp"

#include <filesystem>
#include <fstream>
#include <vector>


/**
 * Get the lengths of all mis-assembled contigs. The contig lengths are needed to map positions
 * from the reverse strand to the forward strand (via contig_len-pos)
 */
std::unordered_map<std::string, uint32_t> contig_lengths(const std::string &fasta_file) {
    std::unordered_map<std::string, uint32_t> contig_lens;
    if (!std::filesystem::exists(fasta_file)) {
        logger()->error("Could not find misassembly fasta: {}", fasta_file);
        std::exit(1);
    }
    std::ifstream f(fasta_file);
    uint32_t seq_len = 0;
    std::string contig_id;
    std::string line;
    while (std::getline(f, line)) {
        rtrim(line);
        if (starts_with(line, ">")) { // sequence ID, new contig starting
            if (contig_id != "") { // not the first contig
                contig_lens[contig_id] = seq_len;
                seq_len = 0;
            }
            contig_id = line.substr(1); // trim the leftmost '>'
        } else { // sequence
            seq_len += line.size();
        }
    }
    // last sequence
    contig_lens[contig_id] = seq_len;
    return contig_lens;
}

inline std::pair<uint32_t, uint32_t>
get_overlap(uint32_t &start1, uint32_t &end1, uint32_t &start2, uint32_t &end2) {
    if (start1 > end1) {
        std::swap(start1, end1);
    }
    if (start2 > end2) {
        std::swap(start2, end2);
    }
    return start1 < start2 ? std::make_pair(start2, end1) : std::make_pair(start1, end2);
}

std::vector<MisassemblyInfo> parse_line(const std::string &line) {
    // parse the metaQUAST info out of line
    uint32_t btw_pos = line.find(" between ");
    uint32_t type_start = std::string("Extensive misassembly ( ").size();
    uint32_t type_stop = line.find(' ', type_start);
    if (line[type_stop - 1] == ',') {
        type_stop--;
    }
    MisassemblyInfo mi;
    mi.set_type(line.substr(type_start, type_stop - type_start));
    std::string positions = line.substr(btw_pos + std::string(" between ").size());
    std::istringstream pos_stream(positions);
    std::string token;

    uint32_t start1, start2, end1, end2;
    pos_stream >> start1 >> end1 >> token >> start2 >> end2;
    bool invert1 = false;
    if (start1 > end1) {
        std::swap(start1, end1);
        invert1 = true;
    }
    // if we have an inversion, one interval should map to the reverse and one to the fwd strand
    assert(mi.type != MisassemblyInfo::INVERSION || (invert1 && start2 <= end2)
           || (!invert1 && start2 >= end2));
    if (start2 > end2) {
        std::swap(start2, end2);
    }
    if (start1 > start2) { // make sure intervals are sorted
        std::swap(start1, start2);
        std::swap(end1, end2);
    }
    if (start2 <= end1) { // intervals overlap
        mi.break_start = start2;
        mi.break_end = end1;
        if (mi.type != MisassemblyInfo::INVERSION) {
            mi.start = start1;
            mi.end = end2;
        } else {
            // in case of an inversion, only the interval mapping to the reverse strand is marked as
            // "bad"
            if (invert1) {
                mi.start = start1;
                mi.end = end1;
            } else {
                mi.start = start2;
                mi.end = end2;
            }
        }
        return { mi };
    }
    // intervals don't overlap

    mi.start = start1;
    mi.end = end1;
    mi.break_start = end1;
    mi.break_end = end1;

    MisassemblyInfo mi2 = mi;
    mi2.start = start2;
    mi2.end = end2;
    mi2.break_start = start2;
    mi2.break_end = start2;
    if (mi.type != MisassemblyInfo::INVERSION) {
        return { mi, mi2 };
    } else {
        return invert1 ? std::vector({ mi }) : std::vector({ mi2 });
    }
}

/**
 * Converting metaQUAST extensive misassembly report file to an interval tree.
 * Metaquast special encodings:
 *      (relocation, inconsistency = 278087)
 *      (relocation, inconsistency = -129 [linear representation of circular genome])
 * @param report_file the name of the metaQuast report file
 * @param contig_lens maps misassembled contig name to contig length
 * @return {contigID : itree}, where itree[start:end] : [misassembly_type, inverted_positions?, pos1
 * or pos2?]
 * @note MetaQUAST position info is 1-indexed
 */
std::unordered_map<std::string, std::vector<MisassemblyInfo>>
parse_misassembly_info(const std::string &report_file) {
    std::unordered_map<std::string, std::vector<MisassemblyInfo>> contig_to_misassembly;
    if (!std::filesystem::exists(report_file)) {
        logger()->error("Could not find metaQUAST misassembly info file: {}", report_file);
        std::exit(1);
    }
    std::ifstream f(report_file);
    std::string contig_id;

    std::string line;
    while (std::getline(f, line)) {
        if (starts_with(line, "Extensive misassembly")) {
            std::vector<MisassemblyInfo> mi = parse_line(line);
            contig_to_misassembly[contig_id].insert(contig_to_misassembly[contig_id].end(),
                                                    mi.begin(), mi.end());
        } else {
            contig_id = line;
        }
    }
    return contig_to_misassembly;
}

std::vector<uint8_t> expand(uint32_t contig_length, const std::vector<MisassemblyInfo> &mis) {
    std::vector<uint8_t> result(contig_length);
    for (const MisassemblyInfo &mi : mis) {
        assert(mi.start > 0 && mi.start <= contig_length && mi.end <= contig_length);
        // positions in metaQuast are 1-based
        std::for_each(result.begin() + mi.start - 1, result.begin() + mi.end,
                      [&mi](uint8_t &v) { v |= (mi.type + 1); });
    }
    return result;
}

std::string type_to_string(uint8_t t) {
    if (t == 0) {
        return "None";
    }
    std::string result;
    if (t & 1) {
        result += "relocation,";
    }
    if (t & 2) {
        result += "translocation,";
    }
    if (t & 4) {
        result += "interspecies translocation,";
    }
    if (t & 8) {
        result += "inversion,";
    }
    return result.substr(0, result.size() - 1);
}
