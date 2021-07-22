#pragma once

#include "kseq.h"
#include <zlib.h>

#include <functional>
#include <string>

class FastaReader {
    gzFile fp;
    FastaReaderC<gzFile> readerC;

    ReadT<gzFile> getReader() {
        return [](gzFile f, void *v, unsigned u) { return gzread(f, v, u); };
    }

  public:
    FastaReader(const std::string &name)
        : fp(gzopen(name.c_str(), "r")), readerC(FastaReaderC<gzFile>(getReader(), fp)) {
        ReadT<gzFile> reader = [](gzFile f, void *v, unsigned u) { return gzread(f, v, u); };
    }

    std::string read(const std::string &name) {
        int l;
        std::string seq;
        std::string seq_name;
        std::tie(l, seq, seq_name) = readerC.get_sequence();
        while (l > 0) {
            if (seq_name == name) {
                return seq;
            }
            std::tie(l, seq, seq_name) = readerC.get_sequence();
        }
        return "";
    }

    ~FastaReader() { gzclose(fp); }
};
