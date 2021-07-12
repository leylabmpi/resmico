#include "util.hpp"

bool starts_with(std::string const &value, std::string const &prefix) {
    if (prefix.size() > value.size()) {
        return false;
    }
    return std::equal(prefix.begin(), prefix.end(), value.begin());
}


bool ends_with(std::string const &value, std::string const &ending) {
    if (ending.size() > value.size()) {
        return false;
    }
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}
