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

// Use this for Python compatibility
std::string round2_python(float v) {
    if (std::isnan(v)) {
        return "NA";
    }
    if (v == 0) {
        return "0.0";
    }
    const std::string &decimals = std::to_string(static_cast<int>(std::round(v * 10)) % 10);
    if (decimals == "0") {
        return std::to_string(int(v));
    }
    return std::to_string(static_cast<int>(v * 10) / 10) + '.' + decimals;
}


#include <iostream> // TODO: remove

std::string round2(float v) {
    if (std::isnan(v)) {
        return "NA";
    }
    const std::string &decimals = std::to_string(static_cast<int>(std::round(v * 100)) % 100);
    return std::to_string(static_cast<int>(std::abs(v) * 100) / 100) + '.'
            + std::string(2 - decimals.length(), '0') + decimals;
}
