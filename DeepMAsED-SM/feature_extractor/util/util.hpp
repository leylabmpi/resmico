#pragma once

#include <sstream>
#include <string>
#include <vector>

#include <cmath>

/**
 * Join a vector's element into sep-separated string.
 */
template <typename T>
std::string join_vec(const std::vector<T> &vec, char sep = ',') {
    std::stringstream out;
    if (vec.empty()) {
        return "";
    }

    for (uint32_t i = 0; i < vec.size() - 1; ++i) {
        out << vec[i] << sep;
    }
    out << vec.back();
    out << std::endl;
    return out.str();
}

template <typename T>
std::tuple<T, double, T> min_mean_max(const std::vector<T> &v) {
    if (v.empty()) {
        return { 0, 0, 0 };
    }
    T min = v[0];
    T max = v[0];
    double mean = v[0];
    for (uint32_t i = 1; i < v.size(); ++i) {
        if (v[i] < min) {
            min = v[i];
        }
        if (v[i] > max) {
            max = v[i];
        }
        mean += v[i];
    }
    return { min, mean / v.size(), max };
}

template <typename T>
double std_dev(const std::vector<T> &v, double mean) {
    if (v.size() < 2) {
        return NAN;
    }
    double var = 0;
    for (T el : v) {
        var += (el - mean) * (el - mean);
    }
    var /= (v.size()-1);
    return sqrt(var);
}

bool starts_with(std::string const &value, std::string const &prefix);
bool ends_with(std::string const &value, std::string const &ending);
