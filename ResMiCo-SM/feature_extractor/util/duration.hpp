#pragma once

#include "logger.hpp"

#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <limits>
#include <ostream>
#include <stdint.h>

namespace util {

/**
 * Factors for conversion between seconds, milliseconds, microseconds, nanoseconds.
 */
static constexpr int64_t NANOSECONDS_PER_SECOND = 1e9;
static constexpr int64_t NANOSECONDS_PER_MILLISECOND = 1e6;
static constexpr int64_t NANOSECONDS_PER_MICROSECOND = 1e3;

/**
 * Class representing a duration.
 */
class Duration {
  private:
    /**
     * The duration in nanoseconds.
     */
    int64_t duration_ns;

    /**
     * Private constructor; to create an instance of the class, use nsec(), msec() or sec().
     */
    Duration(const int64_t duration_ns) : duration_ns(duration_ns) {}

  public:
    /**
     * Factory function to create an instance by specifying the duration in nanoseconds.
     */
    static Duration nsec(const int64_t duration_ns) { return Duration(duration_ns); }

    /**
     * Factory function to create an instance by specifying the duration in microseconds.
     */
    static Duration usec(const int64_t duration_us) {
        return Duration(NANOSECONDS_PER_MICROSECOND * duration_us);
    }

    /**
     * Factory function to create an instance by specifying the duration in milliseconds.
     */
    static Duration msec(const int64_t duration_ms) {
        return Duration(NANOSECONDS_PER_MILLISECOND * duration_ms);
    }

    /**
     * Factory function to create an instance by specifying the duration in seconds.
     */
    static Duration sec(const double duration_s) {
        if (duration_s > std::numeric_limits<double>::max()
            || duration_s < std::numeric_limits<int64_t>::lowest()) {
            const Duration result(duration_s > 0.0 ? std::numeric_limits<int64_t>::max()
                                                   : std::numeric_limits<int64_t>::lowest());
            logger()->warn(
                    "Duration of {} seconds exceeds capacity of int64_t. Setting duration to {} "
                    "nanoseconds.",
                    duration_s, result.nsec());
            return result;
        }
        return Duration(NANOSECONDS_PER_SECOND * duration_s);
    }

    /**
     * Get duration in nanoseconds.
     */
    int64_t nsec() const { return duration_ns; }

    /**
     * Get duration in microseconds.
     */
    template <typename T>
    T usec() const {
        return T(duration_ns / NANOSECONDS_PER_MICROSECOND)
                + T(duration_ns % NANOSECONDS_PER_MICROSECOND) / T(NANOSECONDS_PER_MICROSECOND);
    }

    /**
     * Get duration in milliseconds.
     */
    template <typename T>
    T msec() const {
        return T(duration_ns / NANOSECONDS_PER_MILLISECOND)
                + T(duration_ns % NANOSECONDS_PER_MILLISECOND) / T(NANOSECONDS_PER_MILLISECOND);
    }

    /**
     * Get duration in seconds.
     */
    template <typename T>
    T sec() const {
        return T(duration_ns / NANOSECONDS_PER_SECOND)
                + T(duration_ns % NANOSECONDS_PER_SECOND) / T(NANOSECONDS_PER_SECOND);
    }

    /**
     * Operators
     */
    bool operator>(const Duration &other) const { return duration_ns > other.duration_ns; }
    bool operator>=(const Duration &other) const { return duration_ns >= other.duration_ns; }
    bool operator<(const Duration &other) const { return duration_ns < other.duration_ns; }
    bool operator<=(const Duration &other) const { return duration_ns <= other.duration_ns; }
    bool operator==(const Duration &other) const { return duration_ns == other.duration_ns; }
    bool operator!=(const Duration &other) const { return duration_ns != other.duration_ns; }
    Duration operator+(const Duration &other) const {
        return Duration(duration_ns + other.duration_ns);
    }
    Duration operator-(const Duration &other) const {
        return Duration(duration_ns - other.duration_ns);
    }
    Duration operator*(const int64_t factor) const { return Duration(duration_ns * factor); }
    Duration operator*(const double factor) const {
        const double val = duration_ns * factor + 0.5;

        // Overflow checks before casting back to integer
        const double upper = std::nexttoward(std::numeric_limits<int64_t>::max(), 0);
        const double lower = std::nexttoward(std::numeric_limits<int64_t>::min(), 0);

        if (val > upper) {
            return Duration(std::numeric_limits<int64_t>::max());
        } else if (val < lower) {
            return Duration(std::numeric_limits<int64_t>::min());
        } else {
            return Duration(val);
        }
    }
    Duration operator/(const int64_t denom) const { return Duration(duration_ns / denom); }
    Duration operator/(const double denom) const {
        const double val = duration_ns / denom + 0.5;

        // Overflow checks before casting back to integer
        const double upper = std::nexttoward(std::numeric_limits<int64_t>::max(), 0);
        const double lower = std::nexttoward(std::numeric_limits<int64_t>::min(), 0);

        if (val > upper) {
            return Duration(std::numeric_limits<int64_t>::max());
        } else if (val < lower) {
            return Duration(std::numeric_limits<int64_t>::min());
        } else {
            return Duration(val);
        }
    }
    double operator/(const Duration &other) const {
        return static_cast<double>(duration_ns / other.duration_ns)
                + static_cast<double>(duration_ns % other.duration_ns) / other.duration_ns;
    }

    Duration abs() const { return Duration(std::abs(duration_ns)); }

    /**
     * For printing.
     */
    friend std::ostream &operator<<(std::ostream &os, const Duration &duration) {
        os << duration.duration_ns;
        return os;
    }
};

} // namespace util
