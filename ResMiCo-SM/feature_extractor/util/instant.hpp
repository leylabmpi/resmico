#pragma once

#include "duration.hpp"

#include <ostream>
#include <stdint.h>

namespace util {

/**
 * Class representing an instant of time.
 */
class Instant {
  private:
    /**
     * The time in nanoseconds.
     */
    uint64_t instant_ns;

    /**
     * Private constructor; to create an instance of the class, use nsec() or usec().
     */
    Instant(const uint64_t instant_ns) : instant_ns(instant_ns) {}

  public:
    /**
     * Default constructor, setting time to 0.
     */
    Instant() : instant_ns(0) {};

    /**
     * Factory function to create an instance by specifying the time in nanoseconds.
     */
    static Instant nsec(const uint64_t instant_ns);

    /**
     * Factory function to create an instance by specifying the time in microseconds.
     */
    static Instant usec(const uint64_t instant_us);

    /**
     * Factory function to create an instance by specifying the time in milliseconds.
     */
    static Instant msec(const uint64_t instant_ms);

    /**
     * Get time in nanoseconds.
     */
    uint64_t nsec() const;

    /**
     * Get time in microseconds.
     */
    template <typename T>
    T usec() const {
        return T(instant_ns / NANOSECONDS_PER_MICROSECOND)
                + T(instant_ns % NANOSECONDS_PER_MICROSECOND) / T(NANOSECONDS_PER_MICROSECOND);
    }

    /**
     * Get time in milliseconds.
     */
    template <typename T>
    T msec() const {
        return T(instant_ns / NANOSECONDS_PER_MILLISECOND)
                + T(instant_ns % NANOSECONDS_PER_MILLISECOND) / T(NANOSECONDS_PER_MILLISECOND);
    }

    /**
     * Operators
     */
    bool operator<(const Instant &other) const;
    bool operator<=(const Instant &other) const;
    bool operator>(const Instant &other) const;
    bool operator>=(const Instant &other) const;
    bool operator==(const Instant &other) const;
    bool operator!=(const Instant &other) const;
    Duration operator-(const Instant &other) const;
    Instant operator+(const Duration &duration) const;
    Instant &operator+=(const Duration &duration);
    Instant operator-(const Duration &duration) const;

    /**
     * For printing.
     */
    friend std::ostream &operator<<(std::ostream &os, const Instant &instant) {
        os << instant.instant_ns;
        return os;
    }
};

} // namespace util
