#include "instant.hpp"

#include <chrono>

namespace util {

Instant Instant::nsec(const uint64_t instant_ns) {
    return Instant(instant_ns);
}

Instant Instant::usec(const uint64_t instant_us) {
    return Instant(instant_us * NANOSECONDS_PER_MICROSECOND);
}

Instant Instant::msec(const uint64_t instant_ms) {
    return Instant(instant_ms * NANOSECONDS_PER_MILLISECOND);
}

uint64_t Instant::nsec() const {
    return instant_ns;
}

/**
 * Operators
 */
bool Instant::operator<(const Instant &other) const {
    return instant_ns < other.instant_ns;
}
bool Instant::operator<=(const Instant &other) const {
    return instant_ns <= other.instant_ns;
}
bool Instant::operator>(const Instant &other) const {
    return instant_ns > other.instant_ns;
}
bool Instant::operator>=(const Instant &other) const {
    return instant_ns >= other.instant_ns;
}
bool Instant::operator==(const Instant &other) const {
    return instant_ns == other.instant_ns;
}
bool Instant::operator!=(const Instant &other) const {
    return instant_ns != other.instant_ns;
}
Duration Instant::operator-(const Instant &other) const {
    if (other.instant_ns > instant_ns) {
        return Duration::nsec(-static_cast<int64_t>(other.instant_ns - instant_ns));
    }
    return Duration::nsec(instant_ns - other.instant_ns);
}
Instant Instant::operator+(const Duration &duration) const {
    if (duration.nsec() < 0) {
        const uint64_t inverseDuration_ns = static_cast<uint64_t>(-duration.nsec());
        return inverseDuration_ns >= instant_ns ? Instant(0)
                                                : Instant(instant_ns - inverseDuration_ns);
    }
    return Instant(instant_ns + duration.nsec());
}
Instant &Instant::operator+=(const Duration &duration) {
    if (duration.nsec() < 0) {
        const uint64_t inverseDuration_ns = static_cast<uint64_t>(-duration.nsec());
        if (inverseDuration_ns >= instant_ns) {
            instant_ns = 0;
        } else {
            instant_ns -= inverseDuration_ns;
        }
    } else {
        instant_ns += duration.nsec();
    }
    return *this;
}
Instant Instant::operator-(const Duration &duration) const {
    if (duration.nsec() < 0) {
        return Instant(instant_ns + static_cast<uint64_t>(-duration.nsec()));
    }
    return static_cast<uint64_t>(duration.nsec()) >= instant_ns
            ? Instant(0)
            : Instant(instant_ns - duration.nsec());
}

} // namespace util
