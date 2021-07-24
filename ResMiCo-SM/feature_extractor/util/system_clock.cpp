#include "system_clock.hpp"

namespace util {

void SystemClock::registerThread(const char *name) {
    // No-op
}

Instant SystemClock::getRealTime() {
    const uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count();
    return Instant::nsec(ns);
}

Instant SystemClock::getMonotonicTime() {
    const uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();
    return Instant::nsec(ns);
}

void SystemClock::sleep(Duration duration) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(duration.nsec()));
}

std::cv_status SystemClock::waitForCondition(std::unique_lock<std::mutex> *const lock,
                                             std::condition_variable *const cv,
                                             Duration duration) {
    return cv->wait_for(*lock, std::chrono::nanoseconds(duration.nsec()));
}

int SystemClock::poll(pollfd *pfds, int npfds, Duration timeout) {
    const int timeout_ms = timeout.msec<int>();
    return ::poll(pfds, npfds, timeout_ms);
}

} // namespace util
