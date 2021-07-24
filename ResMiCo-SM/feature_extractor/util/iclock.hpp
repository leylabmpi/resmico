#pragma once

#include "duration.hpp"
#include "instant.hpp"

#include <condition_variable>
#include <mutex>

#include <sys/poll.h>

namespace util {

/**
 * Interface for a clock. The clock can be used to obtain time,
 * or to wait until a certain time has passed.
 */
class IClock {
  public:
    virtual ~IClock() {}

    /**
     * Register the current thread with the clock under the given name.
     * Use this if you want to use any of the functions sleep(),
     * waitForCondition(), waitForConditionTimed(), or poll() with
     * the current thread.
     *
     * This allows for monitoring the thread's activities when debugging,
     * and also makes scheduling easily testable (see the SteppableClock
     * implementation). This is a no-op under normal operating conditions.
     */
    virtual void registerThread(const char *name) = 0;

    /**
     * Obtain the current instant of real (wall) time.
     */
    virtual Instant getRealTime() = 0;

    /**
     * Obtain the current instant of monotonic time.
     */
    virtual Instant getMonotonicTime() = 0;

    /**
     * Sleep for (at least) a certain duration of time.
     */
    virtual void sleep(Duration duration) = 0;

    /**
     * Wait for a condition or until a given duration has expired,
     * whichever happens first.
     */
    virtual std::cv_status waitForCondition(std::unique_lock<std::mutex> *const lock,
                                            std::condition_variable *const cv,
                                            Duration duration)
            = 0;

    /**
     * Wait for a condition or until a given duration has expired,
     * whichever happens first.
     */
    template <typename Pred>
    bool waitForConditionTimed(std::unique_lock<std::mutex> *const lock,
                               std::condition_variable *const cv,
                               Pred pred,
                               Duration duration) {
        Instant now = getMonotonicTime();
        Instant expiresAt = now + duration;
        while (!pred()) {
            if (now >= expiresAt
                || waitForCondition(lock, cv, expiresAt - now) == std::cv_status::timeout) {
                return pred();
            }
            now = getMonotonicTime();
        }

        return true;
    }

    /**
     * Wait for any of the file descriptors to become ready, or until
     * the given duration has expired, whichevery comes first. This is
     * identical to the poll() system call, but uses this clock instead
     * of the system clock for measuring the timeout.
     */
    virtual int poll(pollfd *pfds, int npfds, Duration timeout) = 0;
};

} // namespace util
