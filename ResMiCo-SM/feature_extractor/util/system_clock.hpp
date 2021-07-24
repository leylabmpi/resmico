#pragma once

#include "iclock.hpp"

namespace util {

class SystemClock : public IClock {
  public:
    virtual void registerThread(const char *name) override;

    virtual Instant getRealTime() override;
    virtual Instant getMonotonicTime() override;

    virtual void sleep(Duration duration) override;

    virtual std::cv_status waitForCondition(std::unique_lock<std::mutex> *const lock,
                                            std::condition_variable *const cv,
                                            Duration duration) override;

    virtual int poll(pollfd *pfds, int npfds, Duration timeout) override;
};

} // namespace util
