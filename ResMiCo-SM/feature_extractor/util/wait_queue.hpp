#pragma once

#include "duration.hpp"
#include "iclock.hpp"

#include <cassert>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace util {

/**
 *  A WaitQueue wraps synchronisation primitives needed to share state between threads.
 *  A writer can push_front a value to the queue, readers can pop_back those values in order.
 *  The reader will block if the queue is empty, the writer will block if the queue is full
 *  i.e. if it contains its max_size number of elements (default max size_type).
 *
 *  when the queue is shutdown, the readers will unblock and their pop_back returns false.
 */
template <typename _Tp, typename _Alloc = std::allocator<_Tp>>
class WaitQueue {
  public:
    typedef std::deque<_Tp, _Alloc> queue_type;
    typedef size_t size_type;
    typedef _Tp value_type;

    explicit WaitQueue(size_type max_size = size_type(-1))
        : max_size_(max_size), shutdown_(false) {}

    ~WaitQueue() {
        shutdown();
        std::unique_lock<std::mutex> l(mu_);
        while (!empty()) {
            empty_.wait(l);
        }
    }

    bool empty() const { return queue_.empty(); }

    bool full() const { return queue_.size() == max_size_; }

    /**
     * close the channel. any blocked readers will be woken.
     */
    void shutdown() {
        std::unique_lock<std::mutex> l(mu_);
        shutdown_ = true;
        not_empty_.notify_all();
    }

    /**
     * Enqueues x by moving it into the queue, blocks when full.
     * Note that this function receives its parameter by value, so make sure you std::move it into
     * the queue if the copy construction is expensive.
     */
    void push_front(value_type x) {
        std::unique_lock<std::mutex> l(mu_);
        while (full()) {
            not_full_.wait(l);
        }
        const bool was_empty = empty();
        queue_.push_front(std::move(x));
        if (was_empty) {
            not_empty_.notify_one();
        }
        return;
    }

    /**
     * dequeue, block when empty,
     * returns false if and only if the queue is shut down and empty.
     */
    bool pop_back(value_type *const r) {
        std::unique_lock<std::mutex> l(mu_);
        while (!shutdown_ && empty()) {
            not_empty_.wait(l);
        }

        return pop_back_impl(r);
    }

    /**
     * dequeue, block when empty for up to a period of @timeout as measured by @clock.
     * Returns -1 if and only if the queue is shut down and empty.
     * otherwise, returns the number of items read (0 or 1, depending on
     * whether timeout expired or not).
     */
    int pop_back(value_type *const r, IClock *clock, Duration timeout) {
        std::unique_lock<std::mutex> l(mu_);

        if (!clock->waitForConditionTimed(
                    &l, &not_empty_, [this]() { return shutdown_ || !empty(); }, timeout)) {
            return 0;
        }

        if (!pop_back_impl(r)) {
            return -1;
        }

        return 1;
    }

  private:
    queue_type queue_;
    std::mutex mu_;
    std::condition_variable empty_;
    std::condition_variable not_empty_; // or shut down
    std::condition_variable not_full_;

    const size_type max_size_;
    bool shutdown_;

  private:
    WaitQueue(const WaitQueue &other) = delete; // non construction-copyable
    WaitQueue &operator=(const WaitQueue &) = delete; // non copyable

    /**
     * must hold the lock and the queue must not be empty.
     */
    bool pop_back_impl(value_type *const r) {
        if (empty()) {
            assert(shutdown_);
            return false;
        }
        const bool was_full = full();

        *r = std::move(queue_.back());
        queue_.pop_back();

        if (was_full) {
            not_full_.notify_one();
        }
        if (empty()) {
            empty_.notify_all();
        }

        return true;
    }
};

} // namespace util
