#include "util/util.hpp"

#include <gtest/gtest.h>

namespace {

TEST(Round2, Number) {
    ASSERT_EQ(round2(7.0710678118654755), "7.07");
    ASSERT_EQ(round2(7.0770678118654755), "7.08");
    ASSERT_EQ(round2(7), "7.00");
    ASSERT_EQ(round2(-10.33097), "-10.33");
    ASSERT_EQ(round2(-10.33697), "-10.34");
}

}
