cmake_minimum_required(VERSION 3.13)
project(SomaticVariantCalling)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# no warnings
string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -Wfatal-errors -msse4 ${OpenMP_CXX_FLAGS}")

find_package(OpenMP REQUIRED)
# find_package(ZLIB REQUIRED)

include_directories(.)
include_directories(third_party)
include_directories(third_party/spdlog/include)

set(CMAKE_CXX_FLAGS_DEBUG "-g -D_GLIBCXX_DEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Google Flags Library
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/gflags EXCLUDE_FROM_ALL)

# added later because glags has errors on GCC
string(APPEND CMAKE_CXX_FLAGS " -Werror")

# Spdlog logging library
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/spdlog EXCLUDE_FROM_ALL)

# BAMTools library for parsing BAM files
include_directories(${PROJECT_SOURCE_DIR}/third_party/bamtools/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/bamtools EXCLUDE_FROM_ALL)


file(GLOB util_files util/*.cpp)
add_library(util ${util_files})
target_link_libraries(util spdlog::spdlog )
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(util stdc++fs)
endif()

add_library(stats contig_stats.cpp)
target_link_libraries(stats spdlog::spdlog BamTools BamTools-utils util)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(stats stdc++fs)
endif()

add_executable(bam2feat bam2feat.cpp)
target_link_libraries(bam2feat stats util gflags spdlog::spdlog OpenMP::OpenMP_CXX)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(bam2feat stdc++fs)
endif()

# TESTS
enable_testing()
include(GoogleTest)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/googletest EXCLUDE_FROM_ALL)
target_compile_options(gtest_main PRIVATE -w)
target_compile_options(gtest PRIVATE -w)

file(GLOB test_files "tests/*.cpp")

add_executable(tests ${test_files})
target_link_libraries(tests stats gtest_main gtest gmock)
target_include_directories(tests PRIVATE "include")

gtest_discover_tests(tests)

# copy the data into the binary directory, so that tests can reference it
file(COPY tests/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
