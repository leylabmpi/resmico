cmake_minimum_required(VERSION 3.20)
project(ResMiCo)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_library(contig_reader contig_reader.cpp)
target_link_libraries(contig_reader z)

add_executable(contig_reader_main contig_reader.cpp)

target_link_libraries(contig_reader_main contig_reader)

