#pragma once
#include <spdlog/fmt/ostr.h>  // for logging custom classes
#include <spdlog/spdlog.h>

std::shared_ptr<spdlog::logger> logger();
