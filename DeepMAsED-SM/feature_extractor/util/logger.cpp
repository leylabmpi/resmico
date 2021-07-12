#include "util/logger.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>

std::shared_ptr<spdlog::logger> logger() {
    static const std::shared_ptr<spdlog::logger> LOGGER
            = spdlog::stderr_color_mt("svc", spdlog::color_mode::always);
    return LOGGER;
}
