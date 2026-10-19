#ifndef __SD2R_DEBUG_HPP__
#define __SD2R_DEBUG_HPP__

// sd2R: file-based debug markers, enabled by the GGMLR_DBG_LOG environment
// variable (path of a log file, appended to). Each call opens/closes the file so
// the last marker survives a hard crash — used to localize Windows/Vulkan
// crashes. GGMLR_DBG_NODES=1 additionally dumps the first compute graph.

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static inline void sd2r_dbg_logf(const char* fmt, ...) {
    const char* path = std::getenv("GGMLR_DBG_LOG");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    FILE* f = std::fopen(path, "a");
    if (f == nullptr) {
        return;
    }
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::fwrite(buf, 1, std::strlen(buf), f);
    std::fwrite("\n", 1, 1, f);
    std::fclose(f);
}

#endif  // __SD2R_DEBUG_HPP__
