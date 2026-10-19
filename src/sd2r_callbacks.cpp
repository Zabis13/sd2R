// sd2R: log / progress / preview callbacks, verbosity flags and profiling.
#include <cstdio>
#include <cstring>

#include "sd2r_interface.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// Atomically replace `dst` with `src` (a temp file just written). POSIX
// rename() overwrites an existing target atomically, but Win32 std::rename()
// FAILS when the target exists — which silently froze the live preview after
// the first frame on Windows. Use MoveFileExA(MOVEFILE_REPLACE_EXISTING) there.
// Returns true on success.
static bool sd_atomic_replace(const char* src, const char* dst) {
#ifdef _WIN32
    return MoveFileExA(src, dst,
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return std::rename(src, dst) == 0;
#endif
}

// --- Verbose flag: controls log and progress output ---
bool r_sd_verbose = false;

// --- Log-debug flag: when true, DEBUG-level messages are also written to the
// async log file (used by the Shiny "generation log" diagnostic). Off by
// default to keep the normal log quiet. ---
static bool r_sd_log_debug = false;

// --- Profiling: capture stage events from sd.cpp log messages ---
bool r_sd_profiling = false;

std::vector<ProfileEvent> r_profile_events;
static std::chrono::steady_clock::time_point r_profile_epoch;

double profile_now_ms() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - r_profile_epoch).count();
}

void profile_mark_generate_start() {
    if (r_sd_profiling) {
        double ts = profile_now_ms();
        r_profile_events.push_back({"generate_total", "start", ts});
        r_profile_events.push_back({"text_encode", "start", ts});
    }
}

// Detect stage boundaries from sd.cpp LOG_INFO messages
static void profile_parse_log(const std::string& msg) {
    if (!r_sd_profiling) return;

    double ts = profile_now_ms();

    // --- Model loading (new_sd_ctx) ---
    if (msg.find("loading ") == 0) {
        if (msg.find("diffusion model from") != std::string::npos) {
            r_profile_events.push_back({"load_diffusion", "start", ts});
        } else if (msg.find("clip_l from") != std::string::npos) {
            r_profile_events.push_back({"load_clip_l", "start", ts});
        } else if (msg.find("clip_g from") != std::string::npos) {
            r_profile_events.push_back({"load_clip_g", "start", ts});
        } else if (msg.find("t5xxl from") != std::string::npos) {
            r_profile_events.push_back({"load_t5xxl", "start", ts});
        } else if (msg.find("vae from") != std::string::npos) {
            r_profile_events.push_back({"load_vae", "start", ts});
        } else if (msg.find("model from") != std::string::npos) {
            r_profile_events.push_back({"load_model", "start", ts});
        }
        return;
    }

    // --- Version line marks end of all loading ---
    if (msg.find("Version: ") == 0) {
        r_profile_events.push_back({"load_all", "end", ts});
        return;
    }

    // --- Text encoding per-model ---
    if (msg.find("text_encode_clip starting") != std::string::npos) {
        r_profile_events.push_back({"text_encode_clip", "start", ts});
    }
    else if (msg.find("text_encode_clip completed") != std::string::npos) {
        r_profile_events.push_back({"text_encode_clip", "end", ts});
    }
    else if (msg.find("text_encode_t5 starting") != std::string::npos) {
        r_profile_events.push_back({"text_encode_t5", "start", ts});
    }
    else if (msg.find("text_encode_t5 completed") != std::string::npos) {
        r_profile_events.push_back({"text_encode_t5", "end", ts});
    }
    // Text encoding total
    else if (msg.find("get_learned_condition completed") != std::string::npos ||
             msg.find("PHASE: text-encode END") != std::string::npos) {
        r_profile_events.push_back({"text_encode", "end", ts});
    }
    // Sampling start. NOTE: sd.cpp emits "sampling using <method>" very early —
    // before text encoding even runs (it is logged right after the denoiser is
    // set up). Using it as the sampling start makes the "sampling" interval
    // swallow the whole text_encode stage (observed: sampling 43.40s vs sd.cpp's
    // own 31.49s, the 11.9s gap being text_encode). The real sampling loop only
    // begins at "generating image:", which sd.cpp logs after
    // get_learned_condition completes — anchor the start there instead.
    else if (msg.find("generating image:") != std::string::npos) {
        r_profile_events.push_back({"sampling", "start", ts});
    }
    // Mode markers
    else if (msg.find("IMG2IMG") != std::string::npos) {
        r_profile_events.push_back({"vae_encode", "start", ts});
    }
    // VAE encode done (img2img)
    else if (msg.find("encode_first_stage completed") != std::string::npos) {
        r_profile_events.push_back({"vae_encode", "end", ts});
    }
    // Sampling done
    else if (msg.find("sampling completed") != std::string::npos) {
        r_profile_events.push_back({"sampling", "end", ts});
    }
    // VAE decode
    else if (msg.find("decoding ") != std::string::npos && msg.find("latent") != std::string::npos) {
        r_profile_events.push_back({"vae_decode", "start", ts});
    }
    else if (msg.find("decode_first_stage completed") != std::string::npos) {
        r_profile_events.push_back({"vae_decode", "end", ts});
    }
    // Tiled sampling
    else if (msg.find("Tiled sampling:") != std::string::npos) {
        r_profile_events.push_back({"tiled_sampling", "start", ts});
    }
    // Total
    else if (msg.find("generate_image completed") != std::string::npos) {
        r_profile_events.push_back({"generate_total", "end", ts});
    }
}

// --- Async flag: when true, callbacks must NOT call R API ---
std::atomic<bool> r_sd_async_running{false};

// --- Log file for async status updates (loading stages etc.) ---
static std::string r_sd_log_file;

// --- Log callback: route SD log messages to R or file ---
static void r_sd_log_callback(sd_log_level_t level, const char* text, void* data) {
    (void)data;
    // Remove trailing newline
    std::string msg(text);
    while (!msg.empty() && msg.back() == '\n') msg.pop_back();
    if (msg.empty()) return;

    // Always parse for profiling (even when not verbose)
    if (level == SD_LOG_INFO) {
        profile_parse_log(msg);
    }

    // When async: write to log file instead of R console.
    // Append (not overwrite) so the full generation log accumulates. DEBUG
    // lines (e.g. the Vulkan device list) are written only when the diagnostic
    // log-debug flag is on.
    if (r_sd_async_running.load()) {
        if (!r_sd_log_file.empty() && (level != SD_LOG_DEBUG || r_sd_log_debug)) {
            FILE* fp = std::fopen(r_sd_log_file.c_str(), "a");
            if (fp) {
                std::fprintf(fp, "%s\n", msg.c_str());
                std::fclose(fp);
            }
        }
        return;
    }

    switch (level) {
        case SD_LOG_DEBUG:
            // suppress debug always
            break;
        case SD_LOG_INFO:
            if (r_sd_verbose) Rprintf("%s\n", msg.c_str());
            break;
        case SD_LOG_WARN:
            if (r_sd_verbose) Rprintf("[WARN] %s\n", msg.c_str());
            break;
        case SD_LOG_ERROR:
            // errors always printed
            REprintf("[ERROR] %s\n", msg.c_str());
            break;
        default:
            // SD_LOG_VERBOSE and any future levels: shown only when verbose
            if (r_sd_verbose) Rprintf("%s\n", msg.c_str());
            break;
    }
}

// --- Progress file for async Shiny updates ---
static std::string r_sd_progress_file;

// --- Progress callback: update R console + write file ---
static void r_sd_progress_callback(int step, int steps, float time, void* data) {
    (void)data;
    // Write progress file (always, if path is set) — safe from any thread
    if (!r_sd_progress_file.empty()) {
        float avg = (step > 0) ? time / step : 0.0f;
        float eta = (steps - step) * avg;
        int pct = (steps > 0) ? (int)(100.0f * step / steps) : 0;
        FILE* fp = std::fopen(r_sd_progress_file.c_str(), "w");
        if (fp) {
            std::fprintf(fp,
                "{\"step\":%d,\"steps\":%d,\"pct\":%d,\"elapsed\":%.1f,\"eta_sec\":%.1f}\n",
                step, steps, pct, time, eta);
            std::fclose(fp);
        }
    }
    // No R API calls from worker thread
    if (r_sd_async_running.load()) return;

    // Console output (main thread only)
    if (r_sd_verbose) {
        Rprintf("\rStep %d/%d (%.1fs)", step, steps, time);
        if (step == steps) Rprintf("\n");
        R_FlushConsole();
    }
    R_CheckUserInterrupt();
}

// [[Rcpp::export]]
void sd_set_progress_file(std::string path) {
    r_sd_progress_file = path;
}

// Whether the meta backend (multi-GPU tensor split) was compiled into this
// build, i.e. the ggml_backend_meta_device symbol was present in libggml.a at
// configure time. Lets the R layer warn when meta_backend=TRUE is unsupported.
// [[Rcpp::export]]
bool sd_meta_backend_available() {
#if defined(SD2R_HAVE_META_BACKEND)
    return true;
#else
    return false;
#endif
}

// [[Rcpp::export]]
void sd_clear_progress_file() {
    if (!r_sd_progress_file.empty()) {
        std::remove(r_sd_progress_file.c_str());
        r_sd_progress_file.clear();
    }
}

// --- Preview callback: dump per-step decoded latent to PPM (diagnostic) ---
// Writes the latest preview frame to a single file at r_sd_preview_path,
// atomically: write to "<path>.tmp" then rename() over the target. rename() is
// atomic on POSIX, so a reader polling the file never sees a half-written PPM.
// Pure stdio, safe from the worker thread (no R API calls). PPM P6 = trivial
// raw-RGB, no deps. Only the first frame is written (single-file semantics).
static std::string r_sd_preview_path;

static void r_sd_preview_callback(int step, int frame_count, sd_image_t* frames,
                                  bool /*is_noisy*/, void* /*data*/) {
    (void)step;
    if (r_sd_preview_path.empty() || frames == nullptr || frame_count < 1) return;
    const sd_image_t& img = frames[0];
    if (!img.data || img.width == 0 || img.height == 0) return;

    std::string tmp = r_sd_preview_path + ".tmp";
    FILE* fp = std::fopen(tmp.c_str(), "wb");
    if (!fp) return;
    // PPM P6 expects 3-channel RGB; expand/clamp channel count.
    std::fprintf(fp, "P6\n%u %u\n255\n", img.width, img.height);
    const uint32_t ch = img.channel;
    for (size_t px = 0; px < (size_t)img.width * img.height; ++px) {
        const uint8_t* p = img.data + px * ch;
        uint8_t rgb[3] = {
            p[0],
            ch > 1 ? p[1] : p[0],
            ch > 2 ? p[2] : p[0]
        };
        std::fwrite(rgb, 1, 3, fp);
    }
    std::fclose(fp);
    // Atomic replace works cross-platform (POSIX rename / Win32 MoveFileEx).
    // On failure the stale frame is left in place; the next step retries.
    sd_atomic_replace(tmp.c_str(), r_sd_preview_path.c_str());
}

// [[Rcpp::export]]
void sd_set_preview_dump(std::string path, std::string mode, int interval,
                         bool denoised, bool noisy) {
    r_sd_preview_path = path;
    enum preview_t pm = str_to_preview(mode.c_str());
    sd_set_preview_callback(r_sd_preview_callback, pm, interval,
                            denoised, noisy, nullptr);
}

// [[Rcpp::export]]
void sd_clear_preview_dump() {
    if (!r_sd_preview_path.empty()) {
        std::remove((r_sd_preview_path + ".tmp").c_str());
    }
    r_sd_preview_path.clear();
    sd_set_preview_callback(nullptr, PREVIEW_NONE, 1, false, false, nullptr);
}

// [[Rcpp::export]]
void sd_set_log_file(std::string path) {
    r_sd_log_file = path;
    // Truncate so each load/generation starts with a clean log (the callback
    // appends from here on).
    if (!r_sd_log_file.empty()) {
        FILE* fp = std::fopen(r_sd_log_file.c_str(), "w");
        if (fp) std::fclose(fp);
    }
}

// [[Rcpp::export]]
void sd_set_log_debug(bool enabled) {
    r_sd_log_debug = enabled;
}

// [[Rcpp::export]]
void sd_clear_log_file() {
    if (!r_sd_log_file.empty()) {
        std::remove(r_sd_log_file.c_str());
        r_sd_log_file.clear();
    }
}

// [[Rcpp::export]]
void sd_set_verbose(bool verbose) {
    r_sd_verbose = verbose;
}

// [[Rcpp::export]]
void sd_profile_start() {
    r_profile_events.clear();
    r_profile_epoch = std::chrono::steady_clock::now();
    r_sd_profiling = true;
}

// [[Rcpp::export]]
void sd_profile_stop() {
    r_sd_profiling = false;
}

// [[Rcpp::export]]
Rcpp::DataFrame sd_profile_get() {
    int n = (int)r_profile_events.size();
    Rcpp::CharacterVector stages(n);
    Rcpp::CharacterVector kinds(n);
    Rcpp::NumericVector timestamps(n);

    for (int i = 0; i < n; i++) {
        stages[i] = r_profile_events[i].stage;
        kinds[i] = r_profile_events[i].kind;
        timestamps[i] = r_profile_events[i].timestamp_ms;
    }

    return Rcpp::DataFrame::create(
        Rcpp::Named("stage") = stages,
        Rcpp::Named("kind") = kinds,
        Rcpp::Named("timestamp_ms") = timestamps,
        Rcpp::Named("stringsAsFactors") = false
    );
}

// [[Rcpp::export]]
void sd_init_log() {
    sd_set_log_callback(r_sd_log_callback, nullptr);
    sd_set_progress_callback(r_sd_progress_callback, nullptr);
}
