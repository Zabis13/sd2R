#include <Rcpp.h>
#include <chrono>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <thread>
#include <atomic>
#include <mutex>
#include "sd/stable-diffusion.h"
#include "sd/sd2r_meta_backend.hpp"  // sd2R: meta backend (multi-GPU tensor split, behind a flag)

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
static bool r_sd_verbose = false;

// --- Profiling: capture stage events from sd.cpp log messages ---
static bool r_sd_profiling = false;

struct ProfileEvent {
    std::string stage;
    std::string kind;  // "start" or "end"
    double timestamp_ms;
};

static std::vector<ProfileEvent> r_profile_events;
static std::chrono::steady_clock::time_point r_profile_epoch;

static double profile_now_ms() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - r_profile_epoch).count();
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
    else if (msg.find("get_learned_condition completed") != std::string::npos) {
        r_profile_events.push_back({"text_encode", "end", ts});
    }
    // Sampling setup
    else if (msg.find("sampling using ") != std::string::npos) {
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
static std::atomic<bool> r_sd_async_running{false};

// --- Log file for async status updates (loading stages etc.) ---
static std::string r_sd_log_file;

// --- Log callback: route SD log messages to R or file ---
static void r_sd_log_callback(sd_log_level_t level, const char* text, void* data) {
    // Remove trailing newline
    std::string msg(text);
    while (!msg.empty() && msg.back() == '\n') msg.pop_back();
    if (msg.empty()) return;

    // Always parse for profiling (even when not verbose)
    if (level == SD_LOG_INFO) {
        profile_parse_log(msg);
    }

    // When async: write to log file instead of R console
    if (r_sd_async_running.load()) {
        if (!r_sd_log_file.empty() && level != SD_LOG_DEBUG) {
            FILE* fp = std::fopen(r_sd_log_file.c_str(), "w");
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
    }
}

// --- Progress file for async Shiny updates ---
static std::string r_sd_progress_file;

// --- Progress callback: update R console + write file ---
static void r_sd_progress_callback(int step, int steps, float time, void* data) {
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

// --- Custom deleters for XPtr (avoids delete-incomplete warning) ---
inline void sd_ctx_invoke_free(sd_ctx_t* ctx) { if (ctx) free_sd_ctx(ctx); }
inline void upscaler_ctx_invoke_free(upscaler_ctx_t* ctx) { if (ctx) free_upscaler_ctx(ctx); }
typedef Rcpp::XPtr<sd_ctx_t, Rcpp::PreserveStorage, sd_ctx_invoke_free> SdCtxXPtr;
typedef Rcpp::XPtr<upscaler_ctx_t, Rcpp::PreserveStorage, upscaler_ctx_invoke_free> UpscalerCtxXPtr;

// [[Rcpp::export]]
void sd_init_log() {
    sd_set_log_callback(r_sd_log_callback, nullptr);
    sd_set_progress_callback(r_sd_progress_callback, nullptr);
}

// [[Rcpp::export]]
SEXP sd_create_context(Rcpp::List params) {
    sd_ctx_params_t p;
    sd_ctx_params_init(&p);

    // Required
    if (params.containsElementNamed("model_path"))
        p.model_path = Rcpp::as<std::string>(params["model_path"]).c_str();

    // Optional model paths
    std::string clip_l, clip_g, clip_vision, t5xxl, llm, llm_vision;
    std::string diffusion_model, high_noise_diffusion_model;
    std::string vae, taesd, control_net, photo_maker;
    std::string tensor_type_rules;

    auto set_str = [&](const char* name, std::string& storage, const char*& target) {
        if (params.containsElementNamed(name) && !Rf_isNull(params[name])) {
            storage = Rcpp::as<std::string>(params[name]);
            target = storage.c_str();
        }
    };

    // We need stable storage for strings that outlive the lambda
    std::string model_path_str;
    if (params.containsElementNamed("model_path") && !Rf_isNull(params["model_path"])) {
        model_path_str = Rcpp::as<std::string>(params["model_path"]);
        p.model_path = model_path_str.c_str();
    }

    set_str("clip_l_path", clip_l, p.clip_l_path);
    set_str("clip_g_path", clip_g, p.clip_g_path);
    set_str("clip_vision_path", clip_vision, p.clip_vision_path);
    set_str("t5xxl_path", t5xxl, p.t5xxl_path);
    set_str("llm_path", llm, p.llm_path);
    set_str("llm_vision_path", llm_vision, p.llm_vision_path);
    set_str("diffusion_model_path", diffusion_model, p.diffusion_model_path);
    set_str("high_noise_diffusion_model_path", high_noise_diffusion_model, p.high_noise_diffusion_model_path);
    set_str("vae_path", vae, p.vae_path);
    set_str("taesd_path", taesd, p.taesd_path);
    set_str("control_net_path", control_net, p.control_net_path);
    set_str("photo_maker_path", photo_maker, p.photo_maker_path);
    set_str("tensor_type_rules", tensor_type_rules, p.tensor_type_rules);

    // Numeric/bool params
    if (params.containsElementNamed("n_threads"))
        p.n_threads = Rcpp::as<int>(params["n_threads"]);
    if (params.containsElementNamed("vae_decode_only"))
        p.vae_decode_only = Rcpp::as<bool>(params["vae_decode_only"]);
    if (params.containsElementNamed("free_params_immediately"))
        p.free_params_immediately = Rcpp::as<bool>(params["free_params_immediately"]);
    if (params.containsElementNamed("wtype"))
        p.wtype = static_cast<sd_type_t>(Rcpp::as<int>(params["wtype"]));
    if (params.containsElementNamed("rng_type"))
        p.rng_type = static_cast<rng_type_t>(Rcpp::as<int>(params["rng_type"]));
    if (params.containsElementNamed("prediction"))
        p.prediction = static_cast<prediction_t>(Rcpp::as<int>(params["prediction"]));
    if (params.containsElementNamed("lora_apply_mode"))
        p.lora_apply_mode = static_cast<lora_apply_mode_t>(Rcpp::as<int>(params["lora_apply_mode"]));
    if (params.containsElementNamed("offload_params_to_cpu"))
        p.offload_params_to_cpu = Rcpp::as<bool>(params["offload_params_to_cpu"]);
    if (params.containsElementNamed("max_vram"))
        p.max_vram = Rcpp::as<float>(params["max_vram"]);
    if (params.containsElementNamed("stream_layers"))
        p.stream_layers = Rcpp::as<bool>(params["stream_layers"]);
    if (params.containsElementNamed("enable_mmap"))
        p.enable_mmap = Rcpp::as<bool>(params["enable_mmap"]);
    if (params.containsElementNamed("keep_clip_on_cpu"))
        p.keep_clip_on_cpu = Rcpp::as<bool>(params["keep_clip_on_cpu"]);
    if (params.containsElementNamed("keep_control_net_on_cpu"))
        p.keep_control_net_on_cpu = Rcpp::as<bool>(params["keep_control_net_on_cpu"]);
    if (params.containsElementNamed("keep_vae_on_cpu"))
        p.keep_vae_on_cpu = Rcpp::as<bool>(params["keep_vae_on_cpu"]);
    if (params.containsElementNamed("vae_conv_direct"))
        p.vae_conv_direct = Rcpp::as<bool>(params["vae_conv_direct"]);
    if (params.containsElementNamed("diffusion_conv_direct"))
        p.diffusion_conv_direct = Rcpp::as<bool>(params["diffusion_conv_direct"]);
    if (params.containsElementNamed("diffusion_flash_attn"))
        p.diffusion_flash_attn = Rcpp::as<bool>(params["diffusion_flash_attn"]);
    if (params.containsElementNamed("diffusion_gpu_device"))
        p.diffusion_gpu_device = Rcpp::as<int>(params["diffusion_gpu_device"]);
    if (params.containsElementNamed("clip_gpu_device"))
        p.clip_gpu_device = Rcpp::as<int>(params["clip_gpu_device"]);
    if (params.containsElementNamed("vae_gpu_device"))
        p.vae_gpu_device = Rcpp::as<int>(params["vae_gpu_device"]);

    // sd2R "second path": generate via the meta backend (on/off flag).
    // Default off => normal single-backend path. If meta is unavailable in ggmlR,
    // C++ falls back to the normal path (see get_or_build_diffusion_meta_backend).
    bool meta_on = params.containsElementNamed("meta_backend") &&
                   !Rf_isNull(params["meta_backend"]) &&
                   Rcpp::as<bool>(params["meta_backend"]);
    sd2r::meta::set_meta_backend(meta_on);

    sd_ctx_t* ctx = new_sd_ctx(&p);
    if (!ctx) {
        sd2r::meta::set_meta_backend(false);
        Rcpp::stop("Failed to create stable diffusion context");
    }

    SdCtxXPtr xptr(ctx, true);
    xptr.attr("class") = "sd_ctx";
    return xptr;
}

// [[Rcpp::export]]
void sd_destroy_context(SEXP ctx_sexp) {
    SdCtxXPtr xptr(ctx_sexp);
    if (xptr.get()) {
        free_sd_ctx(xptr.get());
        xptr.release();
    }
}

// ============================================================
// --- Async context creation (std::thread, polled from R) ---
// ============================================================

struct AsyncCtxState {
    std::thread worker;
    sd_ctx_params_t params;

    // Owned string storage (must outlive params)
    std::string model_path, clip_l, clip_g, clip_vision, t5xxl;
    std::string llm, llm_vision, diffusion_model, high_noise_diffusion_model;
    std::string vae, taesd, control_net, photo_maker, tensor_type_rules;

    std::atomic<bool> running{false};
    std::atomic<bool> done{false};
    sd_ctx_t* result = nullptr;
    std::string error_msg;
};

static AsyncCtxState g_async_ctx;

static void async_ctx_worker() {
    r_sd_async_running.store(true);
    try {
        g_async_ctx.result = new_sd_ctx(&g_async_ctx.params);
        if (!g_async_ctx.result) {
            g_async_ctx.error_msg = "Failed to create stable diffusion context";
        }
    } catch (const std::exception& e) {
        g_async_ctx.result = nullptr;
        g_async_ctx.error_msg = std::string("Exception: ") + e.what();
    } catch (...) {
        g_async_ctx.result = nullptr;
        g_async_ctx.error_msg = "Unknown exception during context creation";
    }
    r_sd_async_running.store(false);
    g_async_ctx.done.store(true);
    g_async_ctx.running.store(false);
}

// Helper: copy string param from R list into owned storage, set C pointer
static void async_ctx_set_str(Rcpp::List& params, const char* name,
                               std::string& storage, const char*& target) {
    if (params.containsElementNamed(name) && !Rf_isNull(params[name])) {
        storage = Rcpp::as<std::string>(params[name]);
        target = storage.c_str();
    }
}

// [[Rcpp::export]]
bool sd_create_context_async(Rcpp::List params) {
    if (g_async_ctx.running.load()) {
        Rcpp::stop("Context creation already in progress");
    }

    // Reset
    g_async_ctx.done.store(false);
    g_async_ctx.running.store(true);
    g_async_ctx.result = nullptr;
    g_async_ctx.error_msg.clear();

    sd_ctx_params_t& p = g_async_ctx.params;
    sd_ctx_params_init(&p);

    // String params — copy into owned storage
    async_ctx_set_str(params, "model_path", g_async_ctx.model_path, p.model_path);
    async_ctx_set_str(params, "clip_l_path", g_async_ctx.clip_l, p.clip_l_path);
    async_ctx_set_str(params, "clip_g_path", g_async_ctx.clip_g, p.clip_g_path);
    async_ctx_set_str(params, "clip_vision_path", g_async_ctx.clip_vision, p.clip_vision_path);
    async_ctx_set_str(params, "t5xxl_path", g_async_ctx.t5xxl, p.t5xxl_path);
    async_ctx_set_str(params, "llm_path", g_async_ctx.llm, p.llm_path);
    async_ctx_set_str(params, "llm_vision_path", g_async_ctx.llm_vision, p.llm_vision_path);
    async_ctx_set_str(params, "diffusion_model_path", g_async_ctx.diffusion_model, p.diffusion_model_path);
    async_ctx_set_str(params, "high_noise_diffusion_model_path", g_async_ctx.high_noise_diffusion_model, p.high_noise_diffusion_model_path);
    async_ctx_set_str(params, "vae_path", g_async_ctx.vae, p.vae_path);
    async_ctx_set_str(params, "taesd_path", g_async_ctx.taesd, p.taesd_path);
    async_ctx_set_str(params, "control_net_path", g_async_ctx.control_net, p.control_net_path);
    async_ctx_set_str(params, "photo_maker_path", g_async_ctx.photo_maker, p.photo_maker_path);
    async_ctx_set_str(params, "tensor_type_rules", g_async_ctx.tensor_type_rules, p.tensor_type_rules);

    // Numeric/bool params
    if (params.containsElementNamed("n_threads"))
        p.n_threads = Rcpp::as<int>(params["n_threads"]);
    if (params.containsElementNamed("vae_decode_only"))
        p.vae_decode_only = Rcpp::as<bool>(params["vae_decode_only"]);
    if (params.containsElementNamed("free_params_immediately"))
        p.free_params_immediately = Rcpp::as<bool>(params["free_params_immediately"]);
    if (params.containsElementNamed("wtype"))
        p.wtype = static_cast<sd_type_t>(Rcpp::as<int>(params["wtype"]));
    if (params.containsElementNamed("rng_type"))
        p.rng_type = static_cast<rng_type_t>(Rcpp::as<int>(params["rng_type"]));
    if (params.containsElementNamed("prediction"))
        p.prediction = static_cast<prediction_t>(Rcpp::as<int>(params["prediction"]));
    if (params.containsElementNamed("lora_apply_mode"))
        p.lora_apply_mode = static_cast<lora_apply_mode_t>(Rcpp::as<int>(params["lora_apply_mode"]));
    if (params.containsElementNamed("offload_params_to_cpu"))
        p.offload_params_to_cpu = Rcpp::as<bool>(params["offload_params_to_cpu"]);
    if (params.containsElementNamed("max_vram"))
        p.max_vram = Rcpp::as<float>(params["max_vram"]);
    if (params.containsElementNamed("stream_layers"))
        p.stream_layers = Rcpp::as<bool>(params["stream_layers"]);
    if (params.containsElementNamed("enable_mmap"))
        p.enable_mmap = Rcpp::as<bool>(params["enable_mmap"]);
    if (params.containsElementNamed("keep_clip_on_cpu"))
        p.keep_clip_on_cpu = Rcpp::as<bool>(params["keep_clip_on_cpu"]);
    if (params.containsElementNamed("keep_control_net_on_cpu"))
        p.keep_control_net_on_cpu = Rcpp::as<bool>(params["keep_control_net_on_cpu"]);
    if (params.containsElementNamed("keep_vae_on_cpu"))
        p.keep_vae_on_cpu = Rcpp::as<bool>(params["keep_vae_on_cpu"]);
    if (params.containsElementNamed("vae_conv_direct"))
        p.vae_conv_direct = Rcpp::as<bool>(params["vae_conv_direct"]);
    if (params.containsElementNamed("diffusion_conv_direct"))
        p.diffusion_conv_direct = Rcpp::as<bool>(params["diffusion_conv_direct"]);
    if (params.containsElementNamed("diffusion_flash_attn"))
        p.diffusion_flash_attn = Rcpp::as<bool>(params["diffusion_flash_attn"]);
    if (params.containsElementNamed("diffusion_gpu_device"))
        p.diffusion_gpu_device = Rcpp::as<int>(params["diffusion_gpu_device"]);
    if (params.containsElementNamed("clip_gpu_device"))
        p.clip_gpu_device = Rcpp::as<int>(params["clip_gpu_device"]);
    if (params.containsElementNamed("vae_gpu_device"))
        p.vae_gpu_device = Rcpp::as<int>(params["vae_gpu_device"]);

    // sd2R "second path": generate via the meta backend (on/off flag).
    // The global flag is read in the worker thread at new_sd_ctx → set it BEFORE launch.
    bool meta_on = params.containsElementNamed("meta_backend") &&
                   !Rf_isNull(params["meta_backend"]) &&
                   Rcpp::as<bool>(params["meta_backend"]);
    sd2r::meta::set_meta_backend(meta_on);

    // Launch
    if (g_async_ctx.worker.joinable()) g_async_ctx.worker.join();
    g_async_ctx.worker = std::thread(async_ctx_worker);
    return true;
}

// [[Rcpp::export]]
Rcpp::List sd_create_context_poll() {
    return Rcpp::List::create(
        Rcpp::Named("running") = g_async_ctx.running.load(),
        Rcpp::Named("done") = g_async_ctx.done.load()
    );
}

// [[Rcpp::export]]
SEXP sd_create_context_result() {
    if (!g_async_ctx.done.load()) {
        Rcpp::stop("Context creation not finished yet");
    }
    if (g_async_ctx.worker.joinable()) g_async_ctx.worker.join();

    if (!g_async_ctx.error_msg.empty()) {
        std::string err = g_async_ctx.error_msg;
        g_async_ctx.error_msg.clear();
        Rcpp::stop(err);
    }

    sd_ctx_t* ctx = g_async_ctx.result;
    g_async_ctx.result = nullptr;

    SdCtxXPtr xptr(ctx, true);
    xptr.attr("class") = "sd_ctx";
    return xptr;
}

// Helper: convert sd_image_t to R raw matrix (RGBA -> raw vector + dims)
static Rcpp::List sd_image_to_r(const sd_image_t& img) {
    size_t n = (size_t)img.width * img.height * img.channel;
    Rcpp::RawVector data(n);
    if (img.data && n > 0) {
        std::memcpy(&data[0], img.data, n);
    }
    return Rcpp::List::create(
        Rcpp::Named("width") = (int)img.width,
        Rcpp::Named("height") = (int)img.height,
        Rcpp::Named("channel") = (int)img.channel,
        Rcpp::Named("data") = data
    );
}

// Helper: convert R raw vector + dims to sd_image_t (caller must manage lifetime)
static sd_image_t r_to_sd_image(Rcpp::List img_list) {
    sd_image_t img;
    img.width = Rcpp::as<uint32_t>(img_list["width"]);
    img.height = Rcpp::as<uint32_t>(img_list["height"]);
    img.channel = Rcpp::as<uint32_t>(img_list["channel"]);
    Rcpp::RawVector data = Rcpp::as<Rcpp::RawVector>(img_list["data"]);
    img.data = (uint8_t*)&data[0];
    return img;
}

// ===========================================================================
// Low-level pipeline steps (TODO 9.2) — R bridge.
// sd_tensor <-> R list(type, ne, data); f32 payload carried as R numeric
// (float<->double conversion at the boundary). sd_cond <-> list of 3 tensors.
// ===========================================================================

// sd_tensor -> R list. Empty tensor (data == NULL) becomes R NULL.
static SEXP sd_tensor_to_r(const sd_tensor& t) {
    if (t.data == nullptr) {
        return R_NilValue;
    }
    size_t n_elem = t.nbytes / sizeof(float);  // Variant B steps emit f32
    Rcpp::NumericVector data(n_elem);
    const float* src = reinterpret_cast<const float*>(t.data);
    for (size_t i = 0; i < n_elem; i++) {
        data[i] = static_cast<double>(src[i]);
    }
    return Rcpp::List::create(
        Rcpp::Named("type") = (int)t.type,
        Rcpp::Named("ne")   = Rcpp::IntegerVector::create((int)t.ne[0], (int)t.ne[1],
                                                          (int)t.ne[2], (int)t.ne[3]),
        Rcpp::Named("data") = data
    );
}

// R list -> sd_tensor with a freshly malloc'd f32 buffer (caller frees via
// sd_tensor_free). R NULL -> empty tensor.
static sd_tensor r_to_sd_tensor(SEXP obj) {
    sd_tensor t;
    t.type = SD_TYPE_F32;
    t.ne[0] = t.ne[1] = t.ne[2] = t.ne[3] = 0;
    t.data = nullptr;
    t.nbytes = 0;
    if (Rf_isNull(obj)) {
        return t;
    }
    Rcpp::List l(obj);
    Rcpp::IntegerVector ne = l["ne"];
    Rcpp::NumericVector data = l["data"];
    t.type = (enum sd_type_t)Rcpp::as<int>(l["type"]);
    for (int i = 0; i < 4; i++) t.ne[i] = ne[i];
    size_t n_elem = data.size();
    t.nbytes = n_elem * sizeof(float);
    t.data = malloc(t.nbytes);
    if (t.data != nullptr) {
        float* dst = reinterpret_cast<float*>(t.data);
        for (size_t i = 0; i < n_elem; i++) dst[i] = static_cast<float>(data[i]);
    } else {
        t.nbytes = 0;
    }
    return t;
}

static SEXP sd_cond_to_r(const sd_cond& c) {
    return Rcpp::List::create(
        Rcpp::Named("crossattn") = sd_tensor_to_r(c.crossattn),
        Rcpp::Named("vector")    = sd_tensor_to_r(c.vector),
        Rcpp::Named("concat")    = sd_tensor_to_r(c.concat)
    );
}

// Returns a sd_cond whose tensors own malloc'd buffers; free with sd_cond_free.
static sd_cond r_to_sd_cond(Rcpp::List l) {
    sd_cond c;
    c.crossattn = r_to_sd_tensor(l.containsElementNamed("crossattn") ? l["crossattn"] : R_NilValue);
    c.vector    = r_to_sd_tensor(l.containsElementNamed("vector")    ? l["vector"]    : R_NilValue);
    c.concat    = r_to_sd_tensor(l.containsElementNamed("concat")    ? l["concat"]    : R_NilValue);
    return c;
}

// [[Rcpp::export]]
int sd_ctx_version_cpp(SEXP ctx_sexp) {
    SdCtxXPtr xptr(ctx_sexp);
    if (!xptr.get()) return -1;
    return sd_ctx_get_version(xptr.get());
}

// [[Rcpp::export]]
bool sd_ctx_supports_ref_cpp(SEXP ctx_sexp) {
    SdCtxXPtr xptr(ctx_sexp);
    if (!xptr.get()) return false;
    return sd_ctx_supports_ref_images(xptr.get());
}

// [[Rcpp::export]]
SEXP sd_encode_text_cpp(SEXP ctx_sexp, std::string prompt, int clip_skip,
                        int width, int height) {
    SdCtxXPtr xptr(ctx_sexp);
    if (!xptr.get()) Rcpp::stop("Invalid sd_ctx (NULL pointer)");
    sd_cond c = sd_encode_text(xptr.get(), prompt.c_str(), clip_skip, width, height);
    SEXP out = sd_cond_to_r(c);
    sd_cond_free(&c);
    return out;
}

// [[Rcpp::export]]
SEXP sd_encode_image_cpp(SEXP ctx_sexp, Rcpp::List image) {
    SdCtxXPtr xptr(ctx_sexp);
    if (!xptr.get()) Rcpp::stop("Invalid sd_ctx (NULL pointer)");
    sd_image_t img = r_to_sd_image(image);
    sd_tensor lat = sd_encode_image(xptr.get(), img);
    SEXP out = sd_tensor_to_r(lat);
    sd_tensor_free(&lat);
    return out;
}

// [[Rcpp::export]]
Rcpp::List sd_decode_latent_cpp(SEXP ctx_sexp, SEXP latent) {
    SdCtxXPtr xptr(ctx_sexp);
    if (!xptr.get()) Rcpp::stop("Invalid sd_ctx (NULL pointer)");
    sd_tensor lat = r_to_sd_tensor(latent);
    sd_image_t img = sd_decode_latent(xptr.get(), lat);
    sd_tensor_free(&lat);
    Rcpp::List out = sd_image_to_r(img);
    if (img.data) free(img.data);
    return out;
}

// [[Rcpp::export]]
SEXP sd_sample_cpp(SEXP ctx_sexp, SEXP init_latent, SEXP noise,
                   Rcpp::List cond, Rcpp::List uncond,
                   int sample_method, int scheduler, int sample_steps,
                   double cfg_scale, double eta, double strength,
                   Rcpp::Nullable<Rcpp::NumericVector> custom_sigmas) {
    SdCtxXPtr xptr(ctx_sexp);
    if (!xptr.get()) Rcpp::stop("Invalid sd_ctx (NULL pointer)");

    sd_tensor init_t = r_to_sd_tensor(init_latent);
    sd_tensor noise_t = r_to_sd_tensor(noise);
    sd_cond c  = r_to_sd_cond(cond);
    sd_cond uc = r_to_sd_cond(uncond);

    sd_sample_params_t sp;
    sd_sample_params_init(&sp);
    sp.sample_method     = (enum sample_method_t)sample_method;
    sp.scheduler         = (enum scheduler_t)scheduler;
    sp.sample_steps      = sample_steps;
    sp.eta               = (float)eta;
    sp.guidance.txt_cfg  = (float)cfg_scale;
    std::vector<float> sig;
    if (custom_sigmas.isNotNull()) {
        Rcpp::NumericVector cs(custom_sigmas);
        sig.assign(cs.begin(), cs.end());
        sp.custom_sigmas       = sig.data();
        sp.custom_sigmas_count = (int)sig.size();
    }

    sd_tensor x0 = sd_sample_loop(xptr.get(), init_t, noise_t, c, uc, &sp, (float)strength);
    SEXP out = sd_tensor_to_r(x0);

    sd_tensor_free(&init_t);
    sd_tensor_free(&noise_t);
    sd_cond_free(&c);
    sd_cond_free(&uc);
    sd_tensor_free(&x0);
    return out;
}

// --- step-wise sampling (Stage 2) bridge -----------------------------------

// [[Rcpp::export]]
Rcpp::NumericVector sd_sampler_sigmas_cpp(SEXP ctx_sexp, int scheduler,
                                          int sample_method, int sample_steps,
                                          int width, int height) {
    SdCtxXPtr xptr(ctx_sexp);
    if (!xptr.get()) Rcpp::stop("Invalid sd_ctx (NULL pointer)");
    int len = 0;
    float* sig = sd_sampler_sigmas(xptr.get(), (enum scheduler_t)scheduler,
                                   (enum sample_method_t)sample_method,
                                   sample_steps, width, height, &len);
    if (sig == nullptr) {
        Rcpp::stop("sd_sampler_sigmas() failed");
    }
    Rcpp::NumericVector out(len);
    for (int i = 0; i < len; i++) out[i] = static_cast<double>(sig[i]);
    sd_floats_free(sig);
    return out;
}

// [[Rcpp::export]]
void sd_sampler_begin_cpp(SEXP ctx_sexp) {
    SdCtxXPtr xptr(ctx_sexp);
    if (!xptr.get()) Rcpp::stop("Invalid sd_ctx (NULL pointer)");
    sd_sampler_begin(xptr.get());
}

// [[Rcpp::export]]
void sd_sampler_end_cpp(SEXP ctx_sexp) {
    SdCtxXPtr xptr(ctx_sexp);
    if (!xptr.get()) Rcpp::stop("Invalid sd_ctx (NULL pointer)");
    sd_sampler_end(xptr.get());
}

// [[Rcpp::export]]
SEXP sd_noise_scale_cpp(SEXP ctx_sexp, SEXP init_latent, SEXP noise,
                        double sigma0) {
    SdCtxXPtr xptr(ctx_sexp);
    if (!xptr.get()) Rcpp::stop("Invalid sd_ctx (NULL pointer)");
    sd_tensor init_t  = r_to_sd_tensor(init_latent);
    sd_tensor noise_t = r_to_sd_tensor(noise);
    sd_tensor x = sd_noise_scale(xptr.get(), init_t, noise_t, (float)sigma0);
    SEXP out = sd_tensor_to_r(x);
    sd_tensor_free(&init_t);
    sd_tensor_free(&noise_t);
    sd_tensor_free(&x);
    return out;
}

// [[Rcpp::export]]
SEXP sd_inverse_noise_scale_cpp(SEXP ctx_sexp, SEXP x_in, double sigma_last) {
    SdCtxXPtr xptr(ctx_sexp);
    if (!xptr.get()) Rcpp::stop("Invalid sd_ctx (NULL pointer)");
    sd_tensor x_t = r_to_sd_tensor(x_in);
    sd_tensor r = sd_inverse_noise_scale(xptr.get(), x_t, (float)sigma_last);
    SEXP out = sd_tensor_to_r(r);
    sd_tensor_free(&x_t);
    sd_tensor_free(&r);
    return out;
}

// [[Rcpp::export]]
SEXP sd_denoise_step_cpp(SEXP ctx_sexp, SEXP x_in, double sigma,
                         Rcpp::List cond, Rcpp::List uncond,
                         double cfg_scale, int step, int total_steps) {
    SdCtxXPtr xptr(ctx_sexp);
    if (!xptr.get()) Rcpp::stop("Invalid sd_ctx (NULL pointer)");
    sd_tensor x_t = r_to_sd_tensor(x_in);
    sd_cond c  = r_to_sd_cond(cond);
    sd_cond uc = r_to_sd_cond(uncond);
    sd_tensor den = sd_denoise_step(xptr.get(), x_t, (float)sigma, c, uc,
                                    (float)cfg_scale, step, total_steps);
    SEXP out = sd_tensor_to_r(den);
    sd_tensor_free(&x_t);
    sd_cond_free(&c);
    sd_cond_free(&uc);
    sd_tensor_free(&den);
    return out;
}

// [[Rcpp::export]]
Rcpp::List sd_generate_image(SEXP ctx_sexp, Rcpp::List params) {
    SdCtxXPtr xptr(ctx_sexp);
    if (!xptr.get()) {
        Rcpp::stop("Invalid sd_ctx (NULL pointer)");
    }

    sd_img_gen_params_t p;
    sd_img_gen_params_init(&p);

    // Prompt strings - need stable storage
    std::string prompt_str, neg_prompt_str;

    if (params.containsElementNamed("prompt")) {
        prompt_str = Rcpp::as<std::string>(params["prompt"]);
        p.prompt = prompt_str.c_str();
    }
    if (params.containsElementNamed("negative_prompt")) {
        neg_prompt_str = Rcpp::as<std::string>(params["negative_prompt"]);
        p.negative_prompt = neg_prompt_str.c_str();
    }

    if (params.containsElementNamed("width"))
        p.width = Rcpp::as<int>(params["width"]);
    if (params.containsElementNamed("height"))
        p.height = Rcpp::as<int>(params["height"]);
    if (params.containsElementNamed("clip_skip"))
        p.clip_skip = Rcpp::as<int>(params["clip_skip"]);
    if (params.containsElementNamed("strength"))
        p.strength = Rcpp::as<float>(params["strength"]);
    if (params.containsElementNamed("seed"))
        p.seed = Rcpp::as<int64_t>(params["seed"]);
    if (params.containsElementNamed("batch_count"))
        p.batch_count = Rcpp::as<int>(params["batch_count"]);
    if (params.containsElementNamed("control_strength"))
        p.control_strength = Rcpp::as<float>(params["control_strength"]);

    // Sample params
    if (params.containsElementNamed("sample_method"))
        p.sample_params.sample_method = static_cast<sample_method_t>(Rcpp::as<int>(params["sample_method"]));
    if (params.containsElementNamed("sample_steps"))
        p.sample_params.sample_steps = Rcpp::as<int>(params["sample_steps"]);
    if (params.containsElementNamed("scheduler"))
        p.sample_params.scheduler = static_cast<scheduler_t>(Rcpp::as<int>(params["scheduler"]));
    if (params.containsElementNamed("cfg_scale"))
        p.sample_params.guidance.txt_cfg = Rcpp::as<float>(params["cfg_scale"]);
    if (params.containsElementNamed("eta"))
        p.sample_params.eta = Rcpp::as<float>(params["eta"]);
    if (params.containsElementNamed("flow_shift"))
        p.sample_params.flow_shift = Rcpp::as<float>(params["flow_shift"]);

    // VAE tiling
    if (params.containsElementNamed("vae_tiling") && Rcpp::as<bool>(params["vae_tiling"])) {
        p.vae_tiling_params.enabled = true;
        if (params.containsElementNamed("vae_tile_size")) {
            int ts = Rcpp::as<int>(params["vae_tile_size"]);
            p.vae_tiling_params.tile_size_x = ts;
            p.vae_tiling_params.tile_size_y = ts;
        }
        if (params.containsElementNamed("vae_tile_overlap"))
            p.vae_tiling_params.target_overlap = Rcpp::as<float>(params["vae_tile_overlap"]);
        if (params.containsElementNamed("vae_tile_rel_x"))
            p.vae_tiling_params.rel_size_x = Rcpp::as<float>(params["vae_tile_rel_x"]);
        if (params.containsElementNamed("vae_tile_rel_y"))
            p.vae_tiling_params.rel_size_y = Rcpp::as<float>(params["vae_tile_rel_y"]);
    }

    // Tiled sampling (MultiDiffusion)
    if (params.containsElementNamed("tiled_sampling") && Rcpp::as<bool>(params["tiled_sampling"])) {
        p.tiled_sample_params.enabled = true;
        if (params.containsElementNamed("sample_tile_size"))
            p.tiled_sample_params.tile_size = Rcpp::as<int>(params["sample_tile_size"]);
        if (params.containsElementNamed("sample_tile_overlap"))
            p.tiled_sample_params.tile_overlap = Rcpp::as<float>(params["sample_tile_overlap"]);
    }

    // Step caching (EasyCache / UCache / etc.)
    if (params.containsElementNamed("cache_mode")) {
        p.cache.mode = static_cast<sd_cache_mode_t>(Rcpp::as<int>(params["cache_mode"]));
        if (params.containsElementNamed("cache_threshold"))
            p.cache.reuse_threshold = Rcpp::as<float>(params["cache_threshold"]);
        if (params.containsElementNamed("cache_start"))
            p.cache.start_percent = Rcpp::as<float>(params["cache_start"]);
        if (params.containsElementNamed("cache_end"))
            p.cache.end_percent = Rcpp::as<float>(params["cache_end"]);
    }

    // Init image (for img2img)
    if (params.containsElementNamed("init_image") && !Rf_isNull(params["init_image"])) {
        p.init_image = r_to_sd_image(Rcpp::as<Rcpp::List>(params["init_image"]));
    }

    // Mask image (for inpaint). 1-channel grayscale: white (255) = generate,
    // black (0) = keep original. If absent, stable-diffusion.cpp creates an
    // all-white mask at the correct aligned size (plain img2img).
    if (params.containsElementNamed("mask_image") && !Rf_isNull(params["mask_image"])) {
        p.mask_image = r_to_sd_image(Rcpp::as<Rcpp::List>(params["mask_image"]));
    }

    // Control image
    if (params.containsElementNamed("control_image") && !Rf_isNull(params["control_image"])) {
        p.control_image = r_to_sd_image(Rcpp::as<Rcpp::List>(params["control_image"]));
    }

    // Reference images (multi-reference, TODO 9.7). ref_images is a list of
    // sd_image lists. r_to_sd_image points at R-owned RawVector data, so the
    // source R lists (held in `ref_list`) and the sd_image_t array
    // (`ref_vec`) must outlive the generate_image() call below — both are
    // function-local and stay in scope until return.
    std::vector<sd_image_t> ref_vec;
    Rcpp::List ref_list;  // keeps RawVector data alive
    if (params.containsElementNamed("ref_images") && !Rf_isNull(params["ref_images"])) {
        ref_list = Rcpp::as<Rcpp::List>(params["ref_images"]);
        for (R_xlen_t i = 0; i < ref_list.size(); i++) {
            ref_vec.push_back(r_to_sd_image(Rcpp::as<Rcpp::List>(ref_list[i])));
        }
        if (!ref_vec.empty()) {
            p.ref_images       = ref_vec.data();
            p.ref_images_count = (int)ref_vec.size();
        }
    }
    if (params.containsElementNamed("auto_resize_ref_image"))
        p.auto_resize_ref_image = Rcpp::as<bool>(params["auto_resize_ref_image"]);
    if (params.containsElementNamed("increase_ref_index"))
        p.increase_ref_index = Rcpp::as<bool>(params["increase_ref_index"]);

    // Profile: mark text_encode start and generate_total start
    if (r_sd_profiling) {
        double ts = profile_now_ms();
        r_profile_events.push_back({"generate_total", "start", ts});
        r_profile_events.push_back({"text_encode", "start", ts});
    }

    sd_image_t* results = generate_image(xptr.get(), &p);

    if (!results) {
        Rcpp::stop("Image generation failed");
    }

    // Convert results to R list
    int batch = (p.batch_count > 0) ? p.batch_count : 1;
    Rcpp::List output(batch);
    for (int i = 0; i < batch; i++) {
        output[i] = sd_image_to_r(results[i]);
        free(results[i].data);
    }
    free(results);

    return output;
}

// [[Rcpp::export]]
std::string sd_system_info_cpp() {
    const char* info = sd_get_system_info();
    return info ? std::string(info) : "";
}

// [[Rcpp::export]]
std::string sd_version_cpp() {
    const char* v = sd_version();
    return v ? std::string(v) : "";
}

// [[Rcpp::export]]
int sd_num_physical_cores_cpp() {
    return sd_get_num_physical_cores();
}

// [[Rcpp::export]]
std::string sd_type_name_cpp(int type) {
    const char* name = sd_type_name(static_cast<sd_type_t>(type));
    return name ? std::string(name) : "";
}

// [[Rcpp::export]]
std::string sd_sample_method_name_cpp(int method) {
    const char* name = sd_sample_method_name(static_cast<sample_method_t>(method));
    return name ? std::string(name) : "";
}

// [[Rcpp::export]]
std::string sd_scheduler_name_cpp(int sched) {
    const char* name = sd_scheduler_name(static_cast<scheduler_t>(sched));
    return name ? std::string(name) : "";
}

// --- Upscaler ---
// [[Rcpp::export]]
SEXP sd_create_upscaler(std::string esrgan_path, int n_threads = 0,
                         bool offload_params_to_cpu = false,
                         bool direct = false, int tile_size = 0) {
    upscaler_ctx_t* ctx = new_upscaler_ctx(
        esrgan_path.c_str(), offload_params_to_cpu, direct, n_threads, tile_size,
        nullptr, nullptr  // backend / params_backend: use defaults
    );
    if (!ctx) {
        Rcpp::stop("Failed to create upscaler context");
    }
    UpscalerCtxXPtr xptr(ctx, true);
    xptr.attr("class") = "upscaler_ctx";
    return xptr;
}

// [[Rcpp::export]]
Rcpp::List sd_upscale(SEXP upscaler_sexp, Rcpp::List image, int upscale_factor) {
    UpscalerCtxXPtr xptr(upscaler_sexp);
    if (!xptr.get()) {
        Rcpp::stop("Invalid upscaler_ctx (NULL pointer)");
    }

    sd_image_t input = r_to_sd_image(image);
    sd_image_t result = upscale(xptr.get(), input, (uint32_t)upscale_factor);
    Rcpp::List out = sd_image_to_r(result);
    free(result.data);
    return out;
}

// [[Rcpp::export]]
bool sd_convert_model(std::string input_path, std::string output_path,
                      int output_type, std::string vae_path = "",
                      std::string tensor_type_rules = "",
                      bool convert_name = false) {
    return convert(
        input_path.c_str(),
        vae_path.empty() ? nullptr : vae_path.c_str(),
        output_path.c_str(),
        static_cast<sd_type_t>(output_type),
        tensor_type_rules.empty() ? nullptr : tensor_type_rules.c_str(),
        convert_name
    );
}

// ============================================================
// --- Async generation (std::thread, polled from R/Shiny) ---
// ============================================================

// State shared between worker thread and R polling
struct AsyncGenState {
    std::mutex mtx;
    std::thread worker;

    // inputs (owned copies — safe to use from thread)
    sd_ctx_t* ctx = nullptr;           // borrowed pointer, lives in R XPtr
    SEXP ctx_sexp_protected = nullptr;  // prevent GC of XPtr during async
    sd_img_gen_params_t params;
    std::string prompt_str;
    std::string neg_prompt_str;
    std::vector<uint8_t> init_image_data;
    std::vector<uint8_t> mask_image_data;
    std::vector<uint8_t> control_image_data;
    int batch_count = 1;

    // outputs
    std::atomic<bool> running{false};
    std::atomic<bool> done{false};
    sd_image_t* results = nullptr;     // owned by thread, consumed by R
    std::string error_msg;
};

static AsyncGenState g_async;

// Release GC protection on ctx XPtr (call from main R thread only)
static void async_release_ctx() {
    if (g_async.ctx_sexp_protected != nullptr) {
        R_ReleaseObject(g_async.ctx_sexp_protected);
        g_async.ctx_sexp_protected = nullptr;
    }
}

// Ensure previous worker thread is fully finished before reuse
static void async_join_worker() {
    if (g_async.worker.joinable()) g_async.worker.join();
}

// SIGSEGV handler for worker thread — catch crash before R's handler
#ifndef _WIN32
#include <signal.h>
#include <setjmp.h>
static thread_local sigjmp_buf worker_jmpbuf;
static thread_local bool worker_has_jmpbuf = false;

static void worker_sigsegv_handler(int sig) {
    if (worker_has_jmpbuf) {
        worker_has_jmpbuf = false;
        siglongjmp(worker_jmpbuf, sig);
    }
    // If no jmpbuf, let it crash normally
    signal(sig, SIG_DFL);
    raise(sig);
}
#endif

// Worker function — runs in std::thread
static void async_worker() {
    r_sd_async_running.store(true);
    try {
        if (!sd_ctx_is_valid(g_async.ctx)) {
            g_async.results = nullptr;
            g_async.error_msg = "ctx is invalid (NULL or sd==NULL)";
            r_sd_async_running.store(false);
            g_async.done.store(true);
            g_async.running.store(false);
            return;
        }

#ifndef _WIN32
        // Install thread-local SIGSEGV handler to catch crash before R does
        struct sigaction sa_new, sa_old;
        memset(&sa_new, 0, sizeof(sa_new));
        sa_new.sa_handler = worker_sigsegv_handler;
        sigemptyset(&sa_new.sa_mask);
        sa_new.sa_flags = 0;
        sigaction(SIGSEGV, &sa_new, &sa_old);

        int sig = sigsetjmp(worker_jmpbuf, 1);
        if (sig == 0) {
            worker_has_jmpbuf = true;
            g_async.results = generate_image(g_async.ctx, &g_async.params);
            worker_has_jmpbuf = false;
        } else {
            g_async.results = nullptr;
            g_async.error_msg = "SIGSEGV in generate_image";
        }
        // Restore R's signal handler
        sigaction(SIGSEGV, &sa_old, nullptr);
#else
        g_async.results = generate_image(g_async.ctx, &g_async.params);
#endif
        if (!g_async.results) {
            g_async.error_msg = "Image generation failed";
        }
    } catch (const std::exception& e) {
        g_async.results = nullptr;
        g_async.error_msg = std::string("Exception: ") + e.what();
    } catch (...) {
        g_async.results = nullptr;
        g_async.error_msg = "Unknown exception during generation";
    }
    // NOTE: r_sd_async_running cleared here, but GC protection released
    // only from main R thread in sd_generate_result() or next sd_generate_async()
    r_sd_async_running.store(false);
    g_async.done.store(true);
    g_async.running.store(false);
}

// [[Rcpp::export]]
bool sd_generate_async(SEXP ctx_sexp, Rcpp::List params) {
    if (g_async.running.load()) {
        Rcpp::stop("Generation already in progress");
    }

    // Extract raw pointer without creating a temporary XPtr
    // (XPtr constructor/destructor with PreserveStorage causes extra
    // R_PreserveObject/R_ReleaseObject that interfere with our GC protection)
    sd_ctx_t* ctx_raw = reinterpret_cast<sd_ctx_t*>(R_ExternalPtrAddr(ctx_sexp));
    if (!ctx_raw) {
        Rcpp::stop("Invalid sd_ctx (NULL pointer)");
    }

    // Ensure previous worker thread is fully finished
    async_join_worker();

    // Release GC protection from previous run (if not already released)
    async_release_ctx();

    // Reset state completely
    g_async.done.store(false);
    g_async.running.store(false);
    g_async.error_msg.clear();
    g_async.init_image_data.clear();
    g_async.mask_image_data.clear();
    g_async.control_image_data.clear();
    g_async.prompt_str.clear();
    g_async.neg_prompt_str.clear();
    if (g_async.results) {
        // Clean up unretrieved results from previous run
        for (int i = 0; i < g_async.batch_count; i++) {
            free(g_async.results[i].data);
        }
        free(g_async.results);
        g_async.results = nullptr;
    }
    g_async.batch_count = 1;
    g_async.running.store(true);
    g_async.ctx = ctx_raw;

    // Protect XPtr SEXP from GC during async generation
    R_PreserveObject(ctx_sexp);
    g_async.ctx_sexp_protected = ctx_sexp;

    // Copy params (same logic as sd_generate_image)
    sd_img_gen_params_t& p = g_async.params;
    sd_img_gen_params_init(&p);

    if (params.containsElementNamed("prompt")) {
        g_async.prompt_str = Rcpp::as<std::string>(params["prompt"]);
        p.prompt = g_async.prompt_str.c_str();
    }
    if (params.containsElementNamed("negative_prompt")) {
        g_async.neg_prompt_str = Rcpp::as<std::string>(params["negative_prompt"]);
        p.negative_prompt = g_async.neg_prompt_str.c_str();
    }

    if (params.containsElementNamed("width"))
        p.width = Rcpp::as<int>(params["width"]);
    if (params.containsElementNamed("height"))
        p.height = Rcpp::as<int>(params["height"]);
    if (params.containsElementNamed("clip_skip"))
        p.clip_skip = Rcpp::as<int>(params["clip_skip"]);
    if (params.containsElementNamed("strength"))
        p.strength = Rcpp::as<float>(params["strength"]);
    if (params.containsElementNamed("seed"))
        p.seed = Rcpp::as<int64_t>(params["seed"]);
    if (params.containsElementNamed("batch_count"))
        p.batch_count = Rcpp::as<int>(params["batch_count"]);
    if (params.containsElementNamed("control_strength"))
        p.control_strength = Rcpp::as<float>(params["control_strength"]);

    if (params.containsElementNamed("sample_method"))
        p.sample_params.sample_method = static_cast<sample_method_t>(Rcpp::as<int>(params["sample_method"]));
    if (params.containsElementNamed("sample_steps"))
        p.sample_params.sample_steps = Rcpp::as<int>(params["sample_steps"]);
    if (params.containsElementNamed("scheduler"))
        p.sample_params.scheduler = static_cast<scheduler_t>(Rcpp::as<int>(params["scheduler"]));
    if (params.containsElementNamed("cfg_scale"))
        p.sample_params.guidance.txt_cfg = Rcpp::as<float>(params["cfg_scale"]);
    if (params.containsElementNamed("eta"))
        p.sample_params.eta = Rcpp::as<float>(params["eta"]);
    if (params.containsElementNamed("flow_shift"))
        p.sample_params.flow_shift = Rcpp::as<float>(params["flow_shift"]);

    // VAE tiling
    if (params.containsElementNamed("vae_tiling") && Rcpp::as<bool>(params["vae_tiling"])) {
        p.vae_tiling_params.enabled = true;
        if (params.containsElementNamed("vae_tile_size")) {
            int ts = Rcpp::as<int>(params["vae_tile_size"]);
            p.vae_tiling_params.tile_size_x = ts;
            p.vae_tiling_params.tile_size_y = ts;
        }
        if (params.containsElementNamed("vae_tile_overlap"))
            p.vae_tiling_params.target_overlap = Rcpp::as<float>(params["vae_tile_overlap"]);
    }

    // Tiled sampling
    if (params.containsElementNamed("tiled_sampling") && Rcpp::as<bool>(params["tiled_sampling"])) {
        p.tiled_sample_params.enabled = true;
        if (params.containsElementNamed("sample_tile_size"))
            p.tiled_sample_params.tile_size = Rcpp::as<int>(params["sample_tile_size"]);
        if (params.containsElementNamed("sample_tile_overlap"))
            p.tiled_sample_params.tile_overlap = Rcpp::as<float>(params["sample_tile_overlap"]);
    }

    // Step caching
    if (params.containsElementNamed("cache_mode")) {
        p.cache.mode = static_cast<sd_cache_mode_t>(Rcpp::as<int>(params["cache_mode"]));
        if (params.containsElementNamed("cache_threshold"))
            p.cache.reuse_threshold = Rcpp::as<float>(params["cache_threshold"]);
        if (params.containsElementNamed("cache_start"))
            p.cache.start_percent = Rcpp::as<float>(params["cache_start"]);
        if (params.containsElementNamed("cache_end"))
            p.cache.end_percent = Rcpp::as<float>(params["cache_end"]);
    }

    // Init image — deep copy pixel data
    if (params.containsElementNamed("init_image") && !Rf_isNull(params["init_image"])) {
        Rcpp::List img_list = Rcpp::as<Rcpp::List>(params["init_image"]);
        p.init_image.width = Rcpp::as<uint32_t>(img_list["width"]);
        p.init_image.height = Rcpp::as<uint32_t>(img_list["height"]);
        p.init_image.channel = Rcpp::as<uint32_t>(img_list["channel"]);
        Rcpp::RawVector data = Rcpp::as<Rcpp::RawVector>(img_list["data"]);
        g_async.init_image_data.assign(data.begin(), data.end());
        p.init_image.data = g_async.init_image_data.data();
    }

    // Mask image (for inpaint) — deep copy. 1-channel grayscale; white = generate.
    if (params.containsElementNamed("mask_image") && !Rf_isNull(params["mask_image"])) {
        Rcpp::List img_list = Rcpp::as<Rcpp::List>(params["mask_image"]);
        p.mask_image.width = Rcpp::as<uint32_t>(img_list["width"]);
        p.mask_image.height = Rcpp::as<uint32_t>(img_list["height"]);
        p.mask_image.channel = Rcpp::as<uint32_t>(img_list["channel"]);
        Rcpp::RawVector data = Rcpp::as<Rcpp::RawVector>(img_list["data"]);
        g_async.mask_image_data.assign(data.begin(), data.end());
        p.mask_image.data = g_async.mask_image_data.data();
    }

    // Control image — deep copy
    if (params.containsElementNamed("control_image") && !Rf_isNull(params["control_image"])) {
        Rcpp::List img_list = Rcpp::as<Rcpp::List>(params["control_image"]);
        p.control_image.width = Rcpp::as<uint32_t>(img_list["width"]);
        p.control_image.height = Rcpp::as<uint32_t>(img_list["height"]);
        p.control_image.channel = Rcpp::as<uint32_t>(img_list["channel"]);
        Rcpp::RawVector data = Rcpp::as<Rcpp::RawVector>(img_list["data"]);
        g_async.control_image_data.assign(data.begin(), data.end());
        p.control_image.data = g_async.control_image_data.data();
    }

    g_async.batch_count = (p.batch_count > 0) ? p.batch_count : 1;

    // Profiling
    if (r_sd_profiling) {
        double ts = profile_now_ms();
        r_profile_events.push_back({"generate_total", "start", ts});
        r_profile_events.push_back({"text_encode", "start", ts});
    }

    // Launch worker thread (join already done above)
    g_async.worker = std::thread(async_worker);

    return true;
}

// [[Rcpp::export]]
Rcpp::List sd_generate_poll() {
    bool done = g_async.done.load();
    bool running = g_async.running.load();
    return Rcpp::List::create(
        Rcpp::Named("running") = running,
        Rcpp::Named("done") = done
    );
}

// [[Rcpp::export]]
Rcpp::List sd_generate_result() {
    if (!g_async.done.load()) {
        Rcpp::stop("Generation not finished yet");
    }
    async_join_worker();

    // NOTE: Do NOT release GC protection here — ctx must stay protected
    // until the next sd_generate_async() call (which does its own release+reprotect).
    // Releasing here allows GC to collect the XPtr between generations,
    // causing segfault on the next call.
    // async_release_ctx();  // REMOVED — was causing use-after-free

    if (!g_async.error_msg.empty()) {
        std::string err = g_async.error_msg;
        g_async.error_msg.clear();
        Rcpp::stop(err);
    }

    int batch = g_async.batch_count;
    Rcpp::List output(batch);
    for (int i = 0; i < batch; i++) {
        output[i] = sd_image_to_r(g_async.results[i]);
        free(g_async.results[i].data);
    }
    free(g_async.results);
    g_async.results = nullptr;

    return output;
}
