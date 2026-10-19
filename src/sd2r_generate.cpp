// sd2R: image generation (sync + async worker thread polled from R/Shiny).
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#include "sd2r_interface.h"

// Converts generate_image() output to an R list and frees it.
static Rcpp::List images_to_r(sd_image_t* images, int count) {
    Rcpp::List output(count > 0 ? count : 0);
    for (int i = 0; i < count; i++) {
        output[i] = sd_image_to_r(images[i]);
    }
    free_sd_images(images, count);
    return output;
}

// [[Rcpp::export]]
Rcpp::List sd_generate_image(SEXP ctx_sexp, Rcpp::List params) {
    SdCtxXPtr xptr(ctx_sexp);
    if (!xptr.get()) {
        Rcpp::stop("Invalid sd_ctx (NULL pointer)");
    }

    sd_img_gen_params_t p;
    ImgGenStorage storage;
    sd2r_fill_img_gen_params(params, p, storage);

    profile_mark_generate_start();

    sd_image_t* results = nullptr;
    int count           = 0;
    if (!generate_image(xptr.get(), &p, &results, &count) || results == nullptr) {
        free_sd_images(results, count);
        Rcpp::stop("Image generation failed");
    }
    return images_to_r(results, count);
}

// ============================================================
// --- Async generation (std::thread, polled from R/Shiny) ---
// ============================================================

// State shared between worker thread and R polling
struct AsyncGenState {
    std::mutex mtx;
    std::thread worker;

    // inputs (owned copies — safe to use from thread)
    sd_ctx_t* ctx = nullptr;            // borrowed pointer, lives in R XPtr
    SEXP ctx_sexp_protected = nullptr;  // prevent GC of XPtr during async
    sd_img_gen_params_t params;
    ImgGenStorage storage;

    // outputs
    std::atomic<bool> running{false};
    std::atomic<bool> done{false};
    sd_image_t* results = nullptr;  // owned by thread, consumed by R
    int result_count    = 0;
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

static bool async_run_generate() {
    return generate_image(g_async.ctx, &g_async.params, &g_async.results, &g_async.result_count) &&
           g_async.results != nullptr;
}

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

        bool ok = false;
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
            ok = async_run_generate();
            worker_has_jmpbuf = false;
        } else {
            g_async.results = nullptr;
            g_async.result_count = 0;
            g_async.error_msg = "SIGSEGV in generate_image";
        }
        // Restore R's signal handler
        sigaction(SIGSEGV, &sa_old, nullptr);
#else
        ok = async_run_generate();
#endif
        if (!ok && g_async.error_msg.empty()) {
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
    if (g_async.results) {
        // Clean up unretrieved results from previous run
        free_sd_images(g_async.results, g_async.result_count);
        g_async.results = nullptr;
    }
    g_async.result_count = 0;

    // Parse params into owned storage (same logic as sd_generate_image)
    sd2r_fill_img_gen_params(params, g_async.params, g_async.storage);

    g_async.running.store(true);
    g_async.ctx = ctx_raw;

    // Protect XPtr SEXP from GC during async generation
    R_PreserveObject(ctx_sexp);
    g_async.ctx_sexp_protected = ctx_sexp;

    profile_mark_generate_start();

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

    if (!g_async.error_msg.empty()) {
        std::string err = g_async.error_msg;
        g_async.error_msg.clear();
        if (g_async.results) {
            free_sd_images(g_async.results, g_async.result_count);
            g_async.results = nullptr;
            g_async.result_count = 0;
        }
        Rcpp::stop(err);
    }

    Rcpp::List output = images_to_r(g_async.results, g_async.result_count);
    g_async.results = nullptr;
    g_async.result_count = 0;
    return output;
}
