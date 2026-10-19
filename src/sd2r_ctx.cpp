// sd2R: context creation (sync + async) and info helpers.
#include <thread>

#include "sd/sd2r_meta_backend.hpp"  // sd2R: meta backend (multi-GPU tensor split, behind a flag)
#include "sd2r_interface.h"

// [[Rcpp::export]]
SEXP sd_create_context(Rcpp::List params) {
    sd_ctx_params_t p;
    CtxParamStorage storage;
    sd2r_fill_ctx_params(params, p, storage);

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
void sd_destroy_context_impl(SEXP ctx_sexp) {
    // Free the context explicitly and clear the external pointer IN the SEXP so
    // the XPtr finalizer (sd_ctx_invoke_free -> free_sd_ctx) cannot run a second
    // time during GC. XPtr::release() only nulls the temporary wrapper, not the
    // underlying SEXP, which left the finalizer armed and caused a double free
    // (segfault) when the same ctx was later garbage-collected — e.g. on the
    // second "Load Model" in the Shiny app. R_ClearExternalPtr zeroes the
    // address in the SEXP itself, so the finalizer sees NULL and is a no-op.
    if (TYPEOF(ctx_sexp) != EXTPTRSXP) {
        return;
    }
    sd_ctx_t* ctx = reinterpret_cast<sd_ctx_t*>(R_ExternalPtrAddr(ctx_sexp));
    if (ctx) {
        free_sd_ctx(ctx);
        R_ClearExternalPtr(ctx_sexp);
    }
}

// ============================================================
// --- Async context creation (std::thread, polled from R) ---
// ============================================================

struct AsyncCtxState {
    std::thread worker;
    sd_ctx_params_t params;
    CtxParamStorage storage;  // must outlive params

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

    // Parse on the R thread (the meta-backend flag read by new_sd_ctx is set here, before launch)
    sd2r_fill_ctx_params(params, g_async_ctx.params, g_async_ctx.storage);

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
