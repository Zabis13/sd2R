// sd2R: Rcpp bridge for the low-level step-wise API (sd/sd2r_lowlevel.inc).
#include <cstdlib>
#include <vector>

#include "sd2r_interface.h"

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
int sd_ctx_lora_apply_mode_cpp(SEXP ctx_sexp) {
    SdCtxXPtr xptr(ctx_sexp);
    if (!xptr.get()) return -1;
    return sd_ctx_lora_apply_mode(xptr.get());
}

// [[Rcpp::export]]
Rcpp::DataFrame sd_ctx_lora_stats_cpp(SEXP ctx_sexp) {
    SdCtxXPtr xptr(ctx_sexp);
    if (!xptr.get()) Rcpp::stop("Invalid sd_ctx (NULL pointer)");
    int n = sd_ctx_lora_stats(xptr.get(), nullptr, 0);
    std::vector<sd_lora_stat_t> stats(n > 0 ? n : 0);
    if (n > 0) n = sd_ctx_lora_stats(xptr.get(), stats.data(), n);
    Rcpp::CharacterVector path(n);
    Rcpp::NumericVector total(n), applied(n), skipped(n);
    for (int i = 0; i < n; i++) {
        path[i]    = stats[i].path;
        total[i]   = static_cast<double>(stats[i].total);
        applied[i] = static_cast<double>(stats[i].applied);
        skipped[i] = static_cast<double>(stats[i].skipped);
    }
    return Rcpp::DataFrame::create(Rcpp::Named("path") = path, Rcpp::Named("total") = total,
                                   Rcpp::Named("applied") = applied, Rcpp::Named("skipped") = skipped,
                                   Rcpp::Named("stringsAsFactors") = false);
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
    if (img.data == nullptr) Rcpp::stop("sd_decode_latent failed (see log)");
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
                         double cfg_scale, int step, int total_steps,
                         double sigma_next) {
    SdCtxXPtr xptr(ctx_sexp);
    if (!xptr.get()) Rcpp::stop("Invalid sd_ctx (NULL pointer)");
    sd_tensor x_t = r_to_sd_tensor(x_in);
    sd_cond c  = r_to_sd_cond(cond);
    sd_cond uc = r_to_sd_cond(uncond);
    sd_tensor den = sd_denoise_step(xptr.get(), x_t, (float)sigma, (float)sigma_next,
                                    c, uc, (float)cfg_scale, step, total_steps);
    SEXP out = sd_tensor_to_r(den);
    sd_tensor_free(&x_t);
    sd_cond_free(&c);
    sd_cond_free(&uc);
    sd_tensor_free(&den);
    if (out == R_NilValue) {
        Rcpp::stop("sd_denoise_step failed (see log; FLUX.3 flow-map models need sigma_next)");
    }
    return out;
}

// [[Rcpp::export]]
SEXP sd_euler_step_cpp(SEXP x_in, SEXP den_in, double sigma, double sigma_next) {
    sd_tensor x_t   = r_to_sd_tensor(x_in);
    sd_tensor den_t = r_to_sd_tensor(den_in);
    sd_tensor r     = sd_euler_step(x_t, den_t, (float)sigma, (float)sigma_next);
    SEXP out = sd_tensor_to_r(r);
    sd_tensor_free(&x_t);
    sd_tensor_free(&den_t);
    sd_tensor_free(&r);
    if (out == R_NilValue) {
        Rcpp::stop("sd_euler_step failed: x and denoised must be non-empty and of equal size");
    }
    return out;
}
