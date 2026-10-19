// sd2R: shared declarations for the Rcpp bridge (src/sd2r_*.cpp).
#ifndef SD2R_INTERFACE_H
#define SD2R_INTERFACE_H

#include <Rcpp.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "sd/stable-diffusion.h"

// --- Flags and profiling state (sd2r_callbacks.cpp) ---
extern bool r_sd_verbose;
extern bool r_sd_profiling;
// When true, callbacks run on a worker thread and must NOT call the R API.
extern std::atomic<bool> r_sd_async_running;

struct ProfileEvent {
    std::string stage;
    std::string kind;  // "start" or "end"
    double timestamp_ms;
};

extern std::vector<ProfileEvent> r_profile_events;
double profile_now_ms();
// Marks generate_total/text_encode start before a generate_image() call.
void profile_mark_generate_start();

// --- XPtr types with custom deleters (avoids delete-incomplete warning) ---
inline void sd_ctx_invoke_free(sd_ctx_t* ctx) {
    if (ctx) free_sd_ctx(ctx);
}
inline void upscaler_ctx_invoke_free(upscaler_ctx_t* ctx) {
    if (ctx) free_upscaler_ctx(ctx);
}
typedef Rcpp::XPtr<sd_ctx_t, Rcpp::PreserveStorage, sd_ctx_invoke_free> SdCtxXPtr;
typedef Rcpp::XPtr<upscaler_ctx_t, Rcpp::PreserveStorage, upscaler_ctx_invoke_free> UpscalerCtxXPtr;

// --- Image conversion (sd2r_params.cpp) ---
// sd_image_t -> R list(width, height, channel, data = raw).
Rcpp::List sd_image_to_r(const sd_image_t& img);
// R image list -> sd_image_t pointing at R-owned raw data (caller keeps the list alive).
sd_image_t r_to_sd_image(Rcpp::List img_list);

// --- Parameter parsing (sd2r_params.cpp) ---
// Owned storage for every string the C structs point to; must outlive the params.
struct CtxParamStorage {
    std::string model_path, clip_l, clip_g, clip_vision, t5xxl, llm, llm_vision;
    std::string diffusion_model, high_noise_diffusion_model, vae, taesd, control_net;
    std::string photo_maker, tensor_type_rules;
    std::string backend, params_backend, max_vram;
};

// R list (from sd_ctx()) -> sd_ctx_params_t. Also sets the meta-backend flag.
void sd2r_fill_ctx_params(Rcpp::List params, sd_ctx_params_t& p, CtxParamStorage& storage);

// Deep copies of everything sd_img_gen_params_t points to, so the params stay
// valid on a worker thread after the R call returns.
struct ImgGenStorage {
    std::string prompt, negative_prompt, ref_image_args;
    std::vector<uint8_t> init_image_data, mask_image_data, control_image_data;
    std::vector<std::vector<uint8_t>> ref_image_data;
    std::vector<sd_image_t> ref_images;
    std::vector<std::string> lora_paths;
    std::vector<sd_lora_t> loras;

    void clear();
};

// R list (from sd_generate()/sd_txt2img()/...) -> sd_img_gen_params_t.
void sd2r_fill_img_gen_params(Rcpp::List params, sd_img_gen_params_t& p, ImgGenStorage& storage);

#endif  // SD2R_INTERFACE_H
