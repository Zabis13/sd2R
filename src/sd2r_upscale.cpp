// sd2R: ESRGAN upscaler and model conversion.
#include "sd2r_interface.h"

// [[Rcpp::export]]
SEXP sd_create_upscaler(std::string esrgan_path, int n_threads = 0,
                        bool offload_params_to_cpu = false,
                        bool direct = false, int tile_size = 0) {
    // offload_params_to_cpu: keep the weights on the CPU (params backend) and
    // run on the default runtime backend.
    upscaler_ctx_t* ctx = new_upscaler_ctx(
        esrgan_path.c_str(), direct, n_threads, tile_size,
        nullptr,                                   // backend: default device
        offload_params_to_cpu ? "cpu" : nullptr);  // params_backend
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

    sd_image_t input    = r_to_sd_image(image);
    sd_image_t* results = nullptr;
    int count           = 0;
    if (!upscale(xptr.get(), input, (uint32_t)upscale_factor, &results, &count) ||
        results == nullptr || count < 1) {
        free_sd_images(results, count);
        Rcpp::stop("Upscaling failed");
    }
    Rcpp::List out = sd_image_to_r(results[0]);
    free_sd_images(results, count);
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
