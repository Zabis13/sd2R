#define R_GGML_IO_IMPL  // disable r_ggml_compat.h macros (we use R API directly)

#include <Rcpp.h>
#include "sd/stable-diffusion.h"

// --- Verbose flag: controls log and progress output ---
static bool r_sd_verbose = false;

// --- Log callback: route SD log messages to R ---
static void r_sd_log_callback(sd_log_level_t level, const char* text, void* data) {
    // Remove trailing newline for Rprintf
    std::string msg(text);
    while (!msg.empty() && msg.back() == '\n') msg.pop_back();
    if (msg.empty()) return;

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

// --- Progress callback: update R console ---
static void r_sd_progress_callback(int step, int steps, float time, void* data) {
    if (!r_sd_verbose) return;
    Rprintf("\rStep %d/%d (%.1fs)", step, steps, time);
    if (step == steps) Rprintf("\n");
    R_FlushConsole();
    R_CheckUserInterrupt();
}

// [[Rcpp::export]]
void sd_set_verbose(bool verbose) {
    r_sd_verbose = verbose;
}

// --- Destructor for XPtr ---
static void sd_ctx_destructor(sd_ctx_t* ctx) {
    if (ctx) {
        free_sd_ctx(ctx);
    }
}

static void upscaler_ctx_destructor(upscaler_ctx_t* ctx) {
    if (ctx) {
        free_upscaler_ctx(ctx);
    }
}

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
    if (params.containsElementNamed("enable_mmap"))
        p.enable_mmap = Rcpp::as<bool>(params["enable_mmap"]);
    if (params.containsElementNamed("keep_clip_on_cpu"))
        p.keep_clip_on_cpu = Rcpp::as<bool>(params["keep_clip_on_cpu"]);
    if (params.containsElementNamed("keep_control_net_on_cpu"))
        p.keep_control_net_on_cpu = Rcpp::as<bool>(params["keep_control_net_on_cpu"]);
    if (params.containsElementNamed("keep_vae_on_cpu"))
        p.keep_vae_on_cpu = Rcpp::as<bool>(params["keep_vae_on_cpu"]);
    if (params.containsElementNamed("diffusion_flash_attn"))
        p.diffusion_flash_attn = Rcpp::as<bool>(params["diffusion_flash_attn"]);
    if (params.containsElementNamed("flow_shift"))
        p.flow_shift = Rcpp::as<float>(params["flow_shift"]);

    sd_ctx_t* ctx = new_sd_ctx(&p);
    if (!ctx) {
        Rcpp::stop("Failed to create stable diffusion context");
    }

    Rcpp::XPtr<sd_ctx_t> xptr(ctx, false);
    R_RegisterCFinalizerEx(xptr, [](SEXP p) {
        sd_ctx_t* c = (sd_ctx_t*)R_ExternalPtrAddr(p);
        if (c) { free_sd_ctx(c); R_ClearExternalPtr(p); }
    }, TRUE);
    xptr.attr("class") = "sd_ctx";
    return xptr;
}

// [[Rcpp::export]]
void sd_destroy_context(SEXP ctx_sexp) {
    Rcpp::XPtr<sd_ctx_t> xptr(ctx_sexp);
    if (xptr.get()) {
        free_sd_ctx(xptr.get());
        xptr.release();
    }
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

// [[Rcpp::export]]
Rcpp::List sd_generate_image(SEXP ctx_sexp, Rcpp::List params) {
    Rcpp::XPtr<sd_ctx_t> xptr(ctx_sexp);
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

    // Init image (for img2img)
    // Note: mask_image is left empty (sd_image_t{}) — stable-diffusion.cpp
    // creates an all-white mask at the correct aligned size if none is provided.
    if (params.containsElementNamed("init_image") && !Rf_isNull(params["init_image"])) {
        p.init_image = r_to_sd_image(Rcpp::as<Rcpp::List>(params["init_image"]));
    }

    // Control image
    if (params.containsElementNamed("control_image") && !Rf_isNull(params["control_image"])) {
        p.control_image = r_to_sd_image(Rcpp::as<Rcpp::List>(params["control_image"]));
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
        esrgan_path.c_str(), offload_params_to_cpu, direct, n_threads, tile_size
    );
    if (!ctx) {
        Rcpp::stop("Failed to create upscaler context");
    }
    Rcpp::XPtr<upscaler_ctx_t> xptr(ctx, false);
    R_RegisterCFinalizerEx(xptr, [](SEXP p) {
        upscaler_ctx_t* c = (upscaler_ctx_t*)R_ExternalPtrAddr(p);
        if (c) { free_upscaler_ctx(c); R_ClearExternalPtr(p); }
    }, TRUE);
    xptr.attr("class") = "upscaler_ctx";
    return xptr;
}

// [[Rcpp::export]]
Rcpp::List sd_upscale(SEXP upscaler_sexp, Rcpp::List image, int upscale_factor) {
    Rcpp::XPtr<upscaler_ctx_t> xptr(upscaler_sexp);
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
