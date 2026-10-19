// sd2R: R list -> sd_ctx_params_t / sd_img_gen_params_t, image conversion.
#include <cstdio>
#include <cstring>

#include "sd/sd2r_meta_backend.hpp"  // sd2R: meta backend (multi-GPU tensor split, behind a flag)
#include "sd2r_interface.h"

Rcpp::List sd_image_to_r(const sd_image_t& img) {
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

sd_image_t r_to_sd_image(Rcpp::List img_list) {
    sd_image_t img;
    img.width = Rcpp::as<uint32_t>(img_list["width"]);
    img.height = Rcpp::as<uint32_t>(img_list["height"]);
    img.channel = Rcpp::as<uint32_t>(img_list["channel"]);
    Rcpp::RawVector data = Rcpp::as<Rcpp::RawVector>(img_list["data"]);
    img.data = (uint8_t*)&data[0];
    return img;
}

static bool has(const Rcpp::List& params, const char* name) {
    return params.containsElementNamed(name) && !Rf_isNull(params[name]);
}

static bool flag(const Rcpp::List& params, const char* name) {
    return has(params, name) && Rcpp::as<bool>(params[name]);
}

static void set_str(const Rcpp::List& params, const char* name,
                    std::string& storage, const char*& target) {
    if (has(params, name)) {
        storage = Rcpp::as<std::string>(params[name]);
        target  = storage.c_str();
    }
}

// Copies an R image list into owned storage and points `img` at it.
static void copy_image(const Rcpp::List& params, const char* name,
                       std::vector<uint8_t>& storage, sd_image_t& img) {
    if (!has(params, name)) {
        return;
    }
    Rcpp::List img_list  = Rcpp::as<Rcpp::List>(params[name]);
    img.width            = Rcpp::as<uint32_t>(img_list["width"]);
    img.height           = Rcpp::as<uint32_t>(img_list["height"]);
    img.channel          = Rcpp::as<uint32_t>(img_list["channel"]);
    Rcpp::RawVector data = Rcpp::as<Rcpp::RawVector>(img_list["data"]);
    storage.assign(data.begin(), data.end());
    img.data = storage.data();
}

void sd2r_fill_ctx_params(Rcpp::List params, sd_ctx_params_t& p, CtxParamStorage& s) {
    sd_ctx_params_init(&p);

    set_str(params, "model_path", s.model_path, p.model_path);
    set_str(params, "clip_l_path", s.clip_l, p.clip_l_path);
    set_str(params, "clip_g_path", s.clip_g, p.clip_g_path);
    set_str(params, "clip_vision_path", s.clip_vision, p.clip_vision_path);
    set_str(params, "t5xxl_path", s.t5xxl, p.t5xxl_path);
    set_str(params, "llm_path", s.llm, p.llm_path);
    set_str(params, "llm_vision_path", s.llm_vision, p.llm_vision_path);
    set_str(params, "diffusion_model_path", s.diffusion_model, p.diffusion_model_path);
    set_str(params, "high_noise_diffusion_model_path", s.high_noise_diffusion_model, p.high_noise_diffusion_model_path);
    set_str(params, "vae_path", s.vae, p.vae_path);
    set_str(params, "taesd_path", s.taesd, p.taesd_path);
    set_str(params, "control_net_path", s.control_net, p.control_net_path);
    set_str(params, "photo_maker_path", s.photo_maker, p.photo_maker_path);
    set_str(params, "tensor_type_rules", s.tensor_type_rules, p.tensor_type_rules);

    if (has(params, "n_threads"))
        p.n_threads = Rcpp::as<int>(params["n_threads"]);
    if (has(params, "wtype"))
        p.wtype = static_cast<sd_type_t>(Rcpp::as<int>(params["wtype"]));
    if (has(params, "rng_type"))
        p.rng_type = static_cast<rng_type_t>(Rcpp::as<int>(params["rng_type"]));
    if (has(params, "prediction"))
        p.prediction = static_cast<prediction_t>(Rcpp::as<int>(params["prediction"]));
    if (has(params, "lora_apply_mode"))
        p.lora_apply_mode = static_cast<lora_apply_mode_t>(Rcpp::as<int>(params["lora_apply_mode"]));
    if (has(params, "enable_mmap"))
        p.enable_mmap = Rcpp::as<bool>(params["enable_mmap"]);
    if (has(params, "eager_load"))
        p.eager_load = Rcpp::as<bool>(params["eager_load"]);
    if (has(params, "vae_conv_direct"))
        p.vae_conv_direct = Rcpp::as<bool>(params["vae_conv_direct"]);
    if (has(params, "diffusion_conv_direct"))
        p.diffusion_conv_direct = Rcpp::as<bool>(params["diffusion_conv_direct"]);
    if (has(params, "diffusion_flash_attn"))
        p.diffusion_flash_attn = Rcpp::as<bool>(params["diffusion_flash_attn"]);
    // Not in the 2026-10 C API (weights are prepared lazily / segmented by the
    // ModelManager): vae_decode_only, free_params_immediately, stream_layers.

    // max_vram: R keeps the old numeric meaning (0 = no budget, -1 = auto,
    // > 0 = GiB). The C API takes a per-device string where "0" means
    // "use live free VRAM".
    if (has(params, "max_vram")) {
        double mv = Rcpp::as<double>(params["max_vram"]);
        if (mv > 0) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%g", mv);
            s.max_vram = buf;
            p.max_vram = s.max_vram.c_str();
        } else if (mv < 0) {
            s.max_vram = "0";
            p.max_vram = s.max_vram.c_str();
        }
    }

    // offload_params_to_cpu: keep weights on the CPU and stream them to the
    // runtime backend per graph (params backend spec).
    if (flag(params, "offload_params_to_cpu")) {
        s.params_backend = "cpu";
        p.params_backend = s.params_backend.c_str();
    }

    // Runtime backend spec. Per-module GPU indices and keep_*_on_cpu flags map
    // onto "default,module=device,..." (e.g. "Vulkan0,te=cpu,vae=Vulkan1").
    // With nothing set p.backend stays null so get_default_backend_name()
    // (prefers a discrete GPU) picks the device.
    {
        auto gpu_idx = [&](const char* name) -> int {
            return has(params, name) ? Rcpp::as<int>(params[name]) : -1;
        };
        int diff_gpu = gpu_idx("diffusion_gpu_device");
        int clip_gpu = gpu_idx("clip_gpu_device");
        int vae_gpu  = gpu_idx("vae_gpu_device");

        std::vector<std::string> parts;
        if (diff_gpu >= 0 && clip_gpu < 0 && vae_gpu < 0) {
            // Only the diffusion index given: use it as the global default
            // (clip/vae follow the same device unless overridden).
            parts.push_back("Vulkan" + std::to_string(diff_gpu));
        } else {
            if (diff_gpu >= 0)
                parts.push_back("diffusion=Vulkan" + std::to_string(diff_gpu));
            if (clip_gpu >= 0)
                parts.push_back("te=Vulkan" + std::to_string(clip_gpu));
            if (vae_gpu >= 0)
                parts.push_back("vae=Vulkan" + std::to_string(vae_gpu));
        }
        if (flag(params, "keep_clip_on_cpu"))
            parts.push_back("te=cpu");
        if (flag(params, "keep_vae_on_cpu"))
            parts.push_back("vae=cpu");
        if (flag(params, "keep_control_net_on_cpu"))
            parts.push_back("controlnet=cpu");

        s.backend.clear();
        for (size_t i = 0; i < parts.size(); ++i) {
            if (i) s.backend += ",";
            s.backend += parts[i];
        }
        if (!s.backend.empty()) {
            p.backend = s.backend.c_str();
        }
    }

    // sd2R "second path": generate via the meta backend (on/off flag).
    // Default off => normal single-backend path. If meta is unavailable in ggmlR,
    // C++ falls back to the normal path (see get_or_build_diffusion_meta_backend).
    // The flag is read in new_sd_ctx, so it must be set before that call.
    sd2r::meta::set_meta_backend(flag(params, "meta_backend"));
}

void ImgGenStorage::clear() {
    prompt.clear();
    negative_prompt.clear();
    ref_image_args.clear();
    init_image_data.clear();
    mask_image_data.clear();
    control_image_data.clear();
    ref_image_data.clear();
    ref_images.clear();
    lora_paths.clear();
    loras.clear();
}

void sd2r_fill_img_gen_params(Rcpp::List params, sd_img_gen_params_t& p, ImgGenStorage& s) {
    sd_img_gen_params_init(&p);
    s.clear();

    set_str(params, "prompt", s.prompt, p.prompt);
    set_str(params, "negative_prompt", s.negative_prompt, p.negative_prompt);

    if (has(params, "width"))
        p.width = Rcpp::as<int>(params["width"]);
    if (has(params, "height"))
        p.height = Rcpp::as<int>(params["height"]);
    if (has(params, "clip_skip"))
        p.clip_skip = Rcpp::as<int>(params["clip_skip"]);
    if (has(params, "strength"))
        p.strength = Rcpp::as<float>(params["strength"]);
    if (has(params, "seed"))
        p.seed = Rcpp::as<int64_t>(params["seed"]);
    if (has(params, "batch_count"))
        p.batch_count = Rcpp::as<int>(params["batch_count"]);
    if (has(params, "control_strength"))
        p.control_strength = Rcpp::as<float>(params["control_strength"]);

    // Sample params
    if (has(params, "sample_method"))
        p.sample_params.sample_method = static_cast<sample_method_t>(Rcpp::as<int>(params["sample_method"]));
    if (has(params, "sample_steps"))
        p.sample_params.sample_steps = Rcpp::as<int>(params["sample_steps"]);
    if (has(params, "scheduler"))
        p.sample_params.scheduler = static_cast<scheduler_t>(Rcpp::as<int>(params["scheduler"]));
    if (has(params, "cfg_scale"))
        p.sample_params.guidance.txt_cfg = Rcpp::as<float>(params["cfg_scale"]);
    if (has(params, "eta"))
        p.sample_params.eta = Rcpp::as<float>(params["eta"]);
    if (has(params, "flow_shift"))
        p.sample_params.flow_shift = Rcpp::as<float>(params["flow_shift"]);

    // VAE tiling
    if (flag(params, "vae_tiling")) {
        p.vae_tiling_params.enabled = true;
        if (has(params, "vae_tile_size")) {
            // Image pixels, passed to the core as is (it converts to latent cells
            // with the model's VAE scale factor). Before 0.2.2 the R value was in
            // latent cells (default 64); such a value now means a tiny tile.
            int ts = Rcpp::as<int>(params["vae_tile_size"]);
            if (ts > 0 && ts < 128)
                Rcpp::warning("vae_tile_size = %d is in image pixels (since sd2R 0.2.2; "
                              "it used to be latent cells). Tiles this small are very slow; "
                              "the default is 512.", ts);
            p.vae_tiling_params.tile_size_w = ts;
            p.vae_tiling_params.tile_size_h = ts;
        }
        if (has(params, "vae_tile_overlap"))
            p.vae_tiling_params.target_overlap = Rcpp::as<float>(params["vae_tile_overlap"]);
        if (has(params, "vae_tile_rel_x"))
            p.vae_tiling_params.rel_size_w = Rcpp::as<float>(params["vae_tile_rel_x"]);
        if (has(params, "vae_tile_rel_y"))
            p.vae_tiling_params.rel_size_h = Rcpp::as<float>(params["vae_tile_rel_y"]);
    }
    // tiled_sampling / sample_tile_*: no C-side MultiDiffusion in the current
    // core (the old struct field was never read); tiling is driven from R.

    // Step caching (EasyCache / UCache / etc.)
    if (has(params, "cache_mode")) {
        p.cache.mode = static_cast<sd_cache_mode_t>(Rcpp::as<int>(params["cache_mode"]));
        if (has(params, "cache_threshold"))
            p.cache.reuse_threshold = Rcpp::as<float>(params["cache_threshold"]);
        if (has(params, "cache_start"))
            p.cache.start_percent = Rcpp::as<float>(params["cache_start"]);
        if (has(params, "cache_end"))
            p.cache.end_percent = Rcpp::as<float>(params["cache_end"]);
    }

    // Init image (img2img), mask (inpaint: 1-channel grayscale, white = generate,
    // black = keep; if absent the core skips the mask), control image.
    copy_image(params, "init_image", s.init_image_data, p.init_image);
    copy_image(params, "mask_image", s.mask_image_data, p.mask_image);
    copy_image(params, "control_image", s.control_image_data, p.control_image);

    // Reference images (multi-reference, TODO 9.7): list of image lists.
    if (has(params, "ref_images")) {
        Rcpp::List ref_list = Rcpp::as<Rcpp::List>(params["ref_images"]);
        s.ref_image_data.resize(ref_list.size());
        for (R_xlen_t i = 0; i < ref_list.size(); i++) {
            Rcpp::List one = Rcpp::List::create(Rcpp::Named("img") = ref_list[i]);
            sd_image_t img{};
            copy_image(one, "img", s.ref_image_data[i], img);
            s.ref_images.push_back(img);
        }
        if (!s.ref_images.empty()) {
            p.ref_images       = s.ref_images.data();
            p.ref_images_count = (int)s.ref_images.size();
        }
    }

    // LoRAs: list(path = chr, multiplier = num, is_high_noise = lgl), paths already
    // resolved and validated in R (.resolve_loras).
    if (has(params, "loras")) {
        Rcpp::List l                 = Rcpp::as<Rcpp::List>(params["loras"]);
        Rcpp::CharacterVector paths  = l["path"];
        Rcpp::NumericVector mult     = l["multiplier"];
        Rcpp::LogicalVector high     = l["is_high_noise"];
        R_xlen_t n                   = paths.size();
        if (mult.size() != n || high.size() != n)
            Rcpp::stop("loras: path, multiplier and is_high_noise must have the same length");
        s.lora_paths.reserve(n);
        for (R_xlen_t i = 0; i < n; i++)
            s.lora_paths.push_back(Rcpp::as<std::string>(paths[i]));
        for (R_xlen_t i = 0; i < n; i++) {
            sd_lora_t lora;
            lora.is_high_noise = high[i] == TRUE;
            lora.multiplier    = static_cast<float>(mult[i]);
            lora.path          = s.lora_paths[i].c_str();
            s.loras.push_back(lora);
        }
        if (!s.loras.empty()) {
            p.loras      = s.loras.data();
            p.lora_count = static_cast<uint32_t>(s.loras.size());
        }
    }

    // Reference image options -> ref_image_args ("key=value,..." parsed by the core).
    {
        std::vector<std::string> args;
        if (flag(params, "increase_ref_index"))
            args.push_back("ref_index_mode=increase");
        if (flag(params, "auto_resize_ref_image"))
            args.push_back("resize_before_vae=true");
        for (size_t i = 0; i < args.size(); ++i) {
            if (i) s.ref_image_args += ",";
            s.ref_image_args += args[i];
        }
        if (!s.ref_image_args.empty()) {
            p.ref_image_args = s.ref_image_args.c_str();
        }
    }
}
