# High-level R API wrapping stable-diffusion.cpp via Rcpp

#' Create a Stable Diffusion context
#'
#' Loads a model and creates a context for image generation.
#'
#' @param model_path Path to the model file (safetensors, gguf, or checkpoint)
#' @param vae_path Optional path to a separate VAE model
#' @param taesd_path Optional path to TAESD model for preview
#' @param clip_l_path Optional path to CLIP-L model
#' @param clip_g_path Optional path to CLIP-G model
#' @param t5xxl_path Optional path to T5-XXL model
#' @param llm_path Optional path to an LLM text encoder (Qwen3 / Mistral-Small).
#'   Required for models that use an LLM conditioner, e.g. FLUX.2 Klein (Qwen3),
#'   FLUX.2 (Mistral-Small), Z-Image and Qwen-Image. Loaded into the
#'   \code{text_encoders.llm} slot.
#' @param diffusion_model_path Optional path to separate diffusion model
#' @param control_net_path Optional path to ControlNet model
#' @param n_threads Number of CPU threads (0 = auto-detect)
#' @param wtype Weight type for quantization (see \code{SD_TYPE})
#' @param vae_decode_only If TRUE, only load VAE decoder (saves memory)
#' @param free_params_immediately Free model params after first computation.
#'   If TRUE, the context can only be used for a single generation — subsequent
#'   calls will crash. Set to TRUE only when you need to save memory and will
#'   not reuse the context. Default is FALSE.
#' @param keep_clip_on_cpu Keep CLIP model on CPU even when using GPU
#' @param keep_vae_on_cpu Keep VAE on CPU even when using GPU
#' @param offload_params_to_cpu Keep model weights in CPU RAM and stream them to
#'   the GPU on demand during compute (default FALSE). Lowers VRAM usage at the
#'   cost of CPU<->GPU transfers each step. Use when the model does not fit in
#'   GPU memory.
#' @param max_vram GiB budget for graph-cut segmented parameter offload
#'   (default 0 = disabled). A positive value caps GPU memory used by the compute
#'   graph; \code{-1} means "auto" (free VRAM minus ~1 GiB). Required for
#'   \code{stream_layers} to take effect.
#' @param stream_layers Enable residency + prefetch streaming of layers on top of
#'   \code{max_vram} (default FALSE). Has no effect unless \code{max_vram} is set
#'   (a non-zero budget); automatically disabled otherwise.
#' @param enable_mmap Memory-map model weights from disk instead of reading them
#'   into a malloc'd buffer (default FALSE). Lowers RAM footprint for large
#'   models (e.g. Flux); pages are loaded on demand by the OS and shared across
#'   processes. Ignored for zip-archived weights. May slow the first generation
#'   slightly as pages fault in.
#' @param vae_conv_direct Use direct Conv2d implementation in VAE (default TRUE).
#'   Faster on GPU; skips im2col and uses direct convolution kernels.
#' @param diffusion_conv_direct Use direct Conv2d in diffusion model (default FALSE).
#' @param diffusion_flash_attn Enable flash attention for diffusion model
#'   (default TRUE). Set to FALSE if you experience issues with specific
#'   GPU drivers or backends.
#' @param rng_type RNG type (see \code{RNG_TYPE})
#' @param prediction Prediction type override (see \code{PREDICTION}), NULL = auto
#' @param lora_apply_mode LoRA application mode (see \code{LORA_APPLY_MODE})
#' @param model_type Model architecture hint: \code{"sd1"}, \code{"sd2"},
#'   \code{"sdxl"}, \code{"flux"}, \code{"flux2"}, \code{"sd3"}, or
#'   \code{"auto"}. Used by
#'   \code{\link{sd_generate}} to determine native resolution and tile sizes.
#'   With \code{"auto"}, the type is detected from a sibling \code{config.json}
#'   then the filename (GGUF-metadata detection is a future hook); detection
#'   errors with a hint if it cannot decide. Default \code{"sd1"}.
#' @param vram_gb Override available VRAM in GB. When set, disables auto-detection
#'   and uses this value for strategy routing. Default \code{NULL} (auto-detect
#'   from Vulkan device).
#' @param device_layout GPU layout preset for multi-GPU systems. One of:
#'   \describe{
#'     \item{\code{"mono"}}{All models on one GPU (default).}
#'     \item{\code{"split_encoders"}}{Text encoders (CLIP/T5) on GPU 1,
#'       diffusion + VAE on GPU 0.}
#'     \item{\code{"split_vae"}}{Text encoders + VAE on GPU 1,
#'       diffusion on GPU 0. Maximizes VRAM for diffusion.}
#'     \item{\code{"encoders_cpu"}}{Text encoders on CPU,
#'       diffusion + VAE on GPU. Saves GPU memory at the cost of slower
#'       text encoding.}
#'   }
#'   Ignored when \code{diffusion_gpu}, \code{clip_gpu}, or \code{vae_gpu}
#'   are explicitly set (>= 0).
#' @param diffusion_gpu Vulkan GPU device index for the diffusion model.
#'   Default \code{-1} (use \code{SD_VK_DEVICE} env or device 0).
#'   Overrides \code{device_layout}.
#' @param clip_gpu Vulkan GPU device index for CLIP/T5 text encoders.
#'   Default \code{-1} (same device as diffusion).
#'   Overrides \code{device_layout}.
#' @param vae_gpu Vulkan GPU device index for VAE encoder/decoder.
#'   Default \code{-1} (same device as diffusion).
#'   Overrides \code{device_layout}.
#' @param meta_backend Logical flag to run the diffusion model through the ggml
#'   meta backend ("second path", multi-GPU tensor split across all available
#'   GPUs). Requires meta-backend support compiled in at install time (ggmlR
#'   >= 0.7.8 exporting \code{ggml_backend_meta_device}); if the build lacks it,
#'   a warning is emitted and the normal single-backend path is used. Default
#'   \code{FALSE} keeps existing behaviour unchanged. Distinct from
#'   \code{diffusion_gpu}/\code{vae_gpu} (per-component placement) and
#'   \code{sd_generate_multi_gpu()} (per-prompt batch parallelism).
#' @param tensor_type_rules Optional per-component weight type override, as a
#'   comma-separated string of \code{pattern=type} rules. Each pattern is a
#'   regex matched against tensor names; the first match wins. Use this to
#'   load specific model components at a different precision than \code{wtype}.
#'   Examples:
#'   \itemize{
#'     \item \code{"first_stage_model=f16"} — load VAE at F16
#'     \item \code{"first_stage_model=f16,model.diffusion_model=q8_0"} — VAE F16, UNet Q8_0
#'   }
#'   Type names match ggml type names (\code{"f16"}, \code{"f32"}, \code{"q8_0"}, etc.).
#' @param verbose If \code{TRUE}, print model loading progress and sampling
#'   steps. Default \code{FALSE}.
#' @return An external pointer to the SD context (class "sd_ctx") with
#'   attributes \code{model_type}, \code{vae_decode_only}, \code{vram_gb},
#'   \code{vram_total_gb}, and \code{vram_device}.
#' @export
#' @examples
#' \dontrun{
#' ctx <- sd_ctx("model.safetensors")
#' imgs <- sd_txt2img(ctx, "a cat sitting on a chair")
#' sd_save_image(imgs[[1]], "cat.png")
#' }
sd_ctx <- function(model_path = NULL,
                   vae_path = NULL,
                   taesd_path = NULL,
                   clip_l_path = NULL,
                   clip_g_path = NULL,
                   t5xxl_path = NULL,
                   llm_path = NULL,
                   diffusion_model_path = NULL,
                   control_net_path = NULL,
                   n_threads = 0L,
                   wtype = SD_TYPE$COUNT,
                   tensor_type_rules = NULL,
                   vae_decode_only = TRUE,
                   free_params_immediately = FALSE,
                   keep_clip_on_cpu = FALSE,
                   keep_vae_on_cpu = FALSE,
                   offload_params_to_cpu = FALSE,
                   max_vram = 0,
                   stream_layers = FALSE,
                   enable_mmap = FALSE,
                   vae_conv_direct = TRUE,
                   diffusion_conv_direct = FALSE,
                   diffusion_flash_attn = TRUE,
                   rng_type = RNG_TYPE$CUDA,
                   prediction = NULL,
                   lora_apply_mode = LORA_APPLY_MODE$AUTO,
                   model_type = "sd1",
                   vram_gb = NULL,
                   device_layout = "mono",
                   diffusion_gpu = -1L,
                   clip_gpu = -1L,
                   vae_gpu = -1L,
                   meta_backend = FALSE,
                   verbose = FALSE) {

  sd_set_verbose(verbose)

  if (!is.null(model_path) && !file.exists(model_path)) {
    stop("Model file not found: ", model_path, call. = FALSE)
  }
  if (is.null(model_path) && is.null(diffusion_model_path)) {
    stop("Either model_path or diffusion_model_path must be provided", call. = FALSE)
  }
  model_type <- match.arg(model_type,
                          c("sd1", "sd2", "sdxl", "flux", "flux2", "sd3",
                            "auto"))
  if (identical(model_type, "auto")) {
    detect_path <- if (!is.null(model_path)) model_path else diffusion_model_path
    model_type <- .resolve_model_type("auto", detect_path)
    if (verbose) message("Detected model_type: ", model_type)
  }

  params <- list(
    model_path = if (!is.null(model_path)) normalizePath(model_path) else "",
    n_threads = as.integer(n_threads),
    wtype = as.integer(wtype),
    vae_decode_only = vae_decode_only,
    free_params_immediately = free_params_immediately,
    keep_clip_on_cpu = keep_clip_on_cpu,
    keep_vae_on_cpu = keep_vae_on_cpu,
    offload_params_to_cpu = offload_params_to_cpu,
    max_vram = as.numeric(max_vram),
    stream_layers = stream_layers,
    enable_mmap = enable_mmap,
    vae_conv_direct = vae_conv_direct,
    diffusion_conv_direct = diffusion_conv_direct,
    diffusion_flash_attn = diffusion_flash_attn,
    rng_type = as.integer(rng_type),
    lora_apply_mode = as.integer(lora_apply_mode)
  )

  # Optional string params (paths — normalized)
  str_params <- list(
    vae_path = vae_path,
    taesd_path = taesd_path,
    clip_l_path = clip_l_path,
    clip_g_path = clip_g_path,
    t5xxl_path = t5xxl_path,
    llm_path = llm_path,
    diffusion_model_path = diffusion_model_path,
    control_net_path = control_net_path
  )
  for (nm in names(str_params)) {
    if (!is.null(str_params[[nm]])) {
      params[[nm]] <- normalizePath(str_params[[nm]], mustWork = TRUE)
    }
  }

  # tensor_type_rules: passed as-is (not a path)
  if (!is.null(tensor_type_rules)) {
    params$tensor_type_rules <- tensor_type_rules
  }

  if (!is.null(prediction)) {
    params$prediction <- as.integer(prediction)
  }

  # GPU device layout
  layout <- .resolve_device_layout(device_layout, diffusion_gpu, clip_gpu,
                                    vae_gpu, keep_clip_on_cpu, keep_vae_on_cpu)
  if (layout$diffusion >= 0L) params$diffusion_gpu_device <- layout$diffusion
  if (layout$clip >= 0L)      params$clip_gpu_device      <- layout$clip
  if (layout$vae >= 0L)       params$vae_gpu_device       <- layout$vae
  if (layout$clip_on_cpu)     params$keep_clip_on_cpu     <- TRUE
  if (layout$vae_on_cpu)      params$keep_vae_on_cpu      <- TRUE

  # "Second path": run diffusion through the ggml meta backend (on/off flag).
  # Default FALSE => normal single-backend path (full backward compatibility).
  # If meta is unavailable in the linked ggmlR, C++ falls back to the normal path.
  if (isTRUE(meta_backend)) {
    if (isTRUE(sd_meta_backend_available())) {
      params$meta_backend <- TRUE
      if (verbose) message("Meta backend ON (multi-GPU; falls back to single if unavailable)")
    } else {
      warning("meta_backend = TRUE but this sd2R build was compiled without ",
              "meta backend support (ggmlR lacks ggml_backend_meta_device; ",
              "need ggmlR >= 0.7.8). Using the normal single-backend path.",
              call. = FALSE)
    }
  }

  ctx <- sd_create_context(params)
  attr(ctx, "model_type") <- model_type
  attr(ctx, "vae_decode_only") <- vae_decode_only
  attr(ctx, "vram_gb") <- vram_gb

  # Cache total VRAM for auto-routing (one-time Vulkan query)
  device <- as.integer(Sys.getenv("SD_VK_DEVICE", "0"))
  attr(ctx, "vram_device") <- device
  attr(ctx, "vram_total_gb") <- tryCatch({
    mem <- ggmlR::ggml_vulkan_device_memory(device)
    mem$total / 1e9
  }, error = function(e) NULL)

  ctx
}

#' Release a stable diffusion context and free its VRAM
#'
#' Immediately destroys an \code{sd_ctx} object created by \code{\link{sd_ctx}},
#' freeing the GPU memory held by its model weights and compute buffers. Use this
#' before loading a different model so the two models do not pile up in VRAM.
#'
#' The context's external pointer also has a finalizer that frees it during R's
#' garbage collection, but that is non-deterministic and may not run promptly —
#' on a memory-constrained GPU, loading a second model before the first is
#' collected can exhaust VRAM and make the next Vulkan device init fail. Calling
#' \code{sd_destroy_context()} makes the release deterministic.
#'
#' After this call the \code{ctx} object is dead; do not pass it to
#' \code{\link{sd_generate}} or other functions. Calling it twice on the same
#' object, or on an already-finalized one, is a safe no-op.
#'
#' @param ctx An \code{sd_ctx} object from \code{\link{sd_ctx}}.
#' @return \code{NULL}, invisibly.
#' @seealso \code{\link{sd_ctx}}
#' @export
#' @examples
#' \dontrun{
#' ctx <- sd_ctx("flux1.safetensors", model_type = "flux")
#' img <- sd_generate(ctx, "a cat")
#' sd_destroy_context(ctx)              # free VRAM before the next model
#' ctx <- sd_ctx("flux2.safetensors", model_type = "flux2")
#' }
sd_destroy_context <- function(ctx) {
  invisible(.Call(`_sd2R_sd_destroy_context_impl`, ctx))
}

#' Generate images (unified entry point)
#'
#' Automatically selects the best generation strategy based on output resolution
#' and available VRAM (set via \code{vram_gb} in \code{\link{sd_ctx}}). For
#' txt2img, routes between direct generation, tiled sampling (MultiDiffusion),
#' or highres fix. For img2img (when \code{init_image} is provided), routes
#' between direct and tiled img2img.
#'
#' When \code{vram_gb} is not set on the context, defaults to direct generation
#' (equivalent to calling \code{\link{sd_txt2img}} or \code{\link{sd_img2img}}
#' directly).
#'
#' @param ctx SD context created by \code{\link{sd_ctx}}
#' @param prompt Text prompt describing desired image
#' @param negative_prompt Negative prompt (default "")
#' @param width Image width in pixels (default 512)
#' @param height Image height in pixels (default 512)
#' @param init_image Optional init image for img2img. If provided, runs img2img
#'   instead of txt2img. Requires \code{vae_decode_only = FALSE}.
#' @param strength Denoising strength for img2img (default 0.75). Ignored for
#'   txt2img.
#' @param sample_method Sampling method (see \code{SAMPLE_METHOD})
#' @param sample_steps Number of sampling steps (default 20)
#' @param cfg_scale Classifier-free guidance scale (default 7.0)
#' @param seed Random seed (-1 for random)
#' @param batch_count Number of images to generate (default 1)
#' @param scheduler Scheduler type (see \code{SCHEDULER})
#' @param clip_skip Number of CLIP layers to skip (-1 = auto)
#' @param eta Eta parameter for DDIM-like samplers
#' @param flow_shift Flow shift for flow-matching models (Flux, SD3). \code{NULL}
#'   (default) lets the model pick an architecture-specific value; set a numeric
#'   value to override. Ignored by non-flow models.
#' @param hr_strength Denoising strength for highres fix refinement pass
#'   (default 0.4). Only used when auto-routing selects highres fix.
#' @param vae_mode VAE processing mode: \code{"normal"}, \code{"tiled"}, or
#'   \code{"auto"} (VRAM-aware: queries free GPU memory and enables tiling
#'   only when estimated peak VAE usage exceeds available VRAM minus a 50 MB
#'   reserve). Default \code{"auto"}.
#' @param vae_tile_size Tile size for VAE tiling (default 64)
#' @param vae_tile_overlap Overlap for VAE tiling (default 0.25)
#' @param cache_mode Step caching mode: \code{"off"} (default), \code{"easy"}
#'   (EasyCache), or \code{"ucache"} (UCache).
#' @param cache_config Optional fine-tuned cache config from
#'   \code{\link{sd_cache_params}}.
#' @param params Optional baseline list from \code{\link{sd_default_params}}.
#'   Supplies defaults for any generation argument not passed explicitly;
#'   explicitly named arguments to \code{sd_generate()} always take precedence.
#'   \code{NULL} (default) keeps the built-in defaults.
#' @param preview If \code{TRUE}, write intermediate preview frames during
#'   generation to \code{preview_path}; poll with \code{\link{sd_read_preview}}.
#'   Default \code{FALSE} (zero cost). See \code{\link{sd_preview_start}}.
#' @param preview_path File path for the preview PPM. Defaults to a tempfile
#'   when \code{preview = TRUE}.
#' @param preview_mode Preview decode mode (see \code{PREVIEW}); default
#'   \code{"proj"}.
#' @param preview_interval Emit a preview every N steps (default 1).
#' @return List of SD images (or single image for highres fix path).
#' @export
#' @examples
#' \dontrun{
#' # Simple — auto-routes based on detected VRAM
#' ctx <- sd_ctx("model.safetensors", model_type = "sd1",
#'               vae_decode_only = FALSE)
#' imgs <- sd_generate(ctx, "a cat", width = 2048, height = 2048)
#'
#' # Manual override — force 4 GB VRAM limit
#' ctx4 <- sd_ctx("model.safetensors", model_type = "sd1",
#'                vram_gb = 4, vae_decode_only = FALSE)
#' imgs <- sd_generate(ctx4, "a cat", width = 2048, height = 2048)
#' }
sd_generate <- function(ctx,
                        prompt,
                        negative_prompt = "",
                        width = 512L,
                        height = 512L,
                        init_image = NULL,
                        strength = 0.75,
                        sample_method = SAMPLE_METHOD$EULER,
                        sample_steps = 20L,
                        cfg_scale = 7.0,
                        seed = 42L,
                        batch_count = 1L,
                        scheduler = SCHEDULER$DISCRETE,
                        clip_skip = -1L,
                        eta = 0.0,
                        flow_shift = NULL,
                        hr_strength = 0.4,
                        vae_mode = "auto",
                        vae_tile_size = 64L,
                        vae_tile_overlap = 0.25,
                        cache_mode = c("off", "easy", "ucache"),
                        cache_config = NULL,
                        params = NULL,
                        preview = FALSE,
                        preview_path = NULL,
                        preview_mode = PREVIEW$PROJ,
                        preview_interval = 1L) {
  # Merge a params baseline (sd_default_params()) under the explicit args.
  # An argument counts as "explicit" only when the caller named it; for those,
  # the passed value wins. Everything else is taken from `params`, falling back
  # to the built-in defaults. This keeps every existing call site unchanged.
  if (!is.null(params)) {
    called   <- names(match.call())[-(1:2)]  # drop fn name + ctx (prompt may stay)
    defs     <- sd_default_params()
    pv <- function(nm, val) {
      if (nm %in% called) return(val)
      if (nm %in% names(params)) return(params[[nm]])
      defs[[nm]]
    }
    negative_prompt  <- pv("negative_prompt",  negative_prompt)
    width            <- pv("width",            width)
    height           <- pv("height",           height)
    strength         <- pv("strength",         strength)
    sample_method    <- pv("sample_method",    sample_method)
    sample_steps     <- pv("sample_steps",     sample_steps)
    cfg_scale        <- pv("cfg_scale",        cfg_scale)
    seed             <- pv("seed",             seed)
    batch_count      <- pv("batch_count",      batch_count)
    scheduler        <- pv("scheduler",        scheduler)
    clip_skip        <- pv("clip_skip",        clip_skip)
    eta              <- pv("eta",              eta)
    flow_shift       <- pv("flow_shift",       flow_shift)
    hr_strength      <- pv("hr_strength",      hr_strength)
    vae_mode         <- pv("vae_mode",         vae_mode)
    vae_tile_size    <- pv("vae_tile_size",    vae_tile_size)
    vae_tile_overlap <- pv("vae_tile_overlap", vae_tile_overlap)
    cache_mode       <- pv("cache_mode",       cache_mode)
    cache_config     <- pv("cache_config",     cache_config)
  }

  # Resolve string names to integer enum values
  if (is.character(sample_method)) {
    sm <- SAMPLE_METHOD[[sample_method]]
    if (is.null(sm)) stop("Unknown sample_method: ", sample_method, call. = FALSE)
    sample_method <- sm
  }
  if (is.character(scheduler)) {
    sc <- SCHEDULER[[scheduler]]
    if (is.null(sc)) stop("Unknown scheduler: ", scheduler, call. = FALSE)
    scheduler <- sc
  }

  # Live preview: zero cost unless preview = TRUE. Installs the file-based
  # preview callback for the duration of this call; poll preview_path with
  # sd_read_preview() from another process/handler. The path is stored on the
  # returned object's attribute so callers know where to look.
  if (isTRUE(preview)) {
    if (is.null(preview_path)) {
      preview_path <- tempfile("sd_preview_", fileext = ".ppm")
    }
    sd_preview_start(preview_path, mode = preview_mode,
                     interval = preview_interval)
    on.exit(sd_preview_stop(), add = TRUE)
  }

  # img2img: default to init_image dimensions when width/height not specified
  if (!is.null(init_image)) {
    if (missing(width))  width  <- init_image$width
    if (missing(height)) height <- init_image$height
  }
  width <- as.integer(width)
  height <- as.integer(height)
  model_type <- attr(ctx, "model_type") %||% "sd1"

  # Flux / Flux.2 use guidance-distilled models; cfg_scale should default to 1.0
  if (model_type %in% c("flux", "flux2") && cfg_scale == 7.0) {
    cfg_scale <- 1.0
  }
  is_img2img <- !is.null(init_image)

  # Determine strategy
  vae_decode_only <- attr(ctx, "vae_decode_only") %||% TRUE
  strategy <- .select_strategy(width, height, ctx, model_type, is_img2img,
                               vae_decode_only)

  if (is_img2img) {
    if (strategy == "tiled") {
      sd_img2img_tiled(ctx, prompt,
                       init_image = init_image,
                       negative_prompt = negative_prompt,
                       width = width, height = height,
                       sample_method = sample_method,
                       sample_steps = sample_steps,
                       cfg_scale = cfg_scale, seed = seed,
                       batch_count = batch_count,
                       scheduler = scheduler, clip_skip = clip_skip,
                       strength = strength, eta = eta,
                       flow_shift = flow_shift,
                       vae_mode = vae_mode,
                       vae_tile_size = vae_tile_size,
                       vae_tile_overlap = vae_tile_overlap,
                       cache_mode = cache_mode,
                       cache_config = cache_config)
    } else {
      sd_img2img(ctx, prompt,
                 init_image = init_image,
                 negative_prompt = negative_prompt,
                 width = width, height = height,
                 sample_method = sample_method,
                 sample_steps = sample_steps,
                 cfg_scale = cfg_scale, seed = seed,
                 batch_count = batch_count,
                 scheduler = scheduler, clip_skip = clip_skip,
                 strength = strength, eta = eta,
                 flow_shift = flow_shift,
                 vae_mode = vae_mode,
                 vae_tile_size = vae_tile_size,
                 vae_tile_overlap = vae_tile_overlap,
                 cache_mode = cache_mode,
                 cache_config = cache_config)
    }
  } else {
    if (strategy == "highres_fix") {
      img <- sd_highres_fix(ctx, prompt,
                            negative_prompt = negative_prompt,
                            width = width, height = height,
                            sample_method = sample_method,
                            sample_steps = sample_steps,
                            cfg_scale = cfg_scale, seed = seed,
                            scheduler = scheduler, clip_skip = clip_skip,
                            eta = eta, flow_shift = flow_shift,
                            hr_strength = hr_strength,
                            vae_mode = vae_mode,
                            vae_tile_size = vae_tile_size,
                            vae_tile_overlap = vae_tile_overlap,
                            cache_mode = cache_mode,
                            cache_config = cache_config)
      list(img)
    } else if (strategy == "tiled") {
      sd_txt2img_tiled(ctx, prompt,
                       negative_prompt = negative_prompt,
                       width = width, height = height,
                       sample_method = sample_method,
                       sample_steps = sample_steps,
                       cfg_scale = cfg_scale, seed = seed,
                       batch_count = batch_count,
                       scheduler = scheduler, clip_skip = clip_skip,
                       eta = eta, flow_shift = flow_shift,
                       vae_mode = vae_mode,
                       vae_tile_size = vae_tile_size,
                       vae_tile_overlap = vae_tile_overlap,
                       cache_mode = cache_mode,
                       cache_config = cache_config)
    } else {
      sd_txt2img(ctx, prompt,
                 negative_prompt = negative_prompt,
                 width = width, height = height,
                 sample_method = sample_method,
                 sample_steps = sample_steps,
                 cfg_scale = cfg_scale, seed = seed,
                 batch_count = batch_count,
                 scheduler = scheduler, clip_skip = clip_skip,
                 eta = eta, flow_shift = flow_shift,
                 vae_mode = vae_mode,
                 vae_tile_size = vae_tile_size,
                 vae_tile_overlap = vae_tile_overlap,
                 cache_mode = cache_mode,
                 cache_config = cache_config)
    }
  }
}

#' Select generation strategy based on resolution and VRAM
#'
#' @param width Target width
#' @param height Target height
#' @param ctx SD context with VRAM attributes
#' @param model_type Model type string
#' @param is_img2img Whether this is an img2img call
#' @param vae_decode_only Whether context has VAE encoder (FALSE = has encoder)
#' @return One of "direct", "tiled", "highres_fix"
#' @keywords internal
.select_strategy <- function(width, height, ctx, model_type, is_img2img,
                             vae_decode_only = TRUE) {
  # Manual vram_gb takes priority
  vram_gb <- attr(ctx, "vram_gb")

  if (is.null(vram_gb)) {
    # Auto-detect from Vulkan device
    device <- attr(ctx, "vram_device") %||% 0L
    vram_gb <- tryCatch({
      free <- ggmlR::ggml_vulkan_device_memory(device)$free / 1e9
      total <- attr(ctx, "vram_total_gb") %||% free
      # Protect against UMA/shared memory: driver reserves ~10%
      min(free, total * 0.9)
    }, error = function(e) {
      warning("VRAM autodetect failed, assuming unlimited: ",
              conditionMessage(e))
      Inf
    })
  }

  native_px <- .native_tile_size(model_type)
  pixels <- as.numeric(width) * as.numeric(height)
  native_pixels <- as.numeric(native_px) * as.numeric(native_px)

  # Estimated VRAM: ~4 GB per 262144 pixels (512x512) with +10% safety margin
  # 512x512 -> 4.4 GB, 1024x1024 -> 17.6 GB, 2048x2048 -> 70.4 GB
  vram_needed <- pixels / 262144 * 4.0 * 1.1

  if (vram_needed <= vram_gb) return("direct")

  if (is_img2img) {
    if (pixels > native_pixels) return("tiled")
    return("direct")
  }

  # txt2img: prefer highres fix over tiled (global coherence via base gen + upscale),
  # but only when VAE encoder is available
  if (!vae_decode_only && pixels > native_pixels) {
    return("highres_fix")
  }

  # fallback: tiled sampling (no global coherence, but works without encoder)
  "tiled"
}

# Internal: apply cache_mode / cache_config to params list
.apply_cache_params <- function(params, cache_mode, cache_config) {
  if (!is.null(cache_config)) {
    # Custom config overrides everything
    params$cache_mode <- as.integer(cache_config$cache_mode)
    params$cache_threshold <- as.numeric(cache_config$cache_threshold)
    params$cache_start <- as.numeric(cache_config$cache_start)
    params$cache_end <- as.numeric(cache_config$cache_end)
  } else {
    mode <- match.arg(cache_mode, c("off", "easy", "ucache"))
    if (mode != "off") {
      params$cache_mode <- switch(mode,
        easy   = SD_CACHE_MODE$EASYCACHE,
        ucache = SD_CACHE_MODE$UCACHE
      )
      # Use C++ defaults for threshold/start/end
      params$cache_threshold <- 0.3
      params$cache_start <- 0.3
      params$cache_end <- 0.8
    }
  }
  params
}

#' Build an executable step plan for sd_generate (async orchestration)
#'
#' Mirrors the routing logic of \code{\link{sd_generate}} (cfg auto-1.0 for
#' guidance-distilled Flux/Flux.2, strategy selection, VAE tiling resolution)
#' but instead of running the pipeline, returns a list of \emph{steps}. Each
#' step is one of:
#' \itemize{
#'   \item \code{type = "gen"}  — a single \code{sd_generate_async()} call.
#'     Carries a ready-to-use \code{params} list, \code{width}/\code{height},
#'     a \code{label}, and \code{uses_init} (whether it consumes the previous
#'     step's image as \code{init_image}).
#'   \item \code{type = "upscale"} — a synchronous R-side resize/ESRGAN step
#'     run between two gen steps. Carries \code{width}/\code{height},
#'     \code{upscaler}, \code{upscale_factor}.
#' }
#' The final step (the image returned to the caller) has \code{final = TRUE}.
#'
#' This lets the Shiny GUI drive the multi-step highres-fix pipeline through the
#' single-shot async engine: run a gen step, poll it, feed its result into the
#' next step, all without blocking the R session.
#'
#' @inheritParams sd_generate
#' @return List of step descriptors (see above).
#' @keywords internal
.sd_generate_plan <- function(ctx, prompt,
                              negative_prompt = "",
                              width = 512L, height = 512L,
                              init_image = NULL,
                              strength = 0.75,
                              sample_method = SAMPLE_METHOD$EULER,
                              sample_steps = 20L,
                              cfg_scale = 7.0,
                              seed = 42L,
                              batch_count = 1L,
                              scheduler = SCHEDULER$DISCRETE,
                              clip_skip = -1L,
                              eta = 0.0,
                              flow_shift = NULL,
                              hr_strength = 0.4,
                              hr_steps = NULL,
                              upscaler = NULL,
                              upscale_factor = 4L,
                              vae_mode = "auto",
                              vae_auto_threshold = 1048576L,
                              vae_tile_size = 64L,
                              vae_tile_overlap = 0.25,
                              cache_mode = c("off", "easy", "ucache"),
                              cache_config = NULL) {
  if (is.character(sample_method)) sample_method <- SAMPLE_METHOD[[sample_method]]
  if (is.character(scheduler))     scheduler     <- SCHEDULER[[scheduler]]
  width  <- as.integer(width)
  height <- as.integer(height)

  model_type <- attr(ctx, "model_type") %||% "sd1"
  # Flux / Flux.2 are guidance-distilled: cfg_scale must be ~1.0, not 7.0.
  if (model_type %in% c("flux", "flux2") && cfg_scale == 7.0) cfg_scale <- 1.0

  is_img2img      <- !is.null(init_image)
  vae_decode_only <- attr(ctx, "vae_decode_only") %||% TRUE
  strategy <- .select_strategy(width, height, ctx, model_type, is_img2img,
                               vae_decode_only)

  # Helper: assemble a single-shot params list for sd_generate_async, replicating
  # the bodies of sd_txt2img / sd_img2img / sd_txt2img_tiled.
  mk_gen <- function(w, h, steps, strength_v, tiled_sampling = FALSE,
                     sample_tile_size = NULL, sample_tile_overlap = 0.25) {
    vae_tiling_resolved <- .resolve_vae_tiling(
      vae_mode = vae_mode, vae_tiling = NULL, width = w, height = h,
      vae_auto_threshold = vae_auto_threshold, ctx = ctx, batch = batch_count)
    p <- list(
      prompt = prompt, negative_prompt = negative_prompt,
      width = as.integer(w), height = as.integer(h),
      sample_method = as.integer(sample_method),
      sample_steps = as.integer(steps),
      cfg_scale = as.numeric(cfg_scale), seed = as.integer(seed),
      batch_count = as.integer(batch_count), scheduler = as.integer(scheduler),
      clip_skip = as.integer(clip_skip), strength = as.numeric(strength_v),
      eta = as.numeric(eta), control_strength = 0.9,
      vae_tiling = vae_tiling_resolved,
      vae_tile_size = as.integer(vae_tile_size),
      vae_tile_overlap = as.numeric(vae_tile_overlap))
    if (tiled_sampling) {
      if (is.null(sample_tile_size))
        sample_tile_size <- .native_latent_tile_size(model_type)
      p$tiled_sampling <- TRUE
      p$sample_tile_size <- as.integer(sample_tile_size)
      p$sample_tile_overlap <- as.numeric(sample_tile_overlap)
    }
    if (!is.null(flow_shift)) p$flow_shift <- as.numeric(flow_shift)
    .apply_cache_params(p, cache_mode, cache_config)
  }

  gen_step <- function(w, h, steps, strength_v, tiled_sampling = FALSE,
                       uses_init = FALSE, label = "Generating", final = FALSE) {
    list(type = "gen", label = label, width = as.integer(w),
         height = as.integer(h), uses_init = uses_init, final = final,
         params = mk_gen(w, h, steps, strength_v, tiled_sampling = tiled_sampling))
  }

  # --- img2img (single gen step; tiled vs direct only changes sample tiling) ---
  if (is_img2img) {
    return(list(gen_step(width, height, sample_steps, strength,
                         tiled_sampling = (strategy == "tiled"),
                         uses_init = TRUE, label = "img2img", final = TRUE)))
  }

  # --- txt2img direct / tiled: a single gen step ---
  if (strategy != "highres_fix") {
    return(list(gen_step(width, height, sample_steps, 0.0,
                         tiled_sampling = (strategy == "tiled"),
                         label = "Generating", final = TRUE)))
  }

  # --- highres_fix: base gen -> upscale (R-side) -> img2img refine ---
  if (is.null(hr_steps)) hr_steps <- sample_steps
  native_px <- .native_tile_size(model_type)
  aspect <- width / height
  if (aspect >= 1) {
    base_w <- native_px
    base_h <- as.integer(round(native_px / aspect / 8) * 8)
  } else {
    base_h <- native_px
    base_w <- as.integer(round(native_px * aspect / 8) * 8)
  }
  base_w <- max(base_w, 64L); base_h <- max(base_h, 64L)

  # DiT denoisers (Flux/Flux.2/SD3) can't use tiled sampling for the refine pass.
  refine_tiled <- !(model_type %in% c("flux", "flux2", "sd3"))

  list(
    gen_step(base_w, base_h, sample_steps, 0.0,
             label = "Highres: base", final = FALSE),
    list(type = "upscale", label = "Highres: upscale",
         width = width, height = height,
         upscaler = upscaler, upscale_factor = as.integer(upscale_factor)),
    gen_step(width, height, hr_steps, hr_strength,
             tiled_sampling = refine_tiled, uses_init = TRUE,
             label = "Highres: refine", final = TRUE)
  )
}

#' Generate images from text prompt
#'
#' @param ctx SD context created by \code{\link{sd_ctx}}
#' @param prompt Text prompt describing desired image
#' @param negative_prompt Negative prompt (default "")
#' @param width Image width in pixels (default 512)
#' @param height Image height in pixels (default 512)
#' @param sample_method Sampling method (see \code{SAMPLE_METHOD})
#' @param sample_steps Number of sampling steps (default 20)
#' @param cfg_scale Classifier-free guidance scale (default 7.0)
#' @param seed Random seed (-1 for random)
#' @param batch_count Number of images to generate (default 1)
#' @param scheduler Scheduler type (see \code{SCHEDULER})
#' @param clip_skip Number of CLIP layers to skip (-1 = auto)
#' @param eta Eta parameter for DDIM-like samplers
#' @param flow_shift Flow shift for flow-matching models (Flux, SD3). \code{NULL}
#'   (default) lets the model pick an architecture-specific value; set a numeric
#'   value to override. Ignored by non-flow models.
#' @param control_image Optional control image for ControlNet (sd_image format)
#' @param control_strength ControlNet strength (default 0.9)
#' @param vae_mode VAE processing mode: \code{"normal"} (no tiling),
#'   \code{"tiled"} (always tile), or \code{"auto"} (VRAM-aware: queries free
#'   GPU memory via Vulkan and compares against estimated peak VAE usage;
#'   tiles only when VRAM is insufficient). Default \code{"auto"}.
#' @param vae_auto_threshold Pixel area fallback threshold for
#'   \code{vae_mode = "auto"} when VRAM query is unavailable (no Vulkan, CPU
#'   backend, etc.). Tiling activates when \code{width * height} exceeds this
#'   value. Default \code{1048576L} (1024x1024 pixels).
#' @param vae_tile_size Tile size in latent pixels for tiled VAE (default 64).
#'   Ignored when \code{vae_tile_rel_x}/\code{vae_tile_rel_y} are set.
#' @param vae_tile_overlap Overlap ratio between tiles, 0.0-0.5 (default 0.25)
#' @param vae_tile_rel_x Relative tile width as fraction of latent width (0-1)
#'   or number of tiles (>1). NULL = use \code{vae_tile_size}. Takes priority
#'   over \code{vae_tile_size}.
#' @param vae_tile_rel_y Relative tile height as fraction of latent height (0-1)
#'   or number of tiles (>1). NULL = use \code{vae_tile_size}. Takes priority
#'   over \code{vae_tile_size}.
#' @param vae_tiling \strong{Deprecated.} Use \code{vae_mode} instead.
#'   If \code{TRUE}, equivalent to \code{vae_mode = "tiled"}.
#' @param cache_mode Step caching mode: \code{"off"} (default), \code{"easy"}
#'   (EasyCache — skips redundant denoising steps), or \code{"ucache"} (UCache).
#'   Can speed up sampling 20-40\% with minor quality impact.
#' @param cache_config Optional fine-tuned cache config from
#'   \code{\link{sd_cache_params}}. Overrides \code{cache_mode} when provided.
#' @return List of SD images. Each image is a list with
#'   width, height, channel, and data (raw vector of RGB pixels).
#'   Use \code{\link{sd_save_image}} to save or \code{\link{sd_image_to_array}} to convert.
#' @export
sd_txt2img <- function(ctx,
                       prompt,
                       negative_prompt = "",
                       width = 512L,
                       height = 512L,
                       sample_method = SAMPLE_METHOD$EULER,
                       sample_steps = 20L,
                       cfg_scale = 7.0,
                       seed = 42L,
                       batch_count = 1L,
                       scheduler = SCHEDULER$DISCRETE,
                       clip_skip = -1L,
                       eta = 0.0,
                       flow_shift = NULL,
                       control_image = NULL,
                       control_strength = 0.9,
                       vae_mode = "auto",
                       vae_auto_threshold = 1048576L,
                       vae_tile_size = 64L,
                       vae_tile_overlap = 0.25,
                       vae_tile_rel_x = NULL,
                       vae_tile_rel_y = NULL,
                       vae_tiling = NULL,
                       cache_mode = c("off", "easy", "ucache"),
                       cache_config = NULL) {
  # Resolve string names to integer enum values
  if (is.character(sample_method)) {
    sm <- SAMPLE_METHOD[[sample_method]]
    if (is.null(sm)) stop("Unknown sample_method: ", sample_method, call. = FALSE)
    sample_method <- sm
  }
  if (is.character(scheduler)) {
    sc <- SCHEDULER[[scheduler]]
    if (is.null(sc)) stop("Unknown scheduler: ", scheduler, call. = FALSE)
    scheduler <- sc
  }

  vae_tiling_resolved <- .resolve_vae_tiling(
    vae_mode = vae_mode,
    vae_tiling = vae_tiling,
    width = width,
    height = height,
    vae_auto_threshold = vae_auto_threshold,
    ctx = ctx,
    batch = batch_count
  )

  params <- list(
    prompt = prompt,
    negative_prompt = negative_prompt,
    width = as.integer(width),
    height = as.integer(height),
    sample_method = as.integer(sample_method),
    sample_steps = as.integer(sample_steps),
    cfg_scale = as.numeric(cfg_scale),
    seed = as.integer(seed),
    batch_count = as.integer(batch_count),
    scheduler = as.integer(scheduler),
    clip_skip = as.integer(clip_skip),
    strength = 0.0,
    eta = as.numeric(eta),
    control_strength = as.numeric(control_strength),
    vae_tiling = vae_tiling_resolved,
    vae_tile_size = as.integer(vae_tile_size),
    vae_tile_overlap = as.numeric(vae_tile_overlap)
  )
  if (!is.null(vae_tile_rel_x)) {
    params$vae_tile_rel_x <- as.numeric(vae_tile_rel_x)
  }
  if (!is.null(vae_tile_rel_y)) {
    params$vae_tile_rel_y <- as.numeric(vae_tile_rel_y)
  }
  if (!is.null(flow_shift)) {
    params$flow_shift <- as.numeric(flow_shift)
  }
  if (!is.null(control_image)) {
    params$control_image <- control_image
  }
  params <- .apply_cache_params(params, cache_mode, cache_config)

  sd_generate_image(ctx, params)
}

#' Generate images with img2img
#'
#' @inheritParams sd_txt2img
#' @param init_image Init image in sd_image format. Use \code{\link{sd_load_image}}
#'   to load from file.
#' @param strength Denoising strength (0.0 = no change, 1.0 = full denoise, default 0.75)
#' @param mask Optional inpainting mask. A PNG file path, a numeric matrix
#'   \code{[H, W]} (values in 0..1 or 0..255), or a 1-channel SD image list.
#'   White (255) = regenerate that region, black (0) = keep the original. Must
#'   match the init image dimensions. When \code{NULL} (default) the whole image
#'   is denoised (plain img2img).
#' @return List of SD images
#' @export
sd_img2img <- function(ctx,
                       prompt,
                       init_image,
                       negative_prompt = "",
                       mask = NULL,
                       width = NULL,
                       height = NULL,
                       sample_method = SAMPLE_METHOD$EULER,
                       sample_steps = 20L,
                       cfg_scale = 7.0,
                       seed = 42L,
                       batch_count = 1L,
                       scheduler = SCHEDULER$DISCRETE,
                       clip_skip = -1L,
                       strength = 0.75,
                       eta = 0.0,
                       flow_shift = NULL,
                       vae_mode = "auto",
                       vae_auto_threshold = 1048576L,
                       vae_tile_size = 64L,
                       vae_tile_overlap = 0.25,
                       vae_tile_rel_x = NULL,
                       vae_tile_rel_y = NULL,
                       vae_tiling = NULL,
                       cache_mode = c("off", "easy", "ucache"),
                       cache_config = NULL) {
  # FIX: sd_ctx() defaults to vae_decode_only=TRUE, but img2img needs the VAE
  # encoder (encode_first_stage). Without this check, the C++ code hits
  # GGML_ASSERT(!decode_only || decode_graph) in vae.hpp:719.
  if (isTRUE(attr(ctx, "vae_decode_only"))) {
    stop("img2img requires VAE encoder. Recreate context with vae_decode_only = FALSE.",
         call. = FALSE)
  }
  # Resolve string names to integer enum values
  if (is.character(sample_method)) {
    sm <- SAMPLE_METHOD[[sample_method]]
    if (is.null(sm)) stop("Unknown sample_method: ", sample_method, call. = FALSE)
    sample_method <- sm
  }
  if (is.character(scheduler)) {
    sc <- SCHEDULER[[scheduler]]
    if (is.null(sc)) stop("Unknown scheduler: ", scheduler, call. = FALSE)
    scheduler <- sc
  }

  if (is.null(width)) width <- init_image$width
  if (is.null(height)) height <- init_image$height

  mask_image <- NULL
  if (!is.null(mask)) {
    mask_image <- .sd_to_mask_image(mask)
    if (mask_image$width != init_image$width ||
        mask_image$height != init_image$height) {
      stop(sprintf(
        "mask size (%dx%d) must match init_image size (%dx%d).",
        mask_image$width, mask_image$height,
        init_image$width, init_image$height), call. = FALSE)
    }
  }

  vae_tiling_resolved <- .resolve_vae_tiling(
    vae_mode = vae_mode,
    vae_tiling = vae_tiling,
    width = width,
    height = height,
    vae_auto_threshold = vae_auto_threshold,
    ctx = ctx,
    batch = batch_count
  )

  params <- list(
    prompt = prompt,
    negative_prompt = negative_prompt,
    init_image = init_image,
    width = as.integer(width),
    height = as.integer(height),
    sample_method = as.integer(sample_method),
    sample_steps = as.integer(sample_steps),
    cfg_scale = as.numeric(cfg_scale),
    seed = as.integer(seed),
    batch_count = as.integer(batch_count),
    scheduler = as.integer(scheduler),
    clip_skip = as.integer(clip_skip),
    strength = as.numeric(strength),
    eta = as.numeric(eta),
    mask_image = mask_image,
    vae_tiling = vae_tiling_resolved,
    vae_tile_size = as.integer(vae_tile_size),
    vae_tile_overlap = as.numeric(vae_tile_overlap)
  )
  if (!is.null(vae_tile_rel_x)) {
    params$vae_tile_rel_x <- as.numeric(vae_tile_rel_x)
  }
  if (!is.null(vae_tile_rel_y)) {
    params$vae_tile_rel_y <- as.numeric(vae_tile_rel_y)
  }
  if (!is.null(flow_shift)) {
    params$flow_shift <- as.numeric(flow_shift)
  }
  params <- .apply_cache_params(params, cache_mode, cache_config)

  sd_generate_image(ctx, params)
}

#' Tiled diffusion sampling (MultiDiffusion)
#'
#' Generates images at any resolution using tiled sampling: at each denoising
#' step the latent is split into overlapping tiles, each tile is denoised
#' independently by the UNet, and results are merged with Gaussian weighting.
#' VRAM usage is bounded by tile size, not output resolution.
#'
#' Requires tiled VAE (enabled automatically via \code{vae_mode = "auto"}).
#'
#' @inheritParams sd_txt2img
#' @param width Target image width in pixels (can exceed model native resolution)
#' @param height Target image height in pixels
#' @param sample_tile_size Tile size in latent pixels (default \code{NULL} =
#'   auto from \code{model_type}: 64 for SD1/SD2, 128 for SDXL/Flux/SD3).
#'   One latent pixel = \code{vae_scale_factor} image pixels (typically 8).
#' @param sample_tile_overlap Overlap between tiles as fraction of tile size,
#'   0.0-0.5 (default 0.25).
#' @return List of SD images
#' @export
#' @examples
#' \dontrun{
#' ctx <- sd_ctx("sd15.safetensors", model_type = "sd1")
#' imgs <- sd_txt2img_tiled(ctx, "a vast mountain landscape",
#'                          width = 2048, height = 1024)
#' sd_save_image(imgs[[1]], "landscape.png")
#' }
sd_txt2img_tiled <- function(ctx,
                              prompt,
                              negative_prompt = "",
                              width = 2048L,
                              height = 2048L,
                              sample_tile_size = NULL,
                              sample_tile_overlap = 0.25,
                              sample_method = SAMPLE_METHOD$EULER,
                              sample_steps = 20L,
                              cfg_scale = 7.0,
                              seed = 42L,
                              batch_count = 1L,
                              scheduler = SCHEDULER$DISCRETE,
                              clip_skip = -1L,
                              eta = 0.0,
                              flow_shift = NULL,
                              vae_mode = "auto",
                              vae_auto_threshold = 1048576L,
                              vae_tile_size = 64L,
                              vae_tile_overlap = 0.25,
                              vae_tile_rel_x = NULL,
                              vae_tile_rel_y = NULL,
                              cache_mode = c("off", "easy", "ucache"),
                              cache_config = NULL) {
  # Auto-detect sample tile size from model type
  if (is.null(sample_tile_size)) {
    model_type <- attr(ctx, "model_type") %||% "sd1"
    sample_tile_size <- .native_latent_tile_size(model_type)
  }
  sample_tile_size <- as.integer(sample_tile_size)
  stopifnot(sample_tile_size >= 8L)

  vae_tiling_resolved <- .resolve_vae_tiling(
    vae_mode = vae_mode,
    vae_tiling = NULL,
    width = width,
    height = height,
    vae_auto_threshold = vae_auto_threshold,
    ctx = ctx,
    batch = batch_count
  )

  params <- list(
    prompt = prompt,
    negative_prompt = negative_prompt,
    width = as.integer(width),
    height = as.integer(height),
    sample_method = as.integer(sample_method),
    sample_steps = as.integer(sample_steps),
    cfg_scale = as.numeric(cfg_scale),
    seed = as.integer(seed),
    batch_count = as.integer(batch_count),
    scheduler = as.integer(scheduler),
    clip_skip = as.integer(clip_skip),
    strength = 0.0,
    eta = as.numeric(eta),
    control_strength = 0.9,
    vae_tiling = vae_tiling_resolved,
    vae_tile_size = as.integer(vae_tile_size),
    vae_tile_overlap = as.numeric(vae_tile_overlap),
    tiled_sampling = TRUE,
    sample_tile_size = sample_tile_size,
    sample_tile_overlap = as.numeric(sample_tile_overlap)
  )
  if (!is.null(vae_tile_rel_x)) {
    params$vae_tile_rel_x <- as.numeric(vae_tile_rel_x)
  }
  if (!is.null(vae_tile_rel_y)) {
    params$vae_tile_rel_y <- as.numeric(vae_tile_rel_y)
  }
  if (!is.null(flow_shift)) {
    params$flow_shift <- as.numeric(flow_shift)
  }
  params <- .apply_cache_params(params, cache_mode, cache_config)

  sd_generate_image(ctx, params)
}

#' Tiled img2img (MultiDiffusion with init image)
#'
#' Runs img2img with tiled sampling: at each denoising step the latent is
#' split into overlapping tiles, each denoised independently, then merged.
#' The init image provides global composition; tiles add detail.
#'
#' @inheritParams sd_img2img
#' @param sample_tile_size Tile size in latent pixels (default auto from model)
#' @param sample_tile_overlap Overlap fraction 0.0-0.5 (default 0.25)
#' @return List of SD images
#' @keywords internal
sd_img2img_tiled <- function(ctx,
                              prompt,
                              init_image,
                              negative_prompt = "",
                              width = NULL,
                              height = NULL,
                              sample_tile_size = NULL,
                              sample_tile_overlap = 0.25,
                              sample_method = SAMPLE_METHOD$EULER,
                              sample_steps = 20L,
                              cfg_scale = 7.0,
                              seed = 42L,
                              batch_count = 1L,
                              scheduler = SCHEDULER$DISCRETE,
                              clip_skip = -1L,
                              strength = 0.5,
                              eta = 0.0,
                              flow_shift = NULL,
                              vae_mode = "auto",
                              vae_auto_threshold = 1048576L,
                              vae_tile_size = 64L,
                              vae_tile_overlap = 0.25,
                              cache_mode = c("off", "easy", "ucache"),
                              cache_config = NULL) {
  # FIX: same vae_decode_only guard as sd_img2img (see vae.hpp:719)
  if (isTRUE(attr(ctx, "vae_decode_only"))) {
    stop("img2img requires VAE encoder. Recreate context with vae_decode_only = FALSE.",
         call. = FALSE)
  }
  if (is.null(width)) width <- init_image$width
  if (is.null(height)) height <- init_image$height

  if (is.null(sample_tile_size)) {
    model_type <- attr(ctx, "model_type") %||% "sd1"
    sample_tile_size <- .native_latent_tile_size(model_type)
  }
  sample_tile_size <- as.integer(sample_tile_size)
  stopifnot(sample_tile_size >= 8L)

  vae_tiling_resolved <- .resolve_vae_tiling(
    vae_mode = vae_mode,
    vae_tiling = NULL,
    width = width,
    height = height,
    vae_auto_threshold = vae_auto_threshold,
    ctx = ctx,
    batch = batch_count
  )

  params <- list(
    prompt = prompt,
    negative_prompt = negative_prompt,
    init_image = init_image,
    width = as.integer(width),
    height = as.integer(height),
    sample_method = as.integer(sample_method),
    sample_steps = as.integer(sample_steps),
    cfg_scale = as.numeric(cfg_scale),
    seed = as.integer(seed),
    batch_count = as.integer(batch_count),
    scheduler = as.integer(scheduler),
    clip_skip = as.integer(clip_skip),
    strength = as.numeric(strength),
    eta = as.numeric(eta),
    control_strength = 0.9,
    vae_tiling = vae_tiling_resolved,
    vae_tile_size = as.integer(vae_tile_size),
    vae_tile_overlap = as.numeric(vae_tile_overlap),
    tiled_sampling = TRUE,
    sample_tile_size = sample_tile_size,
    sample_tile_overlap = as.numeric(sample_tile_overlap)
  )
  if (!is.null(flow_shift)) {
    params$flow_shift <- as.numeric(flow_shift)
  }
  params <- .apply_cache_params(params, cache_mode, cache_config)

  sd_generate_image(ctx, params)
}

#' High-resolution image generation (Highres Fix)
#'
#' Two-pass generation: first creates a base image at native model resolution,
#' then upscales and refines with tiled img2img to produce a high-resolution
#' result with coherent global composition.
#'
#' @inheritParams sd_txt2img
#' @param width Target output width in pixels (default 2048)
#' @param height Target output height in pixels (default 2048)
#' @param hr_strength Denoising strength for the refinement pass (0.0-1.0,
#'   default 0.4). Lower = more faithful to base, higher = more detail/change.
#' @param hr_steps Sample steps for refinement pass (default same as sample_steps)
#' @param sample_tile_size Tile size in latent pixels for refinement (default auto)
#' @param sample_tile_overlap Tile overlap fraction (default 0.25)
#' @param upscaler Path to ESRGAN model for upscaling. If NULL, uses bilinear.
#' @param upscale_factor ESRGAN upscale factor (default 4, only used with upscaler)
#' @return SD image (single image, not list)
#' @keywords internal
sd_highres_fix <- function(ctx,
                            prompt,
                            negative_prompt = "",
                            width = 2048L,
                            height = 2048L,
                            sample_method = SAMPLE_METHOD$EULER,
                            sample_steps = 20L,
                            cfg_scale = 7.0,
                            seed = 42L,
                            scheduler = SCHEDULER$DISCRETE,
                            clip_skip = -1L,
                            eta = 0.0,
                            flow_shift = NULL,
                            hr_strength = 0.4,
                            hr_steps = NULL,
                            sample_tile_size = NULL,
                            sample_tile_overlap = 0.25,
                            upscaler = NULL,
                            upscale_factor = 4L,
                            vae_mode = "auto",
                            vae_auto_threshold = 1048576L,
                            vae_tile_size = 64L,
                            vae_tile_overlap = 0.25,
                            cache_mode = c("off", "easy", "ucache"),
                            cache_config = NULL) {
  width <- as.integer(width)
  height <- as.integer(height)
  if (is.null(hr_steps)) hr_steps <- sample_steps

  model_type <- attr(ctx, "model_type") %||% "sd1"
  native_px <- .native_tile_size(model_type)

  # Step 1: base generation at native resolution
  aspect <- width / height
  if (aspect >= 1) {
    base_w <- native_px
    base_h <- as.integer(round(native_px / aspect / 8) * 8)
  } else {
    base_h <- native_px
    base_w <- as.integer(round(native_px * aspect / 8) * 8)
  }
  base_w <- max(base_w, 64L)
  base_h <- max(base_h, 64L)

  message(sprintf("[highres_fix] Step 1: base %dx%d", base_w, base_h))
  base_imgs <- sd_txt2img(ctx, prompt,
                           negative_prompt = negative_prompt,
                           width = base_w, height = base_h,
                           sample_method = sample_method,
                           sample_steps = sample_steps,
                           cfg_scale = cfg_scale,
                           seed = seed,
                           scheduler = scheduler,
                           clip_skip = clip_skip,
                           eta = eta,
                           flow_shift = flow_shift,
                           cache_mode = cache_mode,
                           cache_config = cache_config)
  base_img <- base_imgs[[1]]

  # Step 2: upscale to target resolution
  if (!is.null(upscaler) && file.exists(upscaler)) {
    message(sprintf("[highres_fix] Step 2: ESRGAN upscale %dx", upscale_factor))
    upscaled <- sd_upscale_image(upscaler, base_img,
                                  upscale_factor = upscale_factor)
    if (upscaled$width != width || upscaled$height != height) {
      upscaled <- .resize_sd_image(upscaled, width, height)
    }
  } else {
    message(sprintf("[highres_fix] Step 2: bilinear upscale to %dx%d", width, height))
    upscaled <- .resize_sd_image(base_img, width, height)
  }

  # Step 3: img2img refinement.
  # Tiled sampling (MultiDiffusion) only works for convolutional UNets. DiT
  # denoisers (Flux, Flux.2, SD3) patchify the latent and do not preserve the
  # tile's spatial layout, so tiling the sampler produces garbage/black output.
  # For those, refine with direct img2img (VAE tiling still applies).
  if (model_type %in% c("flux", "flux2", "sd3")) {
    message(sprintf("[highres_fix] Step 3: img2img (DiT, no tiled sampling) (strength=%.2f, steps=%d)",
                    hr_strength, hr_steps))
    result <- sd_img2img(ctx, prompt,
                         init_image = upscaled,
                         negative_prompt = negative_prompt,
                         width = width,
                         height = height,
                         sample_method = sample_method,
                         sample_steps = hr_steps,
                         cfg_scale = cfg_scale,
                         seed = seed,
                         scheduler = scheduler,
                         clip_skip = clip_skip,
                         strength = hr_strength,
                         eta = eta,
                         flow_shift = flow_shift,
                         vae_mode = vae_mode,
                         vae_auto_threshold = vae_auto_threshold,
                         vae_tile_size = vae_tile_size,
                         vae_tile_overlap = vae_tile_overlap,
                         cache_mode = cache_mode,
                         cache_config = cache_config)
    return(result[[1]])
  }

  message(sprintf("[highres_fix] Step 3: tiled img2img (strength=%.2f, steps=%d)",
                  hr_strength, hr_steps))
  result <- sd_img2img_tiled(ctx, prompt,
                              init_image = upscaled,
                              negative_prompt = negative_prompt,
                              width = width,
                              height = height,
                              sample_tile_size = sample_tile_size,
                              sample_tile_overlap = sample_tile_overlap,
                              sample_method = sample_method,
                              sample_steps = hr_steps,
                              cfg_scale = cfg_scale,
                              seed = seed,
                              scheduler = scheduler,
                              clip_skip = clip_skip,
                              strength = hr_strength,
                              eta = eta,
                              flow_shift = flow_shift,
                              vae_mode = vae_mode,
                              vae_auto_threshold = vae_auto_threshold,
                              vae_tile_size = vae_tile_size,
                              vae_tile_overlap = vae_tile_overlap,
                              cache_mode = cache_mode,
                              cache_config = cache_config)
  result[[1]]
}

#' Get native latent tile size for a model type
#' @param model_type One of "sd1", "sd2", "sdxl", "flux", "flux2", "sd3"
#' @return Integer tile size in latent pixels
#' @keywords internal
.native_latent_tile_size <- function(model_type) {
  switch(model_type,
    sd1  = 64L,   # 64 * 8 = 512px
    sd2  = 64L,   # 64 * 8 = 512px
    sdxl  = 128L,  # 128 * 8 = 1024px
    flux  = 128L,
    flux2 = 128L,
    sd3   = 128L,
    64L
  )
}

#' High-resolution image generation via patch-based pipeline
#'
#' Generates a large image by independently rendering overlapping patches at
#' the model's native resolution, then stitching them with linear blending.
#' An optional \code{img2img} harmonization pass can smooth seams further.
#'
#' @param ctx SD context created by \code{\link{sd_ctx}}
#' @param prompt Text prompt
#' @param negative_prompt Negative prompt (default "")
#' @param width Target image width in pixels
#' @param height Target image height in pixels
#' @param tile_size Patch size in pixels. \code{NULL} = auto-detect from
#'   \code{model_type} attribute on \code{ctx} (512 for SD1/SD2, 1024 for
#'   SDXL/Flux/SD3). Must be divisible by 8.
#' @param overlap Overlap between patches as fraction of \code{tile_size},
#'   0.0-0.5 (default 0.125).
#' @param img2img_strength If not \code{NULL}, run a final \code{img2img} pass
#'   over the stitched image at this denoising strength (e.g. 0.3) to
#'   harmonize seams. Requires \code{vae_decode_only = FALSE} in the context.
#'   Default \code{NULL} (disabled).
#' @param sample_method Sampling method (see \code{SAMPLE_METHOD})
#' @param sample_steps Number of sampling steps (default 20)
#' @param cfg_scale Classifier-free guidance scale (default 7.0)
#' @param seed Base random seed. Each patch gets \code{seed + patch_index}.
#'   Use -1 for random.
#' @param scheduler Scheduler type (see \code{SCHEDULER})
#' @param clip_skip Number of CLIP layers to skip (-1 = auto)
#' @param eta Eta parameter for DDIM-like samplers
#' @param vae_mode VAE tiling mode for the harmonization pass
#'   (default \code{"auto"}: VRAM-aware, see \code{\link{sd_txt2img}}).
#' @param vae_auto_threshold Pixel area fallback threshold for auto VAE tiling
#'   when VRAM query is unavailable
#' @param vae_tile_size Tile size for VAE tiling (default 64)
#' @param vae_tile_overlap Overlap for VAE tiling (default 0.25)
#' @return SD image (list with width, height, channel, data)
#' @export
#' @examples
#' \dontrun{
#' ctx <- sd_ctx("sd15.safetensors", model_type = "sd1")
#' img <- sd_txt2img_highres(ctx, "a panoramic mountain landscape",
#'                           width = 2048, height = 1024)
#' sd_save_image(img, "panorama.png")
#' }
sd_txt2img_highres <- function(ctx,
                                prompt,
                                negative_prompt = "",
                                width = 2048L,
                                height = 2048L,
                                tile_size = NULL,
                                overlap = 0.125,
                                img2img_strength = NULL,
                                sample_method = SAMPLE_METHOD$EULER,
                                sample_steps = 20L,
                                cfg_scale = 7.0,
                                seed = 42L,
                                scheduler = SCHEDULER$DISCRETE,
                                clip_skip = -1L,
                                eta = 0.0,
                                vae_mode = "auto",
                                vae_auto_threshold = 1048576L,
                                vae_tile_size = 64L,
                                vae_tile_overlap = 0.25) {
  width <- as.integer(width)
  height <- as.integer(height)

  # Determine tile size from model type
  if (is.null(tile_size)) {
    model_type <- attr(ctx, "model_type") %||% "sd1"
    tile_size <- .native_tile_size(model_type)
  }
  tile_size <- as.integer(tile_size)
  stopifnot(tile_size %% 8L == 0L, tile_size >= 64L)

  # If target fits in a single tile, just use sd_txt2img
  if (width <= tile_size && height <= tile_size) {
    return(sd_txt2img(ctx, prompt,
                      negative_prompt = negative_prompt,
                      width = width, height = height,
                      sample_method = sample_method,
                      sample_steps = sample_steps,
                      cfg_scale = cfg_scale, seed = seed,
                      scheduler = scheduler, clip_skip = clip_skip,
                      eta = eta, vae_mode = vae_mode,
                      vae_auto_threshold = vae_auto_threshold,
                      vae_tile_size = vae_tile_size,
                      vae_tile_overlap = vae_tile_overlap)[[1]])
  }

  # Compute patch grid
  overlap_px <- as.integer(round(tile_size * overlap))
  grid <- .compute_patch_grid(width, height, tile_size, overlap_px)

  # Allocate output canvas [H, W, 3]
  canvas <- array(0, dim = c(height, width, 3L))
  weights <- array(0, dim = c(height, width, 1L))

  base_seed <- as.integer(seed)

  for (i in seq_len(nrow(grid))) {
    g <- grid[i, ]
    patch_seed <- if (base_seed < 0L) -1L else base_seed + i - 1L

    patch_imgs <- sd_txt2img(ctx, prompt,
                              negative_prompt = negative_prompt,
                              width = tile_size, height = tile_size,
                              sample_method = sample_method,
                              sample_steps = sample_steps,
                              cfg_scale = cfg_scale,
                              seed = patch_seed,
                              batch_count = 1L,
                              scheduler = scheduler,
                              clip_skip = clip_skip, eta = eta,
                              vae_mode = "normal")
    patch_arr <- sd_image_to_array(patch_imgs[[1]])  # [H, W, 3]

    # Build linear blend mask for this patch
    mask <- .blend_mask(tile_size, tile_size, overlap_px,
                        is_left = (g$x == 0),
                        is_top = (g$y == 0),
                        is_right = (g$x + tile_size >= width),
                        is_bottom = (g$y + tile_size >= height))

    # Crop patch if it extends beyond canvas (edge patches)
    ph <- min(tile_size, height - g$y)
    pw <- min(tile_size, width - g$x)
    ys <- (g$y + 1L):(g$y + ph)
    xs <- (g$x + 1L):(g$x + pw)

    patch_crop <- patch_arr[1:ph, 1:pw, , drop = FALSE]
    mask_crop <- mask[1:ph, 1:pw, drop = FALSE]

    for (ch in 1:3) {
      canvas[ys, xs, ch] <- canvas[ys, xs, ch] + patch_crop[, , ch] * mask_crop
    }
    weights[ys, xs, 1] <- weights[ys, xs, 1] + mask_crop
  }

  # Normalize by weights
  for (ch in 1:3) {
    canvas[, , ch] <- canvas[, , ch] / pmax(weights[, , 1], 1e-8)
  }
  canvas <- pmin(pmax(canvas, 0), 1)

  # Convert to sd_image
  result <- .array_to_sd_image(canvas)

  # Optional harmonization pass
  if (!is.null(img2img_strength) && img2img_strength > 0) {
    harmonized <- sd_img2img(ctx, prompt,
                              init_image = result,
                              negative_prompt = negative_prompt,
                              width = width, height = height,
                              sample_method = sample_method,
                              sample_steps = sample_steps,
                              cfg_scale = cfg_scale,
                              seed = base_seed,
                              batch_count = 1L,
                              scheduler = scheduler,
                              clip_skip = clip_skip,
                              strength = img2img_strength,
                              eta = eta,
                              vae_mode = vae_mode,
                              vae_auto_threshold = vae_auto_threshold,
                              vae_tile_size = vae_tile_size,
                              vae_tile_overlap = vae_tile_overlap)
    result <- harmonized[[1]]
  }

  result
}

#' Resolve device layout preset to concrete GPU indices
#'
#' @param layout One of "mono", "split_encoders", "split_vae", "encoders_cpu"
#' @param diffusion_gpu Manual override (-1 = use layout)
#' @param clip_gpu Manual override (-1 = use layout)
#' @param vae_gpu Manual override (-1 = use layout)
#' @param keep_clip_on_cpu Existing keep_clip_on_cpu flag
#' @param keep_vae_on_cpu Existing keep_vae_on_cpu flag
#' @return List with diffusion, clip, vae (GPU indices), clip_on_cpu, vae_on_cpu
#' @keywords internal
.resolve_device_layout <- function(layout, diffusion_gpu, clip_gpu, vae_gpu,
                                    keep_clip_on_cpu, keep_vae_on_cpu) {
  layout <- match.arg(layout, c("mono", "split_encoders", "split_vae",
                                 "encoders_cpu"))
  has_manual <- any(c(diffusion_gpu, clip_gpu, vae_gpu) >= 0L)

  if (has_manual) {
    return(list(
      diffusion = as.integer(diffusion_gpu),
      clip      = as.integer(clip_gpu),
      vae       = as.integer(vae_gpu),
      clip_on_cpu = keep_clip_on_cpu,
      vae_on_cpu  = keep_vae_on_cpu
    ))
  }

  switch(layout,
    mono = list(
      diffusion = -1L, clip = -1L, vae = -1L,
      clip_on_cpu = keep_clip_on_cpu, vae_on_cpu = keep_vae_on_cpu
    ),
    split_encoders = list(
      diffusion = 0L, clip = 1L, vae = -1L,
      clip_on_cpu = FALSE, vae_on_cpu = keep_vae_on_cpu
    ),
    split_vae = list(
      diffusion = 0L, clip = 1L, vae = 1L,
      clip_on_cpu = FALSE, vae_on_cpu = FALSE
    ),
    encoders_cpu = list(
      diffusion = -1L, clip = -1L, vae = -1L,
      clip_on_cpu = TRUE, vae_on_cpu = keep_vae_on_cpu
    )
  )
}

#' Get native tile size for a model type
#' @param model_type One of "sd1", "sd2", "sdxl", "flux", "flux2", "sd3"
#' @return Integer tile size in pixels
#' @keywords internal
.native_tile_size <- function(model_type) {
  switch(model_type,
    sd1  = 512L,
    sd2  = 512L,
    sdxl  = 1024L,
    flux  = 1024L,
    flux2 = 1024L,
    sd3   = 1024L,
    768L
  )
}

#' Compute patch grid positions
#' @param width Target width
#' @param height Target height
#' @param tile_size Tile size in pixels
#' @param overlap_px Overlap in pixels
#' @return Data frame with columns x, y (0-based top-left of each patch)
#' @importFrom utils tail
#' @keywords internal
.compute_patch_grid <- function(width, height, tile_size, overlap_px) {
  stride <- tile_size - overlap_px

  xs <- seq(0L, max(0L, width - tile_size), by = stride)
  if (tail(xs, 1) + tile_size < width) {
    xs <- c(xs, width - tile_size)
  }

  ys <- seq(0L, max(0L, height - tile_size), by = stride)
  if (tail(ys, 1) + tile_size < height) {
    ys <- c(ys, height - tile_size)
  }

  grid <- expand.grid(x = xs, y = ys)
  grid$x <- as.integer(grid$x)
  grid$y <- as.integer(grid$y)
  grid
}

#' Build linear blend mask for a patch
#' @param h Patch height
#' @param w Patch width
#' @param overlap Overlap in pixels
#' @param is_left,is_top,is_right,is_bottom Whether patch is at canvas edge
#' @return Matrix [h, w] with blend weights in [0, 1]
#' @keywords internal
.blend_mask <- function(h, w, overlap, is_left, is_top, is_right, is_bottom) {
  mask <- matrix(1, nrow = h, ncol = w)

  if (overlap > 0L) {
    ramp <- seq(0, 1, length.out = overlap + 1L)[-1]  # (0, 1]

    # Left ramp
    if (!is_left && overlap <= w) {
      mask[, 1:overlap] <- mask[, 1:overlap] * rep(ramp, each = h)
    }
    # Right ramp
    if (!is_right && overlap <= w) {
      mask[, (w - overlap + 1L):w] <- mask[, (w - overlap + 1L):w] *
        rep(rev(ramp), each = h)
    }
    # Top ramp
    if (!is_top && overlap <= h) {
      mask[1:overlap, ] <- mask[1:overlap, ] * ramp
    }
    # Bottom ramp
    if (!is_bottom && overlap <= h) {
      mask[(h - overlap + 1L):h, ] <- mask[(h - overlap + 1L):h, ] * rev(ramp)
    }
  }

  mask
}

#' Bilinear resize of an SD image
#' @param image SD image list
#' @param target_w Target width
#' @param target_h Target height
#' @return Resized SD image
#' @keywords internal
.resize_sd_image <- function(image, target_w, target_h) {
  arr <- sd_image_to_array(image)  # [H, W, C] in [0,1]
  src_h <- dim(arr)[1]
  src_w <- dim(arr)[2]
  ch <- dim(arr)[3]

  out <- array(0, dim = c(target_h, target_w, ch))

  # Coordinate mapping: target pixel -> source pixel (center-aligned)
  sy <- (seq_len(target_h) - 0.5) * src_h / target_h
  sx <- (seq_len(target_w) - 0.5) * src_w / target_w

  y0 <- as.integer(pmax(floor(sy), 1))
  y1 <- as.integer(pmin(y0 + 1L, src_h))
  fy <- sy - floor(sy)

  x0 <- as.integer(pmax(floor(sx), 1))
  x1 <- as.integer(pmin(x0 + 1L, src_w))
  fx <- sx - floor(sx)

  # FIX: arr is 3D [H, W, C]. Indexing arr[y0, , c, drop=FALSE] on a 3D array
  # returns a 3D result, then top[, x0] crashes with "wrong number of dimensions".
  # Solution: extract 2D matrix per channel first, then interpolate on [H, W].
  for (ci in seq_len(ch)) {
    mat <- arr[, , ci]  # [src_h, src_w]
    # Interpolate along Y: top[i,j] = mat[y0[i], j] * (1-fy[i]) + mat[y1[i], j] * fy[i]
    top <- mat[y0, , drop = FALSE] * (1 - fy) + mat[y1, , drop = FALSE] * fy
    # top is [target_h, src_w]. Now interpolate along X:
    out[, , ci] <- top[, x0, drop = FALSE] * (1 - rep(fx, each = target_h)) +
                   top[, x1, drop = FALSE] * rep(fx, each = target_h)
  }

  .array_to_sd_image(out)
}

#' Convert R array [H, W, 3] to sd_image list
#' @param arr 3D numeric array [height, width, channels] in [0, 1]
#' @return SD image list (width, height, channel, data)
#' @keywords internal
.array_to_sd_image <- function(arr) {
  h <- dim(arr)[1]
  w <- dim(arr)[2]
  ch <- dim(arr)[3]
  # R array [H, W, C] → row-major interleaved [y][x][c]
  interleaved <- aperm(arr, c(3, 2, 1))
  bytes <- as.raw(as.integer(pmin(pmax(as.numeric(interleaved) * 255, 0), 255)))
  list(width = as.integer(w), height = as.integer(h),
       channel = as.integer(ch), data = bytes)
}

#' Upscale an image using ESRGAN
#'
#' @param esrgan_path Path to ESRGAN model file
#' @param image SD image to upscale (list with width, height, channel, data)
#' @param upscale_factor Upscale factor (default 4)
#' @param n_threads Number of CPU threads (0 = auto-detect)
#' @return Upscaled SD image
#' @export
sd_upscale_image <- function(esrgan_path, image, upscale_factor = 4L,
                              n_threads = 0L) {
  if (!file.exists(esrgan_path)) {
    stop("ESRGAN model not found: ", esrgan_path, call. = FALSE)
  }
  upscaler <- sd_create_upscaler(
    normalizePath(esrgan_path),
    n_threads = as.integer(n_threads)
  )
  on.exit(rm(upscaler), add = TRUE)
  sd_upscale(upscaler, image, as.integer(upscale_factor))
}

#' Convert model to different quantization format
#'
#' @param input_path Path to input model file
#' @param output_path Path for output model file
#' @param output_type Target quantization type (see \code{SD_TYPE})
#' @param vae_path Optional path to separate VAE model
#' @param tensor_type_rules Optional tensor type rules string
#' @return TRUE on success
#' @export
sd_convert <- function(input_path, output_path, output_type = SD_TYPE$F16,
                       vae_path = NULL, tensor_type_rules = NULL) {
  if (!file.exists(input_path)) {
    stop("Input model not found: ", input_path, call. = FALSE)
  }
  sd_convert_model(
    normalizePath(input_path),
    output_path,
    as.integer(output_type),
    vae_path = if (!is.null(vae_path)) normalizePath(vae_path) else "",
    tensor_type_rules = tensor_type_rules %||% ""
  )
}

#' Estimate peak VAE VRAM usage in bytes
#'
#' Analytic upper bound on the peak compute-buffer size of the VAE decoder.
#' The peak occurs in the ResNet block that runs at full pixel resolution
#' (W x H) with the decoder's base channel width. The per-pixel cost is
#' derived from architecture (base channels x dtype bytes); the only
#' empirical constant is \code{live_tensors} — how many such full-res
#' tensors ggml's graph allocator keeps alive simultaneously. That value
#' is calibrated against an observed Flux failure: a 2048x1024 decode
#' requested 19238223904 bytes, i.e. 19238223904 / (2048*1024) ~= 9175
#' B/px, and 9175 / (128 ch * 4 B) ~= 17.9 live full-res tensors. We round
#' up to 18 for a safe over-estimate (tiling should engage rather than OOM).
#'
#' @param width Image width in pixels
#' @param height Image height in pixels
#' @param model_type Model type string ("sd1", "sd2", "sdxl", "flux", etc.)
#' @param batch Batch size (default 1)
#' @return Estimated peak VRAM in bytes
#' @keywords internal
.estimate_vae_vram <- function(width, height, model_type = "sd1", batch = 1L) {
  # Decoder base channel width at the full-resolution level. SD/SDXL/Flux
  # VAEs all use 128 channels at this stage.
  base_channels <- 128
  dtype_bytes   <- 4    # f32 compute buffer
  # Empirically calibrated: count of simultaneously-live full-res tensors
  # in ggml's gallocr reservation for the VAE decode graph (see @details).
  live_tensors  <- 18

  per_pixel <- base_channels * dtype_bytes * live_tensors
  as.numeric(width) * as.numeric(height) * per_pixel * as.numeric(batch)
}

#' Resolve VAE tiling mode to boolean
#'
#' In \code{"auto"} mode, queries free VRAM from the Vulkan backend and
#' compares against \code{\link{.estimate_vae_vram}}. Falls back to the
#' pixel-area \code{vae_auto_threshold} when VRAM query is unavailable.
#'
#' @param vae_mode One of "normal", "tiled", "auto"
#' @param vae_tiling Deprecated boolean flag (NULL if not set)
#' @param width Image width in pixels
#' @param height Image height in pixels
#' @param vae_auto_threshold Pixel area threshold — fallback for auto mode
#'   when VRAM query fails
#' @param ctx SD context (used to read device index and model_type).
#'   NULL disables VRAM-aware logic.
#' @param batch Batch size for VRAM estimation (default 1)
#' @param system_reserve Bytes to keep free as safety margin (default 50 MB)
#' @return Logical, TRUE if tiling should be enabled
#' @keywords internal
.resolve_vae_tiling <- function(vae_mode, vae_tiling, width, height,
                                vae_auto_threshold, ctx = NULL, batch = 1L,
                                system_reserve = 50 * 1024^2) {
  if (!is.null(vae_tiling)) {
    warning("'vae_tiling' is deprecated. Use vae_mode = \"tiled\" instead.",
            call. = FALSE)
    return(isTRUE(vae_tiling))
  }
  vae_mode <- match.arg(vae_mode, c("normal", "tiled", "auto"))
  if (vae_mode != "auto") {
    return(vae_mode == "tiled")
  }

  # Optional diagnostics: set SD2R_DEBUG_VAE=1 to trace the auto decision.
  # Helps localize why a large decode was (not) tiled (see highres_fix OOM).
  .dbg <- nzchar(Sys.getenv("SD2R_DEBUG_VAE"))

  # --- auto mode: try VRAM-aware decision first ---
  if (!is.null(ctx)) {
    device <- attr(ctx, "vram_device") %||% 0L
    model_type <- attr(ctx, "model_type") %||% "sd1"
    free_vram <- tryCatch({
      ggmlR::ggml_vulkan_device_memory(device)$free
    }, error = function(e) NULL)

    if (!is.null(free_vram) && is.numeric(free_vram) && free_vram > 0) {
      required <- .estimate_vae_vram(width, height, model_type, batch) +
        system_reserve
      decision <- required > free_vram
      if (.dbg) {
        message(sprintf(
          "[vae_tiling] auto/VRAM: %dx%d %s dev=%d free=%.2fGB required=%.2fGB -> tiled=%s",
          as.integer(width), as.integer(height), model_type, device,
          free_vram / 1e9, required / 1e9, decision))
      }
      return(decision)
    }
  }

  # --- fallback: static pixel-area threshold ---
  decision <- as.integer(width) * as.integer(height) >= as.numeric(vae_auto_threshold)
  if (.dbg) {
    message(sprintf(
      "[vae_tiling] auto/pixel-area (no VRAM query): %dx%d area=%.0f threshold=%.0f -> tiled=%s",
      as.integer(width), as.integer(height),
      as.numeric(width) * as.numeric(height),
      as.numeric(vae_auto_threshold), decision))
  }
  decision
}

#' Parallel generation across multiple GPUs
#'
#' Distributes prompts across available Vulkan GPUs, running one process per
#' GPU via \code{callr}. Each process creates its own \code{\link{sd_ctx}} and
#' calls \code{\link{sd_generate}}. Requires the \code{callr} package.
#'
#' @param model_path Path to the model file (single-file models like SD 1.x/2.x/SDXL)
#' @param prompts Character vector of prompts (one image per prompt)
#' @param negative_prompt Negative prompt applied to all images (default "")
#' @param devices Integer vector of Vulkan device indices (0-based). Default
#'   \code{NULL} auto-detects all available devices.
#' @param seeds Integer vector of seeds, same length as \code{prompts}. Default
#'   \code{NULL} generates random seeds.
#' @param width Image width (default 512)
#' @param height Image height (default 512)
#' @param model_type Model type (default "sd1")
#' @param vram_gb VRAM per GPU for auto-routing (default NULL)
#' @param vae_decode_only VAE decode only (default TRUE)
#' @param progress Print progress messages (default TRUE)
#' @param diffusion_model_path Path to diffusion model (Flux/multi-file models)
#' @param vae_path Path to VAE model
#' @param clip_l_path Path to CLIP-L model
#' @param t5xxl_path Path to T5-XXL model
#' @param llm_path Path to an LLM text encoder (Qwen3 / Mistral), e.g. FLUX.2
#' @param ... Additional arguments passed to \code{\link{sd_generate}}
#' @return List of SD images, one per prompt, in original order.
#' @note Release any existing SD context (\code{rm(ctx); gc()}) before calling
#'   this function. Holding a Vulkan context in the main process while
#'   subprocesses try to use the same GPU can produce corrupted (grey) images.
#' @export
#' @examples
#' \dontrun{
#' # Single-file model (SD 1.x/2.x/SDXL)
#' imgs <- sd_generate_multi_gpu(
#'   "model.safetensors",
#'   prompts = c("a cat", "a dog", "a bird", "a fish"),
#'   devices = 0:1
#' )
#'
#' # Multi-file model (Flux)
#' imgs <- sd_generate_multi_gpu(
#'   diffusion_model_path = "flux1-dev-Q4_K_S.gguf",
#'   vae_path = "ae.safetensors",
#'   clip_l_path = "clip_l.safetensors",
#'   t5xxl_path = "t5-v1_1-xxl-encoder-Q5_K_M.gguf",
#'   prompts = c("a cat", "a dog"),
#'   model_type = "flux", devices = 0:1
#' )
#' }
sd_generate_multi_gpu <- function(model_path = NULL,
                                  prompts,
                                  negative_prompt = "",
                                  devices = NULL,
                                  seeds = NULL,
                                  width = 512L,
                                  height = 512L,
                                  model_type = "sd1",
                                  vram_gb = NULL,
                                  vae_decode_only = TRUE,
                                  progress = TRUE,
                                  diffusion_model_path = NULL,
                                  vae_path = NULL,
                                  clip_l_path = NULL,
                                  t5xxl_path = NULL,
                                  llm_path = NULL,
                                  ...) {
  if (!requireNamespace("callr", quietly = TRUE)) {
    stop("Package 'callr' is required for multi-GPU generation. ",
         "Install it with: install.packages('callr')", call. = FALSE)
  }

  # Warn about potential Vulkan conflicts with existing contexts
  if (progress) {
    message("Note: ensure all sd_ctx() contexts are released (rm(ctx); gc()) ",
            "before calling sd_generate_multi_gpu() to avoid GPU conflicts.")
  }

  # Auto-detect devices
  if (is.null(devices)) {
    n_gpu <- tryCatch(ggmlR::ggml_vulkan_device_count(), error = function(e) 1L)
    if (n_gpu < 1L) stop("No Vulkan devices found", call. = FALSE)
    devices <- seq(0L, n_gpu - 1L)
  }
  devices <- as.integer(devices)
  n_gpu <- length(devices)

  n_prompts <- length(prompts)
  if (n_prompts == 0L) return(list())

  # Generate seeds if not provided
  if (is.null(seeds)) {
    seeds <- sample.int(.Machine$integer.max, n_prompts)
  }
  stopifnot(length(seeds) == n_prompts)

  # Validate model paths
  if (is.null(model_path) && is.null(diffusion_model_path)) {
    stop("Either 'model_path' or 'diffusion_model_path' must be provided", call. = FALSE)
  }
  if (!is.null(model_path)) model_path <- normalizePath(model_path)
  if (!is.null(diffusion_model_path)) diffusion_model_path <- normalizePath(diffusion_model_path)
  if (!is.null(vae_path)) vae_path <- normalizePath(vae_path)
  if (!is.null(clip_l_path)) clip_l_path <- normalizePath(clip_l_path)
  if (!is.null(t5xxl_path)) t5xxl_path <- normalizePath(t5xxl_path)
  if (!is.null(llm_path)) llm_path <- normalizePath(llm_path)

  # Capture extra args
  extra_args <- list(...)

  if (progress) message(sprintf("Multi-GPU: %d prompts on %d device(s)", n_prompts, n_gpu))

  # Worker pool: max n_gpu concurrent processes
  running <- list()  # list of list(job, idx, dev_idx)
  results <- vector("list", n_prompts)
  queue <- seq_len(n_prompts)
  done_count <- 0L

  while (length(queue) > 0L || length(running) > 0L) {
    # Launch new jobs on free devices
    busy_devs <- vapply(running, function(x) x$dev_idx, integer(1))
    for (d in seq_len(n_gpu)) {
      if (length(queue) == 0L) break
      if (d %in% busy_devs) next

      idx <- queue[1L]
      queue <- queue[-1L]
      dev <- devices[d]

      job <- callr::r_bg(
        function(model_path, diffusion_model_path, vae_path, clip_l_path,
                 t5xxl_path, llm_path, prompt, negative_prompt, width, height, seed,
                 model_type, vram_gb, vae_decode_only, dev, extra_args) {
          Sys.setenv(SD_VK_DEVICE = as.character(dev))
          library(sd2R)
          ctx <- sd_ctx(model_path = model_path,
                        diffusion_model_path = diffusion_model_path,
                        vae_path = vae_path,
                        clip_l_path = clip_l_path,
                        t5xxl_path = t5xxl_path,
                        llm_path = llm_path,
                        model_type = model_type,
                        vram_gb = vram_gb,
                        vae_decode_only = vae_decode_only)
          args <- c(list(ctx = ctx, prompt = prompt,
                         negative_prompt = negative_prompt,
                         width = as.integer(width), height = as.integer(height),
                         seed = as.integer(seed)),
                    extra_args)
          imgs <- do.call(sd_generate, args)
          imgs[[1]]
        },
        args = list(
          model_path = model_path,
          diffusion_model_path = diffusion_model_path,
          vae_path = vae_path,
          clip_l_path = clip_l_path,
          t5xxl_path = t5xxl_path,
          llm_path = llm_path,
          prompt = prompts[idx],
          negative_prompt = negative_prompt,
          width = width, height = height, seed = seeds[idx],
          model_type = model_type, vram_gb = vram_gb,
          vae_decode_only = vae_decode_only,
          dev = dev, extra_args = extra_args
        ),
        supervise = TRUE
      )
      running <- c(running, list(list(job = job, idx = idx, dev_idx = d)))
    }

    if (length(running) == 0L) break

    # Poll for completed jobs
    Sys.sleep(0.5)
    finished <- vapply(running, function(x) !x$job$is_alive(), logical(1))

    for (x in running[finished]) {
      res <- tryCatch(x$job$get_result(), error = function(e) e)
      results[[x$idx]] <- res
      done_count <- done_count + 1L
      if (progress) {
        status <- if (inherits(res, "error")) "FAILED" else "done"
        message(sprintf("[%d/%d] GPU%d: %s", done_count, n_prompts,
                        devices[x$dev_idx], status))
      }
    }
    running <- running[!finished]
  }

  results
}

#' @keywords internal
`%||%` <- function(x, y) if (is.null(x)) y else x
