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
#' @param diffusion_flash_attn Enable flash attention for diffusion model
#' @param rng_type RNG type (see \code{RNG_TYPE})
#' @param prediction Prediction type override (see \code{PREDICTION}), NULL = auto
#' @param lora_apply_mode LoRA application mode (see \code{LORA_APPLY_MODE})
#' @param flow_shift Flow shift value for Flux models
#' @param model_type Model architecture hint: \code{"sd1"}, \code{"sd2"},
#'   \code{"sdxl"}, \code{"flux"}, or \code{"sd3"}. Used by
#'   \code{\link{sd_generate}} to determine native resolution and tile sizes.
#'   Default \code{"sd1"}.
#' @param vram_gb Override available VRAM in GB. When set, disables auto-detection
#'   and uses this value for strategy routing. Default \code{NULL} (auto-detect
#'   from Vulkan device).
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
                   diffusion_model_path = NULL,
                   control_net_path = NULL,
                   n_threads = 0L,
                   wtype = SD_TYPE$COUNT,
                   vae_decode_only = TRUE,
                   free_params_immediately = FALSE,
                   keep_clip_on_cpu = FALSE,
                   keep_vae_on_cpu = FALSE,
                   diffusion_flash_attn = FALSE,
                   rng_type = RNG_TYPE$CUDA,
                   prediction = NULL,
                   lora_apply_mode = LORA_APPLY_MODE$AUTO,
                   flow_shift = 0.0,
                   model_type = "sd1",
                   vram_gb = NULL,
                   verbose = FALSE) {

  sd_set_verbose(verbose)

  if (!is.null(model_path) && !file.exists(model_path)) {
    stop("Model file not found: ", model_path, call. = FALSE)
  }
  if (is.null(model_path) && is.null(diffusion_model_path)) {
    stop("Either model_path or diffusion_model_path must be provided", call. = FALSE)
  }
  model_type <- match.arg(model_type, c("sd1", "sd2", "sdxl", "flux", "sd3"))

  params <- list(
    model_path = if (!is.null(model_path)) normalizePath(model_path) else "",
    n_threads = as.integer(n_threads),
    wtype = as.integer(wtype),
    vae_decode_only = vae_decode_only,
    free_params_immediately = free_params_immediately,
    keep_clip_on_cpu = keep_clip_on_cpu,
    keep_vae_on_cpu = keep_vae_on_cpu,
    diffusion_flash_attn = diffusion_flash_attn,
    rng_type = as.integer(rng_type),
    lora_apply_mode = as.integer(lora_apply_mode),
    flow_shift = as.numeric(flow_shift)
  )

  # Optional string params
  str_params <- list(
    vae_path = vae_path,
    taesd_path = taesd_path,
    clip_l_path = clip_l_path,
    clip_g_path = clip_g_path,
    t5xxl_path = t5xxl_path,
    diffusion_model_path = diffusion_model_path,
    control_net_path = control_net_path
  )
  for (nm in names(str_params)) {
    if (!is.null(str_params[[nm]])) {
      params[[nm]] <- normalizePath(str_params[[nm]], mustWork = TRUE)
    }
  }

  if (!is.null(prediction)) {
    params$prediction <- as.integer(prediction)
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
#' @param hr_strength Denoising strength for highres fix refinement pass
#'   (default 0.4). Only used when auto-routing selects highres fix.
#' @param vae_mode VAE processing mode (default "auto")
#' @param vae_tile_size Tile size for VAE tiling (default 64)
#' @param vae_tile_overlap Overlap for VAE tiling (default 0.25)
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
                        hr_strength = 0.4,
                        vae_mode = "auto",
                        vae_tile_size = 64L,
                        vae_tile_overlap = 0.25) {
  # img2img: default to init_image dimensions when width/height not specified
  if (!is.null(init_image)) {
    if (missing(width))  width  <- init_image$width
    if (missing(height)) height <- init_image$height
  }
  width <- as.integer(width)
  height <- as.integer(height)
  model_type <- attr(ctx, "model_type") %||% "sd1"

  # Flux uses guidance-distilled models; cfg_scale should default to 1.0
  if (model_type == "flux" && cfg_scale == 7.0) {
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
                       vae_mode = vae_mode,
                       vae_tile_size = vae_tile_size,
                       vae_tile_overlap = vae_tile_overlap)
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
                 vae_mode = vae_mode,
                 vae_tile_size = vae_tile_size,
                 vae_tile_overlap = vae_tile_overlap)
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
                            eta = eta, hr_strength = hr_strength,
                            vae_mode = vae_mode,
                            vae_tile_size = vae_tile_size,
                            vae_tile_overlap = vae_tile_overlap)
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
                       eta = eta, vae_mode = vae_mode,
                       vae_tile_size = vae_tile_size,
                       vae_tile_overlap = vae_tile_overlap)
    } else {
      sd_txt2img(ctx, prompt,
                 negative_prompt = negative_prompt,
                 width = width, height = height,
                 sample_method = sample_method,
                 sample_steps = sample_steps,
                 cfg_scale = cfg_scale, seed = seed,
                 batch_count = batch_count,
                 scheduler = scheduler, clip_skip = clip_skip,
                 eta = eta, vae_mode = vae_mode,
                 vae_tile_size = vae_tile_size,
                 vae_tile_overlap = vae_tile_overlap)
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
#' @param control_image Optional control image for ControlNet (sd_image format)
#' @param control_strength ControlNet strength (default 0.9)
#' @param vae_mode VAE processing mode: \code{"normal"} (no tiling),
#'   \code{"tiled"} (always tile), or \code{"auto"} (tile when
#'   \code{width * height > vae_auto_threshold}). Default \code{"auto"}.
#' @param vae_auto_threshold Pixel area threshold for \code{vae_mode = "auto"}.
#'   Tiling activates when \code{width * height} exceeds this value.
#'   Default \code{1048576L} (1024x1024 pixels). Adjust for your VRAM budget.
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
                       control_image = NULL,
                       control_strength = 0.9,
                       vae_mode = "auto",
                       vae_auto_threshold = 1048576L,
                       vae_tile_size = 64L,
                       vae_tile_overlap = 0.25,
                       vae_tile_rel_x = NULL,
                       vae_tile_rel_y = NULL,
                       vae_tiling = NULL) {
  vae_tiling_resolved <- .resolve_vae_tiling(
    vae_mode = vae_mode,
    vae_tiling = vae_tiling,
    width = width,
    height = height,
    vae_auto_threshold = vae_auto_threshold
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
  if (!is.null(control_image)) {
    params$control_image <- control_image
  }

  sd_generate_image(ctx, params)
}

#' Generate images with img2img
#'
#' @inheritParams sd_txt2img
#' @param init_image Init image in sd_image format. Use \code{\link{sd_load_image}}
#'   to load from file.
#' @param strength Denoising strength (0.0 = no change, 1.0 = full denoise, default 0.75)
#' @return List of SD images
#' @export
sd_img2img <- function(ctx,
                       prompt,
                       init_image,
                       negative_prompt = "",
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
                       vae_mode = "auto",
                       vae_auto_threshold = 1048576L,
                       vae_tile_size = 64L,
                       vae_tile_overlap = 0.25,
                       vae_tile_rel_x = NULL,
                       vae_tile_rel_y = NULL,
                       vae_tiling = NULL) {
  # FIX: sd_ctx() defaults to vae_decode_only=TRUE, but img2img needs the VAE
  # encoder (encode_first_stage). Without this check, the C++ code hits
  # GGML_ASSERT(!decode_only || decode_graph) in vae.hpp:719.
  if (isTRUE(attr(ctx, "vae_decode_only"))) {
    stop("img2img requires VAE encoder. Recreate context with vae_decode_only = FALSE.",
         call. = FALSE)
  }
  if (is.null(width)) width <- init_image$width
  if (is.null(height)) height <- init_image$height

  vae_tiling_resolved <- .resolve_vae_tiling(
    vae_mode = vae_mode,
    vae_tiling = vae_tiling,
    width = width,
    height = height,
    vae_auto_threshold = vae_auto_threshold
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
#' @keywords internal
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
                              vae_mode = "auto",
                              vae_auto_threshold = 1048576L,
                              vae_tile_size = 64L,
                              vae_tile_overlap = 0.25,
                              vae_tile_rel_x = NULL,
                              vae_tile_rel_y = NULL) {
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
    vae_auto_threshold = vae_auto_threshold
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
                              vae_mode = "auto",
                              vae_auto_threshold = 1048576L,
                              vae_tile_size = 64L,
                              vae_tile_overlap = 0.25) {
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
    vae_auto_threshold = vae_auto_threshold
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
                            hr_strength = 0.4,
                            hr_steps = NULL,
                            sample_tile_size = NULL,
                            sample_tile_overlap = 0.25,
                            upscaler = NULL,
                            upscale_factor = 4L,
                            vae_mode = "auto",
                            vae_auto_threshold = 1048576L,
                            vae_tile_size = 64L,
                            vae_tile_overlap = 0.25) {
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
                           eta = eta)
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

  # Step 3: tiled img2img refinement
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
                              vae_mode = vae_mode,
                              vae_auto_threshold = vae_auto_threshold,
                              vae_tile_size = vae_tile_size,
                              vae_tile_overlap = vae_tile_overlap)
  result[[1]]
}

#' Get native latent tile size for a model type
#' @param model_type One of "sd1", "sd2", "sdxl", "flux", "sd3"
#' @return Integer tile size in latent pixels
#' @keywords internal
.native_latent_tile_size <- function(model_type) {
  switch(model_type,
    sd1  = 64L,   # 64 * 8 = 512px
    sd2  = 64L,   # 64 * 8 = 512px
    sdxl = 128L,  # 128 * 8 = 1024px
    flux = 128L,
    sd3  = 128L,
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
#'   (default \code{"auto"}, see \code{\link{sd_txt2img}}).
#' @param vae_auto_threshold Pixel area threshold for auto VAE tiling
#' @param vae_tile_size Tile size for VAE tiling (default 64)
#' @param vae_tile_overlap Overlap for VAE tiling (default 0.25)
#' @return SD image (list with width, height, channel, data)
#' @keywords internal
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

#' Get native tile size for a model type
#' @param model_type One of "sd1", "sd2", "sdxl", "flux", "sd3"
#' @return Integer tile size in pixels
#' @keywords internal
.native_tile_size <- function(model_type) {
  switch(model_type,
    sd1  = 512L,
    sd2  = 512L,
    sdxl = 1024L,
    flux = 1024L,
    sd3  = 1024L,
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

#' Resolve VAE tiling mode to boolean
#'
#' @param vae_mode One of "normal", "tiled", "auto"
#' @param vae_tiling Deprecated boolean flag (NULL if not set)
#' @param width Image width in pixels
#' @param height Image height in pixels
#' @param vae_auto_threshold Pixel area threshold for auto mode
#' @return Logical, TRUE if tiling should be enabled
#' @keywords internal
.resolve_vae_tiling <- function(vae_mode, vae_tiling, width, height,
                                vae_auto_threshold) {
  if (!is.null(vae_tiling)) {
    warning("'vae_tiling' is deprecated. Use vae_mode = \"tiled\" instead.",
            call. = FALSE)
    return(isTRUE(vae_tiling))
  }
  vae_mode <- match.arg(vae_mode, c("normal", "tiled", "auto"))
  switch(vae_mode,
    normal = FALSE,
    tiled  = TRUE,
    auto   = as.integer(width) * as.integer(height) >= as.numeric(vae_auto_threshold)
  )
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
                 t5xxl_path, prompt, negative_prompt, width, height, seed,
                 model_type, vram_gb, vae_decode_only, dev, extra_args) {
          Sys.setenv(SD_VK_DEVICE = as.character(dev))
          library(sdR)
          ctx <- sd_ctx(model_path = model_path,
                        diffusion_model_path = diffusion_model_path,
                        vae_path = vae_path,
                        clip_l_path = clip_l_path,
                        t5xxl_path = t5xxl_path,
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
