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
#' @return An external pointer to the SD context (class "sd_ctx")
#' @export
#' @examples
#' \dontrun{
#' ctx <- sd_ctx("model.safetensors")
#' imgs <- sd_txt2img(ctx, "a cat sitting on a chair")
#' sd_save_image(imgs[[1]], "cat.png")
#' }
sd_ctx <- function(model_path,
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
                   flow_shift = 0.0) {

  if (!file.exists(model_path)) {
    stop("Model file not found: ", model_path, call. = FALSE)
  }

  params <- list(
    model_path = normalizePath(model_path),
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

  sd_create_context(params)
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
#' @param vae_tiling Enable tiled VAE decode to reduce VRAM usage (default FALSE)
#' @param vae_tile_size Tile size in latent pixels for tiled VAE (default 64)
#' @param vae_tile_overlap Overlap ratio between tiles, 0.0-0.5 (default 0.25)
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
                       vae_tiling = FALSE,
                       vae_tile_size = 64L,
                       vae_tile_overlap = 0.25) {
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
    vae_tiling = isTRUE(vae_tiling),
    vae_tile_size = as.integer(vae_tile_size),
    vae_tile_overlap = as.numeric(vae_tile_overlap)
  )
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
                       vae_tiling = FALSE,
                       vae_tile_size = 64L,
                       vae_tile_overlap = 0.25) {
  if (is.null(width)) width <- init_image$width
  if (is.null(height)) height <- init_image$height

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
    vae_tiling = isTRUE(vae_tiling),
    vae_tile_size = as.integer(vae_tile_size),
    vae_tile_overlap = as.numeric(vae_tile_overlap)
  )

  sd_generate_image(ctx, params)
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

#' @keywords internal
`%||%` <- function(x, y) if (is.null(x)) y else x
