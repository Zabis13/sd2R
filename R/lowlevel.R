# ===========================================================================
# Low-level pipeline steps (TODO 9.2)
# ===========================================================================
# Explicit per-step entry points exposing what sd_generate() does internally:
# text/image encode, sampling loop, latent decode. Building blocks for graph
# pipelines (TODO 7) and multiref (TODO 9.7).
#
# Tensors cross the R boundary as plain lists: an sd_tensor is
# list(type, ne, data) with data a numeric (f32) vector; a conditioning is
# list(crossattn, vector, concat), each an sd_tensor or NULL. See the C side
# in src/sd/stable-diffusion.cpp (Variant B ownership: results are RAM copies).

#' Encode a text prompt into conditioning (low-level)
#'
#' Runs only the text-encoder stage of the pipeline, returning the
#' conditioning tensors (analogue of \code{SDCondition}). Building block for
#' custom pipelines; most users want \code{\link{sd_generate}}.
#'
#' @param ctx SD context from \code{\link{sd_ctx}}
#' @param prompt Text prompt
#' @param clip_skip CLIP layers to skip (-1 = model default)
#' @param width,height Intended generation size (affects size-conditioning for
#'   some models, e.g. SDXL). -1 lets the model decide.
#' @return A conditioning list with elements \code{crossattn}, \code{vector},
#'   \code{concat}; each is an sd_tensor list (\code{type}, \code{ne},
#'   \code{data}) or \code{NULL} when the model does not produce it.
#' @export
#' @seealso \code{\link{sd_sample}}, \code{\link{sd_decode_latent}}
sd_encode_text <- function(ctx, prompt, clip_skip = -1L, width = -1L, height = -1L) {
  sd_encode_text_cpp(ctx, as.character(prompt), as.integer(clip_skip),
                     as.integer(width), as.integer(height))
}

#' Encode an image into a latent (low-level VAE encode)
#'
#' @param ctx SD context (must be built with \code{vae_decode_only = FALSE})
#' @param image An sd_image list (\code{width}, \code{height}, \code{channel},
#'   \code{data}) as produced by \code{\link{sd_load_image}}.
#' @return An sd_tensor list (\code{type}, \code{ne}, \code{data}) — the latent.
#' @export
#' @seealso \code{\link{sd_decode_latent}}, \code{\link{sd_sample}}
sd_encode_image <- function(ctx, image) {
  sd_encode_image_cpp(ctx, image)
}

#' Decode a latent into a pixel image (low-level VAE decode)
#'
#' @param ctx SD context
#' @param latent An sd_tensor list (e.g. the output of \code{\link{sd_sample}}
#'   or \code{\link{sd_encode_image}}).
#' @return An sd_image list (\code{width}, \code{height}, \code{channel},
#'   \code{data}).
#' @export
#' @seealso \code{\link{sd_save_image}}
sd_decode_latent <- function(ctx, latent) {
  sd_decode_latent_cpp(ctx, latent)
}

#' Run the sampling loop (low-level)
#'
#' Runs the full denoising loop given pre-computed conditioning and an explicit
#' noise tensor. Noise is supplied by the caller for determinism; use
#' \code{seed} to generate it reproducibly, or pass \code{noise} directly.
#'
#' @param ctx SD context
#' @param cond Positive conditioning from \code{\link{sd_encode_text}}
#' @param uncond Negative conditioning from \code{\link{sd_encode_text}}.
#'   Pass an empty conditioning (all \code{NULL}) to disable CFG.
#' @param latent_shape Integer vector \code{c(W, H, C)} in latent space; used
#'   to generate noise when \code{noise} is not supplied. Ignored if
#'   \code{noise} is given.
#' @param init_latent Optional starting latent for img2img (from
#'   \code{\link{sd_encode_image}}); \code{NULL} for txt2img.
#' @param noise Optional explicit noise sd_tensor. When \code{NULL}, standard
#'   normal noise of \code{latent_shape} is generated using \code{seed}.
#' @param strength img2img denoising strength (ignored for txt2img)
#' @param sample_method Sampling method (name or \code{SAMPLE_METHOD} value)
#' @param scheduler Scheduler (name or \code{SCHEDULER} value)
#' @param sample_steps Number of steps
#' @param cfg_scale CFG scale
#' @param eta Eta for DDIM-like samplers
#' @param seed Seed for noise generation when \code{noise} is \code{NULL}
#' @param custom_sigmas Optional explicit sigma schedule (overrides scheduler)
#' @return An sd_tensor list — the denoised latent x_0. Pass to
#'   \code{\link{sd_decode_latent}}.
#' @export
#' @seealso \code{\link{sd_encode_text}}, \code{\link{sd_decode_latent}}
sd_sample <- function(ctx, cond, uncond = list(crossattn = NULL, vector = NULL, concat = NULL),
                      latent_shape = NULL, init_latent = NULL, noise = NULL,
                      strength = 1.0,
                      sample_method = SAMPLE_METHOD$EULER,
                      scheduler = SCHEDULER$DISCRETE,
                      sample_steps = 20L, cfg_scale = 7.0, eta = 0.0,
                      seed = 42L, custom_sigmas = NULL) {
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

  if (is.null(noise)) {
    if (is.null(latent_shape) || length(latent_shape) < 3L) {
      stop("Either `noise` or `latent_shape = c(W, H, C)` must be supplied.",
           call. = FALSE)
    }
    n <- prod(as.numeric(latent_shape[1:3]))
    set.seed(seed)
    noise <- list(
      type = SD_TYPE$F32,
      ne   = as.integer(c(latent_shape[1], latent_shape[2], latent_shape[3], 1L)),
      data = stats::rnorm(n)
    )
  }

  sd_sample_cpp(ctx, init_latent, noise, cond, uncond,
                as.integer(sample_method), as.integer(scheduler),
                as.integer(sample_steps), as.numeric(cfg_scale),
                as.numeric(eta), as.numeric(strength),
                if (is.null(custom_sigmas)) NULL else as.numeric(custom_sigmas))
}

# ===========================================================================
# Step-wise sampling (TODO 9.2 Stage 2)
# ===========================================================================
# Expose one denoise step so the sampler loop can live in R, giving per-step
# preview / interruption. Euler family only; for caches / tiled / control /
# multistep samplers use sd_sample() (the whole loop). See
# dev/9.2-stage2-design.md.

#' Sigma schedule for a sampler (low-level)
#'
#' Returns the sigma schedule that \code{\link{sd_sample_stepwise}} iterates
#' over, for a given scheduler / step count / generation size.
#'
#' @param ctx SD context from \code{\link{sd_ctx}}
#' @param scheduler Scheduler (name or \code{SCHEDULER} value)
#' @param sample_steps Number of steps
#' @param width,height Generation size in PIXELS (same as passed to generation)
#' @param sample_method Sampling method (name or \code{SAMPLE_METHOD} value);
#'   only used to pick a default scheduler when \code{scheduler} is a default.
#' @return Numeric vector of length \code{sample_steps + 1}; the last value is 0.
#' @export
#' @seealso \code{\link{sd_denoise_step}}, \code{\link{sd_sample_stepwise}}
sd_sampler_sigmas <- function(ctx, scheduler = SCHEDULER$DISCRETE,
                              sample_steps = 20L, width = 512L, height = 512L,
                              sample_method = SAMPLE_METHOD$EULER) {
  if (is.character(scheduler)) {
    sc <- SCHEDULER[[scheduler]]
    if (is.null(sc)) stop("Unknown scheduler: ", scheduler, call. = FALSE)
    scheduler <- sc
  }
  if (is.character(sample_method)) {
    sm <- SAMPLE_METHOD[[sample_method]]
    if (is.null(sm)) stop("Unknown sample_method: ", sample_method, call. = FALSE)
    sample_method <- sm
  }
  sd_sampler_sigmas_cpp(ctx, as.integer(scheduler), as.integer(sample_method),
                        as.integer(sample_steps), as.integer(width),
                        as.integer(height))
}

#' Run a single denoise step (low-level)
#'
#' Runs the diffusion model once on \code{x} at \code{sigma} and returns the
#' denoised x_0 estimate. The Euler update of \code{x} is done by the caller
#' (see \code{\link{sd_sample_stepwise}} for the full loop). Must be called
#' between \code{\link{sd_sampler_begin}} and \code{\link{sd_sampler_end}}.
#'
#' @param ctx SD context
#' @param x Current latent sd_tensor
#' @param sigma Current sigma (scalar)
#' @param cond Positive conditioning from \code{\link{sd_encode_text}}
#' @param uncond Negative conditioning; empty (all \code{NULL}) disables CFG
#' @param cfg_scale CFG scale (1 disables CFG)
#' @param step,total_steps 1-based step index / total, for progress hooks
#' @return An sd_tensor list — the denoised x_0 estimate.
#' @export
#' @seealso \code{\link{sd_sample_stepwise}}
sd_denoise_step <- function(ctx, x, sigma, cond,
                            uncond = list(crossattn = NULL, vector = NULL, concat = NULL),
                            cfg_scale = 7.0, step = 1L, total_steps = 1L) {
  sd_denoise_step_cpp(ctx, x, as.numeric(sigma), cond, uncond,
                      as.numeric(cfg_scale), as.integer(step),
                      as.integer(total_steps))
}

#' Open / close a step-wise sampling window (low-level)
#'
#' Between begin and end the diffusion model keeps its GPU compute buffer alive
#' across \code{\link{sd_denoise_step}} calls, avoiding a large realloc per
#' step. Must be paired; \code{sd_sampler_end} frees the buffer. Not reentrant.
#' \code{\link{sd_sample_stepwise}} manages this for you.
#'
#' @param ctx SD context
#' @return Invisibly \code{NULL}.
#' @export
#' @rdname sd_sampler_window
sd_sampler_begin <- function(ctx) {
  sd_sampler_begin_cpp(ctx)
  invisible(NULL)
}

#' @export
#' @rdname sd_sampler_window
sd_sampler_end <- function(ctx) {
  sd_sampler_end_cpp(ctx)
  invisible(NULL)
}

#' Scale noise into the starting latent (low-level)
#'
#' Applies the denoiser's noise scaling for the first sigma, producing the
#' starting \code{x} for the sampling loop. For txt2img pass \code{init_latent
#' = NULL}.
#'
#' @param ctx SD context
#' @param noise Noise sd_tensor (defines geometry)
#' @param sigma0 First sigma of the schedule
#' @param init_latent Optional starting latent (img2img); \code{NULL} for txt2img
#' @return An sd_tensor — the scaled starting latent.
#' @export
sd_noise_scale <- function(ctx, noise, sigma0, init_latent = NULL) {
  sd_noise_scale_cpp(ctx, init_latent, noise, as.numeric(sigma0))
}

#' Undo final-step latent scaling (low-level)
#'
#' Applies the denoiser's inverse noise scaling after the last step. A no-op for
#' discrete CompVis denoisers (SD1/SD2/SDXL).
#'
#' @param ctx SD context
#' @param x Latent sd_tensor after the last step
#' @param sigma_last Last sigma of the schedule (typically 0)
#' @return An sd_tensor.
#' @export
sd_inverse_noise_scale <- function(ctx, x, sigma_last) {
  sd_inverse_noise_scale_cpp(ctx, x, as.numeric(sigma_last))
}

#' Run the sampling loop step-by-step in R (low-level)
#'
#' Equivalent to \code{\link{sd_sample}} for the Euler / Euler-a samplers, but
#' runs the loop in R so a callback can observe or interrupt each step (e.g.
#' live preview). For Euler (no ancestral noise) the result is bit-for-bit equal
#' to \code{sd_sample}; Euler-a differs (R RNG vs ggml RNG for the ancestral
#' term). Other samplers are not supported here — use \code{\link{sd_sample}}.
#'
#' @param ctx SD context
#' @param cond Positive conditioning from \code{\link{sd_encode_text}}
#' @param uncond Negative conditioning; empty (all \code{NULL}) disables CFG
#' @param latent_shape Integer \code{c(W, H, C)} in latent space, used to make
#'   noise when \code{noise} is \code{NULL}
#' @param init_latent Optional starting latent (img2img); \code{NULL} for txt2img
#' @param noise Optional explicit noise sd_tensor; generated from \code{seed}
#'   and \code{latent_shape} when \code{NULL}
#' @param width,height Generation size in PIXELS (for the sigma schedule)
#' @param sample_method \code{SAMPLE_METHOD$EULER} or \code{$EULER_A}
#' @param scheduler Scheduler (name or \code{SCHEDULER} value)
#' @param sample_steps Number of steps
#' @param cfg_scale CFG scale
#' @param seed Seed for noise generation when \code{noise} is \code{NULL}
#' @param custom_sigmas Optional explicit sigma schedule (overrides scheduler)
#' @param on_step Optional callback \code{function(step, total, x, denoised)}
#'   called after each step; return \code{FALSE} to stop early.
#' @return An sd_tensor — the denoised latent x_0.
#' @export
#' @seealso \code{\link{sd_sample}}, \code{\link{sd_decode_latent}}
sd_sample_stepwise <- function(ctx, cond,
                               uncond = list(crossattn = NULL, vector = NULL, concat = NULL),
                               latent_shape = NULL, init_latent = NULL, noise = NULL,
                               width = 512L, height = 512L,
                               sample_method = SAMPLE_METHOD$EULER,
                               scheduler = SCHEDULER$DISCRETE,
                               sample_steps = 20L, cfg_scale = 7.0,
                               seed = 42L, custom_sigmas = NULL,
                               on_step = NULL) {
  if (is.character(sample_method)) {
    sm <- SAMPLE_METHOD[[sample_method]]
    if (is.null(sm)) stop("Unknown sample_method: ", sample_method, call. = FALSE)
    sample_method <- sm
  }
  if (!sample_method %in% c(SAMPLE_METHOD$EULER, SAMPLE_METHOD$EULER_A)) {
    stop("sd_sample_stepwise() supports only EULER / EULER_A. ",
         "Use sd_sample() for other samplers.", call. = FALSE)
  }
  if (is.character(scheduler)) {
    sc <- SCHEDULER[[scheduler]]
    if (is.null(sc)) stop("Unknown scheduler: ", scheduler, call. = FALSE)
    scheduler <- sc
  }

  if (is.null(noise)) {
    if (is.null(latent_shape) || length(latent_shape) < 3L) {
      stop("Either `noise` or `latent_shape = c(W, H, C)` must be supplied.",
           call. = FALSE)
    }
    n <- prod(as.numeric(latent_shape[1:3]))
    set.seed(seed)
    noise <- list(
      type = SD_TYPE$F32,
      ne   = as.integer(c(latent_shape[1], latent_shape[2], latent_shape[3], 1L)),
      data = stats::rnorm(n)
    )
  }

  sigmas <- if (is.null(custom_sigmas)) {
    sd_sampler_sigmas(ctx, scheduler, sample_steps, width, height, sample_method)
  } else {
    as.numeric(custom_sigmas)
  }
  n_steps <- length(sigmas) - 1L

  x <- sd_noise_scale(ctx, noise, sigmas[1], init_latent = init_latent)

  sd_sampler_begin(ctx)
  on.exit(sd_sampler_end(ctx), add = TRUE)

  for (i in seq_len(n_steps)) {
    sigma <- sigmas[i]
    den <- sd_denoise_step(ctx, x, sigma, cond, uncond, cfg_scale, i, n_steps)
    # d = (x - denoised) / sigma ; x = x + d * (sigma_{i+1} - sigma)
    d  <- (x$data - den$data) / sigma
    dt <- sigmas[i + 1L] - sigma
    x$data <- x$data + d * dt

    if (sample_method == SAMPLE_METHOD$EULER_A && sigmas[i + 1L] > 0) {
      # ancestral: split the next sigma into deterministic + noise parts and
      # add fresh noise. NOTE: not bit-exact with sd_sample (R RNG != ggml RNG).
      s_i  <- sigma
      s_i1 <- sigmas[i + 1L]
      sigma_up   <- min(s_i1, sqrt(s_i1^2 * (s_i^2 - s_i1^2) / s_i^2))
      x$data <- x$data + stats::rnorm(length(x$data)) * sigma_up
    }

    if (!is.null(on_step) && isFALSE(on_step(i, n_steps, x, den))) break
  }

  sd_inverse_noise_scale(ctx, x, sigmas[length(sigmas)])
}

# ===========================================================================
# Multi-reference generation (TODO 9.7)
# ===========================================================================

#' Does the loaded model support reference images?
#'
#' Reports whether the model in \code{ctx} consumes reference images (edit /
#' control / DiT families: Flux, Flux.2, SD3, Qwen-Image, Z-Image). Passing
#' refs to other models aborts inside ggml, so \code{\link{sd_generate_multiref}}
#' uses this to fail cleanly first.
#'
#' @param ctx SD context from \code{\link{sd_ctx}}
#' @return Logical scalar.
#' @export
sd_supports_ref_images <- function(ctx) {
  sd_ctx_supports_ref_cpp(ctx)
}

#' Generate an image conditioned on multiple reference images
#'
#' Runs generation with one or more reference images, as used by edit /
#' reference-conditioned models (e.g. Qwen-Image, FLUX control/edit variants).
#' The references are passed straight through to the underlying
#' \code{generate_image} C-API (\code{ref_images}); the active model decides how
#' to use them, so this only has effect on models that support reference
#' conditioning.
#'
#' @param ctx SD context from \code{\link{sd_ctx}}
#' @param prompt Text prompt
#' @param refs A list of sd_image lists (each with \code{width}, \code{height},
#'   \code{channel}, \code{data}), e.g. from \code{\link{sd_load_image}}.
#' @param negative_prompt Negative prompt (default "")
#' @param width,height Output size in pixels
#' @param auto_resize_ref_image If \code{TRUE} (default), references are resized
#'   to fit the model's expected reference size.
#' @param increase_ref_index If \code{TRUE}, reference latents get increasing
#'   positional indices (model-specific; default \code{FALSE}).
#' @param sample_method,scheduler Sampler / scheduler (name or enum value)
#' @param sample_steps,cfg_scale,seed,clip_skip,eta Standard sampling controls
#' @param batch_count Number of images (default 1)
#' @return List of sd_image lists.
#' @export
#' @seealso \code{\link{sd_generate}}, \code{\link{sd_encode_image}}
sd_generate_multiref <- function(ctx, prompt, refs,
                                 negative_prompt = "",
                                 width = 512L, height = 512L,
                                 auto_resize_ref_image = TRUE,
                                 increase_ref_index = FALSE,
                                 sample_method = SAMPLE_METHOD$EULER,
                                 scheduler = SCHEDULER$DISCRETE,
                                 sample_steps = 20L, cfg_scale = 7.0,
                                 seed = 42L, clip_skip = -1L, eta = 0.0,
                                 batch_count = 1L) {
  if (!is.list(refs) || length(refs) == 0L) {
    stop("`refs` must be a non-empty list of sd_image lists.", call. = FALSE)
  }
  # Accept a single image passed directly (not wrapped in a list).
  if (!is.null(refs$width) && !is.null(refs$data)) refs <- list(refs)

  # Guard: passing reference images to a model that does not consume them
  # aborts inside ggml and kills the R process. Validate up-front and fail
  # cleanly instead. (Edit / control / DiT families support refs.)
  if (!sd_ctx_supports_ref_cpp(ctx)) {
    stop("This model does not support reference images. ",
         "sd_generate_multiref() requires an edit/control or DiT model ",
         "(Flux, Flux.2, SD3, Qwen-Image, Z-Image). For plain ",
         "SD1/SD2/SDXL use sd_generate() / sd_img2img().", call. = FALSE)
  }

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
    ref_images = refs,
    auto_resize_ref_image = isTRUE(auto_resize_ref_image),
    increase_ref_index = isTRUE(increase_ref_index)
  )

  sd_generate_image(ctx, params)
}
