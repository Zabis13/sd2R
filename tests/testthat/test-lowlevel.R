# Low-level pipeline steps (TODO 9.2). The end-to-end test needs a real model;
# it is skipped unless SD2R_TEST_MODEL points at an SD1.x checkpoint.

test_that("low-level step functions are exported", {
  expect_true(is.function(sd_encode_text))
  expect_true(is.function(sd_encode_image))
  expect_true(is.function(sd_sample))
  expect_true(is.function(sd_decode_latent))
})

test_that("sd_sample errors without noise or latent_shape", {
  # ctx is unused before the argument check, so a dummy is fine.
  expect_error(
    sd_sample(ctx = NULL, cond = list(), latent_shape = NULL, noise = NULL),
    "noise|latent_shape"
  )
})

test_that("sd_generate_multiref refuses empty refs", {
  expect_error(sd_generate_multiref(NULL, "p", refs = list()),
               "non-empty list")
})

test_that("multiref guard: non-ref model raises a clean R error (no abort)", {
  model <- Sys.getenv("SD2R_TEST_MODEL", "")
  skip_if(model == "" || !file.exists(model),
          "Set SD2R_TEST_MODEL to an SD1.x checkpoint to run this test")
  ctx <- sd_ctx(model, model_type = "sd1")
  expect_false(sd_supports_ref_images(ctx))
  ref <- list(width = 32L, height = 32L, channel = 3L,
              data = as.raw(rep(128L, 32 * 32 * 3)))
  expect_error(sd_generate_multiref(ctx, "a cat", refs = list(ref),
                                    width = 128, height = 128, sample_steps = 2L),
               "does not support reference")
})

test_that("step-wise sampling functions are exported", {
  expect_true(is.function(sd_sampler_sigmas))
  expect_true(is.function(sd_denoise_step))
  expect_true(is.function(sd_sampler_begin))
  expect_true(is.function(sd_sampler_end))
  expect_true(is.function(sd_noise_scale))
  expect_true(is.function(sd_inverse_noise_scale))
  expect_true(is.function(sd_sample_stepwise))
})

test_that("sd_sample_stepwise rejects unsupported samplers", {
  expect_error(
    sd_sample_stepwise(ctx = NULL, cond = list(),
                       sample_method = SAMPLE_METHOD$DPMPP_2M,
                       latent_shape = c(32L, 32L, 4L)),
    "EULER"
  )
})

test_that("sd_sample_stepwise errors without noise or latent_shape", {
  expect_error(
    sd_sample_stepwise(ctx = NULL, cond = list(),
                       latent_shape = NULL, noise = NULL),
    "noise|latent_shape"
  )
})

test_that("sd_sampler_sigmas returns a valid schedule on a real model", {
  model <- Sys.getenv("SD2R_TEST_MODEL", "")
  skip_if(model == "" || !file.exists(model),
          "Set SD2R_TEST_MODEL to an SD1.x checkpoint to run this test")
  ctx <- sd_ctx(model, model_type = "sd1")
  sig <- sd_sampler_sigmas(ctx, scheduler = SCHEDULER$DISCRETE,
                           sample_steps = 6L, width = 256, height = 256)
  expect_length(sig, 7L)                      # steps + 1
  expect_equal(sig[length(sig)], 0)           # trailing sigma is 0
  expect_true(all(diff(sig) <= 0))            # monotonically non-increasing
})

# The R-side Euler loop replays sd_sample's math exactly. Verified: a SINGLE
# step matches the monolith to ~3e-6 (the f32 limit), proving denoise_once is
# mathematically identical. Over multiple steps the latents diverge slightly
# because the diffusion model runs in a fresh per-step ggml context and the GPU
# f32 reduction order differs between graph builds (bit-exact multi-step is
# physically unachievable on Vulkan, not a bug). So we assert: 1 step strict,
# N steps a numeric closeness bound. See dev/9.2-stage2-design.md §7.

test_that("sd_sample_stepwise (Euler) single step matches sd_sample (f32-exact)", {
  model <- Sys.getenv("SD2R_TEST_MODEL", "")
  skip_if(model == "" || !file.exists(model),
          "Set SD2R_TEST_MODEL to an SD1.x checkpoint to run this test")
  ctx <- sd_ctx(model, model_type = "sd1", vae_decode_only = FALSE)

  cond   <- sd_encode_text(ctx, "a red apple on a table", width = 256, height = 256)
  uncond <- sd_encode_text(ctx, "", width = 256, height = 256)

  shape <- c(32L, 32L, 4L)
  set.seed(123L)
  noise <- list(type = SD_TYPE$F32, ne = as.integer(c(shape, 1L)),
                data = stats::rnorm(prod(shape)))

  # Drive both paths over the SAME single-step schedule (sigmas[1:2]) so only
  # one diffusion-model evaluation happens — isolates the per-step math.
  sig <- sd_sampler_sigmas(ctx, SCHEDULER$DISCRETE, 6L, 256, 256)
  one <- c(sig[1], sig[2])

  x_mono <- sd_sample(ctx, cond, uncond, noise = noise,
                      sample_method = SAMPLE_METHOD$EULER,
                      scheduler = SCHEDULER$DISCRETE,
                      custom_sigmas = one, cfg_scale = 7.0)
  x_step <- sd_sample_stepwise(ctx, cond, uncond, noise = noise,
                               width = 256, height = 256,
                               sample_method = SAMPLE_METHOD$EULER,
                               scheduler = SCHEDULER$DISCRETE,
                               custom_sigmas = one, cfg_scale = 7.0)

  expect_equal(x_step$ne, x_mono$ne)
  # Single step: identical math, only f32 reduction-order noise (~3e-6).
  expect_lt(max(abs(x_step$data - x_mono$data)), 1e-5)
})

test_that("sd_sample_stepwise (Euler) multi-step is numerically close to sd_sample", {
  model <- Sys.getenv("SD2R_TEST_MODEL", "")
  skip_if(model == "" || !file.exists(model),
          "Set SD2R_TEST_MODEL to an SD1.x checkpoint to run this test")
  ctx <- sd_ctx(model, model_type = "sd1", vae_decode_only = FALSE)

  cond   <- sd_encode_text(ctx, "a red apple on a table", width = 256, height = 256)
  uncond <- sd_encode_text(ctx, "", width = 256, height = 256)

  shape <- c(32L, 32L, 4L)
  set.seed(123L)
  noise <- list(type = SD_TYPE$F32, ne = as.integer(c(shape, 1L)),
                data = stats::rnorm(prod(shape)))

  x_mono <- sd_sample(ctx, cond, uncond, noise = noise,
                      sample_method = SAMPLE_METHOD$EULER,
                      scheduler = SCHEDULER$DISCRETE,
                      sample_steps = 20L, cfg_scale = 7.0)
  x_step <- sd_sample_stepwise(ctx, cond, uncond, noise = noise,
                               width = 256, height = 256,
                               sample_method = SAMPLE_METHOD$EULER,
                               scheduler = SCHEDULER$DISCRETE,
                               sample_steps = 20L, cfg_scale = 7.0)

  expect_equal(x_step$ne, x_mono$ne)
  # Multi-step: catch systematic divergence (a wrong formula tanks corr) while
  # tolerating GPU f32 accumulation across per-step contexts. Measured on
  # SD1.5/radv at 20 steps: latent corr ~0.9989, mean|diff|/255 ~0.0028. The
  # decoded images are visually identical.
  expect_gt(stats::cor(x_step$data, x_mono$data), 0.998)
  expect_lt(mean(abs(x_step$data - x_mono$data)) / 255, 0.005)
})

test_that("sd_sample_stepwise on_step fires per step and can interrupt", {
  model <- Sys.getenv("SD2R_TEST_MODEL", "")
  skip_if(model == "" || !file.exists(model),
          "Set SD2R_TEST_MODEL to an SD1.x checkpoint to run this test")
  ctx <- sd_ctx(model, model_type = "sd1", vae_decode_only = FALSE)
  cond   <- sd_encode_text(ctx, "a cat", width = 256, height = 256)
  uncond <- sd_encode_text(ctx, "", width = 256, height = 256)

  calls <- 0L
  sd_sample_stepwise(ctx, cond, uncond, latent_shape = c(32L, 32L, 4L),
                     width = 256, height = 256, sample_steps = 6L,
                     on_step = function(step, total, x, denoised) {
                       calls <<- calls + 1L
                       step < 3L  # FALSE at step 3 -> stop early
                     })
  expect_equal(calls, 3L)
})

test_that("low-level encode/sample/decode round-trip on a real model", {
  model <- Sys.getenv("SD2R_TEST_MODEL", "")
  skip_if(model == "" || !file.exists(model),
          "Set SD2R_TEST_MODEL to an SD1.x checkpoint to run this test")

  ctx <- sd_ctx(model, model_type = "sd1", vae_decode_only = FALSE)

  cond   <- sd_encode_text(ctx, "a red apple on a table", width = 256, height = 256)
  uncond <- sd_encode_text(ctx, "", width = 256, height = 256)
  expect_false(is.null(cond$crossattn))
  expect_equal(cond$crossattn$ne[1:2], c(768L, 77L))

  x0 <- sd_sample(ctx, cond, uncond, latent_shape = c(32L, 32L, 4L),
                  sample_steps = 6L, cfg_scale = 7.0, seed = 123L)
  expect_false(is.null(x0$data))
  expect_equal(x0$ne[1:3], c(32L, 32L, 4L))

  img <- sd_decode_latent(ctx, x0)
  expect_equal(c(img$width, img$height, img$channel), c(256L, 256L, 3L))
  # non-trivial output (not all one value)
  expect_gt(stats::sd(as.integer(img$data)), 1)
})

# --- sd_sample: mask / control_image checks (before C++) ---

test_that("sd_sample validates mask / control inputs up front", {
  local_mocked_bindings(
    sd_ctx_latent_info_cpp = function(ctx) c(8L, 4L),
    sd_ctx_is_inpaint_model_cpp = function(ctx) FALSE,
    sd_ctx_has_control_net_cpp = function(ctx) TRUE)
  noise <- list(type = 0L, ne = c(4L, 2L, 4L, 1L), data = rep(0, 32))
  init <- noise
  f <- function(...) sd2R:::.sample_extra_inputs("ctx", noise, init, ...)
  m_ok <- matrix(1, 16, 32)                  # [H, W] = 16 x 32 pixels
  expect_equal(f(m_ok, NULL, 0.9)$mask$width, 32L)
  expect_error(sd2R:::.sample_extra_inputs("ctx", noise, NULL, m_ok, NULL, 0.9),
               "'mask' needs 'init_latent'")
  expect_error(f(matrix(1, 8, 8), NULL, 0.9),
               "mask is 8x8, the generation is 32x16; resize it first")
  gray <- sd2R:::.array_to_sd_image(array(0.5, c(16, 32, 1)))
  ci <- f(NULL, gray, 0.9)$control
  expect_equal(ci$channel, 3L)               # grayscale expanded to RGB
  expect_error(f(NULL, gray, 3), "'control_strength' must be a number in \\[0, 2\\]")
  expect_error(f(NULL, sd2R:::.array_to_sd_image(array(0, c(8, 8, 3))), 0.9),
               "control image is 8x8")
  expect_equal(f(NULL, NULL, 99), list(mask = NULL, control = NULL))
})

test_that("sd_sample refuses inpaint checkpoints and missing ControlNet", {
  local_mocked_bindings(
    sd_ctx_latent_info_cpp = function(ctx) c(8L, 4L),
    sd_ctx_is_inpaint_model_cpp = function(ctx) TRUE,
    sd_ctx_has_control_net_cpp = function(ctx) FALSE)
  noise <- list(type = 0L, ne = c(4L, 2L, 4L, 1L), data = rep(0, 32))
  f <- function(...) sd2R:::.sample_extra_inputs("ctx", noise, noise, ...)
  expect_error(f(matrix(1, 16, 32), NULL, 0.9),
               "inpainting checkpoints are not supported.*sd_img2img\\(mask = \\)")
  expect_error(f(NULL, sd2R:::.array_to_sd_image(array(0, c(16, 32, 3))), 0.9),
               "the context has no ControlNet")
})

test_that("sd_sample passes mask / control to C++ and errors on NULL result", {
  got <- NULL
  local_mocked_bindings(
    sd_ctx_latent_info_cpp = function(ctx) c(8L, 4L),
    sd_ctx_is_inpaint_model_cpp = function(ctx) FALSE,
    sd_ctx_has_control_net_cpp = function(ctx) TRUE,
    sd_sample_cpp = function(...) {
      a <- list(...); got <<- a[13:15]; NULL
    })
  noise <- list(type = 0L, ne = c(4L, 2L, 4L, 1L), data = rep(0, 32))
  ctrl <- sd2R:::.array_to_sd_image(array(0, c(16, 32, 3)))
  expect_error(sd_sample("ctx", cond = list(), noise = noise, init_latent = noise,
                         mask = matrix(1, 16, 32), control_image = ctrl,
                         control_strength = 0.4),
               "sd_sample: sampling failed")
  expect_equal(got[[1]]$channel, 1L)
  expect_identical(got[[2]], ctrl)
  expect_equal(got[[3]], 0.4)
})

# --- sd_sample batch ---

test_that("batch noise: element 1 is the batch = 1 noise, seeds per element", {
  n1 <- sd2R:::.batch_noise(c(4L, 2L, 3L), 42L, 1L)
  set.seed(42L)
  expect_equal(n1$data, stats::rnorm(24))                  # unchanged
  expect_equal(n1$ne, c(4L, 2L, 3L, 1L))
  n3 <- sd2R:::.batch_noise(c(4L, 2L, 3L), 42L, 3L)
  expect_equal(n3$ne, c(4L, 2L, 3L, 3L))
  expect_equal(sd2R:::.tensor_element(n3, 1L)$data, n1$data)
  set.seed(44L)
  expect_equal(sd2R:::.tensor_element(n3, 3L)$data, stats::rnorm(24))
  expect_equal(sd2R:::.tensor_bind(lapply(1:3, function(k)
    sd2R:::.tensor_element(n3, k))), n3)
})

test_that("sd_sample batch: one call on batch models, loop otherwise", {
  calls <- list()
  local_mocked_bindings(
    sd_ctx_latent_info_cpp = function(ctx) c(8L, 4L),
    sd_sample_cpp = function(ctx, init, noise, ...) {
      calls[[length(calls) + 1L]] <<- list(init = init, noise = noise)
      noise                                                # x0 := noise
    })
  cond <- list(crossattn = NULL, vector = NULL, concat = NULL)

  local_mocked_bindings(sd_ctx_supports_batch_cpp = function(ctx) TRUE)
  b <- sd_sample("ctx", cond, latent_shape = c(4L, 2L, 3L), seed = 7L, batch = 2L)
  expect_equal(length(calls), 1L)                          # one call
  expect_equal(b$ne, c(4L, 2L, 3L, 2L))

  calls <- list()
  local_mocked_bindings(sd_ctx_supports_batch_cpp = function(ctx) FALSE)
  l <- sd_sample("ctx", cond, latent_shape = c(4L, 2L, 3L), seed = 7L, batch = 2L)
  expect_equal(length(calls), 2L)                          # one per element
  expect_true(all(vapply(calls, function(x) x$noise$ne[4] == 1L, logical(1L))))
  expect_identical(l, b)                                   # same result format

  # batch = 1: exactly the old single call
  calls <- list()
  one <- sd_sample("ctx", cond, latent_shape = c(4L, 2L, 3L), seed = 7L)
  expect_equal(length(calls), 1L)
  expect_equal(one$data, sd2R:::.tensor_element(b, 1L)$data)
})

test_that("sd_sample batch: init_latent 1 or N, explicit noise, checks", {
  calls <- list()
  local_mocked_bindings(
    sd_ctx_latent_info_cpp = function(ctx) c(8L, 4L),
    sd_ctx_supports_batch_cpp = function(ctx) TRUE,
    sd_sample_cpp = function(ctx, init, noise, ...) {
      calls[[length(calls) + 1L]] <<- list(init = init, noise = noise); noise
    })
  cond <- list(crossattn = NULL, vector = NULL, concat = NULL)
  init1 <- list(type = 0L, ne = c(4L, 2L, 3L, 1L), data = rep(1, 24))
  sd_sample("ctx", cond, latent_shape = c(4L, 2L, 3L), init_latent = init1,
            batch = 3L, strength = 0.5)
  expect_equal(calls[[1]]$init$ne, c(4L, 2L, 3L, 3L))      # repeated to N
  init2 <- list(type = 0L, ne = c(4L, 2L, 3L, 2L), data = rep(1, 48))
  expect_error(sd_sample("ctx", cond, latent_shape = c(4L, 2L, 3L),
                         init_latent = init2, batch = 3L),
               "'init_latent' has 2 elements, need 1 or 3")
  nz <- sd2R:::.batch_noise(c(4L, 2L, 3L), 1L, 2L)
  out <- sd_sample("ctx", cond, noise = nz)
  expect_equal(out$ne[4], 2L)                              # batch from noise
  expect_error(sd_sample("ctx", cond, noise = nz, batch = 3L),
               "'batch' = 3 but 'noise' has 2 element")
  expect_error(sd_sample("ctx", cond, latent_shape = c(4L, 2L, 3L), batch = 0),
               "'batch' must be a positive integer")
})

test_that("sd_sample batch with mask / control always loops", {
  calls <- 0L
  local_mocked_bindings(
    sd_ctx_latent_info_cpp = function(ctx) c(8L, 4L),
    sd_ctx_supports_batch_cpp = function(ctx) TRUE,
    sd_ctx_is_inpaint_model_cpp = function(ctx) FALSE,
    sd_sample_cpp = function(ctx, init, noise, ...) { calls <<- calls + 1L; noise })
  cond <- list(crossattn = NULL, vector = NULL, concat = NULL)
  init1 <- list(type = 0L, ne = c(4L, 2L, 3L, 1L), data = rep(1, 24))
  sd_sample("ctx", cond, latent_shape = c(4L, 2L, 3L), init_latent = init1,
            mask = matrix(1, 16, 32), batch = 2L)
  expect_equal(calls, 2L)
})
