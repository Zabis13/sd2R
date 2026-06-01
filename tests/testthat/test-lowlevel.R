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
