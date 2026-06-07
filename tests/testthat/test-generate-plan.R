# Tests for .sd_generate_plan() — the step planner that drives the Shiny GUI's
# async state machine. Pure R, no GPU/models: a fake ctx (list with attributes)
# selects the strategy deterministically via the manual `vram_gb` attribute.

# Helper: fake context. vram_gb is honoured by .select_strategy (manual override
# takes priority over Vulkan autodetect), so strategy is deterministic offline.
fake_ctx <- function(model_type = "sd1", vram_gb = 1e6,
                     vae_decode_only = TRUE) {
  ctx <- list()
  attr(ctx, "model_type") <- model_type
  attr(ctx, "vram_gb") <- vram_gb
  attr(ctx, "vae_decode_only") <- vae_decode_only
  ctx
}

# --- cfg auto-1.0 for guidance-distilled models (the VAE-crash root cause) ---

test_that("flux/flux2 force cfg_scale to 1.0 when left at the 7.0 default", {
  for (mt in c("flux", "flux2")) {
    plan <- sd2R:::.sd_generate_plan(fake_ctx(mt), "p", width = 512L, height = 512L,
                                     cfg_scale = 7.0)
    expect_equal(plan[[1]]$params$cfg_scale, 1.0)
  }
})

test_that("explicit non-default cfg_scale is preserved for flux", {
  plan <- sd2R:::.sd_generate_plan(fake_ctx("flux"), "p", width = 512L, height = 512L,
                                   cfg_scale = 3.5)
  expect_equal(plan[[1]]$params$cfg_scale, 3.5)
})

test_that("non-flux models keep cfg_scale = 7.0", {
  plan <- sd2R:::.sd_generate_plan(fake_ctx("sd1"), "p", width = 512L, height = 512L,
                                   cfg_scale = 7.0)
  expect_equal(plan[[1]]$params$cfg_scale, 7.0)
})

# --- direct strategy: single final gen step ---

test_that("direct strategy yields one final gen step", {
  plan <- sd2R:::.sd_generate_plan(fake_ctx("sd1", vram_gb = 1e6), "p",
                                   width = 512L, height = 512L)
  expect_length(plan, 1L)
  expect_equal(plan[[1]]$type, "gen")
  expect_true(plan[[1]]$final)
  expect_false(plan[[1]]$uses_init)
  expect_null(plan[[1]]$params$tiled_sampling)
})

# --- tiled strategy: single gen step with tiled sampling enabled ---

test_that("tiled strategy enables tiled_sampling in the gen step", {
  # Tiny VRAM + large res + decode-only ctx -> .select_strategy returns "tiled"
  plan <- sd2R:::.sd_generate_plan(fake_ctx("sd1", vram_gb = 0.5,
                                            vae_decode_only = TRUE),
                                   "p", width = 2048L, height = 2048L)
  expect_length(plan, 1L)
  expect_true(plan[[1]]$final)
  expect_true(isTRUE(plan[[1]]$params$tiled_sampling))
})

# --- highres_fix: base -> upscale -> refine ---

test_that("highres_fix expands into base, upscale and final refine steps", {
  # Tiny VRAM + large res + encoder available (vae_decode_only = FALSE) -> highres
  plan <- sd2R:::.sd_generate_plan(fake_ctx("sd1", vram_gb = 0.5,
                                            vae_decode_only = FALSE),
                                   "p", width = 2048L, height = 2048L)
  expect_length(plan, 3L)
  expect_equal(plan[[1]]$type, "gen")     # base
  expect_false(plan[[1]]$final)
  expect_equal(plan[[2]]$type, "upscale")
  expect_equal(plan[[2]]$width, 2048L)
  expect_equal(plan[[2]]$height, 2048L)
  expect_equal(plan[[3]]$type, "gen")     # refine
  expect_true(plan[[3]]$final)
  expect_true(plan[[3]]$uses_init)
})

test_that("highres base step runs at the model's native resolution", {
  plan <- sd2R:::.sd_generate_plan(fake_ctx("sd1", vram_gb = 0.5,
                                            vae_decode_only = FALSE),
                                   "p", width = 2048L, height = 2048L)
  native <- sd2R:::.native_tile_size("sd1")   # 512 for sd1
  expect_equal(plan[[1]]$width, native)
  expect_equal(plan[[1]]$height, native)
})

test_that("highres refine is tiled for UNets but direct for DiT (flux2)", {
  # sd1 (UNet) -> tiled refine
  plan_unet <- sd2R:::.sd_generate_plan(fake_ctx("sd1", vram_gb = 0.5,
                                                 vae_decode_only = FALSE),
                                        "p", width = 2048L, height = 2048L)
  expect_true(isTRUE(plan_unet[[3]]$params$tiled_sampling))

  # flux2 (DiT) -> direct refine (tiled sampling would garble patchified latents)
  plan_dit <- sd2R:::.sd_generate_plan(fake_ctx("flux2", vram_gb = 0.5,
                                                vae_decode_only = FALSE),
                                       "p", width = 2048L, height = 2048L)
  expect_null(plan_dit[[3]]$params$tiled_sampling)
})

# --- img2img: single gen step consuming an init image ---

test_that("img2img produces one final gen step that uses the init image", {
  init <- list(width = 64L, height = 64L, channel = 3L,
               data = as.raw(rep(0L, 64 * 64 * 3)))
  plan <- sd2R:::.sd_generate_plan(fake_ctx("sd1", vram_gb = 1e6), "p",
                                   width = 64L, height = 64L,
                                   init_image = init, strength = 0.6)
  expect_length(plan, 1L)
  expect_true(plan[[1]]$final)
  expect_true(plan[[1]]$uses_init)
  expect_equal(plan[[1]]$params$strength, 0.6)
})

# --- enum resolution from string names ---

test_that("string sample_method / scheduler are resolved to integer codes", {
  plan <- sd2R:::.sd_generate_plan(fake_ctx("sd1"), "p", width = 512L, height = 512L,
                                   sample_method = "EULER", scheduler = "DISCRETE")
  expect_equal(plan[[1]]$params$sample_method, sd2R::SAMPLE_METHOD[["EULER"]])
  expect_equal(plan[[1]]$params$scheduler, sd2R::SCHEDULER[["DISCRETE"]])
})
