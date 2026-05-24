# Tests for VAE tiling mode resolution and parameters

test_that(".resolve_vae_tiling handles 'normal' mode", {
  expect_false(sd2R:::.resolve_vae_tiling("normal", NULL, 512, 512, 1048576L))
  expect_false(sd2R:::.resolve_vae_tiling("normal", NULL, 4096, 4096, 1048576L))
})

test_that(".resolve_vae_tiling handles 'tiled' mode", {
  expect_true(sd2R:::.resolve_vae_tiling("tiled", NULL, 512, 512, 1048576L))
  expect_true(sd2R:::.resolve_vae_tiling("tiled", NULL, 64, 64, 1048576L))
})

test_that(".resolve_vae_tiling handles 'auto' mode with area threshold", {
  threshold <- 1048576L  # 1024*1024

  # Below threshold: 512*512 = 262144 < 1048576
  expect_false(sd2R:::.resolve_vae_tiling("auto", NULL, 512, 512, threshold))

  # At threshold: 1024*1024 = 1048576, equals threshold -> tile
  expect_true(sd2R:::.resolve_vae_tiling("auto", NULL, 1024, 1024, threshold))

  # Above threshold: 1025*1024 = 1049600 > 1048576
  expect_true(sd2R:::.resolve_vae_tiling("auto", NULL, 1025, 1024, threshold))

  # Large: 2048*2048 = 4194304 > 1048576
  expect_true(sd2R:::.resolve_vae_tiling("auto", NULL, 2048, 2048, threshold))

  # 4096*4096 = 16777216 > 1048576
  expect_true(sd2R:::.resolve_vae_tiling("auto", NULL, 4096, 4096, threshold))

  # Non-square landscape: 2048*512 = 1048576, equals threshold -> tile
  expect_true(sd2R:::.resolve_vae_tiling("auto", NULL, 2048, 512, threshold))

  # Non-square landscape just over: 2048*513 = 1050624 > 1048576
  expect_true(sd2R:::.resolve_vae_tiling("auto", NULL, 2048, 513, threshold))
})

test_that(".resolve_vae_tiling custom threshold works", {
  # Low threshold for small VRAM
  expect_true(sd2R:::.resolve_vae_tiling("auto", NULL, 512, 512, 100000L))

  # High threshold for big VRAM
  expect_false(sd2R:::.resolve_vae_tiling("auto", NULL, 2048, 2048, 20000000L))
})

test_that("deprecated vae_tiling=TRUE triggers warning and enables tiling", {
  expect_warning(
    result <- sd2R:::.resolve_vae_tiling("auto", TRUE, 64, 64, 1048576L),
    "deprecated"
  )
  expect_true(result)
})

test_that("deprecated vae_tiling=FALSE triggers warning and disables tiling", {
  expect_warning(
    result <- sd2R:::.resolve_vae_tiling("auto", FALSE, 4096, 4096, 1048576L),
    "deprecated"
  )
  expect_false(result)
})

test_that("deprecated vae_tiling overrides vae_mode", {
  # Even with vae_mode="tiled", deprecated vae_tiling=FALSE wins
  expect_warning(
    result <- sd2R:::.resolve_vae_tiling("tiled", FALSE, 4096, 4096, 1048576L),
    "deprecated"
  )
  expect_false(result)
})

test_that(".resolve_vae_tiling rejects invalid vae_mode", {
  expect_error(
    sd2R:::.resolve_vae_tiling("invalid", NULL, 512, 512, 1048576L)
  )
})

test_that("vae_mode='auto' activates for 2048x2048 decode scenario", {
  # Simulates the production scenario: decode 2048x2048 image
  # Area = 4194304, well above default threshold 1048576
  expect_true(sd2R:::.resolve_vae_tiling("auto", NULL, 2048, 2048, 1048576L))
})

test_that("vae_mode='auto' activates for 4096x4096 decode scenario", {
  # Simulates the production scenario: decode 4096x4096 image
  # Area = 16777216, well above default threshold
  expect_true(sd2R:::.resolve_vae_tiling("auto", NULL, 4096, 4096, 1048576L))
})

# --- .estimate_vae_vram tests ---

test_that(".estimate_vae_vram returns correct values for sd1", {
  # per_pixel = 128 ch * 4 B * 18 live_tensors = 9216
  # 512x512 * 9216 * 1 = 2415919104 (~2.25 GB)
  expect_equal(sd2R:::.estimate_vae_vram(512, 512, "sd1", 1L), 512 * 512 * 9216)
})

test_that(".estimate_vae_vram returns correct values for flux", {
  # per_pixel = 128 ch * 4 B * 18 live_tensors = 9216
  # 1024x1024 * 9216 * 1 = 9663676416 (~9 GB)
  expect_equal(sd2R:::.estimate_vae_vram(1024, 1024, "flux", 1L), 1024 * 1024 * 9216)
})

test_that(".estimate_vae_vram scales with batch", {
  single <- sd2R:::.estimate_vae_vram(512, 512, "sd1", 1L)
  double <- sd2R:::.estimate_vae_vram(512, 512, "sd1", 2L)
  expect_equal(double, single * 2)
})

test_that(".estimate_vae_vram sdxl uses same factor as flux", {
  expect_equal(
    sd2R:::.estimate_vae_vram(512, 512, "sdxl", 1L),
    sd2R:::.estimate_vae_vram(512, 512, "flux", 1L)
  )
})

# --- VRAM-aware auto mode tests ---

test_that("auto mode with ctx uses VRAM when available", {
  skip_if_not(sd2R::sd_vulkan_device_count() > 0, "No Vulkan device")

  # Create a fake ctx with attributes
  ctx <- list()
  attr(ctx, "vram_device") <- 0L
  attr(ctx, "model_type") <- "sd1"

  # Small image with plenty of VRAM should not tile
  # 64x64 * 2048 = ~8 MB — any GPU has more than that
  result <- sd2R:::.resolve_vae_tiling("auto", NULL, 64, 64, 1048576L, ctx = ctx)
  expect_false(result)
})

test_that("auto mode falls back to threshold when ctx is NULL", {
  # Without ctx, uses pixel-area threshold
  expect_true(sd2R:::.resolve_vae_tiling("auto", NULL, 2048, 2048, 1048576L))
  expect_false(sd2R:::.resolve_vae_tiling("auto", NULL, 512, 512, 1048576L))
})
