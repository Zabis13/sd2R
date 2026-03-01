test_that("package loads", {
  expect_true(requireNamespace("sdR", quietly = TRUE))
})

test_that("sd_system_info works", {
  info <- sd_system_info()
  expect_s3_class(info, "sd_system_info")
  expect_true(nchar(info$sdR_version) > 0)
  expect_true(info$num_cores > 0)
})

test_that("constants are defined and 0-based", {
  expect_equal(SAMPLE_METHOD$EULER, 0L)
  expect_equal(SCHEDULER$DISCRETE, 0L)
  expect_equal(PREDICTION$EPS, 0L)
  expect_equal(SD_TYPE$F32, 0L)
  expect_equal(RNG_TYPE$STD_DEFAULT, 0L)
  expect_equal(LORA_APPLY_MODE$AUTO, 0L)
  expect_equal(SD_CACHE_MODE$DISABLED, 0L)
})

test_that("constants have correct count", {
  expect_equal(length(SAMPLE_METHOD), 12)
  expect_equal(length(SCHEDULER), 10)
  expect_equal(length(PREDICTION), 6)
  expect_equal(length(RNG_TYPE), 3)
})

test_that("C++ utility functions work", {
  expect_true(is.character(sd_version_cpp()))
  expect_true(is.character(sd_system_info_cpp()))
  expect_true(is.integer(sd_num_physical_cores_cpp()))
})

test_that("type name functions work", {
  expect_equal(sd_type_name_cpp(SD_TYPE$F32), "f32")
  expect_equal(sd_type_name_cpp(SD_TYPE$F16), "f16")
  expect_equal(sd_type_name_cpp(SD_TYPE$Q4_0), "q4_0")
})

test_that("sample method name functions work", {
  expect_equal(sd_sample_method_name_cpp(SAMPLE_METHOD$EULER), "euler")
})

test_that("scheduler name functions work", {
  expect_equal(sd_scheduler_name_cpp(SCHEDULER$DISCRETE), "discrete")
  expect_equal(sd_scheduler_name_cpp(SCHEDULER$KARRAS), "karras")
})

test_that("sd_ctx rejects missing model", {
  expect_error(sd_ctx("/nonexistent/model.safetensors"), "not found")
})

test_that("sd_txt2img accepts vae_tiling parameters", {
  # Verify that the function signature accepts tiling params without error
  args <- formals(sd_txt2img)
  expect_true("vae_tiling" %in% names(args))
  expect_true("vae_tile_size" %in% names(args))
  expect_true("vae_tile_overlap" %in% names(args))
  expect_false(eval(args$vae_tiling))
  expect_equal(eval(args$vae_tile_size), 64L)
  expect_equal(eval(args$vae_tile_overlap), 0.25)
})

test_that("sd_img2img accepts vae_tiling parameters", {
  args <- formals(sd_img2img)
  expect_true("vae_tiling" %in% names(args))
  expect_true("vae_tile_size" %in% names(args))
  expect_true("vae_tile_overlap" %in% names(args))
  expect_false(eval(args$vae_tiling))
  expect_equal(eval(args$vae_tile_size), 64L)
  expect_equal(eval(args$vae_tile_overlap), 0.25)
})

test_that("sd_load_image and sd_image_to_array roundtrip", {
  skip_if_not_installed("png")
  # Create a small test image
  arr <- array(runif(8 * 8 * 3), dim = c(8, 8, 3))
  tmp <- tempfile(fileext = ".png")
  on.exit(unlink(tmp))
  png::writePNG(arr, tmp)

  img <- sd_load_image(tmp)
  expect_equal(img$width, 8L)
  expect_equal(img$height, 8L)
  expect_equal(img$channel, 3L)
  expect_equal(length(img$data), 8 * 8 * 3)
  expect_true(is.raw(img$data))

  # Convert back to array
  arr2 <- sd_image_to_array(img)
  expect_equal(dim(arr2), c(8, 8, 3))
  # Values should be close (uint8 quantization)
  expect_true(max(abs(arr2 - arr)) < 1/255 + 0.01)
})

test_that("sd_save_image works", {
  skip_if_not_installed("png")
  img <- list(
    width = 4L, height = 4L, channel = 3L,
    data = as.raw(sample(0:255, 48, replace = TRUE))
  )
  tmp <- tempfile(fileext = ".png")
  on.exit(unlink(tmp))

  sd_save_image(img, tmp)
  expect_true(file.exists(tmp))
  expect_true(file.size(tmp) > 0)
})
