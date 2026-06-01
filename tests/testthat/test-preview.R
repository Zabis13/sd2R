# Live preview (TODO 9.4). The PPM parser is testable without a model; the
# end-to-end generation test needs SD2R_TEST_MODEL.

test_that("preview functions and PREVIEW constant are exported", {
  expect_true(is.function(sd_preview_start))
  expect_true(is.function(sd_preview_stop))
  expect_true(is.function(sd_read_preview))
  expect_identical(PREVIEW$PROJ, "proj")
})

test_that("sd_read_preview returns NULL for a missing file", {
  expect_null(sd_read_preview(tempfile(fileext = ".ppm")))
})

test_that(".read_ppm_p6 parses a valid P6 file with binary body", {
  # Build a 2x1 RGB PPM by hand: header + 6 pixel bytes, including a NUL.
  f <- tempfile(fileext = ".ppm")
  con <- file(f, "wb")
  writeBin(charToRaw("P6\n2 1\n255\n"), con)
  writeBin(as.raw(c(0x00, 0x10, 0x20, 0xff, 0x80, 0x00)), con)  # embedded NUL
  close(con)
  img <- sd2R:::.read_ppm_p6(f)
  expect_false(is.null(img))
  expect_equal(c(img$width, img$height, img$channel), c(2L, 1L, 3L))
  expect_equal(length(img$data), 6L)
  expect_equal(as.integer(img$data), c(0, 16, 32, 255, 128, 0))
})

test_that(".read_ppm_p6 returns NULL on a truncated body", {
  f <- tempfile(fileext = ".ppm")
  con <- file(f, "wb")
  writeBin(charToRaw("P6\n4 4\n255\n"), con)
  writeBin(as.raw(rep(0L, 10)), con)  # needs 48 bytes, only 10
  close(con)
  expect_null(sd2R:::.read_ppm_p6(f))
})

test_that("generation with preview writes a readable frame", {
  model <- Sys.getenv("SD2R_TEST_MODEL", "")
  skip_if(model == "" || !file.exists(model),
          "Set SD2R_TEST_MODEL to an SD1.x checkpoint to run this test")
  ctx <- sd_ctx(model, model_type = "sd1")
  pp <- tempfile("prev_", fileext = ".ppm")
  imgs <- sd_generate(ctx, "a red apple", width = 256, height = 256,
                      sample_steps = 6L, seed = 7L,
                      preview = TRUE, preview_path = pp, preview_mode = "proj")
  expect_equal(imgs[[1]]$width, 256L)
  pv <- sd_read_preview(pp)
  expect_false(is.null(pv))
  expect_equal(pv$channel, 3L)
  expect_false(file.exists(paste0(pp, ".tmp")))  # cleaned by on.exit
})
