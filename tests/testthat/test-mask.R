# Tests for inpainting mask conversion (no GPU required).
# Mask semantics: white (255) = generate, black (0) = keep original.
# Engine expects a 1-channel, row-major [y][x] uint8 buffer.

test_that("matrix mask 0..1 is scaled to 0..255 and row-major", {
  # H = 3 rows, W = 4 cols
  m <- matrix(c(0, 0, 1, 1,
                0, 1, 1, 0,
                0, 0, 0, 1), nrow = 3, byrow = TRUE)
  img <- sd2R:::.sd_to_mask_image(m)

  expect_equal(img$width, 4L)
  expect_equal(img$height, 3L)
  expect_equal(img$channel, 1L)
  expect_equal(length(img$data), 12L)
  # First row of the mask must occupy the first W bytes (row-major).
  expect_equal(as.integer(img$data[1:4]), c(0L, 0L, 255L, 255L))
})

test_that("matrix mask already in 0..255 is passed through unscaled", {
  m <- matrix(c(0, 128, 255, 255), nrow = 1)
  img <- sd2R:::.sd_to_mask_image(m)
  expect_equal(as.integer(img$data), c(0L, 128L, 255L, 255L))
})

test_that("PNG mask path loads as a single channel", {
  skip_if_not_installed("png")
  tmp <- tempfile(fileext = ".png")
  on.exit(unlink(tmp), add = TRUE)

  m <- matrix(c(0, 0, 1, 1,
                0, 1, 1, 0,
                0, 0, 0, 1), nrow = 3, byrow = TRUE)
  png::writePNG(m, tmp)

  img <- sd2R:::sd_load_mask(tmp)
  expect_equal(img$channel, 1L)
  expect_equal(img$width, 4L)
  expect_equal(img$height, 3L)
  expect_equal(as.integer(img$data[1:4]), c(0L, 0L, 255L, 255L))
})

test_that("RGB PNG mask is averaged to one channel", {
  skip_if_not_installed("png")
  tmp <- tempfile(fileext = ".png")
  on.exit(unlink(tmp), add = TRUE)

  rgb <- array(0, dim = c(2, 2, 3))
  rgb[1, 1, ] <- 1   # white pixel
  rgb[2, 2, ] <- 1
  png::writePNG(rgb, tmp)

  img <- sd2R:::sd_load_mask(tmp)
  expect_equal(img$channel, 1L)
  expect_equal(length(img$data), 4L)
})

test_that(".sd_to_mask_image reduces a multi-channel SD image to one channel", {
  # A 2x2 RGB SD image list (row-major interleaved): white, black, black, white
  white <- as.raw(c(255, 255, 255))
  black <- as.raw(c(0, 0, 0))
  sd_img <- list(width = 2L, height = 2L, channel = 3L,
                 data = c(white, black, black, white))
  img <- sd2R:::.sd_to_mask_image(sd_img)
  expect_equal(img$channel, 1L)
  expect_equal(length(img$data), 4L)
  expect_equal(as.integer(img$data), c(255L, 0L, 0L, 255L))
})

test_that(".sd_to_mask_image keeps a 1-channel SD image untouched", {
  sd_img <- list(width = 2L, height = 1L, channel = 1L,
                 data = as.raw(c(0, 255)))
  img <- sd2R:::.sd_to_mask_image(sd_img)
  expect_identical(img, sd_img)
})

test_that(".sd_to_mask_image rejects unsupported input", {
  expect_error(sd2R:::.sd_to_mask_image(42), "PNG path")
})
