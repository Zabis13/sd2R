# Tests for background removal (R/rmbg.R) — inference mocked, no model

fake_rmbg <- function(size = 8L) {
  structure(list(model = "fake", input = "input", size = size,
                 path = "rmbg.onnx"), class = "sd_rmbg")
}

# 8x8 image: top half red, bottom half black
top_red_image <- function(h = 8L, w = 8L) {
  arr <- array(0, c(h, w, 3L))
  arr[seq_len(h %/% 2L), , 1] <- 1
  sd2R:::.array_to_sd_image(arr)
}

# mock: mask = red channel of the (row-major NCHW) input
mock_red_mask <- function(env = parent.frame()) {
  seen <- new.env()
  local_mocked_bindings(
    .onnx_run = function(model, inputs) {
      seen$names <- names(inputs)
      seen$x <- inputs[[1]]
      n <- length(inputs[[1]]) / 3
      list(output = array(inputs[[1]][seq_len(n)] + 0.5,
                          c(sqrt(n), sqrt(n), 1, 1)))
    },
    .env = env
  )
  seen
}

test_that("input is NCHW row-major, normalized to [-0.5, 0.5]", {
  seen <- mock_red_mask()
  sd_remove_background(top_red_image(), fake_rmbg())
  expect_equal(seen$names, "input")
  expect_length(seen$x, 8 * 8 * 3)
  expect_true(all(seen$x >= -0.5 & seen$x <= 0.5))
  # first plane = R; row-major: first 8 values are row 1 (red), last 8 row 8
  expect_true(all(seen$x[1:8] > 0.4))
  expect_true(all(seen$x[57:64] < -0.4))
})

test_that("mask keeps image orientation (rows vs columns)", {
  mock_red_mask()
  m <- sd_image_to_array(sd_remove_background(top_red_image(), fake_rmbg(),
                                              output = "mask"))
  expect_equal(dim(m), c(8L, 8L, 1L))
  expect_true(all(m[1, , 1] > 0.9))     # top row = foreground
  expect_true(all(m[8, , 1] < 0.1))     # bottom row = background
})

test_that("rgba output keeps RGB and adds the mask as alpha", {
  mock_red_mask()
  img <- top_red_image()
  out <- sd_remove_background(img, fake_rmbg())
  expect_equal(out$channel, 4L)
  expect_equal(c(out$width, out$height), c(8L, 8L))
  a <- sd_image_to_array(out)
  expect_equal(a[, , 1:3], sd_image_to_array(img), tolerance = 1 / 255)
  expect_true(all(a[1, , 4] > 0.9))
  expect_true(all(a[8, , 4] < 0.1))
})

test_that("mask is resized back to a non-square original", {
  mock_red_mask()
  out <- sd_remove_background(top_red_image(h = 6L, w = 10L), fake_rmbg(),
                              output = "mask")
  expect_equal(c(out$width, out$height, out$channel), c(10L, 6L, 1L))
})

test_that("threshold binarizes the mask", {
  mock_red_mask()
  m <- sd_image_to_array(sd_remove_background(top_red_image(), fake_rmbg(),
                                              output = "mask", threshold = 0.5))
  expect_setequal(unique(as.vector(m)), c(0, 1))
  expect_error(sd_remove_background(top_red_image(), fake_rmbg(), threshold = 1),
               "must be a number in \\(0, 1\\)")
})

test_that("accepts a PNG path and rejects a bad model", {
  mock_red_mask()
  tmp <- tempfile(fileext = ".png")
  on.exit(unlink(tmp))
  sd_save_image(top_red_image(), tmp)
  out <- sd_remove_background(tmp, fake_rmbg(), output = "mask")
  expect_equal(out$channel, 1L)
  expect_error(sd_remove_background(top_red_image(), list()),
               "must be an sd_rmbg object")
})

test_that("unexpected output size is reported", {
  local_mocked_bindings(.onnx_run = function(model, inputs) list(1:10))
  expect_error(sd_remove_background(top_red_image(), fake_rmbg()),
               "RMBG output has 10 values, expected 64")
})

test_that("remove_background graph node runs and is validated", {
  mock_red_mask()
  local_mocked_bindings(.rmbg_cached = function(path, device = NULL) fake_rmbg())
  tmp <- tempfile(fileext = ".png")
  on.exit(unlink(tmp))
  sd_save_image(top_red_image(), tmp)
  pipe <- sd_pipeline(sd_node("load_image", path = tmp),
                      sd_node("remove_background", model_path = "rmbg.onnx"))
  out <- sd_run_pipeline(pipe, ctx = NULL)
  expect_equal(out$channel, 4L)

  bad <- sd_pipeline(sd_node("remove_background"))
  err <- tryCatch(sd_run_pipeline(bad, ctx = NULL), error = conditionMessage)
  expect_match(err, "node 1 \\(remove_background\\): 'model_path' is not set")
  expect_match(err, "node 1 \\(remove_background\\): needs 'image'")
})
