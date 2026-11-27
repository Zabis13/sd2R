# ControlNet preprocessors (canny, depth) and the control inputs

fake_img <- function(h = 4L, w = 6L, ch = 3L) {
  sd2R:::.array_to_sd_image(array(seq(0, 1, length.out = h * w * ch), c(h, w, ch)))
}

test_that("sd_canny checks arguments and calls the core", {
  got <- NULL
  local_mocked_bindings(sd_preprocess_canny_cpp = function(image, hi, lo, weak,
                                                           strong, inverse) {
    got <<- list(hi, lo, weak, strong, inverse); image
  })
  img <- fake_img()
  expect_identical(sd_canny(img), img)
  expect_equal(got, list(0.08, 0.08, 0.8, 1, FALSE))
  sd_canny(img, high_threshold = 0.2, low_threshold = 0.1, inverse = TRUE)
  expect_equal(got[c(1, 2, 5)], list(0.2, 0.1, TRUE))
  expect_error(sd_canny(img, high_threshold = 2),
               "'high_threshold' must be a number in \\[0, 1\\]")
  expect_error(sd_canny(img, weak = -1), "'weak' must be")
})

test_that("sd_estimate_depth: normalization, layout, near = white, size", {
  seen <- NULL
  size <- 4L
  local_mocked_bindings(.onnx_run = function(model, inputs) {
    seen <<- inputs[[1]]
    # output storage [W, H]: value grows with x (column) -> right side nearer
    list(rep(seq_len(size), times = size))
  })
  model <- structure(list(model = NULL, input = "pixel_values", size = size,
                          mean = c(0.5, 0.5, 0.5), std = c(0.5, 0.5, 0.5),
                          path = "d.onnx"), class = "sd_depth")
  img <- sd2R:::.array_to_sd_image(array(1, c(4L, 4L, 3L)))   # white
  d <- sd_estimate_depth(img, model)
  expect_equal(c(d$width, d$height, d$channel), c(4L, 4L, 3L))
  expect_true(all(seen == 1))                                 # (1 - 0.5) / 0.5
  a <- sd_image_to_array(d)
  expect_equal(a[1, , 1], c(0, 1/3, 2/3, 1), tolerance = 1/255)  # left far, right near
  expect_equal(a[, , 1], a[, , 3])
  expect_error(sd_estimate_depth(img, list()), "must be an sd_depth object")
  local_mocked_bindings(.onnx_run = function(model, inputs) list(1:3))
  expect_error(sd_estimate_depth(img, model), "Depth output has 3 values")
})

test_that("canny / depth nodes run in a graph", {
  local_mocked_bindings(
    sd_preprocess_canny_cpp = function(image, ...) {
      image$data <- as.raw(255L - as.integer(image$data)); image
    },
    .depth_cached = function(path, device = NULL, size = 518L) "MODEL",
    sd_estimate_depth = function(image, model) {
      stopifnot(identical(model, "MODEL")); image
    },
    sd_load_image = function(...) fake_img())
  pipe <- sd_pipeline(
    sd_node("load_image", id = "src", path = "a.png"),
    sd_node("canny", id = "edges", image = "src", inverse = TRUE),
    sd_node("depth", id = "dep", image = "src", model_path = "d.onnx")
  )
  res <- sd_run_pipeline(pipe, ctx = NULL, outputs = c("src", "edges", "dep"))
  expect_equal(as.integer(res$edges$data), 255L - as.integer(res$src$data))
  expect_identical(res$dep, res$src)
  bad <- sd_pipeline(sd_node("load_image", path = "a.png"), sd_node("depth"))
  expect_error(sd2R:::validate_pipeline(bad$nodes),
               "node 2 \\(depth\\): 'model_path' is not set")
})

test_that("txt2img control input: explicit, passed on, checked against ctx", {
  got <- NULL
  local_mocked_bindings(
    sd_txt2img = function(ctx, ...) { got <<- list(...); list(fake_img()) },
    sd_load_image = function(...) fake_img(),
    sd_preprocess_canny_cpp = function(image, ...) image)
  pipe <- sd_pipeline(
    sd_node("load_image", id = "src", path = "a.png"),
    sd_node("canny", id = "edges", image = "src"),
    sd_node("txt2img", prompt = "p", control = "edges", control_strength = 0.7)
  )
  sd_run_pipeline(pipe, ctx = NULL)
  expect_identical(got$control_image, fake_img())
  expect_equal(got$control_strength, 0.7)

  # not bound implicitly: an earlier image does not become the control
  plain <- sd_pipeline(sd_node("load_image", path = "a.png"),
                       sd_node("txt2img", prompt = "p"))
  sd_run_pipeline(plain, ctx = NULL)
  expect_null(got$control_image)

  local_mocked_bindings(sd_ctx_has_control_net_cpp = function(ctx) FALSE)
  expect_error(sd_run_pipeline(pipe, methods::new("externalptr")),
               "node 3 \\(txt2img\\): input 'control' needs a ControlNet")
})
