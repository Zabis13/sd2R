# ControlNet preprocessors: canny (sd.cpp core) and depth (ggmlR ONNX).
# The hint images they produce go to the `control` input of txt2img / sample.

#' Canny edge map for ControlNet
#'
#' Computes a Canny edge map with the stable-diffusion.cpp core (the same code
#' as its \code{--canny} option): grayscale, Gaussian blur, Sobel gradients,
#' non-maximum suppression and hysteresis thresholding.
#'
#' @param image SD image (list with \code{width}, \code{height},
#'   \code{channel}, \code{data}) or a path to a PNG file.
#' @param high_threshold,low_threshold Hysteresis thresholds on the
#'   normalized gradient, in [0, 1] (defaults 0.08, as in sd.cpp).
#' @param weak,strong Output values for weak and strong edges, in [0, 1]
#'   (defaults 0.8 and 1).
#' @param inverse If \code{TRUE}, dark edges on white.
#' @return An SD image with the same size and channels as the input, the
#'   edge map in every colour channel (alpha is kept).
#' @examples
#' \dontrun{
#' edges <- sd_canny("photo.png")
#' sd_save_image(edges, "edges.png")
#' }
#' @export
sd_canny <- function(image, high_threshold = 0.08, low_threshold = 0.08,
                     weak = 0.8, strong = 1, inverse = FALSE) {
  if (is.character(image)) image <- sd_load_image(image)
  unit <- function(v, nm) {
    if (!is.numeric(v) || length(v) != 1L || is.na(v) || v < 0 || v > 1) {
      stop("sd_canny: '", nm, "' must be a number in [0, 1]", call. = FALSE)
    }
    as.numeric(v)
  }
  sd_preprocess_canny_cpp(image, unit(high_threshold, "high_threshold"),
                          unit(low_threshold, "low_threshold"),
                          unit(weak, "weak"), unit(strong, "strong"),
                          isTRUE(inverse))
}

#' Load a depth estimation model (ONNX)
#'
#' Loads a monocular depth model through ggmlR's ONNX runtime. The defaults
#' fit Depth Anything V2 (input 518 x 518, ImageNet normalization, output
#' relative inverse depth); \code{size}, \code{mean} and \code{std} can be
#' changed for other models of the same kind (e.g. MiDaS / DPT: 384, mean
#' and std 0.5).
#'
#' @param path Path to the \code{.onnx} file.
#' @param device ggmlR device (\code{NULL}: default device).
#' @param dtype Weight type for ggmlR (default \code{"f32"}).
#' @param size Square input size in pixels (a multiple of 14 for Depth
#'   Anything). Default 518.
#' @param mean,std Per-channel (RGB) normalization of the [0, 1] input.
#' @return An object of class \code{"sd_depth"}.
#' @seealso \code{\link{sd_estimate_depth}}
#' @export
sd_depth_load <- function(path, device = NULL, dtype = "f32", size = 518L,
                          mean = c(0.485, 0.456, 0.406),
                          std = c(0.229, 0.224, 0.225)) {
  size <- as.integer(size)
  if (length(size) != 1L || is.na(size) || size < 1L) {
    stop("sd_depth_load: 'size' must be a positive integer", call. = FALSE)
  }
  if (length(mean) != 3L || length(std) != 3L || any(std <= 0)) {
    stop("sd_depth_load: 'mean' and 'std' must have 3 values (std > 0)",
         call. = FALSE)
  }
  model <- ggmlR::onnx_load(path, device = device, dtype = dtype)
  inputs <- ggmlR::onnx_inputs(model)
  if (length(inputs) != 1L) {
    stop("Depth model must have exactly one input, found ", length(inputs),
         call. = FALSE)
  }
  name <- names(inputs)[1L]
  dims <- inputs[[1L]]
  if (length(dims) != 4L || any(dims != c(1L, 3L, size, size))) {
    # dynamic dims (reported as -1 or 1 by ggmlR) or another size: rebuild
    # the graph for a fixed 1 x 3 x size x size input
    model <- ggmlR::onnx_load(
      path, device = device, dtype = dtype,
      input_shapes = stats::setNames(list(c(1L, 3L, size, size)), name))
  }
  structure(list(model = model, input = name, size = size,
                 mean = as.numeric(mean), std = as.numeric(std), path = path),
            class = "sd_depth")
}

#' @export
print.sd_depth <- function(x, ...) {
  cat("sd_depth:", basename(x$path), sprintf("(input '%s', %dx%d)\n",
                                             x$input, x$size, x$size))
  invisible(x)
}

#' Depth map for ControlNet
#'
#' Estimates relative depth with a model loaded by \code{\link{sd_depth_load}}
#' and returns it as a hint image for depth ControlNets: near = white, far =
#' black, min-max normalized per image, resized back to the input size
#' (bilinear).
#'
#' @param image SD image or a path to a PNG file.
#' @param model An \code{sd_depth} object or a path to the \code{.onnx} file.
#' @return A 3-channel SD image of the input size (gray depth in every
#'   channel).
#' @examples
#' \dontrun{
#' depth <- sd_depth_load("depth_anything_v2_small.onnx")
#' sd_save_image(sd_estimate_depth("photo.png", depth), "depth.png")
#' }
#' @export
sd_estimate_depth <- function(image, model) {
  if (is.character(image)) image <- sd_load_image(image)
  if (is.character(model)) model <- sd_depth_load(model)
  if (!inherits(model, "sd_depth")) {
    stop("`model` must be an sd_depth object or a path to the .onnx file",
         call. = FALSE)
  }
  rgb <- sd_image_to_array(image)                     # [H, W, C] in [0, 1]
  if (dim(rgb)[3] < 3L) rgb <- rgb[, , rep(1L, 3L), drop = FALSE]
  rgb <- rgb[, , 1:3, drop = FALSE]
  h <- dim(rgb)[1]; w <- dim(rgb)[2]
  d <- .depth_predict(model, rgb)                     # [size, size] in [0, 1]
  d <- .resize_array(array(d, c(dim(d), 1L)), w, h)[, , 1L]
  d <- pmin(pmax(d, 0), 1)
  .array_to_sd_image(array(d, c(h, w, 3L)))
}

# rgb: [H, W, 3] in [0, 1] -> depth [size, size] ([H, W]), min-max to [0, 1]
# with larger model output (nearer, for inverse-depth models) = brighter
.depth_predict <- function(model, rgb) {
  size <- model$size
  x <- .resize_array(rgb, size, size)
  for (ci in 1:3) x[, , ci] <- (x[, , ci] - model$mean[ci]) / model$std[ci]
  # ONNX NCHW is row-major (W fastest): R column-major [W, H, C] matches
  x <- aperm(x, c(2L, 1L, 3L))
  out <- .onnx_run(model$model, stats::setNames(list(as.numeric(x)), model$input))
  y <- out[[1L]]
  if (length(y) != size * size) {
    stop(sprintf("Depth output has %d values, expected %d (1 x %d x %d)",
                 length(y), size * size, size, size), call. = FALSE)
  }
  m <- t(matrix(as.numeric(y), size, size))           # storage [W, H] -> [H, W]
  rng <- range(m)
  if (diff(rng) > 0) (m - rng[1]) / diff(rng) else m * 0
}

# Loaded depth models for the graph node, keyed by path, device and size
.depth_cache <- new.env(parent = emptyenv())
.depth_cached <- function(path, device = NULL, size = 518L) {
  key <- paste(normalizePath(path, mustWork = TRUE), device %||% "", size)
  if (is.null(.depth_cache[[key]])) {
    .depth_cache[[key]] <- sd_depth_load(path, device = device, size = size)
  }
  .depth_cache[[key]]
}
