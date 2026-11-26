# Background removal with RMBG-1.4 (BRIA AI) through ggmlR's ONNX runtime.
# Post-processing models run on ggmlR ONNX; diffusion stays in sd.cpp.

#' Load an RMBG background-removal model
#'
#' Loads an RMBG-1.4 ONNX model (\code{model.onnx} from the
#' \code{briaai/RMBG-1.4} repository; download it yourself) with
#' \code{ggmlR::onnx_load()}.
#'
#' @param path Path to the \code{.onnx} file.
#' @param device \code{"vulkan"} or \code{"cpu"}; \code{NULL} lets ggmlR pick
#'   (Vulkan when available).
#' @param dtype Weight precision, \code{"f32"} (default) or \code{"f16"}.
#' @return An object of class \code{"sd_rmbg"}, for
#'   \code{\link{sd_remove_background}}.
#' @export
#' @seealso \code{\link{sd_remove_background}}
sd_rmbg_load <- function(path, device = NULL, dtype = "f32") {
  v <- .ggmlr_version()
  if (v <= .rmbg_max_bad_ggmlr) {
    stop("Background removal needs ggmlR > ", .rmbg_max_bad_ggmlr,
         " (ONNX MaxPool ceil_mode fix); installed: ", v, call. = FALSE)
  }
  size <- .rmbg_size
  model <- ggmlR::onnx_load(path, device = device, dtype = dtype)
  inputs <- ggmlR::onnx_inputs(model)
  if (length(inputs) != 1L) {
    stop("RMBG model must have exactly one input, found ", length(inputs),
         call. = FALSE)
  }
  name <- names(inputs)[1L]
  if (any(inputs[[1L]] < 0L)) {
    # dynamic dims: rebuild the graph for a fixed 1 x 3 x 1024 x 1024 input
    model <- ggmlR::onnx_load(
      path, device = device, dtype = dtype,
      input_shapes = stats::setNames(list(c(1L, 3L, size, size)), name))
  }
  structure(list(model = model, input = name, size = size, path = path),
            class = "sd_rmbg")
}

#' @export
print.sd_rmbg <- function(x, ...) {
  cat("sd_rmbg:", basename(x$path), sprintf("(input '%s', %dx%d)\n",
                                            x$input, x$size, x$size))
  invisible(x)
}

#' Remove the background from an image
#'
#' Runs RMBG-1.4: the image is resized to 1024x1024, normalized
#' (\code{x / 255 - 0.5}), and the predicted foreground mask is min-max
#' normalized and resized back to the original size.
#'
#' @param image An SD image (list with \code{width}, \code{height},
#'   \code{channel}, \code{data}) or a path to a PNG file.
#' @param model An \code{sd_rmbg} object from \code{\link{sd_rmbg_load}}, or
#'   a path to the \code{.onnx} file (loaded for this call).
#' @param output \code{"rgba"} (default): the image with the mask as alpha
#'   channel; \code{"mask"}: the mask alone as a 1-channel image
#'   (white = foreground).
#' @param threshold Optional number in (0, 1): binarize the mask at this
#'   value instead of keeping soft edges.
#' @return An SD image: 4 channels for \code{"rgba"}, 1 for \code{"mask"}.
#'   Save with \code{\link{sd_save_image}}.
#' @export
#' @examples
#' \dontrun{
#' rmbg <- sd_rmbg_load("models/RMBG-1.4.onnx")
#' cut <- sd_remove_background("cat.png", rmbg)
#' sd_save_image(cut, "cat_rgba.png")
#' }
sd_remove_background <- function(image, model, output = c("rgba", "mask"),
                                 threshold = NULL) {
  output <- match.arg(output)
  if (is.character(image)) image <- sd_load_image(image)
  if (is.character(model)) model <- sd_rmbg_load(model)
  if (!inherits(model, "sd_rmbg")) {
    stop("`model` must be an sd_rmbg object or a path to the .onnx file",
         call. = FALSE)
  }
  if (!is.null(threshold) &&
      (!is.numeric(threshold) || threshold <= 0 || threshold >= 1)) {
    stop("`threshold` must be a number in (0, 1)", call. = FALSE)
  }

  rgb <- sd_image_to_array(image)                     # [H, W, C] in [0, 1]
  if (dim(rgb)[3] == 1L) rgb <- rgb[, , c(1L, 1L, 1L), drop = FALSE]
  rgb <- rgb[, , 1:3, drop = FALSE]
  h <- dim(rgb)[1]
  w <- dim(rgb)[2]

  mask <- .rmbg_predict(model, rgb)                   # [size, size] in [0, 1]
  mask <- .resize_array(array(mask, c(dim(mask), 1L)), w, h)[, , 1L]
  mask <- pmin(pmax(mask, 0), 1)
  if (!is.null(threshold)) mask <- (mask >= threshold) * 1

  if (output == "mask") {
    return(.array_to_sd_image(array(mask, c(h, w, 1L))))
  }
  out <- array(0, c(h, w, 4L))
  out[, , 1:3] <- rgb
  out[, , 4] <- mask
  .array_to_sd_image(out)
}

.rmbg_size <- 1024L

# ggmlR <= 0.9.0 mis-sizes ONNX MaxPool with ceil_mode, so RMBG's graph does
# not build (the package itself still works with older ggmlR)
.rmbg_max_bad_ggmlr <- "0.9.0"
.ggmlr_version <- function() utils::packageVersion("ggmlR")

# rgb: [H, W, 3] in [0, 1] -> foreground mask [size, size] ([H, W]) in [0, 1]
.rmbg_predict <- function(model, rgb) {
  size <- model$size
  x <- .resize_array(rgb, size, size) - 0.5
  # ONNX NCHW is row-major (W fastest): R column-major [W, H, C] has the
  # same memory layout
  x <- aperm(x, c(2L, 1L, 3L))
  out <- .onnx_run(model$model, stats::setNames(list(as.numeric(x)), model$input))
  y <- out[[1L]]
  if (length(y) != size * size) {
    stop(sprintf("RMBG output has %d values, expected %d (1 x 1 x %d x %d)",
                 length(y), size * size, size, size), call. = FALSE)
  }
  m <- t(matrix(as.numeric(y), size, size))           # storage [W, H] -> [H, W]
  rng <- range(m)
  if (diff(rng) > 0) (m - rng[1]) / diff(rng) else m * 0
}

# indirection so tests can mock inference without a model
.onnx_run <- function(model, inputs) ggmlR::onnx_run(model, inputs)

# Loaded RMBG models for the graph node, keyed by path and device
.rmbg_cache <- new.env(parent = emptyenv())

.rmbg_cached <- function(path, device = NULL) {
  key <- paste(normalizePath(path, mustWork = TRUE), device %||% "")
  if (is.null(.rmbg_cache[[key]])) {
    .rmbg_cache[[key]] <- sd_rmbg_load(path, device = device)
  }
  .rmbg_cache[[key]]
}
