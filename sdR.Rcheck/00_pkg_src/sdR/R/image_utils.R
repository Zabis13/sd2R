# Image utility functions for sdR
# C++ sd_image_t uses row-major RGB interleaved: byte[y*w*ch + x*ch + c]
# R arrays are column-major [H, W, C]: element[h, w, c]
# Conversion requires aperm to reorder dimensions.

#' Save SD image to PNG file
#'
#' @param image SD image (list with width, height, channel, data) as returned
#'   by \code{sd_txt2img()} or \code{sd_img2img()}. Can also be a 3D numeric
#'   array [height, width, channels] with values in [0, 1].
#' @param path Output file path (should end in .png)
#' @export
sd_save_image <- function(image, path) {
  if (!requireNamespace("png", quietly = TRUE)) {
    stop("Package 'png' is required. Install with install.packages('png')", call. = FALSE)
  }

  if (is.array(image) && length(dim(image)) == 3) {
    img_array <- image
  } else {
    img_array <- sd_image_to_array(image)
  }

  img_array <- pmin(pmax(img_array, 0), 1)

  dir <- dirname(path)
  if (!dir.exists(dir) && dir != ".") {
    dir.create(dir, recursive = TRUE)
  }

  png::writePNG(img_array, path)
  invisible(path)
}

#' Convert SD image to R numeric array
#'
#' Converts the raw uint8 SD image format to a [height, width, channels]
#' numeric array with values in [0, 1] suitable for R image processing.
#'
#' @param image SD image list (width, height, channel, data)
#' @return 3D numeric array [height, width, channels] in [0, 1]
#' @export
sd_image_to_array <- function(image) {
  w <- image$width
  h <- image$height
  ch <- image$channel

  vals <- as.integer(image$data) / 255.0

  # Raw data is row-major interleaved: [y][x][c]
  # R array() fills column-major (first dim fastest).
  # We want [H, W, C] array.
  # Row-major [y][x][c] means c varies fastest, then x, then y.
  # So array(vals, dim=c(ch, w, h)) fills c first, then w, then h.
  # Then aperm(c(3,2,1)) → [h, w, ch]
  arr <- array(vals, dim = c(ch, w, h))
  aperm(arr, c(3, 2, 1))
}

#' Load image from file as SD image
#'
#' Reads a PNG file and converts it to the SD image format
#' (list with width, height, channel, data) suitable for img2img.
#'
#' @param path Path to image file (PNG)
#' @param channels Number of output channels (3 for RGB, default)
#' @return SD image list (width, height, channel, data as raw vector)
#' @export
sd_load_image <- function(path, channels = 3L) {
  if (!requireNamespace("png", quietly = TRUE)) {
    stop("Package 'png' is required. Install with install.packages('png')", call. = FALSE)
  }

  img <- png::readPNG(path)
  if (length(dim(img)) == 2) {
    img <- array(rep(img, 3), dim = c(dim(img), 3L))
  }
  if (dim(img)[3] == 4L && channels == 3L) {
    img <- img[, , 1:3]
  }

  h <- dim(img)[1]
  w <- dim(img)[2]
  ch <- dim(img)[3]

  # R array [H, W, C] is column-major.
  # Need row-major interleaved [y][x][c] (c varies fastest).
  # aperm [H,W,C] → [C,W,H] then as.vector reads c-first, w-next, h-last = row-major
  interleaved <- aperm(img, c(3, 2, 1))
  bytes <- as.raw(as.integer(pmin(pmax(as.numeric(interleaved) * 255, 0), 255)))

  list(
    width = as.integer(w),
    height = as.integer(h),
    channel = as.integer(ch),
    data = bytes
  )
}
