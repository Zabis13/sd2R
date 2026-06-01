# ===========================================================================
# Live preview (TODO 9.4)
# ===========================================================================
# The C side writes the latest in-progress preview frame to a single PPM file,
# atomically (write <path>.tmp, then rename over <path> — atomic on POSIX), so
# this is safe to poll from R while generation runs in a worker thread. R reads
# the PPM here and converts to an sd_image. The C callback is never an R
# callback (calling R from the worker thread is forbidden), matching the
# file-based progress mechanism.

#' Enable live generation previews
#'
#' Installs the preview callback so that, during the next generation, the most
#' recent intermediate frame is written to \code{path} (a single PPM file,
#' updated atomically). Poll it with \code{\link{sd_read_preview}}. Call
#' \code{\link{sd_preview_stop}} when done.
#'
#' Most users pass \code{preview = TRUE} to \code{\link{sd_generate}} instead,
#' which wires this up automatically.
#'
#' @param path File path for the preview PPM (e.g. a tempfile).
#' @param mode Decode mode, one of \code{PREVIEW}: \code{"proj"} (fast, rough),
#'   \code{"tae"} (tiny autoencoder; needs \code{taesd_path} in
#'   \code{\link{sd_ctx}}), \code{"vae"} (full VAE; slow). Default \code{"proj"}.
#' @param interval Emit a preview every N sampling steps (default 1).
#' @param denoised If \code{TRUE} (default), preview the denoised estimate;
#'   otherwise the noisy latent.
#' @return Invisibly, \code{path}.
#' @export
#' @seealso \code{\link{sd_read_preview}}, \code{\link{sd_preview_stop}}
sd_preview_start <- function(path, mode = PREVIEW$PROJ, interval = 1L,
                             denoised = TRUE) {
  mode <- match.arg(as.character(mode), unlist(PREVIEW, use.names = FALSE))
  sd_set_preview_dump(path, mode, as.integer(interval), isTRUE(denoised),
                      !isTRUE(denoised))
  invisible(path)
}

#' Disable live generation previews
#'
#' Removes the preview callback and cleans up the temporary \code{.tmp} file.
#' @return Invisibly \code{NULL}.
#' @export
#' @seealso \code{\link{sd_preview_start}}
sd_preview_stop <- function() {
  sd_clear_preview_dump()
  invisible(NULL)
}

#' Read the current preview frame
#'
#' Reads the latest preview PPM written by the running generation and returns
#' it as an sd_image list. Returns \code{NULL} if no preview exists yet (e.g.
#' generation has not produced a frame). Optionally writes a PNG copy.
#'
#' @param path The preview PPM path passed to \code{\link{sd_preview_start}}.
#' @param png_path Optional path; if set, the frame is also written there as
#'   PNG via \code{\link{sd_save_image}}.
#' @return An sd_image list (\code{width}, \code{height}, \code{channel},
#'   \code{data}), or \code{NULL} if unavailable.
#' @export
#' @seealso \code{\link{sd_preview_start}}
sd_read_preview <- function(path, png_path = NULL) {
  if (!file.exists(path)) return(NULL)
  img <- .read_ppm_p6(path)
  if (is.null(img)) return(NULL)
  if (!is.null(png_path)) sd_save_image(img, png_path)
  img
}

# Minimal PPM P6 reader -> sd_image list. Tolerant of a partially written file
# (returns NULL) so a racing reader degrades gracefully rather than erroring.
# Header: "P6\n<w> <h>\n<maxval>\n" then w*h*3 raw bytes. We don't handle
# comments (# ...) because our own writer never emits them.
#
# The header is parsed byte-wise (not via rawToChar): the binary pixel body can
# contain embedded NUL bytes, and rawToChar() would truncate at the first NUL.
.read_ppm_p6 <- function(path) {
  tryCatch({
    raw_all <- readBin(path, "raw", n = file.info(path)$size)
    if (length(raw_all) < 11L) return(NULL)              # too short for a header
    if (!identical(raw_all[1:2], as.raw(c(0x50, 0x36)))) # "P6"
      return(NULL)
    # Collect the four header tokens (P6, w, h, maxval) by scanning bytes up to
    # and including the single whitespace that terminates maxval.
    ws <- as.raw(c(0x20, 0x09, 0x0a, 0x0d))              # space, tab, LF, CR
    tokens <- character(0)
    cur <- raw(0)
    i <- 1L
    n <- length(raw_all)
    while (i <= n && length(tokens) < 4L) {
      b <- raw_all[i]
      if (b %in% ws) {
        if (length(cur) > 0L) { tokens <- c(tokens, rawToChar(cur)); cur <- raw(0) }
      } else {
        cur <- c(cur, b)
      }
      i <- i + 1L
    }
    if (length(tokens) < 4L) return(NULL)                # partial header
    w <- suppressWarnings(as.integer(tokens[2]))
    h <- suppressWarnings(as.integer(tokens[3]))
    if (is.na(w) || is.na(h) || w <= 0L || h <= 0L) return(NULL)
    # i now points just past the whitespace after maxval -> start of pixels.
    need <- as.numeric(w) * h * 3
    body <- raw_all[i:n]
    if (length(body) < need) return(NULL)                # partial file
    list(width = w, height = h, channel = 3L, data = body[seq_len(need)])
  }, error = function(e) NULL)
}
