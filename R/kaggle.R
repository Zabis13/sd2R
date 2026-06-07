# Download Stable Diffusion models from Kaggle Models
#
# Kaggle's public Models API serves model variations as a single compressed
# bundle (.tar.gz) per version -- there is no public endpoint for fetching an
# individual file, so the whole bundle is downloaded and unpacked. Only public
# models are supported (no Kaggle credentials required).

# Kaggle public API base.
.kaggle_api <- "https://www.kaggle.com/api/v1"

# Parse a kagglehub-style handle "owner/model/framework/variation" into parts.
.parse_kaggle_handle <- function(handle) {
  parts <- strsplit(handle, "/", fixed = TRUE)[[1]]
  if (length(parts) != 4L || any(!nzchar(parts))) {
    stop(
      "Invalid Kaggle handle '", handle, "'. Expected ",
      "'owner/model/framework/variation', e.g. ",
      "'lbsbmsu/flux-2/gguf/default'.",
      call. = FALSE
    )
  }
  list(owner = parts[1L], model = parts[2L],
       framework = parts[3L], variation = parts[4L])
}

# Look up the latest version number for a given model instance via the public
# metadata endpoint (no authentication needed for public models).
.kaggle_latest_version <- function(h) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required.", call. = FALSE)
  }
  url <- sprintf("%s/models/%s/%s/get", .kaggle_api, h$owner, h$model)
  meta <- tryCatch(
    jsonlite::fromJSON(url, simplifyVector = FALSE),
    error = function(e) {
      stop("Failed to query Kaggle model metadata for '",
           h$owner, "/", h$model, "': ", conditionMessage(e), call. = FALSE)
    }
  )
  instances <- meta$instances
  for (inst in instances) {
    if (identical(tolower(inst$framework), tolower(h$framework)) &&
        identical(inst$slug, h$variation)) {
      return(inst$versionNumber)
    }
  }
  stop("No instance '", h$framework, "/", h$variation,
       "' found for model '", h$owner, "/", h$model, "'.", call. = FALSE)
}

#' Download a Stable Diffusion model from Kaggle Models
#'
#' Downloads a model bundle from the public Kaggle Models registry and unpacks
#' it into \code{dest}. Mirrors the behaviour of the Python \code{kagglehub}
#' package (\code{kagglehub.model_download("owner/model/framework/variation")})
#' but uses only base R -- no Python dependency.
#'
#' Kaggle serves each model version as a single \code{.tar.gz} bundle; the whole
#' bundle is downloaded even when only some files are needed. Only public models
#' are supported.
#'
#' @param handle Model handle in kagglehub form
#'   \code{"owner/model/framework/variation"}. Defaults to
#'   \code{"lbsbmsu/flux-2/gguf/default"} -- a ready-to-use FLUX 2 (GGUF) model,
#'   so newcomers can call \code{sd_download_model(dest = "models/flux2")}.
#' @param dest Destination directory for the unpacked files. Created if it does
#'   not exist. Required.
#' @param version Integer version number. If \code{NULL} (default) the latest
#'   version is resolved automatically from Kaggle.
#' @param files Optional character vector of file names to extract from the
#'   bundle. If \code{NULL} (default) all files are extracted.
#' @param verbose Logical; print progress messages. Defaults to \code{FALSE}.
#' @return The path to \code{dest} (invisibly), containing the model files.
#' @export
sd_download_model <- function(handle = "lbsbmsu/flux-2/gguf/default", dest,
                              version = NULL, files = NULL, verbose = FALSE) {
  if (missing(dest) || !is.character(dest) || length(dest) != 1L || !nzchar(dest)) {
    stop("'dest' must be a single non-empty directory path.", call. = FALSE)
  }
  h <- .parse_kaggle_handle(handle)

  if (is.null(version)) {
    version <- .kaggle_latest_version(h)
    if (verbose) message("Resolved latest version: ", version)
  }

  if (!dir.exists(dest)) {
    dir.create(dest, recursive = TRUE, showWarnings = FALSE)
  }

  # Skip download when the requested files are already present.
  if (!is.null(files)) {
    present <- file.exists(file.path(dest, files))
    if (all(present)) {
      if (verbose) message("All requested files already present; skipping download.")
      return(invisible(dest))
    }
  } else if (length(list.files(dest, recursive = TRUE)) > 0L) {
    if (verbose) message("Destination already populated; skipping download.")
    return(invisible(dest))
  }

  url <- sprintf("%s/models/%s/%s/%s/%s/%s/download",
                 .kaggle_api, h$owner, h$model, h$framework, h$variation, version)
  archive <- file.path(
    dest,
    sprintf("%s-%s-%s-v%s.tar.gz", h$model, h$framework, h$variation, version)
  )

  if (!file.exists(archive)) {
    if (verbose) message("Downloading ", url)
    ok <- tryCatch(
      utils::download.file(url, archive, mode = "wb",
                           quiet = !verbose, method = "libcurl"),
      error = function(e) {
        stop("Download failed: ", conditionMessage(e), call. = FALSE)
      }
    )
    if (!identical(ok, 0L) || !file.exists(archive)) {
      stop("Download failed for ", url, call. = FALSE)
    }
  } else if (verbose) {
    message("Archive already downloaded: ", archive)
  }

  if (verbose) message("Extracting bundle into ", dest)
  res <- utils::untar(archive, files = files, exdir = dest)
  if (!identical(res, 0L)) {
    stop("Failed to extract '", archive, "'.", call. = FALSE)
  }

  invisible(dest)
}
