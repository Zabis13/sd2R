# LoRA support: explicit `loras` argument + webui-style <lora:name:weight> tags.
#
# The explicit named vector is the single source of truth; prompt tags are
# syntactic sugar over it. Order: extract tags -> merge with `loras` -> resolve
# paths (one validation point). Nothing of this lives in C++: the core only
# receives the resolved list (sd_img_gen_params_t.loras).

# Extension search order when a name is looked up in lora_model_dir.
.LORA_EXTS <- c(".safetensors", ".gguf", ".pt", ".ckpt")
.LORA_HIGH_NOISE <- "|high_noise|"

#' LoRA in sd2R
#'
#' LoRA adapters are passed per generation, either explicitly or as
#' webui-style tags in the prompt. Both forms can be mixed.
#'
#' \strong{Explicit:} \code{loras = c("style.safetensors" = 0.8, "detail" = 0.5)}
#' — a named numeric vector, names are paths or names, values are weights.
#'
#' \strong{Prompt tags:} \code{"a cat <lora:style:0.8>"}. Tags are removed from
#' the prompt before encoding. The name may contain spaces and colons (e.g.
#' Windows paths \code{C:\\loras\\style.safetensors}); the weight is the part
#' after the last colon.
#'
#' \strong{Path resolution} (same for both forms):
#' \enumerate{
#'   \item an existing file path is used as is (no directory search);
#'   \item otherwise the name is looked up in \code{lora_model_dir} (see
#'     \code{\link{sd_ctx}}), first as given, then with the extensions
#'     \code{.safetensors}, \code{.gguf}, \code{.pt}, \code{.ckpt} in this order.
#' }
#' A LoRA that cannot be found is an error. This deliberately differs from the
#' stable-diffusion.cpp CLI, which only warns and silently generates without it.
#'
#' A name prefixed with \code{|high_noise|} targets the high-noise expert of
#' two-expert video models (Wan 2.2), as in the CLI.
#'
#' \strong{Repeated LoRA:} weights of the same file are summed (as in the CLI),
#' and a message reports the sum.
#'
#' \strong{Application check:} after generation the number of LoRA tensors that
#' were actually applied is queried from the core (\code{\link{sd_lora_stats}}).
#' A LoRA with less than \code{getOption("sd2R.lora_min_applied", 0.5)} of its
#' compatible tensors applied (e.g. trained for another architecture whose
#' block names partly coincide, such as a FLUX.1 LoRA on FLUX.2) is an error and
#' no image is returned; partial application above that share gives a warning
#' with the counts (legitimate for LoRAs that touch only some blocks). Set the
#' option lower (e.g. \code{0}) to accept such LoRAs. The check runs after the
#' generation, so a wrong LoRA costs one full generation.
#'
#' \strong{Apply modes} (\code{lora_apply_mode} in \code{\link{sd_ctx}}):
#' \describe{
#'   \item{\code{AT_RUNTIME}}{base weights stay untouched; the low-rank delta is
#'     added during every forward pass. Exact for quantized models, slightly
#'     slower.}
#'   \item{\code{IMMEDIATELY}}{the delta is merged into the weights once. For
#'     quantized weights this is dequantize -> add -> requantize, so part of the
#'     (small) LoRA delta is lost to quantization — the usual reason for
#'     "LoRA looks weaker / worse than expected". Not available for INT8
#'     tensorwise weights, row-split tensors, or weight types the backend
#'     cannot write back to (Vulkan: K-quants and most IQ types); then the whole
#'     context falls back to at_runtime with a logged warning. The mode in
#'     effect is reported by \code{\link{sd_lora_apply_mode}}.}
#'   \item{\code{AUTO}}{(default) at_runtime if the model has quantized
#'     weights, weights are offloaded to CPU/disk or row-split is active;
#'     immediately otherwise.}
#' }
#'
#' @name sd_lora
NULL

#' Extract <lora:name:weight> tags from a prompt
#' @param prompt Prompt string
#' @return list(prompt = prompt without tags, loras = named numeric vector)
#' @keywords internal
.parse_lora_tags <- function(prompt) {
  if (is.null(prompt) || !nzchar(prompt)) return(list(prompt = prompt, loras = NULL))
  # name: anything up to the LAST colon before '>' (keeps C:\... and spaces)
  re <- "<lora:([^>]+):([^:>]+)>"
  tags <- regmatches(prompt, gregexpr(re, prompt, perl = TRUE))[[1]]
  if (!length(tags)) return(list(prompt = prompt, loras = NULL))
  m <- regmatches(tags, regexec(re, tags, perl = TRUE))
  names_ <- trimws(vapply(m, `[`, "", 2L))
  w_raw  <- trimws(vapply(m, `[`, "", 3L))
  w <- suppressWarnings(as.numeric(w_raw))
  if (anyNA(w)) {
    stop("Invalid LoRA weight in prompt tag: ", paste(tags[is.na(w)], collapse = ", "),
         call. = FALSE)
  }
  clean <- gsub(re, "", prompt, perl = TRUE)
  clean <- gsub("[ \t]+([,.;])", "\\1", gsub("[ \t]{2,}", " ", clean))
  clean <- trimws(clean)
  list(prompt = clean, loras = stats::setNames(w, names_))
}

#' Resolve a LoRA name to an existing file
#' @param name Path or name (without the |high_noise| prefix)
#' @param dir lora_model_dir or NULL
#' @return Normalized path; errors if not found
#' @keywords internal
.resolve_lora_path <- function(name, dir) {
  if (file.exists(name) && !dir.exists(name)) {
    return(normalizePath(name, mustWork = TRUE))
  }
  bases <- if (!is.null(dir) && nzchar(dir)) file.path(dir, name) else name
  for (b in bases) {
    for (cand in c(b, paste0(b, .LORA_EXTS))) {
      if (file.exists(cand) && !dir.exists(cand)) {
        return(normalizePath(cand, mustWork = TRUE))
      }
    }
  }
  stop("LoRA not found: '", name, "'",
       if (is.null(dir) || !nzchar(dir)) " (no lora_model_dir set in sd_ctx())"
       else paste0(" (searched in lora_model_dir = '", dir, "' with extensions ",
                   paste(.LORA_EXTS, collapse = ", "), ")"),
       call. = FALSE)
}

#' Merge prompt tags and explicit loras, resolve paths
#' @param ctx SD context (attribute lora_model_dir)
#' @param prompt Prompt string (tags are removed)
#' @param loras Named numeric vector or NULL
#' @return list(prompt, loras = NULL or list(path, multiplier, is_high_noise))
#' @keywords internal
.resolve_loras <- function(ctx, prompt, loras = NULL) {
  tags <- .parse_lora_tags(prompt)
  if (!is.null(loras) && length(loras)) {
    if (!is.numeric(loras) || is.null(names(loras)) || any(!nzchar(names(loras)))) {
      stop("`loras` must be a named numeric vector: c(\"path_or_name\" = weight, ...)",
           call. = FALSE)
    }
    if (anyNA(loras)) stop("`loras` contains NA weights", call. = FALSE)
  }
  all_w <- c(tags$loras, loras)
  if (!length(all_w)) return(list(prompt = tags$prompt, loras = NULL))

  dir  <- attr(ctx, "lora_model_dir")
  nm   <- names(all_w)
  high <- startsWith(nm, .LORA_HIGH_NOISE)
  nm[high] <- substring(nm[high], nchar(.LORA_HIGH_NOISE) + 1L)
  paths <- vapply(nm, .resolve_lora_path, "", dir = dir, USE.NAMES = FALSE)

  key <- paste0(ifelse(high, .LORA_HIGH_NOISE, ""), paths)
  ukey <- unique(key)
  mult <- vapply(ukey, function(k) sum(all_w[key == k]), 0, USE.NAMES = FALSE)
  for (k in ukey[vapply(ukey, function(k) sum(key == k) > 1L, NA)]) {
    w <- all_w[key == k]
    message(sprintf("LoRA '%s' given %d times, weights summed: %s = %g",
                    basename(k), length(w), paste(format(w), collapse = " + "), sum(w)))
  }
  uhigh <- startsWith(ukey, .LORA_HIGH_NOISE)
  upath <- ifelse(uhigh, substring(ukey, nchar(.LORA_HIGH_NOISE) + 1L), ukey)
  list(prompt = tags$prompt,
       loras = list(path = upath, multiplier = as.numeric(mult),
                    is_high_noise = uhigh))
}

#' LoRA application statistics of the last generation
#'
#' Number of LoRA tensors that target the model, were applied, or were skipped
#' as incompatible (shape mismatch), per LoRA file, for the last generation on
#' \code{ctx}. Same counting as the "(applied / total) LoRA tensors" log line
#' of stable-diffusion.cpp.
#'
#' @param ctx SD context from \code{\link{sd_ctx}}
#' @return data.frame with columns \code{path}, \code{total}, \code{applied},
#'   \code{skipped}. A LoRA none of whose tensors target the model has no row.
#' @export
#' @seealso \code{\link{sd_lora}}
sd_lora_stats <- function(ctx) {
  sd_ctx_lora_stats_cpp(ctx)
}

#' Effective LoRA apply mode of a context
#'
#' The mode actually used by \code{ctx}: \code{AUTO} resolved, and
#' \code{IMMEDIATELY} replaced by \code{AT_RUNTIME} when the backend cannot
#' merge into the model's weight type (e.g. Vulkan with K-quants; a warning is
#' logged). Decided when the model is loaded, i.e. in \code{\link{sd_ctx}} with
#' \code{eager_load = TRUE} (default), otherwise at the first generation.
#'
#' @param ctx SD context from \code{\link{sd_ctx}}
#' @return \code{"immediately"} or \code{"at_runtime"}.
#' @export
#' @seealso \code{\link{sd_lora}}, \code{LORA_APPLY_MODE}
sd_lora_apply_mode <- function(ctx) {
  m <- sd_ctx_lora_apply_mode_cpp(ctx)
  if (m == LORA_APPLY_MODE$IMMEDIATELY) "immediately"
  else if (m == LORA_APPLY_MODE$AT_RUNTIME) "at_runtime"
  else stop("Invalid sd_ctx", call. = FALSE)
}

#' Check LoRA application after a generation
#' @param ctx SD context
#' @param loras Resolved loras (list(path, multiplier, is_high_noise)) or NULL
#' @return The stats data.frame (invisibly), or NULL without loras
#' @keywords internal
.check_lora_stats <- function(ctx, loras) {
  if (is.null(loras) || !length(loras$path)) return(invisible(NULL))
  st <- sd_ctx_lora_stats_cpp(ctx)
  min_share <- getOption("sd2R.lora_min_applied", 0.5)
  for (p in unique(loras$path)) {
    row <- st[st$path == p, , drop = FALSE]
    total   <- if (nrow(row)) row$total[1] else 0
    applied <- if (nrow(row)) row$applied[1] else 0
    skipped <- if (nrow(row)) row$skipped[1] else 0
    if (applied == 0) {
      stop(sprintf(paste0("LoRA '%s': 0 of %d tensors applied — it does not match ",
                          "this model (different architecture or unsupported ",
                          "naming format). The image was discarded."),
                   basename(p), as.integer(total)), call. = FALSE)
    }
    if (total - skipped > 0 && applied / (total - skipped) < min_share) {
      stop(sprintf(paste0("LoRA '%s': only %d of %d tensors applied (< %g%%) — it most ",
                          "likely targets another architecture with partly matching ",
                          "block names. The image was discarded. Set ",
                          "options(sd2R.lora_min_applied = 0) to accept it."),
                   basename(p), as.integer(applied), as.integer(total - skipped),
                   100 * min_share), call. = FALSE)
    }
    if (applied < total - skipped) {
      warning(sprintf("LoRA '%s': only %d of %d tensors applied",
                      basename(p), as.integer(applied), as.integer(total - skipped)),
              call. = FALSE)
    }
    if (skipped > 0) {
      warning(sprintf("LoRA '%s': %d of %d tensors skipped as incompatible (shape mismatch)",
                      basename(p), as.integer(skipped), as.integer(total)),
              call. = FALSE)
    }
  }
  invisible(st)
}

#' Run one generation with LoRA resolution and the post-generation check
#' @param ctx SD context
#' @param params Params list for sd_generate_image (prompt may contain tags)
#' @param loras Named numeric vector or NULL
#' @return List of images
#' @keywords internal
.sd_generate_with_loras <- function(ctx, params, loras = NULL) {
  lr <- .resolve_loras(ctx, params$prompt, loras)
  params$prompt <- lr$prompt
  if (!is.null(lr$loras)) params$loras <- lr$loras
  res <- sd_generate_image(ctx, params)
  .check_lora_stats(ctx, lr$loras)
  res
}
