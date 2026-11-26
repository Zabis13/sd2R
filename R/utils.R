# Utility functions for sd2R

#' Get system information
#'
#' Returns information about the stable-diffusion.cpp backend.
#'
#' @return List with system info, version, and core count
#' @export
sd_system_info <- function() {
  info <- list(
    sd2R_version = as.character(utils::packageVersion("sd2R")),
    sd_cpp_version = sd_version_cpp(),
    system_info = sd_system_info_cpp(),
    num_cores = sd_num_physical_cores_cpp(),
    vulkan_available = ggmlR::ggml_vulkan_available()
  )
  class(info) <- "sd_system_info"
  info
}

#' Get number of Vulkan GPU devices
#'
#' Returns the number of Vulkan-capable GPU devices available on the system.
#' Useful for deciding whether to use \code{\link{sd_generate_multi_gpu}}.
#'
#' @return Integer, number of Vulkan devices (0 if Vulkan is not available)
#' @export
sd_vulkan_device_count <- function() {
  tryCatch(ggmlR::ggml_vulkan_device_count(), error = function(e) 0L)
}

#' @export
print.sd_system_info <- function(x, ...) {
  cat("sd2R System Information\n")
  cat("  sd2R version:   ", x$sd2R_version, "\n")
  cat("  sd.cpp version: ", x$sd_cpp_version, "\n")
  cat("  Physical cores: ", x$num_cores, "\n")
  cat("  Backend info:   ", x$system_info, "\n")
  cat("  Vulkan GPU:     ", if (x$vulkan_available) "available" else "not available", "\n")
  invisible(x)
}

#' Start profiling
#'
#' Clears the event buffer and begins capturing stage timings from sd.cpp.
#'
#' @return No return value, called for side effects.
#' @export
#' @usage sd_profile_start()
#' @name sd_profile_start
NULL

#' Stop profiling
#'
#' Stops capturing stage events. Call \code{\link{sd_profile_get}} to retrieve.
#'
#' @return No return value, called for side effects.
#' @export
#' @usage sd_profile_stop()
#' @name sd_profile_stop
NULL

#' Get raw profile events
#'
#' Returns a data frame of captured events with columns \code{stage},
#' \code{kind} (\code{"start"}/\code{"end"}), and \code{timestamp_ms}.
#'
#' @return Data frame of profile events.
#' @export
#' @usage sd_profile_get()
#' @name sd_profile_get
NULL

#' Build a profile summary from raw events
#'
#' Matches start/end events by stage and computes durations.
#'
#' @param events Data frame from \code{sd_profile_get()} with columns
#'   \code{stage}, \code{kind}, \code{timestamp_ms}.
#' @return Data frame with columns \code{stage}, \code{start_ms},
#'   \code{end_ms}, \code{duration_ms}, \code{duration_s}.
#'   Has class \code{"sd_profile"} for pretty printing.
#' @export
sd_profile_summary <- function(events) {
  if (nrow(events) == 0L) return(events)
  starts <- events[events$kind == "start", , drop = FALSE]
  ends   <- events[events$kind == "end",   , drop = FALSE]

  # Infer end times for load_* stages: each load ends when the next load starts,

  # or at load_all end
  load_starts <- events[grepl("^load_", events$stage) & events$kind == "start", ,
                         drop = FALSE]
  load_all_end <- events[events$stage == "load_all" & events$kind == "end", ,
                          drop = FALSE]
  if (nrow(load_starts) > 1L || (nrow(load_starts) == 1L && nrow(load_all_end) > 0L)) {
    load_starts <- load_starts[order(load_starts$timestamp_ms), , drop = FALSE]
    for (i in seq_len(nrow(load_starts))) {
      end_ts <- if (i < nrow(load_starts)) {
        load_starts$timestamp_ms[i + 1L]
      } else if (nrow(load_all_end) > 0L) {
        load_all_end$timestamp_ms[1L]
      } else NA_real_
      if (!is.na(end_ts)) {
        events <- rbind(events, data.frame(
          stage = load_starts$stage[i], kind = "end",
          timestamp_ms = end_ts, stringsAsFactors = FALSE
        ))
      }
    }
  }

  starts <- events[events$kind == "start", , drop = FALSE]
  ends   <- events[events$kind == "end",   , drop = FALSE]

  stages <- unique(c(starts$stage, ends$stage))
  rows <- list()
  for (s in stages) {
    if (s == "load_all") next
    s_starts <- starts$timestamp_ms[starts$stage == s]
    s_ends   <- ends$timestamp_ms[ends$stage == s]
    if (length(s_starts) > 0L && length(s_ends) > 0L) {
      # Pair each start with the first end after it and before the next start
      # of the same stage; surplus ends (e.g. two log markers for one stage
      # end) are ignored instead of shifting the pairing of later passes.
      s_starts <- sort(s_starts)
      s_ends   <- sort(s_ends)
      for (i in seq_along(s_starts)) {
        next_start <- if (i < length(s_starts)) s_starts[i + 1L] else Inf
        cand <- s_ends[s_ends >= s_starts[i] & s_ends < next_start]
        if (length(cand) == 0L) next
        dur <- cand[1L] - s_starts[i]
        rows[[length(rows) + 1L]] <- data.frame(
          stage = s,
          start_ms = s_starts[i],
          end_ms = cand[1L],
          duration_ms = dur,
          duration_s = round(dur / 1000, 2),
          stringsAsFactors = FALSE
        )
      }
    } else if (length(s_ends) > 0L) {
      for (i in seq_along(s_ends)) {
        rows[[length(rows) + 1L]] <- data.frame(
          stage = s,
          start_ms = NA_real_,
          end_ms = s_ends[i],
          duration_ms = NA_real_,
          duration_s = NA_real_,
          stringsAsFactors = FALSE
        )
      }
    }
  }
  result <- do.call(rbind, rows)
  result <- result[order(result$end_ms), , drop = FALSE]
  rownames(result) <- NULL
  class(result) <- c("sd_profile", "data.frame")
  result
}

#' Create cache configuration for step caching
#'
#' Constructs a list of cache parameters for fine-tuning step caching behavior.
#' Pass the result as \code{cache_config} to generation functions.
#'
#' @param mode Cache mode integer from \code{SD_CACHE_MODE} (default EASYCACHE)
#' @param threshold Reuse threshold (default 1.0). Lower = more aggressive caching
#' @param start_percent Start caching after this fraction of steps (default 0.15)
#' @param end_percent Stop caching after this fraction of steps (default 0.95)
#' @return Named list of cache parameters
#' @export
sd_cache_params <- function(mode = SD_CACHE_MODE$EASYCACHE,
                            threshold = 1.0,
                            start_percent = 0.15,
                            end_percent = 0.95) {
  list(
    cache_mode = as.integer(mode),
    cache_threshold = as.numeric(threshold),
    cache_start = as.numeric(start_percent),
    cache_end = as.numeric(end_percent)
  )
}

#' @export
print.sd_profile <- function(x, ...) {
  if (nrow(x) == 0L) {
    cat("(no profile events)\n")
    return(invisible(x))
  }

  # Identify passes by generate_total intervals
  passes <- x[x$stage == "generate_total" & !is.na(x$duration_s), , drop = FALSE]
  n_passes <- nrow(passes)

  # Compute-stage names (excluding load_* and generate_total)
  compute_stages <- c("text_encode", "text_encode_clip", "text_encode_t5",
                       "vae_encode", "sampling", "tiled_sampling", "vae_decode")

  # Helper: assign each row to a pass based on time interval
  .assign_pass <- function(rows, passes) {
    pass_id <- integer(nrow(rows))
    for (i in seq_len(nrow(rows))) {
      ts <- rows$end_ms[i]
      for (p in seq_len(nrow(passes))) {
        if (ts >= passes$start_ms[p] && ts <= passes$end_ms[p]) {
          pass_id[i] <- p
          break
        }
      }
    }
    pass_id
  }

  .pct <- function(dur, total) {
    if (!is.na(total) && !is.na(dur) && total > 0) {
      sprintf(" (%4.1f%%)", dur / total * 100)
    } else ""
  }

  .line <- function(label, dur, total, indent = 2L) {
    pad <- paste(rep(" ", indent), collapse = "")
    cat(sprintf("%s%-20s %7.2fs%s\n", pad, label, dur, .pct(dur, total)))
  }

  # Print one pass worth of stages
  .print_pass_stages <- function(pdata, total_s, indent = 2L) {
    # Load stages
    load_rows <- pdata[grepl("^load_", pdata$stage) & !is.na(pdata$duration_s), ,
                        drop = FALSE]
    if (nrow(load_rows) > 0L) {
      for (i in seq_len(nrow(load_rows))) {
        .line(load_rows$stage[i], load_rows$duration_s[i], total_s, indent)
      }
    }
    # Text encoding
    te <- pdata[pdata$stage == "text_encode" & !is.na(pdata$duration_s), , drop = FALSE]
    te_clip <- pdata[pdata$stage == "text_encode_clip" & !is.na(pdata$duration_s), ,
                      drop = FALSE]
    te_t5 <- pdata[pdata$stage == "text_encode_t5" & !is.na(pdata$duration_s), ,
                    drop = FALSE]
    if (nrow(te) > 0L) {
      .line("text_encode", sum(te$duration_s), total_s, indent)
      if (nrow(te_clip) > 0L) .line("clip", sum(te_clip$duration_s), total_s, indent + 2L)
      if (nrow(te_t5) > 0L)   .line("t5", sum(te_t5$duration_s), total_s, indent + 2L)
    }
    # VAE encode
    ve <- pdata[pdata$stage == "vae_encode" & !is.na(pdata$duration_s), , drop = FALSE]
    if (nrow(ve) > 0L) .line("vae_encode", sum(ve$duration_s), total_s, indent)
    # Sampling
    samp <- pdata[pdata$stage == "sampling" & !is.na(pdata$duration_s), , drop = FALSE]
    if (nrow(samp) > 0L) .line("sampling", sum(samp$duration_s), total_s, indent)
    # VAE decode
    vd <- pdata[pdata$stage == "vae_decode" & !is.na(pdata$duration_s), , drop = FALSE]
    if (nrow(vd) > 0L) .line("vae_decode", sum(vd$duration_s), total_s, indent)
  }

  cat("sd2R Profile\n")

  if (n_passes <= 1L) {
    # Single pass: original compact format
    total_s <- if (n_passes == 1L) passes$duration_s[1L] else NA_real_
    .print_pass_stages(x, total_s)
    if (!is.na(total_s)) {
      cat(sprintf("  %-20s %7.2fs\n", "TOTAL", total_s))
    }
  } else {
    # Multi-pass: show each pass, then summary
    compute_rows <- x[x$stage %in% compute_stages & !is.na(x$duration_s), , drop = FALSE]
    pass_ids <- .assign_pass(compute_rows, passes)
    compute_rows$pass <- pass_ids

    for (p in seq_len(n_passes)) {
      pass_total <- passes$duration_s[p]
      cat(sprintf("  Pass %d:\n", p))
      pdata <- compute_rows[compute_rows$pass == p, , drop = FALSE]
      .print_pass_stages(pdata, pass_total, indent = 4L)
      cat(sprintf("    %-20s %7.2fs\n", "pass_total", pass_total))
    }

    # Summary: aggregate across all passes
    grand_total <- sum(passes$duration_s)
    cat("  --- Summary ---\n")
    for (s in c("text_encode", "vae_encode", "sampling", "vae_decode")) {
      srows <- compute_rows[compute_rows$stage == s & !is.na(compute_rows$duration_s), ,
                             drop = FALSE]
      if (nrow(srows) > 0L) {
        .line(s, sum(srows$duration_s), grand_total)
      }
    }
    cat(sprintf("  %-20s %7.2fs\n", "TOTAL", grand_total))
  }
  invisible(x)
}
