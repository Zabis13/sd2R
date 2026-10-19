#!/usr/bin/env Rscript
# Sampling bottleneck profiler — Flux.1 768x768, 1 step, per-op Vulkan timings
# Run: Rscript /mnt/Data2/DS_projects/sd2R/inst/examples/test_sampling_profile.R

Sys.setenv(GGML_VK_PERF_LOGGER = "1")

library(sd2R)

models_dir <- "/mnt/Data3/DS_Data/sd_models"

# Split lines into named sections by "----------------" separator.
# Section names: "clip", "t5", "dit_double", "dit_single", "vae", "other" —
# inferred from dominant ops in each block.
split_sections <- function(lines) {
  sep_idx <- c(which(grepl("^-{4,}", lines)), length(lines) + 1L)
  if (length(sep_idx) < 2) return(list(all = lines))

  sections <- list()
  for (i in seq_len(length(sep_idx) - 1L)) {
    block <- lines[seq(sep_idx[i] + 1L, sep_idx[i + 1L] - 1L)]
    block <- block[nzchar(trimws(block))]
    if (length(block) == 0) next

    # name heuristic: look at dominant op signatures
    has <- function(p) any(grepl(p, block, ignore.case = TRUE))
    name <- if (has("FLASH_ATTN"))                        "dit_single"
            else if (has("q4_K.*n=2560|n=2304.*q4_K"))   "dit_double"
            else if (has("q5_K|q6_K"))                    "t5"
            else if (has("f16.*n=77|n=77.*f16"))          "clip"
            else if (has("CONV_2D"))                      "vae"
            else                                          sprintf("sec%d", i)

    # sections can repeat (multi-step), accumulate
    if (is.null(sections[[name]])) sections[[name]] <- block
    else sections[[name]] <- c(sections[[name]], block)
  }
  sections
}

parse_section <- function(lines) {
  # "ADD: 72 x 1523.12 us = 109664.64 us"  →  op, count, avg_us, total_us
  pat  <- "^(.+):\\s+(\\d+) x ([0-9.]+) us = ([0-9.]+) us"
  hits <- regmatches(lines, regexec(pat, lines))
  rows <- lapply(hits, function(m) {
    if (length(m) < 5) return(NULL)
    data.frame(op       = m[[2]],
               count    = as.integer(m[[3]]),
               avg_us   = as.numeric(m[[4]]),
               total_us = as.numeric(m[[5]]),
               stringsAsFactors = FALSE)
  })
  do.call(rbind, Filter(Negate(is.null), rows))
}

print_top <- function(df, top_n, label) {
  if (is.null(df) || nrow(df) == 0) return(invisible(NULL))
  agg <- aggregate(total_us ~ op, data = df, FUN = sum)
  agg <- agg[order(-agg$total_us), ]
  agg$pct <- agg$total_us / sum(agg$total_us) * 100
  total_ms <- sum(agg$total_us) / 1e3
  cat(sprintf("\n--- %s  (total %.1fms) ---\n", label, total_ms))
  cat(sprintf("  %-53s %9s %6s\n", "Op", "ms", "%"))
  for (i in seq_len(min(top_n, nrow(agg)))) {
    cat(sprintf("  %-53s %9.2f %5.1f%%\n",
                agg$op[i], agg$total_us[i] / 1e3, agg$pct[i]))
  }
}

parse_vk_timings <- function(lines, top_n = 10) {
  secs <- split_sections(lines)

  # per-section breakdown
  for (nm in names(secs)) {
    df <- parse_section(secs[[nm]])
    print_top(df, top_n, nm)
  }

  # global summary across all sections
  all_df <- do.call(rbind, lapply(secs, parse_section))
  if (is.null(all_df) || nrow(all_df) == 0) return(invisible(NULL))
  agg <- aggregate(total_us ~ op, data = all_df, FUN = sum)
  agg <- agg[order(-agg$total_us), ]
  agg$pct <- agg$total_us / sum(agg$total_us) * 100
  cat(sprintf("\n=== GLOBAL TOP %d (%.1fms total) ===\n", top_n, sum(agg$total_us) / 1e3))
  cat(sprintf("  %-53s %9s %6s\n", "Op", "ms", "%"))
  for (i in seq_len(min(top_n, nrow(agg)))) {
    cat(sprintf("  %-53s %9.2f %5.1f%%\n",
                agg$op[i], agg$total_us[i] / 1e3, agg$pct[i]))
  }

  # per-section detail for ADD and SCALE
  detail_ops <- c("ADD", "SCALE", "CONT", "MUL")
  cat(sprintf("\n=== ADD / SCALE / CONT / MUL per section ===\n"))
  cat(sprintf("  %-12s %-10s %8s %10s %10s\n", "section", "op", "count", "avg_us", "total_ms"))
  cat(strrep("-", 56), "\n")
  for (nm in names(secs)) {
    df <- parse_section(secs[[nm]])
    if (is.null(df)) next
    df <- df[df$op %in% detail_ops, , drop = FALSE]
    if (nrow(df) == 0) next
    df <- df[order(match(df$op, detail_ops)), ]
    for (i in seq_len(nrow(df))) {
      cat(sprintf("  %-12s %-10s %8d %10.1f %10.2f\n",
                  nm, df$op[i], df$count[i], df$avg_us[i], df$total_us[i] / 1e3))
    }
  }
}

ctx <- sd_ctx(
  diffusion_model_path = file.path(models_dir, "flux1-dev-Q4_K_S.gguf"),
  vae_path             = file.path(models_dir, "ae.safetensors"),
  clip_l_path          = file.path(models_dir, "clip_l.safetensors"),
  t5xxl_path           = file.path(models_dir, "t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
  tensor_type_rules    = "first_stage_model=f16",
  n_threads            = 4L,
  model_type           = "flux",
  vae_decode_only      = TRUE,
  verbose              = FALSE
)

cat("\n=== 1 step ===\n")
t0 <- proc.time()
vk_out <- capture.output({
  imgs <- sd_generate(
    ctx,
    prompt        = "a red apple on a white table",
    width         = 768L, height = 768L,
    sample_steps  = 1L, seed = 42L,
    sample_method = SAMPLE_METHOD$EULER,
    scheduler     = SCHEDULER$DISCRETE
  )
}, type = "output")
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("wall time: %.2fs\n", elapsed))
parse_vk_timings(vk_out, top_n = 15)
