#!/usr/bin/env Rscript
# FLUX.1 load + minimal generate, to catch the Windows deadlock when uploading
# encoder weights (CLIP-L + T5-XXL) to non-host-visible GPU memory (RTX 3090).
#
# Diagnostics go through the ggmlR crash-survivable file logger (r_dbg_logf),
# enabled via the GGMLR_DBG_LOG env var. Plain stderr is macro-redirected to
# buffered REprintf under the R build and is LOST on abort() — so the file log
# is the only reliable channel for the [vk_w2d] / vk_set_tensor / vk_graph_compute
# markers. The last line in that file before the hang/crash is the stall point.
#
# Run (Windows):
#   Rscript test_vk_upload_debug.R C:/models 1 C:/tmp C:/models/ggmlr_dbg.log
# Args: models_dir  n_threads  out_dir  dbg_log_path

suppressMessages(library(sd2R))

args        <- commandArgs(trailingOnly = TRUE)
models_dir  <- if (length(args) >= 1) args[1] else "/mnt/Data2/DS_projects/sd_models"
n_threads   <- if (length(args) >= 2) as.integer(args[2]) else 1L
out_dir     <- if (length(args) >= 3) args[3] else tempdir()
dbg_log     <- if (length(args) >= 4) args[4] else file.path(out_dir, "ggmlr_dbg.log")

# Enable the crash-survivable file logger and start from a clean file.
Sys.setenv(GGMLR_DBG_LOG = dbg_log)
if (file.exists(dbg_log)) unlink(dbg_log)

logf <- function(fmt, ...) {
  cat(sprintf(fmt, ...))
  flush.console()                 # force flush — important on Windows
  flush(stderr())
}

logf("=== FLUX.1 upload/deadlock probe ===\n")
print(sd_system_info())
logf("Vulkan devices : %d\n", sd_vulkan_device_count())
logf("models_dir     : %s\n", models_dir)
logf("n_threads      : %d\n", n_threads)
logf("dbg log file   : %s   <-- tail this file after a hang/crash\n\n", dbg_log)

# --- 1. Load FLUX.1: this allocates CLIP-L + T5-XXL + diffusion on the GPU ---
logf("[step] sd_ctx() — loading FLUX.1 (CLIP-L + T5-XXL) ...\n")
ctx <- sd_ctx(
  diffusion_model_path = file.path(models_dir, "flux1-dev-Q4_K_S.gguf"),
  vae_path             = file.path(models_dir, "ae.safetensors"),
  clip_l_path          = file.path(models_dir, "clip_l.safetensors"),
  t5xxl_path           = file.path(models_dir, "t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
  n_threads            = n_threads,
  model_type           = "flux",
  vae_decode_only      = FALSE,
  verbose              = TRUE,
  device_layout        = "mono"
)
logf("[step] sd_ctx() DONE — context created\n\n")

# --- 2. Minimal generate: smallest viable resolution + 1 step ---
# Text encode (get_learned_condition) runs here and triggers encoder weight
# upload to the GPU — the suspected hang point.
logf("[step] sd_generate() — 256x256, 1 step (reaches text encode) ...\n")
imgs <- sd_generate(
  ctx,
  prompt        = "a cat",
  width         = 256L, height = 256L,
  sample_steps  = 1L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE,
  vae_mode      = "tiled"
)
logf("[step] sd_generate() DONE — %dx%d\n", imgs[[1]]$width, imgs[[1]]$height)

sd_save_image(imgs[[1]], file.path(out_dir, "flux1_probe.png"))
logf("[step] saved: %s\n", file.path(out_dir, "flux1_probe.png"))

rm(ctx, imgs); gc()
logf("\n=== Done (no deadlock) ===\n")
