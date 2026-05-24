#!/usr/bin/env Rscript
# Flux DIAGNOSTIC: ENTIRE pipeline on CPU (Vulkan hidden via empty
# GGML_VK_VISIBLE_DEVICES -> vk device count 0 -> sd.cpp CPU fallback).
#   OK / structured -> bug is strictly in Vulkan DiT forward
#   still noise      -> bug is in shared logic / vendored sd.cpp (not Vulkan)
# Small (256px, 4 steps) since CPU is slow; partial denoise still distinguishable.
# Run: Rscript /mnt/Data2/DS_projects/sd2R/inst/examples/test_flux_full_cpu.R

library(sd2R)
models_dir <- "/mnt/Data2/DS_projects/sd_models"

ctx <- sd_ctx(
  diffusion_model_path = file.path(models_dir, "flux1-dev-Q4_K_S.gguf"),
  vae_path             = file.path(models_dir, "ae.safetensors"),
  clip_l_path          = file.path(models_dir, "clip_l.safetensors"),
  t5xxl_path           = file.path(models_dir, "t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
  n_threads            = 8L,
  model_type           = "flux",
  vae_decode_only      = FALSE,
  verbose              = TRUE
)

cat("\n--- Flux FULL CPU (256x256, 4 steps) ---\n")
imgs <- sd_generate(
  ctx,
  prompt        = "a cat sitting on a chair, oil painting",
  width         = 256L, height = 256L,
  sample_steps  = 4L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE
)
sd_save_image(imgs[[1]], "/tmp/sd2R_flux_full_cpu.png")
cat("Saved: /tmp/sd2R_flux_full_cpu.png\n")
