#!/usr/bin/env Rscript
# Flux DIAGNOSTIC: swap T5 quant Q5_K_M -> Q3_K_S.
# If noise character changes/clears -> T5 quant path is the bug.
# If identical noise -> T5 quant not the cause (DiT or conditioning wiring).
# Run: Rscript /mnt/Data2/DS_projects/sd2R/inst/examples/test_flux_t5q3.R

library(sd2R)
models_dir <- "/mnt/Data2/DS_projects/sd_models"

ctx <- sd_ctx(
  diffusion_model_path = file.path(models_dir, "flux1-dev-Q4_K_S.gguf"),
  vae_path             = file.path(models_dir, "ae.safetensors"),
  clip_l_path          = file.path(models_dir, "clip_l.safetensors"),
  t5xxl_path           = file.path(models_dir, "t5-v1_1-xxl-encoder-Q3_K_S.gguf"),
  n_threads            = 4L,
  model_type           = "flux",
  vae_decode_only      = FALSE,
  verbose              = FALSE
)

cat("\n--- Flux with T5 Q3_K_S ---\n")
imgs <- sd_generate(
  ctx,
  prompt        = "a cat sitting on a chair, oil painting",
  width         = 768L, height = 768L,
  sample_steps  = 10L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE,
  vae_mode      = "tiled"
)
sd_save_image(imgs[[1]], "/tmp/sd2R_flux_t5q3.png")
cat("Saved: /tmp/sd2R_flux_t5q3.png\n")
