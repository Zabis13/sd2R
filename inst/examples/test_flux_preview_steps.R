#!/usr/bin/env Rscript
# Flux DIAGNOSTIC: dump per-step decoded latent (PREVIEW_PROJ) to PPM.
# Shows WHERE the pipeline diverges into noise:
#   - step 0/1 already noise   -> bad initial latent / conditioning (T5/CLIP)
#   - degrades step by step    -> DiT forward (Q4_K matmul / RoPE)
#   - latents OK, final noise  -> VAE decode
# Run: Rscript /mnt/Data2/DS_projects/sd2R/inst/examples/test_flux_preview_steps.R

library(sd2R)

models_dir <- "/mnt/Data2/DS_projects/sd_models"
PREFIX <- "/tmp/flux_step_"

# Wire the preview callback BEFORE generation. PROJ = fast linear latent->RGB
# projection (no VAE), so it isolates the latent from VAE bugs.
sd2R:::sd_set_preview_dump(PREFIX, "proj", 1L, TRUE, TRUE)

ctx <- sd_ctx(
  diffusion_model_path = file.path(models_dir, "flux1-dev-Q4_K_S.gguf"),
  vae_path             = file.path(models_dir, "ae.safetensors"),
  clip_l_path          = file.path(models_dir, "clip_l.safetensors"),
  t5xxl_path           = file.path(models_dir, "t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
  n_threads            = 4L,
  model_type           = "flux",
  vae_decode_only      = FALSE,
  verbose              = TRUE
)

cat("\n--- Flux 768x768, per-step preview dump ---\n")
imgs <- sd_generate(
  ctx,
  prompt        = "a cat sitting on a chair, oil painting",
  width         = 768L, height = 768L,
  sample_steps  = 10L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE,
  vae_mode      = "tiled"
)

sd2R:::sd_clear_preview_dump()
sd_save_image(imgs[[1]], "/tmp/sd2R_flux_quick.png")
cat("Saved final: /tmp/sd2R_flux_quick.png\n")
cat("Per-step PPMs: ls /tmp/flux_step_*.ppm\n")
