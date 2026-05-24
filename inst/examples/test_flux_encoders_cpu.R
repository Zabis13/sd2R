#!/usr/bin/env Rscript
# Flux DIAGNOSTIC: text encoders (T5+CLIP) on CPU, DiT+VAE on Vulkan.
#   OK    -> bug is in Vulkan conditioning (T5/CLIP on GPU produce garbage)
#   noise -> conditioning fine; bug is in Vulkan DiT forward (Q4_K/RoPE/attn)
# Run: Rscript /mnt/Data2/DS_projects/sd2R/inst/examples/test_flux_encoders_cpu.R

library(sd2R)
models_dir <- "/mnt/Data2/DS_projects/sd_models"

ctx <- sd_ctx(
  diffusion_model_path = file.path(models_dir, "flux1-dev-Q4_K_S.gguf"),
  vae_path             = file.path(models_dir, "ae.safetensors"),
  clip_l_path          = file.path(models_dir, "clip_l.safetensors"),
  t5xxl_path           = file.path(models_dir, "t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
  n_threads            = 4L,
  model_type           = "flux",
  vae_decode_only      = FALSE,
  device_layout        = "encoders_cpu",
  verbose              = TRUE
)

cat("\n--- Flux: encoders on CPU, DiT+VAE on Vulkan ---\n")
imgs <- sd_generate(
  ctx,
  prompt        = "a cat sitting on a chair, oil painting",
  width         = 768L, height = 768L,
  sample_steps  = 10L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE,
  vae_mode      = "tiled"
)
sd_save_image(imgs[[1]], "/tmp/sd2R_flux_encoders_cpu.png")
cat("Saved: /tmp/sd2R_flux_encoders_cpu.png\n")
