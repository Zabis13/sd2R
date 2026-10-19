#!/usr/bin/env Rscript
# Flux.1 quick profiler — 768x768, 10 steps
# Run: Rscript /mnt/Data2/DS_projects/sd2R/inst/examples/test_generate_flux_quick.R

library(sd2R)

models_dir <- "/mnt/Data3/DS_Data/sd_models"

ctx <- sd_ctx(
  diffusion_model_path = file.path(models_dir, "flux1-dev-Q4_K_S.gguf"),
  vae_path             = file.path(models_dir, "ae.safetensors"),
  clip_l_path          = file.path(models_dir, "clip_l.safetensors"),
  t5xxl_path           = file.path(models_dir, "t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
  n_threads            = 4L,
  model_type           = "flux",
  vae_decode_only      = FALSE,
  verbose              = FALSE
)

cat("\n--- Flux 768x768 -> direct ---\n")
sd_profile_start()
t0 <- proc.time()
imgs <- sd_generate(
  ctx,
  prompt        = "a cat sitting on a chair, oil painting",
  width         = 768L, height = 768L,
  sample_steps  = 10L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE,
  vae_mode      = "tiled"
)
elapsed <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("wall time: %.2fs | %dx%d\n", elapsed, imgs[[1]]$width, imgs[[1]]$height))
print(sd_profile_summary(sd_profile_get()))
sd_save_image(imgs[[1]], "/tmp/sd2R_flux_quick.png")
cat("Saved: /tmp/sd2R_flux_quick.png\n")
