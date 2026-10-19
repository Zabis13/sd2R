#!/usr/bin/env Rscript
# VAE decode benchmark with Vulkan per-op profiling
# Run: Rscript /mnt/Data2/DS_projects/sd2R/inst/examples/test_vae_f16.R

Sys.setenv(GGML_VK_PERF_LOGGER = "1")

library(sd2R)

models_dir <- "/mnt/Data3/DS_Data/sd_models"

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

sd_profile_start()
t0 <- proc.time()

imgs <- sd_generate(
  ctx,
  prompt        = "a red apple on a white table, studio photo",
  width         = 768L, height = 768L,
  sample_steps  = 4L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE
)

sd_profile_stop()
cat(sprintf("\nwall time: %.2fs\n", (proc.time() - t0)[["elapsed"]]))
print(sd_profile_summary(sd_profile_get()))
