#!/usr/bin/env Rscript
# T5 encoding profiler — isolates text_encode_t5 cost across prompt lengths
# Run: Rscript /mnt/Data2/DS_projects/sd2R/inst/examples/test_t5_profile.R

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

.t5_time <- function(prompt) {
  sd_profile_start()
  sd_generate(ctx, prompt = prompt,
              width = 64L, height = 64L,
              sample_steps = 1L, seed = 42L,
              sample_method = SAMPLE_METHOD$EULER,
              scheduler     = SCHEDULER$DISCRETE)
  sd_profile_stop()
  prof <- sd_profile_summary(sd_profile_get())
  rows <- prof[prof$stage == "text_encode_t5" & !is.na(prof$duration_s), , drop = FALSE]
  if (nrow(rows) > 0L) sum(rows$duration_s) else NA_real_
}

prompts <- c(
  short  = "a cat",
  medium = "a red apple on a white table, studio lighting, 4k, photorealistic",
  long   = paste(
    "a hyperrealistic photo of a futuristic city at night, neon lights reflecting",
    "on wet streets, flying cars, enormous skyscrapers, cyberpunk style,",
    "ultra detailed, 8k resolution, dramatic lighting, foggy atmosphere,",
    "ray tracing, cinematic composition, bokeh depth of field"
  )
)

cat("=== T5 encoding profile ===\n")
cat(sprintf("%-8s  %6s  %6s\n", "prompt", "tokens", "t5_s"))

for (name in names(prompts)) {
  p   <- prompts[[name]]
  tok <- length(strsplit(p, "\\s+")[[1L]])   # приблизительно
  t5  <- .t5_time(p)
  cat(sprintf("%-8s  %6d  %6.3fs\n", name, tok, t5))
}
