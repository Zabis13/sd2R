library(sd2R)

cat("=== Sync repeat test (2 generations, no async) ===\n\n")

models_dir <- "/mnt/Data3/DS_Data/sd_models"
ctx <- sd_ctx(
  diffusion_model_path = file.path(models_dir, "flux1-dev-Q4_K_S.gguf"),
  vae_path = file.path(models_dir, "ae.safetensors"),
  clip_l_path = file.path(models_dir, "clip_l.safetensors"),
  t5xxl_path = file.path(models_dir, "t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
  n_threads = 4L, model_type = "flux",
  vae_decode_only = TRUE, verbose = TRUE
)
cat("Model loaded\n\n")

for (i in 1:2) {
  cat(sprintf("--- Sync Gen %d ---\n", i))
  img <- sd_txt2img(ctx,
    prompt = "a cat",
    width = 512L, height = 512L,
    sample_steps = 5L, seed = 42L,
    sample_method = SAMPLE_METHOD$EULER, scheduler = SCHEDULER$SIMPLE
  )
  cat(sprintf("Gen %d done: %dx%d\n\n", i, img$width, img$height))
  gc(full = TRUE)
}
cat("All sync generations succeeded!\n")
