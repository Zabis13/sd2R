library(sd2R)

cat("=== Async repeat test (5 generations with GC pressure) ===\n\n")

models_dir <- "/mnt/Data3/DS_Data/sd_models"
ctx <- sd_ctx(
  diffusion_model_path = file.path(models_dir, "flux1-dev-Q4_K_S.gguf"),
  vae_path = file.path(models_dir, "ae.safetensors"),
  clip_l_path = file.path(models_dir, "clip_l.safetensors"),
  t5xxl_path = file.path(models_dir, "t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
  n_threads = 4L, model_type = "flux",
  vae_decode_only = TRUE, verbose = FALSE
)
cat("Model loaded\n")

for (i in 1:5) {
  cat(sprintf("\n--- Async Gen %d ---\n", i))

  # Aggressive GC before each run (simulate Shiny)
  gc(full = TRUE)

  pf <- tempfile(fileext = ".json")
  sd2R:::sd_set_progress_file(pf)
  sd2R:::sd_generate_async(ctx, list(
    prompt = "a cat", width = 100L, height = 100L,
    sample_steps = 5L, seed = as.integer(i),
    sample_method = 0L, scheduler = 6L
  ))

  # Poll with GC pressure between polls (simulate Shiny event loop)
  repeat {
    Sys.sleep(0.3)
    gc()
    s <- sd2R:::sd_generate_poll()
    if (s$done) break
  }

  imgs <- sd2R:::sd_generate_result()
  cat(sprintf("Gen %d done: %dx%d\n", i, imgs[[1]]$width, imgs[[1]]$height))
  sd2R:::sd_clear_progress_file()

  # GC after result retrieval
  gc(full = TRUE)
}

cat("\nAll 5 async generations succeeded!\n")
