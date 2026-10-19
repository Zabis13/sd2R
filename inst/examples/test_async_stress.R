library(sd2R)
library(jsonlite)

cat("=== Async stress test (heap pressure during generation) ===\n\n")

# Stress R heap like Shiny does (JSON serialization, allocation)
stress_r_heap <- function() {
  for (i in 1:100) {
    x <- list(a = runif(1000), b = letters, c = complex(real = 1:10))
    tmp <- toJSON(x)
    fromJSON(tmp)
  }
}

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

for (i in 1:3) {
  cat(sprintf("\n--- Stress Gen %d ---\n", i))

  pf <- tempfile(fileext = ".json")
  sd2R:::sd_set_progress_file(pf)
  sd2R:::sd_generate_async(ctx, list(
    prompt = "a cat", width = 512L, height = 512L,
    sample_steps = 5L, seed = as.integer(i),
    sample_method = 0L, scheduler = 6L
  ))

  # Poll with heavy R heap stress (simulates Shiny JSON + websocket activity)
  repeat {
    stress_r_heap()
    gc()
    Sys.sleep(0.2)
    s <- sd2R:::sd_generate_poll()
    if (s$done) break
  }

  res <- sd2R:::sd_generate_result()
  cat(sprintf("Gen %d done: %dx%d\n", i, res[[1]]$width, res[[1]]$height))
  sd2R:::sd_clear_progress_file()
  gc(full = TRUE)
}

cat("\nAll 3 stress generations succeeded!\n")
