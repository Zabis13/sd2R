library(sd2R)
library(later)

cat("=== Async test with R event loop (simulates Shiny) ===\n\n")

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

gen_count <- 0L
max_gens <- 3L

start_generation <- function() {
  gen_count <<- gen_count + 1L
  cat(sprintf("\n--- Starting async gen %d ---\n", gen_count))

  gc(full = TRUE)

  pf <- tempfile(fileext = ".json")
  sd2R:::sd_set_progress_file(pf)
  sd2R:::sd_generate_async(ctx, list(
    prompt = "a cat", width = 512L, height = 512L,
    sample_steps = 5L, seed = as.integer(gen_count),
    sample_method = 0L, scheduler = 6L
  ))

  poll_gen(pf)
}

poll_gen <- function(pf) {
  later::later(function() {
    # Simulate Shiny-like R heap activity between polls
    tmp_data <- replicate(10, list(a = rnorm(100), b = letters))
    rm(tmp_data)
    gc()

    s <- sd2R:::sd_generate_poll()
    if (s$done) {
      imgs <- sd2R:::sd_generate_result()
      cat(sprintf("Gen %d done: %dx%d\n", gen_count, imgs[[1]]$width, imgs[[1]]$height))
      sd2R:::sd_clear_progress_file()

      if (gen_count < max_gens) {
        start_generation()
      } else {
        cat("\nAll generations succeeded!\n")
        stopApp <- function() invisible(NULL)
      }
    } else {
      poll_gen(pf)
    }
  }, delay = 0.5)
}

start_generation()

# Run event loop (like Shiny does)
cat("Running event loop...\n")
while (gen_count < max_gens || !sd2R:::sd_generate_poll()$done) {
  later::run_now(timeoutSecs = 1)
}
# Final poll to collect last result
for (i in 1:5) later::run_now(timeoutSecs = 1)

cat("\nDone.\n")
