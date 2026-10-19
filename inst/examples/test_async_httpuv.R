library(sd2R)
library(later)
library(httpuv)
library(jsonlite)

cat("=== Async test with httpuv event loop (closest to Shiny) ===\n\n")

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

# Start a minimal httpuv server (like Shiny does)
srv <- httpuv::startServer("127.0.0.1", 9999, list(
  call = function(req) {
    list(status = 200L, headers = list("Content-Type" = "text/plain"), body = "ok")
  }
))
cat("httpuv server started on port 9999\n")

gen_count <- 0L
max_gens <- 3L
all_done <- FALSE

start_generation <- function() {
  gen_count <<- gen_count + 1L
  cat(sprintf("\n--- Async Gen %d ---\n", gen_count))
  gc(full = TRUE)

  pf <- tempfile(fileext = ".json")
  sd2R:::sd_set_progress_file(pf)
  sd2R:::sd_generate_async(ctx, list(
    prompt = "a cat", width = 100L, height = 100L,
    sample_steps = 5L, seed = as.integer(gen_count),
    sample_method = 0L, scheduler = 6L
  ))
  poll_gen(pf)
}

poll_gen <- function(pf) {
  later::later(function() {
    # Simulate Shiny JSON traffic
    for (i in 1:20) {
      x <- list(a = runif(100), b = sample(letters, 10))
      toJSON(x)
    }
    gc()

    s <- sd2R:::sd_generate_poll()
    if (s$done) {
      imgs <- sd2R:::sd_generate_result()
      cat(sprintf("Gen %d done: %dx%d\n", gen_count, imgs[[1]]$width, imgs[[1]]$height))
      sd2R:::sd_clear_progress_file()

      if (gen_count < max_gens) {
        start_generation()
      } else {
        all_done <<- TRUE
      }
    } else {
      poll_gen(pf)
    }
  }, delay = 0.5)
}

start_generation()

# Run httpuv event loop (this is what Shiny actually uses)
cat("Running httpuv event loop...\n")
while (!all_done) {
  httpuv::service(timeoutMs = 200)
  later::run_now(timeoutSecs = 0)
}
# Drain remaining callbacks
for (i in 1:10) {
  httpuv::service(timeoutMs = 100)
  later::run_now(timeoutSecs = 0)
}

httpuv::stopServer(srv)
cat("\nAll done. httpuv server stopped.\n")
