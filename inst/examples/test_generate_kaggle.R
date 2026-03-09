library(sdR)
library(png)
library(grid)

cat("=== sdR sd_generate() — Kaggle Test ===\n\n")
print(sd_system_info())

# Kaggle paths
models_dir <- "/kaggle/input/models/lbsbmsu/flux1-dev-q4-k-s/gguf/default/2"
out_dir <- "/kaggle/working"

# Model file paths
flux_diffusion <- file.path(models_dir, "flux1-dev-Q4_K_S.gguf")
flux_vae       <- file.path(models_dir, "ae.safetensors")
flux_clip_l    <- file.path(models_dir, "clip_l.safetensors")
flux_t5xxl     <- file.path(models_dir, "t5-v1_1-xxl-encoder-Q5_K_M.gguf")

# GPU count
n_gpu <- sd_vulkan_device_count()

# Helper: create Flux context
flux_ctx <- function(vae_decode_only = TRUE) {
  sd_ctx(diffusion_model_path = flux_diffusion,
         vae_path = flux_vae,
         clip_l_path = flux_clip_l,
         t5xxl_path = flux_t5xxl,
         n_threads = 4L, model_type = "flux",
         vae_decode_only = vae_decode_only, verbose = FALSE)
}

# Helper: save + display in notebook
show_image <- function(img, filename) {
  path <- file.path(out_dir, filename)
  sd_save_image(img, path)
  cat(sprintf("Saved: %s\n", path))
  img_data <- readPNG(path)
  grid.newpage()
  grid.raster(img_data)
}

# --- 1. Flux 768x768 (direct) ---
cat("\n--- 1. Flux 768x768 -> direct ---\n")
ctx <- flux_ctx()
t0 <- proc.time()
imgs_flux <- sd_generate(
  ctx,
  prompt = "a cat sitting on a chair, oil painting",
  width = 768L, height = 768L,
  sample_steps = 8L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(imgs_flux), imgs_flux[[1]]$width, imgs_flux[[1]]$height, elapsed))
show_image(imgs_flux[[1]], "sdR_flux_768.png")
rm(ctx); gc()

# --- 2. Flux 1024x1024, forced tiled VAE ---
cat("\n--- 2. Flux 1024x1024 -> tiled VAE ---\n")
ctx <- flux_ctx()
t0 <- proc.time()
imgs_flux_tiled <- sd_generate(
  ctx,
  prompt = "a vast mountain landscape, dramatic sky, photorealistic",
  width = 1024L, height = 1024L,
  sample_steps = 8L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE,
  vae_mode = "tiled"
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(imgs_flux_tiled), imgs_flux_tiled[[1]]$width, imgs_flux_tiled[[1]]$height, elapsed))
show_image(imgs_flux_tiled[[1]], "sdR_flux_tiled_1k.png")
rm(ctx); gc()

# --- 3. Flux 2048x1024 -> auto highres fix ---
cat("\n--- 3. Flux 2048x1024 -> auto highres fix ---\n")
ctx <- flux_ctx(vae_decode_only = FALSE)
t0 <- proc.time()
imgs_flux_hr <- sd_generate(
  ctx,
  prompt = "a panoramic mountain landscape, dramatic sky, photorealistic",
  width = 2048L, height = 1024L,
  sample_steps = 8L, seed = 42L,
  hr_strength = 0.4,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(imgs_flux_hr), imgs_flux_hr[[1]]$width, imgs_flux_hr[[1]]$height, elapsed))
show_image(imgs_flux_hr[[1]], "sdR_flux_highres_panorama.png")
rm(ctx); gc()

# --- 4. Flux img2img 768x768 (direct) ---
cat("\n--- 4. Flux img2img 768x768 -> direct ---\n")
ctx <- flux_ctx(vae_decode_only = FALSE)
t0 <- proc.time()
imgs_flux_i2i <- sd_generate(
  ctx,
  prompt = "a cat sitting on a chair, oil painting, masterpiece",
  init_image = imgs_flux[[1]],
  strength = 0.4,
  sample_steps = 8L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(imgs_flux_i2i), imgs_flux_i2i[[1]]$width, imgs_flux_i2i[[1]]$height, elapsed))
show_image(imgs_flux_i2i[[1]], "sdR_flux_img2img.png")

# --- 5. Flux 1024x1024 -> direct (auto-routed) ---
cat("\n--- 5. Flux 1024x1024 -> direct ---\n")
t0 <- proc.time()
imgs_flux_1k <- sd_generate(
  ctx,
  prompt = "a vast mountain landscape, dramatic sky, photorealistic",
  width = 1024L, height = 1024L,
  sample_steps = 8L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(imgs_flux_1k), imgs_flux_1k[[1]]$width, imgs_flux_1k[[1]]$height, elapsed))
show_image(imgs_flux_1k[[1]], "sdR_flux_direct_1k.png")
rm(ctx); gc()

# --- 6. Flux Multi-GPU (if available) ---
if (n_gpu > 1L) {
  cat(sprintf("\n--- 6. Flux Multi-GPU: %d Vulkan device(s) ---\n", n_gpu))
  multi_prompts <- c(
    "a cat in a garden, watercolor",
    "a dog on a beach, oil painting",
    "a bird in the sky, digital art",
    "a fish underwater, photorealistic"
  )
  t0 <- proc.time()
  imgs_flux_multi <- sd_generate_multi_gpu(
    diffusion_model_path = flux_diffusion,
    vae_path = flux_vae,
    clip_l_path = flux_clip_l,
    t5xxl_path = flux_t5xxl,
    prompts = multi_prompts,
    width = 768L, height = 768L,
    model_type = "flux",
    sample_steps = 8L,
    sample_method = SAMPLE_METHOD$EULER,
    scheduler = SCHEDULER$DISCRETE
  )
  elapsed <- (proc.time() - t0)[["elapsed"]]
  cat(sprintf("Multi-GPU: generated %d image(s) across %d GPUs in %.1fs\n",
              length(imgs_flux_multi), n_gpu, elapsed))
  for (i in seq_along(imgs_flux_multi)) {
    if (!inherits(imgs_flux_multi[[i]], "error")) {
      show_image(imgs_flux_multi[[i]], sprintf("sdR_flux_multi_gpu_%d.png", i))
    } else {
      cat(sprintf("  Image %d failed: %s\n", i, conditionMessage(imgs_flux_multi[[i]])))
    }
  }
  rm(imgs_flux_multi)
} else {
  cat("\n--- 6. Flux Multi-GPU: skipped (only 1 GPU) ---\n")
}

rm(imgs_flux, imgs_flux_tiled, imgs_flux_hr, imgs_flux_i2i, imgs_flux_1k)
gc()

cat("\n=== Done ===\n")
