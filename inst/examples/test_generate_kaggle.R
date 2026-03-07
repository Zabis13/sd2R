library(sdR)
library(png)
library(grid)

cat("=== sdR sd_generate() — Kaggle Test ===\n\n")
print(sd_system_info())

# Kaggle paths
model_path <- "/kaggle/input/sd-models/v1-5-pruned-emaonly.safetensors"
out_dir <- "/kaggle/working"

# Helper: save + display in notebook
show_image <- function(img, filename) {
  path <- file.path(out_dir, filename)
  sd_save_image(img, path)
  cat(sprintf("Saved: %s\n", path))
  img_data <- readPNG(path)
  grid.newpage()
  grid.raster(img_data)
}

# --- 1. Basic 512x512 (direct) ---
cat("\n--- 1. Basic 512x512, vram_gb=16 -> direct ---\n")
ctx <- sd_ctx(model_path, n_threads = 4L, model_type = "sd1", vram_gb = 16)
t0 <- proc.time()
imgs <- sd_generate(
  ctx,
  prompt = "a cat sitting on a chair, oil painting",
  negative_prompt = "blurry, bad quality",
  width = 512L, height = 512L,
  sample_steps = 20L, cfg_scale = 7.0, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(imgs), imgs[[1]]$width, imgs[[1]]$height, elapsed))
show_image(imgs[[1]], "sdR_gen_512.png")
rm(ctx); gc()

# --- 2. 1024x1024, vram_gb=8 -> auto tiled ---
cat("\n--- 2. 1024x1024, vram_gb=8 -> auto tiled ---\n")
ctx <- sd_ctx(model_path, n_threads = 4L, model_type = "sd1", vram_gb = 8)
t0 <- proc.time()
imgs_tiled <- sd_generate(
  ctx,
  prompt = "a vast mountain landscape, dramatic sky, photorealistic",
  negative_prompt = "blurry, bad quality, text",
  width = 1024L, height = 1024L,
  sample_steps = 20L, cfg_scale = 7.0, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE,
  vae_mode = "tiled"
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(imgs_tiled), imgs_tiled[[1]]$width, imgs_tiled[[1]]$height, elapsed))
show_image(imgs_tiled[[1]], "sdR_gen_tiled_1k.png")
rm(ctx); gc()

# --- 3. 2048x1024, vram_gb=8 -> auto highres fix ---
cat("\n--- 3. 2048x1024, vram_gb=8 -> auto highres fix ---\n")
ctx <- sd_ctx(model_path, n_threads = 4L, model_type = "sd1",
              vram_gb = 8, vae_decode_only = FALSE)
t0 <- proc.time()
imgs_hr <- sd_generate(
  ctx,
  prompt = "a panoramic mountain landscape, dramatic sky, photorealistic",
  negative_prompt = "blurry, bad quality, text",
  width = 2048L, height = 1024L,
  sample_steps = 20L, cfg_scale = 7.0, seed = 42L,
  hr_strength = 0.4,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(imgs_hr), imgs_hr[[1]]$width, imgs_hr[[1]]$height, elapsed))
show_image(imgs_hr[[1]], "sdR_gen_highres_panorama.png")
rm(ctx); gc()

# --- 4. img2img 512x512 (direct) ---
cat("\n--- 4. img2img 512x512, vram_gb=16 -> direct ---\n")
ctx <- sd_ctx(model_path, n_threads = 4L, model_type = "sd1",
              vram_gb = 16, vae_decode_only = FALSE)
t0 <- proc.time()
refined <- sd_generate(
  ctx,
  prompt = "a cat sitting on a chair, oil painting, masterpiece",
  init_image = imgs[[1]],
  negative_prompt = "blurry, bad quality",
  strength = 0.4,
  sample_steps = 20L, cfg_scale = 7.0, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(refined), refined[[1]]$width, refined[[1]]$height, elapsed))
show_image(refined[[1]], "sdR_gen_img2img.png")

# --- 5. 1024x1024, vram_gb=16 -> direct (no tiling) ---
cat("\n--- 5. 1024x1024, vram_gb=16 -> direct (no tiling) ---\n")
t0 <- proc.time()
imgs_1k <- sd_generate(
  ctx,
  prompt = "a vast mountain landscape, dramatic sky, photorealistic",
  negative_prompt = "blurry, bad quality, text",
  width = 1024L, height = 1024L,
  sample_steps = 20L, cfg_scale = 7.0, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(imgs_1k), imgs_1k[[1]]$width, imgs_1k[[1]]$height, elapsed))
show_image(imgs_1k[[1]], "sdR_gen_direct_1k.png")

# Cleanup
rm(ctx, imgs, imgs_tiled, imgs_hr, refined, imgs_1k)
gc()

cat("\n=== Done ===\n")
