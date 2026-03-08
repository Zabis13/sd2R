library(sdR)

cat("=== sdR sd_generate() Test ===\n\n")
print(sd_system_info())

model_path <- "/mnt/Data2/DS_projects/sd_models/v1-5-pruned-emaonly.safetensors"

# --- 1. Basic 512x512 (direct, VRAM auto-detected) ---
cat("\n--- 1. Basic 512x512 -> direct ---\n")
ctx <- sd_ctx(model_path, n_threads = 4L, model_type = "sd1")
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
sd_save_image(imgs[[1]], "/tmp/sdR_gen_512.png")
cat("Saved: /tmp/sdR_gen_512.png\n")
rm(ctx); gc()

# --- 2. 1024x1024, forced tiled VAE ---
cat("\n--- 2. 1024x1024 -> tiled VAE ---\n")
ctx <- sd_ctx(model_path, n_threads = 4L, model_type = "sd1")
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
sd_save_image(imgs_tiled[[1]], "/tmp/sdR_gen_tiled_1k.png")
cat("Saved: /tmp/sdR_gen_tiled_1k.png\n")
rm(ctx); gc()

# --- 3. 2048x1024 -> auto highres fix (vae_decode_only=FALSE) ---
cat("\n--- 3. 2048x1024 -> auto highres fix ---\n")
ctx <- sd_ctx(model_path, n_threads = 4L, model_type = "sd1",
              vae_decode_only = FALSE)
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
sd_save_image(imgs_hr[[1]], "/tmp/sdR_gen_highres_panorama.png")
cat("Saved: /tmp/sdR_gen_highres_panorama.png\n")
rm(ctx); gc()

# --- 4. img2img 512x512 (direct) ---
cat("\n--- 4. img2img 512x512 -> direct ---\n")
ctx <- sd_ctx(model_path, n_threads = 4L, model_type = "sd1",
              vae_decode_only = FALSE)
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
sd_save_image(refined[[1]], "/tmp/sdR_gen_img2img.png")
cat("Saved: /tmp/sdR_gen_img2img.png\n")

# --- 5. 1024x1024 -> direct (auto-routed) ---
cat("\n--- 5. 1024x1024 -> direct ---\n")
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
sd_save_image(imgs_1k[[1]], "/tmp/sdR_gen_direct_1k.png")
cat("Saved: /tmp/sdR_gen_direct_1k.png\n")

# Cleanup
rm(ctx, imgs, imgs_tiled, imgs_hr, refined, imgs_1k)
gc()

cat("\n=== Done ===\n")
