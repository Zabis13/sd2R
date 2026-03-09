library(sdR)

cat("=== sdR sd_generate() Flux Test ===\n\n")
print(sd_system_info())

models_dir <- "/mnt/Data2/DS_projects/sd_models"

flux_ctx <- function(vae_decode_only = TRUE) {
  sd_ctx(diffusion_model_path = file.path(models_dir, "flux1-dev-Q4_K_S.gguf"),
         vae_path = file.path(models_dir, "ae.safetensors"),
         clip_l_path = file.path(models_dir, "clip_l.safetensors"),
         t5xxl_path = file.path(models_dir, "t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
         n_threads = 4L, model_type = "flux",
         vae_decode_only = vae_decode_only, verbose = FALSE)
}

# --- 1. Basic 768x768 (direct) ---
cat("\n--- 1. Basic 768x768 -> direct ---\n")
ctx <- flux_ctx()
t0 <- proc.time()
imgs <- sd_generate(
  ctx,
  prompt = "a cat sitting on a chair, oil painting",
  width = 768L, height = 768L,
  sample_steps = 8L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(imgs), imgs[[1]]$width, imgs[[1]]$height, elapsed))
sd_save_image(imgs[[1]], "/tmp/sdR_flux_768.png")
cat("Saved: /tmp/sdR_flux_768.png\n")
rm(ctx); gc()

# --- 2. 1024x1024, forced tiled VAE ---
cat("\n--- 2. 1024x1024 -> tiled VAE ---\n")
ctx <- flux_ctx()
t0 <- proc.time()
imgs_tiled <- sd_generate(
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
            length(imgs_tiled), imgs_tiled[[1]]$width, imgs_tiled[[1]]$height, elapsed))
sd_save_image(imgs_tiled[[1]], "/tmp/sdR_flux_tiled_1k.png")
cat("Saved: /tmp/sdR_flux_tiled_1k.png\n")
rm(ctx); gc()

# --- 3. 2048x1024 -> auto highres fix (vae_decode_only=FALSE) ---
cat("\n--- 3. 2048x1024 -> auto highres fix ---\n")
ctx <- flux_ctx(vae_decode_only = FALSE)
t0 <- proc.time()
imgs_hr <- sd_generate(
  ctx,
  prompt = "panoramic mountain landscape, snow-capped peaks, alpine meadow with wildflowers in foreground, winding river through valley, dramatic cumulus clouds lit by golden hour sunlight, volumetric light rays, mist in the valley, photorealistic, 8k, sharp detail",
  width = 2048L, height = 1024L,
  sample_steps = 8L, seed = 42L,
  hr_strength = 0.4,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(imgs_hr), imgs_hr[[1]]$width, imgs_hr[[1]]$height, elapsed))
sd_save_image(imgs_hr[[1]], "/tmp/sdR_flux_highres_panorama.png")
cat("Saved: /tmp/sdR_flux_highres_panorama.png\n")
rm(ctx); gc()

# --- 4. img2img 768x768 (direct) ---
cat("\n--- 4. img2img 768x768 -> direct ---\n")
ctx <- flux_ctx(vae_decode_only = FALSE)
t0 <- proc.time()
refined <- sd_generate(
  ctx,
  prompt = "a cat sitting on a chair, oil painting, masterpiece",
  init_image = imgs[[1]],
  strength = 0.4,
  sample_steps = 8L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(refined), refined[[1]]$width, refined[[1]]$height, elapsed))
sd_save_image(refined[[1]], "/tmp/sdR_flux_img2img.png")
cat("Saved: /tmp/sdR_flux_img2img.png\n")

# --- 5. 1024x1024 -> direct (auto-routed) ---
cat("\n--- 5. 1024x1024 -> direct ---\n")
t0 <- proc.time()
imgs_1k <- sd_generate(
  ctx,
  prompt = "a vast mountain landscape, dramatic sky, photorealistic",
  width = 1024L, height = 1024L,
  sample_steps = 8L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(imgs_1k), imgs_1k[[1]]$width, imgs_1k[[1]]$height, elapsed))
sd_save_image(imgs_1k[[1]], "/tmp/sdR_flux_direct_1k.png")
cat("Saved: /tmp/sdR_flux_direct_1k.png\n")

# Cleanup
rm(ctx, imgs, imgs_tiled, imgs_hr, refined, imgs_1k)
gc()

cat("\n=== Done ===\n")
