library(sd2R)

cat("=== Highres Fix Test ===\n\n")

model_path <- "/mnt/Data3/DS_Data/sd_models/v1-5-pruned-emaonly.safetensors"

cat("--- Loading SD 1.5 ---\n")
ctx <- sd_ctx(model_path, n_threads = 4L, model_type = "sd1", vae_decode_only = FALSE)
cat("Context created\n")

cat("\n--- Highres Fix 2048x2048 ---\n")
t0 <- proc.time()
img <- sd_highres_fix(
  ctx,
  prompt = "a vast mountain landscape, dramatic sky, photorealistic",
  negative_prompt = "blurry, bad quality, text",
  width = 2048L,
  height = 2048L,
  sample_steps = 20L,
  cfg_scale = 7.0,
  seed = 42L,
  hr_strength = 0.4,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE,
  vae_mode = "tiled"
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated: %dx%dx%d in %.1fs\n",
            img$width, img$height, img$channel, elapsed))
sd_save_image(img, "/tmp/sdR_highres_fix_2k.png")
cat("Saved: /tmp/sdR_highres_fix_2k.png\n")

cat("\n=== Done ===\n")
