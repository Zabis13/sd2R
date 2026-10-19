library(sd2R)

cat("=== Tiled-only test ===\n\n")

model_path <- "/mnt/Data3/DS_Data/sd_models/v1-5-pruned-emaonly.safetensors"

cat("--- Loading SD 1.5 ---\n")
ctx <- sd_ctx(model_path, n_threads = 4L, model_type = "sd1")
cat("Context created\n")

cat("\n--- Tiled sampling (MultiDiffusion) 2048x2048 ---\n")
t0 <- proc.time()
imgs_tiled <- sd_txt2img_tiled(
  ctx,
  prompt = "a vast mountain landscape, dramatic sky, photorealistic",
  negative_prompt = "blurry, bad quality, text",
  width = 2048L,
  height = 2048L,
  sample_tile_size = 96L,
  sample_tile_overlap = 0.25,
  sample_steps = 20L,
  cfg_scale = 7.0,
  seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE,
  vae_mode = "tiled"
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%dx%d in %.1fs\n",
            length(imgs_tiled), imgs_tiled[[1]]$width, imgs_tiled[[1]]$height,
            imgs_tiled[[1]]$channel, elapsed))

cat("\n=== Done ===\n")
