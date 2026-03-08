library(sdR)

model_path <- "/mnt/Data2/DS_projects/sd_models/v1-5-pruned-emaonly.safetensors"

ctx <- sd_ctx(model_path, n_threads = 4L, model_type = "sd1",
              vae_decode_only = FALSE)

t0 <- proc.time()
imgs <- sd_generate(
  ctx,
  prompt = "Depict John Connor in a gritty post-nuclear wasteland, leading human rebels against Skynet's T-600 Terminators under a stormy sky, with explosions and hydrobots in the background, cinematic style like a movie poster.",
  negative_prompt = "blurry, bad quality, text, watermark, deformed",
  width = 1024L, height = 1024L,
  sample_steps = 10L, cfg_scale = 7.5, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE
)
elapsed <- (proc.time() - t0)[["elapsed"]]

cat(sprintf("Generated %dx%d in %.1fs\n", imgs[[1]]$width, imgs[[1]]$height, elapsed))
sd_save_image(imgs[[1]], "/tmp/sdR_terminator.png")
cat("Saved: /tmp/sdR_terminator.png\n")
