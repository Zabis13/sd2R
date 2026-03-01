library(sdR)

cat("=== sdR Generation Test ===\n\n")
print(sd_system_info())

# --- SD 1.5 ---
cat("\n--- Loading SD 1.5 ---\n")
ctx <- sd_ctx(
  "/mnt/Data2/DS_projects/sd_models/v1-5-pruned-emaonly.safetensors",
  n_threads = 4L
)
cat("Context created\n")

cat("\n--- Generating 512x512, 20 steps ---\n")
imgs <- sd_txt2img(
  ctx,
  prompt = "a cat sitting on a chair, oil painting",
  negative_prompt = "blurry, bad quality",
  width = 512L,
  height = 512L,
  sample_steps = 20L,
  cfg_scale = 7.0,
  seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE
)

cat("Generated", length(imgs), "image(s)\n")
cat("Size:", imgs[[1]]$width, "x", imgs[[1]]$height, "x", imgs[[1]]$channel, "\n")

sd_save_image(imgs[[1]], "/tmp/sdR_sd15_cat.png")
cat("Saved: /tmp/sdR_sd15_cat.png\n")

# Cleanup
rm(ctx)
gc()

cat("\n=== Done ===\n")
