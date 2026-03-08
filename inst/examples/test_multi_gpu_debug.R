library(sdR)
library(ggmlR)

cat("=== Multi-GPU Debug Test ===\n\n")

# Vulkan status
cat("--- Vulkan Status ---\n")
ggml_vulkan_status()

# Device count and memory
n_gpu <- tryCatch(ggml_vulkan_device_count(), error = function(e) 0L)
cat(sprintf("\nDetected %d Vulkan device(s)\n", n_gpu))

for (i in seq(0L, max(0L, n_gpu - 1L))) {
  mem <- tryCatch(ggml_vulkan_device_memory(i), error = function(e) NULL)
  if (!is.null(mem)) {
    cat(sprintf("  Device %d: total=%.1f GB, free=%.1f GB\n",
                i, mem$total / 1e9, mem$free / 1e9))
  } else {
    cat(sprintf("  Device %d: memory query failed\n", i))
  }
}

# Kaggle model path
model_path <- "/kaggle/input/sd-models/v1-5-pruned-emaonly.safetensors"
if (!file.exists(model_path)) {
  cat("\nModel not found, skipping generation tests\n")
  quit(status = 0)
}

# Test each GPU individually
cat("\n--- Single GPU tests ---\n")
for (dev in seq(0L, max(0L, n_gpu - 1L))) {
  cat(sprintf("\nTesting device %d...\n", dev))
  Sys.setenv(SD_VK_DEVICE = as.character(dev))

  ctx <- tryCatch(
    sd_ctx(model_path, n_threads = 4L, model_type = "sd1"),
    error = function(e) { cat(sprintf("  sd_ctx FAILED: %s\n", e$message)); NULL }
  )
  if (is.null(ctx)) next

  t0 <- proc.time()
  imgs <- tryCatch(
    sd_generate(ctx, prompt = "a red circle on white background",
                width = 256L, height = 256L,
                sample_steps = 5L, cfg_scale = 7.0, seed = 42L,
                sample_method = SAMPLE_METHOD$EULER,
                scheduler = SCHEDULER$DISCRETE),
    error = function(e) { cat(sprintf("  sd_generate FAILED: %s\n", e$message)); NULL }
  )
  elapsed <- (proc.time() - t0)[["elapsed"]]

  if (!is.null(imgs)) {
    img <- imgs[[1]]
    img_arr <- sd_image_to_array(img)
    # Check if image is empty (all zeros or near-zero)
    pixel_sum <- sum(img_arr)
    pixel_mean <- mean(img_arr)
    cat(sprintf("  Device %d: %dx%d in %.1fs, pixel_sum=%.0f, pixel_mean=%.4f %s\n",
                dev, img$width, img$height, elapsed, pixel_sum, pixel_mean,
                if (pixel_mean < 0.01) "** EMPTY **" else "OK"))

    out_path <- sprintf("/kaggle/working/debug_gpu%d.png", dev)
    sd_save_image(img, out_path)
    cat(sprintf("  Saved: %s (%d bytes)\n", out_path, file.info(out_path)$size))
  }

  rm(ctx); gc()
}

# Reset to default device
Sys.unsetenv("SD_VK_DEVICE")

cat("\n=== Done ===\n")
