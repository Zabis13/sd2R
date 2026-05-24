#!/usr/bin/env Rscript
# Flux.1 generate profiler — tests 1-5, per-op Vulkan timings
# Run: Rscript /mnt/Data2/DS_projects/sd2R/inst/examples/test_generate_flux.R

#Sys.setenv(GGML_VK_PERF_LOGGER = "1")

library(sd2R)

cat("=== sd2R sd_generate() Flux Test ===\n\n")
print(sd_system_info())

models_dir <- "/mnt/Data2/DS_projects/sd_models"
out_dir    <- "/tmp"

n_gpu <- sd_vulkan_device_count()
cat(sprintf("Vulkan devices: %d\n", n_gpu))

ctx <- sd_ctx(
  diffusion_model_path = file.path(models_dir, "flux1-dev-Q4_K_S.gguf"),
  vae_path             = file.path(models_dir, "ae.safetensors"),
  clip_l_path          = file.path(models_dir, "clip_l.safetensors"),
  t5xxl_path           = file.path(models_dir, "t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
  n_threads            = 4L,
  model_type           = "flux",
  vae_decode_only      = FALSE,
  verbose              = FALSE,
  device_layout        = if (n_gpu > 1L) "split_vae" else "mono"
)

elapsed <- numeric(5)

# --- 1. Flux 768x768 (direct) ---
cat("\n--- 1. Flux 768x768 -> direct ---\n")
sd_profile_start()
t0 <- proc.time()
imgs_1 <- sd_generate(
  ctx,
  prompt        = "a cat sitting on a chair, oil painting",
  width         = 768L, height = 768L,
  sample_steps  = 10L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE,
  vae_mode      = "tiled"
)
elapsed[1] <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("wall time: %.2fs | %dx%d\n", elapsed[1], imgs_1[[1]]$width, imgs_1[[1]]$height))
print(sd_profile_summary(sd_profile_get()))
sd_save_image(imgs_1[[1]], file.path(out_dir, "sd2R_flux_768.png"))
cat("Saved: /tmp/sd2R_flux_768.png\n")

# --- 2. Flux 1024x1024, tiled VAE ---
cat("\n--- 2. Flux 1024x1024 -> tiled VAE ---\n")
sd_profile_start()
t0 <- proc.time()
imgs_2 <- sd_generate(
  ctx,
  prompt = "Busy street in a vibrant Chinese quarter, street food vendors with steaming woks,
colorful lanterns hanging overhead, crowded market stalls with exotic fruits and
spices, pedestrians in casual clothing, neon signs in Chinese characters, wet
pavement reflections, steam rising from food carts, photorealistic, 8k,
hyperdetailed, street photography style, golden hour lighting",
  width         = 1024L, height = 1024L,
  sample_steps  = 10L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE,
  vae_mode      = "tiled"
)
elapsed[2] <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("wall time: %.2fs | %dx%d\n", elapsed[2], imgs_2[[1]]$width, imgs_2[[1]]$height))
print(sd_profile_summary(sd_profile_get()))
sd_save_image(imgs_2[[1]], file.path(out_dir, "sd2R_flux_tiled_1k.png"))
cat("Saved: /tmp/sd2R_flux_tiled_1k.png\n")

# --- 3. Flux 2048x1024 -> auto highres fix ---
cat("\n--- 3. Flux 2048x1024 -> auto highres fix ---\n")
sd_profile_start()
t0 <- proc.time()
imgs_3 <- sd_generate(
  ctx,
  prompt = "Bustling Chinatown street market, food stalls, dim sum vendors, crowded alley,
hanging red lanterns, neon signs, steam from cooking, photorealistic, 8k,
cinematic, shot on Sony A7R, f/8, sharp focus, high detail",
  width         = 2048L, height = 1024L,
  sample_steps  = 10L, seed = 42L,
  hr_strength   = 0.4,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE
)
elapsed[3] <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("wall time: %.2fs | %dx%d\n", elapsed[3], imgs_3[[1]]$width, imgs_3[[1]]$height))
print(sd_profile_summary(sd_profile_get()))
sd_save_image(imgs_3[[1]], file.path(out_dir, "sd2R_flux_highres_panorama.png"))
cat("Saved: /tmp/sd2R_flux_highres_panorama.png\n")

# --- 4. Flux img2img 768x768 ---
cat("\n--- 4. Flux img2img 768x768 ---\n")
sd_profile_start()
t0 <- proc.time()
imgs_4 <- sd_generate(
  ctx,
  prompt        = "a cat sitting on a chair, oil painting, masterpiece",
  init_image    = imgs_1[[1]],
  strength      = 0.4,
  sample_steps  = 10L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE,
  vae_mode      = "tiled"
)
elapsed[4] <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("wall time: %.2fs | %dx%d\n", elapsed[4], imgs_4[[1]]$width, imgs_4[[1]]$height))
print(sd_profile_summary(sd_profile_get()))
sd_save_image(imgs_4[[1]], file.path(out_dir, "sd2R_flux_img2img.png"))
cat("Saved: /tmp/sd2R_flux_img2img.png\n")

# --- 5. Flux 1024x1024 -> direct ---
cat("\n--- 5. Flux 1024x1024 -> direct ---\n")
sd_profile_start()
t0 <- proc.time()
imgs_5 <- sd_generate(
  ctx,
  prompt = "Busy traditional Japanese shopping street (shotengai), yakitori and ramen vendors,
paper lanterns, wooden shop signs in kanji, tourists and locals in yukata,
takoyaki stall with sizzling batter, narrow alley lined with izakayas, steam
from hot food, wet cobblestones, neon signs, photorealistic, 8k, hyperdetailed,
street photography, shot on Sony A7R, f/8, sharp focus",
  width         = 1024L, height = 1024L,
  sample_steps  = 10L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE,
  vae_mode      = "tiled"
)
elapsed[5] <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("wall time: %.2fs | %dx%d\n", elapsed[5], imgs_5[[1]]$width, imgs_5[[1]]$height))
print(sd_profile_summary(sd_profile_get()))
sd_save_image(imgs_5[[1]], file.path(out_dir, "sd2R_flux_direct_1k.png"))
cat("Saved: /tmp/sd2R_flux_direct_1k.png\n")

# --- Summary ---
cat("\n=== Summary ===\n")
cat(sprintf("1. 768x768  direct:           %.2fs\n", elapsed[1]))
cat(sprintf("2. 1024x1024 tiled VAE:        %.2fs\n", elapsed[2]))
cat(sprintf("3. 2048x1024 highres fix:      %.2fs\n", elapsed[3]))
cat(sprintf("4. img2img 768x768:            %.2fs\n", elapsed[4]))
cat(sprintf("5. 1024x1024 direct:           %.2fs\n", elapsed[5]))

rm(ctx, imgs_1, imgs_2, imgs_3, imgs_4, imgs_5)
gc()

cat("\n=== Done ===\n")
