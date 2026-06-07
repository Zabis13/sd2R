#!/usr/bin/env Rscript
# FLUX.2 (Klein / 4B / 9B) generate profiler — tests 1-5, per-op Vulkan timings
# Run: Rscript /mnt/Data2/DS_projects/sd2R/inst/examples/test_generate_flux2.R
#
# FLUX.2 specifics vs FLUX.1:
#   * model_type = "flux2" (auto-detected from tensors in C++; the R hint only
#     drives native tile/resolution and the cfg_scale default)
#   * guidance-distilled -> cfg_scale auto-defaults to 1.0 (same as flux)
#   * text encoder is an LLM: FLUX.2 Klein uses Qwen3, full FLUX.2 uses
#     Mistral-Small (conditioner.hpp:1623-1627). Pass it via llm_path, NOT
#     t5xxl_path. Diffusion, VAE and the LLM encoder are three separate files.
#
# Set SD2R_DEBUG_VAE=1 to trace the auto VAE-tiling decision (useful for the
# 2048x1024 highres-fix path that can OOM the VAE compute buffer).
# Sys.setenv(SD2R_DEBUG_VAE = "1")

library(sd2R)

# ---- Paths (edit to match your setup) -------------------------------------
models_dir <- "/mnt/Data2/DS_projects/sd_models"
out_dir    <- "/tmp"

# FLUX.2 Klein is distributed as a *diffusion-only* file with bare tensor
# names (double_blocks.*, img_in.*, ...). sd.cpp only adds the
# "model.diffusion_model." prefix (needed for version detection) when the file
# is loaded via diffusion_model_path, NOT via model_path — so use the former.
# The VAE and the LLM text encoder are separate files (NOT bundled).
diffusion_file <- file.path(models_dir, "flux-2-klein-4b.safetensors")

# FLUX.2 has its own VAE (different from the FLUX.1 ae.safetensors).
vae_file       <- file.path(models_dir, "flux2-vae.safetensors")

# LLM text encoder: FLUX.2 Klein uses Qwen3 (full FLUX.2 would use Mistral).
llm_file       <- file.path(models_dir, "Qwen3-4B-Q4_K_S.gguf")

# Alternatively, a true single-file model (everything bundled) can be loaded
# via model_path. Set this and leave diffusion_file = NULL to use that path.
model_file     <- NULL

# ---------------------------------------------------------------------------

cat("=== sd2R sd_generate() FLUX.2 Test ===\n\n")
print(sd_system_info())

n_gpu <- sd_vulkan_device_count()
cat(sprintf("Vulkan devices: %d\n", n_gpu))

# Build sd_ctx() args: single-file via model_path, else split components.
ctx_args <- list(
  n_threads       = 4L,
  model_type      = "flux2",
  vae_decode_only = FALSE,
  verbose         = FALSE,
  device_layout   = if (n_gpu > 1L) "split_vae" else "mono"
)
if (!is.null(model_file)) {
  # True single-file model (bundles diffusion + encoder + VAE).
  ctx_args$model_path <- model_file
} else if (!is.null(diffusion_file)) {
  # Diffusion-only file: MUST go through diffusion_model_path so sd.cpp adds
  # the "model.diffusion_model." prefix and can detect VERSION_FLUX2_KLEIN.
  ctx_args$diffusion_model_path <- diffusion_file
  if (!is.null(vae_file)) ctx_args$vae_path <- vae_file
  if (!is.null(llm_file)) ctx_args$llm_path <- llm_file  # Qwen3 / Mistral slot
} else {
  stop("Set diffusion_file (split layout) or model_file (single-file).")
}

ctx <- do.call(sd_ctx, ctx_args)

elapsed <- numeric(5)

# --- 1. FLUX.2 768x768 (direct) ---
cat("\n--- 1. FLUX.2 768x768 -> direct ---\n")
sd_profile_start()
t0 <- proc.time()
imgs_1 <- sd_generate(
  ctx,
  prompt        = "a cat sitting on a chair, oil painting",
  width         = 768L, height = 768L,
  sample_steps  = 4L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE,
  vae_mode      = "tiled"
)
elapsed[1] <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("wall time: %.2fs | %dx%d\n", elapsed[1], imgs_1[[1]]$width, imgs_1[[1]]$height))
print(sd_profile_summary(sd_profile_get()))
sd_save_image(imgs_1[[1]], file.path(out_dir, "sd2R_flux2_768.png"))
cat("Saved: /tmp/sd2R_flux2_768.png\n")

# --- 2. FLUX.2 1024x1024, tiled VAE ---
cat("\n--- 2. FLUX.2 1024x1024 -> tiled VAE ---\n")
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
  sample_steps  = 4L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE,
  vae_mode      = "tiled"
)
elapsed[2] <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("wall time: %.2fs | %dx%d\n", elapsed[2], imgs_2[[1]]$width, imgs_2[[1]]$height))
print(sd_profile_summary(sd_profile_get()))
sd_save_image(imgs_2[[1]], file.path(out_dir, "sd2R_flux2_tiled_1k.png"))
cat("Saved: /tmp/sd2R_flux2_tiled_1k.png\n")

# --- 3. FLUX.2 2048x1024 -> auto highres fix ---
cat("\n--- 3. FLUX.2 2048x1024 -> auto highres fix ---\n")
sd_profile_start()
t0 <- proc.time()
imgs_3 <- sd_generate(
  ctx,
  prompt = "Bustling Chinatown street market, food stalls, dim sum vendors, crowded alley,
hanging red lanterns, neon signs, steam from cooking, photorealistic, 8k,
cinematic, shot on Sony A7R, f/8, sharp focus, high detail",
  width         = 2048L, height = 1024L,
  sample_steps  = 4L, seed = 42L,
  hr_strength   = 0.4,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE
)
elapsed[3] <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("wall time: %.2fs | %dx%d\n", elapsed[3], imgs_3[[1]]$width, imgs_3[[1]]$height))
print(sd_profile_summary(sd_profile_get()))
sd_save_image(imgs_3[[1]], file.path(out_dir, "sd2R_flux2_highres_panorama.png"))
cat("Saved: /tmp/sd2R_flux2_highres_panorama.png\n")

# --- 4. FLUX.2 img2img 768x768 ---
cat("\n--- 4. FLUX.2 img2img 768x768 ---\n")
sd_profile_start()
t0 <- proc.time()
imgs_4 <- sd_generate(
  ctx,
  prompt        = "a cat sitting on a chair, oil painting, masterpiece",
  init_image    = imgs_1[[1]],
  strength      = 0.4,
  sample_steps  = 4L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE,
  vae_mode      = "tiled"
)
elapsed[4] <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("wall time: %.2fs | %dx%d\n", elapsed[4], imgs_4[[1]]$width, imgs_4[[1]]$height))
print(sd_profile_summary(sd_profile_get()))
sd_save_image(imgs_4[[1]], file.path(out_dir, "sd2R_flux2_img2img.png"))
cat("Saved: /tmp/sd2R_flux2_img2img.png\n")

# --- 5. FLUX.2 1024x1024 -> direct ---
cat("\n--- 5. FLUX.2 1024x1024 -> direct ---\n")
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
  sample_steps  = 4L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE,
  vae_mode      = "tiled"
)
elapsed[5] <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("wall time: %.2fs | %dx%d\n", elapsed[5], imgs_5[[1]]$width, imgs_5[[1]]$height))
print(sd_profile_summary(sd_profile_get()))
sd_save_image(imgs_5[[1]], file.path(out_dir, "sd2R_flux2_direct_1k.png"))
cat("Saved: /tmp/sd2R_flux2_direct_1k.png\n")

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
