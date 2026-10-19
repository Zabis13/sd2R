#!/usr/bin/env Rscript
# FLUX.3 Image Klein generate profiler — tests 1-5 (as FLUX.2) + edit with a
# ref image + low-level step-wise API checks
# Run: Rscript /mnt/Data2/DS_projects/sd2R/inst/examples/test_generate_flux3.R
#
# FLUX.3 specifics vs FLUX.2:
#   * model_type = "flux3" (auto-detected from tensors in C++; the R hint only
#     drives native tile/resolution and the cfg_scale / steps defaults)
#   * defaults: cfg_scale 1.0, 16 steps, euler (applied when left at 7.0 / 20)
#   * own DiT, FLUX.2 VAE, text encoder = Qwen3-VL 4B (LLM, via llm_path)
#   * up to 10 reference images (edit)
#   * flow-map (distilled Klein) checkpoints need the next sigma as target
#     timestep: sd_denoise_step(..., sigma_next =) — test 7; the no-target guard is in test_flux3_stepwise.R
#
# Set SD2R_DEBUG_VAE=1 to trace the auto VAE-tiling decision.
# Sys.setenv(SD2R_DEBUG_VAE = "1")

library(sd2R)

# ---- Paths (edit to match your setup) -------------------------------------
models_dir <- "/mnt/Data3/DS_Data/sd_models"
out_dir    <- "/tmp"

# Diffusion-only file: load via diffusion_model_path (see test_generate_flux2.R).
diffusion_file <- file.path(models_dir, "flux3_image_klein_sd_Q8_0.gguf")
# diffusion_file <- file.path(models_dir, "flux3-image-klein-bf16_comfy.safetensors")

# FLUX.3 uses the FLUX.2 VAE.
vae_file       <- file.path(models_dir, "flux2-vae.safetensors")

# LLM text encoder: Qwen3-VL 4B (NOT the plain Qwen3-4B used by FLUX.2 Klein).
llm_file       <- file.path(models_dir, "Qwen3VL-4B-Instruct-Q4_K_M.gguf")

model_file     <- NULL

# ---------------------------------------------------------------------------

for (f in c(diffusion_file, vae_file, llm_file)) {
  if (!is.null(f) && !file.exists(f)) stop("File not found: ", f)
}

cat("=== sd2R sd_generate() FLUX.3 Test ===\n\n")
print(sd_system_info())

n_gpu <- sd_vulkan_device_count()
cat(sprintf("Vulkan devices: %d\n", n_gpu))

ctx_args <- list(
  n_threads       = 4L,
  model_type      = "flux3",
  vae_decode_only = FALSE,
  verbose         = FALSE,
  device_layout   = if (n_gpu > 1L) "split_vae" else "mono"
)
if (!is.null(model_file)) {
  ctx_args$model_path <- model_file
} else if (!is.null(diffusion_file)) {
  ctx_args$diffusion_model_path <- diffusion_file
  if (!is.null(vae_file)) ctx_args$vae_path <- vae_file
  if (!is.null(llm_file)) ctx_args$llm_path <- llm_file
} else {
  stop("Set diffusion_file (split layout) or model_file (single-file).")
}

ctx <- do.call(sd_ctx, ctx_args)

elapsed <- numeric(7)

# --- 1. FLUX.3 768x768 (direct, default steps/cfg) ---
cat("\n--- 1. FLUX.3 768x768 -> direct (default 16 steps, cfg 1.0) ---\n")
sd_profile_start()
t0 <- proc.time()
imgs_1 <- sd_generate(
  ctx,
  prompt        = "a cat sitting on a chair, oil painting",
  width         = 768L, height = 768L,
  seed          = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE,
  vae_mode      = "normal"
)
elapsed[1] <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("wall time: %.2fs | %dx%d\n", elapsed[1], imgs_1[[1]]$width, imgs_1[[1]]$height))
print(sd_profile_summary(sd_profile_get()))
sd_save_image(imgs_1[[1]], file.path(out_dir, "sd2R_flux3_768.png"))
cat("Saved: /tmp/sd2R_flux3_768.png\n")

# --- 2. FLUX.3 1024x1024, tiled VAE (512 px tiles) ---
# Same prompt/seed as test 5 (normal VAE) -> 2 vs 5 compares only the VAE mode.
# The core's verbose log is captured to check that tiling really engaged
# (image larger than the tile -> more than one tile).
cat("\n--- 2. FLUX.3 1024x1024 -> tiled VAE (512 px tiles) ---\n")
prompt_city <- "Busy street in a vibrant Chinese quarter, street food vendors with steaming woks,
colorful lanterns hanging overhead, crowded market stalls with exotic fruits and
spices, pedestrians in casual clothing, neon signs in Chinese characters, wet
pavement reflections, steam rising from food carts, photorealistic, 8k,
hyperdetailed, street photography style, golden hour lighting"
sd_profile_start()
t0 <- proc.time()
sd2R:::sd_set_verbose(TRUE)
log_2 <- capture.output(imgs_2 <- sd_generate(
  ctx,
  prompt        = prompt_city,
  width         = 1024L, height = 1024L,
  seed          = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE,
  vae_mode      = "tiled",
  vae_tile_size = 512L
))
sd2R:::sd_set_verbose(FALSE)
elapsed[2] <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("wall time: %.2fs | %dx%d\n", elapsed[2], imgs_2[[1]]$width, imgs_2[[1]]$height))

# Tiling check: "VAE decode tile size: 512x512 pixels (...)" + "processing N tiles", N > 1
tile_line <- grep("VAE decode tile size", log_2, value = TRUE)
n_tiles   <- as.integer(sub(".*processing ([0-9]+) tiles.*", "\\1",
                            grep("processing [0-9]+ tiles", log_2, value = TRUE)))
tiling_ok <- length(tile_line) > 0 && grepl("512x512 pixels", tile_line[1]) &&
             length(n_tiles) > 0 && max(n_tiles) > 1
cat(sprintf("tiling: %s | tiles: %s -> %s\n",
            if (length(tile_line)) trimws(tile_line[1]) else "<no tile log line>",
            if (length(n_tiles)) paste(n_tiles, collapse = ",") else "?",
            if (tiling_ok) "ENGAGED (ok)" else "NOT ENGAGED (FAIL)"))
print(sd_profile_summary(sd_profile_get()))
sd_save_image(imgs_2[[1]], file.path(out_dir, "sd2R_flux3_tiled_1k.png"))
cat("Saved: /tmp/sd2R_flux3_tiled_1k.png\n")

# --- 3. FLUX.3 2048x1024 -> auto highres fix ---
cat("\n--- 3. FLUX.3 2048x1024 -> auto highres fix ---\n")
sd_profile_start()
t0 <- proc.time()
imgs_3 <- sd_generate(
  ctx,
  prompt = "Bustling Chinatown street market, food stalls, dim sum vendors, crowded alley,
hanging red lanterns, neon signs, steam from cooking, photorealistic, 8k,
cinematic, shot on Sony A7R, f/8, sharp focus, high detail",
  width         = 2048L, height = 1024L,
  seed          = 42L,
  hr_strength   = 0.4,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE
)
elapsed[3] <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("wall time: %.2fs | %dx%d\n", elapsed[3], imgs_3[[1]]$width, imgs_3[[1]]$height))
print(sd_profile_summary(sd_profile_get()))
sd_save_image(imgs_3[[1]], file.path(out_dir, "sd2R_flux3_highres_panorama.png"))
cat("Saved: /tmp/sd2R_flux3_highres_panorama.png\n")

# --- 4. FLUX.3 img2img 768x768 ---
cat("\n--- 4. FLUX.3 img2img 768x768 ---\n")
sd_profile_start()
t0 <- proc.time()
imgs_4 <- sd_generate(
  ctx,
  prompt        = "a cat sitting on a chair, oil painting, masterpiece",
  init_image    = imgs_1[[1]],
  strength      = 0.4,
  seed          = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE,
  vae_mode      = "normal"
)
elapsed[4] <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("wall time: %.2fs | %dx%d\n", elapsed[4], imgs_4[[1]]$width, imgs_4[[1]]$height))
print(sd_profile_summary(sd_profile_get()))
sd_save_image(imgs_4[[1]], file.path(out_dir, "sd2R_flux3_img2img.png"))
cat("Saved: /tmp/sd2R_flux3_img2img.png\n")

# --- 5. FLUX.3 1024x1024 -> normal VAE (same prompt/seed as test 2) ---
cat("\n--- 5. FLUX.3 1024x1024 -> normal VAE ---\n")
sd_profile_start()
t0 <- proc.time()
imgs_5 <- sd_generate(
  ctx,
  prompt        = prompt_city,
  width         = 1024L, height = 1024L,
  seed          = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE,
  vae_mode      = "normal"
)
elapsed[5] <- (proc.time() - t0)[["elapsed"]]
sd_profile_stop()
cat(sprintf("wall time: %.2fs | %dx%d\n", elapsed[5], imgs_5[[1]]$width, imgs_5[[1]]$height))
print(sd_profile_summary(sd_profile_get()))
sd_save_image(imgs_5[[1]], file.path(out_dir, "sd2R_flux3_direct_1k.png"))
cat("Saved: /tmp/sd2R_flux3_direct_1k.png\n")

# Tiled (2) vs normal (5), same prompt/seed: tiles are blended over the overlap,
# so a small but non-zero difference is expected. 0 means tiling did not run;
# a large value points to seams. Threshold: mean |diff| of 8-bit pixels.
TILED_VS_NORMAL_MAX_MEAN_DIFF <- 2.0
px_2 <- as.integer(imgs_2[[1]]$data)
px_5 <- as.integer(imgs_5[[1]]$data)
diff_25     <- mean(abs(px_2 - px_5))
max_diff_25 <- max(abs(px_2 - px_5))
cat(sprintf("tiled vs normal: mean |diff| %.3f (max %d), threshold (0, %.1f] -> %s\n",
            diff_25, max_diff_25, TILED_VS_NORMAL_MAX_MEAN_DIFF,
            if (diff_25 == 0) "IDENTICAL (tiling did not run, FAIL)"
            else if (diff_25 <= TILED_VS_NORMAL_MAX_MEAN_DIFF) "OK"
            else "TOO DIFFERENT (seams?, FAIL)"))

# --- 6. FLUX.3 edit with a reference image ---
cat("\n--- 6. FLUX.3 edit 768x768 (ref image = test 1) ---\n")
cat(sprintf("supports ref images: %s\n", sd_supports_ref_images(ctx)))
t0 <- proc.time()
imgs_6 <- sd_generate_multiref(
  ctx,
  prompt        = "turn the cat into a dog, keep the chair and the oil painting style",
  refs          = list(imgs_1[[1]]),
  width         = 768L, height = 768L,
  sample_steps  = 16L, cfg_scale = 1.0, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler     = SCHEDULER$DISCRETE
)
elapsed[6] <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("wall time: %.2fs | %dx%d\n", elapsed[6], imgs_6[[1]]$width, imgs_6[[1]]$height))
sd_save_image(imgs_6[[1]], file.path(out_dir, "sd2R_flux3_edit.png"))
cat("Saved: /tmp/sd2R_flux3_edit.png\n")

# --- 7. Low-level: sd_sample (C++ loop) vs sd_sample_stepwise (R loop) ---
# Same explicit noise -> Euler must match (stepwise passes sigma_next =
# sigmas[i + 1], the C++ sampler uses sigmas[step] as target timestep).
cat("\n--- 7. Low-level: sd_sample vs sd_sample_stepwise (768x768, Euler) ---\n")
prompt_ll <- "a cat sitting on a chair, oil painting"
cond  <- sd_encode_text(ctx, prompt_ll, width = 768L, height = 768L)
shape <- sd_encode_image(ctx, imgs_1[[1]])$ne[1:3]
cat(sprintf("latent shape: %s\n", paste(shape, collapse = " x ")))
set.seed(42L)
noise <- list(type = SD_TYPE$F32, ne = as.integer(c(shape, 1L)),
              data = stats::rnorm(prod(as.numeric(shape))))

t0 <- proc.time()
lat_loop <- sd_sample(ctx, cond, noise = noise,
                      sample_method = SAMPLE_METHOD$EULER,
                      scheduler = SCHEDULER$DISCRETE,
                      sample_steps = 16L, cfg_scale = 1.0)
t_loop <- (proc.time() - t0)[["elapsed"]]

t0 <- proc.time()
lat_step <- sd_sample_stepwise(ctx, cond, noise = noise,
                               width = 768L, height = 768L,
                               sample_method = SAMPLE_METHOD$EULER,
                               scheduler = SCHEDULER$DISCRETE,
                               sample_steps = 16L, cfg_scale = 1.0)
t_step <- (proc.time() - t0)[["elapsed"]]
elapsed[7] <- t_loop + t_step

max_diff <- max(abs(lat_loop$data - lat_step$data))
cat(sprintf("sd_sample: %.2fs | stepwise: %.2fs | max |diff| latent: %.3g -> %s\n",
            t_loop, t_step, max_diff, if (max_diff < 1e-3) "MATCH" else "MISMATCH"))
img_loop <- sd_decode_latent(ctx, lat_loop)
img_step <- sd_decode_latent(ctx, lat_step)
sd_save_image(img_loop, file.path(out_dir, "sd2R_flux3_ll_loop.png"))
sd_save_image(img_step, file.path(out_dir, "sd2R_flux3_ll_stepwise.png"))
cat("Saved: /tmp/sd2R_flux3_ll_loop.png, /tmp/sd2R_flux3_ll_stepwise.png\n")

# --- Summary ---
cat("\n=== Summary ===\n")
cat(sprintf("1. 768x768  direct:           %.2fs\n", elapsed[1]))
cat(sprintf("2. 1024x1024 tiled VAE:        %.2fs  (tiling %s)\n", elapsed[2],
            if (tiling_ok) "engaged" else "NOT engaged"))
cat(sprintf("3. 2048x1024 highres fix:      %.2fs\n", elapsed[3]))
cat(sprintf("4. img2img 768x768:            %.2fs\n", elapsed[4]))
cat(sprintf("5. 1024x1024 normal VAE:       %.2fs  (vs tiled: mean diff %.3f)\n",
            elapsed[5], diff_25))
cat(sprintf("6. edit (1 ref) 768x768:       %.2fs\n", elapsed[6]))
cat(sprintf("7. sd_sample vs stepwise:      %.2fs  (max diff %.3g)\n", elapsed[7], max_diff))

rm(ctx, imgs_1, imgs_2, imgs_3, imgs_4, imgs_5, imgs_6)
gc()

cat("\n=== Done ===\n")
