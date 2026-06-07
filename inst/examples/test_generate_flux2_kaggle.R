# FLUX.2 (Klein / 4B) generate — Kaggle notebook version
#
# FLUX.2 specifics vs FLUX.1:
#   * model_type = "flux2" (auto-detected from tensors in C++)
#   * guidance-distilled -> cfg_scale auto-defaults to 1.0
#   * text encoder is an LLM: FLUX.2 Klein uses Qwen3 (full FLUX.2 -> Mistral).
#     Pass it via llm_path, NOT t5xxl_path. Diffusion, VAE and the LLM encoder
#     are three separate files.
#   * Diffusion-only file MUST go through diffusion_model_path so sd.cpp adds the
#     "model.diffusion_model." prefix and can detect VERSION_FLUX2_KLEIN.

library(sd2R)
library(png)
library(grid)

cat("=== sd2R sd_generate() FLUX.2 — Kaggle Test ===\n\n")
print(sd_system_info())

# Kaggle paths
models_dir <- "/kaggle/input/models/lbsbmsu/flux-2/gguf/default/1"
out_dir    <- "/kaggle/working"

# Model file paths.
# NOTE: only the diffusion filename is confirmed. Edit the VAE and LLM names
# below to match the actual files in your Kaggle dataset.
flux2_diffusion <- file.path(models_dir, "flux-2-klein-4b.safetensors")
flux2_vae       <- file.path(models_dir, "flux2-vae.safetensors")        # <-- edit if needed
flux2_llm       <- file.path(models_dir, "Qwen3-4B-Q4_K_S.gguf")         # <-- edit if needed (Qwen3 / Mistral slot)

# GPU count
n_gpu <- sd_vulkan_device_count()
cat(sprintf("Vulkan devices: %d\n", n_gpu))

# Helper: create FLUX.2 context
flux2_ctx <- function(vae_decode_only = TRUE) {
  sd_ctx(diffusion_model_path = flux2_diffusion,
         vae_path = flux2_vae,
         llm_path = flux2_llm,
         n_threads = 4L, model_type = "flux2",
         vae_decode_only = vae_decode_only, verbose = FALSE)
}

# Helper: save + display in notebook
show_image <- function(img, filename) {
  path <- file.path(out_dir, filename)
  sd_save_image(img, path)
  cat(sprintf("Saved: %s\n", path))
  img_data <- readPNG(path)
  grid.newpage()
  grid.raster(img_data)
}

# --- 1. FLUX.2 768x768 (direct) ---
cat("\n--- 1. FLUX.2 768x768 -> direct ---\n")
ctx <- flux2_ctx()
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
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(imgs_1), imgs_1[[1]]$width, imgs_1[[1]]$height, elapsed))
show_image(imgs_1[[1]], "sd2R_flux2_768.png")
rm(ctx); gc()

# --- 2. FLUX.2 1024x1024, forced tiled VAE ---
cat("\n--- 2. FLUX.2 1024x1024 -> tiled VAE ---\n")
ctx <- flux2_ctx()
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
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(imgs_2), imgs_2[[1]]$width, imgs_2[[1]]$height, elapsed))
show_image(imgs_2[[1]], "sd2R_flux2_tiled_1k.png")
rm(ctx); gc()

# --- 3. FLUX.2 2048x1024 -> auto highres fix ---
cat("\n--- 3. FLUX.2 2048x1024 -> auto highres fix ---\n")
ctx <- flux2_ctx(vae_decode_only = FALSE)
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
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(imgs_3), imgs_3[[1]]$width, imgs_3[[1]]$height, elapsed))
show_image(imgs_3[[1]], "sd2R_flux2_highres_panorama.png")
rm(ctx); gc()

# --- 4. FLUX.2 img2img 768x768 ---
cat("\n--- 4. FLUX.2 img2img 768x768 ---\n")
ctx <- flux2_ctx(vae_decode_only = FALSE)
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
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(imgs_4), imgs_4[[1]]$width, imgs_4[[1]]$height, elapsed))
show_image(imgs_4[[1]], "sd2R_flux2_img2img.png")

# --- 5. FLUX.2 1024x1024 -> direct (auto-routed) ---
cat("\n--- 5. FLUX.2 1024x1024 -> direct ---\n")
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
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Generated %d image(s): %dx%d in %.1fs\n",
            length(imgs_5), imgs_5[[1]]$width, imgs_5[[1]]$height, elapsed))
show_image(imgs_5[[1]], "sd2R_flux2_direct_1k.png")
rm(ctx); gc()

# --- 6. FLUX.2 Multi-GPU (if available) ---
if (n_gpu > 1L) {
  cat(sprintf("\n--- 6. FLUX.2 Multi-GPU: %d Vulkan device(s) ---\n", n_gpu))
  multi_prompts <- c(
    "a cat in a garden, watercolor",
    "a dog on a beach, oil painting",
    "a bird in the sky, digital art",
    "a fish underwater, photorealistic"
  )
  t0 <- proc.time()
  imgs_flux2_multi <- sd_generate_multi_gpu(
    diffusion_model_path = flux2_diffusion,
    vae_path = flux2_vae,
    llm_path = flux2_llm,
    prompts = multi_prompts,
    width = 768L, height = 768L,
    model_type = "flux2",
    sample_steps = 4L,
    sample_method = SAMPLE_METHOD$EULER,
    scheduler = SCHEDULER$DISCRETE,
    vae_mode = "tiled"
  )
  elapsed <- (proc.time() - t0)[["elapsed"]]
  cat(sprintf("Multi-GPU: generated %d image(s) across %d GPUs in %.1fs\n",
              length(imgs_flux2_multi), n_gpu, elapsed))
  for (i in seq_along(imgs_flux2_multi)) {
    if (!inherits(imgs_flux2_multi[[i]], "error")) {
      show_image(imgs_flux2_multi[[i]], sprintf("sd2R_flux2_multi_gpu_%d.png", i))
    } else {
      cat(sprintf("  Image %d failed: %s\n", i, conditionMessage(imgs_flux2_multi[[i]])))
    }
  }
  rm(imgs_flux2_multi)
} else {
  cat("\n--- 6. FLUX.2 Multi-GPU: skipped (only 1 GPU) ---\n")
}

rm(imgs_1, imgs_2, imgs_3, imgs_4, imgs_5)
gc()

cat("\n=== Done ===\n")
