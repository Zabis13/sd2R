#!/usr/bin/env Rscript
# FLUX.2 inpainting demo for sd2R.
#
# Regenerates only the masked region of an existing image. The mask drives the
# denoise schedule (denoise_mask), so this works on plain FLUX.2 weights — no
# dedicated inpaint/Fill model required (FLUX.2 has no Fill variant anyway).
#
# Mask semantics: white (255) = regenerate this region, black (0) = keep.
#
# FLUX.2 Klein stack: diffusion + flux2-vae + Qwen3 LLM text encoder.
#
# Run: Rscript inst/examples/test_inpaint_flux2.R

library(sd2R)

cat("=== sd2R FLUX.2 inpaint demo ===\n")
print(sd_system_info())

models_dir <- "/mnt/Data2/DS_projects/sd_models"
out_dir    <- "/tmp"

n_gpu <- sd_vulkan_device_count()
cat(sprintf("Vulkan devices: %d\n", n_gpu))

# vae_decode_only = FALSE is required: img2img/inpaint needs the VAE encoder.
ctx <- sd_ctx(
  diffusion_model_path = file.path(models_dir, "flux-2-klein-4b.safetensors"),
  vae_path             = file.path(models_dir, "flux2-vae.safetensors"),
  llm_path             = file.path(models_dir, "Qwen3-4B-Q4_K_S.gguf"),
  n_threads            = 4L,
  model_type           = "flux2",
  vae_decode_only      = FALSE,
  verbose              = FALSE,
  device_layout        = if (n_gpu > 1L) "split_vae" else "mono"
)

W <- 768L
H <- 768L

# --- Step 1: base image -----------------------------------------------------
cat("\n--- Step 1: base image ---\n")
base <- sd_txt2img(
  ctx, "a cat sitting on a wooden chair, oil painting",
  width = W, height = H,
  sample_steps = 10L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE
)[[1]]
sd_save_image(base, file.path(out_dir, "sd2R_inpaint2_base.png"))
cat("Saved: /tmp/sd2R_inpaint2_base.png\n")

# --- Step 2: build a mask — a centred square to regenerate ------------------
# Matrix is [H, W], values in 0..1: 1 = inpaint, 0 = keep.
mask <- matrix(0, nrow = H, ncol = W)
y0 <- H %/% 4; y1 <- 3 * H %/% 4
x0 <- W %/% 4; x1 <- 3 * W %/% 4
mask[y0:y1, x0:x1] <- 1
sd_save_image(array(mask, dim = c(H, W, 1L)), file.path(out_dir, "sd2R_inpaint2_mask.png"))
cat("Saved: /tmp/sd2R_inpaint2_mask.png\n")

# --- Step 3: inpaint — only the masked square changes -----------------------
cat("\n--- Step 3: inpaint (replace the cat with a dog) ---\n")
result <- sd_img2img(
  ctx, "a dog sitting on a wooden chair, oil painting",
  init_image = base,
  mask = mask,                 # matrix; a PNG path is also accepted
  strength = 0.9,
  sample_steps = 10L, seed = 42L,
  sample_method = SAMPLE_METHOD$EULER,
  scheduler = SCHEDULER$DISCRETE
)[[1]]
sd_save_image(result, file.path(out_dir, "sd2R_inpaint2_result.png"))
cat("Saved: /tmp/sd2R_inpaint2_result.png\n")

cat("\nDone. Compare base vs result: outside the centre square they should match.\n")

rm(ctx, base, result)
gc()
