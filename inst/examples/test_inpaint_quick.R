library(sd2R)

# Inpainting demo: regenerate only the masked region of an existing image.
# Mask semantics: white (255) = generate, black (0) = keep original.
# Works on any model (the mask drives the denoise schedule); on FLUX 1/2 too.

model_path <- "/mnt/Data2/DS_projects/sd_models/v1-5-pruned-emaonly.safetensors"
ctx <- sd_ctx(model_path, n_threads = 4L, model_type = "sd1", vae_decode_only = FALSE)

W <- 512L
H <- 512L

# Step 1: generate a base image.
cat("--- Step 1: base 512x512 ---\n")
base <- sd_txt2img(ctx, "a mountain landscape, photorealistic",
                   negative_prompt = "blurry", width = W, height = H,
                   sample_steps = 20L, cfg_scale = 7.0, seed = 42L,
                   sample_method = SAMPLE_METHOD$EULER,
                   scheduler = SCHEDULER$DISCRETE)[[1]]
sd_save_image(base, "/tmp/sdR_inpaint_base.png")
cat("Saved base: /tmp/sdR_inpaint_base.png\n")

# Step 2: build a programmatic mask — a centred square (the region to regenerate).
# Matrix is [H, W] with values in 0..1: 1 = inpaint, 0 = keep.
mask <- matrix(0, nrow = H, ncol = W)
y0 <- H %/% 4; y1 <- 3 * H %/% 4
x0 <- W %/% 4; x1 <- 3 * W %/% 4
mask[y0:y1, x0:x1] <- 1
sd_save_image(array(mask, dim = c(H, W, 1L)), "/tmp/sdR_inpaint_mask.png")
cat("Saved mask: /tmp/sdR_inpaint_mask.png\n")

# Step 3: inpaint — only the masked square changes, the rest is preserved.
cat("--- Step 3: inpaint masked region ---\n")
result <- sd_img2img(ctx, "a small wooden cabin, photorealistic",
                     init_image = base,
                     mask = mask,                 # matrix; PNG path also accepted
                     negative_prompt = "blurry",
                     strength = 0.8, sample_steps = 20L, cfg_scale = 7.0,
                     seed = 42L, sample_method = SAMPLE_METHOD$EULER,
                     scheduler = SCHEDULER$DISCRETE)[[1]]
sd_save_image(result, "/tmp/sdR_inpaint_result.png")
cat("Saved result: /tmp/sdR_inpaint_result.png\n")

cat("Done. Outside the centre square the image should match the base.\n")
