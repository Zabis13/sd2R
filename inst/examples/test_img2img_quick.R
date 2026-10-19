library(sd2R)

model_path <- "/mnt/Data3/DS_Data/sd_models/v1-5-pruned-emaonly.safetensors"
ctx <- sd_ctx(model_path, n_threads = 4L, model_type = "sd1", vae_decode_only = FALSE)

# Step 1: generate base 512x512
cat("--- Step 1: base 512x512 ---\n")
base <- sd_txt2img(ctx, "a mountain landscape, photorealistic",
                   negative_prompt = "blurry", width = 512L, height = 512L,
                   sample_steps = 20L, cfg_scale = 7.0, seed = 42L,
                   sample_method = SAMPLE_METHOD$EULER,
                   scheduler = SCHEDULER$DISCRETE)[[1]]
sd_save_image(base, "/tmp/sdR_base_512.png")
cat("Saved base: /tmp/sdR_base_512.png\n")

# Step 2: img2img at same size, strength=0.4
cat("--- Step 2: img2img 512x512 strength=0.4 ---\n")
refined <- sd_img2img(ctx, "a mountain landscape, photorealistic",
                      init_image = base,
                      negative_prompt = "blurry",
                      strength = 0.4, sample_steps = 20L, cfg_scale = 7.0,
                      seed = 42L, sample_method = SAMPLE_METHOD$EULER,
                      scheduler = SCHEDULER$DISCRETE)[[1]]
sd_save_image(refined, "/tmp/sdR_img2img_512.png")
cat("Saved img2img: /tmp/sdR_img2img_512.png\n")

cat("Done. Compare the two images.\n")
