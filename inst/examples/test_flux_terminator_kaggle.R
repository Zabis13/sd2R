library(sdR)
library(png)
library(grid)

# Kaggle paths
models_dir <- "/kaggle/input/models/lbsbmsu/flux1-dev-q4-k-s/gguf/default/2"
out_dir <- "/kaggle/working"

ctx <- sd_ctx(diffusion_model_path = file.path(models_dir, "flux1-dev-Q4_K_S.gguf"),
              vae_path = file.path(models_dir, "ae.safetensors"),
              clip_l_path = file.path(models_dir, "clip_l.safetensors"),
              t5xxl_path = file.path(models_dir, "t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
              n_threads = 4L, model_type = "flux",
              vae_decode_only = FALSE, verbose = TRUE)

prompt <- paste(
  "Depict John Connor in a gritty post-nuclear wasteland,",
  "leading human rebels against Skynet's T-600 Terminators",
  "under a stormy sky, with explosions and hydrobots",
  "in the background, cinematic style like a movie poster."
)

for (i in seq_len(2)) {
  t0 <- proc.time()
  imgs <- sd_generate(
    ctx,
    prompt = prompt,
    width = 768L, height = 768L,
    sample_steps = 8L, cfg_scale = 1.0, seed = as.integer(i),
    sample_method = SAMPLE_METHOD$EULER,
    scheduler = SCHEDULER$DISCRETE,
    vae_mode = "tiled"
  )
  elapsed <- (proc.time() - t0)[["elapsed"]]
  cat(sprintf("[%d/2] Generated %dx%d in %.1fs\n", i, imgs[[1]]$width, imgs[[1]]$height, elapsed))

  filename <- sprintf("sdR_flux_terminator_%02d.png", i)
  path <- file.path(out_dir, filename)
  sd_save_image(imgs[[1]], path)
  cat(sprintf("Saved: %s\n", path))

  img_data <- readPNG(path)
  grid.newpage()
  grid.raster(img_data)
}
