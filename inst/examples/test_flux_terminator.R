library(sdR)

models_dir <- "/mnt/Data2/DS_projects/sd_models"

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
  imgs <- sd_generate(
    ctx,
    prompt = prompt,
    width = 768L, height = 768L,
    sample_steps = 8L, cfg_scale = 1.0, seed = as.integer(i),
    sample_method = SAMPLE_METHOD$EULER,
    scheduler = SCHEDULER$DISCRETE
  )
  path <- sprintf("/tmp/sdR_flux_terminator_%02d.png", i)
  sd_save_image(imgs[[1]], path)
  cat(sprintf("[%02d/02] Saved: %s\n", i, path))
}
