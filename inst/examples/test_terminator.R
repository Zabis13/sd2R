library(sd2R)

model_path <- "/mnt/Data3/DS_Data/sd_models/v1-5-pruned-emaonly.safetensors"

ctx <- sd_ctx(model_path, n_threads = 4L, model_type = "sd1",
              vae_decode_only = FALSE, verbose = FALSE)

prompt <- paste(
  "Depict John Connor in a gritty post-nuclear wasteland,",
  "leading human rebels against Skynet's T-600 Terminators",
  "under a stormy sky, with explosions and hydrobots",
  "in the background, cinematic style like a movie poster."
)
negative <- "blurry, bad quality, text, watermark, deformed"

for (i in seq_len(10)) {
  imgs <- sd_generate(
    ctx,
    prompt = prompt,
    negative_prompt = negative,
    width = 1024L, height = 1024L,
    sample_steps = 10L, cfg_scale = 7.5, seed = as.integer(i),
    sample_method = SAMPLE_METHOD$EULER,
    scheduler = SCHEDULER$DISCRETE
  )
  path <- sprintf("/tmp/sdR_terminator_%02d.png", i)
  sd_save_image(imgs[[1]], path)
  cat(sprintf("[%02d/10] Saved: %s\n", i, path))
}
