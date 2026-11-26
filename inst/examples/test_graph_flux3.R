#!/usr/bin/env Rscript
# Graph pipeline on a real model (FLUX.3 Image Klein): low-level nodes
#   1. latent geometry from C++ vs the shape of a real VAE-encoded latent
#   2. txt2img:  encode_text -> sample -> decode -> save
#   3. graph vs direct low-level calls (same seed) -> latents must match
#   4. img2img:  load_image -> encode_image -> encode_text -> sample -> decode -> save
#   5. reference: sd_generate() with the same settings (timing only; its noise
#      comes from the C++ RNG, so the image differs)
#   6. JSON round-trip of the txt2img pipeline
# Run: Rscript /mnt/Data2/DS_projects/sd2R/inst/examples/test_graph_flux3.R

library(sd2R)

# ---- Paths (edit to match your setup) -------------------------------------
models_dir     <- "/mnt/Data3/DS_Data/sd_models"
out_dir        <- "/tmp"
diffusion_file <- file.path(models_dir, "flux3_image_klein_sd_Q8_0.gguf")
vae_file       <- file.path(models_dir, "flux2-vae.safetensors")
llm_file       <- file.path(models_dir, "Qwen3VL-4B-Instruct-Q4_K_M.gguf")
# ---------------------------------------------------------------------------

for (f in c(diffusion_file, vae_file, llm_file)) {
  if (!file.exists(f)) stop("File not found: ", f)
}

cat("=== sd2R graph pipeline FLUX.3 test ===\n")
cat("sd2R version:", as.character(utils::packageVersion("sd2R")), "\n\n")

ctx <- sd_ctx(diffusion_model_path = diffusion_file, vae_path = vae_file,
              llm_path = llm_file, model_type = "flux3", n_threads = 4L,
              vae_decode_only = FALSE, verbose = FALSE)

W <- 768L; H <- 768L
prompt <- "a cat sitting on a chair, oil painting"
steps <- 16L; cfg <- 1.0; seed <- 42L
sampler <- list(sample_method = SAMPLE_METHOD$EULER,
                scheduler = SCHEDULER$DISCRETE,
                sample_steps = steps, cfg_scale = cfg, seed = seed)
timed <- function(expr) {
  t0 <- proc.time()
  val <- force(expr)
  list(val = val, t = (proc.time() - t0)[["elapsed"]])
}
res <- list()

# --- 1. latent geometry --------------------------------------------------
cat("--- 1. latent geometry (C++) vs VAE-encoded latent ---\n")
geo <- sd2R:::latent_geometry(ctx)
cat(sprintf("C++: vae_scale_factor = %d, latent_channels = %d -> shape %s\n",
            geo[["vae_scale_factor"]], geo[["latent_channels"]],
            paste(c(W %/% geo[["vae_scale_factor"]], H %/% geo[["vae_scale_factor"]],
                    geo[["latent_channels"]]), collapse = " x ")))
probe <- list(width = W, height = H, channel = 3L,
              data = as.raw(rep(128L, W * H * 3L)))
enc_shape <- sd_encode_image(ctx, probe)$ne[1:3]
geo_shape <- c(W %/% geo[["vae_scale_factor"]], H %/% geo[["vae_scale_factor"]],
               geo[["latent_channels"]])
res$geometry <- identical(as.integer(enc_shape), as.integer(geo_shape))
cat(sprintf("encode_image: %s -> %s\n", paste(enc_shape, collapse = " x "),
            if (res$geometry) "MATCH" else "MISMATCH (noise shape for txt2img may be wrong)"))

# --- 2. txt2img via graph --------------------------------------------------
cat("\n--- 2. graph txt2img: encode_text -> sample -> decode -> save ---\n")
out_t2i <- file.path(out_dir, "sd2R_graph_flux3_t2i.png")
pipe_t2i <- do.call(sd_pipeline, list(
  sd_node("encode_text", prompt = prompt, width = W, height = H),
  do.call(sd_node, c(list("sample", width = W, height = H), sampler)),
  sd_node("decode"),
  sd_node("save", path = out_t2i)
))
r2 <- timed(sd_run_pipeline(pipe_t2i, ctx, verbose = TRUE))
img_t2i <- r2$val
cat(sprintf("wall time: %.2fs | %dx%d | saved %s\n", r2$t,
            img_t2i$width, img_t2i$height, out_t2i))

# --- 3. graph vs direct low-level calls -----------------------------------
cat("\n--- 3. graph vs direct sd_encode_text + sd_sample (same seed) ---\n")
pipe_lat <- sd_pipeline(
  sd_node("encode_text", prompt = prompt, width = W, height = H),
  do.call(sd_node, c(list("sample", width = W, height = H), sampler))
)
lat_graph <- sd_run_pipeline(pipe_lat, ctx)
cond   <- sd_encode_text(ctx, prompt, width = W, height = H)
uncond <- sd_encode_text(ctx, "", width = W, height = H)
lat_direct <- do.call(sd_sample, c(list(ctx, cond, uncond = uncond,
                                        latent_shape = geo_shape), sampler))
max_diff <- max(abs(lat_graph$data - lat_direct$data))
res$graph_vs_direct <- max_diff < 1e-3
cat(sprintf("max |diff| latent: %.3g -> %s\n", max_diff,
            if (res$graph_vs_direct) "MATCH" else "MISMATCH"))

# --- 4. img2img via graph --------------------------------------------------
cat("\n--- 4. graph img2img: load_image -> encode_image -> sample(strength) ---\n")
out_i2i <- file.path(out_dir, "sd2R_graph_flux3_i2i.png")
pipe_i2i <- sd_pipeline(
  sd_node("load_image", path = out_t2i),
  sd_node("encode_image"),
  sd_node("encode_text", prompt = "a dog sitting on a chair, oil painting",
          width = W, height = H),
  do.call(sd_node, c(list("sample", strength = 0.6), sampler)),
  sd_node("decode"),
  sd_node("save", path = out_i2i)
)
r4 <- timed(sd_run_pipeline(pipe_i2i, ctx, verbose = TRUE))
cat(sprintf("wall time: %.2fs | %dx%d | saved %s\n", r4$t,
            r4$val$width, r4$val$height, out_i2i))

# --- 5. reference: sd_generate ----------------------------------------------
cat("\n--- 5. reference: sd_generate() same settings ---\n")
r5 <- timed(sd_generate(ctx, prompt = prompt, width = W, height = H,
                        sample_method = SAMPLE_METHOD$EULER,
                        scheduler = SCHEDULER$DISCRETE,
                        sample_steps = steps, cfg_scale = cfg, seed = seed,
                        vae_mode = "normal"))
out_ref <- file.path(out_dir, "sd2R_graph_flux3_ref.png")
sd_save_image(r5$val[[1]], out_ref)
cat(sprintf("wall time: %.2fs | saved %s\n", r5$t, out_ref))

# --- 6. JSON round-trip -----------------------------------------------------
cat("\n--- 6. JSON round-trip ---\n")
json <- tempfile(fileext = ".json")
sd_save_pipeline(pipe_t2i, json)
pipe_back <- sd_load_pipeline(json)
res$json <- identical(lapply(pipe_back$nodes, `[[`, "type"),
                      lapply(pipe_t2i$nodes, `[[`, "type")) &&
  isTRUE(all.equal(pipe_back$nodes[[2]]$params, pipe_t2i$nodes[[2]]$params))
cat(readLines(json), sep = "\n")
cat(if (res$json) "round-trip OK\n" else "round-trip MISMATCH\n")
unlink(json)

# --- Summary ----------------------------------------------------------------
cat("\n=== Summary ===\n")
cat(sprintf("1. geometry C++ vs encode_image: %s\n", if (res$geometry) "MATCH" else "MISMATCH"))
cat(sprintf("2. graph txt2img:                %.2fs\n", r2$t))
cat(sprintf("3. graph vs direct latent:       %s (max diff %.3g)\n",
            if (res$graph_vs_direct) "MATCH" else "MISMATCH", max_diff))
cat(sprintf("4. graph img2img:                %.2fs\n", r4$t))
cat(sprintf("5. sd_generate reference:        %.2fs\n", r5$t))
cat(sprintf("6. JSON round-trip:              %s\n", if (res$json) "OK" else "MISMATCH"))
cat("\n=== Done ===\n")
