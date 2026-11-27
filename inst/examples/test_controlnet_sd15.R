#!/usr/bin/env Rscript
# ControlNet (canny) on SD 1.5, through the graph
#   1. hint: load_image -> resize 512 -> canny -> save
#   2. txt2img with / without control (same seed): how well the result's
#      edges follow the hint (share of hint edges with a result edge within
#      2 px); with control it must be clearly higher
#   3. the same through the low-level path: encode_text -> sample(control)
#      -> decode
#   4. upscale_latent on a real 4-channel latent: base 256 -> upscale_latent
#      x2 -> sample(strength) -> decode (512x512, not noise)
#   5. tensor batch (UNet: one call): sample(batch = 3) -> decode ->
#      unbatch; element 1 equals batch = 1, timing vs 3 single runs
# Input: the cat from test_graph_flux3.R (or any PNG); if missing, a source
# image is generated first with SD 1.5 (no control).
# Run: Rscript /mnt/Data2/DS_projects/sd2R/inst/examples/test_controlnet_sd15.R

library(sd2R)

# ---- Paths (edit to match your setup) -------------------------------------
models_dir <- "/mnt/Data3/DS_Data/sd_models"
model_file <- file.path(models_dir, "v1-5-pruned-emaonly.safetensors")
cn_file    <- file.path(models_dir, "control_v11p_sd15_canny_fp16.safetensors")
in_file    <- "/tmp/sd2R_graph_flux3_t2i.png"
out_dir    <- "/tmp"
# ---------------------------------------------------------------------------

for (f in c(model_file, cn_file)) {
  if (!file.exists(f)) stop("File not found: ", f)
}

cat("=== sd2R ControlNet (SD 1.5, canny) test ===\n")
cat("sd2R version:", as.character(utils::packageVersion("sd2R")), "\n\n")

ctx <- sd_ctx(model_path = model_file, control_net_path = cn_file,
              model_type = "sd1", vae_decode_only = FALSE, verbose = FALSE)

S <- 512L
prompt <- "a cat sitting on a chair, oil painting, detailed"
neg <- "blurry, low quality"
gen <- list(sample_method = SAMPLE_METHOD$EULER, scheduler = SCHEDULER$DISCRETE,
            sample_steps = 20L, cfg_scale = 7.0, seed = 7L)

if (!file.exists(in_file)) {
  in_file <- file.path(out_dir, "sd2R_cn_source.png")
  cat("source image not found, generating", in_file, "\n")
  src_img <- do.call(sd_txt2img, c(list(ctx, "a cat sitting on a wooden chair in a room, photo",
                                        negative_prompt = neg, width = S, height = S),
                                   modifyList(gen, list(seed = 123L))))[[1]]
  sd_save_image(src_img, in_file)
}
timed <- function(expr) {
  t0 <- proc.time()
  val <- force(expr)
  list(val = val, t = (proc.time() - t0)[["elapsed"]])
}
res <- list()

# share of hint edges that have a result edge within r pixels
edge_follow <- function(img, hint, r = 2L) {
  e_hint <- sd_image_to_array(hint)[, , 1] > 0.99
  e_res <- sd_image_to_array(sd_canny(img))[, , 1] > 0.99
  h <- nrow(e_res); w <- ncol(e_res)
  dil <- e_res
  for (dy in -r:r) for (dx in -r:r) {
    ys <- pmin(pmax(seq_len(h) + dy, 1L), h)
    xs <- pmin(pmax(seq_len(w) + dx, 1L), w)
    dil <- dil | e_res[ys, xs]
  }
  mean(dil[e_hint])
}

# --- 1. hint -------------------------------------------------------------
cat("--- 1. hint: load_image -> resize -> canny ---\n")
out_hint <- file.path(out_dir, "sd2R_cn_hint.png")
pipe_hint <- sd_pipeline(
  sd_node("load_image", id = "src", path = in_file),
  sd_node("resize", id = "small", image = "src", width = S, height = S),
  sd_node("canny", id = "edges", image = "small"),
  sd_node("save", image = "edges", path = out_hint)
)
hint <- sd_run_pipeline(pipe_hint, ctx, outputs = "edges")$edges
cat(sprintf("hint %dx%d, edges %.1f%% | saved %s\n", hint$width, hint$height,
            100 * mean(sd_image_to_array(hint)[, , 1] > 0.99), out_hint))
hint_file <- out_hint

# --- 2. txt2img with / without control -----------------------------------
cat("\n--- 2. txt2img: control vs no control (same seed) ---\n")
out_cn <- file.path(out_dir, "sd2R_cn_txt2img.png")
out_free <- file.path(out_dir, "sd2R_cn_txt2img_free.png")
t2i <- function(...) do.call(sd_node, c(list("txt2img", prompt = prompt,
                                             negative_prompt = neg,
                                             width = S, height = S, ...), gen))
pipe_t2i <- sd_pipeline(
  sd_node("load_image", id = "hint", path = hint_file),
  t2i(id = "cn", control = "hint", control_strength = 0.9),
  sd_node("save", image = "cn", path = out_cn),
  t2i(id = "free"),
  sd_node("save", image = "free", path = out_free)
)
print(pipe_t2i)
r2 <- timed(sd_run_pipeline(pipe_t2i, ctx, outputs = c("cn", "free"),
                            verbose = TRUE))
f_cn <- edge_follow(r2$val$cn, hint)
f_free <- edge_follow(r2$val$free, hint)
res$t2i <- f_cn > f_free + 0.15
cat(sprintf("wall time: %.2fs | hint edges followed: control %.1f%%, none %.1f%% -> %s\n",
            r2$t, 100 * f_cn, 100 * f_free, if (res$t2i) "OK" else "WEAK"))
cat("saved", out_cn, out_free, "\n")

# --- 3. low-level: sample(control) ---------------------------------------
cat("\n--- 3. low-level: encode_text -> sample(control) -> decode ---\n")
out_ll <- file.path(out_dir, "sd2R_cn_lowlevel.png")
pipe_ll <- sd_pipeline(
  sd_node("load_image", id = "hint", path = hint_file),
  sd_node("encode_text", id = "txt", prompt = prompt, negative_prompt = neg,
          width = S, height = S),
  do.call(sd_node, c(list("sample", id = "lat", cond = "txt", uncond = "txt$uncond",
                          control = "hint", control_strength = 0.9,
                          width = S, height = S), gen)),
  sd_node("decode", id = "img", latent = "lat"),
  sd_node("save", image = "img", path = out_ll)
)
print(pipe_ll)
r3 <- timed(sd_run_pipeline(pipe_ll, ctx, outputs = "img", verbose = TRUE))
f_ll <- edge_follow(r3$val$img, hint)
res$ll <- f_ll > f_free + 0.15
cat(sprintf("wall time: %.2fs | hint edges followed: %.1f%% (txt2img without control: %.1f%%) -> %s\n",
            r3$t, 100 * f_ll, 100 * f_free, if (res$ll) "OK" else "WEAK"))
cat("saved", out_ll, "\n")

# --- 4. upscale_latent on SD 1.5 -----------------------------------------
cat("\n--- 4. upscale_latent: 256 -> x2 -> sample(strength 0.6) ---\n")
out_ul <- file.path(out_dir, "sd2R_cn_upscale_latent.png")
pipe_ul <- sd_pipeline(
  sd_node("encode_text", id = "txt", prompt = prompt, negative_prompt = neg,
          width = S, height = S),
  do.call(sd_node, c(list("sample", id = "base", cond = "txt", uncond = "txt$uncond",
                          width = S %/% 2L, height = S %/% 2L), gen)),
  sd_node("upscale_latent", id = "up", latent = "base", scale = 2),
  do.call(sd_node, c(list("sample", id = "hr", cond = "txt", uncond = "txt$uncond",
                          latent = "up", strength = 0.6), gen)),
  sd_node("decode", id = "img", latent = "hr"),
  sd_node("save", image = "img", path = out_ul)
)
r4 <- timed(sd_run_pipeline(pipe_ul, ctx, outputs = c("base", "up", "img"),
                            verbose = TRUE))
o4 <- r4$val
px_sd <- stats::sd(as.integer(o4$img$data))
res$ul <- identical(o4$base$ne[1:3], c(32L, 32L, 4L)) &&
  identical(o4$up$ne[1:3], c(64L, 64L, 4L)) &&
  o4$img$width == S && o4$img$height == S && px_sd > 20
cat(sprintf("wall time: %.2fs | latent %s -> %s | image %dx%d, pixel sd %.1f -> %s\n",
            r4$t, paste(o4$base$ne[1:3], collapse = "x"),
            paste(o4$up$ne[1:3], collapse = "x"), o4$img$width, o4$img$height,
            px_sd, if (res$ul) "OK" else "MISMATCH"))
cat("saved", out_ul, "\n")

# --- 5. tensor batch -------------------------------------------------------
cat("\n--- 5. sample(batch = 3) -> decode -> unbatch ---\n")
bpipe <- function(batch) sd_pipeline(
  sd_node("encode_text", id = "txt", prompt = prompt, negative_prompt = neg,
          width = S, height = S),
  do.call(sd_node, c(list("sample", id = "lat", cond = "txt", uncond = "txt$uncond",
                          width = S, height = S, batch = batch), gen)),
  sd_node("decode", id = "imgs", latent = "lat"),
  sd_node("unbatch", id = "first", image = "imgs", index = 1)
)
r5a <- timed(sd_run_pipeline(bpipe(1L), ctx, outputs = c("imgs", "first")))
r5b <- timed(sd_run_pipeline(bpipe(3L), ctx, outputs = c("imgs", "first")))
imgs3 <- sd_unbatch(r5b$val$imgs)
for (k in seq_along(imgs3)) {
  sd_save_image(imgs3[[k]], file.path(out_dir, sprintf("sd2R_cn_batch_%d.png", k)))
}
same1 <- mean(abs(as.integer(r5b$val$first$data) - as.integer(r5a$val$first$data))) / 255
res$batch <- isTRUE(r5b$val$imgs$batch == 3L) && length(imgs3) == 3L &&
  same1 < 0.01 && length(unique(lapply(imgs3, `[[`, "data"))) == 3L
cat(sprintf(paste0("batch = 1: %.2fs | batch = 3: %.2fs (3 single runs ~ %.2fs) | ",
                   "element 1 vs batch = 1: mean |diff| %.4f -> %s\n"),
            r5a$t, r5b$t, 3 * r5a$t, same1, if (res$batch) "OK" else "MISMATCH"))

# --- Summary ----------------------------------------------------------------
cat("\n=== Summary ===\n")
cat(sprintf("2. txt2img + control:   %s (edges followed %.1f%% vs %.1f%% without)\n",
            if (res$t2i) "OK" else "WEAK", 100 * f_cn, 100 * f_free))
cat(sprintf("3. sample + control:    %s (edges followed %.1f%%)\n",
            if (res$ll) "OK" else "WEAK", 100 * f_ll))
cat(sprintf("4. upscale_latent x2:   %s (%.2fs)\n",
            if (res$ul) "OK" else "MISMATCH", r4$t))
cat(sprintf("5. batch = 3:           %s (%.2fs vs %.2fs for 3 single runs)\n",
            if (res$batch) "OK" else "MISMATCH", r5b$t, 3 * r5a$t))
cat("\n=== Done ===\n")
