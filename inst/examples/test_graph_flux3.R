#!/usr/bin/env Rscript
# Graph pipeline on a real model (FLUX.3 Image Klein): low-level nodes,
# linked explicitly by node id ("id" / "id$output")
#   1. latent geometry from C++ vs the shape of a real VAE-encoded latent
#   2. txt2img:  encode_text -> sample -> decode -> save
#   3. graph vs direct low-level calls (same seed) -> latents must match
#   4. img2img:  load_image -> encode_image -> encode_text -> sample -> decode -> save
#   5. branching: one encode_text -> two samples (seeds) -> two decodes/saves;
#      the branch with the step-2 seed must reproduce the step-2 image
#   6. reference: sd_generate() with the same settings (timing only; its noise
#      comes from the C++ RNG, so the image differs)
#   7. JSON round-trip of the branching pipeline (ids and links preserved)
#   8. intermediate results: outputs = c("txt", "lat_a", "img_b") on the
#      branching pipeline; culling: the same graph without save nodes,
#      outputs = "img_a" -> the seed+1 branch must not run
#   9. masks (needs RMBG-1.4.onnx, skipped if missing): background
#      replacement load_image -> remove_background -> invert_mask ->
#      img2img(mask) -> save, plus save_mask of both masks
#  10. composite (needs RMBG too): cat from step 2 over a generated beach
#      generated at half size -> resize -> composite(fg, bg, mask) -> crop
#  11. execution cache: branching pipeline run 3 times with one sd_cache():
#      cold; unchanged (both samples cached, only decode/save run); seed of
#      one branch changed (encode_text and the other sample cached)
#  12. on_event: event log with per-node timing on the save-less branching
#      graph (outputs = "img_a" -> lat_b/img_b skipped); then cancel with
#      sd_pipeline_cancel() after the first sample -> nothing else starts
#  13. highres fix from nodes (image path): base at half size -> decode ->
#      resize -> encode_image -> sample(strength) -> decode; upscale_latent
#      must refuse the packed FLUX.3 latent (taken from the step-11 cache)
#  14. low-level inpaint (needs the step-9 masks): load_image ->
#      encode_image -> sample(latent, mask) -> decode; compared with the
#      step-9 img2img(mask) by how much the kept / regenerated areas change
#      (noise differs: R rnorm vs C++ RNG, so no pixel comparison); a
#      control input must be refused before running (no ControlNet loaded)
#  15. ControlNet preprocessors on the step-2 image: canny (sd.cpp core) and
#      depth (Depth Anything V2 ONNX, skipped if the model is missing); the
#      ControlNet itself is not run (none for FLUX.3)
#  16. sd_run_batch / sd_loopback: (a) zip over 3 seeds, encode_text runs
#      once (cache); (b) 2 x 2 grid seed x strength of img2img on a run-time
#      input; (c) loopback img2img x3 with decaying strength, keep = "all"
# Run: Rscript /mnt/Data2/DS_projects/sd2R/inst/examples/test_graph_flux3.R

library(sd2R)

# ---- Paths (edit to match your setup) -------------------------------------
models_dir     <- "/mnt/Data3/DS_Data/sd_models"
out_dir        <- "/tmp"
diffusion_file <- file.path(models_dir, "flux3_image_klein_sd_Q8_0.gguf")
vae_file       <- file.path(models_dir, "flux2-vae.safetensors")
llm_file       <- file.path(models_dir, "Qwen3VL-4B-Instruct-Q4_K_M.gguf")
rmbg_file      <- file.path(models_dir, "RMBG-1.4.onnx")   # steps 9, 10, 14
depth_file     <- file.path(models_dir, "depth_anything_v2_small.onnx")  # step 15
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

sample_node <- function(id, ..., seed_override = NULL) {
  s <- sampler
  if (!is.null(seed_override)) s$seed <- seed_override
  do.call(sd_node, c(list("sample", id = id, ...), s))
}

# --- 2. txt2img via graph --------------------------------------------------
cat("\n--- 2. graph txt2img: encode_text -> sample -> decode -> save ---\n")
out_t2i <- file.path(out_dir, "sd2R_graph_flux3_t2i.png")
pipe_t2i <- sd_pipeline(
  sd_node("encode_text", id = "txt", prompt = prompt, width = W, height = H),
  sample_node("lat", cond = "txt", uncond = "txt$uncond", width = W, height = H),
  sd_node("decode", id = "img", latent = "lat"),
  sd_node("save", image = "img", path = out_t2i)
)
print(pipe_t2i)
r2 <- timed(sd_run_pipeline(pipe_t2i, ctx, verbose = TRUE))
img_t2i <- r2$val
cat(sprintf("wall time: %.2fs | %dx%d | saved %s\n", r2$t,
            img_t2i$width, img_t2i$height, out_t2i))

# --- 3. graph vs direct low-level calls -----------------------------------
cat("\n--- 3. graph vs direct sd_encode_text + sd_sample (same seed) ---\n")
pipe_lat <- sd_pipeline(
  sd_node("encode_text", id = "txt", prompt = prompt, width = W, height = H),
  sample_node("lat", cond = "txt", width = W, height = H)
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
  sd_node("load_image", id = "src", path = out_t2i),
  sd_node("encode_image", id = "init", image = "src"),
  sd_node("encode_text", id = "txt", prompt = "a dog sitting on a chair, oil painting",
          width = W, height = H),
  sample_node("lat", cond = "txt", latent = "init", strength = 0.6),
  sd_node("decode", id = "img", latent = "lat"),
  sd_node("save", image = "img", path = out_i2i)
)
print(pipe_i2i)
r4 <- timed(sd_run_pipeline(pipe_i2i, ctx, verbose = TRUE))
cat(sprintf("wall time: %.2fs | %dx%d | saved %s\n", r4$t,
            r4$val$width, r4$val$height, out_i2i))

# --- 5. branching: one text encoding -> two seeds --------------------------
cat("\n--- 5. graph branching: encode_text -> {sample seed, sample seed+1} ---\n")
out_a <- file.path(out_dir, "sd2R_graph_flux3_branch_a.png")
out_b <- file.path(out_dir, "sd2R_graph_flux3_branch_b.png")
pipe_br <- sd_pipeline(
  sd_node("encode_text", id = "txt", prompt = prompt, width = W, height = H),
  sample_node("lat_a", cond = "txt", width = W, height = H),
  sample_node("lat_b", cond = "txt", width = W, height = H,
              seed_override = seed + 1L),
  sd_node("decode", id = "img_a", latent = "lat_a"),
  sd_node("decode", id = "img_b", latent = "lat_b"),
  sd_node("save", image = "img_a", path = out_a),
  sd_node("save", image = "img_b", path = out_b)
)
print(pipe_br)
r5 <- timed(sd_run_pipeline(pipe_br, ctx, verbose = TRUE))
img_a <- sd_load_image(out_a)
img_b <- r5$val
res$branch_same <- identical(img_a$data, img_t2i$data)
res$branch_diff <- !identical(img_b$data, img_t2i$data)
cat(sprintf("wall time: %.2fs | branch a == step 2: %s | branch b differs: %s\n",
            r5$t, res$branch_same, res$branch_diff))
cat(sprintf("saved %s %s\n", out_a, out_b))

# --- 6. reference: sd_generate ----------------------------------------------
cat("\n--- 6. reference: sd_generate() same settings ---\n")
r6 <- timed(sd_generate(ctx, prompt = prompt, width = W, height = H,
                        sample_method = SAMPLE_METHOD$EULER,
                        scheduler = SCHEDULER$DISCRETE,
                        sample_steps = steps, cfg_scale = cfg, seed = seed,
                        vae_mode = "normal"))
out_ref <- file.path(out_dir, "sd2R_graph_flux3_ref.png")
sd_save_image(r6$val[[1]], out_ref)
cat(sprintf("wall time: %.2fs | saved %s\n", r6$t, out_ref))

# --- 7. JSON round-trip -----------------------------------------------------
cat("\n--- 7. JSON round-trip (branching pipeline) ---\n")
json <- tempfile(fileext = ".json")
sd_save_pipeline(pipe_br, json)
pipe_back <- sd_load_pipeline(json)
strip <- function(p) lapply(p$nodes, function(n) n[c("type", "id", "inputs")])
res$json <- identical(strip(pipe_back), strip(pipe_br)) &&
  isTRUE(all.equal(pipe_back$nodes[[3]]$params, pipe_br$nodes[[3]]$params))
cat(readLines(json), sep = "\n")
cat(if (res$json) "round-trip OK\n" else "round-trip MISMATCH\n")
unlink(json)

# --- 8. intermediate results + culling -------------------------------------
cat("\n--- 8. outputs = c(\"txt\", \"lat_a\", \"img_b\") + culling ---\n")
r8 <- timed(sd_run_pipeline(pipe_br, ctx, outputs = c("txt", "lat_a", "img_b"),
                            verbose = TRUE))
out8 <- r8$val
lat_diff <- max(abs(out8$lat_a$data - lat_graph$data))
res$outputs <- identical(names(out8), c("txt", "lat_a", "img_b")) &&
  identical(sort(names(out8$txt)), c("cond", "uncond")) &&
  lat_diff < 1e-3 &&
  identical(out8$img_b$data, sd_load_image(out_b)$data)
cat(sprintf(paste0("wall time: %.2fs | names: %s | txt: %s | ",
                   "lat_a vs step 3: %.3g | img_b == branch b file: %s\n"),
            r8$t, paste(names(out8), collapse = ", "),
            paste(names(out8$txt), collapse = ", "), lat_diff,
            identical(out8$img_b$data, sd_load_image(out_b)$data)))

# same graph without save nodes: only txt -> lat_a -> img_a must run
pipe_cull <- do.call(sd_pipeline, pipe_br$nodes[1:5])
msgs <- character(0)
r8c <- timed(withCallingHandlers(
  sd_run_pipeline(pipe_cull, ctx, outputs = "img_a", verbose = TRUE),
  message = function(m) {
    msgs <<- c(msgs, conditionMessage(m))
    invokeRestart("muffleMessage")
  }))
cat(msgs, sep = "")
res$cull <- length(msgs) == 3L &&
  identical(r8c$val$img_a$data, img_t2i$data)
cat(sprintf("wall time: %.2fs | nodes run: %d of 5 | img_a == step 2: %s\n",
            r8c$t, length(msgs), identical(r8c$val$img_a$data, img_t2i$data)))

# --- 9. masks: background replacement --------------------------------------
cat("\n--- 9. masks: remove_background -> invert_mask -> img2img(mask) ---\n")
res$mask <- NA
if (!file.exists(rmbg_file)) {
  cat("skipped: not found", rmbg_file, "\n")
} else {
  out_fg   <- file.path(out_dir, "sd2R_graph_flux3_mask_fg.png")
  out_back <- file.path(out_dir, "sd2R_graph_flux3_mask_back.png")
  out_edit <- file.path(out_dir, "sd2R_graph_flux3_beach.png")
  pipe_mask <- sd_pipeline(
    sd_node("load_image", id = "src", path = out_t2i),
    sd_node("remove_background", id = "bg", image = "src",
            model_path = rmbg_file, threshold = 0.5),
    sd_node("save_mask", mask = "bg$mask", path = out_fg),
    sd_node("invert_mask", id = "back", mask = "bg"),
    sd_node("save_mask", mask = "back", path = out_back),
    do.call(sd_node, c(list("img2img", id = "edit", image = "src",
                            mask = "back",
                            prompt = "a cat sitting on a chair on a sunny beach, oil painting",
                            strength = 0.9), sampler)),
    sd_node("save", image = "edit", path = out_edit)
  )
  print(pipe_mask)
  r9 <- timed(sd_run_pipeline(pipe_mask, ctx,
                              outputs = c("src", "bg", "back", "edit"),
                              verbose = TRUE))
  o9 <- r9$val
  fg <- sd_image_to_array(o9$bg$mask)[, , 1] > 0.5
  inv_ok <- identical(as.integer(o9$back$data),
                      255L - as.integer(o9$bg$mask$data))
  # foreground (kept) should change far less than background (regenerated)
  d <- apply(abs(sd_image_to_array(o9$edit)[, , 1:3] -
                 sd_image_to_array(o9$src)[, , 1:3]), c(1, 2), mean)
  d_fg <- mean(d[fg]); d_bg <- mean(d[!fg])
  res$mask <- o9$bg$mask$channel == 1L && inv_ok &&
    o9$edit$width == W && o9$edit$height == H && d_fg < d_bg
  cat(sprintf(paste0("wall time: %.2fs | mask: %d ch, fg %.1f%% | inverted OK: %s\n",
                     "mean |diff| vs source: foreground %.3f, background %.3f\n"),
              r9$t, o9$bg$mask$channel, 100 * mean(fg), inv_ok, d_fg, d_bg))
  cat("saved", out_fg, out_back, out_edit, "\n")
}

# --- 10. composite: cat over a generated background ------------------------
cat("\n--- 10. resize / composite / crop ---\n")
res$comp <- NA
if (!file.exists(rmbg_file)) {
  cat("skipped: not found", rmbg_file, "\n")
} else {
  out_comp <- file.path(out_dir, "sd2R_graph_flux3_composite.png")
  out_crop <- file.path(out_dir, "sd2R_graph_flux3_composite_crop.png")
  Wc <- W %/% 2L; Hc <- H %/% 2L
  pipe_comp <- sd_pipeline(
    sd_node("load_image", id = "src", path = out_t2i),
    sd_node("remove_background", id = "bg", image = "src",
            model_path = rmbg_file, threshold = 0.5),
    sd_node("encode_text", id = "txt_bg", prompt = "a sunny beach, oil painting",
            width = Wc, height = Hc),
    sample_node("lat_bg", cond = "txt_bg", width = Wc, height = Hc),
    sd_node("decode", id = "beach_small", latent = "lat_bg"),
    sd_node("resize", id = "beach", image = "beach_small", width = W, height = H),
    sd_node("composite", id = "comp", fg = "src", bg = "beach", mask = "bg$mask"),
    sd_node("save", image = "comp", path = out_comp),
    sd_node("crop", id = "cr", image = "comp", x = W %/% 4L, y = H %/% 4L,
            width = Wc, height = Hc),
    sd_node("save", image = "cr", path = out_crop)
  )
  print(pipe_comp)
  r10 <- timed(sd_run_pipeline(pipe_comp, ctx,
                               outputs = c("src", "bg", "beach_small", "beach",
                                           "comp", "cr"),
                               verbose = TRUE))
  o10 <- r10$val
  rgb <- function(img) sd_image_to_array(img)[, , 1:3]
  fg <- sd_image_to_array(o10$bg$mask)[, , 1] > 0.5     # binary (threshold)
  ca <- rgb(o10$comp)
  d_fg <- max(abs(ca - rgb(o10$src))[cbind(which(fg, arr.ind = TRUE), 1L)])
  d_bg <- max(abs(ca - rgb(o10$beach))[cbind(which(!fg, arr.ind = TRUE), 1L)])
  crop_ok <- identical(ca[H %/% 4L + seq_len(Hc), W %/% 4L + seq_len(Wc), ],
                       rgb(o10$cr))
  res$comp <- o10$beach_small$width == Wc && o10$beach$width == W &&
    o10$comp$channel == 3L && d_fg == 0 && d_bg == 0 && crop_ok &&
    o10$cr$width == Wc && o10$cr$height == Hc
  cat(sprintf(paste0("wall time: %.2fs | bg %dx%d -> %dx%d | ",
                     "comp == src on fg: %s, == beach on bg: %s | ",
                     "crop %dx%d matches: %s\n"),
              r10$t, o10$beach_small$width, o10$beach_small$height,
              o10$beach$width, o10$beach$height, d_fg == 0, d_bg == 0,
              o10$cr$width, o10$cr$height, crop_ok))
  cat("saved", out_comp, out_crop, "\n")
}

# --- 11. execution cache ----------------------------------------------------
cat("\n--- 11. sd_cache: cold / unchanged / one branch changed ---\n")
run_logged <- function(pipe) {
  msgs <- character(0)
  r <- timed(withCallingHandlers(
    sd_run_pipeline(pipe, ctx, cache = cache, outputs = c("img_a", "img_b"),
                    verbose = TRUE),
    message = function(m) {
      msgs <<- c(msgs, conditionMessage(m))
      invokeRestart("muffleMessage")
    }))
  cat(msgs, sep = "")
  r$msgs <- trimws(msgs)
  r
}
cached_of <- function(msgs) sub(" \\(cached\\)$", "", sub("^\\[[0-9/]+\\] ", "",
                                 grep("(cached)", msgs, value = TRUE, fixed = TRUE)))
pipe_c <- pipe_br
pipe_c$nodes[[6]]$params$path <- file.path(out_dir, "sd2R_graph_flux3_cache_a.png")
pipe_c$nodes[[7]]$params$path <- file.path(out_dir, "sd2R_graph_flux3_cache_b.png")
cache <- sd_cache()

cat("-- run 1 (cold)\n")
c1 <- run_logged(pipe_c)
cat("-- run 2 (unchanged)\n")
c2 <- run_logged(pipe_c)
cat("-- run 3 (lat_b seed changed)\n")
pipe_c3 <- pipe_c
pipe_c3$nodes[[3]]$params$seed <- seed + 2L
c3 <- run_logged(pipe_c3)
print(cache)

res$cache <- length(cached_of(c1$msgs)) == 0L &&
  identical(sort(cached_of(c2$msgs)), c("sample", "sample")) &&
  identical(c2$val$img_a$data, c1$val$img_a$data) &&
  identical(c2$val$img_b$data, c1$val$img_b$data) &&
  identical(sort(cached_of(c3$msgs)), c("encode_text", "sample")) &&
  identical(c3$val$img_a$data, c1$val$img_a$data) &&
  !identical(c3$val$img_b$data, c1$val$img_b$data)
cat(sprintf(paste0("run 1: %.2fs | run 2: %.2fs (same images: %s) | ",
                   "run 3: %.2fs (img_a same: %s, img_b new: %s)\n"),
            c1$t, c2$t,
            identical(c2$val$img_b$data, c1$val$img_b$data), c3$t,
            identical(c3$val$img_a$data, c1$val$img_a$data),
            !identical(c3$val$img_b$data, c1$val$img_b$data)))

# --- 12. on_event: event log + cancel ---------------------------------------
cat("\n--- 12. on_event: event log, timing, cancel ---\n")
evlog <- list()
log_event <- function(e) evlog[[length(evlog) + 1L]] <<- e
r12 <- timed(sd_run_pipeline(pipe_cull, ctx, outputs = "img_a",
                             on_event = log_event))
fmt_out <- function(o) {
  if (is.null(o)) return("")
  paste(vapply(names(o), function(nm) {
    m <- o[[nm]]
    size <- if (!is.null(m$width)) sprintf("%dx%dx%d", m$width, m$height, m$channel)
            else if (!is.null(m$ne) && !is.list(m$ne)) paste(m$ne, collapse = "x")
            else ""
    sprintf("%s:%s %s", nm, m$class, size)
  }, ""), collapse = ", ")
}
cat(sprintf("%-8s %-6s %-12s %8s  %s\n", "event", "id", "type", "elapsed", "outputs"))
for (e in evlog) {
  cat(sprintf("%-8s %-6s %-12s %8s  %s\n", e$event, e$id, e$type,
              if (is.na(e$elapsed)) "" else sprintf("%.2fs", e$elapsed),
              fmt_out(e$outputs)))
}
seq12 <- vapply(evlog, function(e) paste(e$event, e$id), "")
sum_el <- sum(vapply(evlog, function(e) if (e$event == "end") e$elapsed else 0, 0))
ev_ok <- identical(seq12, c("skipped lat_b", "skipped img_b",
                            "start txt", "end txt", "start lat_a", "end lat_a",
                            "start img_a", "end img_a")) &&
  all(vapply(evlog, function(e) is.null(e$outputs) ||
               all(vapply(e$outputs, function(m) is.null(m$data), TRUE)), TRUE))
cat(sprintf("wall time: %.2fs | sum of node times: %.2fs | sequence OK: %s\n",
            r12$t, sum_el, ev_ok))

started <- character(0)
cancel_after_lat_a <- function(e) {
  if (e$event == "start") started <<- c(started, e$id)
  if (e$event == "end" && e$id == "lat_a") sd_pipeline_cancel("stop after lat_a")
}
r12c <- timed(tryCatch(
  sd_run_pipeline(pipe_cull, ctx, outputs = c("img_a", "img_b"),
                  on_event = cancel_after_lat_a),
  sd_pipeline_cancel = function(c) c))
cancel_ok <- inherits(r12c$val, "sd_pipeline_cancel") &&
  identical(started, c("txt", "lat_a"))
cat(sprintf("cancel: %.2fs | condition: %s (\"%s\") | started: %s\n", r12c$t,
            class(r12c$val)[1], conditionMessage(r12c$val),
            paste(started, collapse = ", ")))
res$events <- ev_ok && cancel_ok

# --- 13. highres fix via the image path ------------------------------------
cat("\n--- 13. highres fix: base -> decode -> resize -> encode_image -> sample ---\n")
out_hr <- file.path(out_dir, "sd2R_graph_flux3_highres.png")
pipe_hr <- sd_pipeline(
  sd_node("encode_text", id = "txt", prompt = prompt, width = W, height = H),
  sample_node("base", cond = "txt", width = W %/% 2L, height = H %/% 2L),
  sd_node("decode", id = "small", latent = "base"),
  sd_node("resize", id = "big", image = "small", width = W, height = H),
  sd_node("encode_image", id = "init", image = "big"),
  sample_node("hr", cond = "txt", latent = "init", strength = 0.5),
  sd_node("decode", id = "img", latent = "hr"),
  sd_node("save", image = "img", path = out_hr)
)
print(pipe_hr)
r13 <- timed(sd_run_pipeline(pipe_hr, ctx, outputs = c("small", "big", "img"),
                             verbose = TRUE))
o13 <- r13$val
hr_ok <- o13$small$width == W %/% 2L && o13$big$width == W &&
  o13$img$width == W && o13$img$height == H
cat(sprintf("wall time: %.2fs (direct %dx%d in step 2: %.2fs) | %dx%d -> %dx%d -> %dx%d | saved %s\n",
            r13$t, W, H, r2$t, o13$small$width, o13$small$height,
            o13$big$width, o13$big$height, o13$img$width, o13$img$height, out_hr))

# upscale_latent on FLUX.3: lat_a comes from the step-11 cache, the node
# itself must refuse the 128-channel (2x2-packed) latent
pipe_ul <- sd_pipeline(pipe_c$nodes[[1]], pipe_c$nodes[[2]],
                       sd_node("upscale_latent", latent = "lat_a", scale = 2))
msgs <- character(0)
err_ul <- tryCatch(withCallingHandlers(
  sd_run_pipeline(pipe_ul, ctx, cache = cache, verbose = TRUE),
  message = function(m) {
    msgs <<- c(msgs, conditionMessage(m))
    invokeRestart("muffleMessage")
  }), error = function(e) conditionMessage(e))
cat(msgs, sep = "")
ul_ok <- is.character(err_ul) && grepl("128 channels is not supported", err_ul) &&
  any(grepl("sample (cached)", msgs, fixed = TRUE))
cat(sprintf("upscale_latent on FLUX.3: %s\n  %s\n",
            if (ul_ok) "refused as expected" else "UNEXPECTED", err_ul))
res$highres <- hr_ok && ul_ok

# --- 14. low-level inpaint: sample(mask) -----------------------------------
cat("\n--- 14. sample(latent, mask) vs img2img(mask) ---\n")
res$ll_mask <- NA
if (is.na(res$mask)) {
  cat("skipped: needs the step-9 masks (RMBG)\n")
} else {
  out_ll <- file.path(out_dir, "sd2R_graph_flux3_beach_lowlevel.png")
  beach_prompt <- "a cat sitting on a chair on a sunny beach, oil painting"
  pipe_ll <- sd_pipeline(
    sd_node("load_image", id = "src", path = out_t2i),
    sd_node("load_mask", id = "back", path = out_back),
    sd_node("encode_image", id = "init", image = "src"),
    sd_node("encode_text", id = "txt", prompt = beach_prompt, width = W, height = H),
    sample_node("lat", cond = "txt", latent = "init", mask = "back", strength = 0.9),
    sd_node("decode", id = "img", latent = "lat"),
    sd_node("save", image = "img", path = out_ll)
  )
  print(pipe_ll)
  r14 <- timed(sd_run_pipeline(pipe_ll, ctx, outputs = c("src", "img"),
                               verbose = TRUE))
  fg14 <- sd_image_to_array(sd_load_mask(out_fg))[, , 1] > 0.5
  diffs <- function(edit, src) {
    d <- apply(abs(sd_image_to_array(edit)[, , 1:3] -
                   sd_image_to_array(src)[, , 1:3]), c(1, 2), mean)
    c(fg = mean(d[fg14]), bg = mean(d[!fg14]))
  }
  d_ll <- diffs(r14$val$img, r14$val$src)
  d_i2i <- diffs(sd_load_image(out_edit), r14$val$src)
  ll_ok <- d_ll[["fg"]] < d_ll[["bg"]]

  # ControlNet: refused before anything runs (this context has none)
  pipe_cn <- sd_pipeline(
    sd_node("load_image", id = "hint", path = out_t2i),
    sd_node("encode_text", id = "txt", prompt = prompt, width = W, height = H),
    sample_node("lat", cond = "txt", control = "hint", width = W, height = H)
  )
  t_cn <- timed(err_cn <- tryCatch(sd_run_pipeline(pipe_cn, ctx),
                                   error = function(e) conditionMessage(e)))
  cn_ok <- is.character(err_cn) && grepl("needs a ControlNet", err_cn) && t_cn$t < 1
  res$ll_mask <- ll_ok && cn_ok
  cat(sprintf(paste0("wall time: %.2fs | mean |diff| vs source, kept / regenerated:\n",
                     "  sample(mask):  %.3f / %.3f\n  img2img(mask): %.3f / %.3f (step 9)\n"),
              r14$t, d_ll[["fg"]], d_ll[["bg"]], d_i2i[["fg"]], d_i2i[["bg"]]))
  cat(sprintf("control without ControlNet: refused in %.2fs: %s\n  %s\nsaved %s\n",
              t_cn$t, cn_ok, err_cn, out_ll))
}

# --- 15. ControlNet preprocessors ------------------------------------------
cat("\n--- 15. preprocessors: canny / depth ---\n")
out_canny <- file.path(out_dir, "sd2R_graph_flux3_canny.png")
out_depth <- file.path(out_dir, "sd2R_graph_flux3_depth.png")
have_depth <- file.exists(depth_file)
pipe_pre <- do.call(sd_pipeline, c(
  list(sd_node("load_image", id = "src", path = out_t2i),
       sd_node("canny", id = "edges", image = "src"),
       sd_node("save", image = "edges", path = out_canny)),
  if (have_depth) list(
       sd_node("depth", id = "dep", image = "src", model_path = depth_file),
       sd_node("save", image = "dep", path = out_depth))))
print(pipe_pre)
evp <- list()
r15 <- timed(sd_run_pipeline(pipe_pre, ctx,
                             outputs = c("edges", if (have_depth) "dep"),
                             on_event = function(e) if (e$event == "end")
                               evp[[e$id]] <<- e$elapsed))
o15 <- r15$val
if (!have_depth) o15 <- list(edges = o15)
ev <- sort(unique(as.integer(o15$edges$data)))
edge_frac <- mean(as.integer(o15$edges$data) == 255L)   # strong edges
canny_ok <- o15$edges$width == W && o15$edges$height == H &&
  # core hysteresis: weak edges end up strong or 0; sub-threshold residue (<= 2) stays
  255L %in% ev && all(ev <= 2L | ev == 255L) && edge_frac > 0.001 && edge_frac < 0.5
cat(sprintf("canny: %.2fs | %dx%dx%d | values: %s | edges: %.1f%% | saved %s\n",
            evp$edges, o15$edges$width, o15$edges$height, o15$edges$channel,
            paste(ev, collapse = ", "), 100 * edge_frac, out_canny))
depth_ok <- NA
if (have_depth) {
  da <- sd_image_to_array(o15$dep)[, , 1]
  depth_ok <- o15$dep$width == W && o15$dep$height == H &&
    o15$dep$channel == 3L && diff(range(da)) > 0.5
  cat(sprintf("depth: %.2fs | %dx%dx%d | range %.2f..%.2f | saved %s\n",
              evp$dep, o15$dep$width, o15$dep$height, o15$dep$channel,
              min(da), max(da), out_depth))
} else {
  cat("depth: skipped, not found", depth_file, "\n")
}
res$pre <- canny_ok && !isFALSE(depth_ok)
res$pre_depth <- depth_ok

# --- 16. batch / loopback ---------------------------------------------------
cat("\n--- 16. sd_run_batch / sd_loopback ---\n")
ev16 <- data.frame(run = integer(0), event = character(0), type = character(0))
log16 <- function(e) ev16[nrow(ev16) + 1L, ] <<- list(e$run, e$event, e$type)
started <- function(type) {
  sum(ev16$event == "start" & ev16$type == type)
}

# (a) zip over seeds
pipe_b <- sd_pipeline(
  sd_node("encode_text", id = "txt", prompt = prompt, width = W, height = H),
  sample_node("lat", cond = "txt", width = W, height = H),
  sd_node("decode", id = "img", latent = "lat"),
  sd_node("save", image = "img", path = file.path(out_dir, "sd2R_graph_flux3_batch_{i}.png"))
)
r16a <- timed(sd_run_batch(pipe_b, ctx, vary = list(lat = list(seed = c(1L, 2L, 3L))),
                           outputs = "img", on_event = log16))
ba <- r16a$val
print(ba)
imgs_a <- lapply(ba$results, `[[`, "img")
distinct_a <- length(unique(lapply(imgs_a, `[[`, "data"))) == 3L
enc_once <- started("encode_text") == 1L
cat(sprintf("zip: %.2fs | 3 distinct images: %s | encode_text ran %d time(s)\n",
            r16a$t, distinct_a, started("encode_text")))

# (b) grid seed x strength, img2img on a run-time input image
ev16 <- ev16[0, ]
pipe_g <- sd_pipeline(
  sd_node("input_image", id = "src"),
  sd_node("encode_image", id = "init", image = "src"),
  sd_node("encode_text", id = "txt", prompt = "a dog sitting on a chair, oil painting",
          width = W, height = H),
  sample_node("lat", cond = "txt", latent = "init", strength = 0.5),
  sd_node("decode", id = "img", latent = "lat")
)
r16b <- timed(sd_run_batch(pipe_g, ctx, mode = "grid",
                           vary = list(lat = list(seed = c(1L, 2L), strength = c(0.3, 0.6))),
                           inputs = list(src = img_t2i), outputs = "img",
                           on_event = log16))
bg <- r16b$val
print(bg)
grid_ok <- nrow(bg$values) == 4L && !any(bg$failed) &&
  started("encode_image") == 1L && started("encode_text") == 1L
cat(sprintf("grid: %.2fs | 4 runs, encode_image / encode_text ran %d / %d time(s)\n",
            r16b$t, started("encode_image"), started("encode_text")))

# (c) loopback img2img with decaying strength
ev16 <- ev16[0, ]
pipe_l <- sd_pipeline(
  sd_node("input_image", id = "src"),
  sd_node("encode_image", id = "init", image = "src"),
  sd_node("encode_text", id = "txt", prompt = "a cat sitting on a chair, watercolor",
          width = W, height = H),
  sample_node("lat", cond = "txt", latent = "init", strength = 0.5),
  sd_node("decode", id = "img", latent = "lat"),
  sd_node("save", image = "img", path = file.path(out_dir, "sd2R_graph_flux3_loop_{i}.png"))
)
r16c <- timed(sd_loopback(pipe_l, ctx, n = 3, from = "img", to = "src", init = img_t2i,
                          vary = list(lat = list(strength = c(0.5, 0.4, 0.3))),
                          keep = "all", on_event = log16))
lp <- r16c$val
loop_files <- file.exists(file.path(out_dir, sprintf("sd2R_graph_flux3_loop_%d.png", 1:3)))
d_step <- vapply(seq_along(lp), function(k) {
  prev <- if (k == 1L) img_t2i else lp[[k - 1L]]
  mean(abs(as.integer(lp[[k]]$data) - as.integer(prev$data))) / 255
}, 0)
loop_ok <- length(lp) == 3L && all(loop_files) && started("encode_text") == 1L &&
  started("encode_image") == 3L
cat(sprintf(paste0("loopback: %.2fs | 3 iterations saved: %s | encode_text ran %d, ",
                   "encode_image %d time(s)\n  mean |change| per iteration: %s\n"),
            r16c$t, all(loop_files), started("encode_text"), started("encode_image"),
            paste(sprintf("%.3f", d_step), collapse = " -> ")))
res$batch <- distinct_a && enc_once && grid_ok && loop_ok

# --- Summary ----------------------------------------------------------------
cat("\n=== Summary ===\n")
cat(sprintf("1. geometry C++ vs encode_image: %s\n", if (res$geometry) "MATCH" else "MISMATCH"))
cat(sprintf("2. graph txt2img:                %.2fs\n", r2$t))
cat(sprintf("3. graph vs direct latent:       %s (max diff %.3g)\n",
            if (res$graph_vs_direct) "MATCH" else "MISMATCH", max_diff))
cat(sprintf("4. graph img2img:                %.2fs\n", r4$t))
cat(sprintf("5. branching (2 seeds):          %.2fs (a == step 2: %s, b differs: %s)\n",
            r5$t, res$branch_same, res$branch_diff))
cat(sprintf("6. sd_generate reference:        %.2fs\n", r6$t))
cat(sprintf("7. JSON round-trip:              %s\n", if (res$json) "OK" else "MISMATCH"))
cat(sprintf("8. outputs / culling:            %s / %s (%.2fs vs %.2fs for both branches)\n",
            if (res$outputs) "OK" else "MISMATCH", if (res$cull) "OK" else "MISMATCH",
            r8c$t, r5$t))
cat(sprintf("9. masks (bg replacement):       %s\n",
            if (is.na(res$mask)) "SKIPPED" else if (res$mask) "OK" else "MISMATCH"))
cat(sprintf("10. resize/composite/crop:       %s\n",
            if (is.na(res$comp)) "SKIPPED" else if (res$comp) "OK" else "MISMATCH"))
cat(sprintf("11. cache:                       %s (cold %.2fs, unchanged %.2fs, one branch %.2fs)\n",
            if (res$cache) "OK" else "MISMATCH", c1$t, c2$t, c3$t))
cat(sprintf("12. on_event / cancel:           %s (run %.2fs, cancelled after %.2fs)\n",
            if (res$events) "OK" else "MISMATCH", r12$t, r12c$t))
cat(sprintf("13. highres (image path):        %s (%.2fs; upscale_latent refused on FLUX.3)\n",
            if (res$highres) "OK" else "MISMATCH", r13$t))
cat(sprintf("14. sample(mask) / control check: %s\n",
            if (is.na(res$ll_mask)) "SKIPPED" else if (res$ll_mask) "OK" else "MISMATCH"))
cat(sprintf("15. canny / depth:               %s / %s\n",
            if (res$pre) "OK" else "MISMATCH",
            if (is.na(res$pre_depth)) "SKIPPED" else if (res$pre_depth) "OK" else "MISMATCH"))
cat(sprintf("16. batch / grid / loopback:     %s (%.2fs / %.2fs / %.2fs)\n",
            if (res$batch) "OK" else "MISMATCH", r16a$t, r16b$t, r16c$t))
cat("\n=== Done ===\n")
