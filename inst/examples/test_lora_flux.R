#!/usr/bin/env Rscript
# LoRA test (FLUX.1-dev): explicit `loras` vs prompt tags, golden comparison
# with/without LoRA, both apply modes, foreign-architecture LoRA -> error.
# Run: Rscript /mnt/Data2/DS_projects/sd2R/inst/examples/test_lora_flux.R
#
# LoRA: XLabs realism LoRA, ComfyUI-converted (the one from upstream docs/flux.md)
#   https://huggingface.co/XLabs-AI/flux-lora-collection -> realism_lora_comfy_converted.safetensors

library(sd2R)

# ---- Paths (edit to match your setup) -------------------------------------
models_dir <- "/mnt/Data3/DS_Data/sd_models"
out_dir    <- "/tmp"
lora_name  <- "realism_lora_comfy_converted"        # looked up in lora_model_dir
lora_file  <- file.path(models_dir, paste0(lora_name, ".safetensors"))
# ---------------------------------------------------------------------------

if (!file.exists(lora_file)) stop("File not found: ", lora_file)

n_gpu  <- sd_vulkan_device_count()
layout <- if (n_gpu > 1L) "split_vae" else "mono"
prompt <- "a portrait photo of an old fisherman, natural light, detailed skin"
W <- 512L; H <- 512L; steps <- 12L

flux_ctx <- function(mode) {
  sd_ctx(diffusion_model_path = file.path(models_dir, "flux1-dev-Q4_K_S.gguf"),
         vae_path    = file.path(models_dir, "ae.safetensors"),
         clip_l_path = file.path(models_dir, "clip_l.safetensors"),
         t5xxl_path  = file.path(models_dir, "t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
         n_threads = 4L, model_type = "flux", vae_decode_only = FALSE,
         lora_apply_mode = mode, lora_model_dir = models_dir,
         verbose = FALSE, device_layout = layout)
}
gen <- function(ctx, p = prompt, loras = NULL) {
  sd_generate(ctx, p, width = W, height = H, sample_steps = steps, seed = 42L,
              sample_method = SAMPLE_METHOD$EULER, scheduler = SCHEDULER$DISCRETE,
              loras = loras)[[1]]
}
img_diff <- function(a, b) mean(abs(as.integer(a$data) - as.integer(b$data)))
show_stats <- function(ctx) {
  st <- sd_lora_stats(ctx)
  if (nrow(st)) st$path <- basename(st$path)
  print(st, row.names = FALSE)
}
expect_error <- function(label, expr) {
  r <- tryCatch({ force(expr); "NO ERROR (unexpected)" },
                error = function(e) paste("error:", conditionMessage(e)))
  cat(sprintf("%s -> %s\n", label, r))
}

cat("=== sd2R LoRA test (FLUX.1-dev) ===\n")

# --- 1-5. at_runtime (AUTO picks it for the quantized Q4_K_S model) ---
ctx <- flux_ctx(LORA_APPLY_MODE$AT_RUNTIME)

cat("\n--- 1. No LoRA ---\n")
img0 <- gen(ctx)
sd_save_image(img0, file.path(out_dir, "sd2R_lora_none.png"))

cat("\n--- 2. Explicit loras = c(name = 1.0), at_runtime ---\n")
img1 <- gen(ctx, loras = stats::setNames(1.0, lora_name))
show_stats(ctx)
sd_save_image(img1, file.path(out_dir, "sd2R_lora_runtime.png"))

cat("\n--- 3. Same LoRA via prompt tag ---\n")
img2 <- gen(ctx, p = paste0(prompt, " <lora:", lora_name, ":1.0>"))

cat("\n--- 4. Repeat explicit ---\n")
img3 <- gen(ctx, loras = stats::setNames(1.0, lora_name))

cat("\n--- 5. Tag 0.5 + explicit 0.5 (summed -> 1.0, message expected) ---\n")
img4 <- gen(ctx, p = paste0(prompt, " <lora:", lora_name, ":0.5>"),
            loras = stats::setNames(0.5, lora_name))

cat("\n--- 6. Missing LoRA ---\n")
expect_error("loras = c(no_such_lora = 1)", gen(ctx, loras = c(no_such_lora = 1)))

cat("\n--- Golden comparison (mean |diff| of 8-bit pixels) ---\n")
cat(sprintf("no LoRA vs LoRA:       %.3f  -> %s\n", img_diff(img0, img1),
            if (img_diff(img0, img1) > 0) "DIFFERENT (ok)" else "SAME (LoRA had no effect!)"))
cat(sprintf("explicit vs tag:       %.3f  -> %s\n", img_diff(img1, img2),
            if (img_diff(img1, img2) == 0) "MATCH" else "MISMATCH"))
cat(sprintf("explicit vs repeat:    %.3f  -> %s\n", img_diff(img1, img3),
            if (img_diff(img1, img3) == 0) "MATCH" else "MISMATCH"))
cat(sprintf("1.0 vs 0.5+0.5 summed: %.3f  -> %s\n", img_diff(img1, img4),
            if (img_diff(img1, img4) == 0) "MATCH" else "MISMATCH"))
sd_destroy_context(ctx)

# --- 7. immediately requested on Q4_K (Vulkan cannot CPY f32 -> K-quants) ---
# Expected: R warning in sd_ctx(), effective mode at_runtime, image == test 2.
cat("\n--- 7. immediately requested (Q4_K_S model) ---\n")
ctx <- withCallingHandlers(flux_ctx(LORA_APPLY_MODE$IMMEDIATELY),
                           warning = function(w) {
                             cat("warning:", conditionMessage(w), "\n")
                             invokeRestart("muffleWarning")
                           })
cat("effective mode:", sd_lora_apply_mode(ctx), "| attr:", attr(ctx, "lora_apply_mode"), "\n")
img5 <- gen(ctx, loras = stats::setNames(1.0, lora_name))
show_stats(ctx)
sd_save_image(img5, file.path(out_dir, "sd2R_lora_immediately.png"))
cat(sprintf("immediately(requested) vs at_runtime: %.3f -> %s\n", img_diff(img5, img1),
            if (img_diff(img5, img1) == 0) "MATCH (fell back to at_runtime)"
            else "differs (merged: requantization)"))
cat(sprintf("immediately vs no LoRA:    %.3f\n", img_diff(img5, img0)))
img6 <- gen(ctx)  # LoRA must be removed again
cat(sprintf("immediately, then no LoRA vs no LoRA: %.3f -> %s\n", img_diff(img6, img0),
            if (img_diff(img6, img0) == 0) "MATCH" else "MISMATCH"))
sd_destroy_context(ctx)

# --- 8. Foreign LoRA: FLUX LoRA on SD1.5 -> error ---
cat("\n--- 8. FLUX LoRA on SD1.5 ---\n")
ctx <- sd_ctx(model_path = file.path(models_dir, "v1-5-pruned-emaonly.safetensors"),
              n_threads = 4L, model_type = "sd1", lora_model_dir = models_dir,
              verbose = FALSE, device_layout = layout)
cat("effective mode (AUTO, F16/F32 weights):", sd_lora_apply_mode(ctx), "\n")
expect_error("SD1.5 + FLUX LoRA",
             sd_generate(ctx, "a cat", width = 512L, height = 512L, sample_steps = 4L,
                         seed = 42L, loras = stats::setNames(1.0, lora_name)))
show_stats(ctx)
sd_destroy_context(ctx)

# --- 9. Foreign LoRA: FLUX.1 LoRA on FLUX.2 Klein -> error ---
cat("\n--- 9. FLUX.1 LoRA on FLUX.2 Klein ---\n")
ctx <- sd_ctx(diffusion_model_path = file.path(models_dir, "flux-2-klein-4b.safetensors"),
              vae_path = file.path(models_dir, "flux2-vae.safetensors"),
              llm_path = file.path(models_dir, "Qwen3-4B-Q4_K_S.gguf"),
              n_threads = 4L, model_type = "flux2", lora_model_dir = models_dir,
              verbose = FALSE, device_layout = layout)
expect_error("FLUX.2 Klein + FLUX.1 LoRA",
             sd_generate(ctx, "a cat", width = 512L, height = 512L, sample_steps = 4L,
                         seed = 42L, loras = stats::setNames(1.0, lora_name)))
show_stats(ctx)
sd_destroy_context(ctx)

cat("\n=== Done ===\n")
