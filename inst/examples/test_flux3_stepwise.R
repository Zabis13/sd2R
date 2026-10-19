#!/usr/bin/env Rscript
# FLUX.3 low-level check only: sd_sample (C++ loop) vs sd_sample_stepwise
# (R loop) on the same explicit noise, plus diagnostics and the sd_denoise_step no-target guard.
# Run: Rscript /mnt/Data2/DS_projects/sd2R/inst/examples/test_flux3_stepwise.R
#
# Euler must match: stepwise passes sigma_next = sigmas[i + 1] as the flow-map
# target timestep, the C++ sampler uses sigmas[step]. No sd_generate() here —
# the latent shape for 768x768 is fixed (FLUX.2 VAE: /16, 128 channels).

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

cat("=== sd2R FLUX.3 sd_sample vs sd_sample_stepwise ===\n\n")
n_gpu <- sd_vulkan_device_count()
ctx <- sd_ctx(diffusion_model_path = diffusion_file,
              vae_path = vae_file, llm_path = llm_file,
              n_threads = 4L, model_type = "flux3",
              vae_decode_only = FALSE, verbose = FALSE,
              device_layout = if (n_gpu > 1L) "split_vae" else "mono")

W <- 768L; H <- 768L; steps <- 16L
shape <- c(W %/% 16L, H %/% 16L, 128L)
cond  <- sd_encode_text(ctx, "a cat sitting on a chair, oil painting",
                        width = W, height = H)
set.seed(42L)
noise <- list(type = SD_TYPE$F32, ne = as.integer(c(shape, 1L)),
              data = stats::rnorm(prod(as.numeric(shape))))
diff_lat <- function(a, b) max(abs(a$data - b$data))
sigmas_of <- function(n) sd_sampler_sigmas(ctx, SCHEDULER$DISCRETE, n, W, H,
                                           SAMPLE_METHOD$EULER)

# --- 1. Default schedule (as in test_generate_flux3.R, test 7) ---
cat("--- 1. Default schedule, 16 steps ---\n")
sig_before <- sigmas_of(steps)
t0 <- proc.time()
lat_loop <- sd_sample(ctx, cond, noise = noise,
                      sample_method = SAMPLE_METHOD$EULER,
                      scheduler = SCHEDULER$DISCRETE,
                      sample_steps = steps, cfg_scale = 1.0)
t_loop <- (proc.time() - t0)[["elapsed"]]
t0 <- proc.time()
lat_step <- sd_sample_stepwise(ctx, cond, noise = noise, width = W, height = H,
                               sample_method = SAMPLE_METHOD$EULER,
                               scheduler = SCHEDULER$DISCRETE,
                               sample_steps = steps, cfg_scale = 1.0)
t_step <- (proc.time() - t0)[["elapsed"]]
d1 <- diff_lat(lat_loop, lat_step)
cat(sprintf("sd_sample: %.2fs | stepwise: %.2fs | max |diff| latent: %.3g -> %s\n",
            t_loop, t_step, d1, if (d1 < 1e-3) "MATCH" else "MISMATCH"))
sd_save_image(sd_decode_latent(ctx, lat_loop), file.path(out_dir, "sd2R_flux3_ll_loop.png"))
sd_save_image(sd_decode_latent(ctx, lat_step), file.path(out_dir, "sd2R_flux3_ll_stepwise.png"))
cat("Saved: /tmp/sd2R_flux3_ll_loop.png, /tmp/sd2R_flux3_ll_stepwise.png\n")

# --- 2. Sigma schedule before / after sd_sample (set_flow_shift) ---
cat("\n--- 2. Sigma schedule ---\n")
sig_after <- sigmas_of(steps)
cat("before sd_sample:", sprintf("%.4f", sig_before), "\n")
cat("after  sd_sample:", sprintf("%.4f", sig_after), "\n")
cat(sprintf("max |diff| before/after: %.3g\n", max(abs(sig_before - sig_after))))

# --- 3. Same explicit schedule for both ---
cat("\n--- 3. custom_sigmas, 16 steps ---\n")
lat_loop_cs <- sd_sample(ctx, cond, noise = noise,
                         sample_method = SAMPLE_METHOD$EULER,
                         custom_sigmas = sig_after, cfg_scale = 1.0)
lat_step_cs <- sd_sample_stepwise(ctx, cond, noise = noise, width = W, height = H,
                                  sample_method = SAMPLE_METHOD$EULER,
                                  custom_sigmas = sig_after, cfg_scale = 1.0)
cat(sprintf("loop vs stepwise: %.3g | loop vs loop(default): %.3g | stepwise vs stepwise(default): %.3g\n",
            diff_lat(lat_loop_cs, lat_step_cs), diff_lat(lat_loop_cs, lat_loop),
            diff_lat(lat_step_cs, lat_step)))

# --- 4. Single step ---
cat("\n--- 4. Single step ---\n")
sig1 <- sigmas_of(1L)
cat("sigmas:", sprintf("%.4f", sig1), "\n")
lat_loop_1 <- sd_sample(ctx, cond, noise = noise,
                        sample_method = SAMPLE_METHOD$EULER,
                        custom_sigmas = sig1, cfg_scale = 1.0)
lat_step_1 <- sd_sample_stepwise(ctx, cond, noise = noise, width = W, height = H,
                                 sample_method = SAMPLE_METHOD$EULER,
                                 custom_sigmas = sig1, cfg_scale = 1.0)
cat(sprintf("loop vs stepwise: %.3g (range loop [%.3f, %.3f])\n",
            diff_lat(lat_loop_1, lat_step_1),
            min(lat_loop_1$data), max(lat_loop_1$data)))

# --- 5. Low-level: sd_denoise_step without sigma_next -> R error, not abort ---
cat("\n--- 5. sd_denoise_step without sigma_next ---\n")
sigmas <- sd_sampler_sigmas(ctx, SCHEDULER$DISCRETE, steps, W, H,
                            SAMPLE_METHOD$EULER)
x0 <- sd_noise_scale(ctx, noise, sigmas[1])
empty <- list(crossattn = NULL, vector = NULL, concat = NULL)
sd_sampler_begin(ctx)
# 5a. R-level early check (model_type == "flux3")
r5a <- tryCatch({ sd_denoise_step(ctx, x0, sigmas[1], cond, cfg_scale = 1.0); "no error" },
                error = function(e) paste("error:", conditionMessage(e)))
cat("5a (R check):   ", r5a, "\n")
# 5b. C++ guard, bypassing the R check (sigma_next = -1)
r5b <- tryCatch({ sd2R:::sd_denoise_step_cpp(ctx, x0, sigmas[1], cond, empty,
                                             1.0, 1L, 16L, -1); "no error" },
                error = function(e) paste("error:", conditionMessage(e)))
cat("5b (C++ guard): ", r5b, "\n")
cat("  (flow-map Klein: both must be errors; Klein Base: 5b = no error)\n")
# 5c. Last step: sigma_next = 0 is a valid target
r5c <- tryCatch({ sd_denoise_step(ctx, x0, sigmas[16], cond, cfg_scale = 1.0,
                                  step = 16L, total_steps = 16L, sigma_next = 0); "ok" },
                error = function(e) paste("error:", conditionMessage(e)))
cat("5c (sigma_next = 0): ", r5c, "\n")
sd_sampler_end(ctx)

rm(ctx)
gc()
cat("\n=== Done ===\n")
