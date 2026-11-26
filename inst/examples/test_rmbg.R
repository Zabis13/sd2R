#!/usr/bin/env Rscript
# Background removal with RMBG-1.4 (ggmlR ONNX) on a real model
#   1. load model, show inputs
#   2. sd_remove_background() -> RGBA + mask (cold / warm timing)
#   3. threshold (hard mask)
#   4. graph node: load_image -> remove_background -> save
#   5. CPU vs Vulkan mask agreement
# Input: the cat from test_graph_flux3.R (or any PNG).
# Run: Rscript /mnt/Data2/DS_projects/sd2R/inst/examples/test_rmbg.R

library(sd2R)

# ---- Paths (edit to match your setup) -------------------------------------
rmbg_file <- "/mnt/Data3/DS_Data/sd_models/RMBG-1.4.onnx"
in_file   <- "/tmp/sd2R_graph_flux3_t2i.png"
out_dir   <- "/tmp"
# ---------------------------------------------------------------------------

for (f in c(rmbg_file, in_file)) if (!file.exists(f)) stop("File not found: ", f)
timed <- function(expr) {
  t0 <- proc.time()
  val <- force(expr)
  list(val = val, t = (proc.time() - t0)[["elapsed"]])
}
fg_share <- function(mask) mean(sd_image_to_array(mask) > 0.5)

cat("=== sd2R RMBG-1.4 test ===\n")
cat("sd2R version:", as.character(utils::packageVersion("sd2R")),
    "| ggmlR:", as.character(utils::packageVersion("ggmlR")), "\n\n")

# --- 1. load ------------------------------------------------------------------
cat("--- 1. load ---\n")
r1 <- timed(sd_rmbg_load(rmbg_file))
rmbg <- r1$val
print(rmbg)
str(ggmlR::onnx_inputs(rmbg$model))
cat(sprintf("load: %.2fs\n", r1$t))

img <- sd_load_image(in_file)
cat(sprintf("input: %s (%dx%d)\n", in_file, img$width, img$height))

# --- 2. RGBA + mask -------------------------------------------------------------
cat("\n--- 2. sd_remove_background ---\n")
r2a <- timed(sd_remove_background(img, rmbg))
r2b <- timed(sd_remove_background(img, rmbg))
mask <- sd_remove_background(img, rmbg, output = "mask")
out_rgba <- file.path(out_dir, "sd2R_rmbg_rgba.png")
out_mask <- file.path(out_dir, "sd2R_rmbg_mask.png")
sd_save_image(r2b$val, out_rgba)
sd_save_image(mask, out_mask)
cat(sprintf("cold: %.2fs | warm: %.2fs | channels: %d | foreground: %.1f%%\n",
            r2a$t, r2b$t, r2b$val$channel, 100 * fg_share(mask)))
cat("saved:", out_rgba, out_mask, "\n")

# --- 3. threshold ---------------------------------------------------------------
cat("\n--- 3. threshold = 0.5 ---\n")
hard <- sd_remove_background(img, rmbg, output = "mask", threshold = 0.5)
vals <- unique(as.vector(sd_image_to_array(hard)))
out_hard <- file.path(out_dir, "sd2R_rmbg_mask_hard.png")
sd_save_image(hard, out_hard)
cat(sprintf("unique values: %s | foreground: %.1f%% | saved %s\n",
            paste(sort(vals), collapse = ", "), 100 * fg_share(hard), out_hard))

# --- 4. graph node --------------------------------------------------------------
cat("\n--- 4. graph: load_image -> remove_background -> save ---\n")
out_graph <- file.path(out_dir, "sd2R_rmbg_graph.png")
pipe <- sd_pipeline(
  sd_node("load_image", path = in_file),
  sd_node("remove_background", model_path = rmbg_file),
  sd_node("save", path = out_graph)
)
r4a <- timed(sd_run_pipeline(pipe, ctx = NULL, verbose = TRUE))
r4b <- timed(sd_run_pipeline(pipe, ctx = NULL))   # model cached
same <- identical(r4b$val$data, r2b$val$data)
cat(sprintf("first run: %.2fs | cached: %.2fs | same as step 2: %s | saved %s\n",
            r4a$t, r4b$t, same, out_graph))

# --- 5. CPU vs Vulkan -----------------------------------------------------------
cat("\n--- 5. CPU vs default device ---\n")
rmbg_cpu <- sd_rmbg_load(rmbg_file, device = "cpu")
r5 <- timed(sd_remove_background(img, rmbg_cpu, output = "mask"))
d <- max(abs(sd_image_to_array(r5$val) - sd_image_to_array(mask)))
cat(sprintf("CPU: %.2fs | max |diff| mask vs default device: %.3f -> %s\n",
            r5$t, d, if (d <= 2 / 255) "MATCH" else "DIFF"))

# --- Summary --------------------------------------------------------------------
cat("\n=== Summary ===\n")
cat(sprintf("1. load:                 %.2fs\n", r1$t))
cat(sprintf("2. remove bg cold/warm:  %.2fs / %.2fs (fg %.1f%%)\n",
            r2a$t, r2b$t, 100 * fg_share(mask)))
cat(sprintf("3. threshold values:     %s\n", paste(sort(vals), collapse = ", ")))
cat(sprintf("4. graph first/cached:   %.2fs / %.2fs (same: %s)\n", r4a$t, r4b$t, same))
cat(sprintf("5. CPU vs device:        %.2fs, diff %.3f\n", r5$t, d))
cat("\n=== Done ===\n")
