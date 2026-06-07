# Extracted from test-full-coverage.R:281

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "sd2R", path = "..")
attach(test_env, warn.conflicts = FALSE)

# prequel ----------------------------------------------------------------------
test_model_path <- function() {
  p <- Sys.getenv("SD2R_TEST_MODEL", unset = "")
  if (!nzchar(p)) p <- "/mnt/Data2/DS_projects/sd_models/v1-5-pruned-emaonly.safetensors"
  p
}
skip_if_no_model <- function() {
  skip_if(!file.exists(test_model_path()), "SD2R_TEST_MODEL not set or file missing")
}

# test -------------------------------------------------------------------------
skip_if_no_model()
ctx <- sd_ctx(test_model_path(), vae_decode_only = FALSE, verbose = FALSE)
init <- list(width = 256L, height = 256L, channel = 3L,
               data = as.raw(rep(128L, 256 * 256 * 3)))
imgs <- sd_img2img(ctx, init_image = init, prompt = "a blue square",
                     width = 256L, height = 256L, sample_steps = 2L,
                     strength = 0.5, seed = 1L)
