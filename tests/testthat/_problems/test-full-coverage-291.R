# Extracted from test-full-coverage.R:291

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
ctx <- sd_ctx(test_model_path(), verbose = FALSE)
sd_profile_start()
sd_txt2img(ctx, prompt = "test", width = 256L, height = 256L,
             sample_steps = 2L, seed = 1L)
