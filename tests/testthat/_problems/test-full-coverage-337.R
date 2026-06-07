# Extracted from test-full-coverage.R:337

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
progress_file <- tempfile(fileext = ".json")
sd2R:::sd_set_progress_file(progress_file)
params <- list(
    prompt = "test async",
    negative_prompt = "",
    width = 256L, height = 256L,
    sample_method = SAMPLE_METHOD$EULER,
    sample_steps = 3L,
    cfg_scale = 7.0,
    seed = 42L,
    scheduler = SCHEDULER$DISCRETE,
    batch_count = 1L
  )
sd2R:::sd_generate_async(ctx, params)
for (i in 1:120) {
    status <- sd2R:::sd_generate_poll()
    if (status$done) break
    Sys.sleep(0.5)
  }
expect_true(status$done)
imgs <- sd2R:::sd_generate_result()
expect_equal(length(imgs), 1)
expect_equal(imgs[[1]]$width, 256L)
expect_true(file.exists(progress_file))
sd2R:::sd_clear_progress_file()
