# Comprehensive tests for all exported functions
# Functions requiring a model use skip_if based on SD2R_TEST_MODEL env var

# --- Helper: get test model path ---
# Set SD2R_TEST_MODEL env var, or defaults to SD 1.5 in standard location
test_model_path <- function() {
  p <- Sys.getenv("SD2R_TEST_MODEL", unset = "")
  if (!nzchar(p)) p <- "/mnt/Data2/DS_projects/sd_models/v1-5-pruned-emaonly.safetensors"
  p
}

skip_if_no_model <- function() {
  skip_if(!file.exists(test_model_path()), "SD2R_TEST_MODEL not set or file missing")
}


# ============================================================
# Constants
# ============================================================

test_that("LORA_APPLY_MODE constant is defined", {
  expect_true(is.integer(LORA_APPLY_MODE$AUTO))
  expect_equal(LORA_APPLY_MODE$AUTO, 0L)
  expect_true(length(LORA_APPLY_MODE) >= 1)
})

test_that("SD_CACHE_MODE constant has expected entries", {
  expect_equal(SD_CACHE_MODE$DISABLED, 0L)
  expect_true("EASYCACHE" %in% names(SD_CACHE_MODE))
})


# ============================================================
# sd_cache_params
# ============================================================

test_that("sd_cache_params returns correct structure", {
  cp <- sd_cache_params()
  expect_true(is.list(cp))
  expect_true(all(c("cache_mode", "cache_threshold", "cache_start", "cache_end") %in% names(cp)))
  expect_equal(cp$cache_threshold, 1.0)
  expect_equal(cp$cache_start, 0.15)
  expect_equal(cp$cache_end, 0.95)
})

test_that("sd_cache_params respects custom values", {
  cp <- sd_cache_params(mode = SD_CACHE_MODE$DISABLED, threshold = 0.5,
                         start_percent = 0.1, end_percent = 0.8)
  expect_equal(cp$cache_mode, as.integer(SD_CACHE_MODE$DISABLED))
  expect_equal(cp$cache_threshold, 0.5)
  expect_equal(cp$cache_start, 0.1)
  expect_equal(cp$cache_end, 0.8)
})


# ============================================================
# sd_vulkan_device_count
# ============================================================

test_that("sd_vulkan_device_count returns integer >= 0", {
  n <- sd_vulkan_device_count()
  expect_true(is.numeric(n))
  expect_true(n >= 0)
})


# ============================================================
# Profiling API
# ============================================================

test_that("sd_profile_start/stop/get work without data", {
  sd_profile_start()
  sd_profile_stop()
  df <- sd_profile_get()
  expect_true(is.data.frame(df))
  expect_true(all(c("stage", "kind", "timestamp_ms") %in% names(df)))
})

test_that("sd_profile_summary handles empty events", {
  sd_profile_start()
  sd_profile_stop()
  df <- sd_profile_get()
  summ <- sd_profile_summary(df)
  expect_true(is.data.frame(summ) || nrow(summ) == 0L)
})

test_that("sd_profile_summary returns empty for no events", {
  sd_profile_start()
  sd_profile_stop()
  df <- sd_profile_get()
  summ <- sd_profile_summary(df)
  expect_equal(nrow(summ), 0L)
})


# ============================================================
# sd_convert — signature only (needs model for real test)
# ============================================================

test_that("sd_convert rejects missing input", {
  expect_error(sd_convert("/no/such/model.safetensors", "out.gguf"),
               "not found")
})

test_that("sd_convert has correct signature", {
  args <- formals(sd_convert)
  expect_true(all(c("input_path", "output_path", "output_type") %in% names(args)))
})


# ============================================================
# sd_upscale_image — signature only
# ============================================================

test_that("sd_upscale_image rejects missing model", {
  img <- list(width = 4L, height = 4L, channel = 3L,
              data = as.raw(rep(128, 48)))
  expect_error(sd_upscale_image("/no/such/esrgan.safetensors", img),
               "not found")
})

test_that("sd_upscale_image has correct signature", {
  args <- formals(sd_upscale_image)
  expect_true(all(c("esrgan_path", "image", "upscale_factor") %in% names(args)))
})


# ============================================================
# sd_generate / sd_generate_multi_gpu — signature
# ============================================================

test_that("sd_generate has correct signature", {
  args <- formals(sd_generate)
  expect_true(all(c("ctx", "prompt", "width", "height", "seed") %in% names(args)))
})

test_that("sd_generate_multi_gpu has correct signature", {
  args <- formals(sd_generate_multi_gpu)
  expect_true(all(c("prompts", "devices", "model_path") %in% names(args)))
})


# ============================================================
# Model manager (uses temp registry)
# ============================================================

test_that("sd_list_models returns data frame", {
  skip_if_not_installed("jsonlite")
  df <- sd_list_models()
  expect_true(is.data.frame(df))
  expect_true(all(c("id", "model_type") %in% names(df)))
})

test_that("sd_register_model / sd_remove_model roundtrip", {
  skip_if_not_installed("jsonlite")
  # Use a temp model path that "exists"
  tmp <- tempfile(fileext = ".safetensors")
  writeLines("fake", tmp)
  on.exit(unlink(tmp))

  id <- paste0("test-model-", as.integer(Sys.time()))
  sd_register_model(id, "sd1", paths = list(model = tmp), overwrite = TRUE)

  df <- sd_list_models()
  expect_true(id %in% df$id)

  # Remove
  expect_message(sd_remove_model(id), "Removed")
  df2 <- sd_list_models()
  expect_false(id %in% df2$id)
})

test_that("sd_register_model rejects duplicate without overwrite", {
  skip_if_not_installed("jsonlite")
  tmp <- tempfile(fileext = ".safetensors")
  writeLines("fake", tmp)
  on.exit({
    tryCatch(sd_remove_model("dup-test"), error = function(e) NULL)
    unlink(tmp)
  })

  sd_register_model("dup-test", "sd1", paths = list(model = tmp), overwrite = TRUE)
  expect_error(sd_register_model("dup-test", "sd1", paths = list(model = tmp)),
               "already registered")
})

test_that("sd_unload_model on non-loaded is silent", {
  expect_message(sd_unload_model("nonexistent-id"), "not loaded")
})

test_that("sd_unload_all works", {
  expect_message(sd_unload_all(), "Unloaded")
})

test_that("sd_scan_models rejects missing dir", {
  expect_error(sd_scan_models("/nonexistent/dir"), "not found")
})

test_that("sd_scan_models handles empty dir", {
  skip_if_not_installed("jsonlite")
  td <- tempfile()
  dir.create(td)
  on.exit(unlink(td, recursive = TRUE))
  expect_message(sd_scan_models(td), "No model files")
})


# ============================================================
# Async C++ API
# ============================================================

test_that("sd_create_context_poll returns correct structure", {
  status <- sd2R:::sd_create_context_poll()
  expect_true(is.list(status))
  expect_true(all(c("running", "done") %in% names(status)))
  expect_false(status$running)
})

test_that("sd_set_log_file and sd_clear_log_file work", {
  tmp <- tempfile(fileext = ".txt")
  expect_silent(sd2R:::sd_set_log_file(tmp))
  expect_silent(sd2R:::sd_clear_log_file())
})


# ============================================================
# API (plumber) — signature only
# ============================================================

test_that("sd_api_start has correct signature", {
  args <- formals(sd_api_start)
  expect_true("port" %in% names(args))
})

test_that("sd_api_stop is a function", {
  expect_true(is.function(sd_api_stop))
})


# ============================================================
# Model-dependent tests (require SD2R_TEST_MODEL env var)
# ============================================================

test_that("sd_ctx loads real model", {
  skip_if_no_model()
  ctx <- sd_ctx(test_model_path(), verbose = FALSE)
  expect_s3_class(ctx, "sd_ctx")
})

test_that("sd_txt2img generates image", {
  skip_if_no_model()
  ctx <- sd_ctx(test_model_path(), verbose = FALSE)
  imgs <- sd_txt2img(ctx, prompt = "a red circle on white background",
                     width = 256L, height = 256L, sample_steps = 3L, seed = 1L)
  expect_true(is.list(imgs))
  expect_equal(length(imgs), 1)
  expect_equal(imgs[[1]]$width, 256L)
  expect_equal(imgs[[1]]$height, 256L)
  expect_equal(imgs[[1]]$channel, 3L)
  expect_true(length(imgs[[1]]$data) > 0)
})

test_that("sd_generate auto-selects strategy", {
  skip_if_no_model()
  ctx <- sd_ctx(test_model_path(), verbose = FALSE)
  imgs <- sd_generate(ctx, prompt = "test", width = 256L, height = 256L,
                      sample_steps = 2L, seed = 1L)
  expect_true(is.list(imgs))
  expect_true(length(imgs) >= 1)
  expect_equal(imgs[[1]]$width, 256L)
})

test_that("sd_img2img works with init image", {
  skip_if_no_model()
  ctx <- sd_ctx(test_model_path(), vae_decode_only = FALSE, verbose = FALSE)
  # Create a dummy init image
  init <- list(width = 256L, height = 256L, channel = 3L,
               data = as.raw(rep(128L, 256 * 256 * 3)))
  imgs <- sd_img2img(ctx, init_image = init, prompt = "a blue square",
                     width = 256L, height = 256L, sample_steps = 2L,
                     strength = 0.5, seed = 1L)
  expect_true(is.list(imgs))
  expect_equal(imgs[[1]]$width, 256L)
})

test_that("profiling captures events during generation", {
  skip_if_no_model()
  ctx <- sd_ctx(test_model_path(), verbose = FALSE)
  sd_profile_start()
  sd_txt2img(ctx, prompt = "test", width = 256L, height = 256L,
             sample_steps = 2L, seed = 1L)
  sd_profile_stop()
  df <- sd_profile_get()
  expect_true(nrow(df) > 0)

  summ <- sd_profile_summary(df)
  expect_true(nrow(summ) > 0)
  expect_true("sampling" %in% summ$stage || "generate_total" %in% summ$stage)
})

test_that("async generation works end-to-end", {
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

  # Poll until done (with timeout)
  for (i in 1:120) {
    status <- sd2R:::sd_generate_poll()
    if (status$done) break
    Sys.sleep(0.5)
  }
  expect_true(status$done)

  imgs <- sd2R:::sd_generate_result()
  expect_equal(length(imgs), 1)
  expect_equal(imgs[[1]]$width, 256L)

  # Progress file should have been written
  expect_true(file.exists(progress_file))
  sd2R:::sd_clear_progress_file()
})

test_that("async context creation works end-to-end", {
  skip_if_no_model()

  log_file <- tempfile(fileext = ".txt")
  sd2R:::sd_set_log_file(log_file)
  sd2R:::sd_set_verbose(TRUE)

  params <- list(
    model_path = normalizePath(test_model_path()),
    vae_decode_only = TRUE,
    diffusion_flash_attn = TRUE,
    n_threads = 0L,
    wtype = as.integer(SD_TYPE$COUNT),
    rng_type = as.integer(RNG_TYPE$CUDA),
    lora_apply_mode = as.integer(LORA_APPLY_MODE$AUTO),
    flow_shift = 0.0
  )

  sd2R:::sd_create_context_async(params)

  for (i in 1:240) {
    status <- sd2R:::sd_create_context_poll()
    if (status$done) break
    Sys.sleep(0.5)
  }
  expect_true(status$done)

  ctx <- sd2R:::sd_create_context_result()
  expect_s3_class(ctx, "sd_ctx")

  sd2R:::sd_clear_log_file()
  sd2R:::sd_set_verbose(FALSE)
})
