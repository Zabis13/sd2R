# Tests for Shiny app helper functions

# Source only the helper functions from app.R (skip shiny UI/server)
local({
  app_file <- system.file("shiny", "sd2R_app", "app.R", package = "sd2R")
  if (!nzchar(app_file)) {
    app_file <- file.path(getwd(), "../../inst/shiny/sd2R_app/app.R")
  }
  lines <- readLines(app_file)

  # Extract auto_assign_roles function and MODEL_PRESETS
  # Find start/end of auto_assign_roles
  start_presets <- grep("^MODEL_PRESETS <- list", lines)
  start_fn <- grep("^auto_assign_roles <- function", lines)

  # Find the closing brace of auto_assign_roles (matching depth)
  depth <- 0
  end_fn <- NA
  for (i in start_fn:length(lines)) {
    depth <- depth + nchar(gsub("[^{]", "", lines[i])) -
                     nchar(gsub("[^}]", "", lines[i]))
    if (depth == 0) { end_fn <- i; break }
  }

  # Find end of MODEL_PRESETS
  depth <- 0
  end_presets <- NA
  for (i in start_presets:length(lines)) {
    depth <- depth + nchar(gsub("[^(]", "", lines[i])) -
                     nchar(gsub("[^)]", "", lines[i]))
    if (depth == 0) { end_presets <- i; break }
  }

  # auto_assign_roles uses %||%, which lives near the top of app.R (outside the
  # extracted ranges). Define it here so the extracted function is self-contained.
  helper <- "`%||%` <- function(a, b) if (is.null(a)) b else a"

  code <- c(helper, "", lines[start_presets:end_presets], "", lines[start_fn:end_fn])
  eval(parse(text = code), envir = .GlobalEnv)
})


# --- auto_assign_roles tests ---

test_that("auto_assign_roles detects Flux architecture", {
  d <- tempdir()
  td <- file.path(d, "test_flux")
  dir.create(td, showWarnings = FALSE)
  on.exit(unlink(td, recursive = TRUE))

  # Create fake model files with different sizes
  writeLines(rep("x", 1000), file.path(td, "flux1-dev-Q4_K_S.gguf"))
  writeLines(rep("x", 500),  file.path(td, "clip_l.safetensors"))
  writeLines(rep("x", 800),  file.path(td, "t5-v1_1-xxl-encoder-Q5_K_M.gguf"))
  writeLines(rep("x", 300),  file.path(td, "ae.safetensors"))
  writeLines(rep("x", 600),  file.path(td, "v1-5-pruned-emaonly.safetensors"))

  roles <- auto_assign_roles(td)

  expect_equal(roles$arch, "flux")
  expect_equal(roles$diffusion, "flux1-dev-Q4_K_S.gguf")
  expect_equal(roles$clip_l, "clip_l.safetensors")
  expect_equal(roles$t5xxl, "t5-v1_1-xxl-encoder-Q5_K_M.gguf")
  expect_equal(roles$vae, "ae.safetensors")
  # Flux: model should be empty (no single-file checkpoint)
  expect_equal(roles$model, "")
})

test_that("auto_assign_roles detects SD1 architecture", {
  d <- tempdir()
  td <- file.path(d, "test_sd1")
  dir.create(td, showWarnings = FALSE)
  on.exit(unlink(td, recursive = TRUE))

  writeLines(rep("x", 1000), file.path(td, "v1-5-pruned-emaonly.safetensors"))

  roles <- auto_assign_roles(td)

  expect_equal(roles$arch, "sd1")
  expect_equal(roles$model, "v1-5-pruned-emaonly.safetensors")
  expect_equal(roles$diffusion, "")
})

test_that("auto_assign_roles detects SDXL architecture", {
  d <- tempdir()
  td <- file.path(d, "test_sdxl")
  dir.create(td, showWarnings = FALSE)
  on.exit(unlink(td, recursive = TRUE))

  writeLines(rep("x", 1000), file.path(td, "sdxl_base_1.0.safetensors"))
  writeLines(rep("x", 200),  file.path(td, "sdxl_vae_fp16.safetensors"))

  roles <- auto_assign_roles(td)

  expect_equal(roles$arch, "sdxl")
  expect_equal(roles$model, "sdxl_base_1.0.safetensors")
  # SDXL bundles its VAE inside the checkpoint, so no external VAE is assigned.
  expect_equal(roles$vae, "")
})

test_that("auto_assign_roles returns sd1 for empty dir", {
  d <- tempdir()
  td <- file.path(d, "test_empty")
  dir.create(td, showWarnings = FALSE)
  on.exit(unlink(td, recursive = TRUE))

  roles <- auto_assign_roles(td)
  expect_equal(roles$arch, "sd1")
})

test_that("auto_assign_roles does not assign SD1.5 as model for Flux", {
  # Regression: previously SD1.5 checkpoint was assigned as model alongside Flux
  d <- tempdir()
  td <- file.path(d, "test_flux_no_sd15")
  dir.create(td, showWarnings = FALSE)
  on.exit(unlink(td, recursive = TRUE))

  writeLines(rep("x", 2000), file.path(td, "flux1-dev-Q4_K_S.gguf"))
  writeLines(rep("x", 1500), file.path(td, "v1-5-pruned-emaonly.safetensors"))
  writeLines(rep("x", 100),  file.path(td, "ae.safetensors"))

  roles <- auto_assign_roles(td)

  expect_equal(roles$arch, "flux")
  expect_equal(roles$model, "",
    info = "SD1.5 checkpoint should NOT be assigned as model when arch is Flux")
  expect_equal(roles$diffusion, "flux1-dev-Q4_K_S.gguf")
})


# --- MODEL_PRESETS tests ---

test_that("MODEL_PRESETS has expected architectures", {
  expect_true(all(c("sd1", "sd2", "sdxl", "flux", "sd3") %in% names(MODEL_PRESETS)))
})

test_that("all presets have exactly 3 standard resolutions", {
  expected <- c("512x512", "768x768", "1024x1024")
  for (arch in names(MODEL_PRESETS)) {
    expect_equal(MODEL_PRESETS[[arch]]$resolutions, expected,
      info = sprintf("Resolutions for %s", arch))
  }
})

test_that("each preset has required fields", {
  required <- c("label", "width", "height", "steps", "cfg",
                 "sampler", "scheduler", "max_chars", "resolutions")
  for (arch in names(MODEL_PRESETS)) {
    for (field in required) {
      expect_true(field %in% names(MODEL_PRESETS[[arch]]),
        info = sprintf("Missing field '%s' in preset '%s'", field, arch))
    }
  }
})


# --- sd_app function tests ---

test_that("sd_app exists and is a function", {
  expect_true(is.function(sd2R::sd_app))
})

test_that("sd_app requires shiny package", {
  # Just check formals — actual launch would block
  fmls <- formals(sd2R::sd_app)
  expect_true("model_dir" %in% names(fmls))
  expect_true("launch.browser" %in% names(fmls))
  expect_true("port" %in% names(fmls))
})


# --- Async C++ interface tests ---

test_that("sd_generate_poll returns correct structure", {
  status <- sd2R:::sd_generate_poll()
  expect_true(is.list(status))
  expect_true("running" %in% names(status))
  expect_true("done" %in% names(status))
  expect_false(status$running)
})

test_that("sd_set_progress_file and sd_clear_progress_file work", {
  tmp <- tempfile(fileext = ".json")
  expect_silent(sd2R:::sd_set_progress_file(tmp))
  expect_silent(sd2R:::sd_clear_progress_file())
  # After clear, file should not exist
  expect_false(file.exists(tmp))
})
