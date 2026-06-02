# Layer 1 (TODO 9.1/9.3): sd_default_params() + model_type = "auto" detection.

test_that("sd_default_params returns a complete editable baseline", {
  p <- sd_default_params()
  expect_type(p, "list")
  expect_true(all(c("negative_prompt", "width", "height", "strength",
                    "sample_method", "sample_steps", "cfg_scale", "seed",
                    "batch_count", "scheduler", "clip_skip", "eta",
                    "hr_strength", "vae_mode", "vae_tile_size",
                    "vae_tile_overlap", "cache_mode", "cache_config")
                  %in% names(p)))
  expect_identical(p$sample_steps, 20L)
  expect_identical(p$cfg_scale, 7.0)
  expect_identical(p$vae_mode, "auto")
})

test_that("param merge precedence: explicit > params > default", {
  # Emulates the pv() resolver used inside sd_generate().
  called <- "cfg_scale"
  params <- list(sample_steps = 40L)
  defs   <- sd_default_params()
  pv <- function(nm, val) {
    if (nm %in% called) return(val)
    if (nm %in% names(params)) return(params[[nm]])
    defs[[nm]]
  }
  expect_identical(pv("cfg_scale", 2.5), 2.5)       # explicit wins
  expect_identical(pv("sample_steps", 20L), 40L)    # params over default
  expect_identical(pv("seed", 42L), 42L)            # default fallback
})

test_that(".detect_model_type_gguf returns NULL for non-gguf or missing files", {
  # Guards (no model / ggmlR needed): only .gguf files that exist are probed.
  expect_null(.detect_model_type_gguf("model.safetensors")) # not a .gguf
  expect_null(.detect_model_type_gguf(""))                  # empty path
  expect_null(.detect_model_type_gguf("/no/such/file.gguf")) # missing file
})

test_that(".detect_model_type_gguf reads architecture from a real GGUF header", {
  gguf <- Sys.getenv("SD2R_TEST_GGUF", "")
  skip_if(gguf == "" || !file.exists(gguf),
          "Set SD2R_TEST_GGUF to a .gguf file to run this test")
  skip_if_not_installed("ggmlR")
  skip_if(!"meta_only" %in% names(formals(ggmlR::gguf_load)),
          "Installed ggmlR has no meta_only support")

  # Header-only probe maps general.architecture onto sd2R's type vocabulary.
  res <- .detect_model_type_gguf(gguf)
  expect_true(is.null(res) || res %in% c("sd1", "sd2", "sdxl", "flux", "sd3"))
})

test_that("model_type=auto falls back to filename heuristic", {
  expect_identical(.resolve_model_type("auto", "/tmp/flux1-dev-Q4.gguf"), "flux")
  expect_identical(.resolve_model_type("auto", "/tmp/sd_xl_base.safetensors"), "sdxl")
})

test_that("FLUX.2 is detected from filename and not shadowed by flux", {
  # flux2 must win over the generic "flux" match (order-sensitive).
  expect_identical(.guess_model_type("flux2-klein-9B.gguf"), "flux2")
  expect_identical(.guess_model_type("flux.2_base.safetensors"), "flux2")
  expect_identical(.guess_model_type("flux-2-dev.gguf"), "flux2")
  # plain flux1 must still resolve to "flux", never "flux2".
  expect_identical(.guess_model_type("flux1-dev-Q4_K_S.gguf"), "flux")
  expect_identical(.guess_model_type("flux-dev.safetensors"), "flux")
  expect_identical(.resolve_model_type("auto", "/tmp/flux2-klein.gguf"), "flux2")
})

test_that("FLUX.2 has native tile/latent sizes (same as flux)", {
  expect_identical(.native_tile_size("flux2"), 1024L)
  expect_identical(.native_latent_tile_size("flux2"), 128L)
})

test_that("non-auto model_type passes through unchanged", {
  expect_identical(.resolve_model_type("sd1", "irrelevant"), "sd1")
  expect_identical(.resolve_model_type("flux", ""), "flux")
})

test_that("config.json detection beats an unrecognizable filename", {
  d <- tempfile(); dir.create(d)
  on.exit(unlink(d, recursive = TRUE))
  writeLines('{"_class_name": "FluxPipeline"}', file.path(d, "config.json"))
  mp <- file.path(d, "my_model_v3_final.safetensors")
  file.create(mp)
  expect_identical(.detect_model_type_config(mp), "flux")
  expect_identical(.resolve_model_type("auto", mp), "flux")
})

test_that("auto detection errors with a hint when nothing matches", {
  d <- tempfile(); dir.create(d)
  on.exit(unlink(d, recursive = TRUE))
  expect_error(.resolve_model_type("auto", file.path(d, "weird_name.bin")),
               "Cannot detect model_type")
  expect_error(.resolve_model_type("auto", ""), "needs a model path")
})
