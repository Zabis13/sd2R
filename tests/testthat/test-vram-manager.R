# Proactive VRAM management in the model manager. These tests exercise pure R
# logic (size estimation + LRU eviction) without loading any real model, so they
# run everywhere — including on CPU-only / no-Vulkan machines.

# Temporarily replace the free-VRAM probe in the sd2R namespace with a stub that
# returns `value`, restoring the original on exit. Works both under a real
# installation and under pkgload::load_all().
local_free_vram <- function(value, env = parent.frame()) {
  ns   <- asNamespace("sd2R")
  orig <- get(".mm_free_vram_bytes", envir = ns)
  stub <- function(device = NULL) value
  unlockBinding(".mm_free_vram_bytes", ns)
  assign(".mm_free_vram_bytes", stub, envir = ns)
  withr::defer({
    assign(".mm_free_vram_bytes", orig, envir = ns)
    lockBinding(".mm_free_vram_bytes", ns)
  }, envir = env)
}

test_that(".mm_model_size_bytes sums on-disk weight files", {
  f1 <- tempfile(fileext = ".safetensors")
  f2 <- tempfile(fileext = ".safetensors")
  writeBin(raw(4 * 1024^2), f1)   # 4 MB
  writeBin(raw(6 * 1024^2), f2)   # 6 MB
  on.exit(unlink(c(f1, f2)), add = TRUE)

  entry <- list(id = "x", paths = list(diffusion = f1, vae = f2))
  sz <- sd2R:::.mm_model_size_bytes(entry)
  expect_equal(sz, 10 * 1024^2)
})

test_that(".mm_model_size_bytes ignores missing files and returns 0 for none", {
  entry_missing <- list(id = "x", paths = list(diffusion = "/no/such/file.bin"))
  expect_equal(sd2R:::.mm_model_size_bytes(entry_missing), 0)

  entry_empty <- list(id = "x", paths = list())
  expect_equal(sd2R:::.mm_model_size_bytes(entry_empty), 0)
})

test_that(".mm_ensure_vram skips when there is no usable size estimate", {
  entry <- list(id = "x", paths = list())
  expect_true(sd2R:::.mm_ensure_vram(entry, verbose = FALSE))
})

test_that(".mm_ensure_vram evicts LRU then reports failure when still short", {
  # Force a deterministic, tiny free-VRAM value and a real size estimate so the
  # eviction path is taken regardless of the host's actual GPU.
  local_free_vram(0.1 * 1e9)  # 0.1 GB free

  big <- tempfile(fileext = ".safetensors")
  # 2 GB sparse file → need = 2 * 1.2 + 0.5 ≈ 2.9 GB >> 0.1 GB free
  con <- file(big, "wb"); seek(con, 2 * 1024^3 - 1); writeBin(as.raw(0), con); close(con)
  on.exit(unlink(big), add = TRUE)
  entry <- list(id = "big", model_type = "flux", paths = list(diffusion = big))

  # Seed one loaded model to be evicted.
  mm_env <- get(".mm_env", envir = asNamespace("sd2R"))
  mm_env$contexts[["old"]]  <- "FAKE_CTX"
  mm_env$last_used[["old"]] <- Sys.time() - 100
  on.exit({
    mm_env$contexts[["old"]]  <- NULL
    mm_env$last_used[["old"]] <- NULL
  }, add = TRUE)

  res <- suppressMessages(sd2R:::.mm_ensure_vram(entry, verbose = FALSE))

  expect_false(res)                                  # could not free enough
  expect_false("old" %in% names(mm_env$contexts))    # LRU was evicted
})

test_that(".mm_ensure_vram succeeds when free VRAM is ample", {
  local_free_vram(100 * 1e9)  # 100 GB free
  small <- tempfile(fileext = ".safetensors")
  writeBin(raw(8 * 1024^2), small)                         # 8 MB
  on.exit(unlink(small), add = TRUE)
  entry <- list(id = "small", paths = list(diffusion = small))

  expect_true(sd2R:::.mm_ensure_vram(entry, verbose = FALSE))
})

test_that(".mm_ensure_vram does not block when VRAM is unknown", {
  local_free_vram(NULL)   # cannot query (CPU / no Vulkan)
  f <- tempfile(fileext = ".safetensors")
  writeBin(raw(2 * 1024^2), f)
  on.exit(unlink(f), add = TRUE)
  entry <- list(id = "x", paths = list(diffusion = f))

  expect_true(sd2R:::.mm_ensure_vram(entry, verbose = FALSE))
})
