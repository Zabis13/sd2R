# test_fa_probe.R
#
# Standalone flash-attention support probe.
#
# Purpose: detect, on the actual installed Vulkan backend, whether the
# flash-attention (FLASH_ATTN_EXT) path that sd2R requests at runtime is
# really accepted by ggmlR -- or whether it silently falls back to the slow
# manual F32 attention path.
#
# This does NOT simulate a GPU. It asks the real ggmlR backend the exact
# question ggml_ext_attention_ext() / build_kqv() asks per attention layer:
#     ggml_backend_dev_supports_op(dev, <FLASH_ATTN_EXT node>)
# so the answer reflects this machine's driver/device.
#
# The tensor types must MIRROR build_kqv() in src/sd/ggml_extend.hpp:
#     Q -> F32   (the ggmlR Vulkan FA kernel asserts q->type == F32)
#     K -> F16
#     V -> F16
# A mismatch here is exactly the bug class this script is meant to catch:
# if sd2R feeds Q as F16 (as it did before the q_in cast was added),
# supports_op returns FALSE and every attention layer takes the slow path.
#
# Usage:
#   Rscript -e 'source(system.file("examples/test_fa_probe.R", package="sd2R"))'
# or from a source checkout:
#   Rscript inst/examples/test_fa_probe.R

suppressMessages({
  library(sd2R)
  library(ggmlR)
})

# ---- one probe: build a real FLASH_ATTN_EXT node and ask the backend --------
probe_fa <- function(dev, head_dim, q_type, k_type, v_type,
                     n_head = 8L, seq_len = 256L) {
  ctx <- ggmlR::ggml_init(16L * 1024L * 1024L, no_alloc = TRUE)
  on.exit(ggmlR::ggml_free(ctx), add = TRUE)
  q <- ggmlR::ggml_new_tensor_4d(ctx, q_type, head_dim, n_head, seq_len, 1L)
  k <- ggmlR::ggml_new_tensor_4d(ctx, k_type, head_dim, n_head, seq_len, 1L)
  v <- ggmlR::ggml_new_tensor_4d(ctx, v_type, head_dim, n_head, seq_len, 1L)
  fa <- ggmlR::ggml_flash_attn_ext(ctx, q, k, v, NULL, 1.0 / sqrt(head_dim), 0.0, 0.0)
  isTRUE(ggmlR::ggml_backend_dev_supports_op(dev, fa))
}

# ---- locate a Vulkan device by registry, mirroring app.R --------------------
find_dev <- function(want_desc = NULL) {
  ndev <- ggmlR::ggml_backend_dev_count()
  for (d in seq_len(ndev) - 1L) {
    dd <- ggmlR::ggml_backend_dev_get(d)
    desc <- ggmlR::ggml_backend_dev_description(dd)
    if (is.null(want_desc) || identical(desc, want_desc)) {
      return(list(dev = dd, desc = desc))
    }
  }
  NULL
}

main <- function() {
  n_gpu <- sd_vulkan_device_count()
  if (n_gpu < 1) {
    cat("No Vulkan device found - flash-attention probe skipped.\n")
    return(invisible(NULL))
  }

  ndev <- ggmlR::ggml_backend_dev_count()
  cat(sprintf("Backend devices: %d\n\n", ndev))

  # The two head dims that occur in diffusion models, and the type combos that
  # matter. The build_kqv combo (Q=F32,K/V=F16) is the one sd2R actually uses;
  # the Q=F16 row is the historical mistake, kept as a self-test so the script
  # demonstrates the failing case explicitly.
  combos <- list(
    list(label = "build_kqv (Q=F32, K/V=F16)", q = ggmlR::GGML_TYPE_F32, k = ggmlR::GGML_TYPE_F16, v = ggmlR::GGML_TYPE_F16),
    list(label = "BAD     (Q=F16, K/V=F16)",   q = ggmlR::GGML_TYPE_F16, k = ggmlR::GGML_TYPE_F16, v = ggmlR::GGML_TYPE_F16)
  )

  for (d in seq_len(ndev) - 1L) {
    dd   <- ggmlR::ggml_backend_dev_get(d)
    desc <- ggmlR::ggml_backend_dev_description(dd)
    cat(sprintf("Device [%d]: %s\n", d, desc))
    for (hd in c(64L, 128L)) {
      for (cb in combos) {
        ok <- tryCatch(
          probe_fa(dd, hd, cb$q, cb$k, cb$v),
          error = function(e) { cat(sprintf("    probe error: %s\n", conditionMessage(e))); NA }
        )
        cat(sprintf("  head_dim=%-3d %-28s : %s\n",
                    hd, cb$label,
                    if (isTRUE(ok)) "SUPPORTED" else "NOT SUPPORTED (fallback)"))
      }
    }
    cat("\n")
  }

  cat("Interpretation:\n")
  cat("  - build_kqv row SUPPORTED  -> sd2R uses the fast flash-attention path.\n")
  cat("  - build_kqv row NOT SUPP.  -> every attention layer falls back to slow F32;\n")
  cat("                                check that build_kqv casts Q to F32.\n")
  cat("  - BAD row should be NOT SUPPORTED (it proves Q=F16 is rejected).\n")
  invisible(NULL)
}

main()
