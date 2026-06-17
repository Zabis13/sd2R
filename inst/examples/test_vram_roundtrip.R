# =====================================================================
# Тест B: VRAM roundtrip на чистом ggmlR.
# Заливаем известные данные в VRAM через Vulkan-backend, читаем обратно,
# сравниваем. Цель — проверить, исправен ли сам канал заливки/чтения VRAM
# на этой GPU, БЕЗ модели.
#
# Запуск:
#   Rscript inst/examples/test_vram_roundtrip.R [device]
# device — индекс Vulkan-устройства (по умолчанию 0).
# =====================================================================

suppressMessages(library(ggmlR))

args   <- commandArgs(trailingOnly = TRUE)
device <- if (length(args) >= 1) as.integer(args[1]) else 0L

cat(sprintf("Vulkan available: %s\n", ggml_vulkan_available()))
cat(sprintf("Vulkan device count: %d\n", ggml_vulkan_device_count()))
cat(sprintf("Using device %d: %s\n", device,
            tryCatch(ggml_vulkan_device_description(device), error = function(e) "?")))

# --- backend ---
backend <- ggml_vulkan_init(device)
stopifnot(!is.null(backend))

# Размеры для проверки: мелкий, средний, и ~как f16-вес энкодера (4718592 B / 4 = 1179648 f32)
sizes <- c(16L, 4096L, 1179648L)

# Несколько паттернов: ramp, константа, "знаковый" с отрицательными и дробными
make_pattern <- function(n, kind) {
  switch(kind,
    ramp  = as.double(seq_len(n) - 1),                 # 0,1,2,...
    const = rep(3.14159265, n),
    mixed = as.double((seq_len(n) - 1) * ifelse(seq_len(n) %% 2 == 0, -1, 1)) / 7
  )
}

run_one <- function(n, kind) {
  ctx <- ggml_init(mem_size = 16 * 1024 * 1024, no_alloc = TRUE)   # no_alloc: данные в backend-буфере, тут только метаданные
  on.exit(ggml_free(ctx), add = TRUE)

  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)         # аллокация в VRAM

  src <- make_pattern(n, kind)
  ggml_backend_tensor_set_data(t, src)                        # RAM -> VRAM
  back <- ggml_backend_tensor_get_data(t)                     # VRAM -> RAM

  n_read <- length(back)
  ok_len <- (n_read == n)
  # относительная погрешность: повреждение даёт огромные расхождения/NaN,
  # а не уровень точности f32 (~1e-7 относительно величины).
  rel    <- if (ok_len) abs(back - src) / pmax(abs(src), 1e-30) else NA_real_
  diff   <- if (ok_len) abs(back - src) else NA_real_
  max_d  <- if (ok_len) max(diff) else NA_real_
  n_bad  <- if (ok_len) sum(rel > 1e-3) else NA_integer_   # >0.1% = реальная порча
  n_nan  <- sum(is.nan(back) | is.infinite(back))

  cat(sprintf("[n=%-8d %-5s] read=%d len_ok=%s max_abs_diff=%.3g bad(>1e-6)=%s NaN/Inf=%d  %s\n",
              n, kind, n_read, ok_len, max_d, as.character(n_bad), n_nan,
              if (isTRUE(ok_len && n_bad == 0 && n_nan == 0)) "OK" else "*** MISMATCH ***"))

  # при расхождении — покажем первые несовпадающие индексы
  if (isTRUE(ok_len) && !is.na(n_bad) && n_bad > 0) {
    bad_idx <- head(which(diff > 1e-6), 8)
    for (i in bad_idx) {
      cat(sprintf("    idx %d: wrote %.6g  read %.6g\n", i, src[i], back[i]))
    }
  }
  invisible(NULL)
}

cat("\n--- F32 roundtrip ---\n")
for (n in sizes) for (k in c("ramp", "const", "mixed")) run_one(n, k)

ggml_vulkan_free(backend)
cat("\nDONE\n")
