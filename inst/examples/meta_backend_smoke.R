# Meta backend smoke-test — проверка, что ggml meta backend жив через линковку sd2R.
#
# sd2R все вычисления делает через ggmlR (статическая линковка libggml.a). Meta
# backend (ggml 0.11.0) — обёртка над N физическими устройствами: для каждого
# тензора split_fn решает MIRRORED (копия на всех) или split по оси между GPU.
# Это инфраструктура для multi-GPU sharding ОДНОЙ модели (TODO п.15).
#
# Здесь НЕ трогаем diffusion-путь sd2R. Это чистый R-скрипт через ggmlR::,
# доказывающий, что API доступен и считает правильно: matmul на meta-backend
# (2 устройства, MIRRORED) == CPU-эталон.
#
# Почему MIRRORED, а не split: реальный tensor-split веса matmul требует 2
# физических GPU + поддержки в backend; в degenerate CPU-режиме (или 1 GPU)
# split-matmul упирается в ассерт ne0==ne01. MIRRORED же гоняет полный
# multi-buffer путь meta-backend и проверяется на любой машине. Это и есть
# "минимальное использование, что работает". Реальный split — следующий шаг
# (PoC на encoder, meta_backend_poc_encoder.R), требует 2 GPU.
#
# Запуск:  Rscript inst/examples/meta_backend_smoke.R

library(ggmlR)

cat("=== ggml meta backend smoke-test ===\n")

# --- Выбор устройств -------------------------------------------------------
# Предпочитаем 2 Vulkan GPU. Если их нет — degenerate-режим на CPU (как в
# эталонном тесте ggmlR): два ref на одно устройство всё равно гоняют
# multi-buffer путь meta-backend.
n_vk <- tryCatch(ggml_vulkan_device_count(), error = function(e) 0L)
cat(sprintf("Vulkan devices: %d\n", n_vk))

if (n_vk >= 2L) {
  devs <- list(ggml_backend_dev_get(0L), ggml_backend_dev_get(1L))
  mode <- "2x Vulkan GPU"
} else {
  cpu_dev <- ggml_backend_dev_by_type(ggml_backend_device_type_cpu())
  stopifnot(!is.null(cpu_dev))
  devs <- list(cpu_dev, cpu_dev)         # degenerate: 2 ref на CPU
  mode <- "2x CPU (degenerate, no 2nd GPU)"
}
cat(sprintf("Meta devices: %s\n", mode))

# --- Размеры задачи: out = A * B -------------------------------------------
# A: вес [K, M] (ggml ne = c(K, M, 1, 1)); B: активации [K, N]; out = [M, N].
M <- 64L; K <- 32L; N <- 8L

# --- split_fn --------------------------------------------------------------
# Контракт: (tensor_info, n_devs) -> list(axis, ne, n_segments).
#   tensor_info = list(name, type, ne[4], op, flags)
#   axis: 0..3 = split по оси; 10 = MIRRORED; ne длины n_segments*n_devs.
# Минимальная политика: всё MIRRORED (копия каждого тензора на всех устройствах).
MIRRORED <- 10L
split_fn <- function(tensor_info, n_devs) {
  list(axis = MIRRORED, ne = rep(0, n_devs), n_segments = 1L)
}

# --- Эталон на CPU ---------------------------------------------------------
set.seed(42)
a_raw <- matrix(rnorm(K * M, sd = 0.1), nrow = M, ncol = K)  # [M, K]
b_raw <- matrix(rnorm(K * N, sd = 0.1), nrow = K, ncol = N)  # [K, N]

run_cpu <- function() {
  ctx <- ggml_init(64L * 1024 * 1024); ggml_set_no_alloc(ctx, TRUE)
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N)
  out <- ggml_mul_mat(ctx, a, b)
  backend <- ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(backend, 2L)
  ggml_backend_alloc_ctx_tensors(ctx, backend)
  ggml_backend_tensor_set_data(a, as.vector(t(a_raw)))       # row-major [K,M]
  ggml_backend_tensor_set_data(b, as.vector(b_raw))
  gf <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(backend, gf)
  res <- ggml_backend_tensor_get_data(out)
  ggml_free(ctx); ggml_backend_free(backend)
  res
}

# --- Вычисление через meta backend -----------------------------------------
run_meta <- function() {
  meta_dev <- ggml_backend_meta_device(devs, split_fn)
  stopifnot(!is.null(meta_dev))
  backend <- ggml_backend_dev_init(meta_dev)
  stopifnot(!is.null(backend))

  ctx <- ggml_init(64L * 1024 * 1024); ggml_set_no_alloc(ctx, TRUE)
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N)
  out <- ggml_mul_mat(ctx, a, b)
  ggml_backend_alloc_ctx_tensors(ctx, backend)
  ggml_backend_tensor_set_data(a, as.vector(t(a_raw)))
  ggml_backend_tensor_set_data(b, as.vector(b_raw))
  gf <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(backend, gf)
  res <- ggml_backend_tensor_get_data(out)
  ggml_free(ctx); ggml_backend_free(backend)
  res
}

cpu  <- run_cpu()
meta <- run_meta()

cc <- suppressWarnings(stats::cor(cpu, meta))
md <- max(abs(cpu - meta))
cat(sprintf("out length: cpu=%d meta=%d\n", length(cpu), length(meta)))
cat(sprintf("cor(cpu, meta) = %.6f   max|diff| = %.6g\n", cc, md))

ok <- length(cpu) == length(meta) && is.finite(cc) && cc > 0.999 && md < 1e-3
cat(if (ok) "RESULT: PASS — meta backend matmul == CPU\n"
    else    "RESULT: FAIL — meta backend output diverges from CPU\n")
if (!ok) quit(status = 1L)
