# Meta backend PoC — "второй путь" вычислений для multi-GPU (N устройств).
#
# Цель: научиться использовать ggml meta backend (НЕ меняя ggmlR) как
# альтернативный путь генерации, масштабируемый на произвольное число GPU
# (2, 8, ... 100). Здесь имитируем изолированный encoder-стек (несколько
# Linear + GELU — как T5/LLM-энкодер) и гоняем его на meta-backend.
#
# Что проверяет:
#   1. meta-backend строится над N устройствами и считает многослойный граф;
#   2. split_fn спроектирован под N устройств (политика + корректный расчёт
#      ne-сегментов для split по оси, включая неровное деление);
#   3. результат meta == однобэкендный эталон (CPU).
#
# Реальный tensor-split веса требует 2+ физических GPU; на машине с 1 GPU/CPU
# исполняется MIRRORED-ветка (см. meta_backend_smoke.R). Логика split_fn здесь
# написана и юнит-проверена отдельно (compute_split_ne), чтобы перенос в C++
# (TODO п.15, второй путь в GGMLRunner) был механическим.
#
# Запуск:  Rscript inst/examples/meta_backend_poc_encoder.R
#          SD2R_META_DEVS=8 Rscript inst/examples/meta_backend_poc_encoder.R   # имитация 8 устройств

library(ggmlR)
cat("=== meta backend PoC: encoder-стек на N устройствах ===\n")

MIRRORED <- 10L

# --- split_fn: политика размещения тензоров на N устройствах ----------------
# Контракт: (tensor_info, n_devs) -> list(axis, ne, n_segments).
#   tensor_info = list(name, type, ne[4], op, flags).
# Расчёт ne для split по оскольку оси dim_size между n_devs (равномерно, остаток
# раскидываем на первые устройства — корректно для любого N, в т.ч. неровного).
compute_split_ne <- function(dim_size, n_devs) {
  base <- dim_size %/% n_devs
  rem  <- dim_size %%  n_devs
  per  <- rep(base, n_devs) + c(rep(1L, rem), rep(0L, n_devs - rem))
  stopifnot(sum(per) == dim_size)
  per
}

# Политика энкодера: большие Linear-веса (2D, ось-1 = выходная размерность)
# режем по оси 1 между устройствами (tensor parallelism столбцов); всё
# остальное (bias, нормы, мелкие тензоры, входы) — MIRRORED.
SPLIT_MIN <- 32L   # минимальный размер оси, чтобы вообще резать
make_split_fn <- function() {
  function(tensor_info, n_devs) {
    ne   <- as.numeric(tensor_info$ne)
    is2d <- ne[3] == 1 && ne[4] == 1
    out_dim <- ne[2]
    if (is2d && out_dim >= max(SPLIT_MIN, n_devs)) {
      return(list(axis = 1L,
                  ne = compute_split_ne(out_dim, n_devs),
                  n_segments = 1L))
    }
    list(axis = MIRRORED, ne = rep(0, n_devs), n_segments = 1L)
  }
}

# Безопасный split_fn для исполнения на degenerate-устройствах (CPU/1 GPU),
# где split-matmul упирается в ассерт backend: всё MIRRORED.
mirrored_split_fn <- function(tensor_info, n_devs) {
  list(axis = MIRRORED, ne = rep(0, n_devs), n_segments = 1L)
}

# --- Юнит-проверка split-логики (без GPU) -----------------------------------
cat("\n-- split_fn ne-расчёт --\n")
for (case in list(c(64,2), c(64,8), c(100,8), c(7,3), c(16,100))) {
  d <- case[1]; n <- case[2]
  per <- compute_split_ne(d, n)
  cat(sprintf("  dim=%d n_devs=%d -> ne=[%s] sum=%d %s\n",
              d, n, paste(per, collapse=","), sum(per),
              if (sum(per) == d) "OK" else "FAIL"))
}
sf <- make_split_fn()
w_big   <- sf(list(ne = c(512, 1024, 1, 1)), 4L)   # большой Linear -> split
w_small <- sf(list(ne = c(8, 16, 1, 1)),     4L)   # мелкий -> MIRRORED
bias    <- sf(list(ne = c(1024, 1, 1, 1)),   4L)   # 1D -> MIRRORED
cat(sprintf("  big Linear [512,1024]: axis=%d (1=split)  | small: axis=%d (10=mirror)  | bias: axis=%d\n",
            w_big$axis, w_small$axis, bias$axis))
stopifnot(w_big$axis == 1L, w_small$axis == MIRRORED, bias$axis == MIRRORED)

# --- Устройства -------------------------------------------------------------
n_devs <- as.integer(Sys.getenv("SD2R_META_DEVS", "2"))
n_vk   <- tryCatch(ggml_vulkan_device_count(), error = function(e) 0L)
cat(sprintf("\nVulkan devices: %d ; запрошено meta-устройств: %d\n", n_vk, n_devs))

real_multi_gpu <- n_vk >= n_devs && n_devs >= 2L
if (real_multi_gpu) {
  devs <- lapply(seq_len(n_devs) - 1L, function(i) ggml_backend_dev_get(as.integer(i)))
  active_split_fn <- make_split_fn()      # реальный split на N GPU
  mode <- sprintf("%dx Vulkan GPU (tensor split)", n_devs)
} else {
  cpu  <- ggml_backend_dev_by_type(ggml_backend_device_type_cpu())
  devs <- rep(list(cpu), n_devs)
  active_split_fn <- mirrored_split_fn    # degenerate: MIRRORED
  mode <- sprintf("%dx CPU (degenerate, MIRRORED)", n_devs)
}
cat(sprintf("Режим: %s\n", mode))

# --- Encoder-стек: y = gelu(W3 · gelu(W2 · gelu(W1 · x + b1) + b2) + b3) -----
D_IN <- 32L; D_HID <- 64L; D_OUT <- 48L; SEQ <- 4L
set.seed(7)
mk <- function(r, c) matrix(rnorm(r * c, sd = 0.1), nrow = r, ncol = c)
W1 <- mk(D_HID, D_IN); b1 <- rnorm(D_HID, sd = 0.1)
W2 <- mk(D_HID, D_HID); b2 <- rnorm(D_HID, sd = 0.1)
W3 <- mk(D_OUT, D_HID); b3 <- rnorm(D_OUT, sd = 0.1)
X  <- mk(D_IN, SEQ)      # вход [D_IN, SEQ]

build_and_run <- function(backend) {
  ctx <- ggml_init(128L * 1024 * 1024); ggml_set_no_alloc(ctx, TRUE)
  # ggml mul_mat: A[K,M] * B[K,N] -> [M,N]. Linear y=W·x: W[D_in,D_out] (ne K=D_in,M=D_out)
  w1 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_IN,  D_HID)
  w2 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_HID, D_HID)
  w3 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_HID, D_OUT)
  bb1 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D_HID)
  bb2 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D_HID)
  bb3 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D_OUT)
  x  <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_IN, SEQ)

  h1 <- ggml_gelu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w1, x),  bb1))
  h2 <- ggml_gelu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w2, h1), bb2))
  y  <- ggml_gelu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w3, h2), bb3))

  ggml_backend_alloc_ctx_tensors(ctx, backend)
  ggml_backend_tensor_set_data(w1, as.vector(t(W1)))   # row-major [D_IN,D_HID]
  ggml_backend_tensor_set_data(w2, as.vector(t(W2)))
  ggml_backend_tensor_set_data(w3, as.vector(t(W3)))
  ggml_backend_tensor_set_data(bb1, b1)
  ggml_backend_tensor_set_data(bb2, b2)
  ggml_backend_tensor_set_data(bb3, b3)
  ggml_backend_tensor_set_data(x,  as.vector(X))
  gf <- ggml_build_forward_expand(ctx, y)
  ggml_backend_graph_compute(backend, gf)
  res <- ggml_backend_tensor_get_data(y)
  ggml_free(ctx)
  res
}

# Эталон: одиночный CPU backend
cpu_backend <- ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(cpu_backend, 2L)
ref <- build_and_run(cpu_backend)
ggml_backend_free(cpu_backend)

# Второй путь: meta backend над N устройствами
meta_dev <- ggml_backend_meta_device(devs, active_split_fn)
stopifnot(!is.null(meta_dev))
meta_backend <- ggml_backend_dev_init(meta_dev)
stopifnot(!is.null(meta_backend))
got <- build_and_run(meta_backend)
ggml_backend_free(meta_backend)

cc <- suppressWarnings(stats::cor(ref, got))
md <- max(abs(ref - got))
cat(sprintf("\nencoder out: ref=%d meta=%d | cor=%.6f max|diff|=%.6g\n",
            length(ref), length(got), cc, md))
ok <- length(ref) == length(got) && is.finite(cc) && cc > 0.999 && md < 1e-3
cat(if (ok) "RESULT: PASS — encoder через meta backend == CPU-эталон\n"
    else    "RESULT: FAIL — расхождение meta vs CPU\n")
if (!ok) quit(status = 1L)
