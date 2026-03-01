# sdR package initialization and constants

#' @importFrom Rcpp sourceCpp
NULL

# ============================================================================
# Constants — match C++ enums (0-based)
# ============================================================================

#' RNG types
#' @export
RNG_TYPE <- list(
  STD_DEFAULT = 0L,
  CUDA        = 1L,
  CPU         = 2L
)

#' Sampling methods
#' @export
SAMPLE_METHOD <- list(
  EULER       = 0L,
  EULER_A     = 1L,
  HEUN        = 2L,
  DPM2        = 3L,
  DPMPP_2S_A  = 4L,
  DPMPP_2M    = 5L,
  DPMPP_2M_V2 = 6L,
  IPNDM       = 7L,
  IPNDM_V     = 8L,
  LCM         = 9L,
  DDIM        = 10L,
  TCD         = 11L
)

#' Schedulers
#' @export
SCHEDULER <- list(
  DISCRETE    = 0L,
  KARRAS      = 1L,
  EXPONENTIAL = 2L,
  AYS         = 3L,
  GITS        = 4L,
  SGM_UNIFORM = 5L,
  SIMPLE      = 6L,
  SMOOTHSTEP  = 7L,
  KL_OPTIMAL  = 8L,
  LCM         = 9L
)

#' Prediction types
#' @export
PREDICTION <- list(
  EPS             = 0L,
  V_PRED          = 1L,
  EDM_V_PRED      = 2L,
  FLOW_PRED       = 3L,
  FLUX_FLOW_PRED  = 4L,
  FLUX2_FLOW_PRED = 5L
)

#' Weight types (ggml quantization types)
#' @export
SD_TYPE <- list(
  F32     = 0L,
  F16     = 1L,
  Q4_0    = 2L,
  Q4_1    = 3L,
  Q5_0    = 6L,
  Q5_1    = 7L,
  Q8_0    = 8L,
  Q2_K    = 10L,
  Q3_K    = 11L,
  Q4_K    = 12L,
  Q5_K    = 13L,
  Q6_K    = 14L,
  Q8_K    = 15L,
  BF16    = 30L,
  COUNT   = 40L
)

#' LoRA apply modes
#' @export
LORA_APPLY_MODE <- list(
  AUTO        = 0L,
  IMMEDIATELY = 1L,
  AT_RUNTIME  = 2L
)

#' Cache modes
#' @export
SD_CACHE_MODE <- list(
  DISABLED    = 0L,
  EASYCACHE   = 1L,
  UCACHE      = 2L,
  DBCACHE     = 3L,
  TAYLORSEER  = 4L,
  CACHE_DIT   = 5L
)

# ============================================================================
# .onLoad — initialize C++ log callback
# ============================================================================

.onLoad <- function(libname, pkgname) {
  sd_init_log()
}
