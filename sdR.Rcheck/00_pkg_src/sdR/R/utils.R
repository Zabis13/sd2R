# Utility functions for sdR

#' Get system information
#'
#' Returns information about the stable-diffusion.cpp backend.
#'
#' @return List with system info, version, and core count
#' @export
sd_system_info <- function() {
  info <- list(
    sdR_version = as.character(utils::packageVersion("sdR")),
    sd_cpp_version = sd_version_cpp(),
    system_info = sd_system_info_cpp(),
    num_cores = sd_num_physical_cores_cpp(),
    vulkan_available = ggmlR::ggml_vulkan_available()
  )
  class(info) <- "sd_system_info"
  info
}

#' @export
print.sd_system_info <- function(x, ...) {
  cat("sdR System Information\n")
  cat("  sdR version:    ", x$sdR_version, "\n")
  cat("  sd.cpp version: ", x$sd_cpp_version, "\n")
  cat("  Physical cores: ", x$num_cores, "\n")
  cat("  Backend info:   ", x$system_info, "\n")
  cat("  Vulkan GPU:     ", if (x$vulkan_available) "available" else "not available", "\n")
  invisible(x)
}
