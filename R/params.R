# ===========================================================================
# Generation parameters: unified defaults (sd_params)
# ===========================================================================
# Analogue of IRIS_PARAMS_DEFAULT (iris.c): a single editable list holding all
# generation defaults. Returned by sd_default_params(), consumed by
# sd_generate(ctx, prompt, params = ...). Explicit arguments to sd_generate()
# always override the corresponding field here.
#
# This is the R-layer "sd_params" of TODO 9.3. It deliberately does NOT touch
# the C struct sd_ctx_params_t / sd_img_gen_params_t — those keep their own
# init functions on the C++ side. This list covers per-generation knobs only
# (not context-construction knobs, which live in sd_ctx()).

#' Default generation parameters
#'
#' Returns a named list of all per-generation defaults used by
#' \code{\link{sd_generate}}. Edit the returned list and pass it back via the
#' \code{params} argument to set a reusable baseline; any explicit argument to
#' \code{sd_generate()} overrides the matching field.
#'
#' This is the R-level analogue of \code{IRIS_PARAMS_DEFAULT}. It covers
#' generation knobs only; context-construction options (model paths, devices,
#' offload, etc.) belong to \code{\link{sd_ctx}}.
#'
#' @return A named list with fields: \code{negative_prompt}, \code{width},
#'   \code{height}, \code{strength}, \code{sample_method}, \code{sample_steps},
#'   \code{cfg_scale}, \code{seed}, \code{batch_count}, \code{scheduler},
#'   \code{clip_skip}, \code{eta}, \code{hr_strength}, \code{vae_mode},
#'   \code{vae_tile_size}, \code{vae_tile_overlap}, \code{cache_mode},
#'   \code{cache_config}.
#' @export
#' @seealso \code{\link{sd_generate}}
#' @examples
#' p <- sd_default_params()
#' p$sample_steps <- 30
#' p$cfg_scale <- 4.0
#' \dontrun{
#' ctx <- sd_ctx("model.safetensors", model_type = "auto")
#' imgs <- sd_generate(ctx, "a cat", params = p)
#' }
sd_default_params <- function() {
  list(
    negative_prompt  = "",
    width            = 512L,
    height           = 512L,
    strength         = 0.75,
    sample_method    = SAMPLE_METHOD$EULER,
    sample_steps     = 20L,
    cfg_scale        = 7.0,
    seed             = 42L,
    batch_count      = 1L,
    scheduler        = SCHEDULER$DISCRETE,
    clip_skip        = -1L,
    eta              = 0.0,
    hr_strength      = 0.4,
    vae_mode         = "auto",
    vae_tile_size    = 64L,
    vae_tile_overlap = 0.25,
    cache_mode       = "off",
    cache_config     = NULL
  )
}

# Merge an explicit caller value over the params baseline.
# `explicit` is TRUE when the caller passed the argument by name (detected via
# match.call() in sd_generate); in that case the explicit value wins.
# Otherwise the params list provides the value, falling back to the built-in
# default when the supplied params list omits the field.
.param_value <- function(name, explicit, explicit_value, params, defaults) {
  if (explicit) return(explicit_value)
  if (!is.null(params) && name %in% names(params)) return(params[[name]])
  defaults[[name]]
}
