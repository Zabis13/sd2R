# Model manager for sd2R
#
# Registry stored in tools::R_user_dir("sd2R", "config")/models.json
# (overridable via the SD2R_REGISTRY_DIR environment variable).
# Context cache in memory (package-level environment)

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
.mm_env <- new.env(parent = emptyenv())
.mm_env$contexts <- list()       # named list: id -> sd_ctx
.mm_env$last_used <- list()      # named list: id -> timestamp (for LRU)

# Resolve the directory used to store the model registry.
#
# Per CRAN Policy, packages must not write to the user's home directory or
# any other location outside tempdir() without explicit user consent. We
# therefore default to the standard tools::R_user_dir("sd2R", "config")
# location and allow overriding it via the SD2R_REGISTRY_DIR env-var (used
# by the test suite to redirect writes into tempdir()).
#
# This function never creates the directory on its own; creation happens
# only in .write_registry(), i.e. when the user explicitly registers a
# model.
.mm_registry_dir <- function() {
  override <- Sys.getenv("SD2R_REGISTRY_DIR", unset = "")
  if (nzchar(override)) return(override)
  tools::R_user_dir("sd2R", which = "config")
}

.mm_registry_path <- function() {
  file.path(.mm_registry_dir(), "models.json")
}

# ---------------------------------------------------------------------------
# Registry I/O
# ---------------------------------------------------------------------------

.read_registry <- function() {
  path <- .mm_registry_path()
  if (!file.exists(path)) return(list())
  tryCatch(
    jsonlite::fromJSON(path, simplifyVector = FALSE),
    error = function(e) {
      warning("Failed to read model registry: ", conditionMessage(e), call. = FALSE)
      list()
    }
  )
}

.write_registry <- function(registry) {
  dir <- .mm_registry_dir()
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  }
  path <- file.path(dir, "models.json")
  jsonlite::write_json(registry, path, pretty = TRUE, auto_unbox = TRUE)
  invisible(path)
}

# ---------------------------------------------------------------------------
# Model type detection from filename
# ---------------------------------------------------------------------------

#' Guess model type from filename
#' @param filename File basename
#' @return Character: "flux2", "flux", "sdxl", "sd1", "sd2", "sd3", or "unknown"
#' @keywords internal
.guess_model_type <- function(filename) {
  fn <- tolower(filename)
  if (grepl("flux[._-]?2|flux2", fn)) return("flux2")
  if (grepl("flux", fn)) return("flux")
  if (grepl("sdxl|_xl", fn)) return("sdxl")
  if (grepl("sd3", fn)) return("sd3")
  if (grepl("sd2|v2", fn)) return("sd2")
  if (grepl("sd1|v1|sd-v1|v1-5|v1_5", fn)) return("sd1")
  "unknown"
}

#' Detect model type from a GGUF file's KV metadata (header-only probe)
#'
#' Reads \code{general.architecture} (and a few related KV keys) from a GGUF
#' header WITHOUT loading tensor weights, via
#' \code{ggmlR::gguf_load(path, meta_only = TRUE)} (no_alloc header read). Cheap
#' even on multi-GB Flux models.
#'
#' Note: stable-diffusion.cpp itself detects the version from tensor
#' \emph{names/shapes}, not from \code{general.architecture}, so many diffusion
#' GGUFs (e.g. quantized Flux converters) leave that field empty or set it to a
#' sub-component name (e.g. \code{"t5"} for a packed text encoder). This probe
#' is therefore best-effort: it returns a concrete type only on a confident
#' match and otherwise \code{NULL}, so the caller falls through to
#' config.json / filename detection.
#'
#' @param path Path to a .gguf file
#' @return Model type string, or \code{NULL} if unavailable / inconclusive
#' @keywords internal
.detect_model_type_gguf <- function(path) {
  if (is.null(path) || !nzchar(path)) return(NULL)
  if (!grepl("\\.gguf$", path, ignore.case = TRUE)) return(NULL)
  if (!file.exists(path)) return(NULL)
  if (!requireNamespace("ggmlR", quietly = TRUE)) return(NULL)

  # Require the metadata-only load (no_alloc). Older ggmlR without the
  # meta_only arg would read the full tensor blob — too expensive for a probe,
  # so we bail rather than fall back to a heavy load.
  if (!"meta_only" %in% names(formals(ggmlR::gguf_load))) return(NULL)

  meta <- tryCatch({
    g <- ggmlR::gguf_load(path, meta_only = TRUE)
    on.exit(try(ggmlR::gguf_free(g), silent = TRUE), add = TRUE)
    ggmlR::gguf_metadata(g)
  }, error = function(e) NULL)
  if (is.null(meta) || !length(meta)) return(NULL)

  # Collect the architecture-ish KV values into one haystack.
  keys <- c("general.architecture", "general.name", "general.type",
            "general.basename")
  hay <- tolower(paste(unlist(meta[intersect(keys, names(meta))]),
                       collapse = " "))
  if (!nzchar(hay)) return(NULL)

  # Confident matches only; order matters (most specific first).
  if (grepl("flux[._-]?2|flux2", hay))           return("flux2")
  if (grepl("flux", hay))                        return("flux")
  if (grepl("sd3|stable[-_ ]?diffusion[-_ ]?3|mmdit", hay)) return("sd3")
  if (grepl("sdxl|[-_ ]xl\\b", hay))             return("sdxl")
  if (grepl("\\bsd2\\b|stable[-_ ]?diffusion[-_ ]?2|v2", hay)) return("sd2")
  if (grepl("\\bsd1\\b|stable[-_ ]?diffusion(?![-_ ]?[23x])|v1", hay,
            perl = TRUE)) return("sd1")
  NULL
}

#' Detect model type from a sibling config.json (diffusers-style layout)
#'
#' Looks for \code{config.json} next to the model file and maps its
#' \code{model_type} / \code{architectures[1]} / \code{_class_name} fields onto
#' sd2R's type vocabulary. Pure R, no weights read.
#' @param path Path to the model file
#' @return Model type string, or \code{NULL} if not detectable
#' @keywords internal
.detect_model_type_config <- function(path) {
  cfg <- file.path(dirname(path), "config.json")
  if (!file.exists(cfg)) return(NULL)
  if (!requireNamespace("jsonlite", quietly = TRUE)) return(NULL)
  j <- tryCatch(jsonlite::fromJSON(cfg), error = function(e) NULL)
  if (is.null(j)) return(NULL)
  hay <- tolower(paste(c(j$model_type, j$architectures, j[["_class_name"]]),
                       collapse = " "))
  if (!nzchar(hay)) return(NULL)
  if (grepl("flux[._-]?2|flux2", hay))    return("flux2")
  if (grepl("flux", hay))                 return("flux")
  if (grepl("sd3|stablediffusion3|mmdit", hay)) return("sd3")
  if (grepl("xl|sdxl", hay))              return("sdxl")
  if (grepl("v2|sd2", hay))               return("sd2")
  if (grepl("v1|sd1|stablediffusion", hay)) return("sd1")
  NULL
}

#' Resolve model_type, including the "auto" detection hierarchy
#'
#' When \code{model_type == "auto"}, tries, in order: GGUF KV metadata
#' (\code{\link{.detect_model_type_gguf}}, header-only probe), a sibling
#' \code{config.json} (\code{\link{.detect_model_type_config}}), then the
#' filename heuristic (\code{\link{.guess_model_type}}). If all fail it errors
#' with a hint to set \code{model_type} explicitly rather than guessing.
#' @param model_type User-supplied model type ("auto" triggers detection)
#' @param path Path used for detection (model_path or diffusion_model_path)
#' @return A concrete model type string
#' @keywords internal
.resolve_model_type <- function(model_type, path) {
  if (!identical(model_type, "auto")) return(model_type)
  if (is.null(path) || !nzchar(path)) {
    stop("model_type = \"auto\" needs a model path to detect from. ",
         "Pass model_path/diffusion_model_path, or set model_type explicitly.",
         call. = FALSE)
  }
  detected <- .detect_model_type_gguf(path)
  if (is.null(detected)) detected <- .detect_model_type_config(path)
  if (is.null(detected)) {
    g <- .guess_model_type(basename(path))
    if (!identical(g, "unknown")) detected <- g
  }
  if (is.null(detected)) {
    stop("Cannot detect model_type from '", basename(path), "'. ",
         "Set it explicitly, e.g. model_type = \"flux\" ",
         "(one of: sd1, sd2, sdxl, flux, flux2, sd3).", call. = FALSE)
  }
  detected
}

#' Guess component role from filename
#' @param filename File basename
#' @return Character: "diffusion", "vae", "clip_l", "clip_g", "t5xxl",
#'   "taesd", or "unknown"
#' @keywords internal
.guess_component <- function(filename) {
  fn <- tolower(filename)
  if (grepl("^ae[._]|[._]ae[._]|vae|decoder", fn)) return("vae")
  if (grepl("clip_l|clip-l|openclip-vit-l", fn)) return("clip_l")
  if (grepl("clip_g|clip-g|openclip-vit-g", fn)) return("clip_g")
  if (grepl("t5|umt5", fn)) return("t5xxl")
  if (grepl("taesd", fn)) return("taesd")
  if (grepl("control", fn)) return("control_net")
  # Diffusion is the fallback for main model files
  if (grepl("flux|sdxl|sd[_-]?v?[123]|unet|dit|mmdit|transformer", fn)) return("diffusion")
  # Large .gguf/.safetensors without other matches -> likely diffusion
  "unknown"
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#' Register a model in the sd2R model registry
#'
#' Adds or updates a model entry in the sd2R model registry file. The
#' registry lives in \code{tools::R_user_dir("sd2R", "config")} by default
#' and can be overridden via the \code{SD2R_REGISTRY_DIR} environment
#' variable. The directory is created only when a model is actually
#' registered. Paths and defaults are stored for later use by
#' \code{\link{sd_load_model}}.
#'
#' @param id Unique model identifier (e.g. "flux-dev", "sd15-base")
#' @param model_type Model architecture: "sd1", "sd2", "sdxl", "flux", "flux2", "sd3"
#' @param paths Named list of file paths. Recognized names:
#'   \code{diffusion}, \code{model} (alias for diffusion), \code{vae},
#'   \code{clip_l}, \code{clip_g}, \code{t5xxl}, \code{taesd},
#'   \code{control_net}.
#' @param defaults Named list of generation defaults (optional). Recognized:
#'   \code{steps}, \code{cfg_scale}, \code{scheduler}, \code{width},
#'   \code{height}, \code{sample_method}.
#' @param overwrite If FALSE (default), error when id already exists
#' @return Invisible model id
#' @export
#' @examples
#' \dontrun{
#' sd_register_model(
#'   id = "flux-dev",
#'   model_type = "flux",
#'   paths = list(
#'     diffusion = "models/flux1-dev-Q4_K_S.gguf",
#'     vae = "models/ae.safetensors",
#'     clip_l = "models/clip_l.safetensors",
#'     t5xxl = "models/t5xxl_fp16.safetensors"
#'   ),
#'   defaults = list(steps = 25, cfg_scale = 3.5, width = 1024, height = 1024)
#' )
#' }
sd_register_model <- function(id, model_type, paths, defaults = list(),
                               overwrite = FALSE) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required. Install with install.packages('jsonlite')",
         call. = FALSE)
  }
  model_type <- match.arg(model_type, c("sd1", "sd2", "sdxl", "flux", "flux2", "sd3", "unknown"))

  registry <- .read_registry()
  if (!overwrite && id %in% vapply(registry, `[[`, character(1), "id")) {
    stop("Model '", id, "' already registered. Use overwrite = TRUE to replace.",
         call. = FALSE)
  }

  # Normalize 'model' alias to 'diffusion'
  if ("model" %in% names(paths) && !"diffusion" %in% names(paths)) {
    paths$diffusion <- paths$model
    paths$model <- NULL
  }

  # Validate paths exist
  for (nm in names(paths)) {
    if (!file.exists(paths[[nm]])) {
      warning("Path not found for '", nm, "': ", paths[[nm]], call. = FALSE)
    } else {
      paths[[nm]] <- normalizePath(paths[[nm]])
    }
  }

  entry <- list(
    id = id,
    model_type = model_type,
    paths = paths,
    defaults = defaults
  )

  # Replace or append
  idx <- which(vapply(registry, `[[`, character(1), "id") == id)
  if (length(idx) > 0) {
    registry[[idx]] <- entry
  } else {
    registry[[length(registry) + 1L]] <- entry
  }

  .write_registry(registry)
  invisible(id)
}

#' List registered models
#'
#' Returns a data frame of all models recorded in the sd2R model registry,
#' with a column indicating which are currently loaded in memory.
#'
#' @return Data frame with columns: id, model_type, loaded, diffusion_path
#' @export
sd_list_models <- function() {
  registry <- .read_registry()
  if (length(registry) == 0L) {
    return(data.frame(id = character(), model_type = character(),
                      loaded = logical(), diffusion_path = character(),
                      stringsAsFactors = FALSE))
  }
  data.frame(
    id = vapply(registry, `[[`, character(1), "id"),
    model_type = vapply(registry, `[[`, character(1), "model_type"),
    loaded = vapply(registry, function(e) e$id %in% names(.mm_env$contexts),
                    logical(1)),
    diffusion_path = vapply(registry, function(e) {
      e$paths$diffusion %||% e$paths$model %||% ""
    }, character(1)),
    stringsAsFactors = FALSE
  )
}

#' Load a registered model
#'
#' Loads a model by its registry id. Returns a cached context if already
#' loaded, otherwise creates a new \code{\link{sd_ctx}}. Additional
#' arguments override registry defaults.
#'
#' Before loading, the estimated VRAM need (on-disk weight size times a
#' headroom factor plus a reserve) is compared against free GPU memory; if it
#' would not fit, least-recently-used models are unloaded first. If loading
#' still fails due to insufficient VRAM, the LRU model is unloaded and the load
#' is retried once. VRAM estimation/eviction is skipped when GPU memory cannot
#' be queried (e.g. CPU backend). Tunable via environment variables
#' \code{SD2R_VRAM_HEADROOM} (default 1.2) and \code{SD2R_VRAM_RESERVE_MB}
#' (default 512).
#'
#' @param id Model identifier from registry
#' @param ... Additional arguments passed to \code{\link{sd_ctx}}, overriding
#'   registry defaults (e.g. \code{vae_decode_only = FALSE})
#' @return SD context (external pointer)
#' @importFrom utils modifyList
#' @export
#' @examples
#' \dontrun{
#' ctx <- sd_load_model("flux-dev")
#' imgs <- sd_txt2img(ctx, "a cat in space")
#'
#' # Override defaults
#' ctx <- sd_load_model("flux-dev", vae_decode_only = FALSE, verbose = TRUE)
#' }
sd_load_model <- function(id, ...) {
  # Return cached context if available
  if (id %in% names(.mm_env$contexts)) {
    .mm_env$last_used[[id]] <- Sys.time()
    return(.mm_env$contexts[[id]])
  }

  # Find in registry
  registry <- .read_registry()
  idx <- which(vapply(registry, `[[`, character(1), "id") == id)
  if (length(idx) == 0L) {
    stop("Model '", id, "' not found in registry. Use sd_list_models() to see available.",
         call. = FALSE)
  }
  entry <- registry[[idx]]

  # Build sd_ctx arguments from paths
  ctx_args <- list()
  path_map <- list(
    diffusion = "diffusion_model_path",
    model = "model_path",
    vae = "vae_path",
    clip_l = "clip_l_path",
    clip_g = "clip_g_path",
    t5xxl = "t5xxl_path",
    taesd = "taesd_path",
    control_net = "control_net_path"
  )
  for (nm in names(entry$paths)) {
    arg_name <- path_map[[nm]]
    if (!is.null(arg_name)) {
      ctx_args[[arg_name]] <- entry$paths[[nm]]
    }
  }
  ctx_args$model_type <- entry$model_type

  # Apply defaults from registry
  defaults <- entry$defaults
  if (!is.null(defaults$scheduler)) {
    # Store scheduler name for later use, but don't pass to sd_ctx
    defaults$scheduler <- NULL
  }
  if (!is.null(defaults$steps)) defaults$steps <- NULL
  if (!is.null(defaults$cfg_scale)) defaults$cfg_scale <- NULL
  if (!is.null(defaults$width)) defaults$width <- NULL
  if (!is.null(defaults$height)) defaults$height <- NULL
  if (!is.null(defaults$sample_method)) defaults$sample_method <- NULL

  # User overrides via ...
  user_args <- list(...)
  ctx_args <- modifyList(ctx_args, user_args)

  # Proactively evict LRU models if the new one likely won't fit in free VRAM.
  # Runs before sd_ctx() because on Vulkan an OOM can crash the process outright,
  # so the reactive tryCatch fallback below may never get a chance to run.
  .mm_ensure_vram(entry)

  # Try to load, with LRU eviction on failure (reactive fallback)
  ctx <- tryCatch(do.call(sd_ctx, ctx_args), error = function(e) {
    # If memory error, try evicting LRU model
    if (length(.mm_env$contexts) > 0L &&
        grepl("memory|alloc|VRAM|out of", conditionMessage(e), ignore.case = TRUE)) {
      lru_id <- .find_lru()
      if (!is.null(lru_id)) {
        message("VRAM insufficient, unloading '", lru_id, "' (LRU)")
        sd_unload_model(lru_id)
        return(do.call(sd_ctx, ctx_args))
      }
    }
    stop(e)
  })

  # Store defaults as attributes for sd_generate convenience
  defaults <- entry$defaults
  if (length(defaults) > 0L) attr(ctx, "model_defaults") <- defaults

  .mm_env$contexts[[id]] <- ctx
  .mm_env$last_used[[id]] <- Sys.time()
  ctx
}

#' Unload a model from memory
#'
#' Removes the cached context for the given model id. The model remains
#' in the registry and can be reloaded with \code{\link{sd_load_model}}.
#'
#' @param id Model identifier
#' @return No return value, called for side effects.
#' @export
sd_unload_model <- function(id) {
  if (id %in% names(.mm_env$contexts)) {
    .mm_env$contexts[[id]] <- NULL
    .mm_env$last_used[[id]] <- NULL
    gc()
    message("Unloaded: ", id)
  } else {
    message("Model '", id, "' is not loaded")
  }
  invisible(NULL)
}

#' Unload all models from memory
#'
#' Removes all cached contexts. Registry is preserved.
#'
#' @return No return value, called for side effects.
#' @export
sd_unload_all <- function() {
  n <- length(.mm_env$contexts)
  .mm_env$contexts <- list()
  .mm_env$last_used <- list()
  gc()
  message("Unloaded ", n, " model(s)")
  invisible(NULL)
}

#' Remove a model from the registry
#'
#' Removes the model entry from the sd2R model registry and unloads
#' it from memory if loaded.
#'
#' @param id Model identifier
#' @return No return value, called for side effects.
#' @export
sd_remove_model <- function(id) {
  # Unload if loaded
  if (id %in% names(.mm_env$contexts)) sd_unload_model(id)

  registry <- .read_registry()
  idx <- which(vapply(registry, `[[`, character(1), "id") == id)
  if (length(idx) == 0L) {
    message("Model '", id, "' not in registry")
    return(invisible(NULL))
  }
  registry[[idx]] <- NULL
  .write_registry(registry)
  message("Removed '", id, "' from registry")
  invisible(NULL)
}

#' Scan a directory for models and register them
#'
#' Scans for \code{.safetensors} and \code{.gguf} files, guesses component
#' roles and model types from filenames, groups multi-file models (Flux),
#' and registers them.
#'
#' Single-file models (SD 1.5, SDXL) are registered individually.
#' Multi-file Flux models are grouped when diffusion + supporting files
#' (VAE, CLIP, T5) are found in the same directory.
#'
#' @param dir Directory to scan
#' @param overwrite If TRUE, overwrite existing entries (default FALSE)
#' @param recursive Scan subdirectories (default FALSE)
#' @return Character vector of registered model ids (invisible)
#' @export
#' @examples
#' \dontrun{
#' sd_scan_models("/mnt/models/")
#' sd_list_models()
#' }
sd_scan_models <- function(dir, overwrite = FALSE, recursive = FALSE) {
  if (!dir.exists(dir)) stop("Directory not found: ", dir, call. = FALSE)
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required", call. = FALSE)
  }

  files <- list.files(dir, pattern = "\\.(safetensors|gguf)$",
                       full.names = TRUE, recursive = recursive)
  if (length(files) == 0L) {
    message("No model files found in ", dir)
    return(invisible(character()))
  }

  # Classify each file
  file_info <- data.frame(
    path = files,
    basename = basename(files),
    component = vapply(basename(files), .guess_component, character(1)),
    model_type = vapply(basename(files), .guess_model_type, character(1)),
    stringsAsFactors = FALSE
  )

  registered <- character()

  # Group by directory for multi-file models
  dirs <- unique(dirname(files))
  for (d in dirs) {
    dir_files <- file_info[dirname(file_info$path) == d, , drop = FALSE]

    # Find diffusion models in this dir
    diffusion_files <- dir_files[dir_files$component == "diffusion", , drop = FALSE]
    support_files <- dir_files[dir_files$component %in%
                                 c("vae", "clip_l", "clip_g", "t5xxl", "taesd",
                                   "control_net"), , drop = FALSE]

    for (i in seq_len(nrow(diffusion_files))) {
      df <- diffusion_files[i, ]
      model_type <- df$model_type
      id <- tools::file_path_sans_ext(df$basename)

      paths <- list(diffusion = df$path)

      # Attach only relevant support files based on model type
      relevant <- switch(model_type,
        flux = c("vae", "clip_l", "clip_g", "t5xxl", "taesd"),
        sdxl = c("vae", "clip_l", "clip_g", "taesd"),
        c("vae", "clip_l", "taesd")  # sd1, sd2, unknown
      )
      for (j in seq_len(nrow(support_files))) {
        sf <- support_files[j, ]
        if (sf$component %in% relevant && !sf$component %in% names(paths)) {
          paths[[sf$component]] <- sf$path
        }
      }

      # Generate defaults based on model type
      defaults <- switch(model_type,
        flux = list(steps = 25L, cfg_scale = 1.0, width = 1024L, height = 1024L,
                    sample_method = "euler", scheduler = "discrete"),
        sdxl = list(steps = 25L, cfg_scale = 7.0, width = 1024L, height = 1024L,
                    sample_method = "euler_a", scheduler = "normal"),
        sd1 = list(steps = 20L, cfg_scale = 7.0, width = 512L, height = 512L,
                   sample_method = "euler_a", scheduler = "normal"),
        sd2 = list(steps = 20L, cfg_scale = 7.0, width = 768L, height = 768L,
                   sample_method = "euler_a", scheduler = "normal"),
        list(steps = 20L, cfg_scale = 7.0, width = 512L, height = 512L)
      )

      tryCatch({
        sd_register_model(id, model_type, paths, defaults, overwrite = overwrite)
        registered <- c(registered, id)
        message("Registered: ", id, " (", model_type, ") - ",
                length(paths), " component(s)")
      }, error = function(e) {
        message("Skipped: ", id, " - ", conditionMessage(e))
      })
    }

    # Single-file models that weren't classified as diffusion
    unknown_files <- dir_files[dir_files$component == "unknown", , drop = FALSE]
    for (i in seq_len(nrow(unknown_files))) {
      uf <- unknown_files[i, ]
      # Large files (>100 MB) are likely diffusion models
      fsize <- file.info(uf$path)$size
      if (!is.na(fsize) && fsize > 100 * 1024^2) {
        id <- tools::file_path_sans_ext(uf$basename)
        model_type <- if (uf$model_type != "unknown") uf$model_type else "unknown"
        paths <- list(diffusion = uf$path)

        # Attach only relevant support files based on model type
        # SD1/SD2 don't need t5xxl or clip_g
        relevant <- switch(model_type,
          flux = c("vae", "clip_l", "clip_g", "t5xxl", "taesd"),
          sdxl = c("vae", "clip_l", "clip_g", "taesd"),
          c("vae", "clip_l", "taesd")  # sd1, sd2, unknown
        )
        for (j in seq_len(nrow(support_files))) {
          sf <- support_files[j, ]
          if (sf$component %in% relevant && !sf$component %in% names(paths)) {
            paths[[sf$component]] <- sf$path
          }
        }

        defaults <- switch(model_type,
          flux = list(steps = 25L, cfg_scale = 1.0, width = 1024L, height = 1024L,
                      sample_method = "euler", scheduler = "discrete"),
          sdxl = list(steps = 25L, cfg_scale = 7.0, width = 1024L, height = 1024L,
                      sample_method = "euler_a", scheduler = "normal"),
          sd1 = list(steps = 20L, cfg_scale = 7.0, width = 512L, height = 512L,
                     sample_method = "euler_a", scheduler = "normal"),
          sd2 = list(steps = 20L, cfg_scale = 7.0, width = 768L, height = 768L,
                     sample_method = "euler_a", scheduler = "normal"),
          list(steps = 20L, cfg_scale = 7.0, width = 512L, height = 512L)
        )

        tryCatch({
          sd_register_model(id, model_type, paths, defaults, overwrite = overwrite)
          registered <- c(registered, id)
          message("Registered: ", id, " (", model_type, ", detected by size)")
        }, error = function(e) {
          message("Skipped: ", id, " - ", conditionMessage(e))
        })
      }
    }
  }

  if (length(registered) == 0L) {
    message("No new models registered")
  }
  invisible(registered)
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

#' Find least recently used model id
#' @keywords internal
.find_lru <- function() {
  if (length(.mm_env$last_used) == 0L) return(NULL)
  times <- vapply(.mm_env$last_used, as.numeric, numeric(1))
  names(times)[which.min(times)]
}

# ---------------------------------------------------------------------------
# Proactive VRAM management
# ---------------------------------------------------------------------------

# Headroom multiplier applied to on-disk weight size to estimate peak VRAM,
# plus an absolute reserve (compute buffers, fragmentation). Tunable via env.
.mm_vram_headroom <- function() {
  as.numeric(Sys.getenv("SD2R_VRAM_HEADROOM", "1.2"))
}
.mm_vram_reserve_bytes <- function() {
  as.numeric(Sys.getenv("SD2R_VRAM_RESERVE_MB", "512")) * 1024^2
}

# Sum on-disk size of a registry entry's weight files. Returns 0 if unknown.
.mm_model_size_bytes <- function(entry) {
  paths <- unlist(entry$paths, use.names = FALSE)
  paths <- paths[!is.na(paths) & nzchar(paths)]
  if (length(paths) == 0L) return(0)
  sizes <- file.size(paths)            # NA for missing files
  sum(sizes[!is.na(sizes)])
}

# Free VRAM (bytes) on a device, or NULL if it cannot be queried
# (no Vulkan / CPU backend) — callers must treat NULL as "unknown, skip check".
.mm_free_vram_bytes <- function(device = NULL) {
  if (is.null(device)) {
    device <- as.integer(Sys.getenv("SD_VK_DEVICE", "0"))
  }
  tryCatch({
    mem <- ggmlR::ggml_vulkan_device_memory(device)
    if (is.null(mem$free)) NULL else as.numeric(mem$free)
  }, error = function(e) NULL)
}

# Proactively make room for a model BEFORE loading it: while estimated need
# exceeds free VRAM, evict the LRU loaded model. Returns invisibly TRUE if it
# believes the model will fit (or VRAM is unknown), FALSE if room could not be
# freed. Never throws — loading still proceeds and the reactive OOM fallback in
# sd_load_model() remains as a second line of defence.
.mm_ensure_vram <- function(entry, device = NULL, verbose = TRUE) {
  need <- .mm_model_size_bytes(entry) * .mm_vram_headroom() +
    .mm_vram_reserve_bytes()
  if (need <= .mm_vram_reserve_bytes()) {
    # No usable size estimate (files missing / sizes 0) — can't reason, skip.
    return(invisible(TRUE))
  }

  repeat {
    free <- .mm_free_vram_bytes(device)
    if (is.null(free)) return(invisible(TRUE))   # unknown VRAM — don't block
    if (free >= need) return(invisible(TRUE))    # fits

    lru_id <- .find_lru()
    if (is.null(lru_id)) {
      # Nothing left to evict and still short on VRAM.
      if (verbose) {
        message(sprintf(
          "Insufficient VRAM: model needs ~%.1f GB, %.1f GB free, nothing to evict.",
          need / 1e9, free / 1e9))
      }
      return(invisible(FALSE))
    }
    if (verbose) {
      message(sprintf(
        "Freeing VRAM for new model (~%.1f GB needed, %.1f GB free): unloading '%s' (LRU)",
        need / 1e9, free / 1e9, lru_id))
    }
    sd_unload_model(lru_id)
  }
}
