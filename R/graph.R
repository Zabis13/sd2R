# Graph-based pipeline for sd2R
# Sequential node execution over typed slots (cond, uncond, latent, image):
# each node reads the slots it declares and writes its outputs (see .node_types).

#' Create a pipeline node
#'
#' @param type Node type. High-level: \code{"txt2img"}, \code{"img2img"},
#'   \code{"upscale"}, \code{"save"}, \code{"remove_background"}
#'   (\code{image -> image}; params \code{model_path} (required),
#'   \code{device}, \code{output}, \code{threshold}; see
#'   \code{\link{sd_remove_background}}). Low-level (see Details):
#'   \code{"encode_text"}, \code{"load_image"}, \code{"encode_image"},
#'   \code{"sample"}, \code{"decode"}.
#' @param ... Parameters for the node (passed to the corresponding function).
#' @return A list with class \code{"sd_node"}.
#' @details Slots read -> written, and parameters of the low-level nodes:
#' \describe{
#'   \item{encode_text}{-> \code{cond}, \code{uncond}. \code{prompt}
#'     (required), \code{negative_prompt} (default \code{""}),
#'     \code{clip_skip}, \code{width}, \code{height}; see
#'     \code{\link{sd_encode_text}}.}
#'   \item{load_image}{-> \code{image}. \code{path} (required),
#'     \code{channels}; see \code{\link{sd_load_image}}.}
#'   \item{encode_image}{\code{image} -> \code{latent}; see
#'     \code{\link{sd_encode_image}}.}
#'   \item{sample}{\code{cond} [+ \code{uncond}, \code{latent}] ->
#'     \code{latent}. Other parameters go to \code{\link{sd_sample}}
#'     (\code{sample_steps}, \code{cfg_scale}, \code{strength}, \code{seed},
#'     ...). Without an input latent the noise shape is
#'     \code{c(width, height) / vae_scale_factor} by \code{latent_channels},
#'     both read from the model; \code{width}/\code{height} default to 512,
#'     and \code{vae_scale_factor}, \code{latent_channels} or a full
#'     \code{latent_shape} can be given to override. With an input latent
#'     (img2img) its shape is used; set \code{strength} below 1.}
#'   \item{decode}{\code{latent} -> \code{image}; see
#'     \code{\link{sd_decode_latent}}.}
#' }
#' @examples
#' \dontrun{
#' pipe <- sd_pipeline(
#'   sd_node("encode_text", prompt = "a cat", width = 768L, height = 768L),
#'   sd_node("sample", width = 768L, height = 768L, sample_steps = 20L),
#'   sd_node("decode"),
#'   sd_node("save", path = "cat.png")
#' )
#' sd_run_pipeline(pipe, ctx)
#' }
#' @export
sd_node <- function(type, ...) {
  valid_types <- names(.node_types)
  if (!type %in% valid_types) {
    stop("Unknown node type '", type, "'. Must be one of: ",
         paste(valid_types, collapse = ", "), call. = FALSE)
  }
  node <- list(type = type, params = list(...))
  class(node) <- "sd_node"
  node
}

#' Create a pipeline from nodes
#'
#' Nodes are executed sequentially. The image output of each node is passed
#' as input to the next node.
#'
#' @param ... \code{sd_node} objects in execution order.
#' @return A list with class \code{"sd_pipeline"}.
#' @export
sd_pipeline <- function(...) {
  nodes <- list(...)
  for (i in seq_along(nodes)) {
    if (!inherits(nodes[[i]], "sd_node")) {
      stop("Argument ", i, " is not an sd_node", call. = FALSE)
    }
  }
  if (length(nodes) == 0L) {
    stop("Pipeline must contain at least one node", call. = FALSE)
  }
  pipe <- list(nodes = nodes)
  class(pipe) <- "sd_pipeline"
  pipe
}

#' Run a pipeline
#'
#' Executes nodes sequentially. Data passes between nodes through typed
#' slots (\code{cond}, \code{uncond}, \code{latent}, \code{image}): each
#' node type declares the slots it reads and writes, and reads the most
#' recent value written by an earlier node.
#'
#' The pipeline is validated before any node runs: every slot a node reads
#' must have been written by an earlier node, required parameters must be
#' set (\code{txt2img} and \code{img2img} need their own \code{prompt}; it is
#' not inherited from earlier nodes; \code{save} needs \code{path}), and
#' \code{upscale} needs \code{upscaler_ctx}. All problems are reported in one
#' error.
#'
#' @param pipeline An \code{sd_pipeline} object.
#' @param ctx A Stable Diffusion context created by \code{\link{sd_ctx}}.
#' @param upscaler_ctx Optional upscaler context created by
#'   \code{\link{sd_upscale_image}} setup. Required if the pipeline contains
#'   an \code{"upscale"} node. Pass the result of
#'   \code{sd_create_upscaler(path)}.
#' @param verbose Logical. Print progress messages. Default \code{FALSE}.
#' @return The output of the last node (for the current node types, an
#'   sd_image list).
#' @export
sd_run_pipeline <- function(pipeline, ctx, upscaler_ctx = NULL,
                            verbose = FALSE) {
  if (!inherits(pipeline, "sd_pipeline")) {
    stop("Expected an sd_pipeline object", call. = FALSE)
  }

  nodes <- pipeline$nodes
  validate_pipeline(nodes, upscaler_ctx)

  env <- list(ctx = ctx, upscaler_ctx = upscaler_ctx)
  state <- stats::setNames(vector("list", length(.slot_types)), .slot_types)
  last <- NULL

  for (i in seq_along(nodes)) {
    node <- nodes[[i]]
    spec <- .node_types[[node$type]]

    if (verbose) message("[", i, "/", length(nodes), "] ", node$type)

    inputs <- state[c(spec$inputs, spec$optional)]
    out <- spec$run(env, node$params, inputs)
    if (!identical(sort(names(out)), sort(spec$outputs))) {
      stop(sprintf("node %d (%s) returned slots [%s], declared [%s]", i,
                   node$type, paste(names(out), collapse = ", "),
                   paste(spec$outputs, collapse = ", ")), call. = FALSE)
    }
    state[names(out)] <- out
    last <- out[[length(out)]]
  }

  last
}

#' Save pipeline to JSON
#'
#' @param pipeline An \code{sd_pipeline} object.
#' @param path File path (should end in \code{.json}).
#' @return The file path, invisibly.
#' @export
sd_save_pipeline <- function(pipeline, path) {
  if (!inherits(pipeline, "sd_pipeline")) {
    stop("Expected an sd_pipeline object", call. = FALSE)
  }
  nodes_list <- lapply(pipeline$nodes, function(n) {
    list(type = n$type, params = n$params)
  })
  json <- pipeline_to_json(nodes_list)
  writeLines(json, path)
  invisible(path)
}

#' Load pipeline from JSON
#'
#' @param path Path to a JSON file saved by \code{\link{sd_save_pipeline}}.
#' @return An \code{sd_pipeline} object.
#' @export
sd_load_pipeline <- function(path) {
  json <- paste(readLines(path, warn = FALSE), collapse = "\n")
  nodes_list <- pipeline_from_json(json)
  nodes <- lapply(nodes_list, function(n) {
    do.call(sd_node, c(list(type = n$type), n$params))
  })
  do.call(sd_pipeline, nodes)
}

# --- print methods ---

#' @export
print.sd_node <- function(x, ...) {
  cat("sd_node:", x$type, "\n")
  if (length(x$params) > 0L) {
    for (nm in names(x$params)) {
      val <- x$params[[nm]]
      if (is.character(val) && nchar(val) > 60L) {
        val <- paste0(substr(val, 1, 57), "...")
      }
      cat("  ", nm, "=", format(val), "\n")
    }
  }
  invisible(x)
}

#' @export
print.sd_pipeline <- function(x, ...) {
  cat("sd_pipeline:", length(x$nodes), "node(s)\n")
  for (i in seq_along(x$nodes)) {
    cat("  [", i, "] ", x$nodes[[i]]$type, "\n")
  }
  invisible(x)
}

# =========================================================================
# Internal: typed node registry
# =========================================================================
#
# Data flows through typed slots. The runner keeps a state
# list(cond, uncond, latent, image); each node type declares which slots it
# reads (`inputs` required, `optional` if present) and which it writes
# (`outputs`). No implicit inheritance: a node only sees slots written by
# earlier nodes, and its own params must be stated explicitly.
#
# run(env, params, inputs) -> named list of output slot values, where
#   env    = list(ctx, upscaler_ctx)
#   inputs = named list of the slots the node reads (optional ones may be NULL)

.slot_types <- c("cond", "uncond", "latent", "image")

.node_types <- list(
  txt2img = list(
    inputs = character(0), optional = character(0), outputs = "image",
    required_params = "prompt",
    run = function(env, params, inputs) {
      # sd_txt2img() has no `verbose`; core logging is set on the context
      res <- do.call(sd_txt2img, c(list(ctx = env$ctx), params))
      list(image = first_image(res))
    }
  ),
  img2img = list(
    inputs = "image", optional = character(0), outputs = "image",
    required_params = "prompt",
    run = function(env, params, inputs) {
      res <- do.call(sd_img2img,
                     c(list(ctx = env$ctx, init_image = inputs$image), params))
      list(image = first_image(res))
    }
  ),
  upscale = list(
    inputs = "image", optional = character(0), outputs = "image",
    required_params = character(0), needs_upscaler = TRUE,
    run = function(env, params, inputs) {
      factor <- params$factor %||% 4L
      list(image = sd_upscale(env$upscaler_ctx, inputs$image, as.integer(factor)))
    }
  ),
  save = list(
    inputs = "image", optional = character(0), outputs = "image",
    required_params = "path",
    run = function(env, params, inputs) {
      sd_save_image(inputs$image, params$path)
      list(image = inputs$image)
    }
  ),
  remove_background = list(
    inputs = "image", optional = character(0), outputs = "image",
    required_params = "model_path",
    run = function(env, params, inputs) {
      model <- .rmbg_cached(params$model_path, params$device)
      list(image = sd_remove_background(inputs$image, model,
                                        output = params$output %||% "rgba",
                                        threshold = params$threshold))
    }
  ),
  # --- low-level nodes (encode -> sample -> decode) ---
  encode_text = list(
    inputs = character(0), optional = character(0),
    outputs = c("cond", "uncond"),
    required_params = "prompt",
    run = function(env, params, inputs) {
      enc <- function(text) {
        sd_encode_text(env$ctx, text,
                       clip_skip = params$clip_skip %||% -1L,
                       width = params$width %||% -1L,
                       height = params$height %||% -1L)
      }
      # no negative_prompt -> empty-prompt encoding, as in regular CFG
      list(cond = enc(params$prompt),
           uncond = enc(params$negative_prompt %||% ""))
    }
  ),
  load_image = list(
    inputs = character(0), optional = character(0), outputs = "image",
    required_params = "path",
    run = function(env, params, inputs) {
      list(image = sd_load_image(params$path, params$channels %||% 3L))
    }
  ),
  encode_image = list(
    inputs = "image", optional = character(0), outputs = "latent",
    required_params = character(0),
    run = function(env, params, inputs) {
      list(latent = sd_encode_image(env$ctx, inputs$image))
    }
  ),
  sample = list(
    inputs = "cond", optional = c("uncond", "latent"), outputs = "latent",
    required_params = character(0),
    run = function(env, params, inputs) {
      shape_keys <- c("width", "height", "latent_channels",
                      "vae_scale_factor", "latent_shape")
      init <- inputs$latent
      shape <- if (!is.null(init)) {
        init$ne[1:3]  # img2img: noise must match the input latent
      } else if (!is.null(params$latent_shape)) {
        as.integer(params$latent_shape)
      } else {
        g <- latent_geometry(env$ctx, params$vae_scale_factor,
                             params$latent_channels)
        c(as.integer(params$width %||% 512L) %/% g[["vae_scale_factor"]],
          as.integer(params$height %||% 512L) %/% g[["vae_scale_factor"]],
          g[["latent_channels"]])
      }
      uncond <- inputs$uncond %||% list(crossattn = NULL, vector = NULL,
                                        concat = NULL)
      args <- params[setdiff(names(params), shape_keys)]
      res <- do.call(sd_sample, c(list(ctx = env$ctx, cond = inputs$cond,
                                       uncond = uncond, latent_shape = shape,
                                       init_latent = init), args))
      list(latent = res)
    }
  ),
  decode = list(
    inputs = "latent", optional = character(0), outputs = "image",
    required_params = character(0),
    run = function(env, params, inputs) {
      list(image = sd_decode_latent(env$ctx, inputs$latent))
    }
  )
)

# Latent geometry for the `sample` node: queried from the model, each value
# can be overridden explicitly (needed when ctx is not a live context).
latent_geometry <- function(ctx, vae_scale_factor = NULL,
                            latent_channels = NULL) {
  if (is.null(vae_scale_factor) || is.null(latent_channels)) {
    info <- if (is.null(ctx)) c(NA_integer_, NA_integer_)
            else sd_ctx_latent_info_cpp(ctx)
    vae_scale_factor <- vae_scale_factor %||% info[1L]
    latent_channels <- latent_channels %||% info[2L]
  }
  if (is.na(vae_scale_factor) || is.na(latent_channels)) {
    stop("sample: cannot read latent geometry from ctx; pass ",
         "'latent_channels' and 'vae_scale_factor' (or 'latent_shape')",
         call. = FALSE)
  }
  c(vae_scale_factor = as.integer(vae_scale_factor),
    latent_channels = as.integer(latent_channels))
}

# sd_txt2img / sd_img2img return a list of images (batch); take the first
first_image <- function(res) {
  if (is.list(res) && !is.null(res$width)) res else res[[1L]]
}

# =========================================================================
# Internal: validation (runs before any node executes)
# =========================================================================

# Walks the nodes in order, tracking which slots have been written, and
# collects every problem into one error.
validate_pipeline <- function(nodes, upscaler_ctx = NULL) {
  errs <- character(0)
  available <- character(0)
  for (i in seq_along(nodes)) {
    type <- nodes[[i]]$type
    params <- nodes[[i]]$params
    spec <- .node_types[[type]]
    where <- sprintf("node %d (%s)", i, type)

    if (is.null(spec)) {
      errs <- c(errs, sprintf("%s: unknown node type", where))
      next
    }
    for (p in spec$required_params) {
      v <- params[[p]]
      if (is.null(v) || (is.character(v) && !nzchar(v[1L]))) {
        errs <- c(errs, sprintf("%s: '%s' is not set", where, p))
      }
    }
    for (s in setdiff(spec$inputs, available)) {
      errs <- c(errs, sprintf("%s: needs '%s', but no earlier node produces it",
                              where, s))
    }
    if (isTRUE(spec$needs_upscaler) && is.null(upscaler_ctx)) {
      errs <- c(errs, sprintf("%s: requires 'upscaler_ctx' in sd_run_pipeline()",
                              where))
    }
    available <- union(available, spec$outputs)
  }
  if (length(errs)) {
    stop("Invalid pipeline:\n  ", paste(errs, collapse = "\n  "), call. = FALSE)
  }
  invisible(TRUE)
}

# =========================================================================
# Internal: minimal JSON serialization (no external dependencies)
# =========================================================================

pipeline_to_json <- function(nodes_list) {
  node_strings <- vapply(nodes_list, function(n) {
    params_str <- if (length(n$params) == 0L) {
      "{}"
    } else {
      pairs <- vapply(names(n$params), function(nm) {
        paste0("      ", json_quote(nm), ": ", json_value(n$params[[nm]]))
      }, character(1L))
      paste0("{\n", paste(pairs, collapse = ",\n"), "\n    }")
    }
    paste0("  {\n    \"type\": ", json_quote(n$type),
           ",\n    \"params\": ", params_str, "\n  }")
  }, character(1L))
  paste0("[\n", paste(node_strings, collapse = ",\n"), "\n]\n")
}

pipeline_from_json <- function(json) {
  # Pipeline format: [{"type": "...", "params": {...}}, ...]
  parsed <- json_parse(json)
  bad <- function(msg) stop("Invalid pipeline JSON: ", msg, call. = FALSE)
  if (!is.list(parsed) || !is.null(names(parsed))) {
    bad("expected an array of nodes")
  }
  lapply(seq_along(parsed), function(i) {
    n <- parsed[[i]]
    if (!is.list(n) || is.null(names(n))) bad(sprintf("node %d is not an object", i))
    if (!is.character(n$type) || length(n$type) != 1L) {
      bad(sprintf("node %d has no \"type\" string", i))
    }
    params <- n$params %||% list()
    if (!is.list(params) || (length(params) && is.null(names(params)))) {
      bad(sprintf("node %d: \"params\" is not an object", i))
    }
    list(type = n$type, params = params)
  })
}

# Small recursive-descent JSON parser (RFC 8259). Never evaluates the input.
# object -> named list, array -> unnamed list (simplified to an atomic vector
# when all elements are scalars of one kind), integer literals -> integer.
json_parse <- function(txt) {
  chars <- strsplit(enc2utf8(paste(txt, collapse = "\n")), "")[[1L]]
  n <- length(chars)
  pos <- 1L

  fail <- function(msg) {
    stop(sprintf("Invalid pipeline JSON at position %d: %s", pos, msg),
         call. = FALSE)
  }
  peek <- function() if (pos <= n) chars[pos] else ""
  skip_ws <- function() {
    while (pos <= n && chars[pos] %in% c(" ", "\t", "\n", "\r")) pos <<- pos + 1L
  }
  expect <- function(ch) {
    if (peek() != ch) fail(sprintf("expected '%s'", ch))
    pos <<- pos + 1L
  }
  word <- function(w) {
    k <- nchar(w)
    pos + k - 1L <= n && paste(chars[pos:(pos + k - 1L)], collapse = "") == w
  }

  parse_value <- function() {
    skip_ws()
    ch <- peek()
    if (ch == "{") return(parse_object())
    if (ch == "[") return(parse_array())
    if (ch == "\"") return(parse_string())
    if (ch %in% c("-", 0:9)) return(parse_number())
    if (word("true"))  { pos <<- pos + 4L; return(TRUE) }
    if (word("false")) { pos <<- pos + 5L; return(FALSE) }
    if (word("null"))  { pos <<- pos + 4L; return(NULL) }
    fail(if (pos > n) "unexpected end of input" else "unexpected character")
  }

  parse_string <- function() {
    expect("\"")
    out <- character(0)
    repeat {
      if (pos > n) fail("unterminated string")
      ch <- chars[pos]
      pos <<- pos + 1L
      if (ch == "\"") break
      if (ch == "\\") {
        if (pos > n) fail("unterminated string")
        e <- chars[pos]
        pos <<- pos + 1L
        ch <- switch(e,
          "\"" = "\"", "\\" = "\\", "/" = "/",
          b = "\b", f = "\f", n = "\n", r = "\r", t = "\t",
          u = {
            h <- if (pos + 3L <= n) paste(chars[pos:(pos + 3L)], collapse = "") else ""
            if (!grepl("^[0-9A-Fa-f]{4}$", h)) fail("bad \\u escape")
            pos <<- pos + 4L
            intToUtf8(strtoi(h, 16L))
          },
          fail("bad escape sequence"))
      }
      out <- c(out, ch)
    }
    paste(out, collapse = "")
  }

  parse_number <- function() {
    start <- pos
    while (pos <= n && chars[pos] %in% c("-", "+", ".", "e", "E", 0:9)) {
      pos <<- pos + 1L
    }
    s <- paste(chars[start:(pos - 1L)], collapse = "")
    if (!grepl("^-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][-+]?[0-9]+)?$", s)) {
      fail(sprintf("bad number '%s'", s))
    }
    v <- as.numeric(s)
    if (!grepl("[.eE]", s) && abs(v) <= .Machine$integer.max) as.integer(v) else v
  }

  parse_array <- function() {
    expect("[")
    skip_ws()
    out <- list()
    if (peek() == "]") { pos <<- pos + 1L; return(out) }
    repeat {
      out <- c(out, list(parse_value()))
      skip_ws()
      if (peek() == ",") { pos <<- pos + 1L; next }
      expect("]")
      break
    }
    scalar <- vapply(out, function(v) is.atomic(v) && length(v) == 1L, logical(1L))
    kinds <- unique(vapply(out, function(v) {
      if (is.numeric(v)) "numeric" else class(v)[1L]
    }, character(1L)))
    if (all(scalar) && length(kinds) == 1L) unlist(out) else out
  }

  parse_object <- function() {
    expect("{")
    skip_ws()
    out <- list()
    keys <- character(0)
    if (peek() == "}") {
      pos <<- pos + 1L
      return(stats::setNames(out, character(0)))
    }
    repeat {
      skip_ws()
      if (peek() != "\"") fail("expected a string key")
      keys <- c(keys, parse_string())
      skip_ws()
      expect(":")
      out <- c(out, list(parse_value()))
      skip_ws()
      if (peek() == ",") { pos <<- pos + 1L; next }
      expect("}")
      break
    }
    names(out) <- keys
    out
  }

  value <- parse_value()
  skip_ws()
  if (pos <= n) fail("trailing characters after JSON value")
  value
}

json_quote <- function(x) {
  x <- as.character(x)
  x <- gsub("\\", "\\\\", x, fixed = TRUE)
  x <- gsub("\"", "\\\"", x, fixed = TRUE)
  x <- gsub("\n", "\\n", x, fixed = TRUE)
  x <- gsub("\r", "\\r", x, fixed = TRUE)
  x <- gsub("\t", "\\t", x, fixed = TRUE)
  paste0("\"", x, "\"")
}

json_value <- function(x) {
  if (is.null(x)) return("null")
  if (length(x) != 1L || is.list(x)) {
    return(paste0("[", paste(vapply(x, json_value, character(1L)),
                             collapse = ", "), "]"))
  }
  if (is.logical(x)) return(tolower(as.character(x)))
  if (is.integer(x)) return(as.character(x))
  if (is.numeric(x)) return(format(x, digits = 15, scientific = FALSE))
  json_quote(x)
}

# base R %||% for older R versions
`%||%` <- function(a, b) if (is.null(a)) b else a
