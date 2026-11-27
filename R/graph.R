# Graph-based pipeline for sd2R
# Sequential node execution over typed slots (cond, uncond, latent, image):
# each node reads the slots it declares and writes its outputs (see .node_types).

#' Create a pipeline node
#'
#' @param type Node type. High-level: \code{"txt2img"}, \code{"img2img"}
#'   (\code{image} [+ \code{mask}] -> \code{image}; with a mask it inpaints),
#'   \code{"upscale"}, \code{"save"}, \code{"remove_background"}
#'   (\code{image -> mask, image}; params \code{model_path} (required),
#'   \code{device}, \code{output}, \code{threshold}; see
#'   \code{\link{sd_remove_background}}). Masks: \code{"load_mask"},
#'   \code{"invert_mask"}, \code{"save_mask"}. Image operations:
#'   \code{"resize"}, \code{"resize_mask"}, \code{"crop"},
#'   \code{"crop_mask"}, \code{"composite"}. ControlNet preprocessors
#'   (\code{image -> image}): \code{"canny"} (params of
#'   \code{\link{sd_canny}}), \code{"depth"} (\code{model_path}
#'   required, \code{device}, \code{size}; see
#'   \code{\link{sd_estimate_depth}}); their output goes to the
#'   \code{control} input of \code{txt2img} or \code{sample} (linked
#'   explicitly; needs a context with \code{control_net_path}; strength via
#'   \code{control_strength}) (see Details). Low-level:
#'   \code{"encode_text"}, \code{"load_image"}, \code{"encode_image"},
#'   \code{"sample"}, \code{"decode"}, \code{"upscale_latent"}.
#' @param ... Parameters for the node (passed to the corresponding function).
#'   Arguments named after one of the node's inputs (\code{cond},
#'   \code{uncond}, \code{latent}, \code{image}) are links to earlier nodes:
#'   \code{"id"} or \code{"id$output"} (see \code{\link{sd_run_pipeline}}).
#' @param id Optional node id, used to link other nodes to this one. Assigned
#'   automatically by \code{\link{sd_pipeline}} if omitted.
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
#'     (img2img) its shape is used; set \code{strength} below 1.
#'     Optional inputs, linked explicitly only (never the nearest node):
#'     \code{mask} (inpainting; needs \code{latent}; not for inpainting
#'     checkpoints yet) and \code{control} (an image: ControlNet hint; needs
#'     a context with \code{control_net_path}, checked before the run;
#'     strength via \code{control_strength}, in [0, 2], default 0.9). Both
#'     must have the generation size in pixels (else an error: use
#'     \code{resize} / \code{resize_mask}).}
#'   \item{decode}{\code{latent} -> \code{image}; see
#'     \code{\link{sd_decode_latent}}.}
#'   \item{upscale_latent}{\code{latent} -> \code{latent}, resized in latent
#'     space: \code{scale}, or \code{width} and \code{height} in pixels
#'     (multiples of the VAE scale factor); \code{method} as for
#'     \code{resize}. Only plain 4- or 16-channel latents (SD1/2/XL,
#'     SD3/FLUX.1); FLUX.2/3 latents pack 2x2 sub-pixels into 128 channels
#'     and are refused. A quick option: re-sample with a high
#'     \code{strength} (about 0.5-0.7), and expect lower quality than
#'     upscaling the image.}
#' }
#'
#' Highres fix: generate at the model's native size, upscale, then refine
#' with img2img. Upscaling the image works for every model and gives the
#' best quality (use an \code{"upscale"} node with ESRGAN instead of
#' \code{"resize"} for sharper detail):
#' \preformatted{
#' sd_pipeline(
#'   sd_node("encode_text", id = "txt", prompt = "a castle",
#'           width = 1024L, height = 1024L),
#'   sd_node("sample", id = "base", cond = "txt", width = 512L, height = 512L),
#'   sd_node("decode", id = "small", latent = "base"),
#'   sd_node("resize", id = "big", image = "small", width = 1024L),
#'   sd_node("encode_image", id = "init", image = "big"),
#'   sd_node("sample", id = "hr", cond = "txt", latent = "init",
#'           strength = 0.45),
#'   sd_node("decode", latent = "hr"),
#'   sd_node("save", path = "castle.png")
#' )
#' }
#' For plain latents, \code{decode -> resize -> encode_image} can be replaced
#' by \code{sd_node("upscale_latent", latent = "base", scale = 2)} (skips the
#' VAE round trip; use a higher strength).
#'
#' Run-time inputs: \code{"input_image"}, \code{"input_mask"},
#' \code{"input_latent"} (no parameters) output the value given for their
#' id in \code{sd_run_pipeline(inputs = )}; the graph (and its JSON) holds
#' no data. Used by \code{\link{sd_run_batch}} and \code{\link{sd_loopback}}.
#'
#' Masks are a separate type from images: a 1-channel image with values
#' 0..255 ([0, 1] in R), white = regenerate, black = keep the original. A
#' mask input accepts only a mask output, and only as a link to a node (not
#' a file path).
#' \describe{
#'   \item{load_mask}{-> \code{mask}. \code{path} (required); see
#'     \code{\link{sd_load_mask}}.}
#'   \item{invert_mask}{\code{mask} -> \code{mask} (white <-> black).}
#'   \item{save_mask}{\code{mask} -> \code{mask} (pass-through, always
#'     runs). \code{path} (required).}
#'   \item{remove_background}{\code{image} -> \code{mask} (foreground is
#'     white), \code{image} (RGBA, or the mask if \code{output = "mask"}).}
#' }
#' Image operations (pure R, run on the CPU):
#' \describe{
#'   \item{resize, resize_mask}{\code{image -> image} / \code{mask -> mask}.
#'     \code{width} and/or \code{height} (a missing side keeps the aspect
#'     ratio) or \code{scale} (not both); the size is rounded to the nearest
#'     integer. \code{method}: \code{"bilinear"} (default, also for masks:
#'     keeps soft edges) or \code{"nearest"} (for binary masks).}
#'   \item{crop, crop_mask}{\code{image -> image} / \code{mask -> mask}.
#'     \code{x}, \code{y} (0-based offsets of the top-left corner, default
#'     0), \code{width}, \code{height} (required). A region outside the
#'     input is an error.}
#'   \item{composite}{\code{fg}, \code{bg} (images), \code{mask} ->
#'     \code{image}: \code{fg * mask + bg * (1 - mask)}. All three inputs
#'     must be linked explicitly (no nearest-node default, so fg and bg cannot
#'     be swapped silently) and have the same size (else an error: use
#'     \code{resize} / \code{resize_mask}). fg and bg are used as RGB: an
#'     alpha channel is ignored, grayscale is expanded. The result has 3
#'     channels.}
#' }
#' To replace the background, invert the foreground mask and inpaint:
#' \preformatted{
#' sd_pipeline(
#'   sd_node("load_image", id = "src", path = "photo.png"),
#'   sd_node("remove_background", id = "bg", image = "src",
#'           model_path = "RMBG-1.4.onnx"),
#'   sd_node("invert_mask", id = "back", mask = "bg$mask"),
#'   sd_node("img2img", image = "src", mask = "back",
#'           prompt = "on a beach", strength = 0.9),
#'   sd_node("save", path = "beach.png")
#' )
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
sd_node <- function(type, ..., id = NULL) {
  valid_types <- names(.node_types)
  if (!type %in% valid_types) {
    stop("Unknown node type '", type, "'. Must be one of: ",
         paste(valid_types, collapse = ", "), call. = FALSE)
  }
  if (!is.null(id) && !is_ref_string(id)) {
    stop("node '", type, "': 'id' must be a non-empty string", call. = FALSE)
  }
  if (!is.null(id) && grepl("$", id, fixed = TRUE)) {
    stop("node '", type, "': 'id' must not contain '$'", call. = FALSE)
  }
  spec <- .node_types[[type]]
  args <- list(...)
  arg_names <- names(args) %||% rep("", length(args))
  # arguments named after a declared input are links to other nodes
  is_input <- arg_names %in% c(spec$inputs, spec$optional)
  for (nm in arg_names[is_input]) {
    if (!is_ref_string(args[[nm]])) {
      stop(sprintf(paste0("node '%s': input '%s' must be a node id string ",
                          "(\"id\" or \"id$output\")"), type, nm),
           call. = FALSE)
    }
  }
  node <- list(type = type, id = id, inputs = args[is_input],
               params = args[!is_input])
  class(node) <- "sd_node"
  node
}

is_ref_string <- function(x) {
  is.character(x) && length(x) == 1L && !is.na(x) && nzchar(x)
}

#' Create a pipeline from nodes
#'
#' Nodes run in the given order. Each input of a node is either linked
#' explicitly to an earlier node (\code{sd_node("decode", latent = "s1")})
#' or, if not given, taken from the nearest earlier node that produces it.
#'
#' Nodes without an \code{id} get one automatically
#' (\code{"<type>_<position>"}). Ids must be unique.
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
  ids <- vapply(nodes, function(n) n$id %||% NA_character_, character(1L))
  dup <- unique(ids[!is.na(ids) & duplicated(ids)])
  if (length(dup)) {
    stop("Duplicate node id(s): ", paste(dup, collapse = ", "), call. = FALSE)
  }
  for (i in which(is.na(ids))) {
    id <- paste0(nodes[[i]]$type, "_", i)
    while (id %in% ids) id <- paste0(id, "_")
    ids[i] <- id
    nodes[[i]]$id <- id
  }
  pipe <- list(nodes = nodes)
  class(pipe) <- "sd_pipeline"
  pipe
}

#' Run a pipeline
#'
#' Executes nodes in order. Data passes between nodes through typed outputs
#' (\code{cond}, \code{uncond}, \code{latent}, \code{image}): each node type
#' declares the inputs it reads and the outputs it writes. An input is
#' linked to an earlier node by id (\code{"id"}, or \code{"id$output"} to
#' pick a specific output); without a link it reads the nearest earlier node
#' that produces it. With \code{"id"} alone, the output with the same name
#' as the input is used, or else the only output of a matching type
#' (\code{cond} and \code{uncond} are the same type).
#'
#' The pipeline is validated before any node runs: every input must resolve
#' to an earlier node with a matching output, required parameters must be
#' set (\code{txt2img} and \code{img2img} need their own \code{prompt}; it is
#' not inherited from earlier nodes; \code{save} needs \code{path}), and
#' \code{upscale} needs \code{upscaler_ctx}. All problems are reported in one
#' error.
#'
#' Only the nodes needed for the requested outputs run, plus \code{save}
#' nodes (side effects) and everything they depend on; other nodes are
#' skipped. Intermediate results are released as soon as their last
#' consumer has run, unless requested in \code{outputs}.
#'
#' @param pipeline An \code{sd_pipeline} object.
#' @param ctx A Stable Diffusion context created by \code{\link{sd_ctx}}.
#' @param upscaler_ctx Optional upscaler context created by
#'   \code{\link{sd_upscale_image}} setup. Required if the pipeline contains
#'   an \code{"upscale"} node. Pass the result of
#'   \code{sd_create_upscaler(path)}.
#' @param outputs Which results to return. \code{NULL} (default): the last
#'   output of the last node, as a single value. A character vector of node
#'   ids: a named list with one element per id. \code{"all"}: every node
#'   (all results are kept in R memory until the end).
#' @param cache Optional \code{\link{sd_cache}} object. Nodes whose type,
#'   parameters and inputs are unchanged since an earlier run with the same
#'   cache return the stored result instead of running (see
#'   \code{\link{sd_cache}} for what is cached). \code{NULL} (default): no
#'   caching.
#' @param on_event Optional function called with one argument, an event
#'   list, as the run progresses (see Events). It only observes: its return
#'   value is ignored. An error in it stops the run, naming the handler's
#'   node and event; call \code{\link{sd_pipeline_cancel}} from it to stop
#'   the run as a cancel.
#' @param verbose Logical. Print progress messages (via \code{message()},
#'   built on the same events). Default \code{FALSE}.
#' @param inputs Values for the run-time input nodes (\code{input_image},
#'   \code{input_mask}, \code{input_latent}), a list named by node id. Every
#'   input node needs a value of its type; unknown ids are an error. Checked
#'   before anything runs.
#' @section Events:
#' Events, in order: \code{"skipped"} for every node removed by culling
#' (before anything runs), then per node to run either \code{"cached"}
#' (result taken from \code{cache}), or \code{"start"} followed by
#' \code{"end"} or \code{"error"} (the error is then re-raised). Fields:
#' \describe{
#'   \item{schema}{Event format version, currently \code{1L}. New fields
#'     may be added within a version; existing ones are not renamed.}
#'   \item{event}{One of the event names above.}
#'   \item{id, type, index}{Node id, type and position in the pipeline.}
#'   \item{step, total}{Position among the nodes that run (\code{NA} for
#'     \code{"skipped"}) and their number.}
#'   \item{elapsed}{Seconds the node took (\code{"end"}, \code{"error"};
#'     \code{0} for \code{"cached"}; \code{NA} otherwise), measured with
#'     \code{proc.time()} (wall clock, not guaranteed monotonic).}
#'   \item{outputs}{For \code{"end"} and \code{"cached"}: per output, its
#'     metadata only (\code{class}, size: \code{width}/\code{height}/
#'     \code{channel} or \code{ne}, \code{bytes}), never the value; get
#'     values with \code{outputs}.}
#'   \item{error}{For \code{"error"}: the error message.}
#' }
#' @return With \code{outputs = NULL}, the last output of the last node (for
#'   the current node types, an sd_image list). Otherwise a list named by
#'   node id: for a node with one output, the value itself; for several
#'   (\code{encode_text}), a list of them (\code{res$txt$cond}).
#' @export
sd_run_pipeline <- function(pipeline, ctx, upscaler_ctx = NULL,
                            outputs = NULL, cache = NULL, on_event = NULL,
                            verbose = FALSE, inputs = NULL) {
  if (!inherits(pipeline, "sd_pipeline")) {
    stop("Expected an sd_pipeline object", call. = FALSE)
  }
  run_inputs <- inputs   # `inputs` is reused below for node inputs
  if (!is.null(on_event) && !is.function(on_event)) {
    stop("'on_event' must be a function or NULL", call. = FALSE)
  }
  if (!is.null(cache)) .check_cache(cache)

  nodes <- pipeline$nodes
  links <- validate_pipeline(nodes, upscaler_ctx)
  .check_run_inputs(nodes, run_inputs)
  check_ctx_requirements(nodes, links, ctx)
  ids <- vapply(nodes, function(n) n$id %||% NA_character_, character(1L))
  n <- length(nodes)

  # requested nodes: NULL -> last node (single value), "all", or ids
  if (is.null(outputs)) {
    wanted <- n
  } else {
    if (!is.character(outputs) || !length(outputs) || anyNA(outputs)) {
      stop("'outputs' must be NULL, \"all\" or a character vector of node ids",
           call. = FALSE)
    }
    if (identical(outputs, "all")) outputs <- ids
    wanted <- match(outputs, ids)
    if (anyNA(wanted)) {
      stop("Unknown node id(s) in 'outputs': ",
           paste(outputs[is.na(wanted)], collapse = ", "), call. = FALSE)
    }
  }

  # cache: keys are computed up front; a hit replaces running the node
  cacheable <- vapply(nodes, function(x) isTRUE(.node_types[[x$type]]$cache),
                      logical(1L))
  hit <- logical(n)
  if (!is.null(cache)) {
    keys <- .cache_keys(cache, nodes, links, ctx,
                        extra = .input_hashes(nodes, run_inputs))
    cacheable <- cacheable & keys$stable
    hit <- cacheable & keys$id %in% names(cache$entries)
  }

  # cull: run only requested nodes, side-effect nodes (save) and their
  # ancestors; a cache hit needs no ancestors
  side <- vapply(nodes, function(x) isTRUE(.node_types[[x$type]]$side_effect),
                 logical(1L))
  needed <- logical(n)
  needed[c(wanted, which(side))] <- TRUE
  for (i in rev(seq_len(n))) {
    if (needed[i] && !hit[i]) {
      for (l in links[[i]]) needed[l$node] <- TRUE
    }
  }
  run_idx <- which(needed)

  # last consumer of each node's outputs -> free them after it ran
  last_use <- integer(n)
  for (i in run_idx[!hit[run_idx]]) {
    for (l in links[[i]]) last_use[l$node] <- max(last_use[l$node], i)
  }

  env <- list(ctx = ctx, upscaler_ctx = upscaler_ctx,
              input_values = run_inputs, node_id = NULL)
  results <- vector("list", n)
  kept <- vector("list", n)

  # events: verbose is the built-in handler, then the user's on_event
  handlers <- c(if (verbose) list(.verbose_handler), list(on_event))
  emit <- function(event, i, step = NA_integer_, elapsed = NA_real_,
                   out = NULL, error = NULL) {
    ev <- list(schema = 1L, event = event, id = nodes[[i]]$id,
               type = nodes[[i]]$type, index = i, step = step,
               total = length(run_idx), elapsed = elapsed,
               outputs = if (!is.null(out)) .outputs_meta(out),
               error = error)
    for (h in handlers) .call_handler(h, ev)
  }

  for (i in which(!needed)) emit("skipped", i)

  for (k in seq_along(run_idx)) {
    i <- run_idx[k]
    node <- nodes[[i]]
    spec <- .node_types[[node$type]]

    if (hit[i]) {
      out <- .cache_get(cache, keys$id[i])
      emit("cached", i, k, elapsed = 0, out = out)
      cache$hits <- cache$hits + 1L
      results[i] <- list(out)
      if (i %in% wanted) kept[i] <- list(out)
      if (last_use[i] == 0L) results[i] <- list(NULL)
      next
    }
    emit("start", i, k)
    t0 <- proc.time()[["elapsed"]]

    # every declared input is passed; unlinked optional ones as NULL
    slots <- c(spec$inputs, spec$optional)
    inputs <- stats::setNames(vector("list", length(slots)), slots)
    for (s in names(links[[i]])) {
      l <- links[[i]][[s]]
      inputs[s] <- list(results[[l$node]][[l$port]])
    }
    env$node_id <- node$id
    out <- tryCatch({
      o <- spec$run(env, node$params, inputs)
      if (!identical(sort(names(o)), sort(spec$outputs))) {
        stop(sprintf("node %d (%s) returned slots [%s], declared [%s]", i,
                     node$type, paste(names(o), collapse = ", "),
                     paste(spec$outputs, collapse = ", ")), call. = FALSE)
      }
      o
    }, error = function(e) {
      emit("error", i, k, elapsed = proc.time()[["elapsed"]] - t0,
           error = conditionMessage(e))
      stop(e)
    })
    rm(inputs)
    emit("end", i, k, elapsed = proc.time()[["elapsed"]] - t0, out = out)
    results[i] <- list(out)
    if (i %in% wanted) kept[i] <- list(out)
    if (!is.null(cache) && cacheable[i]) {
      .cache_put(cache, keys$id[i], out)
      cache$misses <- cache$misses + 1L
    }

    # drop inputs whose last consumer was this node
    for (l in links[[i]]) {
      if (last_use[l$node] == i) results[l$node] <- list(NULL)
    }
    if (last_use[i] == 0L) results[i] <- list(NULL)
  }

  if (is.null(outputs)) {
    out <- kept[[n]]
    return(out[[length(out)]])
  }
  # one output -> the value itself; several -> named list of outputs
  stats::setNames(lapply(wanted, function(i) {
    out <- kept[[i]]
    if (length(out) == 1L) out[[1L]] else out
  }), outputs)
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
    list(type = n$type, id = n$id, inputs = n$inputs, params = n$params)
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
    do.call(sd_node, c(list(type = n$type, id = n$id), n$inputs, n$params))
  })
  do.call(sd_pipeline, nodes)
}

# --- print methods ---

#' @export
print.sd_node <- function(x, ...) {
  cat("sd_node:", x$type, if (!is.null(x$id)) paste0("(id: ", x$id, ")"),
      "\n")
  for (nm in names(x$inputs)) cat("  ", nm, "<-", x$inputs[[nm]], "\n")
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
    n <- x$nodes[[i]]
    links <- if (length(n$inputs)) {
      paste0("  <- ", paste(names(n$inputs), unlist(n$inputs), sep = "=",
                            collapse = ", "))
    } else ""
    cat("  [", i, "] ", n$id, ": ", n$type, links, "\n", sep = "")
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
# Optional fields: cache (TRUE: result kept by sd_cache), volatile
# (function(params) -> TRUE if the result is not reproducible: never cached,
# nor anything after it), file_params (params naming files read: size and
# mtime go into the cache key), side_effect (always runs), kinds (inputs
# named differently from their type), explicit (TRUE or input names: no
# implicit binding), needs (c(input = other): input requires other linked),
# check_ctx (function(ctx, params, linked) -> error message or NULL, run
# before anything executes).
#
# run(env, params, inputs) -> named list of output slot values, where
#   env    = list(ctx, upscaler_ctx)
#   inputs = named list of the slots the node reads (optional ones may be NULL)

.slot_types <- c("cond", "uncond", "latent", "image", "mask")
# data type of each slot: an input accepts an output of the same type.
# mask: 1-channel SD image, 0..255 ([0, 1] in R), white = regenerate,
# black = keep; never interchangeable with image.
.slot_kind <- c(cond = "cond", uncond = "cond", latent = "latent",
                image = "image", mask = "mask")

.node_types <- list(
  txt2img = list(
    inputs = character(0), optional = "control", outputs = "image",
    kinds = c(control = "image"), explicit = "control",
    required_params = "prompt",
    cache = TRUE, volatile = function(p) isTRUE((p$seed %||% 42L) < 0),
    check_ctx = function(ctx, params, linked) .check_control_net(ctx, linked),
    run = function(env, params, inputs) {
      # sd_txt2img() has no `verbose`; core logging is set on the context
      res <- do.call(sd_txt2img,
                     c(list(ctx = env$ctx),
                       if (!is.null(inputs$control))
                         list(control_image = inputs$control),
                       params))
      list(image = first_image(res))
    }
  ),
  img2img = list(
    inputs = "image", optional = "mask", outputs = "image",
    required_params = "prompt",
    cache = TRUE, volatile = function(p) isTRUE((p$seed %||% 42L) < 0),
    run = function(env, params, inputs) {
      res <- do.call(sd_img2img,
                     c(list(ctx = env$ctx, init_image = inputs$image,
                            mask = inputs$mask), params))
      list(image = first_image(res))
    }
  ),
  upscale = list(
    inputs = "image", optional = character(0), outputs = "image",
    required_params = character(0), needs_upscaler = TRUE,
    volatile = function(p) TRUE,  # upscaler model has no identity in keys
    run = function(env, params, inputs) {
      factor <- params$factor %||% 4L
      list(image = sd_upscale(env$upscaler_ctx, inputs$image, as.integer(factor)))
    }
  ),
  save = list(
    inputs = "image", optional = character(0), outputs = "image",
    required_params = "path", side_effect = TRUE,  # always runs
    run = function(env, params, inputs) {
      sd_save_image(inputs$image, params$path)
      list(image = inputs$image)
    }
  ),
  remove_background = list(
    inputs = "image", optional = character(0), outputs = c("mask", "image"),
    required_params = "model_path", cache = TRUE, file_params = "model_path",
    run = function(env, params, inputs) {
      model <- .rmbg_cached(params$model_path, params$device)
      output <- match.arg(params$output %||% "rgba", c("rgba", "mask"))
      r <- .rmbg_mask(inputs$image, model, params$threshold)
      mask <- .rmbg_mask_image(r)
      # image last: it is the node's default result
      list(mask = mask,
           image = if (output == "mask") mask else .rmbg_rgba_image(r))
    }
  ),
  # --- masks (see .slot_kind for the convention) ---
  load_mask = list(
    inputs = character(0), optional = character(0), outputs = "mask",
    required_params = "path", file_params = "path",
    run = function(env, params, inputs) {
      list(mask = sd_load_mask(params$path))
    }
  ),
  invert_mask = list(
    inputs = "mask", optional = character(0), outputs = "mask",
    required_params = character(0),
    run = function(env, params, inputs) {
      m <- inputs$mask
      m$data <- as.raw(255L - as.integer(m$data))
      list(mask = m)
    }
  ),
  save_mask = list(
    inputs = "mask", optional = character(0), outputs = "mask",
    required_params = "path", side_effect = TRUE,  # always runs
    run = function(env, params, inputs) {
      sd_save_image(inputs$mask, params$path)
      list(mask = inputs$mask)
    }
  ),
  upscale_latent = list(
    inputs = "latent", optional = character(0), outputs = "latent",
    required_params = character(0),
    run = function(env, params, inputs) {
      list(latent = .op_upscale_latent(env$ctx, inputs$latent, params))
    }
  ),
  # --- ControlNet preprocessors (image -> hint image) ---
  canny = list(
    inputs = "image", optional = character(0), outputs = "image",
    required_params = character(0),
    run = function(env, params, inputs) {
      list(image = do.call(sd_canny, c(list(image = inputs$image), params)))
    }
  ),
  depth = list(
    inputs = "image", optional = character(0), outputs = "image",
    required_params = "model_path", cache = TRUE, file_params = "model_path",
    run = function(env, params, inputs) {
      model <- .depth_cached(params$model_path, params$device,
                             params$size %||% 518L)
      list(image = sd_estimate_depth(inputs$image, model))
    }
  ),
  # --- image operations (pure R; see .op_* helpers) ---
  resize = list(
    inputs = "image", optional = character(0), outputs = "image",
    required_params = character(0),
    run = function(env, params, inputs) {
      list(image = .op_resize(inputs$image, params, "resize"))
    }
  ),
  resize_mask = list(
    inputs = "mask", optional = character(0), outputs = "mask",
    required_params = character(0),
    run = function(env, params, inputs) {
      list(mask = .op_resize(inputs$mask, params, "resize_mask"))
    }
  ),
  crop = list(
    inputs = "image", optional = character(0), outputs = "image",
    required_params = c("width", "height"),
    run = function(env, params, inputs) {
      list(image = .op_crop(inputs$image, params, "crop"))
    }
  ),
  crop_mask = list(
    inputs = "mask", optional = character(0), outputs = "mask",
    required_params = c("width", "height"),
    run = function(env, params, inputs) {
      list(mask = .op_crop(inputs$mask, params, "crop_mask"))
    }
  ),
  composite = list(
    inputs = c("fg", "bg", "mask"), optional = character(0),
    kinds = c(fg = "image", bg = "image", mask = "mask"),
    explicit = TRUE,  # two image inputs: no implicit binding
    outputs = "image", required_params = character(0),
    run = function(env, params, inputs) {
      list(image = .op_composite(inputs$fg, inputs$bg, inputs$mask))
    }
  ),
  # --- low-level nodes (encode -> sample -> decode) ---
  encode_text = list(
    inputs = character(0), optional = character(0),
    outputs = c("cond", "uncond"),
    required_params = "prompt", cache = TRUE,
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
  # --- run-time inputs: the value comes from sd_run_pipeline(inputs = ) ---
  input_image = list(
    inputs = character(0), optional = character(0), outputs = "image",
    required_params = character(0), input_kind = "image",
    run = function(env, params, inputs) {
      list(image = env$input_values[[env$node_id]])
    }
  ),
  input_mask = list(
    inputs = character(0), optional = character(0), outputs = "mask",
    required_params = character(0), input_kind = "mask",
    run = function(env, params, inputs) {
      list(mask = env$input_values[[env$node_id]])
    }
  ),
  input_latent = list(
    inputs = character(0), optional = character(0), outputs = "latent",
    required_params = character(0), input_kind = "latent",
    run = function(env, params, inputs) {
      list(latent = env$input_values[[env$node_id]])
    }
  ),
  load_image = list(
    inputs = character(0), optional = character(0), outputs = "image",
    required_params = "path", file_params = "path",
    run = function(env, params, inputs) {
      list(image = sd_load_image(params$path, params$channels %||% 3L))
    }
  ),
  encode_image = list(
    inputs = "image", optional = character(0), outputs = "latent",
    required_params = character(0), cache = TRUE,
    run = function(env, params, inputs) {
      list(latent = sd_encode_image(env$ctx, inputs$image))
    }
  ),
  sample = list(
    inputs = "cond", optional = c("uncond", "latent", "mask", "control"),
    kinds = c(control = "image"),
    explicit = c("mask", "control"),   # never bound implicitly
    needs = c(mask = "latent"),        # inpainting is img2img
    outputs = "latent",
    required_params = character(0), cache = TRUE,
    # before anything runs: a control input needs a ControlNet in ctx
    check_ctx = function(ctx, params, linked) .check_control_net(ctx, linked),
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
                                       init_latent = init),
                                  # only when linked
                                  if (!is.null(inputs$mask)) list(mask = inputs$mask),
                                  if (!is.null(inputs$control))
                                    list(control_image = inputs$control),
                                  args))
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

# Latent resize for upscale_latent. Only plain latents (4 channels: SD1/2/XL,
# 16: SD3/FLUX.1): their scale/shift is uniform, so resizing commutes with
# it. Others (FLUX.2/3: 128 channels packing 2x2 sub-pixels) are refused.
# Size: scale (latent size * scale, rounded) or width/height in pixels,
# which must be multiples of the VAE scale factor.
.op_upscale_latent <- function(ctx, lat, params) {
  stop_op <- function(...) stop("upscale_latent: ", ..., call. = FALSE)
  ne <- as.integer(lat$ne)
  if (!ne[3] %in% c(4L, 16L)) {
    stop_op(sprintf(paste0("latent with %d channels is not supported (only ",
                           "plain 4- or 16-channel latents; FLUX.2/3 pack ",
                           "2x2 sub-pixels into channels). Upscale the image ",
                           "instead: decode -> resize/upscale -> encode_image"),
                    ne[3]))
  }
  scale <- params$scale; w <- params$width; h <- params$height
  if (!is.null(scale) && (!is.null(w) || !is.null(h))) {
    stop_op("give either 'scale' or 'width'/'height', not both")
  }
  if (!is.null(scale)) {
    if (!is.numeric(scale) || length(scale) != 1L || is.na(scale) || scale <= 0) {
      stop_op("'scale' must be a positive number")
    }
    lw <- max(1L, as.integer(round(ne[1] * scale)))
    lh <- max(1L, as.integer(round(ne[2] * scale)))
  } else if (!is.null(w) && !is.null(h)) {
    vsf <- latent_geometry(ctx, params$vae_scale_factor,
                           params$latent_channels %||% ne[3])[["vae_scale_factor"]]
    for (v in list(list(w, "width"), list(h, "height"))) {
      if (!is.numeric(v[[1]]) || length(v[[1]]) != 1L || v[[1]] <= 0 ||
          v[[1]] %% vsf != 0) {
        stop_op(sprintf("'%s' must be a positive multiple of %d (VAE scale factor)",
                        v[[2]], vsf))
      }
    }
    lw <- as.integer(w %/% vsf); lh <- as.integer(h %/% vsf)
  } else {
    stop_op("give 'scale', or both 'width' and 'height' (pixels)")
  }
  method <- params$method %||% "bilinear"
  if (!method %in% c("bilinear", "nearest")) {
    stop_op("'method' must be \"bilinear\" or \"nearest\"")
  }
  # data is ggml order (x fastest): array [W, H, C]; .resize_array treats
  # dim 1 as height, so target_h = new width, target_w = new height
  arr <- array(as.numeric(lat$data), ne[1:3])
  out <- .resize_array(arr, target_w = lh, target_h = lw, method = method)
  lat$ne <- c(lw, lh, ne[3], if (length(ne) >= 4L) ne[4] else 1L)
  lat$data <- as.vector(out)
  lat
}

# --- image operation helpers (resize / crop / composite nodes) ---

# Target size: width and/or height (a missing side keeps the aspect ratio),
# or scale; result rounded to the nearest integer, at least 1.
.op_resize <- function(img, params, what) {
  stop_op <- function(...) stop(what, ": ", ..., call. = FALSE)
  w0 <- img$width; h0 <- img$height
  w <- params$width; h <- params$height; scale <- params$scale
  if (!is.null(scale) && (!is.null(w) || !is.null(h))) {
    stop_op("give either 'scale' or 'width'/'height', not both")
  }
  pos <- function(v, nm) {
    if (!is.numeric(v) || length(v) != 1L || is.na(v) || v <= 0) {
      stop_op("'", nm, "' must be a positive number")
    }
    v
  }
  if (!is.null(scale)) {
    pos(scale, "scale")
    w <- w0 * scale; h <- h0 * scale
  } else if (!is.null(w) && !is.null(h)) {
    pos(w, "width"); pos(h, "height")
  } else if (!is.null(w)) {
    pos(w, "width"); h <- h0 * w / w0
  } else if (!is.null(h)) {
    pos(h, "height"); w <- w0 * h / h0
  } else {
    stop_op("give 'width', 'height' or 'scale'")
  }
  w <- max(1L, as.integer(round(w))); h <- max(1L, as.integer(round(h)))
  method <- params$method %||% "bilinear"
  if (!method %in% c("bilinear", "nearest")) {
    stop_op("'method' must be \"bilinear\" or \"nearest\"")
  }
  .array_to_sd_image(.resize_array(sd_image_to_array(img), w, h, method))
}

# Crop: x, y are 0-based offsets of the top-left corner; the region must lie
# inside the image (checked here, the input size is known only at run time).
.op_crop <- function(img, params, what) {
  x <- params$x %||% 0L; y <- params$y %||% 0L
  w <- params$width; h <- params$height
  int <- function(v, nm, min) {
    if (!is.numeric(v) || length(v) != 1L || is.na(v) || v != round(v) ||
        v < min) {
      stop(what, ": '", nm, "' must be an integer >= ", min, call. = FALSE)
    }
    as.integer(v)
  }
  x <- int(x, "x", 0); y <- int(y, "y", 0)
  w <- int(w, "width", 1); h <- int(h, "height", 1)
  if (x + w > img$width || y + h > img$height) {
    stop(sprintf(paste0("%s: region x = %d, y = %d, %dx%d is outside the ",
                        "%dx%d input"), what, x, y, w, h, img$width,
                 img$height), call. = FALSE)
  }
  arr <- sd_image_to_array(img)
  .array_to_sd_image(arr[y + seq_len(h), x + seq_len(w), , drop = FALSE])
}

# fg * mask + bg * (1 - mask). fg and bg are used as RGB: a 4th (alpha)
# channel is ignored, 1-channel grayscale is expanded to RGB. All three
# inputs must have the same size. Output: 3-channel image.
.op_composite <- function(fg, bg, mask) {
  sizes <- vapply(list(fg, bg, mask), function(x) {
    sprintf("%dx%d", x$width, x$height)
  }, character(1L))
  if (length(unique(sizes)) != 1L) {
    stop(sprintf(paste0("composite: inputs differ in size (fg %s, bg %s, ",
                        "mask %s); use 'resize' / 'resize_mask' nodes first"),
                 sizes[1], sizes[2], sizes[3]), call. = FALSE)
  }
  rgb <- function(img) {
    a <- sd_image_to_array(img)
    if (dim(a)[3] < 3L) a <- a[, , rep(1L, 3L), drop = FALSE]
    a[, , 1:3, drop = FALSE]
  }
  m <- sd_image_to_array(mask)[, , 1L]
  f <- rgb(fg); b <- rgb(bg)
  .array_to_sd_image(f * as.vector(m) + b * (1 - as.vector(m)))
}

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

# Walks the nodes in order, resolves every input to (source node, output)
# and collects every problem into one error. Returns the links invisibly:
# a list per node, each a named list input -> list(node = <index>, port).
validate_pipeline <- function(nodes, upscaler_ctx = NULL) {
  errs <- character(0)
  ids <- vapply(nodes, function(n) n$id %||% NA_character_, character(1L))
  outputs_of <- function(j) .node_types[[nodes[[j]]$type]]$outputs
  links <- vector("list", length(nodes))
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
    lk <- list()
    # explicit = TRUE: all inputs; a character vector: those inputs only
    explicit <- if (isTRUE(spec$explicit)) c(spec$inputs, spec$optional)
                else as.character(spec$explicit)
    for (s in c(spec$inputs, spec$optional)) {
      ref <- nodes[[i]]$inputs[[s]]
      if (is.null(ref)) {
        if (s %in% explicit) {
          # several inputs of one type (composite): never bound implicitly
          if (s %in% spec$inputs) {
            errs <- c(errs, sprintf(
              "%s: needs '%s', link it explicitly (%s = \"<node id>\")",
              where, s, s))
          }
          next
        }
        # implicit link: nearest earlier node producing this slot
        src <- Find(function(j) s %in% outputs_of(j), rev(seq_len(i - 1L)))
        if (!is.null(src)) {
          lk[[s]] <- list(node = src, port = s)
        } else if (s %in% spec$inputs) {
          errs <- c(errs, sprintf(
            "%s: needs '%s', but no earlier node produces it", where, s))
        }
        next
      }
      res <- resolve_ref(ref, s, input_kind(spec, s), i, ids, outputs_of)
      if (is.character(res)) {
        errs <- c(errs, sprintf("%s: input '%s' = \"%s\": %s",
                                where, s, ref, res))
      } else {
        lk[[s]] <- res
      }
    }
    links[[i]] <- lk
    # an input that only makes sense with another one (sample: mask -> latent)
    for (s in intersect(names(spec$needs), names(lk))) {
      if (is.null(lk[[spec$needs[[s]]]])) {
        errs <- c(errs, sprintf("%s: input '%s' needs input '%s'",
                                where, s, spec$needs[[s]]))
      }
    }
    if (isTRUE(spec$needs_upscaler) && is.null(upscaler_ctx)) {
      errs <- c(errs, sprintf("%s: requires 'upscaler_ctx' in sd_run_pipeline()",
                              where))
    }
  }
  if (length(errs)) {
    stop("Invalid pipeline:\n  ", paste(errs, collapse = "\n  "), call. = FALSE)
  }
  invisible(links)
}

# Data type of input `s` of a node: declared in spec$kinds for inputs named
# differently from their type (composite: fg, bg), else the slot of that name.
input_kind <- function(spec, s) {
  if (s %in% names(spec$kinds)) spec$kinds[[s]] else .slot_kind[[s]]
}

# Run-time inputs (input_* nodes): every declared one needs a value of its
# type, and no value may be given for an unknown id. Checked before running.
.check_run_inputs <- function(nodes, values) {
  kinds <- vapply(nodes, function(n) .node_types[[n$type]]$input_kind %||% "",
                  character(1L))
  ids <- vapply(nodes, function(n) n$id %||% "", character(1L))
  want <- stats::setNames(kinds[nzchar(kinds)], ids[nzchar(kinds)])
  errs <- character(0)
  if (!is.null(values) && (!is.list(values) || is.null(names(values)) ||
                           any(!nzchar(names(values))))) {
    stop("'inputs' must be a named list (names = input node ids)",
         call. = FALSE)
  }
  for (id in setdiff(names(values), names(want))) {
    errs <- c(errs, sprintf("inputs: '%s' is not an input node of the pipeline", id))
  }
  for (id in names(want)) {
    v <- values[[id]]
    if (is.null(v)) {
      errs <- c(errs, sprintf("inputs: no value for input node '%s' (%s)",
                              id, want[[id]]))
      next
    }
    msg <- .check_input_value(v, want[[id]])
    if (!is.null(msg)) errs <- c(errs, sprintf("inputs: '%s': %s", id, msg))
  }
  if (length(errs)) {
    stop("Invalid inputs:\n  ", paste(errs, collapse = "\n  "), call. = FALSE)
  }
  invisible(TRUE)
}

.check_input_value <- function(v, kind) {
  if (kind %in% c("image", "mask")) {
    ok <- is.list(v) && all(c("width", "height", "channel", "data") %in% names(v)) &&
      is.raw(v$data)
    if (!ok) return(sprintf("expected an SD image (%s)", kind))
    if (length(v$data) != v$width * v$height * v$channel) {
      return("data length does not match width * height * channel")
    }
    if (kind == "mask" && v$channel != 1L) return("a mask must have 1 channel")
    if (kind == "image" && !v$channel %in% c(1L, 3L, 4L)) {
      return("an image must have 1, 3 or 4 channels")
    }
  } else if (kind == "latent") {
    ok <- is.list(v) && all(c("ne", "data") %in% names(v)) &&
      length(v$ne) == 4L && is.numeric(v$data)
    if (!ok) return("expected a latent sd_tensor (type, ne[4], data)")
    if (length(v$data) != prod(as.numeric(v$ne))) {
      return("data length does not match prod(ne)")
    }
    if (any(!is.finite(v$data))) return("latent contains non-finite values")
  }
  NULL
}

# Content hashes of run-time input values, by node index, for cache keys
# (same value -> same key, so the nodes after an input stay cacheable).
.input_hashes <- function(nodes, values) {
  out <- character(length(nodes))
  for (i in seq_along(nodes)) {
    if (!is.null(.node_types[[nodes[[i]]$type]]$input_kind)) {
      out[i] <- .value_hash(values[[nodes[[i]]$id]])
    }
  }
  out
}

.value_hash <- function(x) {
  tf <- tempfile()
  on.exit(unlink(tf))
  writeBin(serialize(x, NULL, xdr = FALSE), tf)
  unname(tools::md5sum(tf))
}

# check_ctx of nodes with a `control` input
.check_control_net <- function(ctx, linked) {
  if ("control" %in% linked && !sd_ctx_has_control_net_cpp(ctx)) {
    "input 'control' needs a ControlNet (sd_ctx(control_net_path = ))"
  }
}

# Node requirements on the model context (e.g. a ControlNet), checked before
# any node runs; skipped without a live context (ctx NULL in tests).
check_ctx_requirements <- function(nodes, links, ctx) {
  if (typeof(ctx) != "externalptr") {
    return(invisible(TRUE))
  }
  errs <- character(0)
  for (i in seq_along(nodes)) {
    spec <- .node_types[[nodes[[i]]$type]]
    if (!is.function(spec$check_ctx)) next
    msg <- spec$check_ctx(ctx, nodes[[i]]$params, names(links[[i]]))
    if (!is.null(msg)) {
      errs <- c(errs, sprintf("node %d (%s): %s", i, nodes[[i]]$type, msg))
    }
  }
  if (length(errs)) {
    stop("Invalid pipeline for this context:\n  ",
         paste(errs, collapse = "\n  "), call. = FALSE)
  }
  invisible(TRUE)
}

# Resolves an explicit link "id" / "id$output" for input `slot` (of type
# `kind`) of node `i`.
# Returns list(node, port) or an error message string.
resolve_ref <- function(ref, slot, kind, i, ids, outputs_of) {
  parts <- strsplit(ref, "$", fixed = TRUE)[[1L]]
  if (length(parts) > 2L) return("expected \"id\" or \"id$output\"")
  src <- match(parts[1L], ids)
  if (is.na(src)) {
    hint <- switch(kind,
      mask  = "; inputs are links to nodes, to read a file use a 'load_mask' node",
      image = "; inputs are links to nodes, to read a file use a 'load_image' node",
      "")
    return(sprintf("no node with id '%s'%s", parts[1L], hint))
  }
  if (src >= i) return(sprintf("node '%s' does not run earlier", parts[1L]))
  outs <- outputs_of(src)
  if (length(parts) == 2L) {
    port <- parts[2L]
    if (!port %in% outs) {
      return(sprintf("node '%s' has no output '%s' (outputs: %s)",
                     parts[1L], port, paste(outs, collapse = ", ")))
    }
    if (.slot_kind[[port]] != kind) {
      return(sprintf("output '%s' is %s, input needs %s",
                     port, .slot_kind[[port]], kind))
    }
  } else if (slot %in% outs) {
    port <- slot
  } else {
    cand <- outs[.slot_kind[outs] == kind]
    if (length(cand) != 1L) {
      return(sprintf("node '%s' has %s output of type %s (outputs: %s)%s",
                     parts[1L], if (length(cand)) "more than one" else "no",
                     kind, paste(outs, collapse = ", "),
                     if (length(cand)) "; use \"id$output\"" else ""))
    }
    port <- cand
  }
  list(node = src, port = port)
}

# =========================================================================
# Internal: minimal JSON serialization (no external dependencies)
# =========================================================================

pipeline_to_json <- function(nodes_list) {
  obj <- function(x) {
    if (length(x) == 0L) return("{}")
    pairs <- vapply(names(x), function(nm) {
      paste0("      ", json_quote(nm), ": ", json_value(x[[nm]]))
    }, character(1L))
    paste0("{\n", paste(pairs, collapse = ",\n"), "\n    }")
  }
  node_strings <- vapply(nodes_list, function(n) {
    paste0("  {\n    \"type\": ", json_quote(n$type),
           if (!is.null(n$id)) paste0(",\n    \"id\": ", json_quote(n$id)),
           if (length(n$inputs)) paste0(",\n    \"inputs\": ", obj(n$inputs)),
           ",\n    \"params\": ", obj(n$params), "\n  }")
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
    if (!is.null(n$id) && !is_ref_string(n$id)) {
      bad(sprintf("node %d: \"id\" is not a string", i))
    }
    inputs <- n$inputs %||% list()
    if (!is.list(inputs) || (length(inputs) && is.null(names(inputs))) ||
        !all(vapply(inputs, is_ref_string, logical(1L)))) {
      bad(sprintf("node %d: \"inputs\" is not an object of strings", i))
    }
    list(type = n$type, id = n$id, inputs = inputs, params = params)
  })
}

# =========================================================================
# Internal: run events (on_event / verbose)
# =========================================================================

# Built-in handler behind verbose = TRUE: progress lines via message().
.verbose_handler <- function(e) {
  if (e$event == "start") {
    message("[", e$step, "/", e$total, "] ", e$type)
  } else if (e$event == "cached") {
    message("[", e$step, "/", e$total, "] ", e$type, " (cached)")
  }
}

# Calls a handler; a cancel from it stops the run as a cancel, any other
# error stops it with the handler, node and event named.
.call_handler <- function(h, ev) {
  if (is.null(h)) return(invisible(NULL))
  tryCatch(h(ev),
    error = function(e) {
      if (inherits(e, "sd_pipeline_cancel")) stop(e)
      stop(sprintf("on_event handler failed on node %d '%s' (%s), event '%s': %s",
                   ev$index, ev$id, ev$type, ev$event, conditionMessage(e)),
           call. = FALSE)
    })
  invisible(NULL)
}

# Metadata of node outputs for events (never the values themselves, so an
# event does not keep a result alive).
.outputs_meta <- function(out) {
  lapply(out, function(v) {
    if (is.list(v) && !is.null(v$width) && !is.null(v$channel)) {
      list(class = "sd_image", width = v$width, height = v$height,
           channel = v$channel, bytes = length(v$data))
    } else if (is.list(v) && !is.null(v$ne)) {
      list(class = "sd_tensor", ne = v$ne,
           bytes = as.numeric(utils::object.size(v)))
    } else if (is.list(v) && any(c("crossattn", "vector", "concat") %in% names(v))) {
      parts <- Filter(Negate(is.null), v[c("crossattn", "vector", "concat")])
      list(class = "sd_cond", parts = names(parts),
           ne = lapply(parts, `[[`, "ne"),
           bytes = as.numeric(utils::object.size(v)))
    } else {
      list(class = class(v)[1L], bytes = as.numeric(utils::object.size(v)))
    }
  })
}

#' Cancel a running pipeline
#'
#' Call from an \code{on_event} handler of \code{\link{sd_run_pipeline}} to
#' stop the run. Signals an error of class \code{"sd_pipeline_cancel"}, so a
#' cancel can be told apart from a failure:
#' \code{tryCatch(..., sd_pipeline_cancel = function(c) ...)}.
#'
#' @param message Reason shown in the error.
#' @return Does not return.
#' @export
sd_pipeline_cancel <- function(message = "pipeline cancelled") {
  stop(structure(class = c("sd_pipeline_cancel", "error", "condition"),
                 list(message = message, call = NULL)))
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
