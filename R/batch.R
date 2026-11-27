# Batch runs and loopback over a pipeline graph (pure R, built on
# sd_run_pipeline, run-time inputs and sd_cache).

#' Run a pipeline several times with varying parameters or inputs
#'
#' Runs \code{pipeline} once per combination of the values in \code{vary}.
#' Nodes that do not change between runs are taken from \code{cache}. Every
#' variant is built and validated before the first run, so a typo fails at
#' once, not on run 5 of 20.
#'
#' \code{vary} is a list named by node id. For a parameter node, an element
#' is a named list of parameters, each a vector (or list) of values:
#' \code{list(lat = list(seed = 1:4, cfg_scale = c(5, 7, 7, 9)))}. A varied
#' parameter must already be set on the node (its value there is the type
#' reference and guards against typos). For a run-time input node
#' (\code{input_image}, \code{input_mask}, \code{input_latent}) the element is
#' a list of values: \code{list(src = list(img1, img2))}.
#'
#' Seeds change only if varied explicitly. Side-effect nodes (\code{save},
#' \code{save_mask}) would overwrite their file on every run: give their path
#' a \code{\{i\}} placeholder (replaced by the run number) or vary it.
#'
#' @param pipeline An \code{sd_pipeline}.
#' @param ctx Model context (see \code{\link{sd_run_pipeline}}).
#' @param vary Named list of variations (see Details).
#' @param mode \code{"zip"}: run i takes the i-th value of every variation
#'   (all must have the same length); \code{"grid"}: every combination
#'   (cartesian product, e.g. for an X/Y plot).
#' @param inputs Fixed run-time inputs (named by input node id), used in
#'   every run unless varied.
#' @param outputs,upscaler_ctx,verbose As in \code{\link{sd_run_pipeline}}.
#' @param cache An \code{\link{sd_cache}} shared by the runs (default: a new
#'   one); \code{NULL} disables caching.
#' @param on_error \code{"stop"} (default): the first failing run stops the
#'   batch; \code{"continue"}: the error is recorded and the next run starts.
#' @param max_runs Upper limit on the number of runs (default 100), a guard
#'   against huge grids.
#' @param on_event Event handler as in \code{\link{sd_run_pipeline}}; events
#'   get an extra field \code{run} (run number).
#' @return An object of class \code{"sd_batch"}: \code{results} (one
#'   \code{sd_run_pipeline} result per run, or the error condition),
#'   \code{values} (data frame, one row per run, the varied values; input
#'   variations by index), \code{failed} (logical per run).
#' @examples
#' \dontrun{
#' b <- sd_run_batch(pipe, ctx,
#'   vary = list(lat = list(seed = 1:3, cfg_scale = c(4, 7)),
#'               save_4 = list(path = "out_{i}.png")),
#'   mode = "grid")
#' b$values
#' }
#' @export
sd_run_batch <- function(pipeline, ctx, vary, mode = c("zip", "grid"),
                         inputs = NULL, outputs = NULL, cache = sd_cache(),
                         on_error = c("stop", "continue"), max_runs = 100L,
                         upscaler_ctx = NULL, on_event = NULL,
                         verbose = FALSE) {
  mode <- match.arg(mode)
  on_error <- match.arg(on_error)
  if (!inherits(pipeline, "sd_pipeline")) {
    stop("Expected an sd_pipeline object", call. = FALSE)
  }
  axes <- .vary_axes(pipeline, vary)
  lens <- vapply(axes, function(a) length(a$values), integer(1L))
  if (mode == "zip") {
    if (length(unique(lens)) > 1L) {
      stop("sd_run_batch: in mode \"zip\" all variations need the same length; got ",
           paste(sprintf("%s = %d", names(axes), lens), collapse = ", "),
           call. = FALSE)
    }
    idx <- as.data.frame(lapply(lens, seq_len))
  } else {
    n_grid <- prod(as.numeric(lens))
    if (n_grid > max_runs) {
      stop(sprintf("sd_run_batch: the grid has %s runs, more than max_runs = %d",
                   format(n_grid, big.mark = " "), as.integer(max_runs)),
           call. = FALSE)
    }
    idx <- expand.grid(lapply(lens, seq_len), KEEP.OUT.ATTRS = FALSE)
  }
  names(idx) <- names(axes)
  n <- nrow(idx)
  if (n > max_runs) {
    stop(sprintf("sd_run_batch: %d runs, more than max_runs = %d", n,
                 as.integer(max_runs)), call. = FALSE)
  }

  # build and validate every variant before the first run
  varied <- lapply(axes, function(a) if (is.null(a$param)) NULL else c(a$id, a$param))
  .check_side_effect_paths(pipeline, varied, what = "run")
  plans <- lapply(seq_len(n), function(r) {
    .apply_variant(pipeline, axes, idx[r, , drop = FALSE], inputs, r)
  })
  for (r in seq_len(n)) {
    tryCatch({
      validate_pipeline(plans[[r]]$pipeline$nodes, upscaler_ctx)
      .check_run_inputs(plans[[r]]$pipeline$nodes, plans[[r]]$inputs)
    }, error = function(e) {
      stop(sprintf("sd_run_batch: run %d is invalid: %s", r, conditionMessage(e)),
           call. = FALSE)
    })
  }

  results <- vector("list", n)
  failed <- logical(n)
  for (r in seq_len(n)) {
    ev <- if (is.null(on_event)) NULL else local({
      rr <- r
      function(e) { e$run <- rr; on_event(e) }
    })
    if (verbose) message("-- run ", r, "/", n)
    res <- tryCatch(
      sd_run_pipeline(plans[[r]]$pipeline, ctx, upscaler_ctx = upscaler_ctx,
                      outputs = outputs, cache = cache, on_event = ev,
                      verbose = verbose, inputs = plans[[r]]$inputs),
      error = function(e) {
        if (on_error == "stop" || inherits(e, "sd_pipeline_cancel")) {
          stop(sprintf("sd_run_batch: run %d failed: %s", r, conditionMessage(e)),
               call. = FALSE)
        }
        e
      })
    if (inherits(res, "error")) {
      failed[r] <- TRUE
      results[r] <- list(res)
    } else {
      results[r] <- list(res)
    }
  }
  structure(list(results = results, values = .vary_values(axes, idx),
                 failed = failed), class = "sd_batch")
}

#' @export
print.sd_batch <- function(x, ...) {
  cat(sprintf("sd_batch: %d run(s), %d failed\n", length(x$results), sum(x$failed)))
  print(x$values)
  invisible(x)
}

#' Loopback: feed a pipeline's output back into its input
#'
#' Runs \code{pipeline} \code{n} times; the output \code{from} of iteration
#' i becomes the run-time input \code{to} of iteration i + 1 (the first
#' iteration gets \code{init}). Typical use: repeated img2img with a
#' decaying strength.
#'
#' Parameters change per iteration through \code{vary}: a list named by node
#' id of named lists of vectors of length \code{n}, e.g.
#' \code{list(lat = list(strength = seq(0.6, 0.3, length.out = n)))}. Seeds
#' follow \code{seed_mode} unless varied. Side-effect nodes need a
#' \code{\{i\}} placeholder (iteration number) in their path, or a varied
#' path, so iterations do not overwrite each other.
#'
#' @param pipeline An \code{sd_pipeline} with a run-time input node
#'   (\code{input_image}, \code{input_mask} or \code{input_latent}).
#' @param ctx Model context.
#' @param n Number of iterations (>= 1).
#' @param from Output fed back: \code{"id"} or \code{"id$output"}; its type
#'   must match the input node \code{to}.
#' @param to Id of the run-time input node that receives it.
#' @param init Value of \code{to} for the first iteration.
#' @param vary Per-iteration parameters (vectors of length \code{n}).
#' @param seed_mode \code{"fixed"} (default): seeds stay as set;
#'   \code{"increment"}: every node \code{seed} that is not varied gets
#'   \code{+ (i - 1)} on iteration i.
#' @param keep \code{"last"} (default): return the final value only;
#'   \code{"all"}: a list with the value of every iteration.
#' @param inputs Other fixed run-time inputs.
#' @param cache An \code{\link{sd_cache}} shared by the iterations (default:
#'   a new one; nodes not depending on \code{to} are computed once).
#' @param on_event Event handler; events get an extra field \code{run}
#'   (iteration number).
#' @param upscaler_ctx,verbose As in \code{\link{sd_run_pipeline}}.
#' @return The value of \code{from} after the last iteration, or (with
#'   \code{keep = "all"}) a list of the values of all iterations.
#' @examples
#' \dontrun{
#' pipe <- sd_pipeline(
#'   sd_node("input_image", id = "src"),
#'   sd_node("img2img", id = "out", image = "src", prompt = "a castle"),
#'   sd_node("save", image = "out", path = "loop_{i}.png"))
#' sd_loopback(pipe, ctx, n = 4, from = "out", to = "src",
#'             init = sd_load_image("start.png"),
#'             vary = list(out = list(strength = c(0.6, 0.5, 0.4, 0.3))))
#' }
#' @export
sd_loopback <- function(pipeline, ctx, n, from, to, init, vary = NULL,
                        seed_mode = c("fixed", "increment"),
                        keep = c("last", "all"), inputs = NULL,
                        cache = sd_cache(), on_event = NULL,
                        upscaler_ctx = NULL, verbose = FALSE) {
  seed_mode <- match.arg(seed_mode)
  keep <- match.arg(keep)
  if (!inherits(pipeline, "sd_pipeline")) {
    stop("Expected an sd_pipeline object", call. = FALSE)
  }
  if (!is.numeric(n) || length(n) != 1L || is.na(n) || n < 1 || n != round(n)) {
    stop("sd_loopback: 'n' must be a positive integer", call. = FALSE)
  }
  n <- as.integer(n)
  nodes <- pipeline$nodes
  ids <- vapply(nodes, function(x) x$id, character(1L))

  # `to`: a run-time input node
  ti <- match(to, ids)
  to_kind <- if (!is.na(ti)) .node_types[[nodes[[ti]]$type]]$input_kind
  if (is.na(ti) || is.null(to_kind)) {
    stop("sd_loopback: 'to' must be the id of an input_image / input_mask / ",
         "input_latent node", call. = FALSE)
  }
  # `from`: an output of matching type
  parts <- strsplit(from, "$", fixed = TRUE)[[1L]]
  fi <- match(parts[1L], ids)
  if (is.na(fi)) stop("sd_loopback: no node with id '", parts[1L], "'", call. = FALSE)
  outs <- .node_types[[nodes[[fi]]$type]]$outputs
  port <- if (length(parts) == 2L) parts[2L] else {
    cand <- outs[.slot_kind[outs] == to_kind]
    if (length(cand) != 1L) NA_character_ else cand
  }
  if (is.na(port) || !port %in% outs || .slot_kind[[port]] != to_kind) {
    stop(sprintf("sd_loopback: 'from' = \"%s\" has no %s output to feed '%s'",
                 from, to_kind, to), call. = FALSE)
  }
  msg <- .check_input_value(init, to_kind)
  if (!is.null(msg)) stop("sd_loopback: 'init': ", msg, call. = FALSE)

  axes <- if (length(vary)) .vary_axes(pipeline, vary) else list()
  for (a in axes) {
    if (is.null(a$param)) {
      stop("sd_loopback: vary input nodes through 'to' / 'inputs', not 'vary'",
           call. = FALSE)
    }
    if (length(a$values) != n) {
      stop(sprintf("sd_loopback: vary %s$%s has %d values, need n = %d",
                   a$id, a$param, length(a$values), n), call. = FALSE)
    }
  }
  varied <- lapply(axes, function(a) c(a$id, a$param))
  .check_side_effect_paths(pipeline, varied, what = "iteration")

  # per-iteration pipelines, validated up front
  plans <- lapply(seq_len(n), function(i) {
    # plain named list: as.data.frame() would mangle "id$param" names
    p <- .apply_variant(pipeline, axes,
                        stats::setNames(as.list(rep(i, length(axes))), names(axes)),
                        inputs, i)$pipeline
    if (seed_mode == "increment") {
      for (k in seq_along(p$nodes)) {
        s <- p$nodes[[k]]$params$seed
        is_varied <- any(vapply(varied, function(v) identical(v, c(ids[k], "seed")),
                                logical(1L)))
        if (is.numeric(s) && !is_varied) p$nodes[[k]]$params$seed <- s + (i - 1L)
      }
    }
    p
  })
  for (i in seq_len(n)) {
    tryCatch({
      validate_pipeline(plans[[i]]$nodes, upscaler_ctx)
      .check_run_inputs(plans[[i]]$nodes,
                        c(inputs, stats::setNames(list(init), to)))
    }, error = function(e) {
      stop(sprintf("sd_loopback: iteration %d is invalid: %s", i,
                   conditionMessage(e)), call. = FALSE)
    })
  }

  cur <- init
  all <- if (keep == "all") vector("list", n)
  for (i in seq_len(n)) {
    ev <- if (is.null(on_event)) NULL else local({
      ii <- i
      function(e) { e$run <- ii; on_event(e) }
    })
    if (verbose) message("-- iteration ", i, "/", n)
    res <- sd_run_pipeline(plans[[i]], ctx, upscaler_ctx = upscaler_ctx,
                           outputs = parts[1L], cache = cache, on_event = ev,
                           verbose = verbose,
                           inputs = c(inputs, stats::setNames(list(cur), to)))
    v <- res[[1L]]
    cur <- if (length(outs) > 1L) v[[port]] else v
    if (keep == "all") all[i] <- list(cur)
  }
  if (keep == "all") all else cur
}

# --- helpers ---------------------------------------------------------------

# vary -> list of axes list(id, param (NULL for input nodes), values (list)),
# named "id" or "id$param"; checks ids, params and value types.
.vary_axes <- function(pipeline, vary) {
  if (!is.list(vary) || (length(vary) && (is.null(names(vary)) ||
                                          any(!nzchar(names(vary)))))) {
    stop("'vary' must be a list named by node id", call. = FALSE)
  }
  nodes <- pipeline$nodes
  ids <- vapply(nodes, function(x) x$id, character(1L))
  axes <- list()
  errs <- character(0)
  for (id in names(vary)) {
    k <- match(id, ids)
    if (is.na(k)) {
      errs <- c(errs, sprintf("vary: no node with id '%s'", id))
      next
    }
    spec <- .node_types[[nodes[[k]]$type]]
    v <- vary[[id]]
    if (!is.null(spec$input_kind)) {
      if (!is.list(v) || !is.null(names(v)) || !length(v)) {
        errs <- c(errs, sprintf("vary: '%s' is an input node, give an unnamed list of values", id))
        next
      }
      for (j in seq_along(v)) {
        m <- .check_input_value(v[[j]], spec$input_kind)
        if (!is.null(m)) errs <- c(errs, sprintf("vary: '%s' value %d: %s", id, j, m))
      }
      axes[[id]] <- list(id = id, param = NULL, values = v)
      next
    }
    if (!is.list(v) || is.null(names(v)) || any(!nzchar(names(v))) || !length(v)) {
      errs <- c(errs, sprintf("vary: '%s' must be a named list of parameters", id))
      next
    }
    for (p in names(v)) {
      cur <- nodes[[k]]$params[[p]]
      if (is.null(cur)) {
        errs <- c(errs, sprintf(paste0("vary: node '%s' has no parameter '%s'; set a ",
                                       "default on the node first"), id, p))
        next
      }
      vals <- if (is.list(v[[p]])) v[[p]] else as.list(v[[p]])
      if (!length(vals)) {
        errs <- c(errs, sprintf("vary: %s$%s has no values", id, p))
        next
      }
      same_type <- vapply(vals, function(x) {
        (is.numeric(cur) && is.numeric(x)) || (is.character(cur) && is.character(x)) ||
          (is.logical(cur) && is.logical(x)) || identical(class(cur), class(x))
      }, logical(1L))
      if (!all(same_type)) {
        errs <- c(errs, sprintf("vary: %s$%s values must have the type of the node's value (%s)",
                                id, p, class(cur)[1L]))
        next
      }
      axes[[paste0(id, "$", p)]] <- list(id = id, param = p, values = vals)
    }
  }
  if (length(errs)) stop("Invalid vary:\n  ", paste(errs, collapse = "\n  "), call. = FALSE)
  if (!length(axes)) stop("'vary' is empty", call. = FALSE)
  axes
}

# One variant: pipeline with the chosen parameter values (and {i} filled in
# side-effect paths) plus the run-time inputs.
.apply_variant <- function(pipeline, axes, idx_row, inputs, i) {
  ids <- vapply(pipeline$nodes, function(x) x$id, character(1L))
  run_inputs <- inputs
  for (nm in names(axes)) {
    a <- axes[[nm]]
    val <- a$values[[idx_row[[nm]]]]
    if (is.null(a$param)) {
      run_inputs[[a$id]] <- val
    } else {
      pipeline$nodes[[match(a$id, ids)]]$params[[a$param]] <- val
    }
  }
  for (k in seq_along(pipeline$nodes)) {
    if (isTRUE(.node_types[[pipeline$nodes[[k]]$type]]$side_effect)) {
      p <- pipeline$nodes[[k]]$params$path
      if (is.character(p)) {
        pipeline$nodes[[k]]$params$path <- gsub("{i}", i, p, fixed = TRUE)
      }
    }
  }
  list(pipeline = pipeline, inputs = run_inputs)
}

# Side-effect nodes must not overwrite their file: path with {i} or varied.
.check_side_effect_paths <- function(pipeline, varied, what) {
  errs <- character(0)
  for (nd in pipeline$nodes) {
    if (!isTRUE(.node_types[[nd$type]]$side_effect)) next
    p <- nd$params$path
    is_varied <- any(vapply(varied, function(v) identical(v, c(nd$id, "path")),
                            logical(1L)))
    if (!is_varied && !(is.character(p) && grepl("{i}", p, fixed = TRUE))) {
      errs <- c(errs, sprintf(paste0("node '%s' (%s) would overwrite '%s' on every %s; ",
                                     "put {i} in its path or vary it"),
                              nd$id, nd$type, p %||% "", what))
    }
  }
  if (length(errs)) stop(paste(errs, collapse = "\n"), call. = FALSE)
  invisible(TRUE)
}

# data frame of the varied values per run (input variations by index)
.vary_values <- function(axes, idx) {
  cols <- lapply(names(axes), function(nm) {
    a <- axes[[nm]]
    if (is.null(a$param)) return(idx[[nm]])
    v <- a$values[idx[[nm]]]
    if (all(vapply(v, function(x) is.atomic(x) && length(x) == 1L, logical(1L)))) {
      unlist(v)
    } else {
      idx[[nm]]
    }
  })
  stats::setNames(as.data.frame(cols, stringsAsFactors = FALSE,
                                col.names = names(axes)), names(axes))
}
