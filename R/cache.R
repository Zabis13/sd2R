# Execution cache for sd_run_pipeline()

#' Create a pipeline execution cache
#'
#' A cache keeps node results between \code{\link{sd_run_pipeline}} calls:
#' on the next run, a node whose type, parameters and inputs are unchanged
#' returns its stored result without running, and so do not need to run the
#' nodes it depends on. Pass the same cache object to every run; it can be
#' shared between different pipelines.
#'
#' A node's key is computed before anything runs from its type, its
#' parameters, the keys of the nodes linked to its inputs, and the model
#' context (each \code{\link{sd_ctx}} gets its own generation number, so a
#' new context never reuses old results). Node ids are not part of the key:
#' renaming a node or moving it to another pipeline keeps its cached result.
#' For nodes that read a file (\code{load_image}, \code{load_mask}, the
#' \code{remove_background} model), the file's size and modification time
#' are part of the key, so a rewritten file is read again (like
#' \code{make}; changes within the file system's time resolution may go
#' unnoticed).
#'
#' What is cached: the expensive nodes \code{encode_text},
#' \code{encode_image}, \code{sample}, \code{txt2img}, \code{img2img} and
#' \code{remove_background}. Other nodes (\code{decode}, loading, image and
#' mask operations) always run, but their keys still let the nodes after
#' them hit the cache. \code{save} and \code{save_mask} always run (the file
#' must exist after the run), taking their inputs from the cache.
#'
#' Not cached, together with every node after them: \code{txt2img} and
#' \code{img2img} with a random seed (\code{seed < 0}), \code{upscale} (its
#' model is passed separately and has no identity in the key), and all
#' nodes when \code{ctx} was not created by \code{sd_ctx()}.
#'
#' Run-time inputs (\code{input_*} nodes) are keyed by a hash of their
#' content, so the same input value keeps the nodes after it cacheable.
#'
#' Results are kept in R memory (not on the GPU). When more than
#' \code{max_entries} results, or more than \code{max_mb} megabytes, are
#' stored, the least recently used ones are dropped (a single result larger
#' than \code{max_mb} is not kept). Large data (arrays) should reach a node through its inputs, not
#' its parameters: parameters are compared in full on every run.
#'
#' @param max_entries Maximum number of stored node results (LRU eviction).
#'   Default 32.
#' @param max_mb Memory budget in megabytes for the stored results
#'   (estimated with \code{object.size()}); \code{NULL} (default): no limit
#'   besides \code{max_entries}.
#' @return An environment of class \code{"sd_cache"}.
#' @seealso \code{\link{sd_cache_clear}}, \code{\link{sd_run_pipeline}}
#' @examples
#' \dontrun{
#' cache <- sd_cache()
#' sd_run_pipeline(pipe, ctx, cache = cache)   # runs everything
#' pipe$nodes[[4]]$params$path <- "other.png"
#' sd_run_pipeline(pipe, ctx, cache = cache)   # encode/sample from cache
#' }
#' @export
sd_cache <- function(max_entries = 32L, max_mb = NULL) {
  if (!is.numeric(max_entries) || length(max_entries) != 1L ||
      is.na(max_entries) || max_entries < 1) {
    stop("'max_entries' must be a positive integer", call. = FALSE)
  }
  if (!is.null(max_mb) && (!is.numeric(max_mb) || length(max_mb) != 1L ||
                           is.na(max_mb) || max_mb <= 0)) {
    stop("'max_mb' must be a positive number or NULL", call. = FALSE)
  }
  cache <- new.env(parent = emptyenv())
  cache$max_entries <- as.integer(max_entries)
  cache$max_bytes <- if (is.null(max_mb)) Inf else max_mb * 1024^2
  .cache_reset(cache)
  class(cache) <- "sd_cache"
  cache
}

#' Clear a pipeline execution cache
#'
#' Drops all stored results and resets the hit/miss counters.
#'
#' @param cache An \code{sd_cache} object.
#' @return The cache, invisibly.
#' @export
sd_cache_clear <- function(cache) {
  .check_cache(cache)
  .cache_reset(cache)
  invisible(cache)
}

#' @export
print.sd_cache <- function(x, ...) {
  mb <- sum(vapply(x$entries, `[[`, numeric(1L), "bytes")) / 1024^2
  cat(sprintf("sd_cache: %d / %d entries, %.1f%s MB | hits: %d, misses: %d\n",
              length(x$entries), x$max_entries, mb,
              if (is.finite(x$max_bytes)) sprintf(" / %.0f", x$max_bytes / 1024^2) else "",
              x$hits, x$misses))
  invisible(x)
}

.check_cache <- function(cache) {
  if (!inherits(cache, "sd_cache")) {
    stop("Expected an sd_cache object (see sd_cache())", call. = FALSE)
  }
}

.cache_reset <- function(cache) {
  cache$entries <- list()          # key -> list(value, used)
  cache$intern <- new.env(parent = emptyenv())  # key string -> short id
  cache$n_intern <- 0
  cache$tick <- 0
  cache$hits <- 0L
  cache$misses <- 0L
}

# Long key strings are interned to short ids, so a node's key embeds its
# parents' ids instead of their full keys (no growth along a chain).
.cache_intern <- function(cache, key) {
  id <- cache$intern[[key]]
  if (is.null(id)) {
    cache$n_intern <- cache$n_intern + 1
    id <- paste0("k", cache$n_intern)
    assign(key, id, envir = cache$intern)
  }
  id
}

# Context identity for keys: NULL ctx (tests) is fine, a context without a
# generation (not made by sd_ctx) is unknown -> nothing is cacheable.
.ctx_generation <- function(ctx) {
  if (is.null(ctx)) return("none")
  g <- attr(ctx, "sd2R_generation")
  if (is.null(g)) NA_character_ else paste0("g", g)
}

# Keys for all nodes: list(id = interned key per node, stable = logical).
# A node is unstable if it is volatile (random seed, unknown model) or any
# node it reads from is unstable; unstable nodes are never cached.
.cache_keys <- function(cache, nodes, links, ctx, extra = NULL) {
  n <- length(nodes)
  gen <- .ctx_generation(ctx)
  ids <- character(n)
  stable <- logical(n)
  for (i in seq_len(n)) {
    node <- nodes[[i]]
    spec <- .node_types[[node$type]]
    params <- node$params
    if (length(params)) params <- params[order(names(params))]
    files <- lapply(spec$file_params, function(p) {
      f <- params[[p]]
      if (is.character(f) && length(f) == 1L && file.exists(f)) {
        fi <- file.info(f)
        list(f, fi$size, as.numeric(fi$mtime))
      } else {
        f
      }
    })
    lk <- links[[i]]
    ins <- vapply(names(lk), function(s) {
      paste(s, ids[lk[[s]]$node], lk[[s]]$port, sep = ":")
    }, character(1L))
    key <- paste(gen, node$type,
                 rawToChar(serialize(list(params, files), NULL, ascii = TRUE)),
                 paste(ins, collapse = ","),
                 if (!is.null(extra)) extra[i] else "", sep = "\x1f")
    ids[i] <- .cache_intern(cache, key)
    volatile <- is.function(spec$volatile) && isTRUE(spec$volatile(params))
    parents_ok <- all(vapply(lk, function(l) stable[l$node], logical(1L)))
    stable[i] <- !is.na(gen) && !volatile && parents_ok
  }
  list(id = ids, stable = stable)
}

.cache_get <- function(cache, key) {
  e <- cache$entries[[key]]
  if (is.null(e)) return(NULL)
  cache$tick <- cache$tick + 1
  cache$entries[[key]]$used <- cache$tick
  e$value
}

.cache_put <- function(cache, key, value) {
  bytes <- as.numeric(utils::object.size(value))
  if (bytes > cache$max_bytes) return(invisible(NULL))   # never fits
  cache$tick <- cache$tick + 1
  cache$entries[[key]] <- list(value = value, used = cache$tick, bytes = bytes)
  # evict least recently used until both limits hold
  repeat {
    total <- sum(vapply(cache$entries, `[[`, numeric(1L), "bytes"))
    if (length(cache$entries) <= cache$max_entries && total <= cache$max_bytes) break
    used <- vapply(cache$entries, `[[`, numeric(1L), "used")
    cache$entries[[names(which.min(used))]] <- NULL
  }
  invisible(NULL)
}
