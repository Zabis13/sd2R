# Execution cache (sd_cache) for sd_run_pipeline

# Mock node types that log every real run into `log$ran`
mock_types <- function(log) {
  list(
    gen = list(
      inputs = character(0), optional = character(0), outputs = "cond",
      required_params = character(0), cache = TRUE,
      volatile = function(p) isTRUE((p$seed %||% 1L) < 0),
      run = function(env, params, inputs) {
        log$ran <- c(log$ran, paste0("gen:", params$tag))
        list(cond = paste0("c", params$tag, if (isTRUE(params$seed < 0)) runif(1)))
      }
    ),
    step = list(
      inputs = "cond", optional = character(0), outputs = "latent",
      required_params = character(0), cache = TRUE,
      run = function(env, params, inputs) {
        log$ran <- c(log$ran, paste0("step:", params$tag))
        list(latent = paste0(inputs$cond, ">", params$tag))
      }
    ),
    cheap = list(
      inputs = "latent", optional = character(0), outputs = "image",
      required_params = character(0),
      run = function(env, params, inputs) {
        log$ran <- c(log$ran, paste0("cheap:", params$tag))
        list(image = paste0(inputs$latent, "|", params$tag))
      }
    ),
    load = list(
      inputs = character(0), optional = character(0), outputs = "cond",
      required_params = "path", file_params = "path",
      run = function(env, params, inputs) {
        log$ran <- c(log$ran, "load")
        list(cond = readLines(params$path))
      }
    ),
    save = list(
      inputs = "image", optional = character(0), outputs = "image",
      required_params = character(0), side_effect = TRUE,
      run = function(env, params, inputs) {
        log$ran <- c(log$ran, "save")
        list(image = inputs$image)
      }
    )
  )
}

chain <- function(tag_gen = "a", tag_step = "s", tag_cheap = "x", ids = NULL,
                  seed = 1L) {
  ids <- ids %||% c("g", "s", "c")
  sd_pipeline(
    sd_node("gen", id = ids[1], tag = tag_gen, seed = seed),
    sd_node("step", id = ids[2], cond = ids[1], tag = tag_step),
    sd_node("cheap", id = ids[3], latent = ids[2], tag = tag_cheap)
  )
}

test_that("sd_cache: constructor, print, clear, argument checks", {
  cache <- sd_cache(max_entries = 5)
  expect_s3_class(cache, "sd_cache")
  expect_output(print(cache), "0 / 5 entries, 0.0 MB \\| hits: 0, misses: 0")
  expect_error(sd_cache(0), "'max_entries' must be a positive integer")
  expect_error(sd_cache_clear(list()), "Expected an sd_cache object")
  expect_error(sd_run_pipeline(sd_pipeline(sd_node("decode")), NULL,
                               cache = list()),
               "Expected an sd_cache object")
})

test_that("second run takes cached nodes and skips their ancestors", {
  log <- new.env(); log$ran <- character(0)
  local_mocked_bindings(.node_types = mock_types(log))
  cache <- sd_cache()
  out1 <- sd_run_pipeline(chain(), ctx = NULL, cache = cache)
  expect_equal(log$ran, c("gen:a", "step:s", "cheap:x"))
  expect_equal(c(cache$hits, cache$misses), c(0L, 2L))

  log$ran <- character(0)
  out2 <- sd_run_pipeline(chain(), ctx = NULL, cache = cache)
  expect_identical(out2, out1)
  expect_equal(log$ran, "cheap:x")          # step hit -> gen not needed
  expect_equal(cache$hits, 1L)

  # change downstream param: upstream still cached
  log$ran <- character(0)
  expect_equal(sd_run_pipeline(chain(tag_cheap = "y"), ctx = NULL,
                               cache = cache), "ca>s|y")
  expect_equal(log$ran, "cheap:y")

  # change upstream param: everything below reruns
  log$ran <- character(0)
  sd_run_pipeline(chain(tag_gen = "b"), ctx = NULL, cache = cache)
  expect_equal(log$ran, c("gen:b", "step:s", "cheap:x"))

  # without a cache: nothing changes
  log$ran <- character(0)
  sd_run_pipeline(chain(), ctx = NULL)
  expect_equal(log$ran, c("gen:a", "step:s", "cheap:x"))

  sd_cache_clear(cache)
  expect_equal(length(cache$entries), 0L)
  log$ran <- character(0)
  sd_run_pipeline(chain(), ctx = NULL, cache = cache)
  expect_equal(log$ran, c("gen:a", "step:s", "cheap:x"))
})

test_that("keys are by content: renamed ids and other pipelines hit", {
  log <- new.env(); log$ran <- character(0)
  local_mocked_bindings(.node_types = mock_types(log))
  cache <- sd_cache()
  sd_run_pipeline(chain(), ctx = NULL, cache = cache)
  log$ran <- character(0)
  sd_run_pipeline(chain(ids = c("G", "S", "C")), ctx = NULL, cache = cache)
  expect_equal(log$ran, "cheap:x")
  # a different pipeline sharing the gen node, requesting it directly
  log$ran <- character(0)
  other <- sd_pipeline(sd_node("gen", tag = "a", seed = 1L),
                       sd_node("step", tag = "t"))
  res <- sd_run_pipeline(other, ctx = NULL, cache = cache,
                         outputs = c("gen_1", "step_2"))
  expect_equal(log$ran, "step:t")
  expect_equal(res$gen_1, "ca")
})

test_that("volatile nodes and everything after them are not cached", {
  log <- new.env(); log$ran <- character(0)
  local_mocked_bindings(.node_types = mock_types(log))
  cache <- sd_cache()
  o1 <- sd_run_pipeline(chain(seed = -1L), ctx = NULL, cache = cache)
  log$ran <- character(0)
  o2 <- sd_run_pipeline(chain(seed = -1L), ctx = NULL, cache = cache)
  expect_equal(log$ran, c("gen:a", "step:s", "cheap:x"))
  expect_false(identical(o1, o2))
  expect_equal(length(cache$entries), 0L)
})

test_that("save always runs, its inputs come from the cache", {
  log <- new.env(); log$ran <- character(0)
  local_mocked_bindings(.node_types = mock_types(log))
  cache <- sd_cache()
  p <- sd_pipeline(sd_node("gen", tag = "a"), sd_node("step", tag = "s"),
                   sd_node("cheap", tag = "x"), sd_node("save"))
  sd_run_pipeline(p, ctx = NULL, cache = cache)
  log$ran <- character(0)
  sd_run_pipeline(p, ctx = NULL, cache = cache)
  expect_equal(log$ran, c("cheap:x", "save"))
})

test_that("file size / mtime is part of the key", {
  log <- new.env(); log$ran <- character(0)
  local_mocked_bindings(.node_types = mock_types(log))
  tmp <- tempfile(); on.exit(unlink(tmp), add = TRUE)
  writeLines("v1", tmp)
  p <- sd_pipeline(sd_node("load", path = tmp), sd_node("step", tag = "s"))
  cache <- sd_cache()
  expect_equal(sd_run_pipeline(p, ctx = NULL, cache = cache), "v1>s")
  log$ran <- character(0)
  sd_run_pipeline(p, ctx = NULL, cache = cache)
  expect_equal(log$ran, character(0))       # step hit, load not needed
  writeLines("v22", tmp)                     # new size
  log$ran <- character(0)
  expect_equal(sd_run_pipeline(p, ctx = NULL, cache = cache), "v22>s")
  expect_equal(log$ran, c("load", "step:s"))
})

test_that("context generation is part of the key", {
  log <- new.env(); log$ran <- character(0)
  local_mocked_bindings(.node_types = mock_types(log))
  ctx1 <- structure(list(), sd2R_generation = 1)
  ctx2 <- structure(list(), sd2R_generation = 2)
  cache <- sd_cache()
  sd_run_pipeline(chain(), ctx = ctx1, cache = cache)
  log$ran <- character(0)
  sd_run_pipeline(chain(), ctx = ctx1, cache = cache)
  expect_equal(log$ran, "cheap:x")
  log$ran <- character(0)
  sd_run_pipeline(chain(), ctx = ctx2, cache = cache)
  expect_equal(log$ran, c("gen:a", "step:s", "cheap:x"))
  # a context without a generation: nothing cached
  ctx3 <- structure(list(), class = "foreign")
  n <- length(cache$entries)
  sd_run_pipeline(chain(), ctx = ctx3, cache = cache)
  expect_equal(length(cache$entries), n)
})

test_that("LRU keeps at most max_entries results", {
  log <- new.env(); log$ran <- character(0)
  local_mocked_bindings(.node_types = mock_types(log))
  cache <- sd_cache(max_entries = 2)
  run_gen <- function(tag) {
    sd_run_pipeline(sd_pipeline(sd_node("gen", tag = tag)), ctx = NULL,
                    cache = cache)
  }
  run_gen("a"); run_gen("b")
  run_gen("a")                               # touch a -> b is the oldest
  run_gen("c")                               # evicts b
  expect_equal(length(cache$entries), 2L)
  log$ran <- character(0)
  run_gen("a"); run_gen("c")
  expect_equal(log$ran, character(0))
  run_gen("b")
  expect_equal(log$ran, "gen:b")
})
