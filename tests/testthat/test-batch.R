# Run-time inputs, sd_run_batch, sd_loopback

img_of <- function(v, h = 2L, w = 2L) {
  sd2R:::.array_to_sd_image(array(v, c(h, w, 3L)))
}

# mock types: `step` adds params$add to the red byte of its input image and
# logs every real run
batch_types <- function(log) {
  list(
    input_image = sd2R:::.node_types$input_image,
    step = list(
      inputs = "image", optional = character(0), outputs = "image",
      required_params = character(0), cache = TRUE,
      run = function(env, params, inputs) {
        log$ran <- c(log$ran, paste0("step:", params$add))
        img <- inputs$image
        img$data[1] <- as.raw(min(255L, as.integer(img$data[1]) + params$add))
        list(image = img)
      }
    ),
    gen = list(
      inputs = character(0), optional = character(0), outputs = "image",
      required_params = character(0), cache = TRUE,
      run = function(env, params, inputs) {
        log$ran <- c(log$ran, paste0("gen:", params$seed))
        list(image = img_of(params$seed / 255))
      }
    ),
    save = list(
      inputs = "image", optional = character(0), outputs = "image",
      required_params = "path", side_effect = TRUE,
      run = function(env, params, inputs) {
        log$saved <- c(log$saved, params$path)
        list(image = inputs$image)
      }
    )
  )
}

test_that("run-time inputs: checked before running, hashed for the cache", {
  log <- new.env(); log$ran <- character(0)
  local_mocked_bindings(.node_types = batch_types(log))
  pipe <- sd_pipeline(sd_node("input_image", id = "src"),
                      sd_node("step", id = "s", image = "src", add = 1))
  expect_error(sd_run_pipeline(pipe, NULL),
               "no value for input node 'src' \\(image\\)")
  expect_error(sd_run_pipeline(pipe, NULL, inputs = list(src = img_of(0), x = 1)),
               "'x' is not an input node")
  expect_error(sd_run_pipeline(pipe, NULL, inputs = list(src = list(a = 1))),
               "'src': expected an SD image")
  m <- sd2R:::.array_to_sd_image(array(0, c(2, 2, 1)))
  expect_error(sd_run_pipeline(pipe, NULL, inputs = list(src = img_of(0)[-4])),
               "expected an SD image")
  expect_null(sd2R:::.check_input_value(m, "mask"))
  expect_match(sd2R:::.check_input_value(img_of(0), "mask"), "1 channel")
  expect_equal(log$ran, character(0))

  cache <- sd_cache()
  out <- sd_run_pipeline(pipe, NULL, cache = cache, inputs = list(src = img_of(0)))
  expect_equal(as.integer(out$data[1]), 1L)
  log$ran <- character(0)
  sd_run_pipeline(pipe, NULL, cache = cache, inputs = list(src = img_of(0)))
  expect_equal(log$ran, character(0))                     # same content: hit
  sd_run_pipeline(pipe, NULL, cache = cache, inputs = list(src = img_of(1)))
  expect_equal(log$ran, "step:1")                         # new content: miss
})

test_that("sd_cache max_mb evicts by size", {
  cache <- sd_cache(max_entries = 100, max_mb = 1)
  big <- numeric(70000)                                    # ~0.53 MB
  sd2R:::.cache_put(cache, "a", big)
  sd2R:::.cache_put(cache, "b", big)                      # over 1 MB -> drop a
  expect_equal(names(cache$entries), "b")
  sd2R:::.cache_put(cache, "huge", numeric(200000))       # never fits
  expect_equal(names(cache$entries), "b")
  expect_error(sd_cache(max_mb = 0), "'max_mb' must be a positive number")
  expect_output(print(cache), "1 / 100 entries, 0.5 / 1 MB")
})

test_that("sd_run_batch: zip / grid, cache reuse, values table", {
  log <- new.env(); log$ran <- character(0); log$saved <- character(0)
  local_mocked_bindings(.node_types = batch_types(log))
  pipe <- sd_pipeline(sd_node("gen", id = "g", seed = 10),
                      sd_node("step", id = "s", image = "g", add = 1),
                      sd_node("save", id = "out", image = "s", path = "o_{i}.png"))
  b <- sd_run_batch(pipe, NULL, vary = list(s = list(add = c(1, 2, 3))),
                    outputs = "s")
  expect_s3_class(b, "sd_batch")
  expect_equal(b$values, data.frame(`s$add` = c(1, 2, 3), check.names = FALSE))
  expect_equal(vapply(b$results, function(r) as.integer(r$s$data[1]), 1L),
               c(11L, 12L, 13L))
  expect_equal(sum(log$ran == "gen:10"), 1L)               # gen from the cache
  expect_equal(log$saved, c("o_1.png", "o_2.png", "o_3.png"))

  log$ran <- character(0)
  g <- sd_run_batch(pipe, NULL, mode = "grid",
                    vary = list(g = list(seed = c(1, 2)), s = list(add = c(5, 6, 7))))
  expect_equal(nrow(g$values), 6L)
  expect_equal(g$values[["g$seed"]], c(1, 2, 1, 2, 1, 2))
  expect_equal(g$values[["s$add"]], c(5, 5, 6, 6, 7, 7))
  expect_error(sd_run_batch(pipe, NULL, mode = "grid", max_runs = 5,
                            vary = list(g = list(seed = 1:3), s = list(add = 1:2))),
               "the grid has 6 runs, more than max_runs = 5")
})

test_that("sd_run_batch validates everything before the first run", {
  log <- new.env(); log$ran <- character(0); log$saved <- character(0)
  local_mocked_bindings(.node_types = batch_types(log))
  pipe <- sd_pipeline(sd_node("gen", id = "g", seed = 10),
                      sd_node("step", id = "s", image = "g", add = 1),
                      sd_node("save", id = "out", image = "s", path = "o_{i}.png"))
  expect_error(sd_run_batch(pipe, NULL, vary = list(zz = list(add = 1))),
               "no node with id 'zz'")
  expect_error(sd_run_batch(pipe, NULL, vary = list(s = list(ad = 1:2))),
               "node 's' has no parameter 'ad'; set a default")
  expect_error(sd_run_batch(pipe, NULL, vary = list(s = list(add = c("a", "b")))),
               "must have the type of the node's value \\(numeric\\)")
  expect_error(sd_run_batch(pipe, NULL,
                            vary = list(g = list(seed = 1:2), s = list(add = 1:3))),
               "same length; got g\\$seed = 2, s\\$add = 3")
  nosave <- sd_pipeline(sd_node("gen", id = "g", seed = 1),
                        sd_node("save", id = "out", image = "g", path = "fixed.png"))
  expect_error(sd_run_batch(nosave, NULL, vary = list(g = list(seed = 1:2))),
               "node 'out' \\(save\\) would overwrite 'fixed.png' on every run")
  # varying the path is fine
  ok <- sd_run_batch(nosave, NULL, vary = list(g = list(seed = 1:2),
                                               out = list(path = c("a.png", "b.png"))))
  expect_equal(log$saved, c("a.png", "b.png"))
  expect_equal(log$ran, c("gen:1", "gen:2"))
})

test_that("sd_run_batch: on_error policy, run index in events, input values", {
  log <- new.env(); log$ran <- character(0)
  types <- batch_types(log)
  types$step$run <- function(env, params, inputs) {
    if (params$add == 2) stop("bad add")
    list(image = inputs$image)
  }
  local_mocked_bindings(.node_types = types)
  pipe <- sd_pipeline(sd_node("input_image", id = "src"),
                      sd_node("step", id = "s", image = "src", add = 1))
  expect_error(sd_run_batch(pipe, NULL, vary = list(s = list(add = 1:3)),
                            inputs = list(src = img_of(0))),
               "run 2 failed: bad add")
  runs <- integer(0)
  b <- sd_run_batch(pipe, NULL, vary = list(s = list(add = 1:3)),
                    inputs = list(src = img_of(0)), on_error = "continue",
                    on_event = function(e) runs <<- c(runs, e$run))
  expect_equal(b$failed, c(FALSE, TRUE, FALSE))
  expect_s3_class(b$results[[2]], "error")
  expect_equal(unique(runs), 1:3)
  # vary over input values
  b2 <- sd_run_batch(pipe, NULL, vary = list(src = list(img_of(0), img_of(1))))
  expect_equal(b2$values$src, 1:2)
  expect_error(sd_run_batch(pipe, NULL, vary = list(src = list(list(a = 1)))),
               "'src' value 1: expected an SD image")
})

test_that("sd_loopback feeds the output back; keep, vary, seeds, paths", {
  log <- new.env(); log$ran <- character(0); log$saved <- character(0)
  local_mocked_bindings(.node_types = batch_types(log))
  pipe <- sd_pipeline(sd_node("input_image", id = "src"),
                      sd_node("step", id = "s", image = "src", add = 1, seed = 5),
                      sd_node("save", image = "s", path = "loop_{i}.png"))
  out <- sd_loopback(pipe, NULL, n = 3, from = "s", to = "src", init = img_of(0))
  expect_equal(as.integer(out$data[1]), 3L)                # 0 + 1 + 1 + 1
  expect_equal(log$saved, c("loop_1.png", "loop_2.png", "loop_3.png"))
  all <- sd_loopback(pipe, NULL, n = 3, from = "s", to = "src", init = img_of(0),
                     vary = list(s = list(add = c(10, 20, 30))), keep = "all")
  expect_equal(vapply(all, function(v) as.integer(v$data[1]), 1L), c(10L, 30L, 60L))
  expect_error(sd_loopback(pipe, NULL, n = 3, from = "s", to = "src",
                           init = img_of(0), vary = list(s = list(add = 1:2))),
               "has 2 values, need n = 3")
  expect_error(sd_loopback(pipe, NULL, n = 2, from = "s", to = "s", init = img_of(0)),
               "'to' must be the id of an input_image")
  expect_error(sd_loopback(pipe, NULL, n = 2, from = "zz", to = "src", init = img_of(0)),
               "no node with id 'zz'")
  fixed <- sd_pipeline(sd_node("input_image", id = "src"),
                       sd_node("step", id = "s", image = "src", add = 1),
                       sd_node("save", image = "s", path = "same.png"))
  expect_error(sd_loopback(fixed, NULL, n = 2, from = "s", to = "src", init = img_of(0)),
               "would overwrite 'same.png' on every iteration")
  # seed_mode = "increment": seeds 5, 6, 7 on the three iterations
  seen <- integer(0)
  local_mocked_bindings(.node_types = within(batch_types(log), {
    step$run <- function(env, params, inputs) {
      seen <<- c(seen, params$seed); list(image = inputs$image)
    }
  }))
  sd_loopback(pipe, NULL, n = 3, from = "s", to = "src", init = img_of(0),
              seed_mode = "increment", cache = NULL)
  expect_equal(seen, c(5, 6, 7))
})
