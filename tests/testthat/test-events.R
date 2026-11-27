# on_event / verbose / sd_pipeline_cancel for sd_run_pipeline

event_types <- function(log) {
  list(
    make = list(
      inputs = character(0), optional = character(0), outputs = "latent",
      required_params = character(0), cache = TRUE,
      run = function(env, params, inputs) {
        if (isTRUE(params$fail)) stop("boom")
        list(latent = list(type = 0L, ne = c(2L, 2L, 1L, 1L), data = rep(0, 4)))
      }
    ),
    show = list(
      inputs = "latent", optional = character(0), outputs = "image",
      required_params = character(0),
      run = function(env, params, inputs) {
        list(image = list(width = 2L, height = 2L, channel = 3L,
                          data = as.raw(rep(0, 12))))
      }
    )
  )
}

collect <- function() {
  log <- new.env(); log$ev <- list()
  log$f <- function(e) log$ev[[length(log$ev) + 1L]] <- e
  log
}

test_that("events: skipped, start/end with metadata, cached", {
  local_mocked_bindings(.node_types = event_types())
  pipe <- sd_pipeline(sd_node("make", id = "m"),
                      sd_node("make", id = "dead", tag = 2),
                      sd_node("show", id = "s", latent = "m"))
  log <- collect()
  sd_run_pipeline(pipe, ctx = NULL, on_event = log$f)
  ev <- vapply(log$ev, function(e) paste(e$event, e$id), "")
  expect_equal(ev, c("skipped dead", "start m", "end m", "start s", "end s"))
  e_end <- log$ev[[3]]
  expect_equal(e_end$schema, 1L)
  expect_equal(c(e_end$index, e_end$step, e_end$total), c(1L, 1L, 2L))
  expect_true(is.numeric(e_end$elapsed) && e_end$elapsed >= 0)
  expect_equal(e_end$outputs$latent$class, "sd_tensor")
  expect_equal(e_end$outputs$latent$ne, c(2L, 2L, 1L, 1L))
  expect_null(e_end$outputs$latent$data)              # metadata only
  expect_equal(log$ev[[5]]$outputs$image[c("class", "width", "channel")],
               list(class = "sd_image", width = 2L, channel = 3L))
  expect_true(is.na(log$ev[[1]]$step))

  cache <- sd_cache()
  sd_run_pipeline(pipe, ctx = NULL, cache = cache)
  log <- collect()
  sd_run_pipeline(pipe, ctx = NULL, cache = cache, on_event = log$f)
  ev <- vapply(log$ev, function(e) paste(e$event, e$id), "")
  expect_equal(ev, c("skipped dead", "cached m", "start s", "end s"))
  expect_equal(log$ev[[2]]$elapsed, 0)
  expect_equal(log$ev[[2]]$outputs$latent$class, "sd_tensor")
})

test_that("node error emits an error event and re-raises", {
  local_mocked_bindings(.node_types = event_types())
  pipe <- sd_pipeline(sd_node("make", id = "m", fail = TRUE),
                      sd_node("show", latent = "m"))
  log <- collect()
  expect_error(sd_run_pipeline(pipe, ctx = NULL, on_event = log$f), "boom")
  ev <- vapply(log$ev, `[[`, "", "event")
  expect_equal(ev, c("start", "error"))
  expect_equal(log$ev[[2]]$error, "boom")
})

test_that("handler errors stop the run; cancel has its own class", {
  local_mocked_bindings(.node_types = event_types())
  pipe <- sd_pipeline(sd_node("make", id = "m"), sd_node("show", id = "s"))
  bad <- function(e) if (e$event == "end" && e$id == "m") stop("oops")
  expect_error(sd_run_pipeline(pipe, ctx = NULL, on_event = bad),
               "on_event handler failed on node 1 'm' \\(make\\), event 'end': oops")
  ran_s <- FALSE
  cancel <- function(e) {
    if (e$event == "start" && e$id == "s") ran_s <<- TRUE
    if (e$event == "end" && e$id == "m") sd_pipeline_cancel("user stop")
  }
  cond <- tryCatch(sd_run_pipeline(pipe, ctx = NULL, on_event = cancel),
                   sd_pipeline_cancel = function(c) c)
  expect_s3_class(cond, "sd_pipeline_cancel")
  expect_equal(conditionMessage(cond), "user stop")
  expect_false(ran_s)
  expect_error(sd_run_pipeline(pipe, ctx = NULL, on_event = 1),
               "'on_event' must be a function or NULL")
})

test_that("handler return value is ignored; verbose uses the same events", {
  local_mocked_bindings(.node_types = event_types())
  pipe <- sd_pipeline(sd_node("make", id = "m"), sd_node("show", id = "s"))
  ref <- sd_run_pipeline(pipe, ctx = NULL)
  expect_identical(sd_run_pipeline(pipe, ctx = NULL,
                                   on_event = function(e) "ignored"), ref)
  expect_message(
    expect_message(sd_run_pipeline(pipe, ctx = NULL, verbose = TRUE),
                   "^\\[1/2\\] make"),
    "^\\[2/2\\] show")
  cache <- sd_cache()
  sd_run_pipeline(pipe, ctx = NULL, cache = cache)
  expect_message(sd_run_pipeline(pipe, ctx = NULL, cache = cache,
                                 verbose = TRUE),
                 "\\[1/2\\] make \\(cached\\)")
  # verbose and on_event together: both see every event
  log <- collect()
  suppressMessages(sd_run_pipeline(pipe, ctx = NULL, verbose = TRUE,
                                   on_event = log$f))
  expect_equal(length(log$ev), 4L)
})
