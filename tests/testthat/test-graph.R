# Tests for graph-based pipeline (R/graph.R)

# --- sd_node ---

test_that("sd_node creates valid node", {
  n <- sd_node("txt2img", prompt = "a cat", width = 512L)
  expect_s3_class(n, "sd_node")
  expect_equal(n$type, "txt2img")
  expect_equal(n$params$prompt, "a cat")
  expect_equal(n$params$width, 512L)
})

test_that("sd_node rejects unknown type", {
  expect_error(sd_node("unknown"), "Unknown node type")
})

test_that("sd_node accepts all valid types", {
  for (type in c("txt2img", "img2img", "upscale", "save")) {
    n <- sd_node(type)
    expect_equal(n$type, type)
  }
})

test_that("sd_node with no params has empty params list", {
  n <- sd_node("txt2img")
  expect_equal(length(n$params), 0L)
})

# --- sd_pipeline ---

test_that("sd_pipeline creates pipeline from nodes", {
  pipe <- sd_pipeline(
    sd_node("txt2img", prompt = "test"),
    sd_node("save", path = "out.png")
  )
  expect_s3_class(pipe, "sd_pipeline")
  expect_equal(length(pipe$nodes), 2L)
  expect_equal(pipe$nodes[[1]]$type, "txt2img")
  expect_equal(pipe$nodes[[2]]$type, "save")
})

test_that("sd_pipeline rejects non-node arguments", {
  expect_error(sd_pipeline("not a node"), "not an sd_node")
})

test_that("sd_pipeline rejects empty input", {
  expect_error(sd_pipeline(), "at least one node")
})

# --- print methods ---

test_that("print.sd_node works", {
  n <- sd_node("txt2img", prompt = "cat", width = 512L)
  out <- capture.output(print(n))
  expect_true(any(grepl("txt2img", out)))
  expect_true(any(grepl("prompt", out)))
})

test_that("print.sd_pipeline works", {
  pipe <- sd_pipeline(sd_node("txt2img"), sd_node("save", path = "x.png"))
  out <- capture.output(print(pipe))
  expect_true(any(grepl("2 node", out)))
})

# --- JSON serialization ---

test_that("sd_save_pipeline and sd_load_pipeline roundtrip", {
  pipe <- sd_pipeline(
    sd_node("txt2img", prompt = "a dog", width = 768L, height = 768L,
            cfg_scale = 7.5),
    sd_node("upscale", factor = 2L),
    sd_node("img2img", strength = 0.3),
    sd_node("save", path = "result.png")
  )

  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp))

  sd_save_pipeline(pipe, tmp)
  expect_true(file.exists(tmp))

  pipe2 <- sd_load_pipeline(tmp)
  expect_s3_class(pipe2, "sd_pipeline")
  expect_equal(length(pipe2$nodes), 4L)

  # Check types preserved
  expect_equal(pipe2$nodes[[1]]$type, "txt2img")
  expect_equal(pipe2$nodes[[2]]$type, "upscale")
  expect_equal(pipe2$nodes[[3]]$type, "img2img")
  expect_equal(pipe2$nodes[[4]]$type, "save")

  # Check params preserved
  expect_equal(pipe2$nodes[[1]]$params$prompt, "a dog")
  expect_equal(pipe2$nodes[[1]]$params$width, 768)
  expect_equal(pipe2$nodes[[1]]$params$height, 768)
  expect_equal(pipe2$nodes[[1]]$params$cfg_scale, 7.5)
  expect_equal(pipe2$nodes[[2]]$params$factor, 2)
  expect_equal(pipe2$nodes[[3]]$params$strength, 0.3)
  expect_equal(pipe2$nodes[[4]]$params$path, "result.png")
})

test_that("sd_save_pipeline rejects non-pipeline", {
  expect_error(sd_save_pipeline("not a pipeline", "x.json"),
               "Expected an sd_pipeline")
})

test_that("JSON file is human-readable", {
  pipe <- sd_pipeline(sd_node("txt2img", prompt = "test"))
  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp))

  sd_save_pipeline(pipe, tmp)
  json <- readLines(tmp)
  # Should contain type and prompt as readable strings
  expect_true(any(grepl("txt2img", json)))
  expect_true(any(grepl("test", json)))
})

test_that("JSON roundtrip keeps special characters in strings", {
  prompt <- "a: {cat} [dog], \"quoted\"\nnew line\ttab"
  path <- "C:\\images\\out (1).png"
  pipe <- sd_pipeline(sd_node("txt2img", prompt = prompt),
                      sd_node("save", path = path))
  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp))
  sd_save_pipeline(pipe, tmp)
  pipe2 <- sd_load_pipeline(tmp)
  expect_identical(pipe2$nodes[[1]]$params$prompt, prompt)
  expect_identical(pipe2$nodes[[2]]$params$path, path)
})

test_that("JSON roundtrip keeps types and vectors", {
  pipe <- sd_pipeline(sd_node("txt2img", prompt = "x", width = 768L,
                              cfg_scale = 7.5, seed = -1L, flag = TRUE,
                              sizes = c(1L, 2L, 3L), weights = c(0.5, 1.25)))
  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp))
  sd_save_pipeline(pipe, tmp)
  p <- sd_load_pipeline(tmp)$nodes[[1]]$params
  expect_identical(p$width, 768L)
  expect_identical(p$seed, -1L)
  expect_identical(p$cfg_scale, 7.5)
  expect_identical(p$flag, TRUE)
  expect_identical(p$sizes, c(1L, 2L, 3L))
  expect_identical(p$weights, c(0.5, 1.25))
})

test_that("pipeline_from_json never evaluates code", {
  marker <- tempfile()
  on.exit(unlink(marker))
  evil <- sprintf('[{"type": file.create("%s"), "params": {}}]',
                  gsub("\\\\", "/", marker))
  expect_error(sd2R:::pipeline_from_json(evil), "Invalid pipeline JSON")
  expect_false(file.exists(marker))
})

test_that("pipeline_from_json rejects malformed JSON", {
  f <- sd2R:::pipeline_from_json
  expect_error(f('[{"type": "txt2img", "params": {}}'), "Invalid pipeline JSON")
  expect_error(f('[{"type": "txt2img"}] extra'), "trailing characters")
  expect_error(f('[{"type": "a\\q"}]'), "bad escape")
  expect_error(f('{"type": "txt2img"}'), "expected an array")
  expect_error(f('[{"params": {}}]'), "no \"type\"")
  expect_error(f('[{"type": "txt2img", "params": [1, 2]}]'), "not an object")
})

test_that("pipeline_from_json parses escapes and empty params", {
  res <- sd2R:::pipeline_from_json(
    '[ {"type":"txt2img","params":{"prompt":"caf\\u00e9 \\/ ok","n":null}},
       {"type":"save"} ]')
  expect_equal(res[[1]]$params$prompt, "caf\u00e9 / ok")
  expect_true("n" %in% names(res[[1]]$params))
  expect_null(res[[1]]$params$n)
  expect_equal(res[[2]]$params, list())
})

# --- sd_run_pipeline validation ---

test_that("sd_run_pipeline rejects non-pipeline", {
  expect_error(sd_run_pipeline("not a pipeline", NULL),
               "Expected an sd_pipeline")
})

test_that("sd_run_pipeline img2img without input errors", {
  pipe <- sd_pipeline(sd_node("img2img", prompt = "x", strength = 0.5))
  expect_error(sd_run_pipeline(pipe, ctx = NULL),
               "node 1 \\(img2img\\): needs 'image'")
})

test_that("sd_run_pipeline upscale without ctx errors", {
  pipe <- sd_pipeline(sd_node("txt2img", prompt = "x"),
                      sd_node("upscale", factor = 2L))
  expect_error(sd_run_pipeline(pipe, ctx = NULL),
               "node 2 \\(upscale\\): requires 'upscaler_ctx'")
})

test_that("sd_run_pipeline upscale without image errors", {
  pipe <- sd_pipeline(sd_node("upscale", factor = 2L))
  expect_error(sd_run_pipeline(pipe, ctx = NULL, upscaler_ctx = "fake_ctx"),
               "node 1 \\(upscale\\): needs 'image'")
})

test_that("sd_run_pipeline save without image errors", {
  pipe <- sd_pipeline(sd_node("save", path = "x.png"))
  expect_error(sd_run_pipeline(pipe, ctx = NULL),
               "node 1 \\(save\\): needs 'image'")
})

test_that("sd_run_pipeline save requires path param", {
  pipe <- sd_pipeline(sd_node("txt2img", prompt = "x"), sd_node("save"))
  expect_error(sd_run_pipeline(pipe, ctx = NULL),
               "node 2 \\(save\\): 'path' is not set")
})

# --- sd_run_pipeline: arguments passed to the generation functions ---

test_that("sd_run_pipeline passes valid arguments to txt2img / img2img", {
  fake_img <- list(width = 8L, height = 8L, channel = 3L,
                   data = raw(8L * 8L * 3L))
  seen <- list()
  local_mocked_bindings(
    sd_txt2img = function(ctx, prompt, ...) {
      seen$txt2img <<- list(prompt = prompt, extra = list(...))
      list(fake_img)
    },
    sd_img2img = function(ctx, prompt, init_image, ...) {
      seen$img2img <<- list(prompt = prompt, init_image = init_image,
                            extra = list(...))
      list(fake_img)
    }
  )
  pipe <- sd_pipeline(
    sd_node("txt2img", prompt = "a cat", width = 8L, height = 8L),
    sd_node("img2img", prompt = "a cat, detailed", strength = 0.3)
  )
  out <- sd_run_pipeline(pipe, ctx = NULL, verbose = FALSE)

  expect_identical(out, fake_img)
  expect_equal(seen$txt2img$prompt, "a cat")
  expect_false("verbose" %in% names(seen$txt2img$extra))
  expect_identical(seen$img2img$init_image, fake_img)
  expect_false("image" %in% names(seen$img2img$extra))
  expect_false("verbose" %in% names(seen$img2img$extra))
  expect_equal(seen$img2img$extra$strength, 0.3)
})

# --- sd_run_pipeline: validation before any node runs ---

test_that("sd_run_pipeline rejects img2img without its own prompt", {
  called <- FALSE
  local_mocked_bindings(
    sd_txt2img = function(...) { called <<- TRUE; list(list(width = 1L)) }
  )
  pipe <- sd_pipeline(
    sd_node("txt2img", prompt = "a cat"),
    sd_node("img2img", strength = 0.3)
  )
  expect_error(sd_run_pipeline(pipe, ctx = NULL),
               "node 2 \\(img2img\\): 'prompt' is not set")
  expect_false(called)  # nothing executed
})

test_that("sd_run_pipeline reports all validation errors at once", {
  pipe <- sd_pipeline(
    sd_node("upscale", factor = 2L),
    sd_node("txt2img", prompt = ""),
    sd_node("save")
  )
  err <- tryCatch(sd_run_pipeline(pipe, ctx = NULL), error = conditionMessage)
  expect_match(err, "node 1 \\(upscale\\): needs 'image'")
  expect_match(err, "node 1 \\(upscale\\): requires 'upscaler_ctx'")
  expect_match(err, "node 2 \\(txt2img\\): 'prompt' is not set")
  expect_match(err, "node 3 \\(save\\): 'path' is not set")
})

# --- typed slots / node registry ---

test_that("every registered node type is well-formed", {
  types <- sd2R:::.node_types
  slots <- sd2R:::.slot_types
  expect_setequal(names(types), c("txt2img", "img2img", "upscale", "save",
                                  "encode_text", "load_image", "encode_image",
                                  "sample", "decode"))
  for (nm in names(types)) {
    spec <- types[[nm]]
    expect_true(is.function(spec$run), info = nm)
    expect_true(all(c(spec$inputs, spec$optional, spec$outputs) %in% slots),
                info = nm)
    expect_true(length(spec$outputs) >= 1L, info = nm)
  }
})

test_that("sd_node accepts exactly the registered types", {
  for (type in names(sd2R:::.node_types)) {
    expect_equal(sd_node(type)$type, type)
  }
  expect_error(sd_node("bogus"), "Must be one of: txt2img, img2img")
})

test_that("validation tracks slots written by earlier nodes", {
  ok <- sd_pipeline(
    sd_node("txt2img", prompt = "a"),
    sd_node("img2img", prompt = "b"),
    sd_node("save", path = "x.png")
  )
  expect_invisible(sd2R:::validate_pipeline(ok$nodes))

  bad <- sd_pipeline(sd_node("save", path = "x.png"))
  expect_error(sd2R:::validate_pipeline(bad$nodes),
               "node 1 \\(save\\): needs 'image', but no earlier node produces it")
})

test_that("runner passes slots between custom typed nodes", {
  seen <- NULL
  local_mocked_bindings(.node_types = list(
    make_cond = list(
      inputs = character(0), optional = character(0), outputs = "cond",
      required_params = character(0),
      run = function(env, params, inputs) list(cond = "C")
    ),
    make_latent = list(
      inputs = character(0), optional = character(0), outputs = "latent",
      required_params = character(0),
      run = function(env, params, inputs) list(latent = "L")
    ),
    use_both = list(
      inputs = c("cond", "latent"), optional = "uncond", outputs = "image",
      required_params = character(0),
      run = function(env, params, inputs) {
        seen <<- inputs
        list(image = paste(inputs$cond, inputs$latent))
      }
    )
  ))
  pipe <- sd_pipeline(sd_node("make_cond"), sd_node("make_latent"),
                      sd_node("use_both"))
  out <- sd_run_pipeline(pipe, ctx = NULL)

  expect_equal(out, "C L")
  expect_equal(seen$cond, "C")
  expect_equal(seen$latent, "L")
  expect_true("uncond" %in% names(seen))   # optional slot passed (NULL)
  expect_null(seen$uncond)

  # missing required slot is caught before anything runs
  expect_error(sd_run_pipeline(sd_pipeline(sd_node("use_both")), ctx = NULL),
               "needs 'cond'.*\n.*needs 'latent'")
})

test_that("runner rejects a node that returns undeclared slots", {
  local_mocked_bindings(.node_types = list(
    liar = list(
      inputs = character(0), optional = character(0), outputs = "image",
      required_params = character(0),
      run = function(env, params, inputs) list(latent = 1)
    )
  ))
  expect_error(sd_run_pipeline(sd_pipeline(sd_node("liar")), ctx = NULL),
               "returned slots \\[latent\\], declared \\[image\\]")
})

# --- low-level nodes (mocked, no model) ---

test_that("encode_text -> sample -> decode passes slots and latent shape", {
  seen <- list()
  local_mocked_bindings(
    sd_ctx_latent_info_cpp = function(ctx) c(8L, 16L),
    sd_encode_text = function(ctx, prompt, clip_skip, width, height) {
      seen$enc <<- c(seen$enc, prompt)
      list(text = prompt)
    },
    sd_sample = function(ctx, cond, uncond, latent_shape, init_latent, ...) {
      seen$sample <<- list(cond = cond, uncond = uncond, shape = latent_shape,
                           init = init_latent, extra = list(...))
      list(ne = c(latent_shape, 1L))
    },
    sd_decode_latent = function(ctx, latent) {
      seen$decode <<- latent
      list(width = 1L)
    }
  )
  pipe <- sd_pipeline(
    sd_node("encode_text", prompt = "cat", negative_prompt = "blurry"),
    sd_node("sample", width = 768L, height = 512L, sample_steps = 4L,
            cfg_scale = 3.5),
    sd_node("decode")
  )
  out <- sd_run_pipeline(pipe, ctx = "fake_ctx")

  expect_equal(seen$enc, c("cat", "blurry"))
  expect_equal(seen$sample$cond, list(text = "cat"))
  expect_equal(seen$sample$uncond, list(text = "blurry"))
  expect_equal(seen$sample$shape, c(96L, 64L, 16L))   # 768/8, 512/8, 16 ch
  expect_null(seen$sample$init)
  expect_equal(seen$sample$extra, list(sample_steps = 4L, cfg_scale = 3.5))
  expect_equal(seen$decode$ne, c(96L, 64L, 16L, 1L))
  expect_equal(out, list(width = 1L))
})

test_that("encode_text without negative_prompt encodes empty uncond", {
  enc <- character(0)
  local_mocked_bindings(
    sd_encode_text = function(ctx, prompt, ...) {
      enc <<- c(enc, prompt)
      list(text = prompt)
    }
  )
  sd_run_pipeline(sd_pipeline(sd_node("encode_text", prompt = "cat")),
                  ctx = "fake_ctx")
  expect_equal(enc, c("cat", ""))
})

test_that("sample geometry overrides take precedence over the model", {
  shapes <- list()
  local_mocked_bindings(
    sd_ctx_latent_info_cpp = function(ctx) c(8L, 4L),
    sd_encode_text = function(...) list(),
    sd_sample = function(ctx, cond, uncond, latent_shape, ...) {
      shapes[[length(shapes) + 1L]] <<- latent_shape
      list()
    }
  )
  run <- function(...) {
    sd_run_pipeline(sd_pipeline(sd_node("encode_text", prompt = "x"),
                                sd_node("sample", ...)), ctx = "fake_ctx")
  }
  run(width = 512L, height = 512L, latent_channels = 16L)
  run(width = 512L, height = 512L, vae_scale_factor = 16L)
  run(latent_shape = c(10L, 20L, 3L))
  run()                                   # defaults 512x512
  expect_equal(shapes, list(c(64L, 64L, 16L), c(32L, 32L, 4L),
                            c(10L, 20L, 3L), c(64L, 64L, 4L)))
})

test_that("sample without ctx geometry asks for explicit values", {
  local_mocked_bindings(sd_encode_text = function(...) list())
  pipe <- sd_pipeline(sd_node("encode_text", prompt = "x"), sd_node("sample"))
  expect_error(sd_run_pipeline(pipe, ctx = NULL),
               "pass 'latent_channels' and 'vae_scale_factor'")
})

test_that("load_image -> encode_image -> sample uses input latent shape", {
  seen <- list()
  local_mocked_bindings(
    sd_ctx_latent_info_cpp = function(ctx) stop("must not be queried"),
    sd_load_image = function(path, channels) {
      seen$load <<- c(path, channels)
      list(width = 64L)
    },
    sd_encode_image = function(ctx, image) list(ne = c(8L, 8L, 4L, 1L)),
    sd_encode_text = function(...) list(),
    sd_sample = function(ctx, cond, uncond, latent_shape, init_latent, ...) {
      seen$shape <<- latent_shape
      seen$init <<- init_latent
      seen$extra <<- list(...)
      init_latent
    },
    sd_decode_latent = function(ctx, latent) list(width = 64L)
  )
  pipe <- sd_pipeline(
    sd_node("load_image", path = "in.png"),
    sd_node("encode_image"),
    sd_node("encode_text", prompt = "x"),
    sd_node("sample", strength = 0.4, width = 1024L),   # width ignored
    sd_node("decode")
  )
  sd_run_pipeline(pipe, ctx = "fake_ctx")
  expect_equal(seen$load, c("in.png", "3"))
  expect_equal(seen$shape, c(8L, 8L, 4L))
  expect_equal(seen$init$ne, c(8L, 8L, 4L, 1L))
  expect_equal(seen$extra, list(strength = 0.4))
})

test_that("low-level nodes are validated before running", {
  err <- tryCatch(
    sd_run_pipeline(sd_pipeline(sd_node("sample"), sd_node("encode_image"),
                                sd_node("load_image"), sd_node("decode"),
                                sd_node("encode_text")), ctx = NULL),
    error = conditionMessage)
  expect_match(err, "node 1 \\(sample\\): needs 'cond'")
  expect_match(err, "node 2 \\(encode_image\\): needs 'image'")
  expect_match(err, "node 3 \\(load_image\\): 'path' is not set")
  expect_match(err, "node 5 \\(encode_text\\): 'prompt' is not set")
  expect_false(grepl("node 4 \\(decode\\)", err))   # latent from node 1
})
