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
                                  "remove_background", "load_mask",
                                  "invert_mask", "save_mask",
                                  "resize", "resize_mask", "crop",
                                  "crop_mask", "composite",
                                  "encode_text", "load_image", "encode_image",
                                  "sample", "decode", "upscale_latent",
                                  "canny", "depth", "input_image",
                                  "input_mask", "input_latent",
                                  "unbatch", "unbatch_latent"))
  for (nm in names(types)) {
    spec <- types[[nm]]
    expect_true(is.function(spec$run), info = nm)
    ins <- c(spec$inputs, spec$optional)
    kinds <- vapply(ins, function(s) sd2R:::input_kind(spec, s), "")
    expect_true(all(kinds %in% sd2R:::.slot_kind), info = nm)
    expect_true(all(spec$outputs %in% slots),
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

# --- DAG: explicit links by node id ---

test_that("sd_node splits input links from params and checks id", {
  n <- sd_node("decode", id = "d1", latent = "s1")
  expect_equal(n$id, "d1")
  expect_equal(n$inputs, list(latent = "s1"))
  expect_equal(length(n$params), 0L)
  n2 <- sd_node("sample", cond = "txt$cond", seed = 1L)
  expect_equal(n2$inputs, list(cond = "txt$cond"))
  expect_equal(n2$params, list(seed = 1L))
  expect_error(sd_node("decode", latent = 1), "must be a node id string")
  expect_error(sd_node("decode", id = "a$b"), "must not contain")
  expect_error(sd_node("decode", id = ""), "non-empty string")
})

test_that("sd_pipeline assigns unique ids and rejects duplicates", {
  pipe <- sd_pipeline(sd_node("load_image", path = "a.png"),
                      sd_node("save", path = "b.png"))
  expect_equal(vapply(pipe$nodes, `[[`, "", "id"),
               c("load_image_1", "save_2"))
  pipe2 <- sd_pipeline(sd_node("load_image", path = "a.png"),
                       sd_node("save", id = "load_image_1", path = "b.png"))
  expect_equal(pipe2$nodes[[1]]$id, "load_image_1_")
  expect_error(sd_pipeline(sd_node("decode", id = "x"),
                           sd_node("decode", id = "x")),
               "Duplicate node id")
})

test_that("explicit links route outputs by id (branching)", {
  local_mocked_bindings(.node_types = list(
    make_text = list(
      inputs = character(0), optional = character(0),
      outputs = c("cond", "uncond"), required_params = character(0),
      run = function(env, params, inputs) {
        list(cond = params$tag, uncond = paste0("neg-", params$tag))
      }
    ),
    make_image = list(
      inputs = character(0), optional = character(0), outputs = "image",
      required_params = character(0),
      run = function(env, params, inputs) list(image = params$tag)
    ),
    combine = list(
      inputs = c("cond", "image"), optional = "uncond", outputs = "image",
      required_params = character(0),
      run = function(env, params, inputs) {
        list(image = paste(inputs$cond, inputs$uncond, inputs$image, sep = "|"))
      }
    )
  ))
  pipe <- sd_pipeline(
    sd_node("make_text", id = "t1", tag = "a"),
    sd_node("make_image", id = "i1", tag = "x"),
    sd_node("make_text", id = "t2", tag = "b"),
    sd_node("make_image", id = "i2", tag = "y"),
    # explicit links skip the nearest producers (t2, i2)
    sd_node("combine", id = "c1", cond = "t1", image = "i1"),
    # "id$output": uncond fed into cond; uncond implicit -> nearest (t2)
    sd_node("combine", cond = "t1$uncond", image = "c1")
  )
  expect_equal(sd_run_pipeline(pipe, ctx = NULL), "neg-a|neg-b|a|neg-b|x")
})

test_that("bad links are reported before running", {
  pipe <- sd_pipeline(
    sd_node("load_image", id = "img", path = "a.png"),
    sd_node("encode_text", id = "txt", prompt = "p"),
    sd_node("decode", latent = "img"),          # wrong type
    sd_node("decode", latent = "nope"),         # unknown id
    sd_node("decode", latent = "later"),        # not earlier
    sd_node("sample", cond = "txt$latent"),     # no such output
    sd_node("encode_image", id = "later", image = "img$image$x")
  )
  err <- tryCatch(sd2R:::validate_pipeline(pipe$nodes),
                  error = conditionMessage)
  expect_match(err, "node 3 \\(decode\\): input 'latent' = \"img\": node 'img' has no output of type latent")
  expect_match(err, "node 4 \\(decode\\).*no node with id 'nope'")
  expect_match(err, "node 5 \\(decode\\).*node 'later' does not run earlier")
  expect_match(err, "node 6 \\(sample\\).*has no output 'latent'")
  expect_match(err, "node 7 \\(encode_image\\).*expected \"id\" or \"id\\$output\"")
})

test_that("ids and links survive JSON round-trip", {
  pipe <- sd_pipeline(
    sd_node("encode_text", id = "txt", prompt = "a cat"),
    sd_node("sample", id = "s1", cond = "txt", uncond = "txt$uncond",
            seed = 1L),
    sd_node("decode", latent = "s1")
  )
  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp))
  sd_save_pipeline(pipe, tmp)
  pipe2 <- sd_load_pipeline(tmp)
  expect_equal(vapply(pipe2$nodes, `[[`, "", "id"), c("txt", "s1", "decode_3"))
  expect_equal(pipe2$nodes[[2]]$inputs,
               list(cond = "txt", uncond = "txt$uncond"))
  expect_equal(pipe2$nodes[[2]]$params$seed, 1L)
  expect_equal(pipe2$nodes[[3]]$inputs, list(latent = "s1"))
  expect_error(sd2R:::pipeline_from_json(
    '[{"type": "decode", "inputs": {"latent": 1}, "params": {}}]'),
    "not an object of strings")
})

# --- outputs / culling / freeing ---

test_that("outputs selects results, culls unneeded nodes, keeps save", {
  ran <- character(0)
  local_mocked_bindings(.node_types = list(
    make_text = list(
      inputs = character(0), optional = character(0),
      outputs = c("cond", "uncond"), required_params = character(0),
      run = function(env, params, inputs) {
        ran <<- c(ran, params$tag)
        list(cond = params$tag, uncond = paste0("neg-", params$tag))
      }
    ),
    to_image = list(
      inputs = "cond", optional = character(0), outputs = "image",
      required_params = character(0),
      run = function(env, params, inputs) {
        ran <<- c(ran, params$tag)
        list(image = paste0("img-", inputs$cond))
      }
    ),
    save = list(
      inputs = "image", optional = character(0), outputs = "image",
      required_params = character(0), side_effect = TRUE,
      run = function(env, params, inputs) {
        ran <<- c(ran, params$tag)
        list(image = inputs$image)
      }
    )
  ))
  pipe <- sd_pipeline(
    sd_node("make_text", id = "t", tag = "t"),
    sd_node("to_image", id = "a", cond = "t", tag = "a"),
    sd_node("to_image", id = "b", cond = "t", tag = "b"),   # dead branch
    sd_node("save", id = "s", image = "a", tag = "s"),
    sd_node("to_image", id = "c", cond = "t", tag = "c")
  )

  # default: last node only; dead branch "b" skipped, "save" still runs
  expect_equal(sd_run_pipeline(pipe, ctx = NULL), "img-t")
  expect_equal(ran, c("t", "a", "s", "c"))

  # explicit ids: single output -> value, several -> list
  ran <- character(0)
  res <- sd_run_pipeline(pipe, ctx = NULL, outputs = c("b", "t"))
  expect_equal(names(res), c("b", "t"))
  expect_equal(res$b, "img-t")
  expect_equal(res$t, list(cond = "t", uncond = "neg-t"))
  expect_equal(ran, c("t", "a", "b", "s"))   # "c" culled

  # all
  ran <- character(0)
  res <- sd_run_pipeline(pipe, ctx = NULL, outputs = "all")
  expect_equal(names(res), c("t", "a", "b", "s", "c"))
  expect_equal(ran, c("t", "a", "b", "s", "c"))

  expect_error(sd_run_pipeline(pipe, ctx = NULL, outputs = c("a", "zz")),
               "Unknown node id\\(s\\) in 'outputs': zz")
  expect_error(sd_run_pipeline(pipe, ctx = NULL, outputs = 1),
               "'outputs' must be")
})

test_that("intermediate results are released after their last consumer", {
  fin <- new.env()
  fin$freed <- character(0)
  local_mocked_bindings(.node_types = list(
    make = list(
      inputs = character(0), optional = character(0), outputs = "latent",
      required_params = character(0),
      run = function(env, params, inputs) {
        e <- new.env()
        tag <- params$tag
        reg.finalizer(e, function(x) fin$freed <- c(fin$freed, tag))
        list(latent = e)
      }
    ),
    use = list(
      inputs = "latent", optional = character(0), outputs = "image",
      required_params = character(0),
      run = function(env, params, inputs) {
        gc()
        list(image = paste(fin$freed, collapse = ","))
      }
    )
  ))
  pipe <- sd_pipeline(
    sd_node("make", id = "m1", tag = "m1"),
    sd_node("use", id = "u1", latent = "m1"),
    sd_node("make", id = "m2", tag = "m2"),
    sd_node("use", id = "u2", latent = "m2")
  )
  # by the time u2 runs, m1 (last used by u1) has been released
  res <- sd_run_pipeline(pipe, ctx = NULL, outputs = c("u1", "u2"))
  expect_equal(res$u1, "")     # m1 still in use
  expect_equal(res$u2, "m1")   # released after u1, its last consumer
})

# --- mask slot ---

test_that("mask is its own type: links and hints", {
  pipe <- sd_pipeline(
    sd_node("load_image", id = "img", path = "a.png"),
    sd_node("load_mask", id = "m", path = "m.png"),
    sd_node("img2img", image = "m", prompt = "p"),          # mask -> image
    sd_node("invert_mask", mask = "img"),                   # image -> mask
    sd_node("img2img", image = "img", mask = "m.png", prompt = "p"),
    sd_node("img2img", image = "src.png", prompt = "p"),
    sd_node("save_mask", mask = "m")                        # no path
  )
  err <- tryCatch(sd2R:::validate_pipeline(pipe$nodes),
                  error = conditionMessage)
  expect_match(err, "node 3 \\(img2img\\).*'m' has no output of type image")
  expect_match(err, "node 4 \\(invert_mask\\).*'img' has no output of type mask")
  expect_match(err, "node 5 \\(img2img\\).*no node with id 'm.png'; .*'load_mask' node")
  expect_match(err, "node 6 \\(img2img\\).*no node with id 'src.png'; .*'load_image' node")
  expect_match(err, "node 7 \\(save_mask\\): 'path' is not set")

  # remove_background: "bg" alone picks the output named like the input
  ok <- sd_pipeline(
    sd_node("load_image", id = "src", path = "a.png"),
    sd_node("remove_background", id = "bg", image = "src", model_path = "x"),
    sd_node("invert_mask", id = "back", mask = "bg"),
    sd_node("img2img", image = "src", mask = "back", prompt = "p"),
    sd_node("save", image = "bg", path = "fg.png")
  )
  links <- sd2R:::validate_pipeline(ok$nodes)
  expect_equal(links[[3]]$mask, list(node = 2L, port = "mask"))
  expect_equal(links[[4]]$mask, list(node = 3L, port = "mask"))
  expect_equal(links[[5]]$image, list(node = 2L, port = "image"))
  # img2img without a mask stays plain img2img (optional input left NULL)
  plain <- sd_pipeline(sd_node("load_image", path = "a.png"),
                       sd_node("img2img", prompt = "p"))
  expect_null(sd2R:::validate_pipeline(plain$nodes)[[2]]$mask)
})

test_that("load_mask, invert_mask, save_mask run on real data", {
  m <- matrix(c(0, 1, 0.5, 1, 0, 0), nrow = 2)
  tmp_in <- tempfile(fileext = ".png")
  tmp_out <- tempfile(fileext = ".png")
  on.exit(unlink(c(tmp_in, tmp_out)))
  png::writePNG(m, tmp_in)
  pipe <- sd_pipeline(
    sd_node("load_mask", id = "m", path = tmp_in),
    sd_node("invert_mask", id = "inv", mask = "m"),
    sd_node("save_mask", path = tmp_out)
  )
  res <- sd_run_pipeline(pipe, ctx = NULL, outputs = c("m", "inv"))
  expect_equal(res$m$channel, 1L)
  expect_equal(as.integer(res$inv$data), 255L - as.integer(res$m$data))
  expect_equal(sd_load_mask(tmp_out)$data, res$inv$data)
})

test_that("img2img node passes the mask to sd_img2img", {
  got <- NULL
  local_mocked_bindings(sd_img2img = function(ctx, init_image, mask = NULL,
                                              ...) {
    got <<- list(image = init_image, mask = mask)
    list(init_image)
  })
  img <- list(width = 2L, height = 1L, channel = 3L, data = as.raw(1:6))
  msk <- list(width = 2L, height = 1L, channel = 1L, data = as.raw(c(0, 255)))
  local_mocked_bindings(sd_load_image = function(...) img,
                        sd_load_mask = function(...) msk)
  pipe <- sd_pipeline(sd_node("load_image", id = "src", path = "a.png"),
                      sd_node("load_mask", id = "m", path = "m.png"),
                      sd_node("img2img", image = "src", mask = "m",
                              prompt = "p"))
  sd_run_pipeline(pipe, ctx = NULL)
  expect_identical(got$mask, msk)
  # without a mask input: NULL
  pipe2 <- sd_pipeline(sd_node("load_image", path = "a.png"),
                       sd_node("img2img", prompt = "p"))
  sd_run_pipeline(pipe2, ctx = NULL)
  expect_null(got$mask)
})

# --- image operations: resize / crop / composite ---

img_from <- function(arr) sd2R:::.array_to_sd_image(arr)
grad_img <- function(h, w, ch = 3L) {
  img_from(array(seq(0, 1, length.out = h * w * ch), c(h, w, ch)))
}

test_that("resize: size rules, rounding, methods, errors", {
  img <- grad_img(10L, 20L)
  rs <- function(...) sd2R:::.op_resize(img, list(...), "resize")
  expect_identical(rs(width = 20, height = 10), img)       # same size: exact
  r <- rs(width = 7)                                       # 20x10 -> 7x3.5
  expect_equal(c(r$width, r$height), c(7L, 4L))            # rounded
  r <- rs(height = 5)
  expect_equal(c(r$width, r$height), c(10L, 5L))
  r <- rs(scale = 0.5)
  expect_equal(c(r$width, r$height), c(10L, 5L))
  # nearest on 2x upscale keeps only original values
  r <- rs(scale = 2, method = "nearest")
  expect_true(all(r$data %in% img$data))
  expect_error(rs(scale = 2, width = 5), "either 'scale' or 'width'/'height'")
  expect_error(rs(), "give 'width', 'height' or 'scale'")
  expect_error(rs(width = -1), "'width' must be a positive number")
  expect_error(rs(scale = 2, method = "cubic"), "'method' must be")
})

test_that("crop: 0-based region, exact pixels, bounds checked", {
  arr <- array(runif(6 * 8 * 3), c(6, 8, 3))
  img <- img_from(arr)
  cr <- function(...) sd2R:::.op_crop(img, list(...), "crop")
  r <- cr(x = 2, y = 1, width = 3, height = 4)
  expect_equal(c(r$width, r$height, r$channel), c(3L, 4L, 3L))
  expect_identical(r, img_from(sd_image_to_array(img)[2:5, 3:5, , drop = FALSE]))
  expect_identical(cr(width = 8, height = 6), img)          # full image
  expect_error(cr(x = 6, width = 3, height = 1),
               "region x = 6, y = 0, 3x1 is outside the 8x6 input")
  expect_error(cr(x = -1, width = 1, height = 1), "'x' must be an integer >= 0")
  expect_error(cr(width = 1.5, height = 1), "'width' must be an integer >= 1")
})

test_that("composite: blend, channels, size check", {
  fg <- img_from(array(1, c(2, 2, 4)))                      # white RGBA
  bg <- img_from(array(0, c(2, 2, 1)))                      # black gray
  m  <- img_from(array(c(1, 0, 0.5, 1), c(2, 2, 1)))
  r <- sd2R:::.op_composite(fg, bg, m)
  expect_equal(r$channel, 3L)
  a <- sd_image_to_array(r)
  expect_equal(a[, , 1], matrix(c(1, 0, 0.5, 1), 2), tolerance = 1 / 255)
  expect_equal(a[, , 3], a[, , 1])
  big <- img_from(array(0, c(3, 2, 3)))
  expect_error(sd2R:::.op_composite(fg, big, m),
               "fg 2x2, bg 2x3, mask 2x2\\); use 'resize' / 'resize_mask'")
})

test_that("composite needs explicit links; types are checked", {
  pipe <- sd_pipeline(
    sd_node("load_image", id = "a", path = "a.png"),
    sd_node("load_image", id = "b", path = "b.png"),
    sd_node("load_mask", id = "m", path = "m.png"),
    sd_node("composite", fg = "a", bg = "b"),               # mask not linked
    sd_node("composite", fg = "m", bg = "b", mask = "a")    # wrong types
  )
  err <- tryCatch(sd2R:::validate_pipeline(pipe$nodes),
                  error = conditionMessage)
  expect_match(err, "node 4 \\(composite\\): needs 'mask', link it explicitly")
  expect_match(err, "node 5 \\(composite\\): input 'fg'.*no output of type image")
  expect_match(err, "node 5 \\(composite\\): input 'mask'.*no output of type mask")

  ok <- sd_pipeline(
    sd_node("load_image", id = "a", path = "a.png"),
    sd_node("load_image", id = "b", path = "b.png"),
    sd_node("load_mask", id = "m", path = "m.png"),
    sd_node("composite", fg = "a", bg = "b", mask = "m")
  )
  lk <- sd2R:::validate_pipeline(ok$nodes)[[4]]
  expect_equal(lk$fg$node, 1L)
  expect_equal(lk$bg$node, 2L)
  expect_equal(lk$mask$node, 3L)
  expect_equal(sd_node("composite", fg = "a", bg = "b", mask = "m")$inputs,
               list(fg = "a", bg = "b", mask = "m"))
})

test_that("resize/crop/composite nodes run in a graph", {
  tmp <- tempfile(fileext = ".png"); tmpm <- tempfile(fileext = ".png")
  on.exit(unlink(c(tmp, tmpm)))
  sd_save_image(grad_img(8L, 12L), tmp)
  png::writePNG(matrix(rep(c(0, 1), each = 48), 8), tmpm)   # left 0, right 1
  pipe <- sd_pipeline(
    sd_node("load_image", id = "src", path = tmp),
    sd_node("load_mask", id = "m", path = tmpm),
    sd_node("resize", id = "small", image = "src", width = 6),
    sd_node("resize", id = "back", image = "small", width = 12, height = 8),
    sd_node("crop", id = "c", image = "src", x = 2, y = 2, width = 4, height = 4),
    sd_node("crop_mask", id = "cm", mask = "m", width = 6, height = 8),
    sd_node("resize_mask", id = "rm", mask = "cm", width = 12, height = 8,
            method = "nearest"),
    sd_node("composite", id = "out", fg = "src", bg = "back", mask = "m")
  )
  res <- sd_run_pipeline(pipe, ctx = NULL,
                         outputs = c("small", "c", "rm", "out"))
  expect_equal(c(res$small$width, res$small$height), c(6L, 4L))
  expect_equal(c(res$c$width, res$c$height), c(4L, 4L))
  expect_true(all(as.integer(res$rm$data) == 0L))   # left half of mask = 0
  expect_equal(res$out$channel, 3L)
})

test_that("array -> SD image rounds to the nearest byte", {
  arr <- array(c(0.6, 254.4, 254.6, 100.6) / 255, c(1, 4, 1))
  expect_equal(as.integer(sd2R:::.array_to_sd_image(arr)$data),
               c(1L, 254L, 255L, 101L))
  expect_equal(as.integer(sd2R:::.mask_matrix_to_sd_image(
    matrix(c(0.6, 254.6), 1))$data), c(1L, 255L))
})

# --- upscale_latent ---

test_that("upscale_latent: plain latents, size rules, ggml layout", {
  ul <- function(lat, ...) sd2R:::.op_upscale_latent(NULL, lat, list(...))
  # W = 3, H = 2, C = 4; value encodes (x, y, c) to check the layout
  W <- 3L; H <- 2L; C <- 4L
  vals <- array(0, c(W, H, C))
  for (x in 1:W) for (y in 1:H) for (cc in 1:C) vals[x, y, cc] <- x + 10 * y + 100 * cc
  lat <- list(type = 0L, ne = c(W, H, C, 1L), data = as.vector(vals))
  same <- ul(lat, scale = 1)
  expect_equal(same$ne, lat$ne)
  expect_equal(same$data, lat$data)
  up <- ul(lat, scale = 2, method = "nearest")
  expect_equal(up$ne, c(6L, 4L, 4L, 1L))
  a <- array(up$data, up$ne[1:3])
  expect_equal(a[c(1, 3, 5), c(1, 3), ], vals)   # x stays along dim 1
  # pixels: multiples of the VAE scale factor
  up <- ul(lat, width = 48, height = 32, vae_scale_factor = 8)
  expect_equal(up$ne[1:2], c(6L, 4L))
  expect_error(ul(lat, width = 50, height = 32, vae_scale_factor = 8),
               "'width' must be a positive multiple of 8")
  expect_error(ul(lat, scale = 2, width = 48), "either 'scale' or")
  expect_error(ul(lat, width = 48), "give 'scale', or both")
  lat16 <- list(type = 0L, ne = c(2L, 2L, 16L, 1L), data = rep(1, 64))
  expect_equal(ul(lat16, scale = 1.5)$ne, c(3L, 3L, 16L, 1L))
})

test_that("upscale_latent refuses packed latents (FLUX.2/3, 128 ch)", {
  lat <- list(type = 0L, ne = c(2L, 2L, 128L, 1L), data = rep(0, 512))
  expect_error(sd2R:::.op_upscale_latent(NULL, lat, list(scale = 2)),
               "128 channels is not supported.*decode -> resize/upscale -> encode_image")
})

# --- sample: mask / control inputs ---

test_that("sample mask/control: explicit only, typed, mask needs latent", {
  pipe <- sd_pipeline(
    sd_node("load_image", id = "img", path = "a.png"),
    sd_node("load_mask", id = "m", path = "m.png"),
    sd_node("encode_text", id = "txt", prompt = "p"),
    sd_node("sample", id = "s1", cond = "txt", mask = "m"),        # no latent
    sd_node("encode_image", id = "lat", image = "img"),
    sd_node("sample", id = "s2", cond = "txt", latent = "lat",
            control = "m"),                                         # mask as control
    sd_node("sample", id = "s3", cond = "txt", latent = "lat",
            mask = "img")                                           # image as mask
  )
  err <- tryCatch(sd2R:::validate_pipeline(pipe$nodes), error = conditionMessage)
  expect_match(err, "node 4 \\(sample\\): input 'mask' needs input 'latent'")
  expect_match(err, "node 6 \\(sample\\): input 'control'.*no output of type image")
  expect_match(err, "node 7 \\(sample\\): input 'mask'.*no output of type mask")

  ok <- sd_pipeline(
    sd_node("load_image", id = "img", path = "a.png"),
    sd_node("load_mask", id = "m", path = "m.png"),
    sd_node("encode_text", id = "txt", prompt = "p"),
    sd_node("encode_image", id = "lat", image = "img"),
    sd_node("sample", id = "plain", cond = "txt", latent = "lat"),
    sd_node("sample", id = "inp", cond = "txt", latent = "lat", mask = "m",
            control = "img", control_strength = 0.5)
  )
  lk <- sd2R:::validate_pipeline(ok$nodes)
  expect_null(lk[[5]]$mask)                 # not bound implicitly to "m"
  expect_null(lk[[5]]$control)
  expect_equal(lk[[6]]$mask$node, 2L)
  expect_equal(lk[[6]]$control$node, 1L)
  expect_equal(ok$nodes[[6]]$params$control_strength, 0.5)
})

test_that("control input is checked against the context before running", {
  ran <- FALSE
  local_mocked_bindings(
    sd_ctx_has_control_net_cpp = function(ctx) FALSE,
    sd_load_image = function(...) { ran <<- TRUE; list() })
  fake_ctx <- methods::new("externalptr")
  pipe <- sd_pipeline(
    sd_node("load_image", id = "img", path = "a.png"),
    sd_node("encode_text", id = "txt", prompt = "p"),
    sd_node("sample", cond = "txt", control = "img", width = 64, height = 64)
  )
  expect_error(sd_run_pipeline(pipe, fake_ctx),
               "node 3 \\(sample\\): input 'control' needs a ControlNet")
  expect_false(ran)                          # nothing ran
})

# --- batches in the graph ---

test_that("sd_unbatch and decode of a batch latent", {
  lat <- list(type = 0L, ne = c(2L, 2L, 1L, 3L), data = as.numeric(1:12))
  parts <- sd_unbatch(lat)
  expect_length(parts, 3L)
  expect_equal(parts[[2]]$data, 5:8)
  expect_equal(parts[[2]]$ne, c(2L, 2L, 1L, 1L))
  local_mocked_bindings(sd_decode_latent_cpp = function(ctx, latent) {
    list(width = 1L, height = 1L, channel = 3L,
         data = as.raw(rep(latent$data[1], 3)))
  })
  img <- sd_decode_latent("ctx", lat)
  expect_equal(img$batch, 3L)
  expect_equal(as.integer(img$data), rep(c(1L, 5L, 9L), each = 3))
  single <- sd_decode_latent("ctx", parts[[1]])
  expect_null(single$batch)                      # N = 1: plain image
  ims <- sd_unbatch(img)
  expect_null(ims[[3]]$batch)
  expect_equal(as.integer(ims[[3]]$data), rep(9L, 3))
  expect_error(sd_unbatch(1), "expected an SD image or an sd_tensor")
})

test_that("graph: sample(batch) -> decode -> unbatch; others refuse a batch", {
  local_mocked_bindings(
    sd_sample = function(ctx, cond, ..., batch = 1L) {
      list(type = 0L, ne = c(1L, 1L, 1L, as.integer(batch)),
           data = as.numeric(seq_len(batch)))
    },
    sd_encode_text = function(...) list(crossattn = NULL, vector = NULL, concat = NULL),
    sd_decode_latent_cpp = function(ctx, latent) {
      list(width = 1L, height = 1L, channel = 3L,
           data = as.raw(rep(latent$data[1] * 10, 3)))
    },
    latent_geometry = function(...) c(vae_scale_factor = 8L, latent_channels = 1L))
  pipe <- sd_pipeline(
    sd_node("encode_text", id = "txt", prompt = "p"),
    sd_node("sample", id = "lat", cond = "txt", batch = 3L, width = 8, height = 8),
    sd_node("decode", id = "imgs", latent = "lat"),
    sd_node("unbatch", id = "second", image = "imgs", index = 2),
    sd_node("unbatch_latent", id = "l3", latent = "lat", index = 3)
  )
  res <- sd_run_pipeline(pipe, NULL, outputs = c("imgs", "second", "l3"))
  expect_equal(res$imgs$batch, 3L)
  expect_equal(as.integer(res$second$data), rep(20L, 3))
  expect_null(res$second$batch)
  expect_equal(res$l3$data, 3)

  bad <- sd_pipeline(
    sd_node("encode_text", id = "txt", prompt = "p"),
    sd_node("sample", id = "lat", cond = "txt", batch = 2L, width = 8, height = 8),
    sd_node("decode", id = "imgs", latent = "lat"),
    sd_node("resize", image = "imgs", width = 2)
  )
  expect_error(sd_run_pipeline(bad, NULL),
               "node 4 \\(resize\\): input 'image' is a batch of 2; .*'unbatch'")
  oob <- sd_pipeline(
    sd_node("encode_text", id = "txt", prompt = "p"),
    sd_node("sample", id = "lat", cond = "txt", batch = 2L, width = 8, height = 8),
    sd_node("decode", id = "imgs", latent = "lat"),
    sd_node("unbatch", image = "imgs", index = 5)
  )
  expect_error(sd_run_pipeline(oob, NULL), "'index' must be an integer in 1..2")
})
