# sd2R Shiny GUI — text-to-image generation
# Launch via sd2R::sd_app() or sd2R::sd_app(model_dir = "/path/to/models")

library(shiny)

# Null-coalescing operator (not always exported by shiny)
`%||%` <- function(a, b) if (is.null(a)) b else a

# ---------- Model presets by architecture ----------
MODEL_PRESETS <- list(
  sd1 = list(
    label = "SD 1.x",
    width = 512L, height = 512L,
    steps = 20L, cfg = 7.0,
    sampler = "EULER_A", scheduler = "KARRAS",
    max_chars = 350,
    resolutions = c("512x512", "768x768", "1024x1024")
  ),
  sd2 = list(
    label = "SD 2.x",
    width = 768L, height = 768L,
    steps = 20L, cfg = 7.0,
    sampler = "EULER_A", scheduler = "KARRAS",
    max_chars = 350,
    resolutions = c("512x512", "768x768", "1024x1024")
  ),
  sdxl = list(
    label = "SDXL",
    width = 1024L, height = 1024L,
    steps = 25L, cfg = 5.0,
    sampler = "EULER", scheduler = "KARRAS",
    max_chars = 700,
    resolutions = c("512x512", "768x768", "1024x1024")
  ),
  flux = list(
    label = "Flux",
    width = 1024L, height = 1024L,
    steps = 20L, cfg = 1.0,
    sampler = "EULER", scheduler = "SIMPLE",
    max_chars = 2000,
    resolutions = c("512x512", "768x768", "1024x1024")
  ),
  flux2 = list(
    label = "FLUX.2 (Klein)",
    width = 1024L, height = 1024L,
    steps = 4L, cfg = 1.0,
    sampler = "EULER", scheduler = "SIMPLE",
    max_chars = 2000,
    resolutions = c("512x512", "768x768", "1024x1024")
  ),
  sd3 = list(
    label = "SD 3",
    width = 1024L, height = 1024L,
    steps = 28L, cfg = 5.0,
    sampler = "EULER", scheduler = "SGM_UNIFORM",
    max_chars = 700,
    resolutions = c("512x512", "768x768", "1024x1024")
  )
)

sampler_names  <- names(sd2R::SAMPLE_METHOD)
scheduler_names <- names(sd2R::SCHEDULER)

# ---------- Classify files by role (based on filename) ----------
# Returns a named list of character vectors: $main, $diffusion, $vae, $clip_l,
# $t5xxl, $llm. Each file appears only in dropdowns where it can plausibly be used.
classify_files <- function(files) {
  if (length(files) == 0) {
    return(list(main = character(), diffusion = character(),
                vae = character(), clip_l = character(),
                t5xxl = character(), llm = character()))
  }
  fl <- tolower(files)

  is_vae      <- grepl("(^|[^a-z])(vae|\\bae\\b)", fl)
  is_clip     <- grepl("clip", fl) & !grepl("clip_vision|clip-vision", fl)
  is_t5       <- grepl("t5", fl)
  # LLM text encoder for FLUX.2 (Qwen3 / Mistral-Small) and other DiT LLMs.
  is_llm      <- grepl("qwen|mistral", fl)
  is_diff     <- grepl("flux|sd3|dit|unet", fl) & !is_llm
  is_aux_only <- grepl("upscaler|esrgan|taesd|lora|controlnet|control_net|photo_maker|clip_vision|clip-vision", fl)

  # Main checkpoint = anything that isn't a recognized auxiliary or diffusion-only file
  is_main <- !is_vae & !is_clip & !is_t5 & !is_llm & !is_diff & !is_aux_only

  list(
    main      = files[is_main],
    diffusion = files[is_diff],
    vae       = files[is_vae],
    clip_l    = files[is_clip],
    t5xxl     = files[is_t5],
    llm       = files[is_llm]
  )
}

# ---------- Auto-assign model roles by filename ----------
auto_assign_roles <- function(dir_path) {
  files <- list.files(dir_path, pattern = "\\.(safetensors|gguf|ckpt)$",
                      full.names = FALSE, ignore.case = TRUE)
  if (length(files) == 0) return(list(arch = "sd1"))

  sizes <- file.size(file.path(dir_path, files))
  names(sizes) <- files
  fl <- tolower(files)

  roles <- list(arch = "sd1", model = "", diffusion = "", vae = "",
                clip_l = "", t5xxl = "", llm = "")
  assigned <- rep(FALSE, length(files))

  # Step 1: detect architecture from filenames.
  # flux2 must be checked before flux (flux2 filenames also contain "flux").
  has_flux2 <- any(grepl("flux[._-]?2|flux2", fl))
  has_flux  <- any(grepl("flux", fl))
  has_sd3   <- any(grepl("sd3", fl))
  has_sdxl  <- any(grepl("sdxl|sd_xl", fl))
  has_t5    <- any(grepl("t5", fl))

  if (has_flux2) {
    roles$arch <- "flux2"
  } else if (has_flux) {
    roles$arch <- "flux"
  } else if (has_sd3) {
    roles$arch <- "sd3"
  } else if (has_sdxl) {
    roles$arch <- "sdxl"
  } else {
    # Check sizes: SD2 models are typically >3GB, SD1 ~2-4GB
    # Heuristic: if largest model >5GB and no other markers -> sd2
    roles$arch <- "sd1"
  }

  is_multipart <- roles$arch %in% c("flux", "flux2", "sd3")

  # Step 2: assign auxiliary roles (VAE, CLIP, T5)

  # VAE: "vae" or standalone "ae" in name
  idx <- grep("(^|[^a-z])(vae|\\bae\\b)", fl)
  if (length(idx)) {
    pick <- idx[which.max(sizes[idx])]
    roles$vae <- files[pick]
    assigned[pick] <- TRUE
  }

  # FLUX.2 uses an LLM text encoder (Qwen3 / Mistral) only — it has no CLIP-L
  # or T5-XXL. Assigning those here would pull in incompatible encoders and
  # confuse model loading, so skip them for flux2.
  uses_clip_t5 <- roles$arch != "flux2"

  # CLIP-L: "clip" in name (FLUX.1 / SD3 / SDXL)
  if (uses_clip_t5) {
    idx <- grep("clip", fl)
    idx <- setdiff(idx, which(assigned))
    if (length(idx)) {
      pick <- idx[which.max(sizes[idx])]
      roles$clip_l <- files[pick]
      assigned[pick] <- TRUE
    }
  }

  # T5-XXL: "t5" in name (FLUX.1 / SD3)
  if (uses_clip_t5) {
    idx <- grep("t5", fl)
    idx <- setdiff(idx, which(assigned))
    if (length(idx)) {
      pick <- idx[which.max(sizes[idx])]
      roles$t5xxl <- files[pick]
      assigned[pick] <- TRUE
    }
  }

  # LLM text encoder: Qwen3 (FLUX.2 Klein) / Mistral-Small (full FLUX.2)
  idx <- grep("qwen|mistral", fl)
  idx <- setdiff(idx, which(assigned))
  if (length(idx)) {
    pick <- idx[which.max(sizes[idx])]
    roles$llm <- files[pick]
    assigned[pick] <- TRUE
  }

  # Step 3: assign diffusion model (Flux/SD3 specific files; LLM already taken)
  idx <- grep("flux|sd3|dit|unet", fl)
  idx <- setdiff(idx, which(assigned))
  if (length(idx)) {
    pick <- idx[which.max(sizes[idx])]
    roles$diffusion <- files[pick]
    assigned[pick] <- TRUE
  }

  # Step 4: main model — only for single-file architectures (SD1/SD2/SDXL)
  # For Flux/SD3 skip this to avoid loading incompatible checkpoints
  if (!is_multipart) {
    remaining <- which(!assigned)
    if (length(remaining)) {
      pick <- remaining[which.max(sizes[remaining])]
      roles$model <- files[pick]
    }
  }

  roles
}

# Read initial model_dir from option set by sd_app()
init_model_dir <- getOption("sd2R.model_dir", default = "/mnt/Data2/DS_projects/sd_models")

# ---------- UI ----------
ui <- fluidPage(
  tags$head(tags$style(HTML("
    body { background: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
    .well { background: #16213e; border: 1px solid #2a3a5c; }
    .btn-primary { background: #0f3460; border-color: #1a5276; color: #fff; }
    .btn-primary:hover { background: #1a5276; }
    .btn-danger { background: #c0392b; border-color: #a93226; }

    /* Input fields: light background, black text for readability */
    .form-control,
    .selectize-input,
    .selectize-input input {
      background: #eef1f5 !important;
      color: #111 !important;
      border-color: #2a3a5c;
      font-weight: 500;
    }
    .form-control:focus,
    .selectize-input.focus {
      background: #fff !important;
      color: #000 !important;
      border-color: #e94560;
    }
    textarea.form-control {
      background: #eef1f5 !important;
      color: #111 !important;
    }

    /* Dropdowns */
    .selectize-dropdown {
      background: #eef1f5;
      color: #111;
    }
    .selectize-dropdown-content .option {
      color: #111;
    }
    .selectize-dropdown-content .option.active {
      background: #1a5276;
      color: #fff;
    }

    /* Labels */
    .control-label {
      color: #ccc;
      font-weight: 600;
    }

    h3, h4 { color: #e94560; }
    .progress { background: #0f3460; }
    .progress-bar { background: #e94560; }
    #gpu_info { font-family: monospace; font-size: 0.85em; white-space: pre-wrap;
      background: #0f3460; padding: 8px; border-radius: 4px; margin-bottom: 10px;
      color: #e0e0e0; }
    #char_counter { font-size: 0.85em; margin-top: -8px; margin-bottom: 8px; }
    .img-container { text-align: center; padding: 10px; }
    .img-container img { max-width: 100%; border: 2px solid #2a3a5c; border-radius: 4px; }
    #status_text { font-style: italic; color: #aaa; }

    /* Numeric inputs */
    input[type='number'] {
      background: #eef1f5 !important;
      color: #111 !important;
    }
  "))),

  titlePanel(
    div(
      span("sd2R", style = "color:#e94560; font-weight:bold;"),
      span(" Image Generator", style = "color:#e0e0e0;")
    ),
    windowTitle = "sd2R Image Generator"
  ),

  sidebarLayout(
    sidebarPanel(
      width = 4,

      # GPU info
      h4("GPU"),
      uiOutput("gpu_info"),

      # Model
      h4("Model"),
      selectInput("model_type", "Architecture", names(MODEL_PRESETS),
                  selected = "sd1"),

      # Models folder
      fluidRow(
        column(9, textInput("model_dir", "Models folder", value = init_model_dir)),
        column(3, actionButton("scan_dir", "Scan", class = "btn-primary btn-sm",
                               style = "margin-top: 25px; width: 100%;"))
      ),

      # Auto-assigned dropdowns — visibility depends on architecture
      conditionalPanel(
        condition = "input.model_type != 'flux' && input.model_type != 'flux2' && input.model_type != 'sd3'",
        selectInput("sel_model", "Model", choices = NULL)
      ),
      conditionalPanel(
        condition = "input.model_type == 'flux' || input.model_type == 'flux2' || input.model_type == 'sd3'",
        selectInput("sel_diffusion", "Diffusion model", choices = NULL),
        selectInput("sel_clip_l", "CLIP-L (optional)", choices = NULL),
        selectInput("sel_t5xxl", "T5-XXL (optional)", choices = NULL)
      ),
      # LLM text encoder — FLUX.2 only (Qwen3 / Mistral-Small)
      conditionalPanel(
        condition = "input.model_type == 'flux2'",
        selectInput("sel_llm", "LLM encoder (Qwen3/Mistral)", choices = NULL)
      ),
      selectInput("sel_vae", "VAE (optional)", choices = NULL),

      actionButton("load_model", "Load Model", class = "btn-primary btn-block",
                    style = "width: 100%; margin-bottom: 15px;"),

      hr(),

      # Generation params
      h4("Generation"),

      textAreaInput("prompt", "Prompt", rows = 4,
                    value = "A fox and a bear walking through a misty autumn forest, golden sunlight filtering through the trees, detailed fur, photorealistic"),
      uiOutput("char_counter"),
      textAreaInput("neg_prompt", "Negative prompt", rows = 2,
                    value = "bad quality, blurry, ugly"),

      selectInput("resolution", "Resolution", choices = NULL),
      fluidRow(
        column(6, selectInput("sampler", "Sampler", sampler_names, selected = "EULER_A")),
        column(6, selectInput("scheduler", "Scheduler", scheduler_names, selected = "KARRAS"))
      ),
      fluidRow(
        column(4, numericInput("steps", "Steps", 20, min = 1, max = 100)),
        column(4, numericInput("cfg", "CFG", 7.0, min = 0, max = 30, step = 0.5)),
        column(4, numericInput("seed", "Seed", 42, min = -1))
      ),
      fluidRow(
        column(12,
          checkboxInput("live_preview", "Live preview (fast latent projection)",
                        value = TRUE)
        )
      ),

      hr(),
      fluidRow(
        column(6,
          actionButton("generate", "Generate", class = "btn-primary btn-block",
                       style = "width: 100%;")
        ),
        column(6,
          downloadButton("save_btn", "Save PNG", class = "btn-block",
                         style = "width: 100%;")
        )
      )
    ),

    mainPanel(
      width = 8,
      uiOutput("progress_ui"),
      div(class = "img-container", uiOutput("result_image"))
    )
  )
)

# ---------- Server ----------
server <- function(input, output, session) {

  rv <- reactiveValues(
    generating = FALSE,
    loading_model = FALSE,
    status_msg = "",
    progress_trigger = NULL,
    image_trigger = NULL
  )

  # Non-reactive state for use in later() callbacks
  # IMPORTANT: ctx stored here (not in rv) to avoid Shiny reactive wrapping
  # of XPtr, which can cause GC issues with async C++ threads
  local_state <- new.env(parent = emptyenv())
  local_state$load_t0 <- 0
  local_state$model_type <- "sd1"
  local_state$gen_seed <- 42L
  local_state$ctx <- NULL
  local_state$last_image <- NULL

  # GPU info at startup
  output$gpu_info <- renderUI({
    info <- tryCatch({
      if (!sd2R::sd_vulkan_device_count()) {
        "No Vulkan GPU detected"
      } else {
        devs <- ggmlR::ggml_vulkan_list_devices()
        lines <- vapply(devs, function(d) {
          sprintf("[%d] %s  (%.1f / %.1f GB)",
                  d$index, d$name,
                  d$free_memory / 1e9, d$total_memory / 1e9)
        }, character(1))
        paste(lines, collapse = "\n")
      }
    }, error = function(e) paste("GPU info error:", e$message))
    div(id = "gpu_info", info)
  })

  # --- Scan folder: list files, auto-assign roles, populate dropdowns ---
  scan_model_dir <- function() {
    dir_path <- trimws(input$model_dir)
    if (!nzchar(dir_path) || !dir.exists(dir_path)) {
      showNotification("Folder not found", type = "error")
      return()
    }

    all_files <- list.files(dir_path,
                            pattern = "\\.(safetensors|gguf|ckpt)$",
                            full.names = FALSE, ignore.case = TRUE)
    if (length(all_files) == 0) {
      showNotification("No model files found in folder", type = "warning")
      return()
    }

    none <- c("(none)" = "")
    by_role <- classify_files(all_files)
    mk <- function(v) c(none, setNames(v, v))

    roles <- auto_assign_roles(dir_path)

    # Auto-switch architecture based on detected files
    updateSelectInput(session, "model_type", selected = roles$arch)

    updateSelectInput(session, "sel_model",     choices = mk(by_role$main),      selected = roles$model)
    updateSelectInput(session, "sel_diffusion", choices = mk(by_role$diffusion), selected = roles$diffusion)
    updateSelectInput(session, "sel_vae",       choices = mk(by_role$vae),       selected = roles$vae)
    updateSelectInput(session, "sel_clip_l",    choices = mk(by_role$clip_l),    selected = roles$clip_l)
    updateSelectInput(session, "sel_t5xxl",     choices = mk(by_role$t5xxl),     selected = roles$t5xxl)
    updateSelectInput(session, "sel_llm",       choices = mk(by_role$llm),       selected = roles$llm)

    showNotification(sprintf("Found %d files, detected: %s",
                             length(all_files), toupper(roles$arch)),
                     type = "message")
  }

  # Scan on button click
  observeEvent(input$scan_dir, scan_model_dir())

  # Auto-scan if model_dir was passed via sd_app()
  if (nzchar(init_model_dir) && dir.exists(init_model_dir)) {
    observeEvent(TRUE, scan_model_dir(), once = TRUE, ignoreInit = FALSE)
  }

  # --- Resolve model paths ---
  get_model_paths <- function() {
    dir_path <- trimws(input$model_dir)
    if (!nzchar(dir_path)) return(list())
    full <- function(f) {
      if (is.null(f) || !nzchar(f)) return(NULL)
      file.path(dir_path, f)
    }
    list(
      model_path           = full(input$sel_model),
      diffusion_model_path = full(input$sel_diffusion),
      vae_path             = full(input$sel_vae),
      clip_l_path          = full(input$sel_clip_l),
      t5xxl_path           = full(input$sel_t5xxl),
      llm_path             = full(input$sel_llm)
    )
  }

  # Update controls when preset changes
  observeEvent(input$model_type, {
    p <- MODEL_PRESETS[[input$model_type]]
    updateSelectInput(session, "resolution", choices = p$resolutions,
                      selected = paste0(p$width, "x", p$height))
    updateSelectInput(session, "sampler", selected = p$sampler)
    updateSelectInput(session, "scheduler", selected = p$scheduler)
    updateNumericInput(session, "steps", value = p$steps)
    updateNumericInput(session, "cfg", value = p$cfg)

    # Clear stale role selections from the other branch to avoid sending
    # incompatible path combinations to sd.cpp
    if (input$model_type %in% c("flux", "flux2", "sd3")) {
      updateSelectInput(session, "sel_model", selected = "")
    } else {
      updateSelectInput(session, "sel_diffusion", selected = "")
      updateSelectInput(session, "sel_clip_l",    selected = "")
      updateSelectInput(session, "sel_t5xxl",     selected = "")
    }
    # LLM encoder applies to flux2 only
    if (!identical(input$model_type, "flux2")) {
      updateSelectInput(session, "sel_llm", selected = "")
    }
  })

  # Char counter
  output$char_counter <- renderUI({
    p <- MODEL_PRESETS[[input$model_type]]
    n <- nchar(input$prompt %||% "")
    color <- if (n > p$max_chars) "#e94560" else "#888"
    div(id = "char_counter",
        span(sprintf("%d / %d characters", n, p$max_chars), style = paste0("color:", color)))
  })

  # --- Progress file for async generation ---
  progress_file <- tempfile("sd_progress_", fileext = ".json")

  # --- Live preview file (single PPM, updated atomically by the C callback) ---
  preview_file <- tempfile("sd_preview_", fileext = ".ppm")
  preview_active <- FALSE  # whether preview is wired up for the current run

  # Read progress from temp file written by C++ callback
  read_progress <- function() {
    if (!file.exists(progress_file)) return(NULL)
    tryCatch({
      txt <- readLines(progress_file, warn = FALSE)
      if (length(txt) == 0 || !nzchar(txt[1])) return(NULL)
      jsonlite::fromJSON(txt[1])
    }, error = function(e) NULL)
  }

  # Progress UI (updated by polling)
  output$progress_ui <- renderUI({
    rv$progress_trigger  # dependency for reactivity
    p <- read_progress()
    if (rv$generating) {
      if (!is.null(p) && p$steps > 0) {
        pct <- p$pct
        eta <- round(p$eta_sec, 1)
        tagList(
          div(style = "margin-bottom: 8px; color: #e0e0e0;",
              sprintf("Step %d / %d  —  ETA: %.1f sec", p$step, p$steps, eta)),
          div(style = "background: #0f3460; border-radius: 4px; height: 20px; margin-bottom: 10px;",
              div(style = sprintf(
                "background: #e94560; height: 100%%; border-radius: 4px; width: %d%%; transition: width 0.3s;",
                pct)))
        )
      } else {
        div(style = "color: #aaa; font-style: italic; margin-bottom: 10px;",
            "Starting generation...")
      }
    } else if (rv$loading_model) {
      if (!is.null(p) && p$steps > 0) {
        pct <- p$pct
        tagList(
          div(style = "margin-bottom: 8px; color: #e0e0e0;", rv$status_msg),
          div(style = "background: #0f3460; border-radius: 4px; height: 20px; margin-bottom: 10px;",
              div(style = sprintf(
                "background: #3498db; height: 100%%; border-radius: 4px; width: %d%%; transition: width 0.3s;",
                pct)))
        )
      } else {
        div(style = "color: #aaa; font-style: italic; margin-bottom: 10px;",
            rv$status_msg)
      }
    } else {
      div(style = "color: #aaa; font-style: italic; margin-bottom: 10px;",
          rv$status_msg)
    }
  })

  # --- Log file for async loading status ---
  log_file <- tempfile("sd_log_", fileext = ".txt")

  read_log <- function() {
    if (!file.exists(log_file)) return("")
    tryCatch({
      txt <- readLines(log_file, warn = FALSE)
      if (length(txt)) txt[length(txt)] else ""
    }, error = function(e) "")
  }

  # Load model (async via std::thread)
  observeEvent(input$load_model, {
    paths <- get_model_paths()

    if (is.null(paths$model_path) && is.null(paths$diffusion_model_path)) {
      showNotification("Select a model or diffusion model file", type = "error")
      return()
    }
    if (rv$loading_model || rv$generating) {
      showNotification("Busy", type = "warning")
      return()
    }

    rv$loading_model <- TRUE
    local_state$load_t0 <- as.numeric(Sys.time())
    local_state$model_type <- input$model_type
    rv$status_msg <- "Loading model..."

    # Build params for C++ sd_create_context_async
    ctx_params <- list(
      vae_decode_only = TRUE,
      free_params_immediately = FALSE,
      diffusion_flash_attn = TRUE,
      # sd_ctx_params_init() in C++ leaves vae_conv_direct/diffusion_conv_direct
      # uninitialized, so they MUST be passed explicitly — otherwise the VAE
      # convolution path reads garbage and the decode crashes ("vae not start").
      # Match sd_ctx()'s defaults: VAE on (×24 faster CONV_2D), diffusion off.
      vae_conv_direct = TRUE,
      diffusion_conv_direct = FALSE,
      rng_type = as.integer(sd2R::RNG_TYPE$CUDA),
      wtype = as.integer(sd2R::SD_TYPE$COUNT),
      n_threads = 0L,
      flow_shift = 0.0,
      lora_apply_mode = as.integer(sd2R::LORA_APPLY_MODE$AUTO)
    )
    if (!is.null(paths$model_path))
      ctx_params$model_path <- paths$model_path
    if (!is.null(paths$diffusion_model_path))
      ctx_params$diffusion_model_path <- paths$diffusion_model_path
    if (!is.null(paths$vae_path))
      ctx_params$vae_path <- paths$vae_path
    if (!is.null(paths$clip_l_path))
      ctx_params$clip_l_path <- paths$clip_l_path
    if (!is.null(paths$t5xxl_path))
      ctx_params$t5xxl_path <- paths$t5xxl_path
    if (!is.null(paths$llm_path))
      ctx_params$llm_path <- paths$llm_path

    # FLUX.2: request the meta backend when this build supports it (ggmlR has
    # ggml_backend_meta_device). It only actually engages with >= 2 GPUs (C++
    # falls back to the normal single-backend path on 1 GPU or older builds).
    meta_ok <- isTRUE(tryCatch(sd2R:::sd_meta_backend_available(),
                               error = function(e) FALSE))
    if (identical(input$model_type, "flux2") && meta_ok) {
      ctx_params$meta_backend <- TRUE
    }

    # Set log + progress files and launch async
    sd2R:::sd_set_log_file(log_file)
    sd2R:::sd_set_progress_file(progress_file)
    sd2R:::sd_set_verbose(TRUE)

    tryCatch({
      sd2R:::sd_create_context_async(ctx_params)
      poll_loading()
    }, error = function(e) {
      rv$loading_model <- FALSE
      rv$status_msg <- paste("Load error:", e$message)
      sd2R:::sd_clear_log_file()
    })
  })

  # Poll loading status every 500ms
  poll_loading <- function() {
    later::later(function() {
      status <- sd2R:::sd_create_context_poll()
      elapsed <- round(as.numeric(Sys.time()) - local_state$load_t0, 1)

      # Check tensor loading progress (uses same progress_file as generation)
      p <- read_progress()
      msg <- read_log()

      if (!is.null(p) && p$steps > 0) {
        # Tensor loading in progress — show progress bar style
        rv$status_msg <- sprintf("Loading tensors %d/%d (%.0fs)... %s",
                                 p$step, p$steps, elapsed, msg)
      } else if (nzchar(msg)) {
        rv$status_msg <- sprintf("Loading (%.0fs)... %s", elapsed, msg)
      } else {
        rv$status_msg <- sprintf("Loading model... %.0fs", elapsed)
      }
      rv$progress_trigger <- Sys.time()

      if (status$done) {
        tryCatch({
          ctx <- sd2R:::sd_create_context_result()
          attr(ctx, "model_type") <- local_state$model_type
          attr(ctx, "vae_decode_only") <- TRUE
          local_state$ctx <- ctx
          rv$status_msg <- sprintf("Model loaded in %.1f sec.", elapsed)
        }, error = function(e) {
          rv$status_msg <- paste("Load error:", e$message)
        })
        rv$loading_model <- FALSE
        sd2R:::sd_clear_log_file()
        sd2R:::sd_clear_progress_file()
      } else {
        poll_loading()
      }
    }, delay = 0.5)
  }

  # Generate (async via std::thread)
  observeEvent(input$generate, {
    if (is.null(local_state$ctx)) {
      showNotification("Load a model first", type = "error")
      return()
    }
    if (!nzchar(input$prompt %||% "")) {
      showNotification("Enter a prompt", type = "error")
      return()
    }
    if (rv$generating || rv$loading_model) {
      showNotification("Busy — wait for current operation", type = "warning")
      return()
    }

    dims <- as.integer(strsplit(input$resolution, "x")[[1]])

    rv$generating <- TRUE
    local_state$gen_dims <- dims
    local_state$gen_seed <- as.integer(input$seed)
    local_state$gen_t0 <- as.numeric(Sys.time())
    rv$status_msg <- "Starting generation..."

    # Set progress file path in C++
    sd2R:::sd_set_progress_file(progress_file)

    # Build the executable step plan. This mirrors sd_generate()'s routing:
    # cfg auto-1.0 for Flux/Flux.2 (the root cause of the VAE crash with cfg=7),
    # strategy selection (direct / tiled / highres-fix) and VRAM-aware VAE
    # tiling — none of which the old direct-async path inherited. Highres-fix
    # expands into base -> upscale -> refine steps run by the state machine.
    plan <- tryCatch(
      sd2R:::.sd_generate_plan(
        local_state$ctx,
        prompt          = input$prompt,
        negative_prompt = input$neg_prompt %||% "",
        width           = dims[1], height = dims[2],
        sample_method   = sd2R::SAMPLE_METHOD[[input$sampler]],
        sample_steps    = as.integer(input$steps),
        cfg_scale       = as.numeric(input$cfg),
        seed            = as.integer(input$seed),
        scheduler       = sd2R::SCHEDULER[[input$scheduler]],
        batch_count     = 1L,
        vae_mode        = "auto",
        vae_auto_threshold = 768L * 768L),
      error = function(e) e)
    if (inherits(plan, "error")) {
      rv$generating <- FALSE
      rv$status_msg <- paste("Plan error:", conditionMessage(plan))
      sd2R:::sd_clear_progress_file()
      return()
    }

    local_state$plan <- plan
    local_state$step_idx <- 0L          # index of the step about to run
    local_state$step_image <- NULL      # image carried between steps

    # Live preview: write the latest in-progress frame to preview_file. proj
    # mode is cheap and needs no VAE/taesd, so it is always safe to enable.
    preview_active <<- isTRUE(input$live_preview)
    if (preview_active) {
      if (file.exists(preview_file)) unlink(preview_file)
      local_state$preview_image <- NULL
      sd2R::sd_preview_start(preview_file, mode = sd2R::PREVIEW$PROJ, interval = 1L)
    }

    # Kick off the state machine (runs steps in order, async gen + sync upscale).
    tryCatch({
      run_next_step()
    }, error = function(e) {
      rv$generating <- FALSE
      rv$status_msg <- paste("Error:", e$message)
      sd2R:::sd_clear_progress_file()
      if (preview_active) { sd2R::sd_preview_stop(); preview_active <<- FALSE }
    })
  })

  # Finish the whole run: release preview, report timing, reset state.
  finish_generation <- function(err = NULL) {
    rv$generating <- FALSE
    sd2R:::sd_clear_progress_file()
    if (preview_active) { sd2R::sd_preview_stop(); preview_active <<- FALSE }
    if (!is.null(err)) {
      rv$status_msg <- paste("Error:", err)
      return(invisible())
    }
    elapsed <- round(as.numeric(Sys.time()) - local_state$gen_t0, 1)
    rv$status_msg <- sprintf("Done. %dx%d, seed=%d, %.1fs",
                             local_state$gen_dims[1], local_state$gen_dims[2],
                             local_state$gen_seed, elapsed)
  }

  # State machine driver: advance to and execute the next plan step. Synchronous
  # "upscale" steps run inline (fast) and fall through to the next step; async
  # "gen" steps launch the C++ worker and hand off to poll_step().
  run_next_step <- function() {
    repeat {
      local_state$step_idx <- local_state$step_idx + 1L
      if (local_state$step_idx > length(local_state$plan)) {
        # No final gen step produced an image — shouldn't happen, but be safe.
        finish_generation()
        return(invisible())
      }
      step <- local_state$plan[[local_state$step_idx]]

      if (identical(step$type, "upscale")) {
        rv$status_msg <- step$label
        res <- tryCatch({
          base_img <- local_state$step_image
          up <- if (!is.null(step$upscaler) && nzchar(step$upscaler) &&
                    file.exists(step$upscaler)) {
            sd2R:::sd_upscale_image(step$upscaler, base_img,
                                    upscale_factor = step$upscale_factor)
          } else {
            base_img
          }
          if (up$width != step$width || up$height != step$height) {
            up <- sd2R:::.resize_sd_image(up, step$width, step$height)
          }
          up
        }, error = function(e) e)
        if (inherits(res, "error")) {
          finish_generation(conditionMessage(res)); return(invisible())
        }
        local_state$step_image <- res
        next  # fall through to the next step in the same tick
      }

      # gen step: launch async, optionally feeding the previous image as init.
      rv$status_msg <- step$label
      params <- step$params
      if (isTRUE(step$uses_init) && !is.null(local_state$step_image)) {
        params$init_image <- local_state$step_image
      }
      ok <- tryCatch({
        sd2R:::sd_generate_async(local_state$ctx, params)
        TRUE
      }, error = function(e) { finish_generation(conditionMessage(e)); FALSE })
      if (!ok) return(invisible())
      poll_step(step)
      return(invisible())
    }
  }

  # Poll the currently running gen step every 500ms; on completion store its
  # image and either finish (final step) or advance the machine.
  poll_step <- function(step) {
    later::later(function() {
      status <- sd2R:::sd_generate_poll()
      rv$progress_trigger <- Sys.time()

      # Pull the latest preview frame (if enabled) so the result pane shows the
      # image taking shape. sd_read_preview() returns NULL until a frame exists.
      if (preview_active) {
        pv <- tryCatch(sd2R::sd_read_preview(preview_file), error = function(e) NULL)
        if (!is.null(pv)) {
          local_state$preview_image <- pv
          rv$image_trigger <- Sys.time()
        }
      }

      if (status$done) {
        res <- tryCatch(sd2R:::sd_generate_result(), error = function(e) e)
        if (inherits(res, "error")) {
          finish_generation(conditionMessage(res)); return()
        }
        local_state$step_image <- res[[1]]
        if (isTRUE(step$final)) {
          local_state$last_image <- res[[1]]
          local_state$preview_image <- NULL  # final replaces preview
          rv$image_trigger <- Sys.time()
          finish_generation()
        } else {
          # Show the intermediate result while the next step runs.
          local_state$preview_image <- NULL
          local_state$last_image <- res[[1]]
          rv$image_trigger <- Sys.time()
          run_next_step()
        }
      } else {
        poll_step(step)
      }
    }, delay = 0.5)
  }

  # Display result. While generating with live preview on, show the latest
  # preview frame (small latent-projection image, scaled up with pixelation so
  # it reads as a draft); once done, the final image replaces it.
  output$result_image <- renderUI({
    rv$image_trigger  # reactive dependency to re-render on new image
    final <- local_state$last_image
    showing_preview <- rv$generating && !is.null(local_state$preview_image)
    img <- if (showing_preview) local_state$preview_image else final

    if (is.null(img)) {
      div(style = "color:#555; padding: 100px 0; font-size: 1.3em;",
          "Generated image will appear here")
    } else {
      tmp <- tempfile(fileext = ".png")
      sd2R::sd_save_image(img, tmp)
      b64 <- base64enc::base64encode(tmp)
      style <- "max-width: 100%;"
      if (showing_preview) {
        # nearest-neighbour upscale so a 32x32 draft fills the pane crisply
        style <- paste0(style, " image-rendering: pixelated; width: 100%;",
                        " opacity: 0.92;")
      }
      tagList(
        tags$img(src = paste0("data:image/png;base64,", b64), style = style),
        if (showing_preview)
          div(style = "color:#e94560; font-size:0.85em; margin-top:4px;",
              "live preview…")
      )
    }
  })

  # Download
  output$save_btn <- downloadHandler(
    filename = function() {
      paste0("sd2R_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".png")
    },
    content = function(file) {
      if (!is.null(local_state$last_image)) {
        sd2R::sd_save_image(local_state$last_image, file)
      }
    }
  )
}

shinyApp(ui, server)
