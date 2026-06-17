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
# arch_override: if non-NULL, respect the user-selected architecture instead of
# auto-detecting it from filenames. Files are then matched only against that
# architecture (e.g. "flux" excludes flux2 diffusion files, and vice versa).
auto_assign_roles <- function(dir_path, arch_override = NULL) {
  files <- list.files(dir_path, pattern = "\\.(safetensors|gguf|ckpt)$",
                      full.names = FALSE, ignore.case = TRUE)
  if (length(files) == 0) return(list(arch = arch_override %||% "sd1"))

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

  if (!is.null(arch_override)) {
    # User picked the architecture explicitly — honour it and match files to it.
    roles$arch <- arch_override
  } else if (has_flux2) {
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

  # VAE: "vae" or standalone "ae" in name. SD1/SD2/SDXL bundle the VAE inside the
  # checkpoint, so they need no external file. flux and flux2 use different VAEs
  # (e.g. ae.safetensors vs flux2-vae.safetensors) that must not be swapped.
  vae_idx <- grep("(^|[^a-z])(vae|\\bae\\b)", fl)
  if (roles$arch %in% c("sd1", "sd2", "sdxl")) {
    vae_idx <- integer(0)
  } else if (identical(roles$arch, "flux")) {
    # FLUX.1 VAE must not be a flux2 VAE.
    vae_idx <- setdiff(vae_idx, grep("flux[._-]?2|flux2", fl))
  } else if (identical(roles$arch, "flux2")) {
    # Prefer an explicit flux2 VAE; fall back to any VAE only if none is named.
    f2 <- intersect(vae_idx, grep("flux[._-]?2|flux2", fl))
    if (length(f2)) vae_idx <- f2
  }
  if (length(vae_idx)) {
    pick <- vae_idx[which.max(sizes[vae_idx])]
    roles$vae <- files[pick]
    assigned[pick] <- TRUE
  }

  # CLIP-L / T5-XXL are external encoders only for FLUX.1 and SD3. SD1/SD2/SDXL
  # ship them inside the single checkpoint, and FLUX.2 uses an LLM encoder
  # instead — assigning standalone encoders there would feed sd.cpp incompatible
  # paths, so restrict these roles to the architectures that actually need them.
  uses_clip_t5 <- roles$arch %in% c("flux", "sd3")

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

  # LLM text encoder: Qwen3 (FLUX.2 Klein) / Mistral-Small (full FLUX.2) — only
  # relevant to FLUX.2.
  if (identical(roles$arch, "flux2")) {
    idx <- grep("qwen|mistral", fl)
    idx <- setdiff(idx, which(assigned))
    if (length(idx)) {
      pick <- idx[which.max(sizes[idx])]
      roles$llm <- files[pick]
      assigned[pick] <- TRUE
    }
  }

  # Step 3: assign diffusion model — only the multipart architectures use it,
  # and each one must pick its own kind of file (never another arch's).
  if (is_multipart) {
    is_flux_file  <- grepl("flux", fl)
    is_flux2_file <- grepl("flux[._-]?2|flux2", fl)
    is_sd3_file   <- grepl("sd3", fl)
    is_generic    <- grepl("dit|unet", fl)  # arch-neutral diffusion naming

    if (identical(roles$arch, "flux")) {
      # FLUX.1: flux-named files, excluding flux2.
      cand <- which(is_flux_file & !is_flux2_file)
    } else if (identical(roles$arch, "flux2")) {
      # FLUX.2: requires an actual flux2 file; never fall back to FLUX.1.
      cand <- which(is_flux2_file)
    } else {  # sd3
      cand <- which(is_sd3_file | is_generic)
    }
    idx <- setdiff(cand, which(assigned))
    if (length(idx)) {
      pick <- idx[which.max(sizes[idx])]
      roles$diffusion <- files[pick]
      assigned[pick] <- TRUE
    }
  }

  # Step 4: main model — only for single-file architectures (SD1/SD2/SDXL)
  # For Flux/SD3 skip this to avoid loading incompatible checkpoints.
  # Exclude obvious component files (encoders/VAE/diffusion/aux) so we never
  # hand an encoder or VAE to the "Model" slot when no real checkpoint exists.
  if (!is_multipart) {
    is_component <- grepl(paste0("(^|[^a-z])(vae|\\bae\\b)|clip|t5|qwen|mistral|",
                                 "flux|sd3|dit|unet|upscaler|esrgan|taesd|lora|",
                                 "controlnet|control_net|photo_maker|clip_vision|clip-vision"),
                          fl)
    remaining <- setdiff(which(!assigned), which(is_component))
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
      actionButton("gpu_caps", "GPU caps", class = "btn-default btn-sm",
                   style = "margin: 6px 0 4px 0; width: 100%;"),

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
      fluidRow(
        column(8,
          checkboxInput("gen_log", "Write generation log (diagnostics)",
                        value = FALSE)
        ),
        column(4, uiOutput("download_log_ui"))
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
    image_trigger = NULL,
    show_caps = FALSE,    # toggle: GPU caps text replaces the image pane
    caps_text = "",       # captured output of the Vulkan caps inspector
    log_ready = NULL      # bumped when a generation log is ready to download
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
  local_state$gen_log_on <- FALSE

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

    # Respect the architecture the user picked: match files to it instead of
    # auto-switching the dropdown.
    roles <- auto_assign_roles(dir_path, arch_override = input$model_type)

    updateSelectInput(session, "sel_model",     choices = mk(by_role$main),      selected = roles$model)
    updateSelectInput(session, "sel_diffusion", choices = mk(by_role$diffusion), selected = roles$diffusion)
    updateSelectInput(session, "sel_vae",       choices = mk(by_role$vae),       selected = roles$vae)
    updateSelectInput(session, "sel_clip_l",    choices = mk(by_role$clip_l),    selected = roles$clip_l)
    updateSelectInput(session, "sel_t5xxl",     choices = mk(by_role$t5xxl),     selected = roles$t5xxl)
    updateSelectInput(session, "sel_llm",       choices = mk(by_role$llm),       selected = roles$llm)

    # Warn if the primary file for the chosen architecture is missing, instead
    # of silently leaving a (none) the user might not notice.
    if (roles$arch %in% c("flux", "flux2", "sd3")) {
      primary_ok <- nzchar(roles$diffusion)
      primary_lbl <- "diffusion model"
    } else {
      primary_ok <- nzchar(roles$model)
      primary_lbl <- "model checkpoint"
    }
    if (!primary_ok) {
      showNotification(sprintf("No %s found for %s in this folder",
                               primary_lbl, toupper(roles$arch)),
                       type = "warning", duration = 8)
    } else {
      showNotification(sprintf("Found %d files, matched for %s",
                               length(all_files), toupper(roles$arch)),
                       type = "message")
    }
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

    # Free the previously loaded context BEFORE creating the new one. Without
    # this, loading a second model keeps the first in VRAM (the XPtr finalizer
    # is non-deterministic and may not run for a long time), so two ~11 GB
    # models pile up — on a 24 GB card the GPU ends up nearly full and the next
    # Vulkan createDevice (load or even the GPU-caps probe) throws
    # vk::InitializationFailed and terminates the app. Releasing first means the
    # VRAM peak is one model, not two.
    if (!is.null(local_state$ctx)) {
      tryCatch(sd2R::sd_destroy_context(local_state$ctx),
               error = function(e) NULL)
      local_state$ctx <- NULL
      gc()
    }

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

    # Generation diagnostic log (opt-in). Writes inputs + the device/backend
    # actually selected + per-stage timings to log_file so we can tell whether
    # diffusion ran on the discrete GPU or the integrated one.
    local_state$gen_log_on <- isTRUE(input$gen_log)
    if (local_state$gen_log_on) {
      sd2R:::sd_set_log_file(log_file)   # truncates the file
      sd2R:::sd_set_log_debug(TRUE)      # include Vulkan device list (DEBUG)
      sd2R:::sd_set_verbose(TRUE)
      sd2R::sd_profile_start()
      hdr <- c(
        "=== Generation ===",
        sprintf("time:          %s", format(Sys.time())),
        sprintf("model_type:    %s", input$model_type %||% "?"),
        sprintf("prompt:        %s", input$prompt %||% ""),
        sprintf("negative:      %s", input$neg_prompt %||% ""),
        sprintf("resolution:    %dx%d", dims[1], dims[2]),
        sprintf("steps:         %s", input$steps),
        sprintf("sampler:       %s", input$sampler),
        sprintf("scheduler:     %s", input$scheduler),
        sprintf("cfg:           %s", input$cfg),
        sprintf("seed:          %s", input$seed),
        gen_device_line(local_state$ctx),
        "",
        "--- sd.cpp log ---")
      cat(hdr, file = log_file, sep = "\n", append = TRUE)
    }

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

    # Finalize the diagnostic log: stop profiling and append per-stage timings,
    # then a distilled summary (device / flash-attn / stage wall times).
    if (isTRUE(local_state$gen_log_on)) {
      tryCatch({
        sd2R::sd_profile_stop()
        prof <- utils::capture.output(
          print(sd2R::sd_profile_summary(sd2R::sd_profile_get())))
        cat(c("", "--- Stage timings (profiler) ---", prof),
            file = log_file, sep = "\n", append = TRUE)
      }, error = function(e) {
        cat(c("", paste("[log] profile error:", conditionMessage(e))),
            file = log_file, sep = "\n", append = TRUE)
      })
      if (!is.null(err)) {
        cat(c("", paste("[log] generation error:", err)),
            file = log_file, sep = "\n", append = TRUE)
      }
      tryCatch({
        cat(c("", summarize_gen_log(
                    log_file,
                    dev_idx = attr(local_state$ctx, "vram_device") %||% 0L)),
            file = log_file, sep = "\n", append = TRUE)
      }, error = function(e) {
        cat(c("", paste("[log] summary error:", conditionMessage(e))),
            file = log_file, sep = "\n", append = TRUE)
      })
      sd2R:::sd_set_log_debug(FALSE)
      rv$log_ready <- Sys.time()  # reveal the download button
    }

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

  # --- Device line for the generation log -------------------------------------
  # The R-side view of which Vulkan device this context targets (index + name +
  # free/total VRAM). The authoritative C++ pick is the "Selected main device:"
  # line that sd.cpp logs once at context init; this line makes the device
  # visible in every generation log even when that init line isn't re-emitted.
  gen_device_line <- function(ctx) {
    idx <- tryCatch(attr(ctx, "vram_device") %||% 0L, error = function(e) 0L)
    name <- tryCatch(ggmlR::ggml_vulkan_device_description(idx),
                     error = function(e) "?")
    mem <- tryCatch(ggmlR::ggml_vulkan_device_memory(idx), error = function(e) NULL)
    memstr <- if (!is.null(mem)) {
      sprintf(" [%.1f/%.1f GB free]", mem$free / 1e9, mem$total / 1e9)
    } else ""
    sprintf("device:        [%d] %s%s", idx, name, memstr)
  }

  # --- Generation-log summary -------------------------------------------------
  # Distills the raw sd.cpp INFO log (already accumulated in log_file) into the
  # signals that matter for "why is this slow": which device was picked, whether
  # the flash-attention fast path engaged, and the per-stage wall times that
  # sd.cpp prints as "<stage> completed, taking X.XXs". This is the per-section
  # view of test_sampling_profile.R, built from the stage timings sd.cpp already
  # emits (per-op Vulkan timings need GGML_VK_PERF_LOGGER, which writes to the R
  # console, not this file, and is unsafe from the async worker thread).
  summarize_gen_log <- function(path, dev_idx = 0L) {
    if (!file.exists(path)) return(character(0))
    lines <- readLines(path, warn = FALSE)
    out <- c("=== Summary ===")

    dev  <- grep("Selected main device:", lines, value = TRUE)
    if (length(dev)) out <- c(out, sub(".*Selected main device:", "device:", dev[1]))

    # Flash-attention status. sd.cpp does not reliably print a "Using flash
    # attention" line for every architecture (e.g. flux2), so query the device
    # capability directly (coopmat1_fa_support) as the source of truth, and use
    # any sd.cpp "flash attention" line only as secondary confirmation.
    fa_cap <- tryCatch(
      isTRUE(ggmlR::ggml_vulkan_device_caps(dev_idx)$coopmat1_fa_support),
      error = function(e) NA)
    fa_log <- any(grepl("flash attention", lines, ignore.case = TRUE))
    fa <- if (isTRUE(fa_cap) || fa_log) {
      sprintf("flash-attn: ON (coopmat path%s)",
              if (fa_log) ", confirmed in log" else " per device caps")
    } else if (is.na(fa_cap)) {
      "flash-attn: unknown (could not query device caps)"
    } else {
      "flash-attn: not available on this device"
    }
    out <- c(out, fa)

    # Text-encoder backend (CPU vs GPU). sd.cpp logs the encoder compute buffer
    # as "<name> compute buffer size: N MB(RAM)" or "...MB(VRAM)" — RAM means the
    # encoder ran on CPU (keep_clip_on_cpu, or the platform default), VRAM means
    # it ran on the GPU. This is THE signal that separates a healthy run from the
    # Windows VRAM-spill case, so surface it explicitly.
    te_line <- grep("(qwen|qwen3|mistral|t5|clip|llm).*compute buffer size:.*MB\\((RAM|VRAM)\\)",
                    lines, value = TRUE, ignore.case = TRUE)
    if (length(te_line)) {
      where <- sub(".*MB\\((RAM|VRAM)\\).*", "\\1", te_line[1])
      out <- c(out, sprintf("text encoder backend: %s",
                            if (identical(where, "RAM")) "CPU (RAM)" else "GPU (VRAM)"))
    }

    # All "<label> completed/decoded, taking X.XXs" stage timings, in order.
    # Strip the leading "file.cpp:NNN - " source location sd.cpp prepends.
    pat  <- "(.+?)(?: completed| decoded)?, taking ([0-9.]+)s"
    hits <- regmatches(lines, regexec(pat, lines))
    rows <- Filter(function(m) length(m) == 3, hits)
    if (length(rows)) {
      out <- c(out, "", "stage timings (from sd.cpp):")
      for (m in rows) {
        label <- trimws(m[[2]])
        label <- sub("^[A-Za-z0-9_.-]+:[0-9]+\\s*-\\s*", "", label)  # drop file:line -
        out <- c(out, sprintf("  %-40s %8.2fs", label, as.numeric(m[[3]])))
      }
    }

    total <- grep("generate_image completed in", lines, value = TRUE)
    if (length(total)) {
      tt <- sub(".*completed in ([0-9.]+)s.*", "\\1", total[length(total)])
      out <- c(out, "", sprintf("  %-46s %8.2fs", "TOTAL generate_image", as.numeric(tt)))
    }
    out
  }

  # --- Vulkan capabilities inspector (ported from ggmlR vulkan_caps.R) -------
  # Prints the same report into a string. Used by the "GPU caps" toggle to show
  # coopmat / flash-attention / bf16 support — the key signals for diffusion
  # speed (a missing coopmat1_fa_support means flash-attn silently falls back).
  collect_vulkan_caps <- function() {
    capture.output({
      # --- CPU / build capabilities ----------------------------------------
      # These come from the linked libggml.a (ggmlR), NOT from sd2R's own
      # compile flags — the CPU math kernels (incl. the text encoder when it
      # runs on CPU) live there. A Windows build of ggmlR missing AVX2/FMA or
      # OPENMP makes CPU text_encode collapse to scalar/single-thread and can
      # take minutes (observed: 584s on a strong CPU). This block is the first
      # thing to check when text_encode is slow with the encoder on CPU.
      cat("=== CPU / Build Capabilities (from libggml.a) ===\n\n")
      si <- tryCatch(sd2R::sd_system_info(), error = function(e) NULL)
      if (!is.null(si)) {
        cat("sd2R version :", si$sd2R_version, "\n")
        cat("sd.cpp build :", si$sd_cpp_version, "\n")
        cat("CPU cores    :", si$num_cores, "\n")
        cat("ggml string  :", trimws(si$system_info), "\n")
      }
      cf <- tryCatch(ggmlR::ggml_cpu_features(), error = function(e) NULL)
      if (!is.null(cf)) {
        cat("ggml version :", tryCatch(ggmlR::ggml_version(),
                                       error = function(e) "?"), "\n")
        flag <- function(x) if (isTRUE(cf[[x]])) "YES" else "no"
        # The four that actually move the needle for CPU text_encode speed.
        cat("\n  --- key CPU flags ---\n")
        cat(sprintf("  OPENMP : %s   (multi-thread matmul; OFF => single-thread)\n",
                    if (grepl("OPENMP = 1", si$system_info %||% "")) "YES" else "no"))
        cat(sprintf("  AVX2   : %s   (vectorized matmul; OFF => scalar, ~slow)\n", flag("avx2")))
        cat(sprintf("  FMA    : %s\n", flag("fma")))
        cat(sprintf("  F16C   : %s   (fast f16<->f32 for quantized weights)\n", flag("f16c")))
        cat(sprintf("  AVX512 : %s\n", flag("avx512")))
        cat("\n")
      }

      cat("=== Vulkan Device Capabilities ===\n\n")
      if (!ggmlR::ggml_vulkan_available()) {
        cat("Vulkan: NOT COMPILED\n")
        cat("  Reinstall ggmlR with libvulkan-dev + glslc.\n")
        return(invisible())
      }
      cat("Vulkan: compiled OK\n")
      n <- ggmlR::ggml_vulkan_device_count()
      cat("Devices found:", n, "\n\n")
      if (n == 0) {
        cat("No Vulkan devices. Check driver installation.\n")
        return(invisible())
      }
      for (i in seq_len(n)) {
        idx  <- i - 1L
        desc <- tryCatch(ggmlR::ggml_vulkan_device_description(idx),
                         error = function(e) sprintf("<device %d>", idx))
        mem  <- tryCatch(ggmlR::ggml_vulkan_device_memory(idx),
                         error = function(e) NULL)
        # ggml_vulkan_device_caps() spins up a temporary Vulkan logical device
        # (createDevice). On a near-full GPU that throws vk::InitializationFailed
        # error — which, uncaught, terminates the whole R/Shiny process. Catch it
        # so the diagnostic (whose job is to *help*) never kills the app, and
        # report low VRAM as the likely cause.
        caps <- tryCatch(ggmlR::ggml_vulkan_device_caps(idx),
                         error = function(e) NULL)
        cat(sprintf("Device [%d]: %s\n", idx, desc))
        if (!is.null(mem)) {
          cat(sprintf("  Memory : %.2f GB free / %.2f GB total\n",
                      mem$free / 1e9, mem$total / 1e9))
        } else {
          cat("  Memory : <unavailable>\n")
        }
        if (is.null(caps)) {
          cat("\n  --- Capabilities ---\n")
          cat("  could not query device caps (createDevice failed).\n")
          if (!is.null(mem) && mem$free < 2e9) {
            cat(sprintf("  Likely cause: only %.2f GB VRAM free — release loaded\n",
                        mem$free / 1e9))
            cat("  models (rm(ctx); gc()) and retry.\n")
          }
          cat("\n")
          next
        }
        cat("\n  --- Capabilities ---\n")
        cat(sprintf("  arch               : %s\n", caps$arch))
        cat(sprintf("  fp16               : %s   (fast inference)\n",
                    if (caps$fp16) "YES" else "NO"))
        cat(sprintf("  bf16               : %s   (Flux/SD3 native BF16)\n",
                    if (caps$bf16) "YES" else "NO"))
        cat(sprintf("  integer_dot_product: %s   (Q4/Q8 GEMM)\n",
                    if (caps$integer_dot_product) "YES" else "NO"))
        cat(sprintf("  coopmat_support    : %s   (fast GEMM kernels)\n",
                    if (caps$coopmat_support) "YES" else "NO"))
        cat(sprintf("  coopmat1_fa_support: %s   (flash-attention path)\n",
                    if (caps$coopmat1_fa_support) "YES" else "NO"))
        cat(sprintf("  subgroup_size      : %d\n", caps$subgroup_size))
        if (caps$coopmat_support && caps$coopmat_m > 0) {
          cat(sprintf("  coopmat tile       : M=%d N=%d K=%d\n",
                      caps$coopmat_m, caps$coopmat_n, caps$coopmat_k))
        }

        # --- Direct FLASH_ATTN_EXT support probe -----------------------------
        # caps$coopmat1_fa_support only reflects the coopmat-v1 FA path. On
        # Ampere+/Blackwell NVIDIA the FA path is coopmat2, which that flag
        # does NOT capture. The only honest signal is to build a real
        # flash_attn_ext node and ask the backend the exact question sd2R asks
        # per attention layer (ggml_extend.hpp: ggml_backend_supports_op). If
        # this says NO, diffusion_flash_attn silently falls back to F32 attn.
        cat("\n  --- Flash-attention op probe (supports_op) ---\n")
        fa_probe <- tryCatch({
          # Locate the matching backend device by description.
          dev <- NULL
          ndev <- ggmlR::ggml_backend_dev_count()
          for (d in seq_len(ndev) - 1L) {
            dd <- ggmlR::ggml_backend_dev_get(d)
            if (identical(ggmlR::ggml_backend_dev_description(dd), desc)) {
              dev <- dd; break
            }
          }
          if (is.null(dev)) {
            cat("  device handle not found via backend registry — skipped\n")
          } else {
            # Probe the two head dims that actually occur in diffusion models:
            # 64 (SD1.x/SDXL/Flux DiT) and 128 (some Flux/SD3 blocks).
            #
            # The tensor types MUST mirror what build_kqv() in ggml_extend.hpp
            # actually feeds to ggml_flash_attn_ext at runtime, otherwise this
            # probe lies. The ggmlR Vulkan FA kernel requires Q in F32 and
            # K/V in F16 (ggml-vulkan supports_op + shader assert q->type==F32).
            # Building Q as F16 here is what previously made the probe always
            # report NOT SUPPORTED even though FA was in fact available.
            for (hd in c(64L, 128L)) {
              pctx <- ggmlR::ggml_init(16L * 1024L * 1024L, no_alloc = TRUE)
              on.exit(ggmlR::ggml_free(pctx), add = TRUE)
              n_head <- 8L; seq_len <- 256L
              q <- ggmlR::ggml_new_tensor_4d(pctx, ggmlR::GGML_TYPE_F32, hd, n_head, seq_len, 1L)
              k <- ggmlR::ggml_new_tensor_4d(pctx, ggmlR::GGML_TYPE_F16, hd, n_head, seq_len, 1L)
              v <- ggmlR::ggml_new_tensor_4d(pctx, ggmlR::GGML_TYPE_F16, hd, n_head, seq_len, 1L)
              fa <- ggmlR::ggml_flash_attn_ext(pctx, q, k, v, NULL,
                                               1.0 / sqrt(hd), 0.0, 0.0)
              ok <- ggmlR::ggml_backend_dev_supports_op(dev, fa)
              cat(sprintf("  FLASH_ATTN_EXT head_dim=%-3d : %s\n",
                          hd, if (isTRUE(ok)) "SUPPORTED" else "NOT SUPPORTED (fallback)"))
            }
          }
          TRUE
        }, error = function(e) {
          cat(sprintf("  probe error: %s\n", conditionMessage(e)))
          FALSE
        })

        cat("\n  --- Verdict ---\n")
        if (caps$fp16 && caps$coopmat1_fa_support) {
          cat("  BEST: coopmat flash-attention path active (fastest)\n")
        } else if (caps$fp16 && caps$coopmat_support) {
          cat("  GOOD: coopmat GEMM path, NO flash-attention\n")
          cat("        -> diffusion attention falls back to F32 (slow).\n")
        } else if (caps$fp16) {
          cat("  OK:   FP16 active, no coopmat (scalar/subgroup shaders)\n")
        } else {
          cat("  WARN: FP32 only - slow, check driver/device support\n")
        }
        cat("\n")
      }
    }, type = "output")
  }

  observeEvent(input$gpu_caps, {
    if (isTRUE(rv$show_caps)) {
      rv$show_caps <- FALSE          # toggle back to the image
      updateActionButton(session, "gpu_caps", label = "GPU caps")
      return()
    }
    rv$caps_text <- tryCatch(
      paste(collect_vulkan_caps(), collapse = "\n"),
      error = function(e) paste("GPU caps error:", conditionMessage(e)))
    rv$show_caps <- TRUE
    updateActionButton(session, "gpu_caps", label = "Hide caps")
  })

  output$result_image <- renderUI({
    rv$image_trigger  # reactive dependency to re-render on new image

    # Caps toggle takes over the whole pane (either caps OR image).
    if (isTRUE(rv$show_caps)) {
      return(tags$pre(
        style = paste("text-align:left; white-space:pre-wrap;",
                      "background:#1a1a2e; color:#e0e0e0; padding:14px;",
                      "border-radius:6px; font-size:0.85em; overflow:auto;"),
        rv$caps_text))
    }

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

  # Show the "download log" button only when logging is on and a log exists.
  output$download_log_ui <- renderUI({
    rv$log_ready
    # Show after either a logged generation or a profile run produced a file.
    if (!is.null(rv$log_ready) && file.exists(log_file) &&
        file.info(log_file)$size > 0) {
      downloadButton("download_log", "Download log", class = "btn-block",
                     style = "width: 100%;")
    }
  })

  output$download_log <- downloadHandler(
    filename = function() {
      paste0("sd2R_gen_log_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt")
    },
    content = function(file) {
      if (file.exists(log_file)) file.copy(log_file, file, overwrite = TRUE)
    }
  )
}

shinyApp(ui, server)
