# sd2R 0.2.1

* **Diagnostic logging**: `sd_ctx()` now logs all behavioural parameters from R at startup, the resolved weight-residency mode (resident vs streamed when `offload_params_to_cpu`/`params_backend` put params on a different backend), and a running summary of weight data re-uploaded to the GPU each pass — making silent performance regimes visible.
* FLUX.2 (Klein 4B) support via `model_type = "flux2"`, with auto-detection from tensors/filename.
* New `llm_path` argument in `sd_ctx()` for LLM text encoders (Qwen3 for FLUX.2 Klein, Mistral-Small for full FLUX.2).
* Inpainting: new `mask` argument in `sd_img2img()` regenerates only the masked region. Accepts a PNG path, a numeric matrix, or an SD image (white = generate). Works on plain SD/SDXL/FLUX 1/2 weights via the denoise mask. New helper `sd_load_mask()`.
* Shiny GUI now shares `sd_generate()`'s auto-routing (CFG, VAE tiling, highres-fix), fixing FLUX.2 VAE-decode crashes.
* New `meta_backend` argument in `sd_ctx()`: runs the diffusion model through the ggml meta backend for multi-GPU tensor split (a single model sharded across all GPUs). Requires ggmlR >= 0.7.8; falls back to the normal single-backend path otherwise. The Shiny GUI enables it automatically for FLUX.2.

---

# sd2R 0.2.0

## Performance: VAE Decode

* `vae_conv_direct` now defaults to `TRUE` — VAE decoder uses `GGML_OP_CONV_2D`
  (direct convolution via `conv2d_mm.comp`) instead of the legacy IM2COL + MUL_MAT path.
* On RX 9070 (RDNA4) with Vulkan coopmat (KHR): VAE decode 12.6s → 0.5s for 768×768.
* All convolutions now run through the coopmat cm1 path (~16-17 TFLOPS/s) when
  `coopmat_support` is available; scalar FMA fallback used otherwise.

---

# sd2R 0.1.9

## Shiny GUI
* New `sd_app()` launches an interactive Shiny application for image generation.
  - Auto-detection of model architecture (Flux, SD3, SDXL, SD1/2) from filenames
    in the models folder — no manual configuration needed.
  - Non-blocking async generation via C++ `std::thread`: the UI remains responsive
    during image generation, with a live progress bar and ETA display.
  - Automatic role assignment for multi-file models (diffusion, VAE, CLIP-L, T5-XXL).
  - Prevents loading incompatible model combinations (e.g. SD1.5 + Flux).

## Async C++ Generation API
* New internal functions for non-blocking generation from R:
  - `sd_generate_async()` — launches generation in a background C++ thread.
  - `sd_generate_poll()` — checks completion status (atomic flags).
  - `sd_generate_result()` — retrieves results after completion.
* Progress callback writes JSON to a temp file (`step`, `steps`, `pct`,
  `elapsed`, `eta_sec`), read by Shiny via `later::later()` polling.
* R API calls (`Rprintf`, `R_CheckUserInterrupt`) are suppressed in the
  worker thread to prevent stack corruption.

## Build System
* `tools/patch_sd_sources.sh` rewritten: all `sed` calls replaced with
  `perl -pi -e` for cross-platform compatibility (macOS BSD sed + Linux GNU sed).

---

# sd2R 0.1.8

## Bug Fixes
* Fixed `undefined symbol: ggml_backend_vk_get_device_count` load error on
  CRAN Fedora (clang and gcc). Root cause: ggmlR's shared library (`ggmlR.so`)
  was built with Vulkan, but the static library (`libggml.a`) shipped without
  Vulkan objects. The old `configure` relied on `ggml_vulkan_status()` which
  queries `ggmlR.so` — it reported "AVAILABLE", causing sd2R to compile with
  `-DSD_USE_VULKAN` against a `libggml.a` that lacked the symbols.
  Now `configure` checks `nm libggml.a` for a defined (`T`) symbol directly,
  ignoring the runtime ggmlR check entirely.

---

# sd2R 0.1.7

## Multi-GPU Model Parallelism
* New `device_layout` parameter in `sd_ctx()`: distribute sub-models across
  multiple Vulkan GPUs without separate processes.
  - `"mono"` — all on one GPU (default, backward-compatible).
  - `"split_encoders"` — CLIP/T5 on GPU 1, diffusion + VAE on GPU 0.
  - `"split_vae"` — CLIP/T5 + VAE on GPU 1, diffusion on GPU 0.
  - `"encoders_cpu"` — text encoders on CPU, diffusion + VAE on GPU.
* Low-level `diffusion_gpu`, `clip_gpu`, `vae_gpu` integer arguments for
  manual device assignment (override presets).

## Profiling
* New profiling API for per-stage timing of image generation:
  - `sd_profile_start()` / `sd_profile_stop()` — control event capture.
  - `sd_profile_get()` — raw event data frame.
  - `sd_profile_summary()` — formatted summary with durations and percentages.
* Stages tracked: `text_encode` (with `text_encode_clip` and `text_encode_t5`
  sub-stages), `sampling`, `vae_decode`, `vae_encode`, model loading.
* Pretty-printed output via `print.sd_profile()`.

---

# sd2R 0.1.6

## Pipeline Graph API
* New `sd_pipeline()` / `sd_node()` — sequential graph-based pipeline.
  Node types: `"txt2img"`, `"img2img"`, `"upscale"`, `"save"`.
* `sd_run_pipeline(pipeline, ctx)` — execute pipeline with a single context.
* `sd_save_pipeline()` / `sd_load_pipeline()` — JSON serialization.

---

# sd2R 0.1.5

## Flux Support
* Flux model family (flux1-dev, etc.) fully supported: text-to-image,
  image-to-image, highres fix, tiled sampling, multi-GPU.
* Separate model paths: `diffusion_model_path`, `vae_path`, `clip_l_path`,
  `t5xxl_path` in `sd_ctx()`.
* `cfg_scale` auto-defaults to 1.0 for Flux (guidance-distilled models).

## img2img Improvements
* `sd_generate()` now defaults `width`/`height` to init image dimensions
  when not specified explicitly.

---

# sd2R 0.1.4

## Build System
* `configure.win` rewritten to use template approach (`Makevars.win.in` →
  `Makevars.win`), matching `ggmlR` pattern.

---

# sd2R 0.1.3

## Unified `sd_generate()` Entry Point
* New `sd_generate()` — single function for all generation modes. Automatically
  selects the optimal strategy (direct, tiled sampling, or highres fix) based
  on output resolution and available VRAM.
* `vram_gb` parameter in `sd_ctx()`: set once, auto-routing handles the rest.

## Multi-GPU
* New `sd_generate_multi_gpu()` — parallel generation across multiple Vulkan
  GPUs via `callr`, one process per GPU, with progress reporting.

## Performance
* Batch compute optimization for tiled sampling: pre-allocated compute context
  buffer eliminates ~110 MB malloc/free per UNet call.

---

# sd2R 0.1.2

## Highres Fix
* New `sd_highres_fix()` — classic two-pass highres pipeline:
  txt2img at native resolution → upscale → tiled img2img refinement.
* `hr_strength` parameter (default 0.4) controls refinement intensity.

## Tiled img2img
* New `sd_img2img_tiled()` — img2img with MultiDiffusion tiled sampling for
  large images.

---

# sd2R 0.1.1

## VAE Tiling
* New `vae_mode` parameter: `"normal"`, `"tiled"`, `"auto"` (default).
  Auto-tiles when image area exceeds threshold.
* `vae_tile_rel_x` / `vae_tile_rel_y` for adaptive tile sizing.

## High-Resolution Pipeline
* New `sd_txt2img_highres()` — patch-based generation for 2K, 4K+ images.
* `model_type` parameter in `sd_ctx()`: `"sd1"`, `"sd2"`, `"sdxl"`, `"flux"`,
  `"sd3"`.

## Tiled Sampling (MultiDiffusion)
* New `sd_txt2img_tiled()` — tiled diffusion sampling at any resolution.
  VRAM bounded by tile size, not output resolution.

---

# sd2R 0.1.0

## Core
* Text-to-image generation via stable-diffusion.cpp (C++ backend).
* Support for SD 1.x, SD 2.x, SDXL model versions.
* SafeTensors and GGUF model format loading.
* Vulkan GPU backend via ggmlR.
* Samplers: Euler, Euler A, Heun, DPM2, DPM++ (2M), LCM, DDIM, TCD.
* Schedulers: Discrete, Karras, Exponential, Simple, SGM Uniform, AYS, LCM.

## R API
* `sd_ctx()` — create model context.
* `sd_generate()` — unified entry point.
* `sd_txt2img()`, `sd_img2img()` — low-level generation.
* `sd_save_image()`, `sd_system_info()`.
