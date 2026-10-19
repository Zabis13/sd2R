# sd2R

[![R-hub check on the R Consortium cluster](https://github.com/r-hub2/separate-jaguar-sd2R/actions/workflows/rhub-rc.yaml/badge.svg)](https://github.com/r-hub2/separate-jaguar-sd2R/actions/workflows/rhub-rc.yaml)

**sd2R** is an R package that provides a native, GPU-accelerated Stable Diffusion pipeline by wrapping the C++ implementation from [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and using [ggmlR](https://github.com/Zabis13/ggmlR) as the tensor backend.

## Overview

sd2R exposes a high-level R interface for text-to-image and image-to-image generation, while all heavy computation (tokenization, encoders, denoiser, sampler, VAE, model loading) is implemented in C++. Supports SD 1.x, SD 2.x, SDXL, Flux, FLUX.2 (Klein), and FLUX.3 (Image Klein) model families. Targets local inference on Linux with Vulkan-enabled AMD GPUs (with automatic CPU fallback via ggml), without relying on external Python or web APIs.

## Architecture

Flux without Python:

```
R  →  sd2R  →  ggmlR  →  Vulkan  →  GPU
```

- **C++ core** (`src/sd/`, stable-diffusion.cpp master 2026-10): tokenizers, text encoders (CLIP, Mistral, Qwen, UMT5), diffusion UNet/MMDiT denoiser, samplers, VAE encoder/decoder, and model loading for `.safetensors` and `.gguf` weights. Weights are loaded in `sd_ctx()` by default; `eager_load = FALSE` defers loading to first use.
- **R layer**: user-facing pipeline functions, parameter validation, image helpers, testing, and documentation-friendly API.
- **Backend**: links against ggmlR (headers via `LinkingTo`) and `libggml.a`, reusing the same GGML/Vulkan stack that also powers llamaR and other ggmlR-based packages.

## Key Features

- **Unified `sd_generate()`** — single entry point for all generation modes. Automatically selects the optimal strategy (direct, tiled sampling, or highres fix) based on output resolution and available VRAM (`vram_gb` parameter in `sd_ctx()`). Users don't need to think about tiling at all.
- **CRAN-ready defaults**: `verbose = FALSE` by default — no console output unless explicitly enabled. Cross-platform build system with `configure`/`configure.win` generating `Makevars` from templates.
- **VRAM-aware auto-routing**: queries free GPU memory at runtime and routes to direct generation (fits in VRAM), highres fix (txt2img + upscale + tiled img2img, preferred for coherent large images), or tiled sampling (MultiDiffusion fallback). VAE tiling is also VRAM-aware — enabled automatically only when free memory is insufficient for the given resolution. Set `vram_gb` in `sd_ctx()` to override auto-detection.
- **Multi-GPU data parallelism**: `sd_generate_multi_gpu()` distributes prompts across Vulkan GPUs via `callr`, one process per GPU, with progress reporting.
- **Multi-GPU model parallelism**: `device_layout` parameter in `sd_ctx()` distributes sub-models across multiple Vulkan GPUs within a single process. Presets: `"mono"` (all on one GPU), `"split_encoders"` (CLIP/T5 on GPU 1, diffusion + VAE on GPU 0), `"split_vae"` (CLIP/T5 + VAE on GPU 1, diffusion on GPU 0), `"encoders_cpu"` (text encoders on CPU). Manual override via `diffusion_gpu`, `clip_gpu`, `vae_gpu`.
- **Multi-GPU tensor split**: `meta_backend = TRUE` in `sd_ctx()` shards a single diffusion model across all available GPUs via the ggml meta backend (for models too large for one GPU). Requires ggmlR >= 0.7.8; falls back to the normal single-backend path otherwise.
- **Profiling**: built-in per-stage timing via `sd_profile_start()` / `sd_profile_stop()` / `sd_profile_summary()`. Tracks model loading, text encoding (with CLIP/T5 breakdown), sampling, and VAE decode/encode stages.
- **Text-to-image** generation supporting Stable Diffusion 1.x, 2.x, SDXL, Flux, FLUX.2 (Klein), and FLUX.3 (Image Klein) models with typical generations taking a few seconds on Vulkan-enabled GPUs.
- **Image-to-image** workflows with noise strength control and reuse of the same denoising pipeline as text-to-image. Requires `vae_decode_only = FALSE` in context.
- **Inpainting**: the `mask` argument of `sd_img2img()` regenerates only the masked region while preserving the rest. Accepts a PNG path, a numeric matrix, or an SD image (white = generate, black = keep); `sd_load_mask()` loads a mask file. Works on plain SD/SDXL/FLUX 1/2 weights — no dedicated inpaint model required.
- **Optional upscaling** using a dedicated upscaler context managed entirely in C++ and exposed to R through external pointers.
- **VRAM-aware Tiled VAE** for high-resolution images (2K, 4K+) with bounded VRAM usage. `vae_mode = "auto"` (default) queries free GPU memory before VAE decode and enables tiling only when estimated peak usage exceeds available VRAM (with a 50 MB safety reserve). Falls back to a pixel-area threshold (`vae_auto_threshold`) when Vulkan memory query is unavailable (CPU backend, no GPU). Supports per-axis relative tile sizing (`vae_tile_rel_x`, `vae_tile_rel_y`) for non-square aspect ratios.
- **Tiled diffusion sampling** (MultiDiffusion): at each denoising step the latent is split into overlapping tiles, each denoised independently, and merged with Gaussian weighting. VRAM usage scales with tile size, not output resolution.
- **Highres Fix**: classic two-pass pipeline — generates base image at native model resolution, upscales (bilinear or ESRGAN), then refines with tiled img2img at low denoising strength. Produces coherent high-resolution images (2K, 4K+) with global composition preserved.
- **Image utilities** in R: saving generated images to PNG, converting between internal tensors and R raw vectors, and simple inspection of output tensors.
- **System introspection** via `sd_system_info()`, reporting GGML/Vulkan capabilities as detected by ggmlR at build time.
- **Pipeline graph API**: `sd_pipeline()` + `sd_node()` for composable, sequential multi-step workflows (txt2img → upscale → img2img → save). Pipelines are serializable to JSON via `sd_save_pipeline()` / `sd_load_pipeline()`.
- **Shiny GUI**: `sd_app()` launches an interactive web interface with auto-detection of model architecture, non-blocking async generation (C++ `std::thread`), live progress bar with ETA, and automatic role assignment for multi-file models (Flux, FLUX.2, FLUX.3, SD3).

## Shiny GUI

Launch an interactive web interface for image generation:

```r
# From an R session
sd_app()                                # random port, opens browser
sd_app(model_dir = "/path/to/models")   # pre-scan a model folder
sd_app(port = 3838, host = "127.0.0.1") # fixed port/host
```

From the terminal (one-liners):

```bash
# Simplest
Rscript -e 'sd2R::sd_app()'

# Fixed port + local host, open browser
Rscript -e 'sd2R::sd_app(port = 3838, host = "127.0.0.1", launch.browser = TRUE)'

# Equivalent low-level call (no sd2R helpers)
Rscript -e "shiny::runApp(system.file('shiny/sd2R_app', package = 'sd2R'), port = 3838, host = '127.0.0.1', launch.browser = TRUE)"
```

Features:
- Auto-detects model architecture (Flux, FLUX.2, FLUX.3, SD3, SDXL, SD1/2) and assigns component roles (diffusion, VAE, CLIP, T5, LLM encoder)
- Non-blocking generation with live progress bar and ETA
- Shares `sd_generate()`'s auto-routing: guidance-distilled CFG (Flux/FLUX.2/FLUX.3), VRAM-aware VAE tiling, and multi-step highres-fix all run through the async engine
- Prevents incompatible model combinations

## Pipeline Example

```r
pipe <- sd_pipeline(
  sd_node("txt2img", prompt = "a cat in space", width = 512, height = 512),
  sd_node("upscale", factor = 2),
  sd_node("img2img", strength = 0.3),
  sd_node("save", path = "output.png")
)

# Save / load as JSON
sd_save_pipeline(pipe, "my_pipeline.json")
pipe <- sd_load_pipeline("my_pipeline.json")

# Run
ctx <- sd_ctx("model.safetensors")
sd_run_pipeline(pipe, ctx, upscaler_ctx = upscaler)
```

## Quick Start: Download a Ready-to-Use FLUX 2 Model

New to sd2R? Grab a ready-made FLUX 2 model in one line — no Kaggle account, no Python, no manual file juggling. `sd_download_model()` downloads the bundle from a public [Kaggle](https://www.kaggle.com/models) dataset and unpacks it for you:

```r
# Download FLUX 2 (GGUF) into ./models/flux2
sd_download_model(dest = "models/flux2", verbose = TRUE)

# Then launch the GUI pointed at that folder
sd_app(model_dir = "models/flux2")
```

That's it — the app auto-detects the model and you can start generating. Re-running `sd_download_model()` is safe: it skips the download if the folder is already populated.

## FLUX.3 Image Klein

FLUX.3 has its own DiT, uses the FLUX.2 VAE and a Qwen3-VL 4B LLM text encoder (not the plain Qwen3-4B of FLUX.2 Klein). Defaults: 16 steps, `cfg_scale = 1.0`, Euler.

```r
ctx <- sd_ctx(
  diffusion_model_path = "models/flux3_image_klein_sd_Q8_0.gguf",
  vae_path             = "models/flux2-vae.safetensors",
  llm_path             = "models/Qwen3VL-4B-Instruct-Q4_K_M.gguf",
  model_type           = "flux3"
)
img <- sd_generate(ctx, "a cat sitting on a chair, oil painting",
                   width = 768L, height = 768L)

# Edit with up to 10 reference images
edit <- sd_generate_multiref(ctx, "turn the cat into a dog",
                             refs = list(img[[1]]), width = 768L, height = 768L)
```

- **Model manager / GUI**: `sd_scan_models()` and `sd_app()` detect FLUX.3 and attach the LLM encoder (a file with "vl" in its name for FLUX.3, a plain Qwen3 file for FLUX.2).
- **Step-wise API**: flow-map (distilled Klein) checkpoints need the next sigma as target timestep — `sd_denoise_step(..., sigma_next = )`. `sd_euler_step()` does the Euler update in C++ (float), so `sd_sample_stepwise()` matches `sd_sample()` bit for bit.

**VAE tile size** (`vae_tile_size`) is given in image pixels since 0.2.2 (default 512) and converted internally with the model's VAE scale factor (8 for SD/FLUX.1, 16 for FLUX.2/FLUX.3). Older code that passed latent cells (e.g. 64) gets a warning.

## Implementation Details

- **Rcpp bindings**: `src/sd2r_*.cpp` (callbacks, parameters, context, generation, step-wise API, upscaler) define a thin bridge between R and the C API in `stable-diffusion.h`, returning `XPtr` objects with custom finalizers for correct lifetime management of `sd_ctx_t` and `upscaler_ctx_t`.
- **Build system**: `configure` / `configure.win` generate `Makevars` from `.in` templates, resolving ggmlR paths, OpenMP, and Vulkan at configure time. Per-target `-include r_ggml_compat.h` applied only to `sd/*.cpp` sources to avoid macro conflicts with system headers.
- **Package metadata**: `DESCRIPTION` declares Rcpp and ggmlR in `LinkingTo`, and `NAMESPACE` is generated via roxygen2 with `useDynLib` and Rcpp imports.
- **On load**: `.onLoad()` initializes logging and registers constant values that mirror the underlying C++ enums using 0-based indices.

## CRAN Readiness

- `verbose = FALSE` by default — no output unless requested.
- Per-target compiler flags for cross-platform compatibility (Linux, macOS, Windows).
- All C++ warnings fixed (`-Winconsistent-missing-override`, deprecated `codecvt`).
- Large tokenizer vocabularies (CLIP, Mistral, Qwen, UMT5) downloaded automatically during installation from [GitHub Releases](https://github.com/Zabis13/sd2R/releases/tag/assets), keeping the source tarball small.

## Installation

### Linux

```r
# Install ggmlR first (sd2R >= 0.2.2 needs ggmlR >= 0.8.8)
install.packages("ggmlR", configure.args = "--with-simd")

# Install sd2R
install.packages("sd2R")
```

### Linux (detailed)

Full step-by-step setup on Ubuntu/Debian, from a clean system to a working
GPU build.

**1. R and the Vulkan loader/tools:**

```bash
sudo apt install -y r-base

sudo apt install vulkan-tools libvulkan-dev
```

**2. The `glslc` shader compiler** (needed to build ggmlR's Vulkan backend):

```bash
# Ubuntu 24.04 (Noble)
sudo add-apt-repository universe
sudo apt update
sudo apt install glslc

# Ubuntu 22.04 (Jammy) — install the LunarG Vulkan SDK instead
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | \
  sudo tee /etc/apt/trusted.gpg.d/lunarg.asc

sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list \
  https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list

sudo apt update
sudo apt install -y vulkan-sdk
```

**3. Verify the GPU is visible to Vulkan:**

```bash
vulkaninfo --summary
```

**4. Install ggmlR (the tensor/Vulkan backend) with CPU SIMD acceleration:**

```bash
sudo Rscript -e 'install.packages("ggmlR", configure.args = "--with-simd")'
```

**5. Confirm GPU support from R:**

```bash
Rscript -e 'library(ggmlR)
ggml_vulkan_status()'
```

**6. Install sd2R:**

```bash
sudo Rscript -e 'install.packages("sd2R")'
```

or, to build the development version from GitHub:

```bash
sudo apt install -y libcurl4-openssl-dev libssl-dev libgit2-dev
sudo Rscript -e 'install.packages("remotes")'
sudo Rscript -e 'remotes::install_github("Zabis13/sd2R")'
```

Launch the GUI from a terminal:

```
Rscript -e "sd2R::sd_app()"
```

During installation, the `configure` script automatically downloads tokenizer vocabulary files (~128 MB total) from GitHub Releases. This requires `curl` or `wget`.



### Windows (step-by-step)

Tested configuration:

- **R** 4.6.0 ([R-4.6.0-win](https://cran.r-project.org/bin/windows/base/))
- **Rtools45** ([rtools45-6768-6492](https://cran.r-project.org/bin/windows/Rtools/rtools45/))
- **Vulkan SDK** 1.4.350.0 ([vulkansdk-windows-X64-1.4.350.0](https://vulkan.lunarg.com/sdk/home#windows)) — for GPU acceleration

Install R, Rtools45, and the Vulkan SDK (use the default install paths).

**From CRAN — from source with SIMD (recommended; required for GPU):**

Requires Rtools45. Build from source if you want Vulkan **GPU acceleration**:
the build enables Vulkan only when the Vulkan SDK is present at compile time
(`configure.win` auto-detects `VULKAN_SDK`), so install the Vulkan SDK
*before* running the commands below.

SIMD is a *ggmlR* build option, enabled via the `GGML_USE_SIMD` environment
variable. There is **no** `--with-simd` / `--configure-args="..."` flag —
`configure.win` does not parse those, so set the environment variable instead.

```r
# --- ggmlR (tensor/Vulkan backend) with SIMD ---
unlink("C:/Program Files/R/R-4.6.0/library/00LOCK-ggmlR", recursive = TRUE)
Sys.setenv(GGML_USE_SIMD = "1")
install.packages("ggmlR", type = "source")

# --- sd2R ---
unlink("C:/Program Files/R/R-4.6.0/library/00LOCK-sd2R", recursive = TRUE)
Sys.setenv(MAKEFLAGS = "-j8")   # parallel compile; lower on fewer cores
install.packages("sd2R", type = "source")
```

Launch the GUI from a terminal:

```
"C:\Program Files\R\R-4.6.0\bin\Rscript.exe" -e "library(sd2R); sd_app()"
```


## System Requirements

- R ≥ 4.1.0, C++17 compiler
- `curl` or `wget` (for downloading vocabulary files during installation)
- **Optional GPU**: `libvulkan-dev` + `glslc` (Linux) or Vulkan SDK (Windows)
- Platforms: Linux, macOS, Windows (x86-64, ARM64)

## Benchmarks

### FLUX.1-dev Q4_K_S — 10 steps

CLIP-L + T5-XXL text encoders, VAE. `sample_steps = 10`.

| Test | AMD RX 9070 (16 GB) | Tesla P100 (16 GB) | 2x Tesla T4 (16 GB) |
|---|---|---|---|
| 1. 768x768 direct | 13.72 s | 94.0 s | 62.0 s |
| 2. 1024x1024 tiled VAE | 24.84 s | 151.4 s | 105.6 s |
| 3. 2048x1024 highres fix | 42.70 s | 312.5 s | 222.0 s |
| 4. img2img 768x768 direct | 8.16 s | 51.0 s | 32.8 s |
| 5. 1024x1024 direct | 24.90 s | 152.2 s | 112.1 s |
| 6. Multi-GPU 4 prompts | -- | -- | 141.7 s (4 img) |

### FLUX.2 Klein 4B — 4 steps

Qwen3 LLM text encoder + FLUX.2 VAE. `sample_steps = 4`.

RTX 3090 system: CPU Xeon E5-2666 v3, 32 GB RAM (Windows).

| Test | AMD RX 9070 (16 GB) | RTX 3090 (24 GB) (old) |
|---|---|---|
| 1. 768x768 direct | 3.80 s | 5.10 s |
| 2. 1024x1024 tiled VAE | 6.70 s | 8.59 s |
| 3. 2048x1024 highres fix | 13.99 s | 23.54 s |
| 4. img2img 768x768 direct | 2.92 s | 4.34 s |
| 5. 1024x1024 direct | 6.76 s | 8.74 s |

### FLUX.3 Image Klein Q8_0 — 16 steps

Qwen3-VL 4B (Q4_K_M) LLM text encoder + FLUX.2 VAE. `sample_steps = 16`, Euler.
Tiled VAE uses 512 px tiles.

| Test | AMD RX 9070 (16 GB) |
|---|---|
| 1. 768x768 direct | 11.35 s |
| 2. 1024x1024 tiled VAE | 23.70 s |
| 3. 2048x1024 highres fix | 38.69 s |
| 4. img2img 768x768 direct | 5.82 s |
| 5. 1024x1024 direct | 22.46 s |
| 6. Edit 768x768 (1 reference image) | 25.39 s |

### Model size comparison

| | SD 1.5 | Flux Q4_K_S |
|---|---|---|
| Diffusion params | ~860 MB | ~6.5 GB |
| Text encoders | CLIP ~240 MB | CLIP-L + T5-XXL ~3.9 GB |
| Sampling per step (768x768) | ~0.1–0.3 s | ~3.9 s |
| Architecture | UNet | MMDiT (57 blocks) |

## Examples

For a live, runnable demo see the [Kaggle notebook: Stable Diffusion in R (ggmlR + Vulkan GPU)](https://www.kaggle.com/code/lbsbmsu/stable-diffusion-in-r-ggmlr-vulkan-gpu).

## See Also

- [llamaR](https://github.com/Zabis13/llamaR) — LLM inference in R
- [ggmlR](https://github.com/Zabis13/ggmlR) — tensor/Vulkan backend for R
- [ggml](https://github.com/ggml-org/ggml) — underlying C library

## License

MIT


