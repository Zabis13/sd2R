# sdR

[![R-hub check on the R Consortium cluster](https://github.com/r-hub2/separate-jaguar-sdR/actions/workflows/rhub-rc.yaml/badge.svg)](https://github.com/r-hub2/separate-jaguar-sdR/actions/workflows/rhub-rc.yaml)

**sdR** is an R package that provides a native, GPU-accelerated Stable Diffusion pipeline by wrapping the C++ implementation from [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and using [ggmlR](https://github.com/Zabis13/ggmlR) as the tensor backend.

## Overview

sdR exposes a high-level R interface for text-to-image and image-to-image generation, while all heavy computation (tokenization, encoders, denoiser, sampler, VAE, model loading) is implemented in C++. Supports SD 1.x, SD 2.x, SDXL, and Flux model families. Targets local inference on Linux with Vulkan-enabled AMD GPUs (with automatic CPU fallback via ggml), without relying on external Python or web APIs.

## Architecture

- **C++ core** (`src/sd/`): tokenizers, text encoders (CLIP, Mistral, Qwen, UMT5), diffusion UNet/MMDiT denoiser, samplers, VAE encoder/decoder, and model loading for `.safetensors` and `.gguf` weights.
- **R layer**: user-facing pipeline functions, parameter validation, image helpers, testing, and documentation-friendly API.
- **Backend**: links against ggmlR (headers via `LinkingTo`) and `libggml.a`, reusing the same GGML/Vulkan stack that also powers llamaR and other ggmlR-based packages.

## Key Features

- **Unified `sd_generate()`** — single entry point for all generation modes. Automatically selects the optimal strategy (direct, tiled sampling, or highres fix) based on output resolution and available VRAM (`vram_gb` parameter in `sd_ctx()`). Users don't need to think about tiling at all.
- **CRAN-ready defaults**: `verbose = FALSE` by default — no console output unless explicitly enabled. Cross-platform build system with `configure`/`configure.win` generating `Makevars` from templates.
- **VRAM-aware auto-routing**: estimates VRAM from resolution and routes to direct generation (fits in VRAM), highres fix (txt2img + upscale + tiled img2img, preferred for coherent large images), or tiled sampling (MultiDiffusion fallback). Set `vram_gb` once in `sd_ctx()`.
- **Multi-GPU**: `sd_generate_multi_gpu()` distributes prompts across Vulkan GPUs via `callr`, one process per GPU, with progress reporting.
- **Text-to-image** generation supporting Stable Diffusion 1.x, 2.x, SDXL, and Flux models with typical generations taking a few seconds on Vulkan-enabled GPUs.
- **Image-to-image** workflows with noise strength control and reuse of the same denoising pipeline as text-to-image. Requires `vae_decode_only = FALSE` in context.
- **Optional upscaling** using a dedicated upscaler context managed entirely in C++ and exposed to R through external pointers.
- **Tiled VAE** for high-resolution images (2K, 4K+) with bounded VRAM usage. `vae_mode = "auto"` enables tiling automatically when image area exceeds a configurable threshold. Supports per-axis relative tile sizing (`vae_tile_rel_x`, `vae_tile_rel_y`) for non-square aspect ratios.
- **Tiled diffusion sampling** (MultiDiffusion): at each denoising step the latent is split into overlapping tiles, each denoised independently, and merged with Gaussian weighting. VRAM usage scales with tile size, not output resolution.
- **Highres Fix**: classic two-pass pipeline — generates base image at native model resolution, upscales (bilinear or ESRGAN), then refines with tiled img2img at low denoising strength. Produces coherent high-resolution images (2K, 4K+) with global composition preserved.
- **Image utilities** in R: saving generated images to PNG, converting between internal tensors and R raw vectors, and simple inspection of output tensors.
- **System introspection** via `sd_system_info()`, reporting GGML/Vulkan capabilities as detected by ggmlR at build time.

## Implementation Details

- **Rcpp bindings**: `src/sdR_interface.cpp` defines a thin bridge between R and the C API in `stable-diffusion.h`, returning `XPtr` objects with custom finalizers for correct lifetime management of `sd_ctx_t` and `upscaler_ctx_t`.
- **Build system**: `configure` / `configure.win` generate `Makevars` from `.in` templates, resolving ggmlR paths, OpenMP, and Vulkan at configure time. Per-target `-include r_ggml_compat.h` applied only to `sd/*.cpp` sources to avoid macro conflicts with system headers.
- **Package metadata**: `DESCRIPTION` declares Rcpp and ggmlR in `LinkingTo`, and `NAMESPACE` is generated via roxygen2 with `useDynLib` and Rcpp imports.
- **On load**: `.onLoad()` initializes logging and registers constant values that mirror the underlying C++ enums using 0-based indices.

## CRAN Readiness

- `verbose = FALSE` by default — no output unless requested.
- Per-target compiler flags for cross-platform compatibility (Linux, macOS, Windows).
- All C++ warnings fixed (`-Winconsistent-missing-override`, deprecated `codecvt`).
- Large tokenizer vocabularies (CLIP, Mistral, Qwen, UMT5) downloaded automatically during installation from [GitHub Releases](https://github.com/Zabis13/sdR/releases/tag/assets), keeping the source tarball small.

## Installation

```r
# Install ggmlR first (if not already installed)
remotes::install_github("Zabis13/ggmlR")

# Install sdR
remotes::install_github("Zabis13/sdR")
```

During installation, the `configure` script automatically downloads tokenizer vocabulary files (~128 MB total) from GitHub Releases. This requires `curl` or `wget`.

### Offline / Manual Installation

If you don't have internet access during installation, download the vocabulary files manually and place them into `src/sd/` before building:

```sh
# Download from https://github.com/Zabis13/sdR/releases/tag/assets
# Files: vocab.hpp, vocab_mistral.hpp, vocab_qwen.hpp, vocab_umt5.hpp

wget https://github.com/Zabis13/sdR/releases/download/assets/vocab.hpp -P src/sd/
wget https://github.com/Zabis13/sdR/releases/download/assets/vocab_mistral.hpp -P src/sd/
wget https://github.com/Zabis13/sdR/releases/download/assets/vocab_qwen.hpp -P src/sd/
wget https://github.com/Zabis13/sdR/releases/download/assets/vocab_umt5.hpp -P src/sd/

R CMD INSTALL .
```

## System Requirements

- R ≥ 4.1.0, C++17 compiler
- `curl` or `wget` (for downloading vocabulary files during installation)
- **Optional GPU**: `libvulkan-dev` + `glslc` (Linux) or Vulkan SDK (Windows)
- Platforms: Linux, macOS, Windows (x86-64, ARM64)

## See Also

- [llamaR](https://github.com/Zabis13/llamaR) — LLM inference in R
- [sdR](https://github.com/Zabis13/sdR) — Stable Diffusion in R
- [ggml](https://github.com/ggml-org/ggml) — underlying C library

## License

MIT


