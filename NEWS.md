# sdR 0.1.3

## Unified `sd_generate()` Entry Point
* New `sd_generate()` — single function for all generation modes. Automatically
  selects the optimal strategy (direct, tiled sampling, or highres fix) based
  on output resolution and available VRAM.
* `vram_gb` parameter in `sd_ctx()`: set once, auto-routing handles the rest.
  If VRAM is sufficient for direct generation — uses direct. Otherwise — tiled
  sampling with default tile size (64 latent = 512px).
* Highres fix auto-selected only when VAE encoder is available
  (`vae_decode_only = FALSE`), otherwise falls back to tiled sampling.
* Low-level functions (`sd_txt2img_tiled()`, `sd_img2img_tiled()`,
  `sd_highres_fix()`, `sd_txt2img_highres()`) moved to internal API
  (`@keywords internal`). Still accessible for advanced users via `sdR:::`.

## Multi-GPU
* New `sd_generate_multi_gpu()` — parallel generation across multiple Vulkan
  GPUs. Each prompt runs in a separate process via `callr`, with
  `SD_VK_DEVICE` selecting the GPU. Worker pool limited to `length(devices)`
  concurrent processes (one per GPU). Prompts distributed round-robin as GPUs
  become free. `progress = TRUE` prints `[3/100] GPU0: done` per completion.
  Requires the `callr` package.

## Performance
* Batch compute optimization for tiled sampling: pre-allocated compute context
  buffer eliminates ~110 MB malloc/free per UNet call. Uses `ggml_reset()`
  instead of `ggml_free()` + `ggml_init()` for context reuse.

## Bug Fixes
* **VRAM estimation formula** corrected: `pixels / 262144 * 4.0 * 1.1` (with
  +10% safety margin). Old formula underestimated VRAM for 1024x1024 (12.5 GB
  vs actual ~17 GB), causing OOM on 16 GB GPUs.
* **VAE auto-tiling threshold** changed from `>` to `>=`: images exactly at
  the threshold (e.g. 1024x1024 = 1048576 pixels) now correctly use tiled VAE
  instead of OOM.
* **Highres fix priority**: for txt2img, highres fix is now preferred over
  tiled sampling when VAE encoder is available (`pixels > native_pixels`,
  was `pixels > native_pixels * 4`). Tiled sampling without base image
  produces incoherent "collage" artifacts.
* `.native_tile_size()` for SD1/SD2 corrected from 768 to 512 (native model
  resolution).

---

# sdR 0.1.2

## Highres Fix
* New `sd_highres_fix()` — classic two-pass highres pipeline:
  1. `sd_txt2img()` at native model resolution (auto-detected from `model_type`)
  2. Upscale to target size (bilinear or ESRGAN if `upscaler_path` given)
  3. `sd_img2img_tiled()` with low denoising strength for detail refinement
* `hr_strength` parameter (default 0.4) controls refinement intensity.
* Automatic base resolution calculation preserving target aspect ratio.

## Tiled img2img
* New `sd_img2img_tiled()` — img2img with MultiDiffusion tiled sampling for
  large images. Same API as `sd_img2img()` plus `sample_tile_size` and
  `sample_tile_overlap` parameters.

## Bug Fixes
* **Tiled sampling segfault** (stable-diffusion.cpp): `noise_pred` and
  `weight_map` buffers were local `std::vector<float>` captured by reference
  in lambda, but used after scope ended (use-after-free). Fixed with
  `shared_ptr` captured by value.
* **img2img mask crash** (sdR_interface.cpp): `mask_image` was not initialized
  when no mask provided, causing `GGML_ASSERT(image.width == tensor->ne[0])`
  in `ggml_extend.hpp:454`. Fixed by creating an all-white (255) mask.
* **img2img VAE encoder crash** (pipeline.R): `sd_ctx()` defaults to
  `vae_decode_only = TRUE`, but img2img needs the encoder. Added runtime
  check with clear error message in `sd_img2img()` and `sd_img2img_tiled()`.
* Default latent tile size for SD1/SD2 changed from 64 to 96 (768px) for
  better quality with MultiDiffusion.
* Removed excessive per-tile debug logging from C++ code.

---

# sdR 0.1.1

## VAE Tiling
* New `vae_mode` parameter in `sd_txt2img()` and `sd_img2img()`:
  `"normal"` (no tiling), `"tiled"` (always tile), `"auto"` (tile when
  `width * height > vae_auto_threshold`). Default is `"auto"`.
* `vae_auto_threshold` parameter (default `1048576L` = 1024x1024 pixels).
  Adjustable per user's VRAM budget.
* `vae_tile_rel_x` / `vae_tile_rel_y` parameters for adaptive tile sizing:
  fraction of latent dimension (0-1) or number of tiles (>1).
* `vae_tiling` parameter is deprecated with a warning; use `vae_mode` instead.

## High-Resolution Pipeline
* New `sd_txt2img_highres()` — patch-based generation for images larger than
  the model's native resolution (2K, 4K+). Independently generates overlapping
  patches, stitches them with linear blending, and optionally runs an
  `img2img` harmonization pass over the result.
* `model_type` parameter in `sd_ctx()`: `"sd1"`, `"sd2"`, `"sdxl"`, `"flux"`,
  `"sd3"`. Used to auto-detect native tile size (512 or 1024) for highres.
* Patch overlap as fraction of tile size (default 0.125).

## Tiled Sampling (MultiDiffusion)
* New `sd_txt2img_tiled()` — generates images at any resolution using tiled
  diffusion sampling. At each denoising step the latent is split into
  overlapping tiles, each denoised independently by the UNet, and results
  merged with Gaussian weighting. VRAM bounded by tile size, not output
  resolution.
* C++ implementation: tiled denoise wrapper around the existing scheduler loop,
  Gaussian blend mask pre-computed once, accumulation buffers allocated once
  and zeroed per step (industry-standard allocate-once pattern).
* `sample_tile_size` auto-detected from `model_type` (64 latent px for SD1/SD2,
  128 for SDXL/Flux/SD3).
* `sample_tile_overlap` (default 0.25) controls overlap fraction.

## Tests
* 22 new tests for VAE tiling mode resolution (`test-vae-tiling.R`).
* 35 new tests for highres pipeline helpers (`test-highres.R`):
  patch grid coverage, blend masks, array roundtrips, API signatures.
* 22 new tests for tiled sampling API and helpers (`test-tiled-sampling.R`).

---

# sdR 0.1.0

## Core
* Text-to-image generation via stable-diffusion.cpp (C++ backend).
* Support for SD 1.x, SD 2.x, SDXL model versions.
* SafeTensors and GGUF model format loading.
* Rcpp interface with XPtr and custom finalizer for sd_ctx_t.
* add gitignore for vocab*.hpp files (128 MB total) — downloaded during installation from GitHub Releases.

## GPU
* Vulkan GPU backend via ggmlR — tested on AMD (radv).
* SD 1.5: 512x512 in ~7s (20 Euler steps, Vulkan).

## Sampling
* Methods: Euler, Euler A, Heun, DPM2, DPM++ (2M), LCM, DDIM, TCD.
* Schedulers: Discrete, Karras, Exponential, Simple, SGM Uniform, AYS, LCM, Smoothstep.

## R API
* `sd_ctx()` — create model context (`vram_gb` for auto-routing, `vae_decode_only = FALSE` for img2img).
* `sd_generate()` — unified entry point, auto-selects strategy based on resolution and VRAM.
* `sd_txt2img()` — direct text-to-image (low-level).
* `sd_img2img()` — direct image-to-image (low-level).
* `sd_save_image()` — save to PNG.
* `sd_system_info()` — system and backend information.

## Internals
* CLIP BPE tokenizer and text encoder.
* UNet diffusion model.
* VAE decoder.
* PhiloxRNG and MT19937RNG for PyTorch-compatible reproducibility.
