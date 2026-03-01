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
* `sd_ctx()` — create model context.
* `sd_txt2img()` — generate images from text.
* `sd_img2img()` — generate from image.
* `sd_save_image()` — save to PNG.
* `sd_system_info()` — system and backend information.

## Internals
* CLIP BPE tokenizer and text encoder.
* UNet diffusion model.
* VAE decoder.
* PhiloxRNG and MT19937RNG for PyTorch-compatible reproducibility.
