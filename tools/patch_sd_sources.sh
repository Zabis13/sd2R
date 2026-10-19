#!/bin/sh
# Patch vendored sd/ sources for R package compatibility.
#
# C++ <cstdio> does '#undef printf', so the r_ggml_compat.h macro approach
# does not work for .cpp/.hpp files.  Instead we do direct text replacements.
#
# Usage:  ./tools/patch_sd_sources.sh [src_dir]
#   src_dir defaults to 'src/sd' (relative to package root).
#
# After updating upstream (git subtree pull), re-run this script.
#
# Uses perl instead of sed for cross-platform compatibility (macOS/Linux).

SD_DIR="${1:-src/sd}"

if [ ! -d "$SD_DIR" ]; then
  echo "ERROR: directory '$SD_DIR' not found" >&2
  exit 1
fi

echo "* Patching sd/ sources for R compatibility..."

# --- 1. Replace printf/puts/fflush/putchar with R-safe wrappers ---
#
# We only patch active (non-commented) calls.
#   - Skip lines starting with optional whitespace + '//'
#   - Replace whole-word printf( -> r_ggml_printf(
#   - Replace whole-word puts( -> r_ggml_puts(
#   - Replace whole-word putchar( -> r_ggml_putchar(
#   - Replace fflush(stdout) -> r_ggml_fflush(NULL)
#   - Replace fflush(stderr) -> r_ggml_fflush(NULL)
#   - Do NOT touch snprintf, sprintf, fprintf, vprintf, log_printf

# Process all .cpp, .hpp, .h, .c files (recursive; excluding thirdparty/, model_io/, tokenizers/)
find "$SD_DIR" \( -path "$SD_DIR/thirdparty" -o -path "$SD_DIR/model_io" -o -path "$SD_DIR/tokenizers" \) -prune -o \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.c' \) -print | while read -r f; do
  # Skip already-patched files (idempotent)
  if grep -q 'r_ggml_printf' "$f" 2>/dev/null; then
    continue
  fi

  perl -pi -e '
    next if /^\s*\/\//;
    s/\bstd::(printf|puts|putchar)\s*\(/r_ggml_$1(/g;
    s/\bstd::fflush\s*\(std(out|err)\)/r_ggml_fflush(NULL)/g;
    s/\bprintf\s*\(/r_ggml_printf(/g;
    s/\bputs\s*\(/r_ggml_puts(/g;
    s/\bputchar\s*\(/r_ggml_putchar(/g;
    s/\bfflush\s*\(stdout\)/r_ggml_fflush(NULL)/g;
    s/\bfflush\s*\(stderr\)/r_ggml_fflush(NULL)/g;
  ' "$f"

  echo "  patched: ${f#$SD_DIR/}"
done

# --- 2. Patch thirdparty/ separately (only specific files) ---
if [ -f "$SD_DIR/thirdparty/stb_image_resize.h" ]; then
  if ! grep -q 'r_ggml_printf' "$SD_DIR/thirdparty/stb_image_resize.h" 2>/dev/null; then
    perl -pi -e '
      next if /^\s*\/\//;
      s/\bprintf\s*\(/r_ggml_printf(/g;
    ' "$SD_DIR/thirdparty/stb_image_resize.h"
    echo "  patched: thirdparty/stb_image_resize.h"
  fi
fi

# --- 3. Fix compiler warnings (CRAN -Wall) ---

# 3a. util.cpp: sign-compare warnings (int vs size_t) in parse_prompt_attention
perl -pi -e 's/for \(int p = start_position; p < res\.size\(\)/for (size_t p = (size_t)start_position; p < res.size()/' \
  "$SD_DIR/core/util.cpp"
perl -pi -e 's/while \(i \+ 1 < res\.size\(\)\)/while ((size_t)(i + 1) < res.size())/' \
  "$SD_DIR/core/util.cpp" && echo "  patched: util.cpp (sign-compare)"

# 3b. ggml_tensor_utils.cpp: sign-compare (int vs size_t for N) in timestep embedding
perl -pi -e 's/for \(int i = 0; i < N; \+\+i\) \{/for (size_t i = 0; i < N; ++i) {/' \
  "$SD_DIR/core/ggml_tensor_utils.cpp" && echo "  patched: ggml_tensor_utils.cpp (sign-compare N)"

# 3c. json.hpp.inc: deprecated literal operator with space (C++17)
#     operator "" _json  ->  operator ""_json
perl -pi -e 's/operator ""\s+_json\b/operator ""_json/g' \
  "$SD_DIR/thirdparty/json.hpp.inc" && echo "  patched: thirdparty/json.hpp.inc (literal operator spacing)"

# 3d. util.h: GNU extension ##__VA_ARGS__ -> C99/C++20 __VA_OPT__
perl -pi -e 's/, ##__VA_ARGS__/ __VA_OPT__(,) __VA_ARGS__/g' \
  "$SD_DIR/core/util.h" && echo "  patched: util.h (VA_OPT)"

# Removed with the 2026-10 upstream sync (fixed upstream or code gone):
#   unused 'size2' in util.cpp, unused 'param' in ggml_extend.hpp,
#   [[maybe_unused]] sd_version_is_inpaint_or_unet_edit in model.h,
#   defensive mask creation in generate_image (pipeline/image.cpp now resizes
#   the mask and skips it when mask_image.data == nullptr).

echo "* Done."
