#include "vocab.h"

// ---------------------------------------------------------------------------
// Vocabulary / BPE-merges embedded at compile time.
//
// sd2R ships only the vocab files for models that currently work on the Vulkan
// backend (SD1/2/SDXL, Flux, SD3, Wan video, Qwen-Image). The remaining files
// for less common text encoders are kept out of the package to keep the
// download/binary small, and are gated behind SD2R_FULL_VOCAB.
//
// ALWAYS embedded (downloaded by configure, ~64 MB total):
//   clip_merges.hpp  (2.6 MB) — CLIP tokenizer  (SD1/2/SDXL, Flux clip_l/clip_g)
//   t5.hpp           (7.3 MB) — T5 tokenizer    (Flux, SD3)
//   umt5.hpp        (45.7 MB) — UMT5 tokenizer  (Wan video)
//   qwen_merges.hpp  (8.4 MB) — Qwen2 tokenizer (Qwen-Image)
//
// OPTIONAL — only when SD2R_FULL_VOCAB is defined (files available on the
// assets-v2 GitHub release; add them to configure's VOCAB_FILES to enable):
//   mistral_vocab.hpp / mistral_merges.hpp — Mistral  (Chroma-Radiance, Ernie-Image)
//   gemma_vocab.hpp   / gemma_merges.hpp   — Gemma    (LTX audio/video)
//   gemma2_vocab.hpp  / gemma2_merges.hpp  — Gemma2   (PID)
//   gpt_oss_vocab.hpp / gpt_oss_merges.hpp — GPT-OSS  (Lens)
// ---------------------------------------------------------------------------

#include "clip_merges.hpp"
#include "qwen_merges.hpp"
#include "t5.hpp"
#include "umt5.hpp"

#ifdef SD2R_FULL_VOCAB
#include "gemma2_merges.hpp"
#include "gemma2_vocab.hpp"
#include "gemma_merges.hpp"
#include "gemma_vocab.hpp"
#include "gpt_oss_merges.hpp"
#include "gpt_oss_vocab.hpp"
#include "mistral_merges.hpp"
#include "mistral_vocab.hpp"
#endif

std::string load_clip_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(clip_merges_utf8_c_str), sizeof(clip_merges_utf8_c_str));
    return merges_utf8_str;
}

std::string load_qwen2_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(qwen2_merges_utf8_c_str), sizeof(qwen2_merges_utf8_c_str));
    return merges_utf8_str;
}

std::string load_t5_tokenizer_json() {
    std::string json_str(reinterpret_cast<const char*>(t5_tokenizer_json_str), sizeof(t5_tokenizer_json_str));
    return json_str;
}

std::string load_umt5_tokenizer_json() {
    std::string json_str(reinterpret_cast<const char*>(umt5_tokenizer_json_str), sizeof(umt5_tokenizer_json_str));
    return json_str;
}

#ifdef SD2R_FULL_VOCAB

std::string load_mistral_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(mistral_merges_utf8_c_str), sizeof(mistral_merges_utf8_c_str));
    return merges_utf8_str;
}

std::string load_mistral_vocab_json() {
    std::string json_str(reinterpret_cast<const char*>(mistral_vocab_json_utf8_c_str), sizeof(mistral_vocab_json_utf8_c_str));
    return json_str;
}

std::string load_gemma_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(gemma_merges_utf8_c_str), sizeof(gemma_merges_utf8_c_str));
    return merges_utf8_str;
}

std::string load_gemma_vocab_json() {
    std::string json_str(reinterpret_cast<const char*>(gemma_vocab_json_utf8_c_str), sizeof(gemma_vocab_json_utf8_c_str));
    return json_str;
}

std::string load_gemma2_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(gemma2_merges_utf8_c_str), sizeof(gemma2_merges_utf8_c_str));
    return merges_utf8_str;
}

std::string load_gemma2_vocab_json() {
    std::string json_str(reinterpret_cast<const char*>(gemma2_vocab_json_utf8_c_str), sizeof(gemma2_vocab_json_utf8_c_str));
    return json_str;
}

std::string load_gpt_oss_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(gpt_oss_merges_utf8_c_str), sizeof(gpt_oss_merges_utf8_c_str));
    return merges_utf8_str;
}

std::string load_gpt_oss_vocab_json() {
    std::string json_str(reinterpret_cast<const char*>(gpt_oss_vocab_json_utf8_c_str), sizeof(gpt_oss_vocab_json_utf8_c_str));
    return json_str;
}

#else  // !SD2R_FULL_VOCAB — stubs so the package builds with only the 4 core vocabs.

// These return empty data; the corresponding tokenizers (Mistral/Gemma/Gemma2/
// GPT-OSS) will fail to initialize at runtime. The associated models are not
// shipped-enabled in sd2R yet. To enable: define SD2R_FULL_VOCAB and add the
// matching *.hpp files to configure's VOCAB_FILES.

std::string load_mistral_merges()     { return std::string(); }
std::string load_mistral_vocab_json() { return std::string(); }
std::string load_gemma_merges()       { return std::string(); }
std::string load_gemma_vocab_json()   { return std::string(); }
std::string load_gemma2_merges()      { return std::string(); }
std::string load_gemma2_vocab_json()  { return std::string(); }
std::string load_gpt_oss_merges()     { return std::string(); }
std::string load_gpt_oss_vocab_json() { return std::string(); }

#endif  // SD2R_FULL_VOCAB
