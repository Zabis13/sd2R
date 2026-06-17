#include "clip_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <set>

#include "ggml.h"
#include "tokenize_util.h"
#include "util.h"
#include "vocab/vocab.h"

CLIPTokenizer::CLIPTokenizer(int pad_token_id, const std::string& merges_utf8_str) {
    UNK_TOKEN = "<|endoftext|>";
    BOS_TOKEN = "<|startoftext|>";
    EOS_TOKEN = "<|endoftext|>";
    PAD_TOKEN = "<|endoftext|>";

    UNK_TOKEN_ID = 49407;
    BOS_TOKEN_ID = 49406;
    EOS_TOKEN_ID = 49407;
    PAD_TOKEN_ID = pad_token_id;

    end_of_word_suffix = "</w>";
    add_bos_token      = true;
    add_eos_token      = true;

    if (merges_utf8_str.size() > 0) {
        load_from_merges(merges_utf8_str);
    } else {
        load_from_merges(load_clip_merges());
    }
    add_special_token("<|startoftext|>");
    add_special_token("<|endoftext|>");
}

void CLIPTokenizer::load_from_merges(const std::string& merges_utf8_str) {
    auto byte_unicode_pairs = bytes_to_unicode();
    byte_encoder            = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());
    for (auto& pair : byte_unicode_pairs) {
        byte_decoder[pair.second] = pair.first;
    }

    std::vector<std::u32string> merges = split_utf32(merges_utf8_str);
    GGML_ASSERT(merges.size() == 48895);
    merges = std::vector<std::u32string>(merges.begin() + 1, merges.end());
    std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
    for (const auto& merge : merges) {
        size_t space_pos = merge.find(' ');
        merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
    }
    std::vector<std::u32string> vocab;
    for (const auto& pair : byte_unicode_pairs) {
        vocab.push_back(pair.second);
    }
    for (const auto& pair : byte_unicode_pairs) {
        vocab.push_back(pair.second + utf8_to_utf32("</w>"));
    }
    for (const auto& merge : merge_pairs) {
        vocab.push_back(merge.first + merge.second);
    }
    vocab.push_back(utf8_to_utf32("<|startoftext|>"));
    vocab.push_back(utf8_to_utf32("<|endoftext|>"));
    LOG_DEBUG("vocab size: %zu", vocab.size());
    int i = 0;
    for (const auto& token : vocab) {
        encoder[token] = i;
        decoder[i]     = token;
        i++;
    }
    encoder_len = i;

    int rank = 0;
    for (const auto& merge : merge_pairs) {
        bpe_ranks[merge] = rank++;
    }
    bpe_len = rank;
}

static std::string strip(const std::string& str) {
    std::string::size_type start = str.find_first_not_of(" \t\n\r\v\f");
    std::string::size_type end   = str.find_last_not_of(" \t\n\r\v\f");

    if (start == std::string::npos) {
        return "";
    }

    return str.substr(start, end - start + 1);
}

// Collapse any run of whitespace into a single space, then strip ends.
// Hand-rolled to avoid libstdc++'s std::regex, which on MinGW/Windows hangs
// (catastrophic backtracking / deep recursion) — same class of bug already
// fixed in parse_prompt_attention. Matches std::regex R"(\s+)" with the C
// locale: space, \t, \n, \r, \v, \f.
static std::string whitespace_clean(const std::string& text) {
    std::string collapsed;
    collapsed.reserve(text.size());
    bool in_ws = false;
    for (char c : text) {
        bool is_ws = (c == ' ' || c == '\t' || c == '\n' ||
                      c == '\r' || c == '\v' || c == '\f');
        if (is_ws) {
            if (!in_ws) {
                collapsed.push_back(' ');
                in_ws = true;
            }
        } else {
            collapsed.push_back(c);
            in_ws = false;
        }
    }
    return strip(collapsed);
}

std::string CLIPTokenizer::normalize(const std::string& text) const {
    auto normalized_text = whitespace_clean(text);
    std::transform(normalized_text.begin(), normalized_text.end(), normalized_text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return normalized_text;
}

// Hand-rolled equivalent of the CLIP BPE split pattern:
//   's|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[[:digit:]]|[^[:space:][:alpha:][:digit:]]+
// libstdc++'s std::regex hangs on this alternation-with-quantifiers pattern on
// MinGW/Windows (catastrophic backtracking), so we tokenize manually. Matching
// rules, in priority order, mirror the regex exactly:
//   1. English contractions: 's 't 're 've 'm 'll 'd  (only at an apostrophe)
//   2. run of ASCII letters   [[:alpha:]]+
//   3. a single ASCII digit   [[:digit:]]   (one char per token, as in regex)
//   4. run of "other" bytes   [^space,alpha,digit]+ (punctuation + UTF-8 cont.)
// icase is irrelevant: normalize() has already lowercased the input. Note we
// operate on bytes; non-ASCII (high-bit) bytes are neither alpha nor digit
// under the C locale, so they fall into rule 4 exactly as std::regex did with
// the default (C) locale.
std::vector<std::string> CLIPTokenizer::token_split(const std::string& text) const {
    auto is_alpha = [](unsigned char c) { return std::isalpha(c) != 0; };
    auto is_digit = [](unsigned char c) { return std::isdigit(c) != 0; };
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };

    std::vector<std::string> result;
    const size_t n = text.size();
    size_t i       = 0;

    while (i < n) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        // Rule 1: contractions at an apostrophe. Inspect the next byte(s)
        // explicitly (no startswith) so multi-byte UTF-8 before the apostrophe
        // can never be misread.
        if (c == '\'' && i + 1 < n) {
            unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
            // two-letter: 're 've 'll  (check before single-letter forms)
            if (i + 2 < n) {
                unsigned char c2 = static_cast<unsigned char>(text[i + 2]);
                if ((c1 == 'r' && c2 == 'e') ||
                    (c1 == 'v' && c2 == 'e') ||
                    (c1 == 'l' && c2 == 'l')) {
                    result.emplace_back(text.substr(i, 3));
                    i += 3;
                    continue;
                }
            }
            // single-letter: 's 't 'm 'd
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                result.emplace_back(text.substr(i, 2));
                i += 2;
                continue;
            }
            // bare apostrophe falls through to rule 4 (other run).
        }

        // Rule 2: run of letters.
        if (is_alpha(c)) {
            size_t j = i + 1;
            while (j < n && is_alpha(static_cast<unsigned char>(text[j]))) {
                j++;
            }
            result.emplace_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        // Rule 3: a single digit.
        if (is_digit(c)) {
            result.emplace_back(text.substr(i, 1));
            i += 1;
            continue;
        }

        // Whitespace is a separator: the regex never emits whitespace tokens.
        if (is_space(c)) {
            i += 1;
            continue;
        }

        // Rule 4: run of "other" bytes (punctuation, symbols, UTF-8
        // continuation bytes) — anything that is not space/alpha/digit and not
        // the start of a contraction handled above.
        size_t j = i + 1;
        while (j < n) {
            unsigned char cj = static_cast<unsigned char>(text[j]);
            if (is_space(cj) || is_alpha(cj) || is_digit(cj)) {
                break;
            }
            // stop so a new contraction can be matched at the apostrophe
            if (cj == '\'') {
                break;
            }
            j++;
        }
        result.emplace_back(text.substr(i, j - i));
        i = j;
    }

    return result;
}
