#include "regex.h"

// sd2R: oniguruma-free implementation of sd::Regex.
//
// Upstream compiles arbitrary patterns with oniguruma. sd2R does not vendor
// oniguruma/utf8proc (and std::regex hangs on MinGW), so compile() only accepts
// the fixed pre-tokenizer patterns of the built-in tokenizers (CLIP, Qwen2,
// Mistral) and runs a hand-written matcher with the same leftmost-first,
// greedy-with-backtracking semantics. Any other pattern (e.g. from an external
// tokenizer.json) is rejected with an error.

#include <cstdint>
#include <limits>
#include <vector>

#include "unicode_tables.h"

namespace sd {

    enum class SplitKind {
        CLIP,
        QWEN2,
        MISTRAL,
    };

    struct Regex::Impl {
        SplitKind kind;
    };

    static const char* kClipPattern =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+)";
    static const char* kQwen2Pattern =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    static const char* kMistralPattern =
        R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

    static bool regex_error(std::string* error, const std::string& message) {
        if (error) {
            *error = message;
        }
        return false;
    }

    // Rust str excludes overlong encodings, surrogates and extended UTF-8.
    static bool regex_valid_utf8(const std::string& text) {
        size_t position = 0;
        while (position < text.size()) {
            const auto lead = static_cast<unsigned char>(text[position++]);
            if (lead < 0x80) {
                continue;
            }
            int count = 0;
            if (lead >= 0xC2 && lead <= 0xDF) {
                count = 1;
            } else if (lead >= 0xE0 && lead <= 0xEF) {
                count = 2;
            } else if (lead >= 0xF0 && lead <= 0xF4) {
                count = 3;
            }
            if (count == 0 || text.size() - position < static_cast<size_t>(count)) {
                return false;
            }
            uint32_t codepoint = lead & (0x7F >> count);
            for (int i = 0; i < count; ++i) {
                const auto byte = static_cast<unsigned char>(text[position++]);
                if ((byte & 0xC0) != 0x80) {
                    return false;
                }
                codepoint = (codepoint << 6) | (byte & 0x3F);
            }
            constexpr uint32_t minimum[] = {0, 0x80, 0x800, 0x10000};
            if (codepoint < minimum[count] || codepoint > 0x10FFFF || (codepoint >= 0xD800 && codepoint <= 0xDFFF)) {
                return false;
            }
        }
        return true;
    }

    // ---- Unicode properties -------------------------------------------------

    static bool in_ranges(const unicode::Range* ranges, unsigned count, uint32_t cp) {
        unsigned lo = 0;
        unsigned hi = count;
        while (lo < hi) {
            unsigned mid = lo + (hi - lo) / 2;
            if (cp < ranges[mid].lo) {
                hi = mid;
            } else if (cp > ranges[mid].hi) {
                lo = mid + 1;
            } else {
                return true;
            }
        }
        return false;
    }

    enum : uint16_t {
        P_LU    = 1 << 0,
        P_LL    = 1 << 1,
        P_LT    = 1 << 2,
        P_LM    = 1 << 3,
        P_LO    = 1 << 4,
        P_M     = 1 << 5,
        P_N     = 1 << 6,
        P_SPACE = 1 << 7,
        P_L     = P_LU | P_LL | P_LT | P_LM | P_LO,
    };

    static uint16_t props_of(uint32_t cp) {
        if (cp < 0x80) {
            uint16_t p = 0;
            if (cp >= 'A' && cp <= 'Z') {
                p |= P_LU;
            } else if (cp >= 'a' && cp <= 'z') {
                p |= P_LL;
            } else if (cp >= '0' && cp <= '9') {
                p |= P_N;
            } else if (cp == ' ' || (cp >= 0x09 && cp <= 0x0D)) {
                p |= P_SPACE;
            }
            return p;
        }
        uint16_t p = 0;
        if (in_ranges(unicode::kLu, unicode::kCountLu, cp)) p |= P_LU;
        if (in_ranges(unicode::kLl, unicode::kCountLl, cp)) p |= P_LL;
        if (in_ranges(unicode::kLt, unicode::kCountLt, cp)) p |= P_LT;
        if (in_ranges(unicode::kLm, unicode::kCountLm, cp)) p |= P_LM;
        if (in_ranges(unicode::kLo, unicode::kCountLo, cp)) p |= P_LO;
        if (in_ranges(unicode::kM, unicode::kCountM, cp)) p |= P_M;
        if (in_ranges(unicode::kN, unicode::kCountN, cp)) p |= P_N;
        if (in_ranges(unicode::kWhite_Space, unicode::kCountWhite_Space, cp)) p |= P_SPACE;
        return p;
    }

    // Decoded text: code points, their properties and byte offsets
    // (offset[i] = start of cp i, offset[n] = text size).
    struct Decoded {
        std::vector<uint32_t> cp;
        std::vector<uint16_t> prop;
        std::vector<size_t> offset;
        size_t n = 0;

        explicit Decoded(const std::string& text) {
            size_t pos = 0;
            while (pos < text.size()) {
                const auto lead = static_cast<unsigned char>(text[pos]);
                int extra       = lead < 0x80 ? 0 : lead < 0xE0 ? 1 : lead < 0xF0 ? 2 : 3;
                uint32_t value  = extra == 0 ? lead : (lead & (0x3F >> extra));
                for (int k = 1; k <= extra; ++k) {
                    value = (value << 6) | (static_cast<unsigned char>(text[pos + k]) & 0x3F);
                }
                offset.push_back(pos);
                cp.push_back(value);
                prop.push_back(props_of(value));
                pos += static_cast<size_t>(extra) + 1;
            }
            n = cp.size();
            offset.push_back(text.size());
        }

        bool is(size_t i, uint16_t mask) const { return i < n && (prop[i] & mask) != 0; }
        bool is_cr_lf(size_t i) const { return i < n && (cp[i] == '\r' || cp[i] == '\n'); }
        // [^\s\p{L}\p{N}]
        bool is_other(size_t i) const { return i < n && (prop[i] & (P_SPACE | P_L | P_N)) == 0; }
        // [^\r\n\p{L}\p{N}]
        bool is_prefix(size_t i) const { return i < n && !is_cr_lf(i) && (prop[i] & (P_L | P_N)) == 0; }

        size_t run(size_t i, uint16_t mask) const {
            while (is(i, mask)) {
                ++i;
            }
            return i;
        }
        size_t run_other(size_t i) const {
            while (is_other(i)) {
                ++i;
            }
            return i;
        }
    };

    // (?i:'s|'t|'re|'ve|'m|'ll|'d) — ASCII apostrophe and letters only.
    static size_t match_contraction(const Decoded& d, size_t i) {
        if (i >= d.n || d.cp[i] != '\'') {
            return 0;
        }
        auto lower = [&](size_t k) -> uint32_t {
            if (k >= d.n) {
                return 0;
            }
            uint32_t c = d.cp[k];
            return (c >= 'A' && c <= 'Z') ? c + 32 : c;
        };
        uint32_t c1 = lower(i + 1);
        uint32_t c2 = lower(i + 2);
        if (c1 == 's' || c1 == 't') {
            return i + 2;
        }
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e')) {
            return i + 3;
        }
        if (c1 == 'm') {
            return i + 2;
        }
        if (c1 == 'l' && c2 == 'l') {
            return i + 3;
        }
        if (c1 == 'd') {
            return i + 2;
        }
        return 0;
    }

    // Shared tail of the Qwen2/Mistral patterns: \s*[\r\n]+|\s+(?!\S)|\s+
    static size_t match_whitespace(const Decoded& d, size_t i) {
        if (!d.is(i, P_SPACE)) {
            return 0;
        }
        size_t end = d.run(i, P_SPACE);
        // \s*[\r\n]+ : \s* backtracks to the last CR/LF of the run; [\r\n]+ ends right after it
        for (size_t k = end; k > i; --k) {
            if (d.is_cr_lf(k - 1)) {
                return k;
            }
        }
        // \s+(?!\S) : whole run at end of text, otherwise give back one space
        if (end == d.n) {
            return end;
        }
        if (end - i >= 2) {
            return end - 1;
        }
        // \s+
        return end;
    }

    // \p{L}+|\p{N}|[^\s\p{L}\p{N}]+ (after contractions)
    static size_t match_clip(const Decoded& d, size_t i) {
        if (size_t e = match_contraction(d, i)) {
            return e;
        }
        if (d.is(i, P_L)) {
            return d.run(i, P_L);
        }
        if (d.is(i, P_N)) {
            return i + 1;
        }
        if (d.is_other(i)) {
            return d.run_other(i);
        }
        return 0;  // whitespace: no alternative matches, skipped by the search
    }

    static size_t match_qwen2(const Decoded& d, size_t i) {
        if (size_t e = match_contraction(d, i)) {
            return e;
        }
        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (d.is_prefix(i) && d.is(i + 1, P_L)) {
            return d.run(i + 1, P_L);
        }
        if (d.is(i, P_L)) {
            return d.run(i, P_L);
        }
        // \p{N}
        if (d.is(i, P_N)) {
            return i + 1;
        }
        // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        size_t s = (i < d.n && d.cp[i] == ' ' && d.is_other(i + 1)) ? i + 1 : i;
        if (d.is_other(s)) {
            size_t e = d.run_other(s);
            while (d.is_cr_lf(e)) {
                ++e;
            }
            return e;
        }
        return match_whitespace(d, i);
    }

    static size_t match_mistral(const Decoded& d, size_t i) {
        const uint16_t upper = P_LU | P_LT | P_LM | P_LO | P_M;
        const uint16_t lower = P_LL | P_LM | P_LO | P_M;
        // [^\r\n\p{L}\p{N}]?[upper]*[lower]+
        for (int prefix = 1; prefix >= 0; --prefix) {
            if (prefix && !d.is_prefix(i)) {
                continue;
            }
            size_t s = i + prefix;
            size_t u = d.run(s, upper);
            for (size_t k = u + 1; k-- > s;) {
                if (d.is(k, lower)) {
                    return d.run(k, lower);
                }
            }
        }
        // [^\r\n\p{L}\p{N}]?[upper]+[lower]*
        for (int prefix = 1; prefix >= 0; --prefix) {
            if (prefix && !d.is_prefix(i)) {
                continue;
            }
            size_t s = i + prefix;
            if (d.is(s, upper)) {
                return d.run(d.run(s, upper), lower);
            }
        }
        // \p{N}
        if (d.is(i, P_N)) {
            return i + 1;
        }
        // ` ?[^\s\p{L}\p{N}]+[\r\n/]*`
        size_t s = (i < d.n && d.cp[i] == ' ' && d.is_other(i + 1)) ? i + 1 : i;
        if (d.is_other(s)) {
            size_t e = d.run_other(s);
            while (d.is_cr_lf(e) || (e < d.n && d.cp[e] == '/')) {
                ++e;
            }
            return e;
        }
        return match_whitespace(d, i);
    }

    Regex::Regex()                            = default;
    Regex::~Regex()                           = default;
    Regex::Regex(Regex&&) noexcept            = default;
    Regex& Regex::operator=(Regex&&) noexcept = default;

    bool Regex::compile(const std::string& pattern, std::string* error) {
        if (error) {
            error->clear();
        }
        auto next = std::make_unique<Impl>();
        if (pattern == kClipPattern) {
            next->kind = SplitKind::CLIP;
        } else if (pattern == kQwen2Pattern) {
            next->kind = SplitKind::QWEN2;
        } else if (pattern == kMistralPattern) {
            next->kind = SplitKind::MISTRAL;
        } else {
            return regex_error(error, "unsupported regex pattern (sd2R supports only the built-in tokenizer patterns): " + pattern);
        }
        impl_ = std::move(next);
        return true;
    }

    bool Regex::find_matches(const std::string& text, std::vector<Match>& matches, std::string* error) const {
        matches.clear();
        if (error) {
            error->clear();
        }
        if (!impl_) {
            return regex_error(error, "regex has not been compiled");
        }
        if (!regex_valid_utf8(text)) {
            return regex_error(error, "regex input is not valid UTF-8");
        }
        const Decoded d(text);
        size_t i = 0;
        while (i < d.n) {
            size_t e = 0;
            switch (impl_->kind) {
                case SplitKind::CLIP:
                    e = match_clip(d, i);
                    break;
                case SplitKind::QWEN2:
                    e = match_qwen2(d, i);
                    break;
                case SplitKind::MISTRAL:
                    e = match_mistral(d, i);
                    break;
            }
            if (e > i) {
                matches.emplace_back(d.offset[i], d.offset[e]);
                i = e;
            } else {
                ++i;
            }
        }
        return true;
    }

}  // namespace sd
