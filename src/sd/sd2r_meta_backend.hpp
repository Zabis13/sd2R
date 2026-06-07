// sd2r_meta_backend.hpp — sd2R "second path": multi-GPU compute via the ggml
// meta backend (ggml 0.11.0+). NOT part of upstream stable-diffusion.cpp.
//
// Purpose: let sd2R use the meta backend (sharding a SINGLE model across N
// devices) without modifying ggmlR or touching the upstream core. This file is
// self-contained and header-only — upstream sd.cpp updates do not affect it,
// and it needs no entry in OBJECTS/Makevars.
//
// Logic ported from the verified R PoC (inst/examples/meta_backend_*.R):
//   - split policy: large 2D Linear weights are split along the output axis
//     (axis=1) across N devices (tensor parallelism), everything else MIRRORED;
//   - compute_split_ne(): even split of an axis across N devices, distributing
//     the remainder over the first devices; segments sum to the original size.
//
// Usage (when the context spans >1 device):
//   sd2r::meta::Policy pol;                     // default policy
//   ggml_backend_dev_t meta_dev =
//       sd2r::meta::make_meta_device(devs, n_devs, &pol);
//   ggml_backend_t backend = ggml_backend_dev_init(meta_dev);
//   // backend is then passed to GGMLRunner as runtime_backend — compute()
//   // runs through it transparently (the single-backend path is unchanged).
//
// Compatibility: if the meta symbols are absent from the linked libggml.a, the
// caller must use the normal single-backend path. This file only provides the
// split logic and a helper to build the meta-device.

#ifndef SD2R_META_BACKEND_HPP
#define SD2R_META_BACKEND_HPP

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

namespace sd2r {
namespace meta {

#if defined(SD2R_HAVE_META_BACKEND)
// =========================================================================
// Meta backend AVAILABLE in the linked libggml.a (detected by configure).
// Full implementation: split policy + meta-device builder + global flag.
// =========================================================================

// Tensor placement policy. Mirrors the R PoC make_split_fn().
struct Policy {
    // Master switch for real tensor split. When false (default), ALL tensors
    // are MIRRORED — a full replica of the model on every device. This is the
    // only correct-by-construction policy: arbitrary per-weight axis splitting
    // requires the split axes to be consistent across the whole compute graph
    // (op inputs/outputs), otherwise the meta backend aborts
    // (GGML_ASSERT(src_ss[0].axis == ...)). Real split needs graph-aware,
    // architecture-specific axis selection (incl. fused QKV) validated on 2+
    // physical GPUs; until then we keep MIRRORED (correct, replicated, no
    // speedup). See TODO #15.
    bool tensor_split_enabled = false;
    // Minimum output-axis size required to split a weight (else MIRRORED).
    int64_t split_min = 32;
    // Axis along which 2D weights are split (Linear output dim = ggml ne[1]).
    int split_axis = GGML_BACKEND_SPLIT_AXIS_1;
};

// Evenly divide axis size dim_size across n_devs devices.
// The remainder (dim_size % n_devs) is distributed +1 to the first devices.
// The result sums to dim_size. Valid for any n_devs (incl. > dim_size).
inline std::vector<int64_t> compute_split_ne(int64_t dim_size, size_t n_devs) {
    std::vector<int64_t> per(n_devs, 0);
    if (n_devs == 0) {
        return per;
    }
    int64_t base = dim_size / (int64_t)n_devs;
    int64_t rem  = dim_size % (int64_t)n_devs;
    for (size_t d = 0; d < n_devs; ++d) {
        per[d] = base + ((int64_t)d < rem ? 1 : 0);
    }
    return per;
}

// Decide how to place a single tensor.
//   - if policy.tensor_split_enabled AND 2D weight (ne[2]==1 && ne[3]==1) with
//     output axis >= max(split_min, n_devs) -> split along policy.split_axis;
//   - otherwise -> MIRRORED (the safe, correct-by-construction default).
inline ggml_backend_meta_split_state decide_split(const ggml_tensor* tensor,
                                                  size_t n_devs,
                                                  const Policy& pol) {
    ggml_backend_meta_split_state st;
    std::memset(&st, 0, sizeof(st));

    const int64_t* ne = tensor->ne;
    const bool is_2d  = (ne[2] == 1 && ne[3] == 1);
    const int64_t out_dim = ne[1];  // Linear weight output dimension

    const int64_t min_dim = pol.split_min > (int64_t)n_devs ? pol.split_min : (int64_t)n_devs;

    if (pol.tensor_split_enabled && is_2d && n_devs >= 2 && out_dim >= min_dim) {
        st.axis       = (ggml_backend_meta_split_axis)pol.split_axis;
        st.n_segments = 1;
        std::vector<int64_t> per = compute_split_ne(out_dim, n_devs);
        // ne layout: [seg0_dev0, seg0_dev1, ...] — for n_segments=1 this is just
        // one value per device.
        for (size_t d = 0; d < n_devs && d < (size_t)(16 * GGML_BACKEND_META_MAX_DEVICES); ++d) {
            st.ne[d] = per[d];
        }
        return st;
    }

    // MIRRORED: full copy on every device. ne is ignored by ggml.
    st.axis       = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    st.n_segments = 1;
    return st;
}

// userdata passed to the C callback. Holds the policy and device count.
struct SplitContext {
    Policy policy;
    size_t n_devs = 0;
};

// C callback with the required signature (ggml_backend_meta_get_split_state_t).
// No exceptions/RTTI escape — ggml may call it from any context.
inline ggml_backend_meta_split_state get_split_state_cb(const ggml_tensor* tensor,
                                                        void* userdata) {
    const SplitContext* sc = static_cast<const SplitContext*>(userdata);
    if (sc == nullptr || tensor == nullptr) {
        ggml_backend_meta_split_state st;
        std::memset(&st, 0, sizeof(st));
        st.axis       = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        st.n_segments = 1;
        return st;
    }
    return decide_split(tensor, sc->n_devs, sc->policy);
}

// Build a meta-device over a list of physical devices.
// split_ctx must outlive the meta-device (the caller owns it).
// Returns nullptr if there are fewer than 1 device.
inline ggml_backend_dev_t make_meta_device(ggml_backend_dev_t* devs,
                                           size_t n_devs,
                                           SplitContext* split_ctx) {
    if (devs == nullptr || n_devs < 1 || split_ctx == nullptr) {
        return nullptr;
    }
    split_ctx->n_devs = n_devs;
    return ggml_backend_meta_device(devs, n_devs, &get_split_state_cb, split_ctx);
}

// ---------------------------------------------------------------------------
// Global config — sd2R "second path" under a flag. interface.cpp sets the
// meta_backend flag before new_sd_ctx(); SDBackendManager reads it for DIFFUSION.
// Flag = on/off (default off => the old single-backend path). When on we build
// a meta-backend over ALL available GPU devices (works on 1 GPU too).
// State is process-global (like sd.cpp's backend init); one context at a time
// is already a constraint of sd_generate_multi_gpu.
// ---------------------------------------------------------------------------
inline bool& meta_backend_flag() {
    static bool g_meta_on = false;
    return g_meta_on;
}

inline void set_meta_backend(bool on) {
    meta_backend_flag() = on;
}

// meta is enabled when the flag is set.
inline bool meta_enabled() {
    return meta_backend_flag();
}

// Meta is compiled in, so a request is never "unavailable" here.
inline bool meta_requested_but_unavailable() {
    return false;
}

// Owner of the meta-backend and its split context (they live with the backend).
// Singleton, to cache the built meta-backend for the context's lifetime.
struct MetaBackendOwner {
    ggml_backend_t backend = nullptr;
    SplitContext   split_ctx;       // callback userdata; must outlive the backend
    std::vector<ggml_backend_dev_t> devs;

    static MetaBackendOwner& instance() {
        static MetaBackendOwner inst;
        return inst;
    }

    void free() {
        if (backend != nullptr) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
        devs.clear();
    }
};

// Get (lazily build) a meta-backend over ALL available GPU devices.
// Returns nullptr if meta is unavailable / no GPU present — the caller must
// then use the normal single-backend path (backward compatibility).
inline ggml_backend_t get_or_build_diffusion_meta_backend() {
    if (!meta_enabled()) {
        return nullptr;
    }
    MetaBackendOwner& owner = MetaBackendOwner::instance();
    if (owner.backend != nullptr) {
        return owner.backend;
    }

    // Collect all GPU devices from the ggml registry.
    size_t total = ggml_backend_dev_count();
    owner.devs.clear();
    for (size_t i = 0; i < total; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            owner.devs.push_back(dev);
        }
    }
    // Require >= 2 GPUs. A degenerate 1-device meta backend still routes the
    // whole graph through ggml's meta layer, which aborts on some ops even with
    // a pure-MIRRORED policy (ggml-backend-meta.cpp split-axis assert). With a
    // single GPU there is nothing to shard anyway, so fall back to the normal
    // single-backend path — that is the safe, working configuration.
    if (owner.devs.size() < 2) {
        return nullptr;  // < 2 GPUs -> fall back to the normal path
    }

    ggml_backend_dev_t meta_dev =
        make_meta_device(owner.devs.data(), owner.devs.size(), &owner.split_ctx);
    if (meta_dev == nullptr) {
        return nullptr;
    }
    owner.backend = ggml_backend_dev_init(meta_dev, /*params=*/nullptr);
    return owner.backend;  // may be nullptr -> caller falls back
}

#else  // !SD2R_HAVE_META_BACKEND
// =========================================================================
// Meta backend NOT available in the linked libggml.a (older ggmlR).
// Fallback stubs: the symbols below let other TUs compile, but no meta API
// is ever referenced. meta_enabled() is always false, so callers (e.g.
// SDBackendManager::runtime_backend) take the normal single-backend path.
// A request to enable meta from R is recorded only to emit a warning.
// =========================================================================

// Records whether the user asked for meta (for an R-side warning); has no
// effect on backend selection because meta_enabled() is always false here.
inline bool& meta_backend_requested() {
    static bool g_requested = false;
    return g_requested;
}

inline void set_meta_backend(bool on) {
    meta_backend_requested() = on;
}

// Always false: meta backend was not compiled in.
inline bool meta_enabled() {
    return false;
}

// True when the user asked for meta but it is unavailable — R can warn.
inline bool meta_requested_but_unavailable() {
    return meta_backend_requested();
}

// No meta backend to build; caller falls back to single-backend.
inline ggml_backend_t get_or_build_diffusion_meta_backend() {
    return nullptr;
}

// Owner stub so SDBackendManager::reset() can call free() unconditionally.
struct MetaBackendOwner {
    static MetaBackendOwner& instance() {
        static MetaBackendOwner inst;
        return inst;
    }
    void free() {}
};

#endif  // SD2R_HAVE_META_BACKEND

}  // namespace meta
}  // namespace sd2r

#endif  // SD2R_META_BACKEND_HPP
