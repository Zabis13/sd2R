// sd2R: low-level step-wise C API (sd2R-specific, declared in stable-diffusion.h).
//
// Included at the end of stable-diffusion.cpp so it can use struct sd_ctx_t and
// the StableDiffusionGGML internals without duplicating them. Lets R drive the
// pipeline piece by piece: encode text/image, compute sigmas, scale noise, run
// one denoiser evaluation per step, decode the latent.
//
// Every public call runs inside a StableDiffusionGGML::ExecutionScope (context
// lock + lazy runner loading + end_runners on exit). sd_sampler_begin() opens a
// persistent scope that stays alive until sd_sampler_end(), so the per-step
// calls of an R sampling loop reuse loaded runners/weights instead of preparing
// and releasing them on every step.

#include <memory>
#include <mutex>
#include <unordered_map>

#include "runtime/denoiser.hpp"  // complete type for sd->denoiser (only forward-declared above)
#include "model/diffusion/flux3.hpp"  // Flux3Runner::config.flow_map for sd_denoise_step
#include "model/adapter/lora.hpp"     // LoraModel counters for sd_ctx_lora_stats
#include "model_manager.h"            // ModelManager::lora_apply_stats (sd2R patch)

namespace {

    std::mutex g_sd2r_scope_mutex;
    std::unordered_map<const sd_ctx_t*, std::unique_ptr<StableDiffusionGGML::ExecutionScope>> g_sd2r_scopes;

    bool sd2r_has_persistent_scope(const sd_ctx_t* sd_ctx) {
        std::lock_guard<std::mutex> lock(g_sd2r_scope_mutex);
        return g_sd2r_scopes.find(sd_ctx) != g_sd2r_scopes.end();
    }

    // Runs fn inside the persistent scope (if sd_sampler_begin was called) or a
    // temporary one. Returns fallback if the context cannot be entered.
    template <typename R, typename F>
    R sd2r_run_scoped(sd_ctx_t* sd_ctx, R fallback, F&& fn) {
        if (sd2r_has_persistent_scope(sd_ctx)) {
            return fn();
        }
        StableDiffusionGGML::ExecutionScope execution(*sd_ctx->sd);
        if (!execution.ready) {
            LOG_ERROR("sd2R low-level API: context is busy or failed to prepare runners");
            return fallback;
        }
        return fn();
    }

    sd_tensor sd_tensor_empty() {
        sd_tensor t;
        t.type  = SD_TYPE_F32;
        t.ne[0] = t.ne[1] = t.ne[2] = t.ne[3] = 0;
        t.data   = nullptr;
        t.nbytes = 0;
        return t;
    }

    sd::Tensor<float> sd_tensor_to_tensor(const sd_tensor* t) {
        if (t == nullptr || t->data == nullptr) {
            return {};
        }
        int n_dims = 4;
        while (n_dims > 1 && t->ne[n_dims - 1] == 1) {
            --n_dims;
        }
        std::vector<int64_t> shape(t->ne, t->ne + n_dims);
        sd::Tensor<float> out(shape);
        memcpy(out.data(), t->data, std::min(static_cast<size_t>(t->nbytes), static_cast<size_t>(out.numel()) * sizeof(float)));
        return out;
    }

    sd_tensor tensor_to_sd_tensor(const sd::Tensor<float>& t) {
        if (t.empty()) {
            return sd_tensor_empty();
        }
        sd_tensor out;
        out.type  = SD_TYPE_F32;
        out.ne[0] = out.ne[1] = out.ne[2] = out.ne[3] = 1;
        const std::vector<int64_t>& shape = t.shape();
        for (size_t i = 0; i < shape.size() && i < 4; ++i) {
            out.ne[i] = shape[i];
        }
        out.nbytes = t.numel() * sizeof(float);
        out.data   = malloc(out.nbytes);
        if (out.data != nullptr) {
            memcpy(out.data, t.data(), out.nbytes);
        } else {
            out.nbytes = 0;
        }
        return out;
    }

    SDCondition sd_cond_to_condition(const sd_cond& c) {
        SDCondition out;
        out.c_crossattn = sd_tensor_to_tensor(&c.crossattn);
        out.c_vector    = sd_tensor_to_tensor(&c.vector);
        out.c_concat    = sd_tensor_to_tensor(&c.concat);
        return out;
    }

    // One denoiser evaluation (with CFG) at noise level sigma; returns the
    // denoised prediction x0. Mirrors the per-step core of StableDiffusionGGML::sample()
    // for the plain text-to-image path (no controls, ref latents, masks or caches).
    // sigma_next < 0 means "no FLUX.3 target timestep".
    sd::Tensor<float> sd2r_denoise_once(StableDiffusionGGML* sd,
                                        const sd::Tensor<float>& x,
                                        float sigma,
                                        float sigma_next,
                                        const SDCondition& cond,
                                        const SDCondition& uncond,
                                        float cfg_scale) {
        bool has_unconditioned     = cfg_scale != 1.0f && !uncond.c_crossattn.empty();
        std::vector<float> scaling = sd->denoiser->get_scalings(sigma);
        GGML_ASSERT(scaling.size() == 3);
        float c_skip = scaling[0];
        float c_out  = scaling[1];
        float c_in   = scaling[2];

        std::vector<float> timesteps_vec = sd->prepare_sample_timesteps(sigma, 0);
        sd->adjust_sample_step_scalings(0, timesteps_vec, c_in, &c_skip, &c_out);
        sd::Tensor<float> timesteps_tensor({static_cast<int64_t>(timesteps_vec.size())}, timesteps_vec);
        sd::Tensor<float> guidance_tensor({1}, std::vector<float>{0.f});
        sd::Tensor<float> noised_input = x * c_in;
        sd::Tensor<float> flux3_target_timestep;

        DiffusionParams diffusion_params;
        diffusion_params.x         = &noised_input;
        diffusion_params.timesteps = &timesteps_tensor;
        std::vector<sd::Tensor<float>> empty_controls;
        static const std::vector<sd::Tensor<float>> empty_ref_latents;
        diffusion_params.ref_latents = &empty_ref_latents;
        const SDVersion version      = sd->version;
        if (sd_version_is_unet(version)) {
            diffusion_params.extra = UNetDiffusionExtra{-1, &empty_controls, 0.f};
        } else if (sd_version_is_sd3(version)) {
            diffusion_params.extra = SkipLayerDiffusionExtra{nullptr};
        } else if (sd_version_is_flux(version) || sd_version_is_flux2(version) ||
                   sd_version_is_longcat(version)) {
            diffusion_params.extra = FluxDiffusionExtra{&guidance_tensor, nullptr};
        } else if (sd_version_is_flux3(version)) {
            // Flow-map checkpoints need the next sigma as target_timestep; the runner
            // GGML_ASSERTs on a missing one, so fail softly here instead of aborting R.
            if (sigma_next >= 0.f) {
                flux3_target_timestep = sd::Tensor<float>::from_vector({sigma_next});
            } else {
                auto flux3 = std::dynamic_pointer_cast<Flux3::Flux3Runner>(sd->diffusion_model);
                if (flux3 && flux3->config.flow_map) {
                    LOG_ERROR("sd2R low-level API: FLUX.3 flow-map model needs sigma_next (target timestep)");
                    return {};
                }
            }
            diffusion_params.extra = Flux3DiffusionExtra{flux3_target_timestep.empty() ? nullptr : &flux3_target_timestep};
        }

        auto run_condition = [&](const SDCondition& condition) -> sd::Tensor<float> {
            diffusion_params.context  = condition.c_crossattn.empty() ? nullptr : &condition.c_crossattn;
            diffusion_params.c_concat = condition.c_concat.empty() ? nullptr : &condition.c_concat;
            diffusion_params.y        = condition.c_vector.empty() ? nullptr : &condition.c_vector;
            return sd->diffusion_model->compute(sd->n_threads, diffusion_params);
        };

        sd::Tensor<float> out_cond = run_condition(cond);
        if (out_cond.empty()) {
            return {};
        }
        sd::Tensor<float> pred;
        if (has_unconditioned) {
            sd::Tensor<float> out_uncond = run_condition(uncond);
            if (out_uncond.empty()) {
                return {};
            }
            pred = out_uncond + (out_cond - out_uncond) * cfg_scale;
        } else {
            pred = out_cond;
        }
        return pred * c_out + x * c_skip;
    }

    std::vector<float> sd2r_sigmas(StableDiffusionGGML* sd,
                                   const sd_ctx_t* sd_ctx,
                                   enum scheduler_t scheduler,
                                   enum sample_method_t sample_method,
                                   int sample_steps,
                                   int width,
                                   int height,
                                   const char* extra_sample_args) {
        if (sample_method == SAMPLE_METHOD_COUNT) {
            sample_method = sd_get_default_sample_method(sd_ctx);
        }
        if (scheduler == SCHEDULER_COUNT) {
            scheduler = sd_get_default_scheduler(sd_ctx, sample_method);
        }
        return sd->denoiser->get_sigmas(sample_steps,
                                        sd->get_image_seq_len(height, width),
                                        scheduler,
                                        sd->version,
                                        extra_sample_args);
    }

}  // namespace

// Called from free_sd_ctx(): drop a persistent scope left open by R (e.g. an
// error between sd_sampler_begin and sd_sampler_end) before the engine is deleted.
static void sd2r_lowlevel_release(sd_ctx_t* sd_ctx) {
    std::unique_ptr<StableDiffusionGGML::ExecutionScope> scope;
    {
        std::lock_guard<std::mutex> lock(g_sd2r_scope_mutex);
        auto it = g_sd2r_scopes.find(sd_ctx);
        if (it != g_sd2r_scopes.end()) {
            scope = std::move(it->second);
            g_sd2r_scopes.erase(it);
        }
    }
}

SD_API bool sd_ctx_is_valid(const sd_ctx_t* sd_ctx) {
    return sd_ctx != nullptr && sd_ctx->sd != nullptr;
}

SD_API int sd_ctx_get_version(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return -1;
    }
    return (int)sd_ctx->sd->version;
}

SD_API int sd_ctx_get_vae_scale_factor(sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return 8;
    }
    if (sd_version_is_pid(sd_ctx->sd->version)) {
        return 1;
    }
    if (sd_ctx->sd->first_stage_model == nullptr) {
        return 8;
    }
    return sd_ctx->sd->get_vae_scale_factor();
}

SD_API int sd_ctx_lora_stats(sd_ctx_t* sd_ctx, sd_lora_stat_t* out, int max_count) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return 0;
    }
    StableDiffusionGGML* sd = sd_ctx->sd;
    struct Counts {
        int64_t total = 0, applied = 0, skipped = 0;
    };
    std::map<std::string, Counts> acc;
    if (sd->apply_lora_immediately) {
        // immediately: counters accumulated by ModelManager::apply_loras_to_params (sd2R patch)
        if (sd->model_manager != nullptr) {
            for (const auto& kv : sd->model_manager->lora_apply_stats()) {
                Counts& c = acc[kv.first];
                c.total   = static_cast<int64_t>(kv.second.total);
                c.skipped = static_cast<int64_t>(kv.second.skipped.size());
                for (const auto& name : kv.second.applied) {
                    if (kv.second.skipped.find(name) == kv.second.skipped.end()) {
                        c.applied++;
                    }
                }
            }
        }
    } else {
        // at_runtime: one LoraModel per (LoRA, module); same counting as LoraModel::stat()
        for (const auto& entry : sd->runtime_lora_models) {
            if (entry.model == nullptr) {
                continue;
            }
            Counts& c = acc[entry.spec.path];
            for (const auto& kv : entry.model->lora_tensors) {
                c.total++;
                if (entry.model->skipped_incompatible_lora_tensors.count(kv.first)) {
                    c.skipped++;
                } else if (entry.model->applied_lora_tensors.count(kv.first)) {
                    c.applied++;
                }
            }
        }
    }
    int i = 0;
    for (const auto& kv : acc) {
        if (out != nullptr && i < max_count) {
            snprintf(out[i].path, sizeof(out[i].path), "%s", kv.first.c_str());
            out[i].total   = kv.second.total;
            out[i].applied = kv.second.applied;
            out[i].skipped = kv.second.skipped;
        }
        i++;
    }
    return i;
}

SD_API int sd_ctx_lora_apply_mode(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return -1;
    }
    return sd_ctx->sd->apply_lora_immediately ? (int)LORA_APPLY_IMMEDIATELY : (int)LORA_APPLY_AT_RUNTIME;
}

SD_API bool sd_ctx_supports_ref_images(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    SDVersion v = sd_ctx->sd->version;
    return sd_version_is_unet_edit(v) ||
           sd_version_is_control(v) ||
           sd_version_is_dit(v);
}

SD_API sd_cond sd_encode_text(sd_ctx_t* sd_ctx, const char* prompt, int clip_skip, int width, int height) {
    sd_cond out;
    out.crossattn = sd_tensor_empty();
    out.vector    = sd_tensor_empty();
    out.concat    = sd_tensor_empty();
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || prompt == nullptr) {
        return out;
    }
    return sd2r_run_scoped(sd_ctx, out, [&]() {
        if (sd_ctx->sd->cond_stage_model == nullptr) {
            return out;
        }
        ConditionerParams cp;
        cp.text       = std::string(prompt);
        cp.clip_skip  = clip_skip;
        cp.width      = width;
        cp.height     = height;
        SDCondition c = sd_ctx->sd->get_learned_condition(cp);
        out.crossattn = tensor_to_sd_tensor(c.c_crossattn);
        out.vector    = tensor_to_sd_tensor(c.c_vector);
        out.concat    = tensor_to_sd_tensor(c.c_concat);
        return out;
    });
}

SD_API sd_image_t sd_decode_latent(sd_ctx_t* sd_ctx, sd_tensor latent) {
    sd_image_t out;
    out.width = out.height = out.channel = 0;
    out.data                             = nullptr;
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || latent.data == nullptr) {
        return out;
    }
    return sd2r_run_scoped(sd_ctx, out, [&]() {
        sd::Tensor<float> lat = sd_tensor_to_tensor(&latent);
        // sd_tensor_to_tensor drops trailing singleton dims; the VAE expects a
        // 4D latent [W, H, C, N], otherwise it returns a 3D image.
        while (lat.dim() < 4) {
            lat.unsqueeze_(static_cast<size_t>(lat.dim()));
        }
        sd::Tensor<float> img = sd_ctx->sd->decode_first_stage(lat);
        if (!img.empty()) {
            // tensor_to_sd_image GGML_ASSERTs on other ranks; fail softly instead of aborting R.
            if (img.dim() != 4 && img.dim() != 5) {
                LOG_ERROR("sd2R low-level API: decoded image has %d dims, expected 4 or 5",
                          static_cast<int>(img.dim()));
                return out;
            }
            out = tensor_to_sd_image(img);
        }
        return out;
    });
}

SD_API sd_tensor sd_encode_image(sd_ctx_t* sd_ctx, sd_image_t image) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || image.data == nullptr) {
        return sd_tensor_empty();
    }
    return sd2r_run_scoped(sd_ctx, sd_tensor_empty(), [&]() {
        sd::Tensor<float> img    = sd_image_to_tensor(image);
        sd::Tensor<float> latent = sd_ctx->sd->encode_first_stage(img);
        return tensor_to_sd_tensor(latent);
    });
}

SD_API sd_tensor sd_sample_loop(sd_ctx_t* sd_ctx,
                                sd_tensor init_latent,
                                sd_tensor noise,
                                sd_cond cond,
                                sd_cond uncond,
                                const sd_sample_params_t* sample_params,
                                float strength) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr ||
        sample_params == nullptr || noise.data == nullptr) {
        return sd_tensor_empty();
    }
    return sd2r_run_scoped(sd_ctx, sd_tensor_empty(), [&]() {
        StableDiffusionGGML* sd   = sd_ctx->sd;
        sd::Tensor<float> noise_t = sd_tensor_to_tensor(&noise);
        sd::Tensor<float> x_t     = sd_tensor_to_tensor(&init_latent);
        bool is_img2img           = !x_t.empty();
        if (x_t.empty()) {
            x_t = sd::Tensor<float>(noise_t.shape());  // zero-initialized
        }
        int W = static_cast<int>(noise_t.shape()[0]);
        int H = static_cast<int>(noise_t.shape()[1]);

        SDCondition c  = sd_cond_to_condition(cond);
        SDCondition uc = sd_cond_to_condition(uncond);

        enum sample_method_t sample_method = sample_params->sample_method;
        if (sample_method == SAMPLE_METHOD_COUNT) {
            sample_method = sd_get_default_sample_method(sd_ctx);
        }
        sd->set_flow_shift(sample_params->flow_shift);

        int sample_steps = sample_params->sample_steps;
        std::vector<float> sigmas;
        if (sample_params->custom_sigmas_count > 0) {
            sigmas       = std::vector<float>(sample_params->custom_sigmas,
                                              sample_params->custom_sigmas + sample_params->custom_sigmas_count);
            sample_steps = static_cast<int>(sigmas.size()) - 1;
        } else {
            int vae_scale_factor = sd->get_vae_scale_factor();
            sigmas               = sd2r_sigmas(sd, sd_ctx, sample_params->scheduler, sample_method, sample_steps,
                                               W * vae_scale_factor, H * vae_scale_factor,
                                               sample_params->extra_sample_args);
        }

        if (is_img2img && strength < 1.0f) {
            size_t t_enc = static_cast<size_t>(sample_steps * strength);
            if (t_enc == static_cast<size_t>(sample_steps)) {
                t_enc--;
            }
            std::vector<float> sigma_sched;
            sigma_sched.assign(sigmas.begin() + sample_steps - t_enc - 1, sigmas.end());
            sigmas = sigma_sched;
        }

        sd::Tensor<float> x_0 = sd->sample(sd->diffusion_model,
                                           true,                        // inverse_noise_scaling
                                           x_t,                         // init_latent
                                           std::move(noise_t),
                                           c,
                                           uc,
                                           SDCondition(),               // img_uncond
                                           sd::Tensor<float>(),         // control_image
                                           0.0f,                        // control_strength
                                           sample_params->guidance,
                                           sample_params->eta,
                                           sample_params->shifted_timestep,
                                           sample_method,
                                           sd->is_flow_denoiser(),
                                           sample_params->extra_sample_args,
                                           sigmas,
                                           {},                          // ref_latents
                                           sd->resolve_ref_image_params(""),
                                           sd::Tensor<float>(),         // denoise_mask
                                           sd::Tensor<float>(),         // vace_context
                                           1.0f,                        // vace_strength
                                           0,                           // audio_length
                                           0.0f,                        // frame_rate
                                           nullptr,                     // cache_params
                                           false);                      // preview_final_step
        return tensor_to_sd_tensor(x_0);
    });
}

SD_API void sd_tensor_free(sd_tensor* t) {
    if (t != nullptr && t->data != nullptr) {
        free(t->data);
        t->data   = nullptr;
        t->nbytes = 0;
    }
}

SD_API void sd_cond_free(sd_cond* c) {
    if (c != nullptr) {
        sd_tensor_free(&c->crossattn);
        sd_tensor_free(&c->vector);
        sd_tensor_free(&c->concat);
    }
}

SD_API void sd_floats_free(float* p) {
    free(p);
}

SD_API float* sd_sampler_sigmas(sd_ctx_t* sd_ctx,
                                enum scheduler_t scheduler,
                                enum sample_method_t sample_method,
                                int sample_steps,
                                int width,
                                int height,
                                int* out_len) {
    if (out_len != nullptr) {
        *out_len = 0;
    }
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || sd_ctx->sd->denoiser == nullptr ||
        sample_steps <= 0) {
        return nullptr;
    }
    // Same default flow shift as sd_sample_loop / generate; without it a fresh
    // context keeps the denoiser's constructor shift (e.g. FLUX.3: 3 vs 6.93).
    sd_ctx->sd->set_flow_shift();
    std::vector<float> sigmas = sd2r_sigmas(sd_ctx->sd, sd_ctx, scheduler, sample_method,
                                            sample_steps, width, height, nullptr);
    int n    = static_cast<int>(sigmas.size());
    float* p = (float*)malloc(sizeof(float) * static_cast<size_t>(n));
    if (p == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < n; i++) {
        p[i] = sigmas[i];
    }
    if (out_len != nullptr) {
        *out_len = n;
    }
    return p;
}

SD_API void sd_sampler_begin(sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || sd2r_has_persistent_scope(sd_ctx)) {
        return;
    }
    auto scope = std::make_unique<StableDiffusionGGML::ExecutionScope>(*sd_ctx->sd);
    if (!scope->ready) {
        LOG_ERROR("sd_sampler_begin: context is busy or failed to prepare runners");
        return;
    }
    std::lock_guard<std::mutex> lock(g_sd2r_scope_mutex);
    g_sd2r_scopes[sd_ctx] = std::move(scope);
}

SD_API void sd_sampler_end(sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr) {
        return;
    }
    sd2r_lowlevel_release(sd_ctx);  // ExecutionScope dtor runs end_runners()
}

SD_API sd_tensor sd_noise_scale(sd_ctx_t* sd_ctx,
                                sd_tensor init_latent,
                                sd_tensor noise,
                                float sigma0) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || sd_ctx->sd->denoiser == nullptr ||
        noise.data == nullptr) {
        return sd_tensor_empty();
    }
    sd::Tensor<float> noise_t = sd_tensor_to_tensor(&noise);
    sd::Tensor<float> x_t     = sd_tensor_to_tensor(&init_latent);
    if (x_t.empty()) {
        x_t = sd::Tensor<float>(noise_t.shape());  // zero-initialized
    }
    sd::Tensor<float> x = sd_ctx->sd->denoiser->noise_scaling(sigma0, noise_t, x_t);
    return tensor_to_sd_tensor(x);
}

SD_API sd_tensor sd_inverse_noise_scale(sd_ctx_t* sd_ctx,
                                        sd_tensor x,
                                        float sigma_last) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || sd_ctx->sd->denoiser == nullptr ||
        x.data == nullptr) {
        return sd_tensor_empty();
    }
    sd::Tensor<float> x_t = sd_tensor_to_tensor(&x);
    sd::Tensor<float> r   = sd_ctx->sd->denoiser->inverse_noise_scaling(sigma_last, x_t);
    return tensor_to_sd_tensor(r);
}

SD_API sd_tensor sd_denoise_step(sd_ctx_t* sd_ctx,
                                 sd_tensor x,
                                 float sigma,
                                 float sigma_next,
                                 sd_cond cond,
                                 sd_cond uncond,
                                 float cfg_scale,
                                 int step,
                                 int total_steps) {
    (void)step;
    (void)total_steps;
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || x.data == nullptr) {
        return sd_tensor_empty();
    }
    return sd2r_run_scoped(sd_ctx, sd_tensor_empty(), [&]() {
        sd::Tensor<float> x_t      = sd_tensor_to_tensor(&x);
        SDCondition c              = sd_cond_to_condition(cond);
        SDCondition uc             = sd_cond_to_condition(uncond);
        sd::Tensor<float> denoised = sd2r_denoise_once(sd_ctx->sd, x_t, sigma, sigma_next, c, uc, cfg_scale);
        return tensor_to_sd_tensor(denoised);
    });
}

SD_API sd_tensor sd_euler_step(sd_tensor x,
                               sd_tensor denoised,
                               float sigma,
                               float sigma_next) {
    if (x.data == nullptr || denoised.data == nullptr || x.nbytes != denoised.nbytes) {
        return sd_tensor_empty();
    }
    // Same expressions as sample_euler() (runtime/denoiser.hpp) for bit-exact results.
    sd::Tensor<float> x_t = sd_tensor_to_tensor(&x);
    if (sigma != sigma_next) {
        sd::Tensor<float> den_t = sd_tensor_to_tensor(&denoised);
        sd::Tensor<float> d     = (x_t - den_t) / sigma;
        x_t += d * (sigma_next - sigma);
    }
    return tensor_to_sd_tensor(x_t);
}
