#include "gated_delta_net.cuh"
#include "gated-delta-net-mma.cuh"
#include "gated_delta_net_chunked.cuh"
#include "ggml-cuda/common.cuh"
#include "mk-ops-gdn.cuh"
#include "ggml-cuda-try-fuse.cuh"

#include <cstdint>
#include <unordered_map>

static thread_local std::unordered_map<const ggml_tensor *, ggml_cuda_gdn_state_src> gdn_state_srcs;

struct gdn_prefix_src {
    const ggml_tensor * replay;
    float *             out;
};
static thread_local std::unordered_map<const ggml_tensor *, gdn_prefix_src> gdn_prefix_srcs;

void ggml_cuda_gdn_clear_state_srcs() {
    gdn_state_srcs.clear();
    gdn_prefix_srcs.clear();
}

void ggml_cuda_gdn_set_state_src(const ggml_tensor * gdn, const ggml_cuda_gdn_state_src & src) {
    gdn_state_srcs[gdn] = src;
}

bool ggml_cuda_gdn_get_state_src(const ggml_tensor * gdn, ggml_cuda_gdn_state_src & src) {
    const auto it = gdn_state_srcs.find(gdn);
    if (it == gdn_state_srcs.end()) {
        return false;
    }
    src = it->second;
    return true;
}

// The kernel is latency-bound on DRAM state loads, so four columns per warp keep more loads in flight.
static constexpr int gdn_cols_per_warp(int S_v) {
    return S_v == 128 ? 4 : 1;
}

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__(mk_gdn_step_warp_size<S_v>() * mk_gdn_step_warps, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     int           beta_sigmoid,
                                     const int32_t * s_ids,
                                     int64_t       s_row_stride,
                                     float *       state_pre,
                                     const mk_gdn_prefix prefix) {
    mk_gdn_step_params p = {
        q, k, v, g, beta, curr_state, dst, state, H, n_tokens, n_seqs, gridDim.z,
        sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale,
        state_slot_stride, K, s_ids, s_row_stride, state_pre, nullptr,
    };
    p.beta_sigmoid = beta_sigmoid;
    p.prefix       = prefix;
    ggml_cuda_pdl_sync();
    mk_gdn_step_tile<S_v, KDA, keep_rs_t, gdn_cols_per_warp(S_v)>(p, blockIdx.x, blockIdx.z, blockIdx.y, threadIdx.y*blockDim.x + threadIdx.x);
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, int beta_sigmoid,
        const int32_t * s_ids, int64_t s_row_stride, float * state_pre, cudaStream_t stream,
        const mk_gdn_prefix & prefix = {}) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = mk_gdn_step_warps;
    const int ncols = gdn_cols_per_warp(S_v);
    dim3      grid_dims(H, n_seqs, (S_v + num_warps*ncols - 1) / (num_warps*ncols));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, beta_sigmoid, s_ids, s_row_stride, state_pre, prefix);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, beta_sigmoid, s_ids, s_row_stride, state_pre, prefix);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, beta_sigmoid, s_ids, s_row_stride, state_pre, prefix);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, beta_sigmoid, s_ids, s_row_stride, state_pre, prefix);
            break;
        }
        default:
            GGML_ABORT("fatal error");
    }
}

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    return {
        (const float *) q->data,
        (const float *) k->data,
        (const float *) v->data,
        (const float *) g->data,
        (const float *) beta->data,
        (const float *) state->data,
        (float *) dst->data,
        nullptr,
        H,
        q->ne[1],
        v->ne[2],
        v->ne[3],
        v->ne[3] / q->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)),
        (int64_t) (q->nb[2] / sizeof(float)),
        (int64_t) (q->nb[3] / sizeof(float)),
        (int64_t) (v->nb[1] / sizeof(float)),
        (int64_t) (v->nb[2] / sizeof(float)),
        (int64_t) (v->nb[3] / sizeof(float)),
        (int64_t) (beta->nb[1] / sizeof(float)),
        (int64_t) (beta->nb[2] / sizeof(float)),
        (int64_t) (beta->nb[3] / sizeof(float)),
        1.0f / sqrtf((float) S_v),
        g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) < v->ne[2] && S_v == 128 && q->ne[0] == 128,
    };
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache,
        const ggml_tensor * beta_presig = nullptr) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    // a folded replay: its tokens go first, from its own input state, and the state after them goes to `out`
    const auto prefix_it = gdn_prefix_srcs.find(dst);
    const ggml_tensor * replay = prefix_it != gdn_prefix_srcs.end() ? prefix_it->second.replay : nullptr;
    mk_gdn_prefix prefix = {};
    if (replay != nullptr) {
        src_state = replay->src[5];
        const ggml_tensor * kv = replay->src[2]->view_src;
        const ggml_tensor * gb = replay->src[4]->view_src;
        prefix = {
            (const float *) kv->src[0]->data, (const float *) gb->src[0]->data,
            (const int32_t *) kv->src[1]->data, (const int32_t *) gb->src[1]->data,
            (int64_t) (kv->src[0]->nb[1]/sizeof(float)), (int64_t) (gb->src[0]->nb[1]/sizeof(float)),
            (int64_t) (replay->src[1]->nb[1]/sizeof(float)),
            (int64_t) (((const char *) replay->src[2]->data - (const char *) kv->data)/sizeof(float)),
            (int64_t) (replay->src[2]->nb[1]/sizeof(float)),
            (int) replay->src[2]->ne[2], prefix_it->second.out,
        };
    }
    const ggml_tensor * state_owner = replay != nullptr ? replay : dst;

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    GGML_ASSERT(neq3 > 0 && nev3 >= neq3 && nev3 % neq3 == 0);
    GGML_ASSERT(nek3 == neq3);
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    // beta_presig: the input of a fused UNARY(sigmoid) beta node (contiguous, same element order as the contiguous beta,
    // so beta's strides index it); the sequential kernel applies the
    // sigmoid.  Only the decode/verify band uses it (n_seqs == 1, n_tokens <= 16 never takes the chunked path below).
    const int     beta_sigmoid = beta_presig != nullptr;
    const float * b_d = (const float *) (beta_presig ? beta_presig->data : src_beta->data);
    GGML_ASSERT(!beta_sigmoid || (n_seqs == 1 && n_tokens <= 16 && ggml_is_contiguous(beta_presig) &&
                                  ggml_nelements(beta_presig) == ggml_nelements(src_beta)));

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    // the state's GET_ROWS was deferred by the graph loop (ggml_cuda_gdn_state_gather_deferrable): read the
    // cache rows it would have copied.  Decode/verify band only (checked there), so the sequential kernel runs.
    const int32_t * s_ids        = nullptr;
    int64_t         s_row_stride = 0;
    {
        const ggml_tensor * root = src_state;
        while (root->view_src != nullptr) {
            root = root->view_src;
        }
        if (std::find(ctx.gdn_deferred_gathers.begin(), ctx.gdn_deferred_gathers.end(), root) != ctx.gdn_deferred_gathers.end()) {
            GGML_ASSERT(n_seqs == 1 && n_tokens <= 16 && src_state->data == root->data);
            s_d          = (const float *) root->src[0]->data;
            s_ids        = (const int32_t *) root->src[1]->data;
            s_row_stride = root->src[0]->nb[1] / sizeof(float);
        }
    }
    float *         state_pre    = nullptr;
    ggml_cuda_gdn_state_src redirect;
    if (ggml_cuda_gdn_get_state_src(state_owner, redirect)) {
        GGML_ASSERT(n_seqs == 1 && s_ids == nullptr);
        s_d          = redirect.base;
        s_ids        = redirect.ids;
        s_row_stride = redirect.row_stride;
        state_pre    = redirect.state_pre;
    }

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // op param 1 is the largest per-seq batch that can be rolled back into (0 = unset). The
    // rollback snapshots only cover K tokens and only the sequential kernel writes them, so a
    // batch longer than this bound must not take the chunked path. K alone is NOT a safe bound:
    // the snapshot depth comes from the draft-model n_max, while ngram-style speculators can draft
    // far longer (ngram-mod n_max = 64 by default). A verify batch then holds 1 + n_max tokens and
    // is exactly the batch that gets rolled back into, so the bound must cover the longest draft.
    const int64_t n_rs_batch = keep_rs ? std::max<int64_t>(1, ggml_get_op_params_i32(dst, 1)) : 0;

    // Chunked prefill with a FIXED-D sequential tail.  The chunked kernel is not bit-exact with
    // the sequential kernel (the fp32 path differs only in rounding order; the bf16/WMMA path is
    // materially lossy -- issue #113) and computes NO per-token states, so a rollback into a batch
    // it served would restore a slot it never wrote (measured: logit NMSE 0.09-0.72).  One rule
    // handles both: a single-sequence batch always finishes with D = GDN_CHUNKED_MIN_TOKENS tokens
    // decoded by the SEQUENTIAL kernel, so slots 0..D-1 exist for every batch and the state handed
    // to the next batch is an fp32 sequential result.  D depends on K only above K > 16 (and on
    // n_rs_batch), never on the draft depth inside the clamp: a K-dependent boundary would make
    // plain decode and MTP chunk a prefill differently, and the two post-prefill states would
    // disagree (greedy near-ties flipping between --spec-type none and draft-mtp).
    //
    // The bound is ENFORCED, not merely observed.  `llama_context_params::n_rs_batch` is the widest
    // verify batch any enabled speculator can present -- common sets it to
    // `common_speculative_n_max() + 1`, which covers draft-simple/mtp/adaptive/eagle3/dflash/dspark
    // AND ngram-simple/map/mod/cache (ngram-mod at its default n_max = 64 raises it to 65) -- and
    // this op receives it as op param 1.  GDN_CHUNKED_MIN_TOKENS is floored at that value, so a
    // speculative verify batch can never take the chunked path: it is always at or below the
    // threshold, falls through to the sequential kernel below, and restores only slots it wrote.
    // The tail then covers the remaining source of a partial rollback, a NON-speculator rewind into
    // a prefill batch (server cancel / --cache-reuse) -- without it that case would read an
    // unwritten slot.  Cost stays bounded: D tokens per batch, not per token, no extra memory, no
    // extra graph and no ubatch split (which measured -13.7 % pp512; see
    // archive/work/rs-rollback-boundary/).  GGML_CUDA_GDN_TAIL=0 restores whole-batch chunking.
    //
    // The threshold must be a constant for K <= 16 or the two paths diverge again on short prompts;
    // 16 covers K <= 16 (n_max <= 15, including adaptive MTP's recommended n_max = 12).  The floor
    // at K keeps verify batches sequential for deeper drafts, so those keep the K-dependent
    // boundary and stay correct-but-not-bit-identical, exactly as before.
    //
    // The fused GDN->cpy (cache != nullptr) is still honoured: the chunked kernel writes the new
    // state (slot 0) straight into the cache slot, so the cpy node can be skipped.  n_seqs > 1
    // keeps the whole-ubatch chunked path for K == 1 -- the chunked n_tokens override is also the
    // sequence stride, so only n_seqs == 1 may choose per-batch.  GGML_CUDA_GDN_CHUNKED=0 forces
    // the sequential kernel everywhere (the A/B switch, and the fully-snapshot-safe fallback).
    // The chunked ops report a synchronous launch rejection (ROCm: AQL dispatch refused, e.g. a
    // code object carrying a hidden hostcall buffer on a platform without PCIe atomics ->
    // hipErrorIllegalState, llama-cpp-rdna-boosts#2); a rejected launch has no side effects, so we
    // recompute with the sequential kernel below, bit-identical to GGML_CUDA_GDN_CHUNKED=0.  An
    // async fault inside a launched kernel is NOT caught here and surfaces at the next sync point.
    GGML_ASSERT(replay == nullptr || (!kda && !keep_rs && state_pre == nullptr && n_seqs == 1 && n_tokens <= 16));
    if (replay == nullptr && !kda && n_tokens > 1 &&
        (S_v == 16 || S_v == 32 || S_v == 64 || S_v == 128)) {
        // read once as a value: a cached getenv() pointer is unsafe on Windows (the CRT buffer is reused)
        static const bool chunked_on = [] {
            const char * env = getenv("GGML_CUDA_GDN_CHUNKED");
            return env == nullptr || strcmp(env, "0") != 0;
        }();
        if (chunked_on) {
            // Threshold for the chunked path: a batch above it is chunked on its first
            // n_tokens - D tokens and finished by the sequential tail.  A constant for K <= 16 (see
            // above); the floors at K and n_rs_batch keep deeper drafts and every possible verify
            // batch entirely on the sequential kernel, which writes the rollback snapshots.
            const int64_t GDN_CHUNKED_MIN_TOKENS = std::max<int64_t>(K > 16 ? (int64_t) K : 16, n_rs_batch);
            // GGML_CUDA_GDN_CHUNKED_BF16 / arch selection: the bf16/WMMA chunked kernel is the
            // DEFAULT for S_v == 128 prefill on RDNA3/RDNA4 (opt out with `=0`).  It is
            // near-lossless, NOT bit-exact (bf16 operands, fp32 accumulation, ~1e-5 op NMSE);
            // the r29 default-off was a workaround for the n_seqs > 1 `A_sc` stride aliasing
            // bug (llama-cpp-rdna-boosts#113), fixed in r30.  The fp32 chunked kernel is the
            // opt-out and the non-RDNA fallback.  n_tokens_limit > 0 truncates to a single-seq
            // prefix.
            auto launch_chunked = [&](float * state_d_ext, int64_t n_tokens_limit) -> bool {
#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
                const int cc_c = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
                const bool bf16_rdna_c = GGML_CUDA_CC_IS_RDNA4(cc_c) || GGML_CUDA_CC_IS_RDNA3(cc_c);
                static const bool bf16_env_on = [] {
                    const char * envb_c = getenv("GGML_CUDA_GDN_CHUNKED_BF16");
                    return envb_c == nullptr || strcmp(envb_c, "0") != 0;
                }();
                const bool want_bf16_c = bf16_rdna_c && S_v == 128 && bf16_env_on;
                if (want_bf16_c) {
                    if (GGML_CUDA_CC_IS_RDNA4(cc_c)) {
                        return ggml_cuda_op_gated_delta_net_chunked_bf16(ctx, dst, state_d_ext, n_tokens_limit);
                    }
                    return ggml_cuda_op_gated_delta_net_chunked_bf16_gfx11(ctx, dst, state_d_ext, n_tokens_limit);
                }
#endif
                return ggml_cuda_op_gated_delta_net_chunked(ctx, dst, state_d_ext, n_tokens_limit);
            };

            if (n_seqs == 1) {
                if (n_tokens > GDN_CHUNKED_MIN_TOKENS) {
                    // The snapshot tail (see the long note above): chunk the PREFIX and run the
                    // sequential kernel on the last `D = GDN_CHUNKED_MIN_TOKENS` tokens, in the SAME
                    // launch.  The chunked kernel takes that prefix length directly
                    // (`n_tokens_limit`), and D is the batch size the sequential kernel is anyway used
                    // for, so slots 0..D-1 are written and a rollback of up to D tokens is always a
                    // restore of state this batch produced.  D is fixed (not K-dependent) so plain
                    // decode and MTP chunk a prefill identically -- `none == draft-mtp` needs that.
                    static const bool tail_on = [] {
                        const char * env = getenv("GGML_CUDA_GDN_TAIL");
                        return env == nullptr || strcmp(env, "0") != 0;
                    }();
                    const int64_t D = GDN_CHUNKED_MIN_TOKENS;
                    // the state after the prefix lands here: the fused cache slot when the GDN->cpy is
                    // fused, else the snapshot tail of the output (computed from the FULL n_tokens, which
                    // the chunked op would otherwise shrink to the limit)
                    float * const tail_base = cache ? cache->data : dst_d + S_v * H * n_tokens * n_seqs;
                    const int64_t n_head = tail_on ? n_tokens - D : 0;
                    if (launch_chunked(tail_base, n_head)) {
                        if (n_head > 0) {
                            // the tail: same kernel, last D tokens.  q/k/g/beta advance by sq2 per token,
                            // v by sv2, the attention scores by S_v*H (n_seqs == 1 here).  The input
                            // state is the prefix's final state (slot 0 of `tail_base`); the kernel reads
                            // the state into registers before writing any slot, so aliasing it with the
                            // slot output is safe, and slot 0 is written last.
                            const float * q_t = q_d + n_head * sq2;
                            const float * k_t = k_d + n_head * sq2;
                            const float * v_t = v_d + n_head * sv2;
                            const float * g_t = g_d + n_head * sb2;
                            const float * b_t = b_d + n_head * sb2;
                            float * const dst_t = dst_d + n_head * S_v * H;
                            const int64_t slot_stride = cache ? cache->slot_stride : S_v * S_v * H * n_seqs;
                            if (keep_rs) {
                                launch_gated_delta_net<false, true>(q_t, k_t, v_t, g_t, b_t, tail_base,
                                    dst_t, tail_base, S_v, H, D, n_seqs,
                                    sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3,
                                    scale, slot_stride, K, beta_sigmoid, s_ids, s_row_stride, nullptr, stream);
                            } else {
                                launch_gated_delta_net<false, false>(q_t, k_t, v_t, g_t, b_t, tail_base,
                                    dst_t, tail_base, S_v, H, D, n_seqs,
                                    sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3,
                                    scale, slot_stride, K, beta_sigmoid, s_ids, s_row_stride, nullptr, stream);
                            }
                        }
                        static bool logged_free = false;
                        if (!logged_free) {
                            GGML_LOG_INFO("%s: chunked GDN prefill n=%ld K=%d n_seqs=%ld (chunked %ld + "
                                          "sequential tail %ld, snapshots written; GGML_CUDA_GDN_TAIL=0 "
                                          "restores whole-batch chunking)\n",
                                          __func__, (long) n_tokens, K, (long) n_seqs,
                                          (long) n_head, (long) (n_head > 0 ? D : 0));
                            logged_free = true;
                        }
                        return;
                    }
                    GGML_LOG_WARN("%s: chunked GDN launch rejected by the driver; falling back to the sequential kernel (GGML_CUDA_GDN_CHUNKED=0 equivalent)\n", __func__);
                }
            } else if (K == 1) {
                // Multi-sequence ubatch: the kernel choice must not depend on how many sequences
                // share the ubatch, or a sequence's logits depend on its neighbours.  Use the same
                // per-token threshold as the single-sequence branch above; the whole-batch chunked
                // path is then only taken for batches a solo decode would also chunk.
                if (n_tokens > GDN_CHUNKED_MIN_TOKENS) {
                    if (launch_chunked(cache ? cache->data : nullptr, 0)) {
                        return;
                    }
                    GGML_LOG_WARN("%s: chunked GDN launch rejected by the driver; falling back to the sequential kernel (GGML_CUDA_GDN_CHUNKED=0 equivalent)\n", __func__);
                }
            }
        }
    }

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (replay == nullptr) {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        args.state_out = state_d;
        if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
            return;
        }
    }
    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, beta_sigmoid, s_ids, s_row_stride, state_pre, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, beta_sigmoid, s_ids, s_row_stride, state_pre, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, beta_sigmoid, s_ids, s_row_stride, state_pre, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, beta_sigmoid, s_ids, s_row_stride, state_pre, stream,
                prefix);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}

void ggml_cuda_op_gated_delta_net_beta_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * beta_presig,
                                               const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, cache, beta_presig);
}


// moved out of ggml-cuda.cu (Phase-2 self-contained extraction; see PHASE2-REFACTOR.md)
int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// UNARY(sigmoid) beta -> GATED_DELTA_NET (GGML_CUDA_FUSE_GDN_BETA_SIGMOID=0: off): the sequential
// GDN kernel applies the sigmoid to the beta it loads.  Decode/verify band only (single sequence,
// <= 16 tokens: never the chunked kernels).  GGML_CUDA_DISABLE_GDN_CPY=1 also disables the fused
// snapshot-copy scatter.
int ggml_cuda_gdn_beta_sigmoid_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool disable_gdn_cpy = [] { const char * e = getenv("GGML_CUDA_DISABLE_GDN_CPY"); return e != nullptr && std::atoi(e); }();
    static const bool fuse_gdn_beta_sig = [] { const char * e = getenv("GGML_CUDA_FUSE_GDN_BETA_SIGMOID"); return e == nullptr || atoi(e) != 0; }();
    const ggml_tensor * node = cgraph->nodes[i];
    if (!fuse_gdn_beta_sig || node->op != GGML_OP_UNARY || ggml_get_unary_op(node) != GGML_UNARY_OP_SIGMOID ||
            node->type != GGML_TYPE_F32 || node->src[0]->type != GGML_TYPE_F32 || !ggml_is_contiguous(node) ||
            !ggml_is_contiguous(node->src[0]) || !ggml_are_same_shape(node, node->src[0])) {
        return 0;
    }
    // view/no-op nodes that do not read the sigmoid may sit between it and the GDN (the state reshape does)
    int j = i + 1;
    while (j < cgraph->n_nodes && j <= i + 4 && ggml_cuda_is_view_or_noop(cgraph->nodes[j]) &&
           cgraph->nodes[j]->src[0] != node && cgraph->nodes[j]->view_src != node) {
        ++j;
    }
    if (j < cgraph->n_nodes && cgraph->nodes[j]->op == GGML_OP_GATED_DELTA_NET && cgraph->nodes[j]->src[4] == node &&
            cgraph->nodes[j]->src[2]->ne[2] <= 16 && cgraph->nodes[j]->src[2]->ne[3] == 1 &&
            ggml_node_get_use_count(cgraph, i) == 1 && !(node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ggml_tensor * gdn = cgraph->nodes[j];
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = disable_gdn_cpy ? 0 : ggml_cuda_try_gdn_cache_fusion(cgraph, j, fused_state_cpy);
        ggml_cuda_op_gated_delta_net_beta_sigmoid(ctx, gdn, node->src[0], nodes_to_skip > 0 ? &fused_state_cpy : nullptr);
        return (j - i) + nodes_to_skip;
    }
    return 0;
}

// gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache.
int ggml_cuda_gdn_cache_cpy_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool disable_gdn_cpy = [] { const char * e = getenv("GGML_CUDA_DISABLE_GDN_CPY"); return e != nullptr && std::atoi(e); }();
    const ggml_tensor * node = cgraph->nodes[i];
    if (disable_gdn_cpy || node->op != GGML_OP_GATED_DELTA_NET) {
        return 0;
    }
    ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
    const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
    if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
        GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                      __func__, node->name, nodes_to_skip);
#endif
        ggml_cuda_op_gated_delta_net_fused_cache(ctx, cgraph->nodes[i], fused_state_cpy);
        return nodes_to_skip;
    }
    return 0;
}

// GDN lazy commit: a replay GDN A (state update only) followed by the batch's GDN B, whose input state is A's output
// state, runs as one launch.  B's kernel folds A's tokens in first, reading the records through the GET_ROWS indices,
// and writes the state after them where A's copy to the cache would have.  The gathers do not run and their bytes may
// hold B's output.  K == 1, one sequence, at most 16 tokens (the sequential kernel).  GGML_CUDA_GDN_PREFIX=0 turns it off.
struct gdn_prefix_plan {
    const ggml_tensor * b;
    const ggml_tensor * cpy;
    const ggml_tensor * kv;
    const ggml_tensor * gb;
};

// Returns the F32 GET_ROWS (I32 indices) of a cache tensor behind view, when nothing but views and node ia reads it; nullptr otherwise.
static const ggml_tensor * gdn_prefix_gather(const ggml_cgraph * cgraph, int ia, const ggml_tensor * view) {
    const ggml_tensor * gr = view->view_src;
    if (view->op != GGML_OP_VIEW || gr == nullptr || gr->op != GGML_OP_GET_ROWS || gr->type != GGML_TYPE_F32 ||
            gr->src[0]->type != GGML_TYPE_F32 || gr->src[0]->op != GGML_OP_NONE || gr->src[1]->type != GGML_TYPE_I32 ||
            gr->src[0]->ne[2] != 1 || gr->src[1]->ne[1] != 1 || !ggml_is_contiguous(gr->src[1]) || (gr->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return nullptr;
    }
    const ggml_tensor * a = cgraph->nodes[ia];
    for (int k = 0; k < cgraph->n_nodes; ++k) {
        const ggml_tensor * n = cgraph->nodes[k];
        if (n == a || n == gr || ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] != nullptr && ggml_cuda_view_root(n->src[s]) == gr) {
                return nullptr;
            }
        }
    }
    return gr;
}

static bool gdn_prefix_match(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, gdn_prefix_plan & plan) {
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_GDN_PREFIX"); return e == nullptr || atoi(e) != 0; }();
    const ggml_tensor * a = cgraph->nodes[i];
    auto band_ok = [](const ggml_tensor * gdn) {
        return gdn->op == GGML_OP_GATED_DELTA_NET && ggml_get_op_params_i32(gdn, 0) == 1 && gdn->src[2]->ne[0] == 128 &&
               gdn->src[2]->ne[3] == 1 && gdn->src[2]->ne[2] <= 16 && gdn->src[3]->ne[0] == 1;
    };
    if (!enabled || !band_ok(a) || (a->flags & GGML_TENSOR_FLAG_OUTPUT) || a->src[0] != a->src[1]) {
        return false;
    }
    int j = i + 1;
    while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
        ++j;
    }
    if (j >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * b = cgraph->nodes[j];
    if (!band_ok(b) || ggml_cuda_view_root(b->src[5]) != a || b->src[2]->ne[1] != a->src[2]->ne[1] || b->src[1]->ne[1] != a->src[1]->ne[1]) {
        return false;
    }
    const ggml_tensor * kv = gdn_prefix_gather(cgraph, i, a->src[1]);
    const ggml_tensor * gb = gdn_prefix_gather(cgraph, i, a->src[3]);
    const int64_t n = a->src[2]->ne[2];
    if (kv == nullptr || gb == nullptr || kv == gb || gdn_prefix_gather(cgraph, i, a->src[2]) != kv ||
            gdn_prefix_gather(cgraph, i, a->src[4]) != gb || kv->src[1]->ne[0] != n || gb->src[1]->ne[0] != 2*n ||
            a->src[1]->data != kv->data || a->src[1]->nb[2] != kv->nb[1] || a->src[2]->nb[2] != kv->nb[1] ||
            a->src[3]->data != gb->data || a->src[3]->nb[2] != gb->nb[1] || a->src[3]->nb[1] != sizeof(float) ||
            (const char *) a->src[4]->data != (const char *) gb->data + n*gb->nb[1] || a->src[4]->nb[2] != gb->nb[1]) {
        return false;
    }
    // A's input state, unless it is read in place from the cache, is dead after A: B's output may overlay it
    const ggml_tensor * s_root = ggml_cuda_view_root(a->src[5]);
    ggml_cuda_gdn_state_src redirect;
    const bool in_place = ggml_cuda_gdn_get_state_src(a, redirect) ||
        std::find(ctx.gdn_deferred_gathers.begin(), ctx.gdn_deferred_gathers.end(), s_root) != ctx.gdn_deferred_gathers.end();
    if (!in_place) {
        const char * i0 = (const char *) a->src[5]->data;
        const char * b0 = (const char *) b->data;
        if (i0 < b0 + ggml_nbytes(b) && b0 < i0 + ggml_nbytes(a->src[5])) {
            return false;
        }
    }
    const int64_t S_v = a->src[2]->ne[0], H = a->src[2]->ne[1];
    const char *  state = (const char *) a->data + S_v*H*n*sizeof(float);
    const size_t  D     = (size_t) S_v*S_v*H*sizeof(float);
    const ggml_tensor * cpy = nullptr;
    for (int k = j + 1; k < cgraph->n_nodes; ++k) {
        const ggml_tensor * nd = cgraph->nodes[k];
        if (ggml_cuda_is_view_or_noop(nd)) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (nd->src[s] == nullptr || ggml_cuda_view_root(nd->src[s]) != a) {
                continue;
            }
            if (cpy != nullptr || nd->op != GGML_OP_CPY || s != 0 || nd->src[0]->data != state || ggml_nbytes(nd->src[0]) != D ||
                    !ggml_is_contiguous(nd->src[0]) || nd->type != GGML_TYPE_F32 || !ggml_is_contiguous(nd) ||
                    g_precomputed_nodes.count(nd) != 0) {
                return false;
            }
            cpy = nd;
        }
    }
    plan = { b, cpy, kv, gb };
    return true;
}

int ggml_cuda_gdn_prefix_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * node = cgraph->nodes[i];
    gdn_prefix_plan plan;
    if (node->op == GGML_OP_GET_ROWS) {
        // a record gather a folded replay will not read
        for (int k = i + 1; k < std::min(cgraph->n_nodes, i + 16); ++k) {
            if (cgraph->nodes[k]->op == GGML_OP_GATED_DELTA_NET) {
                return gdn_prefix_match(ctx, cgraph, k, plan) && (plan.kv == node || plan.gb == node) ? -1 : 0;
            }
        }
        return 0;
    }
    if (node->op != GGML_OP_GATED_DELTA_NET || !gdn_prefix_match(ctx, cgraph, i, plan)) {
        return 0;
    }
    gdn_prefix_srcs[plan.b] = { node, plan.cpy != nullptr ? (float *) plan.cpy->data : nullptr };
    if (plan.cpy != nullptr) {
        g_precomputed_nodes.insert(plan.cpy);
    }
    return -1;
}
