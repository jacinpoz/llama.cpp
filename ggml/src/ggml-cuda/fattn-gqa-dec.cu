// Decode/verify attention for head dim 256 with GQA 6 or 8 (n_q <= 8), aimed at RDNA3 where the tile kernel
// re-reads every K/V tile once per pair of query heads and is issue-bound on quantized K/V.
// Opt-in with GGML_HIP_FA_GQA_DEC=1.

#include "fattn-gqa-dec.cuh"
#include "mk-ops-attn.cuh"

#include <cmath>

namespace {

constexpr int kMaxQ = 4;  // query tokens per pass; larger widths loop, so the partial buffer has a fixed size

template <int G, ggml_type T>
__launch_bounds__(WARP_SIZE, 1) __global__ void gqa_dec_partial(const mk_attn_partial_params p) {
    __shared__ __attribute__((aligned(16))) char lds[mk_attn_partial_lds_bytes(G)];
    const int tile = (blockIdx.z*gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x;
    mk_attn_partial<WARP_SIZE>::run(p, mk_attn_partial_variant(G, T), tile, true, lds);
}

template <int G>
__launch_bounds__(mk_attn_combine_waves*WARP_SIZE, 1) __global__ void gqa_dec_combine(const mk_attn_combine_params p) {
    using op = mk_attn_combine<mk_attn_combine_waves*WARP_SIZE>;
    __shared__ __attribute__((aligned(16))) char lds[op::lds_bytes];
    op::run(p, mk_attn_combine_variant(G), blockIdx.y*gridDim.x + blockIdx.x, true, lds);
}

template <int G, ggml_type T>
void gqa_dec_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const gqa_dec_epilogue & ep) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    cudaStream_t stream = ctx.stream();

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_q       = (int) Q->ne[1];
    const int n_head    = (int) Q->ne[2];
    const int n_head_kv = (int) K->ne[2];
    const int n_kv      = (int) K->ne[1];
    const int n_chunks  = mk_attn_chunking(n_kv).n;

    // Fixed size (independent of n_kv and n_q), so the pool returns the same buffer every call.
    ggml_cuda_pool_alloc<float>  part_acc(ctx.pool(), (size_t) kMaxQ*n_head_kv*mk_attn_max_chunks*G*mk_attn_d);
    ggml_cuda_pool_alloc<float2> part_ms (ctx.pool(), (size_t) kMaxQ*n_head_kv*mk_attn_max_chunks*G);

    mk_attn_partial_params pp{};
    pp.Q           = (const char *) Q->data;
    pp.K           = (const char *) K->data;
    pp.V           = (const char *) V->data;
    pp.mask        = (const char *) mask->data;
    pp.part_acc    = part_acc.ptr;
    pp.part_ms     = part_ms.ptr;
    pp.scale       = scale;
    pp.n_kv        = n_kv;
    pp.n_head_kv   = n_head_kv;
    pp.tile_chunks = n_chunks;
    pp.nbq1 = Q->nb[1]; pp.nbq2 = Q->nb[2];
    pp.nbk1 = K->nb[1]; pp.nbk2 = K->nb[2];
    pp.nbv1 = V->nb[1]; pp.nbv2 = V->nb[2];
    pp.nbm1 = mask->nb[1];

    mk_attn_combine_params cp{};
    cp.part_acc  = part_acc.ptr;
    cp.part_ms   = part_ms.ptr;
    cp.dst       = (float *) dst->data;
    cp.n_kv      = n_kv;
    cp.n_head_kv = n_head_kv;
    cp.n_head    = n_head;
    cp.ep        = ep;

    for (int q0 = 0; q0 < n_q; q0 += kMaxQ) {
        const int nq = std::min(kMaxQ, n_q - q0);
        pp.q0 = cp.q0 = q0;
        pp.nq = cp.nq = nq;
        gqa_dec_partial<G, T><<<dim3(n_chunks, n_head_kv, nq), WARP_SIZE, 0, stream>>>(pp);
        CUDA_CHECK(cudaGetLastError());
        gqa_dec_combine<G><<<dim3(n_head, nq), mk_attn_combine_waves*WARP_SIZE, 0, stream>>>(cp);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <int G>
void gqa_dec_launch_type(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const gqa_dec_epilogue & ep) {
    switch (dst->src[1]->type) {
        case GGML_TYPE_F16:  gqa_dec_launch<G, GGML_TYPE_F16> (ctx, dst, ep); break;
        case GGML_TYPE_Q8_0: gqa_dec_launch<G, GGML_TYPE_Q8_0>(ctx, dst, ep); break;
        case GGML_TYPE_Q5_0: gqa_dec_launch<G, GGML_TYPE_Q5_0>(ctx, dst, ep); break;
        default: GGML_ABORT("gqa_dec: unsupported K/V type");
    }
}

} // namespace

bool ggml_cuda_fattn_gqa_dec_supported(const int device, const ggml_tensor * dst) {
    static const bool enabled = getenv("GGML_HIP_FA_GQA_DEC") && atoi(getenv("GGML_HIP_FA_GQA_DEC")) != 0;
    if (!enabled) {
        return false;
    }
#ifndef GGML_USE_HIP
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#else
    if (!GGML_CUDA_CC_IS_RDNA3(ggml_cuda_info().devices[device].cc)) {
        return false;
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));

    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) return false;
    if (Q->ne[0] != mk_attn_d || K->ne[0] != mk_attn_d || V->ne[0] != mk_attn_d) return false;
    if (Q->ne[1] > 8 || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) return false;
    if (K->type != V->type || (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0 && K->type != GGML_TYPE_Q5_0)) return false;
    if (K->ne[2] == 0 || V->ne[2] != K->ne[2] || (Q->ne[2] != 6*K->ne[2] && Q->ne[2] != 8*K->ne[2])) return false;
    if (max_bias != 0.0f || softcap != 0.0f || dst->src[4] != nullptr || dst->src[5] != nullptr) return false;
    if (mask == nullptr || mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] || mask->ne[2] != 1 || mask->ne[3] != 1) return false;
    return true;
#endif // GGML_USE_HIP
}

void ggml_cuda_flash_attn_ext_gqa_dec(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const gqa_dec_epilogue & ep) {
    const int gqa = (int) (dst->src[0]->ne[2]/dst->src[1]->ne[2]);
    if (gqa == 6) {
        gqa_dec_launch_type<6>(ctx, dst, ep);
    } else {
        gqa_dec_launch_type<8>(ctx, dst, ep);
    }
}
