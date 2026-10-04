// Decode/verify attention for head dim 256 with GQA 6 or 8 (n_q <= 8), aimed at RDNA3 where the tile kernel
// re-reads every K/V tile once per pair of query heads and is issue-bound on quantized K/V.
//
// One wave per (KV chunk, KV head, query token), 32 keys per step. For the scores each lane takes one key and
// computes all G heads against Q in LDS (no cross-lane reductions); for V each lane owns 8 of the 256 dims of
// every key. Every K/V row is loaded and decoded once for the whole group. Softmax is online per 32-key step.
// The chunk length depends only on n_kv, and each token is processed independently, so a token gets the same
// result whether it is decoded alone or verified in a speculative batch.
// Opt-in with GGML_HIP_FA_GQA_DEC=1.

#include "fattn-gqa-dec.cuh"

#include <cmath>

namespace {

constexpr int kD         = 256;
constexpr int kMaxChunks = 512;
constexpr int kMaxQ      = 4;  // query tokens per pass; larger widths loop, so the partial buffer has a fixed size

template <ggml_type T>
__device__ __forceinline__ void gqa_dec_load_slice(const char * row, const int lane, float v[8]) {
    const int d8 = lane*8;
    if constexpr (T == GGML_TYPE_F16) {
        const half2 * p = reinterpret_cast<const half2 *>(row) + d8/2;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(p[i]);
            v[2*i + 0] = f.x;
            v[2*i + 1] = f.y;
        }
    } else if constexpr (T == GGML_TYPE_Q8_0) {
        const block_q8_0 * blk = reinterpret_cast<const block_q8_0 *>(row) + d8/QK8_0;
        const float d = __half2float(blk->d);
        const int   o = d8 % QK8_0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = blk->qs[o + i]*d;
        }
    } else {
        static_assert(T == GGML_TYPE_Q5_0, "gqa_dec: unsupported K/V type");
        const block_q5_0 * blk = reinterpret_cast<const block_q5_0 *>(row) + d8/QK5_0;
        const float d = __half2float(blk->d);
        uint32_t qh;
        memcpy(&qh, blk->qh, sizeof(qh));
        const int o  = d8 % QK5_0;
        const int lo = o < QK5_0/2;
        uint64_t qs;
        memcpy(&qs, blk->qs + (o % (QK5_0/2)), sizeof(qs));
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int q   = (int) ((qs >> (8*i)) & 0xFF);
            const int nib = lo ? (q & 0xF) : (q >> 4);
            v[i] = ((nib | (int) (((qh >> (o + i)) & 1) << 4)) - 16)*d;
        }
    }
}

// 16 values of one K row, dims 16*hb..16*hb+15, as float.
template <ggml_type T>
__device__ __forceinline__ void gqa_dec_load_half(const char * row, const int hb, float v[16]) {
    if constexpr (T == GGML_TYPE_F16) {
        const half2 * p = reinterpret_cast<const half2 *>(row) + hb*8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float2 f = __half22float2(p[i]);
            v[2*i + 0] = f.x;
            v[2*i + 1] = f.y;
        }
    } else if constexpr (T == GGML_TYPE_Q8_0) {
        const block_q8_0 * blk = reinterpret_cast<const block_q8_0 *>(row) + hb/2;
        const float d = __half2float(blk->d);
        const int   o = (hb % 2)*16;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            v[i] = blk->qs[o + i]*d;
        }
    } else {
        const block_q5_0 * blk = reinterpret_cast<const block_q5_0 *>(row) + hb/2;
        const float d = __half2float(blk->d);
        uint32_t qh;
        memcpy(&qh, blk->qh, sizeof(qh));
        const bool hi = hb % 2;
        qh >>= hi ? 16 : 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int q   = blk->qs[i];
            const int nib = hi ? (q >> 4) : (q & 0xF);
            v[i] = ((nib | (int) (((qh >> i) & 1) << 4)) - 16)*d;
        }
    }
}

template <int G, ggml_type T>
__launch_bounds__(WARP_SIZE, 1) __global__ void gqa_dec_partial(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V, const char * __restrict__ mask,
        float * __restrict__ part_acc, float2 * __restrict__ part_ms,
        const float scale, const int n_kv, const int chunk, const int n_chunks, const int q0,
        const int64_t nbq1, const int64_t nbq2, const int64_t nbk1, const int64_t nbk2,
        const int64_t nbv1, const int64_t nbv2, const int64_t nbm1) {
    const int c    = blockIdx.x;
    const int h    = blockIdx.y;
    const int t    = blockIdx.z;
    const int lane = threadIdx.x;
    const int tok  = q0 + t;

    __shared__ __attribute__((aligned(16))) float q_lds[G][kD];
    __shared__ float p_lds[WARP_SIZE][G];

#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float * qp = reinterpret_cast<const float *>(Q + tok*nbq1 + (int64_t) (h*G + g)*nbq2);
#pragma unroll
        for (int i = lane; i < kD; i += WARP_SIZE) {
            q_lds[g][i] = qp[i]*scale;
        }
    }
    __syncthreads();

    float m[G], s[G], acc[G][8];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = -INFINITY;
        s[g] = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[g][i] = 0.0f;
        }
    }

    const half * mrow  = reinterpret_cast<const half *>(mask + tok*nbm1);
    const char * Kh    = K + h*nbk2;
    const char * Vh    = V + h*nbv2;
    const int    k_beg = c*chunk;
    const int    k_end = min(n_kv, k_beg + chunk);

    for (int k0 = k_beg; k0 < k_end; k0 += WARP_SIZE) {
        const int nk = min(WARP_SIZE, k_end - k0);

        // scores: lane j computes all G scores of key k0 + j
        float my_s[G];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            my_s[g] = 0.0f;
        }
        const bool live = lane < nk;
        const char * krow = Kh + (int64_t) (live ? k0 + lane : k0)*nbk1;
        for (int hb = 0; hb < kD/16; ++hb) {
            float kv[16];
            gqa_dec_load_half<T>(krow, hb, kv);
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float4 * qv = reinterpret_cast<const float4 *>(&q_lds[g][hb*16]);
                float a = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 qq = qv[i];
                    a += qq.x*kv[4*i + 0] + qq.y*kv[4*i + 1] + qq.z*kv[4*i + 2] + qq.w*kv[4*i + 3];
                }
                my_s[g] += a;
            }
        }
        const float mk = live ? __half2float(mrow[k0 + lane]) : -INFINITY;
#pragma unroll
        for (int g = 0; g < G; ++g) {
            my_s[g] = live ? my_s[g] + mk : -INFINITY;
        }

#pragma unroll
        for (int g = 0; g < G; ++g) {
            float gm = my_s[g];
#pragma unroll
            for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
                gm = fmaxf(gm, __shfl_xor(gm, off, WARP_SIZE));
            }
            const float mn = fmaxf(m[g], gm);
            const float sc = mn == -INFINITY ? 1.0f : expf(m[g] - mn);
            m[g]  = mn;
            s[g] *= sc;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[g][i] *= sc;
            }
            p_lds[lane][g] = my_s[g] == -INFINITY ? 0.0f : expf(my_s[g] - mn);
        }
        __syncthreads();

        // V: lane owns dims 8*lane..8*lane+7 of every key
        for (int j = 0; j < nk; ++j) {
            float vv[8];
            gqa_dec_load_slice<T>(Vh + (int64_t) (k0 + j)*nbv1, lane, vv);
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float pj = p_lds[j][g];
                s[g] += pj;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[g][i] += pj*vv[i];
                }
            }
        }
        __syncthreads();
    }

    const int64_t base = ((int64_t) (t*gridDim.y + h)*n_chunks + c)*G;
#pragma unroll
    for (int g = 0; g < G; ++g) {
        float * pa = part_acc + (base + g)*kD + lane*8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            pa[i] = acc[g][i];
        }
        if (lane == 0) {
            part_ms[base + g] = make_float2(m[g], s[g]);
        }
    }
}

// One block per (query head, token). Wave w merges chunks w, w + kCombineWaves, ... over all 256 dims (8 per
// lane); the waves are then merged in LDS in a fixed order.
constexpr int kCombineWaves = 8;

template <int G>
__launch_bounds__(kCombineWaves*WARP_SIZE, 1) __global__ void gqa_dec_combine(
        const float * __restrict__ part_acc, const float2 * __restrict__ part_ms, float * __restrict__ dst,
        const int n_chunks, const int n_head_kv, const int n_head, const int q0, const gqa_dec_epilogue ep) {
    const int hq   = blockIdx.x;
    const int t    = blockIdx.y;
    const int h    = hq / G;
    const int g    = hq % G;
    const int lane = threadIdx.x % WARP_SIZE;
    const int w    = threadIdx.x / WARP_SIZE;

    __shared__ float w_m[kCombineWaves];
    __shared__ float w_s[kCombineWaves];
    __shared__ float w_acc[kCombineWaves][kD];

    const int64_t base = (int64_t) (t*n_head_kv + h)*n_chunks*G + g;
    float M = -INFINITY;
    for (int c = w; c < n_chunks; c += kCombineWaves) {
        M = fmaxf(M, part_ms[base + (int64_t) c*G].x);
    }
    float S = 0.0f;
    float A[8] = {};
    for (int c = w; c < n_chunks; c += kCombineWaves) {
        const float2 ms = part_ms[base + (int64_t) c*G];
        if (ms.x == -INFINITY) {
            continue;
        }
        const float   wgt = expf(ms.x - M);
        const float * pa  = part_acc + (base + (int64_t) c*G)*kD + lane*8;
        S += wgt*ms.y;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            A[i] += wgt*pa[i];
        }
    }
    if (lane == 0) {
        w_m[w] = M;
        w_s[w] = S;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        w_acc[w][lane*8 + i] = A[i];
    }
    __syncthreads();

    if (w == 0) {
        float MM = -INFINITY;
#pragma unroll
        for (int k = 0; k < kCombineWaves; ++k) {
            MM = fmaxf(MM, w_m[k]);
        }
        float SS = 0.0f;
        float AA[8] = {};
#pragma unroll
        for (int k = 0; k < kCombineWaves; ++k) {
            if (w_m[k] == -INFINITY) {
                continue;
            }
            const float wgt = expf(w_m[k] - MM);
            SS += wgt*w_s[k];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                AA[i] += wgt*w_acc[k][lane*8 + i];
            }
        }
        float * out = dst + ((int64_t) (q0 + t)*n_head + hq)*kD + lane*8;
        float o[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            o[i] = AA[i]/SS;
        }
        if (ep.gate != nullptr) {
            // inverse V rotation: fwht over each 64-dim chunk (8 lanes x 8 dims), stages in the order of the chunk
            // index bits 0..5 as fwht_cuda<64> runs them, so the result is bit-identical; then x sigmoid(gate)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                o[i] *= 0.125f;
            }
#pragma unroll
            for (int hh = 1; hh < 8; hh *= 2) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if ((i & hh) == 0) {
                        const float x = o[i];
                        const float y = o[i + hh];
                        o[i]      = x + y;
                        o[i + hh] = x - y;
                    }
                }
            }
#pragma unroll
            for (int hh = 1; hh < 8; hh *= 2) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float val2 = __shfl_xor(o[i], hh, WARP_SIZE);
                    o[i] = (lane & hh) == 0 ? o[i] + val2 : val2 - o[i];
                }
            }
            const float * gp = reinterpret_cast<const float *>(ep.gate + (int64_t) (q0 + t)*ep.gate_nb2 + (int64_t) hq*ep.gate_nb1) + lane*8;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                o[i] = o[i] * (1.0f / (1.0f + expf(-gp[i])));
            }
            out = ep.out + ((int64_t) (q0 + t)*n_head + hq)*kD + lane*8;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out[i] = o[i];
        }
    }
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
    const int chunk     = std::max(64, GGML_PAD((n_kv + kMaxChunks - 1)/kMaxChunks, WARP_SIZE));
    const int n_chunks  = (n_kv + chunk - 1)/chunk;

    // Fixed size (independent of n_kv and n_q), so the pool returns the same buffer every call.
    ggml_cuda_pool_alloc<float>  part_acc(ctx.pool(), (size_t) kMaxQ*n_head_kv*kMaxChunks*G*kD);
    ggml_cuda_pool_alloc<float2> part_ms (ctx.pool(), (size_t) kMaxQ*n_head_kv*kMaxChunks*G);

    for (int q0 = 0; q0 < n_q; q0 += kMaxQ) {
        const int nq = std::min(kMaxQ, n_q - q0);
        gqa_dec_partial<G, T><<<dim3(n_chunks, n_head_kv, nq), WARP_SIZE, 0, stream>>>(
            (const char *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
            part_acc.ptr, part_ms.ptr, scale, n_kv, chunk, n_chunks, q0,
            Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], mask->nb[1]);
        CUDA_CHECK(cudaGetLastError());
        gqa_dec_combine<G><<<dim3(n_head, nq), kCombineWaves*WARP_SIZE, 0, stream>>>(
            part_acc.ptr, part_ms.ptr, (float *) dst->data, n_chunks, n_head_kv, n_head, q0, ep);
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
    if (Q->ne[0] != kD || K->ne[0] != kD || V->ne[0] != kD) return false;
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
