// R25 (local patch, 2026-10-03): dense causal prefill attention for head dim 256 with GQA 8 on the RDNA3
// WMMA cores, ported from gufo (MIT, github.com/NinjaPear/gufo-Qwen3.6-35B-A3B-Q6dense,
// src/models/qwen36_35b_a3b/kernels/rocm/kernels.hip.cpp, DenseCausalAttentionKernel).
//
// Opt-in with GGML_CUDA_FA_GUFO_256=1. Only taken for prefill-shaped batches (n_q >= GGML_CUDA_FA_GUFO_MIN_Q,
// default 32) with D = 256, n_head/n_head_kv = 8 or 6, F16, Q8_0 or Q5_0 K/V (quantized K/V is converted per KV head),
// no ALiBi/softcap/sinks, and either an explicit F16 mask or the derived (cell_pos/tok_lo/tok_hi) mask.
//
// Differences from the gufo original: visibility comes from the ggml mask (explicit or derived) instead of
// absolute positions, so any KV cell layout and multi-sequence streams work; a prepass computes, per
// 16-query tile, the range of KV cells visible to any of its rows, and the key loop only covers that range;
// Q/K/V/dst use ggml strides; the softmax scale is the op's scale; the output gate is not fused.
// The gufo per-row diagonal path (bit-identical results across prefill chunkings) is not needed: masked
// keys get P = 0 exactly and staged K/V beyond n_kv are zero.

#include "common.cuh"
#include "convert.cuh"
#include "fattn-gufo.cuh"

#include <climits>

#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#define GUFO_WMMA_DEVICE 1
#endif

namespace {

constexpr int kD       = 256;
constexpr int kThreads = 256;

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f  = __attribute__((__vector_size__(8 * sizeof(float)))) float;

#ifdef GUFO_WMMA_DEVICE
__device__ __forceinline__ v8f gufo_wmma(v16h a, v16h b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h gufo_load_frag(const half * p) {
    union { v16h f; uint4 u[2]; } cvt;
    cvt.u[0] = *reinterpret_cast<const uint4 *>(p);
    cvt.u[1] = *reinterpret_cast<const uint4 *>(p + 8);
    return cvt.f;
}
#endif

struct gufo_mask_args {
    const half * mask;      // explicit mask [n_kv, >= n_q, 1, ne33] or nullptr
    int64_t      s31;       // mask row stride (halves)
    int64_t      s33;       // mask stream stride (halves)
    int          ne33;      // mask streams (broadcast when 1)
    const int  * cell_pos;  // derived mask (single stream), or nullptr
    const int  * tok_lo;
    const int  * tok_hi;
};

__device__ __forceinline__ float gufo_mask_value(const gufo_mask_args & m, int seq, int query, int key) {
    if (m.mask) {
        return __half2float(m.mask[(int64_t)(seq % m.ne33)*m.s33 + (int64_t)query*m.s31 + key]);
    }
    const int p = m.cell_pos[key];
    return (p != INT_MIN && m.tok_lo[query] <= p && p <= m.tok_hi[query]) ? 0.0f : -INFINITY;
}

// Per (stream, 16-query tile): [first, last+1) KV cell visible to any live row ({0, 0} when none), plus a bitmap of
// the kKeys-key rounds that contain at least one visible cell. With a unified KV cache the other slots' cells can sit
// inside [first, last+1) (e.g. after a slot restore); the kernel skips their rounds instead of streaming them.
template <int kKeys>
__global__ void gufo_kv_bounds(const gufo_mask_args m, int2 * bounds, uint32_t * rounds, const int n_words,
                               const int n_q, const int n_kv, const int n_tiles) {
    const int tile = blockIdx.x;
    const int seq  = blockIdx.y;
    const int q0   = tile*16;
    const int nr   = min(16, n_q - q0);

    extern __shared__ uint32_t s_bits[];
    for (int w = threadIdx.x; w < n_words; w += blockDim.x) {
        s_bits[w] = 0;
    }
    __syncthreads();

    int lo = INT_MAX;
    int hi = -1;
    const int lane = threadIdx.x % 32;
    for (int base = 0; base < n_kv; base += blockDim.x) {
        const int key = base + threadIdx.x;   // a wave covers 32 consecutive keys (warp-uniform loop)
        bool any = false;
        if (key < n_kv) {
            for (int r = 0; r < nr && !any; ++r) {
                any = gufo_mask_value(m, seq, q0 + r, key) != -INFINITY;
            }
        }
        if (any) {
            lo = min(lo, key);
            hi = max(hi, key);
        }
        const uint32_t vis = (uint32_t) __ballot(any);
        if (lane == 0 && vis) {
            const int key_w = key;   // first key of this wave (lane 0)
            for (int g = 0; g < 32; g += kKeys) {
                const uint32_t grp = kKeys == 32 ? vis : (vis >> g) & 0xFFFFu;
                if (grp) {
                    const int rd = (key_w + g) / kKeys;
                    atomicOr(&s_bits[rd >> 5], 1u << (rd & 31));
                }
            }
        }
    }
    __shared__ int s_lo[kThreads];
    __shared__ int s_hi[kThreads];
    s_lo[threadIdx.x] = lo;
    s_hi[threadIdx.x] = hi;
    __syncthreads();
    for (int off = blockDim.x/2; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off) {
            s_lo[threadIdx.x] = min(s_lo[threadIdx.x], s_lo[threadIdx.x + off]);
            s_hi[threadIdx.x] = max(s_hi[threadIdx.x], s_hi[threadIdx.x + off]);
        }
        __syncthreads();
    }
    uint32_t * out = rounds + (int64_t)(seq*n_tiles + tile)*n_words;
    for (int w = threadIdx.x; w < n_words; w += blockDim.x) {
        out[w] = s_bits[w];
    }
    if (threadIdx.x == 0) {
        bounds[seq*n_tiles + tile] = s_hi[0] < 0 ? make_int2(0, 0) : make_int2(s_lo[0], s_hi[0] + 1);
    }
}

// next round >= kt with a visible cell, or kt_end
__device__ __forceinline__ int gufo_next_round(const uint32_t * __restrict__ bits, int kt, const int kt_end) {
    while (kt < kt_end) {
        const uint32_t w = bits[kt >> 5] >> (kt & 31);
        if (w) {
            return min(kt + __ffs(w) - 1, kt_end);
        }
        kt = (kt | 31) + 1;
    }
    return kt_end;
}

// 8 consecutive K/V values of one row starting at d8, as F16. Quantized rows are decoded here, so no F16 copy of
// the cache is needed (a staged copy grows with n_kv and fragments the HIP buffer pool).
template <ggml_type kKvType>
__device__ __forceinline__ uint4 gufo_load8(const char * row, const int d8) {
    if constexpr (kKvType == GGML_TYPE_F16) {
        return *reinterpret_cast<const uint4 *>(row + (int64_t) d8*sizeof(half));
    } else {
        union { uint4 u; half h[8]; } r;
        if constexpr (kKvType == GGML_TYPE_Q8_0) {
            const block_q8_0 * blk = reinterpret_cast<const block_q8_0 *>(row) + d8/QK8_0;
            const float d = __half2float(blk->d);
            const int   o = d8 % QK8_0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                r.h[i] = __float2half(blk->qs[o + i]*d);
            }
        } else {
            static_assert(kKvType == GGML_TYPE_Q5_0, "gufo: unsupported K/V type");
            const block_q5_0 * blk = reinterpret_cast<const block_q5_0 *>(row) + d8/QK5_0;
            const float d = __half2float(blk->d);
            uint32_t qh;
            memcpy(&qh, blk->qh, sizeof(qh));
            const int o = d8 % QK5_0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int j   = o + i;
                const int q   = blk->qs[j % (QK5_0/2)];
                const int nib = j < QK5_0/2 ? (q & 0xF) : (q >> 4);
                r.h[i] = __float2half(((nib | (int) (((qh >> j) & 1) << 4)) - 16)*d);
            }
        }
        return r.u;
    }
}

// One block: 16 queries x kHeads query heads (one 16-row block each) of one KV head, one stream.
// GQA 8 runs 4 heads per block; GQA 6 (Qwen3.5/3.8 dense 27B) runs 2.
template <int kKeys, int kGqa, int kHeads, ggml_type kKvType>
__launch_bounds__(kThreads, 2) __global__ void gufo_attn_d256(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        float * __restrict__ dst, const gufo_mask_args m, const int2 * __restrict__ bounds,
        const uint32_t * __restrict__ rounds, const int n_words, const float scale, const int n_q, const int n_kv, const int n_head, const int kv_head0,
        const int64_t nb01, const int64_t nb02, const int64_t nb03,   // Q strides (bytes)
        const int64_t sk1, const int64_t sk2, const int64_t sk3,      // K strides (bytes)
        const int64_t sv1, const int64_t sv2, const int64_t sv3) {    // V strides (bytes)
#ifdef GUFO_WMMA_DEVICE
    constexpr int kRowBlocks     = kHeads;
    constexpr int kRows          = 16*kRowBlocks;
    constexpr int kKeyBlocks     = kKeys/16;
    constexpr int kSTiles        = kRowBlocks*kKeyBlocks;
    constexpr int kSplit         = 8/kSTiles;              // waves per S tile
    constexpr int kKStepsPerWave = (kD/16)/kSplit;
    constexpr int kKStride       = kD + 8;
    constexpr int kVtStride      = kKeys + 8;
    constexpr int kLanes         = kThreads/kRows;         // softmax lanes per row
    constexpr int kPerLane       = kKeys/kLanes;
    constexpr int kKRegs         = kKeys*(kD/8)/kThreads;
    constexpr int kVRegs         = kKeys*kD/(kThreads*8);
    static_assert(kKeys == 16 || kKeys == 32);
    static_assert(kLanes*kRows == kThreads && kKeys % kLanes == 0);
    static_assert(kSplit*kSTiles == 8 && kKStepsPerWave*kSplit == kD/16);

    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int wave    = tid >> 5;
    const int sub     = lane & 15;
    const int half_id = lane >> 4;

    const int tile        = blockIdx.x;
    const int query_start = tile*16;
    const int kv_head     = kv_head0 + blockIdx.y / (kGqa/kRowBlocks);
    const int first_head  = kv_head*kGqa + (blockIdx.y % (kGqa/kRowBlocks))*kRowBlocks;
    const int seq         = blockIdx.z;
    const int n_tiles_q   = gridDim.x;

    constexpr int kKvHalves = kKeys*kKStride > kD*kVtStride ? kKeys*kKStride : kD*kVtStride;
    __shared__ __attribute__((aligned(16))) half kv_lds[kKvHalves];
    __shared__ float s_lds[kSplit][kSTiles][16][17];
    __shared__ __attribute__((aligned(16))) half p_lds[kRows][kKeys + 8];
    __shared__ float row_sum[kRows];
    __shared__ float row_scale[kRows];

    const int s_tile = wave % kSTiles;
    const int s_kh   = wave / kSTiles;
    const int s_rb   = s_tile % kRowBlocks;
    const int s_kb   = s_tile / kRowBlocks;

    v16h q_frag[kKStepsPerWave];
    {
        const int  query = query_start + sub;
        const bool live  = query < n_q;
        const float * q_row = (const float *) (Q + seq*nb03 + (int64_t)(live ? query : 0)*nb01 + (int64_t)(first_head + s_rb)*nb02);
#pragma unroll
        for (int ks = 0; ks < kKStepsPerWave; ++ks) {
            const int d0 = (s_kh*kKStepsPerWave + ks)*16;
#pragma unroll
            for (int v = 0; v < 16; ++v) {
                q_frag[ks][v] = (_Float16) (live ? q_row[d0 + v]*scale : 0.0f);
            }
        }
    }

    v8f   o_acc[kRowBlocks][2] = {};
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    const int2 b       = bounds[seq*n_tiles_q + tile];
    const int  kt_end  = (b.y + kKeys - 1) / kKeys;
    const uint32_t * rbits = rounds + (int64_t)(seq*n_tiles_q + tile)*n_words;
    const int  kt_beg  = gufo_next_round(rbits, b.x / kKeys, kt_end);

    const int v_key   = lane % kKeys;
    const int v_slice = (tid / kKeys)*(kVRegs*8);
    const char * k_base = K + seq*sk3 + (int64_t)kv_head*sk2;
    const char * v_base = V + seq*sv3 + (int64_t)kv_head*sv2;

    const auto load_k = [&](int key0, uint4 * dst_r) {
#pragma unroll
        for (int n = 0; n < kKRegs; ++n) {
            const int idx = tid + n*kThreads;
            const int key = key0 + idx/(kD/8);
            const int d8  = (idx % (kD/8))*8;
            dst_r[n] = key < n_kv ? gufo_load8<kKvType>(k_base + (int64_t)key*sk1, d8) : make_uint4(0u, 0u, 0u, 0u);
        }
    };
    const auto load_v = [&](int key0, uint4 * dst_r) {
        const int  key  = key0 + v_key;
        const bool live = key < n_kv;
        const char * src = v_base + (int64_t)(live ? key : 0)*sv1;
#pragma unroll
        for (int j = 0; j < kVRegs; ++j) {
            dst_r[j] = live ? gufo_load8<kKvType>(src, v_slice + j*8) : make_uint4(0u, 0u, 0u, 0u);
        }
    };

    uint4 k_cur[kKRegs], v_cur[kVRegs], k_pre[kKRegs], v_pre[kVRegs];
    if (kt_beg < kt_end) {
        load_k(kt_beg*kKeys, k_cur);
        load_v(kt_beg*kKeys, v_cur);
    }
    for (int kt = kt_beg; kt < kt_end; ) {
        const int key0    = kt*kKeys;
        const int kt_next = gufo_next_round(rbits, kt + 1, kt_end);
        __syncthreads();
#pragma unroll
        for (int n = 0; n < kKRegs; ++n) {
            const int idx = tid + n*kThreads;
            *reinterpret_cast<uint4 *>(&kv_lds[(idx/(kD/8))*kKStride + (idx % (kD/8))*8]) = k_cur[n];
        }
        __syncthreads();
        if (kt_next < kt_end) {
            load_k(kt_next*kKeys, k_pre);
            load_v(kt_next*kKeys, v_pre);
        }
        // S = Q K^T
        {
            v8f s_acc = {};
#pragma unroll
            for (int ks = 0; ks < kKStepsPerWave; ++ks) {
                const int d0 = (s_kh*kKStepsPerWave + ks)*16;
                s_acc = gufo_wmma(q_frag[ks], gufo_load_frag(&kv_lds[(s_kb*16 + sub)*kKStride + d0]), s_acc);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                s_lds[s_kh][s_tile][2*i + half_id][sub] = s_acc[i];
            }
        }
        __syncthreads();
        // V transposed into the K region (one key per lane: conflict-free)
#pragma unroll
        for (int j = 0; j < kVRegs; ++j) {
            const half * packed = reinterpret_cast<const half *>(&v_cur[j]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                kv_lds[(v_slice + j*8 + i)*kVtStride + v_key] = packed[i];
            }
        }
        // online softmax, kLanes threads per row
        {
            const int  rg       = tid / kLanes;
            const int  seg      = tid % kLanes;
            const int  rb       = rg / 16;
            const int  row      = rg % 16;
            const int  query    = query_start + row;
            const bool live_row = query < n_q;
            float vals[kPerLane];
            float part_max = -INFINITY;
#pragma unroll
            for (int mm = 0; mm < kPerLane; ++mm) {
                const int col = seg*kPerLane + mm;
                const int key = key0 + col;
                const int t   = (col/16)*kRowBlocks + rb;
                float sum = 0.0f;
#pragma unroll
                for (int h = 0; h < kSplit; ++h) {
                    sum += s_lds[h][t][row][col % 16];
                }
                const float mv = (live_row && key < n_kv) ? gufo_mask_value(m, seq, query, key) : -INFINITY;
                vals[mm] = sum + mv;
                part_max = fmaxf(part_max, vals[mm]);
            }
#pragma unroll
            for (int off = 1; off < kLanes; off <<= 1) {
                part_max = fmaxf(part_max, __shfl_xor(part_max, off, 32));
            }
            const float prev_max    = running_max;
            const float next_max    = fmaxf(prev_max, part_max);
            const float prior_scale = isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0f;
            float part_sum = 0.0f;
#pragma unroll
            for (int mm = 0; mm < kPerLane; ++mm) {
                const float w = isfinite(vals[mm]) ? __expf(vals[mm] - next_max) : 0.0f;
                part_sum += w;
                p_lds[rg][seg*kPerLane + mm] = __float2half(w);
            }
#pragma unroll
            for (int off = 1; off < kLanes; off <<= 1) {
                part_sum += __shfl_xor(part_sum, off, 32);
            }
            running_max = next_max;
            running_sum = running_sum*prior_scale + part_sum;
            if (seg == 0) {
                row_scale[rg] = prior_scale;
            }
        }
        __syncthreads();
#pragma unroll
        for (int rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float s = row_scale[rb*16 + 2*i + half_id];
                o_acc[rb][0][i] *= s;
                o_acc[rb][1][i] *= s;
            }
        }
        // O += P V
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int dim_tile = wave + t*8;
            v16h v_frag[kKeyBlocks];
#pragma unroll
            for (int kb = 0; kb < kKeyBlocks; ++kb) {
                v_frag[kb] = gufo_load_frag(&kv_lds[(dim_tile*16 + sub)*kVtStride + kb*16]);
            }
#pragma unroll
            for (int rb = 0; rb < kRowBlocks; ++rb) {
                v8f acc = o_acc[rb][t];
#pragma unroll
                for (int kb = 0; kb < kKeyBlocks; ++kb) {
                    acc = gufo_wmma(gufo_load_frag(&p_lds[rb*16 + sub][kb*16]), v_frag[kb], acc);
                }
                o_acc[rb][t] = acc;
            }
        }
#pragma unroll
        for (int n = 0; n < kKRegs; ++n) {
            k_cur[n] = k_pre[n];
        }
#pragma unroll
        for (int n = 0; n < kVRegs; ++n) {
            v_cur[n] = v_pre[n];
        }
        kt = kt_next;
    }
    if (tid % kLanes == 0) {
        row_sum[tid / kLanes] = running_sum;
    }
    __syncthreads();
#pragma unroll
    for (int rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int dim_tile = wave + t*8;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row   = 2*i + half_id;
                const int query = query_start + row;
                if (query >= n_q) {
                    continue;
                }
                const float den = row_sum[rb*16 + row];
                const int64_t off = (((int64_t)seq*n_q + query)*n_head + first_head + rb)*kD + dim_tile*16 + sub;
                dst[off] = den > 0.0f ? o_acc[rb][t][i]/den : 0.0f;
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, dst, m, bounds, rounds, n_words, scale, n_q, n_kv, n_head, kv_head0, nb01, nb02, nb03, sk1, sk2, sk3, sv1, sv2, sv3);
    NO_DEVICE_CODE;
#endif // GUFO_WMMA_DEVICE
}

} // namespace

static int gufo_keys() {
    static const int v = getenv("GGML_CUDA_FA_GUFO_KEYS") && atoi(getenv("GGML_CUDA_FA_GUFO_KEYS")) == 32 ? 32 : 16;
    return v;
}

static int gufo_min_q() {
    static const int v = getenv("GGML_CUDA_FA_GUFO_MIN_Q") ? atoi(getenv("GGML_CUDA_FA_GUFO_MIN_Q")) : 32;
    return v;
}

bool ggml_cuda_fattn_gufo_enabled() {
    static const bool v = getenv("GGML_CUDA_FA_GUFO_256") && atoi(getenv("GGML_CUDA_FA_GUFO_256")) != 0;
    return v;
}

bool ggml_cuda_fattn_gufo_supported(int device, const ggml_tensor * dst) {
    if (!ggml_cuda_fattn_gufo_enabled()) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA3(cc)) {
        return false;
    }
    const ggml_tensor * Q        = dst->src[0];
    const ggml_tensor * K        = dst->src[1];
    const ggml_tensor * V        = dst->src[2];
    const ggml_tensor * mask     = dst->src[3];
    const ggml_tensor * sinks    = dst->src[4];
    const ggml_tensor * cell_pos = dst->src[5];
    const ggml_tensor * tok_lo   = dst->src[6];
    const ggml_tensor * tok_hi   = dst->src[7];

    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));

    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || Q->nb[0] != sizeof(float)) return false;
    if (K->type != V->type || (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0 && K->type != GGML_TYPE_Q5_0)) return false;
    if (Q->ne[0] != kD || K->ne[0] != kD || V->ne[0] != kD) return false;
    if (K->ne[2] == 0 || (Q->ne[2] != 8*K->ne[2] && Q->ne[2] != 6*K->ne[2]) || V->ne[2] != K->ne[2]) return false;
    if (Q->ne[3] != K->ne[3] || V->ne[3] != K->ne[3]) return false;
    if (Q->ne[1] < gufo_min_q()) return false;
    if (max_bias != 0.0f || softcap != 0.0f || sinks) return false;
    if (mask) {
        if (mask->type != GGML_TYPE_F16 || mask->ne[0] != K->ne[1] || mask->ne[1] < Q->ne[1] || mask->ne[2] != 1) return false;
        if (cell_pos) return false;
    } else {
        if (!cell_pos || !tok_lo || !tok_hi || Q->ne[3] != 1) return false;
    }
    return true;
}

template <int kKeys, int kGqa, int kHeads>
static void gufo_launch_attn(
        const ggml_type kv_type, const dim3 grid, cudaStream_t stream, const char * Q, const char * K, const char * V, float * dst,
        const gufo_mask_args & m, const int2 * bounds, const uint32_t * rounds, const int n_words, const float scale,
        const int n_q, const int n_kv, const int n_head, const int kv_head0, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t sk1, const int64_t sk2, const int64_t sk3, const int64_t sv1, const int64_t sv2, const int64_t sv3) {
#define GUFO_LAUNCH(T) gufo_attn_d256<kKeys, kGqa, kHeads, T><<<grid, dim3(kThreads, 1, 1), 0, stream>>>( \
        Q, K, V, dst, m, bounds, rounds, n_words, scale, n_q, n_kv, n_head, kv_head0, nb01, nb02, nb03, sk1, sk2, sk3, sv1, sv2, sv3)
    switch (kv_type) {
        case GGML_TYPE_F16:  GUFO_LAUNCH(GGML_TYPE_F16);  break;
        case GGML_TYPE_Q8_0: GUFO_LAUNCH(GGML_TYPE_Q8_0); break;
        case GGML_TYPE_Q5_0: GUFO_LAUNCH(GGML_TYPE_Q5_0); break;
        default: GGML_ABORT("gufo: unsupported K/V type");
    }
#undef GUFO_LAUNCH
}

void ggml_cuda_flash_attn_ext_gufo(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q        = dst->src[0];
    const ggml_tensor * K        = dst->src[1];
    const ggml_tensor * V        = dst->src[2];
    const ggml_tensor * mask     = dst->src[3];
    const ggml_tensor * cell_pos = dst->src[5];
    const ggml_tensor * tok_lo   = dst->src[6];
    const ggml_tensor * tok_hi   = dst->src[7];

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_q   = (int) Q->ne[1];
    const int n_kv  = (int) K->ne[1];
    const int n_seq = (int) Q->ne[3];

    const int64_t sk1 = K->nb[1], sk2 = K->nb[2], sk3 = K->nb[3];
    const int64_t sv1 = V->nb[1], sv2 = V->nb[2], sv3 = V->nb[3];

    gufo_mask_args m = {};
    if (mask) {
        m.mask = (const half *) mask->data;
        m.s31  = mask->nb[1]/sizeof(half);
        m.s33  = mask->nb[3]/sizeof(half);
        m.ne33 = (int) mask->ne[3];
    } else {
        m.cell_pos = (const int *) cell_pos->data;
        m.tok_lo   = (const int *) tok_lo->data;
        m.tok_hi   = (const int *) tok_hi->data;
        m.ne33     = 1;
    }

    const int n_tiles = (n_q + 15)/16;
    const int keys    = gufo_keys();
    const int n_words = ((n_kv + keys - 1)/keys + 31)/32;
    ggml_cuda_pool_alloc<int2>     bounds(pool, (size_t) n_tiles*n_seq);
    ggml_cuda_pool_alloc<uint32_t> rounds(pool, (size_t) n_tiles*n_seq*n_words);
    const size_t smem_bits = (size_t) n_words*sizeof(uint32_t);
    const int gqa   = (int) (Q->ne[2]/K->ne[2]);
    const int heads = gqa == 8 ? 4 : 2;
    const int  n_kv_head = (int) K->ne[2];
    if (keys == 32) {
        gufo_kv_bounds<32><<<dim3(n_tiles, n_seq, 1), dim3(kThreads, 1, 1), smem_bits, stream>>>(m, bounds.ptr, rounds.ptr, n_words, n_q, n_kv, n_tiles);
    } else {
        gufo_kv_bounds<16><<<dim3(n_tiles, n_seq, 1), dim3(kThreads, 1, 1), smem_bits, stream>>>(m, bounds.ptr, rounds.ptr, n_words, n_q, n_kv, n_tiles);
    }
    CUDA_CHECK(cudaGetLastError());

    const auto launch = [&](ggml_type type, dim3 grid, const void * k, const void * v, int kv_head0,
                            int64_t k1, int64_t k2, int64_t k3, int64_t v1, int64_t v2, int64_t v3) {
#define GUFO_ARGS type, grid, stream, (const char *) Q->data, (const char *) k, (const char *) v, (float *) dst->data, m, bounds.ptr, rounds.ptr, \
        n_words, scale, n_q, n_kv, (int) Q->ne[2], kv_head0, Q->nb[1], Q->nb[2], Q->nb[3], k1, k2, k3, v1, v2, v3
        if (keys == 32) {
            gqa == 8 ? gufo_launch_attn<32, 8, 4>(GUFO_ARGS) : gufo_launch_attn<32, 6, 2>(GUFO_ARGS);
        } else {
            gqa == 8 ? gufo_launch_attn<16, 8, 4>(GUFO_ARGS) : gufo_launch_attn<16, 6, 2>(GUFO_ARGS);
        }
#undef GUFO_ARGS
    };

    // Quantized K/V: one KV head at a time is converted to F16 (decoding in the kernel repeats the work for every
    // block of the same head and is slower at depth). The buffers are sized from the whole cache, not n_kv, so the
    // pool hands back the same allocation every call instead of keeping one per n_kv.
    const bool stage = K->type != GGML_TYPE_F16 && n_seq == 1 && K->view_src && V->view_src;
    if (!stage) {
        launch(K->type, dim3(n_tiles, (unsigned) (n_kv_head*(gqa/heads)), n_seq), K->data, V->data, 0,
               K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3]);
        return;
    }
    const int64_t cells = std::max<int64_t>(n_kv, std::max(K->view_src->ne[1], V->view_src->ne[1]));
    ggml_cuda_pool_alloc<half> K_f16(pool, (size_t) cells*kD), V_f16(pool, (size_t) cells*kD);
    const to_fp16_nc_cuda_t k_to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
    const to_fp16_nc_cuda_t v_to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
    const size_t kts = ggml_type_size(K->type), vts = ggml_type_size(V->type);
    for (int h = 0; h < n_kv_head; ++h) {
        k_to_fp16((const char *) K->data + h*K->nb[2], K_f16.ptr, kD, n_kv, 1, 1, K->nb[1]/kts, K->nb[2]/kts, K->nb[3]/kts, stream);
        v_to_fp16((const char *) V->data + h*V->nb[2], V_f16.ptr, kD, n_kv, 1, 1, V->nb[1]/vts, V->nb[2]/vts, V->nb[3]/vts, stream);
        launch(GGML_TYPE_F16, dim3(n_tiles, (unsigned) (gqa/heads), 1), K_f16.ptr, V_f16.ptr, h,
               kD*sizeof(half), 0, 0, kD*sizeof(half), 0, 0);
    }
    CUDA_CHECK(cudaGetLastError());
}

#else // !HIP

bool ggml_cuda_fattn_gufo_enabled() { return false; }
bool ggml_cuda_fattn_gufo_supported(int, const ggml_tensor *) { return false; }
void ggml_cuda_flash_attn_ext_gufo(ggml_backend_cuda_context &, ggml_tensor *) { GGML_ABORT("gufo FA is HIP-only"); }

#endif
