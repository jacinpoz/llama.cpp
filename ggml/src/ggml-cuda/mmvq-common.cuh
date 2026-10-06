#pragma once

#include "mmvq.cuh"
#include "vecdotq.cuh"

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static __device__ __forceinline__ void mmvq_prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

// The wide-VDR MoE expert entry points (mul_mat_vec_q_moe) are a RDNA4/RDNA3_0
// (gfx12xx / gfx1100) optimisation.  Every other target -- notably RDNA3_5
// (gfx115x), where the wide chunk never verified -- uses the dense VDR, so the
// MoE expert matmul reduces exactly like the rest of that arch.  Scope it here,
// once, for both selectors below: a per-quant gate is how the Q4_K/Q6_K arms
// (the Q4_K_M expert types) once leaked onto gfx1151.
#if !(defined(RDNA4) || defined(RDNA3_0))
#define GGML_CUDA_MMVQ_MOE_DENSE 1
#endif

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type, bool moe = false) {
#ifdef GGML_CUDA_MMVQ_MOE_DENSE
    moe = false;
#endif
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return moe ? vec_dot_q8_0_q8_1_moe : vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return moe ? vec_dot_q4_K_q8_1_vdr4 : vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return moe ? vec_dot_q5_K_q8_1_vdr4 : vec_dot_q5_K_q8_1;
#if defined(RDNA4) || defined(RDNA3_0)
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1_vdr2;
#else
        case GGML_TYPE_Q6_K:    return moe ? vec_dot_q6_K_q8_1_vdr2 : vec_dot_q6_K_q8_1;
#endif
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type, bool moe = false) {
#ifdef GGML_CUDA_MMVQ_MOE_DENSE
    moe = false;
#endif
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return moe ? VDR_Q8_0_Q8_1_MMVQ_MOE : VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return moe ? VDR_Q4_K_Q8_1_MMVQ_MOE : VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return moe ? VDR_Q5_K_Q8_1_MMVQ_MOE : VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return moe ? VDR_Q6_K_Q8_1_MMVQ_MOE : VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA3_5,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr __host__ __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA3_5;
#elif defined(RDNA2)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA3_5;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // Band-uniform nwarps=1 for the whole mmvq band (ncols_dst 1..8).
        // nwarps participates in the K-split accumulation order, so decode and the
        // speculative verify batch must use the same value.  The Q8_0 weight kernel
        // gets a per-(type,K) override from calc_nwarps_weight() below; every other
        // launch (including the GDN/SSM and shared-expert fusions) stays at 1.
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        // Apply to the whole mmvq range (ncols_dst 1..8), not just decode: the
        // speculative verify batch (n_draft+1 tokens) must use the same nwarps
        // as decode so its per-row dot-product accumulation is bit-identical.
        if (ncols_dst <= MMVQ_MAX_BATCH_SIZE) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    // gfx1100 sweep 2026-08-28 (rdna3-boosts R5): nwarps=8 beats
                    // the old 2 by +0.8-1.0% decode; the other widened types
                    // (Q2_K/Q4_K/Q5_K/IQ4_XS at 8) regressed -4..-7% and stay 1.
                    return 8;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_5) {
        // gfx1151 (Strix Halo iGPU): nwarps=1 (the RDNA2 table) underutilizes the
        // wave32 datapath on the large-K decode matmuls; nwarps=8 (the RDNA3_0
        // table) over-parallelizes the small ones. Swept 2025-08: nwarps=2 wins
        // (~+0.6% decode on Qwen3.6-35B-A3B Q8_0), nwarps=4 regresses.
        // Apply to the whole mmvq range (ncols_dst 1..8), not just decode: the
        // speculative verify batch (n_draft+1 tokens) must use the same nwarps
        // as decode so its per-row dot-product accumulation is bit-identical.
        if (ncols_dst <= MMVQ_MAX_BATCH_SIZE) {
            switch (type) {
                case GGML_TYPE_Q8_0:
                    return 2;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

// nwarps for a dense mmvq *weight* launch (the ksplit kernel).  A Q8_0 weight with a
// short K (the MoE attention qkv/gate and the lm_head, K < 4096) wins from the
// pre-2026-09-11 wider block; the long-K Q8_0 hybrid projections (dense 27B
// attn_v/ssm_out, K >= 5120) lose from it and shift the MTP arithmetic, so they stay
// at 1.  The choice is per tensor shape (K is fixed for a weight), so W = 1..8 of the
// same tensor still agree.  Fusion ops (GDN/SSM, shared-expert, the gate fusions)
// MUST keep plain calc_nwarps -- their nwarps is a single-token reduction-order anchor.
static constexpr __host__ __device__ int calc_nwarps_weight(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool long_k) {
    if (table_id == MMVQ_PARAMETERS_RDNA4 && type == GGML_TYPE_Q8_0 && !long_k && ncols_dst <= MMVQ_MAX_BATCH_SIZE) {
        return 8;
    }
    return calc_nwarps(type, ncols_dst, table_id);
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

// rows_per_block override for dispatch-selected kernels (e.g. the small-K MoE
// down projection on RDNA, where a single 544 B row per block underutilizes the
// memory system). 0 = use calc_rows_per_block().
static constexpr __host__ __device__ int calc_rows_per_block_override(int rows_per_block, int ncols_dst, int table_id, bool small_k, int nwarps) {
    return rows_per_block > 0 ? rows_per_block : calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
}

// rows per block for a dense mmvq *weight* launch (the ksplit kernel).  On RDNA4 a
// multi-token (verify) batch computes several rows per block so the ncols_dst q8_1
// activation columns are loaded once per block instead of once per row.  Each row's
// accumulation (thread mapping, K order, warp reduction) is unchanged, so W = 1..8
// stay bit-identical.
#ifndef GGML_MMVQ_RDNA4_WEIGHT_RPB
#define GGML_MMVQ_RDNA4_WEIGHT_RPB 4
#endif
// Multi-row blocks only pay off while the launch still has enough blocks to fill the GPU:
// with fewer than this many blocks (e.g. 1024-row K/V or 48-row SSM projections at 4 rows)
// the one-row launch is faster.
#ifndef GGML_MMVQ_RDNA4_WEIGHT_RPB1
#define GGML_MMVQ_RDNA4_WEIGHT_RPB1 2
#endif
#ifndef GGML_MMVQ_RDNA4_WEIGHT_MIN_BLOCKS
#define GGML_MMVQ_RDNA4_WEIGHT_MIN_BLOCKS 512
#endif
static constexpr __host__ __device__ int calc_rows_per_block_weight(ggml_type type, int ncols_dst, int table_id, bool small_k, int nwarps) {
    // gfx1100, K = 6144, n = 5: Q6_K 45.6 -> 39.5 us and IQ4_XS 23.5 -> 18.5 us at 2 rows; MTP verify +2.1%.
    // IQ4_XS at 8 columns would overflow the megakernel's LDS budget.
    if (table_id == MMVQ_PARAMETERS_RDNA3_0 && ncols_dst >= 2 &&
            (type == GGML_TYPE_Q6_K || (type == GGML_TYPE_IQ4_XS && ncols_dst <= 7))) {
        return 2;
    }
    // Q4_0 (the MTP layer) and Q8_0 gain at every width, decode included: K = 5120 Q4_0 44 -> 28 us at n = 1, MTP +0.36%.
    if (table_id == MMVQ_PARAMETERS_RDNA3_0 && ncols_dst <= 7 && (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0)) {
        return 2;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4 && ncols_dst >= 1 && ncols_dst <= MMVQ_MAX_BATCH_SIZE) {
        // Multi-row blocks only pay off for one-wave blocks: the 8-wave Q8_0 short-K block
        // (calc_nwarps_weight) was measured slower at every row count (K = 2880, 8 tokens:
        // +21..41 %, issue #71).  ncols_dst == 1 keeps RPB1 below (it does not go through
        // this early return).
        if (ncols_dst >= 2 && nwarps > 1) {
            return 1;
        }
        // single-token decode too: a one-row, one-warp block keeps too few weight loads in flight
        // (Q5_K/Q6_K ~270 GB/s cold); rows are independent, so this is bit-identical as well
        const int rpb_max = ncols_dst == 1 ? GGML_MMVQ_RDNA4_WEIGHT_RPB1 : GGML_MMVQ_RDNA4_WEIGHT_RPB;
        switch (type) {
            // gfx1201 sweep (rows 1/2/4 x every type x n = 2..8 x 4 shapes): these gain
            // nothing from more rows (NVFP4 loses 40-60 %, Q1_0 ~10 %) and keep one row.
            case GGML_TYPE_NVFP4:
            case GGML_TYPE_Q1_0:
                return 1;
            // these peak at 2 rows
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_Q2_0:
            case GGML_TYPE_Q5_1:
                return rpb_max < 2 ? rpb_max : 2;
            // Q2_K's dot product exceeds the 256-VGPR budget at 4 rows x 7..8 columns and
            // spills to scratch; it keeps 2 rows there.
            case GGML_TYPE_Q2_K:
                return ncols_dst > 6 && rpb_max > 2 ? 2 : rpb_max;
            default:
                return rpb_max;
        }
    }
    return calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
}


// ---------------------------------------------------------------------------
// Llama-Frankenstein R1 helpers: reduce n per-lane values (n <= width) so that every value ends up summed over
// the whole warp, using the butterfly's pairing tree (offsets width/2 ... 1, own value + partner value).
// Returns the sum held by this lane: value index lane / (width / pow2_ceil(n)); the padding values are zeros that
// are only ever added to each other.
static constexpr int ggml_cuda_lf_pow2_ceil(const int n) {
    int p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

static constexpr int ggml_cuda_lf_log2(const int n) {
    int l = 0;
    while ((1 << l) < n) {
        ++l;
    }
    return l;
}

// One halving step: keep half of the values, exchange the other half with lane ^ offset. half and offset are
// template constants so every index stays a compile-time constant (a runtime index turns v[] into selects).
template <int half, int offset, int width, int p>
static __device__ __forceinline__ void ggml_cuda_lf_halving_step(float (&v)[p], const int lane) {
    const bool upper = (lane & offset) != 0;
#pragma unroll
    for (int k = 0; k < half; ++k) {
        const float send = upper ? v[k]        : v[k + half];
        const float keep = upper ? v[k + half] : v[k];
        v[k] = keep + __shfl_xor_sync(0xffffffff, send, offset, width);
    }
    if constexpr (half > 1) {
        ggml_cuda_lf_halving_step<half/2, offset/2, width, p>(v, lane);
    }
}

template <int offset, int width>
static __device__ __forceinline__ float ggml_cuda_lf_butterfly_rest(float x) {
    if constexpr (offset > 0) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
        return ggml_cuda_lf_butterfly_rest<offset/2, width>(x);
    } else {
        return x;
    }
}

template <int n, int width>
static __device__ __forceinline__ float ggml_cuda_lf_reduce_scatter(const float (&v_in)[n]) {
    constexpr int p = ggml_cuda_lf_pow2_ceil(n);
    static_assert(p >= 2 && p <= width, "2..width values");
    float v[p];
#pragma unroll
    for (int k = 0; k < p; ++k) {
        v[k] = k < n ? v_in[k] : 0.0f;
    }
    ggml_cuda_lf_halving_step<p/2, width/2, width, p>(v, threadIdx.x % width);
    return ggml_cuda_lf_butterfly_rest<width/(2*p), width>(v[0]);
}
