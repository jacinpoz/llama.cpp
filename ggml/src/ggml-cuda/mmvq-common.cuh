#pragma once

// The mmvq parameter tables and device helpers, shared by mmvq.cu and the megakernel FFN ops (mk-ops-ffn.cuh).

#include "mmvq.cuh"
#include "vecdotq.cuh"

#include <cstdint>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
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

#include "mmvq-vecdot.cuh"
#include "mmvq-moe.cuh"

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA3_5,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10,
    // halo-hybrid: real RDNA2 (gfx103x). gfx1151 also uses the RDNA2 table above by default (GGML_CUDA_MMVQ_RDNA35),
    // so the Navi21 tuning (llama-navi21-furnace 0aa20a02b, b5f65d215, 05512bb1d) gets its own id
    MMVQ_PARAMETERS_NAVI21
};

#include "mmvq-rdna.cuh"

static constexpr __host__ __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA3_5;
#elif defined(RDNA2)
    return MMVQ_PARAMETERS_NAVI21;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

// VDR of the dense mmvq kernels.  get_vec_dot_q_cuda() picks vec_dot_q4_K_q8_1_vdr4_dense for dense Q4_K on RDNA3_0,
// and on RDNA4 when built with GGML_HIP_RDNA4_Q4K_DENSE_VDR4 (device side, by arch macro); the host's K-split
// arithmetic must use the same VDR, so both sides read it here.
static constexpr __host__ __device__ int get_vdr_mmvq_dense(ggml_type type, mmvq_parameter_table_id table_id) {
#if defined(GGML_HIP_RDNA4_Q4K_DENSE_VDR4)
    const bool q4k_dense_vdr4 = table_id == MMVQ_PARAMETERS_RDNA3_0 || table_id == MMVQ_PARAMETERS_RDNA4;
#else
    const bool q4k_dense_vdr4 = table_id == MMVQ_PARAMETERS_RDNA3_0;
#endif
    return q4k_dense_vdr4 && type == GGML_TYPE_Q4_K ? VDR_Q4_K_Q8_1_MMVQ_RDNA3_0 : get_vdr_mmvq(type);
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
        return MMVQ_PARAMETERS_NAVI21;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false, bool small_m = false) {
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
                return 2;
            case 2:
            case 3:
                return type == GGML_TYPE_Q3_K ? 1 : 2;
            case 4:
                return type == GGML_TYPE_Q6_K? 1 : 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    // Delivery RDNA band-uniform policy; 0 = not ours (fall through to upstream logic).
    const int rdna_nwarps = mmvq_rdna_nwarps(type, ncols_dst, table_id, small_m);
    if (rdna_nwarps > 0) {
        return rdna_nwarps;
    }

    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
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
    // Navi21 (V620, from llama-navi21-furnace): upstream had no RDNA2 entry, so every dispatch ran as one wave32
    // workgroup. ncols_dst == 1 wants 4 waves to keep enough loads in flight; the verify shapes carry ncols_dst
    // accumulators per wave, so wider blocks stop paying (V620, Qwen3.8-27B: 2..4 at 2 waves +6.3%, 5..8 at 1 wave
    // with 4 rows per block +7.6%)
    if (table_id == MMVQ_PARAMETERS_NAVI21) {
        if (ncols_dst == 1) {
            return 4;
        }
        if (ncols_dst <= 4) {
            return 2;
        }
        return 1;
    }
    return 1;
}

// Low bit types load few weight bytes per thread, a second row per block shares the q8_1 loads and adds memory parallelism.
static constexpr __host__ __device__ bool mmvq_two_rows_ncols1(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    // Navi21 (V620, from llama-navi21-furnace): upstream had no RDNA2 entry, so every dispatch ran as one wave32
    // workgroup. ncols_dst == 1 wants 4 waves to keep enough loads in flight; the verify shapes carry ncols_dst
    // accumulators per wave, so wider blocks stop paying (V620, Qwen3.8-27B: 2..4 at 2 waves +6.3%, 5..8 at 1 wave
    // with 4 rows per block +7.6%)
    if (table_id == MMVQ_PARAMETERS_NAVI21) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
                return 2;
            // 5..8 (block-drafter verify): 4 rows amortise the 8 y-columns each wave holds; 8 rows regress
            case 5:
            case 6:
            case 7:
            case 8:
                return 4;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                if (small_k) {
                    return nwarps;
                }
                if (table_id == MMVQ_PARAMETERS_GB10) {
                    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S ? 2 : 1;
                }
                return table_id == MMVQ_PARAMETERS_GENERIC && mmvq_two_rows_ncols1(type) ? 2 : 1;
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

#include "mmvq-rdna-params.cuh"

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

