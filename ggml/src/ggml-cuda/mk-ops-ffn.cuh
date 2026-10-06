#pragma once

// Ops whose sub-tile width depends on the variant are one struct per width; mk_<op>_dispatch(variant, f) calls f(Op{}).
// run() ends with lds still being read: the caller needs a barrier before the next run() on the same lds.

#include "common.cuh"
#include "megakernel.cuh"
#include "mmvq-common.cuh"
#include "unary.cuh"

#include <cfloat>
#include <cstdint>

template <typename Op, typename F>
static __device__ __forceinline__ void mk_call(F && f) {
    static_assert(MK_THREADS % Op::threads == 0, "sub-tile width must divide MK_THREADS");
    static_assert(MK_THREADS / Op::threads * Op::lds_bytes <= MK_LDS_BYTES, "sub-tile LDS exceeds MK_LDS_BYTES");
    f(Op{});
}

template <int width>
static __device__ __forceinline__ float mk_block_reduce_sum(float val, float * shared_vals, const int lane) {
    val = warp_reduce_sum(val);
    if constexpr (width > WARP_SIZE) {
        const int warp_id = lane / WARP_SIZE;
        const int lane_id = lane % WARP_SIZE;
        if (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        __syncthreads();
        val = 0.0f;
        if (lane_id < width / WARP_SIZE) {
            val = shared_vals[lane_id];
        }
        return warp_reduce_sum(val);
    }
    return val;
}

// MK_OP_RMSNORM_Q8_1: tile = row + nrows*(channel + nchannels*sample).
// variant bit 0: has_add (x = xa + xb is written first; y may then be null). bit 1: width 256.

struct mk_rmsnorm_q8_1_params {
    const float * x;
    float       * dst;
    block_q8_1  * y;
    const float * mul;
    const float * xa;
    const float * xb;
    int64_t stride_row;
    int64_t stride_channel;
    int64_t stride_sample;
    int64_t mul_stride_row;
    int     ncols;
    int     ncols_padded;
    int     nrows;
    int     nchannels;
    float   eps;
};

#define MK_RMSNORM_Q8_1_ADD     1
#define MK_RMSNORM_Q8_1_W256    2

template <int BLOCK, int WIDTH = 1024>
struct mk_rmsnorm_q8_1 {
    static constexpr int threads   = WIDTH;
    static constexpr int lds_bytes = WIDTH > WARP_SIZE ? WARP_SIZE*sizeof(float) : 0;

    template <bool has_add>
    static __device__ __forceinline__ void row(const mk_rmsnorm_q8_1_params & p, const int row, const int channel, const int sample,
            const int tid, bool valid, char * lds) {
        constexpr int block_size = WIDTH;
        const int ncols     = p.ncols;
        // an invalid sub-tile runs the barriers with no memory traffic
        const int ncols_v   = valid ? ncols : 0;
        float * s_sum       = (float *) lds;

        const float * x   = p.x + sample*p.stride_sample + channel*p.stride_channel + row*p.stride_row;
        float       * dst = p.dst + ((sample*p.nchannels + channel)*p.nrows + row)*ncols;
        block_q8_1  * y   = p.y;
        const float * mul = p.mul;
        if (y != nullptr) {
            y += ((sample*p.nchannels + channel)*p.nrows + row)*(p.ncols_padded/QK8_1);
        }
        if (mul != nullptr) {
            mul += row*p.mul_stride_row;
        }

        // Rows of up to kRegs*block_size columns keep each thread's values in registers between the passes
        // (same arithmetic as the reloading loops below, so the results are bit-identical).
        constexpr int kRegs = 8;
        if (ncols <= kRegs*block_size) {
            float v[kRegs];
            float w[kRegs];
            float tmp = 0.0f;
#pragma unroll
            for (int k = 0; k < kRegs; ++k) {
                const int col = tid + k*block_size;
                if (col < ncols_v) {
                    w[k] = mul == nullptr ? 1.0f : mul[col];
                    float xi;
                    if constexpr (has_add) {
                        const int64_t off = sample*p.stride_sample + channel*p.stride_channel + row*p.stride_row;
                        xi = p.xa[off + col] + p.xb[off + col];
                        ((float *) x)[col] = xi;
                    } else {
                        xi = x[col];
                    }
                    v[k] = xi;
                    tmp += xi * xi;
                }
            }

            tmp = mk_block_reduce_sum<block_size>(tmp, s_sum, tid);

            const float mean = tmp / ncols;
            const float scale = rsqrtf(mean + p.eps);

#pragma unroll
            for (int k = 0; k < kRegs; ++k) {
                const int col = tid + k*block_size;
                if (col < ncols_v) {
                    v[k] = mul == nullptr ? scale * v[k] : scale * v[k] * w[k];
                    dst[col] = v[k];
                }
            }

            if constexpr (has_add) {
                if (y == nullptr) {
                    return;
                }
            }

            // a warp's lanes hold the 32 consecutive columns of one Q8_1 block
#pragma unroll
            for (int k = 0; k < kRegs; ++k) {
                const int col = tid + k*block_size;
                if (k*block_size >= ncols_v) {
                    break;
                }
                const int ib = col / QK8_1;
                const int lane = col % QK8_1;
                const float xi = col < ncols ? v[k] : 0.0f;
                float amax = fabsf(xi);
                float sum  = xi;
                amax = warp_reduce_max(amax);
                sum  = warp_reduce_sum(sum);
                if (col < ncols) {
                    const float d = amax / 127.0f;
                    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
                    y[ib].qs[lane] = q;
                    if (lane == 0) {
                        y[ib].ds = make_half2(d, sum);
                    }
                }
            }
            return;
        }

        float tmp = 0.0f;
        if constexpr (has_add) {
            // contiguous rows: xa, xb and the sum x share the row offset
            const int64_t off = sample*p.stride_sample + channel*p.stride_channel + row*p.stride_row;
            float * xs = (float *) x;
            for (int col = tid; col < ncols_v; col += block_size) {
                const float xi = p.xa[off + col] + p.xb[off + col];
                xs[col] = xi;
                tmp += xi * xi;
            }
        } else {
            for (int col = tid; col < ncols_v; col += block_size) {
                const float xi = x[col];
                tmp += xi * xi;
            }
        }

        tmp = mk_block_reduce_sum<block_size>(tmp, s_sum, tid);

        const float mean = tmp / ncols;
        const float scale = rsqrtf(mean + p.eps);

        if (mul == nullptr) {
            for (int col = tid; col < ncols_v; col += block_size) {
                dst[col] = scale * x[col];
            }
        } else {
            for (int col = tid; col < ncols_v; col += block_size) {
                dst[col] = scale * x[col] * mul[col];
            }
        }

        __syncthreads();

        if constexpr (has_add) {
            if (y == nullptr) {
                return;
            }
        }

        for (int col = tid; col < ncols_v; col += block_size) {
            const int ib = col / QK8_1;
            const int lane = col % QK8_1;
            const float xi = dst[col];
            float amax = fabsf(xi);
            float sum  = xi;
            amax = warp_reduce_max(amax);
            sum  = warp_reduce_sum(sum);
            const float d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
            y[ib].qs[lane] = q;
            if (lane == 0) {
                y[ib].ds = make_half2(d, sum);
            }
        }
    }

    static __device__ __forceinline__ void run(const mk_rmsnorm_q8_1_params & p, int variant, int tile, bool valid, char * lds) {
        const int r = tile % p.nrows;
        const int c = tile / p.nrows % p.nchannels;
        const int s = tile / p.nrows / p.nchannels;
        if (variant & MK_RMSNORM_Q8_1_ADD) {
            row<true>(p, r, c, s, threadIdx.x % WIDTH, valid, lds);
        } else {
            row<false>(p, r, c, s, threadIdx.x % WIDTH, valid, lds);
        }
    }
};

template <int BLOCK, typename F>
static __device__ __forceinline__ void mk_rmsnorm_q8_1_dispatch(int variant, F && f) {
    if (variant & MK_RMSNORM_Q8_1_W256) {
        mk_call<mk_rmsnorm_q8_1<BLOCK, 256>>(f);
    } else {
        mk_call<mk_rmsnorm_q8_1<BLOCK, 1024>>(f);
    }
}

// MK_OP_RMSNORM_F32: tile = row + nrows*(channel + nchannels*sample).
// variant bit 0: multiply by mul. bit 1: width 256.
// The add form is reachable from the __global__ wrapper only.

struct mk_rmsnorm_f32_params {
    const float * x;
    float       * dst;
    const float * mul;
    const float * add;
    uint16_t    * dst16;
    int64_t stride_row;
    int64_t stride_channel;
    int64_t stride_sample;
    int64_t mul_stride_row;
    int64_t mul_stride_channel;
    int64_t mul_stride_sample;
    int64_t add_stride_row;
    int64_t add_stride_channel;
    int64_t add_stride_sample;
    uint3   mul_ncols_packed;
    uint3   mul_nrows_packed;
    uint3   mul_nchannels_packed;
    uint3   mul_nsamples_packed;
    uint3   add_ncols_packed;
    uint3   add_nrows_packed;
    uint3   add_nchannels_packed;
    uint3   add_nsamples_packed;
    int     ncols;
    int     nrows;
    int     nchannels;
    float   eps;
    bool    store_f32;
};

#define MK_RMSNORM_F32_MUL  1
#define MK_RMSNORM_F32_W256 2

// RNE f32 -> bf16
static __device__ __forceinline__ uint16_t mk_f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

template <int BLOCK, int WIDTH = 1024>
struct mk_rmsnorm_f32 {
    static constexpr int threads   = WIDTH;
    static constexpr int lds_bytes = WIDTH > WARP_SIZE ? WARP_SIZE*sizeof(float) : 0;

    template <bool do_multiply, bool do_add>
    static __device__ __forceinline__ void row(const mk_rmsnorm_f32_params & p, const int row, const int channel, const int sample,
            const int tid, bool valid, char * lds) {
        static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
        constexpr int block_size = WIDTH;
        const int ncols   = p.ncols;
        const int ncols_v = valid ? ncols : 0;

        const float * x     = p.x + sample*p.stride_sample + channel*p.stride_channel + row*p.stride_row;
        float       * dst   = p.dst + ((sample*p.nchannels + channel)*p.nrows + row)*ncols;
        uint16_t    * dst16 = p.dst16;
        const float * mul   = p.mul;
        const float * add   = p.add;
        if (dst16 != nullptr) {
            dst16 += ((sample*p.nchannels + channel)*p.nrows + row)*ncols;
        }

        if constexpr (do_multiply) {
            const uint32_t mul_row     = fastmodulo(row, p.mul_nrows_packed);
            const uint32_t mul_channel = fastmodulo(channel, p.mul_nchannels_packed);
            const uint32_t mul_sample  = fastmodulo(sample, p.mul_nsamples_packed);
            mul += mul_sample * p.mul_stride_sample + mul_channel * p.mul_stride_channel + mul_row * p.mul_stride_row;
        }

        if constexpr (do_add) {
            const int add_row     = fastmodulo(row, p.add_nrows_packed);
            const int add_channel = fastmodulo(channel, p.add_nchannels_packed);
            const int add_sample  = fastmodulo(sample, p.add_nsamples_packed);
            add += add_sample * p.add_stride_sample + add_channel * p.add_stride_channel + add_row * p.add_stride_row;
        }

        float tmp = 0.0f;

        for (int col = tid; col < ncols_v; col += block_size) {
            const float xi = x[col];
            tmp += xi * xi;
        }

        tmp = mk_block_reduce_sum<block_size>(tmp, (float *) lds, tid);

        const float mean = tmp / ncols;
        const float scale = rsqrtf(mean + p.eps);

        for (int col = tid; col < ncols_v; col += block_size) {
            float v;
            if constexpr (do_multiply && do_add) {
                const int mul_col = fastmodulo(col, p.mul_ncols_packed);
                const int add_col = fastmodulo(col, p.add_ncols_packed);
                v = scale * x[col] * mul[mul_col] + add[add_col];
            } else if constexpr (do_multiply) {
                const int mul_col = fastmodulo(col, p.mul_ncols_packed);
                v = scale * x[col] * mul[mul_col];
            } else {
                v = scale * x[col];
            }
            if (p.store_f32) dst[col] = v;
            if (dst16 != nullptr) {
                dst16[col] = mk_f2bf(v);
            }
        }
    }

    static __device__ __forceinline__ void run(const mk_rmsnorm_f32_params & p, int variant, int tile, bool valid, char * lds) {
        const int r = tile % p.nrows;
        const int c = tile / p.nrows % p.nchannels;
        const int s = tile / p.nrows / p.nchannels;
        if (variant & MK_RMSNORM_F32_MUL) {
            row<true, false>(p, r, c, s, threadIdx.x % WIDTH, valid, lds);
        } else {
            row<false, false>(p, r, c, s, threadIdx.x % WIDTH, valid, lds);
        }
    }
};

template <int BLOCK, typename F>
static __device__ __forceinline__ void mk_rmsnorm_f32_dispatch(int variant, F && f) {
    if (variant & MK_RMSNORM_F32_W256) {
        mk_call<mk_rmsnorm_f32<BLOCK, 256>>(f);
    } else {
        mk_call<mk_rmsnorm_f32<BLOCK, 1024>>(f);
    }
}

// MK_OP_QUANTIZE_Q8_1: tile = bx + nblocks_x*(i1 + ne1*(i2 + ne2*i3)). No variants.

struct mk_quantize_q8_1_params {
    const float * x;
    void        * vy;
    int64_t  ne00;
    int64_t  s01;
    int64_t  s02;
    int64_t  s03;
    int64_t  ne0;
    uint32_t ne1;
    uint3    ne2;
    uint32_t nblocks_x;
};

template <int BLOCK>
struct mk_quantize_q8_1 {
    static constexpr int threads   = 256;
    static constexpr int lds_bytes = 0;

    static __device__ __forceinline__ void run(const mk_quantize_q8_1_params & p, int variant, int tile, bool valid, char * lds) {
        GGML_UNUSED_VARS(variant, lds);
        block(p, tile % p.nblocks_x, tile / p.nblocks_x % p.ne1, tile / p.nblocks_x / p.ne1, threadIdx.x % threads, valid);
    }

    static __device__ __forceinline__ void block(const mk_quantize_q8_1_params & p, const uint32_t bx, const int64_t i1,
            const uint32_t bz, const int tid, bool valid) {
        const float * GGML_CUDA_RESTRICT x  = p.x;
        void        * GGML_CUDA_RESTRICT vy = p.vy;
        const int64_t i0 = (int64_t)threads*bx + tid;

        if (!valid || i0 >= p.ne0) {
            return;
        }

        const int64_t i3 = fastdiv(bz, p.ne2);
        const int64_t i2 = bz - i3*p.ne2.z;

        const int64_t & i00 = i0;
        const int64_t & i01 = i1;
        const int64_t & i02 = i2;
        const int64_t & i03 = i3;

        const int64_t i_cont = ((i3*p.ne2.z + i2) * p.ne1 + i1) * p.ne0 + i0;

        block_q8_1 * y = (block_q8_1 *) vy;

        const int64_t ib  = i_cont / QK8_1;
        const int64_t iqs = i_cont % QK8_1;

        const float xi = i0 < p.ne00 ? x[i03*p.s03 + i02*p.s02 + i01*p.s01 + i00] : 0.0f;
        mk_quantize_q8_1_group(xi, y + ib, iqs);
    }
};

// MK_OP_MMVQ: tile = bx + nblocks_x*(channel_dst + nchannels_dst*sample_dst), bx = row block.

struct mk_mmvq_params {
    const void    * vx;
    const void    * vy;
    const int32_t * ids;
    float         * dst;
    ggml_cuda_mm_fusion_args_device fusion;
    uint3    nchannels_y;
    uint3    channel_ratio;
    uint3    sample_ratio;
    uint32_t ncols_x;
    uint32_t stride_row_x;
    uint32_t stride_col_y;
    uint32_t stride_col_dst;
    uint32_t stride_channel_x;
    uint32_t stride_channel_y;
    uint32_t stride_channel_dst;
    uint32_t stride_sample_x;
    uint32_t stride_sample_y;
    uint32_t stride_sample_dst;
    uint32_t nblocks_x;
    uint32_t nchannels_dst;
};

// long_k = K >= 4096, as the host dispatch picks it. On RDNA4 the builder must also keep nrows % rows_per_block == 0.
static constexpr __host__ __device__ int mk_mmvq_variant(ggml_type type, int ncols_dst, bool has_fusion, bool long_k) {
    return int(type) | ncols_dst << 8 | int(has_fusion) << 12 | int(long_k) << 13;
}

template <int BLOCK, ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, int rows_per_block = 0,
          bool long_k = false>
struct mk_mmvq {
    static constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    static constexpr int nwarps    = calc_nwarps_weight(type, ncols_dst, table_id, long_k);
    static constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static constexpr int rows_per_cuda_block =
        rows_per_block > 0 ? rows_per_block : calc_rows_per_block_weight(type, ncols_dst, table_id, small_k, nwarps);
    static constexpr int shared_rows = nwarps-1 > 0 ? nwarps-1 : 1;
    static constexpr int shared_rows_gate = has_fusion ? shared_rows : 0;
    static constexpr int shared_floats = shared_rows*ncols_dst*rows_per_cuda_block*warp_size;
    static constexpr int threads   = nwarps*warp_size;
    static constexpr int lds_bytes = (shared_floats + shared_rows_gate*ncols_dst*rows_per_cuda_block*warp_size)*sizeof(float);

    static __device__ __forceinline__ void run(const mk_mmvq_params & p, int variant, int tile, bool valid, char * lds) {
        GGML_UNUSED(variant);
        // A sub-tile is whole waves, so tile is wave-uniform. Making that explicit keeps the branch below scalar:
        // block() holds a barrier, which must not sit in control flow the compiler thinks diverges.
        tile = __builtin_amdgcn_readfirstlane(tile);
        const int local = threadIdx.x % threads;
        const int wid   = __builtin_amdgcn_readfirstlane(local / warp_size);
        // Decode tiles are one channel and one sample: skip the three integer divisions, a real cost per one-row
        // block. A single block() call keeps its barrier in one place.
        uint32_t bx = tile, channel = 0, sample = 0;
        if ((uint32_t) tile >= p.nblocks_x) {
            bx      = tile % p.nblocks_x;
            channel = tile / p.nblocks_x % p.nchannels_dst;
            sample  = tile / p.nblocks_x / p.nchannels_dst;
        }
        block(p, bx, channel, sample, local % warp_size, wid, valid, lds);
    }

    static __device__ __forceinline__ void block(const mk_mmvq_params & p, const uint32_t bx, const uint32_t channel_dst,
            const uint32_t sample_dst_in, const int lane, const int wid, bool valid, char * lds) {
        const void    * GGML_CUDA_RESTRICT vx  = p.vx;
        const void    * GGML_CUDA_RESTRICT vy  = p.vy;
        const int32_t * GGML_CUDA_RESTRICT ids = p.ids;
        float         * GGML_CUDA_RESTRICT dst = p.dst;
        const ggml_cuda_mm_fusion_args_device & fusion = p.fusion;

        const uint32_t ncols_x            = p.ncols_x;
        const uint32_t stride_row_x       = p.stride_row_x;
        const uint32_t stride_col_y       = p.stride_col_y;
        const uint32_t stride_col_dst     = p.stride_col_dst;
        const uint32_t stride_channel_x   = p.stride_channel_x;
        const uint32_t stride_channel_y   = p.stride_channel_y;
        const uint32_t stride_channel_dst = p.stride_channel_dst;
        const uint32_t stride_sample_x    = p.stride_sample_x;
        const uint32_t stride_sample_y    = p.stride_sample_y;
        const uint32_t stride_sample_dst  = p.stride_sample_dst;

        constexpr int qk  = ggml_cuda_type_traits<type>::qk;
        constexpr int qi  = ggml_cuda_type_traits<type>::qi;
        constexpr int vdr = get_vdr_mmvq(type);

        constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

        const     int tid   = warp_size*wid + lane;
        const     int row0  = rows_per_cuda_block*bx;
        const     int blocks_per_row_x = ncols_x / qk;
        constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

        uint32_t channel_x;
        uint32_t channel_y;
        uint32_t sample_dst;

        channel_x  = ncols_dst == 1 && ids ? (valid ? ids[channel_dst] : 0)          : fastdiv(channel_dst, p.channel_ratio);
        channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, p.nchannels_y) : channel_dst;
        sample_dst = sample_dst_in;

        const uint32_t sample_x    = fastdiv(sample_dst, p.sample_ratio);
        const uint32_t sample_y    = sample_dst;

        bool use_gate = false;
        bool use_bias = false;
        bool use_gate_bias = false;
        bool use_scale = false;
        bool use_gate_scale = false;
        bool use_dst_gate = false;
        bool use_conv_input = false;
        [[maybe_unused]] const void * vgate = nullptr;
        const float * x_bias = nullptr;
        const float * gate_bias = nullptr;
        const float * x_scale = nullptr;
        const float * gate_scale = nullptr;
        [[maybe_unused]] const void * dst_gate = nullptr;
        [[maybe_unused]] float * conv_input = nullptr;
        [[maybe_unused]] const float * conv_states = nullptr;
        [[maybe_unused]] int conv_kernel_size = 0;
        [[maybe_unused]] const int32_t * conv_state_ids = nullptr;
        [[maybe_unused]] const float * conv_state_src = nullptr;
        [[maybe_unused]] int64_t conv_state_row_stride = 0;
        [[maybe_unused]] float * conv_state_dst = nullptr;
        ggml_glu_op active_glu;
        float glu_limit = 0.0f;

        if constexpr (has_fusion) {
            use_gate      = fusion.gate      != nullptr;
            use_bias      = fusion.x_bias    != nullptr;
            use_gate_bias = fusion.gate_bias != nullptr && use_gate;
            vgate         = fusion.gate;
            x_bias        = (const float *) fusion.x_bias;
            gate_bias     = (const float *) fusion.gate_bias;
            active_glu    = fusion.glu_op;
            glu_limit     = fusion.glu_limit;
            use_dst_gate  = fusion.dst_gate != nullptr && use_gate;
            if (use_dst_gate) {
                dst_gate = fusion.dst_gate;
            }
            use_conv_input = fusion.conv_input != nullptr && (fusion.conv_states != nullptr || fusion.conv_state_src != nullptr);
            if (use_conv_input) {
                conv_input            = (float *) fusion.conv_input;
                conv_states           = (const float *) fusion.conv_states;
                conv_kernel_size      = fusion.conv_kernel_size;
                conv_state_ids        = fusion.conv_state_ids;
                conv_state_src        = fusion.conv_state_src;
                conv_state_row_stride = fusion.conv_state_row_stride;
                conv_state_dst        = fusion.conv_state_dst;
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                use_scale      = fusion.x_scale    != nullptr;
                use_gate_scale = fusion.gate_scale != nullptr && use_gate;
                x_scale        = (const float *) fusion.x_scale;
                gate_scale     = (const float *) fusion.gate_scale;
            }
            // Per-token scale (MoE down x topk weights). Indexed by channel_dst.
            if (fusion.x_scale_channel_dst) {
                use_scale = true;
                x_scale   = (const float *) fusion.x_scale;
            }
        }

        [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
        [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
        [[maybe_unused]] float x_scales = 1.0f;
        [[maybe_unused]] float gate_scales = 1.0f;
        if constexpr (has_fusion) {
            const uint32_t channel_bias = ids ? channel_x : channel_dst;
            if (valid && lane < rows_per_cuda_block && wid == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + lane) < stride_col_dst)) {
                if (use_bias) {
                    x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        x_biases[j] = x_bias[j * stride_col_dst + lane];
                    }
                }
                if (use_gate_bias) {
                    gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        gate_biases[j] = gate_bias[j * stride_col_dst + lane];
                    }
                }
                if (use_scale) {
                    x_scales = fusion.x_scale_channel_dst ? x_scale[channel_dst] : x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }

        // partial sum for each thread
        float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
        float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

        const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
        const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;
        const int kbx_end    = valid ? blocks_per_row_x : 0;

        for (int kbx = tid / (qi/vdr); kbx < kbx_end; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

            // x block quant index when casting the quants to int
            const int kqs = vdr * (tid % (qi/vdr));

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
            // start the next iterations' weight loads early
            if constexpr (mmvq_should_prefetch(type)) {
                constexpr int pf_dist = 2; // loop iterations, not blocks
                const int kbx_pf = kbx + pf_dist*blocks_per_iter;
                if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                    for (int i = 0; i < rows_per_cuda_block; ++i) {
                        const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                        mmvq_prefetch_l2((const char *) vx + off);
                        if constexpr (has_fusion) {
                            if (use_gate) {
                                mmvq_prefetch_l2((const char *) vgate + off);
                            }
                        }
                    }
                }
            }
#endif

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    tmp[j][i] += vec_dot_q_cuda(
                        vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_gate[j][i] += vec_dot_q_cuda(
                                vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                        }
                    }
                }
            }
        }

        typedef float shared_t[ncols_dst][rows_per_cuda_block][warp_size];
        shared_t * tmp_shared = (shared_t *) lds;
        [[maybe_unused]] shared_t * tmp_shared_gate = (shared_t *) (lds + shared_floats*sizeof(float));

        if (wid > 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    tmp_shared[wid-1][j][i][lane] = tmp[j][i];
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_shared_gate[wid-1][j][i][lane] = tmp_gate[j][i];
                        }
                    }
                }
            }
        }
        // A one-wave op has nothing to exchange; a barrier would make every wave of a wider block wait for the slowest row.
        if constexpr (nwarps > 1) {
            __syncthreads();
        }
        if (wid > 0 || !valid) {
            return;
        }

        dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

        // Llama-Frankenstein R1: without fusion, reduce all ncols_dst*rows_per_cuda_block partial sums at once by
        // recursive halving instead of one butterfly per output (8 tokens x 4 rows: 31 lane exchanges instead of 160).
        // Every output is summed by the same pairing tree (lane ^ 16, ^ 8, ... ^ 1, own value first) as
        // warp_reduce_sum, only in a different lane, so the results are bit-identical.
        // One-wave blocks only: in the 8-wave Q8_0 short-K block the gathering warp's serial halving steps were
        // slower on small grids (gpt-oss k/v, 512 rows x 6 tokens: 6.7 -> 11.1 us per launch).
        constexpr int lf_nval = ncols_dst*rows_per_cuda_block;
        if constexpr (!has_fusion && nwarps == 1 && lf_nval >= 2 && lf_nval <= warp_size) {
            float lf_v[lf_nval];
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    float t = tmp[j][i];
#pragma unroll
                    for (int l = 0; l < nwarps-1; ++l) {
                        t += tmp_shared[l][j][i][lane];
                    }
                    lf_v[j*rows_per_cuda_block + i] = t;
                }
            }
            const float lf_sum = ggml_cuda_lf_reduce_scatter<lf_nval, warp_size>(lf_v);
            constexpr int lf_lanes = warp_size / ggml_cuda_lf_pow2_ceil(lf_nval); // lanes holding the same output
            const int lf_idx = lane / lf_lanes;
            if (lane % lf_lanes == 0 && lf_idx < lf_nval) {
                const int j = lf_idx / rows_per_cuda_block;
                const int i = lf_idx % rows_per_cuda_block;
                if (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst) {
                    dst[j*stride_col_dst + i] = lf_sum;
                }
            }
        } else {
        // sum up partial sums and write back result
        #pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
        #pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
        #pragma unroll
                    for (int l = 0; l < nwarps-1; ++l) {
                        tmp[j][i] += tmp_shared[l][j][i][lane];
                        if constexpr (has_fusion) {
                            if (use_gate) {
                                tmp_gate[j][i] += tmp_shared_gate[l][j][i][lane];
                            }
                        }
                    }
                    tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                        }
                    }

                    float result_val = 0.0f;
                    if (lane == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                        result_val = tmp[j][i];
                        if constexpr (has_fusion) {
                            if (use_scale) {
                                result_val *= x_scales;
                            }
                            result_val += x_biases[j];
                            if (use_gate) {
                                float gate_value = tmp_gate[j][i];
                                if constexpr (type == GGML_TYPE_NVFP4) {
                                    gate_value *= gate_scales;
                                }
                                gate_value += gate_biases[j];
                                if (use_dst_gate) {
                                    float * dst_gate_row = (float *) dst_gate + sample_dst*stride_sample_dst +
                                                           channel_dst*stride_channel_dst + row0 + j*stride_col_dst;
                                    dst_gate_row[i] = gate_value;
                                } else {
                                    switch (active_glu) {
                                        case GGML_GLU_OP_SWIGLU:
                                            result_val *= ggml_cuda_op_silu_single(gate_value);
                                            break;
                                        case GGML_GLU_OP_GEGLU:
                                            result_val *= ggml_cuda_op_gelu_single(gate_value);
                                            break;
                                        case GGML_GLU_OP_SWIGLU_OAI:
                                            result_val = ggml_cuda_op_swiglu_oai_single(gate_value, result_val);
                                            break;
                                        case GGML_GLU_OP_SWIGLU_CLAMP:
                                            result_val = ggml_cuda_op_swiglu_clamp_single(gate_value, result_val, glu_limit);
                                            break;
                                        default:
                                            result_val = result_val * gate_value;
                                            break;
                                    }
                                }
                            }
                        }
                    }
                    if (use_conv_input) {
                        // interleaved conv input: [state0, state1, ..., state_{cs-2}, result] per
                        // channel. conv_input [cs, C] has nb1 = cs floats, conv_states [(cs-1), C]
                        // has nb1 = cs-1 floats. Spread the 4 writes over threads 0..cs-1.
                        const float r_bcast = __shfl_sync(0xffffffff, result_val, i, warp_size);
                        const int c = row0 + i;
                        const int cs = conv_kernel_size;
                        const int k = lane;
                        float v = r_bcast;
                        if (k < cs - 1) {
                            v = conv_state_src != nullptr ? conv_state_src[conv_state_ids[0]*conv_state_row_stride + (cs-1)*c + k]
                                                          : conv_states[(cs-1)*c + k];
                        }
                        if (k < cs) {
                            conv_input[cs*c + k] = v;
                        }
                        if (conv_state_dst != nullptr) {
                            // the next state is conv_input columns 1..cs-1; every lane has loaded before any lane stores,
                            // so conv_state_dst may be the row conv_state_src was read from
                            const float v_next = __shfl_sync(0xffffffff, v, k + 1, warp_size);
                            if (k < cs - 1) {
                                conv_state_dst[(cs-1)*c + k] = v_next;
                            }
                        }
                    } else if (lane == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                        dst[j*stride_col_dst + i] = result_val;
                    }
                }
            }
        }

        if constexpr (has_fusion) {
            static_assert(QK8_1 % rows_per_cuda_block == 0, "a block's rows must sit in one Q8_1 group");
            if (fusion.q8_1_out != nullptr) {
                // Publish this block's rows, then count them in their 32-row group.
                // The wave that completes the group quantizes it for every column, reading the other waves' rows after an acquire.
                // The host requires an unpadded row, so column j's Q8_1 row starts at block j*stride_col_dst/QK8_1.
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
                const uint32_t group = row0 / QK8_1;
                uint32_t prev = 0;
                if (lane == 0) {
                    prev = __hip_atomic_fetch_add(fusion.q8_1_group_done + group, (uint32_t) rows_per_cuda_block,
                                                  __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
                }
                prev = __builtin_amdgcn_readfirstlane(prev);
                if (prev == QK8_1 - rows_per_cuda_block) {
                    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        const float xi = __hip_atomic_load(dst + j*stride_col_dst - row0 % QK8_1 + lane,
                                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                        mk_quantize_q8_1_group(xi, (block_q8_1 *) fusion.q8_1_out + j*(stride_col_dst/QK8_1) + group, lane);
                    }
                    if (lane == 0) {
                        fusion.q8_1_group_done[group] = 0;
                    }
                }
            }
        }
        if constexpr (!has_fusion) {
            GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, use_dst_gate, use_conv_input, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate, dst_gate, conv_input, conv_states, conv_kernel_size);
        }
        if constexpr (type != GGML_TYPE_NVFP4) {
            GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
        }
    }
};

// long_k only changes the kernel where calc_nwarps_weight differs, so the other types share one instantiation.
template <int BLOCK, ggml_type type, int ncols_dst, bool has_fusion, bool long_k, typename F>
static __device__ __forceinline__ void mk_mmvq_call(F && f) {
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr bool lk = long_k || calc_nwarps_weight(type, ncols_dst, table_id, false) == calc_nwarps_weight(type, ncols_dst, table_id, true);
    mk_call<mk_mmvq<BLOCK, type, ncols_dst, has_fusion, false, 0, lk>>(f);
}

template <int BLOCK, ggml_type type, bool long_k, typename F>
static __device__ __forceinline__ void mk_mmvq_dispatch_ncols(int ncols_dst, bool has_fusion, F && f) {
    switch (ncols_dst) {
        case 1:
            if (has_fusion) {
                mk_mmvq_call<BLOCK, type, 1, true,  long_k>(f);
            } else {
                mk_mmvq_call<BLOCK, type, 1, false, long_k>(f);
            }
            break;
        case 2: mk_mmvq_call<BLOCK, type, 2, false, long_k>(f); break;
        case 3: mk_mmvq_call<BLOCK, type, 3, false, long_k>(f); break;
        case 4: mk_mmvq_call<BLOCK, type, 4, false, long_k>(f); break;
        case 5: mk_mmvq_call<BLOCK, type, 5, false, long_k>(f); break;
        case 6: mk_mmvq_call<BLOCK, type, 6, false, long_k>(f); break;
        case 7: mk_mmvq_call<BLOCK, type, 7, false, long_k>(f); break;
        case 8: mk_mmvq_call<BLOCK, type, 8, false, long_k>(f); break;
        default: break;
    }
}

template <int BLOCK, ggml_type type, typename F>
static __device__ __forceinline__ void mk_mmvq_dispatch_type(int ncols_dst, bool has_fusion, bool long_k, F && f) {
    if (long_k) {
        mk_mmvq_dispatch_ncols<BLOCK, type, true>(ncols_dst, has_fusion, f);
    } else {
        mk_mmvq_dispatch_ncols<BLOCK, type, false>(ncols_dst, has_fusion, f);
    }
}

// Fused variants exist only at ncols_dst == 1; wider fused launches are recorded as host-only.
template <int BLOCK, typename F>
static __device__ __forceinline__ void mk_mmvq_dispatch(int variant, F && f) {
    const int  ncols_dst  = (variant >> 8) & 0xF;
    const bool has_fusion = (variant >> 12) & 1;
    const bool long_k     = (variant >> 13) & 1;
    switch (ggml_type(variant & 0xFF)) {
        case GGML_TYPE_IQ4_XS: mk_mmvq_dispatch_type<BLOCK, GGML_TYPE_IQ4_XS>(ncols_dst, has_fusion, long_k, f); break;
        case GGML_TYPE_Q6_K:   mk_mmvq_dispatch_type<BLOCK, GGML_TYPE_Q6_K>  (ncols_dst, has_fusion, long_k, f); break;
        case GGML_TYPE_Q8_0:   mk_mmvq_dispatch_type<BLOCK, GGML_TYPE_Q8_0>  (ncols_dst, has_fusion, long_k, f); break;
        case GGML_TYPE_Q4_K:   mk_mmvq_dispatch_type<BLOCK, GGML_TYPE_Q4_K>  (ncols_dst, has_fusion, long_k, f); break;
        case GGML_TYPE_Q5_K:   mk_mmvq_dispatch_type<BLOCK, GGML_TYPE_Q5_K>  (ncols_dst, has_fusion, long_k, f); break;
        default: break;
    }
}

// argmax_f32 at 1024 threads (ncols > 992) breaks ties by the smallest key (bitrev5(warp), bitrev5(lane), c / 1024),
// c % 1024 = 32*warp + lane; values <= -FLT_MAX and NaN never win.

static __device__ __forceinline__ uint32_t mk_bitrev5(uint32_t v) {
    return __brev(v) >> 27;
}

static __device__ __forceinline__ uint64_t mk_argmax_key(int32_t c) {
    const uint32_t t = uint32_t(c) % 1024;
    return (uint64_t) (mk_bitrev5(t / 32)*32 + mk_bitrev5(t % 32)) << 32 | uint32_t(c) / 1024;
}

// i < 0 is the empty candidate
static __device__ __forceinline__ bool mk_argmax_better(float v0, int32_t i0, float v1, int32_t i1) {
    if (i0 < 0) {
        return false;
    }
    if (i1 < 0) {
        return true;
    }
    return v0 > v1 || (v0 == v1 && mk_argmax_key(i0) < mk_argmax_key(i1));
}

template <int threads>
static __device__ __forceinline__ void mk_argmax_reduce(float & val, int32_t & idx, char * lds, const int lane) {
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float   v = __shfl_xor_sync(0xFFFFFFFF, val, offset, WARP_SIZE);
        const int32_t i = __shfl_xor_sync(0xFFFFFFFF, idx, offset, WARP_SIZE);
        if (mk_argmax_better(v, i, val, idx)) {
            val = v;
            idx = i;
        }
    }
    constexpr int nwarps = threads / WARP_SIZE;
    float   * s_val = (float *) lds;
    int32_t * s_idx = (int32_t *) (lds + nwarps*sizeof(float));
    if (lane % WARP_SIZE == 0) {
        s_val[lane / WARP_SIZE] = val;
        s_idx[lane / WARP_SIZE] = idx;
    }
    __syncthreads();
#pragma unroll
    for (int w = 0; w < nwarps; ++w) {
        if (mk_argmax_better(s_val[w], s_idx[w], val, idx)) {
            val = s_val[w];
            idx = s_idx[w];
        }
    }
}

// MK_OP_LM_HEAD_ARGMAX: tile = chunk + nchunks*row, chunk = columns [chunk*chunk_cols, (chunk+1)*chunk_cols) of row.
// idx_offset is added to every column index. Writes part_val/part_idx[tile]. No variants.
struct mk_argmax_partial_params {
    const float * x;
    float       * part_val;
    int32_t     * part_idx;
    int64_t       stride_row;
    int32_t       ncols;
    int32_t       chunk_cols;
    int32_t       nchunks;
    int32_t       idx_offset;
};

template <int BLOCK>
struct mk_argmax_partial {
    static constexpr int threads   = 256;
    static constexpr int lds_bytes = threads/WARP_SIZE*(sizeof(float) + sizeof(int32_t));

    static __device__ __forceinline__ void run(const mk_argmax_partial_params & p, int variant, int tile, bool valid, char * lds) {
        GGML_UNUSED(variant);
        const int lane  = threadIdx.x % threads;
        const int chunk = tile % p.nchunks;
        const int row   = tile / p.nchunks;
        const int c0    = chunk*p.chunk_cols;
        const int c1    = valid ? min(p.ncols, c0 + p.chunk_cols) : c0;
        const float * x = p.x + row*p.stride_row;

        float   val = -FLT_MAX;
        int32_t idx = -1;
        for (int c = c0 + lane; c < c1; c += threads) {
            const float   v = x[c];
            const int32_t i = c + p.idx_offset;
            if (v > -FLT_MAX && mk_argmax_better(v, i, val, idx)) {
                val = v;
                idx = i;
            }
        }
        mk_argmax_reduce<threads>(val, idx, lds, lane);
        if (valid && lane == 0) {
            p.part_val[tile] = val;
            p.part_idx[tile] = idx;
        }
    }
};

// MK_OP_ARGMAX_COMBINE: dst[row] = winner of part_*[row*nchunks, (row+1)*nchunks). tile = row. No variants.
struct mk_argmax_combine_params {
    const float   * part_val;
    const int32_t * part_idx;
    int32_t       * dst;
    int32_t         nchunks;
};

template <int BLOCK>
struct mk_argmax_combine {
    static constexpr int threads   = 256;
    static constexpr int lds_bytes = threads/WARP_SIZE*(sizeof(float) + sizeof(int32_t));

    static __device__ __forceinline__ void run(const mk_argmax_combine_params & p, int variant, int tile, bool valid, char * lds) {
        GGML_UNUSED(variant);
        const int lane = threadIdx.x % threads;
        const int n    = valid ? p.nchunks : 0;

        float   val = -FLT_MAX;
        int32_t idx = -1;
        for (int k = lane; k < n; k += threads) {
            const float   v = p.part_val[tile*p.nchunks + k];
            const int32_t i = p.part_idx[tile*p.nchunks + k];
            if (mk_argmax_better(v, i, val, idx)) {
                val = v;
                idx = i;
            }
        }
        mk_argmax_reduce<threads>(val, idx, lds, lane);
        if (valid && lane == 0) {
            p.dst[tile] = idx;
        }
    }
};
