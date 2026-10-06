#pragma once

// Gated DeltaNet decode ops (megakernel.cuh contract). k_gdn_gates, ssm_conv_f32, rms_norm_scale_pair_f32,
// gated_delta_net_cuda and norm_silu_gate_q8_1_kernel call this device code.

#include "common.cuh"
#include "megakernel.cuh"
#include "unary.cuh"

#define MK_GDN_MAX_T 8

// T comes from launch when the op runs in the megakernel, else from n_tokens.
template <typename P>
static __device__ __forceinline__ int mk_gdn_n_tokens(const P & p) {
    return p.launch != nullptr ? p.launch->n_tokens : p.n_tokens;
}

// Sub-tiles of one megakernel block run back to back: no sub-tile may overwrite its LDS for the next tile
// while another is still reading it for this one. The standalone kernels (BLOCK == threads) skip it.
template <int BLOCK, int threads>
static __device__ __forceinline__ void mk_gdn_lds_release() {
    if constexpr (BLOCK != threads) {
        __syncthreads();
    }
}

// Opaque to the optimizer. With a constant row width or trip count, fp contraction would otherwise fuse across this
// value (x*x into the first shuffle add, x/128 + eps into one fma).
static __device__ __forceinline__ float mk_gdn_fp_fence(float v) {
#if defined(GGML_USE_HIP)
    asm volatile("" : "+v"(v));
#else
    asm volatile("" : "+f"(v));
#endif
    return v;
}

// block_reduce<SUM, threads> split at its barrier. A 128-wide sub-tile sums identically to a 256-thread block, whose
// extra warps only add +0.0f partials.
static __device__ __forceinline__ void mk_gdn_sum_partial(float v, float * part, const int tid) {
    v = warp_reduce_sum(v);
    if (tid % WARP_SIZE == 0) {
        part[tid / WARP_SIZE] = v;
    }
}

template <int threads>
static __device__ __forceinline__ float mk_gdn_sum_total(const float * part, const int tid) {
    const int lane = tid % WARP_SIZE;
    return warp_reduce_sum(lane < threads/WARP_SIZE ? part[lane] : 0.0f);
}

// rms_norm(eps) followed by ggml_scale(s, b), split at the reduction: the GDN q/k l2 norm (build_gdn_l2_norm).
template <int threads>
static __device__ __forceinline__ float mk_gdn_row_sumsq(const float * x, const int ncols, const int tid) {
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += threads) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    return mk_gdn_fp_fence(tmp);
}

template <int threads>
static __device__ __forceinline__ void mk_gdn_row_scale_store(const float * x, float * dst, const int ncols, const int tid,
        const float sumsq, const float eps, const float s, const float b) {
    const float mean  = mk_gdn_fp_fence(sumsq / ncols);
    const float scale = rsqrtf(mean + eps);
    for (int col = tid; col < ncols; col += threads) {
        const float v = scale * x[col];
        dst[col] = s * v + b;
    }
}

// MK_OP_GDN_GATES: gate = softplus(W_alpha x + dt) * A, beta = sigmoid(W_beta x) for 1..8 tokens.
// tile = t*2H + row; rows [0, H) are alpha, [H, 2H) beta. Tiles of tokens >= T are invalid.
struct mk_gdn_gates_params {
    const char  * w_alpha;
    const char  * w_beta;
    const float * x;
    const float * dt;
    const float * a;
    float       * gate;
    float       * beta;
    int           K;
    int           H;
    int64_t       nbw_alpha;
    int64_t       nbw_beta;
    int64_t       sx;
    int64_t       s_gate;
    int64_t       s_beta;
    int           n_tokens;
    const mk_launch_params * launch;
};

template <int BLOCK>
struct mk_gdn_gates {
    static constexpr int threads   = 256;
    static constexpr int lds_bytes = threads/WARP_SIZE*sizeof(float);

    static __device__ void run(const mk_gdn_gates_params & p, int variant, int tile, bool valid, char * lds) {
        GGML_UNUSED(variant);
        run_at(p, tile % (2*p.H), tile / (2*p.H), valid, lds);
    }

    static __device__ void run_at(const mk_gdn_gates_params & p, const int row, const int t, bool valid, char * lds) {
        const int tid = threadIdx.x % threads;
        valid = valid && t < mk_gdn_n_tokens(p);

        const bool is_alpha = row < p.H;
        const int  r        = is_alpha ? row : row - p.H;

        float sum = 0.0f;
        if (valid) {
            const float4 * w  = reinterpret_cast<const float4 *>(is_alpha ? p.w_alpha + r*p.nbw_alpha : p.w_beta + r*p.nbw_beta);
            const float4 * xv = reinterpret_cast<const float4 *>(p.x + t*p.sx);
            for (int k = tid; k < p.K/4; k += threads) {
                const float4 wv = w[k];
                const float4 xx = xv[k];
                sum += wv.x*xx.x + wv.y*xx.y + wv.z*xx.z + wv.w*xx.w;
            }
        }
        float * partial = (float *) lds;
        mk_gdn_sum_partial(sum, partial, tid);
        __syncthreads();
        if (tid == 0 && valid) {
            float v = 0.0f;
#pragma unroll
            for (int j = 0; j < threads/WARP_SIZE; ++j) {
                v += partial[j];
            }

            if (is_alpha) {
                const float z  = v + p.dt[r];
                const float sp = z > 20.0f ? z : logf(1.0f + expf(z));
                p.gate[t*p.s_gate + r] = sp*p.a[r];
            } else {
                p.beta[t*p.s_beta + r] = 1.0f / (1.0f + expf(-v));
            }
        }
        mk_gdn_lds_release<BLOCK, threads>();
    }
};

// One channel of SSM_CONV (+ SILU) over n_t tokens, shared with ssm_conv_f32. x_at(j) is column j of the
// channel's [d_conv-1 state | n_t inputs] window, store(i, v) writes token i.
template <bool apply_silu, int d_conv, typename load_t, typename store_t>
static __device__ __forceinline__ void mk_ssm_conv_channel(const float * w_row, const float b, const int64_t n_t,
        const load_t & x_at, const store_t & store) {
    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

#pragma unroll
    for (int j = 0; j < d_conv; j++) {
        w[j] = w_row[j];
    }

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (int j = 0; j < d_conv; j++) {
                x[j] = x_at(j);
            }
        } else {
            x[(i - 1) % d_conv] = x_at(i + d_conv - 1);
        }

#pragma unroll
        for (int j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        store(i, apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf);
    }
}

// MK_OP_GDN_CONV: the decode conv step on the plain qkv projection. Reads the conv state in place, writes the
// shifted state snapshots, SSM_CONV + SILU, then the q and k l2 norms. T <= MK_GDN_MAX_T; x, w and the state rows
// are contiguous.
// tile = channel group of 128 = one q/k head: tiles [0, H_k) also write q, [H_k, 2*H_k) k.
// Snapshot slot s (0 <= s <= min(n_slots - 1, T)) gets window columns [T-s, T-s+3): slot 0 the new state,
// slot T the pre-batch one (build_conv_state's CPYs).
struct mk_gdn_conv_params {
    const float   * x;                 // [C, T] qkv projection
    const float   * state_src;         // conv state rows, [d_conv-1, C] each
    const int32_t * state_ids;         // nullptr: state_src is the row itself
    int64_t         state_row_stride;  // floats
    float         * state_dst;         // snapshot slot 0 row
    int64_t         state_slot_stride; // floats
    int             n_slots;
    const float   * w;                 // [d_conv, C]
    const float   * bias;              // nullptr: none
    float         * y;                 // [C, T]
    float         * q_out;             // [S_k, H_k, T]
    float         * k_out;
    int             C;
    int             n_qk_heads;
    float           eps;               // rms_norm eps (eps/S_k in build_gdn_l2_norm)
    float           scale;             // ggml_scale s and b
    float           scale_bias;
    int             n_tokens;
    const mk_launch_params * launch;
};

template <int BLOCK>
struct mk_gdn_conv {
    static constexpr int threads   = 128;
    static constexpr int d_conv    = 4;
    static constexpr int lds_bytes = MK_GDN_MAX_T*(threads/WARP_SIZE)*sizeof(float);

    static __device__ void run(const mk_gdn_conv_params & p, int variant, int tile, bool valid, char * lds) {
        GGML_UNUSED(variant);
        const int tid = threadIdx.x % threads;
        const int c   = tile*threads + tid;
        const int n_t = mk_gdn_n_tokens(p);

        if (valid) {
            const float * st = p.state_src + (p.state_ids != nullptr ? p.state_ids[0]*p.state_row_stride : 0) + c*(d_conv - 1);
            const float s0 = st[0];
            const float s1 = st[1];
            const float s2 = st[2];
            const auto x_at = [&](const int64_t j) {
                return j == 0 ? s0 : j == 1 ? s1 : j == 2 ? s2 : p.x[(j - (d_conv - 1))*p.C + c];
            };
            const float b = p.bias != nullptr ? p.bias[c] : 0.0f;
            mk_ssm_conv_channel<true, d_conv>(p.w + c*d_conv, b, n_t, x_at,
                [&](const int64_t i, const float v) { p.y[i*p.C + c] = v; });

            const int last_slot = min(p.n_slots - 1, n_t);
            for (int s = 0; s <= last_slot; ++s) {
                float * dst = p.state_dst + s*p.state_slot_stride + c*(d_conv - 1);
#pragma unroll
                for (int k = 0; k < d_conv - 1; ++k) {
                    dst[k] = x_at(n_t - s + k);
                }
            }
        }

        // each thread reads back only its own y writes
        const bool  norm = valid && tile < 2*p.n_qk_heads;
        const int   head = tile < p.n_qk_heads ? tile : tile - p.n_qk_heads;
        float     * out  = tile < p.n_qk_heads ? p.q_out : p.k_out;
        float     * part = (float *) lds;
#pragma unroll
        for (int t = 0; t < MK_GDN_MAX_T; ++t) {
            if (norm && t < n_t) {
                const float sumsq = mk_gdn_row_sumsq<threads>(p.y + t*p.C + tile*threads, threads, tid);
                mk_gdn_sum_partial(sumsq, part + t*(threads/WARP_SIZE), tid);
            }
        }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < MK_GDN_MAX_T; ++t) {
            if (norm && t < n_t) {
                const float sumsq = mk_gdn_sum_total<threads>(part + t*(threads/WARP_SIZE), tid);
                mk_gdn_row_scale_store<threads>(p.y + t*p.C + tile*threads, out + (t*p.n_qk_heads + head)*threads,
                    threads, tid, sumsq, p.eps, p.scale, p.scale_bias);
            }
        }
        mk_gdn_lds_release<BLOCK, threads>();
    }
};

// rms_norm + scale over two same-shaped inputs, for the standalone rms_norm_scale_pair_f32 only. MK_OP_GDN_CONV
// runs the same row code inline.
// tile = (z*nchannels + channel)*nrows + row, z in [0, 2*nsamples): the first input, then the second.
struct mk_gdn_qk_norm_params {
    const float * x0;
    float       * dst0;
    const float * x1;
    float       * dst1;
    int           ncols;
    int64_t       stride_row;
    int64_t       stride_channel;
    int64_t       stride_sample;
    int           nrows;
    int           nchannels;
    int           nsamples;
    float         eps;
    float         s;
    float         b;
};

template <int BLOCK>
struct mk_gdn_qk_norm {
    static constexpr int threads   = 256;
    static constexpr int lds_bytes = threads/WARP_SIZE*sizeof(float);

    static __device__ void run(const mk_gdn_qk_norm_params & p, int variant, int tile, bool valid, char * lds) {
        GGML_UNUSED(variant);
        run_at(p, tile % p.nrows, (tile / p.nrows) % p.nchannels, tile / (p.nrows*p.nchannels), valid, lds);
    }

    static __device__ void run_at(const mk_gdn_qk_norm_params & p, const int row, const int channel, const int z, bool valid,
            char * lds) {
        const int  tid     = threadIdx.x % threads;
        const bool second  = z >= p.nsamples;
        const int  sample  = second ? z - p.nsamples : z;

        const float * x   = (second ? p.x1 : p.x0) + sample*p.stride_sample + channel*p.stride_channel + row*p.stride_row;
        float       * dst = (second ? p.dst1 : p.dst0) + ((sample*p.nchannels + channel)*p.nrows + row)*p.ncols;

        float * part = (float *) lds;
        mk_gdn_sum_partial(valid ? mk_gdn_row_sumsq<threads>(x, p.ncols, tid) : 0.0f, part, tid);
        __syncthreads();
        const float sumsq = mk_gdn_sum_total<threads>(part, tid);
        if (valid) {
            mk_gdn_row_scale_store<threads>(x, dst, p.ncols, tid, sumsq, p.eps, p.s, p.b);
        }
        mk_gdn_lds_release<BLOCK, threads>();
    }
};

// MK_OP_GDN_STEP: the sequential gated delta rule (gated_delta_net_cuda) with the state read in place.
// One warp per state column, 4 columns per tile. tile = (sequence*n_col_groups + col_group)*H + head.
// variant bit 0: keep the last K per-token states (keep_rs_t).
struct mk_gdn_step_params {
    const float   * q;
    const float   * k;
    const float   * v;
    const float   * g;
    const float   * beta;
    const float   * curr_state;
    float         * dst;
    float         * state;
    int64_t         H;
    int64_t         n_tokens;
    int64_t         n_seqs;
    int64_t         n_col_groups;
    int64_t         sq1, sq2, sq3;
    int64_t         sv1, sv2, sv3;
    int64_t         sb1, sb2, sb3;
    uint3           neqk1_magic;
    uint3           rq3_magic;
    float           scale;
    int64_t         state_slot_stride;
    int             K;
    const int32_t * state_ids;
    int64_t         state_row_stride;
    float         * state_pre;         // nullptr: none. Receives the input state (the pre-batch rollback slot).
    const mk_launch_params * launch;
};

static constexpr int mk_gdn_step_warps = 4;

template <int S_v>
static constexpr __device__ int mk_gdn_step_warp_size() {
    return ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
}

// tid is the thread's linear index within the tile (the standalone kernel uses a 2D block).
// Each warp owns ncols adjacent columns; every column's arithmetic is the same for any ncols.
template <int S_v, bool KDA, bool keep_rs_t, int ncols = 1>
static __device__ __forceinline__ void mk_gdn_step_tile(const mk_gdn_step_params & p, const uint32_t h_idx,
        const uint32_t col_group, const uint32_t sequence, const int tid) {
    constexpr int warp_size = mk_gdn_step_warp_size<S_v>();
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;

    const int64_t  n_tokens  = mk_gdn_n_tokens(p);
    const int      lane      = tid % warp_size;
    const int      col0      = (col_group * mk_gdn_step_warps + tid / warp_size) * ncols;

    const uint32_t iq1 = fastmodulo(h_idx, p.neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, p.rq3_magic);

    const int64_t H = p.H;
    float * attn_data = p.dst;
    float * state     = p.state;

    // input state holds s0 only: [S_v, S_v, H, n_seqs], seq stride D = H * S_v * S_v.
    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float s_shard[ncols][rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    // state_ids: the state is read in place from cache row state_ids[sequence] (the GET_ROWS gather was elided)
    const float * curr_state = p.curr_state + (p.state_ids != nullptr ? p.state_ids[sequence] * p.state_row_stride : state_in_offset) + col0 * S_v;
    if (p.state_ids != nullptr) {
        curr_state += h_idx * S_v * S_v;
    }
#pragma unroll
    for (int c = 0; c < ncols; c++) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[c][r] = curr_state[c * S_v + r * warp_size + lane];
        }
    }

    // The columns are fully loaded before any write, so the slot may alias the rows they were read from.
    if (p.state_pre != nullptr) {
        float * pre = p.state_pre + state_out_offset + col0 * S_v;
#pragma unroll
        for (int c = 0; c < ncols; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                pre[c * S_v + r * warp_size + lane] = s_shard[c][r];
            }
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = p.q + iq3 * p.sq3 + t * p.sq2 + iq1 * p.sq1;
        const float * k_t = p.k + iq3 * p.sq3 + t * p.sq2 + iq1 * p.sq1;
        const float * v_t = p.v + sequence * p.sv3 + t * p.sv2 + h_idx * p.sv1;

        const int64_t gb_offset = sequence * p.sb3 + t * p.sb2 + h_idx * p.sb1;
        const float * beta_t = p.beta + gb_offset;
        const float * g_t    = p.g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

#pragma unroll
        for (int c = 0; c < ncols; c++) {
            const int col = col0 + c;
            if constexpr (!KDA) {
                const float g_val = expf(*g_t);

                // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[c][r] * k_reg[r];
                }
                float kv_col = warp_reduce_sum<warp_size>(kv_shard);

                // delta[col] = (v[col] - g * kv[col]) * beta
                float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

                // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
                // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r] = g_val * s_shard[c][r] + k_reg[r] * delta_col;
                    attn_partial += s_shard[c][r] * q_reg[r];
                }

                float attn_col = warp_reduce_sum<warp_size>(attn_partial);

                if (lane == 0) {
                    attn_data[col] = attn_col * p.scale;
                }
            } else {
                // kv[col] = sum_i g[i] * S[i][col] * k[i]
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    kv_shard += expf(g_t[i]) * s_shard[c][r] * k_reg[r];
                }

                float kv_col = warp_reduce_sum<warp_size>(kv_shard);

                // delta[col] = (v[col] - kv[col]) * beta
                float delta_col = (v_t[col] - kv_col) * beta_val;

                // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
                // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    s_shard[c][r] = expf(g_t[i]) * s_shard[c][r] + k_reg[r] * delta_col;
                    attn_partial += s_shard[c][r] * q_reg[r];
                }

                float attn_col = warp_reduce_sum<warp_size>(attn_partial);

                if (lane == 0) {
                    attn_data[col] = attn_col * p.scale;
                }
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < p.K) {
                float * slot_state = state + target_slot * p.state_slot_stride;
#pragma unroll
                for (int c = 0; c < ncols; c++) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        slot_state[(col0 + c) * S_v + r * warp_size + lane] = s_shard[c][r];
                    }
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < ncols; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                state[(col0 + c) * S_v + r * warp_size + lane] = s_shard[c][r];
            }
        }
    }
}

// Scalar gate only.
template <int BLOCK>
struct mk_gdn_step {
    static constexpr int S_v       = 128;
    static constexpr int threads   = mk_gdn_step_warp_size<S_v>() * mk_gdn_step_warps;
    static constexpr int lds_bytes = 0;

    static __device__ void run(const mk_gdn_step_params & p, int variant, int tile, bool valid, char * lds) {
        GGML_UNUSED(lds);
        if (!valid) {
            return;
        }
        const int      tid       = threadIdx.x % threads;
        const uint32_t h_idx     = tile % p.H;
        const uint32_t col_group = (tile / p.H) % p.n_col_groups;
        const uint32_t sequence  = tile / (p.H*p.n_col_groups);
        if (variant & 1) {
            mk_gdn_step_tile<S_v, false, true>(p, h_idx, col_group, sequence, tid);
        } else {
            mk_gdn_step_tile<S_v, false, false>(p, h_idx, col_group, sequence, tid);
        }
    }
};

// MK_OP_GDN_OUT_GATE: rms_norm(x) * w of a head row, times silu(z), quantized to the out projection's Q8_1 input.
// tile = ggml row of x (head + t*n_heads). Tiles of tokens >= T are invalid.
struct mk_gdn_out_gate_params {
    const float * x;
    int64_t       sx;
    const float * w;
    float         eps;
    const float * z;
    int64_t       sz;
    block_q8_1  * y;
    int64_t       row_len;   // the out projection's ne10
    int           n_heads;
    int           n_tokens;
    const mk_launch_params * launch;
};

template <int BLOCK, int ncols>
struct mk_gdn_out_gate {
    static_assert(ncols % QK8_1 == 0 && ncols <= 256, "ncols");
    static constexpr int threads   = 256;
    static constexpr int lds_bytes = threads/WARP_SIZE*sizeof(float);

    static __device__ void run(const mk_gdn_out_gate_params & p, int variant, int tile, bool valid, char * lds) {
        GGML_UNUSED(variant);
        const int tid = threadIdx.x % threads;
        const int row = tile;
        valid = valid && row < p.n_heads*mk_gdn_n_tokens(p);

        const float xi = valid && tid < ncols ? p.x[row*p.sx + tid] : 0.0f;
        float * part = (float *) lds;
        // the select keeps xi * xi from contracting into the first shuffle add
        mk_gdn_sum_partial(tid < ncols ? xi * xi : 0.0f, part, tid);
        __syncthreads();
        const float tmp   = mk_gdn_sum_total<threads>(part, tid);
        const float mean  = tmp / ncols;
        const float scale = rsqrtf(mean + p.eps);

        if (valid && tid < ncols) {
            const float n = scale * xi * p.w[tid];
            const float v = ggml_cuda_op_silu_single(p.z[row*p.sz + tid]) * n;

            const int64_t i    = (int64_t) row*ncols + tid;
            const int     lane = tid % 32;
            const int64_t blocks_per_row = (GGML_PAD(p.row_len, MATRIX_ROW_PADDING)) / QK8_1;
            const int64_t ib = (i / p.row_len) * blocks_per_row + (i % p.row_len) / QK8_1;

            float amax = fabsf(v);
            float sum = v;
            amax = warp_reduce_max<32>(amax);
            sum  = warp_reduce_sum<32>(sum);

            const float  d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(v / d);

            p.y[ib].qs[lane] = q;
            if (lane == 0) {
                p.y[ib].ds = make_half2(d, sum);
            }
        }
        mk_gdn_lds_release<BLOCK, threads>();
    }
};
