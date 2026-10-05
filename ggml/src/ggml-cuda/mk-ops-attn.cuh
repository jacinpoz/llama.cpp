#pragma once

// Gated full-attention decode/verify ops (Qwen3.5/3.8, head dim 256) in the megakernel op form (megakernel.cuh).
// The standalone kernels in rope.cu and fattn-gqa-dec.cu are thin wrappers around these.
// Include before any file-wide fp contract pragma: quantize_f32_q5_0_block (cpy-utils.cuh) takes the includer's
// setting, and the q5_0 cache bits match the reference kernels only with the default (contraction on).

#include "common.cuh"
#include "cpy-utils.cuh"
#include "fattn-gqa-dec.cuh"
#include "megakernel.cuh"
#include "rope-yarn.cuh"

// block_reduce<SUM, threads> with the warp index taken from the sub-tile lane.
template <int threads>
static __device__ __forceinline__ float mk_attn_block_sum(float val, float * s_sum, const int tid) {
    val = warp_reduce_sum(val);
    if (tid % WARP_SIZE == 0) {
        s_sum[tid / WARP_SIZE] = val;
    }
    __syncthreads();
    val = 0.0f;
    if (tid % WARP_SIZE < threads / WARP_SIZE) {
        val = s_sum[tid % WARP_SIZE];
    }
    return warp_reduce_sum(val);
}

// SET_ROWS of block tid of a 256-row into cache row idx[tok], head slot head.
static __device__ __forceinline__ void mk_attn_set_row_q5_0(const float * y, char * cache, const int64_t cache_nb1,
        const void * idx, const bool idx_i64, const int tok, const int head, const int tid) {
    const int64_t row = idx_i64 ? ((const int64_t *) idx)[tok] : (int64_t) ((const int32_t *) idx)[tok];
    block_q5_0 * dst = (block_q5_0 *) (cache + row*cache_nb1) + head*(256/QK5_0) + tid;
    quantize_f32_q5_0_block(&y[tid*QK5_0], dst);
}

// ---------------------------------------------------------------------------------------------------------------------
// MK_OP_ATTN_PREP_Q / MK_OP_ATTN_PREP_K: one (head, token) row of 256: RMS_NORM * weight -> ROPE (mrope) -> 256-point
// Hadamard, then written to out (Q) or quantized to q5_0 into cache row idx[token] (K, the SET_ROWS). Q and K are the
// same op; out or cache selects the path. Each stage is the code of the kernel it replaces (rms_norm_f32<256>'s
// reduction, rope_multi_cos_sin, fwht_cuda<256>'s layout and stage order, quantize_f32_q5_0_block), so the result is
// bit-identical. Tile = tok*n_head + head. Variant: bit 0 = freq_factors present.

struct mk_attn_prep_params {
    const float *   x;            // [256] rows, head stride sx1, token stride sx2 (floats)
    int64_t         sx1, sx2;
    const float *   w;            // norm weight [256]
    float           eps;
    int             n_dims, n_offs;
    const int32_t * pos;
    int             n_tok;        // rope ne02 (pos is [n_tok * 4] for mrope); launch->n_tokens when launch is set
    int             n_head;
    mrope_sections  sections;
    bool            is_imrope;
    float           theta_scale, freq_scale, ext_factor, attn_factor;
    rope_corr_dims  corr_dims;
    const float *   freq_factors;
    float *         out;          // [256, n_head, n_tok] contiguous, or nullptr
    char *          cache;        // q5_0 cache view data, or nullptr
    int64_t         cache_nb1;
    const void *    idx;          // set_rows row index per token
    bool            idx_i64;
    const mk_launch_params * launch;
};

template <int BLOCK>
struct mk_attn_prep {
    static constexpr int threads   = 256;
    static constexpr int lds_bytes = (32 + 256 + 256)*sizeof(float);
    static_assert(BLOCK % threads == 0, "BLOCK must be a multiple of threads");

    template <bool has_ff>
    static __device__ __forceinline__ void run_t(const mk_attn_prep_params & p, const int tile, const bool valid, char * lds) {
#pragma clang fp contract(off)
        const int  n_tok = p.launch ? p.launch->n_tokens : p.n_tok;
        const int  head  = tile % p.n_head;
        const int  tok   = tile / p.n_head;
        const bool live  = valid && tok < n_tok;
        const int  tid   = threadIdx.x % threads;

        float * s_sum = reinterpret_cast<float *>(lds);
        float * y     = s_sum + 32;
        float * r     = y + 256;

        const float xi = live ? p.x[tok*p.sx2 + head*p.sx1 + tid] : 0.0f;
        float tmp = xi * xi;
        tmp = mk_attn_block_sum<threads>(tmp, s_sum, tid);
        const float mean  = tmp / 256;
        const float scale = rsqrtf(mean + p.eps);
        y[tid] = scale * xi * p.w[tid];
        __syncthreads();

        if (live && tid < 128) {
            const int i0 = 2*tid;
            if (i0 < p.n_offs || i0 >= p.n_offs + p.n_dims) {
                r[i0 + 0] = y[i0 + 0];
                r[i0 + 1] = y[i0 + 1];
            } else {
                const int iw = i0 - p.n_offs;
                float cos_theta;
                float sin_theta;
                rope_multi_cos_sin<true, has_ff>(iw, p.pos, tok, n_tok, p.sections, p.is_imrope, p.theta_scale, p.freq_factors,
                                                 p.freq_scale, p.corr_dims, p.ext_factor, p.attn_factor, cos_theta, sin_theta);
                const float x0 = y[i0/2 + p.n_offs/2 + 0];
                const float x1 = y[i0/2 + p.n_offs/2 + p.n_dims/2];
                r[i0/2 + p.n_offs/2 + 0]          = x0*cos_theta - x1*sin_theta;
                r[i0/2 + p.n_offs/2 + p.n_dims/2] = x0*sin_theta + x1*cos_theta;
            }
        }
        __syncthreads();

        // fwht_cuda<256>: one wave, element i*32 + lane in reg[i]
        if (tid < WARP_SIZE) {
            constexpr int el_w = 256 / WARP_SIZE;
            const int lane = tid;
            float reg[el_w];
#pragma unroll
            for (int i = 0; i < el_w; ++i) {
                reg[i] = r[i*WARP_SIZE + lane] * (1.0f / 16.0f);
            }
#pragma unroll
            for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
                for (int j = 0; j < el_w; j++) {
                    const float val  = reg[j];
                    const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, WARP_SIZE);
                    reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
                }
            }
#pragma unroll
            for (int h = WARP_SIZE; h < 256; h *= 2) {
                const int step = h / WARP_SIZE;
#pragma unroll
                for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
                    for (int k = 0; k < step; k++) {
                        const float x = reg[j + k];
                        const float z = reg[j + k + step];
                        reg[j + k]        = x + z;
                        reg[j + k + step] = x - z;
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < el_w; ++i) {
                y[i*WARP_SIZE + lane] = reg[i];
            }
        }
        __syncthreads();

        if (!live) {
            return;
        }
        if (p.out != nullptr) {
            p.out[((int64_t) tok*p.n_head + head)*256 + tid] = y[tid];
        }
        if (p.cache != nullptr && tid < 256/QK5_0) {
            mk_attn_set_row_q5_0(y, p.cache, p.cache_nb1, p.idx, p.idx_i64, tok, head, tid);
        }
    }

    static __device__ __forceinline__ void run(const mk_attn_prep_params & p, const int variant, const int tile, const bool valid, char * lds) {
        if (variant & 1) {
            run_t<true>(p, tile, valid, lds);
        } else {
            run_t<false>(p, tile, valid, lds);
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// MK_OP_V_HAD_SET_ROWS: fwht_cuda<64> over the four 64-chunks of one (head, token) V row of 256, then quantized to
// q5_0 into cache row idx[token] (the SET_ROWS). Bit-identical to the two kernels it replaces. Tile = tok*n_head + head.
// No variants.

struct mk_v_had_set_rows_params {
    const float *  x;             // [256] rows, head stride sx1, token stride sx2 (floats)
    int64_t        sx1, sx2;
    char *         cache;
    int64_t        cache_nb1;
    const void *   idx;
    bool           idx_i64;
    int            n_head;
    int            n_tok;         // launch->n_tokens when launch is set
    const mk_launch_params * launch;
};

template <int BLOCK>
struct mk_v_had_set_rows {
    static constexpr int threads   = 4*WARP_SIZE;
    static constexpr int lds_bytes = 256*sizeof(float);
    static_assert(BLOCK % threads == 0, "BLOCK must be a multiple of threads");

    static __device__ __forceinline__ void run(const mk_v_had_set_rows_params & p, const int variant, const int tile, const bool valid, char * lds) {
#pragma clang fp contract(off)
        GGML_UNUSED(variant);
        const int  n_tok = p.launch ? p.launch->n_tokens : p.n_tok;
        const int  head  = tile % p.n_head;
        const int  tok   = tile / p.n_head;
        const bool live  = valid && tok < n_tok;
        const int  tid   = threadIdx.x % threads;
        const int  lane  = tid % WARP_SIZE;
        const int  w     = tid / WARP_SIZE;

        float * y = reinterpret_cast<float *>(lds);

        constexpr int el_w = 64 / WARP_SIZE;
        const float * src = p.x + tok*p.sx2 + head*p.sx1 + w*64;
        float reg[el_w];
#pragma unroll
        for (int i = 0; i < el_w; ++i) {
            reg[i] = live ? src[i*WARP_SIZE + lane] * 0.125f : 0.0f;
        }
#pragma unroll
        for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
            for (int j = 0; j < el_w; j++) {
                const float val  = reg[j];
                const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, WARP_SIZE);
                reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
#pragma unroll
        for (int h = WARP_SIZE; h < 64; h *= 2) {
            const int step = h / WARP_SIZE;
#pragma unroll
            for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
                for (int k = 0; k < step; k++) {
                    const float a = reg[j + k];
                    const float b = reg[j + k + step];
                    reg[j + k]        = a + b;
                    reg[j + k + step] = a - b;
                }
            }
        }
#pragma unroll
        for (int i = 0; i < el_w; ++i) {
            y[w*64 + i*WARP_SIZE + lane] = reg[i];
        }
        __syncthreads();
        if (live && tid < 256/QK5_0) {
            mk_attn_set_row_q5_0(y, p.cache, p.cache_nb1, p.idx, p.idx_i64, tok, head, tid);
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// MK_OP_ATTN_PARTIAL / MK_OP_ATTN_COMBINE: GQA-aware decode/verify attention for head dim 256, GQA 6 or 8.
//
// Partial: one wave per (KV chunk, KV head, query token), 32 keys per step. For the scores each lane takes one key and
// computes all G heads against Q in LDS (no cross-lane reductions); for V each lane owns 8 of the 256 dims of every
// key. Every K/V row is loaded and decoded once for the whole group. Softmax is online per 32-key step. The chunk
// length depends only on n_kv, and each token is processed independently, so a token gets the same result whether it
// is decoded alone or verified in a speculative batch.
//
// The chunk count follows n_kv, so a megakernel instruction covers tile_chunks >= the live count (mk_attn_tile_chunks)
// and the tiles past it run with valid == false. The partial buffers are indexed with the live count; the combine
// derives it from n_kv the same way, so the two need not share a launch.

static constexpr int mk_attn_d             = 256;
static constexpr int mk_attn_max_chunks    = 512;
static constexpr int mk_attn_min_chunk     = 64;
static constexpr int mk_attn_combine_waves = 8;

struct mk_attn_chunks {
    int len;
    int n;
};

static __host__ __device__ __forceinline__ mk_attn_chunks mk_attn_chunking(const int n_kv) {
    const int per = GGML_PAD((n_kv + mk_attn_max_chunks - 1)/mk_attn_max_chunks, WARP_SIZE);
    const int len = per > mk_attn_min_chunk ? per : mk_attn_min_chunk;
    return {len, (n_kv + len - 1)/len};
}

// Upper bound of mk_attn_chunking(n_kv).n over n_kv <= n_ctx.
static inline int mk_attn_tile_chunks(const int n_ctx) {
    return std::min(mk_attn_max_chunks, (n_ctx + mk_attn_min_chunk - 1)/mk_attn_min_chunk);
}

static constexpr int mk_attn_partial_variant(const int G, const ggml_type T) {
    return (G == 8 ? 1 : 0) | (T == GGML_TYPE_Q8_0 ? 2 : T == GGML_TYPE_Q5_0 ? 4 : 0);
}

static constexpr int mk_attn_partial_lds_bytes(const int G) {
    return (G*mk_attn_d + WARP_SIZE*G)*sizeof(float);
}

// Tile = (t*n_head_kv + h)*tile_chunks + c for token q0 + t, KV head h, chunk c.
// With launch set, n_kv and n_tokens come from it and the mask row stride is n_kv halves (the dense [n_kv, T] mask).
struct mk_attn_partial_params {
    const char * Q;
    const char * K;
    const char * V;
    const char * mask;
    float *      part_acc;        // [nq][n_head_kv][n_chunks][G][256]
    float2 *     part_ms;         // [nq][n_head_kv][n_chunks][G]
    float        scale;
    int          n_kv;
    int          n_head_kv;
    int          q0;
    int          nq;
    int          tile_chunks;
    int64_t      nbq1, nbq2, nbk1, nbk2, nbv1, nbv2, nbm1;
    const mk_launch_params * launch;
};

template <ggml_type T>
static __device__ __forceinline__ void gqa_dec_load_slice(const char * row, const int lane, float v[8]) {
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
static __device__ __forceinline__ void gqa_dec_load_half(const char * row, const int hb, float v[16]) {
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

// A sub-tile is one wave, so its LDS hand-offs use __syncwarp() and run() has no block barrier.
template <int BLOCK>
struct mk_attn_partial {
    static constexpr int threads   = WARP_SIZE;
    static constexpr int lds_bytes = mk_attn_partial_lds_bytes(8);
    static_assert(BLOCK % threads == 0, "BLOCK must be a multiple of threads");

    template <int G, ggml_type T>
    static __device__ __forceinline__ void run_t(const mk_attn_partial_params & p, const int tile, const bool valid, char * lds) {
#pragma clang fp contract(fast)
        const int     n_kv = p.launch ? p.launch->n_kv : p.n_kv;
        const int     nq   = p.launch ? min(p.nq, p.launch->n_tokens - p.q0) : p.nq;
        const int64_t nbm1 = p.launch ? (int64_t) n_kv*sizeof(half) : p.nbm1;
        const mk_attn_chunks ch = mk_attn_chunking(n_kv);

        const int c    = tile % p.tile_chunks;
        const int h    = tile / p.tile_chunks % p.n_head_kv;
        const int t    = tile / (p.tile_chunks*p.n_head_kv);
        const int lane = threadIdx.x % threads;
        const int tok  = p.q0 + t;
        if (!valid || c >= ch.n || t >= nq) {
            return;
        }

        float (*q_lds)[mk_attn_d] = reinterpret_cast<float (*)[mk_attn_d]>(lds);
        float (*p_lds)[G]         = reinterpret_cast<float (*)[G]>(lds + G*mk_attn_d*sizeof(float));

#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float * qp = reinterpret_cast<const float *>(p.Q + tok*p.nbq1 + (int64_t) (h*G + g)*p.nbq2);
#pragma unroll
            for (int i = lane; i < mk_attn_d; i += WARP_SIZE) {
                q_lds[g][i] = qp[i]*p.scale;
            }
        }
        __syncwarp();

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

        const half * mrow  = reinterpret_cast<const half *>(p.mask + tok*nbm1);
        const char * Kh    = p.K + h*p.nbk2;
        const char * Vh    = p.V + h*p.nbv2;
        const int    k_beg = c*ch.len;
        const int    k_end = min(n_kv, k_beg + ch.len);

        for (int k0 = k_beg; k0 < k_end; k0 += WARP_SIZE) {
            const int nk = min(WARP_SIZE, k_end - k0);

            // scores: lane j computes all G scores of key k0 + j
            float my_s[G];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                my_s[g] = 0.0f;
            }
            const bool live = lane < nk;
            const char * krow = Kh + (int64_t) (live ? k0 + lane : k0)*p.nbk1;
            for (int hb = 0; hb < mk_attn_d/16; ++hb) {
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
            __syncwarp();

            // V: lane owns dims 8*lane..8*lane+7 of every key
            for (int j = 0; j < nk; ++j) {
                float vv[8];
                gqa_dec_load_slice<T>(Vh + (int64_t) (k0 + j)*p.nbv1, lane, vv);
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
            __syncwarp();
        }

        const int64_t base = ((int64_t) (t*p.n_head_kv + h)*ch.n + c)*G;
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float * pa = p.part_acc + (base + g)*mk_attn_d + lane*8;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                pa[i] = acc[g][i];
            }
            if (lane == 0) {
                p.part_ms[base + g] = make_float2(m[g], s[g]);
            }
        }
    }

    static __device__ __forceinline__ void run(const mk_attn_partial_params & p, const int variant, const int tile, const bool valid, char * lds) {
        switch (variant) {
            case mk_attn_partial_variant(6, GGML_TYPE_F16):  run_t<6, GGML_TYPE_F16> (p, tile, valid, lds); break;
            case mk_attn_partial_variant(6, GGML_TYPE_Q8_0): run_t<6, GGML_TYPE_Q8_0>(p, tile, valid, lds); break;
            case mk_attn_partial_variant(6, GGML_TYPE_Q5_0): run_t<6, GGML_TYPE_Q5_0>(p, tile, valid, lds); break;
            case mk_attn_partial_variant(8, GGML_TYPE_F16):  run_t<8, GGML_TYPE_F16> (p, tile, valid, lds); break;
            case mk_attn_partial_variant(8, GGML_TYPE_Q8_0): run_t<8, GGML_TYPE_Q8_0>(p, tile, valid, lds); break;
            case mk_attn_partial_variant(8, GGML_TYPE_Q5_0): run_t<8, GGML_TYPE_Q5_0>(p, tile, valid, lds); break;
            default: break;
        }
    }
};

// Combine: one sub-tile per (query head, token). Wave w merges chunks w, w + mk_attn_combine_waves, ... over all 256
// dims (8 per lane); the waves are then merged in LDS in a fixed order. With ep.gate set, the gated-attention tail is
// applied: out = fwht64(o) * sigmoid(gate). Tile = t*n_head + hq. Variant: bit 0 = GQA 8 (else 6).
static constexpr int mk_attn_combine_variant(const int G) {
    return G == 8 ? 1 : 0;
}

struct mk_attn_combine_params {
    const float *    part_acc;
    const float2 *   part_ms;
    float *          dst;
    int              n_kv;        // launch->n_kv when launch is set
    int              n_head_kv;
    int              n_head;
    int              q0;
    int              nq;          // capped by launch->n_tokens - q0 when launch is set
    gqa_dec_epilogue ep;
    const mk_launch_params * launch;
};

template <int BLOCK>
struct mk_attn_combine {
    static constexpr int threads   = mk_attn_combine_waves*WARP_SIZE;
    static constexpr int lds_bytes = (2*mk_attn_combine_waves + mk_attn_combine_waves*mk_attn_d)*sizeof(float);
    static_assert(BLOCK % threads == 0, "BLOCK must be a multiple of threads");

    template <int G>
    static __device__ __forceinline__ void run_t(const mk_attn_combine_params & p, const int tile, const bool valid, char * lds) {
#pragma clang fp contract(fast)
        const int  n_kv = p.launch ? p.launch->n_kv : p.n_kv;
        const int  nq   = p.launch ? min(p.nq, p.launch->n_tokens - p.q0) : p.nq;
        const int  hq   = tile % p.n_head;
        const int  t    = tile / p.n_head;
        const int  h    = hq / G;
        const int  g    = hq % G;
        const int  lane = threadIdx.x % WARP_SIZE;
        const int  w    = threadIdx.x % threads / WARP_SIZE;
        const bool live = valid && t < nq;
        const int  n_chunks = live ? mk_attn_chunking(n_kv).n : 0;

        float * w_m = reinterpret_cast<float *>(lds);
        float * w_s = w_m + mk_attn_combine_waves;
        float (*w_acc)[mk_attn_d] = reinterpret_cast<float (*)[mk_attn_d]>(w_s + mk_attn_combine_waves);

        const int64_t base = (int64_t) (t*p.n_head_kv + h)*n_chunks*G + g;
        float M = -INFINITY;
        for (int c = w; c < n_chunks; c += mk_attn_combine_waves) {
            M = fmaxf(M, p.part_ms[base + (int64_t) c*G].x);
        }
        float S = 0.0f;
        float A[8] = {};
        for (int c = w; c < n_chunks; c += mk_attn_combine_waves) {
            const float2 ms = p.part_ms[base + (int64_t) c*G];
            if (ms.x == -INFINITY) {
                continue;
            }
            const float   wgt = expf(ms.x - M);
            const float * pa  = p.part_acc + (base + (int64_t) c*G)*mk_attn_d + lane*8;
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

        if (w == 0 && live) {
            float MM = -INFINITY;
#pragma unroll
            for (int k = 0; k < mk_attn_combine_waves; ++k) {
                MM = fmaxf(MM, w_m[k]);
            }
            float SS = 0.0f;
            float AA[8] = {};
#pragma unroll
            for (int k = 0; k < mk_attn_combine_waves; ++k) {
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
            float * out = p.dst + ((int64_t) (p.q0 + t)*p.n_head + hq)*mk_attn_d + lane*8;
            float o[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                o[i] = AA[i]/SS;
            }
            if (p.ep.gate != nullptr) {
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
                const float * gp = reinterpret_cast<const float *>(p.ep.gate + (int64_t) (p.q0 + t)*p.ep.gate_nb2 + (int64_t) hq*p.ep.gate_nb1) + lane*8;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    o[i] = o[i] * (1.0f / (1.0f + expf(-gp[i])));
                }
                out = p.ep.out + ((int64_t) (p.q0 + t)*p.n_head + hq)*mk_attn_d + lane*8;
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                out[i] = o[i];
            }
        }
    }

    static __device__ __forceinline__ void run(const mk_attn_combine_params & p, const int variant, const int tile, const bool valid, char * lds) {
        if (variant & 1) {
            run_t<8>(p, tile, valid, lds);
        } else {
            run_t<6>(p, tile, valid, lds);
        }
    }
};
