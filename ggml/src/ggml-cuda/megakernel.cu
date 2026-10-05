#include "megakernel.cuh"

#if defined(GGML_USE_HIP)

#include "mk-ops-test.cuh"

#if __has_include("mk-ops-ffn.cuh")
#include "mk-ops-ffn.cuh"
#define MK_HAVE_OPS_FFN 1
#endif
#if __has_include("mk-ops-gdn.cuh")
#include "mk-ops-gdn.cuh"
#define MK_HAVE_OPS_GDN 1
#endif
#if __has_include("mk-ops-attn.cuh")
#include "mk-ops-attn.cuh"
#define MK_HAVE_OPS_ATTN 1
#endif

#include <array>

// 0 disables the weight prefetch during waits.
#ifndef GGML_CUDA_MK_PREFETCH
#define GGML_CUDA_MK_PREFETCH 1
#endif

template <typename T>
static __device__ __forceinline__ T mk_uniform(T v) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "mk_uniform takes 32- or 64-bit values");
    if constexpr (sizeof(T) == 4) {
        return __builtin_bit_cast(T, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int32_t, v)));
    } else {
        const uint64_t u = __builtin_bit_cast(uint64_t, v);
        const uint32_t lo = __builtin_amdgcn_readfirstlane((uint32_t) u);
        const uint32_t hi = __builtin_amdgcn_readfirstlane((uint32_t) (u >> 32));
        return __builtin_bit_cast(T, (uint64_t) hi << 32 | lo);
    }
}

// Call arguments arrive in VGPRs, so the uniform values are re-scalarized here; otherwise every param
// read in the op's inner loops becomes a per-lane vector load.
template <typename Op>
static __device__ __noinline__ void mk_exec(const mk_instr & in_ref, const uint8_t * __restrict__ params_ref, char * lds) {
    using P = mk_op_params<Op>;
    constexpr int threads = Op::threads;
    constexpr int nsub    = MK_THREADS / threads;
    static_assert(MK_THREADS % threads == 0, "op threads must divide MK_THREADS");
    static_assert(nsub*Op::lds_bytes <= MK_OP_LDS_BYTES, "op LDS does not fit the megakernel");

    const int tile_begin = mk_uniform(in_ref.tile_begin);
    const int tile_end   = mk_uniform(in_ref.tile_end);
    const int variant    = mk_uniform((int) in_ref.variant);
    const P * pp = mk_uniform(reinterpret_cast<const P *>(mk_uniform(params_ref) + mk_uniform(in_ref.params_off)));
    const P   p  = *pp;
    const int sub = threadIdx.x / threads;
    char * lds_sub = lds + sub*Op::lds_bytes;

    for (int t0 = tile_begin; t0 < tile_end; t0 += nsub) {
        // Separates the previous step's LDS reads from this step's writes.
        if (Op::lds_bytes > 0 && t0 != tile_begin) {
            __syncthreads();
        }
        const int  tile  = t0 + sub;
        const bool valid = tile < tile_end;
        Op::run(p, variant, valid ? tile : tile_end - 1, valid, lds_sub);
    }
}

// Returns false when the block must exit. The opcode is block-uniform, so the return is too.
static __device__ __forceinline__ bool mk_dispatch(const mk_stream_desc & d, const mk_instr & in, char * lds) {
#define MK_CASE(opc, op) case opc: mk_exec<op<MK_THREADS>>(in, d.params, lds); return true
    switch (in.opcode) {
        case MK_OP_NOP:
            return true;
        case MK_OP_EXIT:
            return false;
        MK_CASE(MK_OP_TEST_ADD,  mk_test_add);
        MK_CASE(MK_OP_TEST_SPIN, mk_test_spin);
#ifdef MK_HAVE_OPS_FFN
#define MK_CASE_DISPATCH(opc, dispatch) case opc: \
            dispatch<MK_THREADS>(in.variant, [&](auto op) { mk_exec<decltype(op)>(in, d.params, lds); }); return true
        MK_CASE_DISPATCH(MK_OP_RMSNORM_Q8_1, mk_rmsnorm_q8_1_dispatch);
        MK_CASE_DISPATCH(MK_OP_RMSNORM_F32,  mk_rmsnorm_f32_dispatch);
        MK_CASE_DISPATCH(MK_OP_MMVQ,         mk_mmvq_dispatch);
#undef MK_CASE_DISPATCH
        MK_CASE(MK_OP_QUANTIZE_Q8_1,  mk_quantize_q8_1);
        MK_CASE(MK_OP_LM_HEAD_ARGMAX, mk_argmax_partial);
        MK_CASE(MK_OP_ARGMAX_COMBINE, mk_argmax_combine);
#endif
#ifdef MK_HAVE_OPS_GDN
        MK_CASE(MK_OP_GDN_GATES, mk_gdn_gates);
        MK_CASE(MK_OP_GDN_CONV,  mk_gdn_conv);
        MK_CASE(MK_OP_GDN_STEP,  mk_gdn_step);
        case MK_OP_GDN_OUT_GATE: mk_exec<mk_gdn_out_gate<MK_THREADS, 128>>(in, d.params, lds); return true;
#endif
#ifdef MK_HAVE_OPS_ATTN
        // ATTN_PARTIAL runs as its own kernel between segments: 32 sub-tiles need more than 64 KiB of LDS.
        MK_CASE(MK_OP_ATTN_PREP_Q,    mk_attn_prep);
        MK_CASE(MK_OP_ATTN_PREP_K,    mk_attn_prep);
        MK_CASE(MK_OP_V_HAD_SET_ROWS, mk_v_had_set_rows);
        MK_CASE(MK_OP_ATTN_COMBINE,   mk_attn_combine);
#endif
        default:
            if (threadIdx.x == 0) {
                int32_t expected = MK_ERR_NONE;
                __hip_atomic_compare_exchange_strong(d.error, &expected, MK_ERR_BAD_OPCODE,
                    __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
            return false;
    }
#undef MK_CASE
}

static __device__ __forceinline__ bool mk_poll(const uint32_t * counter, uint32_t target) {
    return mk_counter_reached(__hip_atomic_load(counter, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT), target);
}

// Thread 0 only. Returns false on watchdog timeout or when another block has set *error.
static __device__ bool mk_wait(const mk_stream_desc & d, int counter, uint32_t target) {
    const uint32_t * c = d.counters + counter;
    if (!mk_poll(c, target)) {
        const uint64_t t0 = wall_clock64();
        while (true) {
            __builtin_amdgcn_s_sleep(1);
            if (mk_poll(c, target)) {
                break;
            }
            if (__hip_atomic_load(d.error, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != MK_ERR_NONE) {
                return false;
            }
            if (wall_clock64() - t0 > d.watchdog_cycles) {
                int32_t expected = MK_ERR_NONE;
                __hip_atomic_compare_exchange_strong(d.error, &expected, MK_ERR_WATCHDOG,
                    __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                return false;
            }
        }
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    return true;
}

// Stops as soon as thread 0 reports the wait is over, so the prefetch only fills idle time.
static __device__ __forceinline__ void mk_prefetch(const mk_prefetch_desc & pf, int t, int nt, const volatile int * wait_done) {
    const uint8_t * p = (const uint8_t *) pf.ptr;
    uint32_t sink = 0;
    for (uint64_t off = (uint64_t) t*128; off < pf.bytes && !*wait_done; off += (uint64_t) nt*128) {
        sink ^= *(const volatile uint32_t *) (p + off);
    }
    asm volatile("" :: "v"(sink));
}

__global__ void __launch_bounds__(MK_THREADS, 1) mk_persistent(const mk_stream_desc d) {
    __shared__ __align__(16) char lds[MK_OP_LDS_BYTES];
    __shared__ int abort_block;
    __shared__ int wait_done;
    static_assert(sizeof(lds) + sizeof(abort_block) + sizeof(wait_done) <= MK_LDS_BYTES, "megakernel LDS over budget");

    const int      q_begin = d.queue_begin[blockIdx.x];
    const int      q_end   = d.queue_begin[blockIdx.x + 1];
    const uint32_t epoch   = d.launch->epoch;

    if (threadIdx.x == 0) {
        wait_done = 0;
    }
    __syncthreads();
    mk_instr next = q_begin < q_end ? d.instrs[q_begin] : mk_instr{};
    for (int q = q_begin; q < q_end; ++q) {
        const mk_instr in = next;
        // Issued here so its latency overlaps this instruction's wait and op.
        if (q + 1 < q_end) {
            next = d.instrs[q + 1];
        }

        uint64_t * tr = d.trace ? d.trace + 3*(size_t) q : nullptr;
        if (tr && threadIdx.x == 0) {
            tr[0] = wall_clock64();
        }
        if (in.wait_counter >= 0) {
            if (threadIdx.x == 0) {
                const uint32_t target = mk_counter_target(epoch, d.signals_per_pass[in.wait_counter], in.wait_target);
                abort_block = !mk_wait(d, in.wait_counter, target);
                __hip_atomic_store(&wait_done, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
            }
#if GGML_CUDA_MK_PREFETCH
            else if (in.prefetch_off != UINT32_MAX && threadIdx.x >= WARP_SIZE) {
                mk_prefetch(*(const mk_prefetch_desc *) (d.params + in.prefetch_off), threadIdx.x - WARP_SIZE, MK_THREADS - WARP_SIZE, &wait_done);
            }
#endif
            __syncthreads();
            if (abort_block) {
                return;
            }
        }

        if (tr && threadIdx.x == 0) {
            tr[1] = wall_clock64();
        }
        if (!mk_dispatch(d, in, lds)) {
            return;
        }

        // The barrier also keeps this op's LDS reads before the next op's LDS writes.
        if (in.signal_counter >= 0) {
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        }
        if (threadIdx.x == 0) {
            wait_done = 0;
        }
        __syncthreads();
        if (in.signal_counter >= 0 && threadIdx.x == 0) {
            __hip_atomic_fetch_add(d.counters + in.signal_counter, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
        if (tr && threadIdx.x == 0) {
            tr[2] = wall_clock64();
        }
    }
}

void ggml_cuda_mk_launch(const mk_stream_desc & desc, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    static std::array<int, GGML_CUDA_MAX_DEVICES> max_blocks = {};
    if (max_blocks[id] == 0) {
        GGML_ASSERT(ggml_cuda_info().devices[id].supports_cooperative_launch);
        int occupancy = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, mk_persistent, MK_THREADS, 0));
        max_blocks[id] = occupancy*ggml_cuda_info().devices[id].nsm;
    }
    GGML_ASSERT(desc.n_blocks > 0 && desc.n_blocks <= max_blocks[id]);

    mk_stream_desc d = desc;
    // A cooperative launch costs ~47 us on ROCm 10.x. With one block per WGP and the occupancy check above
    // every block is resident anyway; GGML_CUDA_MK_COOPERATIVE=1 restores the guaranteed form.
    static const bool cooperative = getenv("GGML_CUDA_MK_COOPERATIVE") != nullptr;
    if (!cooperative) {
        mk_persistent<<<desc.n_blocks, MK_THREADS, 0, stream>>>(d);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    void * args[] = { &d };
    CUDA_CHECK(cudaLaunchCooperativeKernel((const void *) mk_persistent, dim3(desc.n_blocks), dim3(MK_THREADS), args, 0, stream));
}

int32_t ggml_cuda_mk_take_error(int32_t * error, cudaStream_t stream) {
    int32_t err = MK_ERR_NONE;
    CUDA_CHECK(cudaMemcpyAsync(&err, error, sizeof(err), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (err != MK_ERR_NONE) {
        CUDA_CHECK(cudaMemsetAsync(error, 0, sizeof(*error), stream));
    }
    return err;
}

void ggml_cuda_mk_reset_counters(const mk_stream_desc & desc, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(desc.counters, 0, desc.n_counters*sizeof(uint32_t), stream));
}

void ggml_cuda_mk_set_recording(std::vector<mk_recorded_op> * rec) {
    g_mk_recording = rec;
}

#endif // defined(GGML_USE_HIP)
