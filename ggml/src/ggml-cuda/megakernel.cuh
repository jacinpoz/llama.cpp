#pragma once

// One block of MK_THREADS per WGP. 1024 threads give 8 waves per SIMD, so VGPRs must stay <= MK_MAX_VGPRS.

#include "common.cuh"

#include <cstdint>

#define MK_THREADS     1024
#define MK_MAX_VGPRS   192
#define MK_LDS_BYTES   (64*1024)
#define MK_PARAM_ALIGN 16

// LDS left to ops; the persistent kernel keeps the rest.
#define MK_OP_LDS_BYTES (MK_LDS_BYTES - 16)

enum mk_opcode : uint16_t {
    MK_OP_NOP = 0,
    MK_OP_EXIT,

    // WS-A: mk-ops-ffn.cuh
    MK_OP_RMSNORM_Q8_1,        // optional residual add; whole row per tile
    MK_OP_MMVQ,                // IQ4_XS / Q6_K / Q8_0, optional GLU and add; rows per tile
    MK_OP_LM_HEAD_ARGMAX,      // mmvq rows + per-tile argmax partials
    MK_OP_ARGMAX_COMBINE,

    // WS-B: mk-ops-gdn.cuh
    MK_OP_GDN_GATES,
    MK_OP_GDN_CONV,            // in-place conv state, silu, q/k l2-norm
    MK_OP_GDN_STEP,            // in-place recurrent state update
    MK_OP_GDN_OUT_GATE,        // norm, silu(z) gate, q8_1

    // WS-C: mk-ops-attn.cuh
    MK_OP_ATTN_PREP_Q,
    MK_OP_ATTN_PREP_K,
    MK_OP_V_HAD_SET_ROWS,
    MK_OP_ATTN_PARTIAL,
    MK_OP_ATTN_COMBINE,        // + gated tail

    MK_OP_COUNT,

    // WS-D: mk-ops-test.cuh, used by test-megakernel
    MK_OP_TEST_ADD = 0xF000,
    MK_OP_TEST_SPIN,
};

struct mk_instr {
    uint16_t opcode;
    uint16_t variant;          // op-specific template selector (type, ncols_dst = T, flags); decoded by the op
    int32_t  tile_begin;
    int32_t  tile_end;
    int32_t  wait_counter;     // -1 = no wait
    uint32_t wait_target;      // per-pass count; see mk_counter_target
    int32_t  signal_counter;   // -1 = no signal
    uint32_t params_off;       // byte offset into the param blob, MK_PARAM_ALIGN-aligned
    uint32_t prefetch_off;     // byte offset of the next weight tile to prefetch while waiting, or UINT32_MAX
};
static_assert(sizeof(mk_instr) == 32, "mk_instr must stay 32 bytes");

// What prefetch_off points at in the param blob (MK_PARAM_ALIGN-aligned). Read one dword per 128-byte line during the wait.
struct mk_prefetch_desc {
    const void * ptr;
    uint64_t     bytes;
};

// Counters are not reset between passes: pass e waits for e*S_c + rel. The compare is wrap-safe.
static __host__ __device__ __forceinline__ uint32_t mk_counter_target(uint32_t epoch, uint32_t signals_per_pass, uint32_t rel) {
    return epoch*signals_per_pass + rel;
}
static __host__ __device__ __forceinline__ bool mk_counter_reached(uint32_t value, uint32_t target) {
    return (int32_t) (value - target) >= 0;
}

struct mk_launch_params {
    uint32_t epoch;
    int32_t  n_kv;
    int32_t  n_tokens;         // T = 1..8
    int32_t  kv_head;          // first KV cell written this pass
};

struct mk_stream_desc {
    const mk_instr       * instrs;          // all queues, concatenated
    const int32_t        * queue_begin;     // [n_blocks + 1]
    const uint8_t        * params;          // param blob
    const uint32_t       * signals_per_pass;// [n_counters]
    uint32_t             * counters;        // [n_counters], device memory, reset only after an error
    int32_t              * error;           // 0 = ok; set by the watchdog or an op
    const mk_launch_params * launch;
    int32_t                n_blocks;
    int32_t                n_counters;
    uint64_t               watchdog_cycles; // wall_clock64() ticks a wait may spin before aborting
};

enum mk_error : int32_t {
    MK_ERR_NONE = 0,
    MK_ERR_WATCHDOG = 1,       // a wait never reached its target
    MK_ERR_BAD_OPCODE = 2,
};

// Op contract (mk-ops-*.cuh): POD mk_<op>_params; template <int BLOCK> struct mk_<op> with
// threads (sub-tile width, divides MK_THREADS), lds_bytes (per sub-tile) and
// static __device__ void run(const mk_<op>_params & p, int variant, int tile, bool valid, char * lds).
// The __global__ wrapper runs one sub-tile per block; the megakernel runs MK_THREADS/threads.
// Every thread hits the same __syncthreads() sequence whatever its tile or valid flag.
// Lane = threadIdx.x % threads; no blockIdx or gridDim in run(). Results must be bit-identical.

template <typename F> struct mk_run_signature;
template <typename P> struct mk_run_signature<void (*)(const P &, int, int, bool, char *)> {
    using params = P;
};
template <typename Op> using mk_op_params = typename mk_run_signature<decltype(&Op::run)>::params;

// Host API (megakernel.cu). Launches are async and never sync, so segments can be queued back to back.
// A set *error makes later waits abort until take_error clears it; then reset the counters and restart at epoch 0.
// A captured launch replays one epoch.
void    ggml_cuda_mk_launch(const mk_stream_desc & desc, cudaStream_t stream);
int32_t ggml_cuda_mk_take_error(int32_t * error, cudaStream_t stream); // syncs the stream, returns and clears *error
void    ggml_cuda_mk_reset_counters(const mk_stream_desc & desc, cudaStream_t stream);
