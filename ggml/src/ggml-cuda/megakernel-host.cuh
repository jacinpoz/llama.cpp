#pragma once

#ifdef GGML_MK_HOST_ONLY
#define __host__
#define __device__
#define __forceinline__ inline
#include "ggml.h"
#else
#include "common.cuh"
#endif

#include "megakernel.cuh"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#define MK_N_BLOCKS 48

//
// stream builder
//

// Per-block queues flattened into instrs, queue_begin[n_blocks + 1].
struct mk_stream_host {
    int32_t               n_blocks = 0;
    std::vector<mk_instr> instrs;
    std::vector<int32_t>  queue_begin;
    std::vector<uint8_t>  params;
    std::vector<uint32_t> signals_per_pass;
};

struct mk_segment {
    bool           external    = false;
    int32_t        external_id = -1;
    mk_stream_host stream;
};

struct mk_counter {
    int32_t segment = -1;
    int32_t id      = -1;
};

// target is the per-pass signal count to reach; UINT32_MAX means every signal appended so far.
struct mk_wait {
    mk_counter counter;
    uint32_t   target = UINT32_MAX;
};

struct mk_op_desc {
    uint16_t             opcode        = MK_OP_NOP;
    uint16_t             variant       = 0;
    int32_t              n_tiles       = 0;
    int32_t              tiles_per_step = 1; // tiles one block runs at once (MK_THREADS / op threads)
    uint32_t             params_off    = 0;
    uint32_t             prefetch_off  = UINT32_MAX;
    std::vector<mk_wait> waits;
    mk_counter           signal;             // invalid: a new counter is allocated
};

class mk_stream_builder {
public:
    explicit mk_stream_builder(int32_t n_blocks = MK_N_BLOCKS);

    uint32_t add_params(const void * data, size_t size);
    template <typename T> uint32_t add_params(const T & p) { return add_params(&p, sizeof(T)); }

    mk_counter new_counter();
    mk_wait    wait_all(mk_counter c) const;

    // Every block that gets part of [0, n_tiles) signals the counter once; extra waits become NOPs in front of the op.
    // Waits on earlier segments are dropped: segments run in stream order.
    mk_counter add_op(const mk_op_desc & d);

    void add_external(int32_t external_id);

    std::vector<mk_segment> finish();

private:
    struct segment_build {
        std::vector<std::vector<mk_instr>> queues;
        std::vector<uint32_t>              signals;
    };

    void flush_segment();

    int32_t                    n_blocks;
    int32_t                    next_block = 0;
    std::vector<uint8_t>       params;
    std::vector<mk_segment>    segments;
    segment_build              building;
};

// Returns the first error, or an empty string when every wait is reachable for any interleaving of the blocks.
std::string mk_validate_stream(const mk_stream_host & s);

//
// graph template matcher
//

enum mk_hop : int32_t {
    MK_HOP_EMBED_ROWS,     // external
    MK_HOP_RMSNORM_Q8_1,
    MK_HOP_RMSNORM_F32,
    MK_HOP_QUANTIZE_Q8_1,
    MK_HOP_MMVQ,
    MK_HOP_MMVQ_ADD,
    MK_HOP_MMVQ_GLU,
    MK_HOP_STATE_COPY,
    MK_HOP_GDN_CONV,
    MK_HOP_GDN_GATES,
    MK_HOP_GDN_STEP,
    MK_HOP_GDN_OUT_GATE,
    MK_HOP_ATTN_PREP_Q,
    MK_HOP_ATTN_PREP_K,
    MK_HOP_V_HAD_SET_ROWS,
    MK_HOP_ATTN_PARTIAL,   // external
    MK_HOP_ATTN_COMBINE,
    MK_HOP_OUT_ROWS,
    MK_HOP_CONCAT,
    MK_HOP_FILL_NEG_INF,
    MK_HOP_COUNT,
};

#define MK_MATCH_MAX_T 10

// t[0] is the tensor the op writes; the other slots are named in mk_hop_slot_names.
struct mk_match_op {
    mk_hop              kind;
    int32_t             layer;   // -1: graph input/output part
    const ggml_tensor * t[MK_MATCH_MAX_T];
};

enum mk_layer_kind : int32_t {
    MK_LAYER_INPUT,
    MK_LAYER_GDN,
    MK_LAYER_ATTN,
    MK_LAYER_OUTPUT,
};

struct mk_match_layer {
    mk_layer_kind kind;
    int32_t       op_begin;
    int32_t       op_end;
};

struct mk_match {
    bool                        ok = false;
    std::string                 reason;
    bool                        mtp = false;
    int32_t                     n_tokens = 0;
    int64_t                     n_kv = 0;
    std::vector<mk_match_op>    ops;
    std::vector<mk_match_layer> layers;
};

const char * mk_hop_name(mk_hop kind);
const char * mk_hop_slot_name(mk_hop kind, int slot);
bool         mk_hop_is_external(mk_hop kind);

mk_match    mk_match_graph(ggml_cgraph * cgraph);
std::string mk_match_dump(const mk_match & m);

#ifndef GGML_MK_HOST_ONLY

//
// ggml-cuda entry points
//

bool ggml_cuda_mk_enabled();

typedef bool (*ggml_cuda_mk_run_node_fn)(ggml_backend_cuda_context & ctx, ggml_tensor * node);

// True when the megakernel path computed the whole graph; false leaves the graph untouched for the normal path.
bool ggml_cuda_mk_try_compute(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, ggml_cuda_mk_run_node_fn run_node);
void ggml_cuda_mk_release(ggml_backend_cuda_context * ctx);

#endif // GGML_MK_HOST_ONLY
