#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// same op with the beta UNARY(sigmoid) fused in: beta_presig is the sigmoid's input (decode/verify band only)
void ggml_cuda_op_gated_delta_net_beta_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * beta_presig,
                                               const ggml_cuda_gated_delta_net_fused_cache * cache);
int ggml_cuda_gdn_beta_sigmoid_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
int ggml_cuda_gdn_cache_cpy_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

int ggml_cuda_try_gdn_cache_fusion(const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy);
// The recurrent state read in place from a cache row instead of a gathered copy (ggml-cuda.cu elides the
// GET_ROWS for single-sequence batches that take the sequential kernel). Registered per graph evaluation.
struct ggml_cuda_gdn_state_src {
    const float *   base;       // cache rows
    const int32_t * ids;        // device row index per sequence
    int64_t         row_stride; // in floats
    float *         state_pre;  // nullptr: none. Receives the input state (the pre-batch rollback slot).
};

void ggml_cuda_gdn_clear_state_srcs();

// GDN lazy commit: fold a replay GDN into the GDN after it (see gated_delta_net.cu); -1 when node i was folded.
int ggml_cuda_gdn_prefix_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
void ggml_cuda_gdn_set_state_src(const ggml_tensor * gdn, const ggml_cuda_gdn_state_src & src);
bool ggml_cuda_gdn_get_state_src(const ggml_tensor * gdn, ggml_cuda_gdn_state_src & src);
