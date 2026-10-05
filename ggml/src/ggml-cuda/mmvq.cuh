#pragma once

#include "common.cuh"

#include <unordered_map>
#include <unordered_set>

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.
// Dedicated MUL_MAT_ID MoE kernel band (mul_mat_vec_q_moe).  That kernel is one warp per token and
// its per-token reduction does not depend on the column count, so it covers the whole speculative
// verify range (--spec-draft-n-max <= 15 -> n_tokens <= 16) on a single accumulation order.  The
// dense MMVQ band above stays at 8; only the MUL_MAT_ID (routed expert) path uses this one.
#define MMVQ_MOE_MAX_BATCH_SIZE 16

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

void ggml_cuda_op_ssm_gate_beta(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0_alpha, const ggml_tensor * src0_beta, const ggml_tensor * src1,
    const ggml_tensor * dt, const ggml_tensor * ssm_a,
    ggml_tensor * dst_gate, ggml_tensor * dst_beta);

void ggml_cuda_op_ssm_conv_l2_gatebeta(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * conv_input, const ggml_tensor * conv_w,
    const ggml_tensor * src0_alpha, const ggml_tensor * src0_beta, const ggml_tensor * src1,
    const ggml_tensor * dt, const ggml_tensor * ssm_a,
    ggml_tensor * q_norm, ggml_tensor * k_norm, ggml_tensor * v_raw,
    ggml_tensor * dst_gate, ggml_tensor * dst_beta,
    const int head_k_dim, const int n_qk_heads,
    const int head_v_dim, const int n_v_heads, const float eps);

void ggml_cuda_op_shexp_down_gate(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * w_down, const ggml_tensor * swiglu,
    const ggml_tensor * w_gate, const ggml_tensor * x_gate,
    const ggml_tensor * moe_out, const ggml_tensor * ffn_residual,
    ggml_tensor * dst);

// qwen4exp IQ4_NL/Q8_0 routed down projection fused with the 10-expert weighted sum (RDNA3.5,
// single-token): ggml_cuda_mul_mat_id_weighted_rdna3_5_ok returns whether the graph pattern's
// experts/weights/dst match the specialized kernel's shape fingerprint.
bool ggml_cuda_mul_mat_id_weighted_rdna3_5_ok(const ggml_tensor * experts, const ggml_tensor * weights, const ggml_tensor * dst);
void ggml_cuda_mul_mat_id_weighted_rdna3_5(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * experts, const ggml_tensor * weights, ggml_tensor * dst);

bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor, const bool verify_band = false, const int64_t max_ncols = 1);

int ggml_cuda_mul_mat_id_weighted_down_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

int ggml_cuda_ssm_conv_in_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
// SSM conv-input fusion target: MUL_MAT i (the qkv projection) feeding the last row of a dim-0 CONCAT j with a
// GET_ROWS of conv states (s0) as the first source. See ggml_cuda_ssm_conv_in_try_fuse.
bool ggml_cuda_match_conv_input_fusion(const ggml_cgraph * cgraph, int i, int & j_out, const ggml_tensor *& s0_out);

// Decode conv state read in place (ggml-cuda.cu elides the conv-states GET_ROWS that the conv-input fusion would
// have read, and the state CPY the fusion then writes itself). Registered per graph evaluation.
struct ggml_cuda_conv_state_src {
    const float *   base;
    const int32_t * ids;
    int64_t         row_stride;
};
extern thread_local std::unordered_map<const ggml_tensor *, ggml_cuda_conv_state_src> g_conv_state_srcs;
extern thread_local std::unordered_set<const ggml_tensor *>                            g_conv_elided_cpys;

// GGML_CUDA_GDN_CONV_FUSED=1: match the GDN conv layers of this graph (once per graph uid) before its nodes run.
void ggml_cuda_gdn_conv_prepare(ggml_cgraph * cgraph, bool single_stream);
bool ggml_cuda_ssm_prescan_match_at(const struct ggml_cgraph * cgraph, int i);
int ggml_cuda_ssm_prescan_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

int ggml_cuda_mul_mat_dual_output_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
int ggml_cuda_mul_mat_row_concat_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
int ggml_cuda_moe_down_fold_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

int ggml_cuda_gate_beta_verify_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
int ggml_cuda_shexp_down_gate_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

bool ggml_cuda_match_shared_expert(const ggml_cgraph * graph, int routed_idx, int shared_idx);
int  ggml_cuda_match_shared_expert_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
void ggml_cuda_glu_q8_1_mark(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

// Path F: run independent matvecs over the same input as one launch. While a capture is set, the ksplit
// launcher stores its params instead of launching; ggml_cuda_mmvq_launch_pair then runs two captures together.
struct ggml_cuda_mmvq_capture {
    bool     valid = false;
    int      variant = 0;
    uint32_t grid_x = 0, grid_y = 0, grid_z = 0;
    uint32_t threads = 0;
    alignas(16) unsigned char params[512];
};

void ggml_cuda_mmvq_set_capture(ggml_cuda_mmvq_capture * cap); // nullptr: launch normally again
bool ggml_cuda_mmvq_pair_supported(const ggml_cuda_mmvq_capture & a, const ggml_cuda_mmvq_capture & b);
// Restores the tensors an early (paired) output was redirected for; the graph loop calls it after the evaluation.
void ggml_cuda_restore_early_outputs();
// GGML_CUDA_MULTI_OP: the attention q/k/v projections at q's node as one launch; -1 when done (k, v precomputed).
int ggml_cuda_try_mmvq_triple(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
void ggml_cuda_mmvq_launch_pair(const ggml_cuda_mmvq_capture & a, const ggml_cuda_mmvq_capture & b, cudaStream_t stream);
bool ggml_cuda_mmvq_triple_supported(const ggml_cuda_mmvq_capture & a, const ggml_cuda_mmvq_capture & b, const ggml_cuda_mmvq_capture & c);
void ggml_cuda_mmvq_launch_triple(const ggml_cuda_mmvq_capture & a, const ggml_cuda_mmvq_capture & b, const ggml_cuda_mmvq_capture & c,
                                  cudaStream_t stream);
