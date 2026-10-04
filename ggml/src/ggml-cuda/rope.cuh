#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows);

// Fused attention head prep (see rope.cu): RMS_NORM*w -> ROPE(mrope) -> Hadamard 256 -> out, or -> SET_ROWS q5_0.
bool ggml_cuda_op_attn_head_prep(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, const ggml_tensor * mul,
        const ggml_tensor * rope, ggml_tensor * hadamard, ggml_tensor * set_rows);
// V path: Hadamard 64 -> SET_ROWS q5_0.
bool ggml_cuda_op_hadamard64_set_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * hadamard, ggml_tensor * set_rows);
