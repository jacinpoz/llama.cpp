#pragma once

#include "common.cuh"

// Optional output epilogue (the Qwen3.5/3.8 gated-attention tail): out = fwht64(o) * sigmoid(gate), written to out
// instead of dst. gate is a [256, n_head, n_q] float view (byte strides gate_nb1/gate_nb2); out is contiguous.
struct gqa_dec_epilogue {
    const char * gate     = nullptr;
    int64_t      gate_nb1 = 0;
    int64_t      gate_nb2 = 0;
    float *      out      = nullptr;
};

bool ggml_cuda_fattn_gqa_dec_supported(int device, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_gqa_dec(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const gqa_dec_epilogue & ep = {});
