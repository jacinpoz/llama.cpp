#pragma once
#include "common.cuh"

// R25: gufo-derived dense causal prefill attention (D = 256, GQA 8, RDNA3 WMMA). Opt-in: GGML_CUDA_FA_GUFO_256=1.
bool ggml_cuda_fattn_gufo_enabled();
bool ggml_cuda_fattn_gufo_supported(int device, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_gufo(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
