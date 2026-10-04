#pragma once

#include "common.cuh"

// Qwen3.5/3.8 GDN gate chain for 1..8 tokens, in one launch instead of four:
//   gate = softplus(W_alpha x + dt) * A,  beta = sigmoid(W_beta x)
// Returns the number of extra graph nodes consumed (0 if the window does not match).
int ggml_cuda_try_fuse_gdn_gates(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
