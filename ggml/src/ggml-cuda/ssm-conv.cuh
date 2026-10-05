#include "common.cuh"
#include "mk-ops-gdn.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// MK_OP_GDN_CONV as one launch of p.C/128 blocks; replaces SSM_CONV + SILU and the q/k l2 norm pair.
void ggml_cuda_gdn_conv(const mk_gdn_conv_params & p, cudaStream_t stream);
