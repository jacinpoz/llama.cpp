#include "gdn-gates.cuh"

#include "ggml-impl.h"
#include "mk-ops-gdn.cuh"

// One block per (output row, token); rows [0, H) are alpha, [H, 2H) beta. Each token is reduced on its own, in the
// same order whatever the batch width, so decode and speculative-verify batches give the same values.
static __global__ void __launch_bounds__(256, 1) k_gdn_gates(const mk_gdn_gates_params p) {
    using op = mk_gdn_gates<256>;
    __shared__ float partial[op::threads/WARP_SIZE];
    op::run_at(p, blockIdx.x, blockIdx.y, true, (char *) partial);
}

static bool gdn_gates_is_f32_vec(const ggml_tensor * t, const int64_t n) {
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) == n;
}

bool ggml_cuda_gdn_gates_prepare(const ggml_cgraph * cgraph, const int i, mk_gdn_gates_params & p_out) {
    static const bool disabled = getenv("GGML_CUDA_DISABLE_GDN_GATES_FUSION") != nullptr && atoi(getenv("GGML_CUDA_DISABLE_GDN_GATES_FUSION")) != 0;
    if (disabled || i + 8 >= cgraph->n_nodes) {
        return false;
    }
    // MUL_MAT(alpha) RESHAPE ADD(dt) SOFTPLUS MUL(A) RESHAPE MUL_MAT(beta) RESHAPE SIGMOID
    static const ggml_op ops[] = {
        GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_RESHAPE,
        GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_UNARY,
    };
    for (int k = 0; k < 9; ++k) {
        if (cgraph->nodes[i + k]->op != ops[k]) {
            return false;
        }
    }
    ggml_tensor * mm_a  = cgraph->nodes[i + 0];
    ggml_tensor * add   = cgraph->nodes[i + 2];
    ggml_tensor * sp    = cgraph->nodes[i + 3];
    ggml_tensor * mul   = cgraph->nodes[i + 4];
    ggml_tensor * mm_b  = cgraph->nodes[i + 6];
    ggml_tensor * sig   = cgraph->nodes[i + 8];

    if (ggml_get_unary_op(sp) != GGML_UNARY_OP_SOFTPLUS || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID) {
        return false;
    }
    const ggml_tensor * w_a = mm_a->src[0];
    const ggml_tensor * w_b = mm_b->src[0];
    const ggml_tensor * x   = mm_a->src[1];
    if (mm_b->src[1] != x || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[2] != 1 || x->ne[3] != 1) {
        return false;
    }
    const int64_t K = x->ne[0];
    const int64_t T = x->ne[1];
    const int64_t H = w_a->ne[1];
    if (T < 1 || T > 8 || K % 4 != 0 || w_a->type != GGML_TYPE_F32 || w_b->type != GGML_TYPE_F32 ||
            w_a->ne[0] != K || w_b->ne[0] != K || w_b->ne[1] != H || w_a->ne[2] != 1 || w_b->ne[2] != 1 ||
            w_a->nb[0] != sizeof(float) || w_b->nb[0] != sizeof(float) || w_a->nb[1] % 16 != 0 || w_b->nb[1] % 16 != 0) {
        return false;
    }
    // chain wiring: ADD(reshape(mm_a), dt), SOFTPLUS(add), MUL(sp, A), SIGMOID(reshape(mm_b))
    const ggml_tensor * dt = add->src[0] == cgraph->nodes[i + 1] ? add->src[1] : add->src[1] == cgraph->nodes[i + 1] ? add->src[0] : nullptr;
    const ggml_tensor * A  = mul->src[0] == sp ? mul->src[1] : mul->src[1] == sp ? mul->src[0] : nullptr;
    if (cgraph->nodes[i + 1]->src[0] != mm_a || sp->src[0] != add || cgraph->nodes[i + 5]->src[0] != mul ||
            cgraph->nodes[i + 7]->src[0] != mm_b || sig->src[0] != cgraph->nodes[i + 7] ||
            dt == nullptr || A == nullptr || !gdn_gates_is_f32_vec(dt, H) || !gdn_gates_is_f32_vec(A, H)) {
        return false;
    }
    if (!ggml_is_contiguous(mul) || !ggml_is_contiguous(sig) || ggml_nelements(mul) != H*T || ggml_nelements(sig) != H*T ||
            mul->type != GGML_TYPE_F32 || sig->type != GGML_TYPE_F32) {
        return false;
    }
    const int outputs[] = { i + 5, i + 8 };
    if (!ggml_can_fuse_subgraph(cgraph, i, 9, ops, outputs, 2)) {
        return false;
    }

    p_out = {
        (const char *) w_a->data, (const char *) w_b->data, (const float *) x->data, (const float *) dt->data,
        (const float *) A->data, (float *) mul->data, (float *) sig->data, (int) K, (int) H,
        (int64_t) w_a->nb[1], (int64_t) w_b->nb[1], (int64_t) (x->nb[1]/sizeof(float)), H, H, (int) T, nullptr,
    };
    return true;
}

int ggml_cuda_try_fuse_gdn_gates(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    mk_gdn_gates_params p;
    if (!ggml_cuda_gdn_gates_prepare(cgraph, i, p)) {
        return 0;
    }
    k_gdn_gates<<<dim3((unsigned) (2*p.H), (unsigned) p.n_tokens), mk_gdn_gates<256>::threads, 0, ctx.stream()>>>(p);
    CUDA_CHECK(cudaGetLastError());
    return 8;
}
