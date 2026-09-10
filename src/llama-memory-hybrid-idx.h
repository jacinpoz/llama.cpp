#pragma once

#include "llama-memory-hybrid.h"

#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                 uint32_t   n_rs_batch,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    ~llama_memory_hybrid_idx() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // incremental block-vector cache access: the F32 pool tensor of indexer layer `il`
    // (nullptr when the model has no indexer); view [idx_dim x n_blocks x n_stream],
    // n_blocks = ceil(n_kv/ratio), rows [0, watermark) valid
    ggml_tensor * get_pool(ggml_context * ctx, int32_t il, uint32_t n_blocks) const;

    // fill the per-step derived-cache host leaves: fill range [from, to) with the score limit == to.
    // n_bid = count of full blocks per stream (from set_input_qsa's grouping).  Called by the const
    // set_input_qsa on the decode append path (advance = the derived path is live this step).
    void qsa_derived_limits(int32_t * dst_fill_from, int32_t * dst_limit, int n_stream, uint32_t ratio,
                            const uint32_t * n_bid, bool advance) const;

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    //
    // blk_idx/blk_tail are the compact (derived) alternative to the bias tensor: they let the
    // top-k derive the per-block half of the bias in-kernel from 4 bytes per block instead of
    // n_tokens/ns.  blk_idx is -1 for a block that is not complete for this stream, INT32_MAX
    // for the spare block holding the unpooled tail cells, else the position of the block's
    // first cell; blk_tail holds the per-token tail start.  The per-sequence half of the bias
    // is not folded in: the visibility (the attention mask, or the derived cell positions)
    // already drops every cell of a foreign block, so the values stay identical.  A caller
    // passing blk_idx must not add the bias into the block score itself.
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, ggml_tensor * blk_idx, ggml_tensor * blk_tail,
                       ggml_tensor * cell_vis, ggml_tensor * q_vis,
                       const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias,
                       int32_t * dst_derived_from = nullptr,
                       int32_t * dst_derived_lim  = nullptr) const;

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    //
    // incremental block-vector cache ("derived cache", qwen4exp QSA decode waste fix)
    //
    // One F32 [indexer_head_size x ceil(kv_size/ratio) x n_stream] tensor per indexer layer
    // holding the pooled + rms-normed + ROTATED vector of each FULL block, computed once when
    // the block completes instead of re-pooling the raw cache every decode token.  Lifecycle =
    // a per-layer host watermark: rows [0, watermark) are valid; any sequence mutation drops the
    // watermarks (the rows are never read above the watermark, so nothing needs memsetting).
    // The graph ops (fill + the derived score path) receive the range via per-step host leaves;
    // set_input_qsa fills them and advances the watermarks (decode-only, env-gated).
    struct llama_mem_pool_layer {
        uint32_t il;                  // model layer id (dense-attention, indexer-carrying)
        uint32_t ratio;               // compress ratio of this layer (blocks = ceil(kv_size/ratio))
        ggml_tensor * pool = nullptr; // F32 [idx_dim, n_blocks, n_stream]
    };

    // per-layer derived tensors + the contexts/buffers owning their memory
    std::vector<llama_mem_pool_layer> pool_layers;
    std::vector<ggml_context_ptr>      pool_ctxs;
    std::vector<ggml_backend_buffer_ptr> pool_bufs;

    // watermark per pool layer (rows [0, wm) are valid); decode advances it, seq ops drop it
    // mutable: set_input_qsa (const) advances it on the decode append path
    mutable std::vector<uint32_t> pool_wm;

    // the derived path is decode-only; ON by default, GGML_CUDA_QSA_INDEXER_CACHE=0 disables
    // (the constructor overwrites this from the env)
    bool derived_enabled = false;

    void pool_invalidate_all();
    void pool_create(const llama_model & model, const layer_filter_cb & filter_idx);
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    // Cells the QSA block metadata must cover.  The KV view is sized by OCCUPIED cells, but blocks
    // are keyed by POSITION, and a cache whose positions run ahead of its cells has blocks past that
    // view: the MTP draft context never receives the cells an M-RoPE image pins to one position, so
    // after an image its highest position leads its cell count by the image's grid size.  Sizing the
    // block tensors from get_n_kv() then makes the fill walk past the window (assert / corrupt read);
    // use max(get_n_kv(), highest stored position + 1), padded to 256 like get_n_kv() so graph reuse
    // keeps its cadence.  (Ported from the other solution's b0f31f587.)
    uint32_t qsa_n_kv_window() const;

    // [QSA_SCORE_BOUNDS] precondition for trimming the indexer scorer to the columns a query strip
    // can actually see: one sequence whose occupied cache cells carry unique non-negative positions
    // (so the complete blocks are enumerated in ascending logical block order and a block's ordinal
    // cannot exceed its logical block number).  M-RoPE images pin several cells to one position, so
    // they fail the uniqueness check and stay unbounded.
    bool qsa_position_prefix(const llama_ubatch & ubatch) const;

    // [QSA_SCORE_BOUNDS] per query strip, the number of leading score columns the strip can see
    // (the complete-block ordinals that are fully inside its causal prefix, plus the incomplete
    // tail block the fused top-k carries as cells), or an empty vector when the bound does not
    // apply.  `strip` is in tokens, `budget` = indexer_top_k / ratio.
    std::vector<int64_t> qsa_score_key_limits(const llama_ubatch & ubatch, int64_t n_blocks,
            int64_t strip, uint32_t ratio, int64_t budget) const;

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, ggml_tensor * blk_idx, ggml_tensor * blk_tail,
                       ggml_tensor * cell_vis, ggml_tensor * q_vis,
                       const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias,
                       int32_t * dst_derived_from = nullptr,
                       int32_t * dst_derived_lim  = nullptr) const;

    // F32 derived-cache view of indexer layer `il` ([idx_dim x n_blocks x n_stream]); nullptr
    // when the model carries no indexer or the layer is not a pool layer
    ggml_tensor * get_pool(ggml_context * ctx, int32_t il, uint32_t n_blocks) const;

private:
    const llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;
};
