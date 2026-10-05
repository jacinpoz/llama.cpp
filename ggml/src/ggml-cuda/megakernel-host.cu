#include "megakernel-host.cuh"

#include "ggml-impl.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_set>

static std::string mk_format(const char * fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    return buf;
}

//
// stream builder
//

mk_stream_builder::mk_stream_builder(int32_t n_blocks_) : n_blocks(n_blocks_) {
    GGML_ASSERT(n_blocks > 0);
    building.queues.resize(n_blocks);
}

uint32_t mk_stream_builder::add_params(const void * data, size_t size) {
    const size_t off = GGML_PAD(params.size(), MK_PARAM_ALIGN);
    GGML_ASSERT(off + size <= UINT32_MAX);
    params.resize(off + std::max<size_t>(size, MK_PARAM_ALIGN), 0);
    memcpy(params.data() + off, data, size);
    return (uint32_t) off;
}

mk_counter mk_stream_builder::new_counter() {
    building.signals.push_back(0);
    return { (int32_t) segments.size(), (int32_t) building.signals.size() - 1 };
}

mk_wait mk_stream_builder::wait_all(mk_counter c) const {
    mk_wait w;
    w.counter = c;
    w.target  = c.segment == (int32_t) segments.size() && c.id >= 0 ? building.signals[c.id] : 0;
    return w;
}

mk_counter mk_stream_builder::add_op(const mk_op_desc & d) {
    GGML_ASSERT(d.n_tiles >= 0 && d.tiles_per_step > 0);
    const int32_t seg = (int32_t) segments.size();

    mk_counter signal = d.signal;
    if (signal.id < 0) {
        signal = new_counter();
    }
    GGML_ASSERT(signal.segment == seg && signal.id < (int32_t) building.signals.size());

    std::vector<mk_wait> waits;
    for (const mk_wait & w : d.waits) {
        if (w.counter.segment != seg) {
            continue;
        }
        GGML_ASSERT(w.counter.id >= 0 && w.counter.id < (int32_t) building.signals.size());
        const uint32_t target = w.target == UINT32_MAX ? building.signals[w.counter.id] : w.target;
        if (target > 0) {
            waits.push_back({ w.counter, target });
        }
    }

    segment_build & s = building;
    const int32_t n_steps  = (d.n_tiles + d.tiles_per_step - 1) / d.tiles_per_step;
    const int32_t n_chunks = std::min(n_steps, n_blocks);
    int32_t step = 0;
    for (int32_t c = 0; c < n_chunks; ++c) {
        const int32_t steps_c = n_steps / n_chunks + (c < n_steps % n_chunks ? 1 : 0);
        std::vector<mk_instr> & q = s.queues[(next_block + c) % n_blocks];

        for (size_t k = 0; k + 1 < waits.size(); ++k) {
            mk_instr nop = {};
            nop.opcode         = MK_OP_NOP;
            nop.wait_counter   = waits[k].counter.id;
            nop.wait_target    = waits[k].target;
            nop.signal_counter = -1;
            nop.prefetch_off   = UINT32_MAX;
            q.push_back(nop);
        }

        mk_instr ins = {};
        ins.opcode         = d.opcode;
        ins.variant        = d.variant;
        ins.tile_begin     = step * d.tiles_per_step;
        ins.tile_end       = std::min(d.n_tiles, (step + steps_c) * d.tiles_per_step);
        ins.wait_counter   = waits.empty() ? -1 : waits.back().counter.id;
        ins.wait_target    = waits.empty() ? 0 : waits.back().target;
        ins.signal_counter = signal.id;
        ins.params_off     = d.params_off;
        ins.prefetch_off   = d.prefetch_off;
        q.push_back(ins);

        s.signals[signal.id]++;
        step += steps_c;
    }
    next_block = (next_block + n_chunks) % n_blocks;
    return signal;
}

void mk_stream_builder::flush_segment() {
    if (building.signals.empty()) {
        return;
    }
    mk_segment seg;
    seg.stream.n_blocks = n_blocks;
    seg.stream.queue_begin.push_back(0);
    for (const std::vector<mk_instr> & q : building.queues) {
        seg.stream.instrs.insert(seg.stream.instrs.end(), q.begin(), q.end());
        seg.stream.queue_begin.push_back((int32_t) seg.stream.instrs.size());
    }
    seg.stream.signals_per_pass = std::move(building.signals);
    segments.push_back(std::move(seg));

    for (std::vector<mk_instr> & q : building.queues) {
        q.clear();
    }
    building.signals.clear();
    next_block = 0;
}

void mk_stream_builder::add_external(int32_t external_id) {
    flush_segment();
    mk_segment seg;
    seg.external    = true;
    seg.external_id = external_id;
    segments.push_back(std::move(seg));
}

std::vector<mk_segment> mk_stream_builder::finish() {
    flush_segment();
    for (mk_segment & s : segments) {
        if (!s.external) {
            s.stream.params = params;
        }
    }
    params.clear();
    return std::move(segments);
}

//
// stream validator
//

std::string mk_validate_stream(const mk_stream_host & s) {
    const int32_t nb = s.n_blocks;
    if (nb <= 0) {
        return mk_format("n_blocks = %d", nb);
    }
    if ((int32_t) s.queue_begin.size() != nb + 1 || s.queue_begin[0] != 0 || s.queue_begin[nb] != (int32_t) s.instrs.size()) {
        return mk_format("queue_begin must have n_blocks + 1 = %d entries from 0 to n_instrs = %zu", nb + 1, s.instrs.size());
    }
    for (int32_t b = 0; b < nb; ++b) {
        if (s.queue_begin[b] > s.queue_begin[b + 1]) {
            return mk_format("queue_begin is not monotonic at block %d", b);
        }
    }

    const int32_t nc = (int32_t) s.signals_per_pass.size();
    std::vector<uint32_t> n_signals(nc, 0);

    for (int32_t b = 0; b < nb; ++b) {
        for (int32_t i = s.queue_begin[b]; i < s.queue_begin[b + 1]; ++i) {
            const mk_instr & ins = s.instrs[i];
            const int32_t    k   = i - s.queue_begin[b];
            if (ins.opcode >= MK_OP_COUNT) {
                return mk_format("block %d #%d: opcode %u out of range", b, k, ins.opcode);
            }
            if (ins.opcode == MK_OP_EXIT && i + 1 != s.queue_begin[b + 1]) {
                return mk_format("block %d #%d: EXIT before the end of the queue", b, k);
            }
            if (ins.tile_begin < 0 || ins.tile_begin > ins.tile_end) {
                return mk_format("block %d #%d: bad tile range [%d, %d)", b, k, ins.tile_begin, ins.tile_end);
            }
            if (ins.wait_counter < -1 || ins.wait_counter >= nc) {
                return mk_format("block %d #%d: wait counter %d out of range [0, %d)", b, k, ins.wait_counter, nc);
            }
            if (ins.signal_counter < -1 || ins.signal_counter >= nc) {
                return mk_format("block %d #%d: signal counter %d out of range [0, %d)", b, k, ins.signal_counter, nc);
            }
            if (ins.params_off % MK_PARAM_ALIGN != 0) {
                return mk_format("block %d #%d: params offset %u not %d-byte aligned", b, k, ins.params_off, MK_PARAM_ALIGN);
            }
            if (ins.opcode != MK_OP_NOP && ins.opcode != MK_OP_EXIT && (size_t) ins.params_off + MK_PARAM_ALIGN > s.params.size()) {
                return mk_format("block %d #%d: params offset %u out of range (blob is %zu bytes)", b, k, ins.params_off, s.params.size());
            }
            if (ins.signal_counter >= 0) {
                n_signals[ins.signal_counter]++;
            }
        }
    }

    for (int32_t c = 0; c < nc; ++c) {
        if (n_signals[c] != s.signals_per_pass[c]) {
            return mk_format("counter %d: signals_per_pass is %u but %u instructions signal it", c, s.signals_per_pass[c], n_signals[c]);
        }
    }

    // signals of each counter still queued behind the instruction being checked in its own block
    std::vector<uint32_t> behind(nc, 0);
    std::vector<int32_t>  touched;
    for (int32_t b = 0; b < nb; ++b) {
        for (int32_t i = s.queue_begin[b + 1] - 1; i >= s.queue_begin[b]; --i) {
            const mk_instr & ins = s.instrs[i];
            const int32_t    k   = i - s.queue_begin[b];
            if (ins.signal_counter >= 0) {
                if (behind[ins.signal_counter]++ == 0) {
                    touched.push_back(ins.signal_counter);
                }
            }
            if (ins.wait_counter < 0) {
                continue;
            }
            const uint32_t total = s.signals_per_pass[ins.wait_counter];
            if (ins.wait_target > total) {
                return mk_format("block %d #%d: wait target %u on counter %d exceeds its signals_per_pass %u",
                        b, k, ins.wait_target, ins.wait_counter, total);
            }
            const uint32_t reachable = total - behind[ins.wait_counter];
            if (ins.wait_target > reachable) {
                return mk_format("block %d #%d: self-wait: target %u on counter %d needs its own later signals (%u of %u come from other work)",
                        b, k, ins.wait_target, ins.wait_counter, reachable, total);
            }
        }
        for (int32_t c : touched) {
            behind[c] = 0;
        }
        touched.clear();
    }

    std::vector<int32_t>              pc(nb);
    std::vector<uint32_t>             count(nc, 0);
    std::vector<std::vector<int32_t>> waiters(nc);
    std::vector<int32_t>              ready;
    for (int32_t b = 0; b < nb; ++b) {
        pc[b] = s.queue_begin[b];
        ready.push_back(b);
    }
    while (!ready.empty()) {
        const int32_t b = ready.back();
        ready.pop_back();
        while (pc[b] < s.queue_begin[b + 1]) {
            const mk_instr & ins = s.instrs[pc[b]];
            if (ins.wait_counter >= 0 && count[ins.wait_counter] < ins.wait_target) {
                waiters[ins.wait_counter].push_back(b);
                break;
            }
            pc[b]++;
            if (ins.signal_counter < 0) {
                continue;
            }
            const int32_t c = ins.signal_counter;
            count[c]++;
            std::vector<int32_t> & w = waiters[c];
            for (size_t j = 0; j < w.size();) {
                if (count[c] >= s.instrs[pc[w[j]]].wait_target) {
                    ready.push_back(w[j]);
                    w[j] = w.back();
                    w.pop_back();
                } else {
                    ++j;
                }
            }
        }
    }

    int32_t first_stuck = -1;
    for (int32_t b = 0; b < nb && first_stuck < 0; ++b) {
        if (pc[b] < s.queue_begin[b + 1]) {
            first_stuck = b;
        }
    }
    if (first_stuck < 0) {
        return "";
    }

    auto describe = [&](int32_t b) {
        const mk_instr & ins = s.instrs[pc[b]];
        return mk_format("block %d #%d waits counter %d at %u/%u", b, pc[b] - s.queue_begin[b], ins.wait_counter,
                count[ins.wait_counter], ins.wait_target);
    };
    auto pending_signaler = [&](int32_t b) {
        const int32_t c = s.instrs[pc[b]].wait_counter;
        for (int32_t d = 0; d < nb; ++d) {
            for (int32_t i = pc[d]; i < s.queue_begin[d + 1]; ++i) {
                if (s.instrs[i].signal_counter == c) {
                    return d;
                }
            }
        }
        return -1;
    };

    std::vector<int32_t> visit_pos(nb, -1);
    std::vector<int32_t> path;
    int32_t b = first_stuck;
    while (b >= 0 && visit_pos[b] < 0) {
        visit_pos[b] = (int32_t) path.size();
        path.push_back(b);
        b = pending_signaler(b);
    }
    if (b < 0) {
        return "deadlock: " + describe(path.back()) + " and no pending instruction signals it";
    }
    std::string msg = "deadlock cycle: ";
    for (size_t j = visit_pos[b]; j < path.size(); ++j) {
        msg += describe(path[j]) + " -> ";
    }
    return msg + mk_format("block %d", b);
}

//
// graph template matcher
//

static const char * const mk_hop_names[MK_HOP_COUNT] = {
    "EMBED_ROWS", "RMSNORM_Q8_1", "RMSNORM_F32", "QUANTIZE_Q8_1", "MMVQ", "MMVQ_ADD", "MMVQ_GLU",
    "STATE_COPY", "GDN_CONV", "GDN_GATES", "GDN_STEP", "GDN_OUT_GATE",
    "ATTN_PREP_Q", "ATTN_PREP_K", "V_HAD_SET_ROWS", "ATTN_PARTIAL", "ATTN_COMBINE", "OUT_ROWS", "CONCAT",
    "FILL_NEG_INF",
};

static const char * const mk_hop_slots[MK_HOP_COUNT][MK_MATCH_MAX_T] = {
    /* EMBED_ROWS     */ { "out", "weight", "ids" },
    /* RMSNORM_Q8_1   */ { "out", "norm", "x", "w" },
    /* RMSNORM_F32    */ { "out", "norm", "x", "w" },
    /* QUANTIZE_Q8_1  */ { "x" },
    /* MMVQ           */ { "out", "weight", "x" },
    /* MMVQ_ADD       */ { "out", "mm", "weight", "x", "residual" },
    /* MMVQ_GLU       */ { "out", "gate_mm", "up_mm", "gate_w", "up_w", "x" },
    /* STATE_COPY     */ { "cpy", "src", "dst" },
    /* GDN_CONV       */ { "out", "q_out", "k_out", "qkv_mm", "state_gather", "snapshot_cpy", "kernel", "conv", "q_norm", "k_norm" },
    /* GDN_GATES      */ { "gate", "beta", "alpha_mm", "beta_mm", "dt", "a", "x" },
    /* GDN_STEP       */ { "out", "state_cpy", "q", "k", "v", "g", "beta", "state_gather", "state_dst" },
    /* GDN_OUT_GATE   */ { "out", "norm", "norm_mul", "silu_z", "z_mm", "w" },
    /* ATTN_PREP_Q    */ { "out", "norm", "norm_mul", "rope", "q_mm", "w", "pos", "rot" },
    /* ATTN_PREP_K    */ { "set_rows", "had", "norm", "norm_mul", "rope", "k_mm", "w", "pos", "rot", "idxs" },
    /* V_HAD_SET_ROWS */ { "set_rows", "had", "v_mm", "rot", "idxs" },
    /* ATTN_PARTIAL   */ { "fa", "q", "k", "v", "mask" },
    /* ATTN_COMBINE   */ { "out", "fa", "had", "rot", "sigmoid", "gate_cont", "q_mm" },
    /* OUT_ROWS       */ { "out", "x", "ids" },
    /* CONCAT         */ { "out", "a", "b" },
    /* FILL_NEG_INF   */ { "out", "src", "lo" },
};

const char * mk_hop_name(mk_hop kind) {
    return kind >= 0 && kind < MK_HOP_COUNT ? mk_hop_names[kind] : "?";
}

const char * mk_hop_slot_name(mk_hop kind, int slot) {
    return kind >= 0 && kind < MK_HOP_COUNT && slot >= 0 && slot < MK_MATCH_MAX_T ? mk_hop_slots[kind][slot] : nullptr;
}

bool mk_hop_is_external(mk_hop kind) {
    return kind == MK_HOP_EMBED_ROWS || kind == MK_HOP_ATTN_PARTIAL;
}

static bool mk_is_transparent(const ggml_tensor * t) {
    return ggml_is_empty(t) || ggml_op_is_empty(t->op) || (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0;
}

static const ggml_tensor * mk_root(const ggml_tensor * t) {
    while (t != nullptr && t->op != GGML_OP_NONE && ggml_op_is_empty(t->op)) {
        t = t->src[0];
    }
    return t;
}

static bool mk_weight_type_ok(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

static const ggml_tensor * mk_other_src(const ggml_tensor * t, const ggml_tensor * a) {
    return t->src[0] == a ? t->src[1] : t->src[1] == a ? t->src[0] : nullptr;
}

static bool mk_is_unary(const ggml_tensor * t, ggml_unary_op op) {
    return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op;
}

static bool mk_is_hadamard(const ggml_tensor * t, int64_t n) {
    return t->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        t->src[0]->ne[0] == n && t->src[0]->ne[1] == n && t->src[0]->type == GGML_TYPE_F32;
}

struct mk_walker {
    ggml_cgraph *        g;
    std::vector<int32_t> cn;
    size_t               pos   = 0;
    int32_t              layer = -1;
    mk_match &           m;

    mk_walker(ggml_cgraph * graph, mk_match & match) : g(graph), m(match) {
        for (int32_t i = 0; i < ggml_graph_n_nodes(g); ++i) {
            if (!mk_is_transparent(ggml_graph_node(g, i))) {
                cn.push_back(i);
            }
        }
    }

    ggml_tensor * peek(size_t k = 0) const {
        return pos + k < cn.size() ? ggml_graph_node(g, cn[pos + k]) : nullptr;
    }

    bool fail(const std::string & what) {
        if (m.reason.empty()) {
            const ggml_tensor * t = peek();
            m.reason = t != nullptr
                ? mk_format("layer %d, node %d (%s '%s'): %s", layer, cn[pos], ggml_op_name(t->op), t->name, what.c_str())
                : mk_format("layer %d, end of graph: %s", layer, what.c_str());
        }
        return false;
    }

    ggml_tensor * take(ggml_op op, const char * what) {
        ggml_tensor * t = peek();
        if (t == nullptr || t->op != op) {
            fail(mk_format("expected %s (%s)", ggml_op_name(op), what));
            return nullptr;
        }
        pos++;
        return t;
    }

    void push(mk_hop kind, std::initializer_list<const ggml_tensor *> ts) {
        mk_match_op op = {};
        op.kind  = kind;
        op.layer = layer;
        int k = 0;
        for (const ggml_tensor * t : ts) {
            GGML_ASSERT(k < MK_MATCH_MAX_T);
            op.t[k++] = t;
        }
        m.ops.push_back(op);
    }
};

#define MK_TAKE(var, op, what)                   \
    ggml_tensor * var = w.take(op, what);        \
    if (var == nullptr) {                        \
        return false;                            \
    }

#define MK_CHECK(cond, what)                     \
    if (!(cond)) {                               \
        return w.fail(what);                     \
    }

static bool mk_check_rows(mk_walker & w, const ggml_tensor * t, const char * what) {
    MK_CHECK(t->type == GGML_TYPE_F32 && t->ne[1] == w.m.n_tokens && t->ne[2] == 1 && t->ne[3] == 1,
        mk_format("%s must be F32 [n, T = %d, 1, 1]", what, w.m.n_tokens));
    return true;
}

static bool mk_check_weight(mk_walker & w, const ggml_tensor * mm, bool allow_f32 = false) {
    const ggml_tensor * wt = mm->src[0];
    MK_CHECK(mk_root(wt)->op == GGML_OP_NONE && wt->ne[2] == 1 && wt->ne[3] == 1 && mm->type == GGML_TYPE_F32,
        mk_format("MUL_MAT '%s' must read a 2D weight", mm->name));
    MK_CHECK(mk_weight_type_ok(wt->type) || (allow_f32 && wt->type == GGML_TYPE_F32),
        mk_format("weight '%s' has unsupported type %s", wt->name, ggml_type_name(wt->type)));
    return true;
}

// RMS_NORM(x) MUL(w)
static bool mk_match_norm(mk_walker & w, const ggml_tensor * x, mk_hop kind, ggml_tensor ** out) {
    MK_TAKE(norm, GGML_OP_RMS_NORM, "norm");
    MK_TAKE(mul, GGML_OP_MUL, "norm weight");
    MK_CHECK(x == nullptr || norm->src[0] == x, "norm does not read the expected input");
    const ggml_tensor * wt = mk_other_src(mul, norm);
    MK_CHECK(wt != nullptr && wt->op == GGML_OP_NONE && wt->type == GGML_TYPE_F32 && wt->ne[0] == norm->ne[0] &&
        ggml_nelements(wt) == wt->ne[0], "norm weight MUL wiring");
    MK_CHECK(norm->src[0]->type == GGML_TYPE_F32 && ggml_are_same_shape(mul, norm), "norm shape");
    w.push(kind, { mul, norm, norm->src[0], wt });
    *out = mul;
    return true;
}

// MUL_MAT(w, x)
static bool mk_match_mmvq(mk_walker & w, const ggml_tensor * x, bool x_is_q8_1, ggml_tensor ** out) {
    MK_TAKE(mm, GGML_OP_MUL_MAT, "matvec");
    MK_CHECK(mk_root(mm->src[1]) == x, mk_format("MUL_MAT '%s' does not read '%s'", mm->name, x->name));
    if (!mk_check_weight(w, mm)) {
        return false;
    }
    if (!x_is_q8_1) {
        w.push(MK_HOP_QUANTIZE_Q8_1, { x });
    }
    w.push(MK_HOP_MMVQ, { mm, mm->src[0], x });
    *out = mm;
    return true;
}

// MUL_MAT(w, x) ADD(residual)
static bool mk_match_mmvq_add(mk_walker & w, const ggml_tensor * x, bool x_is_q8_1, const ggml_tensor * residual, ggml_tensor ** out) {
    MK_TAKE(mm, GGML_OP_MUL_MAT, "matvec");
    MK_TAKE(add, GGML_OP_ADD, "residual add");
    MK_CHECK(mk_root(mm->src[1]) == x, mk_format("MUL_MAT '%s' does not read '%s'", mm->name, x->name));
    if (!mk_check_weight(w, mm)) {
        return false;
    }
    MK_CHECK((mk_root(add->src[0]) == mm && add->src[1] == residual) || (mk_root(add->src[1]) == mm && add->src[0] == residual),
        "residual ADD wiring");
    if (!mk_check_rows(w, add, "residual add")) {
        return false;
    }
    if (!x_is_q8_1) {
        w.push(MK_HOP_QUANTIZE_Q8_1, { x });
    }
    w.push(MK_HOP_MMVQ_ADD, { add, mm, mm->src[0], x, residual });
    *out = add;
    return true;
}

static bool mk_match_ffn(mk_walker & w, const ggml_tensor * x, ggml_tensor ** out) {
    ggml_tensor * xn = nullptr;
    if (!mk_match_norm(w, x, MK_HOP_RMSNORM_Q8_1, &xn)) {
        return false;
    }
    MK_TAKE(gate, GGML_OP_MUL_MAT, "ffn gate");
    MK_TAKE(up, GGML_OP_MUL_MAT, "ffn up");
    MK_TAKE(glu, GGML_OP_GLU, "ffn swiglu");
    MK_CHECK(gate->src[1] == xn && up->src[1] == xn, "ffn gate/up input");
    MK_CHECK(ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU && glu->src[0] == gate && glu->src[1] == up, "ffn swiglu wiring");
    if (!mk_check_weight(w, gate) || !mk_check_weight(w, up)) {
        return false;
    }
    w.push(MK_HOP_MMVQ_GLU, { glu, gate, up, gate->src[0], up->src[0], xn });
    return mk_match_mmvq_add(w, glu, false, x, out);
}

static bool mk_match_gdn_layer(mk_walker & w, const ggml_tensor * x, ggml_tensor ** out) {
    ggml_tensor * xn = nullptr;
    if (!mk_match_norm(w, x, MK_HOP_RMSNORM_Q8_1, &xn)) {
        return false;
    }

    auto take_gather = [&](const char * what) -> ggml_tensor * {
        ggml_tensor * gr = w.take(GGML_OP_GET_ROWS, what);
        if (gr == nullptr) {
            return nullptr;
        }
        const ggml_tensor * src = gr->src[0];
        if (gr->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32 || gr->src[1]->type != GGML_TYPE_I32 ||
                ggml_nelements(gr->src[1]) != 1 || gr->ne[1] != 1 || gr->ne[2] != 1 || gr->ne[3] != 1 ||
                src->nb[0] != sizeof(float) || src->nb[1] != src->ne[0]*sizeof(float)) {
            w.fail(mk_format("%s must gather one contiguous state row", what));
            return nullptr;
        }
        return gr;
    };
    // copies of src_root into the F32 state cache; the first one is returned in *first
    auto take_copies = [&](const ggml_tensor * src_root, const char * what, ggml_tensor ** first, int * n) {
        *first = nullptr;
        *n     = 0;
        while (w.peek() != nullptr && w.peek()->op == GGML_OP_CPY && mk_root(w.peek()->src[0]) == src_root) {
            ggml_tensor * cpy = w.take(GGML_OP_CPY, what);
            if (cpy->src[1]->type != GGML_TYPE_F32 || mk_root(cpy->src[1])->op != GGML_OP_NONE || !ggml_is_contiguous(cpy->src[1])) {
                return w.fail(mk_format("%s must write a contiguous F32 state cache row", what));
            }
            if (*first == nullptr) {
                *first = cpy;
            } else {
                w.push(MK_HOP_STATE_COPY, { cpy, cpy->src[0], cpy->src[1] });
            }
            (*n)++;
        }
        return true;
    };

    ggml_tensor * conv_states = take_gather("conv state gather");
    if (conv_states == nullptr) {
        return false;
    }
    MK_TAKE(qkv, GGML_OP_MUL_MAT, "qkv projection");
    MK_TAKE(cat, GGML_OP_CONCAT, "conv input");
    MK_CHECK(qkv->src[1] == xn && mk_root(cat->src[1]) == qkv && mk_root(cat->src[0]) == conv_states &&
        ggml_get_op_params_i32(cat, 0) == 0 && cat->type == GGML_TYPE_F32 && cat->ne[2] == 1, "conv input wiring");
    if (!mk_check_weight(w, qkv)) {
        return false;
    }
    w.push(MK_HOP_MMVQ, { qkv, qkv->src[0], xn });

    ggml_tensor * snap = nullptr;
    int n_snap = 0;
    if (!take_copies(cat, "conv state snapshot", &snap, &n_snap)) {
        return false;
    }
    MK_CHECK(snap != nullptr, "conv state snapshot copy missing");
    ggml_tensor * conv_pre = nullptr;
    int n_pre = 0;
    if (!take_copies(conv_states, "conv state pre-batch copy", &conv_pre, &n_pre)) {
        return false;
    }

    ggml_tensor * state = take_gather("recurrent state gather");
    ggml_tensor * state_pre = nullptr;
    int n_state_pre = 0;
    if (state == nullptr || !take_copies(state, "recurrent state pre-batch copy", &state_pre, &n_state_pre)) {
        return false;
    }

    MK_TAKE(conv, GGML_OP_SSM_CONV, "conv");
    MK_TAKE(silu, GGML_OP_UNARY, "conv silu");
    const ggml_tensor * kern = conv->src[1];
    MK_CHECK(conv->src[0] == cat && kern->op == GGML_OP_NONE && kern->type == GGML_TYPE_F32 && ggml_is_contiguous(kern) &&
        kern->ne[0] == 4 && mk_is_unary(silu, GGML_UNARY_OP_SILU) && silu->src[0] == conv, "conv wiring (d_conv 4)");

    MK_TAKE(qn, GGML_OP_RMS_NORM, "q l2 norm");
    MK_TAKE(qs, GGML_OP_SCALE, "q l2 norm scale");
    MK_TAKE(kn, GGML_OP_RMS_NORM, "k l2 norm");
    MK_TAKE(ks, GGML_OP_SCALE, "k l2 norm scale");
    MK_CHECK(mk_root(qn->src[0]) == silu && qs->src[0] == qn && mk_root(kn->src[0]) == silu && ks->src[0] == kn &&
        qn->src[0]->ne[0] == 128 && kn->src[0]->ne[0] == 128, "q/k l2 norm wiring (128 channels per head)");
    w.push(MK_HOP_GDN_CONV, { silu, qs, ks, qkv, conv_states, snap, kern, conv, qn, kn });
    if (conv_pre != nullptr) {
        w.push(MK_HOP_STATE_COPY, { conv_pre, conv_pre->src[0], conv_pre->src[1] });
    }
    if (state_pre != nullptr) {
        w.push(MK_HOP_STATE_COPY, { state_pre, state_pre->src[0], state_pre->src[1] });
    }

    MK_TAKE(a_mm, GGML_OP_MUL_MAT, "alpha projection");
    MK_TAKE(a_add, GGML_OP_ADD, "alpha + dt");
    MK_TAKE(a_sp, GGML_OP_UNARY, "softplus");
    MK_TAKE(a_mul, GGML_OP_MUL, "softplus * a");
    MK_TAKE(b_mm, GGML_OP_MUL_MAT, "beta projection");
    MK_TAKE(b_sig, GGML_OP_UNARY, "beta sigmoid");
    const ggml_tensor * dt = mk_root(a_add->src[0]) == a_mm ? a_add->src[1] : mk_root(a_add->src[1]) == a_mm ? a_add->src[0] : nullptr;
    const ggml_tensor * A  = mk_other_src(a_mul, a_sp);
    MK_CHECK(a_mm->src[1] == xn && b_mm->src[1] == xn && dt != nullptr && dt->op == GGML_OP_NONE && A != nullptr &&
        A->op == GGML_OP_NONE && mk_is_unary(a_sp, GGML_UNARY_OP_SOFTPLUS) && a_sp->src[0] == a_add &&
        mk_is_unary(b_sig, GGML_UNARY_OP_SIGMOID) && mk_root(b_sig->src[0]) == b_mm, "gate chain wiring");
    if (!mk_check_weight(w, a_mm, true) || !mk_check_weight(w, b_mm, true)) {
        return false;
    }
    w.push(MK_HOP_GDN_GATES, { a_mul, b_sig, a_mm, b_mm, dt, A, xn });

    MK_TAKE(gdn, GGML_OP_GATED_DELTA_NET, "gated delta net");
    MK_CHECK(mk_root(gdn->src[0]) == qs && mk_root(gdn->src[1]) == ks && mk_root(gdn->src[2]) == silu &&
        mk_root(gdn->src[3]) == a_mul && mk_root(gdn->src[4]) == b_sig && mk_root(gdn->src[5]) == state &&
        gdn->src[2]->ne[3] == 1, "gated delta net wiring");
    MK_TAKE(scpy, GGML_OP_CPY, "state snapshot");
    MK_CHECK(mk_root(scpy->src[0]) == gdn && mk_root(scpy->src[1])->op == GGML_OP_NONE, "state snapshot wiring");
    w.push(MK_HOP_GDN_STEP, { gdn, scpy, gdn->src[0], gdn->src[1], gdn->src[2], gdn->src[3], gdn->src[4], state, scpy->src[1] });

    MK_TAKE(on, GGML_OP_RMS_NORM, "output norm");
    MK_TAKE(onm, GGML_OP_MUL, "output norm weight");
    MK_TAKE(z_mm, GGML_OP_MUL_MAT, "z projection");
    MK_TAKE(zs, GGML_OP_UNARY, "silu(z)");
    MK_TAKE(gm, GGML_OP_MUL, "output gate");
    const ggml_tensor * on_w = mk_other_src(onm, on);
    MK_CHECK(mk_root(on->src[0]) == gdn && on_w != nullptr && on_w->op == GGML_OP_NONE && z_mm->src[1] == xn &&
        mk_is_unary(zs, GGML_UNARY_OP_SILU) && mk_root(zs->src[0]) == z_mm && mk_other_src(gm, onm) == zs, "output gate wiring");
    if (!mk_check_weight(w, z_mm)) {
        return false;
    }
    w.push(MK_HOP_MMVQ, { z_mm, z_mm->src[0], xn });
    w.push(MK_HOP_GDN_OUT_GATE, { gm, on, onm, zs, z_mm, on_w });

    ggml_tensor * res = nullptr;
    if (!mk_match_mmvq_add(w, gm, true, x, &res)) {
        return false;
    }
    return mk_match_ffn(w, res, out);
}

struct mk_head_prep {
    const ggml_tensor * norm;
    const ggml_tensor * mul;
    const ggml_tensor * rope;
    const ggml_tensor * w;
    const ggml_tensor * pos;
    const ggml_tensor * rot;
    ggml_tensor *       had;
};

// MUL_MAT(wq|wk) RMS_NORM MUL ROPE MUL_MAT(hadamard 256)
static bool mk_match_head_prep(mk_walker & w, const ggml_tensor * mm, mk_head_prep & hp) {
    MK_TAKE(norm, GGML_OP_RMS_NORM, "head norm");
    MK_TAKE(mul, GGML_OP_MUL, "head norm weight");
    MK_TAKE(rope, GGML_OP_ROPE, "rope");
    MK_TAKE(had, GGML_OP_MUL_MAT, "hadamard");
    const ggml_tensor * nw = mk_other_src(mul, norm);
    MK_CHECK(mk_root(norm->src[0]) == mm && nw != nullptr && nw->op == GGML_OP_NONE && rope->src[0] == mul &&
        rope->src[2] == nullptr && mk_is_hadamard(had, 256) && mk_root(had->src[1]) == rope, "head prep wiring");
    hp = { norm, mul, rope, nw, rope->src[1], had->src[0], had };
    return true;
}

static bool mk_match_attn_block(mk_walker & w, const ggml_tensor * x, ggml_tensor ** out) {
    ggml_tensor * xn = nullptr;
    if (!mk_match_norm(w, x, MK_HOP_RMSNORM_Q8_1, &xn)) {
        return false;
    }
    MK_TAKE(q_mm, GGML_OP_MUL_MAT, "q projection");
    MK_CHECK(q_mm->src[1] == xn, "q projection input");
    mk_head_prep qp;
    if (!mk_check_weight(w, q_mm) || !mk_match_head_prep(w, q_mm, qp)) {
        return false;
    }

    MK_TAKE(v_mm, GGML_OP_MUL_MAT, "v projection");
    MK_TAKE(vh, GGML_OP_MUL_MAT, "v hadamard");
    MK_CHECK(v_mm->src[1] == xn && mk_is_hadamard(vh, 64) && mk_root(vh->src[1]) == v_mm, "v chain wiring");

    MK_TAKE(k_mm, GGML_OP_MUL_MAT, "k projection");
    MK_CHECK(k_mm->src[1] == xn, "k projection input");
    mk_head_prep kp;
    if (!mk_check_weight(w, v_mm) || !mk_check_weight(w, k_mm) || !mk_match_head_prep(w, k_mm, kp)) {
        return false;
    }

    MK_TAKE(ks, GGML_OP_SET_ROWS, "k cache store");
    MK_TAKE(vs, GGML_OP_SET_ROWS, "v cache store");
    MK_CHECK(mk_root(ks->src[0]) == kp.had && mk_root(vs->src[0]) == vh && ks->type == GGML_TYPE_Q5_0 && vs->type == GGML_TYPE_Q5_0 &&
        ks->src[1]->type != GGML_TYPE_F32 && vs->src[1]->type != GGML_TYPE_F32, "kv store wiring (q5_0 cache)");

    MK_TAKE(fa, GGML_OP_FLASH_ATTN_EXT, "attention");
    const ggml_tensor * kc = fa->src[1];
    const ggml_tensor * vc = fa->src[2];
    MK_CHECK(mk_root(fa->src[0]) == qp.had && mk_root(kc) == mk_root(ks->src[2]) && mk_root(vc) == mk_root(vs->src[2]) &&
        fa->src[3] != nullptr && fa->src[4] == nullptr && kc->type == GGML_TYPE_Q5_0 && vc->type == GGML_TYPE_Q5_0 && kc->ne[0] == 256 &&
        kc->ne[3] == 1 && fa->ne[0] == 256 && fa->ne[2] == w.m.n_tokens && fa->ne[3] == 1, "attention wiring");
    w.m.n_kv = std::max<int64_t>(w.m.n_kv, kc->ne[1]);

    MK_TAKE(oh, GGML_OP_MUL_MAT, "attention output hadamard");
    MK_TAKE(cont, GGML_OP_CONT, "gate");
    MK_TAKE(sig, GGML_OP_UNARY, "gate sigmoid");
    MK_TAKE(gm, GGML_OP_MUL, "gated attention");
    MK_CHECK(mk_is_hadamard(oh, 64) && mk_root(oh->src[1]) == fa && mk_root(cont->src[0]) == q_mm &&
        mk_is_unary(sig, GGML_UNARY_OP_SIGMOID) && sig->src[0] == cont &&
        ((mk_root(gm->src[0]) == oh && gm->src[1] == sig) || (gm->src[0] == sig && mk_root(gm->src[1]) == oh)),
        "attention tail wiring");

    w.push(MK_HOP_MMVQ, { q_mm, q_mm->src[0], xn });
    w.push(MK_HOP_ATTN_PREP_Q, { qp.had, qp.norm, qp.mul, qp.rope, q_mm, qp.w, qp.pos, qp.rot });
    w.push(MK_HOP_MMVQ, { v_mm, v_mm->src[0], xn });
    w.push(MK_HOP_V_HAD_SET_ROWS, { vs, vh, v_mm, vh->src[0], vs->src[1] });
    w.push(MK_HOP_MMVQ, { k_mm, k_mm->src[0], xn });
    w.push(MK_HOP_ATTN_PREP_K, { ks, kp.had, kp.norm, kp.mul, kp.rope, k_mm, kp.w, kp.pos, kp.rot, ks->src[1] });
    w.push(MK_HOP_ATTN_PARTIAL, { fa, fa->src[0], kc, vc, fa->src[3] });
    w.push(MK_HOP_ATTN_COMBINE, { gm, fa, oh, oh->src[0], sig, cont, q_mm });

    ggml_tensor * res = nullptr;
    if (!mk_match_mmvq_add(w, gm, false, x, &res)) {
        return false;
    }
    return mk_match_ffn(w, res, out);
}

static void mk_begin_layer(mk_walker & w, mk_layer_kind kind, int32_t layer) {
    w.layer = layer;
    w.m.layers.push_back({ kind, (int32_t) w.m.ops.size(), (int32_t) w.m.ops.size() });
}

static void mk_end_layer(mk_walker & w) {
    w.m.layers.back().op_end = (int32_t) w.m.ops.size();
}

// head norm, out_ids gather, LM head (plain, or the truncated draft vocabulary: lo rows, -inf fill, tail rows)
static bool mk_match_head(mk_walker & w, const ggml_tensor * x) {
    ggml_tensor * xn = nullptr;
    if (!mk_match_norm(w, x, MK_HOP_RMSNORM_F32, &xn)) {
        return false;
    }
    MK_TAKE(gr, GGML_OP_GET_ROWS, "out_ids gather");
    MK_CHECK(gr->src[0] == xn && gr->src[1]->type == GGML_TYPE_I32 && gr->ne[1] == w.m.n_tokens, "out_ids gather must keep all T rows");
    w.push(MK_HOP_OUT_ROWS, { gr, xn, gr->src[1] });
    w.push(MK_HOP_QUANTIZE_Q8_1, { gr });

    MK_TAKE(lo, GGML_OP_MUL_MAT, "lm head");
    MK_CHECK(lo->src[1] == gr, "lm head input");
    if (!mk_check_weight(w, lo)) {
        return false;
    }
    w.push(MK_HOP_MMVQ, { lo, lo->src[0], gr });
    if (w.peek() == nullptr) {
        return true;
    }

    MK_CHECK(w.m.mtp, "trailing nodes after the lm head");
    ggml_tensor * mid_src = w.peek();
    MK_CHECK(mid_src->op == GGML_OP_CONT || mid_src->op == GGML_OP_PAD, "truncated draft head: expected CONT or PAD");
    w.pos++;
    MK_TAKE(fill, GGML_OP_FILL, "-inf filler");
    MK_TAKE(cat, GGML_OP_CONCAT, "lo + filler");
    MK_CHECK(mk_root(mid_src->src[0]) == lo && fill->src[0] == mid_src && cat->src[0] == lo && cat->src[1] == fill &&
        ggml_get_op_params_i32(cat, 0) == 0, "truncated draft head wiring");
    w.push(MK_HOP_FILL_NEG_INF, { fill, mid_src, lo });
    w.push(MK_HOP_CONCAT, { cat, lo, fill });
    if (w.peek() == nullptr) {
        return true;
    }
    MK_TAKE(hi, GGML_OP_MUL_MAT, "lm head tail rows");
    MK_TAKE(cat2, GGML_OP_CONCAT, "head + tail rows");
    MK_CHECK(hi->src[1] == gr && cat2->src[0] == cat && cat2->src[1] == hi && ggml_get_op_params_i32(cat2, 0) == 0,
        "truncated draft head tail wiring");
    if (!mk_check_weight(w, hi)) {
        return false;
    }
    w.push(MK_HOP_MMVQ, { hi, hi->src[0], gr });
    w.push(MK_HOP_CONCAT, { cat2, cat, hi });
    return w.peek() == nullptr || w.fail("trailing nodes after the lm head");
}

static bool mk_match_graph_impl(mk_walker & w) {
    mk_match & m = w.m;
    mk_begin_layer(w, MK_LAYER_INPUT, -1);

    const ggml_tensor * x = nullptr;
    ggml_tensor * first = w.peek();
    MK_CHECK(first != nullptr, "empty graph");
    if (first->op == GGML_OP_GET_ROWS && first->src[0]->op == GGML_OP_NONE && first->src[1]->type == GGML_TYPE_I32) {
        w.pos++;
        w.push(MK_HOP_EMBED_ROWS, { first, first->src[0], first->src[1] });
    }

    ggml_tensor * n0 = w.peek();
    MK_CHECK(n0 != nullptr && n0->op == GGML_OP_RMS_NORM, "expected the first layer norm");
    m.n_tokens = (int32_t) n0->src[0]->ne[1];
    MK_CHECK(m.n_tokens >= 1 && m.n_tokens <= 8, mk_format("T = %d is outside the decode/verify band 1..8", m.n_tokens));
    MK_CHECK(n0->src[0]->ne[2] == 1 && n0->src[0]->ne[3] == 1, "multi-sequence batch");

    m.mtp = w.peek(2) != nullptr && w.peek(2)->op == GGML_OP_RMS_NORM;
    if (m.mtp) {
        ggml_tensor * e = nullptr;
        ggml_tensor * h = nullptr;
        if (!mk_match_norm(w, nullptr, MK_HOP_RMSNORM_F32, &e) || !mk_match_norm(w, nullptr, MK_HOP_RMSNORM_F32, &h)) {
            return false;
        }
        MK_TAKE(cat, GGML_OP_CONCAT, "mtp concat");
        MK_CHECK(cat->src[0] == e && cat->src[1] == h && ggml_get_op_params_i32(cat, 0) == 0, "mtp concat wiring");
        w.push(MK_HOP_CONCAT, { cat, e, h });
        ggml_tensor * eh = nullptr;
        if (!mk_match_mmvq(w, cat, false, &eh)) {
            return false;
        }
        x = eh;
    } else {
        x = n0->src[0];
        if (!mk_check_rows(w, x, "layer 0 input")) {
            return false;
        }
    }
    mk_end_layer(w);

    for (int32_t il = 0;; ++il) {
        const ggml_tensor * n2 = w.peek(2);
        MK_CHECK(n2 != nullptr, "graph ends inside a layer");
        if (n2->op == GGML_OP_GET_ROWS && mk_root(n2->src[0]) == w.peek(1)) {
            break;
        }
        ggml_tensor * y = nullptr;
        if (n2->op == GGML_OP_GET_ROWS) {
            MK_CHECK(!m.mtp, "recurrent layer in the draft graph");
            mk_begin_layer(w, MK_LAYER_GDN, il);
            if (!mk_match_gdn_layer(w, x, &y)) {
                return false;
            }
        } else {
            MK_CHECK(!m.mtp || il == 0, "draft graph with more than one layer");
            mk_begin_layer(w, MK_LAYER_ATTN, il);
            if (!mk_match_attn_block(w, x, &y)) {
                return false;
            }
        }
        mk_end_layer(w);
        x = y;
    }

    mk_begin_layer(w, MK_LAYER_OUTPUT, -1);
    if (!mk_match_head(w, x)) {
        return false;
    }
    mk_end_layer(w);

    std::unordered_set<const ggml_tensor *> written;
    for (const mk_match_op & op : m.ops) {
        written.insert(op.t[0]);
    }
    for (const mk_match_op & op : m.ops) {
        for (int k = 1; k < MK_MATCH_MAX_T && op.t[k] != nullptr; ++k) {
            if (op.t[k]->op != GGML_OP_NONE && (op.t[k]->flags & GGML_TENSOR_FLAG_OUTPUT) && written.count(op.t[k]) == 0) {
                m.reason = mk_format("layer %d: graph output '%s' is an intermediate of %s", op.layer, op.t[k]->name,
                        mk_hop_name(op.kind));
                return false;
            }
        }
    }
    return true;
}

mk_match mk_match_graph(ggml_cgraph * cgraph) {
    mk_match m;
    mk_walker w(cgraph, m);
    m.ok = mk_match_graph_impl(w);
    if (!m.ok) {
        m.ops.clear();
        m.layers.clear();
    }
    return m;
}

std::string mk_match_dump(const mk_match & m) {
    if (!m.ok) {
        return "MK no match: " + m.reason + "\n";
    }
    std::string s = mk_format("MK match: %s T=%d n_kv=%lld ops=%zu layers=%zu\n", m.mtp ? "mtp" : "trunk", m.n_tokens,
            (long long) m.n_kv, m.ops.size(), m.layers.size());
    for (const mk_match_op & op : m.ops) {
        s += mk_format("MK L%3d %-15s", op.layer, mk_hop_name(op.kind));
        for (int k = 0; k < MK_MATCH_MAX_T && op.t[k] != nullptr; ++k) {
            const ggml_tensor * t = op.t[k];
            s += mk_format(" %s=%s:%s[%lld,%lld,%lld,%lld]%s", mk_hop_slot_name(op.kind, k), ggml_op_name(t->op), t->name,
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], ggml_type_name(t->type));
        }
        s += "\n";
    }
    return s;
}

#ifndef GGML_MK_HOST_ONLY

#include <memory>
#include <unordered_map>

//
// emission
//

struct mk_emit_ctx {
    mk_stream_builder &                          b;
    const mk_match &                             match;
    const mk_launch_params *                     launch;     // device
    std::unordered_map<const void *, mk_counter> producers;  // keyed by the view root's data

    static const void * key(const ggml_tensor * t) {
        return t->view_src != nullptr ? t->view_src->data : t->data;
    }

    std::vector<mk_wait> waits_for(std::initializer_list<const ggml_tensor *> inputs) const {
        std::vector<mk_wait> waits;
        for (const ggml_tensor * t : inputs) {
            const auto it = producers.find(key(t));
            if (it != producers.end()) {
                waits.push_back(b.wait_all(it->second));
            }
        }
        return waits;
    }

    void set_producer(const ggml_tensor * t, mk_counter c) {
        producers[key(t)] = c;
    }
};

static bool mk_emit_not_implemented(mk_emit_ctx & ctx, const mk_match_op & op) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(op);
    return false;
}

static bool mk_emit_embed_rows    (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_rmsnorm_q8_1  (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_rmsnorm_f32   (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_quantize_q8_1 (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_mmvq          (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_mmvq_add      (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_mmvq_glu      (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_state_copy    (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_gdn_conv      (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_gdn_gates     (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_gdn_step      (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_gdn_out_gate  (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_attn_prep_q   (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_attn_prep_k   (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_v_had_set_rows(mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_attn_partial  (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_attn_combine  (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_out_rows      (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_concat        (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }
static bool mk_emit_fill_neg_inf  (mk_emit_ctx & ctx, const mk_match_op & op) { return mk_emit_not_implemented(ctx, op); }

typedef bool (*mk_emit_fn)(mk_emit_ctx & ctx, const mk_match_op & op);

static mk_emit_fn mk_emit_hook(mk_hop kind) {
    switch (kind) {
        case MK_HOP_EMBED_ROWS:      return mk_emit_embed_rows;
        case MK_HOP_RMSNORM_Q8_1:    return mk_emit_rmsnorm_q8_1;
        case MK_HOP_RMSNORM_F32:     return mk_emit_rmsnorm_f32;
        case MK_HOP_QUANTIZE_Q8_1:   return mk_emit_quantize_q8_1;
        case MK_HOP_MMVQ:            return mk_emit_mmvq;
        case MK_HOP_MMVQ_ADD:        return mk_emit_mmvq_add;
        case MK_HOP_MMVQ_GLU:        return mk_emit_mmvq_glu;
        case MK_HOP_STATE_COPY:      return mk_emit_state_copy;
        case MK_HOP_GDN_CONV:        return mk_emit_gdn_conv;
        case MK_HOP_GDN_GATES:       return mk_emit_gdn_gates;
        case MK_HOP_GDN_STEP:        return mk_emit_gdn_step;
        case MK_HOP_GDN_OUT_GATE:    return mk_emit_gdn_out_gate;
        case MK_HOP_ATTN_PREP_Q:     return mk_emit_attn_prep_q;
        case MK_HOP_ATTN_PREP_K:     return mk_emit_attn_prep_k;
        case MK_HOP_V_HAD_SET_ROWS:  return mk_emit_v_had_set_rows;
        case MK_HOP_ATTN_PARTIAL:    return mk_emit_attn_partial;
        case MK_HOP_ATTN_COMBINE:    return mk_emit_attn_combine;
        case MK_HOP_OUT_ROWS:        return mk_emit_out_rows;
        case MK_HOP_CONCAT:          return mk_emit_concat;
        case MK_HOP_FILL_NEG_INF:    return mk_emit_fill_neg_inf;
        default:                 return nullptr;
    }
}

//
// launch hooks
//

typedef bool (*mk_external_fn)(ggml_backend_cuda_context & ctx, const mk_match_op & op, ggml_cuda_mk_run_node_fn run_node);

static bool mk_external_embed_rows(ggml_backend_cuda_context & ctx, const mk_match_op & op, ggml_cuda_mk_run_node_fn run_node) {
    return run_node(ctx, const_cast<ggml_tensor *>(op.t[0]));
}

static bool mk_external_attn_partial(ggml_backend_cuda_context & ctx, const mk_match_op & op, ggml_cuda_mk_run_node_fn run_node) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(op);
    GGML_UNUSED(run_node);
    return false;
}

static mk_external_fn mk_external_hook(mk_hop kind) {
    switch (kind) {
        case MK_HOP_EMBED_ROWS:   return mk_external_embed_rows;
        case MK_HOP_ATTN_PARTIAL: return mk_external_attn_partial;
        default:                  return nullptr;
    }
}

static bool mk_launch_segment(ggml_backend_cuda_context & ctx, const mk_stream_desc & desc) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(desc);
    return false;
}

//
// stream cache
//

struct mk_device_segment {
    bool           external    = false;
    int32_t        external_id = -1;
    void *         buf         = nullptr;
    mk_stream_desc desc        = {};
};

struct mk_cached_stream {
    bool                           ok = false;
    std::string                    reason;
    std::vector<mk_device_segment> segments;
    mk_launch_params *             d_launch = nullptr;
    uint32_t                       epoch    = 0;

    ~mk_cached_stream() {
        for (mk_device_segment & s : segments) {
            if (s.buf != nullptr) {
                CUDA_CHECK(cudaFree(s.buf));
            }
        }
        if (d_launch != nullptr) {
            CUDA_CHECK(cudaFree(d_launch));
        }
    }
};

struct ggml_cuda_mk_state {
    uint64_t                                                           last_uid    = 0;
    mk_match                                                           last_match;
    mk_cached_stream *                                                 last_stream = nullptr;
    std::unordered_map<std::string, std::unique_ptr<mk_cached_stream>> streams;
};

static constexpr size_t   mk_cache_max_streams = 32;
static constexpr uint64_t mk_watchdog_cycles   = 100ull * 1000 * 1000;

template <typename T>
static void mk_key_append(std::string & key, const T & v) {
    key.append((const char *) &v, sizeof(v));
}

// The KV view shapes follow n_kv, which the stream reads from mk_launch_params; the mask feeds the external step.
static std::string mk_cache_key(const mk_match & m) {
    std::string key;
    key.reserve(m.ops.size() * MK_MATCH_MAX_T * 48);
    mk_key_append(key, m.n_tokens);
    for (const mk_match_op & op : m.ops) {
        mk_key_append(key, op.kind);
        for (int k = 0; k < MK_MATCH_MAX_T; ++k) {
            const ggml_tensor * t = op.t[k];
            if (t == nullptr || (op.kind == MK_HOP_ATTN_PARTIAL && k == 4)) {
                continue;
            }
            mk_key_append(key, t->data);
            mk_key_append(key, t->type);
            if (t->op == GGML_OP_NONE || (op.kind == MK_HOP_ATTN_PARTIAL && k >= 2)) {
                continue;
            }
            mk_key_append(key, t->op);
            mk_key_append(key, t->ne);
            mk_key_append(key, t->nb);
            mk_key_append(key, t->op_params);
        }
    }
    return key;
}

static void * mk_upload_segment(const mk_stream_host & s, mk_stream_desc & desc, const mk_launch_params * d_launch) {
    auto align = [](size_t x) { return GGML_PAD(x, 256); };
    const size_t n_counters = s.signals_per_pass.size();

    const size_t off_instrs   = 0;
    const size_t off_queue    = align(off_instrs + s.instrs.size() * sizeof(mk_instr));
    const size_t off_params   = align(off_queue + s.queue_begin.size() * sizeof(int32_t));
    const size_t off_signals  = align(off_params + s.params.size());
    const size_t off_counters = align(off_signals + n_counters * sizeof(uint32_t));
    const size_t off_error    = align(off_counters + n_counters * sizeof(uint32_t));
    const size_t size         = align(off_error + sizeof(int32_t));

    std::vector<uint8_t> host(size, 0);
    memcpy(host.data() + off_instrs, s.instrs.data(), s.instrs.size() * sizeof(mk_instr));
    memcpy(host.data() + off_queue, s.queue_begin.data(), s.queue_begin.size() * sizeof(int32_t));
    memcpy(host.data() + off_params, s.params.data(), s.params.size());
    memcpy(host.data() + off_signals, s.signals_per_pass.data(), n_counters * sizeof(uint32_t));

    void * buf = nullptr;
    CUDA_CHECK(cudaMalloc(&buf, size));
    CUDA_CHECK(cudaMemcpy(buf, host.data(), size, cudaMemcpyHostToDevice));

    uint8_t * d = (uint8_t *) buf;
    desc.instrs           = (const mk_instr *) (d + off_instrs);
    desc.queue_begin      = (const int32_t *) (d + off_queue);
    desc.params           = d + off_params;
    desc.signals_per_pass = (const uint32_t *) (d + off_signals);
    desc.counters         = (uint32_t *) (d + off_counters);
    desc.error            = (int32_t *) (d + off_error);
    desc.launch           = d_launch;
    desc.n_blocks         = s.n_blocks;
    desc.n_counters       = (int32_t) n_counters;
    desc.watchdog_cycles  = mk_watchdog_cycles;
    return buf;
}

static std::unique_ptr<mk_cached_stream> mk_build(const mk_match & m) {
    auto cs = std::make_unique<mk_cached_stream>();
    CUDA_CHECK(cudaMalloc((void **) &cs->d_launch, sizeof(mk_launch_params)));
    CUDA_CHECK(cudaMemset(cs->d_launch, 0, sizeof(mk_launch_params)));

    mk_stream_builder b;
    mk_emit_ctx       ectx{ b, m, cs->d_launch, {} };
    for (size_t i = 0; i < m.ops.size(); ++i) {
        const mk_match_op & op = m.ops[i];
        if (mk_hop_is_external(op.kind)) {
            b.add_external((int32_t) i);
            continue;
        }
        const mk_emit_fn emit = mk_emit_hook(op.kind);
        if (emit == nullptr || !emit(ectx, op)) {
            cs->reason = mk_format("no emitter for %s (layer %d)", mk_hop_name(op.kind), op.layer);
            return cs;
        }
    }

    std::vector<mk_segment> segments = b.finish();
    for (size_t i = 0; i < segments.size(); ++i) {
        if (segments[i].external) {
            continue;
        }
        const std::string err = mk_validate_stream(segments[i].stream);
        if (!err.empty()) {
            cs->reason = mk_format("segment %zu: %s", i, err.c_str());
            return cs;
        }
    }
    for (const mk_segment & seg : segments) {
        mk_device_segment ds;
        ds.external    = seg.external;
        ds.external_id = seg.external_id;
        if (!seg.external) {
            ds.buf = mk_upload_segment(seg.stream, ds.desc, cs->d_launch);
        }
        cs->segments.push_back(ds);
    }
    cs->ok = true;
    return cs;
}

bool ggml_cuda_mk_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("GGML_HIP_MEGAKERNEL");
        return e != nullptr && atoi(e) == 1;
    }();
    return enabled;
}

static bool mk_dump_enabled() {
    static const bool enabled = getenv("GGML_HIP_MEGAKERNEL_DUMP") != nullptr;
    return enabled;
}

bool ggml_cuda_mk_try_compute(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, ggml_cuda_mk_run_node_fn run_node) {
    if (!ctx.stream_context().concurrent_events.empty()) {
        return false;
    }
    if (ctx.mk_state == nullptr) {
        ctx.mk_state = new ggml_cuda_mk_state();
    }
    ggml_cuda_mk_state & st = *ctx.mk_state;

    if (cgraph->uid == 0 || cgraph->uid != st.last_uid) {
        st.last_uid    = cgraph->uid;
        st.last_match  = mk_match_graph(cgraph);
        st.last_stream = nullptr;
        if (mk_dump_enabled()) {
            fputs(mk_match_dump(st.last_match).c_str(), stderr);
        }
        if (st.last_match.ok) {
            const std::string key = mk_cache_key(st.last_match);
            auto it = st.streams.find(key);
            if (it == st.streams.end()) {
                if (st.streams.size() >= mk_cache_max_streams) {
                    st.streams.erase(st.streams.begin());
                }
                it = st.streams.emplace(key, mk_build(st.last_match)).first;
                if (!it->second->ok && mk_dump_enabled()) {
                    fprintf(stderr, "MK fallback: %s\n", it->second->reason.c_str());
                }
            }
            st.last_stream = it->second.get();
        }
    }
    const mk_match & m = st.last_match;
    if (st.last_stream == nullptr || !st.last_stream->ok) {
        return false;
    }
    mk_cached_stream & cs = *st.last_stream;

    mk_launch_params lp = {};
    lp.epoch    = cs.epoch;
    lp.n_kv     = (int32_t) m.n_kv;
    lp.n_tokens = m.n_tokens;
    lp.kv_head  = -1;
    CUDA_CHECK(cudaMemcpyAsync(cs.d_launch, &lp, sizeof(lp), cudaMemcpyHostToDevice, ctx.stream()));

    // external steps only write graph intermediates; a megakernel segment also updates the caches in place
    bool caches_written = false;
    for (size_t i = 0; i < cs.segments.size(); ++i) {
        const mk_device_segment & seg = cs.segments[i];
        bool launched;
        if (seg.external) {
            const mk_match_op & op = m.ops[seg.external_id];
            const mk_external_fn fn = mk_external_hook(op.kind);
            launched = fn != nullptr && fn(ctx, op, run_node);
        } else {
            launched = mk_launch_segment(ctx, seg.desc);
            caches_written |= launched;
        }
        if (!launched) {
            if (!caches_written) {
                return false;
            }
            GGML_ABORT("megakernel: step %zu of %zu failed to launch after earlier steps ran", i, cs.segments.size());
        }
    }
    cs.epoch++;
    return true;
}

void ggml_cuda_mk_release(ggml_backend_cuda_context * ctx) {
    delete ctx->mk_state;
    ctx->mk_state = nullptr;
}

#endif // GGML_MK_HOST_ONLY
