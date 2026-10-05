// Host-only tests for the megakernel stream builder, validator and graph matcher. No GPU is touched.
// Optional argument: a full GGML_CUDA_OP_SEQ_TMP dump (RAW + SEQ lines) to match instead of the bundled fixture.

#define GGML_MK_HOST_ONLY
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "../ggml/src/ggml-cuda/megakernel-host.cu"

#include <chrono>
#include <cstdio>
#include <deque>
#include <fstream>
#include <map>
#include <set>

static int n_failed = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
            n_failed++;                                                   \
        }                                                                 \
    } while (0)

static bool contains(const std::string & s, const char * needle) {
    return s.find(needle) != std::string::npos;
}

static mk_instr instr(uint16_t opcode, int32_t wait_counter, uint32_t wait_target, int32_t signal_counter, uint32_t params_off = 0) {
    mk_instr i = {};
    i.opcode         = opcode;
    i.tile_end       = 1;
    i.wait_counter   = wait_counter;
    i.wait_target    = wait_target;
    i.signal_counter = signal_counter;
    i.params_off     = params_off;
    i.prefetch_off   = UINT32_MAX;
    return i;
}

// queues[b] is block b's queue; signals_per_pass counts the signalling instructions.
static mk_stream_host make_stream(const std::vector<std::vector<mk_instr>> & queues, int32_t n_counters, size_t params_size = 64) {
    mk_stream_host s;
    s.n_blocks = (int32_t) queues.size();
    s.queue_begin.push_back(0);
    for (const auto & q : queues) {
        s.instrs.insert(s.instrs.end(), q.begin(), q.end());
        s.queue_begin.push_back((int32_t) s.instrs.size());
    }
    s.params.resize(params_size, 0);
    s.signals_per_pass.assign(n_counters, 0);
    for (const mk_instr & i : s.instrs) {
        if (i.signal_counter >= 0 && i.signal_counter < n_counters) {
            s.signals_per_pass[i.signal_counter]++;
        }
    }
    return s;
}

static mk_op_desc op(uint16_t opcode, int32_t n_tiles, uint32_t params_off, std::vector<mk_wait> waits = {}, int32_t tiles_per_step = 1) {
    mk_op_desc d;
    d.opcode         = opcode;
    d.n_tiles        = n_tiles;
    d.tiles_per_step = tiles_per_step;
    d.params_off     = params_off;
    d.waits          = std::move(waits);
    return d;
}

static void test_builder_chain() {
    mk_stream_builder b;
    const uint32_t p0 = b.add_params(uint64_t(1));
    const uint32_t p1 = b.add_params(uint8_t(2));
    CHECK(p0 == 0 && p1 == 16);

    const mk_counter c0 = b.add_op(op(MK_OP_RMSNORM_Q8_1, 1, p0));
    const mk_counter c1 = b.add_op(op(MK_OP_MMVQ, 100, p1, { b.wait_all(c0) }));
    const mk_counter c2 = b.add_op(op(MK_OP_MMVQ, 10, p1, { b.wait_all(c1) }, 4));
    b.add_op(op(MK_OP_RMSNORM_Q8_1, 1, p0, { b.wait_all(c2) }));

    std::vector<mk_segment> segs = b.finish();
    CHECK(segs.size() == 1 && !segs[0].external);
    const mk_stream_host & s = segs[0].stream;
    CHECK(s.signals_per_pass.size() == 4);
    CHECK(s.signals_per_pass[c0.id] == 1);
    CHECK(s.signals_per_pass[c1.id] == MK_N_BLOCKS);
    CHECK(s.signals_per_pass[c2.id] == 3);
    CHECK(s.params.size() == 32);

    int32_t covered = 0;
    for (const mk_instr & i : s.instrs) {
        if (i.signal_counter == c1.id) {
            covered += i.tile_end - i.tile_begin;
            CHECK(i.wait_counter == c0.id && i.wait_target == 1);
        }
        if (i.signal_counter == c2.id) {
            CHECK(i.tile_begin % 4 == 0 && i.wait_target == MK_N_BLOCKS);
        }
    }
    CHECK(covered == 100);
    CHECK(mk_validate_stream(s).empty());
}

static void test_builder_multi_wait_and_segments() {
    mk_stream_builder b;
    const uint32_t p = b.add_params(uint32_t(0));
    const mk_counter a  = b.add_op(op(MK_OP_MMVQ, 7, p));
    const mk_counter bb = b.add_op(op(MK_OP_MMVQ, 60, p));
    const mk_counter c  = b.add_op(op(MK_OP_GDN_STEP, 5, p, { b.wait_all(a), b.wait_all(bb) }));

    b.add_external(3);
    const mk_counter d = b.add_op(op(MK_OP_ATTN_COMBINE, 9, p, { b.wait_all(c) }));
    b.add_op(op(MK_OP_MMVQ, 9, p, { b.wait_all(d) }));

    std::vector<mk_segment> segs = b.finish();
    CHECK(segs.size() == 3);
    CHECK(!segs[0].external && segs[1].external && segs[1].external_id == 3 && !segs[2].external);

    int n_nop = 0;
    for (const mk_instr & i : segs[0].stream.instrs) {
        if (i.opcode == MK_OP_NOP) {
            n_nop++;
            CHECK(i.wait_counter == a.id && i.wait_target == 7);
        }
        if (i.opcode == MK_OP_GDN_STEP) {
            CHECK(i.wait_counter == bb.id && i.wait_target == MK_N_BLOCKS);
        }
    }
    CHECK(n_nop == 5);
    CHECK(mk_validate_stream(segs[0].stream).empty());

    for (const mk_instr & i : segs[2].stream.instrs) {
        if (i.opcode == MK_OP_ATTN_COMBINE) {
            CHECK(i.wait_counter == -1);
        }
    }
    CHECK(d.segment == 2 && d.id == 0);
    CHECK(mk_validate_stream(segs[2].stream).empty());
}

static void test_partial_wait_satisfied_by_others() {
    // block 0 waits for one signal of c0 and signals c0 itself later; block 1 supplies the one it needs
    const mk_stream_host s = make_stream({
        { instr(MK_OP_MMVQ, 0, 1, 0) },
        { instr(MK_OP_MMVQ, -1, 0, 0) },
    }, 1);
    CHECK(mk_validate_stream(s).empty());
}

static void test_deadlock_cycle() {
    const mk_stream_host s = make_stream({
        { instr(MK_OP_MMVQ, 1, 1, 0) },
        { instr(MK_OP_MMVQ, 2, 1, 1) },
        { instr(MK_OP_MMVQ, 0, 1, 2) },
        { instr(MK_OP_MMVQ, -1, 0, -1) },
    }, 3);
    const std::string err = mk_validate_stream(s);
    CHECK(contains(err, "deadlock cycle"));
    CHECK(contains(err, "block 0 #0 waits counter 1 at 0/1"));
    CHECK(contains(err, "block 2 #0 waits counter 0 at 0/1"));
}

static void test_deadlock_cycle_behind_progress() {
    // both blocks make progress before the cycle closes
    const mk_stream_host s = make_stream({
        { instr(MK_OP_MMVQ, -1, 0, 2), instr(MK_OP_MMVQ, 1, 1, 0) },
        { instr(MK_OP_MMVQ, 2, 1, -1), instr(MK_OP_MMVQ, 0, 1, 1) },
    }, 3);
    const std::string err = mk_validate_stream(s);
    CHECK(contains(err, "deadlock cycle: block 0 #1"));
    CHECK(contains(err, "block 1 #1"));
}

static void test_self_wait() {
    const mk_stream_host s = make_stream({
        { instr(MK_OP_MMVQ, 0, 1, -1), instr(MK_OP_MMVQ, -1, 0, 0) },
        { instr(MK_OP_NOP, -1, 0, -1) },
    }, 1);
    const std::string err = mk_validate_stream(s);
    CHECK(contains(err, "block 0 #0: self-wait"));

    // the waiting instruction's own signal also lands behind its wait
    const mk_stream_host s2 = make_stream({ { instr(MK_OP_MMVQ, 0, 1, 0) } }, 1);
    CHECK(contains(mk_validate_stream(s2), "self-wait"));
}

static void test_out_of_range() {
    CHECK(contains(mk_validate_stream(make_stream({ { instr(MK_OP_MMVQ, 5, 1, -1) } }, 1)), "wait counter 5 out of range"));
    CHECK(contains(mk_validate_stream(make_stream({ { instr(MK_OP_MMVQ, -1, 0, 3) } }, 1)), "signal counter 3 out of range"));
    CHECK(contains(mk_validate_stream(make_stream({ { instr(MK_OP_MMVQ, -2, 0, -1) } }, 1)), "wait counter -2"));
    CHECK(contains(mk_validate_stream(make_stream({ { instr(MK_OP_MMVQ, -1, 0, -1, 8) } }, 0)), "not 16-byte aligned"));
    CHECK(contains(mk_validate_stream(make_stream({ { instr(MK_OP_MMVQ, -1, 0, -1, 64) } }, 0)), "params offset 64 out of range"));
    CHECK(mk_validate_stream(make_stream({ { instr(MK_OP_NOP, -1, 0, -1, 4096) } }, 0)).empty());
    CHECK(contains(mk_validate_stream(make_stream({ { instr(MK_OP_COUNT, -1, 0, -1) } }, 0)), "opcode"));

    mk_instr bad_tiles = instr(MK_OP_MMVQ, -1, 0, -1);
    bad_tiles.tile_begin = 5;
    bad_tiles.tile_end   = 2;
    CHECK(contains(mk_validate_stream(make_stream({ { bad_tiles } }, 0)), "bad tile range"));

    mk_stream_host q = make_stream({ { instr(MK_OP_MMVQ, -1, 0, -1) } }, 0);
    q.queue_begin.back() = 7;
    CHECK(contains(mk_validate_stream(q), "queue_begin"));

    CHECK(contains(mk_validate_stream(make_stream({ { instr(MK_OP_EXIT, -1, 0, -1), instr(MK_OP_NOP, -1, 0, -1) } }, 0)), "EXIT before"));
}

static void test_target_above_signals_per_pass() {
    const mk_stream_host s = make_stream({
        { instr(MK_OP_MMVQ, -1, 0, 0) },
        { instr(MK_OP_MMVQ, 0, 2, -1) },
    }, 1);
    CHECK(contains(mk_validate_stream(s), "wait target 2 on counter 0 exceeds its signals_per_pass 1"));

    mk_stream_host m = make_stream({ { instr(MK_OP_MMVQ, -1, 0, 0) } }, 1);
    m.signals_per_pass[0] = 2;
    CHECK(contains(mk_validate_stream(m), "signals_per_pass is 2 but 1 instructions signal it"));
}

static void test_large_stream_speed() {
    mk_stream_builder b;
    const uint32_t p = b.add_params(uint32_t(0));
    mk_counter prev = b.add_op(op(MK_OP_RMSNORM_Q8_1, 1, p));
    mk_counter prev2 = prev;
    for (int i = 0; i < 400; ++i) {
        const mk_counter c = b.add_op(op(MK_OP_MMVQ, 1000 + i, p, { b.wait_all(prev), b.wait_all(prev2) }, 2));
        prev2 = prev;
        prev  = c;
    }
    std::vector<mk_segment> segs = b.finish();
    const auto t0 = std::chrono::steady_clock::now();
    const std::string err = mk_validate_stream(segs[0].stream);
    const auto t1 = std::chrono::steady_clock::now();
    CHECK(err.empty());
    CHECK(segs[0].stream.instrs.size() > 30000);
    printf("validated %zu instructions in %.2f ms\n", segs[0].stream.instrs.size(),
        std::chrono::duration<double, std::milli>(t1 - t0).count());
}


//
// matcher, on graphs rebuilt from the RAW lines of an op sequence dump
//

// The dump lacks op params, leaf types and src3+; those follow the Qwen3.5/3.8 graph conventions below.
struct dump_graph {
    std::deque<ggml_tensor>               tensors;
    std::map<std::string, ggml_tensor *>  leafs;
    std::vector<ggml_tensor *>            nodes;
    std::vector<int>                      seq;   // node indices of the executed (post-fusion) nodes
    ggml_cgraph                           graph = {};
};

static std::string trim(const std::string & s) {
    const size_t e = s.find_last_not_of(' ');
    return e == std::string::npos ? "" : s.substr(0, e + 1);
}

static ggml_op op_from_name(const std::string & name) {
    for (int o = 0; o < GGML_OP_COUNT; ++o) {
        if (name == ggml_op_name((ggml_op) o)) {
            return (ggml_op) o;
        }
    }
    return GGML_OP_COUNT;
}

static ggml_type type_from_name(const std::string & name) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n != nullptr && name == n) {
            return (ggml_type) t;
        }
    }
    return GGML_TYPE_COUNT;
}

static ggml_tensor * new_tensor(dump_graph & g, const std::string & name, ggml_op op, ggml_type type, const int64_t ne[4]) {
    g.tensors.emplace_back();
    ggml_tensor * t = &g.tensors.back();
    *t = {};
    t->op   = op;
    t->type = type;
    snprintf(t->name, sizeof(t->name), "%s", name.c_str());
    for (int i = 0; i < 4; ++i) {
        t->ne[i] = ne[i];
    }
    t->nb[0] = ggml_type_size(type);
    t->nb[1] = ggml_row_size(type, ne[0]);
    t->nb[2] = t->nb[1]*ne[1];
    t->nb[3] = t->nb[2]*ne[2];
    t->data  = (void *) (uintptr_t) (0x1000 * g.tensors.size());
    return t;
}

static ggml_type leaf_type(const std::string & n) {
    auto has = [&](const char * s) { return n.find(s) != std::string::npos; };
    if (has("cache_k_l") || has("cache_v_l")) return GGML_TYPE_Q5_0;
    if (has("k_idxs") || has("v_idxs"))       return GGML_TYPE_I64;
    if (has("s_copy") || has("out_ids") || has("inp_pos")) return GGML_TYPE_I32;
    if (has("kq_mask"))                       return GGML_TYPE_F16;
    if (has("output.weight"))                 return GGML_TYPE_Q6_K;
    if (has(".weight") && !has("norm") && !has("ssm_alpha") && !has("ssm_beta") && !has("ssm_conv1d")) return GGML_TYPE_IQ4_XS;
    return GGML_TYPE_F32;
}

static ggml_tensor * leaf(dump_graph & g, const std::string & name, const int64_t ne[4]) {
    auto it = g.leafs.find(name);
    if (it != g.leafs.end()) {
        return it->second;
    }
    ggml_tensor * t = new_tensor(g, name, GGML_OP_NONE, leaf_type(name), ne);
    g.leafs[name] = t;
    return t;
}

// Names repeat in the dump; a src is the latest node of that name with no consumer yet, else the latest one.
static ggml_tensor * resolve(dump_graph & g, const std::string & name, std::map<const ggml_tensor *, int> & uses) {
    ggml_tensor * any = nullptr;
    for (auto it = g.nodes.rbegin(); it != g.nodes.rend(); ++it) {
        if (name == (*it)->name) {
            if (uses[*it] == 0) {
                return *it;
            }
            if (any == nullptr) {
                any = *it;
            }
        }
    }
    return any;
}

static ggml_tensor * latest(dump_graph & g, const std::string & prefix) {
    for (auto it = g.nodes.rbegin(); it != g.nodes.rend(); ++it) {
        if (std::string((*it)->name).rfind(prefix, 0) == 0) {
            return *it;
        }
    }
    return nullptr;
}

static bool parse_dump(std::istream & in, dump_graph & g) {
    std::map<const ggml_tensor *, int> uses;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("SEQ", 0) == 0) {
            g.seq.push_back(atoi(line.c_str() + 3));
            continue;
        }
        if (line.rfind("RAW", 0) != 0) {
            continue;
        }
        size_t p = 3;
        while (line[p] == ' ') p++;
        const int idx = atoi(line.c_str() + p);
        while (line[p] != ' ') p++;
        while (line[p] == ' ') p++;
        const size_t op_begin = p;
        while (line[p] != ' ') p++;
        const std::string op_name = line.substr(op_begin, p - op_begin);
        const size_t name_begin = op_begin + std::max<size_t>(14, op_name.size()) + 1;
        const size_t shape_begin = line.find(" [", name_begin);
        const size_t shape_end   = line.find("] ", shape_begin);
        const size_t arrow       = line.find(" <- ", shape_end);
        if (idx != (int) g.nodes.size() || shape_begin == std::string::npos || shape_end == std::string::npos || arrow == std::string::npos) {
            fprintf(stderr, "bad RAW line: %s\n", line.c_str());
            return false;
        }
        const std::string name = trim(line.substr(name_begin, shape_begin - name_begin));
        long long ne[4];
        sscanf(line.c_str() + shape_begin + 2, "%lld,%lld,%lld,%lld", &ne[0], &ne[1], &ne[2], &ne[3]);
        const int64_t ne64[4] = { ne[0], ne[1], ne[2], ne[3] };
        const std::string type_name = trim(line.substr(shape_end + 2, arrow - shape_end - 2));

        std::vector<std::string> srcs;
        std::string rest = line.substr(arrow + 4);
        for (size_t sep; (sep = rest.find(" | ")) != std::string::npos; rest = rest.substr(sep + 3)) {
            srcs.push_back(rest.substr(0, sep));
        }
        srcs.push_back(trim(rest));

        const ggml_op op = op_from_name(op_name);
        const ggml_type type = type_from_name(type_name);
        if (op == GGML_OP_COUNT || type == GGML_TYPE_COUNT) {
            fprintf(stderr, "unknown op or type: %s\n", line.c_str());
            return false;
        }
        ggml_tensor * t = new_tensor(g, name, op, type, ne64);
        t->flags = GGML_TENSOR_FLAG_COMPUTE;
        if (name == "h_nextn" || name == "result_output") {
            t->flags |= GGML_TENSOR_FLAG_OUTPUT;
        }
        const bool is_rot = op == GGML_OP_MUL_MAT && srcs[0].find("_rot") != std::string::npos;

        for (int s = 0; s < (int) srcs.size() && s < 3; ++s) {
            std::string sn = srcs[s];
            if (sn == "-") {
                continue;
            }
            const bool is_leaf = sn.size() > 3 && sn.compare(sn.size() - 3, 3, "(w)") == 0;
            if (is_leaf) {
                sn = sn.substr(0, sn.size() - 3);
            }
            ggml_tensor * src = is_leaf ? nullptr : resolve(g, sn, uses);
            if (src == nullptr) {
                int64_t lne[4] = { 1, 1, 1, 1 };
                if (op == GGML_OP_MUL_MAT && s == 0) {
                    lne[0] = lne[1] = ne[0];
                } else if (op == GGML_OP_MUL || op == GGML_OP_ADD) {
                    lne[0] = ne[0];
                    if (sn.find("input_embed") != std::string::npos) {
                        lne[1] = ne[1];
                    }
                } else if (op == GGML_OP_SSM_CONV) {
                    lne[0] = 4;
                    lne[1] = ne[0];
                } else if (op == GGML_OP_RMS_NORM) {
                    lne[0] = ne[0];
                    lne[1] = ne[1];
                }
                src = leaf(g, sn, lne);
            }
            t->src[s] = src;
            uses[src]++;
        }
        // the weight's row length is only known once src1 is resolved
        if (op == GGML_OP_MUL_MAT && !is_rot && t->src[0]->op == GGML_OP_NONE) {
            t->src[0]->ne[0] = t->src[1]->ne[0];
        }
        if (op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE) {
            t->view_src = t->src[0]->view_src != nullptr ? t->src[0]->view_src : t->src[0];
        }
        if (op == GGML_OP_UNARY) {
            const ggml_unary_op u = name.find("softplus") != std::string::npos ? GGML_UNARY_OP_SOFTPLUS
                                  : name.find("sigmoid") != std::string::npos ? GGML_UNARY_OP_SIGMOID : GGML_UNARY_OP_SILU;
            t->op_params[0] = (int32_t) u;
        }
        if (op == GGML_OP_GLU) {
            t->op_params[0] = (int32_t) GGML_GLU_OP_SWIGLU;
        }
        if (is_rot) {
            t->op_params[1] = GGML_HINT_SRC0_IS_HADAMARD;
        }
        if (op == GGML_OP_GATED_DELTA_NET) {
            t->src[3] = latest(g, "gate-");
            t->src[4] = latest(g, "beta_sigmoid-");
            t->src[5] = latest(g, "state_predelta-");
        }
        if (op == GGML_OP_FLASH_ATTN_EXT) {
            const int64_t mne[4] = { t->src[1]->ne[1], 1, 1, 1 };
            t->src[3] = leaf(g, "attn_inp_kq_mask", mne);
        }
        g.nodes.push_back(t);
    }
    g.graph.n_nodes = (int) g.nodes.size();
    g.graph.size    = g.graph.n_nodes;
    g.graph.nodes   = g.nodes.data();
    return !g.nodes.empty();
}

static bool load_dump(const char * path, dump_graph & g) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    return parse_dump(f, g);
}

static std::vector<mk_hop> layer_kinds(const mk_match & m, size_t il) {
    std::vector<mk_hop> kinds;
    for (int32_t i = m.layers[il].op_begin; i < m.layers[il].op_end; ++i) {
        kinds.push_back(m.ops[i].kind);
    }
    return kinds;
}

static void check_seq_covered(const dump_graph & g, const mk_match & m) {
    std::set<const ggml_tensor *> claimed;
    for (const mk_match_op & op : m.ops) {
        for (const ggml_tensor * t : op.t) {
            if (t != nullptr) {
                claimed.insert(t);
            }
        }
    }
    int missing = 0;
    for (int idx : g.seq) {
        if (idx < 0 || idx >= (int) g.nodes.size() || claimed.count(g.nodes[idx]) == 0) {
            fprintf(stderr, "executed node %d is not claimed by any matched op\n", idx);
            missing++;
        }
    }
    CHECK(missing == 0);
}

static void test_match_decode_dump(const char * path, bool full) {
    dump_graph g;
    if (!load_dump(path, g)) {
        CHECK(false);
        return;
    }
    const mk_match m = mk_match_graph(&g.graph);
    if (!m.ok) {
        fprintf(stderr, "%s", mk_match_dump(m).c_str());
        CHECK(false);
        return;
    }
    CHECK(!m.mtp && m.n_tokens == 1 && m.n_kv == 768);
    check_seq_covered(g, m);

    const std::vector<mk_hop> gdn = {
        MK_HOP_RMSNORM_Q8_1, MK_HOP_MMVQ, MK_HOP_GDN_CONV, MK_HOP_GDN_GATES, MK_HOP_GDN_STEP, MK_HOP_MMVQ, MK_HOP_GDN_OUT_GATE, MK_HOP_MMVQ_ADD,
        MK_HOP_RMSNORM_Q8_1, MK_HOP_MMVQ_GLU, MK_HOP_QUANTIZE_Q8_1, MK_HOP_MMVQ_ADD,
    };
    const std::vector<mk_hop> attn = {
        MK_HOP_RMSNORM_Q8_1, MK_HOP_MMVQ, MK_HOP_ATTN_PREP_Q, MK_HOP_MMVQ, MK_HOP_V_HAD_SET_ROWS, MK_HOP_MMVQ,
        MK_HOP_ATTN_PREP_K, MK_HOP_ATTN_PARTIAL, MK_HOP_ATTN_COMBINE, MK_HOP_QUANTIZE_Q8_1, MK_HOP_MMVQ_ADD,
        MK_HOP_RMSNORM_Q8_1, MK_HOP_MMVQ_GLU, MK_HOP_QUANTIZE_Q8_1, MK_HOP_MMVQ_ADD,
    };
    const std::vector<mk_hop> head = { MK_HOP_RMSNORM_F32, MK_HOP_OUT_ROWS, MK_HOP_QUANTIZE_Q8_1, MK_HOP_MMVQ };

    const size_t n_layers = full ? 64 : 4;
    CHECK(m.layers.size() == n_layers + 2);
    CHECK(m.layers.front().kind == MK_LAYER_INPUT && m.layers.front().op_begin == m.layers.front().op_end);
    for (size_t il = 1; il <= n_layers && il < m.layers.size(); ++il) {
        const bool is_attn = il % 4 == 0;
        CHECK(m.layers[il].kind == (is_attn ? MK_LAYER_ATTN : MK_LAYER_GDN));
        CHECK(layer_kinds(m, il) == (is_attn ? attn : gdn));
    }
    CHECK(m.layers.back().kind == MK_LAYER_OUTPUT && layer_kinds(m, m.layers.size() - 1) == head);
    printf("matched %s: %zu ops in %zu layers\n", path, m.ops.size(), m.layers.size() - 2);
}

static void test_match_rejects() {
    dump_graph g;
    if (!load_dump(MK_FIXTURE, g)) {
        CHECK(false);
        return;
    }
    auto find = [&](const char * name) {
        for (ggml_tensor * t : g.nodes) {
            if (std::string(t->name) == name) {
                return t;
            }
        }
        return (ggml_tensor *) nullptr;
    };

    ggml_tensor * qkv = find("node_13");
    const ggml_type saved = qkv->src[0]->type;
    qkv->src[0]->type = GGML_TYPE_Q4_0;
    mk_match m = mk_match_graph(&g.graph);
    CHECK(!m.ok && contains(m.reason, "unsupported type q4_0"));
    qkv->src[0]->type = saved;

    ggml_tensor * scale = find("q_conv_predelta-0");
    scale->op = GGML_OP_MUL;
    m = mk_match_graph(&g.graph);
    CHECK(!m.ok && contains(m.reason, "layer 0") && contains(m.reason, "expected SCALE"));
    scale->op = GGML_OP_SCALE;

    ggml_tensor * zero = find("cache_r_l1 (reshaped) (view) (view)");
    zero->ne[0] = 30720;
    m = mk_match_graph(&g.graph);
    CHECK(!m.ok && contains(m.reason, "layer 1"));
    zero->ne[0] = 0;

    ggml_tensor * xn = find("ffn_gate-0");
    xn->flags |= GGML_TENSOR_FLAG_OUTPUT;
    m = mk_match_graph(&g.graph);
    CHECK(!m.ok && contains(m.reason, "graph output 'ffn_gate-0' is an intermediate"));
    xn->flags &= ~GGML_TENSOR_FLAG_OUTPUT;

    ggml_tensor * in = g.nodes[0]->src[0];
    in->ne[1] = 9;
    m = mk_match_graph(&g.graph);
    CHECK(!m.ok && contains(m.reason, "T = 9"));
    in->ne[1] = 1;

    CHECK(mk_match_graph(&g.graph).ok);
}

int main(int argc, char ** argv) {
    test_builder_chain();
    test_builder_multi_wait_and_segments();
    test_partial_wait_satisfied_by_others();
    test_deadlock_cycle();
    test_deadlock_cycle_behind_progress();
    test_self_wait();
    test_out_of_range();
    test_target_above_signals_per_pass();
    test_large_stream_speed();
    test_match_decode_dump(MK_FIXTURE, false);
    test_match_rejects();
    if (argc > 1) {
        test_match_decode_dump(argv[1], true);
    }

    if (n_failed != 0) {
        fprintf(stderr, "%d checks failed\n", n_failed);
        return 1;
    }
    printf("all megakernel host tests passed\n");
    return 0;
}
