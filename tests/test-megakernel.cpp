// Persistent decode megakernel: ordering, epoch reuse, watchdog, soak and timing. HIP only, compiled as HIP.
// Usage: test-megakernel [-n soak_launches] [--timing] [--ffn]

#include "megakernel.cuh"
#include "mk-ops-test.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <functional>
#include <string>
#include <vector>

#define TEST_THREADS (mk_test_add<MK_THREADS>::threads)

struct mk_host_stream {
    int                                n_blocks;
    std::vector<std::vector<mk_instr>> queues;
    std::vector<uint8_t>               params;
    std::vector<uint32_t>              signals_per_pass;

    explicit mk_host_stream(int n_blocks) : n_blocks(n_blocks), queues(n_blocks) {}

    template <typename P>
    uint32_t add_params(const P & p) {
        const size_t off = GGML_PAD(params.size(), MK_PARAM_ALIGN);
        params.resize(off + sizeof(P));
        memcpy(params.data() + off, &p, sizeof(P));
        return (uint32_t) off;
    }

    int add_counter() {
        signals_per_pass.push_back(0);
        return (int) signals_per_pass.size() - 1;
    }

    void push(int block, uint16_t opcode, int tile_begin, int tile_end, int wait_counter, uint32_t wait_target,
              int signal_counter, uint32_t params_off = 0) {
        mk_instr in = {};
        in.opcode         = opcode;
        in.tile_begin     = tile_begin;
        in.tile_end       = tile_end;
        in.wait_counter   = wait_counter;
        in.wait_target    = wait_target;
        in.signal_counter = signal_counter;
        in.params_off     = params_off;
        in.prefetch_off   = UINT32_MAX;
        queues[block].push_back(in);
        if (signal_counter >= 0) {
            signals_per_pass[signal_counter]++;
        }
    }

    void push_add(int block, int tile_begin, int tile_end, int wait_counter, uint32_t wait_target, int signal_counter,
                  float * y, const float * x, int n, int delay) {
        const mk_test_add_params p = { y, x, n, delay };
        push(block, MK_OP_TEST_ADD, tile_begin, tile_end, wait_counter, wait_target, signal_counter, add_params(p));
    }
};

// Device copy of a host stream. launch[e].epoch == e, so a launch selects its epoch through desc.launch alone.
struct mk_dev_stream {
    int n_epochs;

    mk_stream_desc     desc = {};
    mk_instr         * instrs = nullptr;
    int32_t          * queue_begin = nullptr;
    uint8_t          * params = nullptr;
    uint32_t         * spp = nullptr;
    uint32_t         * counters = nullptr;
    int32_t          * error = nullptr;
    mk_launch_params * launch = nullptr;

    mk_dev_stream(const mk_host_stream & h, uint64_t watchdog_ticks, int n_epochs) : n_epochs(std::max(n_epochs, 1)) {
        std::vector<mk_instr> flat;
        std::vector<int32_t>  qb = { 0 };
        for (const auto & q : h.queues) {
            flat.insert(flat.end(), q.begin(), q.end());
            qb.push_back((int32_t) flat.size());
        }
        std::vector<mk_launch_params> lp(this->n_epochs);
        for (int e = 0; e < this->n_epochs; ++e) {
            lp[e] = { (uint32_t) e, 0, 1, 0 };
        }
        const size_t n_counters = h.signals_per_pass.size();

        CUDA_CHECK(hipMalloc(&instrs,      std::max<size_t>(flat.size(), 1)*sizeof(mk_instr)));
        CUDA_CHECK(hipMalloc(&queue_begin, qb.size()*sizeof(int32_t)));
        CUDA_CHECK(hipMalloc(&params,      std::max<size_t>(h.params.size(), 1)));
        CUDA_CHECK(hipMalloc(&spp,         std::max<size_t>(n_counters, 1)*sizeof(uint32_t)));
        CUDA_CHECK(hipMalloc(&counters,    std::max<size_t>(n_counters, 1)*sizeof(uint32_t)));
        CUDA_CHECK(hipMalloc(&error,       sizeof(int32_t)));
        CUDA_CHECK(hipMalloc(&launch,      lp.size()*sizeof(mk_launch_params)));
        CUDA_CHECK(hipMemcpy(instrs, flat.data(), flat.size()*sizeof(mk_instr), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(queue_begin, qb.data(), qb.size()*sizeof(int32_t), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(params, h.params.data(), h.params.size(), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(spp, h.signals_per_pass.data(), n_counters*sizeof(uint32_t), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemset(counters, 0, std::max<size_t>(n_counters, 1)*sizeof(uint32_t)));
        CUDA_CHECK(hipMemset(error, 0, sizeof(int32_t)));
        CUDA_CHECK(hipMemcpy(launch, lp.data(), lp.size()*sizeof(mk_launch_params), hipMemcpyHostToDevice));

        desc.instrs           = instrs;
        desc.queue_begin      = queue_begin;
        desc.params           = params;
        desc.signals_per_pass = spp;
        desc.counters         = counters;
        desc.error            = error;
        desc.launch           = launch;
        desc.n_blocks         = h.n_blocks;
        desc.n_counters       = (int32_t) n_counters;
        desc.watchdog_cycles  = watchdog_ticks;
    }

    mk_dev_stream(const mk_dev_stream &) = delete;
    mk_dev_stream & operator=(const mk_dev_stream &) = delete;

    ~mk_dev_stream() {
        CUDA_CHECK(hipFree(instrs));
        CUDA_CHECK(hipFree(queue_begin));
        CUDA_CHECK(hipFree(params));
        CUDA_CHECK(hipFree(spp));
        CUDA_CHECK(hipFree(counters));
        CUDA_CHECK(hipFree(error));
        CUDA_CHECK(hipFree(launch));
    }
};

// Single launch point for every megakernel launch in this test; stream validation goes here.
static void mk_run(const mk_dev_stream & s, uint32_t epoch, hipStream_t stream) {
    GGML_ASSERT(epoch < (uint32_t) s.n_epochs);
    mk_stream_desc d = s.desc;
    d.launch = s.launch + epoch;
    ggml_cuda_mk_launch(d, stream);
}

// A hung megakernel cannot be recovered from the host, so report it and stop the process.
static void sync_bounded(hipStream_t stream, double timeout_ms, const char * what) {
    const auto t0 = std::chrono::steady_clock::now();
    while (true) {
        const hipError_t err = hipStreamQuery(stream);
        if (err == hipSuccess) {
            return;
        }
        if (err != hipErrorNotReady) {
            CUDA_CHECK(err);
        }
        if (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() > timeout_ms) {
            fprintf(stderr, "HANG: %s did not finish within %.0f ms\n", what, timeout_ms);
            fflush(stderr);
            std::_Exit(2);
        }
    }
}

template <typename F>
static float gpu_time_ms(hipStream_t stream, double timeout_ms, const char * name, F && f) {
    hipEvent_t e0, e1;
    CUDA_CHECK(hipEventCreate(&e0));
    CUDA_CHECK(hipEventCreate(&e1));
    CUDA_CHECK(hipEventRecord(e0, stream));
    f();
    CUDA_CHECK(hipEventRecord(e1, stream));
    sync_bounded(stream, timeout_ms, name);
    float ms = 0.0f;
    CUDA_CHECK(hipEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(hipEventDestroy(e0));
    CUDA_CHECK(hipEventDestroy(e1));
    return ms;
}

static uint64_t ticks_per_ms() {
    int khz = 0;
    CUDA_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, ggml_cuda_get_device()));
    return (uint64_t) khz;
}

static int n_wgp() {
    return ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
}

static float init_value(int buf, int i) {
    return (float) ((buf*31 + i) % 17);
}

struct buffers {
    int                n;
    int                count;
    float            * d = nullptr;
    std::vector<float> init;

    buffers(int n, int count) : n(n), count(count), init((size_t) n*count) {
        for (int b = 0; b < count; ++b) {
            for (int i = 0; i < n; ++i) {
                init[(size_t) b*n + i] = init_value(b, i);
            }
        }
        CUDA_CHECK(hipMalloc(&d, init.size()*sizeof(float)));
    }
    ~buffers() { CUDA_CHECK(hipFree(d)); }

    float * operator[](int b) const { return d + (size_t) b*n; }

    void reset(hipStream_t stream) {
        CUDA_CHECK(hipMemcpyAsync(d, init.data(), init.size()*sizeof(float), hipMemcpyHostToDevice, stream));
    }

    std::vector<float> read(hipStream_t stream) const {
        std::vector<float> out(init.size());
        CUDA_CHECK(hipMemcpyAsync(out.data(), d, out.size()*sizeof(float), hipMemcpyDeviceToHost, stream));
        sync_bounded(stream, 10000, "readback");
        return out;
    }
};

static bool check_adds(const buffers & bufs, const std::vector<float> & got, const std::vector<std::pair<int, int>> & adds,
                       const char * name) {
    std::vector<float> ref = bufs.init;
    for (const auto & [y, x] : adds) {
        for (int i = 0; i < bufs.n; ++i) {
            ref[(size_t) y*bufs.n + i] += ref[(size_t) x*bufs.n + i];
        }
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        if (memcmp(&ref[i], &got[i], sizeof(float)) != 0) {
            fprintf(stderr, "%s: mismatch at buf %zu elem %zu: got %f want %f\n", name, i / bufs.n, i % bufs.n, got[i], ref[i]);
            return false;
        }
    }
    return true;
}

static bool expect_error(hipStream_t stream, const mk_dev_stream & s, int32_t want, const char * name) {
    const int32_t err = ggml_cuda_mk_take_error(s.error, stream);
    if (err != want) {
        fprintf(stderr, "%s: error flag %d, want %d\n", name, err, want);
        return false;
    }
    if (err != MK_ERR_NONE) {
        ggml_cuda_mk_reset_counters(s.desc, stream);
    }
    return true;
}

// Chain: instruction k adds buffer k into buffer k+1 after k-1 signals, spread across blocks.
// 4 tiles per instruction on 8 sub-tiles per step, the last tile partial.
struct stream_case {
    buffers                          bufs;
    mk_host_stream                   h;
    std::vector<std::pair<int, int>> adds;

    stream_case(int n, int count, int n_blocks) : bufs(n, count), h(n_blocks) {}
};

struct chain_test : stream_case {
    static constexpr int K = 101;

    explicit chain_test(int n_blocks) : stream_case(3*TEST_THREADS + 17, K + 1, n_blocks) {
        const int n_tiles = (bufs.n + TEST_THREADS - 1) / TEST_THREADS;
        int prev = -1;
        for (int k = 0; k < K; ++k) {
            const int c = h.add_counter();
            h.push_add((k*7) % n_blocks, 0, n_tiles, prev, 1, c, bufs[k + 1], bufs[k], bufs.n, 64);
            adds.push_back({ k + 1, k });
            prev = c;
        }
    }
};

// Fan-out over every block, then fan-in on one block, R rounds. Every queue ends with EXIT and then an
// unsatisfiable wait, which only runs if EXIT is ignored.
struct dag_test : stream_case {
    static constexpr int R   = 8;
    static constexpr int TPB = 3;

    explicit dag_test(int n_blocks) : stream_case(n_blocks*TPB*TEST_THREADS - 5, 2*R + 1, n_blocks) {
        const int n_tiles = (bufs.n + TEST_THREADS - 1) / TEST_THREADS;
        int fin = -1;
        int src = 0;
        for (int r = 0; r < R; ++r) {
            const int a   = 1 + 2*r;
            const int b   = 2 + 2*r;
            const int out = h.add_counter();
            for (int blk = 0; blk < n_blocks; ++blk) {
                const int t0 = std::min(blk*TPB, n_tiles);
                const int t1 = std::min(t0 + TPB, n_tiles);
                h.push_add(blk, t0, t1, fin, 1, out, bufs[a], bufs[src], bufs.n, 32);
            }
            adds.push_back({ a, src });
            fin = h.add_counter();
            h.push_add((r*13 + 5) % n_blocks, 0, n_tiles, out, n_blocks, fin, bufs[b], bufs[a], bufs.n, 0);
            adds.push_back({ b, a });
            src = b;
        }
        const int never = h.add_counter();
        for (int blk = 0; blk < n_blocks; ++blk) {
            h.push(blk, MK_OP_EXIT, 0, 0, -1, 0, -1);
            h.push(blk, MK_OP_NOP, 0, 0, never, 1, -1);
        }
    }
};

static bool run_passes(stream_case & t, hipStream_t stream, int passes, uint64_t watchdog, const char * name) {
    mk_dev_stream s(t.h, watchdog, passes);
    for (int e = 0; e < passes; ++e) {
        t.bufs.reset(stream);
        mk_run(s, (uint32_t) e, stream);
        sync_bounded(stream, 10000, name);
        if (!expect_error(stream, s, MK_ERR_NONE, name) || !check_adds(t.bufs, t.bufs.read(stream), t.adds, name)) {
            fprintf(stderr, "%s: failed at pass %d\n", name, e);
            return false;
        }
    }
    printf("%s: %d passes OK\n", name, passes);
    return true;
}

static bool test_watchdog(hipStream_t stream, int n_blocks) {
    const uint64_t tpm = ticks_per_ms();
    bool ok = true;

    {
        mk_host_stream h(n_blocks);
        const int never = h.add_counter();
        h.push(0, MK_OP_NOP, 0, 0, never, 1, -1);
        mk_dev_stream s(h, 50*tpm, 1);
        mk_run(s, 0, stream);
        sync_bounded(stream, 5000, "watchdog");
        ok = expect_error(stream, s, MK_ERR_WATCHDOG, "watchdog") && ok;
    }

    // Block 1 starts waiting at 120 ms, so it ends at 200 ms only if it sees block 0's error.
    {
        mk_host_stream h(n_blocks);
        const int never = h.add_counter();
        const mk_test_spin_params spin = { 120*tpm };
        h.push(0, MK_OP_NOP, 0, 0, never, 1, -1);
        h.push(1, MK_OP_TEST_SPIN, 0, 1, -1, 0, -1, h.add_params(spin));
        h.push(1, MK_OP_NOP, 0, 0, never, 1, -1);
        mk_dev_stream s(h, 200*tpm, 1);
        const float ms = gpu_time_ms(stream, 5000, "watchdog-propagate", [&] { mk_run(s, 0, stream); });
        ok = expect_error(stream, s, MK_ERR_WATCHDOG, "watchdog-propagate") && ok;
        if (ms > 290.0f) {
            fprintf(stderr, "watchdog-propagate: kernel took %.1f ms, other blocks did not see the error\n", ms);
            ok = false;
        }
    }

    {
        mk_host_stream h(n_blocks);
        h.push(0, 0x7777, 0, 1, -1, 0, -1);
        mk_dev_stream s(h, 50*tpm, 1);
        mk_run(s, 0, stream);
        sync_bounded(stream, 5000, "bad-opcode");
        ok = expect_error(stream, s, MK_ERR_BAD_OPCODE, "bad-opcode") && ok;
    }

    printf("watchdog: %s\n", ok ? "OK" : "FAILED");
    return ok;
}

template <typename T>
static __global__ void __launch_bounds__(T::threads) k_mk_wrap(const mk_op_params<T> p, int variant) {
    __shared__ __align__(16) char lds[T::lds_bytes > 0 ? T::lds_bytes : 1];
    T::run(p, variant, blockIdx.x, true, lds);
}

// Chain of K full-width instructions with a full dependency between them, as K back to back kernel launches would have.
static void test_timing(hipStream_t stream, int n_blocks) {
    constexpr int K     = 256;
    constexpr int iters = 50;
    using op = mk_test_add<TEST_THREADS>;

    buffers bufs(n_blocks*TEST_THREADS, K + 1);
    const uint64_t watchdog = 1000*ticks_per_ms();

    auto build = [&](mk_host_stream & h, int k0, int k1) {
        int prev = -1;
        for (int k = k0; k < k1; ++k) {
            const int c = h.add_counter();
            for (int blk = 0; blk < n_blocks; ++blk) {
                h.push_add(blk, blk, blk + 1, prev, n_blocks, c, bufs[k + 1], bufs[k], bufs.n, 0);
            }
            prev = c;
        }
    };
    mk_host_stream h1(n_blocks), h2a(n_blocks), h2b(n_blocks);
    build(h1, 0, K);
    build(h2a, 0, K/2);
    build(h2b, K/2 + 1, K);
    mk_dev_stream s1(h1, watchdog, iters + 2), s2a(h2a, watchdog, iters + 2), s2b(h2b, watchdog, iters + 2);

    auto normal = [&](int k) {
        const mk_test_add_params p = { bufs[k + 1], bufs[k], bufs.n, 0 };
        k_mk_wrap<op><<<n_blocks, op::threads, 0, stream>>>(p, 0);
    };

    uint32_t epoch1 = 0, epoch2 = 0;
    auto mk_one = [&] { mk_run(s1, epoch1++, stream); };
    auto mk_two = [&] {
        mk_run(s2a, epoch2, stream);
        normal(K/2);
        mk_run(s2b, epoch2, stream);
        epoch2++;
    };
    auto eager = [&] { for (int k = 0; k < K; ++k) { normal(k); } };
    // Same chain with NOP instructions: the interpreter's own per-hand-off cost.
    mk_host_stream hn(n_blocks);
    {
        int prev = -1;
        for (int k = 0; k < K; ++k) {
            const int c = hn.add_counter();
            for (int blk = 0; blk < n_blocks; ++blk) {
                hn.push(blk, MK_OP_NOP, 0, 1, prev, n_blocks, c);
            }
            prev = c;
        }
    }
    mk_dev_stream sn(hn, watchdog, iters + 2);
    uint32_t epochn = 0;
    auto mk_nop = [&] { mk_run(sn, epochn++, stream); };

    auto time_us = [&](auto && f, const char * name) {
        f();
        sync_bounded(stream, 10000, name);
        const float ms = gpu_time_ms(stream, 60000, name, [&] { for (int i = 0; i < iters; ++i) { f(); } });
        printf("timing: %-36s %9.1f us per pass, %6.2f us per instruction\n", name, 1000.0f*ms/iters, 1000.0f*ms/iters/K);
    };

    auto time_graph = [&](auto && f, const char * name) {
        hipGraph_t     graph = nullptr;
        hipGraphExec_t exec  = nullptr;
        CUDA_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
        f();
        const hipError_t err_end = hipStreamEndCapture(stream, &graph);
        const hipError_t err_ins = err_end == hipSuccess ? hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0) : err_end;
        if (err_ins != hipSuccess) {
            printf("timing: %-36s capture failed: %s\n", name, hipGetErrorString(err_ins));
            (void) hipGetLastError();
        } else {
    time_us([&] { CUDA_CHECK(hipGraphLaunch(exec, stream)); }, name);
        }
        if (exec) {
            CUDA_CHECK(hipGraphExecDestroy(exec));
        }
        if (graph) {
            CUDA_CHECK(hipGraphDestroy(graph));
        }
    };

    time_us(eager, "normal launches, eager");
    time_us(mk_nop, "megakernel, NOP chain");
    time_graph(eager, "normal launches, HIP graph");
    time_us(mk_one, "megakernel, 1 segment");
    time_us(mk_two, "megakernel, 2 segments + 1 kernel");

    // A captured graph replays one epoch, so its counters are reset before each replay.
    time_graph([&] { ggml_cuda_mk_reset_counters(s2a.desc, stream); ggml_cuda_mk_reset_counters(s2b.desc, stream);
                     mk_run(s2a, 0, stream); normal(K/2); mk_run(s2b, 0, stream); },
               "megakernel, 2 segments, HIP graph");

    const int32_t err = std::max({ ggml_cuda_mk_take_error(s1.error, stream), ggml_cuda_mk_take_error(s2a.error, stream),
                                   ggml_cuda_mk_take_error(s2b.error, stream) });
    printf("timing: error flags %s\n", err == MK_ERR_NONE ? "clear" : "SET");
}

#if __has_include("mk-ops-ffn.cuh")
#include "mk-ops-ffn.cuh"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

// Qwen3.8-27B FFN block at T = 1: rms_norm_q8_1 -> mmvq gate/up + GLU -> mmvq down + residual add.
// The reference is the regular ggml-cuda path (3 launches); the megakernel stream must match it bit for bit.
struct ffn_case {
    static constexpr int n_embd = 5120;
    static constexpr int n_ff   = 17408;

    ggml_backend_t        backend = nullptr;
    ggml_context        * ctx     = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    ggml_cgraph         * gf      = nullptr;
    ggml_tensor * x = nullptr, * norm_w = nullptr, * gate = nullptr, * up = nullptr, * down = nullptr, * out = nullptr;

    ffn_case() {
        backend = ggml_backend_cuda_init(ggml_cuda_get_device());
        ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ctx    = ggml_init(ip);
        x      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        gate   = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_XS, n_embd, n_ff);
        up     = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_XS, n_embd, n_ff);
        down   = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_XS, n_ff, n_embd);
        ggml_tensor * cur = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), norm_w);
        ggml_tensor * h   = ggml_swiglu_split(ctx, ggml_mul_mat(ctx, gate, cur), ggml_mul_mat(ctx, up, cur));
        out = ggml_add(ctx, ggml_mul_mat(ctx, down, h), x);
        gf  = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

        std::vector<float> v(n_embd);
        for (int i = 0; i < n_embd; ++i) {
            v[i] = 0.01f*(float) ((i*37) % 101 - 50);
        }
        ggml_backend_tensor_set(x, v.data(), 0, ggml_nbytes(x));
        for (int i = 0; i < n_embd; ++i) {
            v[i] = 1.0f + 0.001f*(float) (i % 13);
        }
        ggml_backend_tensor_set(norm_w, v.data(), 0, ggml_nbytes(norm_w));
        fill_quant(gate);
        fill_quant(up);
        fill_quant(down);
    }

    ~ffn_case() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
    }

    // Quantizes 64 pseudo-random rows and tiles them over the whole matrix.
    static void fill_quant(ggml_tensor * t) {
        const int64_t ncols = t->ne[0];
        const int64_t rows  = 64;
        std::vector<float> f((size_t) (ncols*rows));
        uint32_t s = 12345u + (uint32_t) ncols;
        for (float & e : f) {
            s = s*1664525u + 1013904223u;
            e = (float) ((int) (s >> 9) % 2001 - 1000) * 1e-3f;
        }
        const size_t row_size = ggml_row_size(t->type, ncols);
        std::vector<uint8_t> q(row_size*rows);
        ggml_quantize_chunk(t->type, f.data(), q.data(), 0, rows, ncols, nullptr);
        for (int64_t r = 0; r < t->ne[1]; r += rows) {
            const int64_t n = std::min(rows, t->ne[1] - r);
            ggml_backend_tensor_set(t, q.data(), r*row_size, n*row_size);
        }
    }
};

// One instruction per recorded launch, each waiting for the whole previous op. Tiles go to blocks in
// contiguous runs rounded to the op's sub-tile step, so a block's steps stay full.
static bool build_stream_from_records(const std::vector<mk_recorded_op> & recs, int n_blocks, mk_host_stream & h) {
    int prev_counter = -1;
    uint32_t prev_signals = 0;
    for (const mk_recorded_op & r : recs) {
        if (r.n_tiles <= 0) {
            fprintf(stderr, "records: opcode %u has no megakernel form\n", r.opcode);
            return false;
        }
        const uint32_t off = (uint32_t) GGML_PAD(h.params.size(), MK_PARAM_ALIGN);
        h.params.resize(off + r.params.size());
        memcpy(h.params.data() + off, r.params.data(), r.params.size());

        const int counter = h.add_counter();
        uint32_t signals = 0;
        static const int chunk = getenv("MK_CHUNK") ? atoi(getenv("MK_CHUNK")) : 0;
        if (chunk > 0) {
            // Round-robin chunks: at any moment the blocks work on neighbouring tiles, like hardware dispatch.
            for (int t0 = 0, b = 0; t0 < r.n_tiles; t0 += chunk, b = (b + 1) % n_blocks) {
                h.push(b, r.opcode, t0, std::min<int>(r.n_tiles, t0 + chunk), prev_counter, prev_signals, counter, off);
                h.queues[b].back().variant = r.variant;
                signals++;
            }
        } else {
            const int per_block = (r.n_tiles + n_blocks - 1) / n_blocks;
            for (int b = 0; b < n_blocks; ++b) {
                const int t0 = b*per_block;
                const int t1 = std::min<int>(r.n_tiles, t0 + per_block);
                if (t0 >= t1) {
                    break;
                }
                h.push(b, r.opcode, t0, t1, prev_counter, prev_signals, counter, off);
                h.queues[b].back().variant = r.variant;
                if (getenv("MK_PREFETCH") && r.opcode == MK_OP_MMVQ && prev_counter >= 0) {
                    // Weight rows of this block's tiles (and of the gate matrix for GLU), warmed while it waits.
                    mk_mmvq_params mp;
                    memcpy(&mp, r.params.data(), sizeof(mp));
                    const size_t row_bytes = (size_t) mp.stride_row_x*ggml_type_size(ggml_type(r.variant & 0xFF));
                    const mk_prefetch_desc pf = { (const char *) mp.vx + (size_t) t0*row_bytes, (uint64_t) (t1 - t0)*row_bytes };
                    h.queues[b].back().prefetch_off = h.add_params(pf);
                }
                signals++;
            }
        }
        prev_counter = counter;
        prev_signals = signals;
    }
    return true;
}

static const char * mk_opname(uint16_t opc) {
    switch (opc) {
        case MK_OP_RMSNORM_Q8_1: return "RMSNORM_Q8_1";
        case MK_OP_RMSNORM_F32:  return "RMSNORM_F32";
        case MK_OP_MMVQ:         return "MMVQ";
        case MK_OP_QUANTIZE_Q8_1:return "QUANTIZE_Q8_1";
        default:                 return "?";
    }
}


struct mk_param_blob { alignas(16) uint8_t b[256]; };

// Dynamic schedule: each sub-tile claims its next tile from a global counter (starting at 0),
// so waves that finish early take more work instead of idling through the static split's tail.
template <typename Op>
__global__ void __launch_bounds__(MK_THREADS, 1) k_split_dyn(const mk_op_params<Op> p, int n_tiles, uint32_t * claim) {
    __shared__ __align__(16) char lds[MK_THREADS/Op::threads*Op::lds_bytes + 16];
    constexpr int nsub = MK_THREADS / Op::threads;
    const int sub   = threadIdx.x / Op::threads;
    const int local = threadIdx.x % Op::threads;
    char * my_lds = lds + sub*Op::lds_bytes;

    while (true) {
        int tile = 0;
        if (local == 0) {
            tile = (int) __hip_atomic_fetch_add(claim, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        tile = __shfl(tile, 0, 32);
        if (tile >= n_tiles) {
            break;
        }
        if (p.ncols_x == 0xFFFFFFFFu) {
            continue; // never true; keeps the empty-body variant comparable
        }
        Op::block(p, tile, 0, 0, local % 32, local / 32, true, my_lds);
    }
}

// The claim loop alone, to tell a scheduling hang from an op hang.
__global__ void __launch_bounds__(MK_THREADS, 1) k_claim_only(int n_tiles, uint32_t * claim, uint32_t * done) {
    const int local = threadIdx.x % 32;
    while (true) {
        int tile = 0;
        if (local == 0) {
            tile = (int) __hip_atomic_fetch_add(claim, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        tile = __shfl(tile, 0, 32);
        if (tile >= n_tiles) {
            break;
        }
        if (local == 0) {
            __hip_atomic_fetch_add(done, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

// One recorded op as a plain 1024-thread kernel: same tile split as the stream builder, params by value
// in kernarg. Separates the persistent machinery from the 1024-block op structure.
__global__ void __launch_bounds__(MK_THREADS, 1) k_split_op(const mk_param_blob blob, int opcode, int variant, int n_tiles) {
    __shared__ __align__(16) char lds[MK_OP_LDS_BYTES];
    const int per_block = (n_tiles + (int) gridDim.x - 1) / (int) gridDim.x;
    const int tb = (int) blockIdx.x*per_block;
    const int te = min(n_tiles, tb + per_block);
    auto exec = [&](auto op) {
        using Op = decltype(op);
        using P  = mk_op_params<Op>;
        const P & p = *reinterpret_cast<const P *>(blob.b);
        constexpr int nsub = MK_THREADS / Op::threads;
        const int sub = threadIdx.x / Op::threads;
        for (int t0 = tb; t0 < te; t0 += nsub) {
            if (Op::lds_bytes > 0 && t0 != tb) {
                __syncthreads();
            }
            const int tile = __builtin_amdgcn_readfirstlane(t0 + sub);
            Op::run(p, variant, tile < te ? tile : te - 1, tile < te, lds + sub*Op::lds_bytes);
        }
    };
    switch (opcode) {
        case MK_OP_RMSNORM_Q8_1:  mk_rmsnorm_q8_1_dispatch<MK_THREADS>(variant, exec); break;
        case MK_OP_MMVQ:          mk_mmvq_dispatch<MK_THREADS>(variant, exec); break;
        case MK_OP_QUANTIZE_Q8_1: exec(mk_quantize_q8_1<MK_THREADS>{}); break;
        default: break;
    }
}

template <typename Op, int MINB>
__global__ void __launch_bounds__(Op::threads, MINB > 0 ? MINB : 1) k_direct(const mk_op_params<Op> p) {
    __shared__ __align__(16) char lds[Op::lds_bytes + 16];
    Op::block(p, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x % 32, threadIdx.x / 32, true, lds);
}

static bool test_ffn(hipStream_t stream, int n_blocks, bool timing) {
    ffn_case fc;

    std::vector<mk_recorded_op> recs;
    ggml_cuda_mk_set_recording(&recs);
    GGML_ASSERT(ggml_backend_graph_compute(fc.backend, fc.gf) == GGML_STATUS_SUCCESS);
    ggml_cuda_mk_set_recording(nullptr);
    ggml_backend_synchronize(fc.backend);

    std::vector<float> ref(ffn_case::n_embd);
    ggml_backend_tensor_get(fc.out, ref.data(), 0, ggml_nbytes(fc.out));

    printf("ffn: %zu recorded launches:", recs.size());
    for (const auto & r : recs) {
        printf(" %s(v=0x%x,%d)", mk_opname(r.opcode), r.variant, r.n_tiles);
    }
    printf("\n");

    // MK_FFN_OPS=i,j,... keeps only those recorded launches (timing experiments; the bit check then fails by design).
    if (const char * sel = getenv("MK_FFN_OPS")) {
        std::vector<mk_recorded_op> keep;
        for (const char * c = sel; *c; ) {
            keep.push_back(recs.at(strtol(c, (char **) &c, 10)));
            if (*c == ',') { c++; }
        }
        recs = keep;
    }
    mk_host_stream h(n_blocks);
    if (recs.empty() || !build_stream_from_records(recs, n_blocks, h)) {
        printf("ffn: FAILED (no usable records)\n");
        return false;
    }

    // The stream writes the same buffers as the graph: poison the output so a skipped write shows.
    std::vector<float> poison(ffn_case::n_embd, NAN);
    ggml_backend_tensor_set(fc.out, poison.data(), 0, ggml_nbytes(fc.out));
    ggml_backend_synchronize(fc.backend);

    constexpr int iters = 2000;
    constexpr int reps  = 3;
    // Counters are never reset, so every launch needs a fresh epoch.
    mk_dev_stream s(h, 1000*ticks_per_ms(), 1 + reps*iters + 300);
    // Counters are never reset, so epochs must be consecutive.
    uint32_t epoch = 0;
    mk_run(s, epoch++, stream);
    sync_bounded(stream, 10000, "ffn");
    bool ok = expect_error(stream, s, MK_ERR_NONE, "ffn");
    std::vector<float> got(ffn_case::n_embd);
    ggml_backend_tensor_get(fc.out, got.data(), 0, ggml_nbytes(fc.out));
    if (memcmp(got.data(), ref.data(), ggml_nbytes(fc.out)) != 0) {
        int nbad = 0;
        for (int i = 0; i < ffn_case::n_embd; ++i) {
            nbad += memcmp(&got[i], &ref[i], sizeof(float)) != 0;
        }
        fprintf(stderr, "ffn: megakernel output differs from the reference in %d of %d values (e.g. [0] %.9g vs %.9g)\n",
                nbad, ffn_case::n_embd, got[0], ref[0]);
        ok = false;
    }

    if (getenv("MK_TRACE")) {
        // Per instruction: wait start, op start, op end (wall clock). Printed per op as spread across blocks.
        size_t n_instr = 0;
        for (const auto & q : h.queues) { n_instr += q.size(); }
        uint64_t * tr_d = nullptr;
        CUDA_CHECK(hipMalloc(&tr_d, 3*n_instr*sizeof(uint64_t)));
        CUDA_CHECK(hipMemset(tr_d, 0, 3*n_instr*sizeof(uint64_t)));
        // Warm the clocks with back-to-back passes, then trace the last one.
        for (int i = 0; i < 300; ++i) {
            if (i == 299) {
                s.desc.trace = tr_d;
            }
            mk_run(s, epoch++, stream);
        }
        sync_bounded(stream, 10000, "ffn trace");
        s.desc.trace = nullptr;
        std::vector<uint64_t> tr(3*n_instr);
        CUDA_CHECK(hipMemcpy(tr.data(), tr_d, tr.size()*sizeof(uint64_t), hipMemcpyDeviceToHost));
        CUDA_CHECK(hipFree(tr_d));
        const double tpus = ticks_per_ms()/1000.0;
        uint64_t t_first = UINT64_MAX;
        for (size_t i = 0; i < n_instr; ++i) { t_first = std::min(t_first, tr[3*i]); }
        // group by signal counter = recorded op index
        std::map<int, std::vector<size_t>> by_op;
        size_t q = 0;
        for (const auto & qu : h.queues) { for (const auto & in : qu) { by_op[in.signal_counter].push_back(q++); } }
        for (auto & [op, idx] : by_op) {
            double ws_min = 1e30, ws_max = 0, os_min = 1e30, os_max = 0, oe_min = 1e30, oe_max = 0, dur_sum = 0, dur_max = 0;
            for (size_t i : idx) {
                const double w0 = (tr[3*i] - t_first)/tpus, o0 = (tr[3*i+1] - t_first)/tpus, o1 = (tr[3*i+2] - t_first)/tpus;
                ws_min = std::min(ws_min, w0); ws_max = std::max(ws_max, w0);
                os_min = std::min(os_min, o0); os_max = std::max(os_max, o0);
                oe_min = std::min(oe_min, o1); oe_max = std::max(oe_max, o1);
                dur_sum += o1 - o0; dur_max = std::max(dur_max, o1 - o0);
            }
            printf("trace op %d (%zu blocks): op start %.1f..%.1f us, op end %.1f..%.1f us, op time mean %.1f max %.1f us\n",
                   op, idx.size(), os_min, os_max, oe_min, oe_max, dur_sum/idx.size(), dur_max);
        }
    }
    if (getenv("MK_DYN")) {
        // GLU op (recs[1], IQ4_XS, one wave per row): reference-shaped kernel vs 1024-thread static vs dynamic.
        const mk_recorded_op & r = recs.at(1);
        mk_mmvq_params p0;
        memcpy(&p0, r.params.data(), sizeof(p0));
        using Op32 = mk_mmvq<32,   GGML_TYPE_IQ4_XS, 1, true, false, 0, true>;
        using OpK  = mk_mmvq<1024, GGML_TYPE_IQ4_XS, 1, true, false, 0, true>;
        uint32_t * claim; CUDA_CHECK(hipMalloc(&claim, 4));
        const int nt = r.n_tiles;
        float * dst = (float *) p0.dst;
        std::vector<float> a(nt), b(nt);
        k_direct<Op32, 1><<<nt, 32, 0, stream>>>(p0);
        sync_bounded(stream, 10000, "dyn ref");
        CUDA_CHECK(hipMemcpy(a.data(), dst, nt*4, hipMemcpyDeviceToHost));
        if (getenv("MK_DYN_CLAIM_ONLY")) {
            uint32_t * done; CUDA_CHECK(hipMalloc(&done, 4)); CUDA_CHECK(hipMemset(done, 0, 4)); CUDA_CHECK(hipMemset(claim, 0, 4));
            k_claim_only<<<n_blocks, MK_THREADS, 0, stream>>>(nt, claim, done);
            sync_bounded(stream, 3000, "claim only");
            uint32_t d = 0, c = 0; CUDA_CHECK(hipMemcpy(&d, done, 4, hipMemcpyDeviceToHost)); CUDA_CHECK(hipMemcpy(&c, claim, 4, hipMemcpyDeviceToHost));
            printf("dyn claim only: done %u of %d, claims %u\n", d, nt, c);
        }
        if (getenv("MK_DYN_SMALL")) {
            CUDA_CHECK(hipMemsetAsync(claim, 0, 4, stream));
            k_split_dyn<OpK><<<1, MK_THREADS, 0, stream>>>(p0, 64, claim);
            sync_bounded(stream, 3000, "dyn small");
            uint32_t c = 0; CUDA_CHECK(hipMemcpy(&c, claim, 4, hipMemcpyDeviceToHost));
            printf("dyn small: finished, claim counter %u\n", c);
        }
        CUDA_CHECK(hipMemsetAsync(claim, 0, 4, stream));
        k_split_dyn<OpK><<<n_blocks, MK_THREADS, 0, stream>>>(p0, nt, claim);
        sync_bounded(stream, 10000, "dyn");
        CUDA_CHECK(hipMemcpy(b.data(), dst, nt*4, hipMemcpyDeviceToHost));
        printf("dyn: %s\n", memcmp(a.data(), b.data(), nt*4) == 0 ? "bit-identical" : "DIFFERS");
        // Weights alone fit the infinity cache, so time the GLU op paired with the down op as in the FFN block.
        mk_param_blob bd = {}; memcpy(bd.b, recs.at(3).params.data(), recs.at(3).params.size());
        for (int rep = 0; rep < 3; ++rep) {
            const float ms_r = gpu_time_ms(stream, 60000, "dyn ref", [&] { for (int i = 0; i < 500; ++i) {
                k_direct<Op32, 1><<<nt, 32, 0, stream>>>(p0); k_split_op<<<n_blocks, MK_THREADS, 0, stream>>>(bd, recs[3].opcode, recs[3].variant, recs[3].n_tiles); } });
            const float ms_s = gpu_time_ms(stream, 60000, "dyn static", [&] { for (int i = 0; i < 500; ++i) {
                mk_param_blob bg = {}; memcpy(bg.b, r.params.data(), r.params.size());
                k_split_op<<<n_blocks, MK_THREADS, 0, stream>>>(bg, r.opcode, r.variant, nt); k_split_op<<<n_blocks, MK_THREADS, 0, stream>>>(bd, recs[3].opcode, recs[3].variant, recs[3].n_tiles); } });
            const float ms_d = gpu_time_ms(stream, 60000, "dyn", [&] { for (int i = 0; i < 500; ++i) {
                hipMemsetAsync(claim, 0, 4, stream); k_split_dyn<OpK><<<n_blocks, MK_THREADS, 0, stream>>>(p0, nt, claim);
                k_split_op<<<n_blocks, MK_THREADS, 0, stream>>>(bd, recs[3].opcode, recs[3].variant, recs[3].n_tiles); } });
            printf("dyn rep %d (GLU op + down op, us per pair): one-wave blocks %.1f, 1024-thread static %.1f, 1024-thread dynamic %.1f\n",
                   rep, 2*ms_r, 2*ms_s, 2*ms_d);
        }
        CUDA_CHECK(hipFree(claim));
    }
    if (getenv("MK_SPLIT")) {
        auto run_split = [&] {
            for (const auto & r : recs) {
                mk_param_blob blob = {};
                memcpy(blob.b, r.params.data(), r.params.size());
                k_split_op<<<n_blocks, MK_THREADS, 0, stream>>>(blob, r.opcode, r.variant, r.n_tiles);
            }
        };
        ggml_backend_tensor_set(fc.out, poison.data(), 0, ggml_nbytes(fc.out));
        ggml_backend_synchronize(fc.backend);
        run_split();
        sync_bounded(stream, 10000, "ffn split");
        std::vector<float> got2(ffn_case::n_embd);
        ggml_backend_tensor_get(fc.out, got2.data(), 0, ggml_nbytes(fc.out));
        printf("ffn split: %s\n", memcmp(got2.data(), ref.data(), ggml_nbytes(fc.out)) == 0 ? "bit-identical" : "DIFFERS");
        const float ms = gpu_time_ms(stream, 60000, "ffn split", [&] { for (int i = 0; i < 2000; ++i) { run_split(); } });
        printf("ffn split: %.1f us per pass (4 plain 1024-thread launches)\n", 1000.0f*ms/2000);
    }
    if (getenv("MK_FFN_OPS")) {
        ok = true;
    }
    if (timing && ok) {
        for (int rep = 0; rep < reps; ++rep) {
            ggml_backend_synchronize(fc.backend);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                ggml_backend_graph_compute_async(fc.backend, fc.gf);
            }
            ggml_backend_synchronize(fc.backend);
            const double us_ref = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;

            const float ms = gpu_time_ms(stream, 60000, "ffn timing", [&] {
                for (int i = 0; i < iters; ++i) {
                    mk_run(s, epoch++, stream);
                }
            });
            const double us_mk = 1000.0*ms/iters;
            printf("ffn timing rep %d: reference graph %.1f us, megakernel %.1f us, speedup %.3fx\n", rep, us_ref, us_mk, us_ref/us_mk);
        }
        ok = expect_error(stream, s, MK_ERR_NONE, "ffn timing") && ok;
    }

    printf("ffn: %s\n", ok ? "OK" : "FAILED");
    return ok;
}

// Path F prototype: independent ops side by side in one normal launch. Each block runs one op;
// params travel by value in kernarg and the hardware dispatcher balances the blocks.
template <int BLOCK>
static __device__ void mf_run_op(const mk_param_blob & blob, int opcode, int variant, int n_tiles, int blk, char * lds) {
    auto exec = [&](auto op) {
        using Op = decltype(op);
        using P  = mk_op_params<Op>;
        const P & p = *reinterpret_cast<const P *>(blob.b);
        constexpr int nsub = BLOCK / Op::threads;
        const int sub  = threadIdx.x / Op::threads;
        const int tile = blk*nsub + sub;
        Op::run(p, variant, tile < n_tiles ? tile : n_tiles - 1, tile < n_tiles, lds + sub*Op::lds_bytes);
    };
    switch (opcode) {
        case MK_OP_MMVQ:          mk_mmvq_dispatch<BLOCK>(variant, exec); break;
        case MK_OP_QUANTIZE_Q8_1: exec(mk_quantize_q8_1<BLOCK>{}); break;
        default: break;
    }
}

struct mf_op { mk_param_blob blob; int opcode, variant, n_tiles, n_blocks; };

template <int BLOCK>
__global__ void __launch_bounds__(BLOCK) k_multi2(const mf_op a, const mf_op b) {
    __shared__ __align__(16) char lds[8192];
    if ((int) blockIdx.x < a.n_blocks) {
        mf_run_op<BLOCK>(a.blob, a.opcode, a.variant, a.n_tiles, blockIdx.x, lds);
    } else {
        mf_run_op<BLOCK>(b.blob, b.opcode, b.variant, b.n_tiles, blockIdx.x - a.n_blocks, lds);
    }
}


// rms_norm -> two independent matvecs over the same input: the GDN qkv (Q6_K) and z (IQ4_XS) projections.
struct proj_case {
    static constexpr int n_embd = 5120;
    ggml_backend_t backend = nullptr; ggml_context * ctx = nullptr; ggml_backend_buffer_t buf = nullptr; ggml_cgraph * gf = nullptr;
    ggml_tensor * x = nullptr, * norm_w = nullptr, * wa = nullptr, * wb = nullptr, * oa = nullptr, * ob = nullptr;
    proj_case() {
        backend = ggml_backend_cuda_init(ggml_cuda_get_device());
        ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        x      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        // MK_ROWS_MULT scales both matrices past the 96 MiB infinity cache, as in real decode.
        const int m = getenv("MK_ROWS_MULT") ? atoi(getenv("MK_ROWS_MULT")) : 1;
        wa     = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K,   n_embd, 10240*m);
        wb     = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_XS, n_embd, 6144*m);
        ggml_tensor * cur = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), norm_w);
        oa = ggml_mul_mat(ctx, wa, cur);
        ob = ggml_mul_mat(ctx, wb, cur);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, oa);
        ggml_build_forward_expand(gf, ob);
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::vector<float> v(n_embd);
        for (int i = 0; i < n_embd; ++i) { v[i] = 0.01f*(float) ((i*37) % 101 - 50); }
        ggml_backend_tensor_set(x, v.data(), 0, ggml_nbytes(x));
        for (int i = 0; i < n_embd; ++i) { v[i] = 1.0f + 0.001f*(float) (i % 13); }
        ggml_backend_tensor_set(norm_w, v.data(), 0, ggml_nbytes(norm_w));
        ffn_case::fill_quant(wa);
        ffn_case::fill_quant(wb);
    }
    ~proj_case() { ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(backend); }
};

// Typed form: exact op types are template arguments and params are typed kernel arguments, so the
// compiler sees kernarg pointers (global loads, scalar params) and instantiates only these two bodies.
template <int BLOCK, typename OpA, typename OpB>
__global__ void __launch_bounds__(BLOCK) k_multi_typed(const mk_op_params<OpA> pa, int nta, int nba, const mk_op_params<OpB> pb, int ntb) {
    __shared__ __align__(16) char lds[(BLOCK/OpA::threads*OpA::lds_bytes > BLOCK/OpB::threads*OpB::lds_bytes ?
                                      BLOCK/OpA::threads*OpA::lds_bytes : BLOCK/OpB::threads*OpB::lds_bytes) + 16];
    auto go = [&](auto op, const auto & p, int nt, int blk) {
        using Op = decltype(op);
        constexpr int nsub = BLOCK / Op::threads;
        const int sub  = threadIdx.x / Op::threads;
        const int tile = blk*nsub + sub;
        Op::run(p, 0, tile < nt ? tile : nt - 1, tile < nt, lds + sub*Op::lds_bytes);
    };
    if ((int) blockIdx.x < nba) {
        go(OpA{}, pa, nta, blockIdx.x);
    } else {
        go(OpB{}, pb, ntb, blockIdx.x - nba);
    }
}

// Tuned form: (BLOCK, 1) launch bounds, and decode-shaped tiles (one channel, one sample) passed straight
// to block() without run()'s generic decode.
template <int BLOCK, typename OpA, typename OpB>
__global__ void __launch_bounds__(BLOCK, 1) k_multi_fast(const mk_op_params<OpA> pa, int nta, int nba, const mk_op_params<OpB> pb, int ntb) {
    __shared__ __align__(16) char lds[(BLOCK/OpA::threads*OpA::lds_bytes > BLOCK/OpB::threads*OpB::lds_bytes ?
                                      BLOCK/OpA::threads*OpA::lds_bytes : BLOCK/OpB::threads*OpB::lds_bytes) + 16];
    auto go = [&](auto op, const auto & p, int nt, int blk) {
        using Op = decltype(op);
        constexpr int nsub = BLOCK / Op::threads;
        const int sub   = threadIdx.x / Op::threads;
        const int local = threadIdx.x % Op::threads;
        const int tile  = blk*nsub + sub;
        Op::block(p, tile < nt ? tile : nt - 1, 0, 0, local % 32, local / 32, tile < nt, lds + sub*Op::lds_bytes);
    };
    if ((int) blockIdx.x < nba) {
        go(OpA{}, pa, nta, blockIdx.x);
    } else {
        go(OpB{}, pb, ntb, blockIdx.x - nba);
    }
}

static bool test_multi(hipStream_t stream) {
    proj_case pc;
    std::vector<mk_recorded_op> recs;
    ggml_cuda_mk_set_recording(&recs);
    GGML_ASSERT(ggml_backend_graph_compute(pc.backend, pc.gf) == GGML_STATUS_SUCCESS);
    ggml_cuda_mk_set_recording(nullptr);
    ggml_backend_synchronize(pc.backend);
    printf("multi: %zu recorded launches:", recs.size());
    for (const auto & r : recs) { printf(" %s(v=0x%x,%d)", mk_opname(r.opcode), r.variant, r.n_tiles); }
    printf("\n");
    if (recs.size() != 3 || recs[1].opcode != MK_OP_MMVQ || recs[2].opcode != MK_OP_MMVQ) {
        printf("multi: unexpected launch pattern, skipped\n");
        return false;
    }
    std::vector<float> ra(pc.oa->ne[0]), rb(pc.ob->ne[0]);
    ggml_backend_tensor_get(pc.oa, ra.data(), 0, ggml_nbytes(pc.oa));
    ggml_backend_tensor_get(pc.ob, rb.data(), 0, ggml_nbytes(pc.ob));

    constexpr int BLOCK = 256;
    mf_op op[2];
    for (int i = 0; i < 2; ++i) {
        const mk_recorded_op & r = recs[1 + i];
        memcpy(op[i].blob.b, r.params.data(), r.params.size());
        op[i].opcode = r.opcode; op[i].variant = r.variant; op[i].n_tiles = r.n_tiles;
        op[i].n_blocks = (r.n_tiles + BLOCK/r.threads - 1) / (BLOCK/r.threads);
    }
    std::vector<float> poison(pc.oa->ne[0], NAN);
    ggml_backend_tensor_set(pc.oa, poison.data(), 0, ggml_nbytes(pc.oa));
    ggml_backend_tensor_set(pc.ob, poison.data(), 0, ggml_nbytes(pc.ob));
    ggml_backend_synchronize(pc.backend);
    k_multi2<BLOCK><<<op[0].n_blocks + op[1].n_blocks, BLOCK, 0, stream>>>(op[0], op[1]);
    sync_bounded(stream, 10000, "multi");
    std::vector<float> ga(pc.oa->ne[0]), gb(pc.ob->ne[0]);
    ggml_backend_tensor_get(pc.oa, ga.data(), 0, ggml_nbytes(pc.oa));
    ggml_backend_tensor_get(pc.ob, gb.data(), 0, ggml_nbytes(pc.ob));
    const bool ok = memcmp(ga.data(), ra.data(), ga.size()*4) == 0 && memcmp(gb.data(), rb.data(), gb.size()*4) == 0;
    printf("multi: matvec pair %s\n", ok ? "bit-identical" : "DIFFERS");

    using OpA = mk_mmvq<BLOCK, GGML_TYPE_Q6_K,   1, false, false, 0, true>;
    using OpB = mk_mmvq<BLOCK, GGML_TYPE_IQ4_XS, 1, false, false, 0, true>;
    GGML_ASSERT(recs[1].variant == mk_mmvq_variant(GGML_TYPE_Q6_K, 1, false, true) && recs[2].variant == mk_mmvq_variant(GGML_TYPE_IQ4_XS, 1, false, true));
    mk_mmvq_params pa, pb;
    memcpy(&pa, recs[1].params.data(), sizeof(pa));
    memcpy(&pb, recs[2].params.data(), sizeof(pb));
    // Grid sizes come from the recorded block sizes: OpA::threads in host code uses the generic mmvq table.
    const int nba = (recs[1].n_tiles + BLOCK/recs[1].threads - 1) / (BLOCK/recs[1].threads);
    const int nbb = (recs[2].n_tiles + BLOCK/recs[2].threads - 1) / (BLOCK/recs[2].threads);
    auto typed = [&] { k_multi_typed<BLOCK, OpA, OpB><<<nba + nbb, BLOCK, 0, stream>>>(pa, recs[1].n_tiles, nba, pb, recs[2].n_tiles); };
    ggml_backend_tensor_set(pc.oa, poison.data(), 0, ggml_nbytes(pc.oa));
    ggml_backend_tensor_set(pc.ob, poison.data(), 0, ggml_nbytes(pc.ob));
    ggml_backend_synchronize(pc.backend);
    typed();
    sync_bounded(stream, 10000, "multi typed");
    ggml_backend_tensor_get(pc.oa, ga.data(), 0, ggml_nbytes(pc.oa));
    ggml_backend_tensor_get(pc.ob, gb.data(), 0, ggml_nbytes(pc.ob));
    const bool ok_typed = memcmp(ga.data(), ra.data(), ga.size()*4) == 0 && memcmp(gb.data(), rb.data(), gb.size()*4) == 0;
    printf("multi: recorded threads %d / %d, blocks %d + %d\n", recs[1].threads, recs[2].threads, nba, nbb);
    printf("multi: typed matvec pair %s\n", ok_typed ? "bit-identical" : "DIFFERS");
    ggml_backend_tensor_set(pc.oa, poison.data(), 0, ggml_nbytes(pc.oa));
    ggml_backend_tensor_set(pc.ob, poison.data(), 0, ggml_nbytes(pc.ob));
    ggml_backend_synchronize(pc.backend);
    k_multi_fast<BLOCK, OpA, OpB><<<nba + nbb, BLOCK, 0, stream>>>(pa, recs[1].n_tiles, nba, pb, recs[2].n_tiles);
    sync_bounded(stream, 10000, "multi fast");
    ggml_backend_tensor_get(pc.oa, ga.data(), 0, ggml_nbytes(pc.oa));
    ggml_backend_tensor_get(pc.ob, gb.data(), 0, ggml_nbytes(pc.ob));
    const bool ok_fast = memcmp(ga.data(), ra.data(), ga.size()*4) == 0 && memcmp(gb.data(), rb.data(), gb.size()*4) == 0;
    printf("multi: fast 2-op kernel %s\n", ok_fast ? "bit-identical" : "DIFFERS");
    if (!ok_typed) {
        int da = 0, na = 0, db = 0, nb = 0;
        for (size_t i = 0; i < ga.size(); ++i) { da += memcmp(&ga[i], &ra[i], 4) != 0; na += std::isnan(ga[i]); }
        for (size_t i = 0; i < gb.size(); ++i) { db += memcmp(&gb[i], &rb[i], 4) != 0; nb += std::isnan(gb[i]); }
        printf("multi: qkv differs %d (nan %d) of %zu, z differs %d (nan %d) of %zu; e.g. qkv[0] %.9g vs %.9g\n", da, na, ga.size(), db, nb, gb.size(), ga[0], ra[0]);
    }

    constexpr int iters = 2000;
    for (int rep = 0; rep < 3; ++rep) {
        ggml_backend_synchronize(pc.backend);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) { ggml_backend_graph_compute_async(pc.backend, pc.gf); }
        ggml_backend_synchronize(pc.backend);
        const double us_ref = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        const float ms_pair = gpu_time_ms(stream, 60000, "multi pair", [&] { for (int i = 0; i < iters; ++i) { typed(); } });
        printf("multi timing rep %d: reference graph (norm + 2 mmvq) %.1f us; typed multi-op launch (2 mmvq) %.1f us\n",
               rep, us_ref, 1000.0f*ms_pair/iters);
    }
    return ok && ok_typed && ok_fast;
}
#endif // __has_include("mk-ops-ffn.cuh")

int main(int argc, char ** argv) {
    int  soak   = 20;
    bool timing = false;
    bool ffn    = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-n" && i + 1 < argc) {
            soak = atoi(argv[++i]);
        } else if (a == "--timing") {
            timing = true;
        } else if (a == "--ffn") {
            ffn = true;
        } else {
            fprintf(stderr, "usage: %s [-n soak_launches] [--timing] [--ffn]\n", argv[0]);
            return 1;
        }
    }

    ggml_cuda_set_device(0);
    hipStream_t stream;
    CUDA_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    const int      n_blocks = n_wgp();
    const uint64_t watchdog = 2000*ticks_per_ms();
    printf("megakernel: %d blocks x %d threads\n", n_blocks, MK_THREADS);

    bool ok = true;
    {
        chain_test t(n_blocks);
        ok = run_passes(t, stream, 4, watchdog, "chain") && ok;
    }
    {
        dag_test t(n_blocks);
        ok = run_passes(t, stream, 64, watchdog, "dag-epochs") && ok;
    }
    ok = test_watchdog(stream, n_blocks) && ok;
    {
        chain_test t(n_blocks);
        ok = run_passes(t, stream, 2, watchdog, "chain-after-errors") && ok;
    }
    {
        dag_test t(n_blocks);
        ok = run_passes(t, stream, soak, watchdog, "soak") && ok;
    }
    if (timing) {
        if (!getenv("MK_FFN_ONLY")) { test_timing(stream, n_blocks); }
    }
#if __has_include("mk-ops-ffn.cuh")
    if (getenv("MK_MULTI")) {
        ok = test_multi(stream) && ok;
    } else if (ffn || timing) {
        ok = test_ffn(stream, n_blocks, timing) && ok;
    }
#else
    if (ffn) {
        printf("ffn: mk-ops-ffn.cuh not present, skipped\n");
    }
#endif

    CUDA_CHECK(hipStreamDestroy(stream));
    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
