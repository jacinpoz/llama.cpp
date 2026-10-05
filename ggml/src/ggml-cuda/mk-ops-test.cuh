#pragma once

#include "megakernel.cuh"

// y[i] += x[i] for i in [tile*threads, (tile+1)*threads), lanes swapped through LDS across a barrier.
// delay: s_sleep loops before the store, to widen race windows in ordering tests.
struct mk_test_add_params {
    float       * y;
    const float * x;
    int32_t       n;
    int32_t       delay;
};

template <int BLOCK>
struct mk_test_add {
    static constexpr int threads   = 128;
    static constexpr int lds_bytes = threads*sizeof(float);

    static __device__ void run(const mk_test_add_params & p, int /*variant*/, int tile, bool valid, char * lds) {
        float * buf = (float *) lds;
        const int lane = threadIdx.x % threads;
        const int i    = tile*threads + lane;
        if (valid && i < p.n) {
            buf[lane] = p.x[i];
        }
        __syncthreads();
        const int lane_r = threads - 1 - lane;
        const int j      = tile*threads + lane_r;
        if (valid && j < p.n) {
            for (int k = 0; k < p.delay; ++k) {
                __builtin_amdgcn_s_sleep(8);
            }
            p.y[j] += buf[lane_r];
        }
    }
};

// Spins for ticks of wall_clock64().
struct mk_test_spin_params {
    uint64_t ticks;
};

template <int BLOCK>
struct mk_test_spin {
    static constexpr int threads   = 1024;
    static constexpr int lds_bytes = 0;

    static __device__ void run(const mk_test_spin_params & p, int /*variant*/, int /*tile*/, bool valid, char * /*lds*/) {
        if (!valid) {
            return;
        }
        const uint64_t t0 = wall_clock64();
        while (wall_clock64() - t0 < p.ticks) {
            __builtin_amdgcn_s_sleep(8);
        }
    }
};
