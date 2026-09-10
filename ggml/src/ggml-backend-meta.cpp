#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-moe-weighted-reduction.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// getenv() is cheap on Linux but takes a lock and rescans the environment block on Windows.  These
// debug / A-B gates sit on per-graph, per-split and per-tensor paths, so resolve each one once per
// call site: the static inside the immediately-invoked lambda is unique to each macro expansion.
#define GGML_ENV_STR(name) ([]() -> const char * { static const char * v = getenv(name); return v; }())

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

// TEMP INSTRUMENT (wip/tensor-split-expert-split): is the whole-cache clear the per-graph cost?
static int64_t g_ss_calc = 0, g_ss_hit = 0, g_ss_clear = 0, g_ss_erase = 0;
static bool    g_ss_reg = false;
static void g_ss_dump(void) {
    fprintf(stderr, "SPLITSTATE calc=%lld hit=%lld clears=%lld erase=%lld\n",
        (long long) g_ss_calc, (long long) g_ss_hit, (long long) g_ss_clear, (long long) g_ss_erase);
}
static void g_ss_add(const int which) {
    if (GGML_ENV_STR("GGML_META_SSDBG") == nullptr) { return; }
    if (!g_ss_reg) { g_ss_reg = true; atexit(g_ss_dump); }
    if (which == 0) g_ss_calc++; else if (which == 1) g_ss_hit++; else if (which == 2) g_ss_clear++; else g_ss_erase++;
}

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // Not implemented.
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ false, // Not implemented.
        /* .mmap_support          = */ true,
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

// Op offload (host-resident weights executed on the device) for the tensor split.  The meta backend
// is a first-class host of such ops: it runs the op on every device, and each device uploads its own
// share of the weights -- which is exactly what the scheduler's used-expert pruning and the H2D
// staging ring act on.  Every device has to accept the op on its own, since the meta backend cannot
// run a node that only some of its devices can.  Without this hook `ggml_backend_dev_offload_op()`
// is false here (the device iface defaults to no offload), so `ggml_backend_sched_backend_id_from_cur`
// never places an offloaded node on this backend and `-sm tensor` silently executes the whole MoE on
// the CPU.
static bool ggml_backend_meta_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    if (meta_dev_ctx->simple_devs.empty()) {
        return false;
    }
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_offload_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ ggml_backend_meta_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        for (size_t i = 0; i < simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_meta_buffer_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
struct ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(const ggml_init_params & params, const int n_simple) {
        ctxs.reserve(n_simple);
        for (int i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

struct ggml_backend_meta_buffer_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[2];
    int stc_compute_index      = 0;
    int stc_compute_index_next = 0;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);
    std::map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, char[nbtc]>> split_state_cache;

    int debug;

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            const std::vector<ggml_backend_buffer_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        this->bufs.reserve(bufs.size());
        for (ggml_backend_buffer_t buf : bufs) {
            this->bufs.emplace_back(buf);
        }
        const char * GGML_META_DEBUG = GGML_ENV_STR("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(const ggml_tensor * tensor) {
        if (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            return stc_static;
        }
        return stc_compute[stc_compute_index];
    }
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    delete buf_ctx;
}

static size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->bufs.size();
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    return buf_ctx->bufs[index].get();
}

static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    auto it = stc.simple_tensors.find(tensor);
    if (it == stc.simple_tensors.end()) {
        // exp21: is a meta tensor's simple mapping in the *other* double-buffered compute container?
        if (GGML_ENV_STR("GGML_META_CNDBG") != nullptr) {
            for (int c = 0; c < 2; ++c) {
                auto & sc = buf_ctx->stc_compute[c];
                auto jt = sc.simple_tensors.find(tensor);
                if (jt != sc.simple_tensors.end()) {
                    const ggml_tensor * st = jt->second[index];
                    fprintf(stderr, "CNDBG MISS %-52s cur=%d other=%d data=%p ne=[%lld,%lld,%lld]\n",
                        tensor->name[0] ? tensor->name : "-", buf_ctx->stc_compute_index, c,
                        st ? (void *) st->data : nullptr,
                        (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2]);
                }
            }
        }
        return nullptr;
    }
    return it->second[index];
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync);

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, const struct ggml_tensor * tensor, bool assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto split_states_equal = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            for (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            for (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } else if (!split_states_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                break;
            }
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        return ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        return src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            return src_ss[1];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        if (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (int) src_ss[0].axis, tensor->src[1]->name, (int) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                int64_t base_ne_in = 1;
                for (int dim = 0; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                if (src_ss[0].n_segments == 1) {
                    base_ne_in /= src_ss[0].nr[0];
                    if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        return {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        bool complete_rows = true;
                        for (size_t j = 0; j < n_bufs; j++) {
                            const int64_t ne = src_ss[0].ne[j];
                            complete_rows = complete_rows && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        if (complete_rows) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    base_ne_out *= tensor->ne[dim];
                    if (base_ne_out % base_ne_in == 0) {
                        return {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out/base_ne_in)}, 1};
                    }
                    if (base_ne_out > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_cpy = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            return handle_reshape(src_ss);
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            return handle_reshape(src_ss);
        }
        const int axis = src_ss[0].axis;
        {
            bool all_strides_the_same = true;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    continue;
                }
                if (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return src_ss[0];
            }
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            for (int dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                if (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_permute = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_transpose = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3:
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_get_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        return src_ss[0];
    };

    auto handle_rope = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    auto handle_pad = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        const bool kv_split = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2;
        const bool kv_mirrored = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        GGML_ASSERT(kv_split || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    // qwen4exp QSA sparse attention: same split as FLASH_ATTN_EXT (q by head,
    // K/V by kv-head, result by head); idx and mask are mirrored.
    auto handle_flash_attn_qsa = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        // src4 (mask) plus the optional derived-visibility keys are replicated when present; a
        // null src slot (the mask on the derived path) carries no split state
        for (size_t i = 4; i < 7; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                    src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        const bool kv_split = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2;
        const bool kv_mirrored = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        GGML_ASSERT(kv_split || kv_mirrored);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_lightning_indexer = [&](
            const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        for (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    };

    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_gated_delta_net = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        // Tensors the per-tensor policy can answer for: a weight held in a static buffer, or the
        // scheduler's split-input copy of one.  A host-resident weight that is op-offloaded reaches the
        // meta backend **only** as such a copy (`ggml_dup_tensor_layout` of the weight, so op NONE with
        // no view source, allocated in the split's COMPUTE buffer -- see
        // `ggml_backend_sched_alloc_splits`), and the policy has to be asked for it too: the COMPUTE
        // branch below would otherwise propagate it from `GGML_OP_NONE` and return mirrored, which
        // uploads the whole weight to every device.  A tensor the policy does not recognise (any
        // activation copy) keeps the mirrored default, so this is a no-op for them.
        const bool policy_tensor =
            tensor->view_src == nullptr &&
            (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
             tensor->op == GGML_OP_NONE);
        if (policy_tensor) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                for (size_t s = 0; s < ret.n_segments; s++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            } else if (ret.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            return ret;
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                continue;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ true);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUM_ROWS:
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_REPEAT:
            case GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } break;
            case GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } break;
            case GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } break;
            case GGML_OP_CONT:
            case GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } break;
            case GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            case GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } break;
            case GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } break;
            case GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } break;
            case GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } break;
            case GGML_OP_DIAG:
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_ROPE_BACK: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_1D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } break;
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_ARANGE:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_QSA: {
                split_state = handle_flash_attn_qsa(src_ss);
            } break;
            case GGML_OP_INDEXER_TOPK: {
                // like TOP_K: the row (tps, stream) is the independent unit.  every src is
                // mirrored: the cell map plus the optional derived position/bias compact srcs
                for (int i = 0; i < GGML_MAX_SRC; ++i) {
                    if (tensor->src[i] != nullptr) {
                        GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                    }
                }
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_INDEXER_SCORE: {
                // like INDEXER_TOPK: mirrored indexer cache view + cell map + positions +
                // query + norm weights + bias (each device holds full mirrored copies);
                // the optional derived-cache pool view + range leaf are mirrored too
                for (int i = 0; i < GGML_MAX_SRC; ++i) {
                    if (tensor->src[i] != nullptr) {
                        GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                    }
                }
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_INDEXER_FILL: {
                // mirrored like the other indexer ops: raw cache view + cell map + positions + norm
                // weights + the host range leaf + the pool buffer it writes
                for (int i = 0; i < GGML_MAX_SRC; ++i) {
                    if (tensor->src[i] != nullptr) {
                        GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                    }
                }
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } break;
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_RWKV_WKV7:
            case GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } break;
            case GGML_OP_LIGHTNING_INDEXER: {
                split_state = handle_lightning_indexer(src_ss);
            } break;
            case GGML_OP_DSV4_HC_COMB:
            case GGML_OP_DSV4_HC_PRE:
            case GGML_OP_DSV4_HC_POST: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_HC_MIX: {
                // the hc weights and the per-token tensors are mirrored per
                // device (verified on the qwen4exp decode graph); each device
                // runs the full op and produces its own local copy. The output
                // carries both mixed and the inject scatter (the combine views
                // the tail), all mirrored. src[4] (w_inject) is null at the
                // head call (il = -1), which the splitter marks UNKNOWN.
                GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_HC_COMBINE: {
                // the hc weights and the per-token tensors are mirrored per
                // device (verified on the qwen4exp decode graph); each device
                // runs the full op and produces its own local copy.
                GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } break;
        }
        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                if (first_src_split_by_axis) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        if (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            const int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                } else {
                    GGML_ASSERT(split_state.n_segments == 1);
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t sum = 0;
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = false;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        return split_state;
    };

    const std::pair key = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(key);
    if (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (const char *) tensor, sizeof(it->second.second)) != 0) {
        // TEMP INSTRUMENT: GGML_META_SS_FIX=1 erases only the stale entry instead of the whole cache.
        if (GGML_ENV_STR("GGML_META_SS_FIX") != nullptr) {
            buf_ctx->split_state_cache.erase(it);
            g_ss_add(3);
        } else {
            buf_ctx->split_state_cache.clear();
            g_ss_add(2);
        }
        it = buf_ctx->split_state_cache.end();
    }

    if (it != buf_ctx->split_state_cache.end()) {
        g_ss_add(1);
    }
    if (it == buf_ctx->split_state_cache.end()) {
        g_ss_add(0);
        buf_ctx->split_state_cache[key].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[key].second, tensor, sizeof(buf_ctx->split_state_cache[key].second));
        if (buf_ctx->debug > 0) {
            std::string srcs_info;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                const ggml_backend_meta_split_state split_state =
                        ggml_backend_meta_get_split_state(tensor->src[i], true);
                GGML_ASSERT(split_state.n_segments == 1);
                const char * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                for (size_t j = 0; j < n_bufs; j++) {
                    if (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]) + "x" + std::to_string(split_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                const ggml_backend_meta_split_state & ss = buf_ctx->split_state_cache[key].first;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[key].first.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[key].first;
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        for (size_t s = 0; s < ret.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        assert(ne_ret == tensor->ne[int(ret.axis)]);
    }
#endif // NDEBUG
    return ret;
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    if (ggml_backend_buffer_is_host(tensor->buffer)) {
        // host tensors are not part of the meta split (e.g. a reshaped token
        // embedding or a PLE gather); treat them as unsplit
        return {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
    }
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    return ggml_backend_meta_get_split_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    // exp17: which tensors does the split-copy policy actually split?  Print every scheduler-created
    // copy (`<backend>#<name>#<n>`) with its state, so a view whose parent copy was split while the view
    // itself is MIRRORED is visible.
    if (GGML_ENV_STR("GGML_META_REDDBG") != nullptr && strchr(tensor->name, '#') != nullptr) {
        static int n_cp = 0;
        if (n_cp++ < 400) {
            const bool is_view = tensor->view_src != nullptr;
            fprintf(stderr, "REDDBG COPY %-52s axis=%d view=%d ne=[%lld,%lld,%lld,%lld]\n",
                tensor->name, (int) split_state.axis, (int) is_view,
                (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2], (long long) tensor->ne[3]);
        }
    }
    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get();
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            for (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j] * split_state.nr[s];
            }
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            if (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                const int split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        break;
                    }
                }
                if (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        // TODO: revisit once the graph allocator has been refactored, see https://github.com/ggml-org/llama.cpp/pull/25051#issuecomment-4842873396
        ggml_backend_buffer_t init_buf = simple_buf;
        if (t_ij->view_src != nullptr) {
            // exp18: name the ids view's parent and its buffer/buft, and both pointers, so a bad one is
            // visible directly (the split run's first ids read faults).
            if (GGML_ENV_STR("GGML_META_VIEWDBG") != nullptr && strstr(tensor->name, "topk") != nullptr) {
                const ggml_tensor * vo = tensor->view_src;   // the META parent, before the remap
                fprintf(stderr, "VIEWDBG %-24s dev=%zu split_dim=%d\n", tensor->name, j, (int) split_dim);
                fprintf(stderr, "VIEWDBG   meta parent=%s buf=%p buft=%s is_meta=%d data=%p\n",
                    vo ? vo->name : "-", vo ? (void *) vo->buffer : nullptr,
                    vo && vo->buffer ? ggml_backend_buffer_name(vo->buffer) : "-",
                    vo && vo->buffer ? (int) ggml_backend_buffer_is_meta(vo->buffer) : -1,
                    vo ? vo->data : nullptr);
                fprintf(stderr, "VIEWDBG   simple parent=%s data=%p view_offs=%zu -> t_ij data=%p (nb1=%zu ne1=%lld)\n",
                    t_ij->view_src->name, t_ij->view_src->data, t_ij->view_offs,
                    (void *) ((char *) t_ij->view_src->data + t_ij->view_offs),
                    t_ij->view_src->nb[1], (long long) t_ij->view_src->ne[1]);
            }
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
            // views inherit the source slice's concrete sub-buffer (issue 22197)
            if (tensor->view_src != nullptr && ggml_backend_buffer_is_meta(tensor->view_src->buffer)
                    && t_ij->view_src->buffer != nullptr) {
                t_ij->buffer = t_ij->view_src->buffer;
                init_buf     = t_ij->view_src->buffer;
            }
        } else if (simple_buf != nullptr) {
            {
                const size_t off = size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
                if (GGML_ENV_STR("GGML_META_BUFDBG") != nullptr && strstr(tensor->name, "exps") != nullptr) {
                    const size_t s_need = off + ggml_nbytes(t_ij);
                    const size_t s_have = ggml_backend_buffer_get_size(simple_buf);
                    fprintf(stderr, "BUFDBG %-46s dev=%zu off=%zu need=%zu have=%zu %s\n",
                        tensor->name, j, off, s_need, s_have, s_need > s_have ? "OVERFLOW" : "ok");
                }
            }
            if (ggml_backend_buffer_is_multi_buffer(simple_buf)) {
                GGML_ABORT("multi buffers are not supported by the meta backend");
            }
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }

        if (init_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_buffer_init_tensor(init_buf, t_ij);
        } else {
            t_ij->extra = tensor->extra;
        }

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            continue;
        }

        const ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            for (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            if (ne_sum == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    buf_ctx->stc_compute_index = buf_ctx->stc_compute_index_next;
    return ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

static void ggml_backend_meta_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state =
            ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        for (int64_t row = 0; row < row_count; row++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (row_start + row)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            return;
        }

        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    for (int64_t row = 0; row < row_count; row++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (row_start + row)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        / chunk_size_full;
            const int64_t i_stop  = (offset + size) / chunk_size_full;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size == 0) {
                    continue;
                }
                for (int64_t i = i_start; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_size, chunk_size);
                }
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(offset % sizeof(float) == 0);
            GGML_ASSERT(size   % sizeof(float) == 0);
            const size_t n_values = size / sizeof(float);
            size_t n_contributors = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                n_contributors += split_state.ne[j] != 0;
            }
            const bool has_contributor_mask = n_contributors != 0;
            if (!has_contributor_mask) {
                n_contributors = n_bufs;
            }
            std::vector<float> tmp(n_values);
            for (size_t i = 0; i < n_values; i++) {
                tmp[i] = ((const float *) data)[i] / n_contributors;
            }
            std::vector<float> zero;
            if (has_contributor_mask) {
                zero.resize(n_values, 0.0f);
            }
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const float * partial = has_contributor_mask && split_state.ne[j] == 0 ? zero.data() : tmp.data();
                ggml_backend_tensor_set(simple_tensor, partial, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++){
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

void ggml_backend_meta_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        if (buf_ctx->bufs[i]) {
            ggml_backend_buffer_set_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    const ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != nullptr);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
    }
    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    // Views of the static tensors that are created between graph evals are stored in the compute
    // containers. The number of such views is proportional to the number of tensors in the graph
    // that share the buffer, which for hybrid recurrent models with n_rs_seq snapshotting can be
    // much larger than 16 per static tensor (e.g. Qwen35: ~2*(n_rs_seq+1) views per recurrent
    // layer are created for the conv-state snapshot copies). Size the headroom accordingly.
    constexpr size_t compute_headroom = 128;
    const ggml_init_params params_static = {
        /*.mem_size   =*/ ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    const ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom*ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        bool any_nonzero_slice = false;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_nelements(t) != 0) {
                any_nonzero_slice = true;
                break;
            }
        }
        if (any_nonzero_slice) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } else {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
    }
    return meta_buf;
}

//
// meta backend
//

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

struct ggml_backend_meta_context {
    struct cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        int           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    struct backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;

        backend_config(ggml_backend_t backend, const size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    int                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;
    // the scheduler's eval-callback flag, stashed by graph_optimize and replayed into the child
    // backends' per-device marking pass (graph_compute) so a callback-split graph stands the F32
    // elision down exactly as the single-backend path does
    bool                        opt_has_eval_callback = false;

    // op-offload H2D staging for the tensor split (issue #50 WIP).  The scheduler's own staging path
    // assumes one destination and an event on the split backend's device; under -sm tensor one logical
    // upload is spliced across N devices and the split's consumers read the per-device "simple"
    // tensors, so the meta backend owns the lifecycle: `stage_input` lands each device's chunk in that
    // device's own ring on its auxiliary copy stream (overlapping the previous split's compute), and
    // `graph_compute` waits for those copies, points the simple tensors at the slots and frees them
    // once the child graphs are launched.
    struct stage_chunk {
        ggml_backend_t backend = nullptr; // the simple backend that owns the slot
        size_t         device  = 0;
        void *         slot    = nullptr; // that backend's ring slot for this upload
        size_t         size    = 0;       // bytes per block (gather) or of the whole copy
        size_t         src     = 0;       // offset into the host source (0 for a MIRRORED split)
        size_t         stride  = 0;       // host-source stride between blocks (gather only)
        size_t         n_copies= 0;       // number of blocks to gather (0 = plain contiguous upload)
        void *         orig    = nullptr; // the simple tensor's data, restored after the launch
    };
    struct stage_entry {
        ggml_tensor * tensor = nullptr;   // the split input (a tensor of this backend)
        int            slot   = 0;
        std::vector<stage_chunk> chunks;  // one per device that holds part of the upload
        int64_t        t_issue = 0;       // TEMP INSTRUMENT: when the last chunk was issued
    };
    std::vector<stage_entry> stage_pending; // staged by `stage_input`, drained by `graph_compute`
    int   stage_slot_next = 0;
    int   stage_n_slots   = 8;              // GGML_SCHED_STAGE_SLOTS, clamped to the ring's 16
    std::vector<std::vector<ggml_backend_event_t>> stage_done_ev; // [device][slot], upload complete
    std::vector<std::vector<ggml_backend_event_t>> stage_free_ev; // [device][slot], slot reusable

    void *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        // op-offload H2D staging (issue #50 WIP): the slot count mirrors the single-device ring, so
        // GGML_SCHED_STAGE_SLOTS means the same depth on both paths (the ring itself lives in each
        // simple backend's context and is bounded by its own GGML_SCHED_STAGE_MAX_MB).
        {
            const char * slots_env = GGML_ENV_STR("GGML_SCHED_STAGE_SLOTS");
            stage_n_slots = slots_env != nullptr ? atoi(slots_env) : 8;
            if (stage_n_slots < 1) {
                stage_n_slots = 1;
            }
            if (stage_n_slots > 16) {
                stage_n_slots = 16;
            }
            stage_done_ev.resize(n_devs);
            stage_free_ev.resize(n_devs);
            for (size_t j = 0; j < n_devs; j++) {
                stage_done_ev[j].resize(stage_n_slots, nullptr);
                stage_free_ev[j].resize(stage_n_slots, nullptr);
            }
        }

        if (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            if (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
        }
        if (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
        }
    }

    ~ggml_backend_meta_context() {
        if (comm_ctx != nullptr) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        for (size_t j = 0; j < stage_done_ev.size(); j++) {
            for (size_t s = 0; s < stage_done_ev[j].size(); s++) {
                ggml_backend_event_free(stage_done_ev[j][s]);
                ggml_backend_event_free(stage_free_ev[j][s]);
            }
        }
        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
        // A mirrored tensor holds the same bytes on every device, so no chunk arithmetic is needed and
        // an arbitrary byte range can be forwarded as-is.  The scheduler's used-expert pruning uploads
        // just the experts a ubatch uses -- a range at a non-zero offset that need not be a whole-chunk
        // multiple -- which is what makes op-offload under -sm tensor work at all (see
        // ggml_backend_meta_device_offload_op).
        for (size_t j = 0; j < n_backends; j++) {
            ggml_backend_tensor_set_async(
                ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
        }
        return;
    }

    // The partial split exploits contiguity to splice the tensor across the devices as whole chunks.
    // A chunk-aligned, non-zero `offset` is legal: the used-expert pruning uploads one range per group
    // of consecutive used experts, so `first_id * expert_size` is the common case there.
    GGML_ASSERT(ggml_is_contiguous(tensor));

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            {
                static int n_dbg = 0;
                if (GGML_ENV_STR("GGML_META_SPLITDBG") && n_dbg < 6 && strstr(tensor->name, "exps.weight")) {
                    n_dbg++;
                    fprintf(stderr, "SPLITDBG name=%s axis=%d offset=%zu size=%zu chunk_full=%zu nbytes=%zu ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
                        tensor->name, (int) split_state.axis, offset, size, chunk_size_full, ggml_nbytes(tensor),
                        (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2], (long long) tensor->ne[3],
                        tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
                    for (size_t j = 0; j < n_backends; j++) {
                        ggml_tensor * st = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        fprintf(stderr, "SPLITDBG   dev=%zu chunk_j=%zu nbytes=%zu ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n", j,
                            st->nb[split_state.axis + 1], ggml_nbytes(st),
                            (long long) st->ne[0], (long long) st->ne[1], (long long) st->ne[2], (long long) st->ne[3],
                            st->nb[0], st->nb[1], st->nb[2], st->nb[3]);
                    }
                }
            }
            GGML_ASSERT(offset % chunk_size_full == 0);
            // exp27 diagnostic: is the trigger a cold (non-resident) page in the pageable source that
            // the 2-D blit kernel cannot fault in?  Touch every page on the CPU first.
            if (GGML_ENV_STR("GGML_META_TOUCHSRC") != nullptr) {
                volatile uint8_t sum = 0;
                for (size_t off = 0; off < size; off += 512) {
                    sum = (uint8_t) (sum ^ ((const uint8_t *) data)[off]);
                }
                (void) sum;
            }
            // The scheduler's used-expert pruning copies whole experts plus a trailing
            // `min(expert_size, 512)`-byte pad for MMQ's over-read, and `expert_size` for `MUL_MAT_ID` is
            // exactly this chunk, so the range is chunk-aligned except for that pad.  Copy the aligned
            // part with the spliced 2-D path and the short tail as a plain per-device range.
            //
            // `offset` is a *destination* byte offset: the caller (`sched_compute_splits`' expert
            // pruning) passes `data` already advanced to the start of the range, so the source base is
            // `data` itself and only the destination has to be rebased -- chunk `k` lives at
            // `k*chunk_size_full` in the meta tensor but at `k*chunk_size_j` in device `j`'s simple
            // tensor.  (That is also why the original code insisted `offset == 0`: with a non-zero
            // offset the two readings of the same argument diverge.)
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            const size_t  rem     = (offset + size)%chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t dst_base = (size_t) i_start*chunk_size_j;
                if (GGML_ENV_STR("GGML_META_UPLOADDBG") != nullptr) {
                    const char * sb = (const char *) data + offset_j;
                    const char * se = (const char *) data + offset_j + (size_t) (i_stop - i_start > 0 ? i_stop - i_start - 1 : 0)*chunk_size_full + chunk_size_j;
                    const char * db = (const char *) simple_tensor->data + dst_base;
                    const char * de = db + (size_t) std::max<int64_t>(i_stop - i_start, 0)*chunk_size_j;
                    ggml_backend_buffer_t sbuf = simple_tensor->view_src ? simple_tensor->view_src->buffer : simple_tensor->buffer;
                    fprintf(stderr, "UP2D %-44s axis=%d dev=%zu i=[%lld,%lld) rem=%zu cf=%zu cj=%zu nc=%lld src=[%p,%p) dst=[%p,%p) buf_base=%p buf_size=%zu nbytes_simple=%zu\n",
                        tensor->name[0] ? tensor->name : "-", (int) split_state.axis, j, (long long) i_start, (long long) i_stop, rem, chunk_size_full, chunk_size_j,
                        (long long) (i_stop - i_start), (const void *) sb, (const void *) se, (const void *) db, (const void *) de,
                        (void *) ggml_backend_buffer_get_base(sbuf), ggml_backend_buffer_get_size(sbuf), ggml_nbytes(simple_tensor));
                }
                if (i_stop > i_start) {
                    if (GGML_ENV_STR("GGML_META_NO_2D") == nullptr) {
                        ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (const char *) data + offset_j, dst_base, chunk_size_j,
                            i_stop - i_start, chunk_size_j, chunk_size_full);
                    } else {
                        // exp24 diagnostic: the 2-D pageable H2D copy is the faulting kernel
                        // (`__amd_rocclr_copyBufferRectAligned` under HSA_ENABLE_SDMA=0); do the same
                        // splice with the 1-D copy the mirrored path already uses successfully.
                        for (int64_t k = 0; k < i_stop - i_start; ++k) {
                            ggml_backend_tensor_set_async(simple_backend, simple_tensor,
                                (const char *) data + offset_j + (size_t) k*chunk_size_full,
                                dst_base + (size_t) k*chunk_size_j, chunk_size_j);
                        }
                    }
                }
                if (rem != 0 && offset_j < rem) {
                    // The range's short tail is a *partial* chunk, and each device owns only its own
                    // sub-block inside every chunk.  When the remainder is shorter than this device's
                    // sub-block offset it owns none of the tail; otherwise it owns
                    // `min(rem - offset_j, chunk_size_j)` bytes starting at `data + offset_j`.
                    // (The old code gave every device the whole `rem` from `data + offset_j`, which
                    // read past the range -- and past the tensor for the last expert -- for j > 0.)
                    const size_t tail_len = std::min(rem - offset_j, chunk_size_j);
                    ggml_backend_tensor_set_async(simple_backend, simple_tensor,
                        (const char *) data + offset_j + (size_t) (i_stop - i_start)*chunk_size_full,
                        dst_base + (size_t) (i_stop - i_start)*chunk_size_j, tail_len);
                }
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
            // exp23: drain the H2D copies so a bad destination faults here, not in a later kernel.
            // GGML_META_SYNCUPLOAD: 1 = all devices, 2 = only the second device, 3 = only the first
            // two uploads of the run (is a one-shot lazy init the trigger?).
            {
                const char * su = GGML_ENV_STR("GGML_META_SYNCUPLOAD");
                const int su_mode = su ? atoi(su) : 0;
                static int su_n = 0;
                bool do_sync = su_mode == 1 || (su_mode == 2) || (su_mode == 3 && su_n < 2);
                su_n++;
                if (do_sync) {
                    for (size_t j = 0; j < n_backends; j++) {
                        if (su_mode == 2 && j != n_backends - 1) continue;
                        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, j));
                    }
                    if (su_mode != 3 || su_n <= 2) {
                        fprintf(stderr, "SYNCUPLOAD mode=%d n=%d %s axis=%d offset=%zu size=%zu ok\n",
                            su_mode, su_n, tensor->name[0] ? tensor->name : "-", (int) split_state.axis, offset, size);
                    }
                }
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
        // A mirrored tensor is the same on every device; any device can serve an arbitrary byte range,
        // including a strided view's raw span (the scheduler's used-expert pruning reads the router's
        // ids that way, and their layout is indexed with the view's own strides).
        // TODO other simple backend may be better
        ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
        const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
        ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        return;
    }

    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

// op-offload H2D staging for the tensor split (issue #50 WIP).  The scheduler stages a whole host
// weight so its upload overlaps the previous split's compute; under -sm tensor that upload is spliced
// across N devices and the split's consumers read the per-device "simple" tensors rather than the
// split tensor's `data`, so neither the scheduler's single slot nor its data-pointer redirect can
// reach the op.  This backend therefore owns the whole lifecycle: each device's chunk goes into that
// device's own ring on the device's auxiliary copy stream, `graph_compute` waits for the copies on
// each device's main stream and points the simple tensors at the slots, and the slots are freed once
// the child graphs have been launched (see `ggml_backend_meta_stage_guard`).
static ggml_backend_event_t ggml_backend_meta_stage_ev(ggml_backend_meta_context * backend_ctx,
        std::vector<std::vector<ggml_backend_event_t>> & evs, size_t device, int slot) {
    if (evs[device][slot] == nullptr) {
        evs[device][slot] = ggml_backend_event_new(
            ggml_backend_get_device(backend_ctx->backend_configs[device].backend));
    }
    return evs[device][slot];
}

// TEMP INSTRUMENT (wip/tensor-split-expert-split): is the ring's upload actually overlapped with
// compute?  `wait_us` is the time a device's main stream spent blocked on the upload it needed, i.e.
// upload latency that was NOT hidden.
static int64_t g_ring_wait_us = 0;
static int64_t g_ring_wait_n  = 0;
static int64_t g_ring_gap_us  = 0;
static bool    g_ring_reg     = false;
static void g_ring_dump(void) {
    fprintf(stderr, "RING waits=%lld wait_total=%lld us (%.1f ms) gap_total=%lld us (%.1f ms)\n",
        (long long) g_ring_wait_n, (long long) g_ring_wait_us, double(g_ring_wait_us)/1000.0,
        (long long) g_ring_gap_us, double(g_ring_gap_us)/1000.0);
}
static void g_ring_add(const int64_t wait_us, const int64_t gap_us) {
    if (GGML_ENV_STR("GGML_RING_STATS") == nullptr) {
        return;
    }
    if (!g_ring_reg) { g_ring_reg = true; atexit(g_ring_dump); }
    g_ring_wait_us += wait_us;
    g_ring_gap_us  += gap_us;
    g_ring_wait_n  += 1;
}

static bool ggml_backend_meta_stage_input(ggml_backend_t backend, ggml_tensor * input, ggml_tensor * input_cpy) {
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    const bool stage_dbg = GGML_ENV_STR("GGML_META_STAGEDBG") != nullptr;
    if (stage_dbg) {
        fprintf(stderr, "STAGEDBG enter name=%s nbytes=%zu contiguous=%d\n",
            input_cpy->name, ggml_nbytes(input_cpy), (int) ggml_is_contiguous(input_cpy));
    }

    if (!ggml_backend_buffer_is_meta(input_cpy->buffer) || !ggml_is_contiguous(input_cpy)) {
        return false;
    }

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(input_cpy, /*assume_sync =*/ false);
    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        return false;
    }
    const bool mirrored = split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    if (!mirrored && (split_state.axis < 0 || split_state.axis >= GGML_MAX_DIMS)) {
        if (stage_dbg) fprintf(stderr, "STAGEDBG bad split_state.axis=%d\n", (int) split_state.axis);
        return false;
    }

    // MIRRORED: a single contiguous copy per device covers the whole tensor.
    // Split op-offload (wip/tensor-split-expert-split): the tensor is uploaded whole and each device
    // owns a *strided* slice (every `chunk_size_full`, take `chunk_size_j`), so it is gathered on the
    // host into the ring slot -- which keeps the transfer on the copy stream (queueable) instead of the
    // in-order spliced path, whose pre-upload host synchronize exposed it (§25).
    const size_t size = ggml_nbytes(input_cpy);
    const size_t chunk_size_full = mirrored ? size : input_cpy->nb[split_state.axis + 1];
    if (chunk_size_full == 0) {
        return false;
    }
    const bool gather = !mirrored && chunk_size_full != size;
    if (!mirrored && !gather && chunk_size_full != size) {
        if (stage_dbg) fprintf(stderr, "STAGEDBG not a whole number of chunks name=%s\n", input_cpy->name);
        return false;
    }
    if (gather && (size % chunk_size_full) != 0) {
        if (stage_dbg) fprintf(stderr, "STAGEDBG size%%chunk_full != 0 name=%s size=%zu cf=%zu\n", input_cpy->name, size, chunk_size_full);
        return false;
    }
    const size_t n_chunks = gather ? size / chunk_size_full : 0;
    if (stage_dbg) {
        fprintf(stderr, "STAGEDBG gate name=%s size=%zu chunk_full=%zu mirrored=%d gather=%d n_chunks=%zu axis=%d\n",
            input_cpy->name, size, chunk_size_full, (int) mirrored, (int) gather, n_chunks, (int) split_state.axis);
    }

    const int slot = backend_ctx->stage_slot_next;
    backend_ctx->stage_slot_next = (backend_ctx->stage_slot_next + 1) % backend_ctx->stage_n_slots;

    // Reserve every device's ring slot before anything is uploaded: a ring that cannot hold its chunk
    // aborts the whole attempt, because staging only part of an upload would leave the rest on the
    // pruned path (a full device synchronize plus an ids readback), which measured worse than either
    // path on its own.
    ggml_backend_meta_context::stage_entry entry;
    entry.tensor = input_cpy;
    entry.slot   = slot;
    size_t offset_j = 0;
    for (size_t j = 0; j < n_backends; j++) {
        ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
        ggml_tensor *  simple_tensor  = ggml_backend_meta_buffer_simple_tensor(input_cpy, j);
        if (simple_tensor == nullptr ||
            simple_backend->iface.stage_buffer == nullptr ||
            simple_backend->iface.stage_upload == nullptr ||
            simple_backend->iface.stage_wait == nullptr ||
            (gather && simple_backend->iface.stage_gather == nullptr)) {
            return false;
        }
        const size_t chunk_size_j = mirrored ? ggml_nbytes(simple_tensor) : simple_tensor->nb[split_state.axis + 1];
        if (chunk_size_j == 0) {
            continue;
        }
        const size_t copy_bytes = gather ? n_chunks * chunk_size_j : chunk_size_j;
        void * slot_ptr = simple_backend->iface.stage_buffer(simple_backend, slot, copy_bytes);
        if (slot_ptr == nullptr) {
            if (stage_dbg) fprintf(stderr, "STAGEDBG stage_buffer FAIL dev=%zu bytes=%zu\n", j, copy_bytes);
            return false;
        }
        if (stage_dbg) fprintf(stderr, "STAGEDBG dev=%zu chunk_j=%zu copy_bytes=%zu slot=%p\n", j, chunk_size_j, copy_bytes, slot_ptr);
        ggml_backend_meta_context::stage_chunk chunk;
        chunk.backend  = simple_backend;
        chunk.device   = j;
        chunk.slot     = slot_ptr;
        chunk.size     = chunk_size_j;
        chunk.src      = mirrored ? 0 : offset_j;
        chunk.stride   = gather ? chunk_size_full : 0;
        chunk.n_copies = gather ? n_chunks : 0;
        entry.chunks.push_back(chunk);
        if (!mirrored) {
            offset_j += chunk_size_j;
        }
    }
    if (entry.chunks.empty() || (!mirrored && offset_j != chunk_size_full)) {
        if (stage_dbg) fprintf(stderr, "STAGEDBG chunks empty or offset sum mismatch: offset_j=%zu chunk_full=%zu size=%zu\n", offset_j, chunk_size_full, size);
        return false;
    }
    for (auto & chunk : entry.chunks) {
        if (ggml_backend_meta_stage_ev(backend_ctx, backend_ctx->stage_done_ev, chunk.device, slot) == nullptr ||
            ggml_backend_meta_stage_ev(backend_ctx, backend_ctx->stage_free_ev, chunk.device, slot) == nullptr) {
            return false;
        }
    }

    // Issue the copies on each device's auxiliary copy stream: they are independent of the main
    // streams, so they run concurrently with the previous split's compute.
    for (auto & chunk : entry.chunks) {
        chunk.backend->iface.stage_wait(chunk.backend, backend_ctx->stage_free_ev[chunk.device][slot]);
        if (chunk.n_copies > 0) {
            chunk.backend->iface.stage_gather(chunk.backend, slot, (const char *) input->data, chunk.src,
                                              chunk.size, chunk.stride, chunk.n_copies,
                                              backend_ctx->stage_done_ev[chunk.device][slot]);
        } else {
            chunk.backend->iface.stage_upload(chunk.backend, chunk.slot, (const char *) input->data + chunk.src,
                                              chunk.size, backend_ctx->stage_done_ev[chunk.device][slot]);
        }
    }
    entry.t_issue = ggml_time_us();
    backend_ctx->stage_pending.push_back(std::move(entry));
    return true;
}

static float ggml_backend_meta_stage_h2d_gbps(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    float bw = 0.0f;
    for (size_t j = 0; j < n_backends; j++) {
        ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
        if (simple_backend->iface.stage_h2d_gbps == nullptr) {
            return 0.0f;
        }
        const float b = simple_backend->iface.stage_h2d_gbps(simple_backend);
        if (b <= 0.0f) {
            return 0.0f;
        }
        // the slowest link gates: every staged chunk has to overlap the previous split's compute
        if (bw == 0.0f || b < bw) {
            bw = b;
        }
    }
    return bw;
}

// The fused per-device kernels (see ggml_cuda_try_fuse) must not write into a
// source they still read. The scheduler allocates the graph's tensors before
// the Meta backend dispatches the per-device subgraphs, so the CUDA-side
// graph_optimize (which registers the alloc deps against the scheduler's
// allocator) never runs for graphs owned by this backend. Walk the graph for
// the same structural patterns here so the gallocr keeps the sources alive
// until the fused kernel's destination is written.
static void ggml_backend_meta_graph_optimize(ggml_backend_t backend,
                                             struct ggml_cgraph * cgraph,
                                             struct ggml_backend_graph_optimize_params * params) {
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    backend_ctx->opt_has_eval_callback = params != nullptr && params->has_eval_callback;

    // The scheduler never calls a child backend's graph_optimize for a graph this backend owns, so
    // forward its alloc-deps half here, over the whole scheduled graph (original tensors), while the
    // scheduler's collector is still live and before allocation.  A child's full pass also sets its
    // MMB markings, but those must be keyed by the per-device *simple* tensors the child compute
    // sees; that half is deferred to graph_compute (`marks_only`).  `allocs_only` here keeps the two
    // halves from clobbering each other (the marks would otherwise be built for the original tensor
    // pointers and then never found at compute time).
    if (params != nullptr) {
        struct ggml_backend_graph_optimize_params alloc_params = *params;
        alloc_params.marks_only  = false;
        alloc_params.allocs_only = true;
        for (size_t j = 0; j < backend_ctx->backend_configs.size(); j++) {
            ggml_backend_t b = backend_ctx->backend_configs[j].backend;
            if (b->iface.graph_optimize != nullptr) {
                b->iface.graph_optimize(b, cgraph, &alloc_params);
            }
        }
    }

    static const bool disable_fusion = GGML_ENV_STR("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(GGML_ENV_STR("GGML_CUDA_DISABLE_FUSION"));
    if (disable_fusion) {
        return;
    }

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        if (cgraph->nodes[i]->op != GGML_OP_MUL) {
            continue;
        }

        ggml_moe_weighted_reduction_match match;
        if (!ggml_match_moe_weighted_reduction(cgraph, i, match)) {
            continue;
        }

        params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
        params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
        if (match.expert_scale != nullptr) {
            params->add_alloc_dep(
                params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
        }
        i += match.node_count - 1;
    }
}

// RAII: once a split's child graphs have been launched their kernels hold the staged slot pointers,
// so the simple tensors can be pointed back at their real data and the slots freed -- on every exit
// path, an early error return included.  Recording the free event on the device's main stream is what
// lets a later upload reuse the slot through `stage_wait`.
struct ggml_backend_meta_stage_guard {
    ggml_backend_meta_context * ctx;
    explicit ggml_backend_meta_stage_guard(ggml_backend_meta_context * ctx) : ctx(ctx) {}
    ~ggml_backend_meta_stage_guard() {
        for (auto & entry : ctx->stage_pending) {
            for (auto & chunk : entry.chunks) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(entry.tensor, chunk.device);
                if (simple_tensor != nullptr) {
                    simple_tensor->data = chunk.orig;
                }
                ggml_backend_event_record(ctx->stage_free_ev[chunk.device][entry.slot], chunk.backend);
            }
        }
        ctx->stage_pending.clear();
    }
};

// TEMP INSTRUMENT (wip/tensor-split-expert-split): host-side cost of the meta backend's per-graph
// setup.  With op-offload the scheduler hands the meta backend a *one-node* graph per offloaded op, so
// this runs ~100x per pass -- the suspicion is that this, not the compute, is the tensor path's cost.
static int64_t g_meta_gc_us = 0;
static int64_t g_meta_gc_n  = 0;
static bool    g_meta_gc_reg = false;
static int64_t g_phase_us[5] = {0};
static void g_meta_gc_dump(void) {
    fprintf(stderr, "METAGCPHASE body=%.1fms map=%.1fms stage=%.1fms analysis=%.1fms dispatch=%.1fms\n",
        g_phase_us[0]/1000.0, g_phase_us[1]/1000.0, g_phase_us[2]/1000.0, g_phase_us[3]/1000.0, g_phase_us[4]/1000.0);
    fprintf(stderr, "METAGC calls=%lld host_total=%lld us (%.1f ms) avg=%.1f us\n",
        (long long) g_meta_gc_n, (long long) g_meta_gc_us, double(g_meta_gc_us)/1000.0,
        g_meta_gc_n ? double(g_meta_gc_us)/double(g_meta_gc_n) : 0.0);
}
struct meta_gc_timer {
    int64_t t0;
    meta_gc_timer() : t0(GGML_ENV_STR("GGML_META_GCDBG") ? ggml_time_us() : 0) {
        if (t0 && !g_meta_gc_reg) { g_meta_gc_reg = true; atexit(g_meta_gc_dump); }
    }
    ~meta_gc_timer() { if (t0) { g_meta_gc_us += ggml_time_us() - t0; g_meta_gc_n += 1; } }
};

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    const meta_gc_timer meta_gc_timer_inst;
    int64_t t_ph = ggml_time_us();
    GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    const bool needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    bool max_nnodes_raised = false;
    if (cgraph->n_nodes > backend_ctx->max_nnodes) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
        assert(needs_rebuild);
    }

    if (needs_rebuild) {
        std::set<ggml_backend_buffer_t> used_buffers;
        for (int i = 0; i < cgraph->n_leafs; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
                used_buffers.emplace(cgraph->leafs[i]->buffer);
            }
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
                used_buffers.emplace(cgraph->nodes[i]->buffer);
            }
        }
        for (ggml_backend_buffer_t buf : used_buffers) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
            buf_ctx->stc_compute_index_next = buf_ctx->stc_compute_index ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_index_next];
            for (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
    if (GGML_ENV_STR("GGML_META_GCDBG")) { int64_t _n = ggml_time_us(); g_phase_us[1] += _n - t_ph; t_ph = _n; }
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (ggml_backend_buffer_is_host(node->buffer)) {
                    // FIXME host tensors are not part of the meta split: passing them to
                    // ggml_backend_meta_buffer_simple_tensor crashes. This covers views of
                    // NONE-op leafs (e.g. s_copy_main) as well as views of computed host
                    // tensors (e.g. a reshaped token embedding) and host op results (e.g.
                    // the PLE get_rows); for regular usage these are noops in the split
                    // graph, so keeping the original tensor is fine.
                    bcj.nodes[i] = node;
                    continue;
                }
                bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(node, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        // op-offload H2D staging (issue #50 WIP): the inputs `stage_input` took over are in flight on
        // each device's auxiliary copy stream.  Make that device's main stream wait for its copy and
        // point its simple tensor at the slot so the child graphs consume the staged bytes; both
        // happen before any child graph is launched, and the guard below undoes them once they all
        // are.  The lookup is redone here rather than cached because a rebuild replaces the simple
        // tensors.
        for (auto & entry : backend_ctx->stage_pending) {
            for (auto & chunk : entry.chunks) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(entry.tensor, chunk.device);
                GGML_ASSERT(simple_tensor != nullptr);
                ggml_backend_event_t done_ev = backend_ctx->stage_done_ev[chunk.device][entry.slot];
                GGML_ASSERT(done_ev != nullptr);
                const int64_t t0 = ggml_time_us();
                ggml_backend_event_wait(chunk.backend, done_ev);
                g_ring_add(ggml_time_us() - t0, t0 - entry.t_issue);
                chunk.orig = simple_tensor->data;
                simple_tensor->data = chunk.slot;
            }
        }
    if (GGML_ENV_STR("GGML_META_GCDBG")) { int64_t _n = ggml_time_us(); g_phase_us[2] += _n - t_ph; t_ph = _n; }
        const ggml_backend_meta_stage_guard stage_guard(backend_ctx);

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed_branch = [&](const int i) -> int {
                int id = i; // i_delayed
                int idr = i; // i_delayed return, last safe return value

                ggml_tensor * node = cgraph->nodes[id];
                int32_t n_used = ggml_node_get_use_count(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    while (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            break;
                        }
                        bool safe = true;
                        for (int s = 0; s < GGML_MAX_SRC; s++) {
                            if (next->src[s] == nullptr) {
                                continue;
                            }
                            if (next->src[s] == node) {
                                safe = false;
                                break;
                            }
                            if (ggml_backend_meta_get_split_state(next->src[s], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = false;
                                break;
                            }
                        }
                        if (!safe) {
                            break;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                if (id + 1 >= cgraph->n_nodes) {
                    return idr;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_ADD_ID && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                while (true) {
                    skip_unrelated();
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_MUL && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    } else {
                        break;
                    }
                }

                if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    return idr;
                }
                for (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                            next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                            ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                for (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                idr = id;
                return idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            auto get_i_delayed = [&](const int i) -> int {
                const int i_delayed = get_i_delayed_branch(i);
                ggml_tensor * node = cgraph->nodes[i_delayed];

                if (ggml_node_get_use_count(cgraph, i_delayed) != 1) {
                    return i_delayed;
                }

                for (int id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * next = cgraph->nodes[id];
                    if (next->view_src == node) {
                        return i_delayed;
                    }
                    for (int s = 0; s < GGML_MAX_SRC; s++) {
                        if (next->src[s] == node) {
                            return i_delayed;
                        }
                    }

                    if (next->view_src != nullptr && next->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(next->view_src->buffer)) {
                        continue;
                    }
                    if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                        continue;
                    }

                    const int i_other = id;
                    const int i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    if (ggml_node_get_use_count(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        return i_delayed;
                    }

                    ggml_tensor * sum = cgraph->nodes[i_other_delayed + 1];
                    if (sum->op != GGML_OP_ADD ||
                            !ggml_are_same_shape(node, other) || node->type != other->type || sum->type != node->type ||
                            !((sum->src[0] == node && sum->src[1] == other) ||
                              (sum->src[0] == other && sum->src[1] == node)) ||
                            ggml_backend_meta_get_split_state(sum, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        return i_delayed;
                    }

                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        const bool compute       = bcj.nodes[i]->flags       & GGML_TENSOR_FLAG_COMPUTE;
                        const bool compute_other = bcj.nodes[i_other]->flags & GGML_TENSOR_FLAG_COMPUTE;
                        if (compute != compute_other) {
                            return i_delayed;
                        }
                    }
                    return i_other_delayed + 1;
                }
                return i_delayed;
            };

            int i_start = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    continue;
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false);
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    if (GGML_ENV_STR("GGML_META_REDDBG")) {
                        fprintf(stderr, "REDDBG PARTIAL node[%d]=%s op=%s nbytes=%zu ne=[%lld,%lld,%lld]\n",
                            i, node->name, ggml_op_name(node->op), ggml_nbytes(node),
                            (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2]);
                    }
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
                }
                const bool new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                if (!new_subgraph) {
                    continue;
                }

                const int i_delayed = get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                if (i_delayed > i) {
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        if ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            for (int ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].offset = i_start;
                }
                n_subgraphs++;
                i_start = i + 1;
            }
            GGML_ASSERT(i_start == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
    if (GGML_ENV_STR("GGML_META_GCDBG")) { int64_t _n = ggml_time_us(); g_phase_us[3] += _n - t_ph; t_ph = _n; }
        backend_ctx->n_subgraphs = n_subgraphs;

        if (GGML_ENV_STR("GGML_META_REDDBG")) {
            fprintf(stderr, "REDDBG max_tmp_size=%zu (ctx=%zu) n_subgraphs=%zu n_backends=%zu n_reduce_steps=%zu\n",
                max_tmp_size, backend_ctx->max_tmp_size, n_subgraphs, n_backends, backend_ctx->n_reduce_steps);
        }
        if (max_tmp_size > backend_ctx->max_tmp_size) {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        if (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            const size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            const size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            const size_t mem_per_device_graphs_main = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            const size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            const ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                // allocate ALL historically-used entries: an arena reset orphans any
                // entry in [n_subgraphs, max_subgraphs) and a later graph with fewer
                // subgraphs would reuse the dangling pointer without re-allocating
                for (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), cgraph->n_nodes, /*grads =*/ false);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                const size_t i_node_start = bcj.cgraphs[i_graph].offset;
                const size_t i_node_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_node_stop - i_node_start;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                for (size_t i_node = i_node_start; i_node < i_node_stop; i_node++) {
                    ggml_tensor * node_ij = bcj.nodes[i_node];
                    cgraph_ij->nodes[i_node - i_node_start] = node_ij;
                    const size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                    const size_t hash_pos_ij = ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_graph_next_uid();
            }
        }

        // The child backends' marking half.  The scheduler never calls their graph_optimize under
        // this backend, so the MMB HC16 / DOWN16 / blk16 / res16 markings would be inert.  Run them
        // on the per-device subgraphs now, before any of them is computed (the mark lifetime assumes
        // every subgraph of a graph is optimized before any is computed).  `marks_only` skips the
        // alloc-deps half, which graph_optimize already registered before allocation, and
        // `full_graph == NULL` treats each subgraph as self-contained: a marked producer and every
        // consumer reading through its BF16 activation cache are always on the same device.
        {
            struct ggml_backend_graph_optimize_params mark_params = {
                /* .add_alloc_dep     = */ [](void *, ggml_tensor *, ggml_tensor *) {},
                /* .user_data         = */ nullptr,
                /* .has_eval_callback = */ backend_ctx->opt_has_eval_callback,
                /* .full_graph        = */ nullptr,
                /* .marks_only        = */ true,
                /* .allocs_only       = */ false,
            };
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                if (bcj.backend->iface.graph_optimize == nullptr) {
                    continue;
                }
                for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                    bcj.backend->iface.graph_optimize(bcj.backend, bcj.cgraphs[i_graph].cgraph_main, &mark_params);
                }
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_node_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, sizeof(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    };
    auto set_tmp_data = [&](ggml_tensor * tensor, const size_t j, const size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        if (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
        if (GGML_ENV_STR("GGML_META_REDDBG")) {
            static int n = 0;
            if (n++ < 40) {
                fprintf(stderr, "REDDBG set_tmp dev=%zu i_buf=%zu buf=%p size=%zu tensor=%s nbytes=%zu data=%p\n",
                    j, i_buf, (void *) buf_ptr.get(), ggml_backend_buffer_get_size(buf_ptr.get()),
                    tensor->name, ggml_nbytes(tensor), tensor->data);
            }
        }
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        return ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * node = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
                continue;
            }
            ggml_tensor * node_zero = get_node_aux(node);
            node_zero->op = GGML_OP_SCALE; // FIXME 0.0f * NaN == NaN
            node_zero->src[0] = node;
            ggml_set_op_params_f32(node_zero, 0, 0.0f);
            node_zero->data = node->data;
            node_zero->buffer = node->buffer;
            node_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

            step_cgraphs[j] = get_cgraph_aux();
            step_cgraphs[j]->nodes[0] = node_zero;
            step_cgraphs[j]->n_nodes = 1;
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto push_data = [&](const size_t j_src, const size_t j_dst, const size_t i_buf) {
            assert(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(node_src));
            GGML_ASSERT(ggml_is_contiguous(node_dst));

            ggml_tensor * node_tmp = get_node_aux(node_dst);
            set_tmp_data(node_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_tmp);

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        size_t offset_j = n_backends/2;
        while ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        const size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        for (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            const size_t j_dst = j_src - 2*offset_j_max;
            push_data(j_src, j_dst, i_buf);
            const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        for (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                const size_t j_other = j ^ offset_j;
                if (j_other >= n_backends) {
                    continue;
                }
                push_data(j, j_other, i_buf);
            }

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                if (step_cgraphs[j] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            i_buf++;
        }
        assert(i_buf == backend_ctx->n_reduce_steps);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        for (size_t j = 2*offset_j_max; j < n_backends; j++) {
            auto & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            auto & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_dst);
        }

        return GGML_STATUS_SUCCESS;
    };


    if (GGML_ENV_STR("GGML_META_EXECDBG")) {
        fprintf(stderr, "METAEXEC n_subgraphs=%zu n_backends=%zu reduce_steps=%zu\n",
            backend_ctx->n_subgraphs, n_backends, backend_ctx->n_reduce_steps);
    }
    if (GGML_ENV_STR("GGML_META_GCDBG")) { int64_t _n = ggml_time_us(); g_phase_us[4] += _n - t_ph; t_ph = _n; }
    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            if (GGML_ENV_STR("GGML_META_EXECDBG")) {
                ggml_cgraph * cg = bcj.cgraphs[i].cgraph_main;
                ggml_tensor * f = cg->n_nodes > 0 ? cg->nodes[0] : nullptr;
                ggml_tensor * l = cg->n_nodes > 0 ? cg->nodes[cg->n_nodes - 1] : nullptr;
                fprintf(stderr, "METAEXEC sub=%zu dev=%zu nodes=%d first=%s(%s) last=%s(%s)\n", i, j, cg->n_nodes,
                    f ? f->name : "-", f ? ggml_op_name(f->op) : "-", l ? l->name : "-", l ? ggml_op_name(l->op) : "-");
                // exp15: dump the pointers/geometry the device is about to use, so a fault address can be
                // matched against them.
                for (int q = 0; q < cg->n_nodes; ++q) {
                    ggml_tensor * n = cg->nodes[q];
                    if (n->op != GGML_OP_MUL_MAT_ID && n->op != GGML_OP_MUL_MAT && n->op != GGML_OP_GLU) continue;
                    fprintf(stderr, "METAEXEC   %s op=%s data=%p ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
                        n->name, ggml_op_name(n->op), n->data,
                        (long long)n->ne[0], (long long)n->ne[1], (long long)n->ne[2], (long long)n->ne[3],
                        n->nb[0], n->nb[1], n->nb[2], n->nb[3]);
                    for (int s = 0; s < GGML_MAX_SRC; ++s) {
                        const ggml_tensor * sr = n->src[s];
                        if (sr == nullptr) continue;
                        fprintf(stderr, "METAEXEC     src%d %s op=%s data=%p ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
                            s, sr->name, ggml_op_name(sr->op), sr->data,
                            (long long)sr->ne[0], (long long)sr->ne[1], (long long)sr->ne[2], (long long)sr->ne[3],
                            sr->nb[0], sr->nb[1], sr->nb[2], sr->nb[3]);
                        if (n->op == GGML_OP_MUL_MAT_ID && s == 2 && GGML_ENV_STR("GGML_META_READIDS") != nullptr) {
                            fprintf(stderr, "READEIDS begin %s\n", sr->name);
                            // read the ids the way the kernel does: element (i0,i1) is at
                            // data + i1*nb[1] + i0*nb[0] (the topk view is strided)
                            const size_t n_ids = ggml_nelements(sr);
                            std::vector<int32_t> idsbuf(n_ids, 0);
                            {
                                // a strided view's logical nbytes is far smaller than its stride span, so
                                // read through the parent (view_src) and add the view offset
                                const ggml_tensor * par = sr->view_src != nullptr ? sr->view_src : sr;
                                const size_t base = sr->view_src != nullptr ? sr->view_offs : 0;
                                const size_t span = base + (size_t)(sr->ne[1]-1)*sr->nb[1] + (size_t)(sr->ne[0]-1)*sr->nb[0] + sizeof(int32_t);
                                std::vector<char> raw(std::min<size_t>(span, par->nb[1]*par->ne[1]));
                                ggml_backend_tensor_get(par, raw.data(), 0, raw.size());
                                for (int64_t i1 = 0; i1 < sr->ne[1]; ++i1)
                                for (int64_t i0 = 0; i0 < sr->ne[0]; ++i0) {
                                    const size_t off = base + i1*sr->nb[1] + i0*sr->nb[0];
                                    if (off + 4 <= raw.size()) idsbuf[i1*sr->ne[0] + i0] = *(const int32_t *)(raw.data() + off);
                                    else idsbuf[i1*sr->ne[0] + i0] = -12345;
                                }
                            }
                            const int n_exp = (int) n->src[0]->ne[2];
                            int mn = 1<<30, mx = -(1<<30); size_t bad = 0; int firstbad = -1;
                            for (size_t q = 0; q < n_ids; ++q) {
                                const int v = idsbuf[q];
                                mn = std::min(mn, v); mx = std::max(mx, v);
                                if (v < 0 || v >= n_exp) { bad++; if (firstbad < 0) firstbad = (int) q; }
                            }
                            fprintf(stderr, "METAEXEC       ids n=%zu n_expert=%d min=%d max=%d out_of_range=%zu firstbad=%d\n",
                                n_ids, n_exp, mn, mx, bad, firstbad);
                            fprintf(stderr, "READEIDS end   %s\n", sr->name);
                        }
                    }
                }
            }
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            // exp19 bisect: with GGML_META_SYNCEACH the fault surfaces immediately after the guilty
            // launch instead of at an arbitrary later host-side drain, which names the subgraph.
            if (GGML_ENV_STR("GGML_META_SYNCEACH") != nullptr) {
                ggml_backend_synchronize(bcj.backend);
                fprintf(stderr, "SYNCEACH sub=%zu dev=%zu nodes=%d first=%s last=%s\n",
                    i, j, bcj.cgraphs[i].cgraph_main->n_nodes,
                    bcj.cgraphs[i].cgraph_main->n_nodes ? bcj.cgraphs[i].cgraph_main->nodes[0]->name : "-",
                    bcj.cgraphs[i].cgraph_main->n_nodes ? bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes-1]->name : "-");
            }
        }

        if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            bool backend_allreduce_success = false;
            if (backend_ctx->comm_ctx) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
            }

            if (!backend_allreduce_success) {
                const ggml_status status = allreduce_fallback(i);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            // exp22: the per-subgraph SYNCEACH above runs *before* the reduce, so a fault in the
            // reduce/butterfly surfaces one graph later.  Drain here too, to attribute it.
            if (GGML_ENV_STR("GGML_META_SYNCEACH") != nullptr) {
                for (size_t j = 0; j < n_backends; j++) {
                    ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
                }
                fprintf(stderr, "SYNCEACH REDUCE sub=%zu ok\n", i);
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .stage_buffer            = */ nullptr, // the meta backend stages per device instead: stage_input
    /* .stage_upload            = */ nullptr,
    /* .stage_gather            = */ nullptr,
    /* .stage_wait              = */ nullptr,
    /* .stage_d2d               = */ nullptr,
    /* .stage_h2d_gbps          = */ ggml_backend_meta_stage_h2d_gbps,
    /* .stage_input             = */ ggml_backend_meta_stage_input,
    /* .graph_optimize          = */ ggml_backend_meta_graph_optimize,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}
