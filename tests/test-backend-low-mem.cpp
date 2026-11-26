// CONV_2D on a GPU with almost no free memory, checked against the CPU.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

static std::vector<ggml_backend_buffer_t> fill_memory(ggml_backend_dev_t dev, size_t leave) {
    std::vector<ggml_backend_buffer_t> fill;
    size_t mem_free, mem_total;
    ggml_backend_dev_memory(dev, &mem_free, &mem_total);
    while (mem_free > leave + (16u << 20)) {
        const size_t n = mem_free - leave > (1ull << 30) ? (1ull << 30) : (16u << 20);
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), n);
        if (buf == nullptr) {
            break;
        }
        fill.push_back(buf);
        ggml_backend_dev_memory(dev, &mem_free, &mem_total);
    }
    return fill;
}

// Each case stages about 540 MB of input, more than will be free.
struct conv_case {
    const char * name;
    int          w, c, k, s, p, d, n;
    ggml_type    type_kernel;
};

// Empty if the backend does not support the op.
static std::vector<float> conv_2d(ggml_backend_t backend, const conv_case & cc, const std::vector<float> & x, const std::vector<float> & k, size_t leave) {
    ggml_init_params params = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * kt = ggml_new_tensor_4d(ctx, cc.type_kernel, cc.k, cc.k, cc.c, cc.c);
    ggml_tensor * xt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cc.w, cc.w, cc.c, cc.n);
    ggml_tensor * yt = ggml_conv_2d_direct(ctx, kt, xt, cc.s, cc.s, cc.p, cc.p, cc.d, cc.d);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, yt);

    std::vector<float> y;
    if (ggml_backend_supports_op(backend, yt)) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == nullptr) {
            printf("%s: not enough memory for the tensors, skipping\n", ggml_backend_name(backend));
            ggml_free(ctx);
            return y;
        }
        ggml_backend_tensor_set(xt, x.data(), 0, ggml_nbytes(xt));
        if (cc.type_kernel == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> k16(k.size());
            ggml_fp32_to_fp16_row(k.data(), k16.data(), k.size());
            ggml_backend_tensor_set(kt, k16.data(), 0, ggml_nbytes(kt));
        } else {
            ggml_backend_tensor_set(kt, k.data(), 0, ggml_nbytes(kt));
        }

        std::vector<ggml_backend_buffer_t> fill;
        if (leave > 0) {
            fill = fill_memory(ggml_backend_get_device(backend), leave);
            size_t mem_free, mem_total;
            ggml_backend_dev_memory(ggml_backend_get_device(backend), &mem_free, &mem_total);
            printf("%s: %zu MB free\n", ggml_backend_name(backend), mem_free >> 20);
        }
        // NaN on failure, so the case fails.
        y.resize(ggml_nelements(yt), NAN);
        if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
            ggml_backend_tensor_get(yt, y.data(), 0, ggml_nbytes(yt));
        }

        for (ggml_backend_buffer_t b : fill) {
            ggml_backend_buffer_free(b);
        }
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    return y;
}

int main() {
    ggml_backend_load_all();

    // conv2d stages at most 256 MB at a time, leave room for that.
    const size_t leave = 384u << 20;

    const conv_case cases[] = {
        { "3x3",          4096, 16, 3, 1, 1, 1, 1, GGML_TYPE_F16 },
        { "1x1",          4096, 16, 1, 1, 0, 1, 1, GGML_TYPE_F16 },
        { "3x3 f32",      4096, 16, 3, 1, 1, 1, 1, GGML_TYPE_F32 },
        { "3x3 stride 2", 4096, 16, 3, 2, 1, 1, 1, GGML_TYPE_F16 },
        { "3x3 dilation", 4096, 16, 3, 1, 2, 2, 1, GGML_TYPE_F16 },
        { "3x3 batch 2",  2900, 16, 3, 1, 1, 1, 2, GGML_TYPE_F16 },
    };

    std::vector<ggml_backend_dev_t> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpus.push_back(ggml_backend_dev_get(i));
        }
    }
    if (gpus.empty()) {
        printf("no GPU device, skipping\n");
        return 0;
    }

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads");
    if (set_n_threads) {
        set_n_threads(cpu, std::thread::hardware_concurrency());
    }

    int n_fail = 0;
    for (const conv_case & cc : cases) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> x(size_t(cc.w) * cc.w * cc.c * cc.n);
        for (float & v : x) {
            v = dist(rng);
        }
        std::vector<float> k(size_t(cc.k) * cc.k * cc.c * cc.c);
        for (float & v : k) {
            v = ggml_fp16_to_fp32(ggml_fp32_to_fp16(0.1f * dist(rng)));
        }

        std::vector<float> ref;
        for (ggml_backend_dev_t dev : gpus) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            const std::vector<float> y = conv_2d(backend, cc, x, k, leave);
            ggml_backend_free(backend);
            if (y.empty()) {
                printf("%s: CONV_2D %s not supported, skipping\n", ggml_backend_dev_name(dev), cc.name);
                continue;
            }
            if (ref.empty()) {
                ref = conv_2d(cpu, cc, x, k, 0);
            }

            double err = 0.0;
            double sum = 0.0;
            for (size_t j = 0; j < y.size(); j++) {
                err += (y[j] - ref[j]) * (y[j] - ref[j]);
                sum += ref[j] * ref[j];
            }
            const double nmse = err / sum;
            const bool   ok   = nmse < 5e-4;
            printf("%s: CONV_2D %s with %zu MB free: nmse = %g %s\n", ggml_backend_dev_name(dev), cc.name, leave >> 20, nmse, ok ? "OK" : "FAIL");
            n_fail += !ok;
        }
    }

    ggml_backend_free(cpu);
    return n_fail == 0 ? 0 : 1;
}
