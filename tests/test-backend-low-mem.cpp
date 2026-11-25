// run ops on GPU devices with almost no free memory left and compare against the CPU backend

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstdio>
#include <random>
#include <thread>
#include <vector>

// fill device memory until about `leave` bytes are free
static std::vector<ggml_backend_buffer_t> fill_memory(ggml_backend_dev_t dev, size_t leave) {
    std::vector<ggml_backend_buffer_t> fill;
    size_t                             free, total;
    ggml_backend_dev_memory(dev, &free, &total);
    while (free > leave + (16u << 20)) {
        const size_t          n   = free - leave > (1ull << 30) ? (1ull << 30) : (16u << 20);
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), n);
        if (buf == nullptr) {
            break;
        }
        fill.push_back(buf);
        ggml_backend_dev_memory(dev, &free, &total);
    }
    return fill;
}

// returns an empty vector if the backend does not support the op
static std::vector<float> conv_2d(ggml_backend_t                   backend,
                                  const std::vector<float> &       x,
                                  const std::vector<ggml_fp16_t> & k,
                                  int                              w,
                                  int                              c,
                                  size_t                           leave) {
    ggml_init_params params = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx    = ggml_init(params);

    ggml_tensor * kt = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, c, c);
    ggml_tensor * xt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w, w, c, 1);
    ggml_tensor * yt = ggml_conv_2d_direct(ctx, kt, xt, 1, 1, 1, 1, 1, 1);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, yt);

    std::vector<float> y;
    if (ggml_backend_supports_op(backend, yt)) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(kt, k.data(), 0, ggml_nbytes(kt));
        ggml_backend_tensor_set(xt, x.data(), 0, ggml_nbytes(xt));

        std::vector<ggml_backend_buffer_t> fill;
        if (leave > 0) {
            fill = fill_memory(ggml_backend_get_device(backend), leave);
        }
        ggml_backend_graph_compute(backend, gf);

        y.resize(ggml_nelements(yt));
        ggml_backend_tensor_get(yt, y.data(), 0, ggml_nbytes(yt));

        for (ggml_backend_buffer_t b : fill) {
            ggml_backend_buffer_free(b);
        }
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    return y;
}

int main() {
    // the staged input of this conv is about 270 MB, more than will be free
    // if a backend stages it in one piece, the run aborts instead of passing
    const int    w     = 4096;
    const int    c     = 8;
    const size_t leave = 128u << 20;

    std::mt19937                          rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float>                    x(size_t(w) * w * c);
    for (float & v : x) {
        v = dist(rng);
    }
    std::vector<ggml_fp16_t> k(size_t(3) * 3 * c * c);
    for (ggml_fp16_t & v : k) {
        v = ggml_fp32_to_fp16(0.1f * dist(rng));
    }

    std::vector<float> ref;
    int                n_run = 0;
    int                n_ok  = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_t           backend = ggml_backend_dev_init(dev, nullptr);
        const std::vector<float> y       = conv_2d(backend, x, k, w, c, leave);
        ggml_backend_free(backend);
        if (y.empty()) {
            printf("%s: CONV_2D not supported, skipping\n", ggml_backend_dev_name(dev));
            continue;
        }

        if (ref.empty()) {
            ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
            ggml_backend_cpu_set_n_threads(cpu, std::thread::hardware_concurrency());
            ref = conv_2d(cpu, x, k, w, c, 0);
            ggml_backend_free(cpu);
        }

        double err = 0.0;
        double sum = 0.0;
        for (size_t j = 0; j < y.size(); j++) {
            err += (y[j] - ref[j]) * (y[j] - ref[j]);
            sum += ref[j] * ref[j];
        }
        const double nmse = err / sum;
        const bool   ok   = nmse < 1e-6;
        printf("%s: CONV_2D %dx%dx%d with %zu MB free: nmse = %g %s\n", ggml_backend_dev_name(dev), w, w, c,
               leave >> 20, nmse, ok ? "OK" : "FAIL");
        n_run++;
        n_ok += ok;
    }

    if (n_run == 0) {
        printf("no GPU device with CONV_2D, skipping\n");
    }
    return n_ok == n_run ? 0 : 1;
}
