// GPU correctness test for the logit_bias kernels: the sparse (id,val) scatter and the elementwise
// bias application. Mirrors penalty_gpu_test.cpp's Scratch-struct-plus-direct-kernel-call
// structure, no model needed.
#include "sparkinfer/kernels/fused.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

using sparkinfer::kernels::launch_logit_bias;
using sparkinfer::kernels::launch_scatter_logit_bias;

struct Scratch {
    int vocab = 0;
    float* bias = nullptr;
    int* d_ids = nullptr;
    float* d_vals = nullptr;

    explicit Scratch(int v, int max_entries = 16) : vocab(v) {
        cudaMalloc(&bias, (size_t)vocab * sizeof(float));
        cudaMalloc(&d_ids, (size_t)max_entries * sizeof(int));
        cudaMalloc(&d_vals, (size_t)max_entries * sizeof(float));
        cudaMemset(bias, 0, (size_t)vocab * sizeof(float));
        cudaDeviceSynchronize();
    }
    ~Scratch() { cudaFree(bias); cudaFree(d_ids); cudaFree(d_vals); }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
};

std::vector<float> get_bias(Scratch& s) {
    std::vector<float> out(s.vocab);
    cudaMemcpy(out.data(), s.bias, (size_t)s.vocab * sizeof(float), cudaMemcpyDeviceToHost);
    return out;
}

void scatter(Scratch& s, const std::vector<int>& ids, const std::vector<float>& vals) {
    const int k = (int)ids.size();
    cudaMemcpy(s.d_ids, ids.data(), k * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_vals, vals.data(), k * sizeof(float), cudaMemcpyHostToDevice);
    launch_scatter_logit_bias(s.bias, s.d_ids, s.d_vals, k, s.vocab);
    cudaDeviceSynchronize();
}

std::vector<float> apply_bias(Scratch& s, std::vector<float> logits) {
    const int vocab = (int)logits.size();
    float* d_logits = nullptr;
    cudaMalloc(&d_logits, (size_t)vocab * sizeof(float));
    cudaMemcpy(d_logits, logits.data(), (size_t)vocab * sizeof(float), cudaMemcpyHostToDevice);
    launch_logit_bias(d_logits, s.bias, vocab);
    cudaMemcpy(logits.data(), d_logits, (size_t)vocab * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_logits);
    return logits;
}

bool nearly_equal(float a, float b, float tol = 1e-4f) { return std::fabs(a - b) < tol; }

bool test_scatter_correctness() {
    const int vocab = 16;
    Scratch s(vocab);
    scatter(s, {2, 9, 15}, {-100.f, 5.5f, 42.f});
    const auto bias = get_bias(s);
    bool ok = true;
    for (int v = 0; v < vocab; v++) {
        float expect = (v == 2) ? -100.f : (v == 9) ? 5.5f : (v == 15) ? 42.f : 0.f;
        if (!nearly_equal(bias[v], expect)) {
            printf("FAIL: scatter v=%d got=%.6f expect=%.6f\n", v, bias[v], expect);
            ok = false;
        }
    }
    if (ok) printf("[OK] scatter lands sparse (id,val) pairs at the right offsets, rest stay zero\n");
    return ok;
}

bool test_scatter_out_of_range_id_dropped() {
    const int vocab = 8;
    Scratch s(vocab);
    // ids include one in-range and two out-of-range (negative, >= vocab) entries -- the kernel's
    // defensive bound check must drop the latter two without touching adjacent memory.
    scatter(s, {3, -1, 1000}, {7.f, 999.f, 999.f});
    const auto bias = get_bias(s);
    bool ok = true;
    for (int v = 0; v < vocab; v++) {
        float expect = (v == 3) ? 7.f : 0.f;
        if (!nearly_equal(bias[v], expect)) {
            printf("FAIL: bound-safety v=%d got=%.6f expect=%.6f\n", v, bias[v], expect);
            ok = false;
        }
    }
    if (ok) printf("[OK] out-of-range scatter ids are dropped, no corruption of adjacent memory\n");
    return ok;
}

bool test_apply_correctness() {
    const int vocab = 8;
    Scratch s(vocab);
    scatter(s, {0, 3, 7}, {10.f, -5.f, 2.5f});
    const std::vector<float> logits = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    const auto out = apply_bias(s, logits);
    const std::vector<float> expect = {11.f, 2.f, 3.f, -1.f, 5.f, 6.f, 7.f, 10.5f};
    bool ok = true;
    for (int v = 0; v < vocab; v++) {
        if (!nearly_equal(out[v], expect[v])) {
            printf("FAIL: apply v=%d got=%.6f expect=%.6f\n", v, out[v], expect[v]);
            ok = false;
        }
    }
    if (ok) printf("[OK] apply kernel matches hand-computed logits[v] += bias[v]\n");
    return ok;
}

bool test_reset_clears_prior_scatter() {
    // Mirrors Qwen35Model::set_logit_bias's own sequence: a second call (smaller/empty map) must
    // fully clear entries a PRIOR call set -- the session-0-reuse safety property this feature
    // depends on.
    const int vocab = 8;
    Scratch s(vocab);
    scatter(s, {1, 2, 3, 4}, {1.f, 2.f, 3.f, 4.f});
    auto after_first = get_bias(s);
    bool first_ok = nearly_equal(after_first[1], 1.f) && nearly_equal(after_first[4], 4.f);

    // Same reset-then-scatter sequence set_logit_bias performs: memset zero, then (possibly empty)
    // scatter.
    cudaMemsetAsync(s.bias, 0, (size_t)vocab * sizeof(float));
    cudaDeviceSynchronize();
    scatter(s, {5}, {9.f});
    const auto after_second = get_bias(s);
    bool ok = first_ok;
    for (int v = 0; v < vocab; v++) {
        float expect = (v == 5) ? 9.f : 0.f;
        if (!nearly_equal(after_second[v], expect)) {
            printf("FAIL: reset-then-scatter v=%d got=%.6f expect=%.6f (stale bias survived)\n",
                   v, after_second[v], expect);
            ok = false;
        }
    }
    if (ok) printf("[OK] reset genuinely clears a prior call's bias before the next scatter\n");
    return ok;
}

bool test_combined_pipeline_matches_cpu_reference() {
    const int vocab = 32;
    Scratch s(vocab);
    std::vector<int> ids = {0, 5, 10, 17, 31};
    std::vector<float> vals = {-100.f, 3.25f, -3.25f, 100.f, -1.f};
    scatter(s, ids, vals);

    std::vector<float> cpu_bias(vocab, 0.f);
    for (size_t i = 0; i < ids.size(); i++) cpu_bias[ids[i]] = vals[i];

    std::vector<float> logits(vocab);
    for (int v = 0; v < vocab; v++) logits[v] = (float)v * 0.1f;
    std::vector<float> cpu_expect(vocab);
    for (int v = 0; v < vocab; v++) cpu_expect[v] = logits[v] + cpu_bias[v];

    const auto out = apply_bias(s, logits);
    bool ok = true;
    for (int v = 0; v < vocab; v++) {
        if (!nearly_equal(out[v], cpu_expect[v])) {
            printf("FAIL: combined pipeline v=%d got=%.6f expect=%.6f\n", v, out[v], cpu_expect[v]);
            ok = false;
        }
    }
    if (ok) printf("[OK] scatter+apply pipeline matches a CPU reference\n");
    return ok;
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no CUDA device\n");
        return 0;
    }
    bool ok = true;
    ok = test_scatter_correctness() && ok;
    ok = test_scatter_out_of_range_id_dropped() && ok;
    ok = test_apply_correctness() && ok;
    ok = test_reset_clears_prior_scatter() && ok;
    ok = test_combined_pipeline_matches_cpu_reference() && ok;
    if (!ok) return 1;
    printf("logit_bias_gpu_test: OK\n");
    return 0;
}
