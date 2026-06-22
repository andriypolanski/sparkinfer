# sparkinfer

**Blackwell-native MoE/LLM inference runtime.** The engineering arm of [SN74 on Gittensor](https://github.com/gittensor-ai-lab) — reproducible, hardware-level inference-speed gains for NVIDIA Blackwell consumer/edge GPUs: RTX Spark (`sm_121`), RTX 5090 & RTX PRO 6000 (`sm_120`), Jetson Thor (`sm_121`).

## Proven

Qwen3-30B-A3B (Q4_K_M GGUF) runs end-to-end on an RTX PRO 6000 (sm_120), decode optimized **0.60 → 134 tok/s (≈220×)** across 6 source-verifiable passes, output verified correct, **21.7 GB** resident (experts kept quantized). Independently verified on an **RTX 5090** (CUDA 13): **100% top-1 token agreement** with llama.cpp (KL ≈ 0.14 nats), **163.88 tok/s** — see the [accuracy](bench/results/accuracy_qwen3-30b-a3b_q4km.md) and [RTX 5090](bench/results/qwen3-30b-a3b_q4km_rtx5090.md) results.

## Quickstart

On an NVIDIA Blackwell box (CUDA 12.8+) — the scripts auto-detect your GPU arch, fetch **prebuilt binaries** (or build from source if incompatible), and download the model:

```bash
# decode throughput (fetches Qwen3-30B-A3B Q4_K_M on first run)
bench/scripts/bench.sh --download

# head-to-head vs llama.cpp on the same GGUF + GPU
bench/scripts/bench.sh --download --compare

# accuracy gate — token-match / KL / perplexity vs llama.cpp
bench/scripts/accuracy.sh --download
```

Your own model: `bench/scripts/bench.sh /path/to/model.gguf --tokens 256`. All options: [`bench/scripts/README.md`](bench/scripts/README.md).

## Layout & emission weights

| Path | Weight | What |
|---|--:|---|
| [`kernels/`](kernels) | **0.42** | CUDA kernels — flash-decode (hd128/256/512), decode GEMV, fused quantized MoE expert FFN, GEMM, RMSNorm, RoPE, GGUF dequant |
| [`runtime/`](runtime) | **0.26** | scheduler, paged KV cache, CUDA-graph decode, native GGUF loading, model forward |
| [`moe/`](moe) | **0.21** | sync-free MoE router + expert dispatch (on-device counts, CUDA-graph-ready) |
| [`bench/`](bench) | **0.11** | reproducible benchmarks + eval harness (source-required builds, frozen weights) |

`Weight` = intra-repo emission share for SN74 (**path-based**, sums to 1.0; see [`.gittensor/weights.json`](.gittensor/weights.json)). Performance paths (`kernels`/`runtime`/`moe`) are scored by **verified frontier-delta speedup** (XL/L/M/S/XS); `bench/` by code quality. Weights are **maturity-adaptive** — see the [org reward model](https://github.com/gittensor-ai-lab).

## Build

Requires **CUDA Toolkit 12.8+** (first toolkit with `sm_120` / `sm_121` codegen).

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120   # or 121 for RTX Spark / Jetson Thor
cmake --build build -j
ctest --test-dir build
```

The top-level `CMakeLists.txt` is a superbuild (`kernels → moe → runtime`); each subsystem also builds standalone (the sibling `../kernels` references resolve within the monorepo). A direct `nvcc` build from the repo root works too — see [`bench/scripts`](bench/scripts).

## Targets

**Blackwell only, by design:** `sm_120` (RTX 5090, RTX PRO 6000) and `sm_121` (RTX Spark / GB10, Jetson Thor). **Not** `sm_100` (datacenter B200/GB200 — binary-incompatible).

## Contributing

Source-required and reproducible — the validator builds your PR from source (the
prebuilt binaries are a run convenience, not a submission format). Before a PR, run
`bench/scripts/bench.sh` (speed) and `bench/scripts/accuracy.sh` (accuracy must hold:
~100% top-1 + KL ≈ 0 vs the prior build). Contributions are rewarded on SN74 by the
**verified marginal speedup** added over the live frontier, correctness-gated against a
frozen reference, validated on both basket models (Qwen + Gemma). See
[CONTRIBUTING.md](CONTRIBUTING.md) and the [org reward model](https://github.com/gittensor-ai-lab).

## License

[MIT](LICENSE) · [Changelog](CHANGELOG.md)
