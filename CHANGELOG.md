# Changelog

Notable changes to sparkinfer. Format loosely follows [Keep a Changelog](https://keepachangelog.com);
versions track the GitHub [releases](https://github.com/gittensor-ai-lab/sparkinfer/releases).

## [0.1.0] — 2026-06-22

First release of the consolidated **sparkinfer** monorepo (kernels + MoE engine + runtime + benchmarks).

### Added
- **Native GGUF loading** — mmap parser + on-GPU **byte-exact Q4_K / Q6_K dequant**;
  expert weights kept quantized resident (Q4_K_M-sized footprint, not bf16).
- **Qwen3-MoE runtime** — embed → RMSNorm → QKV → per-head QK-norm → RoPE → paged GQA
  flash-decode → routed top-k MoE (+ optional shared expert) → LM head → greedy decode.
- **Kernels** — flash-decode (hd128/256/512), **flash-decoding (KV-split)** attention,
  **fused quantized MoE expert FFN** (dequant only the routed experts on-read), decode
  GEMV (coalesced `[out,in]`), GEMM, fused RMSNorm, RoPE.
- **CUDA-graph decode** — the per-token compute is captured once and replayed.
- **Turnkey harness** — `bench/scripts/bench.sh` (decode tok/s, `--compare` vs llama.cpp)
  and `accuracy.sh` (token-match / KL / perplexity); auto-detect arch, fetch model.
- **Accuracy gate** — `qwen3_gguf_score` teacher-forced scorer (per-position argmax +
  top-k logprobs + perplexity), for regression-checking optimizations.
- **Prebuilt binaries** attached to this release (sm_120 / CUDA 13 / glibc 2.39), with
  automatic **source-build fallback** when incompatible.

### Verified
- **RTX 5090** (sm_120, CUDA 13): `ctest` 5/5, compute-sanitizer 0 errors,
  **163.88 tok/s** decode, **100% top-1 token agreement** with llama.cpp (KL ≈ 0.14 nats),
  21.4 GB resident.
- **RTX PRO 6000** (sm_120, CUDA 12.8): **0.60 → 134 tok/s** decode across 6 source-verifiable
  optimization passes.

### Fixed (during RTX 5090 / CUDA 13 bring-up)
- CUDA 13 removed `cudaDeviceProp::memoryClockRate` / `memoryBusWidth` → query via
  `cudaDeviceGetAttribute` (portable across CUDA 12.x / 13).
- Flash-decode scratch (`fa_*`) was NULL on the non-GGUF path (allocated only in
  `load_gguf`) → moved to the constructor (caught by compute-sanitizer).
- Top-level superbuild was missing `enable_testing()` → `ctest` found no tests.

[0.1.0]: https://github.com/gittensor-ai-lab/sparkinfer/releases/tag/v0.1.0
