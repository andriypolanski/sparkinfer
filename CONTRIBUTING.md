# Contributing to sparkinfer

sparkinfer is the engineering arm of **SN74 on Gittensor**. Contributions are rewarded
for **real, verified inference-speed engineering** — not benchmark gaming. This guide is
how to make a contribution that counts.

## Principles

- **Source-required & reproducible.** The validator builds your PR from source. No
  opaque prebuilt images — the shipped prebuilt binaries are a *run* convenience, not a
  submission format.
- **Correctness first.** A faster kernel that changes the model's output is worth zero.
  Every change is gated against a frozen reference (see *Accuracy gate* below).
- **General, not overfit.** Optimizations must hold across the basket — **Qwen3-MoE and
  Gemma 4** — and across shapes. A win on one model/shape but not the other is overfitting.
- **Blackwell only, by design.** Targets `sm_120` (RTX 5090, RTX PRO 6000) and `sm_121`
  (RTX Spark / Jetson Thor). CUDA 12.8+ (13 works). Not `sm_100`.

## Before you open a PR

```bash
# 1. build + tests (must be 5/5)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build -j && ctest --test-dir build

# 2. speed — does it actually go faster?
bench/scripts/bench.sh --download            # and --compare for the llama.cpp gap

# 3. accuracy — did it stay correct?  (this is the gate that blocks regressions)
bench/scripts/accuracy.sh --download
```

**Accuracy gate.** Run `bench/scripts/accuracy.sh` (or `qwen3_gguf_score`) on the build
*before* and *after* your change. A correct optimization must keep:
- **≥ 99–100% top-1 token agreement** vs the previous build, and
- **mean KL ≈ 0** (the next-token distributions barely move).

(`accuracy.sh` also compares against llama.cpp; the implementation bar there is ≥ 90%
top-1, which we currently meet at 100%.) If `compute-sanitizer` is available, your kernels
must be clean (0 errors).

## How rewards work (SN74)

You're paid for the **verified marginal speedup** your PR adds over the current best
("frontier"), not your rank — so "copy the leader + ε" pays ≈ ε. Performance PRs
(`kernels/`, `runtime/`, `moe/`) are scored by measured speedup and labeled
**XL / L / M / S / XS** (by the eval loop, from the measured delta — not by hand);
`bench/` and infra PRs are scored by code quality. Label weights are maturity-adaptive
(they rebalance toward smaller gains as the runtime nears the hardware ceiling). See the
[org reward model](https://github.com/gittensor-ai-lab) for the full design.

## Style & scope

- Match the surrounding code (portable CUDA is the production path; CuTe/tensor-core is
  the opt-in ceiling). Keep kernels readable and commented where non-obvious.
- Reference the bench + accuracy numbers in your PR description (before → after).
- Keep changes focused; one optimization per PR makes the measured delta attributable.

By contributing you agree your work is licensed under the repository's [MIT License](LICENSE).
