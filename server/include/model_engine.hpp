#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace sparkinfer_server {

// Hand-mirrored server-layer copy of sparkinfer::Qwen35Model::TokenLogprob -- model_engine.hpp
// deliberately never re-exposes runtime types in its public API, matching CompletionResult's
// existing field-by-field-copy convention below.
struct TokenLogprob {
    int token_id = -1;
    float logprob = 0.f;
    std::vector<std::pair<int, float>> top_alternatives;
};

// Outcome of one complete()/complete_streaming() call. Returned by value so concurrent
// HTTP worker threads cannot observe or clear each other's failure state.
struct CompletionResult {
    std::vector<int> tokens;
    std::string error;  // empty on success
    bool overloaded = false;  // true => caller should return 429, not a generic 4xx
    bool alloc_failed = false;  // true => real device OOM, not capacity -- caller should return
                                 // 503 (permanent until restart), never 429 (implies transient/retry)
    bool timed_out = false;   // true => a per-request deadline was exceeded
    bool cancelled = false;   // true => on_token returned false; not an error
    bool reached_token_limit = false;  // true => max_new_tokens, not EOS, ended generation
    double ttft_ms = -1.0;
    double generation_ms = -1.0;
    double decode_tps = -1.0;
};

// Thread-safe wrapper around sparkinfer::Qwen35Model + GGUF load.
class ModelEngine {
public:
    ModelEngine();
    ~ModelEngine();

    ModelEngine(const ModelEngine&) = delete;
    ModelEngine& operator=(const ModelEngine&) = delete;

    bool load(const std::string& gguf_path, int max_seq = 0);
    bool loaded() const;

    std::string model_path() const;
    int eos_id() const;
    int vocab() const;
    int max_seq() const;
    bool is_museglimmer() const;

    // Optional shared prompt prefix (e.g. system message tokens). When set, each request whose
    // prompt starts with these ids calls cache_prefix() (batched prefill) before generate().
    void set_prefix_tokens(const std::vector<int>& tokens);
    int prefix_token_len() const;

    // Greedy decode. Tokens are in .tokens; on failure .tokens is empty and .error is set.
    CompletionResult complete(const std::vector<int>& prompt_ids, int max_new_tokens);

    // Same, but invokes on_token after each generated token (for SSE streaming). on_token
    // returns false to cancel generation early (client disconnected); the result then comes
    // back with .cancelled = true, not an error.
    //
    // temperature <= 0 (default) is plain greedy argmax. > 0 samples via Gumbel-max -- see
    // ContinuousBatchEngine::Request's doc comment for the reproducibility contract and the
    // known "first token is always greedy" v1 scope limitation. top_k/top_p truncate the
    // candidate set before the Gumbel draw; neither requires temperature > 0 (see
    // ContinuousBatchEngine::Request's doc comment for the inertness proof).
    //
    // presence_penalty/frequency_penalty ([-2.0, 2.0], 0 disables both) -- see
    // ContinuousBatchEngine::Request's doc comment for the semantics and the "no inertness proof,
    // needs its own DFlash check" distinction from top_k/top_p.
    //
    // logit_bias: (token_id, bias in [-100,100]) pairs, empty disables it. Same tier as
    // presence_penalty/frequency_penalty (sampling control, not reporting) -- see
    // ContinuousBatchEngine::Request's doc comment. Unlike every other sampling control here, its
    // value is static for the whole request rather than refreshed per decode step.
    //
    // on_token_logprob (optional) fires once per token, immediately before on_token for that same
    // token, only when logprobs is true AND this callback is non-null -- pass nullptr (not a
    // no-op lambda) when logprobs aren't wanted; see ContinuousBatchEngine::complete_streaming's
    // doc comment for why an always-non-null callback would defeat the "costs nothing extra when
    // unused" property. Same "first token has no logprobs" v1 gap as above.
    CompletionResult complete_streaming(const std::vector<int>& prompt_ids, int max_new_tokens,
                                        const std::function<bool(int)>& on_token,
                                        float temperature = 0.f, uint64_t seed = 0,
                                        int top_k = 0, float top_p = 1.0f,
                                        float presence_penalty = 0.f, float frequency_penalty = 0.f,
                                        const std::vector<std::pair<int, float>>& logit_bias = {},
                                        bool logprobs = false, int top_logprobs = 0,
                                        const std::function<void(const TokenLogprob&)>&
                                            on_token_logprob = nullptr);

    // Live occupancy, for capacity-reporting endpoints. 0/0 if the model isn't loaded yet.
    int active_requests() const;
    int free_kv_blocks() const;
    int max_queue_depth() const;

    // LMCache bridge counters (docs/lmcache_bridge_protocol.md), for GET /metrics'
    // sparkinfer_lmcache_lookup_{hits,misses}_total. enabled=false (both counts 0) when
    // SPARKINFER_LMCACHE_ENABLE wasn't set or the sidecar never came up -- distinct from
    // "enabled but 0 lookups happened yet," which also reports zero counts but enabled=true.
    struct LMCacheStats {
        bool enabled = false;
        uint64_t lookup_hits = 0;
        uint64_t lookup_misses = 0;
    };
    LMCacheStats lmcache_stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    mutable std::mutex mu_;
};

}  // namespace sparkinfer_server
