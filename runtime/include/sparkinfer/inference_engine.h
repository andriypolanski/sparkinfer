#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/scheduler.h"

namespace sparkinfer {

// Continuous-batch serving engine: queues requests, assigns per-request seq_ids,
// right-sizes KV allocation, and interleaves decode steps via the Scheduler.
//
// DFlash multi-token accept in step_job() is intentionally deferred until the
// single-stream dflash_generate path proves SPEC_AGREE=100% and a tok/s win
// (see bench/scripts/dflash_accuracy.sh). Do not wire speculative multi-accept here yet.
class ContinuousBatchEngine {
public:
    struct Request {
        std::vector<int> prompt;
        int max_new_tokens = 0;
        int priority = 0;
        int prefill_start = 0;          // skip tokens already in a shared prefix cache
        bool use_prefix_session = false; // bind to session 0 (cache_prefix KV)
        // <= 0 (default) is plain greedy argmax, byte-identical to pre-sampling behavior. > 0
        // samples via Gumbel-max (Qwen35Model::forward_token). Note: the prefill-phase seed
        // token (the very first token of the response) is always greedy regardless of this
        // value -- see step_job()'s PREFILL branch comment; every token from the second onward
        // respects it.
        float temperature = 0.f;
        uint64_t seed = 0;              // only meaningful when temperature > 0
        // top_k <= 0 or >= vocab disables top_k (no truncation). top_p <= 0 or >= 1.0 disables
        // top_p. Both truncate the candidate set before the Gumbel draw above; neither requires
        // temperature > 0 to be accepted -- see Qwen35Model::forward_token's doc comment for the
        // inertness proof. Same prefill-phase-seed-token caveat as temperature applies unchanged.
        int top_k = 0;
        float top_p = 1.0f;
        // [-2.0, 2.0]; 0 (default) disables both. OpenAI semantics: subtracted from every vocab
        // logit each decode step, weighted by this request's own running per-token generation
        // count (presence: binary "appeared at all"; frequency: linear in count) -- see
        // Qwen35Model::forward_token's doc comment. UNLIKE top_k/top_p above, this has no
        // inertness proof at temperature<=0 -- it can change the greedy winner -- so it needs its
        // own DFlash-incompatibility check (should_reject_dflash_penalty), independent of the
        // existing temperature>0 check.
        float presence_penalty = 0.f;
        float frequency_penalty = 0.f;
        // top_logprobs is only meaningful when logprobs is true. Delivery is via the separate
        // on_token_logprob callback (complete_streaming's new trailing param), not this struct --
        // pure data here, same as every other sampling control.
        //
        // KNOWN v1 GAP: the prefill-phase seed token (this response's very first token) never
        // gets a last_token_logprobs() call -- same root cause as temperature's "first token is
        // always greedy" gap above (ingest_prompt_range() is a funnel shared by other callers,
        // not worth threading through for one token). Callers should expect one fewer logprobs
        // entry than emitted tokens.
        bool logprobs = false;
        int top_logprobs = 0;   // 0-20; only meaningful when logprobs is true
        // OpenAI's logit_bias: (token_id, bias in [-100,100]) pairs, added to every vocab logit
        // each decode step -- see Qwen35Model::forward_token's doc comment. Empty (default)
        // disables it. UNLIKE every other sampling control above, this is NOT refreshed every
        // forward_token() call -- it's static for the whole request, set once at submit time
        // (Qwen35Model::set_logit_bias, called from submit_locked() alongside
        // reset_penalty_counts). Same "no inertness proof at temperature<=0, needs its own DFlash
        // check" story as presence/frequency penalty (should_reject_dflash_logit_bias). First
        // non-scalar sampling-control field here -- deep-copied into Job by submit_locked's
        // `job.req = req;`, same safe-across-the-async-worker-boundary mechanism `prompt` already
        // relies on.
        std::vector<std::pair<int, float>> logit_bias;
    };

    struct Result {
        std::vector<int> tokens;
        std::string error;
        // true => caller should surface 429 (no capacity right now), not a generic 4xx —
        // the request itself was fine, there was just nowhere to run it.
        bool overloaded = false;
        // true => a real device allocation failed (distinct from the KV pool being full, which
        // sets `overloaded` instead) -- caller should surface 503, not 429: this condition is
        // permanent until the process restarts, not a transient "try again shortly" (#779, where
        // a leak eventually exhausted VRAM and every subsequent request got a misleading 429
        // "overloaded" even though the queue was empty and the KV pool had free blocks).
        bool alloc_failed = false;
        // true => a per-request deadline (SPARKINFER_REQUEST_TIMEOUT_S) was exceeded.
        bool timed_out = false;
        // true => on_token returned false (client went away mid-stream); not an error.
        bool cancelled = false;
        // true => generation exhausted max_new_tokens instead of reaching an EOS token.
        // HTTP callers surface this as finish_reason="length"; in particular, a truncated
        // tool-call payload must never be reported as a successful "stop".
        bool reached_token_limit = false;
        // GPU-side timings (exclude SSE/on_token backpressure).
        double ttft_ms = -1.0;
        double generation_ms = -1.0;
        double decode_tps = -1.0;
    };

    ContinuousBatchEngine(Qwen35Model* model, KVCacheManager* kv,
                          int max_tokens_per_batch = 64,
                          SchedulePolicy policy = SchedulePolicy::CONTINUOUS_BATCHING);
    ~ContinuousBatchEngine();

    ContinuousBatchEngine(const ContinuousBatchEngine&) = delete;
    ContinuousBatchEngine& operator=(const ContinuousBatchEngine&) = delete;

    // Blocking completion (used by the HTTP server).
    Result complete(const Request& req);

    // Streaming completion: on_token is invoked on the worker thread as tokens are produced.
    // on_token returns false to cancel generation early (e.g. the client disconnected) --
    // the request then finishes with Result::cancelled = true, not an error.
    //
    // on_token_logprob (optional) is invoked once per token, immediately BEFORE on_token fires
    // for that same token, only when req.logprobs is true AND this callback is non-null -- pass
    // nullptr (not a no-op lambda) when logprobs aren't wanted, since step_job()'s cost gate is
    // `req.logprobs && on_token_logprob`; an always-non-null callback would defeat the
    // "logprobs=false costs nothing extra" property.
    Result complete_streaming(const Request& req, const std::function<bool(int)>& on_token,
                              const std::function<void(const Qwen35Model::TokenLogprob&)>&
                                  on_token_logprob = nullptr);

    int num_active() const;
    int num_free_kv_blocks() const;
    // Admission-time queue depth cap (SPARKINFER_MAX_QUEUE_DEPTH, 0 = unlimited). Requests
    // beyond this are rejected as overloaded before any KV allocation is attempted.
    int max_queue_depth() const;

private:
    struct Job;
    enum class EnqueueError { NONE, BAD_REQUEST, OVERLOADED, ALLOC_FAILED };
    uint64_t submit_locked(Job job, const std::function<bool(int)>& on_token,
                           const std::function<void(const Qwen35Model::TokenLogprob&)>& on_token_logprob,
                           EnqueueError* err_out);
    Result wait_locked(uint64_t request_id);
    void worker_loop();
    bool step_job(Job& job, bool chunked = false);

    Qwen35Model* model_;
    KVCacheManager* kv_;
    Scheduler scheduler_;
    SchedulePolicy policy_ = SchedulePolicy::CONTINUOUS_BATCHING;
    std::thread worker_;
    std::atomic<bool> running_{false};
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::unordered_map<uint64_t, std::unique_ptr<Job>> jobs_;
    std::atomic<uint64_t> next_req_id_{1};
};

}  // namespace sparkinfer
