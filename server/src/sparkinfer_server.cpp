#include "chat_tokenizer.hpp"
#include "model_engine.hpp"

#include <nlohmann/json.hpp>

// Do not define CPPHTTPLIB_OPENSSL_SUPPORT — even `= 0` enables OpenSSL in httplib.
#include "../third_party/httplib.h"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

// Per-request output cap. Independent of context length (checked separately against the
// live engine max_seq below) -- this just bounds how long a single request may run.
int max_output_tokens() {
    static int v = []{
        const char* e = getenv("SPARKINFER_MAX_OUTPUT_TOKENS");
        int n = e ? atoi(e) : 4096;
        return n > 0 ? n : 4096;
    }();
    return v;
}

std::string g_api_key;
std::string g_model_name = "qwen3.6-35b-a3b";
sparkinfer_server::ChatTokenizer g_tokenizer;
const auto g_start_time = std::chrono::steady_clock::now();

// Request/error metrics (GET /metrics). Counters only -- no per-request content is retained.
std::atomic<uint64_t> g_requests_total{0};
std::atomic<uint64_t> g_requests_streaming{0};
std::atomic<uint64_t> g_requests_ok{0};
std::atomic<uint64_t> g_requests_client_error{0};   // 4xx
std::atomic<uint64_t> g_requests_overloaded{0};     // 429
std::atomic<uint64_t> g_requests_alloc_failed{0};   // 503 -- real device OOM, not capacity (#779)
std::atomic<uint64_t> g_requests_timeout{0};
std::atomic<uint64_t> g_requests_cancelled{0};
std::atomic<uint64_t> g_requests_server_error{0};   // 5xx
// 502 -- the model produced tool-call output that failed schema/markup validation for a reason
// other than truncation. Distinct from server_error so monitoring doesn't conflate model-output
// quality variance with a real infrastructure fault -- the exact conflation #779 already
// documented once for overloaded (429) vs alloc_failed (503).
std::atomic<uint64_t> g_requests_invalid_tool_output{0};
// 502 -- same rationale as g_requests_invalid_tool_output above, for response_format: the model
// produced output that failed JSON/schema validation on both the original and the corrective
// retry attempt. Not a server_error -- this is a model-output-quality signal, not an infra fault.
std::atomic<uint64_t> g_requests_invalid_json_output{0};
std::atomic<uint64_t> g_prompt_tokens_total{0};
std::atomic<uint64_t> g_completion_tokens_total{0};

std::atomic<bool> g_shutdown_requested{false};
void on_shutdown_signal(int) { g_shutdown_requested = true; }

std::string repo_root() {
    const char* env = getenv("SPARKINFER_ROOT");
    if (env && *env) return env;
    return ".";
}

std::string json_escape(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (c < 0x20) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                else o << c;
        }
    }
    return o.str();
}

std::string random_id(const char* prefix = "chatcmpl-") {
    thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dist;
    std::ostringstream ss;
    ss << prefix << std::hex << dist(rng);
    return ss.str();
}

bool auth_ok(const httplib::Request& req) {
    if (g_api_key.empty()) return true;
    auto it = req.headers.find("Authorization");
    if (it == req.headers.end()) return false;
    const std::string prefix = "Bearer ";
    return it->second.size() > prefix.size() &&
           it->second.compare(0, prefix.size(), prefix) == 0 &&
           it->second.substr(prefix.size()) == g_api_key;
}

bool encode_messages(const std::string& body, std::vector<int>& ids, bool enable_thinking,
                     std::string& err, sparkinfer_server::ChatRequest* request = nullptr) {
    return g_tokenizer.encode_chat_request(body, ids, enable_thinking, err, request);
}

// RequestControls / parse_request_controls / should_reject_dflash_temperature now live in
// chat_tools.hpp/.cpp (moved so `stop`/`temperature`/`seed` validation has a unit-test seam via
// chat_tools_test, same as ChatRequest/parse_chat_request_json already had).

// One-shot scan over a complete decoded string: is there a stop match anywhere, and where.
// Unlike StopSequenceFilter, this needs no holdback machinery -- it's only ever called once
// generation has already stopped and the full text is available in hand.
bool find_stop_match(const std::string& text, const std::vector<std::string>& stops, size_t& pos) {
    bool found = false;
    for (const auto& s : stops) {
        const size_t p = text.find(s);
        if (p != std::string::npos && (!found || p < pos)) {
            pos = p;
            found = true;
        }
    }
    return found;
}

nlohmann::json stream_chunk_base(const std::string& cid, long long created) {
    return {{"id", cid}, {"object", "chat.completion.chunk"}, {"created", created},
            {"model", g_model_name}};
}

bool write_sse_json(httplib::DataSink& sink, const nlohmann::json& value) {
    const std::string event = "data: " + value.dump() + "\n\n";
    return sink.write(event.c_str(), event.size());
}

bool write_stream_role(httplib::DataSink& sink, const std::string& cid, long long created) {
    auto chunk = stream_chunk_base(cid, created);
    chunk["choices"] = nlohmann::json::array({{{"index", 0},
                                                {"delta", {{"role", "assistant"},
                                                           {"content", nullptr}}},
                                                {"finish_reason", nullptr},
                                                {"logprobs", nullptr}}});
    return write_sse_json(sink, chunk);
}

bool write_stream_delta(httplib::DataSink& sink, const std::string& cid, long long created, const std::string& field,
                        const std::string& piece, const nlohmann::json& logprobs = nullptr) {
    if (piece.empty()) return true;
    auto chunk = stream_chunk_base(cid, created);
    chunk["choices"] = nlohmann::json::array({{{"index", 0},
                                                {"delta", {{field, piece}}},
                                                {"finish_reason", nullptr},
                                                {"logprobs", logprobs}}});
    return write_sse_json(sink, chunk);
}

bool write_stream_tool_call(httplib::DataSink& sink, const std::string& cid, long long created,
                            size_t index, const sparkinfer_server::ToolCall& call) {
    auto chunk = stream_chunk_base(cid, created);
    nlohmann::json delta_call = {{"index", index}, {"id", call.id}, {"type", "function"},
                                {"function", {{"name", call.name},
                                              {"arguments", call.arguments}}}};
    chunk["choices"] = nlohmann::json::array({{{"index", 0},
                                                {"delta", {{"tool_calls",
                                                            nlohmann::json::array({delta_call})}}},
                                                {"finish_reason", nullptr},
                                                {"logprobs", nullptr}}});
    return write_sse_json(sink, chunk);
}

bool write_stream_finish(httplib::DataSink& sink, const std::string& cid, long long created,
                         const std::string& reason) {
    auto chunk = stream_chunk_base(cid, created);
    chunk["choices"] = nlohmann::json::array({{{"index", 0},
                                                {"delta", nlohmann::json::object()},
                                                {"finish_reason", reason},
                                                {"logprobs", nullptr}}});
    return write_sse_json(sink, chunk);
}

bool write_stream_usage(httplib::DataSink& sink, const std::string& cid, long long created,
                        int prompt_tokens, int completion_tokens, double ttft_ms,
                        double generation_ms, double decode_tps) {
    auto chunk = stream_chunk_base(cid, created);
    chunk["choices"] = nlohmann::json::array();
    nlohmann::json usage = {{"prompt_tokens", prompt_tokens},
                            {"completion_tokens", completion_tokens},
                            {"total_tokens", prompt_tokens + completion_tokens}};
    if (ttft_ms >= 0.0) usage["ttft_ms"] = ttft_ms;
    if (generation_ms >= 0.0) usage["generation_ms"] = generation_ms;
    if (decode_tps >= 0.0) usage["decode_tps"] = decode_tps;
    chunk["usage"] = std::move(usage);
    return write_sse_json(sink, chunk);
}

nlohmann::json token_logprob_entry_json(int token_id, float logprob) {
    const auto piece = g_tokenizer.id_to_raw_piece(token_id);
    nlohmann::json bytes = nlohmann::json::array();
    for (uint8_t b : piece.bytes) bytes.push_back((int)b);
    return {{"token", piece.display}, {"logprob", logprob}, {"bytes", bytes}};
}

// Builds the OpenAI-shaped logprobs.content array from a flat, generation-ordered list of
// per-token entries. Each entry's own top_alternatives list is truncated to top_logprobs_n
// (the request's requested top_logprobs value) regardless of how many were captured on-device.
nlohmann::json build_logprobs_content_json(const std::vector<sparkinfer_server::TokenLogprob>& entries,
                                           int top_logprobs_n) {
    nlohmann::json content = nlohmann::json::array();
    for (const auto& e : entries) {
        nlohmann::json item = token_logprob_entry_json(e.token_id, e.logprob);
        nlohmann::json alts = nlohmann::json::array();
        const int n = std::min((int)e.top_alternatives.size(), top_logprobs_n);
        for (int i = 0; i < n; i++)
            alts.push_back(token_logprob_entry_json(e.top_alternatives[i].first, e.top_alternatives[i].second));
        item["top_logprobs"] = alts;
        content.push_back(item);
    }
    return content;
}

bool write_stream_done(httplib::DataSink& sink) {
    static const std::string done = "data: [DONE]\n\n";
    return sink.write(done.c_str(), done.size());
}

bool decode_ids(const std::vector<int>& ids, std::string& text, std::string& err) {
    text = g_tokenizer.decode(ids);
    if (text.empty() && !ids.empty()) {
        err = "detokenize returned empty text";
        return false;
    }
    return true;
}

// Builds the attempt-2 prompt for a response_format validation failure: original conversation +
// the model's own (invalid) attempt-1 output as an assistant turn, followed by a corrective user
// turn describing what was wrong. This runtime's decode is fully deterministic greedy argmax
// with no RNG anywhere -- resubmitting attempt 1's identical prompt_ids would just reproduce the
// exact same invalid output. The retry only has a chance of succeeding with a materially
// different prompt.
sparkinfer_server::ChatRequest build_retry_request(const sparkinfer_server::ChatRequest& original,
                                                    const std::string& failed_content,
                                                    const std::string& validation_error) {
    sparkinfer_server::ChatRequest retry = original;
    sparkinfer_server::ChatMessage assistant_turn;
    assistant_turn.role = "assistant";
    assistant_turn.content = failed_content.empty() ? "(no output)" : failed_content;
    retry.messages.push_back(std::move(assistant_turn));
    sparkinfer_server::ChatMessage correction;
    correction.role = "user";
    correction.content = "Your previous reply did not satisfy the required response_format: " +
                         validation_error +
                         "\nRespond again with ONLY the corrected JSON, matching the required "
                         "format exactly.";
    retry.messages.push_back(std::move(correction));
    return retry;
}

std::vector<int> load_prefix_token_ids() {
    std::vector<int> out;
    if (const char* csv = getenv("SPARKINFER_SERVER_PREFIX_TOKEN_IDS")) {
        const char* p = csv;
        while (*p) {
            char* end = nullptr;
            long v = strtol(p, &end, 10);
            if (end == p) break;
            out.push_back((int)v);
            p = end;
            while (*p == ',' || *p == ' ') p++;
        }
        return out;
    }
    const char* path = getenv("SPARKINFER_SERVER_PREFIX_TOKEN_FILE");
    if (!path || !*path) return out;
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "[sparkinfer-server] WARN: cannot open prefix token file %s\n", path);
        return out;
    }
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (size_t i = 0; i < s.size();) {
        i = s.find_first_of("0123456789", i);
        if (i == std::string::npos) break;
        out.push_back(atoi(s.c_str() + i));
        i = s.find_first_not_of("0123456789", i);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string model_path;
    std::string tokenizer_json;
    int ctx = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char* flag) { return a == flag && i + 1 < argc; };
        if (need("-m") || need("--model")) model_path = argv[++i];
        else if (need("--host")) host = argv[++i];
        else if (need("--port")) port = atoi(argv[++i]);
        else if (need("--ctx")) ctx = atoi(argv[++i]);
        else if (need("--api-key")) g_api_key = argv[++i];
        else if (need("--tokenizer")) tokenizer_json = argv[++i];
        else if (need("--model-name")) g_model_name = argv[++i];
        else if (a == "-h" || a == "--help") {
            fprintf(stderr,
                    "usage: %s -m model.gguf [--host 127.0.0.1] [--port 8080] [--ctx N] "
                    "[--tokenizer path/to/tokenizer.json] [--model-name ID] [--api-key KEY]\n",
                    argv[0]);
            return 0;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "error: -m model.gguf is required\n");
        return 2;
    }

    const std::string root = repo_root();
    std::string tok_path = tokenizer_json.empty() ? root + "/models/tokenizer.json" : tokenizer_json;
    std::string tok_err;
    if (!g_tokenizer.load(tok_path, tok_err)) {
        fprintf(stderr, "[sparkinfer-server] %s\n", tok_err.c_str());
        return 1;
    }

    sparkinfer_server::ModelEngine engine;
    if (!engine.load(model_path, ctx > 0 ? ctx : 0)) return 1;
    g_tokenizer.set_museglimmer(engine.is_museglimmer());

    const std::vector<int> prefix_ids = load_prefix_token_ids();
    if (!prefix_ids.empty()) {
        engine.set_prefix_tokens(prefix_ids);
        fprintf(stderr, "[sparkinfer-server] prefix cache: %zu tokens (batched prefill per request)\n",
                prefix_ids.size());
    }

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/v1/models", [&engine](const httplib::Request&, httplib::Response& res) {
        std::ostringstream body;
        body << "{\"object\":\"list\",\"data\":[{\"id\":\"" << g_model_name
             << "\",\"object\":\"model\",\"owned_by\":\"sparkinfer\",\"context_length\":"
             << engine.max_seq() << "}]}";
        res.set_content(body.str(), "application/json");
    });

    svr.Get("/v1/info", [&engine](const httplib::Request& req, httplib::Response& res) {
        if (!auth_ok(req)) {
            res.status = 401;
            res.set_content("{\"error\":{\"message\":\"unauthorized\"}}", "application/json");
            return;
        }
        std::ostringstream body;
        body << "{\"model\":\"" << g_model_name << "\",\"max_context\":" << engine.max_seq()
             << ",\"max_output_tokens\":" << max_output_tokens() << "}";
        res.set_content(body.str(), "application/json");
    });

    // Live occupancy -- lets an orchestrator (or a human) see whether this worker has room
    // before routing a request to it, and is what a fleet-level capacity/load-balancing layer
    // would poll. Single-process only: this reports this server's own queue, not fleet-wide
    // capacity across other nodes.
    svr.Get("/v1/capacity", [&engine](const httplib::Request&, httplib::Response& res) {
        std::ostringstream body;
        const int cap = engine.max_queue_depth();
        body << "{\"active_requests\":" << engine.active_requests()
             << ",\"free_kv_blocks\":" << engine.free_kv_blocks()
             << ",\"max_queue_depth\":" << cap
             << ",\"accepting_requests\":" << (g_shutdown_requested.load() ? "false" : "true") << "}";
        res.set_content(body.str(), "application/json");
    });

    svr.Get("/metrics", [&engine](const httplib::Request&, httplib::Response& res) {
        const double uptime_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - g_start_time).count();
        std::ostringstream body;
        body << "# HELP sparkinfer_uptime_seconds Process uptime\n"
                "# TYPE sparkinfer_uptime_seconds gauge\n"
             << "sparkinfer_uptime_seconds " << uptime_s << "\n"
                "# HELP sparkinfer_requests_total Chat completion requests received\n"
                "# TYPE sparkinfer_requests_total counter\n"
             << "sparkinfer_requests_total " << g_requests_total.load() << "\n"
             << "sparkinfer_requests_streaming_total " << g_requests_streaming.load() << "\n"
                "# HELP sparkinfer_requests_by_outcome_total Requests by terminal outcome\n"
                "# TYPE sparkinfer_requests_by_outcome_total counter\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"ok\"} " << g_requests_ok.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"client_error\"} "
             << g_requests_client_error.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"overloaded\"} "
             << g_requests_overloaded.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"alloc_failed\"} "
             << g_requests_alloc_failed.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"timeout\"} "
             << g_requests_timeout.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"cancelled\"} "
             << g_requests_cancelled.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"server_error\"} "
             << g_requests_server_error.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"invalid_tool_output\"} "
             << g_requests_invalid_tool_output.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"invalid_json_output\"} "
             << g_requests_invalid_json_output.load() << "\n"
                "# HELP sparkinfer_tokens_total Tokens processed\n"
                "# TYPE sparkinfer_tokens_total counter\n"
             << "sparkinfer_tokens_total{kind=\"prompt\"} " << g_prompt_tokens_total.load() << "\n"
             << "sparkinfer_tokens_total{kind=\"completion\"} " << g_completion_tokens_total.load() << "\n"
                "# HELP sparkinfer_active_requests In-flight requests\n"
                "# TYPE sparkinfer_active_requests gauge\n"
             << "sparkinfer_active_requests " << engine.active_requests() << "\n"
                "# HELP sparkinfer_free_kv_blocks Free KV cache blocks\n"
                "# TYPE sparkinfer_free_kv_blocks gauge\n"
             << "sparkinfer_free_kv_blocks " << engine.free_kv_blocks() << "\n";
        // Only emitted when the LMCache bridge is actually enabled (docs/lmcache_bridge_protocol.md)
        // -- omitting the metric entirely when disabled, rather than always emitting zeros, makes
        // "is this feature even on" visible from /metrics itself, not just server startup logs.
        const auto lmc = engine.lmcache_stats();
        if (lmc.enabled) {
            body << "# HELP sparkinfer_lmcache_lookup_hits_total External KV cache tier lookups "
                    "that restored a matched prefix\n"
                    "# TYPE sparkinfer_lmcache_lookup_hits_total counter\n"
                 << "sparkinfer_lmcache_lookup_hits_total " << lmc.lookup_hits << "\n"
                    "# HELP sparkinfer_lmcache_lookup_misses_total External KV cache tier lookups "
                    "that found nothing usable (includes a genuine miss, a timed-out sidecar, and "
                    "the sidecar being unreachable -- see docs/lmcache_bridge_protocol.md's "
                    "degradation invariant, all three fall back to the same recompute path)\n"
                    "# TYPE sparkinfer_lmcache_lookup_misses_total counter\n"
                 << "sparkinfer_lmcache_lookup_misses_total " << lmc.lookup_misses << "\n";
        }
        res.set_content(body.str(), "text/plain; version=0.0.4");
    });

    svr.Post("/v1/tokenize", [&engine](const httplib::Request& req, httplib::Response& res) {
        if (!auth_ok(req)) {
            res.status = 401;
            res.set_content("{\"error\":{\"message\":\"unauthorized\"}}", "application/json");
            return;
        }
        const bool enable_thinking = sparkinfer_server::parse_enable_thinking(req.body, false);
        std::vector<int> ids;
        std::string err;
        if (!encode_messages(req.body, ids, enable_thinking, err)) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"" + json_escape(err) + "\"}}", "application/json");
            return;
        }
        std::ostringstream body;
        body << "{\"tokens\":" << ids.size() << ",\"max_context\":" << engine.max_seq()
             << ",\"max_output_tokens\":" << max_output_tokens() << ",\"model\":\"" << g_model_name << "\"}";
        res.set_content(body.str(), "application/json");
    });

    svr.Post("/v1/chat/completions",
             [&engine](const httplib::Request& req, httplib::Response& res) {
                 if (!auth_ok(req)) {
                     res.status = 401;
                     res.set_content("{\"error\":{\"message\":\"unauthorized\"}}", "application/json");
                     return;
                 }
                 if (g_shutdown_requested.load()) {
                     res.status = 503;
                     res.set_content("{\"error\":{\"message\":\"server is shutting down\"}}",
                                     "application/json");
                     return;
                 }
                 if (!engine.loaded()) {
                     res.status = 503;
                     res.set_content("{\"error\":{\"message\":\"model not loaded\"}}", "application/json");
                     return;
                 }

                 g_requests_total++;
                 sparkinfer_server::RequestControls controls;
                 std::string err;
                 if (!sparkinfer_server::parse_request_controls(req.body, controls, err, engine.vocab())) {
                     g_requests_client_error++;
                     res.status = 400;
                     res.set_content("{\"error\":{\"message\":\"" + json_escape(err) + "\"}}",
                                     "application/json");
                     return;
                 }
                 // DFlash's verify step requires exact greedy-argmax determinism against the
                 // draft model's own proposal -- real sampling can't coexist with it. DFlash is
                 // a process-wide, env-var-gated toggle (not per-request), so this is checked
                 // once per request against the process env, at validation time, before any
                 // generation is attempted.
                 //
                 // top_k/top_p deliberately have no analogous check here: they can only ever
                 // change output in combination with an actual Gumbel draw, which only happens
                 // at temperature > 0 -- already rejected below regardless of top_k/top_p. A
                 // top_k=5, temperature=0 request under DFlash is safe (masking is provably
                 // inert at temperature<=0 -- see ContinuousBatchEngine::Request's doc comment).
                 // logprobs needs no check either -- it's pure output reporting, never touches
                 // the sampling path at all.
                 const bool dflash_env_on =
                     [] { const char* e = getenv("SPARKINFER_DFLASH"); return e && e[0] == '1'; }();
                 if (sparkinfer_server::should_reject_dflash_temperature(dflash_env_on, controls.temperature)) {
                     g_requests_client_error++;
                     res.status = 400;
                     res.set_content("{\"error\":{\"message\":\"temperature sampling is not supported "
                                     "while SPARKINFER_DFLASH=1 is active on this server instance\"}}",
                                     "application/json");
                     return;
                 }
                 // Unlike top_k/top_p (provably inert at temperature<=0, see above), presence/
                 // frequency penalty can change the greedy-argmax winner even at temperature==0 --
                 // no inertness proof exists, so this needs its own DFlash check, independent of
                 // the temperature check above.
                 if (sparkinfer_server::should_reject_dflash_penalty(
                         dflash_env_on, controls.presence_penalty, controls.frequency_penalty)) {
                     g_requests_client_error++;
                     res.status = 400;
                     res.set_content("{\"error\":{\"message\":\"presence_penalty/frequency_penalty are not "
                                     "supported while SPARKINFER_DFLASH=1 is active on this server instance\"}}",
                                     "application/json");
                     return;
                 }
                 // Same "no inertness proof at temperature<=0" gap as presence/frequency penalty
                 // above -- an arbitrary per-vocab additive bias can change the greedy-argmax
                 // winner on its own, so this needs its own DFlash check too, independent of both
                 // checks above.
                 if (sparkinfer_server::should_reject_dflash_logit_bias(
                         dflash_env_on, !controls.logit_bias.empty())) {
                     g_requests_client_error++;
                     res.status = 400;
                     res.set_content("{\"error\":{\"message\":\"logit_bias is not supported while "
                                     "SPARKINFER_DFLASH=1 is active on this server instance\"}}",
                                     "application/json");
                     return;
                 }
                 if (!controls.seed_set) {
                     static thread_local std::random_device rd;
                     controls.seed = ((uint64_t)rd() << 32) | rd();
                 }
                 const bool stream = controls.stream;
                 if (stream) g_requests_streaming++;
                 const bool enable_thinking = sparkinfer_server::parse_enable_thinking(req.body, false);
                 int max_tokens = controls.max_tokens;
                 if (max_tokens <= 0) max_tokens = 256;
                 if (max_tokens > max_output_tokens()) max_tokens = max_output_tokens();

                 std::vector<int> prompt_ids;
                 sparkinfer_server::ChatRequest chat_request;
                 if (!encode_messages(req.body, prompt_ids, enable_thinking, err, &chat_request)) {
                     g_requests_client_error++;
                     res.status = 400;
                     res.set_content("{\"error\":{\"message\":\"" + json_escape(err) + "\"}}",
                                     "application/json");
                     return;
                 }
                 // Presence of a tool protocol requires strict, buffered parsing even when
                 // tool_choice=none. The latter disables execution, not output validation:
                 // native tool markup must never leak through as ordinary assistant content.
                 const bool tool_protocol = !chat_request.tools.empty();
                 // Mutually exclusive with tool_protocol by construction -- parse_chat_request_json
                 // rejects tools + response_format together at request time.
                 const bool json_mode_active =
                     chat_request.response_format.type != sparkinfer_server::ResponseFormatType::kText;
                 if ((int)prompt_ids.size() + max_tokens > engine.max_seq()) {
                     g_requests_client_error++;
                     res.status = 400;
                     res.set_content(
                         "{\"error\":{\"message\":\"context overflow: prompt=" +
                         std::to_string(prompt_ids.size()) + " max_tokens=" + std::to_string(max_tokens) +
                         " exceeds server ctx=" + std::to_string(engine.max_seq()) + "\"}}",
                         "application/json");
                     return;
                 }

                 const std::string cid = random_id();
                 const auto created = (long long)std::chrono::duration_cast<std::chrono::seconds>(
                                        std::chrono::system_clock::now().time_since_epoch())
                                        .count();

                 // Maps an engine outcome to the metrics bucket + HTTP status a non-2xx response
                 // should use. Overloaded -> 429 (retry elsewhere / later, not a bad request).
                 // Alloc failed -> 503 (real device OOM -- permanent until restart, never imply
                 // "retry shortly" like 429 does; #779, where this used to fall through to the
                 // same 429 as a full queue and sent operators chasing SPARKINFER_MAX_QUEUE_DEPTH
                 // for nothing while the actual queue sat empty).
                 // Timed out -> 504 (the request was valid, the server just didn't finish in time).
                 auto record_and_status = [](const sparkinfer_server::CompletionResult& o) -> int {
                     if (o.overloaded)    { g_requests_overloaded++;   return 429; }
                     if (o.alloc_failed)  { g_requests_alloc_failed++; return 503; }
                     if (o.timed_out)     { g_requests_timeout++;      return 504; }
                     g_requests_server_error++;
                     return 400;
                 };

                 if (stream) {
                     res.set_chunked_content_provider(
                         "text/event-stream",
                         [&engine, prompt_ids, max_tokens, cid, created, enable_thinking,
                          chat_request, tool_protocol, json_mode_active,
                          include_usage = controls.include_usage, stop = controls.stop,
                          temperature = controls.temperature, seed = controls.seed,
                          top_k = controls.top_k, top_p = controls.top_p,
                          presence_penalty = controls.presence_penalty,
                          frequency_penalty = controls.frequency_penalty,
                          logit_bias = controls.logit_bias,
                          logprobs = controls.logprobs, top_logprobs = controls.top_logprobs]
                         (size_t offset, httplib::DataSink& sink) {
                             if (offset > 0) {
                                 sink.done();
                                 return true;
                             }
                             if (!write_stream_role(sink, cid, created)) {
                                 g_requests_cancelled++;
                                 sink.done();
                                 return true;
                             }

                             if (json_mode_active) {
                                 // response_format needs the complete, validated output before
                                 // anything is emitted -- a client can't un-receive already-
                                 // streamed bytes if a retry becomes necessary. Buffer fully
                                 // internally (like tool_protocol below), then emit a single
                                 // content delta once a valid attempt is confirmed.
                                 std::vector<int> cur_prompt_ids = prompt_ids;
                                 sparkinfer_server::ChatRequest cur_request = chat_request;
                                 long long total_prompt_tokens = 0, total_completion_tokens = 0;
                                 sparkinfer_server::CompletionResult outcome;
                                 sparkinfer_server::ParsedAssistantOutput parsed;
                                 bool ok = false;
                                 std::string validation_err;
                                 for (int attempt = 1; attempt <= 2; ++attempt) {
                                     if ((int)cur_prompt_ids.size() + max_tokens > engine.max_seq()) {
                                         validation_err = "retry prompt exceeds server context";
                                         break;
                                     }
                                     std::vector<int> ids;
                                     std::string stop_text;
                                     bool stopped_by_sequence = false;
                                     auto on_tok = [&](int tid) -> bool {
                                         if (stop.empty()) {
                                             ids.push_back(tid);
                                             return sink.is_writable();
                                         }
                                         stop_text += g_tokenizer.decode_delta(ids, tid);
                                         size_t pos;
                                         if (find_stop_match(stop_text, stop, pos)) {
                                             stopped_by_sequence = true;
                                             return false;
                                         }
                                         return sink.is_writable();
                                     };
                                     outcome = engine.complete_streaming(cur_prompt_ids, max_tokens, on_tok, temperature, seed, top_k, top_p, presence_penalty, frequency_penalty, logit_bias);
                                     total_prompt_tokens += (long long)cur_prompt_ids.size();
                                     total_completion_tokens += (long long)ids.size();
                                     if (outcome.cancelled && !stopped_by_sequence) {
                                         g_requests_cancelled++;
                                         sink.done();
                                         return true;
                                     }
                                     if (!outcome.error.empty()) {
                                         if (outcome.overloaded) g_requests_overloaded++;
                                         else if (outcome.alloc_failed) g_requests_alloc_failed++;
                                         else if (outcome.timed_out) g_requests_timeout++;
                                         else g_requests_server_error++;
                                         const nlohmann::json error = {{"error", {{"message", outcome.error}}}};
                                         write_sse_json(sink, error);
                                         write_stream_done(sink);
                                         sink.done();
                                         return true;
                                     }
                                     std::string text;
                                     std::string decode_err;
                                     if (!decode_ids(ids, text, decode_err)) {
                                         g_requests_server_error++;
                                         const nlohmann::json error = {{"error", {{"message", decode_err}}}};
                                         write_sse_json(sink, error);
                                         write_stream_done(sink);
                                         sink.done();
                                         return true;
                                     }
                                     if (stopped_by_sequence) {
                                         size_t pos;
                                         if (find_stop_match(text, stop, pos)) text.resize(pos);
                                     }
                                     parsed = sparkinfer_server::parse_assistant_output(
                                         text, enable_thinking, engine.is_museglimmer(), nullptr);
                                     const bool truncated = outcome.reached_token_limit || stopped_by_sequence;
                                     if (truncated) {
                                         validation_err = outcome.reached_token_limit
                                             ? "truncated: hit max_tokens before producing valid output"
                                             : "truncated: hit a stop sequence before producing valid output";
                                     } else if (!sparkinfer_server::validate_response_format(
                                                    parsed.content, cur_request.response_format, validation_err)) {
                                         // validation_err already set
                                     } else {
                                         ok = true;
                                         break;
                                     }
                                     if (attempt == 1) {
                                         cur_request = build_retry_request(chat_request, parsed.content, validation_err);
                                         cur_prompt_ids = g_tokenizer.encode_augmented(cur_request, enable_thinking);
                                     }
                                 }
                                 g_prompt_tokens_total += (uint64_t)total_prompt_tokens;
                                 g_completion_tokens_total += (uint64_t)total_completion_tokens;
                                 if (!ok) {
                                     g_requests_invalid_json_output++;
                                     const nlohmann::json error = {
                                         {"error", {{"message", "model output did not satisfy response_format "
                                                                 "after retry: " + validation_err}}}};
                                     write_sse_json(sink, error);
                                     write_stream_done(sink);
                                     sink.done();
                                     return true;
                                 }
                                 write_stream_delta(sink, cid, created, "reasoning_content", parsed.reasoning_content);
                                 write_stream_delta(sink, cid, created, "content", parsed.content);
                                 g_requests_ok++;
                                 write_stream_finish(sink, cid, created, "stop");
                                 if (include_usage)
                                     write_stream_usage(sink, cid, created, (int)total_prompt_tokens,
                                                        (int)total_completion_tokens, outcome.ttft_ms,
                                                        outcome.generation_ms, outcome.decode_tps);
                                 write_stream_done(sink);
                                 sink.done();
                                 return true;
                             }

                             std::vector<int> stream_ids;
                             stream_ids.reserve((size_t)max_tokens);
                             sparkinfer_server::ThinkingStreamSplitter splitter(enable_thinking, engine.is_museglimmer());
                             sparkinfer_server::StopSequenceFilter stop_filter(stop);
                             std::string tool_stop_text;  // raw accumulator, tool_protocol branch only
                             // Set by either on_tok branch right before it returns false to signal a
                             // matched `stop` sequence -- checked ahead of outcome.cancelled below,
                             // since both now collapse to the same engine-level cancelled flag, but
                             // only a real disconnect means "the client is gone, send nothing".
                             bool stopped_by_sequence = false;
                             // Scoped out of tool-calling responses (buffered, no live content
                             // emission at all -- see the DFlash-check comment above for the
                             // parallel "logprobs never touches sampling" note).
                             const bool want_logprobs = logprobs && !tool_protocol;
                             // Accumulates since the last CONTENT delta actually flushed (not
                             // cleared on reasoning-only rounds -- see build_logprobs_content_json
                             // call sites below for where attach-and-clear happens). Entries never
                             // claimed by a content flush (reasoning-only tail, text swallowed by
                             // a stop match) are simply never emitted.
                             std::vector<sparkinfer_server::TokenLogprob> pending_logprobs;
                             auto on_tok_logprob = [&](const sparkinfer_server::TokenLogprob& tl) {
                                 pending_logprobs.push_back(tl);
                             };
                             // Returning false cancels generation -- the client is gone, so there
                             // is no point spending GPU time finishing the response.
                             auto on_tok = [&](int tid) -> bool {
                                 // Tool-capable responses are intentionally buffered until the
                                 // native Qwen XML is complete and schema-valid. This still emits
                                 // valid atomic OpenAI SSE deltas, and guarantees malformed or
                                 // partial markup can never leak as executable client content.
                                 if (tool_protocol) {
                                     if (stop.empty()) {
                                         stream_ids.push_back(tid);
                                         return sink.is_writable();
                                     }
                                     // Only pay for incremental decode here when the client
                                     // actually set `stop` -- a scoped opt-in, not a blanket cost
                                     // increase for every tool-calling request.
                                     tool_stop_text += g_tokenizer.decode_delta(stream_ids, tid);
                                     size_t pos;
                                     if (find_stop_match(tool_stop_text, stop, pos)) {
                                         stopped_by_sequence = true;
                                         return false;
                                     }
                                     return sink.is_writable();
                                 }
                                 std::string piece = g_tokenizer.decode_delta(stream_ids, tid);
                                 std::string safe = stop_filter.feed(piece);
                                 if (stop_filter.matched()) {
                                     if (!safe.empty()) {
                                         const auto delta = splitter.feed(safe);
                                         if (!delta.reasoning_content.empty())
                                             write_stream_delta(sink, cid, created, "reasoning_content",
                                                                delta.reasoning_content);
                                         if (!delta.content.empty()) {
                                             nlohmann::json lp = nullptr;
                                             if (want_logprobs) {
                                                 lp = nlohmann::json{{"content",
                                                     build_logprobs_content_json(pending_logprobs, top_logprobs)}};
                                                 pending_logprobs.clear();
                                             }
                                             write_stream_delta(sink, cid, created, "content", delta.content, lp);
                                         }
                                     }
                                     stopped_by_sequence = true;
                                     return false;
                                 }
                                 const auto delta = splitter.feed(safe);
                                 bool ok = true;
                                 if (!delta.reasoning_content.empty())
                                     ok = write_stream_delta(sink, cid, created, "reasoning_content",
                                                              delta.reasoning_content) && ok;
                                 if (!delta.content.empty()) {
                                     nlohmann::json lp = nullptr;
                                     if (want_logprobs) {
                                         lp = nlohmann::json{{"content",
                                             build_logprobs_content_json(pending_logprobs, top_logprobs)}};
                                         pending_logprobs.clear();
                                     }
                                     ok = write_stream_delta(sink, cid, created, "content", delta.content, lp) && ok;
                                 }
                                 return ok && sink.is_writable();
                             };
                             const std::function<void(const sparkinfer_server::TokenLogprob&)> maybe_on_tok_logprob =
                                 want_logprobs ? std::function<void(const sparkinfer_server::TokenLogprob&)>(on_tok_logprob)
                                              : nullptr;
                             const auto outcome = engine.complete_streaming(prompt_ids, max_tokens, on_tok,
                                 temperature, seed, top_k, top_p, presence_penalty, frequency_penalty,
                                 logit_bias, logprobs, top_logprobs, maybe_on_tok_logprob);
                             const int prompt_tokens = (int)prompt_ids.size();
                             const int completion_tokens = (int)stream_ids.size();
                             g_prompt_tokens_total += (uint64_t)prompt_tokens;
                             g_completion_tokens_total += (uint64_t)completion_tokens;
                             if (outcome.cancelled && !stopped_by_sequence) {
                                 // Client is already gone -- nothing left to write to, just record it.
                                 g_requests_cancelled++;
                                 sink.done();
                                 return true;
                             }
                             if (!outcome.error.empty()) {
                                 if (outcome.overloaded) g_requests_overloaded++;
                                 else if (outcome.alloc_failed) g_requests_alloc_failed++;
                                 else if (outcome.timed_out) g_requests_timeout++;
                                 else g_requests_server_error++;
                                 std::ostringstream err_chunk;
                                 err_chunk << "data: {\"error\":{\"message\":\"" << json_escape(outcome.error)
                                           << "\"}}\n\n";
                                 sink.write(err_chunk.str().c_str(), (size_t)err_chunk.str().size());
                                 // A client that explicitly asked for usage still deserves the
                                 // token counts even though generation faulted -- ttft/generation
                                 // timing isn't meaningful for a hard engine error, so omit those
                                 // (write_stream_usage only includes a field when its arg is >= 0).
                                 if (include_usage)
                                     write_stream_usage(sink, cid, created, prompt_tokens,
                                                        completion_tokens, -1.0, -1.0, -1.0);
                                 write_stream_done(sink);
                                 sink.done();
                                 return true;
                             }

                             std::string finish_reason = outcome.reached_token_limit ? "length" : "stop";
                             if (tool_protocol) {
                                 std::string text;
                                 std::string decode_err;
                                 if (!decode_ids(stream_ids, text, decode_err)) {
                                     g_requests_server_error++;
                                     const nlohmann::json error = {{"error", {{"message", decode_err}}}};
                                     write_sse_json(sink, error);
                                     write_stream_done(sink);
                                     sink.done();
                                     return true;
                                 }
                                 if (stopped_by_sequence) {
                                     size_t pos;
                                     if (find_stop_match(text, stop, pos)) text.resize(pos);
                                 }
                                 auto parsed = sparkinfer_server::parse_assistant_output(
                                     text, enable_thinking, engine.is_museglimmer(), &chat_request);
                                 if (!parsed.error.empty()) {
                                     if (outcome.reached_token_limit || stopped_by_sequence) {
                                         // A truncated native call is not an executable result,
                                         // but token exhaustion (or a stop sequence landing mid
                                         // tool-call XML) is still a normal completion.
                                         // Emit no buffered markup and finish with "length"/"stop".
                                         parsed = {};
                                     } else {
                                         g_requests_invalid_tool_output++;
                                         const nlohmann::json error = {
                                             {"error", {{"message", "invalid model tool call: " + parsed.error}}}};
                                         write_sse_json(sink, error);
                                         write_stream_done(sink);
                                         sink.done();
                                         return true;
                                     }
                                 }
                                 write_stream_delta(sink, cid, created, "reasoning_content",
                                                    parsed.reasoning_content);
                                 write_stream_delta(sink, cid, created, "content", parsed.content);
                                 // parsed.tool_calls is only non-empty here when parsed.error was
                                 // empty (the reached_token_limit-and-error case already reset
                                 // parsed to {} above) -- i.e. a complete, valid call, even if the
                                 // model also happened to exhaust its token budget emitting it.
                                 // Gating this on !reached_token_limit dropped that call entirely.
                                 for (size_t i = 0; i < parsed.tool_calls.size(); ++i) {
                                     parsed.tool_calls[i].id = random_id("call_");
                                     write_stream_tool_call(sink, cid, created, i, parsed.tool_calls[i]);
                                 }
                                 if (!parsed.tool_calls.empty()) finish_reason = "tool_calls";
                             } else if (!stopped_by_sequence) {
                                 // On a stop-match exit, on_tok already wrote whatever safe text
                                 // stop_filter cleared before returning false. Deliberately skip
                                 // finish() here: it exists to flush the splitter's OWN unrelated
                                 // holdback on a legitimate end of generation, not bytes stop_filter
                                 // never vetted -- generation ends at the match point regardless.
                                 sparkinfer_server::ThinkingStreamSplitter::Delta flush;
                                 splitter.finish(flush);
                                 write_stream_delta(sink, cid, created, "reasoning_content",
                                                    flush.reasoning_content);
                                 write_stream_delta(sink, cid, created, "content", flush.content);
                             }

                             g_requests_ok++;
                             write_stream_finish(sink, cid, created, finish_reason);
                             if (include_usage)
                                 write_stream_usage(sink, cid, created, prompt_tokens,
                                                    completion_tokens, outcome.ttft_ms,
                                                    outcome.generation_ms, outcome.decode_tps);
                             write_stream_done(sink);
                             sink.done();
                             return true;
                         });
                     return;
                 }

                 // engine.complete() is complete_streaming(..., nullptr); pass a real callback
                 // whenever stop was requested so non-streaming requests can halt early too --
                 // previously stop-checking (and the compute savings of stopping early) were
                 // unavailable here entirely.
                 long long total_prompt_tokens = 0, total_completion_tokens = 0;
                 sparkinfer_server::CompletionResult outcome;
                 sparkinfer_server::ParsedAssistantOutput parsed;
                 std::string finish_reason;
                 // Populated only in the plain (non-json_mode, non-tool_protocol) branch below --
                 // see the DFlash-check comment above for why tool-calling/response_format
                 // responses are scoped out of logprobs entirely for v1.
                 std::vector<sparkinfer_server::TokenLogprob> logprob_entries;
                 const bool want_logprobs = controls.logprobs && !tool_protocol;

                 if (json_mode_active) {
                     // response_format validation needs the complete output; a truncated or
                     // schema-violating attempt is retried once with a corrective follow-up
                     // prompt (this runtime's decode is deterministic greedy argmax with no RNG
                     // anywhere -- resubmitting the identical prompt would just reproduce the
                     // same invalid output) before giving up.
                     std::vector<int> cur_prompt_ids = prompt_ids;
                     sparkinfer_server::ChatRequest cur_request = chat_request;
                     bool ok = false;
                     std::string validation_err;
                     for (int attempt = 1; attempt <= 2; ++attempt) {
                         if ((int)cur_prompt_ids.size() + max_tokens > engine.max_seq()) {
                             validation_err = "retry prompt exceeds server context";
                             break;
                         }
                         std::vector<int> ids;
                         std::string stop_text;
                         bool stopped_by_sequence = false;
                         auto on_tok = [&](int tid) -> bool {
                             if (controls.stop.empty()) {
                                 ids.push_back(tid);
                                 return true;
                             }
                             stop_text += g_tokenizer.decode_delta(ids, tid);
                             size_t pos;
                             if (find_stop_match(stop_text, controls.stop, pos)) {
                                 stopped_by_sequence = true;
                                 return false;
                             }
                             return true;
                         };
                         outcome = engine.complete_streaming(cur_prompt_ids, max_tokens, on_tok, controls.temperature, controls.seed, controls.top_k, controls.top_p, controls.presence_penalty, controls.frequency_penalty, controls.logit_bias);
                         total_prompt_tokens += (long long)cur_prompt_ids.size();
                         total_completion_tokens += (long long)ids.size();
                         if (!outcome.error.empty()) {
                             // A hard engine fault (overloaded/alloc_failed/timed_out) is not a
                             // validation failure -- never retry it, propagate immediately.
                             res.status = record_and_status(outcome);
                             res.set_content("{\"error\":{\"message\":\"" + json_escape(outcome.error) + "\"}}",
                                             "application/json");
                             return;
                         }
                         std::string text;
                         if (!decode_ids(ids, text, err)) {
                             g_requests_server_error++;
                             res.status = 500;
                             res.set_content("{\"error\":{\"message\":\"" + json_escape(err) + "\"}}",
                                             "application/json");
                             return;
                         }
                         if (stopped_by_sequence) {
                             size_t pos;
                             if (find_stop_match(text, controls.stop, pos)) text.resize(pos);
                         }
                         parsed = sparkinfer_server::parse_assistant_output(
                             text, enable_thinking, engine.is_museglimmer(), nullptr);
                         // Unlike tool_protocol, truncation is NOT a free pass here -- a
                         // stop/length-truncated response is essentially always invalid JSON, so
                         // it's a normal validation failure subject to the same retry policy.
                         const bool truncated = outcome.reached_token_limit || stopped_by_sequence;
                         if (truncated) {
                             validation_err = outcome.reached_token_limit
                                 ? "truncated: hit max_tokens before producing valid output"
                                 : "truncated: hit a stop sequence before producing valid output";
                         } else if (!sparkinfer_server::validate_response_format(
                                        parsed.content, cur_request.response_format, validation_err)) {
                             // validation_err already set by validate_response_format
                         } else {
                             ok = true;
                             break;
                         }
                         if (attempt == 1) {
                             cur_request = build_retry_request(chat_request, parsed.content, validation_err);
                             cur_prompt_ids = g_tokenizer.encode_augmented(cur_request, enable_thinking);
                         }
                     }
                     g_prompt_tokens_total += (uint64_t)total_prompt_tokens;
                     g_completion_tokens_total += (uint64_t)total_completion_tokens;
                     if (!ok) {
                         g_requests_invalid_json_output++;
                         res.status = 502;
                         const nlohmann::json error = {
                             {"error", {{"message", "model output did not satisfy response_format after "
                                                     "retry: " + validation_err}}}};
                         res.set_content(error.dump(), "application/json");
                         return;
                     }
                     finish_reason = "stop";  // ok only true on a non-truncated, valid attempt
                 } else {
                     std::vector<int> nonstream_ids;
                     std::string nonstream_stop_text;
                     bool stopped_by_sequence = false;
                     auto nonstream_on_tok_logprob = [&](const sparkinfer_server::TokenLogprob& tl) {
                         logprob_entries.push_back(tl);
                     };
                     auto nonstream_on_tok = [&](int tid) -> bool {
                         if (controls.stop.empty()) {
                             nonstream_ids.push_back(tid);
                             return true;
                         }
                         nonstream_stop_text += g_tokenizer.decode_delta(nonstream_ids, tid);
                         size_t pos;
                         if (find_stop_match(nonstream_stop_text, controls.stop, pos)) {
                             stopped_by_sequence = true;
                             // The match-triggering token's text got truncated below (a stop
                             // match can land mid-token) -- drop its logprobs entry too, no
                             // partial-credit modeling for v1. Fires after nonstream_on_tok_logprob
                             // (delivered before on_token for the same token, see step_job()'s
                             // ordering contract), so this correctly removes THIS token's entry.
                             if (want_logprobs && !logprob_entries.empty()) logprob_entries.pop_back();
                             return false;
                         }
                         return true;
                     };
                     const std::function<void(const sparkinfer_server::TokenLogprob&)> maybe_nonstream_on_tok_logprob =
                         want_logprobs ? std::function<void(const sparkinfer_server::TokenLogprob&)>(nonstream_on_tok_logprob)
                                      : nullptr;
                     outcome = engine.complete_streaming(prompt_ids, max_tokens, nonstream_on_tok,
                         controls.temperature, controls.seed, controls.top_k, controls.top_p,
                         controls.presence_penalty, controls.frequency_penalty, controls.logit_bias,
                         controls.logprobs, controls.top_logprobs, maybe_nonstream_on_tok_logprob);
                     // Defensive clamp -- should already hold, cheap insurance against any
                     // subtle off-by-one between the two accumulation paths above.
                     if (logprob_entries.size() > outcome.tokens.size())
                         logprob_entries.resize(outcome.tokens.size());
                     std::string text;
                     if (!outcome.error.empty()) {
                         res.status = record_and_status(outcome);
                         res.set_content("{\"error\":{\"message\":\"" + json_escape(outcome.error) + "\"}}",
                                         "application/json");
                         return;
                     }
                     if (!decode_ids(outcome.tokens, text, err)) {
                         g_requests_server_error++;
                         res.status = 500;
                         res.set_content("{\"error\":{\"message\":\"" + json_escape(err) + "\"}}",
                                         "application/json");
                         return;
                     }
                     total_prompt_tokens = (long long)prompt_ids.size();
                     total_completion_tokens = (long long)outcome.tokens.size();
                     g_prompt_tokens_total += (uint64_t)prompt_ids.size();
                     g_completion_tokens_total += (uint64_t)outcome.tokens.size();
                     if (stopped_by_sequence) {
                         size_t pos;
                         if (find_stop_match(text, controls.stop, pos)) text.resize(pos);
                     }

                     parsed = sparkinfer_server::parse_assistant_output(
                         text, enable_thinking, engine.is_museglimmer(),
                         tool_protocol ? &chat_request : nullptr);
                     if (!parsed.error.empty()) {
                         if (outcome.reached_token_limit || stopped_by_sequence) {
                             // Never expose a truncated native tag sequence. A length/stop end is a
                             // valid completion, so return an empty assistant result instead of 5xx.
                             parsed = {};
                         } else {
                             g_requests_invalid_tool_output++;
                             res.status = 502;
                             const nlohmann::json error = {
                                 {"error", {{"message", "invalid model tool call: " + parsed.error}}}};
                             res.set_content(error.dump(), "application/json");
                             return;
                         }
                     }
                     finish_reason = outcome.reached_token_limit ? "length" : "stop";
                 }

                 nlohmann::json message = {{"role", "assistant"}};
                 if (!parsed.reasoning_content.empty())
                     message["reasoning_content"] = parsed.reasoning_content;
                 // See the streaming path's identical comment: parsed.tool_calls is only
                 // non-empty here for a complete, successfully-parsed call.
                 if (!parsed.tool_calls.empty()) {
                     message["content"] = parsed.content.empty()
                         ? nlohmann::json(nullptr) : nlohmann::json(parsed.content);
                     message["tool_calls"] = nlohmann::json::array();
                     for (auto& call : parsed.tool_calls) {
                         call.id = random_id("call_");
                         message["tool_calls"].push_back({
                             {"id", call.id}, {"type", "function"},
                             {"function", {{"name", call.name}, {"arguments", call.arguments}}}});
                     }
                     finish_reason = "tool_calls";
                 } else {
                     message["content"] = parsed.content;
                 }

                 nlohmann::json usage = {
                     {"prompt_tokens", (int)total_prompt_tokens},
                     {"completion_tokens", (int)total_completion_tokens},
                     {"total_tokens", (int)(total_prompt_tokens + total_completion_tokens)}};
                 if (outcome.ttft_ms >= 0.0) usage["ttft_ms"] = outcome.ttft_ms;
                 if (outcome.generation_ms >= 0.0) usage["generation_ms"] = outcome.generation_ms;
                 if (outcome.decode_tps >= 0.0) usage["decode_tps"] = outcome.decode_tps;
                 const nlohmann::json logprobs_json = want_logprobs
                     ? nlohmann::json{{"content", build_logprobs_content_json(logprob_entries, controls.top_logprobs)}}
                     : nlohmann::json(nullptr);
                 const nlohmann::json body = {
                     {"id", cid}, {"object", "chat.completion"}, {"created", created},
                     {"model", g_model_name},
                     {"choices", nlohmann::json::array({{{"index", 0},
                                                          {"message", message},
                                                          {"logprobs", logprobs_json},
                                                          {"finish_reason", finish_reason}}})},
                     {"usage", usage}};
                 g_requests_ok++;
                 res.set_content(body.dump(), "application/json");
             });

    // Transport-level deadlines. Defaults are generous, not aggressive: a cold 32k-context
    // prefill has been measured taking ~90s of TTFT alone (see eval/pr_dflash_bot.py's 32k
    // sweep), so a short default here would misfire on legitimate long-context requests.
    // The read timeout resets on each byte received, so a slow streaming response keeps
    // extending it as it goes -- this only fires on a genuinely stalled connection.
    const long read_timeout_s = getenv("SPARKINFER_READ_TIMEOUT_S") ? atol(getenv("SPARKINFER_READ_TIMEOUT_S")) : 300;
    const long write_timeout_s = getenv("SPARKINFER_WRITE_TIMEOUT_S") ? atol(getenv("SPARKINFER_WRITE_TIMEOUT_S")) : 300;
    svr.set_read_timeout(read_timeout_s, 0);
    svr.set_write_timeout(write_timeout_s, 0);

    std::signal(SIGTERM, on_shutdown_signal);
    std::signal(SIGINT, on_shutdown_signal);
    // Signal handlers must stay async-signal-safe (just set the atomic flag above); the actual
    // drain-and-stop happens here, on an ordinary thread. /v1/chat/completions and /v1/capacity
    // already check g_shutdown_requested to refuse new work as soon as the flag flips, so the
    // gap between "signal received" and "svr.stop() called" (at most one poll interval) only
    // means a few new requests might land right before shutdown starts, not that the drain window
    // is unbounded.
    std::thread shutdown_watcher([&svr] {
        while (!g_shutdown_requested.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        fprintf(stderr, "[sparkinfer-server] shutdown signal received, draining in-flight requests...\n");
        svr.stop();
        // svr.stop() only closes the LISTENING socket -- it does not force-close connections
        // already accepted. A client that vanishes without a clean TCP close (RST, dead network,
        // a hard `kill` on a curl process) can leave an httplib worker thread blocked in a read()
        // for up to the read timeout (SPARKINFER_READ_TIMEOUT_S, default 300s) -- measured: this
        // alone can make listen() take minutes to return, defeating the point of a "graceful"
        // shutdown. Bound the total drain time instead: same SIGTERM -> grace period -> force-kill
        // shape as Kubernetes' terminationGracePeriodSeconds.
        const long grace_s = getenv("SPARKINFER_SHUTDOWN_GRACE_S")
                                 ? atol(getenv("SPARKINFER_SHUTDOWN_GRACE_S")) : 30;
        std::this_thread::sleep_for(std::chrono::seconds(grace_s));
        fprintf(stderr, "[sparkinfer-server] shutdown grace period (%lds) elapsed with requests "
                        "still draining -- forcing exit\n", grace_s);
        _exit(0);  // not exit(): other threads may still be mid-flight; skip atexit/static dtors
    });

    const std::string queue_depth_label =
        engine.max_queue_depth() > 0 ? std::to_string(engine.max_queue_depth()) : std::string("unlimited");
    fprintf(stderr,
            "[sparkinfer-server] OpenAI-compatible API on http://%s:%d\n"
            "  GET  /health\n"
            "  GET  /v1/models\n"
            "  GET  /v1/info\n"
            "  GET  /v1/capacity\n"
            "  GET  /metrics\n"
            "  POST /v1/tokenize\n"
            "  POST /v1/chat/completions\n"
            "  read_timeout=%lds write_timeout=%lds max_output_tokens=%d max_queue_depth=%s\n",
            host.c_str(), port, read_timeout_s, write_timeout_s, max_output_tokens(),
            queue_depth_label.c_str());

    const bool bound = svr.listen(host.c_str(), port);
    g_shutdown_requested = true;  // unblock the watcher thread if listen() returned on its own
    shutdown_watcher.join();
    if (!bound) {
        fprintf(stderr, "[sparkinfer-server] failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
