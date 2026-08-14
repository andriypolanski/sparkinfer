#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace sparkinfer_server {

struct ToolCall {
    std::string id;
    std::string name;
    // OpenAI wire representation: a compact JSON object encoded as a string.
    std::string arguments;
};

struct ToolDefinition {
    std::string name;
    // The complete top-level OpenAI tool object ({"type":"function","function":{...}}).
    nlohmann::json spec;
};

enum class ToolChoiceMode {
    kAuto,
    kNone,
};

struct ChatMessage {
    std::string role;
    std::string content;
    bool content_is_null = false;
    std::string reasoning_content;
    std::string name;
    std::string tool_call_id;
    std::vector<ToolCall> tool_calls;
};

enum class ResponseFormatType {
    kText,
    kJsonObject,
    kJsonSchema,
};

struct ResponseFormat {
    ResponseFormatType type = ResponseFormatType::kText;
    std::string schema_name;   // json_schema.name -- steering text / logging only
    nlohmann::json schema;     // json_schema.schema -- empty/ignored unless type == kJsonSchema
    // json_schema.strict -- parsed and stored, but a documented v1 no-op: this backend has no
    // constrained-decoding mechanism, so it cannot honor a real conformance guarantee either way.
    bool strict = false;
};

struct ChatRequest {
    std::vector<ChatMessage> messages;
    std::vector<ToolDefinition> tools;
    ToolChoiceMode tool_choice = ToolChoiceMode::kAuto;
    bool parallel_tool_calls = true;
    ResponseFormat response_format;
};

struct ParsedToolOutput {
    std::string reasoning_content;
    std::string content;
    std::vector<ToolCall> tool_calls;
    std::string error;
};

bool parse_chat_request_json(const std::string& body, ChatRequest& request, std::string& err);

// Decode-control fields that don't influence prompt construction (unlike ChatRequest's
// tools/response_format, which do) -- extracted here rather than kept private to
// sparkinfer_server.cpp so parsing/validation has a unit-test seam without a running server/GPU.
struct RequestControls {
    bool stream = false;
    bool include_usage = false;
    int max_tokens = 256;
    std::vector<std::string> stop;
    // <= 0 (default) is plain greedy argmax, byte-identical to pre-sampling behavior.
    float temperature = 0.f;
    uint64_t seed = 0;       // only meaningful when seed_set
    bool seed_set = false;   // client explicitly supplied `seed`; false => caller should generate one
    // top_k <= 0 disables top_k (0 = no limit, matching llama.cpp/vLLM). top_p >= 1.0 disables
    // top_p (1.0 is OpenAI's own "disabled" default). Neither requires temperature > 0 -- see
    // ContinuousBatchEngine::Request's doc comment for the inertness proof.
    int top_k = 0;
    float top_p = 1.0f;
    // [-2.0, 2.0]; 0 (default, OpenAI's own default) disables both. Sampling controls, same tier
    // as temperature/top_k/top_p (NOT logprobs, which is pure output reporting) -- threaded
    // through every complete_streaming call site, including the json_mode/tool-calling retry
    // loops that top_k/top_p already reach but logprobs/top_logprobs do not.
    float presence_penalty = 0.f;
    float frequency_penalty = 0.f;
    // logprobs=false (default) attaches no logprobs field anywhere in the response. top_logprobs
    // is only meaningful when logprobs is true -- unlike top_k/top_p, this IS cross-validated
    // against a sibling field: parse_request_controls rejects top_logprobs supplied without
    // logprobs=true (matches OpenAI's own documented constraint).
    bool logprobs = false;
    int top_logprobs = 0;   // 0-20 when logprobs=true
    // OpenAI's logit_bias: (token_id, bias) pairs, bias in [-100.0, 100.0]. Empty (default)
    // disables it -- same self-describing-default convention as top_p/top_k, no `_set` bool
    // needed. Sampling-control tier, same as presence_penalty/frequency_penalty -- threaded
    // through every complete_streaming call site, including the json_mode/tool-calling retry
    // loops.
    std::vector<std::pair<int, float>> logit_bias;
};

// vocab, when > 0, additionally rejects any logit_bias token id >= vocab with a real validation
// error (400) instead of silently letting it through -- 0 (the default) skips that upper-bound
// check, so existing callers that don't have a vocab size handy are unaffected. Negative token
// ids and malformed keys are always rejected regardless of vocab.
bool parse_request_controls(const std::string& body, RequestControls& out, std::string& err,
                            int vocab = 0);

// Pure decision function for the DFlash+temperature-sampling incompatibility (kept separate from
// getenv() so it's unit-testable without a process-wide env var): true => the request should be
// rejected with 400. temperature<=0 is always accepted regardless of dflash_env_on.
bool should_reject_dflash_temperature(bool dflash_env_on, float temperature);

// Pure decision function for the DFlash+presence/frequency-penalty incompatibility -- mirrors
// should_reject_dflash_temperature, but is a SEPARATE check (not folded into it) because penalty
// has no inertness proof at temperature<=0 the way top_k/top_p do: a nonzero penalty CAN change
// the greedy-argmax winner on its own. true => the request should be rejected with 400.
// presence_penalty==0 && frequency_penalty==0 is always accepted regardless of dflash_env_on.
bool should_reject_dflash_penalty(bool dflash_env_on, float presence_penalty, float frequency_penalty);

// Pure decision function for the DFlash+logit_bias incompatibility -- mirrors
// should_reject_dflash_penalty; logit_bias has the same "no inertness proof at temperature<=0" gap
// (an arbitrary per-vocab additive bias CAN change the greedy-argmax winner on its own). true =>
// the request should be rejected with 400. An empty logit_bias is always accepted regardless of
// dflash_env_on.
bool should_reject_dflash_logit_bias(bool dflash_env_on, bool has_logit_bias);

// Best-effort validation only -- there is no constrained decoding in this backend, so this
// cannot guarantee the model's output actually conforms; it just checks after the fact.
// format.type == kText always returns true (no-op).
bool validate_response_format(const std::string& content, const ResponseFormat& format,
                              std::string& err);

std::string apply_qwen36_tools_template(const ChatRequest& request,
                                        bool enable_thinking = false);

ParsedToolOutput parse_qwen36_tool_output(const std::string& raw,
                                          bool enable_thinking,
                                          const ChatRequest& request);

}  // namespace sparkinfer_server
