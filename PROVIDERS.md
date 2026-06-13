# ai-infra custom providers — isolation map

This fork (`indradeep/litellm-proxy`, based on upstream `BerriAI/litellm` @ `v1.88.1`)
adds three custom providers and a provider-agnostic Cursor/Responses layer.

Each provider is **self-contained**: it lives in its own module and is wired in
through a small, fixed set of *registration touchpoints* that every LiteLLM
provider uses. No provider hardcodes another provider, and the shared
Cursor/Responses plumbing contains **no provider-specific branches**.

## How Cursor connects

Cursor → `POST /cursor/chat/completions` (or `/v1/responses`) on the proxy →
LiteLLM routes by **model-name prefix** to the matching provider:

| Cursor sends `model` | Routes to provider |
|----------------------|--------------------|
| `gpt-5.5` (router `model_group_alias` → `oca/gpt-5.5`) | `oca` |
| `oca/*`               | `oca` |
| `clip/*`              | `clip` |
| `codex/*`             | `openai` (local proxy via `CODEX_API_BASE`) |
| `oci/*`               | `oci` |

The `/cursor/chat/completions` endpoint and the Responses→chat-completions
stream transform are **provider-agnostic** — they normalize the Cursor request
(`messages`→`input`, `-extra`→`xhigh`) and translate the stream back. The only
thing that selects OCA vs Clip is the `model_name` Cursor sends.

## Shared Cursor / Responses layer (NOT provider-specific)

These files benefit every Responses-API provider (OCA, Clip, Codex). Keep them
generic — they are the most likely to conflict on the next upstream merge.

- `litellm/proxy/response_api_endpoints/endpoints.py` — `/cursor/*`, request
  normalization, SSE success-logging.
- `litellm/completion_extras/litellm_responses_transformation/transformation.py`
  — Responses→chat-completions stream transform (tool-call index mapping,
  argument backfill).
- `litellm/llms/openai/responses/transformation.py` — tolerant SSE-framed
  payload parsing (helps Clip and Codex).
- `litellm/llms/base_llm/responses/transformation.py` — generic hooks:
  `async_prepare_responses_api_request`, `requires_streaming_upstream`,
  `should_buffer_streaming_upstream_response`.
- `litellm/llms/custom_httpx/llm_http_handler.py` — stream-upstream / buffer
  handling for providers that only support SSE.
- `litellm/responses/utils.py` — `collect_{sync,async}_responses_stream`.
- `litellm/proxy/litellm_pre_call_utils.py` — `cursor` added to metadata routes.
- `litellm/llms/anthropic/experimental_pass_through/**` — reasoning effort /
  thinking-display mapping (used by Cursor's Anthropic-style requests).

Provider hook contract: a provider that needs to rewrite the request before it
is sent (e.g. OCA Zero Data Retention) overrides
`BaseResponsesAPIConfig.async_prepare_responses_api_request(...)`. Core code
(`litellm/responses/main.py`) calls the hook generically and never imports a
provider module.

## Provider: OCA (Oracle Code Assist)

Custom OAuth (`client_credentials`) Responses/chat provider with Zero Data
Retention input rebuild.

**Module (delete to remove):** `litellm/llms/oca/**`

**Registration touchpoints (standard for any provider):**
- `litellm/types/utils.py` — `LlmProviders.OCA = "oca"`
- `litellm/litellm_core_utils/get_llm_provider_logic.py` — `model.startswith("oca/")`
- `litellm/utils.py` — `ProviderConfigManager` chat + responses config entries
- `litellm/__init__.py` + `litellm/_lazy_imports_registry.py` — config exports
- `litellm/main.py` — `elif custom_llm_provider == "oca":` chat dispatch branch
- cost map: `register_oca_upstream_model_costs()` runs on `oca` module import

**Self-contained behaviors (no shared-file edits needed):**
- ZDR input rebuild → `OCAResponsesAPIConfig.async_prepare_responses_api_request`
- SSE-only upstream → `requires_streaming_upstream` / buffer override
- OAuth headers, effort/service-tier normalization → `oca/common_utils.py`

## Provider: OCI (Oracle Cloud Infrastructure GenAI)

Native OCI GenAI chat + embeddings (request signing).

**Module (delete to remove):** `litellm/llms/oci/**`, `litellm/types/llms/oci.py`

**Registration touchpoints:**
- `litellm/types/utils.py` — `LlmProviders.OCI`
- `litellm/litellm_core_utils/get_llm_provider_logic.py` — `model.startswith("oci/")`
- `litellm/utils.py` — chat + embed config entries
- `litellm/__init__.py` + `litellm/_lazy_imports_registry.py` — config exports + `oci_models`
- `litellm/main.py` — `oci` chat + embedding dispatch branches
- `litellm/constants.py` — `oci` in provider list

Independent of Cursor/Clip.

## Provider: Clip (local OpenAI-compatible proxy, port 8317)

JSON-driven `openai_like` provider — **no dedicated module**.

**Definition (delete to remove):**
- `litellm/llms/openai_like/providers.json` — the `clip` entry
- `litellm/types/utils.py` — `LlmProviders.CLIP = "clip"`

**Shared generic helpers in `litellm/llms/openai_like/dynamic_config.py`**
(driven by `special_handling` flags, reusable by any JSON provider):
- `convert_content_list_to_string` (text-only flattening)
- `responses_convert_anthropic_tool_blocks`
- `responses_drop_unsupported_params`

Clip routes through the shared Cursor/Responses layer with explicit `clip/*`
model names (see `config/litellm-proxy/config.yaml` in the outer repo).

## Removal checklist (any provider)

1. Delete the provider module (or its `providers.json` entry for Clip).
2. Remove its registration touchpoints listed above.
3. No edits to the shared Cursor/Responses layer are required.
4. `make ai-check` and the provider unit tests under
   `tests/test_litellm/llms/<provider>/` confirm the rest is intact.
