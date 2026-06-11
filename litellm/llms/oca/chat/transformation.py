"""
OCA (Oracle Code Assist) — Chat Completions Configuration.

Extends OpenAIGPT5Config to handle:
- In-process OAuth client_credentials bearer token authentication
- OCA-specific custom headers (opc-request-id, client, etc.)
- GPT-5 reasoning_effort / service_tier passthrough (e.g. xhigh + priority)
- Forced streaming for OCA's SSE-only response format
- Stripping unsupported parameters from the request body
"""

from typing import List, Optional, cast

import httpx

from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config
from litellm.llms.openai.common_utils import OpenAIError
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import LlmProviders

from ..common_utils import (
    apply_oca_auth_headers,
    normalize_oca_request_params,
    strip_oca_unsupported_request_params,
)

# Parameters that OCA's /chat/completions endpoint accepts.
_OCA_SUPPORTED_PARAMS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "n",
    "stop",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "user",
    "stream",
    "tools",
    "tool_choice",
    "reasoning_effort",
    "verbosity",
    "service_tier",
    "response_format",
}


class OCAChatConfig(OpenAIGPT5Config):
    """Configuration for Oracle Code Assist (OCA) chat completions."""

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OCA

    @classmethod
    def _supports_reasoning_effort_level(cls, model: str, level: str) -> bool:
        # OCA validates effort levels server-side; do not strip xhigh/minimal/low.
        return True

    @classmethod
    def _is_reasoning_effort_level_explicitly_disabled(
        cls, model: str, level: str
    ) -> bool:
        return False

    def get_supported_openai_params(self, model: str) -> list:
        params = super().get_supported_openai_params(model=model)
        if "service_tier" not in params:
            params.append("service_tier")
        return [param for param in params if param != "stream_options"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        normalize_oca_request_params(non_default_params, prefer_reasoning_dict=False)
        normalize_oca_request_params(optional_params, prefer_reasoning_dict=False)
        strip_oca_unsupported_request_params(non_default_params)
        strip_oca_unsupported_request_params(optional_params)
        return super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        try:
            apply_oca_auth_headers(headers=headers, model=model)
        except Exception as exc:
            raise OpenAIError(
                status_code=401,
                message=f"Failed to acquire OCA OAuth token: {exc}",
                headers=cast(httpx.Headers, {}),
            ) from exc
        return headers

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        normalize_oca_request_params(optional_params, prefer_reasoning_dict=False)
        strip_oca_unsupported_request_params(optional_params)
        data = super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        return {k: v for k, v in data.items() if k in _OCA_SUPPORTED_PARAMS}

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        return False

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        api_base = api_base or get_secret_str("OCA_API_BASE")
        if not api_base:
            raise OpenAIError(
                status_code=400,
                message="OCA_API_BASE is required but not set.",
                headers=cast(httpx.Headers, {}),
            )

        api_base = api_base.rstrip("/")
        endpoint = "chat/completions"
        if endpoint in api_base:
            return api_base
        return f"{api_base}/{endpoint}"
