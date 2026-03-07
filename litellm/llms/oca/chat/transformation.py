"""
OCA (Oracle Code Assist) — Chat Completions Configuration.

Extends OpenAIGPTConfig to handle:
- File-based bearer token authentication
- OCA-specific custom headers (opc-request-id, client, etc.)
- Forced streaming for OCA's SSE-only response format
- Stripping unsupported parameters from the request body
"""

from typing import Any, List, Optional, cast

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai.common_utils import OpenAIError
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import LlmProviders

from ..common_utils import add_oca_headers, read_oca_access_token

# Parameters that OCA's /chat/completions endpoint accepts.
# Everything else is stripped from the request body.
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
}


class OCAChatConfig(OpenAIGPTConfig):
    """
    Configuration for Oracle Code Assist (OCA) chat completions.

    OCA exposes an OpenAI-compatible ``/chat/completions`` endpoint but has
    two quirks that require provider-level handling:

    1. **File-based auth** — The bearer token lives in a file that is
       periodically refreshed by a host-side token refresher service.
       The token file path is passed via ``api_key`` or the
       ``OCA_ACCESS_TOKEN_FILE`` environment variable.

    2. **SSE-only responses** — OCA always returns ``text/event-stream``
       even for non-streaming requests.  We force ``stream=True`` at the
       SDK level and reassemble chunks for callers that didn't request
       streaming (``should_fake_stream`` returns ``True``).
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OCA

    def get_supported_openai_params(self, model: str) -> list:
        """Return only the subset of OpenAI params that OCA accepts."""
        return [
            "temperature",
            "top_p",
            "n",
            "stop",
            "max_tokens",
            "max_completion_tokens",
            "presence_penalty",
            "frequency_penalty",
            "user",
        ]

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
        """Read OCA token from file and set Authorization + OCA headers."""
        oca_token = read_oca_access_token(api_key=api_key)
        if not oca_token:
            raise OpenAIError(
                status_code=401,
                message=(
                    "Missing OCA bearer token. Ensure OCA_ACCESS_TOKEN_FILE is set "
                    "and the token refresher is running, or enable "
                    "OCA_ENABLE_API_KEY_FALLBACK with OCA_API_KEY."
                ),
                headers=cast(httpx.Headers, {}),
            )

        headers["Authorization"] = f"Bearer {oca_token}"
        add_oca_headers(headers=headers, model=model, token=oca_token)

        # Ensure Content-Type
        if "content-type" not in headers and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        return headers

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """Build OCA request body, stripping params OCA doesn't support."""
        # Start with the parent's transform (which builds a standard OpenAI body)
        data = super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Strip any keys that OCA doesn't accept
        return {k: v for k, v in data.items() if k in _OCA_SUPPORTED_PARAMS}

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        """OCA always returns SSE — never fake the stream.

        We always let LiteLLM treat the response as a real stream
        (``stream=True`` at the HTTP level).  For callers that did NOT
        request streaming, LiteLLM's ``CustomStreamWrapper`` reassembles
        the chunks into a single ``ModelResponse`` automatically.
        """
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
        """Build the OCA chat completions URL."""
        api_base = (
            api_base
            or get_secret_str("OCA_API_BASE")
        )
        if not api_base:
            raise OpenAIError(
                status_code=400,
                message="OCA_API_BASE is required but not set.",
                headers=cast(httpx.Headers, {}),
            )

        api_base = api_base.rstrip("/")
        endpoint = "chat/completions"

        # Don't double-append if already present
        if endpoint in api_base:
            return api_base

        return f"{api_base}/{endpoint}"
