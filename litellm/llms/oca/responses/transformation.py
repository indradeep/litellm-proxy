"""
OCA (Oracle Code Assist) — Responses API Configuration.

Extends OpenAIResponsesAPIConfig to handle:
- In-process OAuth client_credentials bearer token authentication
- OCA-specific custom headers
- SSE streaming (OCA rejects non-streaming /responses)
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..common_utils import (
    apply_oca_auth_headers,
    normalize_oca_request_params,
    strip_oca_unsupported_request_params,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


def _env_flag_enabled(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


class OCAResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """Configuration for Oracle Code Assist (OCA) Responses API."""

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OCA

    async def async_prepare_responses_api_request(
        self,
        *,
        input: Any,
        response_api_optional_request_params: Dict,
        litellm_params: Any,
    ):
        # OCA Zero Data Retention: rebuild full input and drop previous_response_id.
        from ..common_utils import prepare_oca_zdr_responses_request

        return await prepare_oca_zdr_responses_request(
            input=input,
            response_api_optional_params=response_api_optional_request_params,
        )

    def requires_streaming_upstream(self, stream: Optional[bool]) -> bool:
        # OCA returns HTTP 400: "Non Streaming Request are not supported".
        return stream is not True

    def should_buffer_streaming_upstream_response(
        self,
        stream: Optional[bool],
    ) -> bool:
        # Default to real streaming. This escape hatch exists only for clients
        # that cannot consume streamed /responses events.
        return stream is not True and _env_flag_enabled(
            get_secret_str("OCA_BUFFER_RESPONSES_STREAM")
        )

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        params = dict(response_api_optional_params)
        normalize_oca_request_params(params, prefer_reasoning_dict=True)
        strip_oca_unsupported_request_params(params)
        return super().map_openai_params(
            response_api_optional_params=params,
            model=model,
            drop_params=drop_params,
        )

    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        try:
            apply_oca_auth_headers(headers=headers, model=model)
        except Exception as exc:
            raise OpenAIError(
                status_code=401,
                message=f"Failed to acquire OCA OAuth token: {exc}",
            ) from exc
        return headers

    def transform_responses_api_request(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        response_api_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        normalize_oca_request_params(
            response_api_optional_request_params, prefer_reasoning_dict=True
        )
        strip_oca_unsupported_request_params(response_api_optional_request_params)
        result = super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        result.pop("stream_options", None)
        return result

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_base = api_base or get_secret_str("OCA_API_BASE")
        if not api_base:
            raise OpenAIError(
                status_code=400,
                message="OCA_API_BASE is required but not set.",
            )

        api_base = api_base.rstrip("/")
        if api_base.endswith("/responses"):
            return api_base
        return f"{api_base}/responses"
