"""Oracle Code Assist Responses API configuration."""

from typing import Dict, Optional, Union

from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..common_utils import apply_oca_auth_headers, normalize_oca_request_params, strip_oca_unsupported_request_params


class OCAResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """Streaming-only, full-input Responses API adapter for Oracle Code Assist."""

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OCA

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        params = dict(response_api_optional_params)
        normalize_oca_request_params(params)
        strip_oca_unsupported_request_params(params)
        params["stream"] = True
        return super().map_openai_params(params, model, drop_params)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        try:
            apply_oca_auth_headers(headers, model)
        except Exception as exc:
            raise OpenAIError(status_code=401, message=f"Failed to acquire OCA OAuth token: {exc}") from exc
        return headers

    def transform_responses_api_request(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        response_api_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        params = dict(response_api_optional_request_params)
        normalize_oca_request_params(params)
        strip_oca_unsupported_request_params(params)
        params["stream"] = True
        data = super().transform_responses_api_request(model, input, params, litellm_params, headers)
        data.pop("stream_options", None)
        data.pop("previous_response_id", None)
        data["stream"] = True
        data["store"] = False
        return data

    def get_complete_url(self, api_base: Optional[str], litellm_params: dict) -> str:
        base = (api_base or get_secret_str("OCA_API_BASE") or "").rstrip("/")
        if not base:
            raise OpenAIError(status_code=400, message="OCA_API_BASE is required")
        return base if base.endswith("/responses") else f"{base}/responses"
