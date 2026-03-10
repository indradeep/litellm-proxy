"""
OCA (Oracle Code Assist) — Responses API Configuration.

Extends OpenAIResponsesAPIConfig to handle:
- File-based bearer token authentication
- OCA-specific custom headers
- SSE fallback parsing (inherited from parent)
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

import litellm
from litellm._logging import verbose_logger
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..common_utils import add_oca_headers, read_oca_access_token

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class OCAResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for Oracle Code Assist (OCA) Responses API.

    Inherits from OpenAIResponsesAPIConfig since OCA's /responses endpoint
    is OpenAI-compatible. Key differences:

    1. **File-based auth** — reads bearer token from file at request time.
    2. **OCA headers** — adds ``opc-request-id``, ``client``, etc.
    3. **SSE fallback** — inherits ``_extract_json_from_sse_payload()`` from
       the parent class, which already handles OCA's SSE-only responses.
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OCA

    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        """Read OCA token from file and set Authorization + OCA headers."""
        litellm_params = litellm_params or GenericLiteLLMParams()
        oca_token = read_oca_access_token(api_key=litellm_params.api_key)
        if not oca_token:
            raise OpenAIError(
                status_code=401,
                message=(
                    "Missing OCA bearer token. Ensure OCA_ACCESS_TOKEN_FILE is set "
                    "and the token refresher is running, or enable "
                    "OCA_ENABLE_API_KEY_FALLBACK with OCA_API_KEY."
                ),
            )

        headers["Authorization"] = f"Bearer {oca_token}"
        add_oca_headers(headers=headers, model=model, token=oca_token)
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """Build the OCA responses API URL."""
        api_base = (
            api_base
            or get_secret_str("OCA_API_BASE")
        )
        if not api_base:
            raise OpenAIError(
                status_code=400,
                message="OCA_API_BASE is required but not set.",
            )

        api_base = api_base.rstrip("/")
        return f"{api_base}/responses"
