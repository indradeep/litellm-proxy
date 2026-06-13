"""
Dynamic configuration class generator for JSON-based providers.
"""

import json
from typing import Any, Coroutine, Dict, List, Literal, Optional, Tuple, Union, cast, overload

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from .json_loader import SimpleProviderConfig

_TEXT_CONTENT_TYPES = frozenset({"text", "input_text", "output_text"})
_NON_TEXT_CONTENT_TYPES = frozenset(
    {
        "tool_use",
        "tool_result",
        "function_call",
        "function_call_output",
        "image_url",
        "input_audio",
        "file",
        "refusal",
    }
)


def _message_has_non_text_content_blocks(message: AllMessageValues) -> bool:
    """Return True when content must stay structured (tools, media, etc.)."""
    if message.get("tool_calls") or message.get("function_call"):
        return True

    content = message.get("content")
    if not isinstance(content, list):
        return False

    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in _NON_TEXT_CONTENT_TYPES:
            return True
        if part_type not in _TEXT_CONTENT_TYPES and part_type is not None:
            return True
    return False


def _convert_text_only_content_lists_to_string(
    messages: List[AllMessageValues],
) -> List[AllMessageValues]:
    """Flatten text-only content lists; preserve tool/media message structure."""
    converted: List[AllMessageValues] = []
    for message in messages:
        if _message_has_non_text_content_blocks(message):
            converted.append(message)
            continue

        msg_copy = dict(message)
        texts = convert_content_list_to_str(msg_copy)
        if texts:
            msg_copy["content"] = texts
        converted.append(cast(AllMessageValues, msg_copy))
    return converted


def _json_dumps_tool_arguments(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), default=str)


def _tool_result_output_to_responses_format(content: Any) -> List[Dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if isinstance(content, list):
        output: List[Dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                output.append({"type": "input_text", "text": part})
            elif isinstance(part, dict):
                part_type = part.get("type")
                if part_type in _TEXT_CONTENT_TYPES or "text" in part:
                    output.append(
                        {"type": "input_text", "text": str(part.get("text", ""))}
                    )
                else:
                    output.append(
                        {
                            "type": "input_text",
                            "text": json.dumps(part, default=str),
                        }
                    )
            else:
                output.append({"type": "input_text", "text": str(part)})
        return output
    return [{"type": "input_text", "text": str(content)}]


def _content_part_to_responses_text(part: Any, role: Optional[str]) -> Dict[str, Any]:
    response_type = "output_text" if role == "assistant" else "input_text"
    if isinstance(part, str):
        return {"type": response_type, "text": part}
    if not isinstance(part, dict):
        return {"type": response_type, "text": str(part)}

    converted = dict(part)
    converted["type"] = response_type
    converted["text"] = str(part.get("text", ""))
    return converted


def _message_contains_anthropic_tool_blocks(message: Dict[str, Any]) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, dict) and part.get("type") in ("tool_use", "tool_result")
        for part in content
    )


def _convert_anthropic_tool_blocks_to_responses_input(
    input: List[Any],
) -> List[Any]:
    """Convert Anthropic tool_use/tool_result history into Responses input items."""
    converted_input: List[Any] = []

    for message in input:
        if not isinstance(message, dict):
            converted_input.append(message)
            continue
        if not _message_contains_anthropic_tool_blocks(message):
            converted_input.append(message)
            continue

        role = cast(Optional[str], message.get("role"))
        content = cast(List[Any], message.get("content"))
        pending_text_parts: List[Dict[str, Any]] = []

        def flush_text_parts() -> None:
            if pending_text_parts:
                converted_input.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": list(pending_text_parts),
                    }
                )
                pending_text_parts.clear()

        for part in content:
            if not isinstance(part, dict):
                pending_text_parts.append(
                    _content_part_to_responses_text(part, role=role)
                )
                continue

            part_type = part.get("type")
            if part_type == "tool_use":
                flush_text_parts()
                tool_call_id = part.get("id") or part.get("tool_use_id") or ""
                converted_input.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call_id,
                        "name": part.get("name", ""),
                        "arguments": _json_dumps_tool_arguments(part.get("input")),
                    }
                )
            elif part_type == "tool_result":
                flush_text_parts()
                converted_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": part.get("tool_use_id") or part.get("id") or "",
                        "output": _tool_result_output_to_responses_format(
                            part.get("content")
                        ),
                    }
                )
            else:
                pending_text_parts.append(
                    _content_part_to_responses_text(part, role=role)
                )

        flush_text_parts()

    return converted_input


def create_config_class(provider: SimpleProviderConfig):
    """Generate config class dynamically from JSON configuration"""

    # Choose base class
    base_class: type = (
        OpenAIGPTConfig if provider.base_class == "openai_gpt" else OpenAILikeChatConfig
    )

    class JSONProviderConfig(base_class):  # type: ignore[valid-type,misc]
        @overload
        def _transform_messages(
            self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
        ) -> Coroutine[Any, Any, List[AllMessageValues]]: ...

        @overload
        def _transform_messages(
            self,
            messages: List[AllMessageValues],
            model: str,
            is_async: Literal[False] = False,
        ) -> List[AllMessageValues]: ...

        def _transform_messages(
            self, messages: List[AllMessageValues], model: str, is_async: bool = False
        ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
            """Transform messages based on special_handling config"""

            # Flatten text-only content lists; keep tool/media blocks intact.
            if provider.special_handling.get("convert_content_list_to_string"):
                messages = _convert_text_only_content_lists_to_string(messages)

            if is_async:
                return super()._transform_messages(
                    messages=messages, model=model, is_async=True
                )
            else:
                return super()._transform_messages(
                    messages=messages, model=model, is_async=False
                )

        def _get_openai_compatible_provider_info(
            self, api_base: Optional[str], api_key: Optional[str]
        ) -> Tuple[Optional[str], Optional[str]]:
            """Get API base and key from JSON config"""

            # Resolve base URL
            resolved_base = api_base
            if not resolved_base and provider.api_base_env:
                resolved_base = get_secret_str(provider.api_base_env)
            if not resolved_base:
                resolved_base = provider.base_url

            # Resolve API key
            resolved_key = api_key or get_secret_str(provider.api_key_env)

            return resolved_base, resolved_key

        def get_complete_url(
            self,
            api_base: Optional[str],
            api_key: Optional[str],
            model: str,
            optional_params: dict,
            litellm_params: dict,
            stream: Optional[bool] = None,
        ) -> str:
            """Build complete URL for the API endpoint"""
            if not api_base:
                api_base = provider.base_url

            if api_base is None:
                raise ValueError(f"api_base is required for provider {provider.slug}")

            if not api_base.endswith("/chat/completions"):
                api_base = f"{api_base}/chat/completions"

            return api_base

        def get_supported_openai_params(self, model: str) -> list:
            """Get supported OpenAI params, excluding tool-related params for models
            that don't support function calling."""
            from litellm.utils import supports_function_calling

            supported_params = super().get_supported_openai_params(model=model)

            _supports_fc = supports_function_calling(
                model=model, custom_llm_provider=provider.slug
            )

            if not _supports_fc:
                tool_params = [
                    "tools",
                    "tool_choice",
                    "function_call",
                    "functions",
                    "parallel_tool_calls",
                ]
                for param in tool_params:
                    if param in supported_params:
                        supported_params.remove(param)
                verbose_logger.debug(
                    f"Model {model} on provider {provider.slug} does not support "
                    f"function calling — removed tool-related params from supported params."
                )

            return supported_params

        def map_openai_params(
            self,
            non_default_params: dict,
            optional_params: dict,
            model: str,
            drop_params: bool,
        ) -> dict:
            """Apply parameter mappings and constraints"""

            supported_params = self.get_supported_openai_params(model)

            # Apply supported params
            for param, value in non_default_params.items():
                # Check parameter mappings first
                if param in provider.param_mappings:
                    optional_params[provider.param_mappings[param]] = value
                elif param in supported_params:
                    optional_params[param] = value

            # Apply temperature constraints if present
            if "temperature" in optional_params:
                temp = optional_params["temperature"]
                constraints = provider.constraints

                # Clamp to max
                if "temperature_max" in constraints:
                    temp = min(temp, constraints["temperature_max"])

                # Clamp to min
                if "temperature_min" in constraints:
                    temp = max(temp, constraints["temperature_min"])

                # Special case: temperature_min_with_n_gt_1
                if "temperature_min_with_n_gt_1" in constraints:
                    n = optional_params.get("n", 1)
                    if n > 1 and temp < constraints["temperature_min_with_n_gt_1"]:
                        temp = constraints["temperature_min_with_n_gt_1"]

                optional_params["temperature"] = temp

            return optional_params

        @property
        def custom_llm_provider(self) -> Optional[str]:
            return provider.slug

    return JSONProviderConfig


_responses_config_cache: dict = {}


def create_responses_config_class(provider: SimpleProviderConfig):
    """Generate a Responses API config class dynamically from JSON configuration.

    Parallel to create_config_class() but for /v1/responses endpoints.
    Classes are cached per provider slug to avoid regeneration on every request.
    """
    if provider.slug in _responses_config_cache:
        return _responses_config_cache[provider.slug]

    from litellm.llms.openai_like.responses.transformation import (
        OpenAILikeResponsesConfig,
    )
    from litellm.types.router import GenericLiteLLMParams

    class JSONProviderResponsesConfig(OpenAILikeResponsesConfig):
        @property
        def custom_llm_provider(self):  # type: ignore[override]
            return provider.slug

        def transform_responses_api_request(
            self,
            model: str,
            input: Union[str, List[AllMessageValues]],
            response_api_optional_request_params: Dict,
            litellm_params: GenericLiteLLMParams,
            headers: dict,
        ) -> Dict:
            if provider.special_handling.get(
                "convert_content_list_to_string"
            ) and isinstance(input, list):
                input = _convert_text_only_content_lists_to_string(input)
            if provider.special_handling.get(
                "responses_convert_anthropic_tool_blocks"
            ) and isinstance(input, list):
                input = _convert_anthropic_tool_blocks_to_responses_input(input)

            return super().transform_responses_api_request(
                model=model,
                input=input,
                response_api_optional_request_params=response_api_optional_request_params,
                litellm_params=litellm_params,
                headers=headers,
            )

        def map_openai_params(
            self,
            response_api_optional_params: dict,
            model: str,
            drop_params: bool,
        ) -> dict:
            params = super().map_openai_params(
                response_api_optional_params=response_api_optional_params,
                model=model,
                drop_params=drop_params,
            )
            for param in provider.special_handling.get(
                "responses_drop_unsupported_params", []
            ):
                params.pop(param, None)
            return params

        def validate_environment(
            self,
            headers: dict,
            model: str,
            litellm_params: Optional[GenericLiteLLMParams],
        ) -> dict:
            litellm_params = litellm_params or GenericLiteLLMParams()
            api_key = litellm_params.api_key or get_secret_str(provider.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            return headers

        def get_complete_url(
            self,
            api_base: Optional[str],
            litellm_params: dict,
        ) -> str:
            if not api_base:
                if provider.api_base_env:
                    api_base = get_secret_str(provider.api_base_env)
                if not api_base:
                    api_base = provider.base_url

            if api_base is None:
                raise ValueError(f"api_base is required for provider {provider.slug}")

            api_base = api_base.rstrip("/")
            return f"{api_base}/responses"

    _responses_config_cache[provider.slug] = JSONProviderResponsesConfig
    return JSONProviderResponsesConfig
