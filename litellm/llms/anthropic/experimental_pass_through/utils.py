import os
from typing import Any, Dict, Optional, Union

import litellm
from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
)
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config

OpenAIReasoningParam = Union[str, Dict[str, Any]]


def is_reasoning_auto_summary_enabled() -> bool:
    """Check whether the default 'summary: detailed' injection is enabled (opt-in)."""
    return (
        litellm.reasoning_auto_summary
        or os.getenv("LITELLM_REASONING_AUTO_SUMMARY", "false").lower() == "true"
    )


def _map_anthropic_effort_to_openai(
    effort: str,
    model: str,
) -> Optional[str]:
    normalized_effort = effort.lower()
    if normalized_effort in {"low", "medium", "high"}:
        return normalized_effort
    if normalized_effort in {"max", "xhigh"}:
        if OpenAIGPT5Config._supports_reasoning_effort_level(model, "xhigh"):
            return "xhigh"
        return "high"
    return None


def _map_anthropic_thinking_to_openai(
    thinking: Optional[Dict[str, Any]],
) -> Optional[str]:
    if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
        return None

    budget_tokens = thinking.get("budget_tokens", 0) or 0
    if budget_tokens <= DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET:
        return "minimal"
    if budget_tokens <= DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET:
        return "low"
    if budget_tokens <= DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET:
        return "medium"
    if budget_tokens >= DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET:
        return "high"
    return "high"


def resolve_anthropic_reasoning_effort(
    *,
    output_config: Optional[Dict[str, Any]],
    thinking: Optional[Dict[str, Any]],
    model: str,
) -> Optional[str]:
    if isinstance(output_config, dict):
        effort = output_config.get("effort")
        if isinstance(effort, str):
            mapped_effort = _map_anthropic_effort_to_openai(effort, model)
            if mapped_effort is not None:
                return mapped_effort

    return _map_anthropic_thinking_to_openai(thinking)


def build_openai_reasoning_param(
    *,
    output_config: Optional[Dict[str, Any]],
    thinking: Optional[Dict[str, Any]],
    model: str,
    always_dict: bool,
) -> Optional[OpenAIReasoningParam]:
    reasoning_effort = resolve_anthropic_reasoning_effort(
        output_config=output_config,
        thinking=thinking,
        model=model,
    )
    if reasoning_effort is None:
        return None

    summary = thinking.get("summary") if isinstance(thinking, dict) else None
    auto_summary = is_reasoning_auto_summary_enabled()
    if summary:
        return {"effort": reasoning_effort, "summary": summary}
    if auto_summary or always_dict:
        result: Dict[str, Any] = {"effort": reasoning_effort}
        if auto_summary:
            result["summary"] = "detailed"
        return result
    return reasoning_effort
