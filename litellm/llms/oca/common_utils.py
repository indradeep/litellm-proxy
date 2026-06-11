"""
OCA (Oracle Code Assist) — shared utilities.

In-process OAuth client_credentials token management, OCA header construction,
and request detection helpers used by chat and responses API configs.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional, Union

from litellm._logging import verbose_logger

REFRESH_SKEW_SECONDS = 300

# UI / client aliases -> OpenAI API values for OCA GPT-5 models.
_EFFORT_ALIASES = {
    "extra high": "xhigh",
    "extra_high": "xhigh",
    "extra-high": "xhigh",
    "extrahigh": "xhigh",
}
_SERVICE_TIER_ALIASES = {
    "fast": "priority",
    "high": "priority",
    "on_demand_priority": "priority",
    "on-demand-priority": "priority",
}
_VALID_EFFORT_LEVELS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)
_VALID_SERVICE_TIERS = frozenset({"auto", "default", "flex", "priority"})
_DEFAULT_REASONING_EFFORT = "xhigh"
_DEFAULT_SERVICE_TIER = "priority"


def _normalize_effort_value(value: Union[str, dict, None]) -> Union[str, dict, None]:
    if value is None:
        return None
    if isinstance(value, dict):
        if "effort" not in value:
            return value
        normalized_effort = _normalize_effort_value(value["effort"])
        if normalized_effort == value["effort"]:
            return value
        return {**value, "effort": normalized_effort}
    if not isinstance(value, str):
        return value
    key = value.strip().lower().replace("_", " ")
    if key in _EFFORT_ALIASES:
        return _EFFORT_ALIASES[key]
    if key.replace(" ", "") == "extrahigh":
        return "xhigh"
    return value


def _coerce_effort_level(value: Union[str, dict, None]) -> Optional[str]:
    normalized = _normalize_effort_value(value)
    if normalized is None:
        return None
    if isinstance(normalized, dict):
        normalized = normalized.get("effort")
    if isinstance(normalized, str):
        level = normalized.strip().lower()
        if level in _VALID_EFFORT_LEVELS:
            return level
    return None


def _normalize_service_tier_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    key = value.strip().lower().replace("-", "_")
    return _SERVICE_TIER_ALIASES.get(key, value.strip().lower())


def _coerce_service_tier(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    normalized = _normalize_service_tier_value(value)
    if isinstance(normalized, str) and normalized in _VALID_SERVICE_TIERS:
        return normalized
    return None


def _apply_oca_effort_defaults(
    params: dict, *, prefer_reasoning_dict: bool
) -> None:
    if "reasoning" in params and isinstance(params["reasoning"], dict):
        reasoning = dict(params["reasoning"])
        effort = _coerce_effort_level(reasoning.get("effort"))
        reasoning["effort"] = effort or _DEFAULT_REASONING_EFFORT
        params["reasoning"] = reasoning
        return

    if "reasoning_effort" in params:
        effort = _coerce_effort_level(params["reasoning_effort"])
        params["reasoning_effort"] = effort or _DEFAULT_REASONING_EFFORT
        return

    if prefer_reasoning_dict:
        params["reasoning"] = {"effort": _DEFAULT_REASONING_EFFORT}
    else:
        params["reasoning_effort"] = _DEFAULT_REASONING_EFFORT


def _apply_oca_service_tier_defaults(params: dict) -> None:
    tier = _coerce_service_tier(params.get("service_tier"))
    params["service_tier"] = tier or _DEFAULT_SERVICE_TIER


def strip_oca_unsupported_request_params(params: dict) -> None:
    """Remove request fields OCA does not accept (e.g. stream_options.include_usage)."""
    params.pop("stream_options", None)
    # OCA Zero Data Retention orgs reject server-side response chaining.
    params.pop("previous_response_id", None)
    if params.get("store") is True:
        params["store"] = False


async def prepare_oca_zdr_responses_request(
    *,
    input: Union[str, Any],
    response_api_optional_params: dict,
) -> tuple[Any, dict]:
    """
    Prepare a Responses API request for OCA Zero Data Retention.

    OCA cannot accept ``previous_response_id``. When the client chains turns with
    that field, rebuild the full ``input`` from proxy spend logs and drop it.
    """
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )
    from litellm.responses.litellm_completion_transformation.session_handler import (
        ResponsesSessionHandler,
    )
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )
    from litellm.responses.utils import ResponsesAPIRequestUtils

    params = dict(response_api_optional_params)
    previous_response_id = params.get("previous_response_id")

    if previous_response_id:
        decoded_previous_response_id = (
            ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(
                previous_response_id
            )
        )
        session = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            previous_response_id=decoded_previous_response_id
        )
        session_messages = session.get("messages") or []
        current_messages = (
            LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                input=input,
                responses_api_request=params,
            )
        )
        combined_messages = session_messages + current_messages
        tools = params.get("tools") or []
        if tools:
            combined_messages = LiteLLMCompletionResponsesConfig._ensure_tool_results_have_corresponding_tool_calls(
                messages=combined_messages,
                tools=tools,
            )

        handler = LiteLLMResponsesTransformationHandler()
        expanded_input, extracted_instructions = (
            handler.convert_chat_completion_messages_to_responses_api(combined_messages)
        )
        input = expanded_input
        if extracted_instructions:
            existing_instructions = params.get("instructions")
            params["instructions"] = (
                f"{existing_instructions} {extracted_instructions}".strip()
                if existing_instructions
                else extracted_instructions
            )

        verbose_logger.debug(
            "OCA ZDR: expanded responses input from previous_response_id=%s (%d prior messages)",
            decoded_previous_response_id,
            len(session_messages),
        )

    strip_oca_unsupported_request_params(params)
    return input, params


def normalize_oca_request_params(
    params: dict,
    *,
    prefer_reasoning_dict: bool = False,
) -> None:
    """Normalize client/UI aliases and apply OCA defaults in-place."""
    if "priority" in params and "service_tier" not in params:
        params["service_tier"] = params.pop("priority")

    if "reasoning_effort" in params:
        params["reasoning_effort"] = _normalize_effort_value(
            params["reasoning_effort"]
        )

    if "reasoning" in params and isinstance(params["reasoning"], dict):
        params["reasoning"] = _normalize_effort_value(params["reasoning"])

    _apply_oca_effort_defaults(params, prefer_reasoning_dict=prefer_reasoning_dict)
    _apply_oca_service_tier_defaults(params)


def is_oca_request(model: str, api_base: Optional[str]) -> bool:
    """Return True when the request targets an OCA endpoint."""
    if model.startswith("oca/") or model.startswith("responses/oca/"):
        return True
    if api_base and "oraclecloud.com" in api_base and "aiservice" in api_base:
        return True
    return False


@dataclass
class _TokenCache:
    access_token: str = ""
    expires_at: int = 0


class OCATokenManager:
    """Thread-safe in-process OAuth token cache for OCA client_credentials."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache = _TokenCache()

    def invalidate(self) -> None:
        with self._lock:
            self._cache = _TokenCache()

    def get_access_token(self, *, force_refresh: bool = False) -> str:
        if not force_refresh:
            cached = self._get_cached_token()
            if cached:
                return cached
        return self._refresh_token()

    def _get_cached_token(self) -> Optional[str]:
        with self._lock:
            if not self._cache.access_token:
                return None
            if int(time.time()) >= (self._cache.expires_at - REFRESH_SKEW_SECONDS):
                return None
            return self._cache.access_token

    def _refresh_token(self) -> str:
        with self._lock:
            if self._cache.access_token and int(time.time()) < (
                self._cache.expires_at - REFRESH_SKEW_SECONDS
            ):
                return self._cache.access_token

            token, expires_at = _fetch_client_credentials_token()
            self._cache = _TokenCache(access_token=token, expires_at=expires_at)
            verbose_logger.debug("OCA OAuth token refreshed; expires_at=%s", expires_at)
            return token


_oca_token_manager = OCATokenManager()


def get_oca_access_token(*, force_refresh: bool = False) -> str:
    """Return a valid OCA bearer token, refreshing via client_credentials when needed."""
    return _oca_token_manager.get_access_token(force_refresh=force_refresh)


def invalidate_oca_access_token() -> None:
    """Drop cached OCA token (e.g. after a 401 from the OCA API)."""
    _oca_token_manager.invalidate()


def _fetch_client_credentials_token() -> tuple[str, int]:
    client_id = os.getenv("OCA_CLIENT_ID", "").strip()
    client_secret = os.getenv("OCA_CLIENT_SECRET", "").strip()
    token_url = os.getenv("OCA_TOKEN_URL", "").strip()
    scope = os.getenv("OCA_SCOPE", "").strip()
    auth_method = os.getenv("OCA_AUTH_METHOD", "basic").strip().lower() or "basic"

    if not client_id or not client_secret or not token_url:
        raise RuntimeError(
            "Missing OCA OAuth configuration. Set OCA_CLIENT_ID, "
            "OCA_CLIENT_SECRET, and OCA_TOKEN_URL."
        )

    form: dict[str, str] = {"grant_type": "client_credentials"}
    if scope:
        form["scope"] = scope

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }

    if auth_method == "basic":
        raw = f"{client_id}:{client_secret}".encode("utf-8")
        headers["Authorization"] = "Basic " + base64.b64encode(raw).decode("utf-8")
    elif auth_method == "body":
        form["client_id"] = client_id
        form["client_secret"] = client_secret
    else:
        raise ValueError("OCA_AUTH_METHOD must be 'basic' or 'body'")

    data = urllib.parse.urlencode(form).encode("utf-8")
    request = urllib.request.Request(
        token_url, data=data, headers=headers, method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OCA token request failed: HTTP {error.code}\n{body}"
        ) from error

    access_token = payload.get("access_token")
    if not access_token:
        raise RuntimeError(
            f"No access_token in OCA token response: {json.dumps(payload, indent=2)}"
        )

    expires_at = payload.get("expires_at")
    if not expires_at and payload.get("expires_in"):
        expires_at = int(time.time()) + int(payload["expires_in"])
    if not expires_at:
        expires_at = int(time.time()) + 3600

    return str(access_token), int(expires_at)


def add_oca_headers(headers: dict, model: str, token: str) -> None:
    """Add OCA-required custom headers to the request."""
    token_hash = hashlib.sha256(token.encode("utf-8")).digest()[:4].hex()
    model_hash = hashlib.sha256(model.encode("utf-8")).digest()[:4].hex()
    ts_hex = f"{int(time.time()):08x}"[-8:]
    rnd_hex = secrets.token_hex(4)
    opc_request_id = f"{token_hash}{model_hash}{ts_hex}{rnd_hex}"

    headers["client"] = os.getenv("OCA_CLIENT_NAME", "litellm-proxy")
    headers["client-version"] = os.getenv("OCA_CLIENT_VERSION", "0.1.0")
    headers["client-ide"] = os.getenv("OCA_CLIENT_IDE", "litellm")
    headers["client-ide-version"] = os.getenv("OCA_CLIENT_IDE_VERSION", "n/a")
    headers["opc-request-id"] = opc_request_id


def apply_oca_auth_headers(
    headers: dict, model: str, *, force_refresh: bool = False
) -> str:
    """Acquire token, set Authorization + OCA headers; return the token."""
    token = get_oca_access_token(force_refresh=force_refresh)
    headers["Authorization"] = f"Bearer {token}"
    add_oca_headers(headers=headers, model=model, token=token)
    if "content-type" not in headers and "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"
    return token


# OCA /responses reports upstream model ids like openai.gpt-5.5 (dot notation).
# Register them so spend-log model_map_information populates in the LiteLLM UI.
_OCA_UPSTREAM_MODEL_COSTS: dict[str, dict[str, object]] = {
    "openai.gpt-5.5": {
        "input_cost_per_token": 5e-06,
        "output_cost_per_token": 3e-05,
        "max_input_tokens": 1_050_000,
        "max_output_tokens": 128_000,
        "mode": "responses",
    },
    # OCA streams dated snapshot ids during in-progress events.
    "openai.gpt-5.5-2026-04-23": {
        "input_cost_per_token": 5e-06,
        "output_cost_per_token": 3e-05,
        "max_input_tokens": 1_050_000,
        "max_output_tokens": 128_000,
        "mode": "responses",
    },
    "openai.gpt-5.4": {
        "input_cost_per_token": 2.5e-06,
        "output_cost_per_token": 1.5e-05,
        "max_input_tokens": 1_050_000,
        "max_output_tokens": 128_000,
        "mode": "responses",
    },
    "openai.gpt-5.4-mini": {
        "input_cost_per_token": 7.5e-07,
        "output_cost_per_token": 4.5e-06,
        "max_input_tokens": 272_000,
        "max_output_tokens": 128_000,
        "mode": "responses",
    },
}


def register_oca_upstream_model_costs() -> None:
    """Register OCA upstream response model names in LiteLLM's cost map."""
    import litellm

    model_cost: dict[str, dict[str, object]] = {}
    for model_name, pricing in _OCA_UPSTREAM_MODEL_COSTS.items():
        entry = {"litellm_provider": "oca", **pricing}
        model_cost[model_name] = entry
        model_cost[f"oca/{model_name}"] = entry
        # Cursor / router aliases (gpt-5.5) and deployment names (oca/gpt-5.5).
        if model_name.startswith("openai."):
            short_name = model_name.split(".", 1)[1]
            model_cost[short_name] = entry
            model_cost[f"oca/{short_name}"] = entry
    litellm.register_model(model_cost=model_cost)


register_oca_upstream_model_costs()
