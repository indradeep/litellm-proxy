"""Provider-local authentication and request normalization for Oracle Code Assist."""

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
_VALID_EFFORT_LEVELS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
_VALID_SERVICE_TIERS = frozenset({"auto", "default", "flex", "priority"})
_DEFAULT_REASONING_EFFORT = "xhigh"
_DEFAULT_SERVICE_TIER = "priority"


def _normalize_effort_value(value: Union[str, dict, None]) -> Union[str, dict, None]:
    if value is None:
        return None
    if isinstance(value, dict):
        if "effort" not in value:
            return value
        return {**value, "effort": _normalize_effort_value(value["effort"])}
    if not isinstance(value, str):
        return value
    key = value.strip().lower().replace("_", " ")
    return _EFFORT_ALIASES.get(key, "xhigh" if key.replace(" ", "") == "extrahigh" else value)


def _coerce_effort_level(value: Union[str, dict, None]) -> Optional[str]:
    normalized = _normalize_effort_value(value)
    if isinstance(normalized, dict):
        normalized = normalized.get("effort")
    if isinstance(normalized, str) and normalized.strip().lower() in _VALID_EFFORT_LEVELS:
        return normalized.strip().lower()
    return None


def _coerce_service_tier(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    key = value.strip().lower().replace("-", "_")
    normalized = _SERVICE_TIER_ALIASES.get(key, key)
    return normalized if normalized in _VALID_SERVICE_TIERS else None


def strip_oca_unsupported_request_params(params: dict) -> None:
    """Remove fields rejected by OCA's streaming-only, zero-retention endpoint."""
    params.pop("stream_options", None)
    params.pop("previous_response_id", None)
    params["store"] = False


def normalize_oca_request_params(params: dict) -> None:
    """Normalize OCA aliases and apply provider defaults in place."""
    if "priority" in params and "service_tier" not in params:
        params["service_tier"] = params.pop("priority")

    reasoning = params.get("reasoning")
    if isinstance(reasoning, dict):
        reasoning = dict(reasoning)
        reasoning["effort"] = _coerce_effort_level(reasoning.get("effort")) or _DEFAULT_REASONING_EFFORT
        params["reasoning"] = reasoning
    else:
        effort = _coerce_effort_level(params.pop("reasoning_effort", None))
        params["reasoning"] = {"effort": effort or _DEFAULT_REASONING_EFFORT}

    params["service_tier"] = _coerce_service_tier(params.get("service_tier")) or _DEFAULT_SERVICE_TIER


@dataclass
class _TokenCache:
    access_token: str = ""
    expires_at: int = 0


class OCATokenManager:
    """Thread-safe, in-process OAuth client-credentials token cache."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache = _TokenCache()

    def invalidate(self) -> None:
        with self._lock:
            self._cache = _TokenCache()

    def get_access_token(self, *, force_refresh: bool = False) -> str:
        with self._lock:
            now = int(time.time())
            if not force_refresh and self._cache.access_token and now < self._cache.expires_at - REFRESH_SKEW_SECONDS:
                return self._cache.access_token
            token, expires_at = _fetch_client_credentials_token()
            self._cache = _TokenCache(access_token=token, expires_at=expires_at)
            verbose_logger.debug("OCA OAuth token refreshed; expires_at=%s", expires_at)
            return token


_oca_token_manager = OCATokenManager()


def get_oca_access_token(*, force_refresh: bool = False) -> str:
    return _oca_token_manager.get_access_token(force_refresh=force_refresh)


def invalidate_oca_access_token() -> None:
    _oca_token_manager.invalidate()


def _fetch_client_credentials_token() -> tuple[str, int]:
    client_id = os.getenv("OCA_CLIENT_ID", "").strip()
    client_secret = os.getenv("OCA_CLIENT_SECRET", "").strip()
    token_url = os.getenv("OCA_TOKEN_URL", "").strip()
    scope = os.getenv("OCA_SCOPE", "").strip()
    auth_method = os.getenv("OCA_AUTH_METHOD", "basic").strip().lower() or "basic"
    if not client_id or not client_secret or not token_url:
        raise RuntimeError(
            "Missing OCA OAuth configuration: OCA_CLIENT_ID, OCA_CLIENT_SECRET, and OCA_TOKEN_URL are required"
        )

    form = {"grant_type": "client_credentials"}
    if scope:
        form["scope"] = scope
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    if auth_method == "basic":
        credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        headers["Authorization"] = f"Basic {credentials}"
    elif auth_method == "body":
        form.update({"client_id": client_id, "client_secret": client_secret})
    else:
        raise ValueError("OCA_AUTH_METHOD must be 'basic' or 'body'")

    request = urllib.request.Request(
        token_url,
        data=urllib.parse.urlencode(form).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(f"OCA token request failed: HTTP {error.code}: {body}") from error

    access_token = payload.get("access_token")
    if not access_token:
        raise RuntimeError("OCA token response did not contain access_token")
    expires_at = payload.get("expires_at") or int(time.time()) + int(payload.get("expires_in", 3600))
    return str(access_token), int(expires_at)


def apply_oca_auth_headers(headers: dict, model: str) -> None:
    token = get_oca_access_token()
    token_hash = hashlib.sha256(token.encode()).digest()[:4].hex()
    model_hash = hashlib.sha256(model.encode()).digest()[:4].hex()
    request_id = f"{token_hash}{model_hash}{int(time.time()):08x}{secrets.token_hex(4)}"
    headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "client": os.getenv("OCA_CLIENT_NAME", "litellm-proxy"),
            "client-version": os.getenv("OCA_CLIENT_VERSION", "0.1.0"),
            "client-ide": os.getenv("OCA_CLIENT_IDE", "litellm"),
            "client-ide-version": os.getenv("OCA_CLIENT_IDE_VERSION", "n/a"),
            "opc-request-id": request_id,
        }
    )
