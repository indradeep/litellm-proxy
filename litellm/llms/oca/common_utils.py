"""
OCA (Oracle Code Assist) — shared utilities.

Token reading, OCA header construction, and request detection helpers
used by both the chat and responses API configs.
"""

import hashlib
import os
import secrets
import time
from pathlib import Path
from typing import Optional

from litellm._logging import verbose_logger


def is_oca_request(model: str, api_base: Optional[str]) -> bool:
    """Return True when the request targets an OCA endpoint."""
    if model.startswith("oca/") or model.startswith("responses/oca/"):
        return True
    if api_base and "oraclecloud.com" in api_base and "aiservice" in api_base:
        return True
    return False


def read_oca_access_token(api_key: Optional[str]) -> Optional[str]:
    """Read the OCA bearer token from a file or environment variable.

    Resolution order:
    1. If *api_key* looks like a file path (contains ``/``), read from that file.
    2. ``OCA_ACCESS_TOKEN_FILE`` env var → read from that file.
    3. If ``OCA_ENABLE_API_KEY_FALLBACK`` is truthy, use ``OCA_API_KEY`` env var.
    """

    # 1. api_key might be the file path (set via config.yaml as os.environ/OCA_ACCESS_TOKEN_FILE)
    if api_key and api_key.strip():
        candidate = api_key.strip()
        # If it looks like a file path, read the file
        if "/" in candidate or candidate.startswith("~"):
            try:
                token_value = Path(candidate).expanduser().read_text(encoding="utf-8").strip()
                if token_value:
                    verbose_logger.debug("OCA token read from file: %s", candidate)
                    return token_value
            except OSError:
                verbose_logger.warning("OCA token file not readable: %s", candidate)
        else:
            # It's a literal token value
            return candidate

    # 2. Fall back to OCA_ACCESS_TOKEN_FILE env var
    token_file = os.getenv("OCA_ACCESS_TOKEN_FILE", "").strip()
    if token_file:
        try:
            token_value = Path(token_file).expanduser().read_text(encoding="utf-8").strip()
            if token_value:
                verbose_logger.debug("OCA token read from OCA_ACCESS_TOKEN_FILE: %s", token_file)
                return token_value
        except OSError:
            verbose_logger.warning("OCA_ACCESS_TOKEN_FILE not readable: %s", token_file)

    # 3. Optional API key fallback
    fallback_enabled = os.getenv("OCA_ENABLE_API_KEY_FALLBACK", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if fallback_enabled:
        fallback_token = os.getenv("OCA_API_KEY", "").strip()
        if fallback_token:
            verbose_logger.debug("OCA token from OCA_API_KEY fallback")
            return fallback_token

    return None


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
