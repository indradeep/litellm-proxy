"""Signed HTTP transport helpers shared by OCI-only modality handlers."""

from __future__ import annotations

import json
from typing import Any, Optional, Union

import httpx

from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.oci.common_utils import OCIError, sign_oci_request_bytes
from litellm.types.utils import LlmProviders


def _raise_for_oci_error(response: httpx.Response) -> None:
    if response.status_code >= 400:
        raise OCIError(status_code=response.status_code, message=response.text, headers=response.headers)


def encode_json(data: dict) -> bytes:
    """Use one stable encoding for both the OCI signature and wire body."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def encode_multipart(data: dict, file_value: Any, url: str) -> tuple[bytes, dict]:
    """Materialize multipart bytes so OCI signs the exact wire representation."""
    request = httpx.Request("POST", url, data=data, files={"file": file_value})
    body = request.read()
    headers = {
        "content-type": request.headers["content-type"],
        "content-length": str(len(body)),
    }
    return body, headers


def post_signed_bytes(
    *,
    url: str,
    body: bytes,
    headers: dict,
    optional_params: dict,
    timeout: Optional[Union[float, httpx.Timeout]],
    client: Optional[HTTPHandler] = None,
) -> httpx.Response:
    signed_headers, signed_body = sign_oci_request_bytes(headers, optional_params, body, url)
    http_client = client if isinstance(client, HTTPHandler) else _get_httpx_client()
    response = http_client.post(url=url, headers=signed_headers, data=signed_body, timeout=timeout)
    _raise_for_oci_error(response)
    return response


async def apost_signed_bytes(
    *,
    url: str,
    body: bytes,
    headers: dict,
    optional_params: dict,
    timeout: Optional[Union[float, httpx.Timeout]],
    client: Optional[AsyncHTTPHandler] = None,
) -> httpx.Response:
    http_client = (
        client if isinstance(client, AsyncHTTPHandler) else get_async_httpx_client(llm_provider=LlmProviders.OCI)
    )
    signed_headers, signed_body = sign_oci_request_bytes(headers, optional_params, body, url)
    response = await http_client.post(url=url, headers=signed_headers, data=signed_body, timeout=timeout)
    _raise_for_oci_error(response)
    return response
