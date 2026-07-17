"""OCI IAM-signed OpenAI-compatible image generation adapter."""

from __future__ import annotations

from typing import Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.oci.common_utils import (
    get_oci_base_url,
    strip_oci_transport_params,
    validate_oci_openai_compatible_environment,
)
from litellm.llms.oci.transport import apost_signed_bytes, encode_json, post_signed_bytes
from litellm.llms.openai.image_generation.gpt_transformation import GPTImageGenerationConfig
from litellm.types.utils import ImageResponse


class OCIImageGenerationConfig(GPTImageGenerationConfig):
    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base = get_oci_base_url(litellm_params, api_base).rstrip("/")
        if base.endswith("/openai/v1/images/generations"):
            return base
        if base.endswith("/openai/v1"):
            return f"{base}/images/generations"
        return f"{base}/openai/v1/images/generations"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return validate_oci_openai_compatible_environment(headers, litellm_params, api_key)

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        params = strip_oci_transport_params(optional_params)
        params.pop("extra_headers", None)
        extra_body = strip_oci_transport_params(params.pop("extra_body", {}))
        data = {"model": model, "prompt": prompt, **params}
        if isinstance(extra_body, dict):
            data.update(extra_body)
        return data


def image_generation(
    *,
    model: str,
    prompt: str,
    optional_params: dict,
    litellm_params: dict,
    headers: dict,
    logging_obj: LiteLLMLoggingObj,
    timeout: Union[float, httpx.Timeout],
    client: Optional[Union[HTTPHandler, AsyncHTTPHandler]],
    is_async: bool,
):
    config = OCIImageGenerationConfig()
    headers = config.validate_environment(dict(headers), model, [], optional_params, litellm_params)
    url = config.get_complete_url(litellm_params.get("api_base"), None, model, optional_params, litellm_params)
    data = config.transform_image_generation_request(model, prompt, optional_params, litellm_params, headers)
    logging_obj.pre_call(
        input=prompt,
        api_key="",
        additional_args={"complete_input_dict": data, "api_base": url, "headers": headers},
    )

    async def _async_call() -> ImageResponse:
        response = await apost_signed_bytes(
            url=url,
            body=encode_json(data),
            headers=headers,
            optional_params=litellm_params,
            timeout=timeout,
            client=client if isinstance(client, AsyncHTTPHandler) else None,
        )
        return config.transform_image_generation_response(
            model, response, ImageResponse(), logging_obj, data, optional_params, litellm_params, None
        )

    if is_async:
        return _async_call()
    response = post_signed_bytes(
        url=url,
        body=encode_json(data),
        headers=headers,
        optional_params=litellm_params,
        timeout=timeout,
        client=client if isinstance(client, HTTPHandler) else None,
    )
    return config.transform_image_generation_response(
        model, response, ImageResponse(), logging_obj, data, optional_params, litellm_params, None
    )
