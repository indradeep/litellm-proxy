"""OCI IAM-signed OpenAI-compatible audio transcription adapter."""

from __future__ import annotations

import json
from typing import Optional, Union

import httpx

from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.audio_transcription.transformation import AudioTranscriptionRequestData
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.oci.common_utils import (
    get_oci_base_url,
    strip_oci_transport_params,
    validate_oci_openai_compatible_environment,
)
from litellm.llms.oci.transport import apost_signed_bytes, encode_multipart, post_signed_bytes
from litellm.llms.openai.transcriptions.gpt_transformation import OpenAIGPTAudioTranscriptionConfig
from litellm.types.utils import (
    FileTypes,
    TranscriptionResponse,
    TranscriptionUsageDurationObject,
    TranscriptionUsageInputTokenDetailsObject,
    TranscriptionUsageTokensObject,
)


class OCIAudioTranscriptionConfig(OpenAIGPTAudioTranscriptionConfig):
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
        if base.endswith("/openai/v1/audio/transcriptions"):
            return base
        if base.endswith("/openai/v1"):
            return f"{base}/audio/transcriptions"
        return f"{base}/openai/v1/audio/transcriptions"

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
        headers = validate_oci_openai_compatible_environment(headers, litellm_params, api_key)
        headers.pop("content-type", None)
        headers.pop("Content-Type", None)
        return headers

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        processed = process_audio_file(audio_file)
        params = strip_oci_transport_params(optional_params)
        params.pop("extra_headers", None)
        extra_body = strip_oci_transport_params(params.pop("extra_body", {}))
        form = {"model": model, **params}
        if isinstance(extra_body, dict):
            form.update(extra_body)
        files = {"file": (processed.filename, processed.file_content, processed.content_type)}
        return AudioTranscriptionRequestData(data=form, files=files)

    def transform_audio_transcription_response(self, raw_response: httpx.Response) -> TranscriptionResponse:
        """Normalize OCI's OpenAI-compatible response without passing usage to its custom constructor."""
        try:
            payload = raw_response.json()
        except json.JSONDecodeError:
            content_type = raw_response.headers.get("content-type", "").lower()
            if "application/json" in content_type:
                raise
            return TranscriptionResponse(text=raw_response.text)

        if not isinstance(payload, dict) or "text" not in payload:
            raise ValueError("Invalid OCI transcription response: expected a JSON object with a 'text' field")

        result = TranscriptionResponse(text=payload["text"])
        usage = payload.get("usage")
        if isinstance(usage, dict):
            if usage.get("type") == "duration" and isinstance(usage.get("seconds"), (int, float)):
                result.usage = TranscriptionUsageDurationObject(seconds=float(usage["seconds"]))
            elif all(isinstance(usage.get(key), int) for key in ("input_tokens", "output_tokens", "total_tokens")):
                details = usage.get("input_token_details")
                if isinstance(details, dict) and all(
                    isinstance(details.get(key), int) for key in ("audio_tokens", "text_tokens")
                ):
                    result.usage = TranscriptionUsageTokensObject(
                        type="tokens",
                        input_tokens=usage["input_tokens"],
                        output_tokens=usage["output_tokens"],
                        total_tokens=usage["total_tokens"],
                        input_token_details=TranscriptionUsageInputTokenDetailsObject(
                            audio_tokens=details["audio_tokens"], text_tokens=details["text_tokens"]
                        ),
                    )
            else:
                result._hidden_params["oci_usage"] = usage
        return result


def audio_transcriptions(
    *,
    model: str,
    audio_file: FileTypes,
    optional_params: dict,
    litellm_params: dict,
    logging_obj: LiteLLMLoggingObj,
    timeout: Union[float, httpx.Timeout],
    api_base: Optional[str],
    headers: dict,
    client: Optional[Union[HTTPHandler, AsyncHTTPHandler]],
    is_async: bool,
):
    config = OCIAudioTranscriptionConfig()
    headers = config.validate_environment(dict(headers), model, [], optional_params, litellm_params)
    url = config.get_complete_url(api_base, None, model, optional_params, litellm_params)
    transformed = config.transform_audio_transcription_request(model, audio_file, optional_params, litellm_params)
    form = dict(transformed.data) if isinstance(transformed.data, dict) else {}
    files = transformed.files or {}
    file_value = files.get("file")
    if file_value is None:
        raise ValueError("OCI transcription request did not contain an audio file")
    body, multipart_headers = encode_multipart(form, file_value, url)
    headers.update(multipart_headers)
    logging_obj.pre_call(
        input=None,
        api_key="",
        additional_args={"complete_input_dict": {**form, "file": "<binary>"}, "api_base": url, "headers": headers},
    )

    async def _async_call() -> TranscriptionResponse:
        response = await apost_signed_bytes(
            url=url,
            body=body,
            headers=headers,
            optional_params=litellm_params,
            timeout=timeout,
            client=client if isinstance(client, AsyncHTTPHandler) else None,
        )
        result = config.transform_audio_transcription_response(response)
        result._hidden_params.update({"model": model, "custom_llm_provider": "oci"})
        logging_obj.post_call(input=None, api_key="", original_response=result.model_dump())
        return result

    if is_async:
        return _async_call()
    response = post_signed_bytes(
        url=url,
        body=body,
        headers=headers,
        optional_params=litellm_params,
        timeout=timeout,
        client=client if isinstance(client, HTTPHandler) else None,
    )
    result = config.transform_audio_transcription_response(response)
    result._hidden_params.update({"model": model, "custom_llm_provider": "oci"})
    logging_obj.post_call(input=None, api_key="", original_response=result.model_dump())
    return result
