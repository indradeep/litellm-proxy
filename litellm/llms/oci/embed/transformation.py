"""
OCI Generative AI Embedding Configuration

Supports embedding models available on Oracle Cloud Infrastructure Generative AI service.
Uses the same authentication mechanisms as OCI chat (manual signing or OCI SDK Signer).

Supported models:
- cohere.embed-english-v3.0
- cohere.embed-english-light-v3.0
- cohere.embed-multilingual-v3.0
- cohere.embed-multilingual-light-v3.0
- cohere.embed-english-image-v3.0
- cohere.embed-english-light-image-v3.0
- cohere.embed-multilingual-light-image-v3.0
- cohere.embed-v4.0

Reference: https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai-inference/latest/EmbedTextResult/EmbedText
"""

from typing import Any, Dict, List, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.llms.oci.chat.transformation import OCIChatConfig
from litellm.llms.oci.common_utils import OCIError
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage

# Input type mapping from OpenAI conventions to OCI/Cohere conventions
_INPUT_TYPE_MAP = {
    "search_document": "SEARCH_DOCUMENT",
    "search_query": "SEARCH_QUERY",
    "classification": "CLASSIFICATION",
    "clustering": "CLUSTERING",
}


class OCIEmbeddingConfig(BaseEmbeddingConfig):
    """
    Configuration for OCI Generative AI Embedding API.

    The OCI embedding endpoint uses the Cohere embed models hosted on OCI.
    Authentication is handled via OCI request signing (manual credentials or OCI SDK Signer).

    Usage:
        ```python
        import litellm

        response = litellm.embedding(
            model="oci/cohere.embed-english-v3.0",
            input=["Hello world", "Goodbye world"],
            oci_compartment_id="ocid1.compartment.oc1..xxx",
            oci_region="us-ashburn-1",
            oci_user="ocid1.user.oc1..xxx",
            oci_fingerprint="xx:xx:xx:xx",
            oci_tenancy="ocid1.tenancy.oc1..xxx",
            oci_key_file="~/.oci/key.pem",
        )
        ```
    """

    def __init__(self) -> None:
        # We reuse OCIChatConfig for signing logic
        self._chat_config = OCIChatConfig()

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base:
            return api_base

        oci_region = optional_params.get("oci_region", "us-ashburn-1")
        return f"https://inference.generativeai.{oci_region}.oci.oraclecloud.com/20231130/actions/embedText"

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "dimensions",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        # Note: OCI Cohere embed does not support custom dimensions natively,
        # but we pass it through in case future models support it
        if "dimensions" in non_default_params:
            optional_params["dimensions"] = non_default_params["dimensions"]
        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate OCI credentials for embedding requests.
        Supports both OCI SDK Signer and manual credential signing.
        """
        oci_signer = optional_params.get("oci_signer")
        oci_region = optional_params.get("oci_region", "us-ashburn-1")

        api_base = (
            api_base
            or f"https://inference.generativeai.{oci_region}.oci.oraclecloud.com"
        )

        if oci_signer is None:
            oci_user = optional_params.get("oci_user")
            oci_fingerprint = optional_params.get("oci_fingerprint")
            oci_tenancy = optional_params.get("oci_tenancy")
            oci_key = optional_params.get("oci_key")
            oci_key_file = optional_params.get("oci_key_file")
            oci_compartment_id = optional_params.get("oci_compartment_id")

            if (
                not oci_user
                or not oci_fingerprint
                or not oci_tenancy
                or not (oci_key or oci_key_file)
                or not oci_compartment_id
            ):
                raise Exception(
                    "Missing required parameters: oci_user, oci_fingerprint, oci_tenancy, oci_compartment_id "
                    "and at least one of oci_key or oci_key_file. "
                    "Alternatively, provide an oci_signer object from the OCI SDK."
                )

        from litellm.llms.custom_httpx.http_handler import version

        headers.update(
            {
                "content-type": "application/json",
                "user-agent": f"litellm/{version}",
            }
        )

        return headers

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ):
        """
        Sign request using compact JSON serialization.

        The base HTTP handler's aembedding() method sends the body via
        httpx's ``json=`` parameter, which serializes with compact separators
        (no spaces: ``separators=(',', ':')``). The OCI signature includes a
        SHA-256 hash of the body, so the signing step must hash the *exact*
        bytes that will be sent on the wire. We achieve this by pre-serializing
        ``request_data`` with compact separators, then passing the resulting
        dict (which preserves key order) to the parent signer. The parent
        signer will call ``json.dumps()`` again with default separators, but
        since we intercept both manual and SDK paths, we directly compute the
        hash from the compact encoding.
        """
        import json as _json
        import hashlib as _hashlib
        import base64 as _base64
        from litellm.llms.oci.chat.transformation import (
            sha256_base64,
            build_signature_string,
            load_private_key_from_str,
            load_private_key_from_file,
            OCIRequestWrapper,
        )
        import datetime as _datetime
        from urllib.parse import urlparse as _urlparse

        # Compact body — matches what httpx sends via json= parameter
        compact_body = _json.dumps(request_data, separators=(',', ':')).encode("utf-8")

        oci_signer = optional_params.get("oci_signer")

        if oci_signer is not None:
            # OCI SDK Signer path
            method = str(optional_params.get("method", "POST")).upper()
            prepared_headers = headers.copy()
            prepared_headers.setdefault("content-type", "application/json")
            prepared_headers.setdefault("content-length", str(len(compact_body)))

            request_wrapper = OCIRequestWrapper(
                method=method, url=api_base, headers=prepared_headers, body=compact_body
            )

            from litellm.llms.oci.common_utils import OCIError
            try:
                oci_signer.do_request_sign(request_wrapper, enforce_content_headers=True)
            except Exception as e:
                raise OCIError(
                    status_code=500,
                    message=f"Failed to sign embedding request with oci_signer: {e}",
                ) from e

            headers.update(request_wrapper.headers)
            return headers, compact_body

        # Manual credential signing path
        oci_region = optional_params.get("oci_region", "us-ashburn-1")
        oci_user = optional_params.get("oci_user")
        oci_fingerprint = optional_params.get("oci_fingerprint")
        oci_tenancy = optional_params.get("oci_tenancy")
        oci_key = optional_params.get("oci_key")
        oci_key_file = optional_params.get("oci_key_file")

        method = str(optional_params.get("method", "POST")).upper()
        parsed = _urlparse(api_base)
        path = parsed.path or "/"
        host = parsed.netloc

        date = _datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
        content_type = headers.get("content-type", "application/json")
        content_length = str(len(compact_body))
        x_content_sha256 = sha256_base64(compact_body)

        headers_to_sign = {
            "date": date,
            "host": host,
            "content-type": content_type,
            "content-length": content_length,
            "x-content-sha256": x_content_sha256,
        }

        signed_headers_list = [
            "date", "(request-target)", "host",
            "content-length", "content-type", "x-content-sha256",
        ]
        signing_string = build_signature_string(
            method, path, headers_to_sign, signed_headers_list
        )

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        oci_key_content = None
        if oci_key and isinstance(oci_key, str):
            oci_key_content = oci_key.replace("\\n", "\n")
            if "\r\n" in oci_key_content:
                oci_key_content = oci_key_content.replace("\r\n", "\n")

        private_key = (
            load_private_key_from_str(oci_key_content)
            if oci_key_content
            else load_private_key_from_file(oci_key_file)
            if oci_key_file
            else None
        )

        if private_key is None:
            from litellm.llms.oci.common_utils import OCIError
            raise OCIError(
                status_code=400,
                message="Private key required for OCI embedding auth.",
            )

        signature = private_key.sign(
            signing_string.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        signature_b64 = _base64.b64encode(signature).decode()

        key_id = f"{oci_tenancy}/{oci_user}/{oci_fingerprint}"
        authorization = (
            'Signature version="1",'
            f'keyId="{key_id}",'
            'algorithm="rsa-sha256",'
            f'headers="{" ".join(signed_headers_list)}",'
            f'signature="{signature_b64}"'
        )

        headers.update({
            "authorization": authorization,
            "date": date,
            "host": host,
            "content-type": content_type,
            "content-length": content_length,
            "x-content-sha256": x_content_sha256,
        })

        return headers, None

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Transform the embedding request to OCI format.

        OCI embedText API expects:
        {
            "compartmentId": "...",
            "servingMode": {"servingType": "ON_DEMAND", "modelId": "..."},
            "inputs": ["text1", "text2"],
            "truncate": "END",
            "inputType": "SEARCH_DOCUMENT"
        }
        """
        oci_compartment_id = optional_params.get("oci_compartment_id")
        if not oci_compartment_id:
            raise Exception(
                "kwarg `oci_compartment_id` is required for OCI embedding requests"
            )

        # Build serving mode
        oci_serving_mode = optional_params.get("oci_serving_mode", "ON_DEMAND")
        if oci_serving_mode == "DEDICATED":
            oci_endpoint_id = optional_params.get("oci_endpoint_id", model)
            serving_mode = {
                "servingType": "DEDICATED",
                "endpointId": oci_endpoint_id,
            }
        else:
            serving_mode = {
                "servingType": "ON_DEMAND",
                "modelId": model,
            }

        # Normalize input to list of strings
        if isinstance(input, str):
            inputs = [input]
        elif isinstance(input, list):
            inputs = []
            for item in input:
                if isinstance(item, str):
                    inputs.append(item)
                elif isinstance(item, list):
                    raise ValueError(
                        "OCI embedding does not support token-array inputs. "
                        "Please convert token lists to strings before calling embedding()."
                    )
                else:
                    inputs.append(str(item))
        else:
            inputs = [str(input)]

        # Build request data — OCI embedText API expects inputs, truncate,
        # and inputType at the top level alongside compartmentId and servingMode
        request_data: Dict[str, Any] = {
            "compartmentId": oci_compartment_id,
            "servingMode": serving_mode,
            "inputs": inputs,
            "truncate": optional_params.get("truncate", "END"),
        }

        # Map input_type if provided
        input_type = optional_params.get("input_type")
        if input_type:
            mapped_type = _INPUT_TYPE_MAP.get(input_type.lower(), input_type.upper())
            request_data["inputType"] = mapped_type

        # Sign the request using the same URL the HTTP handler will POST to
        signing_url = self.get_complete_url(
            api_base=api_base,
            api_key=None,
            model=model,
            optional_params=optional_params,
            litellm_params={},
        )

        signed_headers, body = self.sign_request(
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=signing_url,
        )
        headers.update(signed_headers)

        return request_data

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> EmbeddingResponse:
        """
        Transform OCI embedding response to standard EmbeddingResponse format.

        OCI response format:
        {
            "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
            "modelId": "cohere.embed-english-v3.0",
            "modelVersion": "3.0",
            "inputTextTokenCounts": [5, 4]
        }
        """
        if raw_response.status_code != 200:
            raise OCIError(
                message=raw_response.text,
                status_code=raw_response.status_code,
            )

        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise OCIError(
                message=raw_response.text,
                status_code=raw_response.status_code,
            )

        embeddings = raw_response_json.get("embeddings", [])
        model_id = raw_response_json.get("modelId", model)

        # Build response data in OpenAI format
        embedding_data = []
        for idx, embedding in enumerate(embeddings):
            embedding_data.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": embedding,
                }
            )

        model_response.model = model_id
        model_response.data = embedding_data
        model_response.object = "list"

        # Calculate token usage
        input_token_counts = raw_response_json.get("inputTextTokenCounts", [])
        total_tokens = sum(input_token_counts) if input_token_counts else 0

        usage = Usage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens,
        )
        model_response.usage = usage

        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        return OCIError(
            message=error_message,
            status_code=status_code,
            headers=headers if isinstance(headers, httpx.Headers) else None,
        )
