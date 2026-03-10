"""
OCI GenAI Embedding provider using the official OCI Python SDK.

Uses oci.generative_ai_inference.GenerativeAiInferenceClient.embed_text()
for authentication, request building, and API calls — bypassing manual
HTTP signing entirely.

OCI GenAI embedding models available:
- cohere.embed-english-v3.0
- cohere.embed-english-light-v3.0
- cohere.embed-multilingual-v3.0
- cohere.embed-multilingual-light-v3.0
- cohere.embed-4

Docs: https://docs.oracle.com/en-us/iaas/Content/generative-ai/embed-models.htm
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm import BaseEmbeddingConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage

from ..common_utils import OCIError


# Map of OpenAI input_type values to OCI inputType values
INPUT_TYPE_MAP = {
    "search_document": "SEARCH_DOCUMENT",
    "search_query": "SEARCH_QUERY",
    "classification": "CLASSIFICATION",
    "clustering": "CLUSTERING",
}

DEFAULT_INPUT_TYPE = "SEARCH_DOCUMENT"


class OCIEmbeddingConfig(BaseEmbeddingConfig):
    """
    Minimal config class for OCI GenAI embedding — required by LiteLLM's
    ProviderConfigManager registry.

    The actual embedding work is done by the oci_embed() function below,
    which uses the OCI Python SDK directly.
    """

    def get_supported_openai_params(self, model: str) -> List[str]:
        return ["encoding_format", "dimensions"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool = False,
    ) -> dict:
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
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        oci_region = optional_params.get("oci_region", "us-phoenix-1")
        return f"https://inference.generativeai.{oci_region}.oci.oraclecloud.com/20231130/actions/embedText"

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        return {}  # Not used — oci_embed() handles everything

    def transform_embedding_response(self, *args, **kwargs) -> EmbeddingResponse:
        return EmbeddingResponse()  # Not used — oci_embed() handles everything

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OCIError(status_code=status_code, message=error_message)


def oci_embed(
    model: str,
    input: Union[str, List[str]],
    optional_params: dict,
    logging_obj: Any,
    api_key: Optional[str] = None,
) -> EmbeddingResponse:
    """
    Call OCI GenAI embedText using the OCI Python SDK.

    This mirrors the pattern from Oracle's embed_text_demo.py:
    1. Build OCI config from optional_params credentials
    2. Create GenerativeAiInferenceClient
    3. Call embed_text() with EmbedTextDetails
    4. Transform response to OpenAI EmbeddingResponse format
    """
    try:
        import oci
        from oci.generative_ai_inference import GenerativeAiInferenceClient
        from oci.generative_ai_inference.models import (
            EmbedTextDetails,
            OnDemandServingMode,
            DedicatedServingMode,
        )
    except ImportError as e:
        raise ImportError(
            "OCI Python SDK is required for OCI embedding. "
            "Install with: pip install oci"
        ) from e

    # --- Extract credentials from optional_params ---
    oci_user = optional_params.get("oci_user")
    oci_fingerprint = optional_params.get("oci_fingerprint")
    oci_tenancy = optional_params.get("oci_tenancy")
    oci_key_file = optional_params.get("oci_key_file")
    oci_key = optional_params.get("oci_key")
    oci_region = optional_params.get("oci_region", "us-phoenix-1")
    oci_compartment_id = optional_params.get("oci_compartment_id")

    if not oci_compartment_id:
        raise Exception(
            "kwarg `oci_compartment_id` is required for OCI embedding requests"
        )

    # --- Build OCI config ---
    config: Dict[str, Any] = {
        "user": oci_user,
        "fingerprint": oci_fingerprint,
        "tenancy": oci_tenancy,
        "region": oci_region,
    }

    if oci_key_file:
        config["key_file"] = oci_key_file
    elif oci_key:
        config["key_content"] = oci_key
    else:
        raise Exception(
            "Missing required parameters: at least one of oci_key or oci_key_file "
            "must be provided."
        )

    # Validate required fields
    if not all([oci_user, oci_fingerprint, oci_tenancy]):
        raise Exception(
            "Missing required OCI credentials: oci_user, oci_fingerprint, oci_tenancy."
        )

    # --- Build the client ---
    endpoint = f"https://inference.generativeai.{oci_region}.oci.oraclecloud.com"
    client = GenerativeAiInferenceClient(
        config=config,
        service_endpoint=endpoint,
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240),
    )

    # --- Normalize input ---
    if isinstance(input, str):
        input_texts = [input]
    elif isinstance(input, list):
        if len(input) > 0 and isinstance(input[0], (list, int)):
            raise ValueError(
                "OCI GenAI embeddings only accept string inputs, not token arrays."
            )
        input_texts = [str(t) for t in input]
    else:
        input_texts = [str(input)]

    # --- Build serving mode ---
    oci_serving_mode = optional_params.get("oci_serving_mode", "ON_DEMAND")
    if oci_serving_mode == "DEDICATED":
        oci_endpoint_id = optional_params.get("oci_endpoint_id", model)
        serving_mode = DedicatedServingMode(endpoint_id=oci_endpoint_id)
    else:
        serving_mode = OnDemandServingMode(model_id=model)

    # --- Determine input type ---
    input_type = optional_params.get("input_type", DEFAULT_INPUT_TYPE)
    if input_type.lower() in INPUT_TYPE_MAP:
        input_type = INPUT_TYPE_MAP[input_type.lower()]

    truncate = optional_params.get("truncate", "NONE")

    # --- Build EmbedTextDetails ---
    embed_text_detail = EmbedTextDetails()
    embed_text_detail.serving_mode = serving_mode
    embed_text_detail.inputs = input_texts
    embed_text_detail.truncate = truncate
    embed_text_detail.compartment_id = oci_compartment_id
    embed_text_detail.input_type = input_type

    # --- LOGGING pre-call ---
    logging_obj.pre_call(
        input=input_texts,
        api_key=api_key,
        additional_args={
            "complete_input_dict": {
                "model": model,
                "inputs": input_texts,
                "input_type": input_type,
                "truncate": truncate,
                "compartment_id": oci_compartment_id,
            },
            "api_base": endpoint,
        },
    )

    # --- Call OCI SDK ---
    embed_response = client.embed_text(embed_text_detail)

    # --- Transform response to OpenAI format ---
    embeddings = embed_response.data.embeddings or []
    output_data = []
    for idx, embedding in enumerate(embeddings):
        output_data.append(
            {"object": "embedding", "index": idx, "embedding": embedding}
        )

    model_response = EmbeddingResponse()
    model_response.object = "list"
    model_response.data = output_data
    model_response.model = getattr(embed_response.data, "model_id", model)

    # --- Calculate usage (approximate — OCI doesn't return token counts) ---
    input_tokens = 0
    try:
        for text in input_texts:
            input_tokens += len(litellm.encoding.encode(text))
    except Exception:
        for text in input_texts:
            input_tokens += len(text) // 4

    setattr(
        model_response,
        "usage",
        Usage(
            prompt_tokens=input_tokens,
            completion_tokens=0,
            total_tokens=input_tokens,
        ),
    )

    # --- LOGGING post-call ---
    logging_obj.post_call(
        input=input_texts,
        api_key=api_key,
        additional_args={"complete_input_dict": {"inputs": input_texts}},
        original_response=str(embed_response.data),
    )

    return model_response
