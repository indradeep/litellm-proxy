"""OCI Generative AI native rerankText adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.oci.common_utils import (
    OCI_API_VERSION,
    OCIError,
    get_oci_base_url,
    resolve_oci_credentials,
    validate_oci_environment,
)
from litellm.llms.oci.transport import apost_signed_bytes, encode_json, post_signed_bytes
from litellm.types.rerank import RerankResponse


class OCIRerankConfig(BaseRerankConfig):
    def get_complete_url(self, api_base: Optional[str], model: str, optional_params: Optional[dict] = None) -> str:
        base = get_oci_base_url(optional_params or {}, api_base)
        return f"{base}/{OCI_API_VERSION}/actions/rerankText"

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return [
            "query",
            "documents",
            "top_n",
            "return_documents",
            "max_chunks_per_doc",
            "max_tokens_per_doc",
        ]

    def map_cohere_rerank_params(
        self,
        non_default_params: dict,
        model: str,
        drop_params: bool,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        custom_llm_provider: Optional[str] = None,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
        max_tokens_per_doc: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> Dict:
        if rank_fields and not drop_params:
            raise OCIError(status_code=400, message="rank_fields is not supported by OCI rerankText")
        if instruction and not drop_params:
            raise OCIError(status_code=400, message="instruction is not supported by OCI rerankText")
        return {
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
            "max_chunks_per_doc": max_chunks_per_doc,
            "max_tokens_per_doc": max_tokens_per_doc,
            **non_default_params,
        }

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        optional_params: Optional[dict] = None,
    ) -> dict:
        return validate_oci_environment(headers, optional_params or {}, api_key)

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Dict,
        headers: dict,
        litellm_params: Optional[dict] = None,
    ) -> dict:
        credentials = resolve_oci_credentials(litellm_params or optional_rerank_params)
        compartment_id = credentials.get("oci_compartment_id")
        if not compartment_id:
            raise OCIError(status_code=400, message="OCI compartment ID is required for rerankText")

        serving_type = str(optional_rerank_params.get("oci_serving_mode", "ON_DEMAND")).upper()
        if serving_type == "DEDICATED":
            serving_mode = {
                "servingType": "DEDICATED",
                "endpointId": optional_rerank_params.get("oci_endpoint_id") or model,
            }
        elif serving_type == "ON_DEMAND":
            serving_mode = {"servingType": "ON_DEMAND", "modelId": model}
        else:
            raise OCIError(status_code=400, message="oci_serving_mode must be ON_DEMAND or DEDICATED")

        data = {
            "compartmentId": compartment_id,
            "servingMode": serving_mode,
            "input": optional_rerank_params["query"],
            "documents": optional_rerank_params["documents"],
            "topN": optional_rerank_params.get("top_n"),
            "isEcho": optional_rerank_params.get("return_documents", True),
            "maxChunksPerDocument": optional_rerank_params.get("max_chunks_per_doc"),
            "maxTokensPerDocument": optional_rerank_params.get("max_tokens_per_doc"),
        }
        return {key: value for key, value in data.items() if value is not None}

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> RerankResponse:
        payload = raw_response.json()
        results = []
        for rank in payload.get("documentRanks", []):
            result = {
                "index": rank["index"],
                "relevance_score": rank["relevanceScore"],
            }
            document = rank.get("document")
            if isinstance(document, str):
                result["document"] = {"text": document}
            elif isinstance(document, dict):
                result["document"] = {"text": document.get("text", str(document))}
            results.append(result)

        usage = payload.get("usage") or {}
        meta = None
        if usage:
            meta = {"billed_units": {"total_tokens": usage.get("totalTokens")}}
        response = RerankResponse(id=payload.get("id"), results=results, meta=meta)
        response._hidden_params.update({"model": payload.get("modelId", model), "custom_llm_provider": "oci"})
        logging_obj.post_call(input=request_data.get("input"), api_key="", original_response=payload)
        return response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        return OCIError(status_code=status_code, message=error_message)


def rerank(
    *,
    model: str,
    provider_config: OCIRerankConfig,
    optional_rerank_params: dict,
    logging_obj: LiteLLMLoggingObj,
    timeout: Optional[Union[float, httpx.Timeout]],
    api_base: Optional[str],
    headers: dict,
    litellm_params: dict,
    client: Optional[Union[HTTPHandler, AsyncHTTPHandler]],
    is_async: bool,
):
    request_headers = provider_config.validate_environment(headers, model, optional_params=litellm_params)
    url = provider_config.get_complete_url(api_base, model, litellm_params)
    data = provider_config.transform_rerank_request(model, optional_rerank_params, request_headers, litellm_params)
    logging_obj.pre_call(
        input=optional_rerank_params.get("query", ""),
        api_key="",
        additional_args={"complete_input_dict": data, "api_base": url, "headers": request_headers},
    )

    async def _async_call() -> RerankResponse:
        response = await apost_signed_bytes(
            url=url,
            body=encode_json(data),
            headers=request_headers,
            optional_params=litellm_params,
            timeout=timeout,
            client=client if isinstance(client, AsyncHTTPHandler) else None,
        )
        return provider_config.transform_rerank_response(
            model, response, RerankResponse(), logging_obj, request_data=data
        )

    if is_async:
        return _async_call()
    response = post_signed_bytes(
        url=url,
        body=encode_json(data),
        headers=request_headers,
        optional_params=litellm_params,
        timeout=timeout,
        client=client if isinstance(client, HTTPHandler) else None,
    )
    return provider_config.transform_rerank_response(model, response, RerankResponse(), logging_obj, request_data=data)
