"""
Unit tests for OCI GenAI embedding transformation.

Tests the OCIEmbeddingConfig class which handles:
- URL generation for embedText API
- Request transformation (OpenAI → OCI format)
- Response transformation (OCI → OpenAI format)
- Credential validation
"""

import os
import sys
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm

# Adds the parent directory to the system path
sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.oci.embed.transformation import OCIEmbeddingConfig
from litellm.types.utils import EmbeddingResponse


# ── Test Fixtures ──────────────────────────────────────────────────────────────

TEST_COMPARTMENT_ID = "ocid1.compartment.oc1..aaaatest"
TEST_REGION = "us-phoenix-1"

BASE_OCI_PARAMS = {
    "oci_user": "ocid1.user.oc1..aaaatest",
    "oci_fingerprint": "aa:bb:cc:dd",
    "oci_tenancy": "ocid1.tenancy.oc1..aaaatest",
    "oci_region": TEST_REGION,
    "oci_compartment_id": TEST_COMPARTMENT_ID,
}

TEST_OCI_PARAMS_KEY = {
    **BASE_OCI_PARAMS,
    "oci_key": "<private_key.pem as string>",
}

TEST_OCI_PARAMS_KEY_FILE = {
    **BASE_OCI_PARAMS,
    "oci_key_file": "<private_key.pem as a Path>",
}


@pytest.fixture(params=[TEST_OCI_PARAMS_KEY, TEST_OCI_PARAMS_KEY_FILE])
def supplied_params(request):
    return request.param


# ── Test Class ─────────────────────────────────────────────────────────────────

class TestOCIEmbeddingConfig:
    """Tests for OCI GenAI embedding configuration."""

    def test_get_complete_url(self):
        """Tests that the embedText URL is constructed correctly."""
        config = OCIEmbeddingConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="cohere.embed-english-v3.0",
            optional_params={"oci_region": "us-phoenix-1"},
            litellm_params={},
        )
        assert url == "https://inference.generativeai.us-phoenix-1.oci.oraclecloud.com/20231130/actions/embedText"

    def test_get_complete_url_default_region(self):
        """Tests URL with default region (us-ashburn-1)."""
        config = OCIEmbeddingConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="cohere.embed-english-v3.0",
            optional_params={},
            litellm_params={},
        )
        assert "us-ashburn-1" in url

    def test_validate_environment_with_credentials(self, supplied_params):
        """Tests that validate_environment accepts valid credentials."""
        config = OCIEmbeddingConfig()
        headers = config.validate_environment(
            headers={},
            model="cohere.embed-english-v3.0",
            messages=[],
            optional_params=supplied_params,
            litellm_params={},
        )
        assert headers["content-type"] == "application/json"
        assert "user-agent" in headers

    def test_validate_environment_missing_credentials(self):
        """Tests that missing credentials raise an error."""
        config = OCIEmbeddingConfig()
        with pytest.raises(Exception, match="Missing required parameters"):
            config.validate_environment(
                headers={},
                model="cohere.embed-english-v3.0",
                messages=[],
                optional_params={"oci_region": "us-ashburn-1"},
                litellm_params={},
            )

    def test_validate_environment_with_signer(self):
        """Tests that oci_signer bypasses credential validation."""
        class MockSigner:
            pass

        config = OCIEmbeddingConfig()
        headers = config.validate_environment(
            headers={},
            model="cohere.embed-english-v3.0",
            messages=[],
            optional_params={"oci_signer": MockSigner(), "oci_region": "us-ashburn-1"},
            litellm_params={},
        )
        assert headers["content-type"] == "application/json"

    def test_build_oci_request_body_on_demand(self):
        """Tests that the OCI request body is built correctly for ON_DEMAND mode."""
        config = OCIEmbeddingConfig()
        body = config._build_oci_request_body(
            model="cohere.embed-english-v3.0",
            input_texts=["Hello", "World"],
            optional_params={
                "oci_compartment_id": TEST_COMPARTMENT_ID,
            },
        )

        assert body["compartmentId"] == TEST_COMPARTMENT_ID
        assert body["servingMode"]["servingType"] == "ON_DEMAND"
        assert body["servingMode"]["modelId"] == "cohere.embed-english-v3.0"
        assert body["embedTextDetails"]["inputs"] == ["Hello", "World"]
        assert body["embedTextDetails"]["inputType"] == "SEARCH_DOCUMENT"
        assert body["embedTextDetails"]["truncate"] == "END"

    def test_build_oci_request_body_dedicated(self):
        """Tests DEDICATED serving mode with endpoint ID."""
        config = OCIEmbeddingConfig()
        body = config._build_oci_request_body(
            model="cohere.embed-english-v3.0",
            input_texts=["Test"],
            optional_params={
                "oci_compartment_id": TEST_COMPARTMENT_ID,
                "oci_serving_mode": "DEDICATED",
                "oci_endpoint_id": "ocid1.endpoint.oc1..test",
            },
        )

        assert body["servingMode"]["servingType"] == "DEDICATED"
        assert body["servingMode"]["endpointId"] == "ocid1.endpoint.oc1..test"

    def test_build_oci_request_body_missing_compartment(self):
        """Tests that missing compartment ID raises an error."""
        config = OCIEmbeddingConfig()
        with pytest.raises(Exception, match="oci_compartment_id"):
            config._build_oci_request_body(
                model="cohere.embed-english-v3.0",
                input_texts=["Test"],
                optional_params={},
            )

    def test_transform_embedding_response(self):
        """Tests response transformation from OCI to OpenAI format."""
        config = OCIEmbeddingConfig()

        # Mock OCI response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "embeddings": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
            "modelId": "cohere.embed-english-v3.0",
            "modelVersion": "3.0",
        }
        mock_response.status_code = 200

        model_response = EmbeddingResponse()
        logging_obj = MagicMock()
        logging_obj.model_call_details = {"input": ["Hello", "World"]}

        result = config.transform_embedding_response(
            model="cohere.embed-english-v3.0",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=None,
            request_data={
                "embedTextDetails": {"inputs": ["Hello", "World"]},
            },
            optional_params={},
            litellm_params={},
        )

        assert result.object == "list"
        assert len(result.data) == 2
        assert result.data[0]["object"] == "embedding"
        assert result.data[0]["index"] == 0
        assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert result.data[1]["embedding"] == [0.4, 0.5, 0.6]
        assert result.model == "cohere.embed-english-v3.0"

    def test_supported_openai_params(self):
        """Tests that supported params are returned."""
        config = OCIEmbeddingConfig()
        params = config.get_supported_openai_params(model="cohere.embed-english-v3.0")
        assert "encoding_format" in params
        assert "dimensions" in params

    def test_input_type_mapping(self):
        """Tests custom input_type mapping."""
        config = OCIEmbeddingConfig()
        body = config._build_oci_request_body(
            model="cohere.embed-english-v3.0",
            input_texts=["Query text"],
            optional_params={
                "oci_compartment_id": TEST_COMPARTMENT_ID,
                "input_type": "search_query",
            },
        )
        assert body["embedTextDetails"]["inputType"] == "SEARCH_QUERY"
