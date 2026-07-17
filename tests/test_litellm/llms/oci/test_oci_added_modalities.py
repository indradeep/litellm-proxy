import base64
from unittest.mock import MagicMock

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from litellm.llms.oci.audio_transcription.transformation import OCIAudioTranscriptionConfig
from litellm.llms.oci.common_utils import resolve_oci_credentials, sign_oci_request_bytes
from litellm.llms.oci.image_generation.transformation import OCIImageGenerationConfig
from litellm.llms.oci.rerank.transformation import OCIRerankConfig
from litellm.types.rerank import RerankResponse


def _write_profile(tmp_path, *, security_token=None):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_path = tmp_path / "oci.pem"
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    token_line = ""
    if security_token:
        token_path = tmp_path / "token"
        token_path.write_text(security_token)
        token_line = f"security_token_file={token_path}\n"
    config_path = tmp_path / "config"
    config_path.write_text(
        "[DEFAULT]\n"
        "user=ocid1.user.test\n"
        "fingerprint=aa:bb\n"
        "tenancy=ocid1.tenancy.test\n"
        "region=us-chicago-1\n"
        f"key_file={key_path}\n"
        f"{token_line}"
    )
    return config_path


def test_oci_profile_is_source_for_credentials(tmp_path):
    config_path = _write_profile(tmp_path)
    credentials = resolve_oci_credentials({"oci_config_file": str(config_path)})
    assert credentials["oci_user"] == "ocid1.user.test"
    assert credentials["oci_tenancy"] == "ocid1.tenancy.test"
    assert credentials["oci_region"] == "us-chicago-1"
    assert credentials["oci_compartment_id"] is None


def test_oci_security_token_profile_uses_session_key_id(tmp_path):
    token = base64.urlsafe_b64encode(b"session-token").decode()
    config_path = _write_profile(tmp_path, security_token=token)
    headers, body = sign_oci_request_bytes(
        {"content-type": "application/json"},
        {"oci_config_file": str(config_path)},
        b"{}",
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/openai/v1/images/generations",
    )
    assert body == b"{}"
    assert f'keyId="ST${token}"' in headers["authorization"]


def test_oci_openai_compatible_modality_urls():
    params = {"oci_region": "us-chicago-1"}
    image = OCIImageGenerationConfig()
    audio = OCIAudioTranscriptionConfig()
    assert image.get_complete_url(None, None, "openai.gpt-image-1.5", {}, params).endswith(
        "/openai/v1/images/generations"
    )
    assert audio.get_complete_url(None, None, "openai.gpt-4o-transcribe", {}, params).endswith(
        "/openai/v1/audio/transcriptions"
    )


def test_oci_openai_compatible_requests_do_not_leak_transport_params():
    image = OCIImageGenerationConfig()
    image_request = image.transform_image_generation_request(
        "openai.gpt-image-1.5",
        "draw a square",
        {
            "size": "1024x1024",
            "oci_config_file": "/session/config",
            "oci_config_profile": "DEFAULT",
            "oci_region": "us-chicago-1",
            "oci_compartment_id": "ocid1.compartment.oc1..test",
            "extra_headers": {"x-test": "value"},
            "extra_body": {"quality": "high", "oci_signer": "must-not-be-sent"},
        },
        {},
        {},
    )
    assert image_request == {
        "model": "openai.gpt-image-1.5",
        "prompt": "draw a square",
        "size": "1024x1024",
        "quality": "high",
    }

    audio_request = OCIAudioTranscriptionConfig().transform_audio_transcription_request(
        "openai.gpt-4o-transcribe",
        ("sample.wav", b"audio", "audio/wav"),
        {"language": "en", "oci_config_file": "/session/config", "oci_compartment_id": "test"},
        {},
    )
    assert audio_request.data == {"model": "openai.gpt-4o-transcribe", "language": "en"}


@pytest.mark.parametrize("config_class", [OCIImageGenerationConfig, OCIAudioTranscriptionConfig])
def test_oci_openai_compatible_modalities_add_resolved_compartment_header(config_class):
    config = config_class()
    headers = config.validate_environment(
        {},
        "openai.gpt-4o-transcribe",
        [],
        {},
        {"oci_compartment_id": "ocid1.compartment.oc1..from-session-config"},
    )
    assert headers["opc-compartment-id"] == "ocid1.compartment.oc1..from-session-config"


def test_oci_openai_compatible_modality_preserves_explicit_compartment_header():
    headers = OCIImageGenerationConfig().validate_environment(
        {"OPC-Compartment-ID": "ocid1.compartment.oc1..explicit"},
        "openai.gpt-image-1.5",
        [],
        {},
        {"oci_compartment_id": "ocid1.compartment.oc1..from-session-config"},
    )
    assert headers["OPC-Compartment-ID"] == "ocid1.compartment.oc1..explicit"
    assert "opc-compartment-id" not in headers


def test_oci_openai_compatible_modality_requires_compartment_id(monkeypatch):
    monkeypatch.delenv("OCI_COMPARTMENT_ID", raising=False)
    with pytest.raises(Exception, match="oci_compartment_id"):
        OCIImageGenerationConfig().validate_environment({}, "openai.gpt-image-1.5", [], {}, {})


def test_oci_transcription_response_maps_openai_usage_without_constructor_error():
    response = OCIAudioTranscriptionConfig().transform_audio_transcription_response(
        httpx.Response(
            200,
            json={
                "text": "hello from OCI",
                "usage": {
                    "type": "tokens",
                    "input_tokens": 12,
                    "output_tokens": 3,
                    "total_tokens": 15,
                    "input_token_details": {"audio_tokens": 10, "text_tokens": 2},
                },
            },
        )
    )
    assert response.text == "hello from OCI"
    assert response.usage.total_tokens == 15


def test_oci_rerank_request_and_response_mapping(monkeypatch):
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.test")
    config = OCIRerankConfig()
    request = config.transform_rerank_request(
        "cohere.rerank-v4.0-fast",
        {"query": "q", "documents": ["a", "b"], "top_n": 1, "return_documents": True},
        {},
        {"oci_region": "us-chicago-1"},
    )
    assert request["input"] == "q"
    assert request["servingMode"]["modelId"] == "cohere.rerank-v4.0-fast"
    response = httpx.Response(
        200,
        json={
            "id": "rank-1",
            "modelId": "cohere.rerank-v4.0-fast",
            "documentRanks": [{"index": 1, "relevanceScore": 0.9, "document": "b"}],
            "usage": {"totalTokens": 7},
        },
    )
    result = config.transform_rerank_response(
        "cohere.rerank-v4.0-fast",
        response,
        RerankResponse(),
        MagicMock(),
        request_data=request,
    )
    assert result.results == [{"index": 1, "relevance_score": 0.9, "document": {"text": "b"}}]
    assert result.meta == {"billed_units": {"total_tokens": 7}}
