from unittest.mock import patch

from litellm.llms.oca.common_utils import normalize_oca_request_params, strip_oca_unsupported_request_params
from litellm.llms.oca.responses.transformation import OCAResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams


def test_oca_normalizes_defaults_and_strips_stateful_fields():
    params = {
        "reasoning": {"effort": "extra high"},
        "priority": "fast",
        "previous_response_id": "resp_previous",
        "stream_options": {"include_usage": True},
        "store": True,
    }
    normalize_oca_request_params(params)
    strip_oca_unsupported_request_params(params)
    assert params["reasoning"] == {"effort": "xhigh"}
    assert params["service_tier"] == "priority"
    assert params["store"] is False
    assert "previous_response_id" not in params
    assert "stream_options" not in params


def test_oca_responses_forces_stream_and_full_input_contract():
    config = OCAResponsesAPIConfig()
    params = {
        "stream": False,
        "previous_response_id": "resp_previous",
        "store": True,
    }
    with patch("litellm.llms.oca.responses.transformation.apply_oca_auth_headers"):
        headers = config.validate_environment({}, "gpt-5.4", GenericLiteLLMParams())
    data = config.transform_responses_api_request(
        model="gpt-5.4",
        input="hello",
        response_api_optional_request_params=params,
        litellm_params=GenericLiteLLMParams(),
        headers=headers,
    )
    assert data["stream"] is True
    assert data["store"] is False
    assert "previous_response_id" not in data


def test_oca_responses_url():
    config = OCAResponsesAPIConfig()
    assert config.get_complete_url("https://oca.example.test/root", {}) == "https://oca.example.test/root/responses"
