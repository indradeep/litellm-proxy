from litellm.proxy.response_api_endpoints.endpoints import (
    _normalize_cursor_extra_model,
)


def test_cursor_extra_model_rewrites_model_and_effort():
    data = {
        "model": "gpt-5.5-extra",
        "reasoning": {"effort": "high", "summary": "auto"},
    }

    _normalize_cursor_extra_model(data)

    assert data["model"] == "gpt-5.5"
    assert data["reasoning"] == {"effort": "xhigh", "summary": "auto"}


def test_cursor_non_extra_model_keeps_reasoning_effort():
    data = {
        "model": "gpt-5.5",
        "reasoning": {"effort": "high", "summary": "auto"},
    }

    _normalize_cursor_extra_model(data)

    assert data["model"] == "gpt-5.5"
    assert data["reasoning"] == {"effort": "high", "summary": "auto"}
