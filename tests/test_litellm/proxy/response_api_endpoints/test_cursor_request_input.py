from litellm.proxy.response_api_endpoints.endpoints import (
    _normalize_cursor_request_input,
)


def test_cursor_request_input_prefers_messages_over_empty_input():
    data = {
        "model": "clip/claude-opus-4-8-xhigh",
        "messages": [{"role": "user", "content": "hello"}],
        "input": [],
    }

    _normalize_cursor_request_input(data)

    assert "messages" not in data
    assert data["input"] == [{"role": "user", "content": "hello"}]


def test_cursor_request_input_keeps_non_empty_input():
    data = {
        "model": "clip/gpt-5.5",
        "messages": [{"role": "user", "content": "ignored"}],
        "input": [{"role": "user", "content": "kept"}],
    }

    _normalize_cursor_request_input(data)

    assert "messages" not in data
    assert data["input"] == [{"role": "user", "content": "kept"}]


def test_cursor_request_input_converts_string_input():
    data = {
        "model": "clip/claude-opus-4-8-xhigh",
        "input": "hello",
    }

    _normalize_cursor_request_input(data)

    assert data["input"] == [{"role": "user", "content": "hello"}]
