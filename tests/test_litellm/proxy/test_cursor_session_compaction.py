import json

from litellm.proxy.response_api_endpoints.endpoints import _normalize_cursor_request_input
from litellm.proxy.response_api_endpoints.cursor_session import CursorSessionStore


def test_cursor_normalization_preserves_responses_fields():
    flat_tool = {"type": "function", "name": "lookup", "parameters": {}}
    data = {
        "model": "clip/claude-opus-4-8-high",
        "messages": [{"role": "user", "content": "hello"}],
        "input": [],
        "instructions": "Be brief.",
        "tools": [flat_tool],
        "reasoning": {"effort": "high"},
        "max_output_tokens": 32,
        "previous_response_id": "resp_previous",
    }

    _normalize_cursor_request_input(data)

    assert "messages" not in data
    assert data["input"] == [{"role": "user", "content": "hello"}]
    assert data["instructions"] == "Be brief."
    assert data["tools"] == [flat_tool]
    assert data["reasoning"] == {"effort": "high"}
    assert data["max_output_tokens"] == 32
    assert data["previous_response_id"] == "resp_previous"


def test_cursor_session_compacts_repeated_prefix_after_recorded_turn():
    store = CursorSessionStore(
        max_entries=8,
        ttl_seconds=3600,
        min_estimated_tokens=1,
        raw_tail_messages=1,
    )
    old_context = "important-old-context " * 500
    turn_one = {
        "model": "clip/claude-opus-4-8-high",
        "metadata": {"cursor_session_id": "session-1"},
        "input": [
            {"role": "user", "content": old_context},
            {"role": "assistant", "content": "I captured the context."},
        ],
    }

    decision_one = store.prepare_request(turn_one, headers={})

    assert decision_one.input_mode == "full"
    assert turn_one["input"][0]["content"] == old_context

    store.record_response(decision_one, assistant_text="Ready for the next step.")

    turn_two_input = turn_one["input"] + [
        {"role": "user", "content": "Now answer the follow-up."}
    ]
    turn_two = {
        "model": "clip/claude-opus-4-8-high",
        "metadata": {"cursor_session_id": "session-1"},
        "input": list(turn_two_input),
    }
    original_size = len(json.dumps(turn_two_input))

    decision_two = store.prepare_request(turn_two, headers={})

    assert decision_two.input_mode == "compacted"
    assert turn_two["input"][0]["role"] == "developer"
    assert "server-side compacted" in turn_two["input"][0]["content"]
    assert turn_two["input"][-1]["content"] == "Now answer the follow-up."
    assert len(json.dumps(turn_two["input"])) < original_size


def test_cursor_session_opt_out_preserves_full_input():
    store = CursorSessionStore(
        max_entries=8,
        ttl_seconds=3600,
        min_estimated_tokens=1,
        raw_tail_messages=1,
    )
    first = {
        "model": "clip/claude-opus-4-8-high",
        "metadata": {"cursor_session_id": "session-2"},
        "input": [{"role": "user", "content": "old context " * 100}],
    }
    decision = store.prepare_request(first, headers={})
    store.record_response(decision, assistant_text="ok")

    second_input = first["input"] + [{"role": "user", "content": "follow-up"}]
    second = {
        "model": "clip/claude-opus-4-8-high",
        "metadata": {
            "cursor_session_id": "session-2",
            "cursor_disable_compaction": True,
        },
        "input": list(second_input),
    }

    decision_two = store.prepare_request(second, headers={})

    assert decision_two.input_mode == "full"
    assert second["input"] == second_input


def test_cursor_session_requires_true_prefix_match():
    store = CursorSessionStore(
        max_entries=8,
        ttl_seconds=3600,
        min_estimated_tokens=1,
        raw_tail_messages=1,
    )
    first = {
        "model": "clip/claude-opus-4-8-high",
        "metadata": {"cursor_session_id": "session-3"},
        "input": [{"role": "user", "content": "first context " * 100}],
    }
    decision = store.prepare_request(first, headers={})
    store.record_response(decision, assistant_text="ok")

    divergent = {
        "model": "clip/claude-opus-4-8-high",
        "metadata": {"cursor_session_id": "session-3"},
        "input": [
            {"role": "user", "content": "different context"},
            {"role": "user", "content": "follow-up"},
        ],
    }

    decision_two = store.prepare_request(divergent, headers={})

    assert decision_two.input_mode == "full"
    assert divergent["input"][0]["content"] == "different context"
