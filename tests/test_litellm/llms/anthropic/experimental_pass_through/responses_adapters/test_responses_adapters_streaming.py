"""
Tests for AnthropicResponsesStreamWrapper
(litellm/llms/anthropic/experimental_pass_through/responses_adapters/streaming_iterator.py)

Validates the Responses API → Anthropic SSE event translation, with a focus on
ensuring exactly one message_start is emitted per stream.
"""

import asyncio
import os
import sys
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../../../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesStreamWrapper,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(event_type: str, **kwargs) -> MagicMock:
    """Create a mock Responses API event with the given type and attributes."""
    event = MagicMock()
    event.type = event_type
    for key, value in kwargs.items():
        setattr(event, key, value)
    return event


def _make_item(item_type: str, item_id: str = "item_001", **kwargs) -> MagicMock:
    """Create a mock output item."""
    item = MagicMock()
    item.type = item_type
    item.id = item_id
    for key, value in kwargs.items():
        setattr(item, key, value)
    return item


async def _aiter_from_list(events: list) -> AsyncIterator:
    """Turn a plain list into an async iterator."""
    for event in events:
        yield event


async def _collect_all_chunks(wrapper: AnthropicResponsesStreamWrapper) -> List[Dict[str, Any]]:
    """Drain all chunks from the wrapper's __anext__."""
    chunks = []
    async for chunk in wrapper:
        chunks.append(chunk)
    return chunks


def _make_completed_response(
    status: str = "completed",
    input_tokens: int = 10,
    output_tokens: int = 20,
    has_function_call: bool = False,
) -> MagicMock:
    """Create a mock response object for response.completed events."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 0

    resp = MagicMock()
    resp.status = status
    resp.usage = usage

    if has_function_call:
        fc = MagicMock()
        fc.type = "function_call"
        resp.output = [fc]
    else:
        resp.output = []

    return resp


def _standard_text_events():
    """A standard set of events for a text response."""
    return [
        _make_event("response.created"),
        _make_event(
            "response.output_item.added",
            item=_make_item("message", item_id="msg_01"),
        ),
        _make_event(
            "response.output_text.delta",
            item_id="msg_01",
            delta="Hello!",
        ),
        _make_event(
            "response.output_item.done",
            item=_make_item("message", item_id="msg_01"),
        ),
        _make_event("response.completed", response=_make_completed_response()),
    ]


# ---------------------------------------------------------------------------
# Tests — No duplicate message_start (the critical bug)
# ---------------------------------------------------------------------------


class TestNoDoubleMessageStart:
    """The critical bug: only one message_start should ever be emitted."""

    def test_response_created_does_not_duplicate_message_start(self):
        """
        When __anext__ fallback fires before response.created arrives,
        there must still be exactly one message_start in the output.
        """
        async def _run():
            wrapper = AnthropicResponsesStreamWrapper(
                responses_stream=_aiter_from_list(_standard_text_events()),
                model="gpt-5.3-codex",
            )
            return await _collect_all_chunks(wrapper)

        chunks = asyncio.run(_run())
        message_start_count = sum(1 for c in chunks if c.get("type") == "message_start")
        assert message_start_count == 1, (
            f"Expected exactly 1 message_start, got {message_start_count}. "
            f"Event types: {[c.get('type') for c in chunks]}"
        )

    def test_message_stop_follows_message_start(self):
        """message_stop must come after message_start, never before a second message_start."""
        async def _run():
            wrapper = AnthropicResponsesStreamWrapper(
                responses_stream=_aiter_from_list(_standard_text_events()),
                model="gpt-5.3-codex",
            )
            return await _collect_all_chunks(wrapper)

        chunks = asyncio.run(_run())
        types = [c.get("type") for c in chunks]
        assert types[0] == "message_start"
        assert types[-1] == "message_stop"
        assert types.count("message_start") == 1
        assert types.count("message_stop") == 1


# ---------------------------------------------------------------------------
# Tests — Normal stream flow
# ---------------------------------------------------------------------------


class TestNormalStreamFlow:
    """Verify the happy-path event translation."""

    def test_text_delta_produces_content_block_delta(self):
        async def _run():
            wrapper = AnthropicResponsesStreamWrapper(
                responses_stream=_aiter_from_list(_standard_text_events()),
                model="test-model",
            )
            return await _collect_all_chunks(wrapper)

        chunks = asyncio.run(_run())
        types = [c.get("type") for c in chunks]

        assert "content_block_start" in types
        assert "content_block_delta" in types
        assert "content_block_stop" in types

        deltas = [c for c in chunks if c.get("type") == "content_block_delta"]
        assert len(deltas) == 1
        assert deltas[0]["delta"]["type"] == "text_delta"
        assert deltas[0]["delta"]["text"] == "Hello!"

    def test_function_call_produces_tool_use_blocks(self):
        fc_item = _make_item("function_call", item_id="fc_01", call_id="call_abc", name="get_weather")
        events = [
            _make_event("response.created"),
            _make_event("response.output_item.added", item=fc_item),
            _make_event(
                "response.function_call_arguments.delta",
                item_id="fc_01",
                delta='{"city":"NYC"}',
            ),
            _make_event("response.output_item.done", item=fc_item),
            _make_event("response.completed", response=_make_completed_response(has_function_call=True)),
        ]

        async def _run():
            wrapper = AnthropicResponsesStreamWrapper(
                responses_stream=_aiter_from_list(events),
                model="test-model",
            )
            return await _collect_all_chunks(wrapper)

        chunks = asyncio.run(_run())

        starts = [c for c in chunks if c.get("type") == "content_block_start"]
        assert any(s["content_block"]["type"] == "tool_use" for s in starts)

        deltas = [c for c in chunks if c.get("type") == "content_block_delta"]
        assert any(d["delta"]["type"] == "input_json_delta" for d in deltas)

        msg_deltas = [c for c in chunks if c.get("type") == "message_delta"]
        assert msg_deltas[0]["delta"]["stop_reason"] == "tool_use"

    def test_no_response_created_still_works(self):
        """
        When upstream doesn't emit response.created, the fallback in __anext__
        should still produce exactly one message_start.
        """
        events = [
            _make_event(
                "response.output_item.added",
                item=_make_item("message", item_id="msg_01"),
            ),
            _make_event(
                "response.output_text.delta",
                item_id="msg_01",
                delta="World",
            ),
            _make_event(
                "response.output_item.done",
                item=_make_item("message", item_id="msg_01"),
            ),
            _make_event("response.completed", response=_make_completed_response()),
        ]

        async def _run():
            wrapper = AnthropicResponsesStreamWrapper(
                responses_stream=_aiter_from_list(events),
                model="test-model",
            )
            return await _collect_all_chunks(wrapper)

        chunks = asyncio.run(_run())
        types = [c.get("type") for c in chunks]
        assert types.count("message_start") == 1
        assert types[0] == "message_start"


# ---------------------------------------------------------------------------
# Tests — SSE wrapper
# ---------------------------------------------------------------------------


class TestSseWrapper:
    """Test the SSE byte-encoding wrapper."""

    def test_async_anthropic_sse_wrapper_emits_bytes(self):
        events = [
            _make_event("response.created"),
            _make_event("response.completed", response=_make_completed_response()),
        ]

        async def _run():
            wrapper = AnthropicResponsesStreamWrapper(
                responses_stream=_aiter_from_list(events),
                model="test-model",
            )
            sse_chunks = []
            async for chunk in wrapper.async_anthropic_sse_wrapper():
                sse_chunks.append(chunk)
            return sse_chunks

        sse_chunks = asyncio.run(_run())
        assert all(isinstance(c, bytes) for c in sse_chunks)
        decoded = [c.decode() for c in sse_chunks]
        assert any("event: message_start" in d for d in decoded)
        assert any("event: message_stop" in d for d in decoded)
