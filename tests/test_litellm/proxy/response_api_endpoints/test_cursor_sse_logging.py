import pytest


class _FakeResponsesIterator:
    def __init__(self):
        self.completed_response = {"id": "resp_test"}
        self._completed_response_logged = False
        self.logging_calls = 0

    def _handle_logging_completed_response(self):
        self.logging_calls += 1


class _FakeResponsesWrapper:
    def __init__(self, source: _FakeResponsesIterator):
        self._source_iterator = source
        self.completed_response = None

    def _handle_logging_completed_response(self):
        self._source_iterator._handle_logging_completed_response()


async def _fake_sse_stream():
    yield 'data: {"choices": []}\n\n'
    yield "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_cursor_sse_with_responses_logging_on_normal_completion():
    from litellm.proxy.response_api_endpoints.endpoints import (
        _cursor_sse_with_responses_logging,
    )

    responses_iterator = _FakeResponsesIterator()
    chunks = []
    async for chunk in _cursor_sse_with_responses_logging(
        _fake_sse_stream(), responses_iterator
    ):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert responses_iterator.logging_calls == 1


@pytest.mark.asyncio
async def test_cursor_sse_with_responses_logging_uses_source_iterator():
    from litellm.proxy.response_api_endpoints.endpoints import (
        _cursor_sse_with_responses_logging,
    )

    source = _FakeResponsesIterator()
    wrapper = _FakeResponsesWrapper(source)

    async for _ in _cursor_sse_with_responses_logging(
        _fake_sse_stream(), wrapper
    ):
        pass

    assert source.logging_calls == 1


@pytest.mark.asyncio
async def test_cursor_sse_with_responses_logging_skips_without_completed_response():
    from litellm.proxy.response_api_endpoints.endpoints import (
        _cursor_sse_with_responses_logging,
    )

    responses_iterator = _FakeResponsesIterator()
    responses_iterator.completed_response = None

    async for _ in _cursor_sse_with_responses_logging(
        _fake_sse_stream(), responses_iterator
    ):
        pass

    assert responses_iterator.logging_calls == 0
