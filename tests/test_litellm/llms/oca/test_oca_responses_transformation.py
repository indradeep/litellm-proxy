import pytest

from litellm.llms.oca.responses.transformation import OCAResponsesAPIConfig
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponseCompletedEvent, ResponsesAPIResponse


class TestOCAResponsesAPIConfig:
  def test_requires_streaming_upstream_when_not_streaming(self) -> None:
      config = OCAResponsesAPIConfig()
      assert config.requires_streaming_upstream(False) is True
      assert config.requires_streaming_upstream(None) is True
      assert config.requires_streaming_upstream(True) is False


class TestCollectResponsesStream:
  def test_collect_sync_responses_stream_returns_completed_response(self) -> None:
      completed = ResponsesAPIResponse(
          id="resp_test",
          object="response",
          created_at=1,
          status="completed",
          model="oca/gpt-5.5",
          output=[],
      )
      event = ResponseCompletedEvent(
          type="response.completed",
          response=completed,
      )

      class _FakeIterator:
          def __init__(self) -> None:
              self.completed_response = None
              self._events = [event]

          def __iter__(self):
              return self

          def __next__(self):
              if not self._events:
                  raise StopIteration
              chunk = self._events.pop(0)
              self.completed_response = chunk
              return chunk

      result = ResponsesAPIRequestUtils.collect_sync_responses_stream(_FakeIterator())
      assert result.id == "resp_test"
      assert result.status == "completed"

  @pytest.mark.asyncio
  async def test_collect_async_responses_stream_returns_completed_response(self) -> None:
      completed = ResponsesAPIResponse(
          id="resp_async",
          object="response",
          created_at=1,
          status="completed",
          model="oca/gpt-5.5",
          output=[],
      )
      event = ResponseCompletedEvent(
          type="response.completed",
          response=completed,
      )

      class _FakeAsyncIterator:
          def __init__(self) -> None:
              self.completed_response = None
              self._events = [event]

          def __aiter__(self):
              return self

          async def __anext__(self):
              if not self._events:
                  raise StopAsyncIteration
              chunk = self._events.pop(0)
              self.completed_response = chunk
              return chunk

      result = await ResponsesAPIRequestUtils.collect_async_responses_stream(
          _FakeAsyncIterator()
      )
      assert result.id == "resp_async"
