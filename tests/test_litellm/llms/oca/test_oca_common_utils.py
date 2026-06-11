import json
import time
from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.oca.common_utils import (
    OCATokenManager,
    add_oca_headers,
    get_oca_access_token,
    invalidate_oca_access_token,
    normalize_oca_request_params,
    prepare_oca_zdr_responses_request,
    strip_oca_unsupported_request_params,
)


class TestOCATokenManager:
    def setup_method(self) -> None:
        invalidate_oca_access_token()

    def test_fetch_and_cache_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCA_CLIENT_ID", "client-id")
        monkeypatch.setenv("OCA_CLIENT_SECRET", "client-secret")
        monkeypatch.setenv("OCA_TOKEN_URL", "https://example.com/token")
        monkeypatch.setenv("OCA_SCOPE", "scope")
        monkeypatch.setenv("OCA_AUTH_METHOD", "basic")

        payload = json.dumps(
            {"access_token": "token-123", "expires_in": 3600}
        ).encode("utf-8")
        mock_response = MagicMock()
        mock_response.read.return_value = payload
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_response) as urlopen:
            token = get_oca_access_token()
            assert token == "token-123"
            cached = get_oca_access_token()
            assert cached == "token-123"
            assert urlopen.call_count == 1

    def test_refresh_when_near_expiry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCA_CLIENT_ID", "client-id")
        monkeypatch.setenv("OCA_CLIENT_SECRET", "client-secret")
        monkeypatch.setenv("OCA_TOKEN_URL", "https://example.com/token")

        manager = OCATokenManager()
        manager._cache.access_token = "old-token"
        manager._cache.expires_at = int(time.time()) + 100

        payload = json.dumps(
            {"access_token": "new-token", "expires_in": 3600}
        ).encode("utf-8")
        mock_response = MagicMock()
        mock_response.read.return_value = payload
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_response):
            token = manager.get_access_token()
            assert token == "new-token"

    def test_body_auth_method(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCA_CLIENT_ID", "client-id")
        monkeypatch.setenv("OCA_CLIENT_SECRET", "client-secret")
        monkeypatch.setenv("OCA_TOKEN_URL", "https://example.com/token")
        monkeypatch.setenv("OCA_AUTH_METHOD", "body")

        payload = json.dumps(
            {"access_token": "body-token", "expires_in": 3600}
        ).encode("utf-8")
        mock_response = MagicMock()
        mock_response.read.return_value = payload
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_response) as urlopen:
            token = get_oca_access_token(force_refresh=True)
            assert token == "body-token"
            request = urlopen.call_args[0][0]
            body = request.data.decode("utf-8")
            assert "client_id=client-id" in body
            assert "client_secret=client-secret" in body


class TestStripOCAUnsupportedRequestParams:
    def test_removes_stream_options(self) -> None:
        params = {"stream_options": {"include_usage": True}, "stream": True}
        strip_oca_unsupported_request_params(params)
        assert "stream_options" not in params
        assert params["stream"] is True

    def test_removes_previous_response_id_and_store_true(self) -> None:
        params = {
            "previous_response_id": "resp_123",
            "store": True,
            "stream": True,
        }
        strip_oca_unsupported_request_params(params)
        assert "previous_response_id" not in params
        assert params["store"] is False
        assert params["stream"] is True


class TestPrepareOCAZDRResponsesRequest:
    @pytest.mark.asyncio
    async def test_expands_input_from_session_and_strips_previous_response_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session_messages = [{"role": "user", "content": "hello"}]
        captured: dict = {}

        async def fake_get_history(previous_response_id: str):
            captured["previous_response_id"] = previous_response_id
            return {"messages": session_messages, "litellm_session_id": None}

        from litellm.responses.litellm_completion_transformation import session_handler

        monkeypatch.setattr(
            session_handler.ResponsesSessionHandler,
            "get_chat_completion_message_history_for_previous_response_id",
            fake_get_history,
        )

        input_data = [{"role": "user", "content": "follow up"}]
        params = {"previous_response_id": "resp_prev", "stream": True}

        expanded_input, result_params = await prepare_oca_zdr_responses_request(
            input=input_data,
            response_api_optional_params=params,
        )

        assert captured["previous_response_id"] == "resp_prev"
        assert "previous_response_id" not in result_params
        assert result_params["stream"] is True
        assert isinstance(expanded_input, list)
        assert len(expanded_input) >= 2

    @pytest.mark.asyncio
    async def test_noop_without_previous_response_id(self) -> None:
        input_data = [{"role": "user", "content": "only turn"}]
        params = {"stream": True}

        expanded_input, result_params = await prepare_oca_zdr_responses_request(
            input=input_data,
            response_api_optional_params=params,
        )

        assert expanded_input == input_data
        assert "previous_response_id" not in result_params


class TestNormalizeOCARequestParams:
    def test_maps_extra_high_and_fast_aliases(self) -> None:
        params = {
            "reasoning_effort": "extra high",
            "priority": "fast",
        }
        normalize_oca_request_params(params)
        assert params["reasoning_effort"] == "xhigh"
        assert params["service_tier"] == "priority"
        assert "priority" not in params

    def test_normalizes_reasoning_dict_effort(self) -> None:
        params = {"reasoning": {"effort": "extra_high", "summary": "detailed"}}
        normalize_oca_request_params(params, prefer_reasoning_dict=True)
        assert params["reasoning"]["effort"] == "xhigh"

    def test_defaults_chat_effort_and_priority(self) -> None:
        params: dict = {}
        normalize_oca_request_params(params, prefer_reasoning_dict=False)
        assert params["reasoning_effort"] == "xhigh"
        assert params["service_tier"] == "priority"

    def test_defaults_responses_reasoning_and_priority(self) -> None:
        params: dict = {}
        normalize_oca_request_params(params, prefer_reasoning_dict=True)
        assert params["reasoning"] == {"effort": "xhigh"}
        assert params["service_tier"] == "priority"

    def test_unreconciled_effort_defaults_to_xhigh(self) -> None:
        params = {"reasoning_effort": "not-a-real-level"}
        normalize_oca_request_params(params, prefer_reasoning_dict=False)
        assert params["reasoning_effort"] == "xhigh"

    def test_preserves_explicit_effort_and_tier(self) -> None:
        params = {"reasoning_effort": "medium", "service_tier": "default"}
        normalize_oca_request_params(params, prefer_reasoning_dict=False)
        assert params["reasoning_effort"] == "medium"
        assert params["service_tier"] == "default"


class TestOCAHeaders:
    def test_add_oca_headers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCA_CLIENT_NAME", "test-client")
        monkeypatch.setenv("OCA_CLIENT_VERSION", "1.2.3")
        headers: dict[str, str] = {}
        add_oca_headers(headers=headers, model="oca/gpt-5.4", token="abc")
        assert headers["client"] == "test-client"
        assert headers["client-version"] == "1.2.3"
        assert headers["client-ide"] == "litellm"
        assert "opc-request-id" in headers
