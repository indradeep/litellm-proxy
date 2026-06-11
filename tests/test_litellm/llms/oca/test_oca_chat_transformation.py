from unittest.mock import patch

import pytest

from litellm.llms.oca.chat.transformation import OCAChatConfig


class TestOCAChatConfig:
    def test_provider_enum(self) -> None:
        config = OCAChatConfig()
        assert config.custom_llm_provider.value == "oca"

    def test_get_complete_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "OCA_API_BASE",
            "https://code-internal.example.com/20250206/app/litellm",
        )
        config = OCAChatConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="oca/gpt-5.4",
            optional_params={},
            litellm_params={},
        )
        assert url.endswith("/chat/completions")

    def test_validate_environment_sets_auth_headers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OCA_API_BASE", "https://example.com/litellm")
        config = OCAChatConfig()
        headers: dict[str, str] = {}

        with patch(
            "litellm.llms.oca.chat.transformation.apply_oca_auth_headers",
            return_value="token-xyz",
        ) as mock_apply:
            result = config.validate_environment(
                headers=headers,
                model="oca/gpt-5.4-mini",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={},
                litellm_params={},
            )
            mock_apply.assert_called_once()
            assert result is headers

    def test_should_not_fake_stream(self) -> None:
        config = OCAChatConfig()
        assert config.should_fake_stream("oca/gpt-5.4", False, "oca") is False

    def test_transform_request_passes_effort_and_priority(self) -> None:
        import litellm

        litellm.drop_params = True
        config = OCAChatConfig()
        optional_params = {
            "reasoning_effort": "extra high",
            "priority": "fast",
            "stream": True,
        }
        non_default_params: dict = {}
        config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="oca/gpt-5.5",
            drop_params=True,
        )
        body = config.transform_request(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        assert body["reasoning_effort"] == "xhigh"
        assert body["service_tier"] == "priority"

    def test_transform_request_applies_defaults(self) -> None:
        config = OCAChatConfig()
        body = config.transform_request(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"stream": True},
            litellm_params={},
            headers={},
        )
        assert body["reasoning_effort"] == "xhigh"
        assert body["service_tier"] == "priority"

    def test_transform_request_strips_stream_options(self) -> None:
        config = OCAChatConfig()
        optional_params = {
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        body = config.transform_request(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        assert body["stream"] is True
        assert "stream_options" not in body
