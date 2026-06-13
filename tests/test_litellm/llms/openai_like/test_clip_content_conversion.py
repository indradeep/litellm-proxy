from litellm.llms.openai_like.dynamic_config import (
    _convert_anthropic_tool_blocks_to_responses_input,
    _convert_text_only_content_lists_to_string,
    create_responses_config_class,
)
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.types.router import GenericLiteLLMParams


def test_convert_text_only_content_lists_to_string_flattens_simple_user_text():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    ]

    converted = _convert_text_only_content_lists_to_string(messages)

    assert converted[0]["content"] == "hello"


def test_convert_text_only_content_lists_preserves_tool_use_and_tool_result():
    messages = [
        {"role": "user", "content": "run git status"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll run git status"},
                {
                    "type": "tool_use",
                    "id": "toolu_test1",
                    "name": "Shell",
                    "input": {"command": "git status --short"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_test1",
                    "content": [{"type": "text", "text": " M file.txt"}],
                }
            ],
        },
    ]

    converted = _convert_text_only_content_lists_to_string(messages)

    assert converted[0]["content"] == "run git status"
    assert isinstance(converted[1]["content"], list)
    assert converted[1]["content"][1]["type"] == "tool_use"
    assert isinstance(converted[2]["content"], list)
    assert converted[2]["content"][0]["type"] == "tool_result"


def test_convert_anthropic_tool_blocks_to_responses_input_items():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll inspect that."},
                {
                    "type": "tool_use",
                    "id": "toolu_test1",
                    "name": "Shell",
                    "input": {"command": "git status --short"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_test1",
                    "content": [{"type": "text", "text": " M file.txt"}],
                }
            ],
        },
    ]

    converted = _convert_anthropic_tool_blocks_to_responses_input(messages)

    assert converted == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "I'll inspect that."}],
        },
        {
            "type": "function_call",
            "call_id": "toolu_test1",
            "name": "Shell",
            "arguments": '{"command":"git status --short"}',
        },
        {
            "type": "function_call_output",
            "call_id": "toolu_test1",
            "output": [{"type": "input_text", "text": " M file.txt"}],
        },
    ]


def test_clip_responses_transform_converts_anthropic_tool_history():
    provider = JSONProviderRegistry.get("clip")
    assert provider is not None

    config = create_responses_config_class(provider)()
    request = config.transform_responses_api_request(
        model="clip/claude-opus-4-8(xhigh)",
        input=[
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll inspect that."},
                    {
                        "type": "tool_use",
                        "id": "toolu_test1",
                        "name": "Shell",
                        "input": {"command": "git status --short"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_test1",
                        "content": [{"type": "text", "text": " M file.txt"}],
                    }
                ],
            },
        ],
        response_api_optional_request_params={"metadata": {"source": "cursor"}},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request["input"][-2]["type"] == "function_call"
    assert request["input"][-1]["type"] == "function_call_output"


def test_responses_api_response_accepts_chat_style_usage():
    """clip emits chat-style usage on response.completed; parsing must not raise.

    Forcing ResponseAPIUsage(**value) on a chat-style usage dict previously
    raised a ValidationError (missing input_tokens/output_tokens), surfacing as
    a MidStreamFallbackError mid-stream.
    """
    from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse

    response = ResponsesAPIResponse(
        id="resp_clip",
        object="response",
        created_at=1,
        status="completed",
        model="clip/claude-opus-4-8",
        output=[],
        usage={
            "prompt_tokens": 182240,
            "completion_tokens": 166,
            "total_tokens": 182406,
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "text_tokens": 166,
            },
            "prompt_tokens_details": {"cached_tokens": 179585},
        },
    )

    assert isinstance(response.usage, ResponseAPIUsage)
    assert response.usage.input_tokens == 182240
    assert response.usage.output_tokens == 166
    assert response.usage.total_tokens == 182406


def test_responses_api_response_preserves_response_style_usage():
    """Standard Responses-style usage must continue to parse unchanged."""
    from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse

    response = ResponsesAPIResponse(
        id="resp_std",
        object="response",
        created_at=1,
        status="completed",
        model="oca/gpt-5.5",
        output=[],
        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )

    assert isinstance(response.usage, ResponseAPIUsage)
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 15
