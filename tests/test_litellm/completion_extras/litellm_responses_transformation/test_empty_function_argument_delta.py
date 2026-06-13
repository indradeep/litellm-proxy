from litellm.completion_extras.litellm_responses_transformation.transformation import (
    OpenAiResponsesToChatCompletionStreamIterator,
)


def test_empty_function_call_arguments_delta_is_skipped():
    chunk = {
        "type": "response.function_call_arguments.delta",
        "item_id": "fc_toolu_test",
        "output_index": 2,
        "delta": "",
        "sequence_number": 12,
    }

    result = OpenAiResponsesToChatCompletionStreamIterator.translate_responses_chunk_to_openai_stream(
        chunk
    )

    assert result.choices[0].delta.content is None
    assert result.choices[0].delta.tool_calls is None
    assert result.choices[0].finish_reason is None
