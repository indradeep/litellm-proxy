import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Set


_EXPLICIT_METADATA_KEYS = (
    "cursor_session_id",
    "conversation_id",
    "thread_id",
    "session_id",
)
_EXPLICIT_TOP_LEVEL_KEYS = ("conversation_id", "thread_id", "session_id")
_EXPLICIT_HEADER_KEYS = (
    "x-litellm-cursor-session-id",
    "x-cursor-session-id",
    "x-cursor-chat-id",
    "x-cursor-conversation-id",
)
_OPT_OUT_HEADER = "x-litellm-cursor-compaction"


@dataclass
class CursorSessionDecision:
    session_key: Optional[str]
    session_key_source: str
    input_mode: str
    status: str
    original_input: Any
    original_message_hashes: List[str]
    raw_tail_messages: List[Any]
    estimated_tokens_before: int
    estimated_tokens_after: int


@dataclass
class _CursorSessionState:
    message_hashes: List[str]
    summary_message: Dict[str, Any]
    raw_tail_messages: List[Any]
    last_seen_at: float
    turn_count: int = 0


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _headers_get(headers: Mapping[str, str], key: str) -> Optional[str]:
    lowered = key.lower()
    for header_key, header_value in headers.items():
        if header_key.lower() == lowered and header_value:
            return str(header_value)
    return None


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _is_opted_out(data: Dict[str, Any], headers: Mapping[str, str]) -> bool:
    metadata = data.get("metadata")
    if isinstance(metadata, dict) and _is_truthy(
        metadata.get("cursor_disable_compaction")
    ):
        return True
    header_value = _headers_get(headers, _OPT_OUT_HEADER)
    return bool(header_value and header_value.strip().lower() in {"off", "false", "0"})


def _extract_session_key(
    data: Dict[str, Any],
    headers: Mapping[str, str],
    user_api_key_hash: Optional[str],
) -> tuple[Optional[str], str]:
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        for key in _EXPLICIT_METADATA_KEYS:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return f"explicit:{key}:{_hash_text(value.strip())}", f"metadata.{key}"

    for key in _EXPLICIT_TOP_LEVEL_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return f"explicit:{key}:{_hash_text(value.strip())}", key

    for key in _EXPLICIT_HEADER_KEYS:
        value = _headers_get(headers, key)
        if value and value.strip():
            return f"explicit:{key}:{_hash_text(value.strip())}", f"header.{key}"

    input_value = data.get("input")
    model = data.get("model")
    if not user_api_key_hash or not isinstance(model, str) or not isinstance(
        input_value, list
    ):
        return None, "none"
    if not input_value:
        return None, "none"

    first_message_hash = _hash_value(input_value[0])
    fallback_material = f"{user_api_key_hash}:{model}:{first_message_hash}"
    return f"fallback:{_hash_text(fallback_material)}", "fallback.first_message"


def _estimate_tokens(value: Any) -> int:
    # Fast, deterministic estimate. It is intentionally approximate; it is used
    # only for thresholding and observability.
    return max(1, len(_stable_json(value)) // 4)


def _message_hashes(input_value: Any) -> List[str]:
    if not isinstance(input_value, list):
        return []
    return [_hash_value(message) for message in input_value]


def _is_prefix(prefix: List[str], values: List[str]) -> bool:
    return len(prefix) <= len(values) and values[: len(prefix)] == prefix


def _common_prefix_len(left: List[str], right: List[str]) -> int:
    count = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        count += 1
    return count


def _clip_text(text: str, limit: int = 700) -> str:
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return f"{text[:half]}\n...[{len(text) - limit} chars omitted]...\n{text[-half:]}"


def _summarize_content(content: Any) -> str:
    if isinstance(content, str):
        return _clip_text(content)
    return _clip_text(_stable_json(content))


def _build_summary_message(messages: List[Any], max_messages: int = 24) -> Dict[str, str]:
    older_messages = messages[:max_messages]
    omitted = max(0, len(messages) - len(older_messages))
    lines = [
        "This is a server-side compacted summary of earlier Cursor context.",
        "It preserves bounded excerpts of older messages so the downstream model does not receive the full repeated prefix.",
    ]
    if omitted:
        lines.append(f"{omitted} older messages were omitted from this compacted view.")
    for index, message in enumerate(older_messages, start=1):
        if isinstance(message, dict):
            role = message.get("role") or message.get("type") or "unknown"
            content = message.get("content")
            lines.append(f"[{index}] {role}: {_summarize_content(content)}")
        else:
            lines.append(f"[{index}] {_summarize_content(message)}")
    return {"role": "developer", "content": "\n".join(lines)}


def _add_string_id(ids: Set[str], value: Any) -> None:
    if isinstance(value, str) and value:
        ids.add(value)


def _tool_use_ids_from_dict(value: Dict[str, Any]) -> Set[str]:
    ids: Set[str] = set()
    value_type = value.get("type")
    if value_type == "function_call":
        _add_string_id(ids, value.get("call_id"))
        _add_string_id(ids, value.get("id"))
    elif value_type == "tool_use":
        _add_string_id(ids, value.get("id"))
        _add_string_id(ids, value.get("tool_use_id"))

    tool_calls = value.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                _add_string_id(ids, tool_call.get("id"))
                _add_string_id(ids, tool_call.get("call_id"))

    content = value.get("content")
    if isinstance(content, list):
        for content_block in content:
            if isinstance(content_block, dict):
                ids.update(_tool_use_ids_from_dict(content_block))
    return ids


def _tool_result_ids_from_dict(value: Dict[str, Any]) -> Set[str]:
    ids: Set[str] = set()
    value_type = value.get("type")
    if value_type == "function_call_output":
        _add_string_id(ids, value.get("call_id"))
        _add_string_id(ids, value.get("id"))
    elif value_type == "tool_result":
        _add_string_id(ids, value.get("tool_use_id"))

    _add_string_id(ids, value.get("tool_call_id"))

    content = value.get("content")
    if isinstance(content, list):
        for content_block in content:
            if isinstance(content_block, dict):
                ids.update(_tool_result_ids_from_dict(content_block))
    return ids


def _has_orphaned_tool_results(messages: List[Any]) -> bool:
    previous_tool_use_ids: Set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            previous_tool_use_ids = set()
            continue

        result_ids = _tool_result_ids_from_dict(message)
        if result_ids and not result_ids.issubset(previous_tool_use_ids):
            return True

        previous_tool_use_ids = _tool_use_ids_from_dict(message)
    return False


class CursorSessionStore:
    def __init__(
        self,
        *,
        max_entries: int,
        ttl_seconds: int,
        min_estimated_tokens: int,
        raw_tail_messages: int,
        raw_tail_max_estimated_tokens: int = 12_000,
    ) -> None:
        self.max_entries = max(1, max_entries)
        self.ttl_seconds = max(1, ttl_seconds)
        self.min_estimated_tokens = max(1, min_estimated_tokens)
        self.raw_tail_messages = max(0, raw_tail_messages)
        self.raw_tail_max_estimated_tokens = max(0, raw_tail_max_estimated_tokens)
        self._sessions: "OrderedDict[str, _CursorSessionState]" = OrderedDict()

    def _select_raw_tail_messages(self, input_value: Any) -> List[Any]:
        if (
            not isinstance(input_value, list)
            or self.raw_tail_messages <= 0
            or not input_value
        ):
            return []

        selected: List[Any] = []
        estimated_total = 0
        candidates = input_value[-self.raw_tail_messages :]
        for message in reversed(candidates):
            message_tokens = _estimate_tokens(message)
            if (
                self.raw_tail_max_estimated_tokens
                and estimated_total + message_tokens
                > self.raw_tail_max_estimated_tokens
            ):
                break
            selected.append(message)
            estimated_total += message_tokens

        selected.reverse()
        return selected

    def prepare_request(
        self,
        data: Dict[str, Any],
        *,
        headers: Mapping[str, str],
        user_api_key_hash: Optional[str] = None,
    ) -> CursorSessionDecision:
        input_value = data.get("input")
        hashes = _message_hashes(input_value)
        original_input = list(input_value) if isinstance(input_value, list) else input_value
        estimated_before = _estimate_tokens(original_input)
        session_key, key_source = _extract_session_key(
            data=data,
            headers=headers,
            user_api_key_hash=user_api_key_hash,
        )
        raw_tail = self._select_raw_tail_messages(input_value)

        if _is_opted_out(data, headers):
            return CursorSessionDecision(
                session_key=session_key,
                session_key_source=key_source,
                input_mode="full",
                status="opted_out",
                original_input=original_input,
                original_message_hashes=hashes,
                raw_tail_messages=raw_tail,
                estimated_tokens_before=estimated_before,
                estimated_tokens_after=estimated_before,
            )

        if not session_key or not isinstance(input_value, list) or not hashes:
            return CursorSessionDecision(
                session_key=session_key,
                session_key_source=key_source,
                input_mode="full",
                status="no_session",
                original_input=original_input,
                original_message_hashes=hashes,
                raw_tail_messages=raw_tail,
                estimated_tokens_before=estimated_before,
                estimated_tokens_after=estimated_before,
            )

        self._expire()
        state = self._sessions.get(session_key)
        if state is None:
            return CursorSessionDecision(
                session_key=session_key,
                session_key_source=key_source,
                input_mode="full",
                status="miss",
                original_input=original_input,
                original_message_hashes=hashes,
                raw_tail_messages=raw_tail,
                estimated_tokens_before=estimated_before,
                estimated_tokens_after=estimated_before,
            )

        if _is_prefix(state.message_hashes, hashes) and len(hashes) > len(
            state.message_hashes
        ):
            suffix = input_value[len(state.message_hashes) :]
            compacted_input = (
                [state.summary_message] + list(state.raw_tail_messages) + suffix
            )
            estimated_after = _estimate_tokens(compacted_input)
            if _has_orphaned_tool_results(compacted_input):
                return CursorSessionDecision(
                    session_key=session_key,
                    session_key_source=key_source,
                    input_mode="full",
                    status="tool_history_unsafe",
                    original_input=original_input,
                    original_message_hashes=hashes,
                    raw_tail_messages=raw_tail,
                    estimated_tokens_before=estimated_before,
                    estimated_tokens_after=estimated_before,
                )
            if estimated_after >= estimated_before:
                return CursorSessionDecision(
                    session_key=session_key,
                    session_key_source=key_source,
                    input_mode="full",
                    status="not_smaller",
                    original_input=original_input,
                    original_message_hashes=hashes,
                    raw_tail_messages=raw_tail,
                    estimated_tokens_before=estimated_before,
                    estimated_tokens_after=estimated_before,
                )

            data["input"] = compacted_input
            return CursorSessionDecision(
                session_key=session_key,
                session_key_source=key_source,
                input_mode="compacted",
                status="hit",
                original_input=original_input,
                original_message_hashes=hashes,
                raw_tail_messages=raw_tail,
                estimated_tokens_before=estimated_before,
                estimated_tokens_after=estimated_after,
            )

        common_prefix_len = _common_prefix_len(state.message_hashes, hashes)
        if common_prefix_len <= 0 or common_prefix_len >= len(hashes):
            return CursorSessionDecision(
                session_key=session_key,
                session_key_source=key_source,
                input_mode="full",
                status="prefix_mismatch",
                original_input=original_input,
                original_message_hashes=hashes,
                raw_tail_messages=raw_tail,
                estimated_tokens_before=estimated_before,
                estimated_tokens_after=estimated_before,
            )

        stable_prefix = input_value[:common_prefix_len]
        stable_prefix_tail = self._select_raw_tail_messages(stable_prefix)
        stable_tail_count = len(stable_prefix_tail)
        stable_summary_source = (
            stable_prefix[:-stable_tail_count] if stable_tail_count else stable_prefix
        )
        if not stable_summary_source:
            stable_summary_source = stable_prefix

        suffix = input_value[common_prefix_len:]
        compacted_input = (
            [_build_summary_message(stable_summary_source)]
            + stable_prefix_tail
            + suffix
        )
        estimated_after = _estimate_tokens(compacted_input)
        if _has_orphaned_tool_results(compacted_input):
            return CursorSessionDecision(
                session_key=session_key,
                session_key_source=key_source,
                input_mode="full",
                status="tool_history_unsafe",
                original_input=original_input,
                original_message_hashes=hashes,
                raw_tail_messages=raw_tail,
                estimated_tokens_before=estimated_before,
                estimated_tokens_after=estimated_before,
            )
        if estimated_after >= estimated_before:
            return CursorSessionDecision(
                session_key=session_key,
                session_key_source=key_source,
                input_mode="full",
                status="not_smaller",
                original_input=original_input,
                original_message_hashes=hashes,
                raw_tail_messages=raw_tail,
                estimated_tokens_before=estimated_before,
                estimated_tokens_after=estimated_before,
            )

        data["input"] = compacted_input
        return CursorSessionDecision(
            session_key=session_key,
            session_key_source=key_source,
            input_mode="compacted",
            status="partial_prefix_hit",
            original_input=original_input,
            original_message_hashes=hashes,
            raw_tail_messages=raw_tail,
            estimated_tokens_before=estimated_before,
            estimated_tokens_after=estimated_after,
        )

    def record_response(
        self, decision: CursorSessionDecision, *, assistant_text: Optional[str] = None
    ) -> None:
        # Cursor's next request should include the assistant reply after the
        # repeated prefix; the suffix path preserves it without duplicating it.
        _ = assistant_text
        if (
            not decision.session_key
            or not isinstance(decision.original_input, list)
            or not decision.original_message_hashes
            or decision.estimated_tokens_before < self.min_estimated_tokens
        ):
            return

        input_for_summary = list(decision.original_input)
        raw_tail_count = len(decision.raw_tail_messages)
        summary_source = (
            input_for_summary[:-raw_tail_count]
            if raw_tail_count
            else input_for_summary
        )
        if not summary_source:
            summary_source = input_for_summary

        state = self._sessions.get(decision.session_key)
        turn_count = (state.turn_count + 1) if state else 1
        self._sessions[decision.session_key] = _CursorSessionState(
            message_hashes=list(decision.original_message_hashes),
            summary_message=_build_summary_message(summary_source),
            raw_tail_messages=list(decision.raw_tail_messages),
            last_seen_at=time.monotonic(),
            turn_count=turn_count,
        )
        self._sessions.move_to_end(decision.session_key)
        while len(self._sessions) > self.max_entries:
            self._sessions.popitem(last=False)

    def _expire(self) -> None:
        now = time.monotonic()
        expired = [
            key
            for key, state in self._sessions.items()
            if now - state.last_seen_at > self.ttl_seconds
        ]
        for key in expired:
            self._sessions.pop(key, None)
