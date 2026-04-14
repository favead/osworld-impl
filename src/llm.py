import json
import re
import time
import urllib.error
import urllib.request
from typing import Any


def strip_code_fence(text: str) -> str:
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced[0].strip()
    return text.strip()


def _extract_balanced_json(raw: str) -> dict[str, Any] | None:
    start = raw.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = raw[start : idx + 1]
                try:
                    value = json.loads(snippet)
                except json.JSONDecodeError:
                    return None
                if isinstance(value, dict):
                    return value
                return None
    return None


def _content_to_text(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif part.get("type") == "output_text" and isinstance(part.get("output_text"), str):
                    parts.append(part["output_text"])
        return "\n".join(parts).strip()
    if isinstance(content, str):
        return content
    return str(content)


def _tool_calls_to_actions(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []

    actions: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue

        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            continue

        args_raw = function.get("arguments", "{}")
        parsed_args: dict[str, Any] = {}
        if isinstance(args_raw, str) and args_raw.strip():
            try:
                decoded = json.loads(args_raw)
                if isinstance(decoded, dict):
                    parsed_args = decoded
            except json.JSONDecodeError:
                parsed_args = {}
        elif isinstance(args_raw, dict):
            parsed_args = args_raw

        actions.append({"name": name.strip(), "arguments": parsed_args})
    return actions


def extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = strip_code_fence(text)
    if "</think>" in candidate:
        candidate = candidate.split("</think>", 1)[1].strip()

    tool_call_match = re.search(r"<tool_call>\s*(\{.*\})\s*</tool_call>", candidate, flags=re.DOTALL)
    if tool_call_match:
        try:
            call_obj = json.loads(tool_call_match.group(1))
            if isinstance(call_obj, dict):
                reasoning = candidate[: tool_call_match.start()].strip()
                return {
                    "reasoning": reasoning,
                    "actions": [call_obj],
                }
        except json.JSONDecodeError:
            pass

    for raw in (candidate, text):
        trimmed = raw.strip()
        if not trimmed:
            continue

        try:
            parsed = json.loads(trimmed)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        parsed = _extract_balanced_json(trimmed)
        if isinstance(parsed, dict):
            return parsed

    return None


def call_model(
    *,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str,
    base_url: str,
    max_tokens: int,
    top_p: float,
    top_k: int,
    min_p: float,
    temperature: float,
    presence_penalty: float,
    repetition_penalty: float,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> str:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    request_url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "temperature": temperature,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
    }
    request = urllib.request.Request(
        request_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error: Exception | None = None
    attempts = max(max_retries, 0) + 1
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            is_retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
            last_error = RuntimeError(f"LLM request failed with HTTP {exc.code}: {error_body}")
            if not is_retryable or attempt == attempts:
                raise last_error from exc
        except urllib.error.URLError as exc:
            last_error = RuntimeError(f"LLM request failed: {exc}")
            if attempt == attempts:
                raise last_error from exc

        time.sleep(retry_backoff_seconds * attempt)
    else:
        if last_error:
            raise last_error
        raise RuntimeError("LLM request failed without a specific error")

    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError(f"LLM response missing choices: {body}")

    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return str(message)

    reasoning = _content_to_text(message.get("content", "")).strip()
    actions = _tool_calls_to_actions(message.get("tool_calls"))
    if actions:
        return json.dumps({"reasoning": reasoning, "actions": actions}, ensure_ascii=False)
    return reasoning
