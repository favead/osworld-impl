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


def extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = strip_code_fence(text)
    if candidate.startswith("<think>") and "</think>" in candidate:
        candidate = candidate.split("</think>", 1)[1].strip()

    for raw in (candidate, text):
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            continue
        snippet = raw[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue

    return None


def call_model(
    *,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str,
    base_url: str,
    max_tokens: int,
    top_p: float,
    temperature: float,
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
        "temperature": temperature,
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

    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content)
