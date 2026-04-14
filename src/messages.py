import base64
import json
from typing import Any

from a2a.types import DataPart, FilePart, FileWithBytes, Message, TextPart

from state import AgentState, ParsedInput, StepRecord


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[...]"


def coerce_screenshot_bytes(value: Any) -> bytes | None:
    if isinstance(value, bytearray):
        value = bytes(value)

    if isinstance(value, bytes):
        if value.startswith(b"\x89PNG") or value.startswith(b"\xff\xd8"):
            return value
        try:
            decoded = base64.b64decode(value, validate=True)
            if decoded:
                return decoded
        except Exception:
            pass
        return value

    if isinstance(value, str):
        encoded = value.strip()
        if not encoded:
            return None
        if encoded.startswith("data:image") and "," in encoded:
            encoded = encoded.split(",", 1)[1]
        try:
            return base64.b64decode(encoded)
        except Exception:
            return None

    return None


def parse_message(message: Message) -> ParsedInput:
    instruction = ""
    obs: dict[str, Any] = {}
    env_config: dict[str, Any] = {}

    for part in message.parts:
        root = part.root
        if isinstance(root, TextPart) and root.text.strip():
            instruction = root.text.strip()
        elif isinstance(root, FilePart) and isinstance(root.file, FileWithBytes):
            screenshot_bytes = coerce_screenshot_bytes(root.file.bytes)
            if screenshot_bytes is not None:
                obs["screenshot"] = screenshot_bytes
        elif isinstance(root, DataPart):
            payload = dict(root.data)
            if "env_config" in payload:
                raw_env = payload.pop("env_config")
                if isinstance(raw_env, dict):
                    env_config.update(raw_env)
            obs.update(payload)

    screenshot_value = obs.get("screenshot")
    if screenshot_value is not None:
        screenshot_bytes = coerce_screenshot_bytes(screenshot_value)
        if screenshot_bytes is None:
            obs.pop("screenshot", None)
        else:
            obs["screenshot"] = screenshot_bytes

    return ParsedInput(instruction=instruction, observation=obs, env_config=env_config)


def merge_input(state: AgentState, parsed: ParsedInput) -> AgentState:
    instruction_changed = bool(parsed.instruction and parsed.instruction != state.instruction)
    if instruction_changed:
        state = AgentState(
            instruction=parsed.instruction,
            env_config=parsed.env_config or state.env_config,
        )
    else:
        if parsed.instruction:
            state.instruction = parsed.instruction
        if parsed.env_config:
            state.env_config = parsed.env_config

    if parsed.observation:
        state.last_observation = parsed.observation

    return state


def planner_observation_text(obs: dict[str, Any], a11y_tree_max_chars: int) -> str:
    structured: dict[str, Any] = {}
    for key, value in obs.items():
        if key == "screenshot":
            continue
        if key in {"a11y_tree", "accessibility_tree"} and isinstance(value, str):
            structured[key] = truncate_text(value, a11y_tree_max_chars)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            structured[key] = value
        else:
            try:
                structured[key] = json.loads(json.dumps(value, ensure_ascii=False, default=str))
            except Exception:
                structured[key] = str(value)

    if not structured:
        return ""
    return json.dumps(structured, ensure_ascii=False, indent=2)


def image_message(screenshot_bytes: bytes) -> dict[str, Any]:
    encoded = base64.b64encode(screenshot_bytes).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}"},
    }


def summarize_observation(obs: dict[str, Any]) -> str:
    summary: dict[str, Any] = {}
    if "screenshot" in obs and isinstance(obs["screenshot"], (bytes, bytearray)):
        summary["screenshot_bytes"] = len(obs["screenshot"])
    for key in ["app", "cur_app", "window_title", "cur_window_id"]:
        if key in obs:
            summary[key] = obs[key]
    for key in ["a11y_tree", "accessibility_tree"]:
        if key in obs and isinstance(obs[key], str):
            summary[key] = f"{len(obs[key])} chars"
    return json.dumps(summary, ensure_ascii=False) if summary else "Minimal screenshot-only observation."


def history_text(trajectory: list[StepRecord], history_window: int) -> str:
    if not trajectory:
        return "No previous steps."

    lines = []
    for step in trajectory[-history_window:]:
        lines.append(
            f"Step {step.step_index}: actions={'; '.join(step.actions)} | "
            f"obs={truncate_text(step.observation_summary, 180)} | "
            f"reasoning={truncate_text(step.reasoning, 220)}"
        )
    return "\n".join(lines)
