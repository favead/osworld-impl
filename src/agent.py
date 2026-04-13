import base64
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import (
    DataPart,
    FilePart,
    FileWithBytes,
    Message,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message

from messenger import Messenger


logger = logging.getLogger("osworld_impl.agent")

PYAUTOGUI_FUNCS = {
    "click",
    "doubleClick",
    "tripleClick",
    "rightClick",
    "middleClick",
    "moveTo",
    "move",
    "dragTo",
    "drag",
    "scroll",
    "hscroll",
    "typewrite",
    "write",
    "press",
    "keyDown",
    "keyUp",
    "hotkey",
}


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
EXECUTOR_MODEL = os.environ.get(
    "EXECUTOR_MODEL", os.environ.get("MODEL", "qwen/qwen3.5-plus-02-15")
)
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "google/gemini-3.1-flash-lite-preview")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1500"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.1"))
MAX_TRAJECTORY_LENGTH = int(os.environ.get("MAX_TRAJECTORY_LENGTH", "12"))
A11Y_TREE_MAX_TOKENS = int(os.environ.get("A11Y_TREE_MAX_TOKENS", "10000"))
PLANNER_REVIEW_INTERVAL = int(os.environ.get("PLANNER_REVIEW_INTERVAL", "12"))
TRAJECTORY_SUMMARY_WINDOW = int(os.environ.get("TRAJECTORY_SUMMARY_WINDOW", "8"))
MODEL_TIMEOUT_SECONDS = float(os.environ.get("MODEL_TIMEOUT_SECONDS", "120"))
MODEL_MAX_RETRIES = int(os.environ.get("MODEL_MAX_RETRIES", "1"))
MODEL_RETRY_BACKOFF_SECONDS = float(os.environ.get("MODEL_RETRY_BACKOFF_SECONDS", "1.5"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "30"))


@dataclass
class ParsedInput:
    instruction: str
    observation: dict[str, Any]
    env_config: dict[str, Any]


@dataclass
class StepRecord:
    step_index: int
    goal: str
    executor_prompt: str
    executor_response: str
    action_summary: str
    actions: list[str] = field(default_factory=list)
    observation_summary: str = ""


@dataclass
class PlanDecision:
    goal: str
    reasoning: str
    continue_current_goal: bool = False
    task_completed: bool = False
    failure: bool = False
    executor_guidance: str = ""


@dataclass
class AgentState:
    instruction: str = ""
    env_config: dict[str, Any] = field(default_factory=dict)
    current_goal: str = ""
    plan_reasoning: str = ""
    executor_guidance: str = ""
    steps_since_plan: int = 0
    total_steps: int = 0
    last_trajectory_summary: str = ""
    last_observation: dict[str, Any] = field(default_factory=dict)
    trajectory: list[StepRecord] = field(default_factory=list)


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[...]"


def _strip_code_fence(text: str) -> str:
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced[0].strip()
    return text.strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = _strip_code_fence(text)
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


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.state = AgentState()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        parsed = self._parse_message(message)
        self._merge_input(parsed)

        if not self.state.instruction:
            await updater.failed(
                new_agent_text_message(
                    "Missing task instruction.",
                    context_id=message.context_id,
                    task_id=getattr(message, "task_id", None),
                )
            )
            return

        if MAX_STEPS > 0 and self.state.total_steps >= MAX_STEPS:
            await updater.add_artifact(
                parts=[
                    Part(
                        root=TextPart(
                            text=f"Reached maximum step limit ({MAX_STEPS}). Stopping."
                        )
                    ),
                    Part(
                        root=DataPart(
                            data={
                                "actions": ["FAIL"],
                                "current_goal": self.state.current_goal,
                                "trajectory_summary": self.state.last_trajectory_summary,
                                "plan_reasoning": f"Exceeded max_steps limit of {MAX_STEPS}",
                            }
                        )
                    ),
                ],
                name="prediction",
            )
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Planning the next desktop action..."),
        )

        current_obs = self.state.last_observation
        if "screenshot" not in current_obs:
            await updater.add_artifact(
                parts=[
                    Part(
                        root=TextPart(
                            text="Observation is missing a screenshot; returning WAIT so the planner can retry on the next frame."
                        )
                    ),
                    Part(
                        root=DataPart(
                            data={
                                "actions": ["WAIT"],
                                "current_goal": self.state.current_goal,
                            }
                        )
                    ),
                ],
                name="prediction",
            )
            return

        decision = self._maybe_refresh_plan(current_obs)

        if decision.task_completed:
            await updater.add_artifact(
                parts=[
                    Part(
                        root=TextPart(
                            text=decision.reasoning or "Task appears complete."
                        )
                    ),
                    Part(
                        root=DataPart(
                            data={
                                "actions": ["DONE"],
                                "current_goal": self.state.current_goal,
                                "trajectory_summary": self.state.last_trajectory_summary,
                                "plan_reasoning": decision.reasoning,
                            }
                        )
                    ),
                ],
                name="prediction",
            )
            return

        if decision.failure:
            await updater.add_artifact(
                parts=[
                    Part(
                        root=TextPart(
                            text=decision.reasoning
                            or "Planner marked the task as blocked."
                        )
                    ),
                    Part(
                        root=DataPart(
                            data={
                                "actions": ["FAIL"],
                                "current_goal": self.state.current_goal,
                                "trajectory_summary": self.state.last_trajectory_summary,
                                "plan_reasoning": decision.reasoning,
                            }
                        )
                    ),
                ],
                name="prediction",
            )
            return

        executor_instruction = self._build_executor_instruction(decision)
        executor_result = self._execute_action(executor_instruction, current_obs)
        response = executor_result.get("response", "")
        actions = executor_result.get("actions", [])
        if isinstance(actions, dict):
            actions = [actions]
        if not isinstance(actions, list):
            actions = []
        actions = self._normalize_actions(actions)
        if not actions:
            actions = ["WAIT"]

        action_summary = self._summarize_actions(actions)
        record = StepRecord(
            step_index=self.state.total_steps + 1,
            goal=self.state.current_goal,
            executor_prompt=executor_instruction,
            executor_response=response,
            action_summary=action_summary,
            actions=actions,
            observation_summary=self._summarize_observation(current_obs),
        )
        self.state.trajectory.append(record)
        self.state.trajectory = self.state.trajectory[-MAX_TRAJECTORY_LENGTH:]
        self.state.total_steps += 1
        self.state.steps_since_plan += 1

        await updater.add_artifact(
            parts=[
                Part(
                    root=TextPart(
                        text=response or f"Current goal: {self.state.current_goal}"
                    )
                ),
                Part(
                    root=DataPart(
                        data={
                            "actions": actions,
                            "current_goal": self.state.current_goal,
                            "plan_reasoning": self.state.plan_reasoning,
                            "executor_guidance": self.state.executor_guidance,
                            "trajectory_summary": self.state.last_trajectory_summary,
                            "steps_since_plan": self.state.steps_since_plan,
                            "total_steps": self.state.total_steps,
                        }
                    )
                ),
            ],
            name="prediction",
        )

    def _parse_message(self, message: Message) -> ParsedInput:
        instruction = ""
        obs: dict[str, Any] = {}
        env_config: dict[str, Any] = {}

        for part in message.parts:
            root = part.root
            if isinstance(root, TextPart) and root.text.strip():
                instruction = root.text.strip()
            elif isinstance(root, FilePart) and isinstance(root.file, FileWithBytes):
                obs["screenshot"] = base64.b64decode(root.file.bytes)
            elif isinstance(root, DataPart):
                payload = dict(root.data)
                if "env_config" in payload:
                    raw_env = payload.pop("env_config")
                    if isinstance(raw_env, dict):
                        env_config.update(raw_env)
                obs.update(payload)

        screenshot_value = obs.get("screenshot")
        if isinstance(screenshot_value, str):
            encoded = screenshot_value
            if encoded.startswith("data:image"):
                encoded = encoded.split(",", 1)[1]
            try:
                obs["screenshot"] = base64.b64decode(encoded)
            except Exception:
                obs.pop("screenshot", None)

        return ParsedInput(
            instruction=instruction, observation=obs, env_config=env_config
        )

    def _merge_input(self, parsed: ParsedInput) -> None:
        instruction_changed = bool(
            parsed.instruction and parsed.instruction != self.state.instruction
        )
        if instruction_changed:
            self.state = AgentState(
                instruction=parsed.instruction,
                env_config=parsed.env_config or self.state.env_config,
            )
        else:
            if parsed.instruction:
                self.state.instruction = parsed.instruction
            if parsed.env_config:
                self.state.env_config = parsed.env_config

        if parsed.observation:
            self.state.last_observation = parsed.observation

    def _maybe_refresh_plan(self, current_obs: dict[str, Any]) -> PlanDecision:
        force_review = (
            not self.state.current_goal
            or self.state.steps_since_plan >= PLANNER_REVIEW_INTERVAL
            or self._looks_stuck()
        )

        if not force_review:
            return PlanDecision(
                goal=self.state.current_goal,
                reasoning=self.state.plan_reasoning,
                continue_current_goal=True,
                executor_guidance=self.state.executor_guidance,
            )

        summary = self._summarize_trajectory()
        decision = self._review_and_plan(summary, current_obs)
        if not decision.goal and self.state.current_goal:
            decision.goal = self.state.current_goal
            decision.continue_current_goal = True

        self.state.last_trajectory_summary = summary
        self.state.current_goal = (
            decision.goal or self.state.current_goal or self.state.instruction
        )
        self.state.plan_reasoning = decision.reasoning
        self.state.executor_guidance = decision.executor_guidance
        self.state.steps_since_plan = 0
        return decision

    def _summarize_trajectory(self) -> str:
        if not self.state.trajectory:
            return "No prior trajectory yet. This is the first execution step for the current task."

        recent_steps = self.state.trajectory[-TRAJECTORY_SUMMARY_WINDOW:]
        trajectory_payload = [
            {
                "step": step.step_index,
                "goal": step.goal,
                "action_summary": step.action_summary,
                "actions": step.actions,
                "executor_response": _truncate_text(step.executor_response, 600),
                "observation_summary": step.observation_summary,
            }
            for step in recent_steps
        ]

        prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You summarize a ReAct-style desktop trajectory. "
                            "Return strict JSON with keys summary, status, issues, and evidence. "
                            "status must be one of on_track, stuck, completed, or blocked."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "task": self.state.instruction,
                                "current_goal": self.state.current_goal,
                                "trajectory": trajectory_payload,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                    }
                ],
            },
        ]

        try:
            raw = self._call_model(PLANNER_MODEL, prompt)
            data = _extract_json_object(raw)
            if not data:
                raise ValueError("summary response was not valid JSON")
            return json.dumps(
                {
                    "summary": data.get("summary", ""),
                    "status": data.get("status", "on_track"),
                    "issues": data.get("issues", []),
                    "evidence": data.get("evidence", []),
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            logger.warning("Falling back to heuristic trajectory summary: %s", exc)
            lines = [f"Task: {self.state.instruction}"]
            if self.state.current_goal:
                lines.append(f"Current goal: {self.state.current_goal}")
            for step in recent_steps:
                lines.append(f"Step {step.step_index}: {step.action_summary}")
            return "\n".join(lines)

    def _review_and_plan(
        self, trajectory_summary: str, current_obs: dict[str, Any]
    ) -> PlanDecision:
        observation_text = self._planner_observation_text(current_obs)
        content = [
            {
                "type": "text",
                "text": (
                    "Task:\n"
                    f"{self.state.instruction}\n\n"
                    "Current subgoal:\n"
                    f"{self.state.current_goal or 'None yet'}\n\n"
                    "Trajectory summary:\n"
                    f"{trajectory_summary}\n\n"
                    "Structured observation:\n"
                    f"{observation_text}\n\n"
                    "Decide whether to keep or replace the current subgoal. "
                    "Return strict JSON with keys goal, reasoning, continue_current_goal, task_completed, failure, executor_guidance."
                ),
            }
        ]
        if "screenshot" in current_obs:
            content.append(self._image_message(current_obs["screenshot"]))

        prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a desktop-task planner and verifier. "
                            "First judge whether the existing trajectory is progressing, then choose the best next subgoal. "
                            "Use the screenshot and observation text as evidence. "
                            "Do not output prose outside the JSON object."
                        ),
                    }
                ],
            },
            {"role": "user", "content": content},
        ]

        try:
            raw = self._call_model(PLANNER_MODEL, prompt)
            data = _extract_json_object(raw)
            if not data:
                raise ValueError("planner response was not valid JSON")
            return PlanDecision(
                goal=str(data.get("goal", "") or "").strip(),
                reasoning=str(data.get("reasoning", "") or "").strip(),
                continue_current_goal=bool(data.get("continue_current_goal", False)),
                task_completed=bool(data.get("task_completed", False)),
                failure=bool(data.get("failure", False)),
                executor_guidance=str(data.get("executor_guidance", "") or "").strip(),
            )
        except Exception as exc:
            logger.warning("Falling back to heuristic plan selection: %s", exc)
            fallback_goal = self.state.current_goal or self.state.instruction
            return PlanDecision(
                goal=fallback_goal,
                reasoning="Continuing the current subgoal because planner verification fell back to heuristic mode.",
                continue_current_goal=bool(self.state.current_goal),
                executor_guidance="Advance toward the visible next UI step and avoid repeating the same failed action.",
            )

    def _execute_action(
        self, executor_instruction: str, current_obs: dict[str, Any]
    ) -> dict[str, Any]:
        content = [
            {"type": "text", "text": executor_instruction},
            {
                "type": "text",
                "text": (
                    "Return strict JSON with keys response and actions. "
                    "actions must be an array of 1 to 3 items. "
                    "Each item should be either a fully-qualified pyautogui string like pyautogui.click(100, 200), "
                    "or an object like {\"action\":\"left_click\",\"coordinate\":[100,200]}. "
                    "For OSWorld compatibility, use WAIT to pause, DONE to finish successfully, FAIL to finish unsuccessfully, "
                    "or structured terminate actions like {\"action\":\"terminate\",\"status\":\"success\"}."
                ),
            },
        ]
        if "screenshot" in current_obs:
            content.append(self._image_message(current_obs["screenshot"]))
        observation_text = self._planner_observation_text(current_obs)
        if observation_text:
            content.append(
                {"type": "text", "text": f"Structured observation:\n{observation_text}"}
            )

        prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a desktop action executor. Pick the next best action for the current subgoal. "
                            "Keep actions concise and executable."
                        ),
                    }
                ],
            },
            {"role": "user", "content": content},
        ]

        try:
            raw = self._call_model(EXECUTOR_MODEL, prompt)
            data = _extract_json_object(raw)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning("Executor call failed, falling back to WAIT: %s", exc)
        return {
            "response": "Executor fallback: waiting for the next observation.",
            "actions": ["WAIT"],
        }

    def _call_model(self, model: str, messages: list[dict[str, Any]]) -> str:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required")

        base_url = OPENAI_BASE_URL.rstrip("/")
        request_url = f"{base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "top_p": TOP_P,
            "temperature": TEMPERATURE,
        }
        request = urllib.request.Request(
            request_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        attempts = max(MODEL_MAX_RETRIES, 0) + 1
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=MODEL_TIMEOUT_SECONDS) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                is_retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
                last_error = RuntimeError(
                    f"LLM request failed with HTTP {exc.code}: {error_body}"
                )
                if not is_retryable or attempt == attempts:
                    raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"LLM request failed: {exc}")
                if attempt == attempts:
                    raise last_error from exc

            sleep_seconds = MODEL_RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "LLM call failed for model %s (attempt %s/%s), retrying in %.1fs",
                model,
                attempt,
                attempts,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
        else:
            if last_error:
                raise last_error
            raise RuntimeError("LLM request failed without a specific error")

        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError(f"LLM response missing choices: {body}")
        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            return "\n".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        return str(content)

    def _planner_observation_text(self, obs: dict[str, Any]) -> str:
        structured: dict[str, Any] = {}
        for key, value in obs.items():
            if key == "screenshot":
                continue
            if key in {"a11y_tree", "accessibility_tree"} and isinstance(value, str):
                structured[key] = _truncate_text(value, A11Y_TREE_MAX_TOKENS)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                structured[key] = value
            else:
                try:
                    structured[key] = json.loads(
                        json.dumps(value, ensure_ascii=False, default=str)
                    )
                except Exception:
                    structured[key] = str(value)
        if not structured:
            return ""
        return json.dumps(structured, ensure_ascii=False, indent=2)

    def _image_message(self, screenshot_bytes: bytes) -> dict[str, Any]:
        encoded = base64.b64encode(screenshot_bytes).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        }

    def _build_executor_instruction(self, decision: PlanDecision) -> str:
        sections = [
            f"Overall task: {self.state.instruction}",
            f"Current subgoal: {self.state.current_goal}",
        ]
        if decision.executor_guidance:
            sections.append(f"Planner guidance: {decision.executor_guidance}")
        if self.state.last_trajectory_summary:
            sections.append(
                f"Recent trajectory summary: {self.state.last_trajectory_summary}"
            )
        sections.append(
            "Choose the single next desktop action that best advances the current subgoal. "
            "Do not repeat the same failed action. If the screen is still updating, use WAIT."
        )
        return "\n\n".join(sections)

    def _summarize_observation(self, obs: dict[str, Any]) -> str:
        summary: dict[str, Any] = {}
        if "screenshot" in obs and isinstance(obs["screenshot"], (bytes, bytearray)):
            summary["screenshot_bytes"] = len(obs["screenshot"])
        for key in ["app", "cur_app", "window_title", "cur_window_id"]:
            if key in obs:
                summary[key] = obs[key]
        for key in ["a11y_tree", "accessibility_tree"]:
            if key in obs and isinstance(obs[key], str):
                summary[key] = f"{len(obs[key])} chars"
        return (
            json.dumps(summary, ensure_ascii=False)
            if summary
            else "Minimal screenshot-only observation."
        )

    def _summarize_actions(self, actions: list[str]) -> str:
        cleaned = [
            action if len(action) <= 200 else action[:200] + "..." for action in actions
        ]
        return "; ".join(cleaned)

    def _normalize_actions(self, raw_actions: list[Any]) -> list[str]:
        normalized: list[str] = []
        for raw_action in raw_actions:
            if isinstance(raw_action, dict):
                mapped = self._action_from_object(raw_action)
                if mapped:
                    normalized.append(mapped)
                continue

            action = str(raw_action).strip()
            if not action:
                continue

            marker = action.upper()
            if marker in {"WAIT", "DONE", "FAIL"}:
                normalized.append(marker)
                continue

            match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(", action)
            if match:
                fn_name = match.group(1)
                if fn_name in PYAUTOGUI_FUNCS:
                    normalized.append(f"pyautogui.{action}")
                    continue

            normalized.append(action)

        return normalized

    def _action_from_object(self, action_obj: dict[str, Any]) -> str | None:
        raw_action = action_obj.get(
            "action",
            action_obj.get("type", action_obj.get("action_type", "")),
        )
        action = str(raw_action).strip().lower()
        if not action:
            return None

        if action in {"done", "success"}:
            return "DONE"
        if action in {"fail", "failed", "failure"}:
            return "FAIL"

        if action == "wait":
            return "WAIT"
        if action == "terminate":
            status = str(action_obj.get("status", "failure")).strip().lower()
            return "DONE" if status == "success" else "FAIL"

        def _coord() -> tuple[int, int] | None:
            value = action_obj.get("coordinate")
            if not isinstance(value, list) or len(value) != 2:
                return None
            try:
                return int(value[0]), int(value[1])
            except (TypeError, ValueError):
                return None

        if action in {"left_click", "click"}:
            coord = _coord()
            if coord:
                x, y = coord
                return f"pyautogui.click({x}, {y})"
            return "pyautogui.click()"

        if action == "right_click":
            coord = _coord()
            if coord:
                x, y = coord
                return f"pyautogui.rightClick({x}, {y})"
            return "pyautogui.rightClick()"

        if action == "middle_click":
            coord = _coord()
            if coord:
                x, y = coord
                return f"pyautogui.middleClick({x}, {y})"
            return "pyautogui.middleClick()"

        if action in {"double_click", "triple_click", "mouse_move", "left_click_drag"}:
            coord = _coord()
            if not coord:
                return None
            x, y = coord
            mapping = {
                "double_click": "doubleClick",
                "triple_click": "tripleClick",
                "mouse_move": "moveTo",
                "left_click_drag": "dragTo",
            }
            return f"pyautogui.{mapping[action]}({x}, {y})"

        if action == "type":
            text = action_obj.get("text")
            if isinstance(text, str) and text:
                return f"pyautogui.write({json.dumps(text)})"
            return None

        if action == "key":
            keys = action_obj.get("keys")
            if isinstance(keys, list) and keys:
                cleaned = [str(key).strip() for key in keys if str(key).strip()]
                if not cleaned:
                    return None
                if len(cleaned) == 1:
                    return f"pyautogui.press({json.dumps(cleaned[0])})"
                args = ", ".join(json.dumps(key) for key in cleaned)
                return f"pyautogui.hotkey({args})"
            return None

        if action in {"scroll", "hscroll"}:
            pixels = action_obj.get("pixels")
            try:
                amount = int(float(pixels))
            except (TypeError, ValueError):
                return None
            if action == "hscroll":
                return f"pyautogui.hscroll({amount})"
            return f"pyautogui.scroll({amount})"

        return None

    def _looks_stuck(self) -> bool:
        recent = self.state.trajectory[-3:]
        if len(recent) < 3:
            return False
        recent_summaries = [step.action_summary for step in recent]
        if len(set(recent_summaries)) == 1:
            return True
        waits = sum(
            1 for step in recent if any(action == "WAIT" for action in step.actions)
        )
        return waits >= 2
