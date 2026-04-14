import logging
import os
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import new_agent_text_message

from llm import call_model, extract_json_object
from messages import (
    history_text,
    image_message,
    merge_input,
    parse_message,
    planner_observation_text,
    summarize_observation,
)
from prompts import build_react_system_prompt, build_react_user_text
from state import AgentState, StepRecord
from tools import has_terminal_action, normalize_actions, open_terminal_action, run_command_action


logger = logging.getLogger("osworld_impl.agent")


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
EXECUTOR_MODEL = os.environ.get("EXECUTOR_MODEL", os.environ.get("MODEL", "qwen/qwen3.5-plus-02-15"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1500"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.1"))
MODEL_TIMEOUT_SECONDS = float(os.environ.get("MODEL_TIMEOUT_SECONDS", "120"))
MODEL_MAX_RETRIES = int(os.environ.get("MODEL_MAX_RETRIES", "1"))
MODEL_RETRY_BACKOFF_SECONDS = float(os.environ.get("MODEL_RETRY_BACKOFF_SECONDS", "1.5"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "30"))
HISTORY_WINDOW = int(os.environ.get("HISTORY_WINDOW", "8"))
A11Y_TREE_MAX_CHARS = int(os.environ.get("A11Y_TREE_MAX_TOKENS", "10000"))
TERMINAL_RECOVERY_ENABLED = os.environ.get("TERMINAL_RECOVERY_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class Agent:
    def __init__(self):
        self.state = AgentState()
        self._stuck_recovery_count = 0

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        parsed = parse_message(message)
        previous_instruction = self.state.instruction
        self.state = merge_input(self.state, parsed)
        if self.state.instruction != previous_instruction:
            self._stuck_recovery_count = 0

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
                    Part(root=TextPart(text=f"Reached maximum step limit ({MAX_STEPS}). Stopping.")),
                    Part(root=DataPart(data={"actions": ["FAIL"], "total_steps": self.state.total_steps})),
                ],
                name="action",
            )
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Reasoning about next action..."),
        )

        current_obs = self.state.last_observation
        if "screenshot" not in current_obs:
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text="Observation missing screenshot; returning WAIT.")),
                    Part(root=DataPart(data={"actions": ["WAIT"], "total_steps": self.state.total_steps})),
                ],
                name="action",
            )
            return

        react_result = self._execute_react(current_obs)
        reasoning = str(react_result.get("response", "")).strip()
        raw_actions = react_result.get("actions", [])
        if isinstance(raw_actions, dict):
            raw_actions = [raw_actions]
        if not isinstance(raw_actions, list):
            raw_actions = []

        actions = self._normalize_actions(raw_actions)
        if not actions:
            actions = ["WAIT"]
        actions, recovery_note = self._apply_stuck_recovery(actions)
        if recovery_note:
            reasoning = f"{reasoning}\n\n{recovery_note}".strip()

        record = StepRecord(
            step_index=self.state.total_steps + 1,
            reasoning=reasoning,
            actions=actions,
            observation_summary=summarize_observation(current_obs),
        )
        self.state.trajectory.append(record)
        if HISTORY_WINDOW > 0:
            self.state.trajectory = self.state.trajectory[-HISTORY_WINDOW:]
        self.state.total_steps += 1

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=reasoning or "Computed next action.")),
                Part(
                    root=DataPart(
                        data={
                            "actions": actions,
                            "total_steps": self.state.total_steps,
                            "history_window": HISTORY_WINDOW,
                        }
                    )
                ),
            ],
            name="action",
        )

    def _execute_react(self, current_obs: dict[str, Any]) -> dict[str, Any]:
        user_text = build_react_user_text(
            instruction=self.state.instruction,
            history_text=history_text(self.state.trajectory, HISTORY_WINDOW),
            observation_text=planner_observation_text(current_obs, A11Y_TREE_MAX_CHARS),
            stuck_hint=self._stuck_hint(),
            history_window=HISTORY_WINDOW,
        )
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": build_react_system_prompt()}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    image_message(current_obs["screenshot"]),
                ],
            },
        ]

        try:
            raw = call_model(
                model=EXECUTOR_MODEL,
                messages=messages,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                max_tokens=MAX_TOKENS,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                timeout_seconds=MODEL_TIMEOUT_SECONDS,
                max_retries=MODEL_MAX_RETRIES,
                retry_backoff_seconds=MODEL_RETRY_BACKOFF_SECONDS,
            )
            data = extract_json_object(raw)
            if not isinstance(data, dict):
                raise ValueError("ReAct response was not valid JSON")

            actions = data.get("actions")
            if actions is None and "action" in data:
                actions = [data["action"]]
            if actions is None and ("name" in data or "arguments" in data):
                actions = [data]
            if isinstance(actions, (str, dict)):
                actions = [actions]
            if not isinstance(actions, list):
                actions = ["WAIT"]

            return {
                "response": str(data.get("reasoning", "")).strip(),
                "actions": actions[:3],
            }
        except Exception as exc:
            logger.warning("ReAct call failed, falling back to WAIT: %s", exc)
            err = f"{type(exc).__name__}: {exc}".strip()
            return {
                "response": f"ReAct fallback after model error: {err}",
                "actions": ["WAIT"],
            }

    def _normalize_actions(self, raw_actions: list[Any]) -> list[str]:
        return normalize_actions(raw_actions)

    def _looks_stuck(self) -> bool:
        recent = self.state.trajectory[-3:]
        if len(recent) < 3:
            return False

        action_signatures = ["|".join(step.actions) for step in recent]
        if len(set(action_signatures)) == 1:
            return True

        waits = sum(1 for step in recent if any(action == "WAIT" for action in step.actions))
        return waits >= 2

    def _stuck_hint(self) -> str:
        if not self._looks_stuck():
            return ""
        return (
            "Potentially stuck: recent actions are repetitive. "
            "Switch strategy. Prefer terminal fallback via open_terminal/run_command."
        )

    def _apply_stuck_recovery(self, actions: list[str]) -> tuple[list[str], str]:
        if not TERMINAL_RECOVERY_ENABLED:
            return actions, ""

        if not self._looks_stuck():
            self._stuck_recovery_count = 0
            return actions, ""

        if has_terminal_action(actions):
            return actions, ""

        self._stuck_recovery_count += 1
        if self._stuck_recovery_count == 1:
            return [open_terminal_action()], "Stuck recovery: forcing open_terminal action."

        return [run_command_action("pwd; ls -la", show_output=True)], (
            "Stuck recovery: forcing terminal command to gather state (pwd; ls -la)."
        )
