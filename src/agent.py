import base64
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, FilePart, FileWithBytes, Message, Part, TaskState, TextPart
from a2a.utils import new_agent_text_message

from messenger import Messenger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../osworld"))
from mm_agents.qwen3vl_agent import Qwen3VLAgent


logger = logging.getLogger("osworld_impl.agent")


EXECUTOR_MODEL = os.environ.get("EXECUTOR_MODEL", os.environ.get("MODEL", "qwen/qwen3.5-plus-02-15"))
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "google/gemini-3.1-flash-lite-preview")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1500"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.5"))
MAX_TRAJECTORY_LENGTH = int(os.environ.get("MAX_TRAJECTORY_LENGTH", "12"))
A11Y_TREE_MAX_TOKENS = int(os.environ.get("A11Y_TREE_MAX_TOKENS", "10000"))
PLANNER_REVIEW_INTERVAL = int(os.environ.get("PLANNER_REVIEW_INTERVAL", "12"))
TRAJECTORY_SUMMARY_WINDOW = int(os.environ.get("TRAJECTORY_SUMMARY_WINDOW", "8"))
QWEN_API_BACKEND = os.environ.get("QWEN_API_BACKEND", "openai")
QWEN_ENABLE_THINKING = os.environ.get("QWEN_ENABLE_THINKING", "0").lower() in {"1", "true", "yes", "on"}
QWEN_THINKING_BUDGET = int(os.environ.get("QWEN_THINKING_BUDGET", "32768"))


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
    executor: Qwen3VLAgent | None = None
    planner_backend: Qwen3VLAgent | None = None


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

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Planning the next desktop action..."),
        )

        current_obs = self.state.last_observation
        if "screenshot" not in current_obs:
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text="Observation is missing a screenshot; returning WAIT so the planner can retry on the next frame.")),
                    Part(root=DataPart(data={"actions": ["WAIT"], "current_goal": self.state.current_goal})),
                ],
                name="prediction",
            )
            return

        decision = self._maybe_refresh_plan(current_obs)

        if decision.task_completed:
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=decision.reasoning or "Task appears complete.")),
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
                    Part(root=TextPart(text=decision.reasoning or "Planner marked the task as blocked.")),
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
        executor = self.state.executor
        assert executor is not None
        response, actions = executor.predict(executor_instruction, current_obs)
        if not actions:
            actions = ["WAIT"]

        action_summary = self._summarize_actions(actions)
        record = StepRecord(
            step_index=self.state.total_steps + 1,
            goal=self.state.current_goal,
            executor_prompt=executor_instruction,
            executor_response=response or "",
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
                Part(root=TextPart(text=response or f"Current goal: {self.state.current_goal}")),
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

        return ParsedInput(instruction=instruction, observation=obs, env_config=env_config)

    def _merge_input(self, parsed: ParsedInput) -> None:
        instruction_changed = bool(parsed.instruction and parsed.instruction != self.state.instruction)
        env_changed = bool(parsed.env_config and parsed.env_config != self.state.env_config)

        if instruction_changed:
            self.state = AgentState(instruction=parsed.instruction, env_config=parsed.env_config or self.state.env_config)
        else:
            if parsed.instruction:
                self.state.instruction = parsed.instruction
            if parsed.env_config:
                self.state.env_config = parsed.env_config

        if self.state.executor is None or instruction_changed or env_changed:
            self.state.executor = self._new_qwen_backend(history_n=MAX_TRAJECTORY_LENGTH)
        if self.state.planner_backend is None or instruction_changed or env_changed:
            self.state.planner_backend = self._new_qwen_backend(history_n=1)

        if parsed.observation:
            self.state.last_observation = parsed.observation

    def _new_qwen_backend(self, history_n: int) -> Qwen3VLAgent:
        backend = Qwen3VLAgent(
            model=EXECUTOR_MODEL,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            action_space="pyautogui",
            observation_type="screenshot",
            history_n=history_n,
            api_backend=QWEN_API_BACKEND,
            enable_thinking=QWEN_ENABLE_THINKING,
            thinking_budget=QWEN_THINKING_BUDGET,
        )
        backend.reset(logger)
        return backend

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
        self.state.current_goal = decision.goal or self.state.current_goal or self.state.instruction
        self.state.plan_reasoning = decision.reasoning
        self.state.executor_guidance = decision.executor_guidance
        self.state.steps_since_plan = 0
        return decision

    def _summarize_trajectory(self) -> str:
        if not self.state.trajectory:
            return "No prior trajectory yet. This is the first execution step for the current task."

        recent_steps = self.state.trajectory[-TRAJECTORY_SUMMARY_WINDOW:]
        trajectory_payload = []
        for step in recent_steps:
            trajectory_payload.append(
                {
                    "step": step.step_index,
                    "goal": step.goal,
                    "action_summary": step.action_summary,
                    "actions": step.actions,
                    "executor_response": _truncate_text(step.executor_response, 600),
                    "observation_summary": step.observation_summary,
                }
            )

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
            raw = self._call_llm(prompt)
            data = _extract_json_object(raw)
            if not data:
                raise ValueError("summary response was not valid JSON")
            summary = data.get("summary", "")
            status = data.get("status", "on_track")
            issues = data.get("issues", [])
            evidence = data.get("evidence", [])
            return json.dumps(
                {
                    "summary": summary,
                    "status": status,
                    "issues": issues,
                    "evidence": evidence,
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

    def _review_and_plan(self, trajectory_summary: str, current_obs: dict[str, Any]) -> PlanDecision:
        observation_text = self._planner_observation_text(current_obs)
        message_content: list[dict[str, Any]] = [
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
                    "Return strict JSON with keys goal, reasoning, continue_current_goal, task_completed, failure, executor_guidance. "
                    "executor_guidance should be a short instruction for the low-level executor."
                ),
            }
        ]
        if "screenshot" in current_obs:
            message_content.append(self._image_message(current_obs["screenshot"]))

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
            {"role": "user", "content": message_content},
        ]

        try:
            raw = self._call_llm(prompt)
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

    def _build_executor_instruction(self, decision: PlanDecision) -> str:
        sections = [
            f"Overall task: {self.state.instruction}",
            f"Current subgoal: {self.state.current_goal}",
        ]
        if decision.executor_guidance:
            sections.append(f"Planner guidance: {decision.executor_guidance}")
        if self.state.last_trajectory_summary:
            sections.append(f"Recent trajectory summary: {self.state.last_trajectory_summary}")
        sections.append(
            "Choose exactly one next desktop action that best advances the current subgoal. "
            "If the subgoal is already satisfied, terminate successfully. If the UI needs time to update, wait."
        )
        return "\n\n".join(sections)

    def _call_llm(self, messages: list[dict[str, Any]]) -> str:
        assert self.state.planner_backend is not None
        return self.state.planner_backend.call_llm(
            {
                "model": PLANNER_MODEL,
                "messages": messages,
                "max_tokens": MAX_TOKENS,
                "top_p": TOP_P,
                "temperature": TEMPERATURE,
            },
            PLANNER_MODEL,
        )

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
                    structured[key] = json.loads(json.dumps(value, ensure_ascii=False, default=str))
                except Exception:
                    structured[key] = str(value)
        if not structured:
            return "No structured observation fields besides the screenshot."
        return json.dumps(structured, ensure_ascii=False, indent=2)

    def _image_message(self, screenshot_bytes: bytes) -> dict[str, Any]:
        encoded = base64.b64encode(screenshot_bytes).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        }

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
        return json.dumps(summary, ensure_ascii=False) if summary else "Minimal screenshot-only observation."

    def _summarize_actions(self, actions: list[str]) -> str:
        cleaned = []
        for action in actions:
            cleaned.append(action if len(action) <= 200 else action[:200] + "...")
        return "; ".join(cleaned)

    def _looks_stuck(self) -> bool:
        recent = self.state.trajectory[-3:]
        if len(recent) < 3:
            return False
        recent_summaries = [step.action_summary for step in recent]
        if len(set(recent_summaries)) == 1:
            return True
        waits = sum(1 for step in recent if any(action == "WAIT" for action in step.actions))
        return waits >= 2
