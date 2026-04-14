from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedInput:
    instruction: str
    observation: dict[str, Any]
    env_config: dict[str, Any]


@dataclass
class StepRecord:
    step_index: int
    reasoning: str
    actions: list[str] = field(default_factory=list)
    observation_summary: str = ""


@dataclass
class AgentState:
    instruction: str = ""
    env_config: dict[str, Any] = field(default_factory=dict)
    total_steps: int = 0
    last_observation: dict[str, Any] = field(default_factory=dict)
    trajectory: list[StepRecord] = field(default_factory=list)
