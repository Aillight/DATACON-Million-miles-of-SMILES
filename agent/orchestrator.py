from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


AgentHandler = Callable[["AgentState"], dict[str, Any]]


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str
    handler: AgentHandler


@dataclass
class AgentState:
    task: str
    inputs: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MASOrchestrator:
    def __init__(self, agents: list[AgentSpec]) -> None:
        if not agents:
            raise ValueError("At least one agent is required.")
        self.agents = agents

    def run(self, task: str, inputs: dict[str, Any] | None = None) -> AgentState:
        state = AgentState(task=task, inputs=inputs or {})
        for agent in self.agents:
            output = agent.handler(state)
            state.memory[agent.name] = output
            state.trace.append(
                {
                    "agent": agent.name,
                    "role": agent.role,
                    "summary": output.get("summary", "Completed step."),
                }
            )
        return state
