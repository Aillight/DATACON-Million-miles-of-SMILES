from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


TraceLogFn = Callable[[str], None]


@dataclass
class AgentTraceStep:
    agent: str
    role: str
    status: str
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArticleSupervisor:
    """Records the article-level multi-agent execution trace."""

    def __init__(self, log: TraceLogFn | None = None) -> None:
        self._log = log
        self.steps: list[AgentTraceStep] = []

    def record(
        self,
        agent: str,
        role: str,
        summary: str,
        *,
        status: str = "completed",
        metrics: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> AgentTraceStep:
        step = AgentTraceStep(
            agent=agent,
            role=role,
            status=status,
            summary=summary,
            metrics=metrics or {},
            warnings=warnings or [],
        )
        self.steps.append(step)
        if self._log:
            self._log(f"{agent}: {summary}")
        return step

    def to_dicts(self) -> list[dict[str, Any]]:
        return [step.to_dict() for step in self.steps]
