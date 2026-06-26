from __future__ import annotations

from typing import Any

from agent.orchestrator import AgentSpec, AgentState, MASOrchestrator
from backend.parsing import chunk_article_text


def build_default_orchestrator() -> MASOrchestrator:
    return MASOrchestrator(
        agents=[
            AgentSpec(name="planner", role="Plan the work", handler=_planner_agent),
            AgentSpec(name="chunker", role="Select relevant article context", handler=_chunker_agent),
            AgentSpec(name="synthesizer", role="Prepare final concise answer", handler=_synthesizer_agent),
        ]
    )


def run_agent_task(task: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_default_orchestrator().run(task=task, inputs=inputs).to_dict()


def _planner_agent(state: AgentState) -> dict[str, Any]:
    steps = [
        "identify requested evidence",
        "segment relevant article sections",
        "summarize selected context for downstream reasoning",
    ]
    return {"summary": "Created a deterministic execution plan.", "steps": steps}


def _chunker_agent(state: AgentState) -> dict[str, Any]:
    text = str(state.inputs.get("text", ""))
    target_sections = state.inputs.get("target_sections")
    max_chars = int(state.inputs.get("max_chars", 6000))
    overlap_chars = int(state.inputs.get("overlap_chars", 400))

    chunks = chunk_article_text(
        text=text,
        target_sections=target_sections,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    return {
        "summary": f"Selected {len(chunks)} context chunks.",
        "chunks": [chunk.to_dict() for chunk in chunks],
    }


def _synthesizer_agent(state: AgentState) -> dict[str, Any]:
    chunker_output = state.memory.get("chunker", {})
    chunks = chunker_output.get("chunks", [])
    section_titles = sorted({chunk["section_title"] for chunk in chunks})
    return {
        "summary": "Prepared response context for the next LLM step.",
        "selected_sections": section_titles,
        "context": "\n\n".join(chunk["text"] for chunk in chunks),
    }
