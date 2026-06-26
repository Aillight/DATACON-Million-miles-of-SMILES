from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from agent.core import run_agent_task
from backend.parsing import chunk_article_text, parse_pdf_to_markdown


app = FastAPI(title="DataCon AI Agent MVP", version="0.1.0")


class ChunkTextRequest(BaseModel):
    text: str
    target_sections: list[str] | None = None
    max_chars: int = Field(default=6000, ge=500)
    overlap_chars: int = Field(default=400, ge=0)


class AgentRunRequest(ChunkTextRequest):
    task: str = "Prepare article context for answer generation."


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chunk/text")
def chunk_text(request: ChunkTextRequest) -> dict[str, Any]:
    chunks = chunk_article_text(
        text=request.text,
        target_sections=request.target_sections,
        max_chars=request.max_chars,
        overlap_chars=request.overlap_chars,
    )
    return {"chunks": [chunk.to_dict() for chunk in chunks]}


@app.post("/agent/run")
def run_agent(request: AgentRunRequest) -> dict[str, Any]:
    return run_agent_task(
        task=request.task,
        inputs={
            "text": request.text,
            "target_sections": request.target_sections,
            "max_chars": request.max_chars,
            "overlap_chars": request.overlap_chars,
        },
    )


@app.post("/parse/pdf")
async def parse_pdf(file: UploadFile = File(...)) -> dict:
    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(payload)
            temp_path = Path(temp_file.name)

        parsed = parse_pdf_to_markdown(temp_path)
        result = parsed.to_dict()
        result["source"] = filename
        result["chunks"] = [chunk.to_dict() for chunk in chunk_article_text(parsed.markdown)]
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
