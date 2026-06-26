from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile

from backend.parsing import parse_pdf_to_markdown


app = FastAPI(title="DataCon AI Agent MVP", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
