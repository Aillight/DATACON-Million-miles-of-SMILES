from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.extraction_graph import (
    DEFAULT_HUGGINGFACE_MODEL,
    ExtractorFn,
    make_huggingface_structured_extractor,
    make_openai_structured_extractor,
    resolve_huggingface_token,
)
from agent.nano_extraction_graph import NanoExtractorFn, make_huggingface_nanozyme_extractor
from backend.vision.cv_recognition import write_particle_summary_csv, write_vision_results_json
from ui.pipeline import ArticlePipelineResult, is_nanozyme_domain, run_pdf_pipeline


DEFAULT_OUTPUT_ROOT = Path("outputs/articles")
DEFAULT_DOMAIN = "Oxazolidinones"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    extractor = build_extractor(
        extractor=args.extractor,
        use_openai=args.use_openai,
        domain=args.domain,
        hf_model=args.hf_model,
        openai_model=args.openai_model,
        max_output_tokens=args.max_output_tokens,
    )
    output_dir = args.output_dir or default_output_dir(args.pdf)

    result = run_pdf_pipeline(
        pdf_path=args.pdf,
        filename=args.pdf.name,
        domain=args.domain,
        extractor=extractor,
        max_chunks=args.max_chunks,
        max_attempts=args.max_attempts,
        vision_enabled=args.vision,
        vision_scale_label=args.vision_scale_label,
        vision_max_pages=args.vision_max_pages,
        vision_dpi=args.vision_dpi,
    )
    artifacts = write_pipeline_outputs(result, output_dir)

    print(f"source={result.source}")
    print(f"domain={result.domain}")
    print(f"tables={result.table_count}")
    print(f"chunks={len(result.prepared.chunks)}")
    print(f"processed_chunks={len(result.extraction_states)}")
    print(f"validated={result.validated_count}")
    print(f"clean_rows={len(result.clean)}")
    print(f"conflicts={len(result.conflicts)}")
    for name, path in artifacts.items():
        print(f"{name}={path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local article extraction pipeline for one PDF.")
    parser.add_argument("pdf", type=Path, help="Path to a PDF article.")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="Extraction domain label.")
    parser.add_argument("--output-dir", type=Path, help="Directory for generated artifacts.")
    parser.add_argument("--max-chunks", type=int, default=3, help="Maximum prepared chunks to send to extractor.")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum LangGraph retry attempts per chunk.")
    parser.add_argument(
        "--extractor",
        choices=("auto", "empty", "hf", "openai"),
        default="auto",
        help="Extractor backend. auto uses Hugging Face when HF_TOKEN is available, otherwise empty.",
    )
    parser.add_argument(
        "--hf-model",
        default=os.getenv("HF_MODEL", DEFAULT_HUGGINGFACE_MODEL),
        help="Hugging Face open-weight model.",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Compatibility alias for --extractor openai.",
    )
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--vision", action="store_true", help="Run CV panel/scale/particle analysis.")
    parser.add_argument("--vision-scale-label", help="Manual scale label, for example '100 nm'.")
    parser.add_argument("--vision-max-pages", type=int, default=1)
    parser.add_argument("--vision-dpi", type=int, default=200)
    return parser


def build_extractor(
    extractor: str,
    use_openai: bool,
    domain: str,
    hf_model: str,
    openai_model: str,
    max_output_tokens: int,
) -> ExtractorFn | NanoExtractorFn | None:
    if use_openai:
        extractor = "openai"
    if extractor == "auto":
        extractor = "hf" if resolve_huggingface_token() else "empty"
    if extractor == "empty":
        return None
    if extractor == "hf":
        if is_nanozyme_domain(domain):
            return make_huggingface_nanozyme_extractor(model=hf_model, max_output_tokens=max_output_tokens)
        return make_huggingface_structured_extractor(model=hf_model, max_output_tokens=max_output_tokens)
    if extractor == "openai":
        if is_nanozyme_domain(domain):
            raise ValueError("OpenAI extractor is not wired for Nanozymes yet. Use --extractor hf or empty.")
        return make_openai_structured_extractor(model=openai_model, max_output_tokens=max_output_tokens)
    raise ValueError(f"Unsupported extractor: {extractor}")


def default_output_dir(pdf_path: Path) -> Path:
    return DEFAULT_OUTPUT_ROOT / safe_filename(pdf_path.stem)


def write_pipeline_outputs(result: ArticlePipelineResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_csv = output_dir / "clean.csv"
    conflicts_csv = output_dir / "conflicts.csv"
    logs_txt = output_dir / "logs.txt"
    prepared_md = output_dir / "prepared.md"
    chunks_json = output_dir / "chunks.json"
    states_json = output_dir / "extraction_states.json"
    vision_json = output_dir / "vision_results.json"
    vision_csv = output_dir / "vision_summary.csv"
    manifest_json = output_dir / "manifest.json"

    result.clean.to_csv(clean_csv, index=False)
    result.conflicts.to_csv(conflicts_csv, index=False)
    logs_txt.write_text("\n".join(result.logs), encoding="utf-8")
    prepared_md.write_text(result.prepared.markdown, encoding="utf-8")
    chunks_json.write_text(
        json.dumps([chunk.to_dict() for chunk in result.prepared.chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    states_json.write_text(
        json.dumps(to_jsonable(result.extraction_states), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if result.vision_results:
        write_vision_results_json(vision_json, result.vision_results)
        write_particle_summary_csv(vision_csv, result.vision_results)

    artifacts = {
        "clean_csv": clean_csv,
        "conflicts_csv": conflicts_csv,
        "logs": logs_txt,
        "prepared_markdown": prepared_md,
        "chunks_json": chunks_json,
        "states_json": states_json,
        "manifest": manifest_json,
    }
    if result.vision_results:
        artifacts["vision_json"] = vision_json
        artifacts["vision_summary_csv"] = vision_csv
    manifest_json.write_text(
        json.dumps(build_manifest(result, artifacts), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return artifacts


def build_manifest(result: ArticlePipelineResult, artifacts: dict[str, Path]) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": result.source,
        "domain": result.domain,
        "parser": result.prepared.parser,
        "table_count": result.table_count,
        "chunk_count": len(result.prepared.chunks),
        "processed_chunk_count": len(result.extraction_states),
        "candidate_count": sum(len(state.get("extracted_objects", [])) for state in result.extraction_states),
        "validated_count": result.validated_count,
        "clean_rows": len(result.clean),
        "conflict_rows": len(result.conflicts),
        "vision_result_count": len(result.vision_results),
        "vision_particle_count": sum(item.particle_summary.count for item in result.vision_results),
        "concentration_mentions": len(result.prepared.concentration_mentions),
        "warnings": list(result.prepared.warnings),
        "artifacts": {name: str(path) for name, path in artifacts.items() if name != "manifest"},
    }


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    return str(value)


def safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    text = text.strip("._-")
    return text[:80] or "article"


if __name__ == "__main__":
    raise SystemExit(main())
