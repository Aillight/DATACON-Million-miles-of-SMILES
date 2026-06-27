from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from agent.extraction_graph import (
    ExtractionBatch,
    ExtractionGraphState,
    ExtractorFn,
    run_small_molecule_extraction,
)
from agent.nano_extraction_graph import (
    NanoExtractionGraphState,
    NanoExtractorFn,
    NanozymeExtractionBatch,
    run_nanozyme_extraction,
)
from backend.aggregation import aggregate_extraction_rows
from backend.nano_aggregation import aggregate_nanozyme_rows
from backend.parsing.pdf_parser import parse_pdf_to_markdown
from backend.parsing.text_preparation import PreparedDocument, prepare_markdown_for_extraction
from backend.vision.cv_recognition import VisionAnalysisResult, analyze_pdf_pages


LogFn = Callable[[str], None]
ExtractorFactory = Callable[[str], ExtractorFn | NanoExtractorFn | None]
NANO_HINT_TOKENS = (
    "nanozyme",
    "nanoparticle",
    "nanoparticles",
    "nanomaterial",
    "microsphere",
    "microspheres",
    "oxidase-like",
    "catalase-like",
    "km",
    "vmax",
)


@dataclass
class ArticlePipelineResult:
    source: str
    domain: str
    prepared: PreparedDocument
    table_count: int
    extraction_states: list[ExtractionGraphState | NanoExtractionGraphState] = field(default_factory=list)
    clean: pd.DataFrame = field(default_factory=pd.DataFrame)
    conflicts: pd.DataFrame = field(default_factory=pd.DataFrame)
    vision_results: list[VisionAnalysisResult] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)

    @property
    def validated_count(self) -> int:
        return sum(len(state.get("validated_objects", [])) for state in self.extraction_states)


def run_pdf_pipeline(
    pdf_path: str | Path,
    filename: str,
    domain: str,
    extractor: ExtractorFn | NanoExtractorFn | None = None,
    extractor_factory: ExtractorFactory | None = None,
    max_chunks: int = 3,
    max_attempts: int = 3,
    vision_enabled: bool = False,
    vision_scale_label: str | None = None,
    vision_max_pages: int = 1,
    vision_dpi: int = 200,
    log: LogFn | None = None,
) -> ArticlePipelineResult:
    logs: list[str] = []

    def emit(message: str) -> None:
        logs.append(message)
        if log:
            log(message)

    emit(f"Loaded PDF: {filename}")
    parsed = parse_pdf_to_markdown(pdf_path)
    emit(f"Docling/Camelot parsed markdown chars={len(parsed.markdown)} tables={len(parsed.tables)}")
    for warning in parsed.warnings:
        emit(f"Parser warning: {warning}")

    prepared = prepare_markdown_for_extraction(
        markdown=parsed.markdown,
        source=filename,
        parser=parsed.parser,
        warnings=parsed.warnings,
    )
    emit(f"Prepared text chars={len(prepared.markdown)} chunks={len(prepared.chunks)}")
    emit(f"Concentration mentions={len(prepared.concentration_mentions)}")

    extraction_states: list[ExtractionGraphState | NanoExtractionGraphState] = []
    effective_domain = resolve_effective_domain(domain, prepared) if extractor_factory else domain
    if effective_domain != domain:
        emit(f"Domain auto-switch: selected={domain} effective={effective_domain}")
    nanozyme_domain = is_nanozyme_domain(effective_domain)
    if extractor_factory:
        extractor_fn = extractor_factory(effective_domain)
    else:
        extractor_fn = extractor
    extractor_fn = extractor_fn or (empty_nano_extractor if nanozyme_domain else empty_extractor)
    chunks_to_process = prepared.chunks[:max(0, max_chunks)]
    extraction_kind = "nanozyme" if nanozyme_domain else "small-molecule"
    emit(f"Extractor kind={extraction_kind} input chunks={len(chunks_to_process)}")

    extracted_rows: list[dict[str, Any]] = []
    for chunk in chunks_to_process:
        if nanozyme_domain:
            state = run_nanozyme_extraction(
                chunk_text=chunk.text,
                extractor=extractor_fn,
                max_attempts=max_attempts,
            )
        else:
            state = run_small_molecule_extraction(
                chunk_text=chunk.text,
                extractor=extractor_fn,
                max_attempts=max_attempts,
            )
        extraction_states.append(state)
        candidates = len(state.get("extracted_objects", []))
        validated = len(state.get("validated_objects", []))
        emit(
            f"Chunk {chunk.index}: candidates={candidates} validated={validated} "
            f"attempts={state.get('attempts', 0)} status={state.get('status', 'unknown')}"
        )
        for error in state.get("error_log", []):
            validator_name = "Formula/sanity" if nanozyme_domain else "RDKit"
            emit(f"{validator_name}/error: {error}")

        extracted_rows.extend(
            enrich_validated_rows(
                state.get("validated_objects", []),
                article_id=filename,
                chunk_index=chunk.index,
                evidence=chunk.text,
            )
        )

    aggregation = aggregate_nanozyme_rows(extracted_rows) if nanozyme_domain else aggregate_extraction_rows(extracted_rows)
    emit(f"Aggregation clean_rows={len(aggregation.clean)} conflicts={len(aggregation.conflicts)}")

    vision_results: list[VisionAnalysisResult] = []
    if vision_enabled:
        vision_output_dir = Path("outputs/vision/ui") / safe_filename(Path(filename).stem)
        try:
            vision_results = analyze_pdf_pages(
                pdf_path=pdf_path,
                output_dir=vision_output_dir,
                scale_label=vision_scale_label or None,
                dpi=vision_dpi,
                max_pages=vision_max_pages,
                crop_panels=True,
            )
            emit(
                "Vision "
                f"panels={len(vision_results)} "
                f"scale_bars={sum(1 for item in vision_results if item.scale_bar is not None)} "
                f"particles={sum(item.particle_summary.count for item in vision_results)}"
            )
            for warning in collect_vision_warnings(vision_results):
                emit(f"Vision warning: {warning}")
        except Exception as exc:
            emit(f"Vision error: {exc}")

    return ArticlePipelineResult(
        source=filename,
        domain=effective_domain,
        prepared=prepared,
        table_count=len(parsed.tables),
        extraction_states=extraction_states,
        clean=aggregation.clean,
        conflicts=aggregation.conflicts,
        vision_results=vision_results,
        logs=logs,
    )


def enrich_validated_rows(
    rows: list[dict[str, Any]],
    article_id: str,
    chunk_index: int,
    evidence: str,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        enriched_row = dict(row)
        enriched_row.setdefault("source_type", "text")
        enriched_row.setdefault("article_id", article_id)
        enriched_row.setdefault("source_id", f"{article_id}:chunk-{chunk_index}")
        enriched_row.setdefault("chunk_index", chunk_index)
        enriched_row.setdefault("evidence", evidence[:1000])
        enriched.append(enriched_row)
    return enriched


def empty_extractor(_context) -> ExtractionBatch:
    return ExtractionBatch(rows=[])


def empty_nano_extractor(_context) -> NanozymeExtractionBatch:
    return NanozymeExtractionBatch(rows=[])


def is_nanozyme_domain(domain: str) -> bool:
    return "nano" in domain.lower()


def resolve_effective_domain(domain: str, prepared: PreparedDocument) -> str:
    if is_nanozyme_domain(domain):
        return domain
    if looks_like_nanozyme_text(prepared.markdown):
        return "Nanozymes"
    return domain


def explain_empty_results(result: ArticlePipelineResult) -> list[str]:
    if not result.clean.empty:
        return []

    messages: list[str] = []
    failed_states = [state for state in result.extraction_states if state.get("status") == "failed"]
    candidate_count = sum(len(state.get("extracted_objects", [])) for state in result.extraction_states)
    validated_count = result.validated_count

    if failed_states:
        messages.append("Extractor failed on at least one chunk; open Log for the exact provider or validation error.")
    elif not result.extraction_states:
        messages.append("No chunks were sent to the extractor.")
    elif candidate_count == 0:
        messages.append("Extractor returned zero candidate rows for the processed chunks.")
    elif validated_count == 0:
        messages.append("Extractor found candidates, but the validator rejected all of them.")
    else:
        messages.append("Validated rows exist, but aggregation removed them as incomplete or non-numeric.")

    if not is_nanozyme_domain(result.domain) and looks_like_nanozyme_document(result):
        messages.append(
            "This document looks like a nano/nanoparticle article; choose the Nanozymes domain and run again."
        )
    if is_nanozyme_domain(result.domain) and has_small_molecule_columns(result):
        messages.append("The result table has small-molecule columns; rerun after selecting the Nanozymes domain.")

    recent_errors = collect_recent_errors(result, limit=3)
    if recent_errors:
        if any(is_socket_permission_error(error) for error in recent_errors):
            messages.append(
                "Network access is blocked for the Streamlit process; restart the app with network access before using Hugging Face."
            )
        messages.append("Recent errors: " + " | ".join(recent_errors))

    return messages


def looks_like_nanozyme_document(result: ArticlePipelineResult) -> bool:
    text = result.prepared.markdown if result.prepared else ""
    return looks_like_nanozyme_text(text)


def looks_like_nanozyme_text(text: str) -> bool:
    text = text.lower()
    hits = sum(1 for token in NANO_HINT_TOKENS if token in text)
    has_nm_signal = " nm" in text or "nm " in text
    return hits >= 2 or (hits >= 1 and has_nm_signal)


def has_small_molecule_columns(result: ArticlePipelineResult) -> bool:
    return any(column in result.clean.columns for column in ("canonical_smiles", "smiles"))


def collect_recent_errors(result: ArticlePipelineResult, limit: int = 3) -> list[str]:
    errors: list[str] = []
    for state in result.extraction_states:
        for error in state.get("error_log", []):
            errors.append(str(error))
    return errors[-limit:]


def vision_results_to_rows(results: list[VisionAnalysisResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, result in enumerate(results, start=1):
        scale = result.scale_bar
        panel = result.panels[0] if result.panels else None
        summary = result.particle_summary
        rows.append(
            {
                "index": index,
                "source": result.source,
                "panel_kind": panel.kind if panel else "",
                "panel_bbox": panel.bbox if panel else "",
                "scale_bar_px": scale.length_px if scale else None,
                "scale_label_nm": scale.label_nm if scale else None,
                "nm_per_px": scale.nm_per_px if scale else None,
                "particle_count": summary.count,
                "mean_diameter_nm": summary.mean_diameter_nm,
                "median_diameter_nm": summary.median_diameter_nm,
                "std_diameter_nm": summary.std_diameter_nm,
                "mean_diameter_px": summary.mean_diameter_px,
                "warnings": " | ".join(result.warnings),
            }
        )
    return rows


def csv_download_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def export_dataframe_csv(
    frame: pd.DataFrame,
    source: str,
    suffix: str,
    output_dir: str | Path = Path("outputs/ui_exports"),
) -> Path:
    output_path = Path(output_dir) / f"{safe_filename(Path(source).stem)}_{safe_filename(suffix)}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path.resolve()


def collect_vision_warnings(results: list[VisionAnalysisResult], limit: int = 5) -> list[str]:
    warnings: list[str] = []
    for result in results:
        warnings.extend(result.warnings)
    return list(dict.fromkeys(warnings))[:limit]


def safe_filename(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in ("-", "_", ".") else "_" for character in value)
    return cleaned[:80].strip("._") or "article"


def is_socket_permission_error(message: str) -> bool:
    lowered = message.lower()
    return "winerror 10013" in lowered or "access to a socket" in lowered or "доступа к сокету" in lowered
