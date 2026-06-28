from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.extraction_graph import (
    DEFAULT_HUGGINGFACE_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    ExtractorFn,
    make_huggingface_structured_extractor,
    make_openai_structured_extractor,
    make_openrouter_structured_extractor,
    resolve_openai_token,
    resolve_openrouter_token,
    resolve_huggingface_token,
)
from agent.nano_extraction_graph import (
    NanoExtractorFn,
    make_huggingface_nanozyme_extractor,
    make_openrouter_nanozyme_extractor,
)
from backend.retrieval import DEFAULT_HF_EMBEDDING_MODEL, make_huggingface_embedding_function
from ui.pipeline import (
    ArticlePipelineResult,
    csv_download_bytes,
    export_dataframe_csv,
    explain_empty_results,
    is_nanozyme_domain,
    run_pdf_pipeline,
    vision_results_to_rows,
)
from ui.temp_files import safe_unlink_path


DOMAINS = ("Oxazolidinones", "Benzimidazoles", "Nanozymes")
EXTRACTOR_OPTIONS = ("Hugging Face", "OpenRouter", "None", "OpenAI")
RETRIEVAL_OPTIONS = ("TF-IDF", "Hybrid HF embeddings")
DISPLAY_COLUMN_ORDER = {
    "agents": [
        "agent",
        "status",
        "summary",
        "role",
        "metrics",
        "warnings",
    ],
    "results": [
        "material_formula",
        "canonical_smiles",
        "property_name",
        "value",
        "unit",
        "quality_score",
        "quality_flags",
        "evidence_preview",
        "condition",
        "assay",
        "material_name",
        "source_type",
        "normalized_formula",
        "chunk_index",
        "source_id",
        "evidence",
    ],
    "rejected": [
        "reason",
        "material_formula",
        "property_name",
        "value",
        "unit",
        "evidence_preview",
        "condition",
        "assay",
        "material_name",
        "validator",
        "source_type",
        "normalized_formula",
        "chunk_index",
        "source_id",
        "evidence",
    ],
    "conflicts": [
        "normalized_formula",
        "property_name",
        "unit",
        "assay",
        "condition",
        "chosen_value",
        "chosen_source_type",
        "rejected_value",
        "rejected_source_type",
        "resolution",
        "source_id",
    ],
}


def display_frame(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = add_evidence_preview(frame)
    preferred = [column for column in DISPLAY_COLUMN_ORDER.get(kind, []) if column in frame.columns]
    remaining = [column for column in frame.columns if column not in preferred]
    return frame[preferred + remaining]


def add_evidence_preview(frame: pd.DataFrame, max_chars: int = 180) -> pd.DataFrame:
    if "evidence" not in frame.columns or "evidence_preview" in frame.columns:
        return frame
    preview_frame = frame.copy()
    preview_frame["evidence_preview"] = preview_frame["evidence"].apply(
        lambda value: str(value).replace("\n", " ")[:max_chars]
    )
    return preview_frame


def render_evidence_view(frame: pd.DataFrame) -> None:
    if frame.empty or "evidence" not in frame.columns:
        st.info("No evidence snippets for this run.")
        return

    evidence_frame = add_evidence_preview(frame)
    labels = [evidence_label(row, index) for index, row in evidence_frame.iterrows()]
    selected_label = st.selectbox("Result row", labels)
    selected_index = labels.index(selected_label)
    selected = evidence_frame.iloc[selected_index]

    summary_columns = [
        column
        for column in (
            "material_formula",
            "canonical_smiles",
            "property_name",
            "value",
            "unit",
            "quality_score",
            "source_type",
            "source_id",
        )
        if column in evidence_frame.columns
    ]
    st.dataframe(
        pd.DataFrame([selected[summary_columns].to_dict()]),
        use_container_width=True,
        hide_index=True,
    )
    st.text_area("Evidence", str(selected.get("evidence", "")), height=320)


def evidence_label(row: pd.Series, index: int) -> str:
    identity = row.get("material_formula") or row.get("canonical_smiles") or row.get("normalized_formula") or f"row {index + 1}"
    property_name = row.get("property_name") or "property"
    value = row.get("value")
    unit = row.get("unit") or ""
    return f"{index + 1}: {identity} | {property_name} = {value} {unit}".strip()


def optional_secret(value: str) -> str | None:
    cleaned = value.strip()
    return cleaned or None


def render_csv_actions(frame: pd.DataFrame, source: str, suffix: str, label: str) -> None:
    if frame.empty:
        return
    export_path = export_dataframe_csv(frame, source, suffix)
    action_cols = st.columns([1, 3])
    with action_cols[0]:
        st.download_button(
            label,
            data=csv_download_bytes(frame),
            file_name=f"{Path(source).stem}_{suffix}.csv",
            mime="text/csv",
            on_click="ignore",
            use_container_width=True,
        )
    with action_cols[1]:
        st.caption(f"Local copy: {export_path}")


st.set_page_config(page_title="DataCon Extraction Agent", layout="wide")
st.title("Scientific PDF Extraction Workbench")
hf_token_available = bool(resolve_huggingface_token())
openrouter_token_available = bool(resolve_openrouter_token())
openai_token_available = bool(resolve_openai_token())

with st.sidebar:
    st.header("Input")
    domain = st.selectbox("Domain", DOMAINS)
    uploaded_file = st.file_uploader("PDF article", type=["pdf"])
    extractor_mode = st.selectbox(
        "Extractor",
        EXTRACTOR_OPTIONS,
        index=0 if hf_token_available else (1 if openrouter_token_available else 2),
    )
    provider_status = []
    provider_status.append("HF token detected" if hf_token_available else "HF token not visible")
    provider_status.append("OpenRouter key detected" if openrouter_token_available else "OpenRouter key not visible")
    provider_status.append("OpenAI key detected" if openai_token_available else "OpenAI key not visible")
    st.caption(" | ".join(provider_status))
    with st.expander("API keys for this run", expanded=False):
        st.caption("Optional. These values stay in the current Streamlit session.")
        hf_token_input = st.text_input("HF_TOKEN", type="password", value="")
        openrouter_token_input = st.text_input("OPENROUTER_API_KEY", type="password", value="")
        openai_token_input = st.text_input("OPENAI_API_KEY", type="password", value="")
    run_clicked = st.button("Run", type="primary", use_container_width=True)
    with st.expander("Advanced", expanded=False):
        max_chunks = st.slider("Chunks", min_value=1, max_value=10, value=3)
        max_attempts = st.slider("Attempts", min_value=1, max_value=3, value=3)
        retrieval_mode_label = st.selectbox("Retrieval", RETRIEVAL_OPTIONS)
        embedding_model = st.text_input(
            "Embedding model",
            value=os.getenv("HF_EMBEDDING_MODEL", DEFAULT_HF_EMBEDDING_MODEL),
            disabled=retrieval_mode_label == "TF-IDF",
        )
        vision_enabled = st.checkbox("Vision analysis", value=is_nanozyme_domain(domain))
        vision_max_pages = st.slider("Vision pages", min_value=1, max_value=5, value=1, disabled=not vision_enabled)
        vision_scale_label = st.text_input("Scale label", value="", disabled=not vision_enabled)
        hf_model = st.text_input("HF model", value=os.getenv("HF_MODEL", DEFAULT_HUGGINGFACE_MODEL))
        openrouter_model = st.text_input(
            "OpenRouter model",
            value=os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
        )
        openai_model = st.text_input("OpenAI model", value="gpt-4.1-mini")

result: ArticlePipelineResult | None = st.session_state.get("pipeline_result")

if run_clicked:
    if uploaded_file is None:
        st.error("Upload a PDF article first.")
    else:
        temp_path: Path | None = None
        log_lines: list[str] = []
        log_box = st.empty()

        def append_log(message: str) -> None:
            log_lines.append(message)
            log_box.code("\n".join(log_lines), language="text")

        try:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = Path(temp_file.name)

            def make_extractor_for_domain(effective_domain: str) -> ExtractorFn | NanoExtractorFn | None:
                hf_token = optional_secret(hf_token_input)
                openrouter_token = optional_secret(openrouter_token_input)
                openai_token = optional_secret(openai_token_input)
                if extractor_mode == "Hugging Face":
                    if is_nanozyme_domain(effective_domain):
                        return make_huggingface_nanozyme_extractor(model=hf_model, token=hf_token)
                    return make_huggingface_structured_extractor(model=hf_model, token=hf_token)
                if extractor_mode == "OpenRouter":
                    if is_nanozyme_domain(effective_domain):
                        return make_openrouter_nanozyme_extractor(model=openrouter_model, token=openrouter_token)
                    return make_openrouter_structured_extractor(model=openrouter_model, token=openrouter_token)
                if extractor_mode == "OpenAI":
                    if is_nanozyme_domain(effective_domain):
                        raise ValueError("OpenAI extractor is not wired for Nanozymes yet. Use Hugging Face or None.")
                    return make_openai_structured_extractor(model=openai_model, token=openai_token)
                return None

            if extractor_mode == "Hugging Face":
                if not (hf_token_available or optional_secret(hf_token_input)):
                    st.error("HF_TOKEN is not visible. Add it in the sidebar or .env.")
                    st.stop()
            elif extractor_mode == "OpenRouter":
                if not (openrouter_token_available or optional_secret(openrouter_token_input)):
                    st.error("OPENROUTER_API_KEY is not visible. Add it in the sidebar or .env.")
                    st.stop()
            elif extractor_mode == "OpenAI":
                if is_nanozyme_domain(domain):
                    st.error("OpenAI extractor is not wired for Nanozymes yet. Use Hugging Face or None.")
                    st.stop()
                if not (openai_token_available or optional_secret(openai_token_input)):
                    st.error("OPENAI_API_KEY is not visible. Add it in the sidebar or .env.")
                    st.stop()
            retrieval_mode = "hybrid" if retrieval_mode_label == "Hybrid HF embeddings" else "tfidf"
            embedding_fn = None
            if retrieval_mode == "hybrid":
                embedding_token = optional_secret(hf_token_input) or resolve_huggingface_token()
                if embedding_token:
                    embedding_fn = make_huggingface_embedding_function(model=embedding_model, token=embedding_token)
                else:
                    append_log("Embedding retrieval requested, but HF token is not visible; falling back to TF-IDF.")
            with st.status("Processing", expanded=True) as status:
                result = run_pdf_pipeline(
                    pdf_path=temp_path,
                    filename=uploaded_file.name,
                    domain=domain,
                    extractor_factory=make_extractor_for_domain,
                    max_chunks=max_chunks,
                    max_attempts=max_attempts,
                    vision_enabled=vision_enabled,
                    vision_scale_label=vision_scale_label,
                    vision_max_pages=vision_max_pages,
                    retrieval_mode=retrieval_mode,
                    embedding_fn=embedding_fn,
                    log=append_log,
                )
                status.update(label="Complete", state="complete", expanded=False)
            st.session_state["pipeline_result"] = result
        except Exception as exc:
            st.exception(exc)
        finally:
            if temp_path:
                removed = safe_unlink_path(temp_path)
                if not removed:
                    append_log(f"Temp cleanup postponed; file is still locked by Windows: {temp_path}")

if result is not None:
    st.subheader(Path(result.source).name)
    st.caption(f"Processed as domain: {result.domain}")
    metric_cols = st.columns(7)
    metric_cols[0].metric("Tables", result.table_count)
    metric_cols[1].metric("Chunks", len(result.prepared.chunks))
    metric_cols[2].metric("Candidates", sum(len(state.get("extracted_objects", [])) for state in result.extraction_states))
    metric_cols[3].metric("Validated", result.validated_count)
    metric_cols[4].metric("Conflicts", len(result.conflicts))
    metric_cols[5].metric("Rejected", len(result.rejected))
    metric_cols[6].metric("Vision", len(result.vision_results))

    tab_results, tab_evidence, tab_rejected, tab_conflicts, tab_vision, tab_agents, tab_chunks, tab_log, tab_markdown = st.tabs(
        ["Results", "Evidence", "Rejected", "Conflicts", "Vision", "Agents", "Chunks", "Log", "Markdown"]
    )

    with tab_results:
        empty_messages = explain_empty_results(result)
        if empty_messages:
            st.warning("\n".join(f"- {message}" for message in empty_messages))
        st.dataframe(display_frame(result.clean, "results"), use_container_width=True, hide_index=True, height=430)
        render_csv_actions(result.clean, result.source, "clean", "Download clean CSV")

    with tab_evidence:
        render_evidence_view(result.clean)

    with tab_rejected:
        st.dataframe(display_frame(result.rejected, "rejected"), use_container_width=True, hide_index=True, height=430)
        render_csv_actions(result.rejected, result.source, "rejected", "Download rejected CSV")

    with tab_conflicts:
        st.dataframe(display_frame(result.conflicts, "conflicts"), use_container_width=True, hide_index=True, height=430)
        render_csv_actions(result.conflicts, result.source, "conflicts", "Download conflicts CSV")

    with tab_vision:
        vision_rows = vision_results_to_rows(result.vision_results)
        if not vision_rows:
            st.info("No vision analysis results for this run.")
        else:
            vision_frame = pd.DataFrame(vision_rows)
            st.dataframe(vision_frame, use_container_width=True, hide_index=True, height=360)
            render_csv_actions(vision_frame, result.source, "vision", "Download vision CSV")

            preview_paths = [Path(row["source"]) for row in vision_rows if Path(row["source"]).exists()]
            if preview_paths:
                st.subheader("Panel crops")
                for offset in range(0, min(len(preview_paths), 12), 3):
                    columns = st.columns(3)
                    for column, image_path in zip(columns, preview_paths[offset : offset + 3]):
                        column.image(str(image_path), caption=image_path.name, use_container_width=True)

    with tab_agents:
        agent_frame = pd.DataFrame(getattr(result, "agent_trace", []))
        if agent_frame.empty:
            st.info("No agent trace for this run.")
        else:
            agent_frame = agent_frame.copy()
            for column in ("metrics", "warnings"):
                if column in agent_frame.columns:
                    agent_frame[column] = agent_frame[column].apply(lambda value: str(value) if value else "")
            st.dataframe(display_frame(agent_frame, "agents"), use_container_width=True, hide_index=True, height=360)
            render_csv_actions(agent_frame, result.source, "agent_trace", "Download agent trace CSV")

    with tab_chunks:
        retrieval_results = getattr(result, "retrieval_results", [])
        retrieval_by_chunk = {item.chunk.index: item for item in retrieval_results}
        chunk_rows = [
            {
                "index": chunk.index,
                "selected": chunk.index in retrieval_by_chunk,
                "retrieval_rank": retrieval_by_chunk[chunk.index].rank if chunk.index in retrieval_by_chunk else None,
                "retrieval_score": retrieval_by_chunk[chunk.index].score if chunk.index in retrieval_by_chunk else None,
                "retrieval_method": retrieval_by_chunk[chunk.index].retrieval_method if chunk.index in retrieval_by_chunk else "",
                "section": chunk.section_title,
                "chars": len(chunk.text),
                "preview": chunk.text[:300].replace("\n", " "),
            }
            for chunk in result.prepared.chunks
        ]
        chunk_frame = pd.DataFrame(chunk_rows)
        if not chunk_frame.empty:
            chunk_frame = chunk_frame.sort_values(
                by=["selected", "retrieval_rank", "index"],
                ascending=[False, True, True],
                na_position="last",
            )
        st.dataframe(chunk_frame, use_container_width=True, hide_index=True, height=430)

    with tab_log:
        st.code("\n".join(result.logs), language="text")

    with tab_markdown:
        st.text_area("Prepared Markdown", result.prepared.markdown, height=520)
else:
    st.info("Upload a PDF and run the pipeline.")
