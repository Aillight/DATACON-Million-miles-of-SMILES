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
    ExtractorFn,
    make_huggingface_structured_extractor,
    make_openai_structured_extractor,
    resolve_huggingface_token,
)
from agent.nano_extraction_graph import NanoExtractorFn, make_huggingface_nanozyme_extractor
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
EXTRACTOR_OPTIONS = ("Hugging Face", "None", "OpenAI")


st.set_page_config(page_title="DataCon ChemX Agent", layout="wide")
st.title("ChemX Extraction Workbench")
hf_token_available = bool(resolve_huggingface_token())

with st.sidebar:
    domain = st.selectbox("Domain", DOMAINS)
    uploaded_file = st.file_uploader("PDF article", type=["pdf"])
    max_chunks = st.slider("Chunks", min_value=1, max_value=10, value=3)
    max_attempts = st.slider("Attempts", min_value=1, max_value=3, value=3)
    vision_enabled = st.checkbox("Vision analysis", value=is_nanozyme_domain(domain))
    vision_max_pages = st.slider("Vision pages", min_value=1, max_value=5, value=1, disabled=not vision_enabled)
    vision_scale_label = st.text_input("Scale label", value="", disabled=not vision_enabled)
    extractor_mode = st.selectbox(
        "Extractor",
        EXTRACTOR_OPTIONS,
        index=0 if hf_token_available else 1,
    )
    st.caption("HF token detected" if hf_token_available else "HF token is not visible to this app process")
    hf_model = st.text_input("HF model", value=os.getenv("HF_MODEL", DEFAULT_HUGGINGFACE_MODEL))
    openai_model = st.text_input("OpenAI model", value="gpt-4.1-mini")
    run_clicked = st.button("Run", type="primary", use_container_width=True)

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
                if extractor_mode == "Hugging Face":
                    if is_nanozyme_domain(effective_domain):
                        return make_huggingface_nanozyme_extractor(model=hf_model)
                    return make_huggingface_structured_extractor(model=hf_model)
                if extractor_mode == "OpenAI":
                    if is_nanozyme_domain(effective_domain):
                        raise ValueError("OpenAI extractor is not wired for Nanozymes yet. Use Hugging Face or None.")
                    return make_openai_structured_extractor(model=openai_model)
                return None

            if extractor_mode == "Hugging Face":
                if not hf_token_available:
                    st.error("HF_TOKEN is not visible to this Streamlit process. Restart the app after setting it.")
                    st.stop()
            elif extractor_mode == "OpenAI":
                if is_nanozyme_domain(domain):
                    st.error("OpenAI extractor is not wired for Nanozymes yet. Use Hugging Face or None.")
                    st.stop()
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
    st.caption(f"Processed as domain: {result.domain}")
    metric_cols = st.columns(6)
    metric_cols[0].metric("Tables", result.table_count)
    metric_cols[1].metric("Chunks", len(result.prepared.chunks))
    metric_cols[2].metric("Candidates", sum(len(state.get("extracted_objects", [])) for state in result.extraction_states))
    metric_cols[3].metric("Validated", result.validated_count)
    metric_cols[4].metric("Conflicts", len(result.conflicts))
    metric_cols[5].metric("Vision", len(result.vision_results))

    tab_results, tab_conflicts, tab_vision, tab_log, tab_chunks, tab_markdown = st.tabs(
        ["Results", "Conflicts", "Vision", "Log", "Chunks", "Markdown"]
    )

    with tab_results:
        empty_messages = explain_empty_results(result)
        if empty_messages:
            st.warning("\n".join(f"- {message}" for message in empty_messages))
        st.dataframe(result.clean, use_container_width=True, hide_index=True)
        if not result.clean.empty:
            clean_export_path = export_dataframe_csv(result.clean, result.source, "clean")
            st.caption(f"Local copy: {clean_export_path}")
            st.download_button(
                "Download clean CSV",
                data=csv_download_bytes(result.clean),
                file_name=f"{Path(result.source).stem}_clean.csv",
                mime="text/csv",
                on_click="ignore",
            )

    with tab_conflicts:
        st.dataframe(result.conflicts, use_container_width=True, hide_index=True)
        if not result.conflicts.empty:
            conflicts_export_path = export_dataframe_csv(result.conflicts, result.source, "conflicts")
            st.caption(f"Local copy: {conflicts_export_path}")
            st.download_button(
                "Download conflicts CSV",
                data=csv_download_bytes(result.conflicts),
                file_name=f"{Path(result.source).stem}_conflicts.csv",
                mime="text/csv",
                on_click="ignore",
            )

    with tab_vision:
        vision_rows = vision_results_to_rows(result.vision_results)
        if not vision_rows:
            st.info("No vision analysis results for this run.")
        else:
            vision_frame = pd.DataFrame(vision_rows)
            st.dataframe(vision_frame, use_container_width=True, hide_index=True)
            vision_export_path = export_dataframe_csv(vision_frame, result.source, "vision")
            st.caption(f"Local copy: {vision_export_path}")
            st.download_button(
                "Download vision CSV",
                data=csv_download_bytes(vision_frame),
                file_name=f"{Path(result.source).stem}_vision.csv",
                mime="text/csv",
                on_click="ignore",
            )

            preview_paths = [Path(row["source"]) for row in vision_rows if Path(row["source"]).exists()]
            if preview_paths:
                st.subheader("Panel crops")
                for offset in range(0, min(len(preview_paths), 12), 3):
                    columns = st.columns(3)
                    for column, image_path in zip(columns, preview_paths[offset : offset + 3]):
                        column.image(str(image_path), caption=image_path.name, use_container_width=True)

    with tab_log:
        st.code("\n".join(result.logs), language="text")

    with tab_chunks:
        chunk_rows = [
            {
                "index": chunk.index,
                "section": chunk.section_title,
                "chars": len(chunk.text),
                "preview": chunk.text[:300].replace("\n", " "),
            }
            for chunk in result.prepared.chunks
        ]
        st.dataframe(chunk_rows, use_container_width=True, hide_index=True)

    with tab_markdown:
        st.text_area("Prepared Markdown", result.prepared.markdown, height=520)
else:
    st.info("Upload a PDF and run the pipeline.")
