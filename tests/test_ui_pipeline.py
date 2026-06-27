import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from agent.nano_extraction_graph import NanozymeExtractionBatch
from backend.parsing.pdf_parser import ParsedDocument
from backend.parsing.chunking import TextChunk
from backend.parsing.text_preparation import PreparedDocument
from backend.vision.cv_recognition import (
    ImagePanel,
    ParticleSummary,
    ScaleBarDetection,
    VisionAnalysisResult,
)
from ui.pipeline import (
    ArticlePipelineResult,
    csv_download_bytes,
    empty_extractor,
    empty_nano_extractor,
    enrich_validated_rows,
    export_dataframe_csv,
    explain_empty_results,
    is_nanozyme_domain,
    resolve_effective_domain,
    run_pdf_pipeline,
    vision_results_to_rows,
)


class UiPipelineTests(unittest.TestCase):
    def test_enrich_validated_rows_adds_source_metadata(self) -> None:
        rows = [
            {
                "compound_id": "1",
                "smiles": "CCO",
                "canonical_smiles": "CCO",
                "property_name": "pMIC",
                "value": 5.0,
                "unit": "pMIC",
            }
        ]

        enriched = enrich_validated_rows(rows, article_id="paper.pdf", chunk_index=2, evidence="abc" * 500)

        self.assertEqual("text", enriched[0]["source_type"])
        self.assertEqual("paper.pdf", enriched[0]["article_id"])
        self.assertEqual("paper.pdf:chunk-2", enriched[0]["source_id"])
        self.assertEqual(2, enriched[0]["chunk_index"])
        self.assertEqual(1000, len(enriched[0]["evidence"]))

    def test_empty_extractor_returns_empty_batch(self) -> None:
        batch = empty_extractor(None)

        self.assertEqual([], batch.rows)

    def test_empty_nano_extractor_returns_empty_batch(self) -> None:
        batch = empty_nano_extractor(None)

        self.assertEqual([], batch.rows)

    def test_csv_download_bytes_uses_excel_friendly_utf8(self) -> None:
        frame = pd.DataFrame([{"material": "катализатор", "value": 99.1}])

        payload = csv_download_bytes(frame)

        self.assertTrue(payload.startswith(b"\xef\xbb\xbf"))
        self.assertIn("катализатор", payload.decode("utf-8-sig"))

    def test_export_dataframe_csv_writes_local_copy(self) -> None:
        frame = pd.DataFrame([{"material": "CuO", "value": 99.1}])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = export_dataframe_csv(frame, "paper 1.pdf", "clean", output_dir=Path(temp_dir))

            self.assertTrue(output_path.exists())
            self.assertEqual("paper_1_clean.csv", output_path.name)
            self.assertIn("CuO", output_path.read_text(encoding="utf-8-sig"))

    def test_is_nanozyme_domain(self) -> None:
        self.assertTrue(is_nanozyme_domain("Nanozymes"))
        self.assertFalse(is_nanozyme_domain("Oxazolidinones"))

    def test_resolve_effective_domain_switches_nano_text_to_nanozymes(self) -> None:
        prepared = PreparedDocument(
            source="paper.pdf",
            parser="test",
            markdown="Mn3O4 nanoparticles showed oxidase-like activity and 800 nm microspheres.",
            chunks=[],
        )

        self.assertEqual("Nanozymes", resolve_effective_domain("Oxazolidinones", prepared))

    def test_run_pdf_pipeline_auto_routes_nano_document_to_nano_extractor(self) -> None:
        markdown = "## Results\n\nMn3O4 nanoparticles showed oxidase-like activity with Vmax 126.7 nM s-1."
        seen_domains: list[str] = []

        def extractor_factory(effective_domain: str):
            seen_domains.append(effective_domain)

            def extractor(_context) -> NanozymeExtractionBatch:
                return NanozymeExtractionBatch(
                    rows=[
                        {
                            "material_id": "Mn3O4",
                            "material_name": "Mn3O4 nanoparticles",
                            "material_formula": "Mn3O4",
                            "property_name": "Vmax",
                            "value": 126.7,
                            "unit": "nM s-1",
                            "assay": "oxidase-like",
                            "condition": "",
                        }
                    ]
                )

            return extractor

        with patch(
            "ui.pipeline.parse_pdf_to_markdown",
            return_value=ParsedDocument(source="paper.pdf", parser="test", markdown=markdown),
        ):
            result = run_pdf_pipeline(
                pdf_path="paper.pdf",
                filename="paper.pdf",
                domain="Oxazolidinones",
                extractor_factory=extractor_factory,
                max_chunks=1,
            )

        self.assertEqual(["Nanozymes"], seen_domains)
        self.assertEqual("Nanozymes", result.domain)
        self.assertEqual(1, len(result.clean))
        self.assertIn("normalized_formula", result.clean.columns)

    def test_article_pipeline_result_validated_count(self) -> None:
        result = ArticlePipelineResult(
            source="paper.pdf",
            domain="Oxazolidinones",
            prepared=None,
            table_count=0,
            extraction_states=[
                {"validated_objects": [{"a": 1}, {"b": 2}]},
                {"validated_objects": []},
            ],
            clean=pd.DataFrame(),
            conflicts=pd.DataFrame(),
        )

        self.assertEqual(2, result.validated_count)

    def test_explain_empty_results_suggests_nanozymes_domain_for_nano_text(self) -> None:
        result = ArticlePipelineResult(
            source="paper.pdf",
            domain="Oxazolidinones",
            prepared=PreparedDocument(
                source="paper.pdf",
                parser="test",
                markdown="Mn3O4 nanoparticles showed oxidase-like activity and 800 nm microspheres.",
                chunks=[TextChunk(index=0, section_title="Results", text="text", start_char=0, end_char=4)],
            ),
            table_count=0,
            extraction_states=[{"extracted_objects": [], "validated_objects": [], "status": "valid"}],
            clean=pd.DataFrame(columns=["canonical_smiles", "property_name"]),
            conflicts=pd.DataFrame(),
        )

        messages = explain_empty_results(result)

        self.assertTrue(any("Nanozymes" in message for message in messages))

    def test_explain_empty_results_surfaces_extractor_failure(self) -> None:
        result = ArticlePipelineResult(
            source="paper.pdf",
            domain="Oxazolidinones",
            prepared=PreparedDocument(source="paper.pdf", parser="test", markdown="", chunks=[]),
            table_count=0,
            extraction_states=[
                {
                    "extracted_objects": [],
                    "validated_objects": [],
                    "status": "failed",
                    "error_log": ["missing HF token"],
                }
            ],
            clean=pd.DataFrame(columns=["canonical_smiles", "property_name"]),
            conflicts=pd.DataFrame(),
        )

        messages = explain_empty_results(result)

        self.assertTrue(any("Extractor failed" in message for message in messages))
        self.assertTrue(any("missing HF token" in message for message in messages))

    def test_explain_empty_results_identifies_socket_permission_error(self) -> None:
        result = ArticlePipelineResult(
            source="paper.pdf",
            domain="Nanozymes",
            prepared=PreparedDocument(source="paper.pdf", parser="test", markdown="", chunks=[]),
            table_count=0,
            extraction_states=[
                {
                    "extracted_objects": [],
                    "validated_objects": [],
                    "status": "failed",
                    "error_log": ["ConnectError: [WinError 10013] доступа к сокету"],
                }
            ],
            clean=pd.DataFrame(columns=["normalized_formula", "property_name"]),
            conflicts=pd.DataFrame(),
        )

        messages = explain_empty_results(result)

        self.assertTrue(any("Network access is blocked" in message for message in messages))

    def test_vision_results_to_rows_flattens_summary(self) -> None:
        result = VisionAnalysisResult(
            source="panel.png",
            image_width=100,
            image_height=80,
            panels=[ImagePanel(index=1, bbox=(1, 2, 30, 40), kind="microscopy")],
            scale_bar=ScaleBarDetection(bbox=(10, 70, 50, 3), length_px=50.0, label_nm=100.0, nm_per_px=2.0),
            particle_summary=ParticleSummary(count=3, mean_diameter_nm=20.0, mean_diameter_px=10.0),
            warnings=["check scale"],
        )

        rows = vision_results_to_rows([result])

        self.assertEqual(1, rows[0]["index"])
        self.assertEqual("microscopy", rows[0]["panel_kind"])
        self.assertEqual(3, rows[0]["particle_count"])
        self.assertEqual(2.0, rows[0]["nm_per_px"])
        self.assertEqual("check scale", rows[0]["warnings"])


if __name__ == "__main__":
    unittest.main()
