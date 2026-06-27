import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.parsing.chunking import TextChunk
from backend.parsing.text_preparation import PreparedDocument
from backend.vision.cv_recognition import ParticleSummary, VisionAnalysisResult
from scripts.run_article_pipeline import build_extractor, default_output_dir, safe_filename, write_pipeline_outputs
from ui.pipeline import ArticlePipelineResult


class RunArticlePipelineScriptTests(unittest.TestCase):
    def test_safe_filename_removes_path_unsafe_characters(self) -> None:
        self.assertEqual("A_B_C.pdf", safe_filename(" A/B:C.pdf "))
        self.assertEqual("article", safe_filename("///"))

    def test_default_output_dir_uses_pdf_stem(self) -> None:
        output_dir = default_output_dir(Path("papers") / "paper 1.pdf")

        self.assertEqual(Path("outputs/articles/paper_1"), output_dir)

    def test_openai_extractor_is_not_allowed_for_nanozymes(self) -> None:
        with self.assertRaisesRegex(ValueError, "Nanozymes"):
            build_extractor(
                extractor="openai",
                use_openai=False,
                domain="Nanozymes",
                hf_model="hf",
                openai_model="openai",
                max_output_tokens=100,
            )

    def test_write_pipeline_outputs_writes_summary_artifacts(self) -> None:
        result = make_result()

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_pipeline_outputs(result, Path(temp_dir))

            manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
            chunks = json.loads(artifacts["chunks_json"].read_text(encoding="utf-8"))
            states = json.loads(artifacts["states_json"].read_text(encoding="utf-8"))

            self.assertTrue(artifacts["clean_csv"].exists())
            self.assertTrue(artifacts["conflicts_csv"].exists())
            self.assertEqual("paper.pdf", manifest["source"])
            self.assertEqual("Oxazolidinones", manifest["domain"])
            self.assertEqual(1, manifest["candidate_count"])
            self.assertEqual(1, manifest["validated_count"])
            self.assertEqual(1, manifest["clean_rows"])
            self.assertEqual(1, manifest["conflict_rows"])
            self.assertEqual(0, manifest["vision_result_count"])
            self.assertEqual("Results", chunks[0]["section_title"])
            self.assertEqual("valid", states[0]["status"])

    def test_write_pipeline_outputs_writes_vision_artifacts(self) -> None:
        result = make_result()
        result.vision_results.append(
            VisionAnalysisResult(
                source="panel.png",
                image_width=100,
                image_height=80,
                particle_summary=ParticleSummary(count=2, mean_diameter_px=12.0),
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_pipeline_outputs(result, Path(temp_dir))
            manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))

            self.assertTrue(artifacts["vision_json"].exists())
            self.assertTrue(artifacts["vision_summary_csv"].exists())
            self.assertEqual(1, manifest["vision_result_count"])
            self.assertEqual(2, manifest["vision_particle_count"])


def make_result() -> ArticlePipelineResult:
    prepared = PreparedDocument(
        source="paper.pdf",
        parser="docling",
        markdown="## Results\n\nActivity data.",
        chunks=[
            TextChunk(
                index=0,
                section_title="Results",
                text="## Results\n\nActivity data.",
                start_char=0,
                end_char=25,
            )
        ],
        warnings=["parser fallback"],
    )
    state = {
        "extracted_objects": [{"compound_id": "1", "smiles": "CCO"}],
        "validated_objects": [{"compound_id": "1", "canonical_smiles": "CCO"}],
        "error_log": [],
        "warnings": [],
        "valid": True,
        "attempts": 1,
        "status": "valid",
    }
    clean = pd.DataFrame(
        [
            {
                "canonical_smiles": "CCO",
                "property_name": "pMIC",
                "value": 5.0,
            }
        ]
    )
    conflicts = pd.DataFrame(
        [
            {
                "canonical_smiles": "CCO",
                "property_name": "pMIC",
                "chosen_value": 5.0,
                "rejected_value": 4.0,
            }
        ]
    )
    return ArticlePipelineResult(
        source="paper.pdf",
        domain="Oxazolidinones",
        prepared=prepared,
        table_count=2,
        extraction_states=[state],
        clean=clean,
        conflicts=conflicts,
        logs=["Loaded PDF: paper.pdf"],
    )


if __name__ == "__main__":
    unittest.main()
