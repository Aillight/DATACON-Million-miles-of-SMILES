import unittest
from types import SimpleNamespace

from agent.extraction_graph import ExtractionPromptContext
from agent.nano_extraction_graph import (
    ExtractedNanozymeProperty,
    NanozymeExtractionBatch,
    build_nanozyme_extraction_prompt,
    make_huggingface_nanozyme_extractor,
    make_openrouter_nanozyme_extractor,
    normalize_nano_extractor_output,
    parse_nanozyme_batch_json,
    run_nanozyme_extraction,
    validate_and_normalize_nanozyme_rows,
    validate_material_formula,
)


class NanoExtractionGraphTests(unittest.TestCase):
    def test_valid_formula_and_size_are_accepted(self) -> None:
        rows = [
            ExtractedNanozymeProperty(
                material_id="mn3o4",
                material_name="Mn3O4 microspheres",
                material_formula="Mn3O4",
                property_name="particle_diameter",
                value=800.0,
                unit="nm",
                assay="SEM",
            )
        ]

        validated, errors, warnings = validate_and_normalize_nanozyme_rows(rows)

        self.assertEqual([], errors)
        self.assertEqual([], warnings)
        self.assertEqual("Mn3O4", validated[0].normalized_formula)

    def test_impossible_particle_size_is_rejected(self) -> None:
        rows = [
            ExtractedNanozymeProperty(
                material_id="bad-size",
                material_name="Mn3O4",
                material_formula="Mn3O4",
                property_name="particle_size",
                value=0.01,
                unit="nm",
            )
        ]

        validated, errors, _warnings = validate_and_normalize_nanozyme_rows(rows)

        self.assertEqual([], validated)
        self.assertIn("below 0.1 nm", errors[0])

    def test_process_metric_yield_property_is_accepted(self) -> None:
        rows = [
            ExtractedNanozymeProperty(
                material_id="cuo",
                material_name="CuO",
                material_formula="CuO",
                property_name="sorbitol yield",
                value=99.1,
                unit="%",
            )
        ]

        validated, errors, warnings = validate_and_normalize_nanozyme_rows(rows)

        self.assertEqual([], errors)
        self.assertEqual([], warnings)
        self.assertEqual("sorbitol yield", validated[0].property_name)

    def test_unrecognized_property_is_rejected_without_retry_error(self) -> None:
        rows = [
            ExtractedNanozymeProperty(
                material_id="cuo",
                material_name="CuO",
                material_formula="CuO",
                property_name="random score",
                value=99.1,
                unit="%",
            )
        ]

        validated, errors, warnings = validate_and_normalize_nanozyme_rows(rows)

        self.assertEqual([], validated)
        self.assertEqual([], errors)
        self.assertIn("not an allowed nanozyme endpoint", warnings[0])

    def test_property_name_is_canonicalized(self) -> None:
        rows = [
            ExtractedNanozymeProperty(
                material_id="mn3o4",
                material_name="Mn3O4",
                material_formula="Mn3O4",
                property_name="nanoparticle diameter",
                value=10.0,
                unit="nm",
            )
        ]

        validated, errors, _warnings = validate_and_normalize_nanozyme_rows(rows)

        self.assertEqual([], errors)
        self.assertEqual("particle diameter", validated[0].property_name)

    def test_unknown_formula_element_is_rejected(self) -> None:
        validation = validate_material_formula("Xx2O")

        self.assertIn("unknown element", validation.errors[0])

    def test_composite_catalyst_label_is_accepted(self) -> None:
        validation = validate_material_formula("M-W/SBA-15 (M = Ni, Pd, Zn, Cu)")

        self.assertEqual([], validation.errors)
        self.assertIn("SBA-15", validation.normalized_formula)

    def test_invalid_formula_retries_with_error_log(self) -> None:
        calls: list[ExtractionPromptContext] = []

        def extractor(context: ExtractionPromptContext) -> NanozymeExtractionBatch:
            calls.append(context)
            formula = "Xx2O" if context.attempt == 1 else "Mn3O4"
            return NanozymeExtractionBatch(
                rows=[
                    ExtractedNanozymeProperty(
                        material_id="material-1",
                        material_name="Mn3O4 microspheres",
                        material_formula=formula,
                        property_name="particle_diameter",
                        value=800.0,
                        unit="nm",
                    )
                ]
            )

        result = run_nanozyme_extraction("Mn3O4 microspheres were about 800 nm.", extractor=extractor)

        self.assertTrue(result["valid"])
        self.assertEqual(2, result["attempts"])
        self.assertIn("unknown element", calls[1].error_log[0])
        self.assertEqual("Mn3O4", result["validated_objects"][0]["normalized_formula"])

    def test_extractor_failure_is_not_marked_valid(self) -> None:
        def extractor(_context: ExtractionPromptContext) -> NanozymeExtractionBatch:
            raise RuntimeError("missing token")

        result = run_nanozyme_extraction("text", extractor=extractor, max_attempts=1)

        self.assertFalse(result["valid"])
        self.assertEqual("failed", result["status"])
        self.assertIn("missing token", result["error_log"][0])

    def test_partial_schema_failures_are_rejected_without_losing_valid_rows(self) -> None:
        def extractor(_context: ExtractionPromptContext) -> dict:
            return {
                "rows": [
                    {
                        "material_id": "cuo",
                        "material_name": "CuO",
                        "material_formula": "CuO",
                        "property_name": "sorbitol yield",
                        "value": 99.1,
                        "unit": "%",
                    },
                    {
                        "material_id": "missing-formula",
                        "material_name": "",
                        "material_formula": "",
                        "property_name": "yield",
                        "value": 12.0,
                        "unit": "%",
                    },
                    {
                        "material_id": "missing-unit",
                        "material_name": "Ru",
                        "material_formula": "Ru",
                        "property_name": "yield",
                        "value": 40.0,
                        "unit": "",
                    },
                ]
            }

        result = run_nanozyme_extraction("CuO showed sorbitol yield.", extractor=extractor, max_attempts=1)

        self.assertEqual("valid", result["status"])
        self.assertEqual(1, len(result["validated_objects"]))
        self.assertEqual(2, len(result["rejected_objects"]))
        self.assertEqual("nano_extractor_schema", result["rejected_objects"][0]["validator"])

    def test_parse_nanozyme_batch_json_normalizes_aliases_and_string_values(self) -> None:
        payload = parse_nanozyme_batch_json(
            '{"rows":[{"id":"mn","material":"Mn3O4 microspheres","formula":"Mn3O4",'
            '"property":"Km","value":"0.02715","units":"mM"}]}'
        )

        self.assertEqual("mn", payload["rows"][0]["material_id"])
        self.assertEqual("Km", payload["rows"][0]["property_name"])
        self.assertEqual(0.02715, payload["rows"][0]["value"])

    def test_huggingface_nanozyme_extractor_parses_json_response(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.kwargs = {}

            def chat_completion(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=(
                                    '{"rows":[{"material_id":"mn","material_name":"Mn3O4 microspheres",'
                                    '"material_formula":"Mn3O4","property_name":"particle_diameter",'
                                    '"value":800.0,"unit":"nm","assay":"SEM","condition":""}]}'
                                )
                            )
                        )
                    ]
                )

        fake_client = FakeClient()
        extractor = make_huggingface_nanozyme_extractor(model="free-test-model", client=fake_client)

        batch = normalize_nano_extractor_output(
            extractor(ExtractionPromptContext(chunk_text="Mn3O4 was about 800 nm.", attempt=1))
        )

        self.assertEqual("free-test-model", fake_client.kwargs["model"])
        self.assertEqual("Mn3O4", batch.rows[0].material_formula)

    def test_openrouter_nanozyme_extractor_parses_chat_completion_response(self) -> None:
        class FakeCompletions:
            def __init__(self) -> None:
                self.kwargs = {}

            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=(
                                    '{"rows":[{"material_id":"cuo","material_name":"CuO",'
                                    '"material_formula":"CuO","property_name":"sorbitol yield",'
                                    '"value":99.1,"unit":"%","assay":"","condition":""}]}'
                                )
                            )
                        )
                    ]
                )

        fake_completions = FakeCompletions()
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
        extractor = make_openrouter_nanozyme_extractor(model="openrouter-test-model", client=fake_client)

        batch = normalize_nano_extractor_output(
            extractor(ExtractionPromptContext(chunk_text="CuO showed sorbitol yield.", attempt=1))
        )

        self.assertEqual("openrouter-test-model", fake_completions.kwargs["model"])
        self.assertEqual("json_schema", fake_completions.kwargs["response_format"]["type"])
        self.assertEqual("CuO", batch.rows[0].material_formula)

    def test_prompt_includes_previous_errors(self) -> None:
        prompt = build_nanozyme_extraction_prompt(
            ExtractionPromptContext(chunk_text="text", error_log=["bad formula"], attempt=2)
        )

        self.assertIn("bad formula", prompt)
        self.assertIn("nanozyme", prompt.lower())


if __name__ == "__main__":
    unittest.main()
