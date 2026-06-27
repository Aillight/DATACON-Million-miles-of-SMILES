import unittest
from types import SimpleNamespace

from agent.extraction_graph import (
    ExtractedProperty,
    ExtractionBatch,
    ExtractionPromptContext,
    build_extraction_prompt,
    extract_huggingface_message_text,
    make_huggingface_structured_extractor,
    make_openai_structured_extractor,
    parse_extraction_batch_json,
    run_small_molecule_extraction,
    validate_and_canonicalize_rows,
)


class ExtractionGraphTests(unittest.TestCase):
    def test_valid_rows_are_canonicalized(self) -> None:
        result = run_small_molecule_extraction(
            chunk_text="Compound 1 had pMIC 5.2.",
            extractor=lambda _context: ExtractionBatch(
                rows=[
                    ExtractedProperty(
                        compound_id="1",
                        smiles="C(C)O",
                        property_name="pMIC",
                        value=5.2,
                        unit="pMIC",
                    )
                ]
            ),
        )

        self.assertTrue(result["valid"])
        self.assertEqual(1, result["attempts"])
        self.assertEqual("CCO", result["validated_objects"][0]["canonical_smiles"])

    def test_invalid_smiles_retries_with_error_log(self) -> None:
        calls: list[ExtractionPromptContext] = []

        def extractor(context: ExtractionPromptContext) -> ExtractionBatch:
            calls.append(context)
            smiles = "C1CC" if context.attempt == 1 else "C1CC1"
            return ExtractionBatch(
                rows=[
                    ExtractedProperty(
                        compound_id="ring",
                        smiles=smiles,
                        property_name="pMIC",
                        value=6.0,
                        unit="pMIC",
                    )
                ]
            )

        result = run_small_molecule_extraction("Correct the ring if RDKit fails.", extractor=extractor)

        self.assertTrue(result["valid"])
        self.assertEqual(2, result["attempts"])
        self.assertIn("RDKit could not parse SMILES", calls[1].error_log[0])
        self.assertEqual("C1CC1", result["validated_objects"][0]["canonical_smiles"])

    def test_invalid_smiles_stops_after_max_attempts(self) -> None:
        result = run_small_molecule_extraction(
            chunk_text="Bad row.",
            max_attempts=2,
            extractor=lambda _context: ExtractionBatch(
                rows=[
                    ExtractedProperty(
                        compound_id="bad",
                        smiles="C1CC",
                        property_name="pMIC",
                        value=1.0,
                        unit="pMIC",
                    )
                ]
            ),
        )

        self.assertFalse(result["valid"])
        self.assertEqual(2, result["attempts"])
        self.assertEqual("invalid", result["status"])
        self.assertEqual([], result["validated_objects"])

    def test_extractor_failure_is_not_marked_valid(self) -> None:
        def extractor(_context: ExtractionPromptContext) -> ExtractionBatch:
            raise RuntimeError("missing token")

        result = run_small_molecule_extraction("text", extractor=extractor, max_attempts=1)

        self.assertFalse(result["valid"])
        self.assertEqual("failed", result["status"])
        self.assertIn("missing token", result["error_log"][0])

    def test_duplicate_canonical_property_pairs_are_skipped(self) -> None:
        rows = [
            ExtractedProperty(compound_id="a", smiles="C(C)O", property_name="pMIC", value=1.0, unit="pMIC"),
            ExtractedProperty(compound_id="b", smiles="CCO", property_name="pMIC", value=1.0, unit="pMIC"),
        ]

        validated, errors, warnings = validate_and_canonicalize_rows(rows)

        self.assertEqual([], errors)
        self.assertEqual(1, len(validated))
        self.assertIn("duplicate", warnings[0])

    def test_build_extraction_prompt_includes_previous_errors(self) -> None:
        prompt = build_extraction_prompt(
            ExtractionPromptContext(
                chunk_text="Article text",
                error_log=["row 1: invalid valence"],
                attempt=2,
            )
        )

        self.assertIn("invalid valence", prompt)
        self.assertIn("Article text", prompt)

    def test_openai_structured_extractor_uses_pydantic_batch_schema(self) -> None:
        class FakeResponses:
            def __init__(self) -> None:
                self.kwargs = {}

            def parse(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    output_parsed=ExtractionBatch(
                        rows=[
                            ExtractedProperty(
                                compound_id="1",
                                smiles="CCO",
                                property_name="pMIC",
                                value=5.0,
                                unit="pMIC",
                            )
                        ]
                    )
                )

        fake_responses = FakeResponses()
        fake_client = SimpleNamespace(responses=fake_responses)
        extractor = make_openai_structured_extractor(model="test-model", client=fake_client)

        batch = extractor(ExtractionPromptContext(chunk_text="Article text", attempt=1))

        self.assertEqual("test-model", fake_responses.kwargs["model"])
        self.assertIs(ExtractionBatch, fake_responses.kwargs["text_format"])
        self.assertEqual("CCO", batch.rows[0].smiles)

    def test_huggingface_structured_extractor_parses_json_response(self) -> None:
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
                                    '{"rows":[{"compound_id":"1","smiles":"CCO",'
                                    '"property_name":"pMIC","value":5.0,"unit":"pMIC"}]}'
                                )
                            )
                        )
                    ]
                )

        fake_client = FakeClient()
        extractor = make_huggingface_structured_extractor(model="free-test-model", client=fake_client)

        batch = extractor(ExtractionPromptContext(chunk_text="Compound 1 pMIC 5.0", attempt=1))

        self.assertEqual("free-test-model", fake_client.kwargs["model"])
        self.assertEqual({"type": "json_object"}, fake_client.kwargs["response_format"])
        self.assertEqual("CCO", batch.rows[0].smiles)

    def test_parse_extraction_batch_json_handles_fenced_output(self) -> None:
        payload = parse_extraction_batch_json(
            '```json\n{"rows":[{"compound_id":"1","smiles":"CCO",'
            '"property_name":"pMIC","value":5.0,"unit":"pMIC"}]}\n```'
        )

        self.assertEqual("CCO", payload["rows"][0]["smiles"])

    def test_parse_extraction_batch_json_fills_blank_pmic_unit(self) -> None:
        payload = parse_extraction_batch_json(
            '{"rows":[{"compound_id":"1","smiles":"CCO",'
            '"property_name":"pMIC","value":5.0,"unit":""}]}'
        )

        self.assertEqual("pMIC", payload["rows"][0]["unit"])

    def test_extract_huggingface_message_text_supports_dict_response(self) -> None:
        text = extract_huggingface_message_text({"choices": [{"message": {"content": '{"rows":[]}'}}]})

        self.assertEqual('{"rows":[]}', text)


if __name__ == "__main__":
    unittest.main()
