import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.aggregation import aggregate_extraction_rows


class AggregationTests(unittest.TestCase):
    def test_aggregate_deduplicates_by_canonical_smiles_and_property(self) -> None:
        result = aggregate_extraction_rows(
            [
                {
                    "smiles": "C(C)O",
                    "property_name": "pMIC",
                    "value": "5.1234",
                    "unit": "pMIC",
                    "source_type": "text",
                },
                {
                    "smiles": "CCO",
                    "property_name": "pMIC",
                    "value": 5.1234,
                    "unit": "pMIC",
                    "source_type": "text",
                },
            ]
        )

        self.assertEqual(1, len(result.clean))
        self.assertEqual("CCO", result.clean.iloc[0]["canonical_smiles"])
        self.assertEqual(5.123, result.clean.iloc[0]["value"])
        self.assertTrue(result.conflicts.empty)

    def test_table_value_wins_over_text_conflict(self) -> None:
        result = aggregate_extraction_rows(
            [
                {
                    "canonical_smiles": "CCO",
                    "property_name": "pMIC",
                    "value": 4.0,
                    "unit": "pMIC",
                    "source_type": "text",
                    "source_id": "chunk-1",
                },
                {
                    "canonical_smiles": "CCO",
                    "property_name": "pMIC",
                    "value": 6.0,
                    "unit": "pMIC",
                    "source_type": "docling_table",
                    "source_id": "table-1",
                },
            ]
        )

        self.assertEqual(6.0, result.clean.iloc[0]["value"])
        self.assertEqual("docling_table", result.clean.iloc[0]["source_type"])
        self.assertEqual(1, len(result.conflicts))
        self.assertEqual("table_priority", result.conflicts.iloc[0]["resolution"])
        self.assertEqual(4.0, result.conflicts.iloc[0]["rejected_value"])

    def test_quality_columns_are_added_to_clean_rows(self) -> None:
        result = aggregate_extraction_rows(
            [
                {
                    "canonical_smiles": "CCO",
                    "property_name": "pMIC",
                    "value": 5.0,
                    "unit": "pMIC",
                    "source_type": "docling_table",
                    "evidence": "Table 1 reports pMIC 5.0 for compound 1.",
                }
            ]
        )

        row = result.clean.iloc[0]
        self.assertGreaterEqual(row["quality_score"], 90)
        self.assertIn("from_table", row["quality_flags"])
        self.assertIn("valid_smiles", row["quality_flags"])
        self.assertIn("has_evidence", row["quality_flags"])

    def test_same_priority_conflict_keeps_first_seen(self) -> None:
        result = aggregate_extraction_rows(
            [
                {"canonical_smiles": "CCO", "property_name": "pMIC", "value": 4.0, "source_type": "text"},
                {"canonical_smiles": "CCO", "property_name": "pMIC", "value": 5.0, "source_type": "chunk"},
            ]
        )

        self.assertEqual(4.0, result.clean.iloc[0]["value"])
        self.assertEqual("first_seen_same_priority", result.conflicts.iloc[0]["resolution"])

    def test_invalid_rows_are_skipped(self) -> None:
        result = aggregate_extraction_rows(
            [
                {"smiles": "C1CC", "property_name": "pMIC", "value": 1.0},
                {"smiles": "CCO", "property_name": "pMIC", "value": ""},
                {"smiles": "CCN", "property_name": "pMIC", "value": "NaN"},
            ]
        )

        self.assertTrue(result.clean.empty)
        self.assertTrue(result.conflicts.empty)

    def test_write_csv_writes_clean_and_conflicts(self) -> None:
        result = aggregate_extraction_rows(
            [
                {"canonical_smiles": "CCO", "property_name": "pMIC", "value": 4.0, "source_type": "text"},
                {"canonical_smiles": "CCO", "property_name": "pMIC", "value": 5.0, "source_type": "table"},
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_csv = Path(temp_dir) / "clean.csv"
            conflicts_csv = Path(temp_dir) / "conflicts.csv"
            result.write_csv(output_csv, conflicts_csv)

            clean = pd.read_csv(output_csv)
            conflicts = pd.read_csv(conflicts_csv)

        self.assertEqual(5.0, clean.iloc[0]["value"])
        self.assertEqual(1, len(conflicts))


if __name__ == "__main__":
    unittest.main()
