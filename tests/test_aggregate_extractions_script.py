import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_extractions import read_extraction_rows


class AggregateExtractionsScriptTests(unittest.TestCase):
    def test_read_extraction_rows_from_json_validated_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.json"
            path.write_text(
                json.dumps(
                    {
                        "validated_objects": [
                            {
                                "canonical_smiles": "CCO",
                                "property_name": "pMIC",
                                "value": 5.0,
                                "source_type": "text",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rows = read_extraction_rows([path])

        self.assertEqual(1, len(rows))
        self.assertEqual("CCO", rows[0]["canonical_smiles"])

    def test_read_extraction_rows_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"canonical_smiles": "CCO", "property_name": "pMIC", "value": 5.0}),
                        json.dumps({"rows": [{"canonical_smiles": "CCN", "property_name": "pMIC", "value": 6.0}]}),
                    ]
                ),
                encoding="utf-8",
            )

            rows = read_extraction_rows([path])

        self.assertEqual(["CCO", "CCN"], [row["canonical_smiles"] for row in rows])

    def test_cli_aggregates_nanozyme_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "state.json"
            output_path = temp_path / "clean.csv"
            conflicts_path = temp_path / "conflicts.csv"
            input_path.write_text(
                json.dumps(
                    {
                        "validated_objects": [
                            {
                                "material_id": "mn",
                                "material_name": "Mn3O4 microspheres",
                                "material_formula": "Mn3O4",
                                "normalized_formula": "Mn3O4",
                                "property_name": "Km",
                                "value": 0.02715,
                                "unit": "mM",
                                "assay": "kinetic assay",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/aggregate_extractions.py",
                    str(input_path),
                    "--kind",
                    "nanozyme",
                    "--output",
                    str(output_path),
                    "--conflicts",
                    str(conflicts_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("clean_rows=1", completed.stdout)
            self.assertIn("Mn3O4", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
