import unittest

from backend.nano_aggregation import aggregate_nanozyme_rows


class NanoAggregationTests(unittest.TestCase):
    def test_aggregate_nanozyme_rows_keeps_distinct_properties(self) -> None:
        rows = [
            {
                "material_id": "mn",
                "material_name": "Mn3O4 microspheres",
                "material_formula": "Mn3O4",
                "normalized_formula": "Mn3O4",
                "property_name": "particle diameter",
                "value": 800.0,
                "unit": "nm",
                "assay": "SEM",
            },
            {
                "material_id": "mn",
                "material_name": "Mn3O4 microspheres",
                "material_formula": "Mn3O4",
                "normalized_formula": "Mn3O4",
                "property_name": "Km",
                "value": 0.02715,
                "unit": "mM",
                "assay": "kinetic assay",
            },
        ]

        result = aggregate_nanozyme_rows(rows)

        self.assertEqual(2, len(result.clean))
        self.assertEqual([], result.conflicts.to_dict("records"))

    def test_aggregate_nanozyme_rows_records_conflicts(self) -> None:
        rows = [
            {
                "material_formula": "Mn3O4",
                "property_name": "particle diameter",
                "value": 800.0,
                "unit": "nm",
                "assay": "SEM",
                "source_type": "text",
            },
            {
                "material_formula": "Mn3O4",
                "property_name": "particle diameter",
                "value": 810.0,
                "unit": "nm",
                "assay": "SEM",
                "source_type": "table",
            },
        ]

        result = aggregate_nanozyme_rows(rows)

        self.assertEqual(1, len(result.clean))
        self.assertEqual(1, len(result.conflicts))
        self.assertEqual(810.0, result.clean.iloc[0]["value"])
        self.assertEqual("table_priority", result.conflicts.iloc[0]["resolution"])

    def test_aggregate_nanozyme_rows_skips_invalid_formula(self) -> None:
        result = aggregate_nanozyme_rows(
            [
                {
                    "material_formula": "Xx2O",
                    "property_name": "particle diameter",
                    "value": 800.0,
                    "unit": "nm",
                }
            ]
        )

        self.assertEqual(0, len(result.clean))


if __name__ == "__main__":
    unittest.main()
