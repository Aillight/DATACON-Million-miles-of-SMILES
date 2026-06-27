import unittest

from scripts.evaluate_extraction import ExtractionRecord, build_index, evaluate_indexes, normalize_value


class EvaluateExtractionTests(unittest.TestCase):
    def test_normalize_value_rounds_numeric_values(self) -> None:
        self.assertEqual("1.235", normalize_value("1.23456", decimals=3))

    def test_evaluate_indexes_reports_key_and_value_metrics(self) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")

        gold = build_index(
            [
                ExtractionRecord(smiles="CCO", property_name="pMIC", value="1.000"),
                ExtractionRecord(smiles="CCN", property_name="pMIC", value="2.000"),
            ]
        )
        generated = build_index(
            [
                ExtractionRecord(smiles="CCO", property_name="pMIC", value="1.000"),
                ExtractionRecord(smiles="CCC", property_name="pMIC", value="3.000"),
            ]
        )

        result = evaluate_indexes(gold, generated, gold_record_count=2, generated_record_count=2)

        self.assertEqual(1, result.shared_keys)
        self.assertEqual(1, result.missing_keys)
        self.assertEqual(1, result.extra_keys)
        self.assertEqual(1, result.exact_value_matches)
        self.assertGreater(result.macro_f1, 0)


if __name__ == "__main__":
    unittest.main()
