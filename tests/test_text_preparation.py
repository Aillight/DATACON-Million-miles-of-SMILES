import unittest

from backend.parsing.text_preparation import (
    clean_article_markdown,
    clean_parsed_markdown,
    find_microgram_per_ml_mentions,
    prepare_markdown_for_extraction,
)


class TextPreparationTests(unittest.TestCase):
    def test_clean_article_markdown_cuts_references(self) -> None:
        markdown = """
Introduction

Ignore.

Results and Discussion

The compounds were active.

References

[1] Reference noise.
"""

        cleaned = clean_article_markdown(markdown)

        self.assertIn("The compounds were active.", cleaned)
        self.assertNotIn("Reference noise", cleaned)

    def test_clean_article_markdown_removes_page_numbers_and_artifacts(self) -> None:
        markdown = """
Journal of Example Chemistry
1
Results

Useful result.
2 of 12
Downloaded from publisher.example
"""

        cleaned = clean_article_markdown(markdown)

        self.assertIn("Useful result.", cleaned)
        self.assertNotIn("2 of 12", cleaned)
        self.assertNotIn("Downloaded from", cleaned)

    def test_clean_parsed_markdown_keeps_appended_tables_after_references_cut(self) -> None:
        markdown = """
Results

Useful text.

References

[1] Reference noise.

## Table 1 (page 2)

| compound | MIC |
| --- | --- |
| A | 1 |
"""

        cleaned = clean_parsed_markdown(markdown)

        self.assertIn("Useful text.", cleaned)
        self.assertNotIn("Reference noise", cleaned)
        self.assertIn("## Table 1", cleaned)
        self.assertIn("| compound | MIC |", cleaned)

    def test_find_microgram_per_ml_mentions_without_smiles(self) -> None:
        mentions = find_microgram_per_ml_mentions("MIC was 12.5 ug/mL for compound A.")

        self.assertEqual(1, len(mentions))
        self.assertEqual(12.5, mentions[0].value_ug_ml)
        self.assertIsNone(mentions[0].pmic)

    def test_find_microgram_per_ml_mentions_with_smiles_converts_to_pmic(self) -> None:
        mentions = find_microgram_per_ml_mentions("MIC was 46.069 ug/mL.", smiles="CCO")

        self.assertEqual(1, len(mentions))
        self.assertAlmostEqual(3.0, mentions[0].pmic or 0, places=2)

    def test_find_microgram_per_ml_mentions_handles_ranges(self) -> None:
        mentions = find_microgram_per_ml_mentions("MIC range was 6.25-12.5 ug/mL.")

        self.assertEqual(1, len(mentions))
        self.assertEqual(6.25, mentions[0].value_ug_ml)
        self.assertEqual(12.5, mentions[0].upper_value_ug_ml)

    def test_prepare_markdown_for_extraction_cleans_then_chunks(self) -> None:
        prepared = prepare_markdown_for_extraction(
            markdown="""
Introduction

Ignore this.

Experimental Section

The MIC was 46.069 ug/mL.

References

[1] Noise.
""",
            max_chars=500,
            overlap_chars=50,
            concentration_smiles="CCO",
        )

        self.assertEqual(1, len(prepared.chunks))
        self.assertIn("Experimental Section", prepared.chunks[0].text)
        self.assertNotIn("Noise", prepared.markdown)
        self.assertAlmostEqual(3.0, prepared.concentration_mentions[0].pmic or 0, places=2)


if __name__ == "__main__":
    unittest.main()
