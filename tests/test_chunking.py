import unittest

from backend.parsing import chunk_article_text, split_article_sections


ARTICLE = """
Introduction

Ignore this section for targeted context.

Experimental Section

We mixed the reagents at room temperature. The sample was filtered.

Results and Discussion

The model recovered the expected trend. Tables confirmed the same direction.

References

[1] Placeholder.
"""


class ChunkingTests(unittest.TestCase):
    def test_split_sections_detects_scientific_headings(self) -> None:
        sections = split_article_sections(ARTICLE)
        titles = [section.title for section in sections]

        self.assertIn("Experimental Section", titles)
        self.assertIn("Results and Discussion", titles)

    def test_chunk_article_text_selects_target_sections(self) -> None:
        chunks = chunk_article_text(ARTICLE, max_chars=500, overlap_chars=50)
        combined = "\n".join(chunk.text for chunk in chunks)

        self.assertIn("Experimental Section", combined)
        self.assertIn("Results and Discussion", combined)
        self.assertNotIn("Ignore this section", combined)


if __name__ == "__main__":
    unittest.main()
