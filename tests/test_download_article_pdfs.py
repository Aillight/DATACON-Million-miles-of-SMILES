import csv
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.download_article_pdfs import extract_pdf_from_tgz, looks_like_pdf, read_article_references, safe_filename


class DownloadArticlePdfsTests(unittest.TestCase):
    def test_read_article_references_groups_rows_by_doi(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "articles.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["pdf", "doi", "title", "access"])
                writer.writeheader()
                writer.writerow({"pdf": "abc", "doi": "10.1000/example", "title": "Paper", "access": "1"})
                writer.writerow({"pdf": "abc", "doi": "10.1000/example", "title": "Paper", "access": "1"})
                writer.writerow({"pdf": "def", "doi": "10.1000/closed", "title": "Closed", "access": "0"})

            references = read_article_references([csv_path], access_only=True)

        self.assertEqual(1, len(references))
        self.assertEqual("10.1000/example", references[0].doi)
        self.assertEqual(2, references[0].row_count)

    def test_safe_filename_removes_path_separators(self) -> None:
        self.assertEqual("10.1000_a_b", safe_filename("10.1000/a\\b"))

    def test_looks_like_pdf_requires_pdf_header(self) -> None:
        self.assertTrue(looks_like_pdf(b"%PDF-1.7\n", "text/html", "https://example.test/file"))
        self.assertFalse(looks_like_pdf(b"<html>forbidden</html>", "application/pdf", "https://example.test/file.pdf"))

    def test_extract_pdf_from_tgz(self) -> None:
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w:gz") as archive:
            pdf_bytes = b"%PDF-1.7\nexample"
            info = tarfile.TarInfo("article/example.pdf")
            info.size = len(pdf_bytes)
            archive.addfile(info, io.BytesIO(pdf_bytes))

        self.assertEqual(b"%PDF-1.7\nexample", extract_pdf_from_tgz(payload.getvalue()))


if __name__ == "__main__":
    unittest.main()
