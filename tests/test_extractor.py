import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.pdf_extractor import extract_document


class ExtractTextTests(unittest.TestCase):
    def test_reads_utf8_text_and_removes_bom(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "guidance.txt"
            source.write_text("\ufeffRead the label before use.", encoding="utf-8")

            extraction = extract_document(source)

        self.assertEqual(extraction.media_type, "text/plain")
        self.assertEqual(len(extraction.pages), 1)
        self.assertEqual(extraction.pages[0].text, "Read the label before use.")

    def test_reads_markdown_as_utf8_single_page(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "recommendations.md"
            source.write_text("\ufeff## Recommendation: Product A", encoding="utf-8")

            extraction = extract_document(source)

        self.assertEqual(extraction.media_type, "text/markdown")
        self.assertEqual(len(extraction.pages), 1)
        self.assertEqual(extraction.pages[0].text, "## Recommendation: Product A")

    def test_extracts_text_and_tables_from_every_pdf_page(self) -> None:
        page_one = MagicMock()
        page_one.extract_text.return_value = "Page one text."
        page_one.extract_tables.return_value = [
            [["Crop", "Disease", "Product"], ["Maize", "Rust", "Product A"]]
        ]
        page_two = MagicMock()
        page_two.extract_text.return_value = "Page two text."
        page_two.extract_tables.return_value = []
        pdf = MagicMock(pages=[page_one, page_two])
        pdf_context = MagicMock()
        pdf_context.__enter__.return_value = pdf

        with patch("pdfplumber.open", return_value=pdf_context):
            extraction = extract_document("guidance.pdf")

        self.assertEqual(len(extraction.pages), 2)
        self.assertEqual(extraction.pages[0].text, "Page one text.")
        self.assertEqual(extraction.pages[0].tables[0][1][2], "Product A")
        self.assertEqual(extraction.pages[1].page_number, 2)

    def test_rejects_unsupported_document_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported document type"):
            extract_document("document.docx")


if __name__ == "__main__":
    unittest.main()