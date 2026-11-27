import unittest

from src.data_cleaner import clean_extraction
from src.pdf_extractor import ExtractedDocument, ExtractedPage


class CleanExtractionTests(unittest.TestCase):
    def test_normalizes_text_and_pads_table_rows(self) -> None:
        extraction = ExtractedDocument(
            "application/pdf",
            [
                ExtractedPage(
                    1,
                    "  Crop   guidance\r\n\r\n\r\nUse label. ",
                    [[["Crop", "Disease", "Product"], ["Maize", "Rust"]]],
                )
            ],
        )

        cleaned = clean_extraction(extraction)

        self.assertEqual(cleaned.pages[0].text, "Crop guidance\n\nUse label.")
        self.assertEqual(
            cleaned.pages[0].tables,
            [[["Crop", "Disease", "Product"], ["Maize", "Rust", ""]]],
        )

    def test_rejects_empty_or_nonsequential_page_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "No readable text"):
            clean_extraction(
                ExtractedDocument("application/pdf", [ExtractedPage(1, " \n ", [])])
            )
        with self.assertRaisesRegex(ValueError, "continuous"):
            clean_extraction(
                ExtractedDocument(
                    "application/pdf",
                    [ExtractedPage(2, "Text", [])],
                )
            )


if __name__ == "__main__":
    unittest.main()
