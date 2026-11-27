import tempfile
import unittest

import yaml

from src.okf_builder import build_okf_document, write_okf_document
from src.pdf_extractor import ExtractedDocument, ExtractedPage


class OkfDocumentTests(unittest.TestCase):
    def test_writes_markdown_with_yaml_front_matter_and_tables(self) -> None:
        extraction = ExtractedDocument(
            "application/pdf",
            [
                ExtractedPage(
                    1,
                    "Maize disease guidance.",
                    [
                        [
                            ["Crop", "Disease", "Product"],
                            ["Maize", "Rust", "Product A"],
                        ]
                    ],
                )
            ],
        )
        document = build_okf_document("/tmp/maize-guide.pdf", extraction)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = write_okf_document(document, temporary_directory)
            content = output.read_text(encoding="utf-8")

        front_matter, markdown = content.split("---\n\n", maxsplit=1)
        metadata = yaml.safe_load(front_matter.removeprefix("---\n"))
        self.assertEqual(output.suffix, ".md")
        self.assertEqual(metadata["format"], "okf")
        self.assertEqual(metadata["page_count"], 1)
        self.assertEqual(metadata["table_count"], 1)
        self.assertIn("| Maize | Rust | Product A |", markdown)


if __name__ == "__main__":
    unittest.main()
