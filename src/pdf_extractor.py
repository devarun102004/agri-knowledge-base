from dataclasses import dataclass
from pathlib import Path


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str
    tables: list[list[list[str | None]]]


@dataclass(frozen=True)
class ExtractedDocument:
    media_type: str
    pages: list[ExtractedPage]


def extract_document(path: str | Path) -> ExtractedDocument:
    """Extract text and detected tables from every page of a supported file."""
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix in {".txt", ".md"}:
        return ExtractedDocument(
            media_type="text/markdown" if suffix == ".md" else "text/plain",
            pages=[ExtractedPage(1, source.read_text(encoding="utf-8-sig"), [])],
        )
    if suffix == ".pdf":
        try:
            import pdfplumber
        except ImportError as error:
            raise RuntimeError(
                "PDF ingestion requires pdfplumber. Install project dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from error

        with pdfplumber.open(source) as pdf:
            pages = [
                ExtractedPage(
                    page_number=page_number,
                    text=page.extract_text() or "",
                    tables=page.extract_tables() or [],
                )
                for page_number, page in enumerate(pdf.pages, start=1)
            ]
        if not any(page.text.strip() or page.tables for page in pages):
            raise ValueError(f"No extractable text found in PDF: {source}")
        return ExtractedDocument(media_type="application/pdf", pages=pages)

    raise ValueError(f"Unsupported document type: {source.suffix or '(no extension)'}")