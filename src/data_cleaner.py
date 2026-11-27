import re
import unicodedata

from src.pdf_extractor import ExtractedDocument, ExtractedPage


def clean_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in normalized.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_extraction(extraction: ExtractedDocument) -> ExtractedDocument:
    if not extraction.pages:
        raise ValueError("The document contains no pages")

    page_numbers = [page.page_number for page in extraction.pages]
    if page_numbers != list(range(1, len(page_numbers) + 1)):
        raise ValueError("Extracted page numbers must be continuous and start at 1")

    cleaned_pages: list[ExtractedPage] = []
    has_content = False
    for page in extraction.pages:
        text = clean_text(page.text)
        cleaned_tables: list[list[list[str | None]]] = []
        for table in page.tables:
            rows = [
                [clean_text(cell or "") for cell in row]
                for row in table
            ]
            rows = [row for row in rows if any(cell for cell in row)]
            if rows:
                column_count = max(map(len, rows))
                cleaned_tables.append(
                    [row + [""] * (column_count - len(row)) for row in rows]
                )
        has_content = has_content or bool(text) or bool(cleaned_tables)
        cleaned_pages.append(
            ExtractedPage(
                page_number=page.page_number,
                text=text,
                tables=cleaned_tables,
            )
        )

    if not has_content:
        raise ValueError("No readable text or table data was extracted")
    return ExtractedDocument(media_type=extraction.media_type, pages=cleaned_pages)