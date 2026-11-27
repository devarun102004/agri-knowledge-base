import hashlib
from pathlib import Path
from typing import Any

import yaml

from src.pdf_extractor import ExtractedDocument

OKF_FORMAT = "okf"
OKF_VERSION = "1.1"


def _markdown_table(rows: list[list[str | None]]) -> str:
    column_count = max(map(len, rows))
    padded_rows = [
        [cell or "" for cell in row] + [""] * (column_count - len(row))
        for row in rows
    ]
    if len(padded_rows) == 1:
        headers = [f"Column {index + 1}" for index in range(column_count)]
        body_rows = padded_rows
    else:
        headers = padded_rows[0]
        body_rows = padded_rows[1:]

    def format_row(row: list[str]) -> str:
        cells = [cell.replace("|", r"\|").replace("\n", "<br>") for cell in row]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("---" for _ in headers) + " |"
    return "\n".join([format_row(headers), separator, *(format_row(row) for row in body_rows)])


def build_okf_document(
    source: str | Path, extraction: ExtractedDocument
) -> dict[str, Any]:
    """Build the project's versioned Markdown document with YAML front matter."""
    source_path = Path(source).resolve()
    document_id = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()
    pages = [
        {
            "page_number": page.page_number,
            "text": page.text,
            "tables": page.tables,
        }
        for page in extraction.pages
    ]

    markdown_lines = [f"# {source_path.stem}"]
    table_count = 0
    for page in pages:
        markdown_lines.extend(["", f"## Page {page['page_number']}"])
        if page["text"]:
            markdown_lines.extend(["", page["text"]])
        for table_index, table in enumerate(page["tables"], start=1):
            table_count += 1
            markdown_lines.extend(
                ["", f"### Table {table_index}", "", _markdown_table(table)]
            )
    markdown = "\n".join(markdown_lines).strip() + "\n"

    return {
        "format": OKF_FORMAT,
        "version": OKF_VERSION,
        "document_id": document_id,
        "source": str(source_path),
        "title": source_path.stem,
        "media_type": extraction.media_type,
        "page_count": len(pages),
        "table_count": table_count,
        "pages": pages,
        "markdown": markdown,
        "text": markdown,
    }


def write_okf_document(document: dict[str, Any], directory: str | Path) -> Path:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{document['document_id']}.md"
    metadata = {
        key: value
        for key, value in document.items()
        if key not in {"pages", "markdown", "text"}
    }
    front_matter = yaml.safe_dump(
        metadata,
        allow_unicode=True,
        sort_keys=False,
        width=1000,
    ).rstrip()
    destination.write_text(
        f"---\n{front_matter}\n---\n\n{document['markdown']}",
        encoding="utf-8",
    )
    return destination