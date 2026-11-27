import re
from typing import Any


_TABLE_FIELDS = {
    "crop": {"crop", "crop name", "host crop"},
    "disease": {
        "disease",
        "disease name",
        "pest",
        "pest disease",
        "disease pest",
        "target pest",
        "target disease",
    },
    "product": {
        "product",
        "product name",
        "recommended product",
        "pesticide",
        "trade name",
        "brand",
    },
    "active_ingredient": {
        "active ingredient",
        "active ingredient ai",
        "ingredient",
    },
}
_RECOMMENDATION_HEADING = re.compile(r"(?m)^## Recommendation:\s*.+$")
_MARKDOWN_FIELD = re.compile(r"(?m)^-\s+([^:]+):\s*(.*)$")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[dict[str, Any]]:
    """Split text into bounded, overlapping character chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and smaller than chunk_size")

    normalized = "\n".join(line.strip() for line in text.splitlines()).strip()
    if not normalized:
        return []

    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end + 1)
            if boundary > start:
                end = boundary

        chunk = normalized[start:end].strip()
        if chunk:
            left_trim = len(normalized[start:end]) - len(normalized[start:end].lstrip())
            right_trim = len(normalized[start:end]) - len(normalized[start:end].rstrip())
            chunks.append(
                {
                    "ordinal": len(chunks),
                    "text": chunk,
                    "start_offset": start + left_trim,
                    "end_offset": end - right_trim,
                }
            )

        if end == len(normalized):
            break
        target_start = max(start + 1, end - overlap)
        previous_boundary = normalized.rfind(" ", start, target_start + 1)
        next_start = previous_boundary + 1 if previous_boundary > start else target_start
        start = next_start

    return chunks


def _field_for_header(header: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", header.casefold()).strip()
    for field, aliases in _TABLE_FIELDS.items():
        if normalized in aliases:
            return field
    return None


def _chunk_recommendations(
    text: str,
    default_page_number: int,
    chunk_size: int,
    overlap: int,
) -> list[dict[str, Any]] | None:
    headings = list(_RECOMMENDATION_HEADING.finditer(text))
    if not headings:
        return None

    chunks: list[dict[str, Any]] = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        record_text = text[heading.start():end].strip()
        fields = {
            label.strip().casefold(): value.strip()
            for label, value in _MARKDOWN_FIELD.findall(record_text)
        }
        source_page = re.search(r"PDF pages?\s+(\d+)", fields.get("source page", ""), re.I)
        page_number = int(source_page.group(1)) if source_page else default_page_number
        disease = fields.get("disease") or fields.get("name of insect")
        metadata = {
            "crop": fields.get("crop", ""),
            "disease": disease or "",
            "product": fields.get("pesticide name", ""),
            "dosage": fields.get("dosage per ha") or fields.get("dosage", ""),
            "waiting_period": fields.get("waiting period", ""),
            "application_instructions": fields.get("application instructions", ""),
            "name_of_insect": fields.get("name of insect", ""),
            "habitat": fields.get("habitat", ""),
            "surface": fields.get("surface", ""),
        }
        for chunk in chunk_text(record_text, chunk_size, overlap):
            chunks.append(
                {
                    **chunk,
                    "page_number": page_number,
                    "table_index": None,
                    **metadata,
                }
            )
    return chunks


def chunk_document(
    document: dict[str, Any],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict[str, Any]]:
    """Chunk page text and table rows while retaining page and table metadata."""
    chunks: list[dict[str, Any]] = []
    for page in document["pages"]:
        page_number = page["page_number"]
        recommendation_chunks = _chunk_recommendations(
            page["text"], page_number, chunk_size, overlap
        )
        if recommendation_chunks is None:
            for chunk in chunk_text(page["text"], chunk_size, overlap):
                chunks.append(
                    {
                        **chunk,
                        "ordinal": len(chunks),
                        "page_number": page_number,
                        "table_index": None,
                    }
                )
        else:
            for chunk in recommendation_chunks:
                chunks.append({**chunk, "ordinal": len(chunks)})

        for table_index, table in enumerate(page["tables"], start=1):
            if len(table) > 1:
                headers = table[0]
                data_rows = table[1:]
            else:
                headers = [f"Column {index + 1}" for index in range(len(table[0]))]
                data_rows = table

            for row in data_rows:
                cells = row + [""] * (len(headers) - len(row))
                fields = {
                    field: value.strip()
                    for header, value in zip(headers, cells)
                    if (field := _field_for_header(header)) and value.strip()
                }
                row_text = (
                    f"Page {page_number}, table {table_index}. "
                    + "; ".join(
                        f"{header}: {value}"
                        for header, value in zip(headers, cells)
                        if value.strip()
                    )
                )
                for chunk in chunk_text(row_text, chunk_size, overlap):
                    chunks.append(
                        {
                            **chunk,
                            "ordinal": len(chunks),
                            "page_number": page_number,
                            "table_index": table_index,
                            **fields,
                        }
                    )
    return chunks