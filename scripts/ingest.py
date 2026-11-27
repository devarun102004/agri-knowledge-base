import argparse
from pathlib import Path

from config.settings import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OKF_DIR,
    DEFAULT_RAW_PDF_DIR,
    DEFAULT_QDRANT_COLLECTION,
)
from src.chunker import chunk_document
from src.data_cleaner import clean_extraction
from src.embedding_model import EmbeddingModel
from src.okf_builder import build_okf_document, write_okf_document
from src.pdf_extractor import SUPPORTED_SUFFIXES, extract_document
from src.qdrant_store import QdrantStore


def collect_sources(paths: list[Path]) -> list[Path]:
    sources: set[Path] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ValueError(f"Unsupported document type: {path}")
            sources.add(path.resolve())
        else:
            sources.update(
                item.resolve()
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
            )
    return sorted(sources)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract PDF/text documents, write OKF JSON, chunk, and index them."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=f"PDF/text files or directories (default: {DEFAULT_RAW_PDF_DIR})",
    )
    parser.add_argument("--okf-dir", type=Path, default=DEFAULT_OKF_DIR)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--collection", default=DEFAULT_QDRANT_COLLECTION)
    args = parser.parse_args()

    if args.chunk_size <= 0 or args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-size must be positive and --chunk-overlap must be >= 0 and smaller")

    input_paths = args.paths or [DEFAULT_RAW_PDF_DIR]
    if not args.paths:
        DEFAULT_RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    sources = collect_sources(input_paths)
    if not sources:
        parser.error(f"No PDF or text documents found in {', '.join(map(str, input_paths))}")

    store = QdrantStore(collection_name=args.collection)
    embedding_model = EmbeddingModel()
    for source in sources:
        extraction = clean_extraction(extract_document(source))
        document = build_okf_document(source, extraction)
        okf_path = write_okf_document(document, args.okf_dir)
        chunks = chunk_document(document, args.chunk_size, args.chunk_overlap)
        vectors = embedding_model.embed_many([chunk["text"] for chunk in chunks])
        count = store.upsert_document(document, chunks, vectors)
        print(f"Ingested {source}: {count} chunks in Qdrant; OKF: {okf_path}")


if __name__ == "__main__":
    main()