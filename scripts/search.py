import argparse
import json

from config.settings import DEFAULT_QDRANT_COLLECTION
from src.qdrant_store import QdrantStore
from src.retriever import PesticideRetriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve source-backed pesticide guidance from Qdrant."
    )
    parser.add_argument("query", help="Crop and/or disease name, or a full question")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--collection", default=DEFAULT_QDRANT_COLLECTION)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be greater than zero")

    retriever = PesticideRetriever(store=QdrantStore(collection_name=args.collection))
    recommendation = retriever.recommend(args.query, args.limit)
    if args.format == "json":
        print(json.dumps(recommendation, ensure_ascii=False, indent=2))
        return

    if not recommendation["recommendations"]:
        print("No matching chunks found.")
        return
    for result in recommendation["recommendations"]:
        print(f"Crop: {result['crop'] or 'Not explicitly identified'}")
        print(f"Disease/pest: {result['disease_or_pest'] or 'Not explicitly identified'}")
        print(f"Product: {result['product'] or 'See source evidence'}")
        print(f"Active ingredient: {result['active_ingredient'] or 'Not explicitly identified'}")
        print(f"Evidence: {result['evidence']}")
        print(f"Source: {result['source']} (page {result['page']})")
        print(f"Relevance: {result['relevance_score']:.3f}")
        print()


if __name__ == "__main__":
    main()