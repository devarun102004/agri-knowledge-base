import re
from typing import Any

from src.embedding_model import EmbeddingModel
from src.qdrant_store import QdrantStore

_TOKEN = re.compile(r"[a-z0-9]+")
_QUERY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "concentration",
    "concenteration",
    "crop",
    "disease",
    "diseases",
    "dose",
    "dosage",
    "for",
    "from",
    "give",
    "me",
    "name",
    "of",
    "on",
    "pest",
    "product",
    "recommend",
    "recommendation",
    "show",
    "the",
    "then",
    "waiting",
    "with",
    "period",
    "what",
    "which",
}


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in _TOKEN.findall(value.casefold())
        if token not in _QUERY_STOP_WORDS
    }


class PesticideRetriever:
    def __init__(
        self,
        store: QdrantStore | None = None,
        embedding_model: EmbeddingModel | None = None,
    ):
        self.store = store or QdrantStore()
        self.embedding_model = embedding_model or EmbeddingModel()

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        vector = self.embedding_model.embed(query)
        results = self.store.search(vector, max(limit * 10, 50))
        query_tokens = _tokens(query)
        if query_tokens and any(result.get("product") for result in results):
            def lexical_match(result: dict[str, Any]) -> tuple[int, int]:
                crop_tokens = _tokens(str(result.get("crop") or ""))
                record_tokens = _tokens(
                    " ".join(
                        str(result.get(field) or "")
                        for field in (
                            "crop",
                            "disease",
                            "product",
                            "name_of_insect",
                            "habitat",
                            "surface",
                        )
                    )
                )
                return (
                    len(query_tokens & crop_tokens),
                    len(query_tokens & record_tokens),
                )

            results.sort(
                key=lambda result: (
                    *lexical_match(result),
                    result["score"],
                ),
                reverse=True,
            )
            best_match = lexical_match(results[0])
            if best_match[1] > 0:
                results = [
                    result
                    for result in results
                    if lexical_match(result) == best_match
                ]
        return results[:limit]

    def recommend(self, query: str, limit: int = 5) -> dict[str, Any]:
        results = self.search(query, limit)
        return {
            "query": query,
            "recommendations": [
                {
                    "crop": result.get("crop"),
                    "disease_or_pest": result.get("disease"),
                    "product": result.get("product"),
                    "active_ingredient": result.get("active_ingredient"),
                    "dosage": result.get("dosage"),
                    "waiting_period": result.get("waiting_period"),
                    "application_instructions": result.get("application_instructions"),
                    "name_of_insect": result.get("name_of_insect"),
                    "habitat": result.get("habitat"),
                    "surface": result.get("surface"),
                    "evidence": result["text"],
                    "source": result.get("source"),
                    "page": result.get("page_number"),
                    "relevance_score": result["score"],
                }
                for result in results
            ],
            "note": (
                "Product fields are populated only when clearly labelled in a source "
                "table; otherwise use the evidence passage. Verify current local "
                "registration and follow the product label."
            ),
        }