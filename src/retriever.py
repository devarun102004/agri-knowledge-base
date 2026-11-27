from typing import Any

from src.embedding_model import EmbeddingModel
from src.qdrant_store import QdrantStore


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
        return self.store.search(vector, limit)

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