import hashlib
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from config.settings import (
    DEFAULT_QDRANT_COLLECTION,
    EMBEDDING_SIZE,
    QDRANT_API_KEY,
    QDRANT_URL,
)


class QdrantStore:
    def __init__(
        self,
        collection_name: str = DEFAULT_QDRANT_COLLECTION,
        client: QdrantClient | None = None,
    ):
        self.collection_name = collection_name
        if client is None:
            if not QDRANT_URL or not QDRANT_API_KEY:
                raise RuntimeError(
                    "Set QDRANT_URL and QDRANT_API_KEY in the local .env file "
                    "before using Qdrant Cloud."
                )
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.client = client
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )
        for field_name in ("document_id", "document_revision"):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

    def upsert_document(
        self,
        document: dict[str, Any],
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("The number of chunks must match the number of vectors")
        if not chunks:
            return 0

        document_id = document["document_id"]
        revision = hashlib.sha256(document["text"].encode("utf-8")).hexdigest()
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            payload = {
                "document_id": document_id,
                "document_revision": revision,
                "source": document["source"],
                "title": document["title"],
                "media_type": document["media_type"],
                "ordinal": chunk["ordinal"],
                "text": chunk["text"],
                "start_offset": chunk["start_offset"],
                "end_offset": chunk["end_offset"],
                "page_number": chunk.get("page_number", 1),
                "table_index": chunk.get("table_index"),
            }
            for field in (
                "crop",
                "disease",
                "product",
                "active_ingredient",
                "dosage",
                "application_instructions",
                "name_of_insect",
                "habitat",
                "surface",
            ):
                if field in chunk:
                    payload[field] = chunk[field]
            points.append(
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, f"{document_id}:{chunk['ordinal']}")),
                    vector=vector,
                    payload=payload,
                )
            )
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="document_revision",
                            match=models.MatchValue(value=revision),
                        )
                    ],
                )
            ),
            wait=True,
        )
        return len(points)

    def search(self, vector: list[float], limit: int = 5) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return [
            {"score": point.score, **(point.payload or {})}
            for point in response.points
        ]