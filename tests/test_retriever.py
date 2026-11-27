import unittest
from pathlib import Path
from unittest.mock import Mock

from qdrant_client import QdrantClient, models

from src.chunker import chunk_text
from src.chunker import chunk_document
from src.data_cleaner import clean_extraction
from src.okf_builder import build_okf_document
from src.pdf_extractor import ExtractedDocument, ExtractedPage
from src.qdrant_store import QdrantStore
from src.retriever import PesticideRetriever


class FixedEmbeddingModel:
    def __init__(self, vector: list[float]):
        self.vector = vector
        self.queries: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.queries.append(text)
        return self.vector


class RetrievalTests(unittest.TestCase):
    def test_markdown_recommendations_keep_structured_fields_in_qdrant(self) -> None:
        client = QdrantClient(":memory:")
        vector = [1.0] + [0.0] * 383
        store = QdrantStore(collection_name="test_markdown_recommendations", client=client)
        embedding_model = FixedEmbeddingModel(vector)
        retriever = PesticideRetriever(store=store, embedding_model=embedding_model)
        extraction = clean_extraction(
            ExtractedDocument(
                media_type="text/markdown",
                pages=[
                    ExtractedPage(
                        1,
                        "\n".join(
                            [
                                "## Recommendation: Product A",
                                "",
                                "- Pesticide name: Product A",
                                "- Crop: Tomato",
                                "- Disease: Root-knot nematodes",
                                "- Dosage: 20 gm/kg",
                                "- Application instructions: Treat the seed.",
                                "- Source page: PDF page 10",
                                "",
                                "## Recommendation: Product B",
                                "",
                                "- Pesticide name: Product B",
                                "- Crop: Tomato",
                                "- Disease: Root-knot nematodes",
                                "- Dosage per ha: Active ingredient: 20 g; formulation: 200 g",
                                "- Waiting period: 10 days",
                                "- Application instructions: Treat the seed.",
                                "- Source page: PDF page 11",
                            ]
                        ),
                        [],
                    )
                ],
            )
        )
        document = build_okf_document(Path("/tmp/recommendations.md"), extraction)
        chunks = chunk_document(document)
        store.upsert_document(document, chunks, [vector for _ in chunks])

        recommendation = retriever.recommend("tomato root-knot nematodes")
        result = next(
            item
            for item in recommendation["recommendations"]
            if item["product"] == "Product A"
        )

        self.assertEqual(result["product"], "Product A")
        self.assertEqual(result["crop"], "Tomato")
        self.assertEqual(result["disease_or_pest"], "Root-knot nematodes")
        self.assertEqual(result["dosage"], "20 gm/kg")
        self.assertEqual(result["application_instructions"], "Treat the seed.")
        self.assertEqual(result["page"], 10)

        new_fields_result = next(
            item
            for item in recommendation["recommendations"]
            if item["product"] == "Product B"
        )
        self.assertEqual(
            new_fields_result["dosage"],
            "Active ingredient: 20 g; formulation: 200 g",
        )
        self.assertEqual(new_fields_result["waiting_period"], "10 days")
        self.assertEqual(new_fields_result["page"], 11)

    def test_qdrant_search_returns_source_backed_chunk(self) -> None:
        client = QdrantClient(":memory:")
        vector = [1.0] + [0.0] * 383
        store = QdrantStore(collection_name="test_pesticide_kb", client=client)
        embedding_model = FixedEmbeddingModel(vector)
        retriever = PesticideRetriever(store=store, embedding_model=embedding_model)
        extraction = clean_extraction(
            ExtractedDocument(
                media_type="application/pdf",
                pages=[
                    ExtractedPage(
                        page_number=1,
                        text="For maize affected by fall armyworm, follow the source guidance.",
                        tables=[
                            [
                                ["Crop", "Disease/Pest", "Product", "Active Ingredient"],
                                ["Maize", "Fall armyworm", "Product X", "Ingredient Y"],
                            ]
                        ],
                    )
                ],
            )
        )
        document = build_okf_document(Path("/tmp/maize.pdf"), extraction)
        chunks = chunk_document(document)
        store.upsert_document(document, chunks, [vector for _ in chunks])

        recommendation = retriever.recommend("maize fall armyworm")

        self.assertEqual(embedding_model.queries, ["maize fall armyworm"])
        self.assertEqual(recommendation["query"], "maize fall armyworm")
        result = next(
            item
            for item in recommendation["recommendations"]
            if item["product"] == "Product X"
        )
        self.assertEqual(result["crop"], "Maize")
        self.assertEqual(result["disease_or_pest"], "Fall armyworm")
        self.assertEqual(result["active_ingredient"], "Ingredient Y")
        self.assertEqual(result["source"], "/tmp/maize.pdf")
        self.assertEqual(result["page"], 1)

    def test_reingesting_document_removes_stale_qdrant_chunks(self) -> None:
        client = QdrantClient(":memory:")
        vector = [1.0] + [0.0] * 383
        store = QdrantStore(collection_name="test_pesticide_kb", client=client)
        source = Path("/tmp/guidance.txt")
        old_extraction = clean_extraction(
            ExtractedDocument(
                "text/plain",
                [ExtractedPage(1, "Old guidance. " * 80, [])],
            )
        )
        old_document = build_okf_document(source, old_extraction)
        old_chunks = chunk_document(old_document, chunk_size=100, overlap=10)
        store.upsert_document(old_document, old_chunks, [vector for _ in old_chunks])

        new_extraction = clean_extraction(
            ExtractedDocument(
                "text/plain",
                [ExtractedPage(1, "Updated guidance.", [])],
            )
        )
        new_document = build_okf_document(source, new_extraction)
        new_chunks = chunk_document(new_document)
        store.upsert_document(new_document, new_chunks, [vector for _ in new_chunks])

        self.assertEqual(client.count("test_pesticide_kb", exact=True).count, 1)
        self.assertIn("Updated guidance.", store.search(vector)[0]["text"])

    def test_upsert_document_sends_large_documents_in_batches(self) -> None:
        client = QdrantClient(":memory:")
        store = QdrantStore(collection_name="test_batched_upsert", client=client)
        original_upsert = client.upsert
        client.upsert = Mock(wraps=original_upsert)
        vector = [1.0] + [0.0] * 383
        document = {
            "document_id": "batched-document",
            "text": "Recommendation records",
            "source": "/tmp/recommendations.md",
            "title": "recommendations",
            "media_type": "text/markdown",
        }
        chunks = [
            {
                "ordinal": ordinal,
                "text": f"Record {ordinal}",
                "start_offset": ordinal,
                "end_offset": ordinal + 1,
            }
            for ordinal in range(65)
        ]

        count = store.upsert_document(document, chunks, [vector] * len(chunks))

        self.assertEqual(count, 65)
        self.assertEqual(
            [len(call.kwargs["points"]) for call in client.upsert.call_args_list],
            [64, 1],
        )
        self.assertEqual(
            client.count("test_batched_upsert", exact=True).count,
            65,
        )

    def test_search_prefers_structured_recommendations_over_raw_chunks(self) -> None:
        client = QdrantClient(":memory:")
        vector = [1.0] + [0.0] * 383
        store = QdrantStore(collection_name="test_structured_search", client=client)
        client.upsert(
            collection_name="test_structured_search",
            points=[
                models.PointStruct(
                    id=1,
                    vector=vector,
                    payload={"text": "Raw PDF table row"},
                ),
                models.PointStruct(
                    id=2,
                    vector=vector,
                    payload={"text": "Structured recommendation", "product": "Product A"},
                ),
            ],
            wait=True,
        )

        results = store.search(vector, limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["product"], "Product A")

    def test_search_falls_back_to_raw_chunks_when_no_recommendations_exist(self) -> None:
        client = QdrantClient(":memory:")
        vector = [1.0] + [0.0] * 383
        store = QdrantStore(collection_name="test_raw_search_fallback", client=client)
        client.upsert(
            collection_name="test_raw_search_fallback",
            points=[
                models.PointStruct(
                    id=1,
                    vector=vector,
                    payload={"text": "Raw PDF table row"},
                )
            ],
            wait=True,
        )

        results = store.search(vector, limit=1)

        self.assertEqual(results[0]["text"], "Raw PDF table row")

    def test_search_reranks_matching_crop_and_disease_above_embedding_score(self) -> None:
        client = QdrantClient(":memory:")
        query_vector = [1.0] + [0.0] * 383
        lower_vector = [0.9, (1 - 0.9**2) ** 0.5] + [0.0] * 382
        store = QdrantStore(collection_name="test_crop_disease_rerank", client=client)
        client.upsert(
            collection_name="test_crop_disease_rerank",
            points=[
                models.PointStruct(
                    id=1,
                    vector=query_vector,
                    payload={
                        "text": "Tomato root rot recommendation",
                        "crop": "Tomato",
                        "disease": "Root rot",
                        "product": "Product A",
                    },
                ),
                models.PointStruct(
                    id=2,
                    vector=lower_vector,
                    payload={
                        "text": "Tomato early blight recommendation",
                        "crop": "Tomato",
                        "disease": "Early blight",
                        "product": "Product B",
                    },
                ),
            ],
            wait=True,
        )
        retriever = PesticideRetriever(
            store=store,
            embedding_model=FixedEmbeddingModel(query_vector),
        )

        results = retriever.search("Tomato Early blight", limit=5)

        self.assertEqual([result["product"] for result in results], ["Product B"])

    def test_creates_payload_indexes_for_document_replacement_filters(self) -> None:
        client = Mock(spec=QdrantClient)
        client.collection_exists.return_value = True

        QdrantStore(collection_name="test_pesticide_kb", client=client)

        indexed_fields = {
            call.kwargs["field_name"]
            for call in client.create_payload_index.call_args_list
        }
        self.assertEqual(
            indexed_fields,
            {"document_id", "document_revision", "product"},
        )

    def test_rejects_empty_query_and_nonpositive_limit(self) -> None:
        embedding_model = FixedEmbeddingModel([1.0] + [0.0] * 383)
        retriever = PesticideRetriever(
            store=QdrantStore(
                collection_name="test_pesticide_kb",
                client=QdrantClient(":memory:"),
            ),
            embedding_model=embedding_model,
        )
        with self.assertRaisesRegex(ValueError, "query must not be empty"):
            retriever.search("  ")
        with self.assertRaisesRegex(ValueError, "limit"):
            retriever.search("disease", limit=0)
        self.assertEqual(embedding_model.queries, [])

    def test_chunks_respect_size_and_keep_overlap_on_word_boundaries(self) -> None:
        chunks = chunk_text(
            "Wear gloves when applying pesticide. Follow label directions.",
            chunk_size=30,
            overlap=5,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk["text"]) <= 30 for chunk in chunks))
        self.assertTrue(any("applying" in chunk["text"] for chunk in chunks))
        self.assertTrue(any("pesticide." in chunk["text"] for chunk in chunks))


if __name__ == "__main__":
    unittest.main()