from collections.abc import Sequence

from fastembed import TextEmbedding

from config.settings import DEFAULT_EMBEDDING_MODEL


class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot create embeddings for empty text")
        return [vector.tolist() for vector in self._model.embed(texts)]