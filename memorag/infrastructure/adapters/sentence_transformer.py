import numpy as np
from sentence_transformers import SentenceTransformer

from memorag.domain.ports import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding

    def get_embedding_dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension() or 0
