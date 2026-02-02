from sentence_transformers import CrossEncoder

from memorag.domain.entities import Vector
from memorag.domain.ports import Reranker


class CrossEncoderReranker(Reranker):
    def __init__(self, model: str) -> None:
        self.model = CrossEncoder(model)

    def rerank(self, query: str, vectors: list[Vector]) -> list[Vector]:
        pairs = [(query, doc.content.decode()) for doc in vectors]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(vectors, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )

        return [doc for doc, _ in ranked]
