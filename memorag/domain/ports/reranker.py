from abc import ABC, abstractmethod

from memorag.domain.entities import Vector


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, vectors: list[Vector]) -> list[Vector]:
        """Rerank the list of documents based on their relevance to the query."""
        pass
