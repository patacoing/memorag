from abc import ABC, abstractmethod

import numpy as np

from ..entities import Vector


class VectorStore(ABC):
    @abstractmethod
    def index_vector(self, vector: Vector, collection_name: str) -> None:
        """Index a vector with associated metadata."""
        pass

    @abstractmethod
    def index_vectors(self, vectors: list[Vector], collection_name: str) -> None:
        """Index multiple vectors with associated metadata."""
        pass

    @abstractmethod
    def search_similar(
        self, query_vector: np.ndarray, top_k: int, collection_name: str
    ) -> list[Vector]:
        """Search for the most similar vectors to the query vector."""
        pass
