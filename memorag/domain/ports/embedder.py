from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Convert the input text into a vector embedding."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Return the dimension of the embeddings produced by this embedder."""
        pass
