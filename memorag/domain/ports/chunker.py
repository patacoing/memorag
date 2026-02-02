from abc import ABC, abstractmethod

from memorag.domain.entities import Document


class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> Document:
        pass
