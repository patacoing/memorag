from collections.abc import Generator
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel

from .document_chunk import DocumentChunk


class DocumentType(str, Enum):
    PDF = "pdf"
    TEXT = "text"
    MARKDOWN = "markdown"
    WORD = "word"
    OTHER = "other"


class Document(BaseModel):
    id: UUID
    name: str
    content: bytes
    metadata: dict[str, str] | None = None
    type: DocumentType
    _chunk: DocumentChunk | None = None

    @property
    def chunks(self) -> Generator[DocumentChunk, None, None]:
        if self._chunk is None:
            return
        current_chunk = self.model_copy(deep=True)._chunk
        while current_chunk is not None:
            yield current_chunk
            current_chunk = current_chunk.next_chunk

    @classmethod
    def from_text(cls, content: str) -> "Document":
        return cls(
            id=uuid4(),
            name="text_document",
            content=content.encode("utf-8"),
            type=DocumentType.TEXT,
        )

    @classmethod
    def from_filepath(cls, filepath: str, filetype: DocumentType) -> "Document":
        with open(filepath, "rb") as f:
            content = f.read()
        name = filepath.split("/")[-1]
        return cls(
            id=uuid4(),
            name=name,
            content=content,
            type=filetype,
        )
