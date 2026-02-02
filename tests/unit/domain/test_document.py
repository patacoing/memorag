import uuid

from memorag.domain.entities.document import Document, DocumentType
from memorag.domain.entities.document_chunk import DocumentChunk


class TestDocument:
    def test_document_creation_from_text(self):
        content = "Hello world"
        doc = Document.from_text(content)

        assert doc.name == "text_document"
        assert doc.content == b"Hello world"
        assert doc.type == DocumentType.TEXT
        assert isinstance(doc.id, uuid.UUID)

    def test_document_chunks_generator(self):
        # Create a chain of 3 chunks
        chunk3 = DocumentChunk(id=uuid.uuid4(), content=b"chunk3")
        chunk2 = DocumentChunk(id=uuid.uuid4(), content=b"chunk2", next_chunk=chunk3)
        chunk1 = DocumentChunk(id=uuid.uuid4(), content=b"chunk1", next_chunk=chunk2)

        doc = Document(
            id=uuid.uuid4(), name="test", content=b"full content", type=DocumentType.TEXT
        )
        doc._chunk = chunk1  # Directly set private attribute after initialization
        # because pydantic filters it out

        chunks = list(doc.chunks)

        assert len(chunks) == 3
        assert chunks[0].content == b"chunk1"
        assert chunks[1].content == b"chunk2"
        assert chunks[2].content == b"chunk3"

    def test_document_chunks_empty(self):
        doc = Document(id=uuid.uuid4(), name="test", content=b"content", type=DocumentType.TEXT)

        chunks = list(doc.chunks)
        assert len(chunks) == 0
