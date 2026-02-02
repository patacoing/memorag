import uuid
from unittest.mock import Mock

import numpy as np
import pytest

from memorag.application.use_cases.index_document import IndexDocument
from memorag.domain.entities import Document, DocumentChunk, DocumentType


class TestIndexDocument:
    @pytest.fixture
    def mock_embedder(self):
        embedder = Mock()
        embedder.embed.return_value = np.array([0.1, 0.2, 0.3])
        return embedder

    @pytest.fixture
    def mock_vector_store(self):
        return Mock()

    @pytest.fixture
    def mock_chunker(self):
        return Mock()

    def test_execute_indexes_chunks_correctly(
        self, mock_embedder, mock_vector_store, mock_chunker
    ):
        # Setup
        use_case = IndexDocument(mock_embedder, mock_vector_store, mock_chunker)

        # Create a document and mock chunking result
        doc = Document(
            id=uuid.uuid4(), name="test_doc", content=b"content", type=DocumentType.TEXT
        )

        # Mock the chunker to return a document with chunks attached
        chunk = DocumentChunk(id=uuid.uuid4(), content=b"chunk content", metadata={"page": "1"})
        # Create a new document instance that has the chunk attached
        chunked_doc = doc.model_copy()
        chunked_doc._chunk = chunk
        mock_chunker.chunk.return_value = chunked_doc

        # Execute
        use_case.execute(doc)

        # Verify interactions
        mock_chunker.chunk.assert_called_once_with(doc)
        mock_embedder.embed.assert_called_once_with("chunk content")

        # Verify vector store call
        mock_vector_store.index_vectors.assert_called_once()
        call_args = mock_vector_store.index_vectors.call_args
        vectors = call_args.kwargs["vectors"]
        collection_name = call_args.kwargs["collection_name"]

        assert collection_name == "infos"
        assert len(vectors) == 1
        assert vectors[0].id == chunk.id
        assert vectors[0].content == b"chunk content"
        assert np.array_equal(vectors[0].vector, np.array([0.1, 0.2, 0.3]))
        assert vectors[0].metadata["page"] == "1"
        assert vectors[0].metadata["document_id"] == str(doc.id)
