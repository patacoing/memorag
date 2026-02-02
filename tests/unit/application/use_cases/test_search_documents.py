import uuid
from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pytest

from memorag.application.use_cases.search_documents import SearchDocuments
from memorag.domain.entities import Vector


class TestSearchDocuments:
    @pytest.fixture
    def mock_deps(self):
        return {
            "embedder": Mock(),
            "vector_store": Mock(),
            "reranker": Mock(),
            "generator": Mock(),
        }

    def test_execute_flow(self, mock_deps):
        # Setup
        use_case = SearchDocuments(**mock_deps)
        query = "test query"

        # Mock embedding
        query_embedding = np.array([0.1, 0.2])
        mock_deps["embedder"].embed.return_value = query_embedding

        # Mock vector search results
        initial_vectors = [
            Vector(
                id=uuid.uuid4(),
                vector=np.array([0.1, 0.2]),
                content=b"doc1",
                inserted_at=datetime.now(),
            ),
            Vector(
                id=uuid.uuid4(),
                vector=np.array([0.3, 0.4]),
                content=b"doc2",
                inserted_at=datetime.now(),
            ),
        ]
        mock_deps["vector_store"].search_similar.return_value = initial_vectors

        # Mock reranking (returns subset or reordered)
        reranked_vectors = [initial_vectors[1]]  # Assume second doc is better
        mock_deps["reranker"].rerank.return_value = reranked_vectors

        # Mock generation
        expected_answer = "Generated answer"

        def mock_generator(context, query):
            yield expected_answer

        mock_deps["generator"].generate.side_effect = mock_generator

        # Execute
        response = use_case.execute(query, top_k=5)

        # Verify
        mock_deps["embedder"].embed.assert_called_once_with(query)

        mock_deps["vector_store"].search_similar.assert_called_once_with(
            query_vector=query_embedding, top_k=5, collection_name="infos"
        )

        mock_deps["reranker"].rerank.assert_called_once_with(query=query, vectors=initial_vectors)

        # Verify context construction passed to generator
        mock_deps["generator"].generate.assert_called_once()
        call_kwargs = mock_deps["generator"].generate.call_args.kwargs
        assert call_kwargs["query"] == query
        assert call_kwargs["context"] == "doc2"  # Only reranked vector content

        # Generator returns an iterator, so we need to consume it to check the content
        generated_content = "".join(list(response.answer))
        assert generated_content == expected_answer
        assert response.sources == reranked_vectors
