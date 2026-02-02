from datetime import datetime
from uuid import UUID

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

from memorag.domain.entities import Vector
from memorag.domain.ports import VectorStore


class QdrantVectorStore(VectorStore):
    def __init__(self, host: str, port: int, embedding_dim: int) -> None:
        self._client = QdrantClient(host=host, port=port)
        self.embedding_dim = embedding_dim

    def _create_collection_if_not_exist(self, collection_name: str) -> None:
        if self._client.collection_exists(collection_name):
            return

        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
        )

    def _build_points_from_vectors(self, vectors: list[Vector]) -> list[PointStruct]:
        points = []
        for vector in vectors:
            point = PointStruct(
                id=str(vector.id),
                vector=vector.vector.tolist(),
                payload={
                    "content": vector.content.decode("utf-8"),
                    "inserted_at": vector.inserted_at.isoformat(),
                    "metadata": vector.metadata,
                },
            )
            points.append(point)
        return points

    def _build_vectors_from_points(
        self, points: list[PointStruct] | list[ScoredPoint]
    ) -> list[Vector]:
        vectors = []
        for point in points:
            payload = point.payload or {}
            # Handle cases where vector might be missing or in a different format
            vec_data = point.vector
            if vec_data is None:
                # Should not happen if we request with_vectors=True, but strictly
                # speaking it's optional
                # For now assuming it is present as our domain entity requires it.
                raise ValueError(f"Vector data missing for point {point.id}")

            vector = Vector(
                id=UUID(str(point.id)),
                vector=np.array(vec_data),
                inserted_at=datetime.fromisoformat(payload["inserted_at"]),
                content=payload["content"].encode("utf-8"),
                metadata=payload.get("metadata"),
            )
            vectors.append(vector)
        return vectors

    def index_vector(self, vector: Vector, collection_name: str) -> None:
        return self.index_vectors([vector], collection_name)

    def index_vectors(self, vectors: list[Vector], collection_name: str) -> None:
        self._create_collection_if_not_exist(collection_name)
        points = self._build_points_from_vectors(vectors)
        self._client.upsert(
            collection_name=collection_name,
            points=points,
        )

    def search_similar(
        self, query_vector: np.ndarray, top_k: int, collection_name: str
    ) -> list[Vector]:
        self._create_collection_if_not_exist(collection_name)
        response = self._client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            with_vectors=True,
        )
        return self._build_vectors_from_points(response.points)
