from datetime import datetime, timezone

from memorag.domain.entities import Document, Vector
from memorag.domain.ports import Chunker, Embedder, VectorStore


class IndexDocument:
    COLLECTION_NAME = "infos"

    def __init__(self, embedder: Embedder, vector_store: VectorStore, chunker: Chunker):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunker = chunker

    def execute(self, document: Document) -> None:
        """
        Embeds and indexes the provided document into the vector store.
        """
        chunked_document = self.chunker.chunk(document)
        vectors = []
        for chunk in chunked_document.chunks:
            embedding = self.embedder.embed(chunk.content.decode())
            vector = Vector(
                id=chunk.id,
                vector=embedding,
                content=chunk.content,
                inserted_at=datetime.now(tz=timezone.utc),
                metadata=(chunk.metadata or {})
                | {"document_id": str(document.id), "document_name": document.name},
            )
            vectors.append(vector)

        self.vector_store.index_vectors(
            vectors=vectors,
            collection_name=self.COLLECTION_NAME,
        )
