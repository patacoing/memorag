from memorag.domain.entities import SearchResponse
from memorag.domain.ports import Embedder, Generator, Reranker, VectorStore


class SearchDocuments:
    COLLECTION_NAME = "infos"

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        reranker: Reranker,
        generator: Generator,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.generator = generator

    def execute(self, query: str, top_k: int = 20, raw: bool = False) -> SearchResponse:
        """
        Searches for documents similar to the query, reranks them, and generates an answer.
        """
        query_embedding = self.embedder.embed(query)

        candidates_vectors = self.vector_store.search_similar(
            query_vector=query_embedding,
            top_k=top_k,
            collection_name=self.COLLECTION_NAME,
        )

        reranked_vectors = self.reranker.rerank(
            query=query,
            vectors=candidates_vectors,
        )

        if raw:
            answer = "\n\n".join([doc.content.decode("utf-8") for doc in reranked_vectors])
            return SearchResponse(answer=[answer], sources=reranked_vectors)

        # Build context from reranked vectors
        context = "\n\n".join([doc.content.decode("utf-8") for doc in reranked_vectors])

        # Generate answer
        answer = self.generator.generate(context=context, query=query)

        return SearchResponse(answer=answer, sources=reranked_vectors)
