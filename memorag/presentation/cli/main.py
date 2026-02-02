from typing import Annotated
from unittest.mock import MagicMock

import typer

app = typer.Typer()


def _parse_document_type(value: str):
    from memorag.domain.entities.document import DocumentType

    choices = list(map(str, DocumentType))

    try:
        return DocumentType(value)
    except ValueError as e:
        choices = ", ".join(choices)
        raise typer.BadParameter(f"Invalid filetype '{value}'. Choose one of: {choices}") from e


@app.callback()
def main(
    ctx: typer.Context,
    embedding_model: str = typer.Option(
        "all-MiniLM-L6-v2", help="SentenceTransformer embedding model name"
    ),
    rerank_model: str = typer.Option(
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", help="CrossEncoder reranking model name"
    ),
    llm_model: str = typer.Option(
        "mistral/mistral-tiny", help="LLM model name (e.g., mistral/mistral-tiny, ollama/llama2)"
    ),
    embedding_dim: int = typer.Option(384, help="Embedding vector dimension"),
    qdrant_host: str = typer.Option("localhost", help="Qdrant host"),
    qdrant_port: int = typer.Option(6333, help="Qdrant port"),
):
    """
    Global configuration for the MemoRAG application.
    """
    ctx.obj = dict(
        embedding_model=embedding_model,
        rerank_model=rerank_model,
        llm_model=llm_model,
        embedding_dim=embedding_dim,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
    )


@app.command()
def index(
    ctx: typer.Context,
    text: Annotated[str | None, typer.Argument(help="Text to index")] = None,
    filepath: Annotated[str | None, typer.Option(help="Path to text file to index")] = None,
    filetype: Annotated[
        str,
        typer.Option(
            help="Type of the document. This will help determine how to chunk the document.",
            show_default=True,
            callback=lambda ctx, param, value: _parse_document_type(value),
        ),
    ] = "text",
):
    from memorag.application.use_cases import IndexDocument
    from memorag.domain.entities import Document
    from memorag.domain.ports import Chunker
    from memorag.infrastructure.adapters import QdrantVectorStore, SentenceTransformerEmbedder

    embedding_model = ctx.obj["embedding_model"]
    qdrant_host = ctx.obj["qdrant_host"]
    qdrant_port = ctx.obj["qdrant_port"]
    embedding_dim = ctx.obj["embedding_dim"]

    chunker = MagicMock(spec=Chunker)  # Placeholder for Chunker implementation

    def mock_chunk(document: Document) -> Document:
        # Simple mock chunking: create one chunk with the whole content
        from memorag.domain.entities import DocumentChunk

        chunk = DocumentChunk(
            id=document.id,
            content=document.content,
            metadata=document.metadata,
            next_chunk=None,
        )
        document._chunk = chunk
        return document

    chunker.chunk.side_effect = mock_chunk
    embedder = SentenceTransformerEmbedder(model_name=embedding_model)
    vector_store = QdrantVectorStore(
        host=qdrant_host, port=qdrant_port, embedding_dim=embedding_dim
    )
    index_uc = IndexDocument(embedder=embedder, vector_store=vector_store, chunker=chunker)

    use_case: IndexDocument = index_uc
    if text is not None:
        document = Document.from_text(text)
    else:
        if filepath is None:
            typer.echo("Either text or filepath must be provided.", err=True)
            raise typer.Exit(code=1)

        document = Document.from_filepath(filepath, filetype)  # type: ignore
    use_case.execute(document)
    typer.echo(f"Indexed document: {document.id} - {document.name}")


@app.command()
def search(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="Query text to search for similar documents")],
    top_k: Annotated[
        int, typer.Option(help="Number of top similar documents to retrieve", show_default=True)
    ] = 10,
    raw: Annotated[bool, typer.Option(help="Output raw answer without llm processing")] = False,
):
    from memorag.application.use_cases import SearchDocuments
    from memorag.infrastructure.adapters import (
        CrossEncoderReranker,
        LiteLLMGenerator,
        QdrantVectorStore,
        SentenceTransformerEmbedder,
    )

    embedding_model = ctx.obj["embedding_model"]
    rerank_model = ctx.obj["rerank_model"]
    llm_model = ctx.obj["llm_model"]
    qdrant_host = ctx.obj["qdrant_host"]
    qdrant_port = ctx.obj["qdrant_port"]
    embedding_dim = ctx.obj["embedding_dim"]

    # Instantiate adapters
    reranker = CrossEncoderReranker(model=rerank_model)
    generator = LiteLLMGenerator(model_name=llm_model)
    embedder = SentenceTransformerEmbedder(model_name=embedding_model)
    vector_store = QdrantVectorStore(
        host=qdrant_host, port=qdrant_port, embedding_dim=embedding_dim
    )

    # Instantiate use cases with their specific dependencies
    search_uc = SearchDocuments(
        embedder=embedder, vector_store=vector_store, reranker=reranker, generator=generator
    )

    use_case: SearchDocuments = search_uc
    response = use_case.execute(query, top_k, raw)

    typer.echo("--------------------------------------------------")
    typer.echo("🤖 Answer: ", nl=False)
    for chunk in response.answer:
        typer.echo(chunk, nl=False)
    typer.echo("\n--------------------------------------------------")
    typer.echo("\n📚 Sources (All might not be relevant):")
    for i, source in enumerate(response.sources, 1):
        typer.echo(
            (
                f"  [{i}] - document name : {source.document_name}, chunk id : {source.id},",
                f"document id : {source.document_id}",
            )
        )


if __name__ == "__main__":
    app()
