# MemoRAG

**MemoRAG** is a Retrieval-Augmented Generation (RAG) Command Line Interface (CLI) application designed to store, index, and retrieve technical decisions and documentation efficiently.

It leverages modern vector search technologies and Large Language Models (LLMs) to provide context-aware answers from your indexed documents.

## 🎯 Purpose

Have you ever found yourself wondering, _"Why did we choose this library?"_ or _"How did we configure this service three months ago?"_?

The primary goal of **MemoRAG** is to serve as an intelligent memory for your project's technical history. Instead of manually digging through commit logs, tickets, or scattered markdown files to find the context behind a decision, you can simply ask MemoRAG. It retrieves the specific technical choices and documentation you indexed weeks or months ago, saving you valuable time and context switching.

## 🏗 Architecture

The project follows the principles of **Clean Architecture** (also known as Hexagonal Architecture) to separate concerns and ensure maintainability:

- **Domain**: Contains business entities and interfaces (ports).
- **Application**: Contains the use cases (e.g., Index Document, Search Documents).
- **Infrastructure**: Contains implementations of the adapters (Qdrant, LiteLLM, SentenceTransformers, etc.).
- **Presentation**: Contains the entry points, currently the CLI.

## 🚀 Features

- **Document Indexing**
  - Chunking and embedding of documents.
  - Storage in **Qdrant** vector database.
  - Uses `SentenceTransformers` for high-quality embeddings.

- **Intelligent Search**
  - Semantic search using vector similarity.
  - **Reranking** step using Cross-Encoders for improved relevance.
  - Answer generation using **LiteLLM** (supports various providers like Ollama, Mistral, OpenAI, etc.).

## 🛠 Prerequisites

- **Python 3.10+**
- **Docker** and **Docker Compose** (for running Qdrant)

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/patacoing/memorag.git
   cd memorag
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install .
   ```

## 🏃 Usage

### 1. Start Qdrant

First, you need to have a Qdrant instance running. A `docker-compose.yml` file is provided for convenience.

```bash
docker-compose up -d
```

This will start Qdrant on `localhost:6333`.

### 2. CLI Commands

The application provides a CLI to interact with the system.

#### Global Options
You can configure the models and connection settings via global options:
- `--embedding-model`: Name of the embedding model (default: `all-MiniLM-L6-v2`)
- `--rerank-model`: Name of the reranking model (default: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`)
- `--llm-model`: Name of the LLM model (default: `mistral/mistral-tiny`)
- `--qdrant-host`: Qdrant host (default: `localhost`)
- `--qdrant-port`: Qdrant port (default: `6333`)

#### Indexing a Document
Use the `index` command to index a file.

```bash
memorag index --file path/to/your/document.txt
```

#### Searching
Use the `search` command to query your knowledge base.

```bash
memorag search "What is the architecture of this project?"
```

## 🔮 Roadmap & Next Steps

The project is continually evolving. The following features are planned:

- [ ] **Hybrid Search**: Implement BM25 indexer and hybrid retrieval for better recall.
- [ ] **Contextual Retrieval**: Implement strategies like summing up prepended chunks.
- [ ] **Advanced Chunking**: Specific chunking strategies based on file type (Markdown, Python, etc.).
- [ ] **Bulk Operations**: Bulk indexing capabilities for processing large datasets.
- [ ] **Web UI**: A user-friendly web interface to chat with your data.
- [ ] **Rich Configuration**: Configuration support through files (YAML/TOML).

## 📄 License

This project is licensed under the **MIT License**.
