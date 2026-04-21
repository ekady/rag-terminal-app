# 🔍 Basic RAG Terminal

A terminal-based Retrieval-Augmented Generation (RAG) application using Python, PostgreSQL + pgvector, and OpenAI.

## Architecture

```
documents/       →  indexer.py  →  PostgreSQL (pgvector)
                                        ↓
user query       →    app.py    →  similarity search → LLM → answer
```

| File | Purpose |
|------|---------|
| `config.py` | Configuration management (loads `.env`) |
| `db.py` | Database connection, schema setup |
| `embeddings.py` | Text chunking and embedding generation |
| `indexer.py` | **Stage 1** — File indexing pipeline |
| `rag.py` | RAG retrieval and LLM query |
| `app.py` | **Stage 2** — Interactive terminal app |

## Prerequisites

- Python 3.10+
- PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) extension installed
- OpenAI API key

### Install pgvector

```bash
# macOS (Homebrew)
brew install pgvector

# Or from source (inside PostgreSQL)
# See: https://github.com/pgvector/pgvector#installation
```

### Create the database

```bash
createdb rag_db
psql rag_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Setup

1. **Clone and enter the project:**
   ```bash
   cd basic-rag-terminal
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API key and database credentials
   ```

## Usage

### Stage 1: Index Files

```bash
# Index a single file
python indexer.py path/to/file.txt

# Index an entire directory (recursively)
python indexer.py path/to/documents/

# Reset database and re-index
python indexer.py path/to/documents/ --reset
```

### Stage 2: Query via Terminal

```bash
python app.py
```

**Available commands:**
| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/sources` | Toggle source document display |
| `/clear` | Clear the screen |
| `/quit` | Exit the app |

Type any natural language question to query your indexed documents.

## Configuration

All settings are managed via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `rag_db` | Database name |
| `POSTGRES_USER` | `postgres` | Database user |
| `POSTGRES_PASSWORD` | `postgres` | Database password |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EMBEDDING_DIMENSIONS` | `1536` | Embedding vector size |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
