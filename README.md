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
| `chunking.py` | Text chunking strategies (fixed, sentence, recursive, semantic) |
| `embeddings.py` | Embedding generation |
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
# Index a single file (default: recursive strategy)
python indexer.py path/to/file.txt

# Index with a specific chunking strategy
python indexer.py path/to/documents/ --chunking fixed
python indexer.py path/to/documents/ --chunking sentence
python indexer.py path/to/documents/ --chunking recursive
python indexer.py path/to/documents/ --chunking semantic

# Reset database and re-index
python indexer.py path/to/documents/ --reset

# Combine options
python indexer.py path/to/documents/ --chunking fixed --reset
```

**Chunking Strategies:**

| Strategy | Description |
|----------|-------------|
| `fixed` | Fixed-size with overlap (character-based) |
| `sentence` | Sentence-based using NLTK |
| `recursive` | Recursive character splitting (default) |
| `semantic` | Paragraph/sentence-aware splitting |

### Stage 2: Query via Terminal

```bash
python app.py
```

### Stage 3: Query via Gradio

```bash
python -m gradio-ui.app
```
Application will be available at http://127.0.0.1:7860 after running the command.

## Job Desc Analyzer

```bash
python -m job-desc-analyzer.app
```
Application will be available at http://127.0.0.1:7860 after running the command.


## Multi-Model LLM Conversation

```bash
python -m multi-model.app
```

## Language Buddy

This app uses **few-shot prompting** to guide the LLM to act as a language tutor for Bahasa Indonesia. The tutor will correct your grammar mistakes and provide alternative ways of saying things with explanations and follow-up questions. Instead of just describing the task, it provides several input/output examples directly in the prompt. This helps the model understand the desired format and style more effectively.

### Running the App

```bash
python -m language_buddy.app
```

Application will be available at http://127.0.0.1:7860 after running the command.

## Airline Assistant

```bash
python -m airline_assistant.app
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
| `CHUNK_SIZE` | `500` | Chunk size (characters for fixed/recursive/semantic, tokens for sentence) |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks (fixed/recursive/semantic) |
