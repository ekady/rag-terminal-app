import os
import sys
import numpy as np
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.table import Table

from db import get_connection, init_db, reset_db
from embeddings import chunk_text, get_embeddings_batch

console = Console()

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rb",
    ".php",
    ".sql",
    ".r",
    ".scala",
    ".kt",
    ".csv",
    ".xml",
    ".log",
    ".dockerfile",
}


def read_file(filepath: str) -> str | None:
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        console.print(f"  [red]Error reading {filepath}: {e}[/red]")
        return None


def collect_files(path: str) -> list[str]:
    path = os.path.abspath(path)
    files = []

    if os.path.isfile(path):
        files.append(path)
    elif os.path.isdir(path):
        for root, _, filenames in os.walk(path):
            for filename in sorted(filenames):
                ext = Path(filename).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    files.append(os.path.join(root, filename))
    else:
        console.print(f"[red]Path not found: {path}[/red]")

    return files


def index_file(filepath: str, conn) -> int:
    content = read_file(filepath)
    if not content or not content.strip():
        return 0

    chunks = chunk_text(content)
    if not chunks:
        return 0

    # Generate embeddings in batch
    embeddings = get_embeddings_batch(chunks)

    cur = conn.cursor()
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        embedding_array = np.array(embedding)
        cur.execute(
            """
            INSERT INTO documents (filename, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (filename, chunk_index)
            DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding;
            """,
            (filepath, i, chunk, embedding_array),
        )

    conn.commit()
    cur.close()
    return len(chunks)


def run_indexer(path: str, do_reset: bool = False):
    console.print(Panel("📂 [bold]RAG File Indexer[/bold]", style="blue"))

    if do_reset:
        console.print("[yellow]Resetting database...[/yellow]")
        reset_db()
        console.print("[green]Database reset complete.[/green]")
    else:
        init_db()

    files = collect_files(path)
    if not files:
        console.print("[red]No supported files found.[/red]")
        return

    console.print(f"\nFound [cyan]{len(files)}[/cyan] file(s) to index.\n")

    # Index each file
    total_chunks = 0
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing files...", total=len(files))

        conn = get_connection()
        for filepath in files:
            filename = os.path.basename(filepath)
            progress.update(task, description=f"Indexing {filename}...")
            chunks = index_file(filepath, conn)
            total_chunks += chunks
            results.append((filename, chunks))
            progress.advance(task)
        conn.close()

    table = Table(title="Indexing Summary")
    table.add_column("File", style="cyan")
    table.add_column("Chunks", style="green", justify="right")

    for filename, chunks in results:
        table.add_row(filename, str(chunks))

    table.add_row("[bold]Total[/bold]", f"[bold]{total_chunks}[/bold]")
    console.print()
    console.print(table)
    console.print(
        f"\n[green]✓ Indexing complete![/green] {total_chunks} chunks stored.\n"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]Usage: python indexer.py <path> [--reset][/red]")
        sys.exit(1)

    target_path = sys.argv[1]
    should_reset = "--reset" in sys.argv

    run_indexer(target_path, should_reset)
