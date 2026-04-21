import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from db import init_db
from rag import ask

# Custom theme
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red bold",
        "success": "green",
        "user": "bold magenta",
        "assistant": "bold blue",
    }
)

console = Console(theme=custom_theme)

HELP_TEXT = """
[bold]Commands:[/bold]
  [cyan]/help[/cyan]     Show this help message
  [cyan]/sources[/cyan]  Toggle showing source documents (default: off)
  [cyan]/clear[/cyan]    Clear the screen
  [cyan]/quit[/cyan]     Exit the application

Type any question to query your indexed documents.
"""


def print_banner():
    console.print()
    console.print(
        Panel(
            "[bold blue]🔍 RAG Terminal Assistant[/bold blue]\n"
            "[dim]Ask questions about your indexed documents[/dim]\n"
            "[dim]Type /help for commands[/dim]",
            style="blue",
            padding=(1, 2),
        )
    )
    console.print()


def print_sources(sources: list[dict]):
    table = Table(title="📄 Sources", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("File", style="cyan")
    table.add_column("Chunk", style="green", justify="center")
    table.add_column("Similarity", style="yellow", justify="right")
    table.add_column("Preview", style="dim", max_width=50)

    for i, s in enumerate(sources, 1):
        preview = s["content"][:80].replace("\n", " ") + "..."
        table.add_row(
            str(i),
            s["filename"].split("/")[-1],
            str(s["chunk_index"]),
            f"{s['similarity']:.4f}",
            preview,
        )

    console.print(table)


def main():
    print_banner()

    try:
        init_db()
    except Exception as e:
        console.print(f"[error]Failed to connect to database: {e}[/error]")
        console.print(
            "[warning]Make sure PostgreSQL is running and .env is configured.[/warning]"
        )
        sys.exit(1)

    show_sources = False

    while True:
        try:
            console.print()
            query = console.input("[user]You >[/user] ").strip()

            if not query:
                continue

            if query.startswith("/"):
                cmd = query.lower()

                if cmd == "/quit" or cmd == "/exit":
                    console.print("[dim]Goodbye! 👋[/dim]")
                    break
                elif cmd == "/help":
                    console.print(HELP_TEXT)
                    continue
                elif cmd == "/sources":
                    show_sources = not show_sources
                    state = "ON" if show_sources else "OFF"
                    console.print(f"[info]Source display: {state}[/info]")
                    continue
                elif cmd == "/clear":
                    console.clear()
                    print_banner()
                    continue
                else:
                    console.print(f"[warning]Unknown command: {query}[/warning]")
                    continue

            # RAG query
            with console.status("[bold blue]Thinking...", spinner="dots"):
                answer, sources = ask(query)

            console.print()
            console.print(
                Panel(
                    Markdown(answer),
                    title="[assistant]Assistant[/assistant]",
                    border_style="blue",
                )
            )

            if show_sources and sources:
                console.print()
                print_sources(sources)

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye! 👋[/dim]")
            break
        except Exception as e:
            console.print(f"\n[error]Error: {e}[/error]")


if __name__ == "__main__":
    main()
