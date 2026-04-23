"""
main.py — Interactive CLI for the RAG Q&A Bot.

Usage:
    python src/main.py

Commands inside the loop:
    quit / exit / q   — exit the bot
    /stats            — show vector store stats
    /help             — show help
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
    RICH = True
except ImportError:
    RICH = False

from query_engine import QueryEngine


# ─── Display Helpers ─────────────────────────────────────────────────────────

def print_header():
    if RICH:
        console = Console()
        console.print(Panel(
            "[bold cyan]RAG Document Q&A Bot[/bold cyan]\n"
            "[dim]Retrieval-Augmented Generation | ChromaDB + SentenceTransformers[/dim]",
            border_style="cyan",
        ))
        console.print("[dim]Type your question below. Commands: /stats  /help  quit[/dim]\n")
    else:
        print("\n" + "=" * 60)
        print("  RAG Document Q&A Bot")
        print("  Commands: /stats  /help  quit")
        print("=" * 60 + "\n")


def _best_source(retrieved):
    """Return the single (chunk, score) with the highest relevance score."""
    if not retrieved:
        return None
    return max(retrieved, key=lambda x: x[1])


def display_answer(question: str, answer: str, retrieved, elapsed: float):
    # Check if the bot couldn't answer
    cannot_answer = "I could not find an answer" in answer

    if RICH:
        console = Console()

        # ── Question ──────────────────────────────────────────────────────────
        console.print(Rule("[bold]Question[/bold]", style="green"))
        console.print(f"[bold green]{question}[/bold green]\n")

        # ── Answer ────────────────────────────────────────────────────────────
        console.print(Rule("[bold]Answer[/bold]", style="blue"))
        console.print(answer)

        # ── Source table (only if answered) ───────────────────────────────────
        if not cannot_answer:
            console.print(Rule("[bold]Source[/bold]", style="yellow"))
            table = Table(show_header=True, header_style="bold yellow")
            table.add_column("Source",    min_width=25)
            table.add_column("Page",      width=6)
            table.add_column("Relevance", width=10)
            table.add_column("Preview",   min_width=40)

            best = _best_source(retrieved)
            if best:
                chunk, score = best
                preview = chunk.text[:80].replace("\n", " ") + ("…" if len(chunk.text) > 80 else "")
                table.add_row(
                    chunk.source,
                    str(chunk.page_hint) if chunk.page_hint else "—",
                    f"{score:.3f}",
                    preview,
                )
            console.print(table)

            # ── Retrieved Chunks (full text) ──────────────────────────────────
            console.print(Rule("[bold]Retrieved Chunks[/bold]", style="magenta"))
            for i, (chunk, score) in enumerate(retrieved, 1):
                page_str = f" | Page {chunk.page_hint}" if chunk.page_hint else ""
                console.print(
                    f"\n[bold magenta][Chunk {i}][/bold magenta] "
                    f"[bold]{chunk.source}{page_str}[/bold] "
                    f"[dim](relevance: {score:.3f})[/dim]"
                )
                console.print(
                    Panel(
                        chunk.text,
                        border_style="magenta",
                        padding=(0, 1),
                    )
                )

        console.print(f"\n[dim]⏱  {elapsed:.2f}s[/dim]\n")

    else:
        print(f"\nQuestion: {question}")
        print("-" * 40)
        print(f"Answer:\n{answer}")

        if not cannot_answer:
            best = _best_source(retrieved)
            if best:
                chunk, score = best
                page = f"p.{chunk.page_hint}" if chunk.page_hint else "—"
                print(f"\nSource: {chunk.source} ({page}) | score={score:.3f}")

            print("\n--- Retrieved Chunks ---")
            for i, (chunk, score) in enumerate(retrieved, 1):
                page_str = f" | Page {chunk.page_hint}" if chunk.page_hint else ""
                print(f"\n[Chunk {i}] {chunk.source}{page_str} (relevance: {score:.3f})")
                print("-" * 40)
                print(chunk.text)
                print("-" * 40)

        print(f"\n⏱  {elapsed:.2f}s\n")


def display_no_results():
    msg = ("No relevant chunks found above the similarity threshold.\n"
           "Try rephrasing your question or lower SIMILARITY_THRESHOLD in config.py.")
    if RICH:
        Console().print(Panel(msg, border_style="red", title="No Results"))
    else:
        print(f"\n⚠️  {msg}\n")


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    print_header()

    try:
        engine = QueryEngine()
        _      = engine.vector_store
        _      = engine.embedding_model
        _      = engine.generator
    except RuntimeError as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Initialisation failed: {e}\n")
        sys.exit(1)

    while True:
        try:
            if RICH:
                console = Console()
                console.print("[bold cyan]❓ Question:[/bold cyan] ", end="")
                question = input().strip()
            else:
                question = input("❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break
        if question == "/stats":
            stats = engine.vector_store.stats()
            print(f"\n📊 Vector Store Stats:\n{stats}\n")
            continue
        if question == "/help":
            print("\nCommands:")
            print("  /stats — show vector store statistics")
            print("  quit   — exit\n")
            continue

        import time
        t0 = time.time()

        try:
            answer, retrieved = engine.query(question)
            elapsed           = time.time() - t0

            if not retrieved:
                display_no_results()
            else:
                display_answer(question, answer, retrieved, elapsed)

        except Exception as e:
            print(f"\n❌ Query failed: {e}\n")


if __name__ == "__main__":
    main()