"""
query.py — CLI Interface for DocuQuery

Usage:
    python -m src.query "What is hybrid retrieval?"
    python -m src.query "How does chunking work?" --verbose
    python -m src.query "What is RAG?" --top-k 3
"""

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.config import load_config
from src.orchestrator import RAGOrchestrator

console = Console()


def run_query(question: str, config: dict, top_k: int | None = None, verbose: bool = False):
    """Execute a query and display results."""

    console.print(f"\n[bold cyan]🔍 Query:[/bold cyan] {question}\n")

    with console.status("[bold green]Thinking..."):
        orch = RAGOrchestrator(config)
        result = orch.query(question, top_k=top_k)

    # Show retrieved chunks in verbose mode
    if verbose and result.chunks_used:
        table = Table(title="Retrieved & Re-ranked Chunks", show_lines=True)
        table.add_column("ID", style="cyan", width=10)
        table.add_column("Score", style="green", width=8)
        table.add_column("Source", style="yellow", width=20)
        table.add_column("Preview", style="white", max_width=60)

        for chunk in result.chunks_used:
            preview = chunk.text[:120].replace("\n", " ")
            table.add_row(
                chunk.chunk_id[:8],
                f"{chunk.score:.4f}",
                chunk.source,
                preview,
            )
        console.print(table)
        console.print()

    # Answer
    console.print(Panel(
        Markdown(result.answer),
        title="[bold green]Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    # Metadata
    meta = Table(show_header=False, box=None, padding=(0, 2))
    meta.add_column("Key", style="dim")
    meta.add_column("Value")
    meta.add_row("Model", result.model)
    meta.add_row("Latency", f"{result.latency_ms:.0f}ms")
    meta.add_row("Tokens", f"{result.prompt_tokens} + {result.completion_tokens} = {result.total_tokens}")
    meta.add_row("Citations", str(len(result.citations)))
    meta.add_row("Coverage", f"{result.citation_coverage:.0%}")
    meta.add_row("Chunks", str(len(result.chunks_used)))
    console.print(meta)
    console.print()

    # Session metrics
    if verbose:
        metrics = orch.get_metrics()
        console.print("[dim]Session metrics:[/dim]")
        for k, v in metrics.items():
            if isinstance(v, float):
                console.print(f"  [dim]{k}: {v:.2f}[/dim]")
            else:
                console.print(f"  [dim]{k}: {v}[/dim]")
        console.print()


def main():
    parser = argparse.ArgumentParser(description="Query DocuQuery from the command line")
    parser.add_argument("query", help="Your question")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show chunks and metrics")
    parser.add_argument("--config", default=None, help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    run_query(args.query, config, top_k=args.top_k, verbose=args.verbose)


if __name__ == "__main__":
    main()
