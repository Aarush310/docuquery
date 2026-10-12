"""
ingest.py — Document Ingestion Pipeline

Loads documents (PDF, Markdown, plain text), chunks them with token-aware
splitting and overlap, embeds each chunk with sentence-transformers,
and stores them in ChromaDB with metadata.

Usage:
    python -m src.ingest --source data/
    python -m src.ingest --source data/sample.md
"""

import argparse
import hashlib
import sys
from functools import lru_cache
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.config import load_config

SUPPORTED_EXTENSIONS = (".pdf", ".md", ".txt", ".text")


class IngestError(Exception):
    """Raised when ingestion cannot proceed (bad path, no supported files)."""

console = Console()


# ─── Document Loaders ────────────────────────────────────────────

def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(path: str) -> str:
    import fitz
    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def load_document(path: str) -> str:
    """Route to the correct loader based on file extension."""
    ext = Path(path).suffix.lower()
    loaders = {
        ".pdf": load_pdf,
        ".md": load_text_file,
        ".txt": load_text_file,
        ".text": load_text_file,
    }
    loader = loaders.get(ext)
    if loader is None:
        console.print(f"[yellow]⚠ Skipping unsupported file: {path}[/yellow]")
        return ""
    return loader(path)


# ─── Chunking ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_encoder():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(get_encoder().encode(text))


def chunk_text(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    min_chunk_size: int = 50,
) -> list[dict]:
    """
    Split text into chunks of approximately `chunk_size` tokens with overlap.

    Strategy: split on paragraph boundaries first, then accumulate paragraphs
    into chunks up to the token limit. Falls back to token-level splitting
    for very long paragraphs.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if not paragraphs:
        return []

    chunks = []
    current_paragraphs = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # If a single paragraph exceeds chunk_size, split it by tokens
        if para_tokens > chunk_size:
            # Flush current buffer first
            if current_paragraphs:
                chunk_text_str = "\n\n".join(current_paragraphs)
                chunks.append({"text": chunk_text_str, "token_count": current_tokens})
                # Keep overlap
                overlap_paras, overlap_tokens = _get_overlap(current_paragraphs, chunk_overlap)
                current_paragraphs = overlap_paras
                current_tokens = overlap_tokens

            # Split the long paragraph by tokens
            tokens = get_encoder().encode(para)
            start = 0
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                if len(chunk_tokens) >= min_chunk_size:
                    chunks.append({
                        "text": get_encoder().decode(chunk_tokens),
                        "token_count": len(chunk_tokens),
                    })
                start += chunk_size - chunk_overlap
            continue

        # Would adding this paragraph exceed the limit?
        if current_tokens + para_tokens > chunk_size and current_paragraphs:
            chunk_text_str = "\n\n".join(current_paragraphs)
            chunks.append({"text": chunk_text_str, "token_count": current_tokens})

            # Keep overlap from end of current buffer
            overlap_paras, overlap_tokens = _get_overlap(current_paragraphs, chunk_overlap)
            current_paragraphs = overlap_paras
            current_tokens = overlap_tokens

        current_paragraphs.append(para)
        current_tokens += para_tokens

    # Flush remaining
    if current_paragraphs:
        chunk_text_str = "\n\n".join(current_paragraphs)
        tc = count_tokens(chunk_text_str)
        if tc >= min_chunk_size:
            chunks.append({"text": chunk_text_str, "token_count": tc})

    return chunks


def _get_overlap(paragraphs: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Get trailing paragraphs that fit within the overlap budget."""
    result = []
    total = 0
    for para in reversed(paragraphs):
        pt = count_tokens(para)
        if total + pt > overlap_tokens:
            break
        result.insert(0, para)
        total += pt
    return result, total


# ─── Embedding + Storage ─────────────────────────────────────────

def generate_chunk_id(source: str, index: int) -> str:
    raw = f"{source}::{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def collect_files(source_path: str | Path) -> list[Path]:
    source = Path(source_path)
    if source.is_file():
        files = [source] if source.suffix.lower() in SUPPORTED_EXTENSIONS else []
    elif source.is_dir():
        files = sorted(p for p in source.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    else:
        raise IngestError(f"Path not found: {source_path}")
    if not files:
        raise IngestError(f"No supported documents ({', '.join(SUPPORTED_EXTENSIONS)}) in {source_path}")
    return files


def ingest_documents(source_path: str, config: dict) -> dict:
    """Main ingestion pipeline. Rebuilds the collection; returns ingestion stats."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    chunk_cfg = config["chunking"]
    embed_cfg = config["embedding"]
    store_cfg = config["vector_store"]

    files = collect_files(source_path)

    console.print(f"\n[bold]📂 Found {len(files)} document(s) to ingest[/bold]\n")

    # Load embedding model
    with console.status("[bold green]Loading embedding model..."):
        embedder = SentenceTransformer(embed_cfg["model"])

    # Initialize ChromaDB (fresh collection)
    client = chromadb.PersistentClient(path=store_cfg["persist_directory"])
    try:
        client.delete_collection(store_cfg["collection_name"])
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=store_cfg["collection_name"],
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    total_tokens = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting...", total=len(files))

        for file_path in files:
            text = load_document(str(file_path))
            if not text.strip():
                progress.update(task, advance=1)
                continue

            chunks = chunk_text(
                text,
                chunk_size=chunk_cfg["chunk_size"],
                chunk_overlap=chunk_cfg["chunk_overlap"],
                min_chunk_size=chunk_cfg["min_chunk_size"],
            )

            if not chunks:
                progress.update(task, advance=1)
                continue

            texts = [c["text"] for c in chunks]
            embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

            ids = []
            metadatas = []
            for i, chunk in enumerate(chunks):
                chunk_id = generate_chunk_id(str(file_path), i)
                ids.append(chunk_id)
                metadatas.append({
                    "source": str(file_path.name),
                    "chunk_index": i,
                    "token_count": chunk["token_count"],
                    "total_chunks_in_doc": len(chunks),
                })

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            total_chunks += len(chunks)
            total_tokens += sum(c["token_count"] for c in chunks)
            progress.update(task, advance=1, description=f"[green]✓ {file_path.name}[/green]")

    console.print(f"\n[bold green]✅ Ingestion complete[/bold green]")
    console.print(f"   Documents: {len(files)}")
    console.print(f"   Chunks:    {total_chunks}")
    console.print(f"   Tokens:    {total_tokens:,}")
    console.print(f"   Stored in: {store_cfg['persist_directory']}\n")
    return {"documents": len(files), "chunks": total_chunks, "tokens": total_tokens}


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into DocuQuery")
    parser.add_argument("--source", required=True, help="File or directory path")
    parser.add_argument("--config", default=None, help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        ingest_documents(args.source, config)
    except IngestError as e:
        console.print(f"[red]✗ {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
