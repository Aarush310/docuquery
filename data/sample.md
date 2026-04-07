# Retrieval-Augmented Generation (RAG): A Technical Overview

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances large language model (LLM) outputs by grounding them in external knowledge retrieved from a document store. Instead of relying solely on the model's parametric memory, RAG systems fetch relevant passages at inference time and include them in the prompt context.

The key benefit of RAG is that it significantly reduces hallucination by anchoring the model's responses to verifiable source material. It also allows the system to access up-to-date information without retraining the model.

## How RAG Works

A typical RAG pipeline involves three stages:

1. **Indexing**: Documents are split into chunks, each chunk is embedded into a vector representation using an embedding model (e.g., sentence-transformers), and stored in a vector database.

2. **Retrieval**: When a user asks a question, the query is embedded using the same model, and the top-k most similar chunks are retrieved via approximate nearest neighbor (ANN) search.

3. **Generation**: The retrieved chunks are inserted into the LLM's prompt as context, and the model generates an answer grounded in those chunks.

## Chunking Strategies

Chunking is the process of splitting documents into smaller pieces for embedding. Common strategies include:

- **Fixed-size chunking**: Split text every N tokens with M tokens of overlap. Simple and predictable, but may split sentences or paragraphs mid-thought.
- **Semantic chunking**: Split at natural boundaries (paragraphs, sections, sentences). Preserves meaning but produces variable-length chunks.
- **Recursive chunking**: Try to split at paragraphs first, then sentences, then tokens as a fallback. Used by LangChain's RecursiveCharacterTextSplitter.

A typical chunk size is 500-800 tokens with 100 tokens of overlap to ensure context continuity across chunk boundaries.

## Vector Databases

Vector databases store embeddings and support fast similarity search. Popular options include:

- **ChromaDB**: Open-source, lightweight, Python-native. Good for prototyping and small-to-medium datasets.
- **Pinecone**: Managed cloud service with high scalability. Supports metadata filtering and hybrid search.
- **Weaviate**: Open-source with built-in vectorization modules and GraphQL API.
- **Qdrant**: High-performance open-source vector search engine with filtering.
- **FAISS**: Facebook's library for efficient similarity search. Not a full database but excellent for pure vector operations.

## Hybrid Retrieval

Pure vector search sometimes misses exact keyword matches. Hybrid retrieval combines:

- **Vector search** (semantic similarity): Good at understanding meaning and paraphrases.
- **BM25** (keyword matching): Good at finding exact terms, names, and acronyms.

The scores from both methods are normalized and combined with configurable weights (e.g., 0.7 vector + 0.3 BM25). This approach consistently outperforms either method alone.

## Re-ranking

After initial retrieval, a cross-encoder re-ranker can rescore the candidate chunks. Unlike bi-encoders (used for embedding), cross-encoders process the query and document together, producing more accurate relevance scores at the cost of higher latency.

Common re-rankers include Cohere Rerank, and cross-encoder models from sentence-transformers (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2).

## Evaluation Metrics

Evaluating RAG systems requires measuring both retrieval quality and generation quality:

- **Retrieval Precision@k**: What fraction of retrieved chunks are actually relevant?
- **Retrieval Recall@k**: What fraction of all relevant chunks were retrieved?
- **Citation Coverage**: Does every claim in the answer have a supporting citation?
- **Faithfulness**: Is every cited claim actually supported by the cited chunk?
- **Answer Relevance**: Does the answer address the user's question?

## Common Pitfalls

1. **Chunk size too large**: The model may ignore important details buried in long chunks.
2. **Chunk size too small**: Chunks lack sufficient context to be meaningful.
3. **No overlap**: Important information at chunk boundaries gets lost.
4. **Ignoring metadata**: Timestamps, source files, and section headers provide useful filtering signals.
5. **No evaluation**: Without a golden dataset, you can't measure or improve quality systematically.

## Production Considerations

For production RAG systems, consider:

- **Prompt versioning**: Track changes to system prompts alongside code changes.
- **Observability**: Log which chunks were retrieved, what prompt was sent, and what the model returned.
- **CI/CD gating**: Run evaluation against a golden dataset on every PR; fail the build if quality regresses.
- **Cost tracking**: Monitor token usage and cost per query.
- **Latency budgeting**: Set P50 and P95 latency targets and monitor them.
