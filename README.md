# Adaptive Context Compression for Long-Document RAG

A research-oriented minor project implementing intent-aware context compression for Retrieval-Augmented Generation on technical documents.

## Problem

Technical documents (research papers, API docs) exceed LLM context limits. Standard RAG with fixed chunking loses semantic coherence and critical evidence, leading to hallucinations.

## Solution

Adaptive Context Compression that:
- Detects query intent (METHOD, RESULT, API_USAGE, DEFINITION, COMPARISON)
- Retrieves relevant chunks
- Compresses evidence differently based on intent
- Preserves high-signal content under strict token budget

## Features

- Section-aware document chunking
- Intent-driven retrieval and compression
- Citation-based answer generation
- Refusal mode for insufficient evidence
- Baseline comparison for evaluation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API keys:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the app:
```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
src/
├── ingestion/      # PDF/HTML extraction, section detection
├── indexing/       # Chunking, embeddings, vector store
├── retrieval/      # Intent detection, retrieval, re-ranking
├── compression/    # Adaptive compression logic
└── answering/      # Answer generation, citations
```

## Evaluation

Compare baseline RAG vs adaptive compression on:
- Token reduction ratio
- Answer correctness
- Unsupported answer rate
- Citation quality

Run evaluation:
```bash
python evaluation/metrics.py
```

## Tech Stack

- Python 3.9+
- FAISS for vector storage
- Sentence-BERT for embeddings
- OpenAI API for LLM
- Streamlit for UI

## License

MIT
