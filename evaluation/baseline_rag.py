"""Baseline RAG with fixed chunking (no compression)"""

from sentence_transformers import SentenceTransformer
from src.indexing.vector_store import search_index
from src.answering.llm import generate_answer


def run_baseline_rag(
    query: str,
    faiss_index,
    metadata_list: list,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 5
) -> dict:
    """
    Run baseline RAG pipeline with fixed chunking and top-k retrieval.
    
    Args:
        query: User question
        faiss_index: Pre-built FAISS index
        metadata_list: Metadata aligned with FAISS index
        embedding_model_name: Sentence transformer model name
        top_k: Number of chunks to retrieve
    
    Returns:
        dict with format:
        {
            "query": str,
            "answer": str,
            "retrieved_chunks": list,
            "context": str,
            "token_count": int
        }
    
    Pipeline:
    1. Embed query
    2. Retrieve top-k chunks
    3. Concatenate chunks into context
    4. Generate answer using LLM
    """
    print(f"Running baseline RAG for query: '{query}'")
    print("-" * 70)
    
    # Step 1: Load embedding model and embed query
    print("Step 1: Embedding query...")
    model = SentenceTransformer(embedding_model_name)
    query_embedding = model.encode(query).tolist()
    print("  Query embedded.\n")
    
    # Step 2: Retrieve top-k chunks
    print(f"Step 2: Retrieving top-{top_k} chunks...")
    retrieved_chunks = search_index(faiss_index, metadata_list, query_embedding, top_k)
    print(f"  Retrieved {len(retrieved_chunks)} chunks.\n")
    
    # Step 3: Concatenate chunk texts into context
    print("Step 3: Building context...")
    context = _build_context(retrieved_chunks)
    token_count = _estimate_token_count(context)
    print(f"  Context built: {token_count} tokens (estimated).\n")
    
    # Step 4: Generate answer using LLM
    print("Step 4: Generating answer...")
    answer = generate_answer(context, query)
    print("  Answer generated.\n")
    
    # Return results
    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "context": context,
        "token_count": token_count
    }


def _build_context(retrieved_chunks: list) -> str:
    """
    Concatenate retrieved chunk texts into a single context string.
    
    Args:
        retrieved_chunks: List of chunk metadata dicts
    
    Returns:
        Concatenated context string
    """
    context_parts = []
    
    for i, chunk in enumerate(retrieved_chunks):
        # Format: [Chunk X] (Page Y, Section Z): text
        chunk_text = (
            f"[Chunk {i+1}] (Page {chunk['page']}, Section {chunk['section']}): "
            f"{chunk['text']}"
        )
        context_parts.append(chunk_text)
    
    return "\n\n".join(context_parts)


def _estimate_token_count(text: str) -> int:
    """
    Estimate token count (rough approximation: 1 token ≈ 4 characters).
    
    Args:
        text: Text to count tokens for
    
    Returns:
        Estimated token count
    """
    return len(text) // 4


if __name__ == "__main__":
    # Test baseline RAG
    from sentence_transformers import SentenceTransformer
    from src.indexing.vector_store import build_faiss_index
    
    print("Testing Baseline RAG")
    print("=" * 70)
    
    # Create sample document chunks
    print("Creating sample document index...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    sample_chunks = [
        {
            "chunk_id": 0,
            "page": 1,
            "section": "Abstract",
            "text": "This paper presents a novel approach to adaptive context compression for long-document RAG systems. We propose intent-aware compression that preserves high-signal content.",
            "embedding": model.encode("This paper presents a novel approach to adaptive context compression for long-document RAG systems.").tolist()
        },
        {
            "chunk_id": 1,
            "page": 2,
            "section": "Introduction",
            "text": "Traditional RAG systems use fixed chunking which breaks semantic coherence. Our method detects query intent and compresses evidence accordingly.",
            "embedding": model.encode("Traditional RAG systems use fixed chunking which breaks semantic coherence.").tolist()
        },
        {
            "chunk_id": 2,
            "page": 3,
            "section": "Method",
            "text": "Our pipeline consists of: 1) Intent detection, 2) Retrieval, 3) Adaptive compression, 4) Answer generation. We support five intent types: METHOD, RESULT, API_USAGE, DEFINITION, COMPARISON.",
            "embedding": model.encode("Our pipeline consists of intent detection, retrieval, adaptive compression, and answer generation.").tolist()
        },
        {
            "chunk_id": 3,
            "page": 4,
            "section": "Results",
            "text": "We achieved 45% token reduction while maintaining 92% answer correctness. The baseline RAG used 2000 tokens on average, while our method used 1100 tokens.",
            "embedding": model.encode("We achieved 45% token reduction while maintaining 92% answer correctness.").tolist()
        },
        {
            "chunk_id": 4,
            "page": 5,
            "section": "Conclusion",
            "text": "Adaptive context compression significantly reduces token usage while preserving answer quality. Future work includes multi-document support and real-time compression.",
            "embedding": model.encode("Adaptive context compression significantly reduces token usage while preserving answer quality.").tolist()
        }
    ]
    
    # Build FAISS index
    faiss_index, metadata_list = build_faiss_index(sample_chunks)
    
    # Test query
    test_query = "What is the token reduction achieved by the proposed method?"
    
    print("\n" + "=" * 70)
    result = run_baseline_rag(
        query=test_query,
        faiss_index=faiss_index,
        metadata_list=metadata_list,
        top_k=3
    )
    
    # Display results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Query: {result['query']}\n")
    print(f"Retrieved {len(result['retrieved_chunks'])} chunks:")
    for chunk in result['retrieved_chunks']:
        print(f"  - Rank {chunk['rank']}: Page {chunk['page']}, {chunk['section']}")
    print(f"\nContext token count: {result['token_count']}")
    print(f"\nAnswer:\n{result['answer']}")
    print("=" * 70)
