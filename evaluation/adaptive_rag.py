"""Adaptive RAG with intent-aware compression"""

from src.retrieval.intent_detector import detect_intent
from src.compression.compressor import compress_context
from src.answering.llm import generate_answer
from src.indexing.embedder import EmbeddingModel


def run_adaptive_rag(
    query: str,
    faiss_index,
    metadata_list: list,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 5,
    token_limit: int = 500
) -> dict:
    """
    Run adaptive RAG pipeline with intent-aware compression.
    
    Args:
        query: User question
        faiss_index: Pre-built FAISS index
        metadata_list: Metadata aligned with FAISS index
        embedding_model_name: Sentence transformer model name
        top_k: Number of chunks to retrieve
        token_limit: Maximum tokens for compressed context
    
    Returns:
        dict with format:
        {
            "query": str,
            "intent": dict,
            "answer": str,
            "compressed_context": str,
            "tokens_used": int,
            "num_sentences": int
        }
    
    Pipeline:
    1. Detect query intent
    2. Embed query
    3. Compress context (retrieval + evidence selection + budget)
    4. Generate answer using LLM
    """
    print(f"Running adaptive RAG for query: '{query}'")
    print("-" * 70)
    
    print("Step 1: Detecting intent...")
    intent_info = detect_intent(query)
    print(f"  Intent: {intent_info['intent']} (confidence: {intent_info['confidence']:.2f})\n")
    
    print("Step 2: Embedding query...")
    embedding_model = EmbeddingModel(model_name=embedding_model_name)
    query_embedding = embedding_model.encode_text(query)
    print("  Query embedded.\n")
    
    print(f"Step 3: Compressing context (token limit: {token_limit})...")
    compression_result = compress_context(
        query_embedding=query_embedding,
        query=query,
        intent_info=intent_info,
        faiss_index=faiss_index,
        metadata_list=metadata_list,
        top_k=top_k,
        token_limit=token_limit
    )
    print(f"  Compressed to {compression_result['num_sentences']} sentences")
    print(f"  Tokens used: {compression_result['tokens_used']}\n")
    
    print("Step 4: Generating answer...")
    answer = generate_answer(compression_result['compressed_context'], query)
    print("  Answer generated.\n")
    
    return {
        "query": query,
        "intent": intent_info,
        "answer": answer,
        "compressed_context": compression_result['compressed_context'],
        "tokens_used": compression_result['tokens_used'],
        "num_sentences": compression_result['num_sentences'],
        "selected_evidence": compression_result.get('selected_evidence', []),
    }


if __name__ == "__main__":
    from src.indexing.vector_store import build_faiss_index
    
    print("Testing Adaptive RAG")
    print("=" * 70)
    
    print("Creating sample document index...")
    embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    
    sample_chunks = [
        {
            "chunk_id": 0,
            "page": 1,
            "section": "Abstract",
            "text": "This paper presents a novel approach to adaptive context compression for long-document RAG systems. We propose intent-aware compression that preserves high-signal content.",
            "embedding": embedding_model.encode_text("This paper presents a novel approach to adaptive context compression for long-document RAG systems.")
        },
        {
            "chunk_id": 1,
            "page": 2,
            "section": "Introduction",
            "text": "Traditional RAG systems use fixed chunking which breaks semantic coherence. Our method detects query intent and compresses evidence accordingly.",
            "embedding": embedding_model.encode_text("Traditional RAG systems use fixed chunking which breaks semantic coherence.")
        },
        {
            "chunk_id": 2,
            "page": 3,
            "section": "Method",
            "text": "Our pipeline consists of: 1) Intent detection, 2) Retrieval, 3) Adaptive compression, 4) Answer generation. We support five intent types: METHOD, RESULT, API_USAGE, DEFINITION, COMPARISON.",
            "embedding": embedding_model.encode_text("Our pipeline consists of intent detection, retrieval, adaptive compression, and answer generation.")
        },
        {
            "chunk_id": 3,
            "page": 4,
            "section": "Results",
            "text": "We achieved 45% token reduction while maintaining 92% answer correctness. The baseline RAG used 2000 tokens on average, while our method used 1100 tokens.",
            "embedding": embedding_model.encode_text("We achieved 45% token reduction while maintaining 92% answer correctness.")
        },
        {
            "chunk_id": 4,
            "page": 5,
            "section": "Conclusion",
            "text": "Adaptive context compression significantly reduces token usage while preserving answer quality. Future work includes multi-document support and real-time compression.",
            "embedding": embedding_model.encode_text("Adaptive context compression significantly reduces token usage while preserving answer quality.")
        }
    ]
    
    faiss_index, metadata_list = build_faiss_index(sample_chunks)
    
    test_query = "What is the token reduction achieved by the proposed method?"
    
    print("\n" + "=" * 70)
    result = run_adaptive_rag(
        query=test_query,
        faiss_index=faiss_index,
        metadata_list=metadata_list,
        top_k=3,
        token_limit=300
    )
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Query: {result['query']}\n")
    print(f"Intent: {result['intent']['intent']} (confidence: {result['intent']['confidence']:.2f})")
    print(f"Sentences selected: {result['num_sentences']}")
    print(f"Tokens used: {result['tokens_used']}")
    print(f"\nCompressed context:\n{result['compressed_context']}")
    print(f"\nAnswer:\n{result['answer']}")
    print("=" * 70)
