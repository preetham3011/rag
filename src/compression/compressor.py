"""Adaptive context compression orchestrator for RAG system"""


def compress_context(
    query_embedding: list,
    query: str,
    intent_info: dict,
    faiss_index,
    metadata_list: list,
    top_k: int = 5,
    token_limit: int = 700
) -> dict:
    """
    Compress retrieved context using intent-aware evidence selection and budget management.
    
    Pipeline:
    1. Intent-aware retrieval (retrieve_with_intent)
    2. Sentence-level evidence selection (select_evidence)
    3. Token budget enforcement (apply_budget)
    4. Build compressed context string
    
    Args:
        query_embedding: Query embedding vector (list of floats)
        query: Raw user query string (used for cross-encoder reranking)
        intent_info: Dict with format {"intent": str, "confidence": float, "method": str}
        faiss_index: FAISS index object
        metadata_list: List of metadata dicts aligned with index
        top_k: Number of chunks to retrieve
        token_limit: Maximum allowed tokens in final context
    
    Returns:
        Dict with format:
        {
            "compressed_context": str,      # Final compressed context string
            "selected_evidence": list,      # List of selected evidence dicts
            "tokens_used": int,             # Total tokens used
            "num_sentences": int            # Number of sentences selected
        }
    """
    # Import required functions
    from src.retrieval.retriever import retrieve_with_intent
    from src.compression.evidence_selector import select_evidence
    from src.compression.budget_manager import apply_budget, estimate_tokens
    
    # Step 1: Intent-aware retrieval
    retrieved_chunks = retrieve_with_intent(
        query_embedding=query_embedding,
        intent_info=intent_info,
        faiss_index=faiss_index,
        metadata_list=metadata_list,
        top_k=top_k,
        query=query
    )

    # Debug: show top retrieved chunks before compression
    print("Top retrieved chunks before compression:")
    for c in retrieved_chunks[: min(5, len(retrieved_chunks))]:
        score = c.get("final_score", c.get("rerank_score", c.get("similarity_score", 0.0)))
        preview = (c.get("text", "")[:120] + "...") if len(c.get("text", "")) > 120 else c.get("text", "")
        print(f"  - chunk_id={c.get('chunk_id')} score={score:.4f} page={c.get('page')} section={c.get('section')}: {preview}")

    # Step 1b: protect top chunks so compression cannot drop them
    protected_k = min(3, len(retrieved_chunks))
    protected_chunks = retrieved_chunks[:protected_k]
    remaining_chunks = retrieved_chunks[protected_k:]

    print(f"Protected chunks (always included): {len(protected_chunks)}")
    for c in protected_chunks:
        print(f"  - chunk_id={c.get('chunk_id')} page={c.get('page')} section={c.get('section')}")

    protected_context = "\n\n".join(
        f"[PROTECTED] (Page {c.get('page')}, Section {c.get('section')}):\n{c.get('text', '')}"
        for c in protected_chunks
    )
    protected_tokens = estimate_tokens(protected_context)
    remaining_budget = max(token_limit - protected_tokens, 0)
    print(f"Token budget: {token_limit} | Protected tokens (est): {protected_tokens} | Remaining budget: {remaining_budget}")
    
    # Step 2: Evidence selection (sentence-level scoring)
    # Compression operates only on remaining chunks; protected chunks are included verbatim.
    evidence_list = select_evidence(
        retrieved_chunks=remaining_chunks,
        intent_info=intent_info,
        top_k=top_k * 5,  # Get plenty of candidates for budget selection
        token_budget=remaining_budget
    )
    
    # Step 3: Apply token budget
    budget_result = apply_budget(
        evidence_list=evidence_list,
        token_limit=remaining_budget
    )
    
    # Step 4: Build compressed context string
    selected_evidence = budget_result["selected_evidence"]
    
    compressed_remaining = "\n\n".join(ev["sentence"] for ev in selected_evidence)
    compressed_context = protected_context
    if compressed_remaining.strip():
        compressed_context = f"{protected_context}\n\n{compressed_remaining}".strip()

    # Debug: final context preview
    preview = compressed_context[:800] + ("..." if len(compressed_context) > 800 else "")
    print("Final compressed context preview:")
    print(preview)
    
    return {
        "compressed_context": compressed_context,
        "selected_evidence": selected_evidence,
        "tokens_used": protected_tokens + budget_result["tokens_used"],
        "num_sentences": budget_result["num_sentences"]
    }


if __name__ == "__main__":
    import numpy as np
    import faiss
    
    print("Testing Adaptive Context Compressor")
    print("=" * 70)
    
    # Create mock embeddings (384-dimensional)
    dimension = 384
    num_chunks = 10
    
    np.random.seed(42)
    embeddings = np.random.randn(num_chunks, dimension).astype(np.float32)
    
    # Create mock metadata with realistic content
    metadata_list = [
        {
            "chunk_id": 0,
            "page": 1,
            "section": "Abstract",
            "text": "This paper introduces a new algorithm for machine learning. We present a novel approach to classification tasks."
        },
        {
            "chunk_id": 1,
            "page": 2,
            "section": "Introduction",
            "text": "Machine learning is defined as the study of computer algorithms. Deep learning refers to neural networks with multiple layers."
        },
        {
            "chunk_id": 2,
            "page": 3,
            "section": "Method",
            "text": "Our pipeline consists of three steps. First, we preprocess the data. The algorithm uses a multi-stage architecture for efficiency."
        },
        {
            "chunk_id": 3,
            "page": 4,
            "section": "Method",
            "text": "The algorithm processes data using a convolutional approach. 1. Extract features. 2. Apply transformations. 3. Generate predictions."
        },
        {
            "chunk_id": 4,
            "page": 5,
            "section": "Results",
            "text": "We achieved 95% accuracy on the test set with 1000 samples. The F1 score was 0.92. Performance improved by 15% compared to baseline."
        },
        {
            "chunk_id": 5,
            "page": 6,
            "section": "Results",
            "text": "Our model achieved 88% precision and 90% recall. The accuracy on validation data was 93%."
        },
        {
            "chunk_id": 6,
            "page": 7,
            "section": "Discussion",
            "text": "We compare our approach versus traditional methods. The difference in performance is significant. Our method is better in terms of speed."
        },
        {
            "chunk_id": 7,
            "page": 8,
            "section": "API Reference",
            "text": "Use model.fit(X_train, y_train) to train the model. Call predict(data) with parameter verbose=True. The function returns a tuple (predictions, confidence)."
        },
        {
            "chunk_id": 8,
            "page": 9,
            "section": "API Reference",
            "text": "Set the argument batch_size=32 for optimal performance. The return value is a numpy array."
        },
        {
            "chunk_id": 9,
            "page": 10,
            "section": "Conclusion",
            "text": "This work demonstrates the effectiveness of our approach. Future work will explore additional architectures."
        }
    ]
    
    # Build FAISS index
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    
    print(f"Created mock index with {num_chunks} chunks\n")
    
    # Test different intent types with different budgets
    test_cases = [
        {
            "name": "RESULT Intent (Budget: 200)",
            "intent_info": {"intent": "RESULT", "confidence": 0.8, "method": "keyword"},
            "query_idx": 4,
            "token_limit": 200
        },
        {
            "name": "METHOD Intent (Budget: 150)",
            "intent_info": {"intent": "METHOD", "confidence": 0.9, "method": "keyword"},
            "query_idx": 2,
            "token_limit": 150
        },
        {
            "name": "API_USAGE Intent (Budget: 100)",
            "intent_info": {"intent": "API_USAGE", "confidence": 0.85, "method": "keyword"},
            "query_idx": 7,
            "token_limit": 100
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}")
        print("=" * 70)
        
        # Use one of the embeddings as query
        query_embedding = embeddings[test_case["query_idx"]].tolist()
        
        # Compress context
        result = compress_context(
            query_embedding=query_embedding,
            query="sample query",
            intent_info=test_case["intent_info"],
            faiss_index=faiss_index,
            metadata_list=metadata_list,
            top_k=5,
            token_limit=test_case["token_limit"]
        )
        
        print(f"Intent: {test_case['intent_info']['intent']}")
        print(f"Token Budget: {test_case['token_limit']}")
        print(f"Tokens Used: {result['tokens_used']}")
        print(f"Sentences Selected: {result['num_sentences']}")
        print(f"\nCompressed Context:")
        print("-" * 70)
        print(result["compressed_context"])
        print("-" * 70)
        
        print(f"\nEvidence Details:")
        for i, ev in enumerate(result["selected_evidence"], 1):
            print(f"{i}. [Score: {ev['score']:.2f}] Page {ev['page']} | {ev['section']}")
            print(f"   {ev['sentence'][:80]}...")
    
    print("\n" + "=" * 70)
    print("Adaptive context compressor test complete!")
