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
    remaining_budget = max(token_limit - protected_tokens, int(0.25 * token_limit))
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

