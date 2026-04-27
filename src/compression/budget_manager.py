"""Token budget management module for RAG compression"""


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using simple approximation.
    
    Args:
        text: Input text string
    
    Returns:
        int: Estimated token count (characters / 4)
    
    Note:
        This is a rough approximation. Real tokenizers may vary.
        Rule of thumb: ~4 characters per token for English text.
    """
    return len(text) // 4


def apply_budget(
    evidence_list: list,
    token_limit: int
) -> dict:
    """
    Apply token budget to evidence sentences.
    
    Iteratively adds evidence sentences in ranked order until
    token limit would be exceeded.
    
    Args:
        evidence_list: List of dicts with format:
            [{"sentence": str, "page": int, "section": str, "score": float}]
            Should be pre-sorted by score (descending)
        token_limit: Maximum allowed tokens
    
    Returns:
        Dict with format:
        {
            "selected_evidence": list,  # Subset of evidence_list that fits budget
            "tokens_used": int,          # Total tokens used
            "num_sentences": int         # Number of sentences selected
        }
    """
    selected_evidence = []
    tokens_used = 0
    
    # Iterate through evidence in ranked order
    for evidence in evidence_list:
        sentence = evidence.get("sentence", "")
        
        # Estimate tokens for this sentence
        sentence_tokens = estimate_tokens(sentence)
        
        # Check if adding this sentence would exceed budget
        if tokens_used + sentence_tokens <= token_limit:
            selected_evidence.append(evidence)
            tokens_used += sentence_tokens
        else:
            # Stop adding sentences once budget would be exceeded
            break
    
    return {
        "selected_evidence": selected_evidence,
        "tokens_used": tokens_used,
        "num_sentences": len(selected_evidence)
    }



