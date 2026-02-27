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


if __name__ == "__main__":
    print("Testing Budget Manager")
    print("=" * 70)
    
    # Mock evidence list (pre-sorted by score)
    evidence_list = [
        {
            "sentence": "We achieved 95% accuracy on the test set with 1000 samples.",
            "page": 5,
            "section": "Results",
            "score": 0.55
        },
        {
            "sentence": "The F1 score was 0.92, demonstrating strong performance.",
            "page": 5,
            "section": "Results",
            "score": 0.50
        },
        {
            "sentence": "Our pipeline consists of three main steps for processing.",
            "page": 3,
            "section": "Method",
            "score": 0.35
        },
        {
            "sentence": "First, we preprocess the data using standard normalization techniques.",
            "page": 3,
            "section": "Method",
            "score": 0.30
        },
        {
            "sentence": "The algorithm uses a multi-stage architecture for efficiency.",
            "page": 3,
            "section": "Method",
            "score": 0.25
        },
        {
            "sentence": "Machine learning is defined as the study of computer algorithms.",
            "page": 1,
            "section": "Introduction",
            "score": 0.20
        }
    ]
    
    print(f"Total evidence sentences: {len(evidence_list)}")
    print(f"\nEvidence list:")
    for i, ev in enumerate(evidence_list, 1):
        tokens = estimate_tokens(ev["sentence"])
        print(f"{i}. [Score: {ev['score']:.2f}, ~{tokens} tokens] {ev['sentence'][:60]}...")
    
    # Test different token budgets
    test_budgets = [50, 100, 200, 500]
    
    for budget in test_budgets:
        print(f"\n{'=' * 70}")
        print(f"Token Budget: {budget}")
        print("-" * 70)
        
        result = apply_budget(evidence_list, token_limit=budget)
        
        print(f"Selected: {result['num_sentences']} sentences")
        print(f"Tokens used: {result['tokens_used']} / {budget}")
        print(f"\nSelected evidence:")
        
        for i, ev in enumerate(result["selected_evidence"], 1):
            tokens = estimate_tokens(ev["sentence"])
            print(f"{i}. [Score: {ev['score']:.2f}, ~{tokens} tokens]")
            print(f"   {ev['sentence']}")
    
    print("\n" + "=" * 70)
    print("Budget manager test complete!")
