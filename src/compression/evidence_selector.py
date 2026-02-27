"""Evidence selection module for RAG compression"""

import re


def split_into_sentences(text: str) -> list:
    """
    Split text into sentences using simple heuristics.
    
    Args:
        text: Input text string
    
    Returns:
        List of sentence strings
    """
    # Simple sentence splitting on period, exclamation, question mark
    # followed by space and capital letter or end of string
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Clean up and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def score_sentence(sentence: str, intent: str) -> float:
    """
    Score a sentence based on intent-specific heuristics.
    
    Args:
        sentence: Sentence text
        intent: Intent type (RESULT, METHOD, API_USAGE, DEFINITION, COMPARISON)
    
    Returns:
        float: Score (0.0 to ~1.0)
    """
    score = 0.0
    sentence_lower = sentence.lower()
    
    if intent == "RESULT":
        # Bonus for numerical content
        if re.search(r'\d', sentence):
            score += 0.2
        
        # Bonus for percentages
        if '%' in sentence:
            score += 0.15
        
        # Bonus for metric keywords
        metric_keywords = ["accuracy", "f1", "precision", "recall"]
        for keyword in metric_keywords:
            if keyword in sentence_lower:
                score += 0.2
                break
    
    elif intent == "METHOD":
        # Bonus for methodology keywords
        method_keywords = ["step", "algorithm", "pipeline", "architecture"]
        for keyword in method_keywords:
            if keyword in sentence_lower:
                score += 0.2
                break
        
        # Bonus for numbered list patterns
        if re.match(r'^\d+\.', sentence.strip()) or sentence_lower.startswith("first,"):
            score += 0.15
    
    elif intent == "API_USAGE":
        # Bonus for code-like symbols
        code_symbols = ["(", ")", "="]
        symbol_count = sum(1 for symbol in code_symbols if symbol in sentence)
        
        if symbol_count >= 2:
            score += 0.25
        elif symbol_count >= 1:
            score += 0.15
        
        # Bonus for API-related keywords
        api_keywords = ["parameter", "argument", "return"]
        for keyword in api_keywords:
            if keyword in sentence_lower:
                score += 0.15
                break
    
    elif intent == "DEFINITION":
        # Bonus for definition patterns
        definition_patterns = ["is defined as", "refers to", "means"]
        for pattern in definition_patterns:
            if pattern in sentence_lower:
                score += 0.3
                break
    
    elif intent == "COMPARISON":
        # Bonus for comparison keywords
        comparison_keywords = ["compare", "difference", "versus", "better", "worse"]
        for keyword in comparison_keywords:
            if keyword in sentence_lower:
                score += 0.2
                break
    
    return score


def select_evidence(
    retrieved_chunks: list,
    intent_info: dict,
    top_k: int = 10
) -> list:
    """
    Select high-signal evidence sentences from retrieved chunks.
    
    Args:
        retrieved_chunks: List of dicts with format:
            [{"chunk_id": int, "page": int, "section": str, "text": str, ...}]
        intent_info: Dict with format:
            {"intent": str, "confidence": float, "method": str}
        top_k: Number of top evidence sentences to return
    
    Returns:
        List of dicts with format:
        [{
            "sentence": str,
            "page": int,
            "section": str,
            "score": float
        }]
        Sorted by score (descending)
    """
    intent = intent_info.get("intent", "UNKNOWN")
    confidence = intent_info.get("confidence", 0.0)
    
    # Only apply intent-based scoring if confidence is reasonable
    apply_intent_scoring = confidence > 0.3
    
    evidence_list = []
    
    # Process each chunk
    for chunk in retrieved_chunks:
        text = chunk.get("text", "")
        page = chunk.get("page", 0)
        section = chunk.get("section", "Unknown")
        
        # Split into sentences
        sentences = split_into_sentences(text)
        
        # Score each sentence
        for sentence in sentences:
            if apply_intent_scoring:
                score = score_sentence(sentence, intent)
            else:
                # Default scoring: prefer longer sentences
                score = len(sentence) / 1000.0
            
            # Only include sentences with non-zero score
            if score > 0:
                evidence_list.append({
                    "sentence": sentence,
                    "page": page,
                    "section": section,
                    "score": score
                })
    
    # Sort by score (descending)
    evidence_list.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top_k results
    return evidence_list[:top_k]


if __name__ == "__main__":
    print("Testing Evidence Selector")
    print("=" * 70)
    
    # Mock retrieved chunks with different content types
    retrieved_chunks = [
        {
            "chunk_id": 0,
            "page": 5,
            "section": "Results",
            "text": "We achieved 95% accuracy on the test set. The F1 score was 0.92. Our model outperformed the baseline by 10%."
        },
        {
            "chunk_id": 1,
            "page": 3,
            "section": "Method",
            "text": "Our pipeline consists of three steps. First, we preprocess the data. The algorithm uses a multi-stage architecture."
        },
        {
            "chunk_id": 2,
            "page": 8,
            "section": "API Reference",
            "text": "Call the function with parameter x = 5. The method returns a tuple (result, status). Use argument verbose=True for debugging."
        },
        {
            "chunk_id": 3,
            "page": 1,
            "section": "Introduction",
            "text": "Machine learning is defined as the study of algorithms. Deep learning refers to neural networks with multiple layers."
        },
        {
            "chunk_id": 4,
            "page": 7,
            "section": "Discussion",
            "text": "We compare our approach versus the baseline. The difference in performance is significant. Our method is better in terms of speed."
        }
    ]
    
    # Test different intent types
    test_cases = [
        {"intent": "RESULT", "confidence": 0.8, "method": "keyword"},
        {"intent": "METHOD", "confidence": 0.9, "method": "keyword"},
        {"intent": "API_USAGE", "confidence": 0.85, "method": "keyword"},
        {"intent": "DEFINITION", "confidence": 0.75, "method": "keyword"},
        {"intent": "COMPARISON", "confidence": 0.7, "method": "keyword"}
    ]
    
    for intent_info in test_cases:
        print(f"\nIntent: {intent_info['intent']} (confidence: {intent_info['confidence']})")
        print("-" * 70)
        
        evidence = select_evidence(
            retrieved_chunks=retrieved_chunks,
            intent_info=intent_info,
            top_k=3
        )
        
        print(f"Top 3 evidence sentences:")
        for i, item in enumerate(evidence, 1):
            print(f"\n{i}. [Score: {item['score']:.2f}] Page {item['page']} | {item['section']}")
            print(f"   {item['sentence']}")
    
    print("\n" + "=" * 70)
    print("Evidence selector test complete!")
