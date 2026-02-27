"""Classify query intent using rule-based heuristics"""


def detect_intent(query: str) -> dict:
    """
    Detect query intent using rule-based keyword matching.
    
    Args:
        query: User query string
    
    Returns:
        dict with format:
        {
            "intent": str,  # One of: METHOD, RESULT, API_USAGE, DEFINITION, COMPARISON
            "confidence": float,  # 0.0 to 1.0
            "method": str  # "rule-based"
        }
    
    Intent categories:
    - METHOD: Questions about how something works, architecture, algorithms
    - RESULT: Questions about performance, metrics, outcomes
    - API_USAGE: Questions about function usage, parameters, examples
    - DEFINITION: Questions about what something is or means
    - COMPARISON: Questions comparing multiple things
    """
    query_lower = query.lower()
    
    # Define keyword patterns for each intent
    intent_patterns = {
        "RESULT": [
            "accuracy", "score", "result", "performance", "achieved", 
            "percentage", "metric", "evaluation", "benchmark", "improvement",
            "precision", "recall", "f1", "error rate", "loss"
        ],
        "METHOD": [
            "how does", "how do", "architecture", "approach", "algorithm", 
            "pipeline", "method", "technique", "process", "step", "procedure",
            "implementation", "design", "mechanism", "work"
        ],
        "API_USAGE": [
            "parameter", "argument", "return", "function", "example", 
            "usage", "syntax", "call", "invoke", "signature", "code",
            "how to use", "how to call"
        ],
        "DEFINITION": [
            "what is", "what are", "define", "definition", "meaning of",
            "explain", "describe", "concept of", "term"
        ],
        "COMPARISON": [
            "compare", "comparison", "difference", "versus", "vs", 
            "better than", "worse than", "similar to", "contrast",
            "advantage", "disadvantage"
        ]
    }
    
    # Score each intent based on keyword matches
    intent_scores = {}
    
    for intent, keywords in intent_patterns.items():
        score = 0
        matches = 0
        
        for keyword in keywords:
            if keyword in query_lower:
                matches += 1
                # Weight longer keywords more heavily
                score += len(keyword.split())
        
        # Normalize score
        if matches > 0:
            intent_scores[intent] = score / len(keywords)
    
    # Determine best intent
    if intent_scores:
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = min(intent_scores[best_intent] * 2, 1.0)  # Scale to 0-1
        
        # Boost confidence if multiple keywords matched
        keyword_count = sum(1 for kw in intent_patterns[best_intent] if kw in query_lower)
        if keyword_count >= 2:
            confidence = min(confidence + 0.2, 1.0)
    else:
        # Fallback: default to DEFINITION with low confidence
        best_intent = "DEFINITION"
        confidence = 0.3
    
    return {
        "intent": best_intent,
        "confidence": confidence,
        "method": "rule-based"
    }


if __name__ == "__main__":
    # Test intent detection
    
    test_queries = [
        "What is adaptive context compression?",
        "How does the compression algorithm work?",
        "What accuracy did the model achieve on the test set?",
        "How do I use the authenticate() function?",
        "Compare BERT and GPT architectures",
        "What are the parameters for the encode() method?",
        "Explain the methodology used in the paper",
        "What were the results on ImageNet?",
        "What is the difference between precision and recall?",
        "Show me an example of using the API"
    ]
    
    print("Testing Intent Detection")
    print("=" * 70)
    
    for query in test_queries:
        result = detect_intent(query)
        
        print(f"\nQuery: {query}")
        print(f"  Intent: {result['intent']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Method: {result['method']}")
        print("-" * 70)
    
    # Summary statistics
    print("\nIntent Distribution:")
    from collections import Counter
    intents = [detect_intent(q)["intent"] for q in test_queries]
    intent_counts = Counter(intents)
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count} queries")
