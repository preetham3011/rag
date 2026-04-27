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

    # Generic factual-signal preservation (query-agnostic):
    # keep numerical/entity-heavy technical statements competitive.
    base_score = len(sentence.split()) / 20
    score += base_score

    if re.search(r"\d", sentence):
        score += 0.15
    if re.search(r"\b[A-Z]{2,}\b", sentence):
        score += 0.10
    if re.search(r"\b[A-Za-z]+[-_/][A-Za-z0-9]+\b", sentence):
        score += 0.08
    if re.search(r"\b[A-Za-z]+\d+[A-Za-z0-9]*\b", sentence):
        score += 0.07
    
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


def knapsack_select(sentences: list, token_budget: int) -> list:
    """
    Select sentence subset maximizing total score under token budget (0/1 knapsack).

    Args:
        sentences: List of dicts with "score" and "token_count"
        token_budget: Maximum token budget

    Returns:
        Selected sentence dicts in original candidate order
    """
    if not sentences or token_budget <= 0:
        return []

    n = len(sentences)
    capacity = int(token_budget)

    # dp[i][w] = max score using first i items within budget w
    dp = [[0.0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        item = sentences[i - 1]
        weight = int(item.get("token_count", 0))
        value = float(item.get("score", 0.0))
        for w in range(capacity + 1):
            best_without = dp[i - 1][w]
            best_with = best_without
            if weight <= w and weight > 0:
                best_with = dp[i - 1][w - weight] + value
            dp[i][w] = best_with if best_with > best_without else best_without

    # Backtrack to recover selected items
    selected_indices = set()
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_indices.add(i - 1)
            w -= int(sentences[i - 1].get("token_count", 0))
            if w <= 0:
                break

    # Keep original order of candidates in output
    return [sentences[i] for i in range(n) if i in selected_indices]


def select_evidence(
    retrieved_chunks: list,
    intent_info: dict,
    top_k: int = 10,
    token_budget: int = None
) -> list:
    """
    Select high-signal evidence sentences from retrieved chunks.
    
    Args:
        retrieved_chunks: List of dicts with format:
            [{"chunk_id": int, "page": int, "section": str, "text": str, ...}]
        intent_info: Dict with format:
            {"intent": str, "confidence": float, "method": str}
        top_k: Number of top evidence sentences to return when no budget optimization is used
        token_budget: Optional token budget for optimization-based selection
    
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

    # Add token count metadata for optimization.
    for evidence in evidence_list:
        evidence["token_count"] = len(evidence["sentence"]) // 4

    # Keep DP cost bounded for responsiveness.
    candidate_pool = evidence_list[:40]

    if token_budget is not None:
        print("Using optimization-based compression (knapsack)")
        selected = knapsack_select(candidate_pool, token_budget)
        if len(selected) == 0:
            selected = candidate_pool[:3]
        print(f"Selected {len(selected)} sentences under token budget")
        # Output sorted by score to keep previous downstream expectations.
        selected.sort(key=lambda x: x["score"], reverse=True)
        return selected

    # Backward-compatible path without explicit budget.
    return candidate_pool[:top_k]

