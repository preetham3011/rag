"""Intent-aware retrieval module for RAG system"""

import re

def calculate_intent_bonus(text, section, intent):
    bonus = 0.0
    text_lower = text.lower()
    section_lower = section.lower()
    if intent == "RESULT":
        if "result" in section_lower:
            bonus += 0.15
        if re.search(r'\d', text):
            bonus += 0.1
        if '%' in text:
            bonus += 0.05
    elif intent == "METHOD":
        if "method" in section_lower:
            bonus += 0.15
        for kw in ["algorithm", "pipeline", "step"]:
            if kw in text_lower:
                bonus += 0.05
                break
    elif intent == "API_USAGE":
        count = sum(1 for s in ["(", ")", "=", ":"] if s in text)
        if count >= 3:
            bonus += 0.2
        elif count >= 2:
            bonus += 0.15
        elif count >= 1:
            bonus += 0.1
    elif intent == "DEFINITION":
        if section_lower in ["abstract", "introduction"]:
            bonus += 0.2
        elif "intro" in section_lower:
            bonus += 0.15
    elif intent == "COMPARISON":
        for kw in ["compare", "comparison", "difference", "versus", "vs"]:
            if kw in text_lower:
                bonus += 0.15
                break
    return min(bonus, 0.3)

def retrieve_with_intent(query_embedding, intent_info, faiss_index, metadata_list, top_k=5):
    from src.indexing.vector_store import search_index
    initial_k = top_k * 2
    initial_results = search_index(faiss_index, metadata_list, query_embedding, top_k=initial_k)
    for result in initial_results:
        result["similarity_score"] = -result["score"]
        del result["score"]
    intent = intent_info.get("intent", "UNKNOWN")
    confidence = intent_info.get("confidence", 0.0)
    apply_bonus = confidence > 0.3
    for result in initial_results:
        bonus = calculate_intent_bonus(result["text"], result["section"], intent) if apply_bonus else 0.0
        result["intent_bonus"] = bonus
        result["final_score"] = result["similarity_score"] + bonus
    reranked_results = sorted(initial_results, key=lambda x: x["final_score"], reverse=True)
    final_results = reranked_results[:top_k]
    for rank, result in enumerate(final_results, start=1):
        result["rank"] = rank
    return final_results
