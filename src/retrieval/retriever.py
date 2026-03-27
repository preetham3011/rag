"""Intent-aware retrieval module for RAG system"""

import re


def _normalize_scores(values: list) -> list:
    """Min-max normalize scores into [0, 1]."""
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return [1.0 for _ in values]
    return [(v - min_v) / (max_v - min_v) for v in values]


def _log_chunk_list(header: str, chunks: list, max_items: int = 8) -> None:
    """Debug helper to inspect retrieval stages."""
    print(header)
    for c in chunks[:max_items]:
        preview = c.get("text", "").replace("\n", " ")
        preview = preview[:100] + ("..." if len(preview) > 100 else "")
        print(
            f"  - id={c.get('chunk_id')} sec={c.get('section')} "
            f"dense={c.get('dense_norm', 0):.3f} bm25={c.get('sparse_norm', 0):.3f} "
            f"hybrid={c.get('hybrid_score', 0):.3f} rerank={c.get('rerank_score', 0):.3f} "
            f"final={c.get('final_score', 0):.3f} | {preview}"
        )


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

def retrieve_with_intent(query_embedding, intent_info, faiss_index, metadata_list, top_k=5, query=""):
    from src.indexing.vector_store import search_index
    from src.retrieval.reranker import CrossEncoderReranker

    # Larger candidate pool improves recall on sparse/rare facts.
    initial_k = min(max(top_k * 4, 12), 60)
    faiss_results = search_index(faiss_index, metadata_list, query_embedding, top_k=initial_k)

    # Convert FAISS distance-style score into a higher-is-better similarity_score
    for result in faiss_results:
        result["similarity_score"] = -result["score"]
        del result["score"]

    bm25_results = []
    if query:
        try:
            from src.retrieval.bm25_retriever import BM25Retriever

            print("Running BM25 retrieval...")
            bm25 = BM25Retriever(metadata_list)
            bm25_results = bm25.search(query, top_k=initial_k)
            print(f"BM25 candidates: {len(bm25_results)}")
        except ModuleNotFoundError as e:
            print(f"BM25 disabled (missing dependency): {e}")

    _log_chunk_list("Top retrieved chunks before rerank (dense):", faiss_results)

    # Build normalized dense/sparse views for balanced hybrid fusion.
    dense_map = {r["chunk_id"]: r for r in faiss_results}
    sparse_map = {r["chunk_id"]: r for r in bm25_results}

    dense_ids = list(dense_map.keys())
    dense_norm = _normalize_scores([dense_map[cid]["similarity_score"] for cid in dense_ids])
    for cid, score in zip(dense_ids, dense_norm):
        dense_map[cid]["dense_norm"] = score

    sparse_ids = list(sparse_map.keys())
    sparse_norm = _normalize_scores([sparse_map[cid]["bm25_score"] for cid in sparse_ids])
    for cid, score in zip(sparse_ids, sparse_norm):
        sparse_map[cid]["sparse_norm"] = score

    # Merge unique candidates preserving both semantic and lexical signals.
    combined = {}
    for cid in set(dense_map.keys()) | set(sparse_map.keys()):
        base = {}
        if cid in dense_map:
            base.update(dense_map[cid])
        if cid in sparse_map:
            # Keep metadata if dense is missing, and add bm25_score if present.
            for key, value in sparse_map[cid].items():
                if key not in base or key == "bm25_score":
                    base[key] = value
        base["dense_norm"] = dense_map.get(cid, {}).get("dense_norm", 0.0)
        base["sparse_norm"] = sparse_map.get(cid, {}).get("sparse_norm", 0.0)
        # Balanced pre-rerank hybrid score (general-purpose, no query-specific hacks).
        base["hybrid_score"] = 0.5 * base["dense_norm"] + 0.5 * base["sparse_norm"]
        combined[cid] = base

    combined_results = sorted(combined.values(), key=lambda x: x.get("hybrid_score", 0.0), reverse=True)

    # Diversity-aware shortlist: avoid domination by a single section/theme.
    max_candidates = 40
    per_section_cap = max(2, max_candidates // max(top_k, 1))
    section_counts = {}
    diversified = []
    for item in combined_results:
        section = item.get("section", "Unknown")
        count = section_counts.get(section, 0)
        if count < per_section_cap:
            diversified.append(item)
            section_counts[section] = count + 1
        if len(diversified) >= max_candidates:
            break

    # Fill any remaining slots from the sorted list.
    if len(diversified) < max_candidates:
        selected_ids = {x.get("chunk_id") for x in diversified}
        for item in combined_results:
            if item.get("chunk_id") not in selected_ids:
                diversified.append(item)
            if len(diversified) >= max_candidates:
                break

    combined_results = diversified
    print(f"Combined candidates: {len(combined_results)}")
    _log_chunk_list("After hybrid merge:", combined_results)

    print("Running cross-encoder reranking...")
    print(f"Candidates before reranking: {len(combined_results)}")

    if query and combined_results:
        reranker = CrossEncoderReranker()
        cross_reranked_results = reranker.rerank(query, combined_results, top_k=len(combined_results))
    else:
        # Fallback keeps pipeline behavior when query text is unavailable.
        cross_reranked_results = combined_results

    intent = intent_info.get("intent", "UNKNOWN")
    confidence = intent_info.get("confidence", 0.0)
    apply_bonus = confidence > 0.3

    for result in cross_reranked_results:
        bonus = calculate_intent_bonus(result["text"], result["section"], intent) if apply_bonus else 0.0
        result["intent_bonus"] = bonus
        result["final_score"] = (
            result.get("rerank_score", 0) * 0.6 +
            result.get("dense_norm", 0) * 0.3 +
            result.get("sparse_norm", 0) * 0.1 +
            result.get("intent_bonus", 0)
        )

    reranked_results = sorted(cross_reranked_results, key=lambda x: x["final_score"], reverse=True)
    _log_chunk_list("After rerank:", reranked_results)
    final_results = reranked_results[:top_k]

    print(f"Final selected chunks: {len(final_results)}")

    for rank, result in enumerate(final_results, start=1):
        result["rank"] = rank
    return final_results
