"""Cross-encoder reranking for second-stage retrieval."""


class CrossEncoderReranker:
    """Rerank FAISS candidates using a cross-encoder relevance model."""

    _model = None

    def __init__(self):
        from sentence_transformers import CrossEncoder

        if CrossEncoderReranker._model is None:
            print("Loading cross-encoder reranker...")
            CrossEncoderReranker._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.model = CrossEncoderReranker._model

    def rerank(self, query: str, chunks: list, top_k: int = 5) -> list:
        """Rerank chunks by cross-encoder score and return top_k."""
        if not chunks:
            return []

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        rerank_candidates = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            chunk_copy["rerank_score"] = float(scores[i])
            rerank_candidates.append(chunk_copy)

        ranked = sorted(rerank_candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]
