try:
    from rank_bm25 import BM25Okapi
except ModuleNotFoundError:  # pragma: no cover
    BM25Okapi = None


class BM25Retriever:
    def __init__(self, documents):
        """
        documents: list of dicts with 'text'
        """
        if BM25Okapi is None:
            raise ModuleNotFoundError(
                "rank_bm25 is not installed. Install it with: pip install rank_bm25"
            )
        self.documents = documents
        self.tokenized_corpus = [doc["text"].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=10):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        results = []
        for i, score in enumerate(scores):
            doc = self.documents[i].copy()
            doc["bm25_score"] = float(score)
            results.append(doc)

        results = sorted(results, key=lambda x: x["bm25_score"], reverse=True)
        return results[:top_k]

