"""Evaluation metrics"""

from numpy import dot
from numpy.linalg import norm


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors."""
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return float(dot(a, b) / denom)


def semantic_similarity(embedder, ans1: str, ans2: str) -> float:
    """Semantic answer similarity using embedding cosine similarity."""
    v1 = embedder.encode_text(ans1)
    v2 = embedder.encode_text(ans2)
    return cosine_similarity(v1, v2)


def efficiency_score(similarity: float, tokens: int) -> float:
    """Simple efficiency: semantic quality per token."""
    return float(similarity / max(tokens, 1))


class Metrics:
    """Calculate evaluation metrics"""
    
    @staticmethod
    def token_reduction_ratio(baseline_tokens: int, compressed_tokens: int) -> float:
        """Calculate token reduction ratio"""
        return 1 - (compressed_tokens / baseline_tokens)
    
    @staticmethod
    def calculate_correctness(answer: str, ground_truth: str) -> float:
        """Manual correctness scoring (0-1)"""
        raise NotImplementedError
    
    @staticmethod
    def unsupported_answer_rate(answers: list) -> float:
        """Calculate rate of unsupported/hallucinated answers"""
        raise NotImplementedError
    
    @staticmethod
    def citation_quality(citations: list, evidence: list) -> float:
        """Evaluate if citations truly support the answer"""
        raise NotImplementedError
