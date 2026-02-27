"""Evaluation metrics"""


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
