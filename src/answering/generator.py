"""LLM-based answer generation"""


class AnswerGenerator:
    """Generate grounded answers using compressed evidence"""
    
    def generate(self, query: str, compressed_evidence: str, intent: str) -> dict:
        """
        Generate answer from evidence
        
        Returns:
            dict with 'answer', 'citations', 'confidence'
        """
        raise NotImplementedError
