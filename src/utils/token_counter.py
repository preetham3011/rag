"""Token counting utilities"""


class TokenCounter:
    """Count tokens for budget management"""
    
    @staticmethod
    def count(text: str, model: str = "gpt-3.5-turbo") -> int:
        """Count tokens in text"""
        raise NotImplementedError
