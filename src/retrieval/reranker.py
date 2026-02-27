"""Intent-aware re-ranking"""


class IntentAwareReranker:
    """Re-rank chunks based on query intent"""
    
    def rerank(self, chunks: list, query: str, intent: str) -> list:
        """
        Re-rank chunks based on intent
        
        Args:
            chunks: Retrieved chunks
            query: User query
            intent: Detected intent
            
        Returns:
            Re-ranked chunks
        """
        raise NotImplementedError
