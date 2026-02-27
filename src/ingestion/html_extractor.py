"""HTML text extraction for API documentation"""


class HTMLExtractor:
    """Extract text from HTML documents"""
    
    def extract(self, html_path: str) -> dict:
        """
        Extract text from HTML
        
        Returns:
            dict with 'text', 'metadata', 'structure'
        """
        raise NotImplementedError
