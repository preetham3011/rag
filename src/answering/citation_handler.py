"""Extract citations and detect insufficient evidence"""


class CitationHandler:
    """Handle citations and evidence sufficiency"""
    
    def extract_citations(self, answer: str, evidence: list) -> list:
        """Extract citation references from answer"""
        raise NotImplementedError
    
    def check_sufficiency(self, query: str, evidence: list) -> bool:
        """Check if evidence is sufficient to answer query"""
        raise NotImplementedError
    
    def generate_refusal(self, query: str) -> str:
        """Generate refusal message when evidence is insufficient"""
        raise NotImplementedError
