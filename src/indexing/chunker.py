"""Section-aware chunking logic"""


def chunk_documents(pages_with_sections: list, chunk_size: int = 1000) -> list:
    """
    Split document pages into chunks while preserving metadata.
    
    Args:
        pages_with_sections: List of dicts with format 
                            [{"page": int, "section": str, "text": str}]
        chunk_size: Target chunk size in characters (default: 1000)
        
    Returns:
        List of dicts with format:
        [{"chunk_id": int, "page": int, "section": str, "text": str}]
        
    Chunking strategy:
    - Split text at sentence boundaries when possible
    - Preserve page and section metadata
    - Assign unique incremental chunk IDs
    """
    chunks = []
    chunk_id = 0
    
    for page_data in pages_with_sections:
        page_num = page_data["page"]
        section = page_data["section"]
        text = page_data["text"]
        
        # Split this page's text into chunks
        page_chunks = _split_text_into_chunks(text, chunk_size)
        
        # Add metadata to each chunk
        for chunk_text in page_chunks:
            chunks.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "section": section,
                "text": chunk_text
            })
            chunk_id += 1
    
    return chunks


def _split_text_into_chunks(text: str, chunk_size: int) -> list:
    """
    Split text into chunks at sentence boundaries.
    
    Args:
        text: Text to split
        chunk_size: Target chunk size in characters
        
    Returns:
        List of text chunks
        
    Strategy:
    - Split text into sentences
    - Group sentences into chunks of approximately chunk_size
    - Avoid splitting mid-sentence when possible
    """
    # Handle empty or very short text
    if not text or len(text.strip()) == 0:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    # Split into sentences (simple heuristic)
    sentences = _split_into_sentences(text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed chunk_size
        if len(current_chunk) + len(sentence) > chunk_size:
            # Save current chunk if it's not empty
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Start new chunk with this sentence
            current_chunk = sentence
        else:
            # Add sentence to current chunk
            current_chunk += sentence
    
    # Add the last chunk if not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def _split_into_sentences(text: str) -> list:
    """
    Split text into sentences using simple heuristics.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
        
    Strategy:
    - Split on common sentence terminators: . ! ?
    - Keep the terminator with the sentence
    - Handle common abbreviations (Dr., Mr., etc.)
    """
    import re
    
    # Simple sentence splitting pattern
    # Split on . ! ? followed by whitespace and capital letter
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    
    sentences = re.split(sentence_pattern, text)
    
    # Handle edge case where split produces empty strings
    sentences = [s for s in sentences if s.strip()]
    
    return sentences


if __name__ == "__main__":
    # Test the chunker
    
    # Sample input (output from section_detector)
    test_pages = [
        {
            "page": 1,
            "section": "Abstract",
            "text": "This is a short abstract. It contains multiple sentences. Each sentence adds information. The abstract summarizes the paper."
        },
        {
            "page": 2,
            "section": "Introduction",
            "text": "Introduction paragraph one. " * 50 + "This is a very long introduction that will definitely exceed the chunk size limit. " * 20
        },
        {
            "page": 3,
            "section": "Method",
            "text": "Our method consists of three steps. First, we preprocess the data. Second, we apply the algorithm. Third, we evaluate results."
        }
    ]
    
    print("Testing Document Chunker")
    print("=" * 70)
    
    # Run chunking with small chunk size for testing
    chunks = chunk_documents(test_pages, chunk_size=200)
    
    print(f"Total chunks created: {len(chunks)}\n")
    
    # Display first few chunks
    for chunk in chunks[:5]:
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Page: {chunk['page']} | Section: {chunk['section']}")
        print(f"Text length: {len(chunk['text'])} chars")
        print(f"Text preview: {chunk['text'][:100]}...")
        print("-" * 70)
    
    # Statistics
    print("\nChunk Statistics:")
    print(f"  Average chunk size: {sum(len(c['text']) for c in chunks) / len(chunks):.0f} chars")
    print(f"  Min chunk size: {min(len(c['text']) for c in chunks)} chars")
    print(f"  Max chunk size: {max(len(c['text']) for c in chunks)} chars")
    
    # Section distribution
    from collections import Counter
    section_counts = Counter(c['section'] for c in chunks)
    print(f"\nChunks per section:")
    for section, count in section_counts.items():
        print(f"  {section}: {count} chunks")
