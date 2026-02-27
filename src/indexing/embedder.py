"""Generate embeddings for chunks using sentence-transformers"""

from sentence_transformers import SentenceTransformer


def generate_embeddings(chunks: list, model_name: str = "all-MiniLM-L6-v2") -> list:
    """
    Generate embeddings for document chunks.
    
    Args:
        chunks: List of dicts with format 
                [{"chunk_id": int, "page": int, "section": str, "text": str}]
        model_name: Name of the sentence-transformers model to use
        
    Returns:
        List of dicts with added "embedding" field:
        [{"chunk_id": int, "page": int, "section": str, "text": str, 
          "embedding": list[float]}]
          
    Model info:
    - all-MiniLM-L6-v2: Fast, 384-dimensional embeddings
    - Downloads automatically on first use
    - Runs on CPU by default
    """
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Model loaded successfully.\n")
    
    chunks_with_embeddings = []
    
    print(f"Generating embeddings for {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        # Extract text from chunk
        text = chunk["text"]
        
        # Generate embedding
        embedding = model.encode(text)
        
        # Convert numpy array to list for JSON serialization
        embedding_list = embedding.tolist()
        
        # Add embedding to chunk
        chunk_with_embedding = {
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "section": chunk["section"],
            "text": chunk["text"],
            "embedding": embedding_list
        }
        
        chunks_with_embeddings.append(chunk_with_embedding)
        
        # Progress indicator
        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Processed {i + 1}/{len(chunks)} chunks")
    
    print("Embedding generation complete.\n")
    return chunks_with_embeddings


if __name__ == "__main__":
    # Test the embedder
    
    # Sample input (output from chunker)
    test_chunks = [
        {
            "chunk_id": 0,
            "page": 1,
            "section": "Abstract",
            "text": "This paper presents a novel approach to document retrieval using adaptive context compression."
        },
        {
            "chunk_id": 1,
            "page": 2,
            "section": "Introduction",
            "text": "Retrieval-Augmented Generation (RAG) systems face challenges with long documents that exceed context limits."
        },
        {
            "chunk_id": 2,
            "page": 3,
            "section": "Method",
            "text": "Our method detects query intent and compresses evidence accordingly to preserve high-signal content."
        }
    ]
    
    print("Testing Embedding Generator")
    print("=" * 70)
    
    # Generate embeddings
    chunks_with_embeddings = generate_embeddings(test_chunks)
    
    # Display results
    print("Results:")
    print("-" * 70)
    for chunk in chunks_with_embeddings:
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Page: {chunk['page']} | Section: {chunk['section']}")
        print(f"Text: {chunk['text'][:60]}...")
        print(f"Embedding dimension: {len(chunk['embedding'])}")
        print(f"Embedding preview: {chunk['embedding'][:5]}...")
        print("-" * 70)
    
    # Verify all chunks have embeddings
    all_have_embeddings = all("embedding" in c for c in chunks_with_embeddings)
    print(f"\nAll chunks have embeddings: {all_have_embeddings}")
    print(f"Embedding dimension: {len(chunks_with_embeddings[0]['embedding'])}")
