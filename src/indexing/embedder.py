"""Generate embeddings for chunks using sentence-transformers"""

from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """Reusable embedding model wrapper.

    Centralizes all SentenceTransformer usage so callers never instantiate it directly.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en"):
        # Keep compatibility with older call-sites that may still pass all-MiniLM-L6-v2.
        resolved_model_name = "BAAI/bge-large-en" if model_name == "all-MiniLM-L6-v2" else model_name
        self.model_name = resolved_model_name

        print("Using BGE embeddings with cosine similarity")
        self.model = SentenceTransformer(resolved_model_name)

        self._normalize = self.model_name == "BAAI/bge-large-en"

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings so dot-product equals cosine similarity."""
        if not self._normalize:
            return embedding

        norm = np.linalg.norm(embedding, ord=2)
        if norm == 0:
            return embedding
        return embedding / norm

    def encode_text(self, text: str) -> list:
        """Encode a single text string into a list[float] embedding."""
        embedding = self.model.encode(text)
        embedding_arr = np.array(embedding)
        embedding_arr = self._normalize_embedding(embedding_arr)
        return embedding_arr.tolist()

    def encode_batch(self, texts: list) -> list:
        """Encode a list of texts into list[list[float]] embeddings."""
        embeddings = self.model.encode(texts)
        embeddings_arr = np.array(embeddings)
        if self._normalize:
            norms = np.linalg.norm(embeddings_arr, ord=2, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            embeddings_arr = embeddings_arr / norms
        return embeddings_arr.tolist()

    def get_dimension(self) -> int:
        """Return embedding vector dimensionality."""
        return int(self.model.get_sentence_embedding_dimension())


def generate_embeddings(chunks: list, model_name: str = "BAAI/bge-large-en") -> list:
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
    - BAAI/bge-large-en: Retrieval-oriented embeddings (typically used with cosine similarity)
    - Downloads automatically on first use
    - Runs on CPU by default
    """
    embedder = EmbeddingModel(model_name=model_name)
    
    chunks_with_embeddings = []
    
    print(f"Generating embeddings for {len(chunks)} chunks...")
    if not chunks:
        print("Embedding generation complete.\n")
        return chunks_with_embeddings

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode_batch(texts)

    for i, chunk in enumerate(chunks):
        embedding_list = embeddings[i]
        
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
