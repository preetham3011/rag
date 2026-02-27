"""Vector database wrapper using FAISS"""

import numpy as np
import faiss


def build_faiss_index(chunks_with_embeddings: list):
    """
    Build a FAISS index from chunk embeddings.
    
    Args:
        chunks_with_embeddings: List of dicts with format
            [{"chunk_id": int, "page": int, "section": str, 
              "text": str, "embedding": list[float]}]
    
    Returns:
        tuple: (faiss_index, metadata_list)
            - faiss_index: FAISS IndexFlatL2 object
            - metadata_list: List of metadata dicts aligned with index positions
    
    Index type:
    - IndexFlatL2: Exact L2 distance search (no approximation)
    - Simple and accurate for small to medium datasets
    """
    if not chunks_with_embeddings:
        raise ValueError("Cannot build index from empty chunks list")
    
    # Extract embeddings and metadata
    embeddings = []
    metadata_list = []
    
    for chunk in chunks_with_embeddings:
        embeddings.append(chunk["embedding"])
        
        # Store metadata (everything except embedding)
        metadata = {
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "section": chunk["section"],
            "text": chunk["text"]
        }
        metadata_list.append(metadata)
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Get embedding dimension
    dimension = embeddings_array.shape[1]
    
    print(f"Building FAISS index...")
    print(f"  Number of chunks: {len(embeddings)}")
    print(f"  Embedding dimension: {dimension}")
    
    # Create FAISS index (L2 distance)
    faiss_index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to index
    faiss_index.add(embeddings_array)
    
    print(f"  Index built successfully with {faiss_index.ntotal} vectors\n")
    
    return faiss_index, metadata_list


def search_index(faiss_index, metadata_list: list, query_embedding: list, top_k: int = 5) -> list:
    """
    Search FAISS index for similar chunks.
    
    Args:
        faiss_index: FAISS index object
        metadata_list: List of metadata dicts aligned with index
        query_embedding: Query embedding vector (list of floats)
        top_k: Number of top results to return
    
    Returns:
        List of dicts with format:
        [{"chunk_id": int, "page": int, "section": str, "text": str, 
          "score": float, "rank": int}]
        
    Note:
    - Lower L2 distance = more similar
    - Results are sorted by similarity (best first)
    """
    # Convert query embedding to numpy array
    query_array = np.array([query_embedding], dtype=np.float32)
    
    # Ensure top_k doesn't exceed index size
    top_k = min(top_k, faiss_index.ntotal)
    
    # Search index
    distances, indices = faiss_index.search(query_array, top_k)
    
    # Build results with metadata
    results = []
    for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
        # Get metadata for this index position
        metadata = metadata_list[idx].copy()
        
        # Add search metadata
        metadata["score"] = float(distance)  # L2 distance
        metadata["rank"] = rank + 1  # 1-indexed rank
        
        results.append(metadata)
    
    return results


if __name__ == "__main__":
    # Test the vector store
    from sentence_transformers import SentenceTransformer
    
    print("Testing FAISS Vector Store")
    print("=" * 70)
    
    # Sample chunks with embeddings
    print("Creating sample data...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    sample_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret visual information.",
        "Reinforcement learning trains agents through rewards and penalties."
    ]
    
    chunks_with_embeddings = []
    for i, text in enumerate(sample_texts):
        embedding = model.encode(text).tolist()
        chunks_with_embeddings.append({
            "chunk_id": i,
            "page": i + 1,
            "section": "Introduction",
            "text": text,
            "embedding": embedding
        })
    
    print(f"Created {len(chunks_with_embeddings)} sample chunks\n")
    
    # Build index
    faiss_index, metadata_list = build_faiss_index(chunks_with_embeddings)
    
    # Test search
    print("Testing search...")
    query_text = "What is neural network deep learning?"
    query_embedding = model.encode(query_text).tolist()
    
    print(f"Query: '{query_text}'")
    print("-" * 70)
    
    results = search_index(faiss_index, metadata_list, query_embedding, top_k=3)
    
    print(f"\nTop {len(results)} results:")
    print("-" * 70)
    for result in results:
        print(f"Rank {result['rank']}: (Score: {result['score']:.4f})")
        print(f"  Chunk ID: {result['chunk_id']}")
        print(f"  Page: {result['page']} | Section: {result['section']}")
        print(f"  Text: {result['text']}")
        print("-" * 70)
    
    print("\nVector store test complete!")
