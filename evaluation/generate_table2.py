import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexing.vector_store import build_faiss_index
from src.indexing.embedder import EmbeddingModel
from evaluation.baseline_rag import run_baseline_rag
from evaluation.adaptive_rag import run_adaptive_rag
from tests.integration_test import create_mock_document

def generate():
    # Use the mock document from the integration test
    chunks = create_mock_document()
    faiss_index, metadata_list = build_faiss_index(chunks)
    
    queries = [
        "What accuracy was achieved?",
        "How do I use the compress_context function?",
        "What is the token reduction achieved?",
        "How does the compression pipeline work?",
        "What is defined as Retrieval-Augmented Generation?",
        "How is this method compared to fixed chunking?",
        "What were the precision and recall scores?",
        "How do I detect query intent using the API?",
        "What is the difference in performance?",
        "First step of the algorithm pipeline?"
    ]
    
    print("| Query | Intent (conf) | Baseline Tokens | Adaptive Tokens | Reduction |")
    print("|---|---|---|---|---|")
    
    total_baseline = 0
    total_adaptive = 0
    
    for q in queries:
        b_res = run_baseline_rag(q, faiss_index, metadata_list, top_k=5)
        a_res = run_adaptive_rag(q, faiss_index, metadata_list, top_k=5, token_limit=500)
        
        b_tok = b_res["token_count"]
        a_tok = a_res["tokens_used"]
        intent = a_res["intent"]["intent"]
        conf = a_res["intent"]["confidence"]
        
        reduction = (1 - (a_tok / b_tok)) * 100 if b_tok > 0 else 0
        
        total_baseline += b_tok
        total_adaptive += a_tok
        
        print(f"| {q} | {intent} ({conf:.2f}) | {b_tok} | {a_tok} | {reduction:.1f}\\% |")
        
    avg_reduction = (1 - (total_adaptive / total_baseline)) * 100 if total_baseline > 0 else 0
    print(f"| **Mean** | | **{total_baseline/len(queries):.0f}** | **{total_adaptive/len(queries):.0f}** | **{avg_reduction:.1f}\\%** |")

if __name__ == '__main__':
    generate()
