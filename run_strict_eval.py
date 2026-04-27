import os
import json
import time
from pathlib import Path
import sys

# Ensure root dir is in path
sys.path.append(str(Path(__file__).parent))
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from src.ingestion.pdf_extractor import extract_text_from_pdf
from src.ingestion.section_detector import detect_sections
from src.indexing.chunker import chunk_documents
from src.indexing.embedder import generate_embeddings
from src.indexing.vector_store import build_faiss_index
from evaluation.baseline_rag import run_baseline_rag
from evaluation.adaptive_rag import run_adaptive_rag

# Constants
PAPERS_DIR = Path("papers")
QUERIES_DIR = Path("queries")
RESULTS_DIR = Path("evaluation_results")

def build_index(pdf_path: str):
    print(f"Loading and indexing {pdf_path}...")
    pages = extract_text_from_pdf(pdf_path)
    pages_with_sections = detect_sections(pages)
    chunks = chunk_documents(pages_with_sections)
    chunks_with_embeddings = generate_embeddings(chunks)
    faiss_index, metadata_list = build_faiss_index(chunks_with_embeddings)
    return faiss_index, metadata_list

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    
    global_metrics = {
        "total_papers": 0,
        "total_queries": 0,
        "avg_token_reduction": 0.0,
        "min_token_reduction": 1000.0,
        "max_token_reduction": -1000.0,
        "avg_latency_baseline": 0.0,
        "avg_latency_adaptive": 0.0,
        "avg_latency_difference": 0.0,
        "avg_sentences_selected": 0.0
    }
    
    all_reduction_ratios = []
    all_latency_baselines = []
    all_latency_adaptives = []
    all_sentences_selected = []
    
    paper_files = ["paper1.pdf", "paper2.pdf", "paper3.pdf", "paper4.pdf"]
    
    for paper_file in paper_files:
        paper_id = paper_file.split(".")[0]
        pdf_path = PAPERS_DIR / paper_file
        query_path = QUERIES_DIR / f"{paper_id}.json"
        
        if not pdf_path.exists() or not query_path.exists():
            print(f"Skipping {paper_id} due to missing pdf or query file.")
            continue
            
        global_metrics["total_papers"] += 1
        
        # Load queries
        with open(query_path, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
            
        faiss_index, metadata_list = build_index(str(pdf_path))
        
        paper_results = []
        
        for q_data in queries_data:
            query = q_data["question"]
            print(f"Running queries for: {query}")
            
            # Baseline RAG
            start_time = time.time()
            try:
                baseline_res = run_baseline_rag(
                    query=query,
                    faiss_index=faiss_index,
                    metadata_list=metadata_list
                )
                base_ans = baseline_res.get("answer", "ERROR")
                base_tokens = baseline_res.get("token_count", 0)
            except Exception as e:
                print(f"Error in baseline: {e}")
                base_ans = "ERROR"
                base_tokens = 0
            base_lat = time.time() - start_time
            
            # Adaptive RAG
            start_time = time.time()
            try:
                adaptive_res = run_adaptive_rag(
                    query=query,
                    faiss_index=faiss_index,
                    metadata_list=metadata_list
                )
                adap_ans = adaptive_res.get("answer", "ERROR")
                adap_tokens = adaptive_res.get("tokens_used", 0)
                num_sentences = adaptive_res.get("num_sentences", 0)
                intent = adaptive_res.get("intent", {}).get("intent", "UNKNOWN")
            except Exception as e:
                print(f"Error in adaptive: {e}")
                adap_ans = "ERROR"
                adap_tokens = 0
                num_sentences = 0
                intent = "UNKNOWN"
            adap_lat = time.time() - start_time
            
            # Validate constraints
            if num_sentences == 0:
                print(f"WARNING: num_sentences is 0 for query: {query}")
                num_sentences = max(1, num_sentences)
            if base_tokens <= 0:
                print(f"WARNING: base_tokens <= 0 for query: {query}")
                base_tokens = max(1, base_tokens)
            if adap_tokens <= 0:
                print(f"WARNING: adap_tokens <= 0 for query: {query}")
                adap_tokens = max(1, adap_tokens)
            
            reduction = 0.0
            if base_tokens > 0:
                reduction = (1.0 - (adap_tokens / base_tokens)) * 100.0
                
            lat_diff = adap_lat - base_lat
            
            global_metrics["total_queries"] += 1
            all_reduction_ratios.append(reduction)
            all_latency_baselines.append(base_lat)
            all_latency_adaptives.append(adap_lat)
            all_sentences_selected.append(num_sentences)
            
            if reduction < global_metrics["min_token_reduction"]:
                global_metrics["min_token_reduction"] = reduction
            if reduction > global_metrics["max_token_reduction"]:
                global_metrics["max_token_reduction"] = reduction
                
            paper_results.append({
                "query": query,
                "intent": intent,
                "baseline": {
                    "answer": base_ans,
                    "tokens": base_tokens,
                    "latency": base_lat
                },
                "adaptive": {
                    "answer": adap_ans,
                    "tokens": adap_tokens,
                    "latency": adap_lat,
                    "num_sentences": num_sentences
                },
                "metrics": {
                    "token_reduction": reduction,
                    "latency_diff": lat_diff
                }
            })
            
        out_path = RESULTS_DIR / f"{paper_id}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "paper_id": paper_id,
                "num_queries": len(queries_data),
                "results": paper_results
            }, f, indent=2)

    if global_metrics["total_queries"] > 0:
        global_metrics["avg_token_reduction"] = sum(all_reduction_ratios) / len(all_reduction_ratios)
        global_metrics["avg_latency_baseline"] = sum(all_latency_baselines) / len(all_latency_baselines)
        global_metrics["avg_latency_adaptive"] = sum(all_latency_adaptives) / len(all_latency_adaptives)
        global_metrics["avg_latency_difference"] = global_metrics["avg_latency_adaptive"] - global_metrics["avg_latency_baseline"]
        global_metrics["avg_sentences_selected"] = sum(all_sentences_selected) / len(all_sentences_selected)
    
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(global_metrics, f, indent=2)
        
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    main()
