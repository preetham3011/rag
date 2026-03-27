import streamlit as st

import hashlib
import os
import tempfile

from src.ingestion.pdf_extractor import extract_text_from_pdf
from src.ingestion.section_detector import detect_sections
from src.indexing.chunker import chunk_documents
from src.indexing.embedder import generate_embeddings
from src.indexing.vector_store import build_faiss_index
from evaluation.baseline_rag import run_baseline_rag
from evaluation.adaptive_rag import run_adaptive_rag


@st.cache_resource(show_spinner=False)
def build_index_for_pdf(pdf_bytes: bytes):
    """
    PHASE 1 (run once per PDF):
    - Extract -> section detect -> chunk -> embed -> build FAISS
    Cached by pdf_bytes content (Streamlit hashes function args).
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        print(f"[app] Processing uploaded PDF once: {tmp_path}")

        pages = extract_text_from_pdf(tmp_path)
        pages_with_sections = detect_sections(pages)
        chunks = chunk_documents(pages_with_sections)
        chunks_with_embeddings = generate_embeddings(chunks)
        faiss_index, metadata_list = build_faiss_index(chunks_with_embeddings)

        return faiss_index, metadata_list, len(pages), len(chunks)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                print(f"[app] Failed to delete temp PDF: {e}")


def main():
    st.title("Adaptive RAG System")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    query = st.text_input("Enter your question")
    run_button = st.button("Run Query")

    if uploaded_file is None:
        st.warning("Please upload a PDF.")
        return

    pdf_bytes = uploaded_file.getvalue()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()[:12]
    st.caption(f"Loaded document: `{uploaded_file.name}` (sha256: {pdf_hash})")

    # PHASE 1: build/cache index once per uploaded PDF
    with st.spinner("Processing document (runs once per upload)..."):
        faiss_index, metadata_list, num_pages, num_chunks = build_index_for_pdf(pdf_bytes)
    st.success(f"Document indexed. Pages: {num_pages}, Chunks: {num_chunks}")

    if not run_button:
        return

    if not query or not query.strip():
        st.warning("Please enter a question.")
        return

    # PHASE 2: query-time pipeline only (runs per query)
    with st.spinner("Running query..."):
        print(f"[app] Running query against cached index (doc sha256: {pdf_hash})")

        baseline_result = run_baseline_rag(
            query=query,
            faiss_index=faiss_index,
            metadata_list=metadata_list,
        )

        adaptive_result = run_adaptive_rag(
            query=query,
            faiss_index=faiss_index,
            metadata_list=metadata_list,
        )

    st.subheader("Baseline Answer")
    st.write(baseline_result.get("answer", ""))

    st.subheader("Adaptive Answer")
    st.write(adaptive_result.get("answer", ""))


if __name__ == "__main__":
    main()

