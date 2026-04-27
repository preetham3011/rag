"""Adaptive RAG System — Professional Research Dashboard"""

import hashlib
import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv

from evaluation.adaptive_rag import run_adaptive_rag
from evaluation.baseline_rag import run_baseline_rag
from src.indexing.chunker import chunk_documents
from src.indexing.embedder import generate_embeddings
from src.indexing.vector_store import build_faiss_index
from src.ingestion.pdf_extractor import extract_text_from_pdf
from src.ingestion.section_detector import detect_sections

# Load environment variables explicitly
load_dotenv()

st.set_page_config(
    page_title="Adaptive RAG Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #f8f9fa; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #1e2230 !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label { color: #b0b8d0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e8eaed;
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 700; color: #1a1f36; }
[data-testid="stMetricLabel"] { font-size: 0.8rem; font-weight: 500; color: #6b7280; }

/* File uploader & inputs */
[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #d1d5db;
    border-radius: 10px;
    padding: 10px;
}

/* Buttons */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    font-size: 0.95rem;
    transition: background 0.2s;
}
.stButton > button:hover { background-color: #1d4ed8; }

/* Tabs */
button[data-baseweb="tab"] { font-weight: 600; font-size: 0.85rem; }

/* Section headings */
.sec-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #6b7280;
    text-transform: uppercase;
    margin-bottom: 10px;
    margin-top: 28px;
}
.sec-title {
    font-size: 1.65rem;
    font-weight: 700;
    color: #1a1f36;
    margin-bottom: 4px;
}
.sec-sub {
    font-size: 0.95rem;
    color: #6b7280;
    margin-bottom: 24px;
}

/* Cards */
.card {
    background: white;
    border: 1px solid #e8eaed;
    border-radius: 12px;
    padding: 22px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
.card-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 12px;
}

/* Pipeline nodes */
.pipe-row {
    display: flex;
    align-items: center;
    gap: 8px;
}
.pipe-node {
    flex: 1;
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px 8px;
    text-align: center;
}
.pipe-node.done { background: #eff6ff; border-color: #bfdbfe; }
.pipe-node.active { background: #dcfce7; border-color: #86efac; }
.pipe-node-title { font-size: 0.78rem; font-weight: 700; color: #374151; }
.pipe-node-sub { font-size: 0.7rem; color: #6b7280; margin-top: 3px; }
.pipe-arrow { color: #9ca3af; font-size: 1.1rem; flex-shrink: 0; }

/* Intent badge */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-blue { background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; }
.badge-green { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }

/* Answer panels */
.answer-box {
    background: white;
    border: 1px solid #e8eaed;
    border-radius: 10px;
    padding: 20px 22px;
    min-height: 200px;
    font-size: 0.92rem;
    color: #374151;
    line-height: 1.7;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.answer-box.adaptive { border-top: 3px solid #22c55e; }
.answer-box.baseline { border-top: 3px solid #6b7280; }
.answer-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 12px;
}
.answer-label.bl { color: #6b7280; }
.answer-label.ad { color: #16a34a; }

/* Token bars */
.bar-wrap { margin-bottom: 18px; }
.bar-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 5px;
}
.bar-bg { background: #f3f4f6; border-radius: 6px; height: 16px; overflow: hidden; }
.bar-fill-bl { background: #94a3b8; height: 100%; border-radius: 6px; }
.bar-fill-ad { background: #22c55e; height: 100%; border-radius: 6px; }

/* Evidence */
.ev-item {
    background: #f9fafb;
    border-left: 3px solid #2563eb;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    color: #374151;
    line-height: 1.5;
}
.ev-meta { font-size: 0.72rem; color: #9ca3af; font-weight: 600; margin-bottom: 4px; }

/* Context pre */
.ctx-box {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 16px;
    max-height: 380px;
    overflow-y: auto;
    font-family: 'Menlo', 'Consolas', monospace;
    font-size: 0.82rem;
    color: #374151;
    line-height: 1.6;
    white-space: pre-wrap;
}

div.stSpinner > div { border-top-color: #2563eb !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar (navigation + settings only) ──────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Adaptive RAG")
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.caption("Embedding: BAAI/bge-large-en")
    st.caption("Retriever: FAISS + BM25 Hybrid")
    st.caption("Reranker: MS-MARCO CrossEncoder")
    st.caption("LLM: Gemini 2.5 Flash / Groq Llama 3.3")
    st.markdown("---")
    llm_provider = st.selectbox("LLM Provider", ["gemini", "groq"])
    import src.answering.llm as llm
    llm.LLM_MODE = llm_provider
    debug = st.checkbox("Debug mode", value=False)

# ── Indexing helper ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_index_for_pdf(pdf_bytes: bytes):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
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
            except Exception:
                pass

# ── SECTION 1: Header ─────────────────────────────────────────────────────────
st.markdown("<div class='sec-label'>Research Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sec-title'>Adaptive RAG System</div>", unsafe_allow_html=True)
st.markdown("<div class='sec-sub'>Hybrid Retrieval · Cross-Encoder Reranking · Intent-Aware Context Compression</div>", unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:1px solid #e8eaed;margin-bottom:24px;'>", unsafe_allow_html=True)

# ── SECTION 2: Input controls (main area) ─────────────────────────────────────
st.markdown("<div class='sec-label'>Step 1 — Upload & Query</div>", unsafe_allow_html=True)

inp_col1, inp_col2 = st.columns([1, 1], gap="large")
with inp_col1:
    uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"])
with inp_col2:
    query = st.text_area("Research question", placeholder="e.g. What accuracy was achieved by the model?", height=100)
    run_button = st.button("Run Analysis", use_container_width=True)

st.markdown("<hr style='border:none;border-top:1px solid #e8eaed;margin:20px 0 24px 0;'>", unsafe_allow_html=True)

# ── Guard: no file ─────────────────────────────────────────────────────────────
if not uploaded_file:
    st.info("Upload a PDF document above to get started.")
    st.stop()

pdf_bytes = uploaded_file.getvalue()
pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()[:12]

with st.spinner("Indexing document…"):
    faiss_index, metadata_list, num_pages, num_chunks = build_index_for_pdf(pdf_bytes)

# Workspace ready card
st.markdown(f"""
<div class='card' style='margin-bottom:0;'>
    <div class='card-label'>Workspace</div>
    <span class='badge badge-blue'>{num_pages} pages</span>&nbsp;
    <span class='badge badge-blue'>{num_chunks} chunks indexed</span>&nbsp;
    <span class='badge badge-green'>FAISS ready</span>
    <div style='font-size:0.8rem;color:#9ca3af;margin-top:8px;'>Document: <code>{uploaded_file.name}</code> &nbsp;·&nbsp; SHA-256: <code>{pdf_hash}</code></div>
</div>
""", unsafe_allow_html=True)

if not run_button or not query.strip():
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Enter a question above and click **Run Analysis** to execute the pipeline.")
    st.stop()

# ── Run pipelines ──────────────────────────────────────────────────────────────
with st.spinner("Running Baseline and Adaptive pipelines…"):
    t0 = time.time()
    baseline_result = run_baseline_rag(query=query, faiss_index=faiss_index, metadata_list=metadata_list)
    t1 = time.time()
    adaptive_result = run_adaptive_rag(query=query, faiss_index=faiss_index, metadata_list=metadata_list)
    t2 = time.time()

    latency_baseline = t1 - t0
    latency_adaptive = t2 - t1

baseline_tokens = baseline_result.get("token_count", 0)
adaptive_tokens = adaptive_result.get("tokens_used", 0)
reduction_pct   = round((1 - adaptive_tokens / baseline_tokens) * 100, 1) if baseline_tokens > 0 else 0.0
intent_info     = adaptive_result.get("intent", {})
intent_label    = intent_info.get("intent", "UNKNOWN")
intent_conf     = intent_info.get("confidence", 0.0)
num_retrieved   = len(baseline_result.get("retrieved_chunks", []))
num_sentences   = adaptive_result.get("num_sentences", 0)
selected_ev     = adaptive_result.get("selected_evidence", [])

st.markdown("<hr style='border:none;border-top:1px solid #e8eaed;margin:24px 0;'>", unsafe_allow_html=True)

# ── SECTION 3: KPI metrics ─────────────────────────────────────────────────────
st.markdown("<div class='sec-label'>Step 2 — Performance Metrics</div>", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Baseline Tokens", baseline_tokens)
m2.metric("Adaptive Tokens", adaptive_tokens,
          delta=f"-{baseline_tokens - adaptive_tokens}", delta_color="inverse")
m3.metric("Token Reduction", f"{reduction_pct}%",
          delta=f"{reduction_pct}%", delta_color="normal")
m4.metric("Retrieved Chunks", num_retrieved)
m5.metric("Selected Sentences", num_sentences)

st.markdown("<br>", unsafe_allow_html=True)

evidence_density = num_sentences / adaptive_tokens if adaptive_tokens > 0 else 0
compression_ratio = adaptive_tokens / baseline_tokens if baseline_tokens > 0 else 0
token_savings = baseline_tokens - adaptive_tokens
latency_diff = latency_baseline - latency_adaptive

n1, n2, n3, n4 = st.columns(4)
n1.metric("Evidence Density", f"{evidence_density:.4f}")
n2.metric("Compression Ratio", f"{compression_ratio:.2f}")
n3.metric("Token Savings", token_savings)
n4.metric("Latency Diff", f"{latency_diff:.2f}s", 
          delta=f"{latency_diff:.2f}s", delta_color="normal")

st.markdown("<hr style='border:none;border-top:1px solid #e8eaed;margin:24px 0;'>", unsafe_allow_html=True)

# ── SECTION 4: Pipeline overview ──────────────────────────────────────────────
st.markdown("<div class='sec-label'>Step 3 — Pipeline Execution</div>", unsafe_allow_html=True)

st.markdown(f"""
<div class='card'>
    <div class='pipe-row'>
        <div class='pipe-node done'>
            <div class='pipe-node-title'>Query</div>
            <div class='pipe-node-sub'>Intent: {intent_label}</div>
        </div>
        <div class='pipe-arrow'>›</div>
        <div class='pipe-node done'>
            <div class='pipe-node-title'>Retrieval</div>
            <div class='pipe-node-sub'>FAISS + BM25</div>
        </div>
        <div class='pipe-arrow'>›</div>
        <div class='pipe-node done'>
            <div class='pipe-node-title'>Rerank</div>
            <div class='pipe-node-sub'>{num_retrieved} chunks</div>
        </div>
        <div class='pipe-arrow'>›</div>
        <div class='pipe-node active'>
            <div class='pipe-node-title'>Compression</div>
            <div class='pipe-node-sub'>{num_sentences} sentences</div>
        </div>
        <div class='pipe-arrow'>›</div>
        <div class='pipe-node done'>
            <div class='pipe-node-title'>Answer</div>
            <div class='pipe-node-sub'>Gemini LLM</div>
        </div>
    </div>
    <div style='margin-top:16px;padding-top:14px;border-top:1px solid #f3f4f6;'>
        <span style='font-size:0.82rem;color:#6b7280;'><b>Query:</b> {query}</span>&nbsp;&nbsp;
        <span class='badge badge-blue'>Intent: {intent_label} ({intent_conf:.0%})</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:1px solid #e8eaed;margin:24px 0;'>", unsafe_allow_html=True)

# ── SECTION 5: Answer comparison ──────────────────────────────────────────────
st.markdown("<div class='sec-label'>Step 4 — Answer Comparison</div>", unsafe_allow_html=True)

ans1, ans2 = st.columns(2, gap="large")
with ans1:
    st.markdown(f"""
    <div class='answer-box baseline'>
        <div class='answer-label bl'>Baseline RAG</div>
        {baseline_result.get('answer', 'No answer generated.')}
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"Tokens: **{baseline_tokens}** · Dense retrieval only")

with ans2:
    st.markdown(f"""
    <div class='answer-box adaptive'>
        <div class='answer-label ad'>Adaptive RAG</div>
        {adaptive_result.get('answer', 'No answer generated.')}
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"Tokens: **{adaptive_tokens}** · Intent: **{intent_label}** · Sentences: **{num_sentences}**")

st.markdown("<hr style='border:none;border-top:1px solid #e8eaed;margin:24px 0;'>", unsafe_allow_html=True)

# ── SECTION 6: Token comparison visual ────────────────────────────────────────
st.markdown("<div class='sec-label'>Step 5 — Token Reduction</div>", unsafe_allow_html=True)

tok_col1, tok_col2 = st.columns([2, 1], gap="large")
with tok_col1:
    adaptive_bar_w = max(4, 100 - reduction_pct)
    st.markdown(f"""
    <div class='card'>
        <div class='card-label'>Token Usage Comparison</div>
        <div class='bar-wrap'>
            <div class='bar-header'><span>Baseline RAG</span><span>{baseline_tokens} tokens</span></div>
            <div class='bar-bg'><div class='bar-fill-bl' style='width:100%;'></div></div>
        </div>
        <div class='bar-wrap'>
            <div class='bar-header'><span>Adaptive RAG</span><span>{adaptive_tokens} tokens</span></div>
            <div class='bar-bg'><div class='bar-fill-ad' style='width:{adaptive_bar_w}%;'></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with tok_col2:
    saved = baseline_tokens - adaptive_tokens
    st.markdown(f"""
    <div class='card' style='text-align:center;'>
        <div class='card-label'>Reduction</div>
        <div style='font-size:3rem;font-weight:800;color:#16a34a;line-height:1;'>{reduction_pct}%</div>
        <div style='font-size:0.8rem;color:#6b7280;margin-top:8px;'>{saved} tokens saved per query</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:1px solid #e8eaed;margin:24px 0;'>", unsafe_allow_html=True)

# ── SECTION 7: Evidence & Context ─────────────────────────────────────────────
st.markdown("<div class='sec-label'>Step 6 — Evidence & Context</div>", unsafe_allow_html=True)

tab_ev, tab_ctx_ad, tab_ctx_bl = st.tabs(["Selected Evidence", "Adaptive Context", "Baseline Context"])

with tab_ev:
    if selected_ev:
        for ev in selected_ev[:8]:
            score = ev.get("score", 0.0)
            page  = ev.get("page", "–")
            sec   = ev.get("section", "–")
            text  = ev.get("sentence", "")
            st.markdown(f"""
            <div class='ev-item'>
                <div class='ev-meta'>Page {page} &nbsp;·&nbsp; {sec} &nbsp;·&nbsp; Score: {score:.3f}</div>
                {text}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Compression relied on high-confidence protected chunks. No additional sentence-level evidence was required.")

with tab_ctx_ad:
    adaptive_ctx = adaptive_result.get("compressed_context", "")
    st.caption(f"~{adaptive_tokens} estimated tokens")
    st.markdown(f"<div class='ctx-box'>{adaptive_ctx}</div>", unsafe_allow_html=True)

with tab_ctx_bl:
    baseline_ctx = baseline_result.get("context", "")
    st.caption(f"~{baseline_tokens} estimated tokens")
    st.markdown(f"<div class='ctx-box'>{baseline_ctx}</div>", unsafe_allow_html=True)

# ── Debug ──────────────────────────────────────────────────────────────────────
if debug:
    st.markdown("<hr style='border:none;border-top:1px solid #e8eaed;margin:24px 0;'>", unsafe_allow_html=True)
    with st.expander("Adaptive RAG — full result payload"):
        st.json({k: v for k, v in adaptive_result.items() if k not in ("compressed_context", "selected_evidence")})
    with st.expander("Baseline RAG — full result payload"):
        st.json({k: v for k, v in baseline_result.items() if k not in ("context",)})
    with st.expander("Intent detection breakdown"):
        st.json(intent_info)
    with st.expander("All evidence (full scoring)"):
        st.json(selected_ev)
