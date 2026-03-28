"""Adaptive RAG System — Research Demo Dashboard"""

import hashlib
import os
import tempfile

import pandas as pd
import streamlit as st

from evaluation.adaptive_rag import run_adaptive_rag
from evaluation.baseline_rag import run_baseline_rag
from src.indexing.chunker import chunk_documents
from src.indexing.embedder import generate_embeddings
from src.indexing.vector_store import build_faiss_index
from src.ingestion.pdf_extractor import extract_text_from_pdf
from src.ingestion.section_detector import detect_sections

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive RAG Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Minimal custom CSS — clean dark-card aesthetic
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Global font */
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    /* Metric card accent */
    [data-testid="stMetric"] {
        background: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 14px 18px;
    }
    [data-testid="stMetricLabel"]  { font-size: 0.75rem; color: #9fa8c7; }
    [data-testid="stMetricValue"]  { font-size: 1.6rem; font-weight: 700; }

    /* Column answer cards */
    .answer-card {
        background: #161926;
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 20px 22px;
        min-height: 160px;
        margin-bottom: 8px;
    }
    .baseline-header  { color: #7b8cde; font-size: 1.15rem; font-weight: 700; }
    .adaptive-header  { color: #43d17a; font-size: 1.15rem; font-weight: 700; }

    /* Divider */
    hr { border: 0; border-top: 1px solid #2e3250; margin: 20px 0; }

    /* Intent badge */
    .intent-badge {
        display: inline-block;
        background: #2a3b5e;
        color: #7ab3ff;
        border-radius: 20px;
        padding: 3px 14px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    /* Sentence table */
    .sentence-row   { background: #1a1f33; border-radius: 8px; padding: 8px 12px; margin: 4px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("## 🔬 Adaptive RAG System")
st.markdown(
    "<span style='color:#9fa8c7;font-size:1rem;'>"
    "Hybrid Retrieval &nbsp;·&nbsp; Cross-Encoder Reranking &nbsp;·&nbsp; Intent-Aware Context Compression"
    "</span>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR — debug toggle + info
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    debug = st.checkbox("Show Debug Info", value=False)
    st.markdown("---")
    st.markdown("**Pipeline stages**")
    st.markdown(
        "1. PDF Extraction  \n"
        "2. Section Detection  \n"
        "3. Chunking (850 chars)  \n"
        "4. BGE-Large Embedding  \n"
        "5. FAISS IndexFlatIP  \n"
        "---  \n"
        "**At query time**  \n"
        "6. Intent Detection  \n"
        "7. Dense + BM25 Retrieval  \n"
        "8. Cross-Encoder Rerank  \n"
        "9. Knapsack Compression  \n"
        "10. LLM Answer Generation"
    )

# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
col_up, col_q, col_btn = st.columns([2, 3, 1])

with col_up:
    uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file:
        st.caption(f"📎 `{uploaded_file.name}`")

with col_q:
    query = st.text_input("💬 Enter your question", placeholder="e.g. What accuracy did the model achieve?")

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("▶ Run Query", use_container_width=True, type="primary")

if not uploaded_file:
    st.info("⬆️ Upload a PDF to get started.")
    st.stop()

# ─────────────────────────────────────────────
# INDEXING — cached per PDF content
# ─────────────────────────────────────────────
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


pdf_bytes = uploaded_file.getvalue()
pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()[:12]

with st.spinner("🔄 Indexing document… (cached after first run)"):
    faiss_index, metadata_list, num_pages, num_chunks = build_index_for_pdf(pdf_bytes)

# ─────────────────────────────────────────────
# STATUS PANEL
# ─────────────────────────────────────────────
st.markdown("### 📊 System Status")
s1, s2, s3, s4 = st.columns(4)
s1.success(f"✅ **{num_pages}** pages extracted")
s2.success(f"✅ **{num_chunks}** chunks indexed")
s3.info("🧠 Embedding: **BAAI/bge-large-en**")
s4.info("🔍 Retrieval: **Dense + BM25 Hybrid**")

st.markdown("---")

# ─────────────────────────────────────────────
# QUERY EXECUTION
# ─────────────────────────────────────────────
if not run_button:
    st.markdown(
        "<div style='text-align:center;color:#555;padding:40px 0;font-size:1rem;'>"
        "Enter a question and click <b>▶ Run Query</b> to see the comparison."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

if not query or not query.strip():
    st.warning("⚠️ Please enter a question before running.")
    st.stop()

with st.spinner("⚙️ Running Baseline and Adaptive pipelines…"):
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

# ─── Derived values ───────────────────────────
baseline_tokens  = baseline_result.get("token_count", 0)
adaptive_tokens  = adaptive_result.get("tokens_used", 0)
reduction_pct    = (
    round((1 - adaptive_tokens / baseline_tokens) * 100, 1)
    if baseline_tokens > 0 else 0.0
)
intent_info      = adaptive_result.get("intent", {})
intent_label     = intent_info.get("intent", "UNKNOWN")
intent_conf      = intent_info.get("confidence", 0.0)
num_retrieved    = len(baseline_result.get("retrieved_chunks", []))
num_sentences    = adaptive_result.get("num_sentences", 0)
selected_ev      = adaptive_result.get("selected_evidence", [])

# ─────────────────────────────────────────────
# INTENT DETECTION BANNER
# ─────────────────────────────────────────────
st.markdown("### 🧠 Detected Query Intent")
intent_colors = {
    "RESULT":     "#43d17a",
    "METHOD":     "#7ab3ff",
    "API_USAGE":  "#f5a623",
    "DEFINITION": "#c77dff",
    "COMPARISON": "#ff6b6b",
}
badge_color = intent_colors.get(intent_label, "#9fa8c7")
st.markdown(
    f"<span class='intent-badge' style='background:#1e2130;border:1px solid {badge_color};"
    f"color:{badge_color};font-size:1rem;padding:5px 18px;'>"
    f"🎯 {intent_label}</span>"
    f"&nbsp;&nbsp;<span style='color:#9fa8c7;font-size:0.9rem;'>"
    f"Confidence: <b>{intent_conf:.0%}</b></span>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────
st.markdown("### 📈 Performance Metrics")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Baseline Tokens",   baseline_tokens, help="Estimated tokens sent to LLM by baseline RAG")
m2.metric("Adaptive Tokens",   adaptive_tokens, help="Estimated tokens sent to LLM by adaptive RAG")
m3.metric(
    "Token Reduction",
    f"{reduction_pct}%",
    delta=f"-{reduction_pct}%" if reduction_pct > 0 else "0%",
    delta_color="inverse",
    help="Percentage fewer tokens used by adaptive vs baseline",
)
m4.metric("Retrieved Chunks",   num_retrieved, help="Dense + BM25 hybrid recall pool (baseline top-k)")
m5.metric("Selected Sentences", num_sentences, help="Evidence sentences chosen by knapsack compression")

st.markdown("---")

# ─────────────────────────────────────────────
# TOKEN COMPARISON BAR CHART
# ─────────────────────────────────────────────
st.markdown("### 📊 Token Usage Comparison")
chart_df = pd.DataFrame(
    {"Tokens": [baseline_tokens, adaptive_tokens]},
    index=["Baseline RAG", "Adaptive RAG"],
)
st.bar_chart(chart_df, color="#4e8cff", height=220)

st.markdown("---")

# ─────────────────────────────────────────────
# SIDE-BY-SIDE ANSWER DISPLAY
# ─────────────────────────────────────────────
st.markdown("### 💬 Answer Comparison")
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        "<div class='answer-card'>"
        "<p class='baseline-header'>📄 Baseline RAG</p>"
        f"<p style='color:#cdd3f0;line-height:1.6;'>{baseline_result.get('answer','')}</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"🔢 Tokens: **{baseline_tokens}** &nbsp;|&nbsp; "
        f"📚 Retrieval: Dense-only (top-{num_retrieved})"
    )

with col2:
    st.markdown(
        "<div class='answer-card' style='border-color:#2d5a3d;'>"
        "<p class='adaptive-header'>🚀 Adaptive RAG</p>"
        f"<p style='color:#cdd3f0;line-height:1.6;'>{adaptive_result.get('answer','')}</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"🔢 Tokens: **{adaptive_tokens}** &nbsp;|&nbsp; "
        f"🎯 Intent: **{intent_label}** &nbsp;|&nbsp; "
        f"📝 Sentences: **{num_sentences}**"
    )

st.markdown("---")

# ─────────────────────────────────────────────
# SENTENCE SELECTION INSIGHTS
# ─────────────────────────────────────────────
st.markdown("### 🧩 Top Selected Evidence Sentences")
if selected_ev:
    top_ev = selected_ev[:8]
    ev_df = pd.DataFrame(
        [
            {
                "Sentence": ev.get("sentence", "")[:120] + ("…" if len(ev.get("sentence","")) > 120 else ""),
                "Score":    round(ev.get("score", 0.0), 3),
                "Page":     ev.get("page", "–"),
                "Section":  ev.get("section", "–"),
            }
            for ev in top_ev
        ]
    )
    st.dataframe(
        ev_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0.0, max_value=1.0, format="%.3f"
            )
        },
    )
else:
    st.info("No sentence-level evidence captured (context may be dominated by protected chunks).")

st.markdown("---")

# ─────────────────────────────────────────────
# CONTEXT VISUALIZATION
# ─────────────────────────────────────────────
st.markdown("### 🗂️ Context Visualization")
ctx_tab1, ctx_tab2 = st.tabs(["🚀 Adaptive (Compressed)", "📄 Baseline (Full)"])

with ctx_tab1:
    adaptive_ctx = adaptive_result.get("compressed_context", "")
    st.markdown(
        f"<span style='color:#9fa8c7;font-size:0.8rem;'>~{adaptive_tokens} tokens</span>",
        unsafe_allow_html=True,
    )
    st.code(adaptive_ctx[:3000] + ("\n…[truncated]" if len(adaptive_ctx) > 3000 else ""), language="text")

with ctx_tab2:
    baseline_ctx = baseline_result.get("context", "")
    st.markdown(
        f"<span style='color:#9fa8c7;font-size:0.8rem;'>~{baseline_tokens} tokens</span>",
        unsafe_allow_html=True,
    )
    st.code(baseline_ctx[:3000] + ("\n…[truncated]" if len(baseline_ctx) > 3000 else ""), language="text")

st.markdown("---")

# ─────────────────────────────────────────────
# PIPELINE VISUALIZATION (EXPANDERS)
# ─────────────────────────────────────────────
st.markdown("### 🔍 Pipeline Internals")

with st.expander("📚 Baseline Retrieved Chunks"):
    b_chunks = baseline_result.get("retrieved_chunks", [])
    if b_chunks:
        for c in b_chunks:
            st.markdown(
                f"**Rank {c.get('rank','-')}** &nbsp;·&nbsp; "
                f"Page {c.get('page','-')} &nbsp;·&nbsp; "
                f"Section: `{c.get('section','-')}`"
            )
            st.markdown(
                f"<div style='color:#aab0cc;font-size:0.85rem;padding-left:12px;'>"
                f"{c.get('text','')[:300]}{'…' if len(c.get('text',''))>300 else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("---")
    else:
        st.info("No chunks available.")

with st.expander("🧪 Adaptive Evidence Sentences (Scored)"):
    if selected_ev:
        for i, ev in enumerate(selected_ev, 1):
            score = ev.get("score", 0.0)
            bar = "█" * min(int(score * 20), 20)
            st.markdown(
                f"**#{i}** &nbsp; Score: `{score:.3f}` &nbsp; `{bar}` &nbsp;·&nbsp; "
                f"Page {ev.get('page','-')} &nbsp;·&nbsp; `{ev.get('section','-')}`  \n"
                f"<span style='color:#aab0cc;font-size:0.85rem;'>{ev.get('sentence','')}</span>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No evidence sentences selected.")

# ─────────────────────────────────────────────
# DEBUG MODE
# ─────────────────────────────────────────────
if debug:
    st.markdown("---")
    st.markdown("### 🐛 Debug Information")

    with st.expander("Baseline RAG — Raw Result Dict"):
        debug_b = {k: v for k, v in baseline_result.items() if k not in ("context",)}
        st.json(debug_b)

    with st.expander("Adaptive RAG — Raw Result Dict"):
        debug_a = {
            k: v
            for k, v in adaptive_result.items()
            if k not in ("compressed_context", "selected_evidence")
        }
        st.json(debug_a)

    with st.expander("Intent Detection — Full Breakdown"):
        st.json(intent_info)

    with st.expander("All Selected Evidence — Full Scoring Details"):
        st.json(selected_ev)

    with st.expander("Index Metadata (first 10 chunks)"):
        st.json(metadata_list[:10])

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.8rem;padding:8px 0;'>"
    "Adaptive RAG System &nbsp;·&nbsp; "
    "Hybrid Retrieval + Intent-Aware Compression &nbsp;·&nbsp; "
    f"Document: <code>{uploaded_file.name}</code> &nbsp;·&nbsp; "
    f"SHA-256: <code>{pdf_hash}</code>"
    "</div>",
    unsafe_allow_html=True,
)
