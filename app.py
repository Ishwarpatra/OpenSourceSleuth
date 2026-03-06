"""
SourceSleuth Web UI — Streamlit Frontend.

A premium, modern web interface for searching orphaned quotes across
your academic PDFs using local semantic search.
"""

import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

from src.config import PDF_DIR, DATA_DIR, EMBEDDING_MODEL, TOP_K, MIN_SCORE
from src.vector_store import VectorStore
from src.pdf_processor import process_pdf_directory, extract_text_from_pdf

# --- Page Configuration ---
st.set_page_config(
    page_title="SourceSleuth | Citation Recovery",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Premium CSS Styling ---
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    * { font-family: 'Inter', sans-serif; }

    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
        color: #e0e0e0;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }

    [data-testid="stSidebar"] * {
        color: #c5c5d2 !important;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Hero title */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #8b8b9e;
        font-weight: 400;
        margin-bottom: 2rem;
        letter-spacing: 0.01em;
    }

    /* Stat cards */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }

    .stat-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(99, 102, 241, 0.12);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .stat-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
        background: rgba(99, 102, 241, 0.06);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.08);
    }

    .stat-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }

    .stat-label {
        font-size: 0.8rem;
        color: #6b6b80;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.4rem;
    }

    /* Citation result card */
    .result-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(99, 102, 241, 0.12);
        border-left: 4px solid;
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .result-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        background: rgba(99, 102, 241, 0.04);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.06);
    }

    .result-card.high { border-left-color: #10b981; }
    .result-card.medium { border-left-color: #f59e0b; }
    .result-card.low { border-left-color: #ef4444; }

    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
    }

    .result-filename {
        font-size: 1.1rem;
        font-weight: 600;
        color: #c5c5e0;
    }

    .result-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-high { background: rgba(16, 185, 129, 0.12); color: #10b981; }
    .badge-medium { background: rgba(245, 158, 11, 0.12); color: #f59e0b; }
    .badge-low { background: rgba(239, 68, 68, 0.12); color: #ef4444; }

    .result-context {
        font-size: 0.92rem;
        color: #9b9bb0;
        line-height: 1.7;
        font-style: italic;
        padding: 1rem;
        background: rgba(0, 0, 0, 0.15);
        border-radius: 10px;
        margin: 0.8rem 0;
        border-left: 3px solid rgba(99, 102, 241, 0.2);
    }

    .result-meta {
        display: flex;
        gap: 1.5rem;
        font-size: 0.82rem;
        color: #6b6b80;
        margin-top: 0.6rem;
    }

    .result-meta span {
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }

    /* Search input area */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 12px !important;
        color: #e0e0e0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
        transition: border-color 0.3s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: rgba(99, 102, 241, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.08) !important;
    }

    .stTextArea textarea::placeholder {
        color: #4a4a5e !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.8rem 2rem !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35) !important;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #d0d0e0;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(99, 102, 241, 0.1);
    }

    /* Examples styling */
    .example-card {
        background: rgba(99, 102, 241, 0.04);
        border: 1px dashed rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .example-item {
        color: #8b8ba5;
        font-style: italic;
        padding: 0.5rem 0;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #4a4a5e;
        font-size: 0.8rem;
        padding: 2rem 0;
        border-top: 1px solid rgba(99, 102, 241, 0.08);
        margin-top: 3rem;
    }

    .app-footer a {
        color: #667eea;
        text-decoration: none;
    }

    .app-footer a:hover {
        text-decoration: underline;
    }

    /* Sidebar specific */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(99, 102, 241, 0.08) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        box-shadow: none !important;
        font-size: 0.85rem !important;
        padding: 0.6rem 1rem !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.3) !important;
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        margin-top: 0.5rem;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border-color: rgba(99, 102, 241, 0.15) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 8px !important;
    }

    /* Dividers */
    hr {
        border-color: rgba(99, 102, 241, 0.08) !important;
    }

    /* Success/Warning/Error messages */
    .stAlert {
        border-radius: 12px !important;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Animated glow on search */
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
        50% { box-shadow: 0 0 20px 5px rgba(99, 102, 241, 0.1); }
    }

    .search-active {
        animation: pulse-glow 2s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
@st.cache_resource
def get_vector_store():
    """Load or initialize the vector store."""
    store = VectorStore(model_name=EMBEDDING_MODEL, data_dir=DATA_DIR)
    store.load()
    return store


def format_confidence(score):
    """Format confidence score with tier."""
    if score >= 0.75:
        return "high", f"High ({score:.3f})"
    elif score >= 0.50:
        return "medium", f"Medium ({score:.3f})"
    else:
        return "low", f"Low ({score:.3f})"


# --- Sidebar ---
with st.sidebar:
    st.markdown("### Settings")
    st.caption("Configure search parameters")

    st.divider()

    # Search settings
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=TOP_K)
    min_score = st.slider(
        "Minimum similarity",
        min_value=0.0, max_value=1.0, value=MIN_SCORE, step=0.05,
    )

    st.divider()

    # File upload section
    st.markdown("### Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload academic PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF files to add to the search index",
    )

    if uploaded_files:
        if st.button("Process Uploaded PDFs", use_container_width=True):
            with st.spinner("Processing PDFs ..."):
                temp_dir = tempfile.mkdtemp()
                saved_paths = []

                for uploaded_file in uploaded_files:
                    save_path = Path(temp_dir) / uploaded_file.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_paths.append(save_path)

                chunks = []
                for pdf_path in saved_paths:
                    try:
                        doc = extract_text_from_pdf(pdf_path)
                        from src.pdf_processor import chunk_text
                        doc_chunks = chunk_text(doc)
                        chunks.extend(doc_chunks)
                    except Exception as e:
                        st.error(f"Failed to process {pdf_path.name}: {e}")

                if chunks:
                    store = get_vector_store()
                    added = store.add_chunks(chunks)
                    store.save()
                    st.success(f"Added {added} chunks from {len(uploaded_files)} PDF(s)")
                else:
                    st.warning("No text could be extracted from uploaded PDFs.")

    st.divider()

    # Maintenance section
    st.markdown("### Maintenance")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    with col2:
        if st.button("Clear Index", use_container_width=True):
            store = get_vector_store()
            store.clear()
            index_path = DATA_DIR / "sourcesleuth.index"
            meta_path = DATA_DIR / "sourcesleuth_metadata.json"
            if index_path.exists():
                index_path.unlink()
            if meta_path.exists():
                meta_path.unlink()
            st.cache_resource.clear()
            st.success("Index cleared!")
            st.rerun()

    st.divider()

    st.markdown("### About")
    st.markdown("""
    **SourceSleuth** recovers citations for orphaned quotes using local semantic search.

    - All data stays on your machine
    - No API keys required
    - Powered by FAISS + SentenceTransformers
    - Licensed under Apache 2.0
    """)

    st.caption("v1.0.0 | Apache 2.0")


# --- Main UI ---
st.markdown('<div class="hero-title">SourceSleuth</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Recover citations for orphaned quotes using local semantic search — powered by MCP'
    '</div>',
    unsafe_allow_html=True,
)

# Load vector store and get stats
store = get_vector_store()
stats = store.get_stats()

# Display stats
st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card">
        <div class="stat-value">{stats['total_chunks']:,}</div>
        <div class="stat-label">Total Chunks</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{stats['num_files']}</div>
        <div class="stat-label">Documents</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{stats['embedding_dim']}</div>
        <div class="stat-label">Dimensions</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">CPU</div>
        <div class="stat-label">Runtime</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Input Area
st.markdown(
    '<div class="section-header">Search for Orphaned Quote</div>',
    unsafe_allow_html=True,
)

query = st.text_area(
    "Paste the quote or paraphrase you want to find the source for:",
    placeholder=(
        "e.g., 'The attention mechanism allows models to focus on "
        "specific parts of the input sequence...'"
    ),
    height=120,
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    search_clicked = st.button("Find Sources", use_container_width=True)

# --- Results Section ---
if search_clicked and query:
    st.divider()
    st.markdown(
        '<div class="section-header">Search Results</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Searching your documents ..."):
        results = store.search(query=query, top_k=top_k)

    if not results:
        st.warning(
            "No matching sources found. Try uploading more PDFs "
            "or adjusting your search query."
        )
    else:
        filtered_results = [r for r in results if r["score"] >= min_score]

        if not filtered_results:
            st.warning(
                f"No results above the minimum score threshold ({min_score}). "
                "Try lowering the threshold in the sidebar."
            )
        else:
            st.success(f"Found {len(filtered_results)} potential match(es)!")

            for i, result in enumerate(filtered_results, start=1):
                tier, badge_text = format_confidence(result["score"])
                context_preview = result["text"][:400].replace("\n", " ")
                if len(result["text"]) > 400:
                    context_preview += " ..."

                st.markdown(f"""
                <div class="result-card {tier}">
                    <div class="result-header">
                        <span class="result-filename">
                            #{i} &mdash; {result['filename']}
                        </span>
                        <span class="result-badge badge-{tier}">{badge_text}</span>
                    </div>
                    <div class="result-context">
                        "{context_preview}"
                    </div>
                    <div class="result-meta">
                        <span>Page {result['page']}</span>
                        <span>Chunk #{result.get('chunk_index', 'N/A')}</span>
                        <span>Chars {result.get('start_char', '?')}&ndash;{result.get('end_char', '?')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"View full context for result #{i}"):
                    st.text(result["text"])

elif search_clicked and not query:
    st.warning("Please enter a quote to search for.")

else:
    # Example queries
    st.markdown("""
    <div class="example-card">
        <div class="section-header" style="border: none; margin-bottom: 0.5rem;">
            Example Queries
        </div>
        <div class="example-item">
            "Attention is all you need for sequence transduction"
        </div>
        <div class="example-item">
            "Wave interference produces a pattern of bright and dark fringes"
        </div>
        <div class="example-item">
            "The photoelectric effect demonstrates the particle nature of light"
        </div>
        <div class="example-item">
            "Transformers have replaced recurrent models in most NLP tasks"
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Document List ---
if stats["num_files"] > 0:
    st.divider()
    st.markdown(
        '<div class="section-header">Indexed Documents</div>',
        unsafe_allow_html=True,
    )

    df = pd.DataFrame({
        "Filename": stats["ingested_files"],
        "Status": ["Indexed"] * len(stats["ingested_files"]),
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

# --- Footer ---
st.markdown("""
<div class="app-footer">
    Powered by <strong>Model Context Protocol (MCP)</strong>
    &nbsp;&bull;&nbsp;
    <a href="https://github.com/Ishwarpatra/OpenSourceSleuth" target="_blank">GitHub</a>
    &nbsp;&bull;&nbsp;
    Apache 2.0
</div>
""", unsafe_allow_html=True)
