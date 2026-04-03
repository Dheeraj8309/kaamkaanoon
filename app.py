# app.py
# This is the main file users run to start KaamKaanoon.
# It creates a Streamlit web interface for our RAG pipeline.
# Run it with: streamlit run app.py

import os
import sys

# Suppress all warnings before any other imports
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Add project root to Python path so src imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Auto-download PDFs and build vectorstore on first startup
# This runs automatically on HuggingFace Spaces
vectorstore_empty = not os.path.exists("vectorstore") or not os.listdir("vectorstore")

if vectorstore_empty:
    print("First startup detected — downloading PDFs...")
    from src.download_pdfs import download_pdfs
    pdfs_ok = download_pdfs()

    if pdfs_ok:
        print("Building knowledge base from PDFs...")
        from src.ingest import load_pdfs, split_documents, create_embeddings, store_in_chromadb
        _docs = load_pdfs(os.path.join("data", "pdfs"))
        _chunks = split_documents(_docs)
        _embeddings = create_embeddings()
        store_in_chromadb(_chunks, _embeddings)
        print("Knowledge base built successfully!")
    else:
        print("WARNING: Some PDFs failed to download. Answers may be incomplete.")

import streamlit as st
from src.chain import ask

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# Must be the first Streamlit command in the file
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="KaamKaanoon — Indian Employee Rights Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2e6da4 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .main-header h1 { font-size: 2.5rem; font-weight: 700; margin: 0; color: white; }
    .main-header p { font-size: 1.1rem; margin: 0.5rem 0 0 0; opacity: 0.9; color: white; }
    .answer-box {
        background-color: #f8f9fa;
        border-left: 4px solid #2e6da4;
        padding: 1.5rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .source-box {
        background-color: #e8f4f8;
        border: 1px solid #2e6da4;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #1a1a1a;
    }
    .blocked-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1.5rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .question-box {
        background-color: #e8f4f8;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-weight: 500;
        color: #1a1a1a;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "is_loading" not in st.session_state:
    st.session_state.is_loading = False

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>⚖️ KaamKaanoon</h1>
    <p>Indian Employee Rights Assistant — Powered by AI & Official Government Laws</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📚 Knowledge Base")
    st.markdown("""
    This assistant answers questions based on these **official Indian government documents**:

    - 📄 Labour Code on Wages 2019
    - 📄 Industrial Relations Code 2020
    - 📄 Code on Social Security 2020
    - 📄 OSH Code 2020
    - 📄 EPF Act 1952
    - 📄 POSH Act 2013
    """)

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    st.markdown("Click any question to use it:")

    sample_questions = [
        "What is the notice period for resignation?",
        "How is gratuity calculated?",
        "What are maternity leave rules?",
        "What is the EPF contribution rate?",
        "What counts as workplace harassment?",
        "What are rules for overtime pay?",
        "Can my employer deduct my salary?",
        "What is the minimum wage in India?",
    ]

    for question in sample_questions:
        if st.button(question, key=f"sample_{question}", use_container_width=True):
            st.session_state.selected_question = question

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    ### ⚠️ Disclaimer
    This tool provides **general legal information** only.
    It is **not legal advice**.
    For specific situations, consult a qualified labour law attorney.
    """)

# ─────────────────────────────────────────────
# MAIN INPUT AREA
# ─────────────────────────────────────────────

st.markdown("### 🔍 Ask Your Question")
st.markdown("Ask anything about your rights as an Indian employee — notice period, PF, gratuity, leave, harassment, wages, and more.")

default_question = ""
if "selected_question" in st.session_state:
    default_question = st.session_state.selected_question
    del st.session_state.selected_question

user_question = st.text_area(
    label="Your Question",
    value=default_question,
    placeholder="Example: What is the notice period if I want to resign from my job?",
    height=100,
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    submit_button = st.button(
        "⚖️ Get Legal Answer",
        use_container_width=True,
        type="primary",
    )

# ─────────────────────────────────────────────
# ANSWER PROCESSING
# ─────────────────────────────────────────────

if submit_button and user_question.strip():
    with st.spinner("🔍 Searching through Indian labour laws... This may take 20-30 seconds."):
        result = ask(user_question.strip())

    st.markdown("---")
    st.markdown("### 📋 Answer")

    st.markdown(f"""
    <div class="question-box">
        ❓ <strong>Your Question:</strong> {user_question.strip()}
    </div>
    """, unsafe_allow_html=True)

    if result["blocked"]:
        st.markdown(f"""
        <div class="blocked-box">
            {result["answer"].replace(chr(10), "<br>")}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="answer-box">
            {result["answer"].replace(chr(10), "<br>")}
        </div>
        """, unsafe_allow_html=True)

        if result["sources"]:
            st.markdown("### 📚 Sources Consulted")
            for source in result["sources"]:
                display_name = source["file"].replace("_", " ").replace(".pdf", "").title()
                page_num = source.get("page", "")
                score_val = source.get("score", "")
                st.markdown(f"""
                <div class="source-box">
                    📄 <strong>{display_name}</strong> — Page {page_num}
                    &nbsp;&nbsp;|&nbsp;&nbsp; Relevance: {score_val}
                </div>
                """, unsafe_allow_html=True)

    st.session_state.chat_history.append({
        "question": user_question.strip(),
        "answer": result["answer"],
        "blocked": result["blocked"],
        "sources": result.get("sources", []),
    })

elif submit_button and not user_question.strip():
    st.warning("⚠️ Please type a question before clicking Get Legal Answer.")

# ─────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────

if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 🕐 Previous Questions")

    for i, item in enumerate(reversed(st.session_state.chat_history)):
        if i == 0:
            continue

        with st.expander(f"❓ {item['question'][:80]}..."):
            if item["blocked"]:
                st.markdown(f"""
                <div class="blocked-box">
                    {item["answer"].replace(chr(10), "<br>")}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="answer-box">
                    {item["answer"].replace(chr(10), "<br>")}
                </div>
                """, unsafe_allow_html=True)

                if item["sources"]:
                    for source in item["sources"]:
                        display_name = source["file"].replace("_", " ").replace(".pdf", "").title()
                        page_num = source.get("page", "")
                        st.markdown(f"📄 **{display_name}** — Page {page_num}")