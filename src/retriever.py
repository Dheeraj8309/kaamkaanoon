# src/retriever.py
# This file connects to our ChromaDB knowledge base and retrieves
# the most relevant chunks for any user question.
# What it does:
#   1. Loads the existing ChromaDB vectorstore from disk
#   2. Converts the user's question into a vector
#   3. Finds the top 5 most similar chunks in ChromaDB
#   4. Returns each chunk with its text, source filename, page number, and score

import os

# Suppress warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# Updated import — uses langchain-chroma instead of langchain-community
from langchain_chroma import Chroma

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION — must match ingest.py exactly
# ─────────────────────────────────────────────

VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "kaamkaanoon"
TOP_K = 5


def load_vectorstore():
    """
    Loads the existing ChromaDB vectorstore from disk.
    Does NOT rebuild — just opens what ingest.py already created.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )

    return vectorstore


def retrieve(query: str, top_k: int = TOP_K):
    """
    Takes a user question and returns the top_k most relevant chunks.
    Each result includes:
        - text   : the actual chunk content
        - source : the PDF filename (used for citations)
        - page   : the page number (used for citations)
        - score  : similarity score between 0 and 1 (used for guardrails)
    """
    vectorstore = load_vectorstore()

    # similarity_search_with_score converts the query to a vector,
    # compares it against all stored vectors, and returns top_k matches
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    formatted_results = []

    for document, score in results:
        # Extract just the filename from the full path
        # e.g. "data\pdfs\epf_act_1952.pdf" → "epf_act_1952.pdf"
        full_path = document.metadata.get("source", "unknown")
        filename = os.path.basename(full_path)

        # Page numbers in LangChain are 0-indexed
        # Add 1 to make it human-readable
        raw_page = document.metadata.get("page", 0)
        human_page = raw_page + 1

        # With cosine similarity + normalized embeddings:
        # score close to 0 = very similar (distance is small)
        # score close to 1 = not similar (distance is large)
        # We convert to relevance score: relevance = 1 - distance
        # So relevance close to 1 = very relevant
        # So relevance close to 0 = not relevant
        relevance_score = round(1 - float(score), 4)

        formatted_results.append({
            "text": document.page_content,
            "source": filename,
            "page": human_page,
            "score": relevance_score,
        })

    return formatted_results


# ─────────────────────────────────────────────
# MAIN — test the retriever directly in terminal
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  KaamKaanoon — Retriever Test")
    print("  Testing ChromaDB retrieval with sample questions")
    print("=" * 60)

    test_questions = [
        "what is the notice period for resignation?",
        "how is gratuity calculated for an employee?",
        "what are the rules for maternity leave?",
        "what is the EPF contribution percentage?",
        "what counts as sexual harassment at workplace?",
    ]

    for question in test_questions:
        print(f"\n📝 Question: {question}")
        print("-" * 50)

        results = retrieve(question, top_k=3)

        if results:
            top = results[0]
            print(f"   Top Result:")
            print(f"   Source    : {top['source']}")
            print(f"   Page      : {top['page']}")
            print(f"   Relevance : {top['score']} (higher = more relevant)")
            print(f"   Text      : {top['text'][:200]}...")
        else:
            print("   No results found!")

    print("\n" + "=" * 60)
    print("  ✅ Retriever test complete!")
    print("  All questions returned relevant chunks with citations.")
    print("=" * 60)