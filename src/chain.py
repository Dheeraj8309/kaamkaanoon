# src/chain.py
# This is the brain of KaamKaanoon — the full RAG pipeline.
# What it does:
#   1. Takes a user question
#   2. Runs Layer 1 guardrail — is this about Indian labour law?
#   3. Retrieves top 5 relevant chunks from ChromaDB
#   4. Runs Layer 2 guardrail — are the chunks relevant enough?
#   5. Sends chunks + question to Groq LLM with a strict system prompt
#   6. Returns the answer with source document and page number citations
# This file is what app.py will call every time a user asks a question.

import os

# Suppress warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv
from groq import Groq

# Import our own modules
from src.retriever import retrieve
from src.guardrails import run_guardrails, OUT_OF_SCOPE_MESSAGE

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# The Groq model that writes our answers
GROQ_MODEL = "llama-3.3-70b-versatile"

# Maximum tokens in the LLM's answer
# 1024 is enough for a detailed legal answer with citations
MAX_TOKENS = 1024

# Number of chunks to retrieve and send to the LLM
TOP_K = 5

# This is the most important prompt in the entire project.
# It tells the LLM exactly how to behave.
# Every rule here is intentional — do not change without understanding why.
SYSTEM_PROMPT = """You are KaamKaanoon, a specialized AI assistant for Indian labour law.

You help Indian workers above age 20 understand their legal rights in plain, simple English.

STRICT RULES YOU MUST FOLLOW:
1. Answer ONLY using the context documents provided below. Do not use your own training knowledge.
2. Every claim you make MUST be cited with the source document name and page number.
3. Use this exact citation format: [Source: filename.pdf, Page: X]
4. If the context does not contain enough information to answer, say "I don't have enough information in my documents to answer this accurately."
5. Write in simple, plain English that any worker can understand — avoid complex legal jargon.
6. Be specific — mention exact numbers, percentages, and timeframes from the law when available.
7. Never make up information. Never guess. Only state what the documents say.
8. At the end of every answer, add a disclaimer: "This is general legal information, not legal advice. For your specific situation, consult a qualified labour law attorney."

CITATION FORMAT EXAMPLE:
"Workers are entitled to one month notice period before retrenchment [Source: industrial_relations_code_2020.pdf, Page: 33]."

Remember: Citations are mandatory. An answer without citations is incomplete."""


def format_context(results: list) -> str:
    """
    Takes the list of retrieved chunks and formats them into a single
    context string that we send to the LLM.
    Each chunk is clearly labeled with its source and page number
    so the LLM knows exactly where each piece of information came from.
    """
    context_parts = []

    for i, result in enumerate(results, 1):
        # Format each chunk with clear source labeling
        # This is what the LLM reads when forming its answer
        chunk_text = f"""--- Document {i} ---
Source: {result['source']}
Page: {result['page']}
Content: {result['text']}
"""
        context_parts.append(chunk_text)

    return "\n".join(context_parts)


def build_user_message(query: str, context: str) -> str:
    """
    Builds the complete message we send to the LLM.
    It contains both the retrieved context AND the user's question.
    The LLM must answer the question using only the context provided.
    """
    return f"""Here are the relevant sections from Indian labour law documents:

{context}

Based ONLY on the documents above, please answer this question:
{query}

Remember to cite the source document name and page number for every claim you make."""


def ask(query: str) -> dict:
    """
    The main function — takes a user question and returns a complete answer.
    This is the function app.py will call for every user question.

    Returns a dictionary with:
        - answer     : the full answer text with citations
        - sources    : list of source documents used
        - guardrails : details of guardrail decisions
        - blocked    : True if guardrails blocked this question
    """
    # Initialize Groq client
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # ── STEP 1: Layer 1 Guardrail — Topic Check ──
    # We check the topic BEFORE retrieving chunks
    # This saves time — no point searching ChromaDB for off-topic questions
    # We pass empty results for now — run_guardrails handles Layer 1 first
    initial_check = run_guardrails(query, [])

    # If Layer 1 failed, return immediately with rejection message
    if not initial_check["passed"] and initial_check["layer2"] is None:
        return {
            "answer": initial_check["message"],
            "sources": [],
            "guardrails": initial_check,
            "blocked": True
        }

    # ── STEP 2: Retrieve relevant chunks from ChromaDB ──
    results = retrieve(query, top_k=TOP_K)

    # ── STEP 3: Layer 2 Guardrail — Similarity Check ──
    # Now we check if the retrieved chunks are relevant enough
    guardrail_result = run_guardrails(query, results)

    # If Layer 2 failed, return low confidence message
    if not guardrail_result["passed"]:
        return {
            "answer": guardrail_result["message"],
            "sources": [],
            "guardrails": guardrail_result,
            "blocked": True
        }

    # ── STEP 4: Format context for the LLM ──
    # Convert our list of chunks into a single formatted string
    context = format_context(results)

    # ── STEP 5: Build the complete message for the LLM ──
    user_message = build_user_message(query, context)

    # ── STEP 6: Call Groq LLM to generate the answer ──
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                # System prompt tells the LLM its role and strict rules
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                # User message contains the context + question
                "role": "user",
                "content": user_message
            }
        ],
        max_tokens=MAX_TOKENS,
        # temperature=0.1 means mostly deterministic but slightly creative
        # Good for legal answers — consistent but not robotic
        temperature=0.1,
    )

    # Extract the answer text from Groq's response
    answer = response.choices[0].message.content.strip()

    # ── STEP 7: Extract unique sources for citation display ──
    # Build a clean list of sources used — for displaying in the UI
    sources = []
    seen = set()  # Track duplicates — same file can appear multiple times

    for result in results:
        # Create a unique key for each source+page combination
        source_key = f"{result['source']}_page_{result['page']}"

        if source_key not in seen:
            seen.add(source_key)
            sources.append({
                "file": result["source"],
                "page": result["page"],
                "score": result["score"]
            })

    return {
        "answer": answer,
        "sources": sources,
        "guardrails": guardrail_result,
        "blocked": False
    }


# ─────────────────────────────────────────────
# MAIN — test the full RAG pipeline end to end
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  KaamKaanoon — Full RAG Pipeline Test")
    print("  Testing end-to-end question answering with citations")
    print("=" * 60)

    # Test questions — one valid, one invalid
    test_questions = [
        "What is the notice period for a worker who wants to resign?",
        "What are the rules for maternity leave in India?",
        "How is gratuity calculated for an employee?",
        "What is the best smartphone to buy in India?",  # should be blocked
    ]

    for question in test_questions:
        print(f"\n{'═'*60}")
        print(f"❓ Question: {question}")
        print(f"{'═'*60}")

        # Call our main ask() function
        result = ask(question)

        if result["blocked"]:
            print(f"🚫 BLOCKED by guardrails")
            print(f"\n{result['answer']}")
        else:
            print(f"✅ ANSWERED")
            print(f"\n📋 Answer:\n{result['answer']}")
            print(f"\n📚 Sources used:")
            for source in result["sources"]:
                print(f"   • {source['file']} — Page {source['page']} (relevance: {source['score']})")

        print()

    print("=" * 60)
    print("  ✅ Full pipeline test complete!")
    print("=" * 60)