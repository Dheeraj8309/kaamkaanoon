# src/guardrails.py
# This file implements 2-layer protection against wrong answers.
# Layer 1 — Topic Classifier: Is this question about Indian labour law?
# Layer 2 — Similarity Threshold: Do our documents actually contain the answer?
# Only if BOTH layers pass does the full RAG pipeline run.
# This is what separates a professional RAG system from a basic chatbot.

import os

# Suppress warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Minimum relevance score to consider a chunk as a valid answer
# If the best chunk scores below this, we don't answer
# 0.30 means "at least 30% similar" — low enough to be flexible,
# high enough to avoid completely unrelated answers
SIMILARITY_THRESHOLD = 0.30

# The Groq model we use for the classifier
# We use the same model as our main QA chain for consistency
GROQ_MODEL = "llama-3.3-70b-versatile"

# Friendly message shown when question is out of scope (Layer 1 fails)
OUT_OF_SCOPE_MESSAGE = """I'm KaamKaanoon, an assistant specialized in Indian labour law.

I can only answer questions about:
- Notice periods and termination rules
- Provident Fund (EPF) and gratuity
- Maternity leave and social security benefits
- Workplace safety and working conditions
- Sexual harassment (POSH Act)
- Wages, salary deductions, and overtime

Your question appears to be outside my area of expertise. Please ask me something related to Indian employee rights and labour law."""

# Friendly message shown when our documents don't contain a good answer (Layer 2 fails)
LOW_CONFIDENCE_MESSAGE = """I found some information in my documents, but I'm not confident it directly answers your question.

This could mean:
- Your question is very specific and may not be covered in the laws I have
- The answer may require interpretation by a legal professional

I have the following Indian labour law documents:
- Labour Code on Wages 2019
- Industrial Relations Code 2020
- Code on Social Security 2020
- OSH Code 2020
- EPF Act 1952
- POSH Act 2013

Please try rephrasing your question, or consult a qualified labour law attorney for specific legal advice."""


def check_topic(query: str) -> dict:
    """
    LAYER 1 — Topic Classifier.
    Asks Groq LLM to classify if the question is about Indian labour law.
    Returns a dictionary with:
        - passed: True if question is about Indian labour law, False otherwise
        - reason: explanation of the decision
    """
    # Initialize the Groq client with our API key from .env
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # This is our classifier prompt — very strict and specific
    # We ask for YES or NO only — no other words
    # This makes parsing the response simple and reliable
    classifier_prompt = f"""You are a strict topic classifier for an Indian labour law assistant.

Your job is to determine if the user's question is related to ANY of these topics:
- Indian labour laws and regulations
- Employee rights in India
- Notice period, resignation, termination, retrenchment
- Provident Fund (EPF, PF), gratuity, pension
- Maternity leave, paternity leave, sick leave
- Wages, salary, overtime, minimum wage, salary deductions
- Workplace safety and health
- Sexual harassment at workplace (POSH)
- Social security benefits for Indian workers
- Industrial disputes and worker unions
- Working hours, rest periods, holidays

Respond with ONLY one word — either YES or NO.
YES = the question is related to Indian labour law
NO = the question is not related to Indian labour law

User question: {query}

Your response (YES or NO only):"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": classifier_prompt
                }
            ],
            # max_tokens=5 forces the model to give a very short answer
            # This prevents it from giving long explanations instead of YES/NO
            max_tokens=5,
            temperature=0,  # temperature=0 means deterministic — same input always gives same output
        )

        # Extract the model's response and clean it up
        answer = response.choices[0].message.content.strip().upper()

        # Check if the answer contains YES
        if "YES" in answer:
            return {
                "passed": True,
                "reason": "Question is related to Indian labour law"
            }
        else:
            return {
                "passed": False,
                "reason": f"Question is not related to Indian labour law (classifier said: {answer})"
            }

    except Exception as e:
        # If the API call fails for any reason, we FAIL OPEN
        # Meaning we let the question through rather than blocking it
        # This prevents the system from being completely unusable if Groq has issues
        return {
            "passed": True,
            "reason": f"Classifier failed ({e}) — failing open to allow question through"
        }


def check_similarity(results: list) -> dict:
    """
    LAYER 2 — Similarity Threshold Check.
    Checks if the best retrieved chunk is relevant enough to answer from.
    Takes the formatted results list from retriever.retrieve().
    Returns a dictionary with:
        - passed: True if best score is above threshold, False otherwise
        - best_score: the highest relevance score found
        - reason: explanation of the decision
    """
    # If no results came back at all, definitely fail
    if not results:
        return {
            "passed": False,
            "best_score": 0.0,
            "reason": "No chunks retrieved from ChromaDB"
        }

    # The results are already sorted by relevance — first result is best
    best_score = results[0]["score"]

    if best_score >= SIMILARITY_THRESHOLD:
        return {
            "passed": True,
            "best_score": best_score,
            "reason": f"Best score {best_score} is above threshold {SIMILARITY_THRESHOLD}"
        }
    else:
        return {
            "passed": False,
            "best_score": best_score,
            "reason": f"Best score {best_score} is below threshold {SIMILARITY_THRESHOLD}"
        }


def run_guardrails(query: str, results: list) -> dict:
    """
    Runs BOTH guardrail layers in sequence.
    This is the main function that chain.py will call.

    Returns a dictionary with:
        - passed: True if both layers passed, False if either failed
        - message: the rejection message to show user (only if passed=False)
        - layer1: result from topic classifier
        - layer2: result from similarity check
    """
    # ── LAYER 1 — Topic Check ──
    layer1 = check_topic(query)

    # If Layer 1 fails, return immediately — don't even check Layer 2
    if not layer1["passed"]:
        return {
            "passed": False,
            "message": OUT_OF_SCOPE_MESSAGE,
            "layer1": layer1,
            "layer2": None  # Layer 2 was never run
        }

    # ── LAYER 2 — Similarity Check ──
    layer2 = check_similarity(results)

    # If Layer 2 fails, return low confidence message
    if not layer2["passed"]:
        return {
            "passed": False,
            "message": LOW_CONFIDENCE_MESSAGE,
            "layer1": layer1,
            "layer2": layer2
        }

    # Both layers passed — full RAG pipeline can run
    return {
        "passed": True,
        "message": None,  # No rejection message needed
        "layer1": layer1,
        "layer2": layer2
    }


# ─────────────────────────────────────────────
# MAIN — test both guardrail layers
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # We need the retriever to test Layer 2
    # We import it here instead of top of file to avoid circular imports
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.retriever import retrieve

    print("=" * 60)
    print("  KaamKaanoon — Guardrails Test")
    print("  Testing both layers of protection")
    print("=" * 60)

    # Test cases — covering all scenarios
    test_cases = [
        {
            "query": "what is the notice period for resignation in India?",
            "expected": "PASS",
            "description": "Valid labour law question — should pass both layers"
        },
        {
            "query": "what is the recipe for biryani?",
            "expected": "FAIL Layer 1",
            "description": "Completely unrelated question — should fail Layer 1"
        },
        {
            "query": "what are the cricket rules in IPL?",
            "expected": "FAIL Layer 1",
            "description": "Sports question — should fail Layer 1"
        },
        {
            "query": "what is the maternity leave entitlement?",
            "expected": "PASS",
            "description": "Valid labour law question — should pass both layers"
        },
        {
            "query": "what is the penalty for murder under Indian law?",
            "expected": "FAIL Layer 1",
            "description": "Criminal law question — not labour law, should fail Layer 1"
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─'*50}")
        print(f"Test {i}: {test['description']}")
        print(f"Query   : {test['query']}")
        print(f"Expected: {test['expected']}")

        # Get retrieval results for Layer 2
        results = retrieve(test["query"], top_k=5)

        # Run both guardrail layers
        outcome = run_guardrails(test["query"], results)

        # Display results
        print(f"Layer 1 : {'✓ PASSED' if outcome['layer1']['passed'] else '✗ FAILED'} — {outcome['layer1']['reason']}")

        if outcome["layer2"] is not None:
            print(f"Layer 2 : {'✓ PASSED' if outcome['layer2']['passed'] else '✗ FAILED'} — {outcome['layer2']['reason']}")
        else:
            print(f"Layer 2 : ⏭ SKIPPED (Layer 1 already failed)")

        print(f"Result  : {'✅ ALLOWED' if outcome['passed'] else '🚫 BLOCKED'}")

    print("\n" + "=" * 60)
    print("  ✅ Guardrails test complete!")
    print("=" * 60)