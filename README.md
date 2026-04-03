# ⚖️ KaamKaanoon — Indian Employee Rights RAG Assistant

> An AI-powered assistant that helps Indian workers understand their legal rights using official government law documents.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3.25-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.23-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-red)
![Groq](https://img.shields.io/badge/Groq-Llama3.3--70B-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 What is KaamKaanoon?

KaamKaanoon (meaning "Work Law" in Hindi) is a Retrieval-Augmented Generation (RAG) system that allows Indian workers to ask questions about their legal rights and receive plain English answers with citations from official Indian government law documents.

**No hallucinations. No guessing. Every answer cites the exact source document and page number.**

---

## 🎯 Who is it for?

Indian employees above age 20 who want to understand:
- Their notice period rights
- Provident Fund (EPF) and gratuity entitlements
- Maternity leave rules
- Workplace safety rights
- Sexual harassment protections (POSH Act)
- Wage and salary deduction rules

---

## 🏗️ Architecture
User Question
│
▼
┌─────────────────┐
│  Layer 1        │  ← Groq LLM classifier: "Is this Indian labour law?"
│  Guardrail      │  ← If NO → return rejection message immediately
└────────┬────────┘
│ YES
▼
┌─────────────────┐
│  ChromaDB       │  ← Convert question to vector, find top 5 chunks
│  Retrieval      │  ← Each chunk has source filename + page number
└────────┬────────┘
│
▼
┌─────────────────┐
│  Layer 2        │  ← Check similarity score > 0.30
│  Guardrail      │  ← If NO → return low confidence message
└────────┬────────┘
│ YES
▼
┌─────────────────┐
│  Groq LLM       │  ← Send chunks + question to Llama 3.3 70B
│  Answer Gen     │  ← LLM writes answer using ONLY the chunks
└────────┬────────┘
│
▼
Answer + Citations
[Source: filename.pdf, Page: X]

---

## 🛠️ Complete Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| RAG Orchestration | LangChain 0.3.25 | Connects all pipeline components |
| Vector Database | ChromaDB 0.5.23 | Stores and retrieves document chunks |
| Embedding Model | HuggingFace all-MiniLM-L6-v2 | Converts text to vectors, runs on CPU |
| LLM | Groq Llama 3.3 70B | Generates answers from retrieved chunks |
| Frontend | Streamlit 1.45.1 | Web UI, pure Python |
| Backend | FastAPI 0.115.12 | API layer |
| Containerization | Docker | Reproducible deployment |
| Deployment | HuggingFace Spaces | Free public hosting |
| **Total Cost** | **₹0** | Everything free tier |

---

## 📚 Knowledge Base — 6 Official Government Documents

| Document | Source | Topics Covered |
|----------|--------|---------------|
| Labour Code on Wages 2019 | labour.gov.in | Minimum wage, overtime, salary deductions |
| Industrial Relations Code 2020 | labour.gov.in | Notice period, retrenchment, unions |
| Code on Social Security 2020 | labour.gov.in | Gratuity, maternity leave, EPF, ESIC |
| OSH Code 2020 | labour.gov.in | Workplace safety, working hours, leaves |
| EPF Act 1952 | epfindia.gov.in | Provident fund contributions and rules |
| POSH Act 2013 | indiacode.nic.in | Sexual harassment prevention |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/kaamkaanoon.git
cd kaamkaanoon

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate.bat   # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your Groq API key

# 5. Download the 6 government PDFs into data/pdfs/
# See the Knowledge Base section above for download links

# 6. Build the knowledge base (run once)
python src/ingest.py

# 7. Start the app
streamlit run app.py
```

### Docker
```bash
# Build the container
docker build -t kaamkaanoon .

# Run the container
docker run -p 8501:8501 --env-file .env kaamkaanoon

# Open in browser
# http://localhost:8501
```

---

## 📁 Project Structure
kaamkaanoon/
├── data/
│   └── pdfs/              ← 6 government PDFs (not in repo — download manually)
├── src/
│   ├── init.py        ← Makes src a Python package
│   ├── ingest.py          ← Loads PDFs, chunks, embeds, stores in ChromaDB
│   ├── retriever.py       ← Loads vectorstore, retrieves top 5 chunks with scores
│   ├── chain.py           ← LangChain QA chain with system prompt and citations
│   └── guardrails.py      ← 2-layer out-of-scope protection
├── vectorstore/           ← ChromaDB saves here automatically (not in repo)
├── app.py                 ← Streamlit UI — main file user runs
├── requirements.txt       ← All Python dependencies with pinned versions
├── .env                   ← API keys (never pushed to GitHub)
├── .env.example           ← Template for .env
├── .gitignore             ← Excludes .env, venv, vectorstore
├── Dockerfile             ← Containerization recipe
└── README.md              ← This file

---

## 🛡️ Guardrail Architecture

KaamKaanoon uses 2 layers of protection to prevent wrong answers:

**Layer 1 — Topic Classifier:**
Before touching the vector database, the Groq LLM classifies whether the question is about Indian labour law. If NO, a friendly rejection message is returned immediately. This prevents wasting compute on irrelevant questions.

**Layer 2 — Similarity Threshold:**
After retrieval, the best chunk's relevance score is checked. If the score is below 0.30, the documents don't contain a good answer. A low-confidence message is returned instead of a hallucinated answer.

Only if BOTH layers pass does the full RAG pipeline run.

---

## 💡 Example Questions

- "What is the notice period if I want to resign?"
- "How is gratuity calculated after 5 years of service?"
- "Am I entitled to maternity leave if I have worked for 3 months?"
- "What percentage of my salary goes to EPF?"
- "What should I do if I face sexual harassment at my workplace?"
- "Can my employer deduct salary without giving a reason?"

---

## 🔑 Environment Variables

| Variable | Description | Where to get |
|----------|-------------|-------------|
| `GROQ_API_KEY` | Groq API key for Llama 3.3 70B | [console.groq.com](https://console.groq.com) |

---

## ⚠️ Important Disclaimer

KaamKaanoon provides **general legal information** only. It is **not legal advice**. The answers are based on official government documents but may not apply to every specific situation. For your specific legal situation, always consult a qualified labour law attorney.

---

## 👨‍💻 Author

**Dheeraj Merugu**
M.Tech CSE AI&DS — KL University (Graduating June 2026)

---

## 📄 License

MIT License — free to use, modify, and distribute.

