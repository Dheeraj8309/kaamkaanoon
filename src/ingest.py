# src/ingest.py
# This is the most important file in the project.
# It runs ONCE to build our knowledge base.
# What it does:
#   1. Loads all 6 PDFs from data/pdfs/
#   2. Splits each PDF into 500-character chunks
#   3. Converts each chunk into a vector (embedding)
#   4. Stores all vectors in ChromaDB (our local vector database)
# After this runs successfully, ChromaDB remembers everything forever
# until we delete the vectorstore/ folder.

import os
import sys

# Suppress the HuggingFace symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Disable ChromaDB telemetry — stops the "Failed to send telemetry" messages
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv

# LangChain's PDF loader — reads PDF files and extracts text page by page
from langchain_community.document_loaders import PyPDFLoader

# LangChain's text splitter — cuts long text into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Updated HuggingFace embeddings — uses the newer langchain-huggingface package
from langchain_huggingface import HuggingFaceEmbeddings

# Updated ChromaDB integration — uses the newer langchain-chroma package
# This replaces the deprecated langchain_community.vectorstores.Chroma
from langchain_chroma import Chroma

# Load environment variables from our .env file
load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION — all settings in one place
# ─────────────────────────────────────────────

# Folder where our 6 PDFs are stored
PDF_FOLDER = os.path.join("data", "pdfs")

# Folder where ChromaDB will save the vector database
VECTORSTORE_DIR = "vectorstore"

# The HuggingFace embedding model we are using
# all-MiniLM-L6-v2 converts text into 384-dimensional vectors
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Each chunk will be maximum 500 characters long
CHUNK_SIZE = 500

# Overlap between consecutive chunks = 50 characters
CHUNK_OVERLAP = 50

# ChromaDB collection name — like a table name in a database
COLLECTION_NAME = "kaamkaanoon"


def load_pdfs(pdf_folder):
    """
    Loads all PDF files from the given folder.
    Returns a list of LangChain Document objects.
    Each Document object contains:
        - page_content: the text of that page
        - metadata: {'source': 'filepath', 'page': page_number}
    """
    print(f"\n📂 Loading PDFs from: {pdf_folder}")

    # Get list of all PDF files in the folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print("✗ No PDF files found! Check your data/pdfs/ folder.")
        sys.exit(1)

    print(f"   Found {len(pdf_files)} PDF files:")

    all_documents = []

    for filename in pdf_files:
        filepath = os.path.join(pdf_folder, filename)
        print(f"   Loading: {filename}...")

        loader = PyPDFLoader(filepath)
        documents = loader.load()

        print(f"      ✓ {len(documents)} pages loaded")
        all_documents.extend(documents)

    print(f"\n   Total pages loaded from all PDFs: {len(all_documents)}")
    return all_documents


def split_documents(documents):
    """
    Splits long pages into smaller chunks of 500 characters.
    Each chunk keeps the original metadata (source filename + page number).
    This metadata is what gives us CITATIONS later.
    """
    print(f"\n✂️  Splitting pages into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    print(f"   Total chunks created: {len(chunks)}")
    print(f"   Example chunk preview:")
    print(f"   Source : {chunks[0].metadata.get('source', 'unknown')}")
    print(f"   Page   : {chunks[0].metadata.get('page', 'unknown')}")
    print(f"   Text   : {chunks[0].page_content[:150]}...")

    return chunks


def create_embeddings():
    """
    Loads the HuggingFace embedding model.
    First run downloads ~90MB. Subsequent runs use cache.
    """
    print(f"\n🤖 Loading embedding model: {EMBEDDING_MODEL}")
    print(f"   (First run downloads ~90MB — this may take a few minutes)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        # normalize_embeddings=True ensures cosine similarity scores
        # are always between 0 and 1 — critical for our guardrails
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"   ✓ Embedding model loaded successfully")
    return embeddings


def store_in_chromadb(chunks, embeddings):
    """
    Stores all chunks and their embeddings in ChromaDB.
    Uses cosine similarity so scores are always between 0 and 1.
    ChromaDB saves everything to the vectorstore/ folder on disk.
    """
    print(f"\n💾 Storing chunks in ChromaDB...")
    print(f"   Location: {VECTORSTORE_DIR}/")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   This may take several minutes for {len(chunks)} chunks...")

    # collection_metadata sets the distance function to cosine similarity
    # This guarantees scores between 0 and 1 — required for our guardrails
    # "hnsw:space": "cosine" is the key fix for scores above 1.0
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORSTORE_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"   ✓ ChromaDB populated and saved to disk!")
    return vectorstore


def verify_vectorstore(vectorstore, chunks):
    """
    Quick sanity check — asks ChromaDB a test question
    and prints the top result to confirm everything works.
    """
    print(f"\n🔍 Verifying ChromaDB with a test query...")

    test_query = "what is the notice period for worker resignation"
    results = vectorstore.similarity_search_with_score(test_query, k=3)

    print(f"   Test query: '{test_query}'")
    print(f"   Top result:")

    document, score = results[0]
    print(f"   Source : {document.metadata.get('source', 'unknown')}")
    print(f"   Page   : {document.metadata.get('page', 'unknown')}")
    print(f"   Score  : {round(float(score), 4)} (should be between 0 and 1)")
    print(f"   Text   : {document.page_content[:200]}...")

    print(f"\n   ✓ ChromaDB is working correctly!")
    print(f"   Total chunks stored: {len(chunks)}")


# ─────────────────────────────────────────────
# MAIN — runs everything in order
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  KaamKaanoon — Ingestion Pipeline")
    print("  Building knowledge base from 6 Indian law PDFs")
    print("=" * 60)

    documents = load_pdfs(PDF_FOLDER)
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vectorstore = store_in_chromadb(chunks, embeddings)
    verify_vectorstore(vectorstore, chunks)

    print("\n" + "=" * 60)
    print("  ✅ Ingestion complete! Knowledge base is ready.")
    print("  You can now run retriever.py to query it.")
    print("=" * 60)