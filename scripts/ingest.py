# scripts/ingest.py

import os
import json
import shutil
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# ------------------ CONFIG ------------------
BASE_DIR = Path(__file__).resolve().parent.parent   # backend/ root
DATA_DIR = BASE_DIR / "scraped_data"
PDF_DIR = BASE_DIR / "pdfs"
BRAND_JSON = BASE_DIR / "brand_data.json"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

# Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

# Text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", ".", "!", "?", ";", "•", "—", "- "]
)

# ------------------ UTILS ------------------
def clean_chroma():
    """Delete old chroma_db folder."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print("🗑️ Old chroma_db deleted")
    CHROMA_DIR.mkdir(exist_ok=True)

def load_brand_data() -> list[Document]:
    """Load brand_data.json into documents."""
    if not BRAND_JSON.exists():
        print("⚠️ brand_data.json not found, skipping...")
        return []

    with open(BRAND_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    if "brand" in data:
        docs.append(Document(page_content=json.dumps(data["brand"], indent=2), metadata={"type": "brand"}))
    for p in data.get("products", []):
        docs.append(Document(page_content=json.dumps(p, indent=2), metadata={"type": "product"}))
    for f in data.get("faqs", []):
        docs.append(Document(page_content=f"Q: {f.get('q')}\nA: {f.get('a')}", metadata={"type": "faq"}))

    print(f"📦 Loaded {len(docs)} docs from brand_data.json")
    return docs

def load_scraped_texts() -> list[Document]:
    """Load scraped_data/*.txt files into documents."""
    if not DATA_DIR.exists():
        print("⚠️ scraped_data/ folder not found, skipping...")
        return []

    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()
    if not docs:
        print("⚠️ No .txt files found in scraped_data/")
        return []

    split_docs = splitter.split_documents(docs)
    print(f"📄 Loaded and split into {len(split_docs)} chunks from scraped_data/")
    return split_docs

def load_pdfs() -> list[Document]:
    """Load pdfs/*.pdf files into documents."""
    if not PDF_DIR.exists():
        print("⚠️ pdfs/ folder not found, skipping...")
        return []

    loader = DirectoryLoader(
        str(PDF_DIR),
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()
    if not docs:
        print("⚠️ No PDFs found in pdfs/")
        return []

    split_docs = splitter.split_documents(docs)
    print(f"📚 Loaded and split into {len(split_docs)} chunks from PDFs/")
    return split_docs

def deduplicate(docs: list[Document]) -> list[Document]:
    """Remove duplicate chunks by content hash."""
    seen = set()
    unique = []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique.append(d)
    print(f"✅ Deduplicated: {len(unique)} unique chunks remain")
    return unique

# ------------------ MAIN ------------------
def main():
    print("🚀 Starting ingestion pipeline...")
    clean_chroma()

    docs = []
    docs.extend(load_brand_data())
    docs.extend(load_scraped_texts())
    docs.extend(load_pdfs())  # NEW

    docs = deduplicate(docs)

    if not docs:
        print("❌ No documents to index. Exiting.")
        return

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="brand_kb",
        client=chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    )

    print(f"🎉 Vector DB built successfully with {len(docs)} docs in {CHROMA_DIR}/")

if __name__ == "__main__":
    main()
