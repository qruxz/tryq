# rag_system.py
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Embeddings (Gemini + HF fallback)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_community.embeddings import HuggingFaceEmbeddings


class RAGSystem:
    def __init__(self, gemini_api_key: str, collection_name: str = "brand_kb"):
        """
        RAG system for brand & product knowledge (Gemini + Chroma).
        Falls back to HuggingFace embeddings if Gemini fails (quota, rate-limit, or init error).
        """
        if not gemini_api_key:
            raise ValueError("Gemini API key is required")

        self.gemini_api_key = gemini_api_key
        self.collection_name = collection_name

        # Try to initialize Gemini embeddings (may raise on bad key or local env issues)
        self.use_fallback = False
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.gemini_api_key
            )
            print("✅ Initialized Gemini embeddings (GoogleGenerativeAIEmbeddings).")
        except Exception as e:
            print(f"⚠️ Gemini init failed ({e}). Falling back to HuggingFace embeddings.")
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.use_fallback = True

        # ChromaDB persistent client
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", "•", "—", "- "]
        )

        self.vectorstore = None
        self.profile_summary = ""  # Cached summary for app.py
        self.data_cache: Dict[str, Any] = {}

    # ---------------- Brand JSON Loader ----------------

    def _brand_json_path(self, json_path: Optional[str] = None) -> Path:
        if json_path:
            return Path(json_path)
        return Path(__file__).parent / "brand_data.json"

    def load_brand_data(self, json_path: Optional[str] = None) -> Dict[str, Any]:
        path = self._brand_json_path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Brand data not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Minimal validation
        if "brand" not in data or "name" not in data["brand"]:
            raise ValueError("brand_data.json must include: { 'brand': { 'name': ... } }")

        data.setdefault("products", [])
        data.setdefault("faqs", [])
        data.setdefault("mechanism", {})
        self.data_cache = data
        return data

    # ---------------- TXT Loader ----------------

    def load_scraped_text_data(self, txt_dir: str = "scraped_data") -> List[Document]:
        """
        Load scraped website data from .txt files.
        Returns a list of LangChain Documents.
        """
        loader = DirectoryLoader(
            txt_dir,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()

        if not documents:
            raise ValueError(f"No text files found in {txt_dir}")

        # Split into smaller chunks
        docs = self.text_splitter.split_documents(documents)
        return docs

    # ---------------- Summary ----------------

    def _generate_summary_text(self, data: Dict[str, Any]) -> str:
        brand = data.get("brand", {})
        name = brand.get("name", "Unknown Brand")
        tagline = brand.get("tagline", "")
        desc = brand.get("description", "")

        parts = [f"Brand Knowledge Base for: {name}"]
        if tagline:
            parts.append(f"Tagline: {tagline}")
        if desc:
            parts.append(f"Description: {desc}")

        products = data.get("products", [])
        parts.append(f"Total products/crops: {len(products)}")
        if products:
            crop_list = ", ".join([p.get("crop", "N/A") for p in products[:12]])
            if len(products) > 12:
                crop_list += ", ..."
            parts.append(f"Crops covered: {crop_list}")

        benefits = brand.get("benefits", [])
        if benefits:
            parts.append("Key Benefits:")
            for b in benefits[:8]:
                parts.append(f"- {b}")

        return "\n".join(parts)

    # ---------------- Build Vectorstore ----------------

    def build_vectorstore(self, json_path: Optional[str] = None, use_scraped: bool = True):
        """
        Build or load vector DB with Gemini embeddings (fallback to HuggingFace).
        This function will:
         - try to load an existing collection (so we avoid re-embedding on every run)
         - if no collection exists, build it (this will call embeddings)
         - if Gemini raises a quota/rate-limit error during embedding, fallback to HF
        """
        print("🔧 Loading/Building vector DB...")

        # Attempt to load an existing collection from disk (persistent Chroma)
        try:
            # Note: Chroma(...) loads an existing collection when available
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                client=self.chroma_client
            )
            # Access internal collection count to decide if it's populated
            try:
                count = self.vectorstore._collection.count()
            except Exception:
                # If count not available due to API differences, ignore and try retriever
                count = None

            if count and count > 0:
                print(f"✅ Loaded existing vector DB from disk (items: {count}). Using {'HuggingFace' if self.use_fallback else 'Gemini'} embeddings.")
                return
            # If count is None, we still attempt to use the loaded collection if retriever works
            if count is None:
                # Try a quick retriever call to confirm
                try:
                    retr = self.vectorstore.as_retriever()
                    _ = retr.get_relevant_documents("test")
                    print("✅ Loaded existing vector DB (no count available).")
                    return
                except Exception:
                    # Fallthrough to rebuild
                    pass
        except Exception as e:
            print(f"⚠️ Could not load existing DB cleanly: {e}. Will attempt to rebuild.")

        # Build fresh documents list (brand JSON + products + faqs + scraped)
        data = self.load_brand_data(json_path=json_path)
        self.profile_summary = self._generate_summary_text(data)

        docs: List[Document] = []
        if "brand" in data:
            docs.append(Document(
                page_content=json.dumps(data["brand"], indent=2),
                metadata={"type": "brand"}
            ))

        for p in data.get("products", []):
            docs.append(Document(
                page_content=json.dumps(p, indent=2),
                metadata={"type": "product"}
            ))

        for f in data.get("faqs", []):
            docs.append(Document(
                page_content=f"Q: {f.get('q')}\nA: {f.get('a')}",
                metadata={"type": "faq"}
            ))

        if use_scraped:
            try:
                scraped_docs = self.load_scraped_text_data("scraped_data")
                docs.extend(scraped_docs)
                print(f"📄 Added {len(scraped_docs)} scraped text chunks")
            except Exception as e:
                print(f"⚠️ Skipping scraped data: {e}")

        # Now create embeddings + Chroma. Try Gemini first, fallback to HF on quota/errors.
        tried_gemini = not self.use_fallback  # whether we should/are trying gemini
        if tried_gemini:
            try:
                print("🔁 Attempting to build Chroma with Gemini embeddings...")
                self.vectorstore = Chroma.from_documents(
                    documents=docs,
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    client=self.chroma_client
                )
                self.use_fallback = False
                print("✅ Vector DB built with Gemini embeddings!")
                return
            except GoogleGenerativeAIError as g_err:
                # Explicit Gemini API error (likely quota/rate limit)
                print(f"⚠️ Gemini embedding error caught: {g_err}. Falling back to HuggingFace embeddings.")
                # fallthrough to fallback block below
            except Exception as e:
                # Other unforeseen errors from Gemini embedding attempt
                print(f"⚠️ Error building with Gemini embeddings: {e}. Falling back to HuggingFace embeddings.")

        # Fallback: HuggingFace local embeddings
        try:
            print("🔁 Building Chroma with HuggingFace embeddings (fallback)...")
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                client=self.chroma_client
            )
            self.use_fallback = True
            print("✅ Vector DB built with HuggingFace embeddings (fallback).")
        except Exception as e:
            print(f"❌ Failed to build vector DB with HuggingFace embeddings: {e}")
            raise

    # ---------------- Retrieval ----------------

    def get_summary_document(self) -> str:
        return self.profile_summary

    def search_relevant_context(self, query: str, k: int = 5) -> str:
        if not self.vectorstore:
            raise ValueError("Vector DB not built. Call build_vectorstore() first.")

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={'k': k, 'fetch_k': 25, 'lambda_mult': 0.25}
        )
        docs = retriever.get_relevant_documents(query)

        unique_docs = []
        seen = set()
        for d in docs:
            if d.page_content not in seen:
                unique_docs.append(d)
                seen.add(d.page_content)

        ctx_parts = []
        for i, d in enumerate(unique_docs, 1):
            ctx_parts.append(f"Context {i}:\n{d.page_content}")
        engine = "HuggingFace" if self.use_fallback else "Gemini"
        ctx = "\n\n".join(ctx_parts)
        return f"<!-- embeddings_engine: {engine} -->\n\n{ctx}"

    # ---------------- Backward compatibility ----------------

    def get_personal_info(self) -> Dict[str, Any]:
        if not self.data_cache:
            self.load_brand_data()
        brand = self.data_cache.get("brand", {})
        return {
            "name": brand.get("name", "Brand"),
            "title": brand.get("tagline", "Brand Assistant")
        }

    def get_brand_info(self) -> Dict[str, Any]:
        if not self.data_cache:
            self.load_brand_data()
        return self.data_cache.get("brand", {})
