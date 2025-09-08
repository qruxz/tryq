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
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class RAGSystem:
    def __init__(self, gemini_api_key: str, collection_name: str = "brand_kb"):
        """
        RAG system for brand & product knowledge (Gemini + Chroma).

        Args:
            gemini_api_key: Google AI Studio (Gemini) API key
            collection_name: ChromaDB collection name
        """
        if not gemini_api_key:
            raise ValueError("Gemini API key is required")

        self.gemini_api_key = gemini_api_key
        self.collection_name = collection_name

        # Gemini embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )

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
        Build vector DB from brand_data.json + scraped text files
        """
        print("🔧 Building vector DB...")
        data = self.load_brand_data(json_path=json_path)
        self.profile_summary = self._generate_summary_text(data)

        # Create documents from brand_data.json
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

        # Add scraped text files
        if use_scraped:
            try:
                scraped_docs = self.load_scraped_text_data("scraped_data")
                docs.extend(scraped_docs)
                print(f"📄 Added {len(scraped_docs)} scraped text chunks")
            except Exception as e:
                print(f"⚠️ Skipping scraped data: {e}")

        # Build Chroma
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            client=self.chroma_client
        )
        print("✅ Vector DB built successfully!")

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
        return "\n\n".join(ctx_parts)

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
