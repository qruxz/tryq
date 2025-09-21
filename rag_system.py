# rag_system.py
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

load_dotenv()

class RAGSystem:
    def __init__(self, api_key: str):
        """
        Initialize the RAG system with Gemini embeddings and Neon (pgvector) connection.
        """
        self.api_key = api_key
        self.pg_conn = os.getenv("NEON_DB_URL")
        if not self.pg_conn:
            raise ValueError("❌ NEON_DB_URL missing in .env")

        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

        # Name of the collection in pgvector
        self.collection_name = "brand_kb"

    def search_relevant_context(self, query: str, k: int = 5) -> str:
        """
        Perform similarity search for the given query and return concatenated context.
        """
        try:
            vectorstore = PGVector(
                collection_name=self.collection_name,
                connection_string=self.pg_conn,
                embedding_function=self.embeddings
            )
            results = vectorstore.similarity_search(query, k=k)
            return " ".join([r.page_content for r in results]) if results else ""
        except Exception as e:
            print(f"⚠️ RAG search error: {e}")
            return ""
