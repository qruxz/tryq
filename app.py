# app.py (English + Hinglish only)

from fastapi import FastAPI, Request, Header
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from rag_system import RAGSystem
import logging
from datetime import datetime
import uuid
import uvicorn
import re

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Assistant Backend with RAG (Gemini)")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chat_logs.log"), logging.StreamHandler()],
)
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)


def log_message(user_id, message, request: Request, is_user=True, response=None, error=None, language="en"):
    """Save chat logs with language"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "user_id": user_id,
        "message_type": "user" if is_user else "ai",
        "message": message,
        "response": response,
        "error": error,
        "language": language,
        "ip_address": request.client.host if request.client else "Unknown",
        "user_agent": request.headers.get("user-agent", "Unknown"),
    }
    log_file = logs_dir / f"chat_logs_{datetime.now().strftime('%Y-%m-%d')}.json"
    try:
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to write to log file: {e}")

    if is_user:
        logging.info(f"[{language.upper()}] User {user_id} ({log_entry['ip_address']}): {message}")
    else:
        logging.info(f"[{language.upper()}] AI Response to {user_id}: {response[:100]}...")


def get_user_id(session_id: str = None):
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id


# ------------------ Hinglish Detector ------------------
def is_hinglish_query(text: str) -> bool:
    text_lower = text.lower().strip()

    hinglish_indicators = {
        "kya", "kaise", "kaun", "kab", "kyun", "kyu",
        "karo", "karna", "karte", "karta", "hai", "nahi", "haan",
        "aap", "aapka", "yeh", "woh", "mera", "tera", "batao", "chahiye"
    }

    words = re.findall(r"\b\w+\b", text_lower)
    hinglish_word_count = sum(1 for word in words if word in hinglish_indicators)

    hinglish_patterns = [
        r"\baap\b.*\b(kya|kaise)\b",
        r"\b(kya|kaise)\b.*\b(hai|ho|hoga)\b",
        r"\bmujhe\b.*\bchahiye\b",
    ]

    pattern_matches = sum(1 for p in hinglish_patterns if re.search(p, text_lower))

    total_words = len(words)
    if total_words == 0:
        return False

    hinglish_ratio = hinglish_word_count / total_words

    return (hinglish_ratio > 0.3) or (hinglish_word_count >= 2 and pattern_matches > 0) or (pattern_matches >= 2)


def detect_language_from_query(text: str, header_language: str) -> str:
    """Return 'en' or 'hinglish' only"""
    if is_hinglish_query(text):
        return "hinglish"
    return header_language


# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# ------------------ Init RAG ------------------
rag_system = None
if api_key:
    rag_system = RAGSystem(api_key)
    print("📑 Building vectorstore (brand_data.json + scraped_data/*.txt if available)...")
    rag_system.build_vectorstore(use_scraped=True)


# ----------------------------------------------------------------------

@app.post("/api/chat")
async def chat(
    request: Request,
    session_id: str = Header(default=None),
    x_language: str = Header(default="en"),
):
    try:
        data = await request.json()
        message = data.get("message", "")
        if not message:
            return {"error": "Message is required", "success": False}

        user_id = get_user_id(session_id)

        detected_language = detect_language_from_query(message, x_language)
        log_message(user_id, message, request, is_user=True, language=detected_language)

        if not api_key:
            return {"error": "Gemini API key not configured.", "success": False}
        if not rag_system:
            return {"error": "RAG system not initialized", "success": False}

        personal_info = rag_system.get_personal_info()

        # ---------------- Query Refinement ----------------
        query_refiner_prompt = f"""
Refine the user question into a precise search query.

User's Original Question: "{message}"
Refined Search Query:
"""
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            query_refiner_response = model.generate_content(query_refiner_prompt)
            refined_query = query_refiner_response.text.strip()
        except Exception:
            refined_query = message

        try:
            relevant_context = rag_system.search_relevant_context(refined_query, k=4)
        except Exception:
            relevant_context = "Unable to retrieve relevant information."

        # ---------------- Language-specific Instructions ----------------
        if detected_language == "en":
            lang_instruction = (
                "\n- Answer in **English only**.\n"
                "- Use a respectful tone, keep it short (2–5 sentences).\n"
                "- Use bullet points with emojis for clarity.\n"
                "- ❌ Do NOT respond in Hinglish or Devanagari.\n"
            )
        else:  # Hinglish
            lang_instruction = (
                "\n- Answer **STRICTLY in Romanized Hindi (Hinglish)**.\n"
                "- Mix Hindi with English technical terms.\n"
                "- Keep respectful tone, 2–5 sentences.\n"
                "- Use bullet points with emojis.\n"
                "- ❌ Do NOT respond in pure English or Devanagari script.\n"
            )

        # ---------------- Final Answer ----------------
        final_answer_prompt = f"""
You are a precise FAQ assistant for the brand {personal_info['name']}.

<USER_QUESTION>
{message}
</USER_QUESTION>

<DETAILED_CONTEXT>
{relevant_context}
</DETAILED_CONTEXT>

INSTRUCTIONS:
{lang_instruction}
- If the context has a clearly written answer, return it verbatim.
- If no relevant answer exists, say politely: "🙏 Sorry, I don't have that information right now."
"""

        final_model = genai.GenerativeModel("gemini-1.5-flash")
        final_response = final_model.generate_content(final_answer_prompt)
        ai_response = final_response.text.strip()

        log_message(user_id, message, request, is_user=False, response=ai_response, language=detected_language)

        return {
            "response": ai_response,
            "success": True,
            "refined_query": refined_query,
            "session_id": user_id,
            "language": x_language,
            "detected_language": detected_language,
        }

    except Exception as e:
        return {"error": f"Failed to get AI response: {str(e)}", "success": False}


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "api_key": "configured" if api_key else "not configured",
        "rag_system": "initialized" if rag_system else "not initialized",
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)
