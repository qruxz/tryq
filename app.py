# app.py (English + Hinglish + Hindi support)

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

app = FastAPI(title="AI Assistant Backend with RAG (Gemini) - Multi-language")

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


# ------------------ Enhanced Language Detection ------------------

def is_hindi_devanagari(text: str) -> bool:
    """Detect Hindi text using Devanagari Unicode range"""
    # Devanagari Unicode range: U+0900-U+097F
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    devanagari_chars = len(devanagari_pattern.findall(text))
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        return False
    
    # If more than 30% characters are Devanagari, it's Hindi
    return (devanagari_chars / total_chars) > 0.3


def is_hinglish_query(text: str) -> bool:
    """Enhanced Hinglish detection"""
    text_lower = text.lower().strip()

    hinglish_indicators = {
        # Core Hindi words in Roman script
        "kya", "kaise", "kaun", "kab", "kyun", "kyu", "kahan", "kitna", "kitni",
        "karo", "karna", "karte", "karta", "karenge", "hai", "hain", "nahi", "haan",
        "aap", "aapka", "aapko", "yeh", "ye", "woh", "wo", "mera", "mere", "tera", "tere",
        "batao", "bataiye", "chahiye", "chaahiye", "mein", "main", "hum", "humara",
        "tum", "tumhara", "unka", "uska", "iska", "jaise", "waise", "phir", "fir",
        "abhi", "sabhi", "sab", "kuch", "koi", "agar", "lekin", "par", "aur", "ya",
        "bhi", "bhai", "didi", "sir", "madam", "sahab", "ji", "accha", "theek",
        "samjha", "samjhi", "pata", "malum", "dekho", "dekhe", "suno", "suniye",
        "boliye", "kaho", "kehte", "milta", "milega", "hoga", "hogi", "hogaye",
        "gaya", "gayi", "liya", "diya", "kiya", "hua", "hui", "wala", "wale", "wali"
    }

    words = re.findall(r"\b\w+\b", text_lower)
    hinglish_word_count = sum(1 for word in words if word in hinglish_indicators)

    # Enhanced pattern matching
    hinglish_patterns = [
        r"\baap\b.*\b(kya|kaise)\b",
        r"\b(kya|kaise)\b.*\b(hai|ho|hoga)\b",
        r"\bmujhe\b.*\bchahiye\b",
        r"\b(mera|tera|uska)\b.*\b(hai|hoga)\b",
        r"\b(samjha|pata|malum)\b",
        r"\b(theek|accha)\b.*\b(hai|hoga)\b"
    ]

    pattern_matches = sum(1 for p in hinglish_patterns if re.search(p, text_lower))

    total_words = len(words)
    if total_words == 0:
        return False

    hinglish_ratio = hinglish_word_count / total_words

    # More flexible detection
    return (hinglish_ratio > 0.25) or (hinglish_word_count >= 2 and pattern_matches > 0) or (pattern_matches >= 1)


def detect_language_from_query(text: str, header_language: str) -> str:
    """Enhanced language detection supporting en, hinglish, hi"""
    # First check for Hindi (Devanagari)
    if is_hindi_devanagari(text):
        return "hi"
    
    # Then check for Hinglish (romanized Hindi)
    if is_hinglish_query(text):
        return "hinglish"
    
    # Default to header language or English
    if header_language in ["en", "hinglish", "hi"]:
        return header_language
    
    return "en"


def get_language_instruction(detected_language: str) -> str:
    """Get language-specific instructions for AI responses"""
    if detected_language == "en":
        return (
            "\n- Answer in **English only**.\n"
            "- Use a respectful, professional tone, keep it concise (2–5 sentences).\n"
            "- Use bullet points with emojis for clarity when listing items.\n"
            "- ❌ Do NOT respond in Hinglish or Hindi/Devanagari.\n"
        )
    elif detected_language == "hinglish":
        return (
            "\n- Answer **STRICTLY in Romanized Hindi (Hinglish)**.\n"
            "- Mix Hindi words with English technical terms naturally.\n"
            "- Keep respectful tone, 2–5 sentences, use common Hinglish phrases.\n"
            "- Use bullet points with emojis for clarity.\n"
            "- Example style: 'Yeh product bahut accha hai aur organic farming ke liye best hai.'\n"
            "- ❌ Do NOT respond in pure English or Devanagari script.\n"
        )
    else:  # Hindi (hi)
        return (
            "\n- Answer in **Hindi using Devanagari script only**.\n"
            "- Use respectful Hindi tone with appropriate honorifics (आप, जी).\n"
            "- Keep response concise (2–5 sentences).\n"
            "- Use bullet points with emojis for clarity.\n"
            "- Technical terms can be in English if no Hindi equivalent exists.\n"
            "- ❌ Do NOT respond in English or Romanized Hindi.\n"
        )


def get_fallback_message(language: str) -> str:
    """Get appropriate fallback message based on language"""
    if language == "en":
        return "🙏 Sorry, I don't have that information right now. Please contact our support team for more details."
    elif language == "hinglish":
        return "🙏 Sorry, mere paas yeh information abhi nahi hai. Please support team se contact kariye."
    else:  # Hindi
        return "🙏 क्षमा करें, मेरे पास अभी यह जानकारी उपलब्ध नहीं है। कृपया हमारी सहायता टीम से संपर्क करें।"


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

        # Enhanced language detection
        detected_language = detect_language_from_query(message, x_language)
        log_message(user_id, message, request, is_user=True, language=detected_language)

        if not api_key:
            error_msg = "Gemini API key not configured."
            if detected_language == "hinglish":
                error_msg = "Gemini API key configure nahi hai."
            elif detected_language == "hi":
                error_msg = "जेमिनी API की कॉन्फ़िगर नहीं है।"
            return {"error": error_msg, "success": False}
            
        if not rag_system:
            error_msg = "RAG system not initialized"
            if detected_language == "hinglish":
                error_msg = "RAG system initialize nahi hua hai."
            elif detected_language == "hi":
                error_msg = "RAG सिस्टम प्रारंभ नहीं हुआ है।"
            return {"error": error_msg, "success": False}

        personal_info = rag_system.get_personal_info()

        # ---------------- Query Refinement ----------------
        query_refiner_prompt = f"""
Refine the user question into a precise search query for better information retrieval.
Focus on key terms and main intent.

User's Original Question: "{message}"
Language Context: {detected_language}

Refined Search Query:
"""
        
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            query_refiner_response = model.generate_content(query_refiner_prompt)
            refined_query = query_refiner_response.text.strip()
        except Exception as e:
            logging.warning(f"Query refinement failed: {e}")
            refined_query = message

        # ---------------- RAG Context Retrieval ----------------
        try:
            relevant_context = rag_system.search_relevant_context(refined_query, k=4)
        except Exception as e:
            logging.warning(f"RAG context retrieval failed: {e}")
            relevant_context = "Unable to retrieve relevant information from knowledge base."

        # ---------------- Language-specific Instructions ----------------
        lang_instruction = get_language_instruction(detected_language)
        fallback_message = get_fallback_message(detected_language)

        # ---------------- Final Answer Generation ----------------
        final_answer_prompt = f"""
You are a helpful and knowledgeable FAQ assistant for {personal_info['name']}, specializing in agricultural products and organic fertilizers.

<USER_QUESTION>
{message}
</USER_QUESTION>

<RELEVANT_CONTEXT>
{relevant_context}
</RELEVANT_CONTEXT>

<RESPONSE_GUIDELINES>
{lang_instruction}

CONTENT INSTRUCTIONS:
- If the context contains relevant information, provide a comprehensive but concise answer.
- Focus on practical, actionable information.
- If asking about products, include benefits, usage, and application details.
- For technical questions, explain in simple terms.
- If no relevant information is found, respond with: "{fallback_message}"
- Maintain consistency with the brand voice and agricultural expertise.
</RESPONSE_GUIDELINES>

Provide your response now:
"""

        try:
            final_model = genai.GenerativeModel("gemini-1.5-flash")
            final_response = final_model.generate_content(final_answer_prompt)
            ai_response = final_response.text.strip()
        except Exception as e:
            logging.error(f"AI response generation failed: {e}")
            ai_response = fallback_message

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
        logging.error(f"Chat endpoint error: {e}")
        error_response = "An unexpected error occurred. Please try again."
        if x_language == "hinglish":
            error_response = "Koi unexpected error hua hai. Please try again kariye."
        elif x_language == "hi":
            error_response = "एक अप्रत्याशित त्रुटि हुई है। कृपया पुनः प्रयास करें।"
            
        return {
            "error": error_response,
            "success": False,
            "detected_language": x_language
        }


@app.get("/api/health")
async def health_check():
    """Enhanced health check with language support info"""
    return {
        "status": "healthy",
        "api_key": "configured" if api_key else "not configured",
        "rag_system": "initialized" if rag_system else "not initialized",
        "supported_languages": ["en", "hinglish", "hi"],
        "language_features": {
            "english": "Full support",
            "hinglish": "Romanized Hindi support",
            "hindi": "Devanagari script support"
        }
    }


@app.get("/api/languages")
async def get_supported_languages():
    """Endpoint to get supported languages"""
    return {
        "supported_languages": [
            {
                "code": "en",
                "name": "English",
                "display": "English",
                "description": "Full English language support"
            },
            {
                "code": "hinglish", 
                "name": "Hinglish",
                "display": "English + Hindi",
                "description": "Mixed English and Romanized Hindi"
            },
            {
                "code": "hi",
                "name": "Hindi",
                "display": "हिंदी", 
                "description": "Hindi in Devanagari script"
            }
        ]
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)
