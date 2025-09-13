# Fixed app.py with two-language toggle and smart Hinglish detection

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

app = FastAPI(title="AI Assistant Backend with RAG (Gemini) - Hindi/English")

# Enhanced CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "https://your-deployed-domain.com",
        "*"  # For development only
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Accept",
        "Accept-Language", 
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "X-Language",
        "Session-Id",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
    ],
    expose_headers=["*"],
    max_age=86400,
)

@app.options("/{full_path:path}")
async def options_handler(request: Request, full_path: str):
    return {"message": "OK"}

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
        logging.info(f"[{language.upper()}] AI Response to {user_id}: {response[:100] if response else 'None'}...")

def get_user_id(session_id: str = None):
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id

# Enhanced Language Detection
def is_hindi_devanagari(text: str) -> bool:
    """Detect Hindi text using Devanagari Unicode range"""
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    devanagari_chars = len(devanagari_pattern.findall(text))
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        return False
    
    return (devanagari_chars / total_chars) > 0.3

def is_hinglish_query(text: str) -> bool:
    """Enhanced Hinglish detection with comprehensive indicators"""
    text_lower = text.lower().strip()

    # Comprehensive Hinglish indicators
    hinglish_indicators = {
        # Question words
        "kya", "kaise", "kaun", "kab", "kyun", "kyu", "kahan", "kitna", "kitni", "kab",
        
        # Verbs and actions
        "karo", "karna", "karte", "karta", "karenge", "hai", "hain", "hoga", "hogi", "hogaye",
        "gaya", "gayi", "liya", "diya", "kiya", "hua", "hui", "dekho", "dekhe", "suno", "suniye",
        "boliye", "kaho", "kehte", "milta", "milega", "batao", "bataiye", "samjha", "samjhi",
        
        # Common words
        "nahi", "haan", "aap", "aapka", "aapko", "yeh", "ye", "woh", "wo", "mera", "mere", 
        "tera", "tere", "chahiye", "chaahiye", "mein", "main", "hum", "humara", "tumhara",
        "unka", "uska", "iska", "jaise", "waise", "phir", "fir", "abhi", "sabhi", "sab",
        "kuch", "koi", "agar", "lekin", "par", "aur", "ya", "bhi",
        
        # Respectful terms
        "bhai", "didi", "sir", "madam", "sahab", "ji", "uncle", "aunty",
        
        # Expressions
        "accha", "theek", "pata", "malum", "wala", "wale", "wali", "waala", "waale", "waali",
        
        # Agricultural/fertilizer specific Hinglish
        "navyakosh", "fertilizer", "organic", "benefits", "fayde", "price", "cost", "rate",
        "khareed", "lagana", "apply", "soil", "mitti", "crop", "fasal", "khet", "kheti",
        "ugana", "paida", "quantity", "amount", "milega", "available", "kahan", "where"
    }

    words = re.findall(r"\b\w+\b", text_lower)
    hinglish_word_count = sum(1 for word in words if word in hinglish_indicators)

    # Enhanced pattern matching for Hinglish
    hinglish_patterns = [
        r"\baap\b.*\b(kya|kaise|kab|kahan)\b",
        r"\b(kya|kaise|kab|kahan)\b.*\b(hai|ho|hoga|milega)\b", 
        r"\bmujhe\b.*\bchahiye\b",
        r"\b(mera|tera|uska|hamara)\b.*\b(hai|hoga|chahiye)\b",
        r"\b(samjha|pata|malum)\b",
        r"\b(theek|accha|okay)\b.*\b(hai|hoga)\b",
        r"\b(kitni|kitna)\b.*\b(quantity|price|cost|amount)\b",
        r"\bkahan\b.*\b(milega|khareed|available)\b",
        r"\bnavyakosh\b.*\b(kya|kaise|kahan|kitna)\b",
        r"\b(organic|fertilizer)\b.*\b(hai|hoga|chahiye|kaise)\b"
    ]

    pattern_matches = sum(1 for p in hinglish_patterns if re.search(p, text_lower))
    total_words = len(words) if words else 1
    hinglish_ratio = hinglish_word_count / total_words

    # More aggressive Hinglish detection
    return (hinglish_ratio > 0.1) or (hinglish_word_count >= 1 and pattern_matches > 0) or (pattern_matches >= 1)

def detect_smart_language(text: str, user_preference: str) -> str:
    """
    Smart language detection that responds appropriately:
    - If user writes in Hindi (Devanagari), respond in Hindi
    - If user writes in Hinglish, respond in Hinglish (regardless of toggle)
    - If user writes in English and toggle is English, respond in English
    - If user writes in English and toggle is Hindi, respond in Hindi
    """
    logging.info(f"Smart detection for: '{text}' with user preference: '{user_preference}'")
    
    # First check for Hindi (Devanagari) - always respond in Hindi
    if is_hindi_devanagari(text):
        logging.info("Detected: Hindi (Devanagari) - will respond in Hindi")
        return "hi"
    
    # Check for Hinglish - always respond in Hinglish regardless of toggle
    if is_hinglish_query(text):
        logging.info("Detected: Hinglish - will respond in Hinglish")
        return "hinglish"
    
    # Pure English text - follow user preference
    if user_preference == "hi":
        logging.info("Detected: English text with Hindi preference - will respond in Hindi")
        return "hi"
    else:
        logging.info("Detected: English text with English preference - will respond in English")
        return "en"

def get_language_instruction(response_language: str) -> str:
    """Language-specific response instructions"""
    if response_language == "en":
        return (
            "\n- IMPORTANT: Answer ONLY in pure English language.\n"
            "- Do NOT use any Hindi words, Roman Hindi, or Hinglish at all.\n"
            "- Use professional, clear English that international users can understand.\n"
            "- Keep responses informative but concise (3-6 sentences).\n"
            "- Use bullet points with emojis when listing benefits or instructions.\n"
            "- Focus on practical, actionable information about the products.\n"
            "- Example: 'Navyakosh is an organic fertilizer that provides excellent benefits for crop growth.'\n"
            "- NEVER use words like: hai, kya, kaise, aur, ke, etc.\n"
        )
    elif response_language == "hinglish":
        return (
            "\n- Answer in **natural Hinglish** (mix of English and romanized Hindi).\n"
            "- Use common Hindi words mixed with English technical terms naturally.\n"
            "- Keep tone friendly and conversational, like talking to a friend.\n"
            "- Use bullet points with emojis for clarity.\n"
            "- Example style: 'Navyakosh ek organic fertilizer hai jo crops ke liye bahut beneficial hai.'\n"
            "- Use words like: hai, kya, kaise, aur, ke liye, etc. naturally\n"
            "- NEVER use pure English or Devanagari script.\n"
        )
    else:  # Hindi
        return (
            "\n- Answer in **Hindi using Devanagari script only**.\n"
            "- Use respectful Hindi with proper honorifics (आप, जी).\n"
            "- Keep responses informative and respectful.\n"
            "- Use bullet points with emojis for clarity.\n"
            "- Technical terms can be in English if no Hindi equivalent exists.\n"
            "- NEVER respond in English or romanized Hindi.\n"
        )

def get_fallback_message(language: str) -> str:
    """Enhanced fallback messages"""
    if language == "en":
        return """🌱 I'm here to help with information about Navyakosh Organic Fertilizer and agricultural products. 

Please ask me about:
• Product benefits and features
• Application methods for different crops  
• Pricing and availability
• Soil health and organic farming

Try asking something like "What is Navyakosh?" or "How to use organic fertilizer?"
        
If you need immediate assistance, please contact our support team."""
        
    elif language == "hinglish":
        return """🌱 Main Navyakosh Organic Fertilizer aur agriculture products ke baare mein help kar sakta hun.

Aap mujhse pooch sakte hain:
• Product ke benefits aur features
• Different crops ke liye kaise use karna hai
• Price aur kahan milta hai  
• Soil health aur organic farming

Try karo kuch aise puchna: "Navyakosh kya hai?" ya "Organic fertilizer kaise use karte hain?"

Agar urgent help chahiye toh support team se contact karo."""
        
    else:  # Hindi
        return """🌱 मैं नव्याकोष जैविक उर्वरक और कृषि उत्पादों के बारे में जानकारी देने के लिए यहाँ हूँ।

आप मुझसे पूछ सकते हैं:
• उत्पाद के फायदे और विशेषताएं  
• विभिन्न फसलों के लिए उपयोग की विधि
• मूल्य और उपलब्धता
• मिट्टी का स्वास्थ्य और जैविक खेती

कुछ इस तरह पूछने की कोशिश करें: "नव्याकोष क्या है?" या "जैविक उर्वरक का उपयोग कैसे करें?"

तत्काल सहायता के लिए हमारी सहायता टीम से संपर्क करें।"""

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    try:
        genai.configure(api_key=api_key)
        logging.info("✅ Gemini API configured successfully")
    except Exception as e:
        logging.error(f"❌ Failed to configure Gemini API: {e}")
        api_key = None

# Init RAG
rag_system = None
if api_key:
    try:
        rag_system = RAGSystem(api_key)
        print("📑 Building vectorstore...")
        rag_system.build_vectorstore(use_scraped=True)
        logging.info("✅ RAG System initialized")
    except Exception as e:
        logging.error(f"❌ RAG System failed: {e}")

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
        
        # Smart language detection
        response_language = detect_smart_language(message, x_language)
        log_message(user_id, message, request, is_user=True, language=response_language)

        if not api_key or not rag_system:
            error_msg = get_fallback_message(response_language)
            log_message(user_id, message, request, is_user=False, error="API/RAG not configured", language=response_language)
            return {"error": error_msg, "success": False}

        personal_info = rag_system.get_personal_info()

        # Context retrieval
        try:
            logging.info(f"Searching for context with query: '{message}'")
            relevant_context = rag_system.search_relevant_context(message, k=5)
            logging.info(f"Retrieved context length: {len(relevant_context)} chars")
            
            # If no context found, try with key terms
            if len(relevant_context.strip()) < 50:
                key_terms = ["navyakosh", "organic", "fertilizer", "benefits", "application", "crops"]
                for term in key_terms:
                    if term.lower() in message.lower():
                        relevant_context = rag_system.search_relevant_context(term, k=3)
                        if len(relevant_context.strip()) > 50:
                            break
                            
        except Exception as e:
            logging.warning(f"Context retrieval failed: {e}")
            relevant_context = "Information about Navyakosh Organic Fertilizer and agricultural products."

        # Response generation with smart language selection
        lang_instruction = get_language_instruction(response_language)
        fallback_message = get_fallback_message(response_language)

        final_answer_prompt = f"""You are a helpful agricultural assistant for {personal_info.get('name', 'LCB Fertilizers')}.

USER QUESTION: "{message}"
USER LANGUAGE PREFERENCE: {x_language}
DETECTED RESPONSE LANGUAGE: {response_language}

RELEVANT INFORMATION:
{relevant_context}

RESPONSE GUIDELINES:
{lang_instruction}

IMPORTANT CONTEXT:
- The user has set their language toggle to: {x_language}
- Based on their message, I've determined the best response language is: {response_language}
- If response_language is "hinglish", use natural mix of English and romanized Hindi
- If response_language is "hi", use only Devanagari script
- If response_language is "en", use only English

Always be helpful and provide practical agricultural advice. Focus on Navyakosh benefits and usage.

Provide your response now:"""

        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            final_response = model.generate_content(final_answer_prompt)
            ai_response = final_response.text.strip()
            
            if not ai_response or len(ai_response.strip()) < 10:
                ai_response = fallback_message
                
            logging.info(f"Generated response length: {len(ai_response)} chars")
            
        except Exception as e:
            logging.error(f"AI response generation failed: {e}")
            ai_response = fallback_message

        log_message(user_id, message, request, is_user=False, response=ai_response, language=response_language)

        return {
            "response": ai_response,
            "success": True,
            "session_id": user_id,
            "user_language_preference": x_language,
            "response_language": response_language,
        }

    except Exception as e:
        logging.error(f"Chat endpoint error: {e}")
        return {
            "error": "An error occurred processing your request.",
            "success": False,
            "user_language_preference": x_language
        }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "api_key": "configured" if api_key else "not configured",
        "rag_system": "initialized" if rag_system else "not initialized",
        "supported_languages": ["en", "hi"],
        "smart_detection": "enabled",
        "cors": "enabled"
    }

@app.get("/api/cors-test")
async def cors_test():
    """Test CORS functionality"""
    return {
        "message": "CORS is working!",
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    }

if __name__ == "__main__":
    print("🚀 Starting server with smart language detection (EN/HI toggle)...")
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)
