# Fixed app.py with proper CORS handling and error fixes

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# Fixed CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],
)

# Proper OPTIONS handler for all routes
@app.options("/{path:path}")
async def options_handler(request: Request, path: str):
    """Handle all OPTIONS requests properly"""
    return JSONResponse(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400",
        },
        content={"message": "OK"}
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
        logging.info(f"[{language.upper()}] User {user_id}: {message}")
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
    """Enhanced Hinglish detection"""
    text_lower = text.lower().strip()

    hinglish_indicators = {
        "kya", "kaise", "kaun", "kab", "kyun", "kyu", "kahan", "kitna", "kitni",
        "karo", "karna", "karte", "karta", "karenge", "hai", "hain", "hoga", "hogi",
        "nahi", "haan", "aap", "aapka", "aapko", "yeh", "ye", "woh", "wo", "mera", "mere",
        "chahiye", "chaahiye", "mein", "main", "hum", "bhai", "accha", "theek", "pata",
        "navyakosh", "fertilizer", "organic", "benefits", "fayde", "price", "cost",
        "khareed", "lagana", "apply", "soil", "mitti", "crop", "fasal", "khet", "kheti"
    }

    words = re.findall(r"\b\w+\b", text_lower)
    hinglish_word_count = sum(1 for word in words if word in hinglish_indicators)
    total_words = len(words) if words else 1
    hinglish_ratio = hinglish_word_count / total_words

    return hinglish_ratio > 0.1 or hinglish_word_count >= 1

def detect_smart_language(text: str, user_preference: str) -> str:
    """Smart language detection"""
    if is_hindi_devanagari(text):
        return "hi"
    
    if is_hinglish_query(text):
        return "hinglish"
    
    return user_preference

def get_language_instruction(response_language: str) -> str:
    """Language-specific response instructions"""
    if response_language == "en":
        return (
            "\n- Answer ONLY in pure English language.\n"
            "- Keep responses informative but concise (3-6 sentences).\n"
            "- Use bullet points with emojis when appropriate.\n"
        )
    elif response_language == "hinglish":
        return (
            "\n- Answer in natural Hinglish (mix of English and romanized Hindi).\n"
            "- Use friendly, conversational tone with emoji and pointwise.\n"
            "- Example: 'Navyakosh ek organic fertilizer hai jo crops ke liye beneficial hai.'\n"
        )
    else:  # Hindi
        return (
            "\n- Answer in Hindi using Devanagari script only.\n"
            "- Use respectful Hindi with proper honorifics also in pointwise with emojis.\n"
        )

def get_fallback_message(language: str) -> str:
    """Fallback messages"""
    if language == "en":
        return "I'm here to help with Navyakosh Organic Fertilizer information. Please ask about products, benefits, or usage methods."
    elif language == "hinglish":
        return "Main Navyakosh ke baare mein help kar sakta hun. Aap products, benefits, ya usage ke baare mein pooch sakte hain."
    else:
        return "मैं नव्याकोष के बारे में जानकारी देने के लिए यहाँ हूँ। कृपया उत्पाद, लाभ या उपयोग के बारे में पूछें।"

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
        logging.info("📑 Building vectorstore...")
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
    """Main chat endpoint with proper error handling"""
    try:
        # Parse request body
        try:
            data = await request.json()
        except Exception as e:
            logging.error(f"Failed to parse JSON: {e}")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid JSON in request body", "success": False}
            )

        message = data.get("message", "").strip()
        if not message:
            return JSONResponse(
                status_code=400,
                content={"error": "Message is required", "success": False}
            )

        user_id = get_user_id(session_id)
        
        # Smart language detection
        response_language = detect_smart_language(message, x_language)
        log_message(user_id, message, request, is_user=True, language=response_language)

        # Check if services are available
        if not api_key:
            error_msg = "Gemini API key not configured"
            logging.error(error_msg)
            return JSONResponse(
                status_code=503,
                content={"error": error_msg, "success": False}
            )

        if not rag_system:
            error_msg = "RAG system not initialized"
            logging.error(error_msg)
            return JSONResponse(
                status_code=503,
                content={"error": error_msg, "success": False}
            )

        # Get context from RAG
        try:
            relevant_context = rag_system.search_relevant_context(message, k=5)
            if len(relevant_context.strip()) < 50:
                relevant_context = rag_system.search_relevant_context("navyakosh fertilizer", k=3)
        except Exception as e:
            logging.warning(f"Context retrieval failed: {e}")
            relevant_context = "Navyakosh Organic Fertilizer information."

        # Generate response
        lang_instruction = get_language_instruction(response_language)
        
        prompt = f"""You are an agricultural assistant for LCB Fertilizers.

User Question: "{message}"
Context: {relevant_context}

{lang_instruction}

Provide helpful information about Navyakosh organic fertilizer:"""

        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            ai_response = response.text.strip()
            
            if not ai_response or len(ai_response) < 10:
                ai_response = get_fallback_message(response_language)
                
        except Exception as e:
            logging.error(f"AI generation failed: {e}")
            ai_response = get_fallback_message(response_language)

        log_message(user_id, message, request, is_user=False, response=ai_response, language=response_language)

        return JSONResponse(
            status_code=200,
            content={
                "response": ai_response,
                "success": True,
                "session_id": user_id,
                "language": response_language,
            }
        )

    except Exception as e:
        logging.error(f"Chat endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error. Please try again.",
                "success": False
            }
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "api_key": "configured" if api_key else "not configured",
                "rag_system": "initialized" if rag_system else "not initialized",
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        logging.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "message": "LCB Fertilizer AI Assistant API",
            "status": "running",
            "endpoints": ["/api/chat", "/api/health"],
        }
    )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "success": False}
    )

@app.exception_handler(405)
async def method_not_allowed_handler(request: Request, exc):
    return JSONResponse(
        status_code=405,
        content={"error": "Method not allowed", "success": False}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logging.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "success": False}
    )

if __name__ == "__main__":
    print("🚀 Starting LCB Fertilizer AI Assistant Server...")
    print(f"📡 API Key: {'✅ Configured' if api_key else '❌ Missing'}")
    print(f"🧠 RAG System: {'✅ Ready' if rag_system else '❌ Not ready'}")
    print("🌐 Server: http://localhost:5001")
    
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)
