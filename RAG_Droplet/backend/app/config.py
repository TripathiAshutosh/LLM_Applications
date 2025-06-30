import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    UPLOAD_DIR = "uploads"
    VECOTR_PERSIST_DIR = "vector_db"  # Renamed for clarity, but keeps compatibility
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {".pdf"}

settings = Settings()

# Validate settings
if not settings.OPENAI_API_KEY:
    print("⚠️ WARNING: OPENAI_API_KEY not found in environment variables!")
else:
    print(f"✅ OpenAI API key loaded (starts with: {settings.OPENAI_API_KEY[:8]}...)")