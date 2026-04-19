from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv

load_dotenv()

# Define both models for dynamic switching
gemini_llm = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite-preview')
ollama_llm = ChatOllama(model='gemma4:e2b', base_url="http://localhost:11434")

# Default export for backwards compatibility
LLM = gemini_llm 

VECTOR_STORE_PATH = "vector_store/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNKS_EXPORT_PATH = "vector_store/chunks_export.json"

# Chunking Parameters
MAX_CHARACTERS = 3000
NEW_AFTER_N_CHARS = 2400
COMBINE_TEXT_UNDER_N_CHARS = 500
