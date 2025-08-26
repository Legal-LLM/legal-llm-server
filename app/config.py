from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv, find_dotenv

# Load .env if present (won't override existing environment)
load_dotenv(find_dotenv(), override=False)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set (put it in .env or env var)")

DOCS_DIR = os.getenv("DOCS_DIR", "docs")
INDEX_DIR = os.getenv("INDEX_DIR", "data/faiss_index")

GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_MODEL_QUERY = os.getenv("GEMINI_MODEL_QUERY", "gemini-2.5-flash-lite")
GEMINI_MODEL_ANSWER = os.getenv("GEMINI_MODEL_ANSWER", "gemini-2.5-flash")

# LangChain / Google GenAI clients


def get_query_llm():
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL_QUERY, temperature=0.2)


def get_answer_llm():
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL_ANSWER, temperature=0.2)


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=GEMINI_EMBED_MODEL)
