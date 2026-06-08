import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

COLLECTION_NAME = "documents"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

VECTOR_DIMENSION = 384

TOP_K = 5

MAX_CHUNK_TOKENS = 500

OVERLAP_TOKENS = 100