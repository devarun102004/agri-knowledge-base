import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RAW_PDF_DIR = DEFAULT_DATA_DIR / "raw_pdfs"
DEFAULT_OKF_DIR = DEFAULT_DATA_DIR / "okf"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_QDRANT_COLLECTION = "pesticide_knowledge"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_SIZE = 384
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")