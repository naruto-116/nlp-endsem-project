"""
Configuration module for KG-CiteRAG system.
Loads environment variables and defines system-wide constants.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Root
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCyOvJ6j2BVcBDpZBeGm_D4kqMggfzswJI")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "groq"

# Data Paths
GRAPH_PATH = DATA_DIR / "graph.pickle"
VECTOR_INDEX_PATH = DATA_DIR / "ildc_vector_index.faiss"
METADATA_PATH = DATA_DIR / "metadata.json"
ILDC_PATH = DATA_DIR / "ILDC_single.jsonl"  # Supreme Court dataset (4967 cases)

# Processing Configuration
# Reduced chunk size for better precision (page-aware hierarchical chunking)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "250"))  # Reduced from 500 to 250 words
CHUNK_OVERLAP = 25  # Reduced proportionally from 50 to 25
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "10"))
SUBSET_SIZE = None  # Process ALL cases (4967 available)

# Verification Settings
ENABLE_VERIFICATION = True
CITATION_PATTERN = r'\[(.*?)\]'  # Regex to extract [Case Name]

# Graph Settings
OVERRULED_KEYWORD = "overruled"
CITED_BY_RELATION = "CITES"
