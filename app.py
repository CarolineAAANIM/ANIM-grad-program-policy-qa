
import os, re, json, time, hashlib, io
import requests
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text
import mammoth
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# Configuration via Streamlit Secrets
# ==============================
# Set one of: provider = "openai" or "azure"
PROVIDER = st.secrets.get("LLM_PROVIDER", "openai").lower()

# --- OpenAI ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = st.secrets.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = st.secrets.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# --- Azure OpenAI ---
AZURE_OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_CHAT_DEPLOYMENT = st.secrets.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_EMBED_DEPLOYMENT = st.secrets.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")
AZURE_OPENAI_API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION", "2024-06-01")

# --- Admin Token (optional) ---
ADMIN_TOKEN = st.secrets.get("ADMIN_TOKEN", "")  # if set, required to build index

# Paths (Streamlit Cloud persists files across sessions; redeploy resets them)
DATA_DIR = "data_index"
os.makedirs(DATA_DIR, exist_ok=True)

VEC_PATH = os.path.join(DATA_DIR, "vectors.npy")
TFIDF_PATH = os.path.join(DATA_DIR, "tfidf.npz")
VECTORIZER_PATH = os.path.join(DATA_DIR, "vectorizer.json")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")
META_PATH = os.path.join(DATA_DIR, "meta.json")

TOP_K = 10
CONFIDENCE_THRESHOLD = 0.15  # if lower, say "not certain"

# ==============================
# Utilities
# ==============================
def clean_text(txt: str) -> str:
    if not txt:
        return ""
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

def fetch_html(url: str):
    """Fetch HTML and extract main content using built-in html.parser (no lxml)."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Try to focus on main content; adjust per site if needed
    main = soup.select_one("main") or soup.select_one("article") or soup.select_one("#content") or soup
    text = main.get_text(separator=" ")
    title = soup.title.string if soup.title else url
    return clean_text(title), clean_text(text)

def read_pdf_file(file_bytes: bytes):
    """Extract text from PDF bytes via pdfminer.six."""
    with io.BytesIO(file_bytes) as f:
        text = pdf_extract_text(f) or ""
    return clean_text(text)

def read_docx_file(file_bytes: bytes):
    """Convert DOCX to HTML via mammoth, then strip tags to text (no python-docx / lxml)."""
    with io.BytesIO(file_bytes) as f:
        result = mammoth.convert_to_html(f)
        html = result.value or ""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    return clean_text(text)

def chunk_text(text: str, chunk_words=500, overlap_words=90):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        c = " ".join(words[start:end])
        chunks.append(c)
        start = max(end - overlap_words, end)
    return chunks

def make_id(source: str, idx: int):
    return hashlib.sha256(f"{source}:{idx}".encode()).hexdigest()[:24]

# ==============================
# Embeddings & Chat
# ==============================
def embed_texts(texts):
    """Get embeddings from OpenAI or Azure OpenAI (text-embedding-3-small by default)."""
    if PROVIDER == "azure":
        try:
            from openai import AzureOpenAI
