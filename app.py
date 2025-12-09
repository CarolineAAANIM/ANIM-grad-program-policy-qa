
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

# Paths
DATA_DIR = "data_index"
os.makedirs(DATA_DIR, exist_ok=True)

VEC_PATH = os.path.join(DATA_DIR, "vectors.npy")
TFIDF_PATH = os.path.join(DATA_DIR, "tfidf.npz")
VECTORIZER_PATH = os.path.join(DATA_DIR, "vectorizer.json")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")
META_PATH = os.path.join(DATA_DIR, "meta.json")

TOP_K = 10

# ==============================
# Utilities
# ==============================
def clean_text(txt: str) -> str:
    if not txt:
        return ""
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

def fetch_html(url: str):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    main = soup.select_one("main") or soup.select_one("article") or soup.select_one("#content") or soup
    text = main.get_text(separator=" ")
    title = soup.title.string if soup.title else url
    return clean_text(title), clean_text(text)

def read_pdf_file(file_bytes: bytes):
    with io.BytesIO(file_bytes) as f:
        text = pdf_extract_text(f) or ""
    return clean_text(text)

def read_docx_file(file_bytes: bytes):
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
# Index build & load
# ==============================
def build_index_from_sources(urls, uploaded_files):
    chunks_meta = []
    all_texts = []

    # URLs
    for url in urls:
        if not url.strip():
            continue
        try:
            title, text = fetch_html(url.strip())
        except Exception as e:
            st.warning(f"Failed to fetch {url}: {e}")
            continue
        parts = chunk_text(text)
        for i, p in enumerate(parts):
            cid = make_id(url, i)
            meta = {
                "id": cid,
                "title": title,
                "content": p,
                "source": url,
                "source_type": "url",
                "section": ""
            }
            chunks_meta.append(meta)
            all_texts.append(p)

    # Files
    for uf in uploaded_files:
        name = uf.name
        try:
            data = uf.read()
            if name.lower().endswith(".pdf"):
                text = read_pdf_file(data)
            elif name.lower().endswith(".docx"):
                text = read_docx_file(data)
            else:
                st.warning(f"Unsupported file type: {name}")
                continue
        except Exception as e:
            st.warning(f"Failed to read {name}: {e}")
            continue

        parts = chunk_text(text)
        for i, p in enumerate(parts):
            cid = make_id(name, i)
            meta = {
                "id": cid,
                "title": name,
                "content": p,
                "source": name,
                "source_type": "file",
                "section": ""
            }
            chunks_meta.append(meta)
            all_texts.append(p)

    if not all_texts:
        raise RuntimeError("No content found. Add URLs or upload PDFs/DOCX.")

    # Embeddings (simple TF-IDF + semantic via cosine similarity)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    tfidf = vectorizer.fit_transform(all_texts)

    # Save artifacts
    save_npz(TFIDF_PATH, tfidf)
    with open(VECTORIZER_PATH, "w", encoding="utf-8") as f:
        json.dump({"vocabulary": vectorizer.vocabulary_}, f)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_meta, f, ensure_ascii=False, indent=2)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"created_at": int(time.time()), "num_chunks": len(chunks_meta)}, f, indent=2)

    return len(chunks_meta)

@st.cache_resource
def load_index():
    if not (os.path.exists(TFIDF_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(CHUNKS_PATH)):
        return None

    tfidf = load_npz(TFIDF_PATH)
    with open(VECTORIZER_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)["vocabulary"]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    vectorizer.vocabulary_ = vocab
    vectorizer.fixed_vocabulary_ = True
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {"tfidf": tfidf, "vectorizer": vectorizer, "chunks": chunks}

# ==============================
# Retrieval
# ==============================
def rank_passages(query, index):
    q_tfidf = index["vectorizer"].transform([query])
    scores = (q_tfidf @ index["tfidf"].T).toarray()[0]
    idxs = np.argsort(-scores)[:TOP_K]
    results = []
    for rank, i in enumerate(idxs, start=1):
        meta = index["chunks"][i]
        results.append({
            "rank": rank,
            "score": float(scores[i]),
            "title": meta["title"],
            "url": meta["source"],
            "content": meta["content"]
        })
    return results

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Faculty Policy Q&A", layout="centered")
st.title("Faculty Policy Q&A (Non-AI Version)")

tabs = st.tabs(["Ask", "Admin (build/update index)"])

# ---- Ask tab ----
with tabs[0]:
    idx = load_index()
    if idx is None:
        st.info("The index has not been built yet. An admin needs to add sources in the **Admin** tab.")
    else:
        query = st.text_input("Ask a question (e.g., “How long does a student have to complete the graduate program?”)")
        if st.button("Search") and query.strip():
            with st.spinner("Searching relevant passages..."):
                passages = rank_passages(query.strip(), idx)
            st.markdown("### Top Matches")
            for p in passages:
                st.markdown(f"**{p['title']}** — {p['url']}")
                st.write(p["content"])
                st.markdown("---")

# ---- Admin tab ----
with tabs[1]:
    st.subheader("Build or Update Index")
    urls_text = st.text_area("Paste one URL per line", placeholder="https://example.edu/grad/handbook\nhttps://example.edu/grad/policy-exceptions")
    st.markdown("or upload files (PDF/DOCX)")
    uploaded = st.file_uploader("Upload PDFs/DOCX", type=["pdf", "docx"], accept_multiple_files=True)

    if st.button("Build Index"):
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        try:
            with st.spinner("Indexing sources (this may take a few minutes)..."):
                num = build_index_from_sources(urls, uploaded or [])
            st.success(f"Index built: {num} chunks.")
            st.caption("You can now use the Ask tab.")
        except Exception as e:
            st.error(f"Index build failed: {e}")
