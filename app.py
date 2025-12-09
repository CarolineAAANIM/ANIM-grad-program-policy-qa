
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
# Safe OpenAI/Azure client constructors (no unsupported kwargs)
# ==============================
def _openai_client():
    """
    Construct the OpenAI client safely without passing unsupported kwargs.
    Proxies (if set via environment variables HTTP_PROXY/HTTPS_PROXY) are
    handled internally by the SDK.
    """
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)

def _azure_client():
    """
    Construct the Azure OpenAI client safely without passing unsupported kwargs.
    """
    from openai import AzureOpenAI
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

# ==============================
# Embeddings & Chat
# ==============================
def embed_texts(texts):
    """Get embeddings from OpenAI or Azure OpenAI (text-embedding-3-small by default)."""
    if PROVIDER == "azure":
        try:
            client = _azure_client()
            resp = client.embeddings.create(
                model=AZURE_OPENAI_EMBED_DEPLOYMENT,
                input=texts
            )
            return [d.embedding for d in resp.data]
        except Exception as e:
            raise RuntimeError(f"Azure embeddings failed: {e}")
    else:
        try:
            client = _openai_client()
            resp = client.embeddings.create(
                model=OPENAI_EMBED_MODEL,
                input=texts
            )
            return [d.embedding for d in resp.data]
        except Exception as e:
            raise RuntimeError(f"OpenAI embeddings failed: {e}")

def call_llm(prompt: str):
    """Call chat model (gpt-4o-mini by default) to compose grounded answer."""
    system = "You are a helpful, reliable university policy assistant. Cite sources."
    if PROVIDER == "azure":
        client = _azure_client()
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return resp.choices[0].message.content
    else:
        client = _openai_client()
        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return resp.choices[0].message.content

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
        parts = chunk_text(text, chunk_words=500, overlap_words=90)
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

    # Files (PDF/DOCX)
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

        parts = chunk_text(text, chunk_words=500, overlap_words=90)
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

    # Embeddings for chunks
    vectors = np.array(embed_texts(all_texts), dtype=np.float32)

    # TF-IDF for keyword matching
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    tfidf = vectorizer.fit_transform(all_texts)

    # Save artifacts
    np.save(VEC_PATH, vectors)
    save_npz(TFIDF_PATH, tfidf)
    with open(VECTORIZER_PATH, "w", encoding="utf-8") as f:
        json.dump({"vocabulary": vectorizer.vocabulary_}, f)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_meta, f, ensure_ascii=False, indent=2)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "created_at": int(time.time()),
            "num_chunks": len(chunks_meta),
            "provider": PROVIDER
        }, f, indent=2)

    return len(chunks_meta)

@st.cache_resource
def load_index():
    if not (os.path.exists(VEC_PATH) and os.path.exists(TFIDF_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(CHUNKS_PATH)):
        return None

    vectors = np.load(VEC_PATH)
    tfidf = load_npz(TFIDF_PATH)

    with open(VECTORIZER_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)["vocabulary"]
    # Recreate vectorizer for transform
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    vectorizer.vocabulary_ = vocab
    vectorizer.fixed_vocabulary_ = True

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return {"vectors": vectors, "tfidf": tfidf, "vectorizer": vectorizer, "chunks": chunks}

# ==============================
# Retrieval & Prompting
# ==============================
def rank_passages(query, index):
    q_vec = np.array(embed_texts([query])[0], dtype=np.float32).reshape(1, -1)
    emb_scores = cosine_similarity(q_vec, index["vectors"])[0]  # (n_chunks,)

    q_tfidf = index["vectorizer"].transform([query])
    kw_scores = (q_tfidf @ index["tfidf"].T).toarray()[0]

    scores = 0.6 * emb_scores + 0.4 * kw_scores  # hybrid
    idxs = np.argsort(-scores)[:TOP_K]

    results = []
    for rank, i in enumerate(idxs, start=1):
        meta = index["chunks"][i]
        results.append({
            "rank": rank,
            "score": float(scores[i]),
            "title": meta["title"],
            "url": meta["source"],
            "section": meta.get("section", ""),
            "content": meta["content"]
        })
    return results, float(np.max(scores))

def build_context(passages):
    blocks = []
    for i, p in enumerate(passages, start=1):
        excerpt = p["content"]
        if len(excerpt) > 1200:
            excerpt = excerpt[:1200] + "..."
        blocks.append(f"[{i}] Title: {p['title']}\nURL: {p['url']}\nSection: {p['section']}\nExcerpt: {excerpt}")
    return "\n\n".join(blocks)

def grounded_prompt(query, passages):
    context = build_context(passages)
    return f"""
You are a university policy assistant. Answer the user's question CONCISELY and ONLY using the provided passages.
If the question asks for a duration or number, put the number FIRST (e.g., "5 years") followed by a short clarification.
If the question is about policy exceptions, list each exception clearly with conditions/approvals.
Include citations as [n] with the title and URL after the answer.
If uncertain or conflicting, say "I’m not certain based on the available sources" and summarize the closest references.

Question: {query}

Passages:
{context}

Return format:
<answer on one concise line or short bullets if exceptions>
Sources: [1] Title (URL), [2] Title (URL)...
"""

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Faculty Policy Q&A", layout="centered")
st.title("Faculty Policy Q&A")

tabs = st.tabs(["Ask", "Admin (build/update index)"])

# ---- Ask tab ----
with tabs[0]:
    idx = load_index()
    if idx is None:
        st.info("The index has not been built yet. An admin needs to add sources in the **Admin** tab.")
    else:
        query = st.text_input("Ask a question (e.g., “How long does a student have to complete the graduate program?”)")
        if st.button("Search") and query.strip():
            with st.spinner("Retrieving and composing grounded answer..."):
                passages, max_score = rank_passages(query.strip(), idx)

                if max_score < CONFIDENCE_THRESHOLD:
                    st.markdown("### Answer")
                    st.write("I’m not certain based on the available sources. Please refine your question or add more documents.")
                    st.markdown("### Closest Sources")
                    for i, p in enumerate(passages[:5], start=1):
                        st.write(f"[{i}] {p['title']} — {p['url']}")
                else:
                    prompt = grounded_prompt(query.strip(), passages)
                    try:
                        answer = call_llm(prompt)
                    except Exception as e:
                        st.error(f"LLM call failed: {e}")
                        answer = "Unable to generate answer at the moment."

                    st.markdown("### Answer")
                    st.write(answer)

                    st.markdown("### Sources")
                    for i, p in enumerate(passages, start=1):
                        st.write(f"[{i}] {p['title']} — {p['url']}")

                    with st.expander("View retrieved passages"):
                        for p in passages:
                            st.markdown(f"**{p['title']}** — {p['url']}")
                            st.write(p["content"])
                            st.markdown("---")

# ---- Admin tab ----
with tabs[1]:
    st.subheader("Build or Update Index")
    if ADMIN_TOKEN:
        token_input = st.text_input("Admin token (required to build/update)", type="password")
        if token_input != ADMIN_TOKEN:
            st.warning("Enter the correct admin token to proceed.")
            st.stop()

    st.markdown("**Add sources**")
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

    st.markdown("---")
    st.markdown("**LLM status**")
    if PROVIDER == "azure":
        ok = bool(AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and AZURE_OPENAI_CHAT_DEPLOYMENT and AZURE_OPENAI_EMBED_DEPLOYMENT)
        st.write("Provider: Azure OpenAI")
        st.write("Configured:", "✅" if ok else "❌")
    else:
        ok = bool(OPENAI_API_KEY)
        st.write("Provider: OpenAI")
        st.write("Configured:", "✅" if ok else "❌")

    st.caption("Note: To change provider or models, set Streamlit secrets.")
