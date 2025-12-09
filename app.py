
import json, math, re, csv
from pathlib import Path
from datetime import datetime
import streamlit as st
from PyPDF2 import PdfReader

st.set_page_config(page_title="ANIM Grad Program Policy Q&A App", layout="wide")

LOG_PATH = Path("qa_logs.csv")

# Initialize log file with header if missing
if not LOG_PATH.exists():
    with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","question","answer_summary","document","page","section","url"]) 

# Load config & index
@st.cache_data(show_spinner=False)
def load_meta():
    p = Path("doc_meta.json")
    if p.exists():
        return json.loads(p.read_text())
    return {"password": "ANIMGRAD", "admin_code": "ANIMADMIN", "logo_path": "logo.png", "docs": []}

@st.cache_data(show_spinner=False)
def load_index():
    p = Path("policy_index.json")
    if p.exists():
        return json.loads(p.read_text())
    return {"chunks": [], "documents": [], "stats": {"N":1, "avgdl":1}, "df": {}}

meta = load_meta()
index = load_index()
N = index["stats"].get("N", 1)
avgdl = index["stats"].get("avgdl", 1)
DF = index.get("df", {})
chunks = index.get("chunks", [])

# Password gate (faculty-only)
if "authed" not in st.session_state:
    st.session_state.authed = False

if not st.session_state.authed:
    st.image(meta.get("logo_path"), width=300, use_column_width=False)
    st.title("ANIM Grad Program Policy Q&A App")
    pw = st.text_input("Enter faculty password", type="password")
    if st.button("Enter"):
        if pw == meta.get("password", "ANIMGRAD"):
            st.session_state.authed = True
            st.experimental_rerun()
        else:
            st.error("Incorrect password")
    st.stop()

# Header with logo
cols = st.columns([1,4])
with cols[0]:
    st.image(meta.get("logo_path"), width=180, use_column_width=False)
with cols[1]:
    st.title("ANIM Grad Program Policy Q&A App")
    st.caption("Answers are grounded in uploaded policy documents. Citations include document, page, and section heading.")

# Admin mode (URL mapping + uploads + logo + log export)
st.sidebar.header("Admin Panel")
admin_toggle = st.sidebar.toggle("Admin mode", value=False)
admin_ok = False
if admin_toggle:
    ac = st.sidebar.text_input("Admin code", type="password")
    if ac == meta.get("admin_code", "ANIMADMIN"):
        admin_ok = True
    else:
        st.sidebar.info("Enter admin code to unlock admin features.")

# Utility
word_re = re.compile(r"[A-Za-z][A-Za-z0-9_'\\-]+")
def tokenize(text):
    return [w.lower() for w in word_re.findall(text)]

k1 = 1.5
b = 0.75
def bm25_score(query_tokens, chunk):
    score = 0.0
    tf = {}
    for t in chunk["tokens"]:
        tf[t] = tf.get(t, 0) + 1
    for q in query_tokens:
        df = DF.get(q, 0)
        if df == 0:
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        fq = tf.get(q, 0)
        denom = fq + k1 * (1 - b + b * (chunk["length"] / (avgdl if avgdl else 1)))
        score += idf * (fq * (k1 + 1)) / (denom if denom else 1)
    return score

# Runtime uploads (admin only)
runtime_chunks = []
if admin_ok:
    st.sidebar.subheader("Upload new policy PDFs")
    uploaded = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    existing_names = [d["file"] for d in meta.get("docs", [])]

    if uploaded:
        for uf in uploaded:
            try:
                reader = PdfReader(uf)
                pages = []
                for i,p in enumerate(reader.pages):
                    t = p.extract_text() or ""
                    t = re.sub(r"\\s+", " ", t).strip()
                    pages.append((i+1, t))
                for page_no, text in pages:
                    section = None
                    for cand in re.split(r"(?<=\\.)\\s|\\n", text):
                        if any(k.lower() in cand.lower() for k in ["Thesis","Review","Guidelines","Degree","Program","Preparation","Defense","Submission","Formatting","Timeline"]):
                            section = cand.strip(); break
                    start = 0
                    CHUNK_SIZE = 800
                    OVERLAP = 200
                    while start < len(text):
                        end = start + CHUNK_SIZE
                        chunk_text = text[start:end]
                        if len(chunk_text) < 100:
                            break
                        tokens = tokenize(chunk_text)
                        runtime_chunks.append({
                            "doc": uf.name,
                            "page": page_no,
                            "section": section or "(section heading not detected)",
                            "text": chunk_text,
                            "tokens": tokens,
                            "length": len(tokens),
                        })
                        start = end - OVERLAP
                if uf.name not in existing_names:
                    meta["docs"].append({"file": uf.name, "display": uf.name, "url": ""})
            except Exception as e:
                st.sidebar.error(f"Failed to read {uf.name}: {e}")

    # URL mapping editor
    st.sidebar.subheader("Associate a URL for each document")
    changed = False
    for i, d in enumerate(meta.get("docs", [])):
        new_url = st.sidebar.text_input(f"URL for: {d['display']}", value=d.get("url", ""))
        if new_url != d.get("url", ""):
            meta["docs"][i]["url"] = new_url
            changed = True
    # Logo upload
    st.sidebar.subheader("Branding")
    logo = st.sidebar.file_uploader("Upload logo image", type=["png","jpg","jpeg"], accept_multiple_files=False)
    if logo:
        Path("logo.png").write_bytes(logo.getbuffer())
        meta["logo_path"] = "logo.png"
        changed = True
    if changed:
        Path("doc_meta.json").write_text(json.dumps(meta, indent=2))
        st.sidebar.success("Saved")

    # Q&A History export / clear
    st.sidebar.subheader("Q&A History")
    if LOG_PATH.exists():
        data = LOG_PATH.read_bytes()
        st.sidebar.download_button("Download CSV", data=data, file_name="qa_history.csv", mime="text/csv")
    else:
        st.sidebar.info("No history yet.")
    if st.sidebar.button("Clear history"):
        try:
            LOG_PATH.unlink(missing_ok=True)
            with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp","question","answer_summary","document","page","section","url"]) 
            st.sidebar.success("History cleared")
        except Exception as e:
            st.sidebar.error(f"Could not clear history: {e}")

# What's Included list
st.subheader("What's Included")
for d in meta.get("docs", []):
    url = d.get("url", "")
    if url:
        st.markdown(f"- **{d['display']}** — {url}")
    else:
        st.markdown(f"- **{d['display']}** — *(no URL provided yet)*")

st.divider()

# Search
query = st.text_input("Ask a policy question")
search_chunks = chunks + runtime_chunks

if query:
    q_tokens = tokenize(query)
    scored = []
    for ch in search_chunks:
        s = bm25_score(q_tokens, ch)
        if s > 0:
            scored.append((s, ch))
    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        st.info("No relevant passages found. Try different wording.")
        with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([datetime.utcnow().isoformat(), query, "(no results)", "", "", "", ""]) 
    else:
        topn = scored[:3]
        st.subheader("Answer & citations")
        import textwrap
        answer_summary = textwrap.shorten(topn[0][1]["text"], width=500, placeholder="...")

        for i, (score, ch) in enumerate(topn, start=1):
            base_url = None
            for d in meta.get("docs", []):
                if d["file"] == ch["doc"] or d["display"] == Path(ch["doc"]).name:
                    base_url = d.get("url")
                    break
            link = None
            if base_url:
                if base_url.lower().endswith(".pdf"):
                    link = f"{base_url}#page={ch['page']}"
                else:
                    link = base_url
            st.markdown(f"**{i}. Source:** {Path(ch['doc']).name} — page {ch['page']} — section: _{ch['section']}_")
            if link:
                st.markdown(f"**Read more:** {link}")
            st.write(ch["text"]) 
            st.caption(f"Score: {score:.3f}")

        st.divider()
        st.markdown("**Draft summary (prototype):**")
        st.write(answer_summary)

        top = topn[0][1]
        top_doc = Path(top["doc"]).name
        top_page = top["page"]
        top_section = top.get("section") or ""
        base_url = None
        for d in meta.get("docs", []):
            if d["file"] == top["doc"] or d["display"] == Path(top["doc"]).name:
                base_url = d.get("url")
                break
        link = None
        if base_url:
            link = f"{base_url}#page={top_page}" if base_url.lower().endswith(".pdf") else base_url
        with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([datetime.utcnow().isoformat(), query, answer_summary, top_doc, top_page, top_section, link or ""]) 

st.divider()
st.caption("Prototype: faculty-only access via password; admin can upload docs, set URLs, branding, and export Q&A history.")
