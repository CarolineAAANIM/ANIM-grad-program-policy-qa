
# ANIM Grad Program Policy Q&A App — Deployment Package (v3)

**This version adds Word (.docx) support**, improved PDF text extraction (pdfminer fallback), `st.rerun()` compatibility, an **Index Health** panel, and a **substring fallback** when BM25 returns no results.

## Deploy Steps
1. Create a GitHub repo and upload all files.
2. In Streamlit Cloud, create a new app pointing to `app.py`.
3. Share your generated URL; Faculty password: `ANIMGRAD`, Admin code: `ANIMADMIN`.

## Runtime Uploads
- Upload **PDF or DOCX** via Admin mode. PDFs use `pdfminer.six` first and fall back to PyPDF2; DOCX uses `python-docx`.
- Associate URLs for each document to enable "Read more" links.

## Debugging
- Use the **Index Health** expander to confirm `policy_index.json` and runtime chunks are loaded.
- If you see 0 chunks, ensure `policy_index.json` exists in the repo root and that uploaded files contain text (DOCX or text-layer PDFs).
