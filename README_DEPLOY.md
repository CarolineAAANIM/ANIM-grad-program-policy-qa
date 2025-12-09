
# ANIM Grad Program Policy Q&A App — Deployment Package

This ZIP contains everything you need to deploy a **faculty-only, browser-based** Policy Q&A app to **Streamlit Cloud** (or your server) with **no coding**.

## Files
- `app.py` — Streamlit application.
- `policy_index.json` — Prebuilt search index from your PDFs (includes section headings).
- `doc_meta.json` — App settings (faculty password, admin code, logo path, document list + URLs).
- `Screenshot 2025-11-19 120830.png` — Logo used on the app header.
- `ANIM_Program Guide_2025-26.pdf` — Source document.
- `ANIM_ Graduate Thesis Guidelines_2023-2024_revised 2023.pdf` — Source document.
- `ANIM_Review Guidelines_Revised 2024.pdf` — Source document.
- `requirements.txt` — Python dependencies for Streamlit Cloud.

## Quick Deploy to Streamlit Cloud (≈ 5 minutes)
1. Create (or sign in to) your Streamlit Cloud account: https://streamlit.io/cloud
2. Create a new **public GitHub repository** (e.g., `anim-grad-policy-qa`).
3. Upload **all files** from this ZIP to the repo **root**.
4. In Streamlit Cloud, click **New app** → select your repo → set **Main file path** to `app.py` → **Deploy**.
5. Share the generated URL with faculty. They enter **`ANIMGRAD`** to access.
6. Admins toggle **Admin mode** and enter **`ANIMADMIN`** to upload new PDFs, set URLs, change logo, and **download Q&A history**.

## Configuration
- Faculty password: `ANIMGRAD` (defined in `doc_meta.json`)
- Admin code: `ANIMADMIN` (defined in `doc_meta.json`)
- Logo path: `Screenshot 2025-11-19 120830.png` (can be changed in Admin Panel)
- Document URLs are already set in `doc_meta.json` and can be updated via Admin Panel.

## Notes
- A persistent **Q&A history** CSV (`qa_logs.csv`) will be created automatically in the app’s working directory.
- For PDF URLs, the app automatically adds `#page=<n>` when citing a specific page.
- You can replace the logo with any PNG/JPG under **Admin mode**.

## Optional: Private/Internal Hosting
If you prefer to deploy internally:
1. Ensure Python 3.10+ is installed.
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
4. Share the local/hosted URL; faculty use `ANIMGRAD`.

## Support
If you need help with domain mapping (e.g., `animgrad.scad.edu`) or role-based authentication later, we can add it without changing how faculty use the app.
