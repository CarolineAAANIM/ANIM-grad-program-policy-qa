
# ANIM Grad Program Policy Q&A App — Deployment Package (v2)

This package fixes the logo path issue by using **logo.png** and adds a **fallback placeholder** if the file is missing.

## Deployment Steps
1. Upload all files to a GitHub repo (root).
2. In Streamlit Cloud, create a new app pointing to `app.py`.
3. Share the generated URL. Faculty password: `ANIMGRAD`. Admin code: `ANIMADMIN`.

## Notes
- If you change the logo in Admin Panel, it will be saved as `logo.png`.
- If `logo.png` does not exist, the app shows a placeholder banner.
- Q&A history is persisted to `qa_logs.csv` and downloadable from Admin Panel.
