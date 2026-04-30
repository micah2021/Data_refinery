# Step 5 — Deploy to Streamlit Cloud (Free, Shareable Link)

Follow these steps exactly. Takes about 10 minutes total.

---

## Prerequisites checklist
- [ ] GitHub account (free at github.com)
- [ ] Streamlit Cloud account (free at share.streamlit.io — sign in with GitHub)
- [ ] Your project running locally (you already have this ✅)

---

## Step A — Prepare your repository

In VS Code terminal (venv activated):

```powershell
# 1. Initialise git if not done yet
git init

# 2. Create .gitignore — IMPORTANT: never commit nigeria.db or .env
@"
venv/
.env
*.db
__pycache__/
*.pyc
models/*.joblib
models/shap_plots/
models/*.pkl
.qodo/
"@ | Out-File -Encoding utf8 .gitignore

# 3. Create requirements.txt from your venv
pip freeze > requirements.txt

# 4. Stage and commit everything
git add .
git commit -m "Initial commit — Nigeria Health AI Data Refinery"
```

---

## Step B — Push to GitHub

```powershell
# Create a new repo on github.com first (call it: nigeria-health-ai)
# Then connect and push:

git remote add origin https://github.com/YOUR_USERNAME/nigeria-health-ai.git
git branch -M main
git push -u origin main
```

---

## Step C — Handle nigeria.db on the cloud

The database is too large for GitHub (160MB > 100MB limit).
Two options — pick one:

### Option 1 (Recommended) — Seed the DB in the cloud
Add a `startup.py` that the cloud app runs before starting:

```python
# startup.py — runs on Streamlit Cloud to build nigeria.db from scratch
import subprocess, sys, os

if not os.path.exists("nigeria.db"):
    print("Seeding database...")
    subprocess.run([sys.executable, "setup_db.py"], check=True)
    subprocess.run([sys.executable, "fix_socioeconomic.py"], check=True)
    subprocess.run([sys.executable, "ndhs_maternal_collector.py",
                    "--source", "synthetic"], check=True)
    subprocess.run([sys.executable, "build_feature_store.py"], check=True)
    print("Database ready.")
```

Then add to the top of `app.py`:
```python
import startup   # runs DB seeding on first launch
```

### Option 2 — Use Git LFS (Large File Storage)
```powershell
# Install Git LFS
git lfs install
git lfs track "*.db"
git add .gitattributes
git add nigeria.db
git commit -m "Add nigeria.db via LFS"
git push
```
Note: GitHub LFS free tier = 1GB storage + 1GB bandwidth/month.

---

## Step D — Deploy on Streamlit Cloud

1. Go to **share.streamlit.io**
2. Click **"New app"**
3. Fill in:
   - **Repository:** `YOUR_USERNAME/nigeria-health-ai`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Advanced settings"** → add Secrets (equivalent to your .env):
   ```toml
   DB_PATH = "./nigeria.db"
   LOG_LEVEL = "INFO"
   ```
5. Click **"Deploy"**

Streamlit Cloud will:
- Install packages from `requirements.txt` automatically
- Run your `app.py`
- Give you a URL like: `https://YOUR_USERNAME-nigeria-health-ai.streamlit.app`

---

## Step E — Share your link

Your dashboard will be publicly accessible at:
```
https://YOUR_USERNAME-nigeria-health-ai.streamlit.app
```

Include this URL in your:
- PhD thesis (as a deployable research artefact)
- Research paper (as a supplementary material link)
- Conference presentations
- NGO/government presentations

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError` | Check `requirements.txt` has all packages |
| `FileNotFoundError: nigeria.db` | Use Option 1 (startup.py seeding) |
| App crashes on first load | Check Streamlit Cloud logs → "Manage app" → "Logs" |
| `statsmodels` not found | Add `statsmodels` to requirements.txt |
| Slow first load | Normal — DB seeding takes ~2 min on first launch |

---

## Keep it updated

Every time you push to GitHub, Streamlit Cloud redeploys automatically:
```powershell
git add .
git commit -m "Update model / add feature"
git push
```
That's it. Your live dashboard updates within ~60 seconds.

