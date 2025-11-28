# 🚀 Complete GitHub + Streamlit Deployment Guide

## ✅ Step 1: Create GitHub Repository

1. **Go to GitHub**: https://github.com/new
2. **Fill in details**:
   - Repository name: `nlp-endsem-project`
   - Description: `KG-CiteRAG: Legal Question Answering System for Indian Supreme Court Law`
   - **Make it PUBLIC** (required for free Streamlit deployment)
   - **DON'T check** "Add a README file"
   - **DON'T check** "Add .gitignore"
   - **DON'T check** "Choose a license"
3. **Click**: "Create repository"

---

## ✅ Step 2: Push Your Code

After creating the repository, run this command:

```powershell
git push -u origin main
```

If it asks for credentials:
- **Username**: PKarthik109
- **Password**: Use a **Personal Access Token** (not your GitHub password)

### How to Create Personal Access Token:
1. Go to: https://github.com/settings/tokens
2. Click: "Generate new token" → "Generate new token (classic)"
3. Give it a name: "NLP Project Upload"
4. Select scopes: Check **"repo"** (all)
5. Click: "Generate token"
6. **COPY THE TOKEN** (you won't see it again!)
7. Use this token as password when pushing

---

## ✅ Step 3: Verify Upload

Go to: https://github.com/PKarthik109/nlp-endsem-project

You should see all your files!

---

## 🌐 Step 4: Deploy on Streamlit Cloud

### 4.1 Go to Streamlit Cloud
1. Visit: https://streamlit.io/cloud
2. Click: **"Sign in"** (use GitHub account)
3. Authorize Streamlit to access GitHub

### 4.2 Create New App
1. Click: **"New app"** button
2. **Repository**: Select `PKarthik109/nlp-endsem-project`
3. **Branch**: `main`
4. **Main file path**: `app.py`
5. **Advanced settings** (click to expand):
   - Python version: `3.10` or `3.11`
   - Add secrets if needed (API keys)

### 4.3 Deploy
1. Click: **"Deploy!"**
2. Wait 5-10 minutes (first deploy takes time)
3. Your app will be live at: `https://[app-name].streamlit.app`

---

## ⚠️ Important Notes for Streamlit Deployment

### Issue: Large Data Files

Your `data/` folder (~2GB) is too large for Streamlit Cloud (1GB limit).

### Solutions:

#### Option 1: Host Data on Google Drive (Recommended)
1. Upload your `data/` folder to Google Drive
2. Make it publicly accessible
3. Modify `config.py` to download data on startup:

```python
import os
from pathlib import Path
import gdown

DATA_DIR = Path("data")

# Download data if not exists
if not DATA_DIR.exists() or not (DATA_DIR / "ildc_vector_index.faiss").exists():
    print("📥 Downloading data files...")
    
    # Google Drive file IDs
    files = {
        "ildc_vector_index.faiss": "YOUR_GDRIVE_FILE_ID_1",
        "metadata.json": "YOUR_GDRIVE_FILE_ID_2",
        "entity_index.json": "YOUR_GDRIVE_FILE_ID_3",
        # Add more files...
    }
    
    DATA_DIR.mkdir(exist_ok=True)
    for filename, file_id in files.items():
        url = f"https://drive.google.com/uc?id={file_id}"
        output = DATA_DIR / filename
        gdown.download(url, str(output), quiet=False)
    
    print("✅ Data files downloaded!")
```

Add to `requirements.txt`:
```
gdown>=4.7.0
```

#### Option 2: Use HuggingFace Datasets
1. Upload data to HuggingFace Datasets: https://huggingface.co/datasets
2. Load in your app using `datasets` library

#### Option 3: Use Smaller Dataset
- Reduce number of cases to fit in 1GB
- Use only essential data files

---

## 🔑 Setting API Keys in Streamlit Cloud

Your `config.py` has API keys that shouldn't be public.

### In Streamlit Cloud:
1. Go to app settings (⚙️ icon)
2. Click: **"Secrets"**
3. Add your secrets in TOML format:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
GROQ_API_KEY = "your-groq-api-key"
```

### Update `config.py`:
```python
import os
import streamlit as st

# Try to get from Streamlit secrets, fallback to environment
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    API_KEY = os.getenv("GEMINI_API_KEY", "")
```

---

## 📊 Expected URLs After Deployment

**GitHub Repository**: 
https://github.com/PKarthik109/nlp-endsem-project

**Streamlit App**: 
https://nlp-endsem-project-[random-id].streamlit.app

---

## 🆘 Troubleshooting

### "Repository not found"
- Make sure you created the repository on GitHub first
- Use correct repository name: `nlp-endsem-project`

### "Permission denied"
- Use Personal Access Token as password, not GitHub password
- Token needs "repo" permissions

### "App crashes on Streamlit"
- Check logs in Streamlit Cloud dashboard
- Usually due to missing data files or dependencies
- Make sure all packages in requirements.txt

### "Out of memory on Streamlit"
- Free tier has 1GB RAM limit
- Reduce data size or upgrade to paid tier

---

## ✅ Current Status

✅ Git repository initialized locally  
✅ All code committed (78 files)  
✅ Remote configured: https://github.com/PKarthik109/nlp-endsem-project.git  
⏳ **NEXT**: Create GitHub repository and push  

---

## 🎯 Quick Checklist

- [ ] Create repository on GitHub (https://github.com/new)
- [ ] Push code: `git push -u origin main`
- [ ] Verify files on GitHub
- [ ] Sign in to Streamlit Cloud (https://streamlit.io/cloud)
- [ ] Click "New app"
- [ ] Select your repository
- [ ] Configure data files (Google Drive or HuggingFace)
- [ ] Add API keys in Streamlit secrets
- [ ] Click "Deploy"
- [ ] Wait 5-10 minutes
- [ ] Share your public URL!

---

**Follow the steps above in order. Start with Step 1: Create GitHub Repository** 🚀
