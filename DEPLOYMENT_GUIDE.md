# Deployment Guide for KG-CiteRAG

## 🎯 Quick Summary

Your project now has **TWO working interfaces**:
1. ✅ **Streamlit** (`app.py`) - Original interface
2. ✅ **Gradio** (`app_gradio.py`) - Alternative interface with identical features

---

## 🚀 Option 1: Deploy on GitHub + Streamlit Cloud (RECOMMENDED)

### Step 1: Create GitHub Repository

```powershell
# Navigate to your project
cd "C:\Users\pkart\OneDrive\Desktop\NLP END sem project"

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: KG-CiteRAG Legal QA System"

# Create repository on GitHub (do this on github.com):
# 1. Go to https://github.com/new
# 2. Name: "kg-citerag-legal-qa"
# 3. Make it Public (required for free Streamlit deployment)
# 4. Click "Create repository"

# Connect to GitHub (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/kg-citerag-legal-qa.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Click "New app"
3. Select your repository: `YOUR_USERNAME/kg-citerag-legal-qa`
4. Main file path: `app.py`
5. Click "Deploy"

**⚠️ Important:** Your app will be publicly accessible. The large data files (~2GB) may cause deployment issues. See Option 3 for solutions.

---

## 🌐 Option 2: Deploy Gradio with HuggingFace Spaces (FREE & EASY)

### Step 1: Push to GitHub (same as above)

### Step 2: Deploy on HuggingFace

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Name: `kg-citerag-legal-qa`
4. Select SDK: **Gradio**
5. Select Hardware: **CPU basic (free)**
6. Make it **Public**
7. Click "Create Space"

### Step 3: Connect GitHub to HuggingFace

1. In your Space settings, go to "Files and versions"
2. Click "⚙️ Settings"
3. Scroll to "Repository secrets"
4. Add your API keys if needed

### Step 4: Upload Files

Option A - Manual:
- Upload `app_gradio.py` as `app.py` (HuggingFace looks for app.py)
- Upload `requirements.txt`
- Upload your data files

Option B - Git:
```powershell
git clone https://huggingface.co/spaces/YOUR_USERNAME/kg-citerag-legal-qa
cd kg-citerag-legal-qa
# Copy your files here
git add .
git commit -m "Deploy KG-CiteRAG"
git push
```

**Advantage:** Gradio on HuggingFace automatically generates a public URL!

---

## 💻 Option 3: Run Locally (NO Deployment Needed)

### Run Streamlit (Original)
```powershell
cd "C:\Users\pkart\OneDrive\Desktop\NLP END sem project"
streamlit run app.py
```
Access at: http://localhost:8501

### Run Gradio (New Alternative)
```powershell
cd "C:\Users\pkart\OneDrive\Desktop\NLP END sem project"
python app_gradio.py
```
Access at: http://localhost:7860

**To share temporarily:**
Edit `app_gradio.py` line 600: Change `share=False` to `share=True`
This generates a public URL valid for 72 hours!

---

## 📦 Option 4: Deploy with Docker (Self-Hosted)

### Create Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501 7860

CMD ["streamlit", "run", "app.py"]
```

### Deploy:
```powershell
docker build -t kg-citerag .
docker run -p 8501:8501 kg-citerag
```

---

## 🔧 Troubleshooting Deployment Issues

### Issue: "Large files (>100MB) rejected by GitHub"

**Solution 1: Use Git LFS**
```powershell
git lfs install
git lfs track "*.faiss"
git lfs track "*.json"
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

**Solution 2: Host data files separately**
- Upload data files to Google Drive / Dropbox / AWS S3
- Modify `config.py` to download data on startup
- Keep data files in `.gitignore`

### Issue: "Streamlit deployment fails (out of memory)"

**Solution:** Use smaller dataset or upgrade to paid tier
- Free tier: 1GB RAM limit
- Your FAISS index + data ≈ 2GB
- **Recommendation:** Use Gradio + HuggingFace (2GB free tier)

### Issue: "Module not found errors"

**Solution:** Ensure all dependencies in `requirements.txt`
```powershell
pip freeze > requirements.txt
```

---

## 🎯 RECOMMENDED DEPLOYMENT PATH

For your project size and requirements:

### **Best Option: Gradio + HuggingFace Spaces**

✅ **Why:**
1. FREE 2GB RAM (vs Streamlit's 1GB)
2. Automatic public URL (no GitHub needed)
3. Easy setup (just upload files)
4. Built-in sharing features
5. Better for large models/data

### **Quick Deploy (5 minutes):**

1. Go to https://huggingface.co/spaces
2. Create Space (SDK: Gradio, Hardware: CPU basic)
3. Upload these files:
   - Rename `app_gradio.py` → `app.py`
   - `requirements.txt`
   - All your `src/` folder
   - All your `data/` folder
   - `config.py`
4. Wait 2-3 minutes for build
5. Done! Public URL: `https://huggingface.co/spaces/YOUR_USERNAME/kg-citerag-legal-qa`

---

## 📝 Changes Made to Your Project

### ✅ Streamlit UI (`app.py`)
- ✅ Removed vector weight slider (fixed at 0.7)
- ✅ Removed graph weight display
- ✅ Removed "Dataset" section from About tab
- ✅ Removed "Technology Stack" section
- ✅ Removed "Validity Tracking" feature line
- ✅ Removed "Overruled" metric from citation report

### ✅ Gradio UI (`app_gradio.py`)
- ✅ Complete alternative interface
- ✅ Same functionality as Streamlit
- ✅ Identical features (query, upload, verify)
- ✅ Better for deployment (lightweight)
- ✅ Automatic public URL with `share=True`

### ✅ Requirements
- ✅ Added `gradio>=4.0.0` to `requirements.txt`
- ✅ Installed gradio in your environment

---

## 🚀 Quick Test Commands

### Test Streamlit:
```powershell
streamlit run app.py
```

### Test Gradio:
```powershell
python app_gradio.py
```

### Test with Public URL (Gradio):
Edit `app_gradio.py` line 600: `share=True`, then:
```powershell
python app_gradio.py
```
Copy the public URL that appears (valid 72 hours)!

---

## 📞 Support

If you encounter issues:
1. Check error messages in terminal
2. Verify all data files exist in `data/` folder
3. Ensure API keys are set in `config.py`
4. Check disk space (need ~3GB free)

---

**Ready to deploy! Choose your option above.** 🚀
