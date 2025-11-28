# GitHub Upload Instructions

## ✅ Step 1: Create GitHub Repository

1. Go to https://github.com
2. Click the **"+" icon** in top right → **"New repository"**
3. Fill in:
   - **Repository name**: `nlp-endsem-project`
   - **Description**: "KG-CiteRAG: Legal Question Answering System for Indian Supreme Court Law"
   - **Visibility**: Public (required for free Streamlit deployment)
   - **DON'T initialize** with README (we already have one)
4. Click **"Create repository"**

---

## ✅ Step 2: Push Your Code

After creating the repository, GitHub will show you commands. Use these:

### If you see the repository URL:

```powershell
# Connect to your GitHub repo (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/nlp-endsem-project.git

# Rename branch to main (GitHub default)
git branch -M main

# Push your code
git push -u origin main
```

---

## ✅ Step 3: Verify Upload

1. Go to your repository: `https://github.com/YOUR_USERNAME/nlp-endsem-project`
2. You should see all your files uploaded
3. README.md will be displayed automatically

---

## 📊 What Was Uploaded

✅ **Source Code** (app.py, src/, scripts/)
✅ **Documentation** (all .md files)
✅ **Requirements** (requirements.txt)
✅ **Configuration** (config.py, .gitignore)
✅ **Visualizations** (all .png files)
✅ **Scripts** (setup.ps1, test files)

❌ **NOT Uploaded** (in .gitignore):
- Large data files (data/ folder)
- Python cache (__pycache__/)
- Virtual environments
- API keys

---

## 🚀 Next Steps After Upload

### Option 1: Deploy on Streamlit Cloud
1. Go to https://streamlit.io/cloud
2. Click "New app"
3. Select your repository: `YOUR_USERNAME/nlp-endsem-project`
4. Main file: `app.py`
5. Click "Deploy"

**Note**: You'll need to upload data files separately or configure them to download on startup.

### Option 2: Share GitHub Link
- Share: `https://github.com/YOUR_USERNAME/nlp-endsem-project`
- Others can clone and run locally

---

## 🔑 Important Notes

1. **Data Files**: The large data files (FAISS index, metadata) are NOT in the repo (too large)
   - You can upload them to Google Drive
   - Or use Git LFS for large files
   - Or host on HuggingFace Datasets

2. **API Keys**: config.py is in .gitignore (API keys not exposed)
   - Users need to add their own API keys

3. **Requirements**: All dependencies listed in requirements.txt

---

## 📝 Your Repository Info

**Repository Name**: nlp-endsem-project  
**URL**: https://github.com/YOUR_USERNAME/nlp-endsem-project  
**Branch**: main  
**Files**: 78 files committed  
**Commit Message**: "Initial commit: KG-CiteRAG Legal QA System - NLP End Semester Project"

---

## ✅ Current Status

✅ Git repository initialized  
✅ All Gradio files removed  
✅ All code committed to git  
✅ Ready to push to GitHub  

**Next**: Create GitHub repository and run the push commands above!

---

## 🆘 Troubleshooting

### Error: "Authentication failed"
- Use personal access token instead of password
- Go to GitHub Settings → Developer settings → Personal access tokens
- Generate new token with "repo" permissions
- Use token as password when pushing

### Error: "Repository already exists"
- Either use a different name
- Or delete the existing repository and recreate

### Error: "Large files rejected"
- This shouldn't happen (data/ folder is in .gitignore)
- If it does, check .gitignore includes data/

---

**Ready to upload! Follow Step 1 above to create your GitHub repository.** 🚀
