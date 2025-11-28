# 🚀 Quick Start Guide - KG-CiteRAG

## ✅ You Have 2 Working Interfaces

### 🎯 **Streamlit** (Original)
- Professional interface
- Full features
- Best for data exploration

### 🌟 **Gradio** (New - Better for Deployment)
- Modern, beautiful UI
- Lightweight
- Can generate public URLs
- Better for sharing/demos

---

## 🚀 **How to Run**

### **Option 1: Streamlit** 
```powershell
streamlit run app.py
```
→ Opens at http://localhost:8501

### **Option 2: Gradio (Local Only)**
**Method A - Command Line:**
```powershell
python app_gradio.py
```

**Method B - Double Click:**
- Double-click `start_gradio.bat`

**Method C - PowerShell:**
```powershell
.\start_gradio.ps1
```
→ Opens at http://localhost:7860 (or next available port)

### **Option 3: Gradio (With Public URL)** 🌐
**Method A - Command Line:**
```powershell
python launch_gradio_public.py
```

**Method B - Double Click:**
- Double-click `start_gradio_public.bat`

**Method C - PowerShell:**
```powershell
.\start_gradio_public.ps1
```
→ Generates public URL like: `https://xxxxx.gradio.live`
→ Valid for 72 hours!

---

## 📁 **All Launch Files**

| File | What It Does | How to Use |
|------|-------------|-----------|
| `app.py` | Streamlit UI | `streamlit run app.py` |
| `app_gradio.py` | Gradio UI (local) | `python app_gradio.py` |
| `launch_gradio_public.py` | Gradio with public URL | `python launch_gradio_public.py` |
| `start_gradio.bat` | Easy Gradio launcher (Windows) | Double-click |
| `start_gradio_public.bat` | Easy public URL launcher | Double-click |
| `start_gradio.ps1` | PowerShell launcher | `.\start_gradio.ps1` |
| `start_gradio_public.ps1` | PowerShell public launcher | `.\start_gradio_public.ps1` |

---

## 🎯 **Choose Your Method**

### **For Local Testing:**
✅ **Easiest:** Double-click `start_gradio.bat`  
✅ **Command line:** `python app_gradio.py`  
✅ **Streamlit:** `streamlit run app.py`

### **For Sharing/Demo:**
✅ **Easiest:** Double-click `start_gradio_public.bat`  
✅ **Command line:** `python launch_gradio_public.py`  
✅ **Copy the public URL and share!**

### **For Deployment:**
✅ **Best:** HuggingFace Spaces (see DEPLOYMENT_GUIDE.md)  
✅ **Alternative:** Streamlit Cloud (requires GitHub)

---

## 🔧 **Troubleshooting**

### **"Port already in use" error**
✅ **Solution:** The scripts now auto-detect available ports!  
- If 7860 is busy, it uses 7861, 7862, etc.
- Or use `start_gradio.ps1` to kill existing processes

### **"Module not found" error**
```powershell
pip install -r requirements.txt
```

### **"System not loading" error**
- Check that `data/` folder exists with all files
- Verify `config.py` has correct paths

---

## 📊 **Feature Comparison**

| Feature | Streamlit | Gradio |
|---------|-----------|--------|
| Legal Q&A | ✅ | ✅ |
| Upload PDFs | ✅ | ✅ |
| Citation Check | ✅ | ✅ |
| Knowledge Graph | ✅ | ✅ |
| Beautiful UI | ✅ | ✅ |
| **Public URL** | ❌ | ✅ |
| **Auto Port** | ❌ | ✅ |
| Easy Share | ⚠️ | ✅ |

---

## 🌟 **Quick Demo Setup**

**For Presentation/Demo (5 seconds):**
1. Double-click `start_gradio_public.bat`
2. Wait 30 seconds
3. Copy the `https://xxxxx.gradio.live` URL
4. Share with anyone!

**For Local Testing (3 seconds):**
1. Double-click `start_gradio.bat`
2. Browser opens automatically
3. Start querying!

---

## 💡 **Tips**

✅ **Gradio is recommended** for most use cases (faster, easier to share)  
✅ **Streamlit is better** if you prefer its look and feel  
✅ **Public URLs** are perfect for demos but expire in 72 hours  
✅ **HuggingFace Spaces** for permanent deployment (FREE!)

---

## 📞 **Need Help?**

- **Deployment:** Read `DEPLOYMENT_GUIDE.md`
- **Setup:** Read `SETUP_COMPLETE.md`
- **Features:** Read `RESEARCH_PAPER_COMPLETE_GUIDE.md`

---

**Everything is ready! Pick your favorite method and start using your Legal QA system! 🎉**
