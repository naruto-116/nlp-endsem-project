# KG-CiteRAG Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional)

## Step-by-Step Setup

### 1. Install Dependencies

Open PowerShell in the project directory and run:

```powershell
pip install -r requirements.txt
```

This will install all required packages including:
- NetworkX (for Knowledge Graph)
- FAISS (for vector search)
- Sentence Transformers (for embeddings)
- Streamlit (for UI)
- PyMuPDF (for PDF processing)
- Groq (for LLM API)

### 2. Configure API Keys

Create a `.env` file in the project root:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

**Get a free Groq API key:** https://console.groq.com/

### 3. Download ILDC Dataset

Option A: Download from official source
- Visit: https://github.com/Exploration-Lab/CJPE or HuggingFace
- Download `ILDC_single.jsonl`
- Place it in the `data/` folder

Option B: Create a sample dataset for testing

```powershell
python -c "from scripts.data_loader import create_sample_ildc_file; from pathlib import Path; import config; create_sample_ildc_file(config.ILDC_PATH, 100)"
```

### 4. Build Knowledge Graph

```powershell
python scripts/build_knowledge_graph.py
```

This will:
- Load the ILDC dataset
- Create nodes for each case
- Create edges for citations
- Save the graph to `data/graph.pickle`

Expected time: 2-5 minutes for 1,000 cases

### 5. Build Vector Index

```powershell
python scripts/build_vector_index.py
```

This will:
- Chunk all judgment texts
- Generate embeddings using MiniLM
- Create FAISS index
- Save to `data/ildc_vector_index.faiss`

Expected time: 5-10 minutes for 1,000 cases

### 6. Run the Application

```powershell
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Troubleshooting

### Import Errors

If you see import errors, ensure all packages are installed:

```powershell
pip install --upgrade -r requirements.txt
```

### FAISS Installation Issues

If FAISS fails to install, try:

```powershell
pip install faiss-cpu --no-cache-dir
```

For GPU support (optional):
```powershell
pip install faiss-gpu
```

### Groq API Errors

If you see API errors:
1. Check that your API key is correct in `.env`
2. Verify you have API credits
3. The system will use a dummy generator if no key is provided

### Memory Issues

If you run out of memory:
1. Reduce `SUBSET_SIZE` in config.py (try 500 or 250)
2. Reduce `CHUNK_SIZE` to 300
3. Close other applications

### File Not Found Errors

Ensure these files exist:
- `data/ILDC_single.jsonl` (or create sample)
- `data/graph.pickle` (run build_knowledge_graph.py)
- `data/ildc_vector_index.faiss` (run build_vector_index.py)
- `data/metadata.json` (created by build_vector_index.py)

## Performance Optimization

### For Development (Fast Testing)

In `config.py` or `.env`:
```
SUBSET_SIZE=100
CHUNK_SIZE=300
```

### For Production (Full Dataset)

```
SUBSET_SIZE=35000
CHUNK_SIZE=500
```

### Using Kaggle/Colab

1. Upload the project as a dataset
2. Save outputs to Kaggle Datasets or Google Drive
3. Mount drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Verification

After setup, verify everything works:

```powershell
# Test imports
python -c "import networkx, faiss, sentence_transformers, streamlit; print('All imports successful')"

# Check data files
python -c "import config; print('Graph exists:', config.GRAPH_PATH.exists()); print('Index exists:', config.VECTOR_INDEX_PATH.exists())"
```

## Next Steps

1. Try example queries in the UI
2. Upload a sample PDF document
3. Review the verification reports
4. Experiment with search parameters

## Getting Help

- Check the main README.md for project documentation
- Review error messages carefully
- Ensure all prerequisites are met
- Try with a smaller dataset first (SUBSET_SIZE=100)
