# 📄 Document Upload & Management Guide

## Overview

Your KG-CiteRAG system now supports **persistent PDF document storage** with unified search across both the main ILDC dataset and your uploaded documents.

## Key Features

### ✅ What's New

1. **Persistent Storage**: Uploaded PDFs are saved permanently until you manually delete them
2. **Unified Search**: When you ask a question, the system searches BOTH:
   - Main ILDC dataset (100 Supreme Court cases)
   - Your uploaded PDFs (up to 5 documents)
3. **Source Attribution**: Retrieved documents show whether they're from ILDC or uploaded PDFs
4. **Document Management**: View and delete uploaded documents from the sidebar
5. **Automatic Chunking**: PDFs are automatically split into searchable chunks with embeddings

### 📊 Limits

- **Maximum Documents**: 5 PDFs at a time
- **Duplicate Detection**: System prevents uploading the same file twice (MD5 hash checking)
- **Storage Location**: `data/uploaded_pdfs/`

## How to Use

### 1. Upload a Document

1. Go to the **"📄 Upload Document"** tab
2. Click **"Choose a PDF file"** and select your document
3. Review the analysis (word count, estimated pages, citations found)
4. Click **"💾 Add to System (Persistent)"**
5. Wait for confirmation: "✅ Document added successfully!"

### 2. Search Across All Documents

1. Go to the **"🔍 Query"** tab
2. Enter your question (e.g., "Explain the right to privacy under Article 21")
3. Click **"🔍 Search & Generate Answer"**
4. The system will:
   - Search your uploaded PDFs
   - Search the main ILDC dataset
   - Combine results with weighted scoring
   - Generate an answer using all relevant sources

### 3. View Uploaded Documents

Check the **sidebar** under **"📄 Uploaded Documents"**:
- Shows current count (e.g., "2/5 documents")
- Lists each document with:
  - Filename (truncated)
  - Number of chunks
  - Delete button (🗑️)

### 4. Delete a Document

1. Find the document in the sidebar
2. Click the **🗑️** button next to it
3. Confirm deletion
4. The document is permanently removed from the system

## Technical Details

### How It Works

1. **PDF Processing**:
   - Text extraction using PyMuPDF
   - Chunking with 500-word chunks + 50-word overlap
   - Embedding generation using `all-MiniLM-L6-v2`

2. **Storage**:
   - FAISS vector index: `data/uploaded_pdfs/uploaded_docs_index.faiss`
   - Metadata JSON: `data/uploaded_pdfs/uploaded_docs_metadata.json`
   - Original PDFs: `data/uploaded_pdfs/{doc_id}_{filename}.pdf`

3. **Search Strategy**:
   - Query is encoded using the same embedding model
   - FAISS performs similarity search on uploaded docs (top 5)
   - Main hybrid retrieval searches ILDC dataset (configurable top_k)
   - Results are merged and ranked by score

4. **Source Tracking**:
   - Each chunk has metadata: `source_type`, `source_filename`, `doc_id`
   - UI displays: "🏛️ ILDC Dataset" or "📄 filename.pdf"

### Architecture

```
DocumentManager
├── add_document(filename, pdf_bytes)
│   ├── Extract text from PDF
│   ├── Create chunks
│   ├── Generate embeddings
│   ├── Add to FAISS index
│   └── Save metadata
│
├── delete_document(doc_id)
│   ├── Remove from metadata
│   ├── Rebuild FAISS index
│   └── Delete PDF file
│
├── search(query, top_k=5)
│   ├── Encode query
│   ├── FAISS similarity search
│   └── Return ranked results
│
└── list_documents()
    └── Return metadata list
```

## Example Workflow

### Scenario: You upload "Article-21_12-Feb-2025.pdf"

1. **Upload Phase**:
   ```
   Processing PDF... ✓
   - 6,104 words extracted
   - 29 citations found
   - 13 chunks created
   Click "Add to System" → Success!
   ```

2. **Query Phase**:
   ```
   Question: "Explain the right to privacy under Article 21"
   
   Retrieved Documents:
   1. Article-21_12-Feb-2025.pdf (Score: 0.892)
   2. K.S. Puttaswamy v. Union of India (Score: 0.856) [ILDC]
   3. Maneka Gandhi v. Union of India (Score: 0.782) [ILDC]
   
   Answer: [Generated using content from all 3 sources]
   ```

3. **Management Phase**:
   ```
   Sidebar shows:
   📄 Uploaded Documents (1/5)
   📄 Article-21_12-Feb-2...
      13 chunks [🗑️]
   ```

## Troubleshooting

### "Maximum 5 documents allowed"
- Delete one or more documents from the sidebar before uploading a new one

### "Document already exists in the system"
- The system detected a duplicate file (same MD5 hash)
- Delete the existing version first if you want to re-upload

### "Could not extract text from PDF"
- PDF might be image-based (scanned document)
- Try using OCR tools to create a searchable PDF first

### Documents not appearing in search results
- Check the "Retrieved Documents" expander in query results
- Verify document was successfully added (check sidebar count)
- Try more specific queries related to your document's content

## Best Practices

1. **Upload Strategy**: Upload key documents you'll reference frequently (contracts, landmark judgments, FIRs, etc.)
2. **Query Specificity**: Ask specific questions to get the most relevant chunks from your documents
3. **Document Management**: Periodically review and remove outdated documents to make room for new ones
4. **Source Verification**: Always check the "Retrieved Documents" section to see which sources were used

## API Integration

If you want to programmatically manage documents:

```python
from src.document_manager import DocumentManager

# Initialize
doc_manager = DocumentManager(
    storage_dir="data/uploaded_pdfs",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_documents=5
)

# Add document
with open("my_document.pdf", "rb") as f:
    result = doc_manager.add_document("my_document.pdf", f.read())
    print(result)  # {'success': True, 'doc_id': '...', 'num_chunks': 13}

# Search
results = doc_manager.search("privacy rights", top_k=5)
for r in results:
    print(f"{r['score']:.3f}: {r['text'][:100]}...")

# List documents
docs = doc_manager.list_documents()
print(f"Total: {len(docs)} documents")

# Delete
doc_manager.delete_document(doc_id="abc123...")
```

---

**Your system is now fully integrated!** 🎉

Upload PDFs, ask questions, and get answers from both your documents and the 100-case ILDC dataset.
