"""
Document manager for handling uploaded PDFs with persistent storage.
Manages PDF uploads, embeddings, and integration with main retrieval system.
"""
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from sentence_transformers import SentenceTransformer
from src.pdf_processor import PDFProcessor


class DocumentManager:
    """
    Manages uploaded documents with persistent storage.
    Handles PDF processing, embedding generation, and FAISS index management.
    """
    
    def __init__(self, storage_dir: Path, embedding_model_name: str, max_documents: int = 5):
        """
        Initialize the document manager.
        
        Args:
            storage_dir: Directory to store uploaded PDFs and metadata
            embedding_model_name: Name of sentence transformer model
            max_documents: Maximum number of documents to store
        """
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(exist_ok=True)
        
        self.pdfs_dir = storage_dir / "uploaded_pdfs"
        self.pdfs_dir.mkdir(exist_ok=True)
        
        self.index_path = storage_dir / "uploaded_docs_index.faiss"
        self.metadata_path = storage_dir / "uploaded_docs_metadata.json"
        
        self.max_documents = max_documents
        self.model = SentenceTransformer(embedding_model_name)
        self.pdf_processor = PDFProcessor()
        
        # Load existing index and metadata
        self.index = self._load_or_create_index()
        self.metadata = self._load_metadata()
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create new one."""
        if self.index_path.exists():
            try:
                return faiss.read_index(str(self.index_path))
            except:
                pass
        
        # Create new index
        embedding_dim = self.model.get_sentence_embedding_dimension()
        return faiss.IndexFlatL2(embedding_dim)
    
    def _load_metadata(self) -> List[Dict]:
        """Load document metadata from JSON file."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def _save_index(self):
        """Save FAISS index to disk."""
        faiss.write_index(self.index, str(self.index_path))
    
    def _save_metadata(self):
        """Save metadata to JSON file."""
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def _compute_file_hash(self, pdf_bytes: bytes) -> str:
        """Compute hash of PDF file to detect duplicates."""
        return hashlib.md5(pdf_bytes).hexdigest()
    
    def get_document_count(self) -> int:
        """Get number of uploaded documents."""
        return len(set(m['doc_id'] for m in self.metadata))
    
    def list_documents(self) -> List[Dict]:
        """Get list of all uploaded documents with their info."""
        docs = {}
        for meta in self.metadata:
            doc_id = meta['doc_id']
            if doc_id not in docs:
                docs[doc_id] = {
                    'doc_id': doc_id,
                    'filename': meta['filename'],
                    'upload_date': meta['upload_date'],
                    'num_chunks': 0,
                    'file_path': meta['file_path']
                }
            docs[doc_id]['num_chunks'] += 1
        
        return list(docs.values())
    
    def add_document(self, filename: str, pdf_bytes: bytes) -> Dict:
        """
        Add a new PDF document to the system.
        
        Args:
            filename: Original filename of the PDF
            pdf_bytes: PDF file content as bytes
            
        Returns:
            Dictionary with processing results
        """
        # Check document limit
        if self.get_document_count() >= self.max_documents:
            return {
                'success': False,
                'error': f'Maximum {self.max_documents} documents allowed. Please delete some documents first.'
            }
        
        # Check for duplicates
        file_hash = self._compute_file_hash(pdf_bytes)
        for meta in self.metadata:
            if meta.get('file_hash') == file_hash:
                return {
                    'success': False,
                    'error': f'Document already uploaded: {meta["filename"]}'
                }
        
        # Extract text from PDF
        text = self.pdf_processor.extract_text_from_bytes(pdf_bytes)
        if not text:
            return {
                'success': False,
                'error': 'Could not extract text from PDF'
            }
        
        # Analyze document
        analysis = self.pdf_processor.analyze_document_structure(text)
        
        # Save PDF file
        from datetime import datetime
        doc_id = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_hash[:8]}"
        pdf_path = self.pdfs_dir / f"{doc_id}.pdf"
        with open(pdf_path, 'wb') as f:
            f.write(pdf_bytes)
        
        # Chunk text
        chunks = self.pdf_processor.chunk_pdf_text(text, chunk_size=500, overlap=50)
        
        # Generate embeddings
        embeddings = self.model.encode(chunks, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        
        # Create metadata for each chunk
        upload_date = datetime.now().isoformat()
        for i, chunk in enumerate(chunks):
            self.metadata.append({
                'doc_id': doc_id,
                'filename': filename,
                'chunk_id': i,
                'text': chunk,
                'file_hash': file_hash,
                'upload_date': upload_date,
                'file_path': str(pdf_path),
                'source': 'uploaded',
                'doc_type': analysis['document_type'],
                'index_position': start_idx + i
            })
        
        # Save to disk
        self._save_index()
        self._save_metadata()
        
        return {
            'success': True,
            'doc_id': doc_id,
            'filename': filename,
            'num_chunks': len(chunks),
            'words': analysis['total_words'],
            'doc_type': analysis['document_type'],
            'citations_found': len(self.pdf_processor.extract_citations_from_pdf(text))
        }
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its chunks.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful
        """
        # Find all chunks for this document
        doc_metadata = [m for m in self.metadata if m['doc_id'] == doc_id]
        if not doc_metadata:
            return False
        
        # Get file path and delete PDF
        pdf_path = Path(doc_metadata[0]['file_path'])
        if pdf_path.exists():
            pdf_path.unlink()
        
        # Remove from metadata
        self.metadata = [m for m in self.metadata if m['doc_id'] != doc_id]
        
        # Rebuild FAISS index without this document
        if self.metadata:
            # Collect all remaining chunks
            remaining_texts = [m['text'] for m in self.metadata]
            embeddings = self.model.encode(remaining_texts, show_progress_bar=False)
            embeddings = np.array(embeddings).astype('float32')
            
            # Create new index
            embedding_dim = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.index.add(embeddings)
            
            # Update index positions
            for i, meta in enumerate(self.metadata):
                meta['index_position'] = i
        else:
            # No documents left, create empty index
            embedding_dim = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Save changes
        self._save_index()
        self._save_metadata()
        
        return True
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search uploaded documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        if self.index.ntotal == 0:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])[0].astype('float32')
        
        # Search
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            min(top_k, self.index.ntotal)
        )
        
        # Get results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                result = {
                    'doc_id': meta['doc_id'],
                    'case_name': meta['filename'],  # For compatibility with app.py
                    'text': meta['text'],
                    'score': float(1 / (1 + dist)),
                    'source_type': 'uploaded',
                    'source_filename': meta['filename'],
                    'chunk_id': meta['chunk_id']
                }
                results.append(result)
        
        return results
    
    def get_all_text(self) -> str:
        """Get all text from uploaded documents for context."""
        texts = []
        docs = {}
        
        for meta in self.metadata:
            doc_id = meta['doc_id']
            if doc_id not in docs:
                docs[doc_id] = {
                    'filename': meta['filename'],
                    'chunks': []
                }
            docs[doc_id]['chunks'].append(meta['text'])
        
        for doc_id, doc_info in docs.items():
            texts.append(f"\n--- Document: {doc_info['filename']} ---\n")
            texts.extend(doc_info['chunks'])
        
        return '\n'.join(texts)
