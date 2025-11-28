"""
PDF processing utilities for document upload feature.
"""
import re
from pathlib import Path
from typing import List, Dict, Set

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class PDFProcessor:
    """
    Processes uploaded PDF documents for legal analysis.
    """
    
    def __init__(self):
        if not fitz:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page in doc:
                text += page.get_text()
            
            doc.close()
            return text
            
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def extract_text_from_bytes(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes (for uploaded files).
        
        Args:
            pdf_bytes: PDF file bytes
            
        Returns:
            Extracted text
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            
            for page in doc:
                text += page.get_text()
            
            doc.close()
            return text
            
        except Exception as e:
            print(f"Error extracting text from PDF bytes: {e}")
            return ""
    
    def extract_text_with_pages(self, pdf_bytes: bytes) -> List[Dict]:
        """
        Extract text from PDF with page number tracking.
        
        Args:
            pdf_bytes: PDF file bytes
            
        Returns:
            List of dicts with 'page_num' and 'text' for each page
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = []
            
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text()
                if page_text.strip():
                    pages.append({
                        'page_num': page_num,
                        'text': page_text
                    })
            
            doc.close()
            return pages
            
        except Exception as e:
            print(f"Error extracting text with pages: {e}")
            return []
    
    def extract_citations_from_pdf(self, text: str) -> Set[str]:
        """
        Extract legal citations from PDF text.
        
        Args:
            text: Extracted PDF text
            
        Returns:
            Set of extracted citations
        """
        from src.citation_utils import extract_citations_from_judgment
        return extract_citations_from_judgment(text)
    
    def chunk_pdf_text(self, text: str, chunk_size: int = 500, 
                      overlap: int = 50) -> List[str]:
        """
        Chunk PDF text for embedding.
        
        Args:
            text: Full PDF text
            chunk_size: Words per chunk
            overlap: Overlapping words
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def chunk_pdf_with_pages(self, pages_data: List[Dict], chunk_size: int = 500,
                           overlap: int = 50) -> List[Dict]:
        """
        Chunk PDF text with page number tracking.
        
        Args:
            pages_data: List of dicts with 'page_num' and 'text'
            chunk_size: Words per chunk
            overlap: Overlapping words
            
        Returns:
            List of dicts with 'text' and 'page_num' for each chunk
        """
        chunks = []
        
        for page_data in pages_data:
            page_num = page_data['page_num']
            text = page_data['text']
            words = text.split()
            
            # Create chunks from this page
            for i in range(0, len(words), chunk_size - overlap):
                chunk_text = ' '.join(words[i:i + chunk_size])
                if chunk_text.strip():
                    chunks.append({
                        'text': chunk_text,
                        'page_num': page_num
                    })
        
        return chunks
    
    def analyze_document_structure(self, text: str) -> Dict:
        """
        Analyze the structure of a legal document.
        
        Args:
            text: Document text
            
        Returns:
            Analysis results
        """
        analysis = {
            'total_words': len(text.split()),
            'total_chars': len(text),
            'num_paragraphs': len([p for p in text.split('\n\n') if p.strip()]),
            'has_citations': bool(self.extract_citations_from_pdf(text)),
            'estimated_pages': len(text) // 3000,  # Rough estimate
        }
        
        # Look for common legal document sections
        sections = [
            'plaintiff', 'defendant', 'respondent', 'petitioner',
            'facts', 'issues', 'arguments', 'judgment', 'order',
            'held', 'ratio', 'obiter'
        ]
        
        found_sections = []
        text_lower = text.lower()
        for section in sections:
            if section in text_lower:
                found_sections.append(section)
        
        analysis['sections_found'] = found_sections
        analysis['document_type'] = self._infer_document_type(text_lower)
        
        return analysis
    
    def _infer_document_type(self, text_lower: str) -> str:
        """
        Infer the type of legal document.
        
        Args:
            text_lower: Lowercase document text
            
        Returns:
            Document type
        """
        if 'supreme court' in text_lower and 'judgment' in text_lower:
            return 'Supreme Court Judgment'
        elif 'petition' in text_lower or 'petitioner' in text_lower:
            return 'Petition/Application'
        elif 'contract' in text_lower or 'agreement' in text_lower:
            return 'Contract/Agreement'
        elif 'fir' in text_lower or 'first information report' in text_lower:
            return 'FIR/Police Report'
        elif 'notice' in text_lower:
            return 'Legal Notice'
        else:
            return 'General Legal Document'
