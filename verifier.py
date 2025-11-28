"""
Citation verification module.
Verifies that cited cases exist in the knowledge graph and checks validity.
"""
import re
from typing import List, Dict, Tuple

from src.graph_utils import LegalKnowledgeGraph
from src.citation_utils import normalize_case_name, extract_citations_from_text


class CitationVerifier:
    """
    Verifies citations in generated answers against the knowledge graph.
    """
    
    def __init__(self, graph: LegalKnowledgeGraph):
        """
        Initialize the verifier.
        
        Args:
            graph: Legal knowledge graph
        """
        self.graph = graph
        self.case_name_to_id = self._build_name_index()
        print(f"[OK] Citation verifier initialized with {len(self.case_name_to_id)} cases")
    
    def _build_name_index(self) -> Dict[str, str]:
        """
        Build an index of normalized case names to case IDs.
        
        Returns:
            Dictionary mapping normalized names to case IDs
        """
        index = {}
        for node_id in self.graph.graph.nodes():
            case_name = self.graph.graph.nodes[node_id].get('name', '')
            if case_name:
                normalized = normalize_case_name(case_name)
                index[normalized] = node_id
        return index
    
    def verify_citation(self, citation: str) -> Dict:
        """
        Verify a single citation against the knowledge graph.
        
        Args:
            citation: Case name to verify
            
        Returns:
            Verification result dictionary
        """
        # Special handling for uploaded PDFs (including page numbers)
        if 'Document' in citation or '.pdf' in citation or 'Article-21' in citation or '-page' in citation:
            # Extract page number if present
            page_info = ''
            if '-page' in citation:
                parts = citation.split('-page')
                if len(parts) > 1:
                    page_info = f" (Page {parts[1]})"
            
            return {
                'citation': citation,
                'verified': True,
                'exists': True,
                'status': f'Uploaded Document{page_info}',
                'case_id': citation,
                'overruled': False
            }
        
        normalized = normalize_case_name(citation)
        
        # Check if citation exists
        if normalized in self.case_name_to_id:
            case_id = self.case_name_to_id[normalized]
            
            # Check if overruled
            is_overruled = self.graph.is_overruled(case_id)
            
            return {
                'citation': citation,
                'exists': True,
                'case_id': case_id,
                'overruled': is_overruled,
                'status': 'Overruled' if is_overruled else 'Valid',
                'verified': True
            }
        else:
            return {
                'citation': citation,
                'exists': False,
                'case_id': None,
                'overruled': False,
                'status': 'Not Found',
                'verified': False
            }
    
    def verify_all_citations(self, text: str) -> List[Dict]:
        """
        Extract and verify all citations from text.
        
        Args:
            text: Generated answer text
            
        Returns:
            List of verification results
        """
        # Extract citations
        citations = extract_citations_from_text(text)
        
        # Verify each citation
        results = []
        for citation in citations:
            result = self.verify_citation(citation)
            results.append(result)
        
        return results
    
    def correct_answer(self, answer: str) -> Tuple[str, Dict]:
        """
        Correct an answer by verifying and annotating citations.
        
        Args:
            answer: Generated answer text with citations
            
        Returns:
            Tuple of (corrected_answer, verification_report)
        """
        # Verify all citations
        verification_results = self.verify_all_citations(answer)
        
        # Build correction
        corrected_answer = answer
        
        for result in verification_results:
            citation = result.get('citation', '')
            exists = result.get('exists', False)
            overruled = result.get('overruled', False)
            
            if not exists:
                # Replace non-existent citation with warning
                pattern = re.escape(f"[{citation}]")
                replacement = f"[{citation}] ⚠️ (Citation not verified)"
                corrected_answer = re.sub(pattern, replacement, corrected_answer)
            
            elif overruled:
                # Add warning for overruled cases
                pattern = re.escape(f"[{citation}]")
                replacement = f"[{citation}] ⚠️ (Overruled - No longer valid law)"
                corrected_answer = re.sub(pattern, replacement, corrected_answer)
        
        # Create verification report
        report = {
            'total_citations': len(verification_results),
            'verified': sum(1 for r in verification_results if r.get('exists', False)),
            'not_found': sum(1 for r in verification_results if not r.get('exists', False)),
            'overruled': sum(1 for r in verification_results if r.get('overruled', False)),
            'valid': sum(1 for r in verification_results if r.get('exists', False) and not r.get('overruled', False)),
            'hallucination_rate': sum(1 for r in verification_results if not r.get('exists', False)) / max(len(verification_results), 1),
            'details': verification_results
        }
        
        return corrected_answer, report
    
    def get_citation_context(self, citation: str, graph_depth: int = 1) -> Dict:
        """
        Get additional context about a citation from the graph.
        
        Args:
            citation: Case name
            graph_depth: How many hops to traverse
            
        Returns:
            Context dictionary with related cases
        """
        result = self.verify_citation(citation)
        
        if not result.get('exists', False):
            return {'error': 'Citation not found in knowledge graph'}
        
        case_id = result.get('case_id')
        
        # Get related cases
        related_cases = self.graph.find_related_cases(case_id, depth=graph_depth)
        
        # Get citing and cited cases
        cited_by = self.graph.get_cited_by(case_id, limit=5)
        cites = self.graph.get_cites(case_id, limit=5)
        
        return {
            'case_id': case_id,
            'citation': citation,
            'status': result['status'],
            'cited_by_count': len(list(self.graph.graph.predecessors(case_id))),
            'cites_count': len(list(self.graph.graph.successors(case_id))),
            'top_citing_cases': [
                self.graph.graph.nodes[cid].get('name', cid) for cid in cited_by
            ],
            'top_cited_cases': [
                self.graph.graph.nodes[cid].get('name', cid) for cid in cites
            ],
            'related_cases_count': len(related_cases)
        }
