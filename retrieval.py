"""
Hybrid retrieval engine combining vector search and graph traversal.
"""
import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from src.graph_utils import LegalKnowledgeGraph
from src.citation_utils import normalize_case_name


class HybridRetriever:
    """
    Combines semantic vector search with knowledge graph traversal
    for more accurate legal document retrieval.
    """
    
    def __init__(self, 
                 index_path: Path,
                 metadata_path: Path,
                 graph_path: Path,
                 embedding_model_name: str):
        """
        Initialize the hybrid retriever.
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to chunk metadata JSON
            graph_path: Path to knowledge graph pickle
            embedding_model_name: Sentence transformer model name
        """
        print("Loading retrieval components...")
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        print(f"[OK] Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"[OK] Loaded metadata for {len(self.metadata)} chunks")
        
        # Load knowledge graph
        self.graph = LegalKnowledgeGraph.load(graph_path)
        print(f"[OK] Loaded knowledge graph")
        
        # Load embedding model
        self.model = SentenceTransformer(embedding_model_name)
        print(f"[OK] Loaded embedding model: {embedding_model_name}")
        
        # Build BM25 index for keyword search
        self._build_bm25_index()
        print(f"[OK] Built BM25 index for keyword search")
        
        # Load entity reverse index
        entity_index_path = Path(index_path).parent / "entity_index.json"
        if entity_index_path.exists():
            with open(entity_index_path, 'r', encoding='utf-8') as f:
                self.entity_index = json.load(f)
            print(f"[OK] Loaded entity index ({len(self.entity_index.get('articles', {}))} articles, {len(self.entity_index.get('sections', {}))} sections)")
        else:
            print(f"[WARNING] Entity index not found at {entity_index_path}, entity-based queries will be suboptimal")
            self.entity_index = {'articles': {}, 'sections': {}, 'acts': {}, 'case_name_to_id': {}}
        
        # Load bench index
        bench_index_path = Path(index_path).parent / "bench_index.json"
        if bench_index_path.exists():
            with open(bench_index_path, 'r', encoding='utf-8') as f:
                self.bench_index = json.load(f)
            print(f"[OK] Loaded bench index ({len(self.bench_index.get('judges', {}))} judges)")
        else:
            print(f"[WARNING] Bench index not found, judge-based queries will be suboptimal")
            self.bench_index = {'judges': {}}
    
    def _build_bm25_index(self):
        """Build BM25 index for keyword-based retrieval."""
        # Tokenize all documents
        corpus = []
        for meta in self.metadata:
            text = meta.get('text', '')
            case_name = meta.get('case_name', '')
            # Combine case name (10x weight) with text for better matching
            case_name_repeated = ' '.join([case_name] * 10)
            combined = f"{case_name_repeated} {text}"
            tokens = combined.lower().split()
            corpus.append(tokens)
        
        self.bm25 = BM25Okapi(corpus)
        self.bm25_corpus_tokens = corpus
    
    def _expand_query(self, query: str) -> str:
        """Expand query with legal synonyms and variations."""
        expansions = {
            'judgment': ['judgment', 'judgement', 'decision', 'ruling', 'order', 'held'],
            'interpret': ['interpret', 'interpretation', 'construed', 'construction', 'meaning'],
            'discuss': ['discuss', 'mentioned', 'cited', 'referred', 'addressed', 'considered'],
            'case': ['case', 'matter', 'appeal', 'petition', 'writ'],
            'article': ['article', 'art', 'art.'],
            'section': ['section', 'sec', 'sec.', 's.'],
        }
        
        expanded_terms = [query]
        query_lower = query.lower()
        
        for term, synonyms in expansions.items():
            if term in query_lower:
                for syn in synonyms[:3]:  # Add top 3 synonyms
                    if syn not in query_lower:
                        expanded_terms.append(syn)
        
        return ' '.join(expanded_terms)
    
    def _extract_entities_from_query(self, query: str) -> Dict[str, List[str]]:
        """Extract legal entities (articles, sections, case names, years, judges) from query."""
        entities = {
            'articles': [],
            'sections': [],
            'case_names': [],
            'years': [],
            'judges': []
        }
        
        # Extract Article references
        article_pattern = r'[Aa]rticle\s+(\d+[A-Za-z]?(?:\(\d+\))?)'
        entities['articles'] = re.findall(article_pattern, query)
        
        # Extract Section references
        section_pattern = r'[Ss]ection\s+(\d+[A-Za-z]?(?:\(\d+\))?)'
        entities['sections'] = re.findall(section_pattern, query)
        
        # Extract case name pattern (X v. Y or X vs Y)
        case_pattern = r'([A-Z][A-Za-z\s\.]+?)\s+(?:v\.|vs\.?|versus)\s+([A-Z][A-Za-z\s\.]+)'
        case_matches = re.findall(case_pattern, query)
        if case_matches:
            entities['case_names'] = [f"{m[0].strip()} v. {m[1].strip()}" for m in case_matches]
        
        # Extract years (e.g., "in 2021", "delivered in 2020")
        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        years = re.findall(year_pattern, query)
        entities['years'] = [int(y) for y in years]
        
        # Extract judge names (e.g., "JUSTICE HEMANT GUPTA")
        judge_pattern = r'(?:JUSTICE|J\.?)\s+([A-Z][A-Z\s\.]+?)(?:,|\s+AND|$)'
        judges = re.findall(judge_pattern, query, re.IGNORECASE)
        entities['judges'] = [j.strip() for j in judges if len(j.strip()) > 5]
        
        return entities
    
    def _metadata_filter(self, results: List[Dict], entities: Dict[str, List[str]]) -> List[Dict]:
        """Boost or filter results based on extracted entities."""
        if not any(entities.values()):
            return results  # No entities to filter on
        
        boosted_results = []
        
        for result in results:
            boost = 1.0
            metadata = result.get('metadata', {})
            text = result.get('text', '').lower()
            case_name = result.get('case_name', '').lower()
            
            # Boost for matching articles
            if entities['articles']:
                result_articles = metadata.get('entities', {}).get('articles', [])
                matching_articles = set(entities['articles']) & set(result_articles)
                if matching_articles:
                    boost *= (1.5 + 0.2 * len(matching_articles))
            
            # Boost for matching sections
            if entities['sections']:
                result_sections = metadata.get('entities', {}).get('sections', [])
                matching_sections = set(entities['sections']) & set(result_sections)
                if matching_sections:
                    boost *= (1.5 + 0.2 * len(matching_sections))
            
            # Strong boost for matching case names
            if entities['case_names']:
                for case_query in entities['case_names']:
                    if case_query.lower() in case_name or case_query.lower() in text[:500]:
                        boost *= 20.0  # Very strong boost for case name match (20x)
                        break
            
            result['score'] = result.get('score', 0) * boost
            boosted_results.append(result)
        
        # Re-sort by boosted scores
        boosted_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return boosted_results
    
    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Perform BM25 keyword search."""
        # Expand query
        expanded_query = self._expand_query(query)
        query_tokens = expanded_query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if idx < len(self.metadata) and scores[idx] > 0:
                result = self.metadata[idx].copy()
                result['score'] = float(scores[idx])
                result['source'] = 'bm25'
                results.append(result)
        
        return results
    
    def vector_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform semantic vector search with query expansion.
        
        Args:
            query: User query string
            top_k: Number of results to return
            
        Returns:
            List of retrieved chunks with metadata
        """
        # Expand query for better recall
        expanded_query = self._expand_query(query)
        
        # Encode query
        query_embedding = self.model.encode([expanded_query])[0].astype('float32')
        
        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), 
            top_k
        )
        
        # Retrieve metadata for results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(1 / (1 + dist))  # Convert distance to similarity
                result['source'] = 'vector'
                results.append(result)
        
        return results
    
    def graph_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform graph-based search using entity extraction and traversal.
        
        Args:
            query: User query string
            top_k: Number of results to return
            
        Returns:
            List of case IDs from graph traversal
        """
        # Extract potential case/entity references from query
        # For now, we'll find landmark cases related to query keywords
        
        results = []
        
        # Get landmark cases (highly cited)
        landmark_cases = self.graph.get_landmark_cases(top_n=top_k)
        
        for case_id, citation_count in landmark_cases:
            case_name = self.graph.graph.nodes[case_id].get('name', case_id)
            
            # Find chunks from this case in metadata
            case_chunks = [m for m in self.metadata if m['case_id'] == case_id]
            
            if case_chunks:
                # Take the first chunk as representative
                result = case_chunks[0].copy()
                result['score'] = citation_count / 100  # Normalize citation count
                result['source'] = 'graph'
                result['citation_count'] = citation_count
                results.append(result)
        
        return results[:top_k]
    
    def date_search(self, year: int, top_k: int = 50) -> List[Dict]:
        """Search for cases by year."""
        results = []
        for meta in self.metadata:
            date_str = meta.get('date', '')
            if str(year) in date_str:
                result = meta.copy()
                result['score'] = 1.0
                result['source'] = 'date_filter'
                results.append(result)
        
        return results[:top_k]
    
    def bench_search(self, judge_names: List[str], top_k: int = 50) -> List[Dict]:
        """Search for cases by judge names."""
        candidate_case_ids = set()
        
        # Match judges (case-insensitive partial matching)
        for query_judge in judge_names:
            query_judge_upper = query_judge.upper()
            for indexed_judge, case_ids in self.bench_index.get('judges', {}).items():
                # Check if query judge name appears in indexed judge name
                if query_judge_upper in indexed_judge.upper():
                    candidate_case_ids.update(case_ids)
        
        if not candidate_case_ids:
            return []
        
        # Get chunks from matching cases
        results = []
        case_chunks = {}
        for meta in self.metadata:
            case_id = meta.get('case_id')
            if case_id in candidate_case_ids:
                if case_id not in case_chunks:
                    case_chunks[case_id] = []
                case_chunks[case_id].append(meta)
        
        # Take first few chunks from each case
        for case_id, chunks in case_chunks.items():
            for chunk in chunks[:2]:  # Max 2 chunks per case
                result = chunk.copy()
                result['score'] = 1.0
                result['source'] = 'bench_filter'
                results.append(result)
        
        return results[:top_k]
    
    def date_search(self, year: int, top_k: int = 50) -> List[Dict]:
        """Search for cases by year."""
        results = []
        for meta in self.metadata:
            date_str = meta.get('date', '')
            if str(year) in date_str:
                result = meta.copy()
                result['score'] = 1.0
                result['source'] = 'date_filter'
                results.append(result)
        
        return results[:top_k]
    
    def bench_search(self, judge_names: List[str], top_k: int = 50) -> List[Dict]:
        """Search for cases by judge names."""
        candidate_case_ids = set()
        
        # Match judges (case-insensitive partial matching)
        for query_judge in judge_names:
            query_judge_upper = query_judge.upper()
            for indexed_judge, case_ids in self.bench_index.get('judges', {}).items():
                # Check if query judge name appears in indexed judge name
                if query_judge_upper in indexed_judge.upper():
                    candidate_case_ids.update(case_ids)
        
        if not candidate_case_ids:
            return []
        
        # Get chunks from matching cases
        results = []
        case_chunks = {}
        for meta in self.metadata:
            case_id = meta.get('case_id')
            if case_id in candidate_case_ids:
                if case_id not in case_chunks:
                    case_chunks[case_id] = []
                case_chunks[case_id].append(meta)
        
        # Take first few chunks from each case
        for case_id, chunks in case_chunks.items():
            for chunk in chunks[:2]:  # Max 2 chunks per case
                result = chunk.copy()
                result['score'] = 1.0
                result['source'] = 'bench_filter'
                results.append(result)
        
        return results[:top_k]
    
    def entity_search(self, entities: Dict[str, List[str]], top_k: int = 10) -> List[Dict]:
        """Search using entity reverse index for precise Article/Section matching."""
        candidate_case_ids = set()
        entity_scores = {}  # Track which entities each case matches
        
        # Get cases for each article mentioned
        for article in entities.get('articles', []):
            if article in self.entity_index['articles']:
                case_ids = self.entity_index['articles'][article]
                candidate_case_ids.update(case_ids)
                for cid in case_ids:
                    if cid not in entity_scores:
                        entity_scores[cid] = {'articles': 0, 'sections': 0}
                    entity_scores[cid]['articles'] += 1
        
        # Get cases for each section mentioned
        for section in entities.get('sections', []):
            if section in self.entity_index['sections']:
                case_ids = self.entity_index['sections'][section]
                candidate_case_ids.update(case_ids)
                for cid in case_ids:
                    if cid not in entity_scores:
                        entity_scores[cid] = {'articles': 0, 'sections': 0}
                    entity_scores[cid]['sections'] += 1
        
        if not candidate_case_ids:
            return []
        
        # Get metadata for candidate cases
        # Group chunks by case_id and take best ones
        case_chunks = {}
        for meta in self.metadata:
            case_id = meta.get('case_id')
            if case_id in candidate_case_ids:
                if case_id not in case_chunks:
                    case_chunks[case_id] = []
                case_chunks[case_id].append(meta)
        
        # Create results with proper scoring
        results = []
        for case_id, chunks in case_chunks.items():
            # Score based on entity matches
            scores = entity_scores.get(case_id, {'articles': 0, 'sections': 0})
            score = scores['articles'] * 2.0 + scores['sections'] * 1.5
            
            # Take first few chunks from this case
            for chunk in chunks[:3]:  # Max 3 chunks per case
                result = chunk.copy()
                result['score'] = score
                result['source'] = 'entity_index'
                results.append(result)
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return results[:top_k]
    
    def hybrid_search(self, query: str, top_k: int = 10, 
                     vector_weight: float = 0.5,
                     bm25_weight: float = 0.3,
                     graph_weight: float = 0.2) -> List[Dict]:
        """
        Combine vector, BM25, and graph search with weighted fusion and metadata filtering.
        
        Args:
            query: User query string
            top_k: Number of final results
            vector_weight: Weight for vector search (default 0.5)
            bm25_weight: Weight for BM25 keyword search (default 0.3)
            graph_weight: Weight for graph search (default 0.2)
            
        Returns:
            Fused, filtered, and reranked results
        """
        # Extract entities for metadata filtering
        entities = self._extract_entities_from_query(query)
        
        # Check if this is an entity-focused query
        has_entities = bool(entities.get('articles') or entities.get('sections'))
        has_year = bool(entities.get('years'))
        has_judges = bool(entities.get('judges'))
        
        # Get results from all methods
        vector_results = self.vector_search(query, top_k=top_k * 3)
        bm25_results = self.bm25_search(query, top_k=top_k * 2)
        graph_results = self.graph_search(query, top_k=top_k)
        
        # For entity queries, add entity-based results with high weight
        entity_results = []
        if has_entities:
            entity_results = self.entity_search(entities, top_k=top_k * 2)
        
        # For year queries, add date-filtered results
        date_results = []
        if has_year:
            for year in entities['years']:
                date_results.extend(self.date_search(year, top_k=top_k * 2))
        
        # For judge queries, add bench-filtered results
        bench_results = []
        if has_judges:
            bench_results = self.bench_search(entities['judges'], top_k=top_k * 2)
        
        # Combine results with Reciprocal Rank Fusion (RRF)
        case_scores = {}
        k_rrf = 60  # RRF constant
        
        # Add date filter scores with HIGHEST priority for year queries
        for rank, result in enumerate(date_results, 1):
            case_id = result.get('case_id')
            if not case_id:
                continue
                
            if case_id not in case_scores:
                case_scores[case_id] = {
                    'vector_score': 0,
                    'bm25_score': 0,
                    'graph_score': 0,
                    'entity_score': 0,
                    'date_score': 0,
                    'bench_score': 0,
                    'data': result
                }
            case_scores[case_id]['date_score'] += 3.0 / (k_rrf + rank)  # 3x multiplier
        
        # Add bench filter scores with HIGHEST priority for judge queries
        for rank, result in enumerate(bench_results, 1):
            case_id = result.get('case_id')
            if not case_id:
                continue
                
            if case_id not in case_scores:
                case_scores[case_id] = {
                    'vector_score': 0,
                    'bm25_score': 0,
                    'graph_score': 0,
                    'entity_score': 0,
                    'date_score': 0,
                    'bench_score': 0,
                    'data': result
                }
            case_scores[case_id]['bench_score'] += 3.0 / (k_rrf + rank)  # 3x multiplier
        
        # Add entity search scores with HIGHEST priority (RRF)
        for rank, result in enumerate(entity_results, 1):
            case_id = result.get('case_id')
            if not case_id:
                continue
                
            if case_id not in case_scores:
                case_scores[case_id] = {
                    'vector_score': 0,
                    'bm25_score': 0,
                    'graph_score': 0,
                    'entity_score': 0,
                    'date_score': 0,
                    'bench_score': 0,
                    'data': result
                }
            # Entity matches get VERY high scores
            case_scores[case_id]['entity_score'] += 2.0 / (k_rrf + rank)  # 2x multiplier
        
        # Add vector search scores with RRF
        for rank, result in enumerate(vector_results, 1):
            case_id = result.get('case_id')
            if not case_id:
                continue
                
            if case_id not in case_scores:
                case_scores[case_id] = {
                    'vector_score': 0,
                    'bm25_score': 0,
                    'graph_score': 0,
                    'entity_score': 0,
                    'date_score': 0,
                    'bench_score': 0,
                    'data': result
                }
            # RRF scoring: 1 / (k + rank)
            case_scores[case_id]['vector_score'] += 1 / (k_rrf + rank)
        
        # Add BM25 scores with RRF
        for rank, result in enumerate(bm25_results, 1):
            case_id = result.get('case_id')
            if not case_id:
                continue
                
            if case_id not in case_scores:
                case_scores[case_id] = {
                    'vector_score': 0,
                    'date_score': 0,
                    'bench_score': 0,
                    'bm25_score': 0,
                    'graph_score': 0,
                    'entity_score': 0,
                    'data': result
                }
            case_scores[case_id]['bm25_score'] += 1 / (k_rrf + rank)
        
        # Add graph search scores with RRF
        for rank, result in enumerate(graph_results, 1):
            case_id = result.get('case_id')
            if not case_id:
                continue
                
            if case_id not in case_scores:
                case_scores[case_id] = {
                    'vector_score': 0,
                    'bm25_score': 0,
                    'graph_score': 0,
                    'entity_score': 0,
                    'date_score': 0,
                    'bench_score': 0,
                    'data': result
                }
            case_scores[case_id]['graph_score'] += 1 / (k_rrf + rank)
        
        # Calculate combined scores with dynamic weighting
        # Prioritize specialized search methods when query has specific criteria
        date_weight = 0.6 if has_year else 0.0
        bench_weight = 0.6 if has_judges else 0.0
        entity_weight = 0.5 if has_entities else 0.0
        
        # Reduce base weights when specialized search is active
        total_special_weight = date_weight + bench_weight + entity_weight
        base_multiplier = max(0.1, 1 - total_special_weight)
        
        adjusted_vector_weight = vector_weight * base_multiplier
        adjusted_bm25_weight = bm25_weight * base_multiplier
        adjusted_graph_weight = graph_weight * base_multiplier
        
        for case_id in case_scores:
            case_scores[case_id]['combined_score'] = (
                adjusted_vector_weight * case_scores[case_id]['vector_score'] +
                adjusted_bm25_weight * case_scores[case_id]['bm25_score'] +
                adjusted_graph_weight * case_scores[case_id]['graph_score'] +
                entity_weight * case_scores[case_id]['entity_score'] +
                date_weight * case_scores[case_id].get('date_score', 0) +
                bench_weight * case_scores[case_id].get('bench_score', 0)
            )
        
        # Sort by combined score
        sorted_cases = sorted(
            case_scores.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        # Return top results with metadata
        results = []
        for case_id, scores in sorted_cases[:top_k * 2]:  # Get 2x for filtering
            result = scores['data'].copy()
            result['combined_score'] = scores['combined_score']
            result['vector_score'] = scores['vector_score']
            result['bm25_score'] = scores['bm25_score']
            result['graph_score'] = scores['graph_score']
            result['entity_score'] = scores['entity_score']
            result['date_score'] = scores.get('date_score', 0)
            result['bench_score'] = scores.get('bench_score', 0)
            results.append(result)
        
        # Apply metadata filtering/boosting
        results = self._metadata_filter(results, entities)
        
        return results[:top_k]
    
    def get_context_for_generation(self, results: List[Dict], 
                                   max_tokens: int = 2000) -> str:
        """
        Format retrieved results as context for LLM generation.
        
        Args:
            results: List of retrieved chunks
            max_tokens: Maximum context length (approximate)
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant documents found."
            
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(results, 1):
            chunk_text = result.get('text', '')
            case_name = result.get('case_name', 'Unknown Case')
            source_type = result.get('source_type', 'ildc')
            
            if not chunk_text:
                continue
            
            # Estimate tokens (rough approximation: 1 token ≈ 4 chars)
            chunk_length = len(chunk_text) // 4
            
            if total_length + chunk_length > max_tokens:
                break
            
            # Format with page number for uploaded PDFs
            if source_type == 'uploaded_pdf':
                page_num = result.get('page_num', 1)
                filename = result.get('source_filename', case_name)
                citation = f"{filename}-page{page_num}"
                context_parts.append(
                    f"[Document {i}] Source: [{citation}]\n{chunk_text}\n"
                )
            else:
                context_parts.append(
                    f"[Document {i}] Case: {case_name}\n{chunk_text}\n"
                )
            total_length += chunk_length
        
        return "\n---\n".join(context_parts) if context_parts else "No context available."
