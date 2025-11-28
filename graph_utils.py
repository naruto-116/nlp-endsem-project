"""
Graph utility module for Knowledge Graph operations.
Handles graph construction, querying, and analysis.
"""
import pickle
import networkx as nx
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path


class LegalKnowledgeGraph:
    """
    Knowledge Graph for Indian Supreme Court judgments.
    Models cases as nodes and citations as directed edges.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.case_metadata = {}  # Store additional case information
        
    def add_case(self, case_id: str, case_name: str, metadata: Optional[Dict] = None):
        """
        Add a case node to the graph.
        
        Args:
            case_id: Unique identifier for the case
            case_name: Full case name
            metadata: Additional information (date, judges, etc.)
        """
        self.graph.add_node(case_id, name=case_name, **metadata if metadata else {})
        self.case_metadata[case_id] = metadata or {}
        
    def add_citation(self, citing_case: str, cited_case: str, relation: str = "CITES"):
        """
        Add a citation edge between two cases.
        
        Args:
            citing_case: ID of the case that cites
            cited_case: ID of the case being cited
            relation: Type of citation relationship
        """
        if citing_case in self.graph and cited_case in self.graph:
            self.graph.add_edge(citing_case, cited_case, relation=relation)
    
    def mark_overruled(self, case_id: str):
        """
        Mark a case as overruled (no longer valid law).
        
        Args:
            case_id: Case to mark as overruled
        """
        if case_id in self.graph:
            self.graph.nodes[case_id]['overruled'] = True
    
    def is_overruled(self, case_id: str) -> bool:
        """
        Check if a case has been overruled.
        
        Args:
            case_id: Case to check
            
        Returns:
            True if overruled, False otherwise
        """
        if case_id not in self.graph:
            return False
        return self.graph.nodes[case_id].get('overruled', False)
    
    def case_exists(self, case_id: str) -> bool:
        """
        Check if a case exists in the graph.
        
        Args:
            case_id: Case ID to check
            
        Returns:
            True if case exists
        """
        return case_id in self.graph
    
    def get_cited_by(self, case_id: str, limit: int = 10) -> List[str]:
        """
        Get cases that cite this case (incoming edges).
        
        Args:
            case_id: Case to find citations for
            limit: Maximum number of results
            
        Returns:
            List of case IDs that cite this case
        """
        if case_id not in self.graph:
            return []
        
        predecessors = list(self.graph.predecessors(case_id))
        return predecessors[:limit]
    
    def get_cites(self, case_id: str, limit: int = 10) -> List[str]:
        """
        Get cases cited by this case (outgoing edges).
        
        Args:
            case_id: Case to find citations from
            limit: Maximum number of results
            
        Returns:
            List of case IDs cited by this case
        """
        if case_id not in self.graph:
            return []
        
        successors = list(self.graph.successors(case_id))
        return successors[:limit]
    
    def get_landmark_cases(self, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Get most cited (landmark) cases based on in-degree.
        
        Args:
            top_n: Number of top cases to return
            
        Returns:
            List of (case_id, citation_count) tuples
        """
        in_degrees = dict(self.graph.in_degree())
        sorted_cases = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
        return sorted_cases[:top_n]
    
    def find_related_cases(self, case_id: str, depth: int = 2) -> Set[str]:
        """
        Find cases related to the given case within a certain graph distance.
        
        Args:
            case_id: Starting case
            depth: How many hops to traverse
            
        Returns:
            Set of related case IDs
        """
        if case_id not in self.graph:
            return set()
        
        related = set()
        current_level = {case_id}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                # Add predecessors (cases that cite this one)
                next_level.update(self.graph.predecessors(node))
                # Add successors (cases cited by this one)
                next_level.update(self.graph.successors(node))
            
            related.update(next_level)
            current_level = next_level
        
        related.discard(case_id)  # Remove the starting case
        return related
    
    def get_subgraph(self, case_ids: List[str]) -> nx.DiGraph:
        """
        Extract a subgraph containing only specified cases.
        
        Args:
            case_ids: List of case IDs to include
            
        Returns:
            NetworkX DiGraph subgraph
        """
        valid_ids = [cid for cid in case_ids if cid in self.graph]
        return self.graph.subgraph(valid_ids).copy()
    
    def get_graph_stats(self) -> Dict:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary with graph statistics
        """
        return {
            'num_cases': self.graph.number_of_nodes(),
            'num_citations': self.graph.number_of_edges(),
            'avg_citations_per_case': self.graph.number_of_edges() / max(self.graph.number_of_nodes(), 1),
            'num_overruled': sum(1 for n in self.graph.nodes() if self.graph.nodes[n].get('overruled', False)),
            'density': nx.density(self.graph)
        }
    
    def save(self, path: Path):
        """
        Save the graph to a pickle file.
        
        Args:
            path: File path to save to
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'metadata': self.case_metadata
            }, f)
        print(f"Graph saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'LegalKnowledgeGraph':
        """
        Load a graph from a pickle file.
        
        Args:
            path: File path to load from
            
        Returns:
            LegalKnowledgeGraph instance
        """
        kg = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
            kg.graph = data['graph']
            kg.case_metadata = data['metadata']
        print(f"Graph loaded from {path}")
        return kg
