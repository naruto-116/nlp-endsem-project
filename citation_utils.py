"""
Utility module for citation normalization and extraction.
Handles messy case name variations and standardizes citations.
"""
import re
import string
from typing import List, Set


def normalize_case_name(case_name: str) -> str:
    """
    Normalize case name for consistent matching.
    
    Example:
        "Kesavananda Bharati vs State of Kerala" 
        -> "kesavananda bharati state of kerala"
    
    Args:
        case_name: Raw case name from judgment or citation
        
    Returns:
        Normalized case name (lowercase, no punctuation)
    """
    if not case_name:
        return ""
    
    # Convert to lowercase
    normalized = case_name.lower()
    
    # Remove common legal separators
    normalized = re.sub(r'\s+vs\.?\s+|\s+v\.?\s+', ' ', normalized)
    
    # Remove punctuation
    normalized = normalized.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized


def extract_citations_from_text(text: str, pattern: str = r'\[(.*?)\]') -> List[str]:
    """
    Extract case citations from generated text.
    
    Args:
        text: Generated answer text containing citations in [brackets]
        pattern: Regex pattern to extract citations
        
    Returns:
        List of extracted case names
    """
    matches = re.findall(pattern, text)
    return [match.strip() for match in matches if match.strip()]


def extract_citations_from_judgment(judgment_text: str, max_citations: int = 50) -> Set[str]:
    """
    Extract case citations from a judgment text using common legal patterns.
    
    Looks for patterns like:
    - "In re [Case Name]"
    - "[Case Name] v. [Party]"
    - "AIR [Year] SC [Number]"
    - "[Party] v/vs [Party]"
    - Capitalized case references
    
    Args:
        judgment_text: Full text of a legal judgment
        max_citations: Maximum number of citations to extract (default 50)
        
    Returns:
        Set of extracted case names
    """
    citations = set()
    
    # Pattern 1: Case names with "v." or "vs." - Very common in Indian judgments
    # Matches: "State of Kerala v. N.M. Thomas" or "Kesavananda Bharati vs State of Kerala"
    pattern1 = r'\b([A-Z][A-Za-z\s&\.\(\),]+?)\s+v[s]?\.\s+([A-Z][A-Za-z\s&\.\(\),]+?)(?=\s*(?:[\(\[]\d{4}|AIR|SCC|\,|\.|;|and|held|wherein|in|$))'
    matches1 = re.findall(pattern1, judgment_text, re.MULTILINE)
    for match in matches1:
        party1 = match[0].strip().rstrip('.')
        party2 = match[1].strip().rstrip('.')
        # Filter out very short or very long names
        if 5 < len(party1) < 100 and 5 < len(party2) < 100:
            citation = f"{party1} v. {party2}"
            citations.add(citation)
            if len(citations) >= max_citations:
                break
    
    # Pattern 2: AIR citations (All India Reporter)
    pattern2 = r'AIR\s+\d{4}\s+(?:SC|SCC)\s+\d+'
    matches2 = re.findall(pattern2, judgment_text)
    citations.update(matches2[:20])  # Limit AIR citations
    
    # Pattern 3: SCC citations (Supreme Court Cases)
    pattern3 = r'\(\d{4}\)\s+\d+\s+SCC\s+\d+'
    matches3 = re.findall(pattern3, judgment_text)
    citations.update(matches3[:20])  # Limit SCC citations
    
    # Pattern 4: "In re [Case Name]" pattern
    pattern4 = r'[Ii]n\s+(?:re|the matter of)\s+([A-Z][A-Za-z\s&\.\(\),]{10,80})'
    matches4 = re.findall(pattern4, judgment_text)
    for match in matches4:
        if len(match.strip()) > 10:
            citations.add(f"In re {match.strip()}")
    
    return citations


def validate_case_name(case_name: str) -> bool:
    """
    Check if a string looks like a valid case name.
    
    Args:
        case_name: Case name to validate
        
    Returns:
        True if it looks like a valid case name
    """
    if not case_name or len(case_name) < 5:
        return False
    
    # Should contain at least one capital letter
    if not any(c.isupper() for c in case_name):
        return False
    
    # Should contain at least one space (party vs party)
    if ' ' not in case_name:
        return False
    
    return True


def fuzzy_match_case_name(query_case: str, candidate_cases: List[str], threshold: float = 0.8) -> str:
    """
    Find best matching case name from a list using fuzzy matching.
    
    Args:
        query_case: Case name to match
        candidate_cases: List of case names to search
        threshold: Minimum similarity score (0-1)
        
    Returns:
        Best matching case name or empty string if no match
    """
    from difflib import SequenceMatcher
    
    normalized_query = normalize_case_name(query_case)
    best_match = ""
    best_score = 0.0
    
    for candidate in candidate_cases:
        normalized_candidate = normalize_case_name(candidate)
        score = SequenceMatcher(None, normalized_query, normalized_candidate).ratio()
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match = candidate
    
    return best_match
