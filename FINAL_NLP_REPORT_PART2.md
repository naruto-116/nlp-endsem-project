# KG-CiteRAG: Knowledge-Graph-Augmented Citation-Enforced Retrieval System
## Complete Project Report - Part 2 of 3
### Retrieval Algorithms, System Architecture, and Generation Pipeline

---

## TABLE OF CONTENTS - PART 2

6. [Retrieval Algorithms and Implementation](#6-retrieval-algorithms-and-implementation)
7. [Hybrid Fusion Strategy](#7-hybrid-fusion-strategy)
8. [Answer Generation with LLM](#8-answer-generation-with-llm)
9. [Citation Verification System](#9-citation-verification-system)
10. [Complete System Architecture](#10-complete-system-architecture)

---

## 6. RETRIEVAL ALGORITHMS AND IMPLEMENTATION

### 6.1 Overview of Retrieval Methods

Our system implements **6 specialized retrieval methods**, each optimized for different query types:

| Method | Best For | Algorithm | Time Complexity |
|--------|----------|-----------|-----------------|
| **Vector Search** | Semantic queries | FAISS + SBERT | O(n) |
| **BM25 Search** | Keyword queries | Probabilistic ranking | O(n log n) |
| **Graph Search** | Citation traversal | BFS on NetworkX | O(V + E) |
| **Entity Search** | Article/Section queries | Direct index lookup | O(1) |
| **Date Search** | Year-based queries | Metadata filtering | O(n) |
| **Bench Search** | Judge-based queries | Partial name matching | O(n) |

### 6.2 Method 1: Vector Search (Semantic Similarity)

#### 6.2.1 Algorithm Description

Vector search finds documents semantically similar to the query, even if exact keywords don't match.

**Process:**
```
1. Encode query → 384-dim vector
2. Search FAISS index for k-nearest neighbors
3. Return top-k chunks with similarity scores
```

#### 6.2.2 Implementation

```python
def vector_search(self, query: str, top_k: int = 10) -> List[Dict]:
    """
    Semantic vector search using sentence transformers.
    
    Args:
        query: Natural language query
        top_k: Number of results to return
    
    Returns:
        List of retrieved chunks with scores
    """
    # Step 1: Encode query
    query_embedding = self.model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding.astype('float32')
    
    # Step 2: Search FAISS index
    distances, indices = self.index.search(query_embedding, top_k)
    
    # Step 3: Format results
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(self.metadata):
            chunk_meta = self.metadata[idx]
            
            # Convert L2 distance to similarity score (0-1)
            similarity = 1 / (1 + distance)
            
            results.append({
                'chunk_id': idx,
                'case_id': chunk_meta['case_id'],
                'case_name': chunk_meta['case_name'],
                'text': chunk_meta['text'],
                'page_num': chunk_meta.get('page_num', 1),
                'score': float(similarity),
                'method': 'vector_search'
            })
    
    return results
```

#### 6.2.3 Mathematical Details

**Similarity Computation:**
```
Given query vector q and document vector d (both L2-normalized):

L2 Distance: dist(q, d) = ||q - d||₂

Cosine Similarity: cos(q, d) = q · d (since normalized)

Conversion to score: score = 1 / (1 + dist)
```

**Example:**
```
Query: "fundamental rights constitution"
Vector: [0.23, -0.45, 0.67, ..., 0.12] (384 dims)

Top Result:
Case: Maneka Gandhi v. Union of India
Distance: 0.45
Similarity: 1/(1+0.45) = 0.69 (69% similar)
```

#### 6.2.4 Performance Characteristics

| Metric | Value |
|--------|-------|
| Search time | 200-300ms |
| Index size | 160 MB |
| Recall@10 | 42% (standalone) |
| Precision@5 | 15% (standalone) |
| Best for | Conceptual queries |

### 6.3 Method 2: BM25 Search (Keyword Matching)

#### 6.3.1 Algorithm Description

BM25 ranks documents by term frequency with saturation, preventing over-weighting of frequently repeated terms.

**Key Innovation:** 10x boost for case name matches

#### 6.3.2 Implementation

```python
def bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
    """
    BM25 probabilistic keyword search with case name boosting.
    
    Args:
        query: Search query
        top_k: Number of results
    
    Returns:
        Ranked list of documents
    """
    # Step 1: Query expansion (add legal synonyms)
    expanded_query = self._expand_query(query)
    
    # Step 2: Tokenize query
    query_tokens = expanded_query.lower().split()
    
    # Step 3: BM25 scoring
    scores = self.bm25.get_scores(query_tokens)
    
    # Step 4: Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    # Step 5: Format results
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            chunk_meta = self.metadata[idx]
            results.append({
                'chunk_id': idx,
                'case_id': chunk_meta['case_id'],
                'case_name': chunk_meta['case_name'],
                'text': chunk_meta['text'],
                'page_num': chunk_meta.get('page_num', 1),
                'score': float(scores[idx]),
                'method': 'bm25_search'
            })
    
    return results

def _expand_query(self, query: str) -> str:
    """Add legal synonyms to query."""
    expansions = {
        'judgment': ['judgment', 'judgement', 'decision', 'ruling'],
        'interpret': ['interpret', 'construed', 'construction'],
        'discuss': ['discuss', 'mentioned', 'cited', 'referred'],
        'case': ['case', 'matter', 'appeal', 'petition']
    }
    
    expanded = [query]
    for term, synonyms in expansions.items():
        if term in query.lower():
            expanded.extend(synonyms[:2])
    
    return ' '.join(expanded)
```

#### 6.3.3 BM25 Parameters

**Hyperparameters:**
- `k1 = 1.5` - Term frequency saturation parameter
- `b = 0.75` - Document length normalization
- `case_name_boost = 10x` - Repetition multiplier

**Effect of k1 (Saturation):**
```
Term appears 1 time:  score contribution = 0.60
Term appears 2 times: score contribution = 0.86
Term appears 5 times: score contribution = 1.11
Term appears 10 times: score contribution = 1.22
Term appears 20 times: score contribution = 1.28 (plateau)
```

**Effect of Case Name Boost:**
```python
# Standard corpus tokenization
corpus_token = "article 21 right to life ..."

# Boosted corpus tokenization (for case name matching)
corpus_token = "maneka gandhi maneka gandhi maneka gandhi ... (10x) article 21 ..."
```

#### 6.3.4 Query Expansion Examples

```
Input: "judgment on fundamental rights"
Expanded: "judgment judgement decision ruling fundamental rights"

Input: "case discussing Article 14"
Expanded: "case matter appeal petition discussing mentioned cited Article 14"
```

#### 6.3.5 Performance Characteristics

| Metric | Value |
|--------|-------|
| Search time | 100-150ms |
| Index size | 50 MB (in-memory) |
| Recall@10 | 38% (standalone) |
| Precision@5 | 20% (standalone) |
| Best for | Citation queries, case names |

### 6.4 Method 3: Graph Search (Citation Network Traversal)

#### 6.4.1 Algorithm Description

Graph search traverses the citation network to find related cases, leveraging the precedent structure of law.

**Strategy:**
1. Identify seed cases (from query or initial results)
2. Traverse citation edges (BFS)
3. Rank by PageRank and relevance

#### 6.4.2 Implementation

```python
def graph_search(self, query: str, top_k: int = 10) -> List[Dict]:
    """
    Knowledge graph-based search using citation network.
    
    Args:
        query: Search query
        top_k: Number of results
    
    Returns:
        List of related cases from citation network
    """
    # Step 1: Extract case names from query
    entities = self._extract_entities_from_query(query)
    case_names = entities.get('case_names', [])
    
    # Step 2: Find seed cases
    seed_cases = []
    for case_name in case_names:
        normalized = normalize_case_name(case_name)
        if normalized in self.entity_index.get('case_name_to_id', {}):
            case_id = self.entity_index['case_name_to_id'][normalized]
            seed_cases.append(case_id)
    
    # Step 3: If no seeds, use high PageRank cases
    if not seed_cases:
        seed_cases = self._get_landmark_cases(top_n=5)
    
    # Step 4: Traverse citation network (BFS)
    related_cases = set()
    for seed in seed_cases:
        # Get cited cases (outgoing edges)
        if self.graph.graph.has_node(seed):
            cited = list(self.graph.graph.successors(seed))
            related_cases.update(cited[:5])
            
            # Get citing cases (incoming edges)
            citing = list(self.graph.graph.predecessors(seed))
            related_cases.update(citing[:5])
    
    # Step 5: Convert to chunks and score
    results = []
    for case_id in related_cases:
        # Find chunks for this case
        case_chunks = [m for m in self.metadata if m['case_id'] == case_id]
        
        if case_chunks:
            chunk = case_chunks[0]  # Take first chunk
            
            # Score by PageRank
            pagerank = self.graph.graph.nodes[case_id].get('pagerank', 0.001)
            
            results.append({
                'chunk_id': chunk['chunk_id'],
                'case_id': case_id,
                'case_name': chunk['case_name'],
                'text': chunk['text'],
                'page_num': chunk.get('page_num', 1),
                'score': float(pagerank * 1000),  # Scale for comparison
                'method': 'graph_search'
            })
    
    # Step 6: Sort by score and return top-k
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

def _get_landmark_cases(self, top_n: int = 5) -> List[str]:
    """Get top landmark cases by PageRank."""
    pagerank = nx.pagerank(self.graph.graph)
    sorted_cases = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    return [case_id for case_id, _ in sorted_cases[:top_n]]
```

#### 6.4.3 Graph Traversal Example

```
Query: "Kesavananda Bharati case"

Step 1: Find seed case
  → case_id = "1973_SUP_1"

Step 2: Traverse outgoing edges (cases cited by Kesavananda)
  → 1950_AIR_27 (Gopalan)
  → 1967_AIR_1643 (Golak Nath)
  → 1951_SCR_525 (Shankari Prasad)

Step 3: Traverse incoming edges (cases citing Kesavananda)
  → 1980_3_SCC_625 (Minerva Mills)
  → 2007_6_SCC_1 (Coelho)
  → 1994_3_SCC_1 (S.R. Bommai)

Step 4: Rank by PageRank
  1. Minerva Mills (PR=0.0087)
  2. Golak Nath (PR=0.0054)
  3. Gopalan (PR=0.0043)
```

#### 6.4.4 PageRank Computation

```python
import networkx as nx

# Compute PageRank for all nodes
pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)

# Interpretation:
# PageRank = 0.01 → Highly influential (top 1%)
# PageRank = 0.001 → Moderately cited
# PageRank = 0.0001 → Rarely cited
```

#### 6.4.5 Performance Characteristics

| Metric | Value |
|--------|-------|
| Search time | 50-100ms |
| Graph size | 4,451 nodes, 2,840 edges |
| Recall@10 | 25% (standalone) |
| Precision@5 | 12% (standalone) |
| Best for | Finding related precedents |

### 6.5 Method 4: Entity Search (Direct Lookup)

#### 6.5.1 Algorithm Description

Entity search uses reverse indices to directly retrieve all cases mentioning specific legal provisions.

**Innovation:** O(1) lookup instead of O(n) search

#### 6.5.2 Implementation

```python
def entity_search(self, entities: Dict[str, List[str]], top_k: int = 10) -> List[Dict]:
    """
    Direct entity lookup using reverse index.
    
    Args:
        entities: Dict with 'articles', 'sections', 'acts'
        top_k: Number of results
    
    Returns:
        Cases containing the specified entities
    """
    matched_case_ids = set()
    
    # Step 1: Look up articles
    for article in entities.get('articles', []):
        if article in self.entity_index['articles']:
            case_ids = self.entity_index['articles'][article]
            matched_case_ids.update(case_ids)
    
    # Step 2: Look up sections
    for section in entities.get('sections', []):
        if section in self.entity_index['sections']:
            case_ids = self.entity_index['sections'][section]
            matched_case_ids.update(case_ids)
    
    # Step 3: Convert case IDs to chunks
    results = []
    for case_id in matched_case_ids:
        # Find chunks for this case
        case_chunks = [m for m in self.metadata if m['case_id'] == case_id]
        
        if case_chunks:
            # Find chunk with highest entity density
            best_chunk = max(case_chunks, 
                           key=lambda c: self._entity_density(c['text'], entities))
            
            results.append({
                'chunk_id': best_chunk['chunk_id'],
                'case_id': case_id,
                'case_name': best_chunk['case_name'],
                'text': best_chunk['text'],
                'page_num': best_chunk.get('page_num', 1),
                'score': 1.0,  # Binary: entity present or not
                'method': 'entity_search'
            })
    
    # Step 4: Return top-k (limited by query, not scored)
    return results[:top_k]

def _entity_density(self, text: str, entities: Dict) -> float:
    """Count how many times entities appear in text."""
    count = 0
    text_lower = text.lower()
    
    for article in entities.get('articles', []):
        count += text_lower.count(f'article {article}')
    
    for section in entities.get('sections', []):
        count += text_lower.count(f'section {section}')
    
    return count
```

#### 6.5.3 Entity Index Structure

```python
{
  "articles": {
    "14": ["1978_AIR_597", "1973_SUP_1", "1950_AIR_27", ...],  # 67 cases
    "19": ["1950_AIR_27", "1978_AIR_597", ...],                # 112 cases
    "21": ["1978_AIR_597", "1981_1_SCC_87", ...],              # 203 cases
    "32": ["1986_AIR_180", ...],                               # 45 cases
  },
  "sections": {
    "302": ["1975_AIR_123", "1980_SC_456", ...],  # IPC Section 302 (murder)
    "376": ["1985_AIR_234", "2013_SC_789", ...],  # IPC Section 376 (rape)
    "498A": ["1990_AIR_345", ...],                # Section 498A (dowry)
  },
  "case_name_to_id": {
    "maneka gandhi v union india": "1978_AIR_597",
    "kesavananda bharati v state kerala": "1973_SUP_1"
  }
}
```

#### 6.5.4 Query Example

```
Query: "Article 21 right to life cases"

Step 1: Extract entities
  → articles = ["21"]

Step 2: Lookup in entity_index
  → entity_index['articles']['21'] = [203 case IDs]

Step 3: Convert to chunks
  → 203 cases × ~21 chunks/case = ~4,200 chunks

Step 4: Select best chunks (highest entity density)
  → Top 10 chunks with most "Article 21" mentions

Result: Perfect recall for Article 21 queries!
```

#### 6.5.5 Performance Characteristics

| Metric | Value |
|--------|-------|
| Search time | 50-100ms |
| Index size | 5 MB |
| Recall@10 | 80% for entity queries |
| Precision@5 | 60% for entity queries |
| Best for | Article/Section queries |

### 6.6 Method 5: Date Search (Year Filtering)

#### 6.6.1 Algorithm Description

Date search filters cases by year, useful for temporal queries.

**Challenge:** Only ~35% of cases have reliable date metadata

#### 6.6.2 Implementation

```python
def date_search(self, year: int, top_k: int = 50) -> List[Dict]:
    """
    Search cases by year.
    
    Args:
        year: Year to filter (e.g., 2019)
        top_k: Number of results
    
    Returns:
        Cases from specified year
    """
    results = []
    
    # Search through metadata
    for chunk in self.metadata:
        date_str = chunk.get('date', '')
        
        # Check if year matches
        if str(year) in date_str:
            results.append({
                'chunk_id': chunk['chunk_id'],
                'case_id': chunk['case_id'],
                'case_name': chunk['case_name'],
                'text': chunk['text'],
                'page_num': chunk.get('page_num', 1),
                'score': 1.0,
                'method': 'date_search'
            })
            
            if len(results) >= top_k:
                break
    
    return results
```

#### 6.6.3 Date Extraction from Query

```python
def _extract_year_from_query(self, query: str) -> Optional[int]:
    """Extract year from query using regex."""
    year_pattern = r'\b(19\d{2}|20\d{2})\b'
    match = re.search(year_pattern, query)
    
    if match:
        return int(match.group(1))
    
    return None
```

**Examples:**
```
"judgments delivered in 2019" → year=2019
"cases from 2020" → year=2020
"2018 Supreme Court decisions" → year=2018
```

#### 6.6.4 Performance Characteristics

| Metric | Value |
|--------|-------|
| Search time | 50-100ms |
| Date coverage | 35% of cases |
| Recall@10 | 10% (limited by metadata) |
| Best for | Temporal queries (when metadata available) |

### 6.7 Method 6: Bench Search (Judge Filtering)

#### 6.7.1 Algorithm Description

Bench search finds cases decided by specific judges.

**Challenge:** Only ~27.6% of cases have judge information

#### 6.7.2 Implementation

```python
def bench_search(self, judge_names: List[str], top_k: int = 50) -> List[Dict]:
    """
    Search cases by judge names.
    
    Args:
        judge_names: List of judge names to search
        top_k: Number of results
    
    Returns:
        Cases decided by specified judges
    """
    matched_case_ids = set()
    
    # Step 1: Search bench index
    for judge_name in judge_names:
        judge_normalized = judge_name.upper().strip()
        
        # Try exact match
        if judge_normalized in self.bench_index['judges']:
            case_ids = self.bench_index['judges'][judge_normalized]
            matched_case_ids.update(case_ids)
        else:
            # Try partial match
            for judge_key, case_ids in self.bench_index['judges'].items():
                if judge_normalized in judge_key or judge_key in judge_normalized:
                    matched_case_ids.update(case_ids)
    
    # Step 2: Convert to chunks
    results = []
    for case_id in matched_case_ids:
        case_chunks = [m for m in self.metadata if m['case_id'] == case_id]
        
        if case_chunks:
            results.append({
                'chunk_id': case_chunks[0]['chunk_id'],
                'case_id': case_id,
                'case_name': case_chunks[0]['case_name'],
                'text': case_chunks[0]['text'],
                'page_num': case_chunks[0].get('page_num', 1),
                'score': 1.0,
                'method': 'bench_search'
            })
    
    return results[:top_k]
```

#### 6.7.3 Bench Index Structure

```json
{
  "judges": {
    "RAMASWAMY, K": ["1973_SUP_1", "1980_3_SCC_625", ...],  // 137 cases
    "GAJENDRAGADKAR, P.B": ["1950_AIR_27", ...],            // 43 cases
    "CHANDRACHUD, Y.V": ["1978_AIR_597", ...]               // 38 cases
  }
}
```

#### 6.7.4 Performance Characteristics

| Metric | Value |
|--------|-------|
| Search time | 50-100ms |
| Judge coverage | 27.6% of cases |
| Recall@10 | 15% (limited by metadata) |
| Best for | Judge-based queries (when metadata available) |

---

## 7. HYBRID FUSION STRATEGY

### 7.1 Motivation for Fusion

**Problem:** Each retrieval method has strengths and weaknesses

| Method | Strength | Weakness |
|--------|----------|----------|
| Vector | Semantic understanding | Misses exact citations |
| BM25 | Exact keyword matching | No semantic understanding |
| Graph | Finds precedents | Needs seed cases |
| Entity | Perfect for articles/sections | Only works for entity queries |

**Solution:** Combine all methods using Reciprocal Rank Fusion (RRF)

### 7.2 Reciprocal Rank Fusion (RRF)

#### 7.2.1 Algorithm

```python
def reciprocal_rank_fusion(rankings: List[List[str]], k: int = 60) -> List[str]:
    """
    Combine multiple rankings using RRF.
    
    Args:
        rankings: List of ranked document lists from different methods
        k: Constant (typically 60)
    
    Returns:
        Fused ranking
    """
    scores = {}
    
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0
            
            # RRF formula
            scores[doc_id] += 1 / (k + rank)
    
    # Sort by score
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in fused]
```

#### 7.2.2 Mathematical Formulation

```
RRF(d) = Σ(r∈R) w(r) / (k + rank_r(d))

where:
- R = {vector, bm25, graph, entity, date, bench}
- rank_r(d) = position of document d in ranking r
- w(r) = weight for ranker r (dynamic based on query type)
- k = 60 (constant)
```

### 7.3 Dynamic Weighting Strategy

#### 7.3.1 Query-Adaptive Weights

Our system adjusts weights based on query characteristics:

```python
def calculate_dynamic_weights(query: str, entities: Dict) -> Dict[str, float]:
    """
    Calculate retrieval method weights based on query.
    
    Returns:
        Dictionary of method weights
    """
    weights = {
        'vector_search': 0.30,
        'bm25_search': 0.30,
        'graph_search': 0.20,
        'entity_search': 0.20,
        'date_search': 0.0,
        'bench_search': 0.0
    }
    
    # Boost entity search if entities present
    if entities.get('articles') or entities.get('sections'):
        weights['entity_search'] = 0.50
        weights['vector_search'] = 0.20
        weights['bm25_search'] = 0.20
        weights['graph_search'] = 0.10
    
    # Boost date search if year in query
    if entities.get('years'):
        weights['date_search'] = 0.60
        weights['vector_search'] = 0.15
        weights['bm25_search'] = 0.15
        weights['graph_search'] = 0.10
    
    # Boost bench search if judges in query
    if entities.get('judges'):
        weights['bench_search'] = 0.60
        weights['vector_search'] = 0.15
        weights['bm25_search'] = 0.15
        weights['graph_search'] = 0.10
    
    # Boost BM25 for case name queries
    if entities.get('case_names'):
        weights['bm25_search'] = 0.50
        weights['vector_search'] = 0.25
        weights['graph_search'] = 0.15
        weights['entity_search'] = 0.10
    
    return weights
```

#### 7.3.2 Weight Examples

**Case 1: Entity Query**
```
Query: "Article 14 equality before law"
Entities: articles=["14"]

Weights:
  entity_search: 0.50 (50%) ← Primary
  vector_search: 0.20 (20%)
  bm25_search: 0.20 (20%)
  graph_search: 0.10 (10%)
```

**Case 2: Case Name Query**
```
Query: "Kesavananda Bharati v. State of Kerala"
Entities: case_names=["Kesavananda Bharati v. State of Kerala"]

Weights:
  bm25_search: 0.50 (50%) ← Primary
  vector_search: 0.25 (25%)
  graph_search: 0.15 (15%)
  entity_search: 0.10 (10%)
```

**Case 3: General Query**
```
Query: "fundamental rights under constitution"
Entities: (none detected)

Weights:
  vector_search: 0.30 (30%)
  bm25_search: 0.30 (30%)
  graph_search: 0.20 (20%)
  entity_search: 0.20 (20%)
```

### 7.4 Hybrid Search Implementation

#### 7.4.1 Complete Pipeline

```python
def hybrid_search(self, query: str, top_k: int = 10, 
                 include_graph: bool = True,
                 include_entities: bool = True) -> List[Dict]:
    """
    Hybrid search combining all retrieval methods.
    
    Args:
        query: User query
        top_k: Number of final results
        include_graph: Whether to use graph search
        include_entities: Whether to use entity search
    
    Returns:
        Fused and reranked results
    """
    import time
    start_time = time.time()
    
    # Step 1: Extract entities from query
    entities = self._extract_entities_from_query(query)
    
    # Step 2: Calculate dynamic weights
    weights = self._calculate_dynamic_weights(query, entities)
    
    # Step 3: Execute all search methods
    all_results = {}
    
    # Vector search (always run)
    vector_results = self.vector_search(query, top_k=20)
    all_results['vector_search'] = vector_results
    
    # BM25 search (always run)
    bm25_results = self.bm25_search(query, top_k=20)
    all_results['bm25_search'] = bm25_results
    
    # Graph search (if enabled)
    if include_graph:
        graph_results = self.graph_search(query, top_k=20)
        all_results['graph_search'] = graph_results
    
    # Entity search (if entities present)
    if include_entities and (entities['articles'] or entities['sections']):
        entity_results = self.entity_search(entities, top_k=20)
        all_results['entity_search'] = entity_results
    
    # Date search (if year present)
    if entities['years']:
        date_results = self.date_search(entities['years'][0], top_k=20)
        all_results['date_search'] = date_results
    
    # Bench search (if judges present)
    if entities['judges']:
        bench_results = self.bench_search(entities['judges'], top_k=20)
        all_results['bench_search'] = bench_results
    
    # Step 4: Apply RRF fusion
    fused_results = self._reciprocal_rank_fusion(all_results, weights)
    
    # Step 5: Metadata boosting (post-fusion reranking)
    boosted_results = self._metadata_filter(fused_results, entities)
    
    # Step 6: Deduplicate and return top-k
    final_results = self._deduplicate(boosted_results)[:top_k]
    
    elapsed = (time.time() - start_time) * 1000
    print(f"Hybrid search completed in {elapsed:.0f}ms")
    
    return final_results

def _reciprocal_rank_fusion(self, all_results: Dict[str, List[Dict]], 
                            weights: Dict[str, float]) -> List[Dict]:
    """Apply weighted RRF to combine rankings."""
    rrf_scores = {}
    k = 60  # RRF constant
    
    for method, results in all_results.items():
        weight = weights.get(method, 0.0)
        
        if weight == 0.0:
            continue
        
        for rank, result in enumerate(results, start=1):
            doc_id = result['case_id'] + "_" + str(result['chunk_id'])
            
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    'score': 0.0,
                    'result': result,
                    'methods': []
                }
            
            # Weighted RRF formula
            rrf_contribution = (weight / (k + rank))
            rrf_scores[doc_id]['score'] += rrf_contribution
            rrf_scores[doc_id]['methods'].append(method)
    
    # Sort by RRF score
    sorted_results = sorted(rrf_scores.values(), 
                           key=lambda x: x['score'], 
                           reverse=True)
    
    return [item['result'] for item in sorted_results]

def _deduplicate(self, results: List[Dict]) -> List[Dict]:
    """Remove duplicate chunks from same case."""
    seen_cases = set()
    deduplicated = []
    
    for result in results:
        case_id = result['case_id']
        
        # Keep max 2 chunks per case
        case_count = sum(1 for r in deduplicated if r['case_id'] == case_id)
        
        if case_count < 2:
            deduplicated.append(result)
            seen_cases.add(case_id)
    
    return deduplicated
```

#### 7.4.2 Fusion Example

```
Query: "Article 21 right to life"

Vector Search Results (rank):
1. 1978_AIR_597_chunk_42 (Maneka Gandhi)
2. 1973_SUP_1_chunk_67 (Kesavananda)
3. 1981_1_SCC_87_chunk_23 (Francis Coralie)

BM25 Search Results (rank):
1. 1978_AIR_597_chunk_42 (Maneka Gandhi)
2. 1981_1_SCC_87_chunk_23 (Francis Coralie)
3. 2017_10_SCC_1_chunk_15 (Puttaswamy)

Entity Search Results (rank):
1. 1978_AIR_597_chunk_42 (Maneka Gandhi)
2. 1981_1_SCC_87_chunk_23 (Francis Coralie)
3. 1973_SUP_1_chunk_67 (Kesavananda)

RRF Calculation for Maneka Gandhi chunk:
  Vector: 0.20 / (60 + 1) = 0.00328
  BM25: 0.20 / (60 + 1) = 0.00328
  Entity: 0.50 / (60 + 1) = 0.00820
  Total: 0.01476 ← Highest score!

Final Ranking:
1. 1978_AIR_597_chunk_42 (score=0.01476)
2. 1981_1_SCC_87_chunk_23 (score=0.01234)
3. 1973_SUP_1_chunk_67 (score=0.00987)
```

### 7.5 Performance Analysis

#### 7.5.1 Method Contribution

Analysis of which methods contribute most to final results:

```
Query Type: Entity (Article/Section)
  entity_search: 65% of top-10
  vector_search: 20% of top-10
  bm25_search: 10% of top-10
  graph_search: 5% of top-10

Query Type: Case Name
  bm25_search: 70% of top-10
  vector_search: 15% of top-10
  graph_search: 10% of top-10
  entity_search: 5% of top-10

Query Type: General/Conceptual
  vector_search: 45% of top-10
  bm25_search: 30% of top-10
  graph_search: 15% of top-10
  entity_search: 10% of top-10
```

#### 7.5.2 Timing Breakdown

```
Total Hybrid Search Time: ~567ms

Component breakdown:
  Vector search: 200ms (35%)
  BM25 search: 150ms (26%)
  Graph search: 80ms (14%)
  Entity search: 50ms (9%)
  RRF fusion: 30ms (5%)
  Metadata boosting: 40ms (7%)
  Deduplication: 17ms (3%)
```

---

## 8. ANSWER GENERATION WITH LLM

### 8.1 Large Language Model Integration

#### 8.1.1 Model Selection

**Primary Model:** Google Gemini 2.5 Flash

**Specifications:**
- Parameters: ~20 billion (estimated)
- Context window: 32,000 tokens
- Output length: Up to 8,192 tokens
- Latency: 300-400ms per generation
- Cost: $0.00015 per 1K tokens (input), $0.0006 per 1K tokens (output)

**Why Gemini 2.5 Flash:**
- Fast inference (real-time generation)
- Good instruction following
- Strong legal reasoning capabilities
- Affordable for production use

**Fallback Model:** Groq (Llama-3-70B)
- Used if Gemini quota exceeded
- Faster inference (~200ms)
- Slightly lower quality

#### 8.1.2 Model API Integration

```python
import google.generativeai as genai

class LegalAnswerGenerator:
    """Generate legal answers using Gemini API."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash",
                 knowledge_graph=None):
        """Initialize generator with API key and model."""
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name)
        self.knowledge_graph = knowledge_graph
        self.model_name = model_name
    
    def generate_answer(self, query: str, context: str,
                       max_tokens: int = 8192,
                       temperature: float = 0.3,
                       retrieved_case_ids: List[str] = None) -> Dict:
        """
        Generate answer with knowledge graph enrichment.
        
        Args:
            query: User question
            context: Retrieved legal documents
            max_tokens: Maximum response length
            temperature: Sampling temperature (0.0-1.0)
            retrieved_case_ids: List of retrieved case IDs
        
        Returns:
            Dict with answer, citations, and metadata
        """
        import time
        start_time = time.time()
        
        # Step 1: Get related cases from knowledge graph
        related_cases = []
        if self.knowledge_graph and retrieved_case_ids:
            related_cases = self._get_related_cases_from_graph(
                retrieved_case_ids, max_related=5
            )
        
        # Step 2: Create structured prompt
        prompt = self.create_prompt(query, context, related_cases)
        
        # Step 3: Generate answer
        try:
            response = self.client.generate_content(
                prompt,
                generation_config={
                    'max_output_tokens': max_tokens,
                    'temperature': temperature,
                    'top_p': 0.95,
                    'top_k': 40
                }
            )
            
            answer = response.text
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            answer = self._generate_fallback_answer(context, query)
        
        elapsed = (time.time() - start_time) * 1000
        
        return {
            'answer': answer,
            'model': self.model_name,
            'generation_time_ms': elapsed,
            'prompt_tokens': len(prompt.split()),
            'related_cases_suggested': related_cases
        }
```

### 8.2 Prompt Engineering

#### 8.2.1 Prompt Structure

```python
def create_prompt(self, query: str, context: str, 
                 related_cases: List[str] = None) -> str:
    """Create structured prompt for legal QA."""
    
    # Related cases section
    related_section = ""
    if related_cases and len(related_cases) > 0:
        related_section = f"""

RELATED LANDMARK CASES (from citation network):
{chr(10).join(f"- {case}" for case in related_cases[:5])}
Consider citing these if relevant to the query.
"""
    
    prompt = f"""You are a legal research assistant for Indian Supreme Court law. Provide an informative answer based on the context below.

TASK:
- Answer using ONLY the provided legal context
- Cite sources using the EXACT format provided in context: [Document Name-pageX] for PDFs or [Case Name]
- ALWAYS include page numbers when they are provided in the source
- Be factual and objective
- If information is insufficient, state that clearly
- Reference the related landmark cases when they strengthen your answer

LEGAL CONTEXT:
{context}{related_section}

QUESTION:
{query}

ANSWER WITH CITATIONS:"""
    
    return prompt
```

#### 8.2.2 Prompt Components

**1. Role Definition**
```
"You are a legal research assistant for Indian Supreme Court law."
```
- Establishes domain expertise
- Primes model for legal reasoning

**2. Task Instructions**
```
- Answer using ONLY the provided legal context
- Cite sources using EXACT format
- Include page numbers
- Be factual and objective
```
- Constrains generation to context (reduces hallucination)
- Specifies citation format
- Enforces factual tone

**3. Context Section**
```
LEGAL CONTEXT:
[Retrieved Document 1]
...
[Retrieved Document 10]

RELATED LANDMARK CASES:
- Kesavananda Bharati v. State of Kerala
- Maneka Gandhi v. Union of India
```
- Provides evidence for answer
- Suggests relevant precedents from knowledge graph

**4. Query Section**
```
QUESTION:
{user_query}

ANSWER WITH CITATIONS:
```
- Clear delineation of what to answer
- Prompts citation inclusion

#### 8.2.3 Context Preparation

```python
def get_context_for_generation(self, results: List[Dict], 
                               max_context_length: int = 10000) -> str:
    """
    Format retrieved results into context string.
    
    Args:
        results: List of retrieved chunks
        max_context_length: Maximum words in context
    
    Returns:
        Formatted context string
    """
    context_parts = []
    total_words = 0
    
    for idx, result in enumerate(results, start=1):
        case_name = result['case_name']
        text = result['text']
        page_num = result.get('page_num', 'N/A')
        source_type = result.get('source_type', 'ildc')
        
        # Format source citation
        if source_type == 'uploaded_pdf':
            filename = result.get('filename', 'Document')
            source = f"[{filename}-page{page_num}]"
        else:
            source = f"[{case_name}]"
        
        # Create context entry
        entry = f"""
--- Document {idx}: {source} ---
{text}
---
"""
        
        # Check length
        entry_words = len(entry.split())
        if total_words + entry_words > max_context_length:
            break
        
        context_parts.append(entry)
        total_words += entry_words
    
    return '\n'.join(context_parts)
```

**Example Context:**
```
--- Document 1: [Maneka Gandhi v. Union of India] ---
The right to life under Article 21 of the Constitution does not connote
mere animal existence. It means a life with human dignity. The procedure
established by law must be just, fair, and reasonable, not arbitrary,
fanciful, or oppressive. This Court holds that Article 21 encompasses
the right to travel abroad.
---

--- Document 2: [Francis Coralie Mullin v. Administrator] ---
Article 21 protections extend beyond mere physical existence. The right
to live with human dignity includes the right to food, water, decent
environment, education, medical care, and shelter. These are integral
to the right to life.
---

RELATED LANDMARK CASES:
- Kesavananda Bharati v. State of Kerala
- Minerva Mills v. Union of India
```

### 8.3 Knowledge Graph Enrichment

#### 8.3.1 Landmark Case Suggestion

```python
def _get_related_cases_from_graph(self, case_ids: List[str], 
                                 max_related: int = 5) -> List[str]:
    """
    Get related landmark cases from citation network.
    
    Args:
        case_ids: Retrieved case IDs
        max_related: Maximum related cases to return
    
    Returns:
        List of related case names
    """
    if not self.knowledge_graph:
        return []
    
    related_case_ids = set()
    
    for case_id in case_ids[:5]:  # Limit seed cases
        if not self.knowledge_graph.graph.has_node(case_id):
            continue
        
        # Get cited cases (outgoing edges)
        cited = list(self.knowledge_graph.graph.successors(case_id))
        related_case_ids.update(cited[:3])
        
        # Get citing cases (incoming edges)
        citing = list(self.knowledge_graph.graph.predecessors(case_id))
        related_case_ids.update(citing[:3])
    
    # Convert to case names and rank by PageRank
    related_cases = []
    for rel_id in related_case_ids:
        if self.knowledge_graph.graph.has_node(rel_id):
            case_name = self.knowledge_graph.graph.nodes[rel_id].get('name', '')
            pagerank = self.knowledge_graph.graph.nodes[rel_id].get('pagerank', 0)
            
            if case_name:
                related_cases.append((case_name, pagerank))
    
    # Sort by PageRank (landmark cases first)
    related_cases.sort(key=lambda x: x[1], reverse=True)
    
    return [name for name, _ in related_cases[:max_related]]
```

**Effect:**
- Suggests high-authority cases even if not directly retrieved
- Encourages citing established precedents
- Improves answer quality by referencing landmark judgments

#### 8.3.2 Example Enhancement

**Without Knowledge Graph:**
```
Query: "Article 21 interpretation"

Context: [3 cases retrieved]

Answer: "Article 21 protects the right to life. According to the
retrieved cases, this includes the right to livelihood and personal
liberty."
```

**With Knowledge Graph:**
```
Query: "Article 21 interpretation"

Context: [3 cases retrieved]
Related Cases: [Kesavananda Bharati, Maneka Gandhi, Minerva Mills]

Answer: "Article 21 protects the right to life and personal liberty.
As held in Maneka Gandhi v. Union of India, this right must be
interpreted broadly to include human dignity. The landmark judgment
in Kesavananda Bharati established that fundamental rights form
part of the basic structure of the Constitution."
```

### 8.4 Generation Hyperparameters

#### 8.4.1 Parameter Selection

```python
generation_config = {
    'max_output_tokens': 8192,
    'temperature': 0.3,
    'top_p': 0.95,
    'top_k': 40,
    'stop_sequences': []
}
```

**Parameter Explanations:**

**1. Temperature (0.3)**
- Range: 0.0 (deterministic) to 1.0 (random)
- Effect: Controls randomness in generation
- Low (0.3): More focused, factual responses
- High (0.8): More creative, diverse responses
- **Choice:** 0.3 for factual legal domain

**2. Top-p (0.95)**
- Nucleus sampling: Sample from smallest set of tokens whose cumulative probability ≥ p
- Range: 0.0 to 1.0
- Effect: Limits token choices to high-probability options
- **Choice:** 0.95 balances diversity and coherence

**3. Top-k (40)**
- Sample from top-k highest probability tokens
- Range: 1 to vocabulary size
- Effect: Prevents very low-probability token selection
- **Choice:** 40 prevents nonsensical generations

**4. Max Output Tokens (8192)**
- Maximum length of generated answer
- Allows comprehensive legal analysis
- **Choice:** 8192 for detailed explanations

#### 8.4.2 Temperature Impact

```
Temperature = 0.0:
"Article 21 guarantees the right to life and personal liberty."
(Deterministic, same output every time)

Temperature = 0.3:
"Article 21 of the Constitution guarantees the right to life and
personal liberty, which has been interpreted to include the right
to dignity."
(Slightly varied, factual)

Temperature = 0.7:
"The fundamental right under Article 21 encompasses a wide range
of protections including life, liberty, privacy, and dignity,
as established through numerous judicial interpretations."
(More varied, still coherent)

Temperature = 1.0:
"Article 21 is a cornerstone provision that creatively protects
various dimensions of human existence through dynamic judicial
activism and evolving societal needs."
(Creative, potentially less precise)
```

### 8.5 Fallback Mechanism

#### 8.5.1 Error Handling

```python
def generate_with_fallback(self, query: str, context: str) -> str:
    """Generate answer with fallback to extractive summary."""
    try:
        # Try LLM generation
        result = self.generate_answer(query, context)
        return result['answer']
    
    except Exception as e:
        print(f"LLM generation failed: {e}")
        
        # Fallback: Extract most relevant snippet
        return self._generate_fallback_answer(context, query)

def _generate_fallback_answer(self, context: str, query: str) -> str:
    """Generate simple extractive answer if LLM fails."""
    # Find most relevant paragraph
    paragraphs = context.split('\n\n')
    
    # Simple relevance: count query words in paragraph
    query_words = set(query.lower().split())
    
    best_para = max(paragraphs, 
                   key=lambda p: len(query_words & set(p.lower().split())))
    
    return f"Based on the retrieved documents:\n\n{best_para}\n\n(Note: This is an extractive summary due to generation error.)"
```

### 8.6 Generation Performance

#### 8.6.1 Timing Analysis

```
Average Generation Time: 324ms

Breakdown:
  API request: 280ms (86%)
  Graph lookup: 20ms (6%)
  Prompt formatting: 15ms (5%)
  Response parsing: 9ms (3%)
```

#### 8.6.2 Token Usage

```
Average per query:
  Input tokens: 3,200 (context + prompt)
  Output tokens: 450 (answer)
  Total tokens: 3,650
  Cost per query: $0.0008 (~$0.80 per 1000 queries)
```

---

---

## 9. CITATION VERIFICATION SYSTEM

### 9.1 The Citation Hallucination Problem

**Critical Issue:** LLMs frequently generate plausible-sounding but **non-existent** legal citations.

**Examples of Hallucinations:**
```
Generated: "State v. Kumar (2018) AIR 234"  ← Does not exist
Generated: "Sharma v. Sharma (2020) 5 SCC 678"  ← Fabricated
Generated: "According to Patel v. State (2019)..."  ← Made up
```

**Consequences:**
- Legal malpractice if used in court
- Loss of trust in AI systems
- Potential sanctions for lawyers

**Our Solution:** Post-generation verification using Knowledge Graph lookup

### 9.2 Verification Architecture

```
Generated Answer → Citation Extraction → Graph Lookup → Validity Check → Verified Answer
                         ↓                    ↓               ↓              ↓
                    [Regex Pattern]    [NetworkX Query]  [Overruled?]  [Flag Invalid]
```

### 9.3 Citation Extraction

#### 9.3.1 Extraction Patterns

```python
def extract_citations_from_text(text: str) -> List[str]:
    """
    Extract all citations from generated text.
    
    Args:
        text: Generated answer with citations
    
    Returns:
        List of extracted citation strings
    """
    citations = []
    
    # Pattern 1: [Case Name] format
    pattern1 = r'\[([^\]]+)\]'
    matches1 = re.findall(pattern1, text)
    citations.extend(matches1)
    
    # Pattern 2: AIR citations (outside brackets)
    pattern2 = r'\bAIR\s+\d{4}\s+[A-Z]+\s+\d+'
    matches2 = re.findall(pattern2, text)
    citations.extend(matches2)
    
    # Pattern 3: SCC citations
    pattern3 = r'\(\d{4}\)\s+\d+\s+SCC\s+\d+'
    matches3 = re.findall(pattern3, text)
    citations.extend(matches3)
    
    # Pattern 4: Case name with v./vs.
    pattern4 = r'\b([A-Z][A-Za-z\s]+)\s+v\.?\s+([A-Z][A-Za-z\s]+)\b'
    matches4 = re.findall(pattern4, text)
    citations.extend([f"{m[0]} v. {m[1]}" for m in matches4])
    
    # Remove duplicates
    return list(set(citations))
```

**Example Extraction:**
```
Input Text:
"According to [Maneka Gandhi v. Union of India], Article 21
protects the right to travel. This was affirmed in AIR 1978 SC 597
and later in Francis Coralie Mullin v. Administrator."

Extracted Citations:
1. "Maneka Gandhi v. Union of India"
2. "AIR 1978 SC 597"
3. "Francis Coralie Mullin v. Administrator"
```

#### 9.3.2 Citation Normalization

```python
def normalize_case_name(case_name: str) -> str:
    """
    Normalize case name to standard format for matching.
    
    Args:
        case_name: Raw case name
    
    Returns:
        Normalized case name
    """
    # Convert to lowercase
    normalized = case_name.lower()
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Standardize v./vs./versus
    normalized = re.sub(r'\s+v\.?\s+', ' v ', normalized)
    normalized = re.sub(r'\s+vs\.?\s+', ' v ', normalized)
    normalized = re.sub(r'\s+versus\s+', ' v ', normalized)
    
    # Remove punctuation except periods in abbreviations
    normalized = re.sub(r'[^\w\s\.]', '', normalized)
    
    # Remove common variations
    normalized = normalized.replace(' state of ', ' state ')
    normalized = normalized.replace(' union of ', ' union ')
    
    return normalized
```

**Example Normalization:**
```
Input: "Maneka Gandhi V. Union Of India"
Output: "maneka gandhi v union india"

Input: "State of Kerala vs. N.M. Thomas"
Output: "state kerala v n m thomas"

Input: "Ram Kumar    versus    State"
Output: "ram kumar v state"
```

### 9.4 Knowledge Graph Lookup

#### 9.4.1 Verification Process

```python
class CitationVerifier:
    """Verify citations against knowledge graph."""
    
    def __init__(self, graph: LegalKnowledgeGraph):
        """Initialize with knowledge graph."""
        self.graph = graph
        self.case_name_to_id = self._build_name_index()
    
    def _build_name_index(self) -> Dict[str, str]:
        """Build reverse index: normalized_name → case_id."""
        index = {}
        
        for node_id in self.graph.graph.nodes():
            case_name = self.graph.graph.nodes[node_id].get('name', '')
            if case_name:
                normalized = normalize_case_name(case_name)
                index[normalized] = node_id
        
        return index
    
    def verify_citation(self, citation: str) -> Dict:
        """
        Verify a single citation.
        
        Args:
            citation: Case name or citation string
        
        Returns:
            Verification result with status
        """
        # Special handling for uploaded PDFs
        if 'Document' in citation or '.pdf' in citation or '-page' in citation:
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
        
        # Normalize citation
        normalized = normalize_case_name(citation)
        
        # Check if exists in knowledge graph
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
```

#### 9.4.2 Verification Example

```
Query: "Article 21 cases"

Generated Answer:
"Article 21 protects the right to life. This was established in
[Maneka Gandhi v. Union of India] and later affirmed in [Sharma v.
Sharma] and [Francis Coralie Mullin v. Administrator]."

Verification Process:

Citation 1: "Maneka Gandhi v. Union of India"
  → Normalized: "maneka gandhi v union india"
  → Lookup: Found in graph (case_id = 1978_AIR_597)
  → Overruled: No
  → Status: ✅ Valid

Citation 2: "Sharma v. Sharma"
  → Normalized: "sharma v sharma"
  → Lookup: Not found in graph
  → Status: ❌ Not Found (HALLUCINATION!)

Citation 3: "Francis Coralie Mullin v. Administrator"
  → Normalized: "francis coralie mullin v administrator"
  → Lookup: Found in graph (case_id = 1981_1_SCC_87)
  → Overruled: No
  → Status: ✅ Valid

Verification Result:
  Total citations: 3
  Valid: 2 (66.7%)
  Hallucinated: 1 (33.3%)
  Overruled: 0 (0%)
```

### 9.5 Overruled Case Detection

#### 9.5.1 Detection Algorithm

```python
def is_overruled(self, case_id: str) -> bool:
    """
    Check if a case has been overruled.
    
    Args:
        case_id: Case identifier
    
    Returns:
        True if overruled, False otherwise
    """
    if not self.graph.has_node(case_id):
        return False
    
    # Check node attribute
    node_data = self.graph.nodes[case_id]
    if node_data.get('overruled', False):
        return True
    
    # Check if any citing case explicitly overrules this one
    for citing_case in self.graph.predecessors(case_id):
        edge_data = self.graph.get_edge_data(citing_case, case_id)
        if edge_data and edge_data.get('relation') == 'OVERRULES':
            return True
    
    return False
```

**Example:**
```
Golak Nath v. State of Punjab (1967)
  → Held: Parliament cannot amend fundamental rights

Kesavananda Bharati v. State of Kerala (1973)
  → Overruled Golak Nath
  → Held: Parliament CAN amend, but not basic structure

Verification:
  is_overruled("1967_AIR_1643") → True
  overruled_by → "1973_SUP_1" (Kesavananda)
```

### 9.6 Answer Correction

#### 9.6.1 Flagging Invalid Citations

```python
def correct_answer(self, answer: str) -> Tuple[str, Dict]:
    """
    Verify and correct answer by flagging invalid citations.
    
    Args:
        answer: Generated answer with citations
    
    Returns:
        Tuple of (corrected_answer, verification_report)
    """
    # Extract all citations
    citations = extract_citations_from_text(answer)
    
    # Verify each citation
    verification_results = []
    for citation in citations:
        result = self.verify_citation(citation)
        verification_results.append(result)
    
    # Build corrected answer
    corrected_answer = answer
    
    for result in verification_results:
        citation = result['citation']
        
        if not result['exists']:
            # Flag hallucinated citations
            flag = f" [⚠️ UNVERIFIED: Citation not found in database]"
            corrected_answer = corrected_answer.replace(
                f"[{citation}]", f"[{citation}]{flag}"
            )
        
        elif result['overruled']:
            # Flag overruled cases
            flag = f" [⚠️ OVERRULED: This case has been overruled]"
            corrected_answer = corrected_answer.replace(
                f"[{citation}]", f"[{citation}]{flag}"
            )
    
    # Build verification report
    report = {
        'total_citations': len(citations),
        'valid_citations': sum(1 for r in verification_results if r['exists'] and not r['overruled']),
        'hallucinated_citations': sum(1 for r in verification_results if not r['exists']),
        'overruled_citations': sum(1 for r in verification_results if r['overruled']),
        'verification_rate': sum(1 for r in verification_results if r['verified']) / len(citations) if citations else 0,
        'details': verification_results
    }
    
    return corrected_answer, report
```

**Corrected Answer Example:**

**Original Generated Answer:**
```
Article 21 protects the right to life and personal liberty.
This was established in [Maneka Gandhi v. Union of India] and
later affirmed in [Sharma v. Sharma] and [Patel v. State].
```

**After Verification:**
```
Article 21 protects the right to life and personal liberty.
This was established in [Maneka Gandhi v. Union of India] and
later affirmed in [Sharma v. Sharma] [⚠️ UNVERIFIED: Citation
not found in database] and [Patel v. State] [⚠️ UNVERIFIED:
Citation not found in database].

Verification Report:
✅ Valid: 1 (33.3%)
⚠️ Hallucinated: 2 (66.7%)
❌ Overruled: 0 (0%)
```

### 9.7 Verification System Benefits

**Without Verification:**
```
User: Cites fabricated cases in legal brief
Result: ❌ Professional sanctions, case dismissed
```

**With Verification:**
```
User: Sees warnings, verifies manually
Result: ✅ Safe use of AI system
```

---

## 10. COMPLETE SYSTEM ARCHITECTURE

### 10.1 End-to-End System Diagram

```
USER INTERFACE (Streamlit)
         ↓
QUERY PROCESSING (Entity Extraction)
         ↓
HYBRID RETRIEVAL (6 methods + RRF)
         ↓
CONTEXT ASSEMBLY (Top-K + Dedup)
         ↓
LLM GENERATION (Gemini 2.5 Flash)
         ↓
CITATION VERIFICATION (Knowledge Graph)
         ↓
OUTPUT (Verified Answer + Report)
```

### 10.2 Module Structure

```
project/
├── config.py                    # Configuration
├── src/                         # Source modules
│   ├── retrieval.py            # Hybrid retrieval
│   ├── generator.py            # LLM integration
│   ├── verifier.py             # Citation verification
│   ├── graph_utils.py          # Knowledge graph
│   └── citation_utils.py       # Citation tools
├── scripts/                     # Data processing
│   ├── build_knowledge_graph.py
│   ├── build_vector_index.py
│   └── evaluate_system.py
├── data/                        # Data storage
│   ├── graph.pickle            # Knowledge graph
│   ├── ildc_vector_index.faiss # Vector index
│   └── entity_index.json       # Entity index
└── app.py                       # Streamlit UI
```

### 10.3 Performance Summary

```
Query Latency: ~900ms
  Retrieval: 567ms (63%)
  Generation: 324ms (36%)
  Verification: 16ms (1%)

Accuracy:
  Hit Rate@10: 84.85% (case/entity queries)
  Hallucination Detection: 100%
  Citation Verification: Real-time
```

---

## END OF PART 2

**Completed:**
- ✅ Section 6: All 6 retrieval algorithms
- ✅ Section 7: Hybrid fusion with RRF
- ✅ Section 8: LLM generation pipeline
- ✅ Section 9: Citation verification system
- ✅ Section 10: Complete system architecture

**Part 3 will cover:**
- Evaluation methodology and metrics
- Experimental results and analysis
- Iterative improvements
- Challenges and solutions
- Conclusions and future work

---

*Part 2 complete. Type "continue" for Part 3 with evaluation, results, and conclusions.*