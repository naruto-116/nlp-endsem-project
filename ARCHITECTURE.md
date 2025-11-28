# KG-CiteRAG Project Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                      (Streamlit App)                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Input Processing Layer                    │
├─────────────────────────────────────────────────────────────┤
│  • Text Query Parser                                         │
│  • PDF Processor (PyMuPDF)                                   │
│  • Citation Extractor                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   Vector Search Engine   │  │   Graph Search Engine    │
├──────────────────────────┤  ├──────────────────────────┤
│  • FAISS Index           │  │  • NetworkX Graph        │
│  • Sentence Transformers │  │  • Citation Network      │
│  • Semantic Similarity   │  │  • Structural Relevance  │
└──────────────────────────┘  └──────────────────────────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Fusion & Reranking                        │
├─────────────────────────────────────────────────────────────┤
│  • Weighted Score Combination                               │
│  • Result Deduplication                                     │
│  • Context Assembly                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   LLM Generation Layer                       │
├─────────────────────────────────────────────────────────────┤
│  • Groq API (Llama-3)                                       │
│  • Structured Prompting                                     │
│  • Citation Formatting                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Verification Layer (KEY!)                   │
├─────────────────────────────────────────────────────────────┤
│  • Citation Extraction (Regex)                              │
│  • Graph Lookup Verification                                │
│  • Validity Check (Overruled?)                              │
│  • Answer Correction                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Output Layer                          │
├─────────────────────────────────────────────────────────────┤
│  • Verified Answer                                          │
│  • Citation Report                                          │
│  • Validity Flags                                           │
│  • Graph Visualization                                      │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Layer

**Files:**
- `scripts/data_loader.py`: ILDC dataset loader
- `config.py`: Configuration and paths

**Responsibilities:**
- Load ILDC JSON data
- Parse case metadata
- Extract citations
- Filter and subset data

**Output:**
- Structured case dictionaries
- Citation relationships

### 2. Knowledge Graph (Core Component)

**Files:**
- `src/graph_utils.py`: Graph operations
- `scripts/build_knowledge_graph.py`: Graph builder

**Data Structure:**
```python
Node: {
    'id': 'SC_2023_123',
    'name': 'Case Name v. State',
    'date': '2023-01-15',
    'overruled': False,
    'judges': [...],
    ...
}

Edge: {
    'source': 'citing_case_id',
    'target': 'cited_case_id',
    'relation': 'CITES'
}
```

**Key Operations:**
- `add_case()`: Add case node
- `add_citation()`: Create citation edge
- `is_overruled()`: Check validity
- `get_cited_by()`: Find incoming citations
- `find_related_cases()`: Graph traversal

### 3. Vector Index

**Files:**
- `scripts/build_vector_index.py`: Index builder
- `src/retrieval.py`: Search engine

**Pipeline:**
```
Text → Chunks → Embeddings → FAISS Index
```

**Chunking:**
- 500 words per chunk
- 50 word overlap
- Preserves context

**Embeddings:**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Fast CPU inference

**Index:**
- FAISS IndexFlatL2
- Exact L2 distance search
- ~100ms for 10k vectors

### 4. Hybrid Retrieval

**File:** `src/retrieval.py`

**Algorithm:**
```python
def hybrid_search(query, top_k):
    # Vector search
    vector_results = semantic_search(query)
    
    # Graph search
    graph_results = graph_traverse(query_entities)
    
    # Fusion
    combined_scores = (
        α * vector_scores + 
        β * graph_scores
    )
    
    return top_k(combined_scores)
```

**Default Weights:**
- Vector: 0.7 (text similarity)
- Graph: 0.3 (structural relevance)

### 5. Generation

**File:** `src/generator.py`

**Prompt Template:**
```
You are a legal AI assistant. Answer using ONLY the context.
When citing, use format: [Case Name]

CONTEXT:
{retrieved_documents}

QUESTION:
{user_query}

ANSWER:
```

**LLM:**
- Groq API (Llama-3-8B)
- Temperature: 0.3 (deterministic)
- Max tokens: 500

### 6. Verification (The Innovation!)

**File:** `src/verifier.py`

**Process:**
```python
1. Extract citations: regex r'\[(.*?)\]'
2. Normalize case names
3. Lookup in graph: exists?
4. Check validity: overruled?
5. Annotate answer with flags
```

**Verification Report:**
```python
{
    'total_citations': 5,
    'verified': 4,
    'not_found': 1,
    'overruled': 0,
    'hallucination_rate': 0.20  # 20%
}
```

### 7. Citation Utilities

**File:** `src/citation_utils.py`

**Key Functions:**
- `normalize_case_name()`: Standardize format
- `extract_citations_from_text()`: Regex extraction
- `extract_citations_from_judgment()`: Legal patterns
- `fuzzy_match_case_name()`: Approximate matching

**Normalization Example:**
```python
"Kesavananda Bharati vs State of Kerala"
→ "kesavananda bharati state of kerala"
```

### 8. PDF Processing

**File:** `src/pdf_processor.py`

**Features:**
- Text extraction (PyMuPDF)
- Citation detection
- Document type inference
- Structure analysis

**Workflow:**
```
PDF Upload → Extract Text → Chunk → 
Temporary Index → Merge with Main Index → 
Query Processing
```

### 9. User Interface

**File:** `app.py`

**Tabs:**
1. **Query**: Basic QA interface
2. **Upload**: PDF document analysis
3. **About**: Project information

**Features:**
- Real-time search
- Verification visualization
- Parameter tuning
- Graph statistics

## Data Flow Example

### Query: "What is the right to privacy?"

```
1. INPUT
   └─ User query: "What is the right to privacy?"

2. HYBRID RETRIEVAL
   ├─ Vector Search
   │  ├─ Embed query → [0.23, -0.45, ...]
   │  ├─ FAISS search → Top 10 chunks
   │  └─ Scores: [0.89, 0.85, 0.82, ...]
   │
   └─ Graph Search
      ├─ Extract entities: ["privacy", "Article 21"]
      ├─ Find landmark cases
      └─ Scores: [0.95, 0.88, ...]

3. FUSION
   └─ Combined scores = 0.7*vector + 0.3*graph
   └─ Reranked results: [Case1, Case2, Case3, ...]

4. CONTEXT ASSEMBLY
   └─ "[Doc 1] Puttaswamy v. Union: Privacy is a 
       fundamental right under Article 21..."

5. LLM GENERATION
   └─ Prompt + Context → Groq API
   └─ Raw answer: "According to [Puttaswamy v. Union 
       of India] and [Maneka Gandhi], privacy..."

6. VERIFICATION
   ├─ Extract: ["Puttaswamy v. Union of India", 
   │            "Maneka Gandhi"]
   ├─ Lookup in graph
   │  ├─ Puttaswamy: ✓ EXISTS, VALID
   │  └─ Maneka Gandhi: ✓ EXISTS, VALID
   └─ Report: 2/2 verified (100%)

7. OUTPUT
   └─ Verified answer + Report + Graph viz
```

## Performance Characteristics

### Time Complexity

- **Vector Search**: O(n) for flat index, O(log n) for IVF
- **Graph Search**: O(k·d) where k=top cases, d=depth
- **Verification**: O(m) where m=citations in answer

### Space Complexity

- **FAISS Index**: ~1.5MB per 1000 documents
- **Knowledge Graph**: ~500KB per 1000 cases
- **Metadata**: ~2MB per 1000 cases

### Typical Performance (1000 case subset)

- End-to-end query: **2-3 seconds**
- Retrieval: 0.5s
- Generation: 1-2s
- Verification: 0.1s

## Scalability

### Small (Development)
- Cases: 100-500
- Hardware: CPU only
- RAM: 4GB
- Time: <1 second per query

### Medium (Demo)
- Cases: 1,000-5,000
- Hardware: CPU, 8GB RAM
- Time: 2-3 seconds

### Large (Production)
- Cases: 10,000-35,000
- Hardware: GPU recommended
- RAM: 16GB+
- Time: 3-5 seconds
- Optimization: Use FAISS IVF index

## Extension Points

### Custom Models
- Replace MiniLM with larger model
- Use domain-specific embeddings
- Fine-tune LLM on legal data

### Enhanced Graph
- Add case-statute relationships
- Track overruling chains
- Include dissenting opinions

### Advanced Features
- Multi-language support
- Case recommendation
- Automated brief generation
- Citation network visualization

## Error Handling

### Graceful Degradation
1. No API key → Use dummy generator
2. Missing graph → Vector-only search
3. No ILDC data → Sample generation
4. FAISS error → Fallback to exact search

### Validation
- Input sanitization
- Citation format checking
- Graph consistency verification
- Index integrity checks
