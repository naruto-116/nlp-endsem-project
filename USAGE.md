# KG-CiteRAG: Usage Examples

## Basic Query Examples

### Example 1: Constitutional Law

```python
query = "What are the fundamental rights under Article 21?"
```

Expected output:
- Retrieved cases: Maneka Gandhi, Puttaswamy (Privacy), etc.
- Answer with verified citations
- Validity status for each cited case

### Example 2: Criminal Law

```python
query = "What are the Supreme Court guidelines on arrest procedures?"
```

Expected output:
- DK Basu guidelines
- Relevant Supreme Court judgments
- Citations with verification status

### Example 3: Contract Law

```python
query = "What constitutes a valid contract under the Indian Contract Act?"
```

Expected output:
- Relevant Supreme Court interpretations
- Essential elements of contract
- Case law citations

## Document Upload Examples

### Example 1: Analyzing a Contract

1. Upload a contract PDF
2. Ask: "Does this contract have any clauses that might be void under Section 23?"
3. System will:
   - Extract text from PDF
   - Find relevant clauses
   - Compare with Supreme Court precedents
   - Provide verified answer

### Example 2: FIR Analysis

1. Upload an FIR PDF
2. Ask: "Are the grounds mentioned in this FIR sufficient for arrest?"
3. System will:
   - Analyze FIR content
   - Cross-reference with DK Basu guidelines
   - Cite relevant judgments

## Programmatic Usage

### Using the Retrieval Engine

```python
from src.retrieval import HybridRetriever
import config

# Initialize retriever
retriever = HybridRetriever(
    index_path=config.VECTOR_INDEX_PATH,
    metadata_path=config.METADATA_PATH,
    graph_path=config.GRAPH_PATH,
    embedding_model_name=config.EMBEDDING_MODEL
)

# Search
query = "privacy rights"
results = retriever.hybrid_search(query, top_k=10)

# Process results
for result in results:
    print(f"Case: {result['case_name']}")
    print(f"Score: {result['combined_score']}")
    print(f"Text: {result['text'][:200]}...")
    print("---")
```

### Using the Verification System

```python
from src.verifier import CitationVerifier
from src.graph_utils import LegalKnowledgeGraph
import config

# Load graph and verifier
graph = LegalKnowledgeGraph.load(config.GRAPH_PATH)
verifier = CitationVerifier(graph)

# Verify a citation
result = verifier.verify_citation("Kesavananda Bharati v. State of Kerala")
print(f"Exists: {result['exists']}")
print(f"Status: {result['status']}")

# Verify all citations in text
answer = "According to [Case A] and [Case B], the law is clear."
corrected_answer, report = verifier.correct_answer(answer)
print(f"Verified: {report['verified']} citations")
print(f"Not found: {report['not_found']} citations")
```

### Building a Custom Graph

```python
from src.graph_utils import LegalKnowledgeGraph

# Create new graph
kg = LegalKnowledgeGraph()

# Add cases
kg.add_case("SC_2023_1", "Case A v. State", {"date": "2023-01-15"})
kg.add_case("SC_2023_2", "Case B v. Union", {"date": "2023-02-20"})

# Add citations
kg.add_citation("SC_2023_2", "SC_2023_1")  # Case B cites Case A

# Mark overruled
kg.mark_overruled("SC_2023_1")

# Query
print(f"Case A overruled: {kg.is_overruled('SC_2023_1')}")
cited_by = kg.get_cited_by("SC_2023_1")
print(f"Cases citing Case A: {cited_by}")

# Save
kg.save("custom_graph.pickle")
```

## Advanced Examples

### Custom Retrieval Pipeline

```python
from src.retrieval import HybridRetriever
from src.generator import LegalAnswerGenerator
from src.verifier import CitationVerifier
import config

# Initialize components
retriever = HybridRetriever(...)
generator = LegalAnswerGenerator(api_key=config.GROQ_API_KEY)
verifier = CitationVerifier(graph)

# Full pipeline
query = "What is the doctrine of basic structure?"

# 1. Retrieve
results = retriever.hybrid_search(query, top_k=10)
context = retriever.get_context_for_generation(results)

# 2. Generate
answer_dict = generator.generate_answer(query, context)
raw_answer = answer_dict['answer']

# 3. Verify
corrected_answer, report = verifier.correct_answer(raw_answer)

# 4. Display
print("=" * 60)
print("QUERY:", query)
print("=" * 60)
print(corrected_answer)
print("\n" + "=" * 60)
print("VERIFICATION REPORT")
print("=" * 60)
print(f"Total Citations: {report['total_citations']}")
print(f"Verified: {report['verified']}")
print(f"Hallucination Rate: {report['hallucination_rate']:.2%}")
```

### Batch Processing Queries

```python
queries = [
    "What is the right to privacy?",
    "Grounds for divorce under Hindu law?",
    "Guidelines for police arrest?"
]

results_list = []

for query in queries:
    results = retriever.hybrid_search(query, top_k=5)
    context = retriever.get_context_for_generation(results)
    answer = generator.generate_with_fallback(query, context)
    corrected, report = verifier.correct_answer(answer)
    
    results_list.append({
        'query': query,
        'answer': corrected,
        'verification': report
    })

# Analyze overall performance
total_citations = sum(r['verification']['total_citations'] for r in results_list)
total_verified = sum(r['verification']['verified'] for r in results_list)
overall_accuracy = total_verified / max(total_citations, 1)

print(f"Overall Citation Accuracy: {overall_accuracy:.2%}")
```

### Measuring Hallucination Rate

```python
# Test queries
test_queries = [
    "What is Article 21?",
    "Privacy rights in India?",
    # ... 20 total queries
]

hallucination_counts = {'with_verification': 0, 'without_verification': 0}

for query in test_queries:
    # Without verification (raw LLM output)
    context = retriever.get_context_for_generation(
        retriever.hybrid_search(query, top_k=5)
    )
    raw_answer = generator.generate_answer(query, context)['answer']
    
    # Count citations
    raw_citations = extract_citations_from_text(raw_answer)
    for cite in raw_citations:
        result = verifier.verify_citation(cite)
        if not result['exists']:
            hallucination_counts['without_verification'] += 1
    
    # With verification
    corrected, report = verifier.correct_answer(raw_answer)
    hallucination_counts['with_verification'] = report['not_found']

print(f"Without KG-CiteRAG: {hallucination_counts['without_verification']} hallucinations")
print(f"With KG-CiteRAG: {hallucination_counts['with_verification']} hallucinations")
```

## Tips for Best Results

1. **Be specific**: "Article 21 privacy rights" works better than just "privacy"
2. **Use legal terminology**: The system understands legal concepts
3. **Upload context**: For document-specific queries, upload the PDF first
4. **Check verification**: Always review the verification report
5. **Adjust weights**: Experiment with vector/graph weight ratios
6. **Increase top_k**: For complex queries, retrieve more documents

## Common Issues and Solutions

### Low Quality Answers

- Increase `top_k` to retrieve more documents
- Adjust vector/graph weights
- Ensure your query is specific enough

### No Citations Found

- Check that the Knowledge Graph is loaded correctly
- Verify that ILDC data includes the relevant cases
- Try a broader query

### Slow Performance

- Reduce `SUBSET_SIZE` in config
- Use a smaller embedding model
- Enable GPU for FAISS (if available)

### API Errors

- Check your Groq API key
- Verify you have API credits
- Reduce request frequency
