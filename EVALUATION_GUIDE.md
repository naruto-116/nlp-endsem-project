# KG-CiteRAG Evaluation Guide

This guide explains how to evaluate the KG-CiteRAG system using the automated evaluation script.

## Quick Start

```powershell
# Run evaluation with sample data
python scripts/evaluate_system.py
```

Results will be saved to `evaluation_results.json`.

## Evaluation Components

### 1. Retrieval Evaluation

Tests how well the system finds relevant documents:

- **Precision@K**: What % of retrieved docs are relevant?
- **Recall@K**: What % of relevant docs were found?
- **MRR**: How quickly does it find the first relevant doc?
- **NDCG**: Is the ranking order good?
- **Hit Rate**: Does it find at least one relevant doc?

### 2. Generation Evaluation

Tests answer quality:

- **Token F1**: Word overlap with reference answer
- **ROUGE-L**: Longest matching phrase sequences
- **Exact Match**: Perfect matches

### 3. Citation Evaluation

Tests citation correctness:

- **Citation Precision**: Are the citations real and relevant?
- **Citation Recall**: Did it cite all expected cases?
- **Hallucination Rate**: How many fake citations?

### 4. Performance Evaluation

Tests system speed:

- **Retrieval Time**: Vector search + graph traversal
- **Generation Time**: LLM response time
- **Verification Time**: Citation checking time
- **Total Time**: End-to-end latency

## Creating Your Test Data

Create `test_data.json` with your test queries:

```json
{
  "queries": [
    {
      "query": "Your legal question here",
      "relevant_docs": ["case_id_1", "case_id_2"],
      "expected_citations": ["Case Name 1", "Article X"],
      "reference_answer": "Expected answer text..."
    }
  ]
}
```

### Test Data Guidelines

1. **Include diverse queries**:
   - Constitutional law questions
   - Criminal law queries
   - Civil procedure questions
   - Specific case-based queries

2. **Annotate relevant documents**:
   - Use actual case IDs from your ILDC dataset
   - Include 3-10 relevant cases per query
   - Rank them by relevance if possible

3. **List expected citations**:
   - Cases that should be cited
   - Constitutional articles
   - Legal provisions

4. **Write reference answers**:
   - 2-4 sentences
   - Include key legal points
   - Use proper legal terminology

## Running Custom Evaluation

```python
from scripts.evaluate_system import KGCiteRAGEvaluator

# Initialize evaluator
evaluator = KGCiteRAGEvaluator()

# Load your test data
results = evaluator.run_full_evaluation('path/to/your/test_data.json')

# Access specific metrics
retrieval_scores = results['retrieval']
print(f"Precision@10: {retrieval_scores['precision@10']:.3f}")
print(f"MRR: {retrieval_scores['mrr']:.3f}")

generation_scores = results['generation']
print(f"Token F1: {generation_scores['token_f1']:.3f}")

citation_scores = results['citations']
print(f"Hallucination Rate: {citation_scores['hallucination_rate']:.3f}")
```

## Evaluation Best Practices

### 1. Test Set Size

- **Minimum**: 20-30 queries for basic evaluation
- **Recommended**: 50-100 queries for reliable metrics
- **Comprehensive**: 200+ queries for publication-quality results

### 2. Query Diversity

Include queries of varying:
- **Complexity**: Simple facts → Complex reasoning
- **Specificity**: Broad topics → Specific cases
- **Type**: What, Why, How, Compare, Analyze

### 3. Ground Truth Annotation

- Use legal experts for annotation when possible
- Multiple annotators for inter-annotator agreement
- Document annotation guidelines

### 4. Baseline Comparisons

Always compare against:
- Pure RAG (no knowledge graph)
- Pure graph-based retrieval
- No citation verification

## Interpreting Results

### Good Performance Indicators

✅ **Retrieval**:
- Precision@10 > 0.60
- MRR > 0.75
- Hit Rate@10 > 0.85

✅ **Generation**:
- Token F1 > 0.55
- ROUGE-L > 0.50

✅ **Citations**:
- Hallucination Rate < 0.15
- Citation Precision > 0.80

✅ **Performance**:
- Total Latency < 5000ms
- Retrieval < 1000ms

### Red Flags

🚨 **Low Precision** (< 0.40):
- Vector embeddings may need tuning
- Query reformulation needed
- Index quality issues

🚨 **High Hallucination** (> 0.25):
- Knowledge graph coverage insufficient
- LLM needs better prompting
- Verification logic broken

🚨 **Poor Recall** (< 0.40):
- Missing relevant documents in index
- Graph connectivity too sparse
- Hybrid fusion weights need adjustment

## Advanced Evaluation

### A/B Testing Different Configurations

```python
# Test different fusion weights
configs = [
    {'vector_weight': 0.5, 'graph_weight': 0.5},
    {'vector_weight': 0.7, 'graph_weight': 0.3},
    {'vector_weight': 0.3, 'graph_weight': 0.7},
]

for config in configs:
    print(f"\nTesting config: {config}")
    # Run evaluation with config
    # Compare results
```

### Statistical Significance Testing

```python
import scipy.stats as stats

# Compare two systems
system_a_scores = [0.75, 0.82, 0.68, ...]
system_b_scores = [0.78, 0.85, 0.71, ...]

t_stat, p_value = stats.ttest_rel(system_a_scores, system_b_scores)
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✓ Improvement is statistically significant")
```

### Error Analysis

Look at failed queries:
1. Identify patterns in failures
2. Check which types of queries fail
3. Analyze why retrieval/generation failed
4. Improve system based on findings

## Continuous Evaluation

Set up automated evaluation:

```powershell
# Run weekly evaluation
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 9am
$action = New-ScheduledTaskAction -Execute "python" -Argument "scripts/evaluate_system.py"
Register-ScheduledTask -TaskName "KG-CiteRAG-Evaluation" -Trigger $trigger -Action $action
```

Track metrics over time:
- Monitor performance degradation
- Detect data drift
- Validate improvements

## Example Results

```
==================================================
RETRIEVAL RESULTS (Averaged)
==================================================
precision@5         : 0.7200
precision@10        : 0.6500
recall@5           : 0.5800
recall@10          : 0.7400
mrr                : 0.8100
ndcg@10            : 0.7650
hit_rate@10        : 0.9200

==================================================
GENERATION RESULTS (Averaged)
==================================================
token_f1           : 0.6150
rouge_l            : 0.5820
exact_match        : 0.1800

==================================================
CITATION RESULTS (Averaged)
==================================================
citation_precision : 0.8500
citation_recall    : 0.7800
citation_f1        : 0.8140
hallucination_rate : 0.0850

==================================================
PERFORMANCE RESULTS (Averaged)
==================================================
retrieval_ms       : 456.23 ms
generation_ms      : 1823.45 ms
verification_ms    : 78.12 ms
total_ms           : 2357.80 ms
```

## Troubleshooting

### "No test queries found"
- Check JSON file format
- Ensure 'queries' key exists
- Validate JSON syntax

### "No API key found"
- Set GEMINI_API_KEY or GROQ_API_KEY in .env
- Generation/citation metrics will be skipped without API

### Slow evaluation
- Reduce test set size
- Use faster embedding model
- Run on GPU if available

### Low scores across all metrics
- Check if system is properly initialized
- Verify data files exist (graph.pickle, index.faiss)
- Test with simple queries first

## Publishing Results

When reporting metrics:
1. **State test set size**: "Evaluated on 75 test queries"
2. **Describe test data**: "Covering 5 legal domains..."
3. **Compare baselines**: "30% improvement over pure RAG"
4. **Show variance**: "P@10: 0.65 ± 0.08"
5. **Significance**: "Statistically significant (p < 0.01)"

## Further Reading

- **BEIR Benchmark**: Standard IR evaluation practices
- **KILT**: Knowledge-intensive language tasks evaluation
- **MS MARCO**: Passage ranking evaluation methodology
- **Legal AI Papers**: Domain-specific evaluation approaches
