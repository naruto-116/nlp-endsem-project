"""
Script to extract real ILDC cases and create a proper test dataset
"""
import json
import random

# Load ILDC data
cases = []
with open('data/ILDC_single.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 100:  # Limit to first 100 cases
            break
        case = json.loads(line)
        cases.append(case)

print(f"Loaded {len(cases)} cases from ILDC dataset\n")

# Display sample case
print("="*70)
print("SAMPLE CASE STRUCTURE:")
print("="*70)
sample = cases[0]
print(f"ID: {sample.get('id')}")
print(f"Name: {sample.get('name')}")
print(f"Date: {sample.get('date')}")
print(f"Court: {sample.get('court')}")
print(f"Judges: {sample.get('judges', [])[:2]}")
print(f"Text length: {len(sample.get('text', ''))} characters")
print(f"Citations: {sample.get('citations', [])[:5]}")
print(f"Statutes: {sample.get('statutes', [])[:5]}")
print(f"Outcome: {sample.get('outcome')}")
print(f"\nText preview:\n{sample.get('text', '')[:500]}...")

# Count cases with different features
print("\n" + "="*70)
print("DATASET STATISTICS:")
print("="*70)
cases_with_citations = sum(1 for c in cases if c.get('citations'))
cases_with_statutes = sum(1 for c in cases if c.get('statutes'))
print(f"Cases with citations: {cases_with_citations}/{len(cases)}")
print(f"Cases with statutes: {cases_with_statutes}/{len(cases)}")

# Sample case names
print("\nSample case names:")
for i, case in enumerate(cases[:10]):
    print(f"  {i+1}. {case.get('name')}")
