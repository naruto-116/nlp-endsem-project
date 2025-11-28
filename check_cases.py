import json

# Load metadata
with open('data/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

print(f"Total chunks: {len(metadata)}")
print("\nSample metadata entries:")
for i, m in enumerate(metadata[:5]):
    print(f"\nChunk {i}:")
    print(f"  case_id: {m.get('case_id', 'N/A')}")
    print(f"  text preview: {m.get('text', '')[:80]}...")

print("\n" + "="*60)
print("Unique case IDs:")
case_ids = set(m.get('case_id', 'N/A') for m in metadata)
print(f"  Total unique: {len(case_ids)}")
print(f"  All case IDs: {sorted(list(case_ids))}")
