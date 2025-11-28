"""
Test script to verify KG-CiteRAG installation and components.
Run this after setup to ensure everything is working.
"""
import sys
from pathlib import Path

print("=" * 60)
print("KG-CiteRAG System Test")
print("=" * 60)
print()

# Test 1: Python version
print("Test 1: Python Version")
print(f"  Python: {sys.version.split()[0]}")
major, minor = sys.version_info[:2]
if major >= 3 and minor >= 8:
    print("  ✓ Python version OK (3.8+)")
else:
    print("  ✗ Python version too old (need 3.8+)")
print()

# Test 2: Required imports
print("Test 2: Required Packages")
packages = {
    'networkx': 'NetworkX',
    'faiss': 'FAISS',
    'sentence_transformers': 'Sentence Transformers',
    'streamlit': 'Streamlit',
    'fitz': 'PyMuPDF',
    'groq': 'Groq',
    'numpy': 'NumPy',
    'pandas': 'Pandas'
}

failed_imports = []
for module, name in packages.items():
    try:
        __import__(module)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} - NOT INSTALLED")
        failed_imports.append(name)
print()

# Test 3: Configuration
print("Test 3: Configuration")
try:
    import config
    print(f"  ✓ Config loaded")
    print(f"  Data directory: {config.DATA_DIR}")
    print(f"  Subset size: {config.SUBSET_SIZE}")
    print(f"  Chunk size: {config.CHUNK_SIZE}")
except Exception as e:
    print(f"  ✗ Config error: {e}")
print()

# Test 4: Data files
print("Test 4: Data Files")
try:
    import config
    
    # Check ILDC
    if config.ILDC_PATH.exists():
        print(f"  ✓ ILDC dataset found")
    else:
        print(f"  ⚠ ILDC dataset not found at {config.ILDC_PATH}")
        print(f"    Run: python scripts/build_knowledge_graph.py")
    
    # Check graph
    if config.GRAPH_PATH.exists():
        print(f"  ✓ Knowledge Graph found")
    else:
        print(f"  ✗ Knowledge Graph not found")
        print(f"    Run: python scripts/build_knowledge_graph.py")
    
    # Check index
    if config.VECTOR_INDEX_PATH.exists():
        print(f"  ✓ Vector Index found")
    else:
        print(f"  ✗ Vector Index not found")
        print(f"    Run: python scripts/build_vector_index.py")
    
    # Check metadata
    if config.METADATA_PATH.exists():
        print(f"  ✓ Metadata found")
    else:
        print(f"  ✗ Metadata not found")
        print(f"    Run: python scripts/build_vector_index.py")
        
except Exception as e:
    print(f"  ✗ Error checking files: {e}")
print()

# Test 5: Core modules
print("Test 5: Core Modules")
try:
    from src.citation_utils import normalize_case_name
    from src.graph_utils import LegalKnowledgeGraph
    print("  ✓ Citation utilities")
    print("  ✓ Graph utilities")
    
    # Test normalization
    test_case = "Kesavananda Bharati vs. State of Kerala"
    normalized = normalize_case_name(test_case)
    print(f"  ✓ Citation normalization works")
    
except Exception as e:
    print(f"  ✗ Module error: {e}")
print()

# Test 6: API key
print("Test 6: API Configuration")
try:
    import config
    if config.GROQ_API_KEY:
        print(f"  ✓ Groq API key configured")
    else:
        print(f"  ⚠ No Groq API key (will use dummy generator)")
        print(f"    Add GROQ_API_KEY to .env file")
except Exception as e:
    print(f"  ✗ Config error: {e}")
print()

# Summary
print("=" * 60)
print("Test Summary")
print("=" * 60)

if failed_imports:
    print(f"✗ {len(failed_imports)} package(s) missing:")
    for pkg in failed_imports:
        print(f"  - {pkg}")
    print()
    print("Run: pip install -r requirements.txt")
else:
    print("✓ All packages installed")

print()

try:
    import config
    all_files_exist = (
        config.GRAPH_PATH.exists() and 
        config.VECTOR_INDEX_PATH.exists() and
        config.METADATA_PATH.exists()
    )
    
    if all_files_exist:
        print("✓ All data files ready")
        print()
        print("System is READY! Run: streamlit run app.py")
    else:
        print("⚠ Some data files missing")
        print()
        print("Next steps:")
        if not config.ILDC_PATH.exists():
            print("  1. Get ILDC data or create sample")
        if not config.GRAPH_PATH.exists():
            print("  2. Run: python scripts/build_knowledge_graph.py")
        if not config.VECTOR_INDEX_PATH.exists():
            print("  3. Run: python scripts/build_vector_index.py")
        print("  4. Run: streamlit run app.py")
except:
    pass

print()
print("=" * 60)
