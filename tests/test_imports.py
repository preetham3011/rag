"""Test that all imports work correctly"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing imports...")
print("=" * 70)

try:
    print("1. Importing vector_store...")
    from src.indexing.vector_store import build_faiss_index, search_index
    print("   ✓ Success")
except ImportError as e:
    print(f"   ✗ Failed: {e}")

try:
    print("2. Importing retriever...")
    from src.retrieval.retriever import retrieve_with_intent
    print("   ✓ Success")
except ImportError as e:
    print(f"   ✗ Failed: {e}")

try:
    print("3. Importing intent_detector...")
    from src.retrieval.intent_detector import detect_intent
    print("   ✓ Success")
except ImportError as e:
    print(f"   ✗ Failed: {e}")

try:
    print("4. Importing evidence_selector...")
    from src.compression.evidence_selector import select_evidence
    print("   ✓ Success")
except ImportError as e:
    print(f"   ✗ Failed: {e}")

try:
    print("5. Importing budget_manager...")
    from src.compression.budget_manager import apply_budget
    print("   ✓ Success")
except ImportError as e:
    print(f"   ✗ Failed: {e}")

try:
    print("6. Importing compressor...")
    from src.compression.compressor import compress_context
    print("   ✓ Success")
except ImportError as e:
    print(f"   ✗ Failed: {e}")

try:
    print("7. Importing baseline_rag...")
    from evaluation.baseline_rag import run_baseline_rag
    print("   ✓ Success")
except ImportError as e:
    print(f"   ✗ Failed: {e}")

print("=" * 70)
print("Import test complete!")
print("\nIf all imports succeeded, the module structure is correct.")
print("If sentence_transformers or faiss errors appear, run: pip install -r requirements.txt")
