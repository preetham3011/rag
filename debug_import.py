"""Debug import issue"""
import sys
import traceback

print("Attempting to import retriever module...")
print("=" * 70)

try:
    print("\n1. Importing module...")
    import src.retrieval.retriever as retriever_module
    print("   Module imported successfully")
    print(f"   Module file: {retriever_module.__file__}")
    print(f"   Module contents: {dir(retriever_module)}")
    
    print("\n2. Attempting to import function...")
    from src.retrieval.retriever import retrieve_with_intent
    print("   Function imported successfully!")
    print(f"   Function: {retrieve_with_intent}")
    
except Exception as e:
    print(f"\n✗ Error occurred:")
    print(f"   Type: {type(e).__name__}")
    print(f"   Message: {e}")
    print(f"\nFull traceback:")
    traceback.print_exc()

print("\n" + "=" * 70)
