"""LLM wrapper for answer generation using Google Gemini"""

import os
from google import genai


def generate_answer(context: str, query: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    client = genai.Client(api_key=api_key)
    
    prompt = f"""Answer the question using ONLY the provided context.
If insufficient information, say you cannot answer.

Context:
{context}

Question: {query}"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0.2}
    )
    
    return response.text


if __name__ == "__main__":
    print("Testing LLM Answer Generation")
    print("=" * 70)
    
    test_context = """
    We achieved 45% token reduction on average across 100 test queries.
    The baseline RAG used 2000 tokens per query, while our adaptive method used 1100 tokens.
    Answer correctness was maintained at 92% accuracy.
    """
    
    test_query = "What token reduction was achieved?"
    
    print(f"Context: {test_context.strip()}")
    print(f"\nQuery: {test_query}")
    print("\nGenerating answer...")
    print("-" * 70)
    
    try:
        answer = generate_answer(test_context, test_query)
        print(f"Answer: {answer}")
    except ValueError as e:
        print(f"Error: {e}")
        print("Set GOOGLE_API_KEY environment variable to test")
    except Exception as e:
        print(f"Error: {e}")
    
    print("=" * 70)
