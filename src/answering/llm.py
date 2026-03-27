"""LLM wrapper with mock, Gemini, and Ollama modes."""

import os
import requests
from google import genai


LLM_MODE = "mock"  # options: "mock", "gemini", "ollama"


def generate_with_ollama(prompt: str) -> str:
    """Generate an answer using local Ollama server."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("response", "Ollama returned an empty response.")
    except requests.exceptions.ConnectionError:
        print("Ollama server is not reachable at http://localhost:11434. Start it with: ollama serve")
        return "Ollama server is not running. Please start Ollama and try again."
    except requests.exceptions.Timeout:
        print("Ollama request timed out.")
        return "Ollama request timed out. Please try again."
    except requests.exceptions.RequestException as exc:
        print(f"Ollama request failed: {exc}")
        return "Failed to generate answer with Ollama."
    except ValueError as exc:
        print(f"Invalid JSON response from Ollama: {exc}")
        return "Ollama returned an invalid response."


def generate_mock_answer(context: str, query: str) -> str:
    """Deterministic lightweight answer used for fast tests."""
    del query  # Query is intentionally ignored in deterministic mock mode.
    return context[:200] if context else "No context available."


def _generate_with_gemini(prompt: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0.2}
    )
    return response.text


def generate_answer(context: str, query: str) -> str:
    prompt = f"""
You are a technical QA system.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

    if LLM_MODE == "mock":
        print("Using MOCK LLM")
        return generate_mock_answer(context, query)
    elif LLM_MODE == "gemini":
        print("Using GEMINI LLM")
        return _generate_with_gemini(prompt)
    elif LLM_MODE == "ollama":
        print("Using OLLAMA LLM")
        return generate_with_ollama(prompt)
    else:
        print(f"Unknown LLM_MODE '{LLM_MODE}', defaulting to mock mode.")
        return generate_mock_answer(context, query)


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
