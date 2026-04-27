"""LLM wrapper with mock, Gemini, and Ollama modes."""

import os
import requests
from google import genai
from dotenv import load_dotenv

load_dotenv()


LLM_MODE = "groq"  # options: "mock", "gemini", "ollama", "groq"


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


def generate_groq_answer(context: str, query: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
        
    prompt = f"""
You are a question answering system.

Answer ONLY using the given context.
If the answer is not present, say 'Not found in context'.

Context:
{context}

Question:
{query}

Give a precise and short answer.
"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )
    if response.status_code != 200:
        raise ValueError(f"Groq API Error: {response.text}")
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def generate_answer(context: str, query: str) -> str:
    prompt = f"""
You are a technical QA system.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

    if LLM_MODE == "groq" and os.getenv("GROQ_API_KEY"):
        try:
            print("Using GROQ LLM")
            return generate_groq_answer(context, query)
        except Exception as e:
            print(f"Groq failed: {e}. Falling back to Gemini.")
            # Fallback happens below
    elif LLM_MODE == "groq":
        print("GROQ_API_KEY missing. Falling back to Gemini.")
        
    # Gemini (default and fallback)
    if LLM_MODE in ["gemini", "groq"]:
        print("Using GEMINI LLM")
        try:
            return _generate_with_gemini(prompt)
        except Exception as e:
            print(f"Gemini failed: {e}")
            return f"LLM Generation failed: {e}. Please try again later."
    elif LLM_MODE == "ollama":
        print("Using OLLAMA LLM")
        return generate_with_ollama(prompt)
    else:
        raise ValueError(f"Unknown LLM_MODE '{LLM_MODE}'. Expected 'gemini', 'ollama', or 'groq'.")

