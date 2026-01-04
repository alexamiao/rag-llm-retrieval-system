"""
RAG Helper Module - Pre-built Complex Library Code
This module handles the complex library interactions so students can focus on Python fundamentals.

Students - DO NOT modify this file. This is provided code.
"""

from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json
import time


# ============================================================
# PRE-BUILT: Embeddings with HuggingFace
# ============================================================

class EmbeddingModel:
    """
    Pre-built wrapper for HuggingFace embeddings.
    Students use this class but don't need to understand the implementation.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the embedding model."""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"✓ Model loaded!")

    def embed_text(self, text: str) -> List[float]:
        """
        Convert a single text to embedding vector.

        Args:
            text: Input text string

        Returns:
            List of floats representing the embedding
        """
        return self.model.encode(text).tolist()

    def embed_multiple(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts to embeddings (faster than one-by-one).

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        print(f"Embedding {len(texts)} texts...")
        embeddings = self.model.encode(texts).tolist()
        print(f"✓ Complete!")
        return embeddings


# ============================================================
# PRE-BUILT: Vector Database with ChromaDB
# ============================================================

class VectorDatabase:
    """
    Pre-built wrapper for ChromaDB.
    Students use this class but don't need to understand the implementation.
    """

    def __init__(self, collection_name: str = "student_rag",
                 persist_directory: str = "./student_chroma_db"):
        """Initialize the vector database."""
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Student RAG project collection"}
        )
        print(f"✓ Vector database initialized")
        print(f"  Collection: {collection_name}")
        print(f"  Current documents: {self.collection.count()}")

    def add_chunks(self, chunks: List[str], embeddings: List[List[float]],
                   metadatas: List[Dict]):
        """
        Add document chunks to the database.

        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors (one per chunk)
            metadatas: List of metadata dictionaries (one per chunk)
        """
        # Create unique IDs for each chunk
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"✓ Added {len(chunks)} chunks to database")

    def search(self, query_embedding: List[float], top_k: int = 3) -> Dict:
        """
        Search for similar chunks using a query embedding.

        Args:
            query_embedding: The embedding vector of the query
            top_k: Number of results to return

        Returns:
            Dictionary with 'documents', 'metadatas', and 'distances'
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results

    def clear_database(self):
        """Delete all documents from the database."""
        self.client.delete_collection(self.collection.name)
        print("✓ Database cleared")


# ============================================================
# PRE-BUILT: Ollama LLM Connection
# ============================================================

class LLM:
    """
    Pre-built wrapper for Ollama LLM.
    Students use this class but don't need to understand the implementation.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:11434",
                 model: str = "mistral"):
        """Initialize connection to Ollama LLM."""
        self.base_url = base_url
        self.model = model
        self.generate_url = f"{base_url}/api/generate"
        print(f"✓ LLM initialized: {model} at {base_url}")

    def generate_answer(self, prompt: str) -> str:
        """
        Generate text using the LLM.

        Args:
            prompt: The input prompt

        Returns:
            Generated text
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.generate_url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.RequestException as e:
            error_msg = f"Error: Could not connect to Ollama at {self.base_url}"
            error_msg += f"\nMake sure Docker container is running!"
            error_msg += f"\nError details: {e}"
            return error_msg

    def test_connection(self) -> bool:
        """
        Test if the LLM is accessible.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.generate_answer("Say 'OK' if you can read this.")
            return "OK" in response or "ok" in response.lower()
        except:
            return False


# ============================================================
# PRE-BUILT: Timing Utility
# ============================================================

class Timer:
    """
    Simple timer utility for measuring function execution time.
    Students can use this to measure performance.
    """

    def __init__(self):
        """Initialize timer."""
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer."""
        self.start_time = time.time()

    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.

        Returns:
            Elapsed time in seconds
        """
        self.end_time = time.time()
        return self.end_time - self.start_time

    def __enter__(self):
        """Support 'with' statement."""
        self.start()
        return self

    def __exit__(self, *args):
        """Support 'with' statement."""
        self.stop()

    def elapsed(self) -> float:
        """
        Get elapsed time.

        Returns:
            Elapsed time in seconds
        """
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


# ============================================================
# PRE-BUILT: Simple Visualization Helpers
# ============================================================

def print_separator(title: str = "", char: str = "=", width: int = 60):
    """
    Print a visual separator with optional title.

    Args:
        title: Optional title to display
        char: Character to use for separator
        width: Width of separator
    """
    if title:
        side_len = (width - len(title) - 2) // 2
        print(f"{char * side_len} {title} {char * side_len}")
    else:
        print(char * width)


def print_search_results(results: Dict, max_preview: int = 150):
    """
    Pretty print search results from vector database.

    Args:
        results: Results dictionary from VectorDatabase.search()
        max_preview: Maximum characters to show in preview
    """
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    print(f"\nFound {len(documents)} results:\n")

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        similarity = 1 - dist  # Convert distance to similarity
        print(f"[Result {i+1}]")
        print(f"  Source: {meta.get('source', 'unknown')}")
        print(f"  Similarity: {similarity:.3f}")
        print(f"  Preview: {doc[:max_preview]}...")
        print()


def print_rag_answer(question: str, answer: str, sources: List[str],
                     latency: float = None):
    """
    Pretty print a RAG answer.

    Args:
        question: The question asked
        answer: The generated answer
        sources: List of source documents
        latency: Optional time taken to generate answer
    """
    print_separator("RAG ANSWER")
    print(f"\nQUESTION: {question}\n")
    print(f"ANSWER:\n{answer}\n")
    print(f"SOURCES: {', '.join(set(sources))}")

    if latency:
        print(f"TIME: {latency:.2f} seconds")

    print_separator()


# ============================================================
# HELPER: Check if all required libraries are installed
# ============================================================

def check_setup():
    """
    Check if all required libraries are properly installed.
    Run this at the beginning of your notebook.
    """
    print("Checking setup...")

    required_packages = {
        'chromadb': chromadb,
        'sentence_transformers': SentenceTransformer,
        'requests': requests,
    }

    all_good = True
    for name, package in required_packages.items():
        try:
            print(f"✓ {name} is installed")
        except:
            print(f"✗ {name} is NOT installed")
            all_good = False

    if all_good:
        print("\n✓ All required packages are installed!")
        print("You're ready to start!")
    else:
        print("\n✗ Some packages are missing. Please install them first.")

    return all_good


if __name__ == "__main__":
    # Quick test of the module
    print("RAG Helpers Module - Pre-built Complex Code")
    print("This module provides:")
    print("  - EmbeddingModel: Convert text to vectors")
    print("  - VectorDatabase: Store and search embeddings")
    print("  - LLM: Generate answers using Ollama")
    print("  - Timer: Measure performance")
    print("  - Visualization helpers: Pretty printing")
    print("\nStudents will use these tools to build their RAG system!")
