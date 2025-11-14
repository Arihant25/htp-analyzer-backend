"""
RAG Query Module for HTP Analysis
Retrieves relevant context from FAISS index using Gemini Embeddings.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from google import genai
from google.genai import types

from rag_indexer import RAGIndexer


class RAGQueryEngine:
    """Query engine for retrieving relevant context from FAISS index."""

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        api_key: Optional[str] = None,
        model_name: str = "gemini-embedding-001",
        embedding_dim: int = 768,
    ):
        """
        Initialize RAG query engine.

        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata pickle file
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model_name: Gemini embedding model name
            embedding_dim: Dimension of embeddings (must match index)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY must be set in environment or passed as parameter"
            )

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.embedding_dim = embedding_dim

        # Load index
        indexer = RAGIndexer(api_key=self.api_key, embedding_dim=self.embedding_dim)
        indexer.load_index(index_path, metadata_path)

        self.index = indexer.index
        self.chunks = indexer.chunks
        self.metadata = indexer.metadata

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query using Gemini.

        Args:
            query: Query text

        Returns:
            Numpy array of embedding
        """
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY", output_dimensionality=self.embedding_dim
            ),
        )

        # Extract embedding values
        embedding_values = np.array(result.embeddings[0].values, dtype=np.float32)

        # Normalize for non-3072 dimensions
        if self.embedding_dim != 3072:
            embedding_values = embedding_values / np.linalg.norm(embedding_values)

        return embedding_values.reshape(1, -1)

    def search(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[str, float, dict]]:
        """
        Search for relevant chunks using the query.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of (chunk_text, distance, metadata) tuples
        """
        # Generate query embedding
        query_embedding = self.embed_query(query)

        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)

        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):  # Valid index
                results.append(
                    (
                        self.chunks[idx],
                        float(dist),
                        self.metadata[idx],
                    )
                )

        return results

    def _format_chunk_with_pages(self, chunk_text: str, metadata: dict) -> str:
        """
        Format chunk text with page number references.

        Args:
            chunk_text: The chunk text
            metadata: Metadata dict containing page_numbers

        Returns:
            Formatted chunk with page references
        """
        page_numbers = metadata.get("page_numbers", [])
        if page_numbers:
            pages_str = ", ".join(str(p) for p in page_numbers)
            return f"{chunk_text}\n\n[Reference Pages: {pages_str}]"
        return chunk_text

    def get_context_for_analysis(
        self,
        analysis_features: dict,
        top_k: int = 5,
    ) -> str:
        """
        Get relevant context from knowledge base for HTP analysis.

        Args:
            analysis_features: Dictionary with detected features and characteristics
            top_k: Number of relevant chunks to retrieve

        Returns:
            Combined context string from knowledge base
        """
        # Build query from analysis features
        query_parts = []

        if "house_size_category" in analysis_features:
            query_parts.append(f"house size {analysis_features['house_size_category']}")

        if "missing_features" in analysis_features:
            missing = ", ".join(analysis_features["missing_features"])
            if missing:
                query_parts.append(f"missing {missing}")

        if "detected_features" in analysis_features:
            detected = ", ".join(analysis_features["detected_features"])
            if detected:
                query_parts.append(f"{detected}")

        if "door_present" in analysis_features and not analysis_features["door_present"]:
            query_parts.append("no door missing door")

        if "window_count" in analysis_features:
            window_count = analysis_features["window_count"]
            if window_count == 0:
                query_parts.append("no windows missing windows")
            elif window_count > 4:
                query_parts.append("many windows excessive windows")

        if "chimney_present" in analysis_features and not analysis_features["chimney_present"]:
            query_parts.append("no chimney missing chimney")

        # Add size-specific queries
        if "door_characteristics" in analysis_features:
            door_chars = analysis_features["door_characteristics"]
            if door_chars.get("size_category") == "tiny":
                query_parts.append("tiny door small door fearfulness")
            elif door_chars.get("size_category") == "large":
                query_parts.append("large door oversized door dependency")
        
        if "chimney_characteristics" in analysis_features:
            chimney_chars = analysis_features["chimney_characteristics"]
            if chimney_chars.get("size") == "large":
                query_parts.append("large chimney oversized chimney")
            elif chimney_chars.get("size") == "small":
                query_parts.append("small chimney tiny chimney")
        
        if "roof_characteristics" in analysis_features:
            roof_chars = analysis_features["roof_characteristics"]
            if roof_chars.get("size") == "large":
                query_parts.append("large roof oversized roof fantasy")
            elif roof_chars.get("size") == "small":
                query_parts.append("small roof tiny roof")

        # Combine query parts
        query = " ".join(query_parts)
        if not query:
            query = "house psychological interpretation HTP analysis"

        # DEBUG: Print the query being used
        print(f"\n🔍 DEBUG: RAG Query: '{query}'\n")

        # Search for relevant context
        results = self.search(query, top_k=top_k)

        # Combine top results into context
        context_parts = []
        for i, (chunk, distance, meta) in enumerate(results, 1):
            # DEBUG: Print each search result with distance
            print(f"  Result {i}: distance={distance:.4f}")
            
            # Only include results with reasonable similarity (lower distance = more similar)
            if distance < 1.5:  # Adjust threshold as needed
                formatted_chunk = self._format_chunk_with_pages(chunk, meta)
                context_parts.append(f"[Reference {i}]\n{formatted_chunk}")
                print(f"    ✓ Included in context (distance < 1.5)")
            else:
                print(f"    ✗ Excluded (distance >= 1.5)")

        print(f"\n📊 DEBUG: Retrieved {len(context_parts)} relevant chunks from {len(results)} results\n")

        context = "\n\n".join(context_parts)
        return context


def main():
    """Test RAG query functionality."""
    # Find index files
    rag_dir = Path(__file__).parent.parent / "RAG"
    index_path = rag_dir / "faiss_index.bin"
    metadata_path = rag_dir / "index_metadata.pkl"

    if not index_path.exists() or not metadata_path.exists():
        print("❌ Index not found. Please run rag_indexer.py first to build the index.")
        return

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        return

    # Initialize query engine
    print("🔍 Initializing RAG Query Engine...")
    query_engine = RAGQueryEngine(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        api_key=api_key,
    )

    # Test queries
    test_queries = [
        "missing door psychological meaning",
        "small house interpretation",
        "no windows significance",
        "chimney symbolism",
    ]

    print("\n📊 Testing queries:\n")
    for query in test_queries:
        print(f"Query: '{query}'")
        results = query_engine.search(query, top_k=3)

        for i, (chunk, distance, meta) in enumerate(results, 1):
            print(f"  Result {i} (distance: {distance:.4f}):")
            print(f"    {chunk[:200]}...")
            print()

    # Test analysis context retrieval
    print("\n🏠 Testing analysis context retrieval:\n")
    test_analysis = {
        "house_size_category": "small",
        "missing_features": ["door", "chimney"],
        "detected_features": ["house", "roof", "window"],
        "door_present": False,
        "window_count": 1,
        "chimney_present": False,
    }

    context = query_engine.get_context_for_analysis(test_analysis, top_k=5)
    print("Retrieved Context:")
    print(context[:500] + "..." if len(context) > 500 else context)


if __name__ == "__main__":
    main()
