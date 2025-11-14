"""
RAG Indexer for HTP Analysis
Indexes PDF documents using Gemini Embeddings and FAISS for retrieval.
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader


class RAGIndexer:
    """Index documents using Gemini Embeddings and FAISS."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-embedding-001",
        embedding_dim: int = 768,
    ):
        """
        Initialize RAG indexer.

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model_name: Gemini embedding model name
            embedding_dim: Dimension of embeddings (768, 1536, or 3072)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY must be set in environment or passed as parameter"
            )

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.embedding_dim = embedding_dim

        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[str] = []
        self.metadata: List[dict] = []
        self.term_to_page: dict = {}  # Maps terms to page numbers

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text content
        """
        reader = PdfReader(pdf_path)
        text = ""
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text += f"\n[Page {page_num}]\n{page_text}"

        return text

    def load_term_mapping(self, mapping_path: str = None) -> None:
        """
        Load term-to-page mapping from JSON file.

        Args:
            mapping_path: Path to mapping.json file. If None, searches in parent directory.
        """
        if mapping_path is None:
            # Try to find mapping.json in the parent directory
            mapping_path = Path(__file__).parent.parent / "mapping.json"
        else:
            mapping_path = Path(mapping_path)

        if not mapping_path.exists():
            print(f"⚠️  Mapping file not found at {mapping_path}")
            return

        try:
            with open(mapping_path, 'r') as f:
                mapping_data = json.load(f)

            # Extract all terms and their page numbers
            for chapter in mapping_data:
                if "sections" in chapter:
                    for section_name, entries in chapter["sections"].items():
                        for entry in entries:
                            if "term" in entry and "page" in entry:
                                term = entry["term"].upper()
                                page = entry["page"]
                                # Store mapping (term might appear in multiple sections)
                                if term not in self.term_to_page:
                                    self.term_to_page[term] = []
                                self.term_to_page[term].append(page)
                elif "entries" in chapter:
                    for entry in chapter["entries"]:
                        if "term" in entry and "page" in entry:
                            term = entry["term"].upper()
                            page = entry["page"]
                            if term not in self.term_to_page:
                                self.term_to_page[term] = []
                            self.term_to_page[term].append(page)

            print(f"✅ Loaded {len(self.term_to_page)} terms from mapping file")
        except Exception as e:
            print(f"❌ Error loading mapping file: {e}")

    def _extract_page_references(self, chunk_text: str) -> List[str]:
        """
        Extract page numbers from chunk text by looking for [Page X] markers.

        Args:
            chunk_text: Text chunk to analyze

        Returns:
            List of unique page numbers found in the chunk
        """
        import re
        pages = set()
        
        # Look for [Page N] markers in the text
        page_markers = re.findall(r'\[Page (\d+)\]', chunk_text)
        pages.update(page_markers)
        
        # Also try term-based extraction as fallback if term_to_page is loaded
        if self.term_to_page:
            text_upper = chunk_text.upper()
            for term, page_list in self.term_to_page.items():
                if term in text_upper:
                    pages.update(page_list)

        return sorted(list(pages), key=lambda x: int(x) if x.isdigit() else 0)

    def chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> List[Tuple[str, dict]]:
        """
        Split text into overlapping chunks for embedding.

        Args:
            text: Input text to chunk
            chunk_size: Target size of each chunk (in characters)
            overlap: Overlap between consecutive chunks

        Returns:
            List of (chunk_text, metadata) tuples
        """
        # Split by sentences to maintain context
        sentences = text.replace("\n", " ").split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # If adding this sentence exceeds chunk size, save current chunk
            if current_size + sentence_size > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                page_refs = self._extract_page_references(chunk_text)
                chunks.append(
                    (
                        chunk_text,
                        {
                            "chunk_id": chunk_id,
                            "size": len(chunk_text),
                            "sentence_count": len(current_chunk),
                            "page_numbers": page_refs,
                        },
                    )
                )
                chunk_id += 1

                # Keep last few sentences for overlap
                overlap_sentences = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break

                current_chunk = overlap_sentences
                current_size = overlap_size

            current_chunk.append(sentence)
            current_size += sentence_size

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            page_refs = self._extract_page_references(chunk_text)
            chunks.append(
                (
                    chunk_text,
                    {
                        "chunk_id": chunk_id,
                        "size": len(chunk_text),
                        "sentence_count": len(current_chunk),
                        "page_numbers": page_refs,
                    },
                )
            )

        return chunks

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts using Gemini.

        Args:
            texts: List of text strings to embed

        Returns:
            Numpy array of embeddings (shape: [num_texts, embedding_dim])
        """
        embeddings = []

        # Batch embed texts (Gemini supports batching)
        # Process in batches to avoid API limits
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            result = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT", output_dimensionality=self.embedding_dim
                ),
            )

            # Extract embedding values and normalize for non-3072 dimensions
            for embedding_obj in result.embeddings:
                embedding_values = np.array(embedding_obj.values, dtype=np.float32)

                # Normalize embeddings for dimensions other than 3072
                if self.embedding_dim != 3072:
                    embedding_values = embedding_values / np.linalg.norm(
                        embedding_values
                    )

                embeddings.append(embedding_values)

        return np.array(embeddings, dtype=np.float32)

    def build_index(self, pdf_path: str, output_dir: str = "RAG") -> Tuple[str, str]:
        """
        Build FAISS index from PDF document.

        Args:
            pdf_path: Path to PDF file to index
            output_dir: Directory to save index and metadata

        Returns:
            Tuple of (index_path, metadata_path)
        """
        # Load term mapping first
        self.load_term_mapping()

        print(f"📄 Extracting text from PDF: {pdf_path}")
        text = self.extract_text_from_pdf(pdf_path)

        print(f"📝 Chunking text...")
        chunk_data = self.chunk_text(text)
        self.chunks = [chunk for chunk, _ in chunk_data]
        self.metadata = [meta for _, meta in chunk_data]

        print(f"✂️  Created {len(self.chunks)} text chunks")

        print(f"🔮 Generating embeddings using {self.model_name}...")
        embeddings = self.embed_texts(self.chunks)

        print(f"📊 Building FAISS index...")
        # Create FAISS index (L2 distance for normalized embeddings)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings)

        # Save index and metadata
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        index_path = output_path / "faiss_index.bin"
        metadata_path = output_path / "index_metadata.pkl"

        print(f"💾 Saving index to {index_path}")
        faiss.write_index(self.index, str(index_path))

        print(f"💾 Saving metadata to {metadata_path}")
        with open(metadata_path, "wb") as f:
            pickle.dump({
                "chunks": self.chunks, 
                "metadata": self.metadata,
                "term_to_page": self.term_to_page  # Save the term mapping
            }, f)

        print(f"✅ Indexing complete! Indexed {len(self.chunks)} chunks with {len(self.term_to_page)} terms mapped")

        return str(index_path), str(metadata_path)

    def load_index(self, index_path: str, metadata_path: str):
        """
        Load existing FAISS index and metadata.

        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata pickle file
        """
        print(f"📂 Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)

        print(f"📂 Loading metadata from {metadata_path}")
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]
            # Load term_to_page mapping if available (for backward compatibility)
            self.term_to_page = data.get("term_to_page", {})

        # If term_to_page wasn't in the saved data, try to load it now
        if not self.term_to_page:
            print("⚠️  term_to_page mapping not found in metadata, attempting to load from mapping.json...")
            self.load_term_mapping()
            # Re-extract page references for all metadata entries
            if self.term_to_page and self.chunks:
                print("🔄 Re-extracting page references with loaded mapping...")
                for i, chunk in enumerate(self.chunks):
                    page_refs = self._extract_page_references(chunk)
                    if page_refs:
                        self.metadata[i]["page_numbers"] = page_refs

        print(f"✅ Loaded index with {len(self.chunks)} chunks and {len(self.term_to_page)} mapped terms")


def main():
    """Build index from HTP Catalogue PDF."""
    # Find PDF in RAG folder
    rag_dir = Path(__file__).parent.parent / "RAG"
    pdf_files = list(rag_dir.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found in RAG folder")
        return

    pdf_path = pdf_files[0]
    print(f"🎯 Found PDF: {pdf_path.name}")

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("Please set it using: export GEMINI_API_KEY='your-api-key'")
        return

    # Initialize indexer and build index
    indexer = RAGIndexer(api_key=api_key, embedding_dim=768)
    index_path, metadata_path = indexer.build_index(str(pdf_path), output_dir=str(rag_dir))

    print(f"\n🎉 Index saved to: {rag_dir}")
    print(f"   - Index: {Path(index_path).name}")
    print(f"   - Metadata: {Path(metadata_path).name}")


if __name__ == "__main__":
    main()
