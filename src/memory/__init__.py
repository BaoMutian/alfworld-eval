"""ReasoningBank memory module for self-evolving agent learning.

This module provides:
- Memory storage and management (JSONL + embeddings)
- Embedding-based memory retrieval
- LLM-based memory extraction from trajectories

Usage:
    from src.memory import (
        MemoryStore,
        MemoryRetriever, 
        MemoryExtractor,
        EmbeddingModel,
        Memory,
        MemoryEntry,
        RetrievedMemory,
    )
"""

from .schemas import Memory, MemoryEntry, RetrievedMemory
from .embeddings import EmbeddingModel, cosine_similarity
from .store import MemoryStore
from .retriever import MemoryRetriever
from .extractor import MemoryExtractor

__all__ = [
    # Data structures
    "Memory",
    "MemoryEntry", 
    "RetrievedMemory",
    # Core components
    "EmbeddingModel",
    "MemoryStore",
    "MemoryRetriever",
    "MemoryExtractor",
    # Utilities
    "cosine_similarity",
]

