"""Memory store for persisting and managing memories."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .schemas import Memory, MemoryEntry
from .embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class MemoryStore:
    """Persistent storage for memories with embedding support.
    
    Stores memories in JSONL format and embeddings in numpy format.
    Supports incremental addition of new memories.
    """

    def __init__(
        self,
        memory_dir: str,
        task_name: str,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        """Initialize memory store.
        
        Args:
            memory_dir: Directory to store memory files.
            task_name: Name of the task (used in file naming).
            embedding_model: Embedding model for encoding queries.
        """
        self.memory_dir = Path(memory_dir)
        self.task_name = task_name
        self.embedding_model = embedding_model

        # File paths
        self.memories_path = self.memory_dir / f"{task_name}_memories.jsonl"
        self.embeddings_path = self.memory_dir / f"{task_name}_embeddings.npy"

        # In-memory cache
        self._memories: List[Memory] = []
        self._embeddings: Optional[np.ndarray] = None
        self._memory_id_to_idx: dict = {}

        # Ensure directory exists
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Load existing memories
        self._load()

    def _load(self) -> None:
        """Load memories and embeddings from disk."""
        self._memories = []
        self._memory_id_to_idx = {}

        # Load memories from JSONL
        if self.memories_path.exists():
            try:
                with open(self.memories_path, "r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            memory = Memory.from_dict(data)
                            self._memories.append(memory)
                            self._memory_id_to_idx[memory.memory_id] = idx
                logger.info(f"Loaded {len(self._memories)} memories from {self.memories_path}")
            except Exception as e:
                logger.error(f"Failed to load memories: {e}")
                self._memories = []
                self._memory_id_to_idx = {}

        # Load embeddings from numpy file
        if self.embeddings_path.exists() and self._memories:
            try:
                self._embeddings = np.load(self.embeddings_path)
                if len(self._embeddings) != len(self._memories):
                    logger.warning(
                        f"Embedding count ({len(self._embeddings)}) doesn't match "
                        f"memory count ({len(self._memories)}). Re-encoding..."
                    )
                    self._recompute_embeddings()
                else:
                    logger.info(f"Loaded embeddings from {self.embeddings_path}")
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")
                self._embeddings = None
        elif self._memories:
            # Memories exist but no embeddings - compute them
            self._recompute_embeddings()

    def _recompute_embeddings(self) -> None:
        """Recompute embeddings for all memories."""
        if not self.embedding_model or not self._memories:
            self._embeddings = None
            return

        try:
            queries = [m.query for m in self._memories]
            self._embeddings = self.embedding_model.encode(queries)
            self._save_embeddings()
            logger.info(f"Recomputed embeddings for {len(self._memories)} memories")
        except Exception as e:
            logger.error(f"Failed to recompute embeddings: {e}")
            self._embeddings = None

    def _save_embeddings(self) -> None:
        """Save embeddings to disk."""
        if self._embeddings is not None:
            try:
                np.save(self.embeddings_path, self._embeddings)
            except Exception as e:
                logger.error(f"Failed to save embeddings: {e}")

    def _append_memory_to_file(self, memory: Memory) -> None:
        """Append a single memory to the JSONL file."""
        try:
            with open(self.memories_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(memory.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to append memory to file: {e}")
            raise

    def add(self, memory: Memory) -> bool:
        """Add a new memory to the store.
        
        Args:
            memory: Memory to add.
            
        Returns:
            True if successfully added, False otherwise.
        """
        if memory.memory_id in self._memory_id_to_idx:
            logger.warning(f"Memory {memory.memory_id} already exists, skipping")
            return False

        try:
            # Add to in-memory cache
            idx = len(self._memories)
            self._memories.append(memory)
            self._memory_id_to_idx[memory.memory_id] = idx

            # Append to file
            self._append_memory_to_file(memory)

            # Update embeddings
            if self.embedding_model:
                new_embedding = self.embedding_model.encode_single(memory.query)
                if self._embeddings is None:
                    self._embeddings = new_embedding.reshape(1, -1)
                else:
                    self._embeddings = np.vstack([self._embeddings, new_embedding])
                self._save_embeddings()

            logger.debug(f"Added memory {memory.memory_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            # Rollback in-memory changes
            if memory.memory_id in self._memory_id_to_idx:
                del self._memory_id_to_idx[memory.memory_id]
                self._memories.pop()
            return False

    def get(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve.
            
        Returns:
            Memory if found, None otherwise.
        """
        idx = self._memory_id_to_idx.get(memory_id)
        if idx is not None:
            return self._memories[idx]
        return None

    def get_all(self) -> List[Memory]:
        """Get all memories.
        
        Returns:
            List of all memories.
        """
        return list(self._memories)

    def get_embeddings(self) -> Optional[np.ndarray]:
        """Get all embeddings.
        
        Returns:
            Numpy array of embeddings or None if not available.
        """
        return self._embeddings

    def get_memories_and_embeddings(self) -> Tuple[List[Memory], Optional[np.ndarray]]:
        """Get all memories and their embeddings.
        
        Returns:
            Tuple of (memories list, embeddings array).
        """
        return self._memories, self._embeddings

    def size(self) -> int:
        """Get the number of memories in the store.
        
        Returns:
            Number of memories.
        """
        return len(self._memories)

    def is_empty(self) -> bool:
        """Check if the store is empty.
        
        Returns:
            True if empty, False otherwise.
        """
        return len(self._memories) == 0

    def has_embeddings(self) -> bool:
        """Check if embeddings are available.
        
        Returns:
            True if embeddings are available, False otherwise.
        """
        return self._embeddings is not None and len(self._embeddings) > 0

    def clear(self) -> None:
        """Clear all memories (both in-memory and on disk)."""
        self._memories = []
        self._embeddings = None
        self._memory_id_to_idx = {}

        # Remove files
        if self.memories_path.exists():
            self.memories_path.unlink()
        if self.embeddings_path.exists():
            self.embeddings_path.unlink()

        logger.info("Cleared all memories")

    def get_stats(self) -> dict:
        """Get statistics about the memory store.
        
        Returns:
            Dictionary with store statistics.
        """
        success_count = sum(1 for m in self._memories if m.is_success)
        failure_count = len(self._memories) - success_count

        task_types = {}
        for m in self._memories:
            task_types[m.task_type] = task_types.get(m.task_type, 0) + 1

        return {
            "total_memories": len(self._memories),
            "success_memories": success_count,
            "failure_memories": failure_count,
            "has_embeddings": self.has_embeddings(),
            "embedding_dimension": self._embeddings.shape[1] if self._embeddings is not None else None,
            "task_types": task_types,
            "memories_path": str(self.memories_path),
            "embeddings_path": str(self.embeddings_path),
        }

