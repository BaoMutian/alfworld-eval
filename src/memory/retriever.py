"""Memory retriever for finding relevant memories based on similarity."""

import logging
from typing import List, Optional

import numpy as np

from .schemas import Memory, RetrievedMemory
from .store import MemoryStore
from .embeddings import EmbeddingModel, cosine_similarity

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """Retriever for finding relevant memories based on query similarity.
    
    Uses embedding-based similarity search to find memories that are
    relevant to the current task.
    """

    def __init__(
        self,
        store: MemoryStore,
        embedding_model: EmbeddingModel,
        top_k: int = 1,
        similarity_threshold: float = 0.5,
    ):
        """Initialize memory retriever.
        
        Args:
            store: Memory store to retrieve from.
            embedding_model: Embedding model for encoding queries.
            top_k: Maximum number of memories to retrieve.
            similarity_threshold: Minimum similarity score to consider.
        """
        self.store = store
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedMemory]:
        """Retrieve relevant memories for a query.
        
        Args:
            query: The query string (typically task goal).
            top_k: Override default top_k.
            similarity_threshold: Override default threshold.
            
        Returns:
            List of RetrievedMemory objects sorted by similarity (highest first).
        """
        if top_k is None:
            top_k = self.top_k
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        # Check if store has memories and embeddings
        if self.store.is_empty():
            logger.debug("Memory store is empty, no memories to retrieve")
            return []

        if not self.store.has_embeddings():
            logger.warning("No embeddings available for retrieval")
            return []

        try:
            # Encode query
            query_embedding = self.embedding_model.encode_single(query)

            # Get corpus embeddings
            memories, corpus_embeddings = self.store.get_memories_and_embeddings()

            if corpus_embeddings is None or len(corpus_embeddings) == 0:
                return []

            # Compute similarities
            similarities = cosine_similarity(query_embedding, corpus_embeddings)

            # Get top-k indices above threshold
            retrieved = []
            sorted_indices = np.argsort(similarities)[::-1]  # Descending order

            for idx in sorted_indices[:top_k]:
                score = float(similarities[idx])
                if score >= similarity_threshold:
                    retrieved.append(RetrievedMemory(
                        memory=memories[idx],
                        similarity=score,
                    ))

            if retrieved:
                logger.debug(
                    f"Retrieved {len(retrieved)} memories for query. "
                    f"Top similarity: {retrieved[0].similarity:.4f}"
                )
            else:
                logger.debug(
                    f"No memories above threshold {similarity_threshold}. "
                    f"Max similarity: {float(max(similarities)):.4f}"
                )

            return retrieved

        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    def retrieve_by_task_type(
        self,
        query: str,
        task_type: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedMemory]:
        """Retrieve memories filtered by task type.
        
        Args:
            query: The query string.
            task_type: Task type to filter by.
            top_k: Override default top_k.
            similarity_threshold: Override default threshold.
            
        Returns:
            List of RetrievedMemory objects filtered by task type.
        """
        # First get all relevant memories
        all_retrieved = self.retrieve(
            query,
            top_k=top_k * 3 if top_k else self.top_k * 3,  # Get more to filter
            similarity_threshold=similarity_threshold,
        )

        # Filter by task type
        filtered = [r for r in all_retrieved if r.memory.task_type == task_type]

        # Limit to top_k
        k = top_k if top_k is not None else self.top_k
        return filtered[:k]

    def retrieve_successful(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedMemory]:
        """Retrieve only successful memories.
        
        Args:
            query: The query string.
            top_k: Override default top_k.
            similarity_threshold: Override default threshold.
            
        Returns:
            List of RetrievedMemory objects that are marked as successful.
        """
        all_retrieved = self.retrieve(
            query,
            top_k=top_k * 3 if top_k else self.top_k * 3,
            similarity_threshold=similarity_threshold,
        )

        filtered = [r for r in all_retrieved if r.is_success]

        k = top_k if top_k is not None else self.top_k
        return filtered[:k]

    def update_config(
        self,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> None:
        """Update retrieval configuration.
        
        Args:
            top_k: New top_k value.
            similarity_threshold: New threshold value.
        """
        if top_k is not None:
            self.top_k = top_k
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold

