"""Embedding model wrapper for memory retrieval."""

import logging
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for sentence-transformer embedding model.
    
    Uses lazy loading to initialize the model only when first needed.
    Supports both CPU and GPU inference.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
    ):
        """Initialize embedding model wrapper.
        
        Args:
            model_name: HuggingFace model name for sentence-transformers.
            device: Device to run the model on ('cpu' or 'cuda').
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension: Optional[int] = None

    @property
    def model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension is None:
            # Trigger model loading to get dimension
            _ = self.model
        return self._dimension

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            # Get embedding dimension from a test encoding
            test_embedding = self._model.encode(["test"], convert_to_numpy=True)
            self._dimension = test_embedding.shape[1]
            logger.info(
                f"Embedding model loaded. Dimension: {self._dimension}, "
                f"Device: {self.device}"
            )
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts to encode.
            normalize: Whether to L2-normalize embeddings (for cosine similarity).
            
        Returns:
            Numpy array of shape (n_texts, dimension) with embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([]).reshape(0, self.dimension)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

        return embeddings

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text into an embedding vector.
        
        Args:
            text: Text to encode.
            normalize: Whether to L2-normalize embedding.
            
        Returns:
            Numpy array of shape (dimension,) with the embedding.
        """
        embedding = self.encode([text], normalize=normalize)
        return embedding[0]

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None


def cosine_similarity(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between query and corpus embeddings.
    
    Args:
        query_embedding: Query embedding of shape (dimension,) or (1, dimension).
        corpus_embeddings: Corpus embeddings of shape (n_corpus, dimension).
        
    Returns:
        Similarity scores of shape (n_corpus,).
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    if corpus_embeddings.size == 0:
        return np.array([])

    # For normalized embeddings, cosine similarity = dot product
    similarities = np.dot(corpus_embeddings, query_embedding.T).flatten()
    return similarities

