"""Skapar semantiska embeddings med Sentence-BERT."""

import os
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper kring SentenceTransformer för semantiska vektorrepresentationer.

    Använder paraphrase-multilingual-MiniLM-L12-v2 som default eftersom
    den stödjer svenska och är relativt snabb att köra lokalt.
    """

    def __init__(self, model_name: str | None = None):
        model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        """Kodar en lista texter till embedding-vektorer."""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def encode_single(self, text: str) -> np.ndarray:
        """Kodar en enskild text till en embedding-vektor."""
        return self.encode([text])[0]

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Beräknar cosinuslikhet mellan två vektorer."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
