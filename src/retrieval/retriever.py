"""Semantisk retrieval med numpy-baserad cosinuslikhet."""

import os
from dataclasses import dataclass

import numpy as np

from src.data.product_loader import Product, ProductLoader
from src.retrieval.embeddings import EmbeddingModel


@dataclass
class SearchResult:
    """Ett sökresultat med produkt och relevansscore."""
    product: Product
    score: float
    rank: int


class ProductRetriever:
    """Bygger och frågar ett semantiskt produktindex med numpy.

    Flöde:
    1. Produkter laddas och konverteras till sökbar text.
    2. Embeddings beräknas med SentenceTransformer.
    3. Vektorer lagras i minnet som numpy-matris.
    4. Vid sökning kodas frågan och cosinuslikhet beräknas mot alla produkter.
    """

    def __init__(self, embedding_model: EmbeddingModel | None = None):
        self.embedding_model = embedding_model or EmbeddingModel()
        self._product_map: dict[str, Product] = {}
        self._product_ids: list[str] = []
        self._embeddings: np.ndarray | None = None

    def index_products(self, products: list[Product]) -> None:
        """Indexerar produkter och beräknar embeddings."""
        texts = [p.to_search_text() for p in products]
        self._embeddings = self.embedding_model.encode(texts)
        self._product_ids = [p.id for p in products]
        for p in products:
            self._product_map[p.id] = p

    def _ensure_product_map(self, products: list[Product]) -> None:
        for p in products:
            self._product_map[p.id] = p

    def search(
        self,
        query: str,
        top_k: int | None = None,
        category_filter: str | None = None,
        max_price: float | None = None,
    ) -> list[SearchResult]:
        """Semantisk sökning via cosinuslikhet.

        Args:
            query: Användarens fritextfråga.
            top_k: Antal resultat att returnera.
            category_filter: Begränsa till en produktkategori.
            max_price: Filtrera bort produkter dyrare än detta värde.
        """
        if self._embeddings is None or len(self._product_ids) == 0:
            return []

        top_k = top_k or int(os.getenv("TOP_K", "5"))
        query_vec = self.embedding_model.encode_single(query)

        # Beräkna cosinuslikhet mot alla produkter
        norms = np.linalg.norm(self._embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        scores = self._embeddings @ query_vec / (norms * query_norm + 1e-10)

        # Sortera efter score (högst först)
        ranked_indices = np.argsort(scores)[::-1]

        search_results: list[SearchResult] = []
        rank = 1
        for idx in ranked_indices:
            pid = self._product_ids[idx]
            product = self._product_map.get(pid)
            if product is None:
                continue
            if category_filter and product.category.lower() != category_filter.lower():
                continue
            if max_price is not None and product.price > max_price:
                continue
            search_results.append(SearchResult(product=product, score=float(scores[idx]), rank=rank))
            rank += 1
            if len(search_results) >= top_k:
                break

        return search_results

    @classmethod
    def from_data_file(cls, **kwargs) -> "ProductRetriever":
        """Skapar och indexerar en retriever från DummyJSON API."""
        retriever = cls()
        products = ProductLoader().load()
        retriever.index_products(products)
        return retriever
