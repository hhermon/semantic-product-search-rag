"""Nyckelordsbaserad sökning (TF-IDF) som baseline för jämförelse."""

import math
import re
from collections import Counter
from dataclasses import dataclass

from src.data.product_loader import Product


@dataclass
class KeywordResult:
    """Ett sökresultat från nyckelordssökning."""
    product: Product
    score: float
    rank: int


class KeywordSearcher:
    """TF-IDF-baserad nyckelordssökning.

    Implementerar klassisk informationssökning (avsnitt 2.2 i rapporten)
    som baslinjemodell att jämföra mot semantisk retrieval.
    """

    def __init__(self):
        self._products: list[Product] = []
        self._index: dict[str, dict[int, float]] = {}  # term → {doc_idx: tf-idf}
        self._doc_texts: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        """Enkel tokenisering: lowercase + dela på icke-alfanumeriska tecken."""
        return re.findall(r"[a-zåäöA-ZÅÄÖ0-9]+", text.lower())

    def index_products(self, products: list[Product]) -> None:
        """Bygger TF-IDF-index över produkterna."""
        self._products = products
        self._doc_texts = [self._tokenize(p.to_search_text()) for p in products]
        n_docs = len(products)

        # Räkna dokumentfrekvens per term
        df: dict[str, int] = {}
        for tokens in self._doc_texts:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        # Bygg inverterat index med TF-IDF-vikter
        self._index = {}
        for doc_idx, tokens in enumerate(self._doc_texts):
            tf = Counter(tokens)
            doc_len = len(tokens)
            for term, count in tf.items():
                tf_score = count / doc_len
                idf_score = math.log((n_docs + 1) / (df[term] + 1)) + 1
                tfidf = tf_score * idf_score
                if term not in self._index:
                    self._index[term] = {}
                self._index[term][doc_idx] = tfidf

    def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: str | None = None,
        max_price: float | None = None,
    ) -> list[KeywordResult]:
        """Söker med TF-IDF-likhet mot frågan."""
        query_terms = self._tokenize(query)
        scores: dict[int, float] = {}

        for term in query_terms:
            if term in self._index:
                for doc_idx, tfidf in self._index[term].items():
                    scores[doc_idx] = scores.get(doc_idx, 0) + tfidf

        # Filtrera och sortera
        results: list[KeywordResult] = []
        rank = 1
        for doc_idx, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            product = self._products[doc_idx]
            if category_filter and product.category.lower() != category_filter.lower():
                continue
            if max_price is not None and product.price > max_price:
                continue
            results.append(KeywordResult(product=product, score=score, rank=rank))
            rank += 1
            if len(results) >= top_k:
                break

        return results

    @classmethod
    def from_products(cls, products: list[Product]) -> "KeywordSearcher":
        searcher = cls()
        searcher.index_products(products)
        return searcher
