"""Hybrid retrieval: Okapi BM25 fused with hashed embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from basket.config import LEX_WEIGHT, SEM_WEIGHT
from basket.models import Goal, Product, allowed
from basket.textutil import BM25, embed


@dataclass(frozen=True)
class Hit:
    product: Product
    score: float
    lexical: float
    semantic: float


def fuse(lexical: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    """Blend channels, trusting BM25 more when one product takes most of the lexical mass.

    A query with no token overlap falls through to embeddings. A query whose
    lexical score is spread across many products (a shared word like "chicken")
    leans on the embedding channel, which is where synonyms and typos live.
    """
    lex = np.asarray(lexical, dtype=np.float64)
    sem = np.asarray(semantic, dtype=np.float64)
    if lex.size == 0 or float(lex.max()) <= 0:
        return sem
    lex_n = lex / lex.max()
    spread = float(sem.max() - sem.min())
    sem_n = (sem - sem.min()) / spread if spread > 1e-9 else np.zeros_like(sem)
    share = float(lex.max() / lex.sum()) if float(lex.sum()) > 0 else 0.0
    weight = min(0.85, LEX_WEIGHT + SEM_WEIGHT * share)
    return (1 - weight) * sem_n + weight * lex_n


class Retriever:
    def __init__(self, products: list[Product]):
        self.products = products
        self.bm25 = BM25([p.search_text for p in products])
        self.vectors = np.stack([embed(p.search_text) for p in products]) if products else np.zeros((0, 1))

    def search(self, query: str, k: int = 8, goal: Goal | None = None, in_stock: bool | None = None) -> list[Hit]:
        if not self.products or not query.strip():
            return []
        lexical = self.bm25.scores(query)
        semantic = self.vectors @ embed(query)
        keep = []
        for i, product in enumerate(self.products):
            if goal is not None and not allowed(product, goal):
                continue
            if in_stock is True and product.stock <= 0:
                continue
            if in_stock is False and product.stock > 0:
                continue
            keep.append(i)
        if not keep:
            return []
        idx = np.array(keep)
        fused = fuse(lexical[idx], semantic[idx])
        order = sorted(range(len(idx)), key=lambda n: (-fused[n], self.products[idx[n]].sku))
        hits = []
        for n in order[:k]:
            product = self.products[idx[n]]
            hits.append(Hit(
                product=product,
                score=float(fused[n]),
                lexical=float(lexical[idx[n]]),
                semantic=float(semantic[idx[n]]),
            ))
        return hits
