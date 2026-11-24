"""Item2vec via positive PMI profiles.

Skip-gram with negative sampling factorizes a PMI matrix (Levy and Goldberg).
Products that show up in the same kinds of baskets end up close together even
when a shopper buys one or the other: chicken thighs near other poultry, feta
near other cheeses. Raw co-occurrence would instead pull in complements
(thighs near rice). PMI down-weights those.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from basket.config import ARTIFACT_DIR


def train_item2vec(baskets: list[list[str]], skus: list[str], dim: int = 32, shift_k: float = 1.0):
    index = {sku: i for i, sku in enumerate(skus)}
    n = len(skus)
    cooc = np.zeros((n, n), dtype=np.float64)
    freq = np.zeros(n, dtype=np.float64)
    used = 0
    for basket in baskets:
        ids = sorted({index[sku] for sku in basket if sku in index})
        if len(ids) < 2:
            continue
        used += 1
        for i in ids:
            freq[i] += 1.0
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                cooc[ids[a], ids[b]] += 1.0
                cooc[ids[b], ids[a]] += 1.0
    pmi = np.zeros((n, n), dtype=np.float64)
    shift = math.log(shift_k)
    for i in range(n):
        if freq[i] <= 0:
            continue
        row = cooc[i]
        for j in range(n):
            if row[j] <= 0 or freq[j] <= 0:
                continue
            value = math.log((row[j] * used) / (freq[i] * freq[j])) - shift
            if value > 0:
                pmi[i, j] = value
    norms = np.linalg.norm(pmi, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    profiles = pmi / norms
    u, singular, _vt = np.linalg.svd(profiles, full_matrices=False)
    width = min(dim, u.shape[1])
    vectors = u[:, :width] * singular[:width]
    for col in range(width):
        pivot = vectors[int(np.argmax(np.abs(vectors[:, col]))), col]
        if pivot < 0:
            vectors[:, col] *= -1
    vnorm = np.linalg.norm(vectors, axis=1, keepdims=True)
    vnorm[vnorm == 0] = 1.0
    vectors = (vectors / vnorm).astype(np.float32)
    return vectors, freq.astype(np.float32), cooc.astype(np.float32), int(used)


class ItemVectors:
    def __init__(self, skus: list[str], vectors: np.ndarray, freq: np.ndarray, cooc: np.ndarray, n_baskets: int):
        self.skus = list(skus)
        self.index = {sku: i for i, sku in enumerate(self.skus)}
        self.vectors = vectors
        self.freq = freq
        self.cooc = cooc
        self.n_baskets = n_baskets

    def cosine(self, left: str, right: str) -> float:
        if left not in self.index or right not in self.index:
            return 0.0
        return float(self.vectors[self.index[left]] @ self.vectors[self.index[right]])

    def nearest(self, sku: str, k: int = 8) -> list[tuple[str, float]]:
        if sku not in self.index:
            return []
        sims = self.vectors @ self.vectors[self.index[sku]]
        sims[self.index[sku]] = -2.0
        order = np.argsort(-sims)[:k]
        return [(self.skus[int(i)], float(sims[int(i)])) for i in order]

    def save(self, folder: Path | None = None) -> None:
        folder = folder or ARTIFACT_DIR
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / "item2vec.npz",
            skus=np.array(self.skus),
            vectors=self.vectors,
            freq=self.freq,
            cooc=self.cooc,
            n_baskets=np.array([self.n_baskets]),
        )

    @classmethod
    def load(cls, folder: Path | None = None) -> "ItemVectors":
        path = (folder or ARTIFACT_DIR) / "item2vec.npz"
        if not path.exists():
            raise FileNotFoundError("Missing artifacts/item2vec.npz. Run: python scripts/train.py")
        data = np.load(path, allow_pickle=False)
        skus = [str(s) for s in data["skus"].tolist()]
        return cls(skus, data["vectors"], data["freq"], data["cooc"], int(data["n_baskets"][0]))
