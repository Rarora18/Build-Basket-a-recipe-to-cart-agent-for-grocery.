"""LightGBM ranker over item2vec, PMI, nutrition, price, and name overlap.

Labels are same-group substitutes. Held-out ingredients never appear as training
anchors, so the reported hit rate is not fit on those queries. The group id
itself is not a feature.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np

from basket.catalog import Catalog
from basket.config import ARTIFACT_DIR
from basket.item2vec import ItemVectors
from basket.models import Product
from basket.textutil import tokenize

FEATURES = [
    "cosine",
    "pmi",
    "same_aisle",
    "protein_diff",
    "kcal_diff",
    "log_price_ratio",
    "log_pop",
    "jaccard",
    "cand_protein",
    "deal",
]


def held_out_ingredient(ingredient_id: str) -> bool:
    digest = hashlib.md5(ingredient_id.encode()).hexdigest()
    return int(digest[:2], 16) % 5 == 0


def _pmi(vectors: ItemVectors, left: str, right: str) -> float:
    if left not in vectors.index or right not in vectors.index:
        return 0.0
    i, j = vectors.index[left], vectors.index[right]
    count = float(vectors.cooc[i, j])
    fi, fj = float(vectors.freq[i]), float(vectors.freq[j])
    if count <= 0 or fi <= 0 or fj <= 0 or vectors.n_baskets <= 0:
        return 0.0
    return max(0.0, math.log((count * vectors.n_baskets) / (fi * fj)))


def _jaccard(left: str, right: str) -> float:
    a, b = set(tokenize(left)), set(tokenize(right))
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_features(anchor: Product, cand: Product, vectors: ItemVectors) -> np.ndarray:
    price_ratio = (cand.unit_cents + 1) / (anchor.unit_cents + 1)
    pop = float(vectors.freq[vectors.index[cand.sku]]) if cand.sku in vectors.index else 0.0
    return np.array([
        vectors.cosine(anchor.sku, cand.sku),
        _pmi(vectors, anchor.sku, cand.sku),
        1.0 if anchor.aisle == cand.aisle else 0.0,
        abs(anchor.protein_100 - cand.protein_100),
        abs(anchor.kcal_100 - cand.kcal_100),
        math.log(price_ratio),
        math.log1p(pop),
        _jaccard(anchor.name, cand.name),
        cand.protein_100,
        cand.deal_pct,
    ], dtype=np.float32)


def train_ranker(catalog: Catalog, vectors: ItemVectors, seed: int = 7):
    rng = np.random.default_rng(seed)
    products = catalog.products
    by_group: dict[str, list[Product]] = {}
    for product in products:
        by_group.setdefault(product.group, []).append(product)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for anchor in products:
        if held_out_ingredient(anchor.ingredient_id):
            continue
        positives = [p for p in by_group[anchor.group] if p.sku != anchor.sku]
        if len(positives) > 6:
            pick = rng.choice(len(positives), size=6, replace=False)
            positives = [positives[int(i)] for i in pick]
        for cand in positives:
            rows.append(pair_features(anchor, cand, vectors))
            labels.append(1)
        negatives = [p for p in products if p.group != anchor.group]
        if not negatives:
            continue
        take = min(4, len(negatives))
        for idx in rng.choice(len(negatives), size=take, replace=False):
            rows.append(pair_features(anchor, negatives[int(idx)], vectors))
            labels.append(0)
    x = np.vstack(rows)
    y = np.array(labels, dtype=np.float32)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "learning_rate": 0.08,
        "num_leaves": 15,
        "min_data_in_leaf": 8,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "scale_pos_weight": neg / pos,
    }
    model = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=80)
    return model


class Ranker:
    def __init__(self, model: lgb.Booster, catalog: Catalog, vectors: ItemVectors):
        self.model = model
        self.catalog = catalog
        self.vectors = vectors

    def score(self, anchor: Product, cand: Product) -> float:
        features = pair_features(anchor, cand, self.vectors).reshape(1, -1)
        return float(self.model.predict(features)[0])

    def rank_pool(self, anchor: Product, pool: list[Product]) -> list[Product]:
        ordered = sorted(pool, key=lambda cand: (-self.score(anchor, cand), cand.sku))
        return ordered

    def save(self, folder: Path | None = None) -> None:
        folder = folder or ARTIFACT_DIR
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(folder / "ranker.txt"))

    @classmethod
    def load(cls, catalog: Catalog, vectors: ItemVectors, folder: Path | None = None) -> "Ranker":
        path = (folder or ARTIFACT_DIR) / "ranker.txt"
        if not path.exists():
            raise FileNotFoundError("Missing artifacts/ranker.txt. Run: python scripts/train.py")
        return cls(lgb.Booster(model_file=str(path)), catalog, vectors)
