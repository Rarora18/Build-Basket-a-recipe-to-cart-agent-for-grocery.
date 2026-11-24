"""Offline evals for matching, substitutions, constraint holds, and latency."""

from __future__ import annotations

import time

import numpy as np

from basket.catalog import Catalog, load_labels, load_orders
from basket.config import FOCUS_GROUPS
from basket.item2vec import ItemVectors
from basket.models import Goal
from basket.ranker import Ranker, held_out_ingredient
from basket.retrieve import Retriever, fuse
from basket.textutil import embed

_OPEN = Goal(
    raw="",
    meals_requested=1,
    servings=2,
    budget=None,
    high_protein=False,
    exclude_pork=False,
    exclude_beef=False,
    vegetarian=False,
    vegan=False,
    gluten_free=False,
    dairy_free=False,
    nut_free=False,
    shellfish_free=False,
    soy_free=False,
)


def _summarize(hits: list[int], precision: list[float]) -> dict:
    arr_h = np.array(hits, dtype=np.float64)
    arr_p = np.array(precision, dtype=np.float64)
    return {
        "hit_at_5": round(float(arr_h.mean()), 4) if len(arr_h) else 0.0,
        "precision_at_5": round(float(arr_p.mean()), 4) if len(arr_p) else 0.0,
        "n": int(len(arr_h)),
    }


def matching_report(catalog: Catalog, retriever: Retriever, labels: list[dict] | None = None) -> dict:
    labels = labels if labels is not None else load_labels()
    channels = {"bm25": [], "embed": [], "hybrid": []}
    precision = {name: [] for name in channels}
    by_kind: dict[str, dict[str, list]] = {}
    products = retriever.products
    for label in labels:
        relevant = set(label["relevant_skus"])
        query = label["query"]
        lexical = retriever.bm25.scores(query)
        semantic = retriever.vectors @ embed(query)
        hybrid = fuse(lexical, semantic)
        ranked = {
            "bm25": np.argsort(-lexical),
            "embed": np.argsort(-semantic),
            "hybrid": np.argsort(-hybrid),
        }
        kind = label.get("kind", "all")
        by_kind.setdefault(kind, {name: [] for name in channels})
        for name, order in ranked.items():
            top = [products[int(i)].sku for i in order[:5]]
            hit = int(any(sku in relevant for sku in top))
            prec = len([sku for sku in top if sku in relevant]) / 5
            channels[name].append(hit)
            precision[name].append(prec)
            by_kind[kind][name].append(prec)
    report = {
        name: {**_summarize(channels[name], precision[name])}
        for name in channels
    }
    report["by_kind"] = {
        kind: {name: round(float(np.mean(values)), 4) for name, values in rows.items()}
        for kind, rows in by_kind.items()
    }
    report["n"] = len(labels)
    return report


def _hit_at(ranked_skus: list[str], gold: set[str], k: int) -> int:
    return int(any(sku in gold for sku in ranked_skus[:k]))


def substitution_report(catalog: Catalog, vectors: ItemVectors, ranker: Ranker) -> dict:
    """Held-out baskets. Gold is another SKU in the same substitute group."""
    test_orders = [order for order in load_orders() if order["split"] == "test"]
    aisle_of: dict[str, list] = {}
    for product in catalog.products:
        aisle_of.setdefault(product.aisle, []).append(product)
    counts = {"n": 0, "pop@1": 0, "pop@5": 0, "item2vec@1": 0, "item2vec@5": 0, "ranker@1": 0, "ranker@5": 0}
    held = {"n": 0, "pop@1": 0, "pop@5": 0, "item2vec@1": 0, "item2vec@5": 0, "ranker@1": 0, "ranker@5": 0}
    for order in test_orders:
        seen = set()
        for sku in order["skus"]:
            if sku in seen or sku not in catalog.by_sku:
                continue
            seen.add(sku)
            anchor = catalog.by_sku[sku]
            if anchor.group not in FOCUS_GROUPS:
                continue
            pool = [product for product in aisle_of[anchor.aisle] if product.sku != anchor.sku]
            gold = {product.sku for product in pool if product.group == anchor.group}
            if not gold or not pool:
                continue
            popularity = sorted(pool, key=lambda product: (-float(vectors.freq[vectors.index[product.sku]]) if product.sku in vectors.index else 0.0, product.sku))
            embedded = sorted(pool, key=lambda product: (-vectors.cosine(anchor.sku, product.sku), product.sku))
            learned = ranker.rank_pool(anchor, pool)
            buckets = [counts]
            if held_out_ingredient(anchor.ingredient_id):
                buckets.append(held)
            pop_skus = [p.sku for p in popularity]
            vec_skus = [p.sku for p in embedded]
            rank_skus = [p.sku for p in learned]
            for bucket in buckets:
                bucket["n"] += 1
                bucket["pop@1"] += _hit_at(pop_skus, gold, 1)
                bucket["pop@5"] += _hit_at(pop_skus, gold, 5)
                bucket["item2vec@1"] += _hit_at(vec_skus, gold, 1)
                bucket["item2vec@5"] += _hit_at(vec_skus, gold, 5)
                bucket["ranker@1"] += _hit_at(rank_skus, gold, 1)
                bucket["ranker@5"] += _hit_at(rank_skus, gold, 5)
    # Headline numbers are held-out ingredients the ranker never trained on as anchors.
    counts = held if held["n"] >= 50 else counts
    n = max(counts["n"], 1)
    def rate(key: str) -> float:
        return round(counts[key] / n, 4)
    pop1, model1 = rate("pop@1"), rate("ranker@1")
    lift = round((model1 - pop1) / pop1 * 100, 1) if pop1 else None
    return {
        "n": counts["n"],
        "popularity_hit_at_1": pop1,
        "popularity_hit_at_5": rate("pop@5"),
        "item2vec_hit_at_1": rate("item2vec@1"),
        "item2vec_hit_at_5": rate("item2vec@5"),
        "ranker_hit_at_1": model1,
        "ranker_hit_at_5": rate("ranker@5"),
        "ranker_lift_vs_popularity_pct": lift,
        "held_out_ingredient_anchors_only": held["n"] >= 50,
        "note": "Candidates are other SKUs in the same aisle. A hit is a SKU from the same substitute group.",
    }


def constraint_report(build, queries: list[str] | None = None) -> dict:
    queries = queries or [
        "high-protein dinners for 2 this week, under $60, no pork.",
        "vegetarian dinners for 2, 4 meals, under $45",
        "vegan dinners for 2 this week under $50",
        "gluten-free high-protein dinners for 2, 4 meals, under $55, no pork",
        "dinners for 4 this week under $100, no beef, no pork",
        "high protein meal for 2 under $25 no pork",
        "nut-free dairy-free dinners for 2, 3 meals, under $40",
        "shellfish-free high-protein dinners for 2 this week under $70",
        "soy-free dinners for 2, 4 meals, under $55, no pork",
    ]
    violations = {"budget": 0, "diet": 0}
    rows = []
    for query in queries:
        cart = build(query)
        budget_ok = bool(cart["within_budget"])
        diet_ok = bool(cart["diet_ok"])
        if not budget_ok:
            violations["budget"] += 1
        if not diet_ok:
            violations["diet"] += 1
        rows.append({
            "query": query,
            "total": cart["total"],
            "budget": cart["budget"],
            "kept": sum(1 for meal in cart["meals"] if meal["kept"]),
            "within_budget": budget_ok,
            "diet_ok": diet_ok,
            "substitutions": sum(1 for line in cart["lines"] if line["substitute"]),
        })
    n = len(queries)
    broken = sum(1 for row in rows if not row["within_budget"] or not row["diet_ok"])
    return {
        "n": n,
        "budget_violations": violations["budget"],
        "diet_violations": violations["diet"],
        "carts_breaking_a_constraint_pct": round(100 * broken / n, 2) if n else 0,
        "rows": rows,
    }


def latency_report(build, query: str = "high-protein dinners for 2 this week, under $60, no pork.", repeats: int = 15) -> dict:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        cart = build(query)
        samples.append((time.perf_counter() - started) * 1000)
        planner = cart["planner"]
    arr = np.array(samples)
    return {
        "n": repeats,
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "planner": planner,
        "estimated_cost_usd": 0.0 if planner == "heuristic" else None,
    }


def held_out_count(catalog: Catalog) -> int:
    return sum(1 for ingredient_id in catalog.ingredients if held_out_ingredient(ingredient_id))
