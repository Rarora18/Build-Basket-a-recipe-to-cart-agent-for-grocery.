#!/usr/bin/env python3
"""Train item2vec and the substitution ranker on the seed baskets."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basket.catalog import load_catalog, load_orders
from basket.item2vec import ItemVectors, train_item2vec
from basket.ranker import held_out_ingredient, train_ranker, Ranker


def main() -> None:
    catalog = load_catalog()
    orders = [order for order in load_orders() if order["split"] == "train"]
    baskets = [order["skus"] for order in orders]
    skus = [product.sku for product in catalog.products]
    vectors, freq, cooc, used = train_item2vec(baskets, skus)
    store = ItemVectors(skus, vectors, freq, cooc, used)
    store.save()
    model = train_ranker(catalog, store)
    Ranker(model, catalog, store).save()
    print(f"trained on {used} baskets, {len(skus)} skus")
    probes = ["chicken_thigh", "feta", "cilantro", "lentils", "milk_2"]
    for ingredient_id in probes:
        anchor = next(product for product in catalog.products if product.ingredient_id == ingredient_id)
        print(f"\n{anchor.ingredient}  held_out={held_out_ingredient(ingredient_id)}")
        for sku, score in store.nearest(anchor.sku, k=5):
            other = catalog.by_sku[sku]
            print(f"  {score:.3f}  {other.group:16}  {other.brand} {other.ingredient}")


if __name__ == "__main__":
    main()
