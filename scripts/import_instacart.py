#!/usr/bin/env python3
"""Convert an Instacart Market Basket extract into basket order files.

Expects the Kaggle files in data/external/instacart/:
  products.csv, order_products__prior.csv

The demo trains without this. Point a later training run at the output if you
want item2vec fit on the public dump instead of the synthetic baskets.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "external" / "instacart"
DEST = ROOT / "data" / "external" / "instacart_orders.jsonl"


def main() -> None:
    products = SOURCE / "products.csv"
    orders = SOURCE / "order_products__prior.csv"
    missing = [str(path) for path in (products, orders) if not path.exists()]
    if missing:
        print("Instacart CSVs were not found.")
        print("Download the Market Basket dataset and unzip it into data/external/instacart/.")
        print("Missing:")
        for path in missing:
            print(" ", path)
        sys.exit(1)
    names = {}
    with products.open() as fh:
        for row in csv.DictReader(fh):
            names[row["product_id"]] = row["product_name"]
    grouped: dict[str, list[tuple[int, str]]] = {}
    with orders.open() as fh:
        for row in csv.DictReader(fh):
            grouped.setdefault(row["order_id"], []).append((int(row["add_to_cart_order"]), row["product_id"]))
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with DEST.open("w") as fh:
        for index, (order_id, items) in enumerate(grouped.items()):
            items.sort()
            fh.write(json.dumps({
                "order_id": order_id,
                "split": "test" if index % 5 == 0 else "train",
                "skus": [product_id for _pos, product_id in items],
                "names": [names.get(product_id, product_id) for _pos, product_id in items],
            }) + "\n")
    print(f"wrote {len(grouped)} orders to {DEST}")


if __name__ == "__main__":
    main()
