#!/usr/bin/env python3
"""Run the offline evals and write evals/latest.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basket.catalog import load_catalog
from basket.evals import constraint_report, latency_report, matching_report, substitution_report
from basket.item2vec import ItemVectors
from basket.ranker import Ranker
from basket.retrieve import Retriever
from basket.service import BasketService


def main() -> None:
    catalog = load_catalog()
    vectors = ItemVectors.load()
    ranker = Ranker.load(catalog, vectors)
    retriever = Retriever(catalog.products)
    service = BasketService()
    report = {
        "matching": matching_report(catalog, retriever),
        "substitution": substitution_report(catalog, vectors, ranker),
        "constraints": constraint_report(service.build),
        "latency": latency_report(service.build),
    }
    # The service build includes its own timer. Keep the harness number too.
    out = ROOT / "evals" / "latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    matching = report["matching"]
    subs = report["substitution"]
    print(
        f"precision@5  bm25 {matching['bm25']['precision_at_5']}  "
        f"embed {matching['embed']['precision_at_5']}  "
        f"hybrid {matching['hybrid']['precision_at_5']}"
    )
    print(
        f"substitution hit@1  popularity {subs['popularity_hit_at_1']}  "
        f"item2vec {subs['item2vec_hit_at_1']}  ranker {subs['ranker_hit_at_1']}  "
        f"lift {subs['ranker_lift_vs_popularity_pct']}%"
    )
    print(
        f"constraints broken {report['constraints']['carts_breaking_a_constraint_pct']}%  "
        f"latency p50 {report['latency']['p50_ms']} ms"
    )


if __name__ == "__main__":
    main()
