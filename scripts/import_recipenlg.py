#!/usr/bin/env python3
"""Read a RecipeNLG CSV and write title plus ingredient lines.

RecipeNLG rows look like: title, ingredients, directions, NER.
`ingredients` is a stringified list. The demo planner uses the curated
recipes in data/seed, which are already aligned to the catalog.
"""

from __future__ import annotations

import ast
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/import_recipenlg.py path/to/RecipeNLG.csv [limit]")
        sys.exit(1)
    source = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    if not source.exists():
        print(f"CSV not found: {source}")
        sys.exit(1)
    dest = ROOT / "data" / "external" / "recipenlg.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with source.open(newline="") as fh, dest.open("w") as out:
        for row in csv.DictReader(fh):
            if kept >= limit:
                break
            raw = row.get("ingredients") or "[]"
            try:
                ingredients = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                continue
            if not isinstance(ingredients, list):
                continue
            out.write(json.dumps({
                "title": row.get("title"),
                "ingredients": [str(item) for item in ingredients],
            }) + "\n")
            kept += 1
    print(f"wrote {kept} recipes to {dest}")


if __name__ == "__main__":
    main()
