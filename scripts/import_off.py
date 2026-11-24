#!/usr/bin/env python3
"""Stream an Open Food Facts JSONL dump and keep products sold in Canada.

The full dump is large. This writes a slim catalog the retrieval loader can
be pointed at later. The demo ships a curated seed so it runs offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/import_off.py path/to/openfoodfacts.jsonl [limit]")
        sys.exit(1)
    source = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    if not source.exists():
        print(f"Dump not found: {source}")
        sys.exit(1)
    dest = ROOT / "data" / "external" / "off_canada.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with source.open() as fh, dest.open("w") as out:
        for line in fh:
            if kept >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            countries = row.get("countries_tags") or []
            if "en:canada" not in countries:
                continue
            nutriments = row.get("nutriments") or {}
            out.write(json.dumps({
                "code": row.get("code"),
                "name": row.get("product_name") or row.get("product_name_en"),
                "brands": row.get("brands"),
                "allergens": row.get("allergens_tags") or [],
                "kcal_100": nutriments.get("energy-kcal_100g"),
                "protein_100": nutriments.get("proteins_100g"),
                "carbs_100": nutriments.get("carbohydrates_100g"),
                "fat_100": nutriments.get("fat_100g"),
            }) + "\n")
            kept += 1
    print(f"wrote {kept} Canadian products to {dest}")


if __name__ == "__main__":
    main()
