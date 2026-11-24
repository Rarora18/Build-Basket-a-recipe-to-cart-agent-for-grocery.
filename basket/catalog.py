from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from basket.config import SEED_DIR
from basket.models import Ingredient, Product, Recipe

_BOOLS = (
    "pork", "beef", "dairy", "nuts", "shellfish", "soy", "fish", "egg",
    "vegetarian", "vegan", "gluten_free",
)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ingredient(row: dict) -> Ingredient:
    return Ingredient(
        id=row["id"],
        name=row["name"],
        aisle=row["aisle"],
        group=row["group"],
        kcal_100=float(row["kcal_100"]),
        protein_100=float(row["protein_100"]),
        carbs_100=float(row["carbs_100"]),
        fat_100=float(row["fat_100"]),
        allergens=tuple(row["allergens"]),
        **{key: bool(row[key]) for key in _BOOLS},
    )


def _product(row: dict) -> Product:
    return Product(
        sku=row["sku"],
        ingredient_id=row["ingredient_id"],
        ingredient=row["ingredient"],
        name=row["name"],
        brand=row["brand"],
        aisle=row["aisle"],
        group=row["group"],
        size_g=int(row["size_g"]),
        price_cents=int(row["price_cents"]),
        stock=int(row["stock"]),
        deal_pct=float(row["deal_pct"]),
        kcal_100=float(row["kcal_100"]),
        protein_100=float(row["protein_100"]),
        carbs_100=float(row["carbs_100"]),
        fat_100=float(row["fat_100"]),
        allergens=tuple(row["allergens"]),
        search_text=row["search_text"],
        **{key: bool(row[key]) for key in _BOOLS},
    )


def _recipe(row: dict) -> Recipe:
    items = tuple((item["ingredient_id"], int(item["grams"])) for item in row["ingredients"])
    return Recipe(
        id=row["id"],
        name=row["name"],
        blurb=row["blurb"],
        minutes=int(row["minutes"]),
        tags=tuple(row["tags"]),
        ingredients=items,
    )


@dataclass
class Catalog:
    ingredients: dict[str, Ingredient]
    products: list[Product]
    recipes: list[Recipe]
    by_sku: dict[str, Product]

    def products_for(self, ingredient_id: str) -> list[Product]:
        return [p for p in self.products if p.ingredient_id == ingredient_id]


def load_catalog(seed_dir: Path | None = None) -> Catalog:
    folder = seed_dir or SEED_DIR
    ingredients = {row.id: row for row in (_ingredient(r) for r in _read_jsonl(folder / "ingredients.jsonl"))}
    products = [_product(r) for r in _read_jsonl(folder / "products.jsonl")]
    recipes = [_recipe(r) for r in _read_jsonl(folder / "recipes.jsonl")]
    return Catalog(
        ingredients=ingredients,
        products=products,
        recipes=recipes,
        by_sku={p.sku: p for p in products},
    )


def load_orders(seed_dir: Path | None = None) -> list[dict]:
    folder = seed_dir or SEED_DIR
    return _read_jsonl(folder / "orders.jsonl")


def load_labels(seed_dir: Path | None = None) -> list[dict]:
    folder = seed_dir or SEED_DIR
    return _read_jsonl(folder / "labels.jsonl")
