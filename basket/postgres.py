"""Postgres + pgvector retrieval. Lexical rank is ts_rank; fusion matches the in-process retriever."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import psycopg

from basket.catalog import Catalog, load_orders
from basket.config import EMBED_DIM, ROOT, database_url
from basket.models import Goal
from basket.retrieve import Hit, fuse

_SCHEMA = ROOT / "sql" / "001_schema.sql"
_conn: psycopg.Connection | None = None


def _statements(script: str) -> list[str]:
    chunks: list[str] = []
    buffer: list[str] = []
    for line in script.splitlines():
        if line.strip().startswith("--"):
            continue
        buffer.append(line)
        if line.rstrip().endswith(";"):
            statement = "\n".join(buffer).strip()
            if statement:
                chunks.append(statement)
            buffer = []
    tail = "\n".join(buffer).strip()
    if tail:
        chunks.append(tail)
    return chunks


def connect() -> psycopg.Connection:
    global _conn
    url = database_url()
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    if _conn is None or _conn.closed:
        _conn = psycopg.connect(url, autocommit=True)
    return _conn


def _vec(values: np.ndarray) -> str:
    return "[" + ",".join(format(float(value), ".6f") for value in values) + "]"


def sync_catalog(catalog: Catalog, embeddings: np.ndarray) -> None:
    if embeddings.shape[0] != len(catalog.products) or embeddings.shape[1] != EMBED_DIM:
        raise RuntimeError("embedding matrix does not match the catalog")
    conn = connect()
    schema = Path(_SCHEMA).read_text()
    with conn.cursor() as cur:
        for statement in _statements(schema):
            cur.execute(statement)
        cur.execute("TRUNCATE order_items, orders, recipe_ingredients, recipes, products CASCADE")
        cur.executemany(
            """
            INSERT INTO products (
                sku, ingredient_id, ingredient, name, brand, aisle, grp, size_g, price_cents,
                stock, deal_pct, kcal_100, protein_100, carbs_100, fat_100, allergens,
                pork, beef, dairy, nuts, shellfish, soy, fish, egg, vegetarian, vegan,
                gluten_free, search_text, embedding
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s::vector
            )
            """,
            [
                (
                    product.sku, product.ingredient_id, product.ingredient, product.name, product.brand,
                    product.aisle, product.group, product.size_g, product.price_cents, product.stock,
                    product.deal_pct, product.kcal_100, product.protein_100, product.carbs_100, product.fat_100,
                    list(product.allergens), product.pork, product.beef, product.dairy, product.nuts,
                    product.shellfish, product.soy, product.fish, product.egg, product.vegetarian,
                    product.vegan, product.gluten_free, product.search_text, _vec(embeddings[i]),
                )
                for i, product in enumerate(catalog.products)
            ],
        )
        cur.executemany(
            "INSERT INTO recipes (id, name, blurb, minutes, tags) VALUES (%s, %s, %s, %s, %s)",
            [(recipe.id, recipe.name, recipe.blurb, recipe.minutes, list(recipe.tags)) for recipe in catalog.recipes],
        )
        cur.executemany(
            "INSERT INTO recipe_ingredients (recipe_id, ingredient_id, grams) VALUES (%s, %s, %s)",
            [
                (recipe.id, ingredient_id, grams)
                for recipe in catalog.recipes
                for ingredient_id, grams in recipe.ingredients
            ],
        )
        orders = load_orders()
        cur.executemany(
            "INSERT INTO orders (order_id, split) VALUES (%s, %s)",
            [(order["order_id"], order["split"]) for order in orders],
        )
        cur.executemany(
            "INSERT INTO order_items (order_id, position, sku) VALUES (%s, %s, %s)",
            [
                (order["order_id"], position, sku)
                for order in orders
                for position, sku in enumerate(order["skus"])
            ],
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS products_embed_hnsw ON products USING hnsw (embedding vector_cosine_ops)"
        )


def _where(goal: Goal | None, in_stock: bool | None) -> str:
    clauses = ["TRUE"]
    if goal is not None:
        if goal.vegan:
            clauses.append("vegan")
        if goal.vegetarian:
            clauses.append("vegetarian")
        if goal.gluten_free:
            clauses.append("gluten_free")
        if goal.dairy_free:
            clauses.append("NOT dairy")
        if goal.nut_free:
            clauses.append("NOT nuts")
        if goal.shellfish_free:
            clauses.append("NOT shellfish")
        if goal.soy_free:
            clauses.append("NOT soy")
        if goal.exclude_pork:
            clauses.append("NOT pork")
        if goal.exclude_beef:
            clauses.append("NOT beef")
    if in_stock is True:
        clauses.append("stock > 0")
    elif in_stock is False:
        clauses.append("stock <= 0")
    return " AND ".join(clauses)


class PostgresRetriever:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.products = catalog.products

    def search(self, query: str, k: int = 8, goal: Goal | None = None, in_stock: bool | None = None) -> list[Hit]:
        if not query.strip():
            return []
        from basket.textutil import embed

        where = _where(goal, in_stock)
        sql = f"""
            SELECT sku,
                   ts_rank_cd(to_tsvector('english', search_text), plainto_tsquery('english', %s)) AS lex,
                   1 - (embedding <=> %s::vector) AS sem
            FROM products
            WHERE {where}
        """
        with connect().cursor() as cur:
            cur.execute(sql, (query, _vec(embed(query))))
            rows = cur.fetchall()
        if not rows:
            return []
        skus = [row[0] for row in rows]
        lexical = np.array([row[1] for row in rows], dtype=np.float64)
        semantic = np.array([row[2] for row in rows], dtype=np.float64)
        fused = fuse(lexical, semantic)
        order = sorted(range(len(skus)), key=lambda i: (-fused[i], skus[i]))
        hits = []
        for index in order[:k]:
            product = self.catalog.by_sku[skus[index]]
            hits.append(Hit(product, float(fused[index]), float(lexical[index]), float(semantic[index])))
        return hits
