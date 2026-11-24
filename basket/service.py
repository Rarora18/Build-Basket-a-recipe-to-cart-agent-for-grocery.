from __future__ import annotations

import time

from basket.cart import build_cart
from basket.catalog import load_catalog
from basket.config import database_url
from basket.item2vec import ItemVectors
from basket.ranker import Ranker
from basket.retrieve import Retriever


class BasketService:
    def __init__(self):
        self.catalog = load_catalog()
        self.vectors = ItemVectors.load()
        self.ranker = Ranker.load(self.catalog, self.vectors)
        self.backend = "memory"
        self.retriever = Retriever(self.catalog.products)
        if database_url():
            from basket.postgres import PostgresRetriever, sync_catalog

            sync_catalog(self.catalog, self.retriever.vectors)
            self.retriever = PostgresRetriever(self.catalog)
            self.backend = "postgres"

    def build(self, query: str) -> dict:
        started = time.perf_counter()
        cart = build_cart(query, self.catalog, self.retriever, self.ranker)
        cart["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        cart["backend"] = self.backend
        return cart
