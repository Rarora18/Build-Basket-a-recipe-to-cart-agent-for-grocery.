import json
from pathlib import Path

import pytest

from basket.catalog import load_catalog, load_labels
from basket.evals import constraint_report, matching_report, substitution_report
from basket.item2vec import ItemVectors
from basket.models import allowed
from basket.planner import parse_goal
from basket.ranker import Ranker
from basket.service import BasketService

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def service():
    return BasketService()


def test_demo_request_parses():
    goal = parse_goal("high-protein dinners for 2 this week, under $60, no pork.")
    assert goal.meals_requested == 5
    assert goal.servings == 2
    assert goal.budget == 60
    assert goal.high_protein
    assert goal.exclude_pork
    assert not goal.vegetarian


def test_labelled_set_has_200_queries():
    labels = load_labels()
    assert len(labels) == 200
    assert {row["kind"] for row in labels} >= {"clean", "synonym", "typo"}


def test_synonym_and_typo_retrieval(service):
    hits = service.retriever.search("coriander", k=3)
    assert hits[0].product.ingredient_id == "cilantro"
    hits = service.retriever.search("chicken thghs", k=3)
    assert hits[0].product.ingredient_id == "chicken_thigh"
    hits = service.retriever.search("icing sugar", k=3)
    assert hits[0].product.ingredient_id == "powdered_sugar"


def test_hybrid_beats_either_channel_alone(service):
    report = matching_report(service.catalog, service.retriever)
    hybrid = report["hybrid"]["precision_at_5"]
    assert hybrid >= report["bm25"]["precision_at_5"]
    assert hybrid >= report["embed"]["precision_at_5"]
    assert report["hybrid"]["hit_at_5"] >= report["bm25"]["hit_at_5"]


def test_ranker_beats_popularity(service):
    report = substitution_report(service.catalog, service.vectors, service.ranker)
    assert report["n"] > 50
    assert report["ranker_hit_at_1"] >= report["popularity_hit_at_1"]
    assert report["item2vec_hit_at_1"] >= report["popularity_hit_at_1"]


def test_feta_substitute_stays_in_cheese(service):
    anchor = next(product for product in service.catalog.products if product.ingredient_id == "feta")
    goal = parse_goal("dinners for 2")
    pool = [
        product for product in service.catalog.products
        if product.stock > 0 and product.sku != anchor.sku and allowed(product, goal)
        and (product.aisle == anchor.aisle or product.group == anchor.group)
    ]
    top = service.ranker.rank_pool(anchor, pool)[0]
    assert top.group == "cheese"
    assert top.ingredient_id != "feta"


def test_demo_cart_holds_budget_and_swaps_lentils(service):
    cart = service.build("high-protein dinners for 2 this week, under $60, no pork.")
    assert cart["within_budget"]
    assert cart["diet_ok"]
    assert cart["total"] <= 60
    assert sum(1 for meal in cart["meals"] if meal["kept"]) >= 3
    assert any(line["substitute"] for line in cart["lines"])
    for line in cart["lines"]:
        product = service.catalog.by_sku[line["sku"]]
        assert not product.pork
        assert product.stock > 0
        assert "pork" not in product.name.lower()
        assert "bacon" not in product.name.lower()


def test_diet_carts_do_not_break_constraints(service):
    report = constraint_report(service.build)
    assert report["budget_violations"] == 0
    assert report["diet_violations"] == 0
    vegan = service.build("vegan dinners for 2 this week under $50")
    assert all(service.catalog.by_sku[line["sku"]].vegan for line in vegan["lines"])
    gluten = service.build("gluten-free high-protein dinners for 2, 4 meals, under $55, no pork")
    assert all(service.catalog.by_sku[line["sku"]].gluten_free for line in gluten["lines"])


def test_recorded_metrics_match_a_fresh_run(service):
    recorded = json.loads((ROOT / "evals" / "latest.json").read_text())
    fresh = matching_report(service.catalog, service.retriever)
    assert fresh["hybrid"]["precision_at_5"] == pytest.approx(
        recorded["matching"]["hybrid"]["precision_at_5"], abs=1e-4
    )


def test_vectors_and_ranker_are_present():
    catalog = load_catalog()
    vectors = ItemVectors.load()
    Ranker.load(catalog, vectors)
    assert len(vectors.skus) == len(catalog.products)
