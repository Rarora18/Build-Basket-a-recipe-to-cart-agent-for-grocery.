"""Turn a request into meals, then one priced, diet-safe cart."""

from __future__ import annotations

import math

from basket.catalog import Catalog
from basket.models import Goal, Ingredient, Product, Recipe, allowed
from basket.optimize import Option, solve
from basket.planner import estimate_cost, plan, protein_grams
from basket.ranker import Ranker
from basket.retrieve import Retriever


def _money(cents: int) -> float:
    return round(cents / 100, 2)


def _option(
    product: Product,
    need_g: float,
    ingredient_id: str,
    substitute: bool,
    match: float,
    covers: tuple[str, ...] = (),
) -> Option | None:
    packs = max(1, math.ceil(need_g / product.size_g - 1e-9))
    if packs > 6:
        return None
    used = min(need_g, packs * product.size_g)
    protein = product.protein_100 * used / 100
    cost = packs * product.unit_cents
    utility = (
        2.6 * match
        + 0.02 * protein
        + 1.6 * product.deal_pct
        + (0.9 if not substitute else 0.0)
        - 0.04 * (cost / 100)
    )
    return Option(
        product=product,
        packs=packs,
        cost_cents=cost,
        utility=utility,
        protein_g=protein,
        substitute=substitute,
        ingredient_id=ingredient_id,
        need_g=need_g,
        covers=covers,
    )


def _options_for(
    ingredient: Ingredient,
    need_g: float,
    goal: Goal,
    catalog: Catalog,
    retriever: Retriever,
) -> list[Option]:
    in_stock = [
        product for product in catalog.products_for(ingredient.id)
        if product.stock > 0 and allowed(product, goal)
    ]
    hits = {hit.product.sku: hit.score for hit in retriever.search(ingredient.name, k=8, goal=goal)}
    options = []
    for product in in_stock:
        option = _option(product, need_g, ingredient.id, False, hits.get(product.sku, 0.4))
        if option:
            options.append(option)
    return options


def _buy_instead(ingredient: Ingredient, goal: Goal, catalog: Catalog, ranker: Ranker) -> str | None:
    """Ingredient to purchase. None when nothing in stock can stand in."""
    in_stock = [
        product for product in catalog.products_for(ingredient.id)
        if product.stock > 0 and allowed(product, goal)
    ]
    if in_stock:
        return ingredient.id
    anchors = catalog.products_for(ingredient.id)
    if not anchors:
        return None
    pool = [
        product for product in catalog.products
        if product.stock > 0
        and product.ingredient_id != ingredient.id
        and allowed(product, goal)
        and (product.aisle == ingredient.aisle or product.group == ingredient.group)
    ]
    ranked = ranker.rank_pool(anchors[0], pool)
    if not ranked:
        return None
    return ranked[0].ingredient_id


def _line_note(option: Option, catalog: Catalog) -> str | None:
    product = option.product
    note = None
    if option.covers:
        covered = ", ".join(name.lower() for name in option.covers)
        note = f"No {covered} in stock. Substituted {product.brand} {product.name}."
    oos = next((item for item in catalog.products_for(option.ingredient_id) if item.stock <= 0), None)
    if oos and not option.covers:
        note = (
            f"{oos.brand} {oos.name} is out of stock, so the cart uses "
            f"{product.brand} {product.name}."
        )
    if product.deal_pct >= 0.2:
        flyer = f"On this week's flyer, {int(round(product.deal_pct * 100))}% off."
        note = f"{note} {flyer}" if note else flyer
    return note


def _aggregate(meals: list[Recipe], servings: int) -> dict[str, float]:
    scale = servings / 2
    need: dict[str, float] = {}
    for meal in meals:
        for ingredient_id, grams in meal.ingredients:
            need[ingredient_id] = need.get(ingredient_id, 0.0) + grams * scale
    return need


def _meal_value(meal: Recipe, catalog: Catalog, servings: int) -> float:
    protein = protein_grams(meal, catalog, servings)
    cost = estimate_cost(meal, catalog, servings)
    return protein / max(cost, 0.5)


def build_cart(query: str, catalog: Catalog, retriever: Retriever, ranker: Ranker) -> dict:
    if not query or not query.strip():
        raise ValueError("Type a meal request.")
    goal, selected, planner_name, _skipped = plan(query, catalog)
    notes: list[str] = []
    budget_cents = int(round(goal.budget * 100)) if goal.budget is not None else 1_000_000
    current = list(selected)
    dropped: list[tuple[Recipe, str]] = []

    def solvable(meals: list[Recipe]) -> list[Option] | None:
        if not meals:
            return []
        raw = _aggregate(meals, goal.servings)
        merged: dict[str, float] = {}
        covered: dict[str, list[str]] = {}
        for ingredient_id, grams in raw.items():
            ingredient = catalog.ingredients[ingredient_id]
            dest = _buy_instead(ingredient, goal, catalog, ranker)
            if not dest:
                return None
            merged[dest] = merged.get(dest, 0.0) + grams
            if dest != ingredient_id:
                covered.setdefault(dest, []).append(ingredient.name)
        groups: list[list[Option]] = []
        for ingredient_id, grams in merged.items():
            options = _options_for(
                catalog.ingredients[ingredient_id], grams, goal, catalog, retriever
            )
            covers = tuple(covered.get(ingredient_id, ()))
            for option in options:
                option.covers = covers
                option.substitute = bool(covers)
            if not options:
                return None
            groups.append(options)
        return solve(groups, budget_cents)

    while True:
        picked = solvable(current)
        if picked is not None:
            break
        if not current:
            picked = []
            break
        victim = min(current, key=lambda meal: _meal_value(meal, catalog, goal.servings))
        current.remove(victim)
        reason = "budget" if goal.budget is not None else "unavailable"
        dropped.append((victim, reason))
        if goal.budget is not None:
            notes.append(f"Dropped {victim.name} to stay under ${goal.budget:.0f}.")
        else:
            notes.append(f"Dropped {victim.name} because an ingredient could not be filled.")

    lines = []
    total_cents = 0
    savings_cents = 0
    protein = 0.0
    for option in picked or []:
        product = option.product
        full = option.packs * product.price_cents
        savings_cents += full - option.cost_cents
        total_cents += option.cost_cents
        protein += option.protein_g
        ingredient = catalog.ingredients[option.ingredient_id]
        lines.append({
            "ingredient_id": option.ingredient_id,
            "ingredient": ingredient.name,
            "sku": product.sku,
            "brand": product.brand,
            "product": product.name,
            "aisle": product.aisle,
            "packs": option.packs,
            "size_g": product.size_g,
            "unit_price": _money(product.unit_cents),
            "line_total": _money(option.cost_cents),
            "deal_pct": product.deal_pct,
            "protein_g": round(option.protein_g, 1),
            "substitute": option.substitute,
            "note": _line_note(option, catalog),
        })
    lines.sort(key=lambda line: (-line["line_total"], line["ingredient"]))

    kept_ids = {meal.id for meal in current}
    meals_out = []
    for meal in selected:
        meals_out.append({
            "id": meal.id,
            "name": meal.name,
            "blurb": meal.blurb,
            "minutes": meal.minutes,
            "protein_g": round(protein_grams(meal, catalog, goal.servings), 1),
            "kept": meal.id in kept_ids,
            "drop_reason": None if meal.id in kept_ids else "budget",
        })
    for meal, reason in dropped:
        if any(row["id"] == meal.id for row in meals_out):
            continue
        meals_out.append({
            "id": meal.id,
            "name": meal.name,
            "blurb": meal.blurb,
            "minutes": meal.minutes,
            "protein_g": round(protein_grams(meal, catalog, goal.servings), 1),
            "kept": False,
            "drop_reason": reason,
        })

    diet_ok = all(allowed(catalog.by_sku[line["sku"]], goal) for line in lines) if lines else True
    kept_count = max(1, sum(1 for meal in meals_out if meal["kept"])) if any(m["kept"] for m in meals_out) else 0
    per_dinner = round(protein / goal.servings / kept_count, 1) if kept_count else 0.0
    within = goal.budget is None or total_cents <= budget_cents + 1
    return {
        "query": goal.raw,
        "planner": planner_name,
        "goal": goal.as_dict(),
        "meals": meals_out,
        "lines": lines,
        "total": _money(total_cents),
        "savings": _money(savings_cents),
        "protein_g": round(protein, 1),
        "protein_each_dinner": per_dinner,
        "budget": goal.budget,
        "within_budget": within,
        "diet_ok": diet_ok,
        "notes": notes,
    }
