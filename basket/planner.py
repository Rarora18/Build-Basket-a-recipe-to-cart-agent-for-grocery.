"""Parse a meal request and choose recipes that respect it."""

from __future__ import annotations

import json
import os
import re

import httpx

from basket.catalog import Catalog
from basket.models import Goal, Recipe, allowed

_PROTEIN_GROUPS = {"poultry", "beef", "pork", "finfish", "shellfish", "plant_protein", "eggs"}


def parse_goal(text: str) -> Goal:
    raw = " ".join(text.split())
    q = raw.lower().replace("-", " ")
    q = " ".join(q.split())
    numbered = re.search(r"(\d+)\s+(dinner|meal|night|lunch)", q)
    week = "this week" in q or "for the week" in q
    if numbered:
        meals = int(numbered.group(1))
    elif week:
        meals = 5
    else:
        meals = 3
    people = re.search(r"for\s+(\d+)", q)
    servings = int(people.group(1)) if people else 2
    budget_match = re.search(r"(?:under|below|less than)\s*\$?\s*(\d+(?:\.\d+)?)", q)
    if not budget_match:
        budget_match = re.search(r"\$\s*(\d+(?:\.\d+)?)", q)
    budget = float(budget_match.group(1)) if budget_match else None

    def has(*phrases: str) -> bool:
        return any(phrase in q for phrase in phrases)

    vegan = has("vegan")
    return Goal(
        raw=raw,
        meals_requested=max(1, min(7, meals)),
        servings=max(1, min(8, servings)),
        budget=budget,
        high_protein=has("high protein"),
        exclude_pork=has("no pork", "pork free", "without pork") or vegan,
        exclude_beef=has("no beef", "beef free", "without beef") or vegan,
        vegetarian=vegan or has("vegetarian"),
        vegan=vegan,
        gluten_free=has("gluten free", "celiac"),
        dairy_free=has("dairy free", "no dairy", "lactose") or vegan,
        nut_free=has("nut free", "no nuts", "peanut allergy", "nut allergy"),
        shellfish_free=has("no shellfish", "shellfish free", "shellfish allergy"),
        soy_free=has("soy free", "no soy"),
    )


def recipe_ok(recipe: Recipe, goal: Goal, catalog: Catalog) -> bool:
    return all(allowed(catalog.ingredients[iid], goal) for iid, _grams in recipe.ingredients)


def protein_grams(recipe: Recipe, catalog: Catalog, servings: int) -> float:
    base = sum(catalog.ingredients[iid].protein_100 * grams / 100 for iid, grams in recipe.ingredients)
    return base * servings / 2


def estimate_cost(recipe: Recipe, catalog: Catalog, servings: int) -> float:
    scale = servings / 2
    total = 0.0
    for iid, grams in recipe.ingredients:
        stocked = [p for p in catalog.products_for(iid) if p.stock > 0]
        pool = stocked or catalog.products_for(iid)
        if not pool:
            return 1e6
        best = min(pool, key=lambda p: p.unit_cents / p.size_g)
        total += grams * scale * best.unit_cents / best.size_g
    return total / 100


def _primary(recipe: Recipe, catalog: Catalog) -> str:
    for iid, _grams in recipe.ingredients:
        group = catalog.ingredients[iid].group
        if group in _PROTEIN_GROUPS:
            return group
    return ""


def select_meals(goal: Goal, catalog: Catalog) -> tuple[list[Recipe], Recipe | None]:
    feasible = [recipe for recipe in catalog.recipes if recipe_ok(recipe, goal, catalog)]

    def score(recipe: Recipe) -> float:
        protein = protein_grams(recipe, catalog, goal.servings)
        cost = estimate_cost(recipe, catalog, goal.servings)
        if goal.high_protein:
            return protein / max(cost, 0.8)
        return -cost

    ranked = sorted(feasible, key=score, reverse=True)
    chosen: list[Recipe] = []
    used: set[str] = set()
    spent = 0.0
    for recipe in ranked:
        if len(chosen) >= goal.meals_requested:
            break
        group = _primary(recipe, catalog)
        if group and group in used:
            continue
        cost = estimate_cost(recipe, catalog, goal.servings)
        if goal.budget is not None and chosen and spent + cost > goal.budget * 1.08:
            continue
        chosen.append(recipe)
        spent += cost
        if group:
            used.add(group)
    if len(chosen) < goal.meals_requested:
        for recipe in ranked:
            if recipe in chosen or len(chosen) >= goal.meals_requested:
                continue
            cost = estimate_cost(recipe, catalog, goal.servings)
            if goal.budget is not None and chosen and spent + cost > goal.budget * 1.2:
                continue
            chosen.append(recipe)
            spent += cost
    skipped = next((recipe for recipe in ranked if recipe not in chosen), None)
    return chosen, skipped


def _llm_ids(goal: Goal, recipes: list[Recipe]) -> list[str] | None:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    brief = [{"id": r.id, "name": r.name, "tags": list(r.tags), "minutes": r.minutes} for r in recipes]
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Pick dinner recipes for a grocery cart. Reply with JSON "
                    '{"meal_ids": ["..."]} using only ids from the list. '
                    "Honor the budget, servings, and diet flags. Prefer higher-protein "
                    "recipes when high_protein is true."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "count": goal.meals_requested,
                    "goal": goal.as_dict(),
                    "recipes": brief,
                }),
            },
        ],
    }
    try:
        response = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        ids = json.loads(content).get("meal_ids", [])
    except (httpx.HTTPError, KeyError, json.JSONDecodeError, TypeError, ValueError):
        return None
    allowed_ids = {recipe.id for recipe in recipes}
    picked: list[str] = []
    for meal_id in ids:
        if meal_id in allowed_ids and meal_id not in picked:
            picked.append(meal_id)
    return picked[: goal.meals_requested] or None


def plan(query: str, catalog: Catalog) -> tuple[Goal, list[Recipe], str, Recipe | None]:
    goal = parse_goal(query)
    feasible = [recipe for recipe in catalog.recipes if recipe_ok(recipe, goal, catalog)]
    llm_ids = _llm_ids(goal, feasible)
    if llm_ids:
        by_id = {recipe.id: recipe for recipe in feasible}
        return goal, [by_id[meal_id] for meal_id in llm_ids], "llm", None
    meals, skipped = select_meals(goal, catalog)
    return goal, meals, "heuristic", skipped
