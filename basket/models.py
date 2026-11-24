from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"[a-z]+")
_PORK_WORDS = {"pork", "bacon", "ham", "prosciutto"}
_BEEF_WORDS = {"beef", "steak", "sirloin"}


@dataclass(frozen=True)
class Ingredient:
    id: str
    name: str
    aisle: str
    group: str
    kcal_100: float
    protein_100: float
    carbs_100: float
    fat_100: float
    allergens: tuple[str, ...]
    pork: bool
    beef: bool
    dairy: bool
    nuts: bool
    shellfish: bool
    soy: bool
    fish: bool
    egg: bool
    vegetarian: bool
    vegan: bool
    gluten_free: bool


@dataclass(frozen=True)
class Product:
    sku: str
    ingredient_id: str
    ingredient: str
    name: str
    brand: str
    aisle: str
    group: str
    size_g: int
    price_cents: int
    stock: int
    deal_pct: float
    kcal_100: float
    protein_100: float
    carbs_100: float
    fat_100: float
    allergens: tuple[str, ...]
    pork: bool
    beef: bool
    dairy: bool
    nuts: bool
    shellfish: bool
    soy: bool
    fish: bool
    egg: bool
    vegetarian: bool
    vegan: bool
    gluten_free: bool
    search_text: str

    @property
    def unit_cents(self) -> int:
        return int(round(self.price_cents * (1 - self.deal_pct)))


@dataclass(frozen=True)
class Recipe:
    id: str
    name: str
    blurb: str
    minutes: int
    tags: tuple[str, ...]
    ingredients: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Goal:
    raw: str
    meals_requested: int
    servings: int
    budget: float | None
    high_protein: bool
    exclude_pork: bool
    exclude_beef: bool
    vegetarian: bool
    vegan: bool
    gluten_free: bool
    dairy_free: bool
    nut_free: bool
    shellfish_free: bool
    soy_free: bool

    def as_dict(self) -> dict:
        return {
            "raw": self.raw,
            "meals_requested": self.meals_requested,
            "servings": self.servings,
            "budget": self.budget,
            "high_protein": self.high_protein,
            "exclude_pork": self.exclude_pork,
            "exclude_beef": self.exclude_beef,
            "vegetarian": self.vegetarian,
            "vegan": self.vegan,
            "gluten_free": self.gluten_free,
            "dairy_free": self.dairy_free,
            "nut_free": self.nut_free,
            "shellfish_free": self.shellfish_free,
            "soy_free": self.soy_free,
        }


def _blob(obj: Ingredient | Product) -> str:
    parts = [
        getattr(obj, "name", ""),
        getattr(obj, "ingredient", ""),
        getattr(obj, "id", ""),
        getattr(obj, "ingredient_id", ""),
    ]
    return " ".join(parts).lower()


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def allowed(obj: Ingredient | Product, goal: Goal) -> bool:
    if goal.vegan and not obj.vegan:
        return False
    if goal.vegetarian and not obj.vegetarian:
        return False
    if goal.gluten_free and not obj.gluten_free:
        return False
    if goal.dairy_free and obj.dairy:
        return False
    if goal.nut_free and obj.nuts:
        return False
    if goal.shellfish_free and obj.shellfish:
        return False
    if goal.soy_free and obj.soy:
        return False
    words = _words(_blob(obj))
    if goal.exclude_pork and (obj.pork or words & _PORK_WORDS):
        return False
    if goal.exclude_beef and (obj.beef or words & _BEEF_WORDS):
        return False
    return True
