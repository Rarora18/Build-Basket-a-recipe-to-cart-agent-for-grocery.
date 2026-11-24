"""Synonyms and common misspellings for the embedding channel.

BM25 stays literal. This lexicon is applied only when embedding a query or a
product name, so "coriander" can find cilantro without pretending the tokens match.
"""

from __future__ import annotations

# Clique: every token expands to the others.
_CLIQUES = [
    ["cilantro", "coriander", "cilanto"],
    ["zucchini", "courgette", "zuchini"],
    ["eggplant", "aubergine"],
    ["arugula", "rocket"],
    ["beet", "beetroot"],
    ["shrimp", "prawn"],
    ["yogurt", "yoghurt"],
    ["chickpea", "garbanzo"],
    ["canola", "rapeseed"],
    ["broth", "stock"],
    ["chicken", "chkn", "chiken"],
    ["broccoli", "brocoli", "brocolli"],
    ["avocado", "avacado"],
    ["turmeric", "tumeric"],
    ["thigh", "thgh", "thygh"],
    ["sriracha", "siracha"],
]

# Directed extras that should not form a full clique.
_DIRECTED = {
    "scallion": ("green", "onion"),
    "capsicum": ("bell",),
    "bell": ("capsicum",),
    "bicarb": ("baking", "soda"),
    "bicarbonate": ("baking", "soda"),
    "tinned": ("canned",),
    "canned": ("tinned",),
    "mince": ("ground",),
    "minced": ("ground",),
    "icing": ("powdered",),
    "powdered": ("icing",),
    "rapini": ("broccoli", "raab"),
    "raab": ("rapini",),
    "sriracha": ("hot", "sauce"),
    "siracha": ("sriracha", "hot", "sauce"),
}

# Phrase on the raw lowercase string -> extra tokens.
PHRASES: list[tuple[str, tuple[str, ...]]] = [
    ("spring onion", ("green", "onion", "scallion")),
    ("green onion", ("scallion", "spring", "onion")),
    ("icing sugar", ("powdered", "sugar")),
    ("powdered sugar", ("icing", "sugar")),
    ("baking soda", ("bicarb", "bicarbonate", "soda")),
    ("bicarbonate of soda", ("baking", "soda", "bicarb")),
    ("all-purpose flour", ("plain", "flour")),
    ("all purpose flour", ("plain", "flour")),
    ("plain flour", ("flour", "purpose")),
    ("beef mince", ("ground", "beef")),
    ("minced beef", ("ground", "beef")),
    ("tinned tomato", ("canned", "tomato")),
    ("canned tomato", ("tinned", "tomato")),
    ("chick pea", ("chickpea",)),
    ("bell pepper", ("capsicum",)),
    ("rapeseed oil", ("canola", "oil")),
    ("canola oil", ("rapeseed", "oil")),
    ("vegetable stock", ("vegetable", "broth")),
    ("vegetable broth", ("vegetable", "stock")),
    ("chicken stock", ("chicken", "broth")),
    ("chicken broth", ("chicken", "stock")),
    ("greek yoghurt", ("greek", "yogurt")),
    ("plain yoghurt", ("plain", "yogurt")),
    ("hot sauce", ("sriracha",)),
    ("ground beef", ("mince", "beef")),
    ("ground turkey", ("mince", "turkey")),
    ("broccoli raab", ("rapini",)),
]


def token_map() -> dict[str, tuple[str, ...]]:
    out: dict[str, set[str]] = {}
    for group in _CLIQUES:
        for tok in group:
            out.setdefault(tok, set()).update(t for t in group if t != tok)
    for tok, extras in _DIRECTED.items():
        out.setdefault(tok, set()).update(extras)
    return {k: tuple(sorted(v)) for k, v in out.items()}


TOKEN_MAP = token_map()
