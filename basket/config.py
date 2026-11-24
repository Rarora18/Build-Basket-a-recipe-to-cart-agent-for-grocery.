from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = ROOT / "data" / "seed"
ARTIFACT_DIR = ROOT / "artifacts"

EMBED_DIM = 128
# Adaptive fusion: lexical weight is LEX_WEIGHT + SEM_WEIGHT * (top lexical share).
LEX_WEIGHT = 0.15
SEM_WEIGHT = 0.40

# Groups where a wrong swap changes the meal. Substitution eval focuses here.
FOCUS_GROUPS = frozenset({
    "poultry",
    "beef",
    "pork",
    "finfish",
    "shellfish",
    "cheese",
    "milk",
    "yogurt",
    "plant_protein",
    "greens",
    "grain",
})


def database_url() -> str | None:
    url = os.getenv("DATABASE_URL", "").strip()
    return url or None
