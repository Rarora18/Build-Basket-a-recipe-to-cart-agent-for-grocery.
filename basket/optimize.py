"""Multiple-choice integer program: one SKU per ingredient, under the budget."""

from __future__ import annotations

from dataclasses import dataclass

import pulp

from basket.models import Product


@dataclass
class Option:
    product: Product
    packs: int
    cost_cents: int
    utility: float
    protein_g: float
    substitute: bool
    ingredient_id: str
    need_g: float
    covers: tuple[str, ...] = ()


def solve(groups: list[list[Option]], budget_cents: int) -> list[Option] | None:
    if not groups or any(not group for group in groups):
        return None
    problem = pulp.LpProblem("basket", pulp.LpMaximize)
    choice: list[list[pulp.LpVariable]] = []
    for group_index, options in enumerate(groups):
        variables = []
        for option_index, _option in enumerate(options):
            variables.append(pulp.LpVariable(f"g{group_index}_o{option_index}", cat="Binary"))
        problem += pulp.lpSum(variables) == 1
        choice.append(variables)
    problem += (
        pulp.lpSum(
            choice[group_index][option_index] * option.cost_cents
            for group_index, options in enumerate(groups)
            for option_index, option in enumerate(options)
        )
        <= budget_cents
    )
    problem += pulp.lpSum(
        choice[group_index][option_index] * option.utility
        for group_index, options in enumerate(groups)
        for option_index, option in enumerate(options)
    )
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return None
    picked: list[Option] = []
    for group_index, options in enumerate(groups):
        chosen = None
        for option_index, option in enumerate(options):
            value = choice[group_index][option_index].value()
            if value is not None and value > 0.5:
                chosen = option
                break
        if chosen is None:
            return None
        picked.append(chosen)
    return picked
