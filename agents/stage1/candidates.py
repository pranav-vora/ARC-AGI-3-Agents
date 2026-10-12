"""Legal actions for the pinned toolkit's 64×64 click interface."""

from functools import lru_cache

import torch
from arcengine import GameAction

from .state import Action, State


@lru_cache(maxsize=3)
def click_points(stage: int) -> tuple[tuple[int, int], ...]:
    points: set[tuple[int, int]] = set()
    for count in (8, 16, 64)[: stage + 1]:
        coordinates = torch.linspace(0, 63, count).round().to(torch.int64).tolist()
        points.update((x, y) for y in coordinates for x in coordinates)
    return tuple(sorted(points, key=lambda p: (p[1], p[0])))


def candidates(state: State, stage: int = 0) -> list[Action]:
    if state.terminal or state.status == "NOT_PLAYED":
        return []
    actions: list[Action] = []
    for action_id in state.available:
        action = GameAction.from_id(action_id)
        if action is GameAction.RESET:
            continue
        if action.is_complex():
            actions.extend(Action(action_id, x, y) for x, y in click_points(stage))
        else:
            actions.append(Action(action_id))
    return actions


def can_expand(state: State, stage: int) -> bool:
    return stage < 2 and any(
        GameAction.from_id(a).is_complex() for a in state.available
    )
