"""Canonical observations and immutable action keys."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, order=True)
class Action:
    id: int
    x: int | None = None
    y: int | None = None

    @property
    def data(self) -> dict[str, int]:
        if self.x is None or self.y is None:
            return {}
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True)
class State:
    id: str
    game: str
    level: int
    status: str
    available: tuple[int, ...]
    frames: tuple[torch.Tensor, ...]

    @property
    def terminal(self) -> bool:
        return self.status in {"WIN", "GAME_OVER"}

    def payload(self) -> dict[str, Any]:
        return {
            "game": self.game,
            "level": self.level,
            "status": self.status,
            "available": list(self.available),
            "frames": [frame.tolist() for frame in self.frames],
        }


def normalize(frames: Any, game: str, level: int, status: str, available: Any) -> State:
    tensors = []
    for frame in frames:
        value = torch.as_tensor(frame, device="cpu")
        if value.ndim != 2:
            raise ValueError("Each observation frame must be a two-dimensional grid")
        if value.is_floating_point() and (
            not bool(torch.isfinite(value).all())
            or not bool((value == value.trunc()).all())
        ):
            raise ValueError("Observation cells must be finite integers")
        tensors.append(value.to(dtype=torch.int64).contiguous().clone())
    actions = tuple(sorted({int(getattr(a, "value", a)) for a in available}))
    digest = hashlib.sha256()
    metadata = [game, int(level), status, actions, [list(t.shape) for t in tensors]]
    digest.update(json.dumps(metadata, separators=(",", ":")).encode())
    for tensor in tensors:
        digest.update(tensor.numpy().astype("<i8", copy=False).tobytes())
    return State(digest.hexdigest(), game, int(level), status, actions, tuple(tensors))


def encode(observation: Any) -> State:
    return normalize(
        observation.frame,
        observation.game_id or "",
        observation.levels_completed,
        observation.state.name,
        observation.available_actions,
    )


def changed_cells(before: State, after: State) -> list[tuple[int, int, int]]:
    """Return (frame, row, column), including added/removed cells and frames."""
    changed: list[tuple[int, int, int]] = []
    for i in range(max(len(before.frames), len(after.frames))):
        a = before.frames[i] if i < len(before.frames) else torch.empty((0, 0))
        b = after.frames[i] if i < len(after.frames) else torch.empty((0, 0))
        if a.shape == b.shape:
            changed.extend((i, y, x) for y, x in torch.nonzero(a != b).tolist())
        else:
            for y in range(max(a.shape[0], b.shape[0])):
                for x in range(max(a.shape[1], b.shape[1])):
                    in_a = y < a.shape[0] and x < a.shape[1]
                    in_b = y < b.shape[0] and x < b.shape[1]
                    if in_a != in_b or (in_a and in_b and a[y, x] != b[y, x]):
                        changed.append((i, y, x))
    return changed
