"""Observed directed multigraph; no inferred reverse edges."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .state import Action, State, changed_cells, normalize


@dataclass
class Node:
    state: State
    visits: int = 0
    tested: set[Action] = field(default_factory=set)
    expansion: int = 0
    retries: dict[Action, int] = field(default_factory=dict)


@dataclass
class Outcome:
    count: int = 0
    cost: float = 0
    changes: int = 0
    progress: int = 0
    failures: int = 0


class StateGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, dict[Action, dict[str, Outcome]]] = {}

    def get_or_create_node(self, state: State) -> Node:
        if state.id not in self.nodes:
            self.nodes[state.id] = Node(state)
        return self.nodes[state.id]

    def visit(self, state: State) -> Node:
        node = self.get_or_create_node(state)
        node.visits += 1
        return node

    def mark_action_tested(self, state_id: str, action: Action) -> None:
        self.nodes[state_id].tested.add(action)

    def get_untested_actions(
        self, state_id: str, actions: list[Action]
    ) -> list[Action]:
        return [a for a in actions if a not in self.nodes[state_id].tested]

    def record_transition(
        self, source: State, action: Action, result: State, cost: float = 1
    ) -> Outcome:
        self.get_or_create_node(source)
        self.get_or_create_node(result)
        self.mark_action_tested(source.id, action)
        outcomes = self.edges.setdefault(source.id, {}).setdefault(action, {})
        outcome = outcomes.setdefault(result.id, Outcome())
        outcome.count += 1
        outcome.cost += cost
        outcome.changes += bool(changed_cells(source, result))
        outcome.progress += int(action.id != 0 and result.level > source.level)
        outcome.failures += result.status == "GAME_OVER"
        return outcome

    def get_neighbors(self, state_id: str) -> list[tuple[Action, str]]:
        """Safe replay edges: one observed outcome, same level, not terminal/reset."""
        source = self.nodes[state_id].state
        result = []
        for action, outcomes in sorted(self.edges.get(state_id, {}).items()):
            if action.id == 0 or len(outcomes) != 1:
                continue
            target = next(iter(outcomes))
            state = self.nodes[target].state
            if (
                target != state_id
                and not state.terminal
                and (state.game, state.level) == (source.game, source.level)
            ):
                result.append((action, target))
        return result

    def paths(self, start: str) -> dict[str, list[tuple[Action, str]]]:
        paths: dict[str, list[tuple[Action, str]]] = {start: []}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for action, target in self.get_neighbors(current):
                if target not in paths:
                    paths[target] = [*paths[current], (action, target)]
                    queue.append(target)
        return paths

    def find_shortest_path(
        self, start: str, goal: str
    ) -> list[tuple[Action, str]] | None:
        return self.paths(start).get(goal)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "nodes": [
                {
                    "id": sid,
                    "state": node.state.payload(),
                    "visits": node.visits,
                    "tested": [asdict(a) for a in sorted(node.tested)],
                    "expansion": node.expansion,
                    "retries": [
                        {"action": asdict(a), "count": n}
                        for a, n in sorted(node.retries.items())
                    ],
                }
                for sid, node in sorted(self.nodes.items())
            ],
            "edges": [
                {"source": src, "action": asdict(a), "target": dst, **asdict(o)}
                for src, actions in sorted(self.edges.items())
                for a, outcomes in sorted(actions.items())
                for dst, o in sorted(outcomes.items())
            ],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> StateGraph:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("version") != 1:
            raise ValueError("Unsupported state graph version")
        graph = cls()
        for record in raw["nodes"]:
            state = normalize(**record["state"])
            if state.id != record["id"]:
                raise ValueError("State digest mismatch")
            node = graph.get_or_create_node(state)
            node.visits = record["visits"]
            node.expansion = record["expansion"]
            node.tested = {Action(**a) for a in record["tested"]}
            node.retries = {
                Action(**r["action"]): r["count"] for r in record["retries"]
            }
        for record in raw["edges"]:
            values = record.copy()
            src, dst = values.pop("source"), values.pop("target")
            action = Action(**values.pop("action"))
            if src not in graph.nodes or dst not in graph.nodes:
                raise ValueError("Transition references a missing state")
            graph.edges.setdefault(src, {}).setdefault(action, {})[dst] = Outcome(
                **values
            )
        return graph
