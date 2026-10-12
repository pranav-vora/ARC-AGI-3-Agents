"""Bounded exploration, with explicit observation/action handshakes."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Callable

from .candidates import can_expand, candidates
from .graph import StateGraph
from .state import Action, State, changed_cells


class Explorer:
    def __init__(
        self,
        mode: str = "graph",
        seed: int = 0,
        budget: int = 1000,
        emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if mode not in {"random", "norepeat", "graph"} or budget < 0:
            raise ValueError("Invalid exploration mode or negative budget")
        self.mode = mode
        self.rng = random.Random(seed)
        self.budget = budget
        self.graph = StateGraph()
        self.current: State | None = None
        self.pending: tuple[State, Action] | None = None
        self.route: list[tuple[Action, str]] = []
        self.expected: str | None = None
        self.actions = 0
        self.resets = 0
        self.started = False
        self.stop_reason: str | None = None
        self.decision_reason = "initial"
        self.emit = emit or (lambda record: None)
        self.metrics: dict[str, int] = defaultdict(int)
        # Per-game, per-level, exact-action evidence: count, progress, novel, failure.
        self.evidence: dict[tuple[str, int, Action], list[int]] = {}
        self.completed_at: dict[int, int] = {}
        self.max_level = 0

    def event(self, event: str, **values: Any) -> None:
        self.emit({"event": event, "step": self.actions, **values})

    def observe(self, state: State) -> None:
        if self.pending is None:
            if self.current is None or self.current.id != state.id:
                self.graph.visit(state)
            self.current = state
            self.max_level = max(self.max_level, state.level)
            return
        source, action = self.pending
        self.pending = None
        new = state.id not in self.graph.nodes
        repeated = action in self.graph.nodes[source.id].tested
        explained = self.decision_reason in {
            "random_baseline",
            "frontier_replay",
            "uncertain_retry",
            "local_recovery",
            "recovery_reset",
            "initial_reset",
        }
        self.metrics["unexplained_repeats"] += repeated and not explained
        changes = changed_cells(source, state)
        self.graph.record_transition(source, action, state)
        self.graph.visit(state)
        self.metrics["outcomes"] += 1
        self.metrics["repeated_states"] += not new
        self.metrics["repeated_state_actions"] += repeated
        self.metrics["noops"] += not changes
        evidence = self.evidence.setdefault(
            (source.game, source.level, action), [0] * 4
        )
        evidence[0] += 1
        evidence[1] += action.id != 0 and state.level > source.level
        evidence[2] += new
        evidence[3] += state.status == "GAME_OVER"
        if action.id != 0:
            for level in range(self.max_level + 1, state.level + 1):
                self.completed_at[level] = self.actions
        self.max_level = max(self.max_level, state.level)
        if self.expected is not None:
            if state.id != self.expected:
                self.metrics["replay_mismatches"] += 1
                self.event(
                    "unexpected_transition", expected=self.expected, actual=state.id
                )
                self.route.clear()
            elif not self.route:
                self.metrics["frontier_recoveries"] += 1
                self.event("frontier_reached", state=state.id)
            self.expected = None
        if action.id == 0 or state.level != source.level or state.terminal:
            self.route.clear()
        self.current = state
        self.event(
            "transition",
            game=state.game,
            source=source.id,
            target=state.id,
            action=asdict(action),
            new_state=new,
            changed_cells=len(changes),
            changed_coordinates=changes,
            previously_tested=repeated,
            decision_reason=self.decision_reason,
            levels_completed=state.level,
            status=state.status,
            level_complete=action.id != 0 and state.level > source.level,
            frontier_actions_remaining=len(self.untested(state)),
        )
        if repeated and not changes:
            self.event("repeated_noop", state=source.id, action=asdict(action))

    def untested(self, state: State) -> list[Action]:
        node = self.graph.nodes[state.id]
        return self.graph.get_untested_actions(
            state.id, candidates(state, node.expansion)
        )

    def expand(self, state: State) -> bool:
        node = self.graph.nodes[state.id]
        if not can_expand(state, node.expansion):
            return False
        node.expansion += 1
        self.event("candidates_expanded", state=state.id, stage=node.expansion)
        return True

    def rank(self, state: State, actions: list[Action]) -> Action:
        def priority(action: Action) -> tuple[float, float, float, float]:
            count, progress, novel, failure = self.evidence.get(
                (state.game, state.level, action), [0, 0, 0, 0]
            )
            denominator = max(count, 1)
            return (
                -progress / denominator,
                failure / denominator,
                -novel / denominator,
                self.rng.random(),
            )

        return min(actions, key=priority)

    def frontier(self, state: State) -> list[tuple[Action, str]] | None:
        paths = self.graph.paths(state.id)
        for broaden in (False, True):
            choices = []
            for sid, path in paths.items():
                if not path:
                    continue
                target = self.graph.nodes[sid].state
                untested = self.untested(target)
                if broaden and not untested and self.expand(target):
                    untested = self.untested(target)
                if not untested:
                    continue
                outcomes = [
                    o
                    for values in self.graph.edges.get(sid, {}).values()
                    for o in values.values()
                ]
                progress = sum(o.progress for o in outcomes)
                failure = sum(o.failures for o in outcomes) / max(
                    1, sum(o.count for o in outcomes)
                )
                choices.append(
                    (len(path), -progress, failure, -len(untested), sid, path)
                )
            if choices:
                return min(choices)[-1]
        return None

    def _reset_or_stop(self, state: State) -> Action | None:
        initial = state.status == "NOT_PLAYED" and not self.started
        if initial or self.resets < 2:
            if not initial:
                self.resets += 1
            self.route.clear()
            self.event("reset", recovery=not initial)
            self.decision_reason = "initial_reset" if initial else "recovery_reset"
            return Action(0)
        self.stop_reason = "exhausted"
        self.event("stalled", state=state.id)
        return None

    def choose(self) -> Action | None:
        if self.pending is not None:
            raise RuntimeError(
                "Observe the previous action's result before choosing again"
            )
        state = self.current
        if state is None:
            raise RuntimeError("An initial observation is required")
        if self.stop_reason:
            return None
        if state.status == "WIN":
            self.stop_reason = "win"
            return None
        if self.actions >= self.budget:
            self.stop_reason = "budget"
            return None
        action = self._choose(state)
        if action is not None:
            self.started = True
            self.pending = (state, action)
            self.actions += 1
        return action

    def _choose(self, state: State) -> Action | None:
        if state.status in {"NOT_PLAYED", "GAME_OVER"}:
            return self._reset_or_stop(state)
        if self.route:
            action, target = self.route.pop(0)
            if (action, target) in self.graph.get_neighbors(state.id):
                self.expected = target
                self.decision_reason = "frontier_replay"
                return action
            self.route.clear()
        node = self.graph.nodes[state.id]
        actions = self.untested(state)
        if self.mode == "random":
            if not actions:
                self.expand(state)
            legal = candidates(state, node.expansion)
            self.decision_reason = "random_baseline"
            return self.rng.choice(legal) if legal else self._reset_or_stop(state)
        if actions:
            self.decision_reason = "untested_action"
            # Match the baseline's priorities and RNG consumption: frontier recovery
            # should be the experimental difference, not a different local policy.
            return self.rank(state, actions)
        if self.mode == "graph":
            path = self.frontier(state)
            if path:
                self.route = path
                self.metrics["frontier_attempts"] += 1
                self.event("frontier_selected", target=path[-1][1], distance=len(path))
                self.event("replay_started", target=path[-1][1])
                action, self.expected = self.route.pop(0)
                self.decision_reason = "frontier_replay"
                return action
        while self.expand(state):
            actions = self.untested(state)
            if actions:
                self.decision_reason = "expanded_untested_action"
                return self.rank(state, actions)
        for action, outcomes in sorted(self.graph.edges.get(state.id, {}).items()):
            if action.id and len(outcomes) > 1 and node.retries.get(action, 0) < 2:
                node.retries[action] = node.retries.get(action, 0) + 1
                self.event("uncertain_retry", action=asdict(action))
                self.decision_reason = "uncertain_retry"
                return action
        if self.mode == "norepeat":
            # Local fallback can wander along known useful edges, without graph routing.
            moves = [a for a, _ in self.graph.get_neighbors(state.id)]
            if moves:
                self.decision_reason = "local_recovery"
                return self.rng.choice(moves)
        return self._reset_or_stop(state)

    def summary(self) -> dict[str, Any]:
        n = max(self.metrics["outcomes"], 1)
        return {
            "actions": self.actions,
            "levels_completed": self.max_level,
            "actions_per_completed_level": self.actions / self.max_level
            if self.max_level
            else None,
            "completion_action_indices": self.completed_at,
            "unique_states": len(self.graph.nodes),
            "repeated_state_rate": self.metrics["repeated_states"] / n,
            "repeated_state_action_rate": self.metrics["repeated_state_actions"] / n,
            "noop_rate": self.metrics["noops"] / n,
            "frontier_recovery_success": self.metrics["frontier_recoveries"]
            / self.metrics["frontier_attempts"]
            if self.metrics["frontier_attempts"]
            else None,
            "recovery_resets": self.resets,
            "unexplained_repeats": self.metrics["unexplained_repeats"],
            "stop_reason": self.stop_reason,
            **self.metrics,
        }
