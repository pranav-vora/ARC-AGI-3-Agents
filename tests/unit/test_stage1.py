"""Deterministic contracts for the Stage 1 explorer."""

import json

import numpy as np
import pytest
import torch
from arcengine import FrameData, FrameDataRaw, GameState

from agents.stage1.agent import Stage1Graph, Stage1NoRepeat, Stage1Random
from agents.stage1.candidates import candidates, click_points
from agents.stage1.explorer import Explorer
from agents.stage1.graph import StateGraph
from agents.stage1.state import Action, changed_cells, normalize


def state(value=0, *, level=0, status="NOT_FINISHED", available=(1, 2)):
    return normalize([[[value, 0], [0, 0]]], "toy", level, status, available)


def test_normalization_and_identity():
    array = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    a = normalize([array], "toy", 0, "NOT_FINISHED", [2, 1, 1])
    b = normalize(
        [torch.tensor(array, dtype=torch.int32)], "toy", 0, "NOT_FINISHED", [1, 2]
    )
    assert a.id == b.id == state(1).id
    array[0, 0] = 9
    assert a.frames[0][0, 0] == 1
    assert a.frames[0].dtype == torch.int64
    assert a.frames[0].device.type == "cpu"
    assert a.frames[0].is_contiguous()
    assert (
        len(
            {
                a.id,
                state(2).id,
                state(1, level=1).id,
                state(1, status="WIN").id,
                state(1, available=(1,)).id,
            }
        )
        == 5
    )
    assert changed_cells(a, state(2)) == [(0, 0, 0)]


def test_all_frames_and_shape_changes():
    a = normalize([[[1]], [[2]]], "toy", 0, "NOT_FINISHED", [1])
    b = normalize([[[1]], [[3, 4]]], "toy", 0, "NOT_FINISHED", [1])
    assert a.id != b.id
    assert changed_cells(a, b) == [(1, 0, 0), (1, 0, 1)]
    empty = normalize([], "toy", 0, "NOT_PLAYED", [])
    assert changed_cells(empty, a) == [(0, 0, 0), (1, 0, 0)]
    with pytest.raises(ValueError):
        normalize([[[0.5]]], "toy", 0, "NOT_FINISHED", [])


def test_scripted_graph_and_roundtrip(tmp_path):
    graph = StateGraph()
    a, b, c = state(0), state(1), state(2)
    right, up, left = Action(1), Action(2), Action(3)
    transitions = [(a, right, b), (a, up, a), (b, right, c), (c, left, b)]
    for _ in range(30):
        for src, action, dst in transitions:
            graph.visit(src)
            graph.record_transition(src, action, dst)
    assert len(graph.nodes) == 3
    assert (
        sum(
            len(outcomes)
            for actions in graph.edges.values()
            for outcomes in actions.values()
        )
        == 4
    )
    assert graph.edges[a.id][up][a.id].count == 30
    assert graph.find_shortest_path(a.id, c.id) == [(right, b.id), (right, c.id)]
    assert graph.find_shortest_path(c.id, a.id) is None
    assert graph.find_shortest_path(a.id, a.id) == []
    assert graph.get_untested_actions(a.id, [right, up, left]) == [left]
    graph.nodes[a.id].expansion = 1
    graph.nodes[a.id].retries[right] = 2
    path = tmp_path / "graph.json"
    graph.save(path)
    assert StateGraph.load(path).to_dict() == graph.to_dict()
    raw = json.loads(path.read_text())
    raw["version"] = 2
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="version"):
        StateGraph.load(path)


def test_ambiguous_terminal_and_cross_level_edges():
    graph = StateGraph()
    a, b, c = state(0), state(1), state(2)
    graph.record_transition(a, Action(1), b)
    graph.record_transition(a, Action(1), c)
    graph.record_transition(a, Action(2), state(0, level=1))
    graph.record_transition(a, Action(3), state(0, status="GAME_OVER"))
    graph.record_transition(a, Action(0), b)
    assert len(graph.edges[a.id][Action(1)]) == 2
    assert graph.get_neighbors(a.id) == []
    assert graph.edges[a.id][Action(2)][state(0, level=1).id].progress == 1


def test_click_candidates_are_legal_cumulative_and_deduplicated():
    s = state(available=(0, 1, 6))
    assert len(click_points(0)) == 64
    assert len(click_points(2)) == 4096
    assert set(click_points(0)) < set(click_points(1)) < set(click_points(2))
    actions = candidates(s, 2)
    assert len(actions) == 4097 == len(set(actions))
    assert Action(0) not in actions
    assert all(0 <= a.x <= 63 and 0 <= a.y <= 63 for a in actions if a.id == 6)
    assert candidates(state(available=())) == []
    assert candidates(state(status="WIN")) == []


def test_noop_memory_and_handshake():
    explorer = Explorer(seed=0)
    s = state()
    explorer.observe(s)
    first = explorer.choose()
    with pytest.raises(RuntimeError, match="Observe"):
        explorer.choose()
    explorer.observe(s)
    second = explorer.choose()
    assert second != first
    explorer.observe(s)
    explorer.observe(s)
    assert explorer.metrics["outcomes"] == 2
    assert explorer.graph.nodes[s.id].visits == 3
    assert explorer.metrics["noops"] == 2


def test_shortest_frontier_recovery():
    e = Explorer()
    a, b, c = state(0, available=(1,)), state(1, available=(1,)), state(2)
    e.graph.record_transition(a, Action(1), b)
    e.graph.record_transition(b, Action(1), c)
    e.observe(a)
    assert e.choose() == Action(1)
    assert e.expected == b.id
    assert e.route == [(Action(1), c.id)]
    e.observe(b)
    assert e.choose() == Action(1)
    e.observe(c)
    assert e.metrics["frontier_recoveries"] == 1
    assert e.choose() in candidates(c)


def test_unexpected_transition_invalidates_replay():
    e = Explorer()
    a, b, c, d = [state(i, available=(1,)) for i in range(4)]
    e.graph.record_transition(a, Action(1), b)
    e.graph.record_transition(b, Action(1), c)
    e.observe(a)
    e.choose()
    e.observe(d)
    assert e.metrics["replay_mismatches"] == 1
    assert not e.route and e.expected is None
    assert e.graph.get_neighbors(a.id) == []
    assert e.choose() == Action(1)  # New experiment at D, not stale B -> C.


def test_click_expansion_preserves_action_history():
    e = Explorer()
    s = state(available=(6,))
    e.observe(s)
    e.graph.nodes[s.id].tested.update(candidates(s))
    action = e.choose()
    assert action in candidates(s, 1) and action not in candidates(s, 0)
    e.observe(s)
    assert e.graph.nodes[s.id].expansion == 1
    assert len(e.graph.nodes[s.id].tested) == 65


@pytest.mark.parametrize("mode", ["norepeat", "graph"])
def test_uncertain_trials_and_resets_are_bounded(mode):
    e = Explorer(mode=mode)
    s = state(available=(1,))
    e.graph.record_transition(s, Action(1), state(1, status="GAME_OVER"))
    e.graph.record_transition(s, Action(1), state(2, status="GAME_OVER"))
    e.observe(s)
    assert e.choose() == Action(1)
    e.observe(s)
    assert e.choose() == Action(1)
    e.observe(s)
    assert e.choose() == Action(0)
    e.observe(s)
    assert e.choose() == Action(0)
    e.observe(s)
    assert e.choose() is None
    assert e.stop_reason == "exhausted"
    assert e.graph.nodes[s.id].retries[Action(1)] == 2
    assert e.resets == 2


def test_level_progress_and_terminal_flush():
    e = Explorer(budget=1)
    e.observe(state(available=(1,)))
    e.choose()
    winner = state(level=1, status="WIN")
    e.observe(winner)
    assert e.choose() is None
    assert e.stop_reason == "win"
    assert e.max_level == 1
    assert e.completed_at == {1: 1}
    assert e.metrics["outcomes"] == 1
    assert not e.route


class ToyEnvironment:
    def __init__(self, win_after=None):
        self.calls = []
        self.win_after = win_after
        self.observation_space = self.frame()

    def frame(self):
        won = self.win_after is not None and len(self.calls) >= self.win_after
        raw = FrameDataRaw(
            game_id="toy",
            state=GameState.WIN if won else GameState.NOT_FINISHED,
            levels_completed=int(won),
            available_actions=[1],
            win_levels=1,
        )
        raw.frame = [np.array([[len(self.calls)]], dtype=np.int8)]
        return raw

    def step(self, action, data=None, reasoning=None):
        self.calls.append((action, data))
        self.observation_space = self.frame()
        return self.observation_space


@pytest.mark.parametrize("cls", [Stage1Graph, Stage1NoRepeat, Stage1Random])
@pytest.mark.parametrize(
    "budget,win_after,expected", [(0, None, 0), (3, None, 3), (3, 2, 2)]
)
def test_agent_exact_budget_and_terminal_recording(cls, budget, win_after, expected):
    env = ToyEnvironment(win_after)
    agent = cls(
        card_id="test",
        game_id="toy",
        agent_name="toy",
        ROOT_URL="",
        record=False,
        arc_env=env,
        budget=budget,
    )
    agent.main()
    assert len(env.calls) == expected == agent.action_counter
    assert agent.explorer.metrics["outcomes"] == expected
    assert agent.explorer.pending is None
    assert len(agent.frames) <= 2
    assert agent.explorer.stop_reason == ("win" if win_after else "budget")


def test_initial_reset_and_request_identifiers():
    from agents.stage1.state import encode

    a = FrameData(game_id="toy", frame=[[[1]]], guid="first", available_actions=[1])
    b = a.model_copy(update={"guid": "second"})
    assert encode(a).id == encode(b).id
    e = Explorer(budget=3)
    e.observe(encode(a))
    assert e.choose() == Action(0)
    assert e.resets == 0
    e.observe(state(status="GAME_OVER"))
    assert e.choose() == Action(0)
    assert e.resets == 1
    e.observe(state(status="GAME_OVER"))
    assert e.choose() == Action(0)
    e.observe(state(status="GAME_OVER"))
    assert e.choose() is None
    assert e.actions == 3


def test_click_adapter_passes_immutable_data():
    env = ToyEnvironment(win_after=1)
    env.observation_space.available_actions = [6]
    agent = Stage1Graph(
        card_id="test",
        game_id="toy",
        agent_name="toy",
        ROOT_URL="",
        record=False,
        arc_env=env,
        budget=1,
    )
    agent.main()
    action, data = env.calls[0]
    assert action.value == 6 and set(data) == {"x", "y"}
    assert 0 <= data["x"] <= 63 and 0 <= data["y"] <= 63


def test_frontier_tie_prefers_observed_progress():
    e = Explorer()
    a, b, c = state(0), state(1), state(2)
    e.graph.record_transition(a, Action(1), b)
    e.graph.record_transition(a, Action(2), c)
    e.graph.record_transition(b, Action(1), b)
    e.graph.record_transition(c, Action(1), state(3, level=1))
    e.observe(a)
    assert e.choose() == Action(2)
    assert e.expected == c.id


def test_multiple_routes_and_terminal_only_path():
    graph = StateGraph()
    a, b, c, d = [state(i) for i in range(4)]
    for _ in range(25):
        for src, action, dst in [
            (a, Action(1), b),
            (b, Action(1), c),
            (a, Action(2), d),
            (d, Action(1), c),
            (c, Action(2), a),
        ]:
            graph.record_transition(src, action, dst)
    assert graph.find_shortest_path(a.id, c.id) == [
        (Action(1), b.id),
        (Action(1), c.id),
    ]
    graph.record_transition(a, Action(3), c)
    assert graph.find_shortest_path(a.id, c.id) == [(Action(3), c.id)]
    terminal = state(9, status="GAME_OVER")
    graph.record_transition(a, Action(4), terminal)
    graph.record_transition(terminal, Action(1), state(10))
    assert graph.find_shortest_path(a.id, state(10).id) is None


def test_norepeat_recovers_locally_without_bfs():
    e = Explorer(mode="norepeat")
    a, b = state(0, available=(1,)), state(1)
    e.graph.record_transition(a, Action(1), b)
    e.observe(a)
    assert e.choose() == Action(1)
    assert not e.route and e.expected is None
    assert e.metrics["frontier_attempts"] == 0


def test_progress_ranking_and_seed_reproducibility():
    e = Explorer(seed=123)
    a, b = state(0), state(1)
    e.observe(a)
    action = e.choose()
    e.observe(state(level=1))
    e.observe(b)
    assert e.choose() == action  # Confirmed progress transfers within level zero.
    sequences = []
    for _ in range(2):
        random_explorer = Explorer(mode="random", seed=42, budget=10)
        random_explorer.observe(a)
        sequence = []
        for _ in range(10):
            sequence.append(random_explorer.choose())
            random_explorer.observe(a)
        sequences.append(sequence)
    assert sequences[0] == sequences[1]


@pytest.mark.parametrize("seed", range(5))
def test_ablation_matches_local_decisions_until_frontier_recovery(seed):
    graph, baseline = Explorer(seed=seed), Explorer(mode="norepeat", seed=seed)
    for step in range(20):
        observation = state(step, available=(1, 2, 3, 6))
        graph.observe(observation)
        baseline.observe(observation)
        assert graph.choose() == baseline.choose()
    graph.observe(state(20))
    baseline.observe(state(20))
    assert (
        graph.summary()["unexplained_repeats"]
        == baseline.summary()["unexplained_repeats"]
        == 0
    )


def test_replayed_actions_have_auditable_reasons():
    records = []
    e = Explorer(emit=records.append)
    a, b, c = state(0, available=(1,)), state(1, available=(1,)), state(2)
    e.graph.record_transition(a, Action(1), b)
    e.graph.record_transition(b, Action(1), c)
    e.observe(a)
    e.choose()
    e.observe(b)
    e.choose()
    e.observe(c)
    transitions = [r for r in records if r["event"] == "transition"]
    assert [r["decision_reason"] for r in transitions] == [
        "frontier_replay",
        "frontier_replay",
    ]
    assert all(r["previously_tested"] for r in transitions)
    assert e.summary()["unexplained_repeats"] == 0


@pytest.mark.parametrize("distance", [1, 2, 5, 8])
def test_every_scripted_reachable_frontier_is_recovered(distance):
    e = Explorer()
    route = [state(i, available=(1,)) for i in range(distance + 1)]
    for source, target in zip(route, route[1:]):
        e.graph.record_transition(source, Action(1), target)
    e.observe(route[0])
    for target in route[1:]:
        assert e.choose() == Action(1)
        assert e.decision_reason == "frontier_replay"
        e.observe(target)
    assert e.metrics["frontier_attempts"] == e.metrics["frontier_recoveries"] == 1
    assert e.summary()["unexplained_repeats"] == 0
    assert e.expected is None and not e.route


def test_budget_exhaustion_during_frontier_replay_is_clean():
    e = Explorer(budget=2)
    route = [state(i, available=(1,)) for i in range(6)]
    for source, target in zip(route, route[1:]):
        e.graph.record_transition(source, Action(1), target)
    e.observe(route[0])
    for target in route[1:3]:
        e.choose()
        e.observe(target)
    assert e.choose() is None and e.stop_reason == "budget"
    assert e.pending is None and e.actions == 2
    assert e.metrics["frontier_recoveries"] == 0


def test_artifact_audit_detects_corrupt_scores_and_missing_runs(tmp_path):
    from evaluation.audit_results import audit

    records = []
    e = Explorer(budget=1, emit=records.append)
    e.observe(state(available=(1,)))
    e.choose()
    e.observe(state(1, level=1, status="WIN"))
    e.choose()
    manifest = {
        "games": ["toy"],
        "agents": ["graph"],
        "seeds": [0],
        "budget": 1,
        "timeout_seconds": 10,
        "policy_version": 2,
    }
    row = {
        "game": "toy",
        "agent": "graph",
        "seed": 0,
        "budget": 1,
        "error": None,
        "runtime_seconds": 1,
        "peak_process_memory_bytes": 1024,
        "official_score": 1,
        **e.summary(),
    }
    folder = tmp_path / "toy-graph-0"
    folder.mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "results.json").write_text(json.dumps([row]))
    (folder / "scorecard.json").write_text(
        json.dumps({"score": 1, "total_actions": 1, "total_levels_completed": 1})
    )
    (folder / "actions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    assert audit(tmp_path)["passed"]
    # A replay label alone must not justify an edge absent from the trace history.
    records[-1]["decision_reason"] = "frontier_replay"
    (folder / "actions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    assert any("replay does not reference" in s for s in audit(tmp_path)["issues"])
    records[-1]["decision_reason"] = "untested_action"
    (folder / "actions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    (folder / "scorecard.json").write_text(
        json.dumps({"score": 1, "total_actions": 1, "total_levels_completed": 2})
    )
    assert any("official completed-level" in s for s in audit(tmp_path)["issues"])
    (folder / "scorecard.json").write_text(json.dumps({"score": 2}))
    result = audit(tmp_path)
    assert not result["passed"] and any("official score" in s for s in result["issues"])
    manifest["seeds"] = [0, 1]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert any("coverage" in s for s in audit(tmp_path)["issues"])


def test_environment_error_is_reported_without_retrying():
    env = ToyEnvironment()
    env.step = lambda *args, **kwargs: None
    agent = Stage1Graph(
        card_id="test",
        game_id="toy",
        agent_name="toy",
        ROOT_URL="",
        record=False,
        arc_env=env,
        budget=5,
    )
    with pytest.raises(ValueError, match="None frame"):
        agent.main()
    assert agent.action_counter == 1
    assert agent.explorer.stop_reason == "error"
    assert agent.explorer.metrics["outcomes"] == 0


def test_report_uses_paired_official_scores(tmp_path):
    from evaluation.analyze_results import report
    from evaluation.run_benchmark import peak_memory_bytes

    rows = []
    for mode, score in [("graph", 0), ("norepeat", 1)]:
        rows.append(
            {
                "game": "toy",
                "seed": 0,
                "agent": mode,
                "error": None,
                "official_score": score,
                "levels_completed": score,
                "actions": 10,
                "runtime_seconds": 1,
                "peak_process_memory_bytes": 1024,
                "unique_states": 3,
                "repeated_state_rate": 0.5,
                "repeated_state_action_rate": 0.1,
                "noop_rate": 0.2,
            }
        )
    (tmp_path / "results.json").write_text(json.dumps(rows))
    text = report(tmp_path)
    assert "-1.0000 across 1 runs" in text
    assert "no-go" in text
    assert "n/a" in text
    assert peak_memory_bytes() > 0
    rows[0]["official_score"] = 2
    (tmp_path / "results.json").write_text(json.dumps(rows))
    text = report(tmp_path)
    assert "1.0000 across 1 runs" in text
    assert "no-go" in text and "No graph run invoked frontier recovery" in text


@pytest.mark.parametrize("seed", range(5))
def test_explorer_solves_branching_ring_with_noops(seed):
    """End-to-end synthetic puzzle: loop, reversals, dead actions, one winning exit."""
    e = Explorer(seed=seed, budget=100)
    position = 0
    e.observe(state(position, available=(1, 2, 3)))
    while (action := e.choose()) is not None:
        assert action.id != 0  # The connected graph always has a reachable frontier.
        if position == 8 and action.id == 3:
            e.observe(state(9, level=1, status="WIN"))
        else:
            position = (position + {1: 1, 2: -1, 3: 0}[action.id]) % 9
            e.observe(state(position, available=(1, 2, 3)))
    assert e.stop_reason == "win"
    assert e.actions < 100
    assert e.summary()["levels_completed"] == 1
    assert e.summary()["unexplained_repeats"] == 0
    for actions in e.graph.edges.values():
        for action, outcomes in actions.items():
            if action.id == 3:
                assert sum(outcome.count for outcome in outcomes.values()) == 1
