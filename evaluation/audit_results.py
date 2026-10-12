"""Check complete evaluation artifacts and classify incomplete runs from evidence."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any


def audit(folder: Path, memory_mib: int = 1024) -> dict[str, Any]:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    expected = set(
        itertools.product(manifest["games"], manifest["agents"], manifest["seeds"])
    )
    actual = [(r["game"], r["agent"], r["seed"]) for r in rows]
    issues: list[str] = []
    if set(actual) != expected or len(actual) != len(expected):
        issues.append("Scheduled game/agent/seed coverage is incomplete or duplicated")
    inspected = []
    for row in rows:
        name = "{game}-{agent}-{seed}".format(**row)
        prefix = folder / name

        def check(condition: bool, message: str) -> None:
            if not condition:
                issues.append(f"{name}: {message}")

        check(not row.get("error"), "run did not terminate cleanly")
        if row.get("error"):
            continue
        check(row["budget"] == manifest["budget"], "wrong action budget")
        check(row["actions"] <= row["budget"], "action budget exceeded")
        check(row["actions"] == row["outcomes"], "submitted action missing an outcome")
        check(
            row["runtime_seconds"] <= manifest["timeout_seconds"],
            "runtime budget exceeded",
        )
        check(
            row["peak_process_memory_bytes"] <= memory_mib * 1048576,
            "memory audit limit exceeded",
        )
        check(row["recovery_resets"] <= 2, "too many recovery resets")
        scorecard = json.loads((prefix / "scorecard.json").read_text(encoding="utf-8"))
        check(
            scorecard["score"] == row["official_score"],
            "official score differs from saved scorecard",
        )
        check(
            scorecard.get("total_actions") == row["actions"],
            "official action count differs from result",
        )
        check(
            scorecard.get("total_levels_completed") == row["levels_completed"],
            "official completed-level count differs from result",
        )
        check("api_key" not in scorecard, "scorecard includes an API key field")
        seen_states: set[str] = set()
        seen_actions: set[tuple[Any, ...]] = set()
        seen_edges: set[tuple[Any, ...]] = set()
        count = repeats = noops = repeated_states = unexplained = max_level = 0
        recovery_resets = 0
        previous_target = None
        last_status = None
        replay_remaining = 0
        retry = reset = mismatch = False
        mismatch_expected = None
        events: Counter[str] = Counter()
        with (prefix / "actions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                kind = record["event"]
                events[kind] += 1
                if kind == "replay_started":
                    # frontier_selected immediately precedes replay_started.
                    pass
                elif kind == "frontier_selected":
                    replay_remaining = record["distance"]
                elif kind == "uncertain_retry":
                    retry = True
                elif kind == "reset":
                    reset = True
                    recovery_resets += bool(record["recovery"])
                elif kind == "unexpected_transition":
                    mismatch = True
                    mismatch_expected = record["expected"]
                if kind != "transition":
                    continue
                count += 1
                source, target = record["source"], record["target"]
                action = record["action"]
                key = (source, action["id"], action["x"], action["y"])
                if previous_target is not None:
                    check(
                        source == previous_target,
                        f"discontinuous observation history at step {count}",
                    )
                seen_states.add(source)
                repeated = key in seen_actions
                new_state = target not in seen_states
                check(record["step"] == count, "duplicated or missing transition step")
                check(
                    record["previously_tested"] == repeated,
                    "incorrect tested-action flag",
                )
                check(record["new_state"] == new_state, "incorrect state novelty flag")
                check(
                    record["changed_cells"] == len(record["changed_coordinates"]),
                    "changed-cell count mismatch",
                )
                check(action["id"] in range(8), "unknown action ID")
                if action["id"] == 6:
                    check(
                        all(
                            isinstance(action[c], int) and 0 <= action[c] <= 63
                            for c in ("x", "y")
                        ),
                        "illegal click coordinate",
                    )
                else:
                    check(
                        action["x"] is None and action["y"] is None,
                        "simple action carries coordinates",
                    )
                reason = record.get("decision_reason")
                if manifest.get("policy_version", 1) >= 2:
                    check(reason is not None, "missing explicit decision reason")
                if reason == "frontier_replay" or (
                    reason is None and replay_remaining > 0
                ):
                    known = {e[4] for e in seen_edges if e[:4] == key}
                    check(
                        len(known) == 1 and source not in known and action["id"] != 0,
                        "replay does not reference an observed unambiguous changing edge",
                    )
                    check(
                        (mismatch_expected if mismatch else target) in known,
                        "replay destination differs without a correct mismatch event",
                    )
                if repeated:
                    justified = (
                        (
                            row["agent"] == "random"
                            and reason in (None, "random_baseline")
                        )
                        or (
                            reset
                            and action["id"] == 0
                            and reason in (None, "initial_reset", "recovery_reset")
                        )
                        or (retry and reason in (None, "uncertain_retry"))
                        or (
                            replay_remaining > 0 and reason in (None, "frontier_replay")
                        )
                        or (
                            row["agent"] == "norepeat"
                            and reason in (None, "local_recovery")
                            and any(e[:4] == key and e[4] != source for e in seen_edges)
                        )
                    )
                    unexplained += not justified
                repeats += repeated
                repeated_states += not new_state
                noops += record["changed_cells"] == 0
                max_level = max(max_level, record["levels_completed"])
                seen_states.add(target)
                seen_actions.add(key)
                seen_edges.add((*key, target))
                previous_target, last_status = target, record["status"]
                replay_remaining = 0 if mismatch else max(0, replay_remaining - 1)
                retry = reset = mismatch = False
                mismatch_expected = None
        check(count == row["actions"], "trace action count differs from result")
        check(
            repeats == row["repeated_state_actions"], "repeat count differs from result"
        )
        check(noops == row["noops"], "no-op count differs from result")
        check(
            repeated_states == row["repeated_states"],
            "repeated-state count differs from result",
        )
        check(
            max_level == row["levels_completed"], "level completion differs from trace"
        )
        check(
            recovery_resets == row["recovery_resets"], "reset count differs from trace"
        )
        check(unexplained == 0, "unexplained repeated state-action pairs")
        check(
            events["frontier_selected"] == row.get("frontier_attempts", 0),
            "frontier attempt count differs from trace",
        )
        check(
            events["frontier_reached"] == row.get("frontier_recoveries", 0),
            "frontier recovery count differs from trace",
        )
        if count:
            check(
                len(seen_states) == row["unique_states"],
                "unique-state count differs from trace",
            )
        if row["stop_reason"] == "budget":
            check(
                count == manifest["budget"],
                "budget termination occurred at the wrong action",
            )
        elif row["stop_reason"] == "win":
            check(last_status == "WIN", "win termination lacks a winning observation")
        else:
            check(row["stop_reason"] == "exhausted", "unrecognized stop reason")
        category = "completed"
        if row["stop_reason"] != "win":
            if row["noop_rate"] >= 0.75:
                category = "action_generation"
            elif row["unique_states"] / max(count + 1, 1) >= 0.85:
                category = "state_encoding"
            elif last_status == "GAME_OVER":
                category = "environment_specific_mechanic"
            elif (
                row.get("frontier_attempts", 0)
                and events["frontier_reached"] == events["frontier_selected"]
            ):
                category = "lack_of_goal_knowledge"
            else:
                category = "frontier_selection"
        inspected.append(
            {
                "run": name,
                "primary_failure_hypothesis": category,
                "last_status": last_status,
                "unexplained_repeats": unexplained,
                "trace_actions": count,
                "frontier_recoveries": events["frontier_reached"],
            }
        )
    result = {
        "scheduled_runs": len(expected),
        "inspected_runs": len(inspected),
        "passed": not issues,
        "issues": issues,
        "memory_audit_limit_mib": memory_mib,
        "max_runtime_seconds": max(
            (r.get("runtime_seconds", 0) for r in rows), default=0
        ),
        "peak_process_memory_mib": max(
            (r.get("peak_process_memory_bytes", 0) for r in rows), default=0
        )
        / 1048576,
        "failure_classification_note": "Hypotheses from traces, not proof of hidden mechanics; see representative source/observation inspection in FAILURE_MODES.md.",
        "runs": inspected,
    }
    (folder / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--memory-mib", type=int, default=1024)
    args = parser.parse_args()
    result = audit(args.folder, args.memory_mib)
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
