"""Summarize measured performance without substituting diagnostic metrics for score."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def report(folder: Path) -> str:
    rows = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors = [r for r in rows if r.get("error")]
    for row in rows:
        if not row.get("error"):
            groups[row["agent"]].append(row)
    lines = [
        "# Stage 1 benchmark results",
        "",
        "Official toolkit scores are reported as returned, without rescaling. "
        "Each run uses a fresh local environment and process. Runtime includes environment "
        "construction and agent execution; peak resident memory includes Python and native libraries.",
        "",
        "| Agent | Runs | Mean official score | Total levels | Mean actions | Mean seconds | Peak MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, runs in sorted(groups.items()):
        scores = [
            r["official_score"] for r in runs if r.get("official_score") is not None
        ]
        score = f"{mean(scores):.4f}" if scores else "unavailable"
        lines.append(
            f"| {mode} | {len(runs)} | {score} | {sum(r['levels_completed'] for r in runs)} | "
            f"{mean(r['actions'] for r in runs):.1f} | {mean(r['runtime_seconds'] for r in runs):.2f} | "
            f"{max(r['peak_process_memory_bytes'] for r in runs) / 1048576:.1f} |"
        )
    lines.extend(["", f"Failed runs: {len(errors)}.", ""])
    lines.extend(
        [
            "| Agent | Runs with completion | Total actions / completed level | Mean first completion action |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for mode, runs in sorted(groups.items()):
        levels = sum(r["levels_completed"] for r in runs)
        efficiency = (
            f"{sum(r['actions'] for r in runs) / levels:.1f}" if levels else "n/a"
        )
        first = [
            min(int(v) for v in r.get("completion_action_indices", {}).values())
            for r in runs
            if r.get("completion_action_indices")
        ]
        first_action = f"{mean(first):.1f}" if first else "n/a"
        lines.append(
            f"| {mode} | {sum(r['levels_completed'] > 0 for r in runs)} | {efficiency} | {first_action} |"
        )
    lines.extend(
        [
            "",
            "Aggregate actions per completed level includes unsuccessful runs; n/a means no completion. "
            "Per-run efficiency and completion indices are preserved in results.json and summary.csv.",
            "",
        ]
    )
    paired = {(r["game"], r["seed"]): r for r in groups.get("norepeat", [])}
    comparisons = [
        (r, paired[(r["game"], r["seed"])])
        for r in groups.get("graph", [])
        if (r["game"], r["seed"]) in paired
    ]
    scored = [
        (a, b)
        for a, b in comparisons
        if a.get("official_score") is not None and b.get("official_score") is not None
    ]
    if scored:
        delta = mean(a["official_score"] - b["official_score"] for a, b in scored)
        wins = sum(a["official_score"] > b["official_score"] for a, b in scored)
        losses = sum(a["official_score"] < b["official_score"] for a, b in scored)
        by_game: dict[str, dict[int, float]] = defaultdict(dict)
        for a, b in scored:
            by_game[a["game"]][a["seed"]] = a["official_score"] - b["official_score"]
        repeatable = [
            g
            for g, deltas in by_game.items()
            if len(deltas) >= 3 and all(d > 0 for d in deltas.values())
        ]
        graph_attempts = sum(
            r.get("frontier_attempts", 0) for r in groups.get("graph", [])
        )
        lines.append(
            f"Paired graph minus no-repeat mean official score: {delta:.4f} across {len(scored)} runs."
        )
        lines.append(
            f"Graph score wins/ties/losses: {wins}/{len(scored) - wins - losses}/{losses}."
        )
        if not graph_attempts:
            lines.append(
                "Stage 2 assessment: no-go. No graph run invoked frontier recovery. "
                "Score differences therefore do not demonstrate a benefit from BFS; "
                "investigate candidate exhaustion and exact-state fragmentation first."
            )
        elif delta > 0 and not errors and repeatable:
            lines.append(
                "Stage 2 assessment: repeated score improvement observed in "
                + ", ".join(sorted(repeatable))
                + "; inspect completion efficiency and regressions before adding learning."
            )
        else:
            lines.append(
                "Stage 2 assessment: no-go. Repeatable improvement across at least three paired seeds is unproven. "
                "Diagnose exact-state fragmentation, click coverage, "
                "and frontier availability before adding learned representations."
            )
    else:
        lines.append(
            "Stage 2 assessment: inconclusive; a complete paired comparison is required."
        )
    lines.extend(
        [
            "",
            "## Exploration diagnostics",
            "",
            "| Agent | Mean unique states | Repeat states | Repeat actions | No-op rate | Frontier success |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode, runs in sorted(groups.items()):
        attempts = sum(r.get("frontier_attempts", 0) for r in runs)
        recovered = sum(r.get("frontier_recoveries", 0) for r in runs)
        rate = f"{recovered / attempts:.3f}" if attempts else "n/a"
        lines.append(
            f"| {mode} | {mean(r['unique_states'] for r in runs):.1f} | "
            f"{mean(r['repeated_state_rate'] for r in runs):.3f} | "
            f"{mean(r['repeated_state_action_rate'] for r in runs):.3f} | "
            f"{mean(r['noop_rate'] for r in runs):.3f} | {rate} |"
        )
    lines.extend(
        [
            "",
            "Observation hashes retain UI, animation frames, and all visible mechanics. "
            "Different hidden states may still share observations, and UI changes can prevent revisits. "
            "The graph does not establish underlying Markov state identity.",
            "",
        ]
    )
    text = "\n".join(lines)
    (folder / "report.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(report(parser.parse_args().folder))
