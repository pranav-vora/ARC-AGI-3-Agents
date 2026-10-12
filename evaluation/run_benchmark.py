"""Sequential, process-isolated comparisons using official ARC environments."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

MODES = ("random", "norepeat", "graph")


def peak_memory_bytes() -> int:
    """OS high-water resident memory, including native PyTorch allocations."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("faults", wintypes.DWORD),
                ("peak_working_set", ctypes.c_size_t),
                ("working_set", ctypes.c_size_t),
                ("peak_paged", ctypes.c_size_t),
                ("paged", ctypes.c_size_t),
                ("peak_nonpaged", ctypes.c_size_t),
                ("nonpaged", ctypes.c_size_t),
                ("pagefile", ctypes.c_size_t),
                ("peak_pagefile", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.peak_working_set)
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def worker(args: argparse.Namespace) -> None:
    import torch
    from arc_agi import Arcade, OperationMode  # type: ignore[import-untyped]

    from agents import AVAILABLE_AGENTS
    from agents.stage1.agent import Stage1Agent

    torch.set_num_threads(1)
    logger = logging.getLogger("stage1.benchmark")
    logger.setLevel(logging.WARNING)
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE, arc_api_key="stage1-local", logger=logger
    )
    card = arc.open_scorecard(tags=["stage1", args.mode, f"seed-{args.seed}"])
    result: dict[str, Any] = {
        "game": args.worker,
        "agent": args.mode,
        "seed": args.seed,
        "budget": args.budget,
        "error": None,
    }
    start = time.perf_counter()
    agent = None
    try:
        env = arc.make(args.worker, seed=args.seed, scorecard_id=card)
        if env is None or env.observation_space is None:
            raise RuntimeError("Official local environment could not be initialized")
        with (args.output / "actions.jsonl").open("w", encoding="utf-8") as stream:

            def emit(record: dict[str, Any]) -> None:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")

            agent_class = cast(
                type[Stage1Agent], AVAILABLE_AGENTS[f"stage1{args.mode}"]
            )
            agent = agent_class(
                card_id=card,
                game_id=args.worker,
                agent_name=f"stage1{args.mode}",
                ROOT_URL="",
                record=False,
                arc_env=env,
                seed=args.seed,
                budget=args.budget,
                emit=emit,
            )
            agent.main()
        if args.save_graphs:
            agent.explorer.graph.save(args.output / "graph.json")
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if agent is not None:
            result.update(agent.explorer.summary())
            if result.get("unexplained_repeats", 0):
                result["error"] = (
                    "Controller repeated a tested action without a recorded recovery reason"
                )
        result["runtime_seconds"] = time.perf_counter() - start
        result["peak_process_memory_bytes"] = peak_memory_bytes()
        scorecard = arc.close_scorecard(card)
        result["official_score"] = None
        if scorecard is not None:
            write_json(
                args.output / "scorecard.json",
                scorecard.model_dump(mode="json", exclude={"api_key"}),
            )
            result["official_score"] = scorecard.score
        write_json(args.output / "result.json", result)


def discover_and_prepare(offline: bool) -> list[str]:
    from arc_agi import Arcade, OperationMode

    logger = logging.getLogger("stage1.prepare")
    logger.setLevel(logging.WARNING)
    # NORMAL is the toolkit default and can otherwise be overridden by .env.
    os.environ["OPERATION_MODE"] = "offline" if offline else "normal"
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE if offline else OperationMode.NORMAL,
        logger=logger,
        **({"arc_api_key": "stage1-local"} if offline else {}),
    )
    return sorted(g.game_id for g in arc.get_environments())


def prepare_games(games: list[str]) -> None:
    from arc_agi import Arcade

    os.environ["OPERATION_MODE"] = "normal"
    logger = logging.getLogger("stage1.prepare")
    logger.setLevel(logging.WARNING)
    arc = Arcade(logger=logger)
    card = arc.open_scorecard(tags=["stage1-download"])
    try:
        for game in games:
            if arc.make(game, scorecard_id=card) is None:
                raise RuntimeError(f"Could not download public environment {game}")
    finally:
        arc.close_scorecard(card)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--games", nargs="+", help="Game IDs or unique prefixes")
    parser.add_argument("--agents", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--list-games", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("evaluation/runs/latest"))
    parser.add_argument("--save-graphs", action="store_true")
    parser.add_argument(
        "--timeout", type=int, default=300, help="Seconds allowed per game/agent/seed"
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.budget < 0 or args.timeout <= 0:
        parser.error("Budget must be nonnegative and timeout positive")
    if args.worker:
        worker(args)
        return
    games = discover_and_prepare(args.offline)
    if args.list_games:
        print("\n".join(games))
        return
    if args.games:
        selected = []
        for prefix in args.games:
            matches = [g for g in games if g == prefix or g.startswith(prefix + "-")]
            if len(matches) != 1:
                parser.error(f"Game prefix {prefix!r} matches {len(matches)} games")
            selected.extend(matches)
        games = sorted(set(selected))
    if not games:
        parser.error(
            "No public environments available; configure toolkit access or download environments first"
        )
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; choose a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.offline:
        prepare_games(games)
    from importlib.metadata import version

    write_json(
        args.output / "manifest.json",
        {
            "games": games,
            "agents": args.agents,
            "seeds": args.seeds,
            "budget": args.budget,
            "timeout_seconds": args.timeout,
            "python": platform.python_version(),
            "packages": {p: version(p) for p in ("torch", "arc-agi", "arcengine")},
            "execution": "sequential isolated offline processes",
            "policy_version": 2,
            "local_ranking": "matched progress/failure/novelty priorities for graph and norepeat",
            "torch_threads": 1,
        },
    )
    rows = []
    total = len(games) * len(args.agents) * len(args.seeds)
    for game in games:
        for seed in args.seeds:
            for mode in args.agents:
                folder = args.output / f"{game}-{mode}-{seed}"
                folder.mkdir()
                command = [
                    sys.executable,
                    "-m",
                    "evaluation.run_benchmark",
                    "--worker",
                    game,
                    "--mode",
                    mode,
                    "--seed",
                    str(seed),
                    "--budget",
                    str(args.budget),
                    "--output",
                    str(folder),
                ]
                if args.save_graphs:
                    command.append("--save-graphs")
                error = None
                with (folder / "worker.log").open("w", encoding="utf-8") as log:
                    try:
                        subprocess.run(
                            command,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=args.timeout,
                        )
                    except (subprocess.SubprocessError, OSError) as exc:
                        error = f"{type(exc).__name__}: worker failed; see worker.log"
                result_file = folder / "result.json"
                result = (
                    json.loads(result_file.read_text())
                    if result_file.exists()
                    else {
                        "game": game,
                        "agent": mode,
                        "seed": seed,
                        "budget": args.budget,
                        "error": error or "Worker produced no result",
                    }
                )
                if error:
                    result["error"] = error
                rows.append(result)
                write_json(args.output / "results.json", rows)
                print(
                    f"{len(rows)}/{total} {game} {mode} seed={seed} "
                    f"score={result.get('official_score')} levels={result.get('levels_completed')} "
                    f"error={result.get('error')}",
                    flush=True,
                )
    with (args.output / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=sorted({key for row in rows for key in row})
        )
        writer.writeheader()
        writer.writerows(rows)
    from .analyze_results import report

    report(args.output)
    if any(row.get("error") for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
