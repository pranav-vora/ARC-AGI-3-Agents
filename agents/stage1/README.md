# Stage 1 exact-state explorer

This package uses **PyTorch CPU tensors**, SHA-256 observation identity, and an
observed directed graph. There is no training, learned representation, or neural
policy. Existing agents retain their original behavior.

## Setup and running on Windows

Install Python 3.12 and `uv`, then run these commands from the repository root:

```powershell
uv sync --locked
uv run python -m pytest tests/unit/test_stage1.py
uv run python -m evaluation.run_benchmark --list-games
uv run python -m evaluation.run_benchmark --games ar25 ls20 --budget 50 --seeds 0 --output evaluation/runs/smoke
uv run python -m evaluation.run_benchmark --output evaluation/runs/public-matched-1000
```

The discovery/download phase uses the official toolkit and the existing `.env`
configuration (`ARC_API_KEY` if needed). It selects normal download-and-local-run
mode explicitly, even if the existing application is configured for online play.
After downloading the environments, workers run **offline**, with no LLM API
calls. `--offline` also skips online discovery/download; its suite is only the
games already present in `environment_files/`. Check the manifest for coverage.
Do not commit credentials or downloaded game source.

The existing runner also accepts these agents:

```powershell
uv run main.py --agent=stage1graph --game=ls20
uv run main.py --agent=stage1norepeat --game=ls20
uv run main.py --agent=stage1random --game=ls20
```

That runner retains its existing server-based discovery and `.env` configuration.
Use the benchmark command for reproducible local experiments. All three new
agents default to 1,000 actions; the benchmark defaults to seeds 0, 1, and 2.
Optional arguments include `--agents graph`, `--budget 100`, `--seeds 0`,
`--timeout 300`, and `--save-graphs`. Output directories must be empty to prevent
mixing separate experiments. No Kaggle submission is performed.

## State and transition contracts

- `state.encode(observation)` accepts the official `FrameData`/`FrameDataRaw`
  attributes. `normalize(...)` supports lists, NumPy arrays, and tensors. It copies
  each 2D frame into contiguous CPU `int64` storage; equivalent representations
  have identical IDs. All animation frames and UI pixels participate in identity.
- Identity includes game ID, completed-level count, status, available actions,
  frame shapes, and canonical little-endian cell bytes. GUIDs/request identifiers
  are excluded. No similarity-based state merging is performed. Callers should
  treat encoded tensors as immutable.
- `changed_cells` returns `(frame, row, column)` locations, including added or
  removed cells. Visible change is a diagnostic; only increased completed-level
  counts count as progress.
- `StateGraph` exposes node lookup, visits, transition recording, tested actions,
  neighbor lookup, and BFS. Each `(source, action, target)` aggregates observations,
  cost, visible changes, progress, and failure counts. An action may have multiple
  observed destinations. A visit must be recorded separately from a transition.
- Replay neighbors exclude resets, self-loops, terminals, cross-level edges, and
  actions with multiple observed outcomes. Recorded evidence remains available
  even when an edge cannot be replayed.

```python
from agents.stage1.graph import StateGraph

graph = StateGraph.load("evaluation/runs/example/game-graph-0/graph.json")
print(len(graph.nodes))
graph.save("graph-copy.json")
```

Graphs use versioned JSON with tensor values represented as lists. Loading checks
the schema version, state digests, and edge endpoints. The export is for inspection
and graph round trips; it is not a full controller checkpoint (RNG state, pending
decisions, and controller statistics are not restored).

## Controller behavior

`Explorer.observe(state)` consumes a pending action result once. `choose()` issues
one immutable `Action`, or returns `None` when done. Choosing twice without an
observation raises an error. The adapter records terminal outcomes and enforces
the exact action cap, including every explicitly submitted reset. The toolkit's
environment constructor already supplies the initial reset observation; that
setup operation is outside the agent's submitted-action budget for all variants.

Only advertised actions are candidates. Clicks use a cumulative lattice of 8×8,
then 16×16, then all 64×64 pixels; duplicates are removed. The bounds come from the
locked toolkit's `ComplexAction` schema. Empty action lists are not interpreted
as permission to try arbitrary keys; recovery resets remain available as the
toolkit's lifecycle operation.

- **Random:** seeded uniform selection from current legal candidates; it broadens
  the lattice after all current candidates have been tried at that state.
- **No-repeat:** remembers tested actions per state and uses the same local ranking
  and RNG consumption as the graph agent. Once exhausted, it broadens clicks, revisits uncertain actions, then
  wanders along known changing local edges without BFS.
- **Graph:** ranks untested actions by within-level observed progress, failure,
  and novelty rates, with seeded tie-breaking. With no local untested actions, it
  seeks the nearest reachable frontier by BFS. Equal-distance ties favor progress,
  lower failure rates, more untested actions, then stable state IDs. It broadens
  reachable frontiers and local click candidates when necessary.

Every replay step is checked against its expected observation. Mismatches cancel
the remaining route and preserve the alternative outcome; that action is no
longer eligible for deterministic routing. Ambiguous actions have at most two
additional exploratory trials per state-action pair. There are at most two
recovery resets per game, plus the initial reset only when the initial observation
is `NOT_PLAYED`. Exhaustion ends the run. Reset and level transitions clear routes,
while graph evidence and action histories remain scoped by game and level.

## Evaluation artifacts and interpretation

The benchmark discovers public games, downloads them using `Arcade.make`, and
runs each game/agent/seed sequentially in an isolated process. Seeds apply both
to the official environment and the agent. Workers limit PyTorch to one CPU
thread. A configurable per-run timeout bounds unexpected hangs.

Each run contains `actions.jsonl`, the official `scorecard.json` (without API key),
`result.json`, and a worker log. `--save-graphs` adds `graph.json`. Root artifacts
include the game/version and dependency manifest, `summary.csv`, `results.json`,
and `report.md`. Only compact evidence is committed under `evaluation/results/`.

```powershell
uv run python -m evaluation.analyze_results evaluation/runs/public-matched-1000
```

Official score is copied directly from the toolkit. Do not interpret local
scores as a submitted Kaggle leaderboard result. Runtime includes environment
construction and agent execution, excluding process/import startup. Peak memory
is the OS-reported resident high-water mark of the whole isolated worker,
including Python, imported agent libraries, and native PyTorch allocations.

Actions per completed level is total submitted actions divided by the maximum
level count reached (null if none). Completion action indices show when progress
first occurred. Repetition and no-op rates include submitted resets. A no-op means
no visible pixel change; metadata can still have changed. Frontier recovery
success divides fully verified routes by routes started, including interrupted
routes. No frontier attempts yields null, not a misleading perfect success rate.

Every transition records its decision reason. Tested actions may be repeated for
frontier replay, an uncertain-outcome trial, recovery reset, or a known local
movement in the no-repeat baseline. Repeating a tested action under an untested
action decision is an invariant failure and causes benchmark failure.

```powershell
uv run python -m evaluation.audit_results evaluation/runs/public-matched-1000
```

The audit verifies complete scheduled coverage, each action/outcome and trace
counter, scorecard agreement, reset limits, termination, and justification for
every repeated state-action pair. Its default resident-memory audit limit is
1 GiB per isolated worker. It also records failure hypotheses for every incomplete
run; these hypotheses require representative observation/source inspection.

Exact observations can fragment the graph when timers/UI/animation change, and
identical observations can conceal different internal states. These limitations
are intentional diagnostics for Stage 1. Do not move to learned representations
merely because the implementation passes unit tests: inspect completion, official
score, and paired results against no-repeat first.
