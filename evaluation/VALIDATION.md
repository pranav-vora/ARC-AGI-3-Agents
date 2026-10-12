# Stage 1 validation

Validated on Windows with Python 3.12.14, PyTorch 2.14.1 CPU,
`arc-agi` 0.9.1, and `arcengine` 0.9.3. Versions are locked in `uv.lock`.

## Correctness and exploration reliability

```powershell
uv run python -m pytest -q
uv run python -m ruff check .
uv run python -m ruff format --check .
uv run python -m mypy --follow-imports=silent agents/stage1 evaluation
uv lock --check --offline
```

All **138 tests pass**, including 46 Stage 1 cases. Repository-wide Ruff lint and
format checks pass (41 files). Targeted strict mypy passes for all ten new source
files. Full `mypy agents main.py` reports 93 errors in 17 legacy files, including
missing third-party type stubs. Running the same check against an archived clean
base commit `4743e7d` reproduces the same 93 errors; normalized error messages
have no additions. Coverage includes 120 and 125 scripted graph transitions, copied contiguous
CPU integer observations, stable hashes, separate levels, directed edges,
self-loops, repeated counts, alternative outcomes, serialization, BFS, candidate
legality and expansion, uncertain trials, bounded resets, terminal recording,
and exact action budgets for all three adapters.

Reachable-frontier scripts at distances 1, 2, 5, and 8 recover successfully.
Unexpected replay outcomes preserve both edges and cancel the route. A branching
nine-state synthetic puzzle with reversals and no-ops completes under five seeds.
These deterministic tests have zero unexplained repeated state-action pairs.
Graph and no-repeat make identical local decisions before frontier recovery in
five seeded comparisons, preventing a local-ranking confound.

Two legacy test files originally failed collection by importing the removed
`agents.structs` module. They now use the installed toolkit's public types and
mock the current Arcade interface. The shared recordings fixture now uses pytest
temporary directories instead of deleting a shared OneDrive directory. No tests
are excluded, and existing agents' execution semantics were not changed.

## Official environment checks

The official toolkit discovered 25 public environments; this is an observed
inventory, not a hard-coded game count. A direct local `ar25` smoke test inspected
its 64x64 observation and available actions, executed a valid action, and closed
an official scorecard. Installed click bounds are 0-63. `Arcade.make` initializes
local environments, so agents do not submit a redundant reset.

All six smoke comparisons (three agents, `ar25` and `ls20`, seed 0, 50 actions)
completed without errors. None completed a level. Rendered `ls20` observations
confirmed that UI and board changes are preserved. An independent official replay
of the successful initial LP85 trace reproduced all 327 source/destination hashes
through the first level completion.

## Evaluation cohorts

- `results/initial-public-1000/`: all 225 scheduled runs completed and passed the
  complete trace audit. Peak resident memory was 408.7 MiB and maximum measured
  runtime 36.68 seconds per worker. This diagnostic cohort predates matched local
  ranking for no-repeat; it does not isolate BFS value.
- The matched-policy suite uses all discovered games, all three agents, seeds
  0-2, 1,000 submitted actions per game, identical legal candidates and reset
  limits. All 225 scheduled runs completed and passed the full trace audit.
  Maximum measured runtime was 43.65 seconds and peak resident memory
  408.0 MiB per worker. See [matched results](results/public-matched-1000/report.md).
- A separate SB26 sensitivity experiment allows 5,000 actions for graph and
  no-repeat under seeds 0-2; it is not substituted for the required 1,000-action
  suite.

```powershell
uv run python -m evaluation.run_benchmark --offline --output evaluation/runs/public-matched-1000
uv run python -m evaluation.audit_results evaluation/runs/public-matched-1000
```

Every submitted action, including resets, consumes budget. Audit checks coverage,
action/outcome continuity, completion accounting, legal click coordinates,
repeat justifications, scorecard correspondence, reset limits, timeouts, and a
documented 1 GiB resident-memory limit. Complete traces and per-run scorecards
remain in ignored `evaluation/runs/`; compact results are committed separately.

## Gates and Stage 2 decision

| Gate | Outcome | Evidence |
| --- | --- | --- |
| 1: State and graph correctness | Pass | All 138 unit tests; exact identity, directed transitions, alternative outcomes, BFS, and mismatch replanning. |
| 2: Exploration reliability | Pass | All scripted reachable frontiers recovered; zero unexplained repeats in deterministic tests and all audited public runs. |
| 3: Runtime stability | Pass | All 225 matched runs completed cleanly; no budget violations, missing outcomes, scorecard inconsistencies, or resource-limit violations. |
| 4: Measured value | **Not met** | Graph and matched no-repeat tie in all 75 score pairs, each with one completed level. The three paired 5,000-action SB26 probes also produce no levels. |
| 5: Known failure modes | Pass | Representative observations, exact successful replay, and source/trace diagnoses in [FAILURE_MODES.md](FAILURE_MODES.md). |

Graph verifies three frontier recoveries in the matched public suite, all in
SB26. This demonstrates reliable routing, not improved solving. Both graph and
no-repeat complete LP85 level 1 at action 327, with mean official score 0.0087
across 75 runs and aggregate 26,677 submitted actions per completed level. The
initial graph-only win disappears when local ranking matches.

Stage 2 is **no-go**. Gate 4 and the overall performance target remain unmet;
implementation correctness and successful frontier recovery do not satisfy that
target. Legal but unhelpful click candidates, sparse goal knowledge, and limited
reuse of exact states explain the major observed limitations. No performance
improvement is claimed.
