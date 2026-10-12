# Matched-policy public comparison

All 25 officially discovered public environments, all three agents, seeds 0, 1,
and 2, and a 1,000-submitted-action cap per run. Graph and no-repeat use identical
local ranking, candidate generation, reset limits, and completion handling; only
graph performs BFS frontier recovery. All 225 scheduled runs and complete traces
passed the audit. Individual official scorecard payloads are preserved in
`scorecards.json` with API-key fields removed. Full traces remain outside Git.

Graph versus no-repeat: zero score wins, 75 ties, zero losses. Both completed one
level (LP85 seed 1 at action 327); neither completed a full game. No repeatable
meaningful improvement was demonstrated. See `../../FAILURE_MODES.md` and the
separate 5,000-action SB26 sensitivity result for the failure investigation.

Workers within this suite ran sequentially in isolated processes. A separate
sensitivity experiment overlapped an early portion of the suite on the same
machine, so timings are descriptive and subject to contention. Runtime excludes
process/import startup; peak resident memory includes native libraries.
