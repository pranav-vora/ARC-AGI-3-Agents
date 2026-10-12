# Stage 1 benchmark results

> Diagnostic cohort: graph and no-repeat used different local ranking. This does not isolate BFS value. See [provenance](PROVENANCE.md) and the [matched comparison](../public-matched-1000/report.md).

Official toolkit scores are reported as returned, without rescaling. Each run uses a fresh local environment and process. Runtime includes environment construction and agent execution; peak resident memory includes Python and native libraries.

| Agent | Runs | Mean official score | Total levels | Mean actions | Mean seconds | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| graph | 75 | 0.0087 | 1 | 355.7 | 2.68 | 408.7 |
| norepeat | 75 | 0.0000 | 0 | 439.2 | 5.10 | 395.1 |
| random | 75 | 0.0204 | 1 | 442.6 | 4.73 | 384.8 |

Failed runs: 0.

| Agent | Runs with completion | Total actions / completed level | Mean first completion action |
| --- | ---: | ---: | ---: |
| graph | 1 | 26677.0 | 327.0 |
| norepeat | 0 | n/a | n/a |
| random | 1 | 33198.0 | 469.0 |

Aggregate actions per completed level includes unsuccessful runs; n/a means no completion. Per-run efficiency and completion indices are preserved in results.json and summary.csv.

Paired graph minus no-repeat mean official score: 0.0087 across 75 runs.
Graph score wins/ties/losses: 1/74/0.
Stage 2 assessment: no-go. Repeatable improvement across at least three paired seeds is unproven. Diagnose exact-state fragmentation, click coverage, and frontier availability before adding learned representations.

## Exploration diagnostics

| Agent | Mean unique states | Repeat states | Repeat actions | No-op rate | Frontier success |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph | 177.4 | 0.384 | 0.001 | 0.251 | 1.000 |
| norepeat | 184.6 | 0.405 | 0.000 | 0.309 | n/a |
| random | 178.8 | 0.419 | 0.078 | 0.316 | n/a |

Observation hashes retain UI, animation frames, and all visible mechanics. Different hidden states may still share observations, and UI changes can prevent revisits. The graph does not establish underlying Markov state identity.
