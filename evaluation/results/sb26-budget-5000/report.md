# Stage 1 benchmark results

Official toolkit scores are reported as returned, without rescaling. Each run uses a fresh local environment and process. Runtime includes environment construction and agent execution; peak resident memory includes Python and native libraries.

| Agent | Runs | Mean official score | Total levels | Mean actions | Mean seconds | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| graph | 3 | 0.0000 | 0 | 4532.7 | 93.98 | 410.0 |
| norepeat | 3 | 0.0000 | 0 | 4815.0 | 58.67 | 421.8 |

Failed runs: 0.

| Agent | Runs with completion | Total actions / completed level | Mean first completion action |
| --- | ---: | ---: | ---: |
| graph | 0 | n/a | n/a |
| norepeat | 0 | n/a | n/a |

Aggregate actions per completed level includes unsuccessful runs; n/a means no completion. Per-run efficiency and completion indices are preserved in results.json and summary.csv.

Paired graph minus no-repeat mean official score: 0.0000 across 3 runs.
Graph score wins/ties/losses: 0/3/0.
Stage 2 assessment: no-go. Repeatable improvement across at least three paired seeds is unproven. Diagnose exact-state fragmentation, click coverage, and frontier availability before adding learned representations.

## Exploration diagnostics

| Agent | Mean unique states | Repeat states | Repeat actions | No-op rate | Frontier success |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph | 130.0 | 0.972 | 0.014 | 0.929 | 1.000 |
| norepeat | 386.0 | 0.920 | 0.000 | 0.880 | n/a |

Observation hashes retain UI, animation frames, and all visible mechanics. Different hidden states may still share observations, and UI changes can prevent revisits. The graph does not establish underlying Markov state identity.
