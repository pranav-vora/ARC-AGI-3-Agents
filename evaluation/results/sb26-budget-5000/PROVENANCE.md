# SB26 budget sensitivity

Six additional matched-policy runs use the same SB26 public environment and
seeds 0, 1, and 2 with a 5,000-action cap. All six passed the complete trace audit.
Neither agent completed a level. Graph verified 64 frontier recoveries per seed
but averaged 92.9% no-ops versus 88.0% for no-repeat, and visited 130 distinct
states versus 386. This is negative performance evidence, not a budget-scaled
success claim. Both controllers exhausted their candidates/recovery options
before the cap; fewer actions are not completion efficiency when nothing solves.

The separate probe overlapped part of the main suite on the same machine.
Process time and memory are measured, but timings are subject to contention;
use the score/completion/decision evidence rather than claiming a speed gain.
Full traces remain in the ignored run directory. `scorecards.json` preserves the
individual official scorecard payloads with API-key fields removed.
