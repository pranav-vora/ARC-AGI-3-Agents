# Initial diagnostic cohort

This archive contains 225 cold-start runs (25 discovered games, three agents,
seeds 0-2, 1,000-action cap). Its no-repeat controller selected local untested
actions uniformly; graph ranked them using progress/failure/novelty evidence.
The difference confounds attribution of score changes to BFS. The final policy
matches local ranking and is evaluated in `../public-matched-1000/`.

Preserve this cohort for failure investigation and runtime evidence. Do not use
its single graph score win as a repeatable or isolated frontier-recovery gain.
