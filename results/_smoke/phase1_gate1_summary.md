# Phase 1 — Gate 1 summary

Trajectories: 1, total events: 6

- Extra prefill tokens per 100-step trajectory: 29503.3 (n=1, too few runs for an IQR — spec §0.2 wants >= 5 seeds)
- Share of total prefill compute spent on compaction events: 56.3 (n=1, too few runs for an IQR — spec §0.2 wants >= 5 seeds)%
- Share of wall clock (ttft+decode) spent on compaction events: 26.4 (n=1, too few runs for an IQR — spec §0.2 wants >= 5 seeds)%

Per-trajectory detail:

| trajectory | steps | events | total prefill | event prefill | event prefill % |
|---|---|---|---|---|---|
| traj-000 | 60 | 6 | 31417 | 17702 | 56.3% |
