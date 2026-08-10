# Phase 1 — Gate 1 summary

Trajectories: 14, total events: 280

- Extra prefill tokens per 100-step trajectory: median 36611.1, IQR [34693.0, 40515.8] (n=14)
- Share of total prefill compute spent on compaction events: median 63.0, IQR [61.2, 64.6] (n=14)%
- Share of wall clock (ttft+decode) spent on compaction events: median 31.4, IQR [29.4, 33.2] (n=14)%

Per-trajectory detail:

| trajectory | steps | events | total prefill | event prefill | event prefill % |
|---|---|---|---|---|---|
| traj-000 | 164 | 22 | 102106 | 67074 | 65.7% |
| traj-001 | 169 | 22 | 103670 | 66530 | 64.2% |
| traj-002 | 166 | 20 | 96300 | 60888 | 63.2% |
| traj-003 | 163 | 22 | 102897 | 68582 | 66.7% |
| traj-004 | 165 | 22 | 104461 | 67595 | 64.7% |
| traj-005 | 164 | 16 | 76777 | 47028 | 61.3% |
| traj-006 | 162 | 19 | 90112 | 56510 | 62.7% |
| traj-008 | 161 | 19 | 91172 | 55812 | 61.2% |
| traj-009 | 162 | 20 | 94449 | 60102 | 63.6% |
| traj-010 | 164 | 18 | 87031 | 53187 | 61.1% |
| traj-011 | 161 | 19 | 91675 | 55987 | 61.1% |
| traj-012 | 161 | 22 | 102355 | 67255 | 65.7% |
| traj-013 | 164 | 20 | 98221 | 59930 | 61.0% |
| traj-014 | 165 | 19 | 90588 | 55997 | 61.8% |
