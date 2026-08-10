# Phase 4.4 — hybrid prefill-cost summary (vs. naive, append_only, kv_evict)

Trajectories compared: 14

- Prefill-token reduction, append_only vs. naive: median 0.7, IQR [-0.9, 1.9] (n=14)%
- Paired Wilcoxon signed-rank test on total prefill tokens (naive vs. append_only): W=42.0, z=-0.628, p=0.5302 (n_pairs=14, n_nonzero=14) -> not significant at alpha=0.05

- Prefill-token reduction, kv_evict vs. naive: median -19.8, IQR [-27.2, -7.5] (n=14)%
- Paired Wilcoxon signed-rank test on total prefill tokens (naive vs. kv_evict): W=2.0, z=-3.139, p=0.001696 (n_pairs=14, n_nonzero=14) -> SIGNIFICANT at alpha=0.05

- Prefill-token reduction, hybrid vs. naive: median 16.6, IQR [15.5, 20.1] (n=14)%
- Paired Wilcoxon signed-rank test on total prefill tokens (naive vs. hybrid): W=0.0, z=-3.264, p=0.001097 (n_pairs=14, n_nonzero=14) -> SIGNIFICANT at alpha=0.05

Per-trajectory detail:

| trajectory | naive prefill | append_only prefill | kv_evict prefill | hybrid prefill |
|---|---|---|---|---|
| traj-000 | 93276 | 92872 | 111048 | 80116 |
| traj-001 | 96350 | 93306 | 122885 | 81067 |
| traj-002 | 85544 | 84042 | 100382 | 68666 |
| traj-003 | 94264 | 92409 | 119122 | 79071 |
| traj-004 | 92963 | 93934 | 132813 | 73576 |
| traj-005 | 64073 | 65535 | 64574 | 54343 |
| traj-006 | 75844 | 80303 | 111882 | 62631 |
| traj-008 | 83174 | 81118 | 85053 | 63433 |
| traj-009 | 84073 | 84495 | 89181 | 69781 |
| traj-010 | 75781 | 75221 | 74895 | 59538 |
| traj-011 | 81426 | 86496 | 98165 | 68923 |
| traj-012 | 91000 | 90348 | 111875 | 77290 |
| traj-013 | 90417 | 89856 | 101011 | 72100 |
| traj-014 | 83862 | 80902 | 189365 | 70302 |
