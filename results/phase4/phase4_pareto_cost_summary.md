# Phase 4.5 — prefill-cost summary, all 5 policies (vs. naive)

Trajectories compared: 14

- Prefill-token reduction, append_only vs. naive: median -10.5, IQR [-12.2, -7.5] (n=14)%
- Paired Wilcoxon signed-rank test on total prefill tokens (naive vs. append_only): W=1.0, z=-3.202, p=0.001367 (n_pairs=14, n_nonzero=14) -> SIGNIFICANT at alpha=0.05

- Prefill-token reduction, kv_evict vs. naive: median -28.5, IQR [-41.2, -19.2] (n=14)%
- Paired Wilcoxon signed-rank test on total prefill tokens (naive vs. kv_evict): W=0.0, z=-3.264, p=0.001097 (n_pairs=14, n_nonzero=14) -> SIGNIFICANT at alpha=0.05

- Prefill-token reduction, hybrid vs. naive: median 9.8, IQR [6.4, 13.2] (n=14)%
- Paired Wilcoxon signed-rank test on total prefill tokens (naive vs. hybrid): W=0.0, z=-3.264, p=0.001097 (n_pairs=14, n_nonzero=14) -> SIGNIFICANT at alpha=0.05

- none vs. naive: SKIPPED — 14/14 trajectories hit `context_exceeded` (ran out of context before the trajectory finished: traj-000, traj-001, traj-002, traj-003, traj-004, traj-005, traj-006, traj-008, traj-009, traj-010, traj-011, traj-012, traj-013, traj-014). A reduction-vs-naive percentage or paired test on `total_prefill_tokens` would be comparing a full-length run against a truncated one, which would look like a cost *win* for the wrong reason. See the per-trajectory table below and the `none` baseline discussion in `phase4_pareto_summary.md` for what this actually shows.

Per-trajectory detail (a `*` marks a run that hit `context_exceeded` — the total is only over the steps actually completed, not the full trajectory):

| trajectory | naive prefill | append_only prefill | kv_evict prefill | hybrid prefill | none prefill |
|---|---|---|---|---|---|
| traj-000 | 93276 | 92872 | 111048 | 80116 | 16630* |
| traj-001 | 86974 | 93290 | 122885 | 81067 | 15776* |
| traj-002 | 76040 | 84042 | 100382 | 68666 | 15786* |
| traj-003 | 84616 | 92409 | 119122 | 79071 | 16577* |
| traj-004 | 83459 | 93934 | 132813 | 73576 | 16341* |
| traj-005 | 64073 | 65551 | 64574 | 54343 | 16718* |
| traj-006 | 66116 | 80223 | 111882 | 62631 | 16416* |
| traj-008 | 73462 | 81118 | 85053 | 63449 | 16602* |
| traj-009 | 74521 | 84495 | 89181 | 69765 | 16304* |
| traj-010 | 66085 | 75237 | 74895 | 59538 | 16267* |
| traj-011 | 81426 | 86496 | 98165 | 68923 | 16686* |
| traj-012 | 81672 | 90284 | 111875 | 77306 | 16158* |
| traj-013 | 80833 | 89856 | 101011 | 72100 | 16360* |
| traj-014 | 74854 | 80918 | 189365 | 70286 | 16510* |
