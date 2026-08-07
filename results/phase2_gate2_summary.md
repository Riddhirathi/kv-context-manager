# Phase 2 — Gate 2 summary (prefill-reduction half; action agreement is Phase 3)

Trajectories compared: 14

- Prefill-token reduction, append_only vs. naive: median 0.7, IQR [-0.9, 1.9] (n=14)%
- Paired Wilcoxon signed-rank test on total prefill tokens (naive vs. append_only): W=42.0, z=-0.628, p=0.5302 (n_pairs=14, n_nonzero=14) -> not significant at alpha=0.05
- Monotonic prefix growth (successive append_only events' divergence index non-decreasing): 14/14 trajectories (spec §2.2's headline systems property)

Per-trajectory detail:

| trajectory | naive prefill | append_only prefill | reduction % |
|---|---|---|---|
| traj-000 | 93276 | 92872 | 0.4% |
| traj-001 | 96350 | 93306 | 3.2% |
| traj-002 | 85528 | 84042 | 1.7% |
| traj-003 | 94264 | 92393 | 2.0% |
| traj-004 | 92947 | 93934 | -1.1% |
| traj-005 | 64073 | 65535 | -2.3% |
| traj-006 | 75828 | 80303 | -5.9% |
| traj-008 | 83174 | 81102 | 2.5% |
| traj-009 | 84057 | 84495 | -0.5% |
| traj-010 | 75781 | 75205 | 0.8% |
| traj-011 | 81410 | 86496 | -6.2% |
| traj-012 | 91000 | 90332 | 0.7% |
| traj-013 | 90401 | 89856 | 0.6% |
| traj-014 | 83862 | 80902 | 3.5% |
