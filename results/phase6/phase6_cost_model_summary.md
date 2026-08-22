# Phase 6 (§6.1) — Analytical Cost Model Report

No GPU/vLLM used to produce this document — every number below comes
from `serving/cost_model.py` applied to the already-committed
`results/phase4/phase4_pareto_cost_steps.parquet` (9,905 step rows, 5 policies, 14 trajectories) and `configs/cost_model.yaml`.

## 1. Validation (against this project's own measurements)

Fitting one efficiency constant on all 9,905 real steps unfiltered gives median **18.02%** (IQR [5.28%, 51.71%]) — and validates badly: median absolute error **91.9%**, p90 237.1%. The reason shows up in a stratified breakdown by step size:

| step size | n steps | median efficiency | efficiency IQR | validation median error |
|---|---|---|---|---|
| tiny (<=50) | 4,864 | 5.22% | [3.51%, 8.08%] | 41.4% |
| small (50-200) | 327 | 20.89% | [18.27%, 25.09%] | 15.6% |
| medium (200-1000) | 4,000 | 50.94% | [47.47%, 55.12%] | 7.3% |
| large (>1000) | 714 | 81.21% | [62.63%, 89.69%] | 12.9% |

Efficiency rises monotonically with step size — fixed per-request overhead (HTTP round-trip, vLLM scheduling, kernel launch) dominates tiny steps and amortizes away for large ones, so a single blended constant mostly reflects how many tiny steps happen to be in the sample (4,864 of 9,905 here), not a stable hardware property. Since compaction's actual savings are dominated by large reprefill-after-compaction events, not tiny single-turn appends, this report calibrates only on steps with `prefill_tokens > 200` from here on.

**Corrected fit** (Qwen/Qwen3-0.6B on RTX 4060 Laptop 8GB (this project's own GPU), `prefill_tokens > 200`): efficiency median **52.14%** (IQR [48.05%, 58.32%]), validates against 4,714 held-in steps at median absolute error **9.1%**, p90 53.1%. This is the efficiency used for every extrapolation below.

**Aggregate check (not clean — reported anyway):** predicted GPU-seconds saved by hybrid vs. naive on the *same* hardware/model actually measured, summed over *every* step including tiny ones (15.9s) vs. the real measured wall-clock saved (sum of `ttft_ms`, -15.7s) — these don't even agree in sign. Checked whether GPU thermal throttling (this exact sweep is documented elsewhere in this project as having hit sustained throttling — `phase4_pareto_summary.md`) explains it: it doesn't cleanly — hybrid's steps ran at a *higher* mean SM clock than naive's (2496 vs. 2412 MHz) despite processing fewer total prefill tokens (980,837 vs. 1,087,407), yet still show higher mean `ttft_ms` (68.2ms vs. 61.3ms). The likely explanation this report doesn't fully chase down: `ttft_ms` is wall-clock to first token, which includes policy-side CPU work (hybrid's per-turn routing logic does more than naive's single threshold check) and engine queueing, not just GPU prefill compute — exactly the kind of factor a pure-FLOPs model was never built to capture. This is why the per-step, large-step-only validation above (not this aggregate check) is what this report treats as the model's real evidence.

## 2. Extrapolation: naive vs. hybrid, replayed at 70B on H100

Same real per-step token sequences (all 14 trajectories) as measured in Phase 4.5, recomputed under Llama-3-70B (extrapolation target — never downloaded or run)'s architecture (80 layers, hidden=8192, 70,552,387,584 params) instead of Qwen/Qwen3-0.6B's.

### Point estimate (this project's fitted efficiency) (efficiency=0.5214)

- Llama-3-70B (extrapolation target — never downloaded or run) on H100 SXM 80GB (extrapolation target)
- naive FLOPs: 1.625e+17
- hybrid FLOPs: 1.460e+17
- FLOPs saved: 1.654e+16
- GPU-seconds saved: 32.1s (0.009 GPU-hours)
- **Dollars saved (this dataset, this hardware's `$/GPU-hour`): $0.0223**

### Low sensitivity bound (fit's IQR low) (efficiency=0.4805)

- Llama-3-70B (extrapolation target — never downloaded or run) on H100 SXM 80GB (extrapolation target)
- naive FLOPs: 1.625e+17
- hybrid FLOPs: 1.460e+17
- FLOPs saved: 1.654e+16
- GPU-seconds saved: 34.8s (0.010 GPU-hours)
- **Dollars saved (this dataset, this hardware's `$/GPU-hour`): $0.0242**

### High sensitivity bound (fit's IQR high) (efficiency=0.5832)

- Llama-3-70B (extrapolation target — never downloaded or run) on H100 SXM 80GB (extrapolation target)
- naive FLOPs: 1.625e+17
- hybrid FLOPs: 1.460e+17
- FLOPs saved: 1.654e+16
- GPU-seconds saved: 28.7s (0.008 GPU-hours)
- **Dollars saved (this dataset, this hardware's `$/GPU-hour`): $0.0199**

**Cross-check against Gate 4's already-published result:** the FLOPs reduction here (10.2%) closely matches Gate 4's independently-measured token-count reduction (`phase4_pareto_cost_summary.md`: "hybrid vs. naive: median 9.8%") — expected, since both FLOPs terms scale with token count, but a real consistency check between two differently-derived numbers, not one number restated as two.

**Making the dollar figure tangible:** this dataset is only 14 short trajectories, so $0.0223 is real but small. Scaling illustratively to 100,000 similar trajectories/day (purely a multiplication of this measured per-trajectory saving — not a new measurement, and production traffic would not actually look like this project's synthetic on-call trajectories): **$159.08/day**, $58,062.61/year, at the point-estimate efficiency and this hardware's illustrative `$/GPU-hour`.

## 3. Cross-check: industry-typical H100 MFU, independent of this project's fit

Using a commonly-cited production-serving MFU figure (40%) instead of anything measured on this project's own hardware, as an independent sanity check on the point estimate above:

### Industry-typical H100 MFU (40%) (efficiency=0.4000)

- Llama-3-70B (extrapolation target — never downloaded or run) on H100 SXM 80GB (extrapolation target)
- naive FLOPs: 1.625e+17
- hybrid FLOPs: 1.460e+17
- FLOPs saved: 1.654e+16
- GPU-seconds saved: 41.8s (0.012 GPU-hours)
- **Dollars saved (this dataset, this hardware's `$/GPU-hour`): $0.0290**

## Assumptions and limitations

- **The FLOPs formula is an approximation** (spec's own "2 × N_params × n_tokens plus attention terms") — it ignores embedding/softmax/norm FLOPs and treats every new token as attending the full context rather than its own causal-masked prefix. See `serving/cost_model.py`'s module docstring for the full derivation.
- **The model only covers GPU prefill compute, not policy-side CPU overhead or engine queueing** — the aggregate check in §1 suggests this matters: hybrid's per-turn routing logic (evict/keep/summarize classification on every candidate turn) plausibly costs more client-side CPU time than naive's single threshold check, which would show up in measured `ttft_ms` but not in this FLOPs-only model. Not confirmed with a dedicated measurement — flagged as the likely explanation, not established as fact.
- **The efficiency fit is from batch-size-1 prefill on an underutilized 8GB laptop GPU** — production H100 serving (larger batches, continuous batching, a very different compute/memory-bandwidth ratio) will not necessarily hit the same fraction of peak FLOPS. That's exactly why an independent industry-MFU cross-check is reported alongside the fitted point estimate, not instead of it.
- **Llama-3-70B was never downloaded or run** — its architecture numbers come from Meta's published config, its parameter count from this project's own parameter-counting formula (cross-checked against the commonly-cited ~70.6B figure, matches within 0.2%). Nothing here is a measurement of Llama-3-70B; it is a projection.
- **`$/GPU-hour` is illustrative** (see `configs/cost_model.yaml`'s own comment) — 2026 on-demand H100 market rates span roughly $2-12/hr depending on provider; the dollar figures above scale linearly with whatever rate is actually paid.
- **This report extrapolates naive-vs-hybrid's *cost* difference only.** Every policy in this project's Gate 4 measurement sits at 0/5 task success (`results/phase4/phase4_pareto_summary.md`) — nothing here claims the quality tradeoff extrapolates, only the cost one.
