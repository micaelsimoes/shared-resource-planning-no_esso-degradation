# P5.12-Y — TSO comparator reconstruction and sensitivity

**Authorship note.** Planner-executed (Worker unavailable: external API spend limit).
Reconstruction plan, historical option set, ladder, acceptance rule, planning weight
and execution order were frozen to `data/SRP1/Results/P512Y/frozen_tso_plan.json`
(SHA-256 prefix `c0f6035a396c2ea4b77bbd75`) **before** the first solve. Classification
thresholds reused verbatim from P5.12-W; none altered afterwards.

## Perturbation family versus absolute values

The perturbation **family** is common across all fixtures: `warm_start_bound_push`.
The **absolute values differ** because this fixture's historically configured
baseline is `1e-6`, not `1e-5` (case9 configures `bound_push = 1e-6`; its cycle-7
echo also shows `acceptable_iter = 0`, `acceptable_tol = 1e-5`, `compl_inf_tol =
5e-4`, all from the production TSO warm-start override and case9's own settings).
A ±10x multiplicative perturbation around the fixture's own baseline was applied:
LOW `1e-7`, BASELINE `1e-6`, HIGH `1e-5`. `1e-4` was explicitly **not** tested here.
TSO options were **not** harmonized with the DSO fixtures.

## Baseline reconstruction gate — PASSED (byte-identical)

The BASELINE `1e-6` replay reproduced the historical cycle-7 trace
(`optim_log_case9_2025_Summer.log`, 8th solve, lines 28780-31897)
**byte-identically: 0 differing lines in 3118**, after normalizing only the log path,
elapsed time and the fresh-file leading blank line. Option echo, iteration-0 row,
full iteration table, 37 iterations, termination, final objective
`3.3794600264311919e+05`, residuals and evaluation counts (39 / 37) all match.

## Sensitivity result

NL hash identical across all three runs (`9145feeb21fc8728`); only
`warm_start_bound_push` and `output_file` differed. Three fresh processes.

| | LOW `1e-7` | BASELINE `1e-6` | HIGH `1e-5` |
|---|---|---|---|
| termination | Optimal | Optimal | Optimal |
| iterations | 38 | 37 | 38 |
| objective (unscaled) | 337946.00264311914 | 337946.00264311919 | 337946.00264311919 |
| dual infeasibility | 7.470e-04 | 7.470e-04 | 7.470e-04 |
| constraint violation | 1.01e-14 | 7.87e-15 | 1.23e-14 |
| complementarity | 4.552e-05 | 4.552e-05 | 4.552e-05 |
| safeguard / restoration | 0 / 0 | 0 / 0 | 0 / 0 |
| relative objective difference | 1.72e-16 | — | 0.0 |
| max scaled primal distance | 1.46e-14 | — | 7.80e-11 |
| columns > 1e-3 | 0 | — | 0 |
| interface max scaled difference | 2.14e-16 | — | 2.14e-16 |
| iteration change | +2.7% | — | +2.7% |

### Classification (multi-axis, frozen thresholds)

- **solve validity: ROBUST**
- **objective equivalence: EQUIVALENT** (1.7e-16 and 0.0, far inside 1e-6)
- **interface/propagating-output equivalence: EQUIVALENT** (2.1e-16)
- **branch/regime: EQUIVALENT** (no column exceeds 1e-3)
- **path sensitivity: NOT MATERIAL** (+2.7%, zero safeguard/restoration)

### Planning-weighted objective effect (descriptive only)

Weight for 2025/Summer = `N_2025 x D_Summer x annualization` = `5 x 91 x 1.0` = **455**.
Weighted objective difference: **2.6e-08 planning units** (LOW), **0** (HIGH) —
negligible against the accepted 22.09 cross-depth uncertainty. Reported descriptively;
it does not redefine any threshold.

## Verdict

BREADTH PROBE INCONCLUSIVE
