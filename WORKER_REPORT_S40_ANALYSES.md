# Worker Report — P5.15 Addendum 22 item (1): two zero-solve analyses (S40)

## Task received

Planner task (this session): two zero-solve analyses authorized by
`PLANNER_BRIEF_2026-09-13.md` Addendum 22 and frozen spec v11
(`data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json`, item
`1_zero_solve_analyses`):

- **Task A** — component decomposition of the run1-vs-A/C/D `gross_operational_cost`
  differences (−39,149 / −57,743 / −72,191 vs run1; C−D = +14,448) from committed
  terminal artifacts, reconciled to double precision, per component and (where the
  artifacts allow) per block, with the "different ADMM configurations" statement.
- **Task B** — a period-by-period cross-check of the largest PF-residual entries
  against the at-rating periods at node 7 (contingency table, coincidence rate,
  base rate under independence, 2030 concentration) plus the RESULT table Addendum 22
  asks for (node 7 utilization and shared-storage dispatch in the binding periods,
  nodes 5/9 as comparators), for arms D, C and (continuity) v9 A/B.

No solves permitted anywhere; no edits to production files, the case file,
`p515_g_g1_g4_admm_gates.py`, or any existing `p515_s36_*`/`p515_s39_*` script or
committed output.

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` in full (Addenda 1–22; Addendum 22 is the authority).
- `data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json`.
- `WORKER_REPORT_S39_NODE7.md` and `p515_s39_node7_interface.py` (prior worker's node-7
  look at v9 arms A/B; read to understand the existing per-arm artifact schema and the
  "consistent with, does not prove" reading this task's Task B was asked to tighten).
- Per run (`P515S35_REF_run` = run1/baseline, `P515S38_A_TAU0_run` = A,
  `P515S38_B_PFBAL_run` = B, `P515S39_C_run` = C, `P515S39_D_run` = D):
  `g_<label>.json` (instance, cycle_trajectory, `recourse_change` per cycle),
  `component_levels_terminal.json` (`totals_weighted`, `blocks`, `recourse_components`),
  `pf_entry_stride_<label>.jsonl` (per-cycle PF consensus entries; run1 has none — it
  predates the per-entry PF capture, confirmed by directory listing, so run1 is used
  only in Task A, never in Task B), `ess_entry_stride_baseline.jsonl`, and (for C/D)
  `s39_evaluation.json` (the s39 evaluator's own precomputed rule-nine/ten bar, used as
  an independent cross-check of this script's own bar computation).
- `shared_energy_storage_data.py:721-728` (read-only) to confirm the sign convention of
  `es_pnet` (`agg_pnet = pch_per_unit − pdch_per_unit`, i.e. **positive = net charge**),
  used to label the RESULT table's storage dispatch rows.
- `p513_solve_profile_guard.py` (guard API, matched to the existing S39 script's usage).

## Files modified

None (production, case file, `p515_g_*`, `p515_s36_*`, `p515_s39_*` all untouched —
verified by `git status`/`git diff --cached --stat` before every commit, only the files
listed below were ever staged).

## Files created

- `p515_s40_cost_decomposition.py` (Task A script).
- `p515_s40_node7_crosscheck.py` (Task B script; new file, does not edit
  `p515_s39_node7_interface.py` or touch its committed outputs under
  `data/SRP1/Results/P515S39/node7_interface/{s38_A_tau0,s38_B_pfbal}/`).
- `data/SRP1/Results/P515S40/cost_decomposition/{cost_decomposition.json,run_log.txt,
  evidence_manifest_sha256.json}`.
- `data/SRP1/Results/P515S40/node7_result/{s38_A_tau0,s38_B_pfbal,s39_C,s39_D}/
  {node7_crosscheck_<label>.json, run_log_<label>.txt, evidence_manifest_sha256.json}`.
- This report.

Nothing under `data/SRP1/Results/P515S36/`, `.p515_g_gate.lock`, or any other process
was touched (checked before and after every commit).

## Changes made

Wrote the two scripts described above (both zero-solve, `SolveProfileGuard(permitted=())`
installed before any file is read, `guard.verify(0)` checked before any output write,
write-once outputs). No other file was edited.

## Commands / experiments run

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B p515_s40_cost_decomposition.py --dry-run
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B p515_s40_cost_decomposition.py
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B p515_s40_node7_crosscheck.py --dry-run
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B p515_s40_node7_crosscheck.py
```

Both real runs completed with `SolveProfileGuard.verify(0): OK`, wrote exactly their
declared write-once outputs (no pre-existing output overwritten), no exceptions.

Commit order followed CLAUDE.md (script before outputs): `p515_s40_cost_decomposition.py`
(commit `11075352`) → `p515_s40_node7_crosscheck.py` (commit `e4db9033`) → both scripts'
outputs together (commit `d73a2b22`, 15 files, only the paths this task produced —
confirmed the concurrent Worker's `data/SRP1/Results/P515S40/clone_preflight/` and
`clone_preflight_launch.log` were never staged).

## Results

**Objective convention, stated once (repeated on every table in the JSON outputs):**
all figures are `gross_operational_cost` as captured in each run's own
`component_levels_terminal.json` (cross-checked bit-for-bit against `g_<label>.json`'s
own field in all four runs, `abs_diff = 0.0` exactly). `gross_operational_cost` equals
`net_operational_recourse` in all four runs (`terminal_salvage_value` is ~1e-36 to
~1e-50, i.e. zero to machine precision — no salvage reinstated on any of these runs);
this is *verified*, not assumed, in the output. `gross_operational_cost` includes the
five category-D detector penalties (per Addendum 10 they stay in the solver objective);
the manuscript's `economic_recourse_all_D_excluded` convention is also carried in the
output and differs from `gross_operational_cost` by `detector_penalty_total`
(≈ −12,801.47, numerically stable to <0.001 across all four runs — it does not affect
which components carry the differences).

**All four runs (run1, A, C, D) cost the SAME candidate/instance** (0.96875 MVA /
3.875 MWh, uniform across nodes 5/7/9, invested 2025 — verified identical across all
four `g_<label>.json['instance']` fields) **under FOUR DIFFERENT ADMM configurations**
(run1: τ=1, balancing on all channels, freeze-after-cycle-30, cap 500, certified cycle
477; A: τ=0, PF/ESS balancing off, cap 300, certified cycle 133; C: τ=0, PF balancing
live, ESS fixed/exempt, cap 300, certified cycle 131/stopped 191; D: C + two-phase ESS
schedule, certified cycle 125/stopped 139 — the adopted oracle). Every table produced
here states this explicitly: **these are a diagnosis of four differently-configured,
differently-stopped runs on one candidate, not a candidate comparison.**

### Task A — cost decomposition

**Reconciliation** (per run): `priced_component_sum + detector_penalty_total` reproduces
the reported `gross_operational_cost` to a residual of **1.2e-7 to 2.4e-7** in every run
(float noise, not a discrepancy — nothing unaccounted). The 48-block sum of each
component's `weighted` value reproduces `totals_weighted` **exactly** (residual `0.0`,
every key, every run) — i.e. every component in `totals_weighted` (generation_cost,
flexibility_cost_internal, the five detector slacks, etc.) is per-block-summable; only
`interface_settlement_tso/dso` (per-DSO-node, not per-block) and `terminal_salvage_value`
are system-level-only, and both are excluded from `gross_operational_cost` by
construction (row-3′, Addendum 12) — confirmed by the reconciliation residual above.

**Headline diffs (arm − run1) reproduced exactly** from the task statement:
A = −39,148.97, C = −57,743.05, D = −72,190.74, C−D = +14,447.69.

**What carries the differences — two components account for the entire difference
to <0.001, in every arm:**

| vs run1 | headline diff | generation_cost + flexibility_cost_internal | other priced | detector_penalty_total Δ | unaccounted |
|---|---:|---:|---:|---:|---:|
| A | −39,148.97 | −39,148.97 | 0.00 | −0.0003 | 0.00 |
| C | −57,743.05 | −57,743.05 | 0.00 | −0.0004 | 0.00 |
| D | −72,190.74 | −72,190.74 | 0.00 | −0.0005 | 0.00 |
| C−D | +14,447.69 | +14,447.69 | 0.00 | ~0 | 0.00 |

Every other component present in the file (`load_curtailment_cost`,
`res_curtailment_penalty`, `ess_usage_cost`, `flexibility_cost_tso_adn_interface_definitional`,
the four "orphan" slacks) is **exactly 0.0 in every run** — not a source of the
difference, confirmed rather than assumed. `ess_complementarity_bilinear_definitional`
and `res_curtailment_definitional_at_weight_1` are **pure reporting/detector fields at
weight 1, not part of `gross_operational_cost` at all** (they move by only a few units
across arms, e.g. 8.0→16.5, immaterial against the ~40–72k differences).

**`generation_cost` is 100% TSO** (DSOs carry zero generation cost in this case file);
**`flexibility_cost_internal` is 100% DSO** — confirmed by the per-block aggregation
(`by_kind`). Per-DSO-node breakdown of the flexibility-cost difference shows it is
**not concentrated at node 7**: e.g. D vs run1, node 5 = −2,963, node 7 = +19,986,
node 9 = +10,692 (node 7 is the largest single mover for D, but not exclusively so —
comparable magnitudes appear at nodes 5 and 9 across the four runs, and the sign
even flips between A and C/D at node 5). Per-year breakdown (generation_cost) shows the
2025/2030/2035 contributions are all of comparable magnitude (no single year dominates).

**C and D agree in sign but not magnitude**: both `generation_cost` and
`flexibility_cost_internal` diffs vs run1 are negative for both C and D
(cheaper on both components), but their absolute sizes differ (e.g.
`flexibility_cost_internal`: C = −3,038, D = +27,715 vs run1 — **opposite sign** for
this one component even though the *headline* totals both come out cheaper than run1,
because `generation_cost` moves the other way for each and dominates differently:
C's generation_cost diff is +60,781 while D's is +44,476). **This is a genuine finding,
not noise**: C and D reach their (both cheaper-than-run1) totals through different
splits between the TSO's generation cost and the DSOs' flexibility cost — consistent
with them being different, incompletely-converged ADMM trajectories on the same
candidate, not the same point reached twice.

**Error bars (rule nine/ten), independently computed and cross-checked:** using the
same construction the s39 evaluator already used for C/D (`bar = arm's own max
|Δgross_operational_cost| over its last 10 cycles + run1's`, run1's own value =
256.26), this script's independent recomputation from `g_<label>.json['cycle_trajectory']`
**exactly reproduces** the s39 evaluator's own numbers for C (bar 734.22) and D
(bar 8,154.89) — `agrees: true` in both cases, cross-check in the JSON. For A (no
pre-existing evaluator bar), this script computed it fresh: arm step 2,883.88, bar
3,140.14. **All three differences are outside their bar** (A: 39,149 vs bar 3,140;
C: 57,743 vs bar 734; D: 72,191 vs bar 8,155) — i.e. **determinate, not stopping
slack**, consistent with the CLAUDE.md rule-ten refinement (a difference must clear
the bar to be reported as more than slack; here every one of the three does, by
roughly an order of magnitude or more).

Full per-component, per-block (all 48 blocks × all runs), per-node, per-year detail is
in `data/SRP1/Results/P515S40/cost_decomposition/cost_decomposition.json`.

### Task B — node 7 cross-check and RESULT table

**Cross-check (quantified, not asserted) — the association the prior worker flagged as
"consistent with, though does not prove" does NOT hold up once quantified:**

For every arm (A, B, C, D), at the stop cycle, ranking node 7's 576 PF-residual entries
(2 power types × 288 period-slots) by `s²` and mapping the top-25 / top-10% to
period-slots, against the period-slots at ≥99%/≥95% of the 100 MVA interface rating
(22–23 / 24–27 of 288 period-slots, depending on arm):

| arm | top25 vs ≥99% | top25 vs ≥95% | top10% vs ≥99% | top10% vs ≥95% |
|---|---|---|---|---|
| A | 0/25 coincide (expected 1.91, enrichment 0.0) | 1/25 (expected 2.17, enrichment 0.46) | 0/57 (expected 4.35, enrichment 0.0) | 1/57 (expected 4.95, enrichment 0.20) |
| B | 0/25 (expected 1.91, enrichment 0.0) | 0/25 (expected 2.08, enrichment 0.0) | 0/58 (expected 4.43, enrichment 0.0) | 0/58 (expected 4.83, enrichment 0.0) |
| C | 0/25 (expected 2.00, enrichment 0.0) | 1/25 (expected 2.34, enrichment 0.43) | 0/58 (expected 4.63, enrichment 0.0) | 1/58 (expected 5.44, enrichment 0.18) |
| D | 0/25 (expected 2.00, enrichment 0.0) | 2/25 (expected 2.34, enrichment 0.85) | 0/57 (expected 4.55, enrichment 0.0) | 2/57 (expected 5.34, enrichment 0.37) |

**Every enrichment value is ≤1** (at or below what independence would predict; ≥99%
coincidence is exactly 0 in every arm). The top-1 entry in A/B (`2030, Spring, period 9`,
matching the prior report) sits at 97.1% utilization — just inside the ≥95% band but
short of ≥99% — but ranks 2 and 3 (91.9%, 94.9%) fall *outside* even the looser band.
**Reading: the PF consensus residual's magnitude and the interface's closeness to its
physical rating are not associated at node 7, at any arm's stop cycle, at either
threshold or either top-set size tested.** This corrects the earlier "consistent with
(though does not prove)" framing to "tested and not supported" — an observation, not a
proof of absence (only these four arms, these two thresholds, this ranking, were
tested; see Remaining issues).

**2030 concentration:** mild in every arm and every set (fractions 0.32–0.60 vs the
1/3 = 0.333 baseline, enrichment 0.95×–1.8×); the at-rating sets sit close to baseline
(0.95×–1.11×) while the top-s² sets run somewhat higher (1.3×–1.8×) — a much weaker
concentration than the qualitative "2030 dominant" framing in earlier reports implied
once the population is restricted to node 7 alone and expressed as a base-rate ratio.

**RESULT table (Addendum 22's ask):** node 7 is confirmed as the **only** node whose
interface reaches its own rating (nodes 5, 9: 0/288 periods ≥95% or ≥99%, in every
arm). At node 7 (D, the oracle): **23/288 periods ≥99%, 27/288 ≥95%** of the 100 MVA
rating (C: 23/27; A: 22/25; B: 22/24 — stable across arms). **Of those binding periods,
only 1 (≥99% interface) to 2 (≥95% interface) coincide with the shared storage itself
at ≥99% of its own 0.96875 MVA rating**, out of 22–27 — e.g. for D, the two coincident
periods are `(2030, Summer, 14)` and `(2035, Autumn, 11)`, both with the storage
**charging** at ~0.9997–0.9999 of its own rating while the interface sits at
98.5%/99.6% of its rating. In the large majority of node-7's binding interface periods
(21–25 of 23–27, depending on arm/threshold), the storage is dispatching at a small
fraction of its own rating (often <1%) even while the interface itself is at/near its
rating. **Reading: the storage's own-rating saturation coincides with the interface's
rating-binding periods in only a small minority of those periods — the "storage's
congestion-relief value" is observable but numerically modest by this specific
coincidence measure**, smaller than the framing in Addendum 22 might suggest taken on
its own; the full binding-period detail (all 22–27 rows per arm, P/Q/|S|/rating/
utilization/storage dispatch/mode) is in each arm's
`node7_crosscheck_<label>.json` under `result_table_by_node`.

Sign convention used throughout: `es_pnet = pch_per_unit − pdch_per_unit`
(`shared_energy_storage_data.py:723`), so **positive `storage_p_z_mw` = net charge**,
negative = net discharge; both coincident periods above are charging, not discharging
(i.e. the storage is absorbing power at node 7 while the interface is near its import/
export limit in the same period, not relieving it by discharging — stated as an
observation; interpreting the direction against the interface's own P sign was not
requested and is left to the Planner).

## Validation

- **Zero-solve claim enforced, not asserted**, in both scripts: `SolveProfileGuard
  (permitted=())` installed before any input file is touched; `guard.verify(0)`
  checked before any output write; both real runs printed `SolveProfileGuard.verify(0):
  OK`.
- **Cross-checks that passed:** Task A's `gross_operational_cost` reproduces
  `g_<label>.json`'s own field bit-for-bit (`abs_diff = 0.0`) in all four runs; the
  independently-recomputed rule-nine/ten bar for C and D matches the s39 evaluator's
  own precomputed bar exactly (`agrees: true`); the 48-block sum reproduces
  `totals_weighted` exactly for every component and every run. Task B's node-7 entry
  count (576) and top-1 entry (`2030, Spring, period 9`, s=2.287e-4) for arm A match
  the prior worker's report exactly.
- **What this validates vs what it does not:** both scripts correctly read and
  cross-check already-committed artifacts (code executes correctly, diagnostic works,
  reconciliation holds to float precision). Task A's decomposition is a bookkeeping
  identity (verified, not a physical claim) — it says *which accounting components*
  carry the differences, not *why* the ADMM trajectories differ there. Task B's
  cross-check is an association test on one ranking/threshold choice at one cycle per
  arm; it does not establish that no relationship exists between PF residuals and
  interface rating more generally (see Remaining issues).
- **Limitations:** (i) Task A's per-block-vs-node-vs-year breakdown is limited to the
  two dominant components (full detail for all 19 component keys × 48 blocks × 4 runs
  is in the JSON, but the printed summary focuses on the two that matter); (ii) Task
  B's "top-s²" population is restricted to node 7's own 576 entries (not the full
  1728-entry stride), a scoping choice stated in the script's docstring and JSON
  `methodology_note`; (iii) the RESULT table's "binding periods" list is capped at
  ≥95% of rating (27 rows max per arm at node 7) — this was the natural cutoff given
  Addendum 22's "at rating" framing, not exhaustive of all 288 periods (full utilization
  detail for the ≥95% set only; the raw per-period P/Q for the remaining periods is in
  `pf_entry_stride_<label>.jsonl`, hash-recorded but not extracted further here).

## Unexpected findings

- **The zero-solve quantification contradicts the earlier qualitative reading.** The
  prior worker's report (`WORKER_REPORT_S39_NODE7.md`, v9 arms A/B) stated the
  interface-rating mechanism was "consistent with (though does not prove)" the PF
  dual-residual concentration at node 7. Tested directly here (four arms, two
  thresholds, two top-set sizes), the coincidence is at or below the rate independence
  would predict in **every one of the 16 arm×threshold×top-set combinations** — the
  top-s² entries are, if anything, mildly *under*-represented among the at-rating
  periods, not over-represented. This should be flagged to the Planner before it is
  carried further as support for the node-7 siting mechanism in the manuscript.
- **C and D reach similar (both cheaper-than-run1) totals through opposite-signed
  moves in `flexibility_cost_internal`** (C: −3,038 vs run1; D: +27,715 vs run1), with
  `generation_cost` compensating in the other direction in each case. This is evidence
  the two arms are different, still-moving ADMM trajectories, not two near-identical
  near-converged points — consistent with both differences being outside their rule-
  nine/ten bar (i.e., determinate, not stopping slack) but also consistent with neither
  arm having reached a single common limit yet.
- **The storage's own-rating saturation and the interface's rating-binding periods
  coincide only rarely** (1–2 of 22–27 periods, all arms) even though both are common
  events individually (storage ≥99% of its own rating in 52–64 of 288 periods per node,
  per the prior S39 report; interface ≥99% in 22–23 of 288 at node 7). This is worth
  flagging alongside the cross-check finding above: neither the PF-residual mechanism
  nor the storage-saturation mechanism shows the tight coincidence with the interface
  rating that the "congestion relief" framing implies at the single-period level, even
  though node 7 remains the only node whose interface reaches its rating at all.

## Remaining issues

- The cross-check above tests one specific operationalization (rank by `s²`, node-7-only
  population, thresholds 0.99/0.95, top-25/top-10%). A different ranking (e.g., `|s|`
  trend/persistence over a window, as the prior worker's monotonicity analysis did) or a
  different population (all-node top entries) might show a different association; this
  was not tested and would need explicit authorization if the Planner wants it tightened
  further.
- The RESULT table's storage-coincidence count (1–2 of 22–27) is a point-in-time
  (single-period) coincidence test; it does not test whether the storage is providing
  congestion relief in a broader sense (e.g., shifting energy into/out of adjacent
  periods around a binding one) — that would need a different, likely non-zero-solve,
  analysis.
- Per Addendum 22's own framing, "the interface rating is NOT changed for convergence" —
  this report does not touch the case file or propose any change; both findings above
  are reported as observations for the Planner's and author's use in the manuscript,
  not as recommendations.

## Questions for Planner

1. The node-7/PF-residual coincidence finding (near-zero or sub-independence enrichment
   in every arm) revises the prior worker's "consistent with, does not prove" reading
   toward "tested, not supported." Should this be carried into the manuscript's
   node-7 section as stated here, or would you like the alternative rankings/populations
   in "Remaining issues" tried before it is treated as settled?
2. The storage-own-rating / interface-at-rating coincidence is small (1–2 of 22–27
   periods). Is this the granularity you want in the manuscript's RESULT table, or
   should the "congestion relief" claim be scoped down to "node 7 is the only node
   whose interface reaches its rating, and the storage is present there" without the
   period-level coincidence claim?
3. Task A's finding that C and D reach their totals via opposite-signed
   `flexibility_cost_internal` moves (while generation_cost compensates) — is this
   relevant to how Step 3 closure characterizes D as "the" oracle, or is it expected
   variation between two different, still-descending ADMM configurations at the same
   candidate?

## Evidence

- Scripts: `p515_s40_cost_decomposition.py` (commit `11075352`),
  `p515_s40_node7_crosscheck.py` (commit `e4db9033`).
- Outputs: `data/SRP1/Results/P515S40/cost_decomposition/` and
  `data/SRP1/Results/P515S40/node7_result/{s38_A_tau0,s38_B_pfbal,s39_C,s39_D}/`
  (commit `d73a2b22`), each with its own `evidence_manifest_sha256.json`.
- This report: `WORKER_REPORT_S40_ANALYSES.md`.
