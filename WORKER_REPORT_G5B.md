# Worker Report G5-B — Physical-Variable Discriminator for the Failed Gate G5

## Verdict: I2 supported (per the predeclared rule), with an important qualification

Per the rule declared in advance ("if any physical variable disagrees by more than 1e-6
relative -> I2 supported, and that is a genuine finding regardless of what the slack
arithmetic shows"): **`es_soh_per_unit_cumul`, `es_D_per_unit`, and
`es_avg_ch_dch_per_unit` disagree between A1 and A3 (and between A3 and A4) by up to
7.633e-05 relative on BOTH node 7 and node 9** — 76x the 1e-6 threshold. The physical
variables decide, so the verdict is **I2**.

**Qualification that must be reported alongside the verdict, not instead of it:** the
disagreement is confined to the SoH/degradation family. The quantity that actually
represents what the ESSO *does* with real power — the per-cohort **net power**
`es_pch_per_unit - es_pdch_per_unit`, and the control `es_pnet` — is **identical across
all three arms to 1e-16 relative** (floating-point noise) on both nodes. Section 3 below
gives the mechanism: the SoH/D/avg_ch_dch disagreement is fully consistent with
(the algebraic constraints make it a **necessary** consequence of) the same mu_strategy-
dependent complementarity leak G5 already diagnosed in the objective, propagating through
a **different** accounting path than the objective term does. I did not independently
verify this to full numerical precision (I did not capture raw individual `pch`/`pdch`,
only their difference) — flagged as a limitation. The verdict itself does not depend on
this mechanism; it is reported under the literal rule as stated.

## 1. Physical-variable comparison table (A1 vs A3, A1 vs A4, A3 vs A4; nodes 7 and 9 — IDENTICAL results on both nodes since it is the same problem instance)

Relative-difference convention: `max(abs_diff) / max(abs(a), abs(b), floor)`, floor = 1e-300 (raw,
matches G5's own convention) and floor = 1e-9 (to show the raw number is not an artifact of
near-zero denominators) — **both floors gave the identical result for every family below; there
was no floor-dependence.**

| variable family | A1 vs A3 max\|Δ\| | A1 vs A3 max rel | A1 vs A4 max rel | A3 vs A4 max rel | agrees to 1e-6? |
|---|---|---|---|---|---|
| `es_soh_per_unit_cumul` (cohort-year) | 1.639e-05 | **2.182e-05** | 1.993e-12 | 2.182e-05 | **NO** |
| `es_D_per_unit` (cohort-year) | 7.286e-06 | **7.633e-05** | 5.655e-12 | 7.633e-05 | **NO** |
| `es_avg_ch_dch_per_unit` (cohort-year) | 1.843e-04 | **7.633e-05** | 5.655e-12 | 7.633e-05 | **NO** |
| net power `pch - pdch` (cohort-period, 864 keys) | 4.163e-17 | 4.163e-16 | 2.776e-16 | 4.163e-16 | yes |
| `es_pnet` (control, 288 keys) | 0.0 | 0.0 | 0.0 | 0.0 | yes (exact, as expected — it is fixed) |

Argmax location for the three disagreeing families (both nodes): `(y_inv=0, y=2)` — the
**active** investment cohort at its terminal (3rd) year, i.e. the cumulative,
most-compounded quantity, not an inactive-cohort placeholder:

- `es_D_per_unit[0,2]`: A1 = 0.09543661056307962, A3 = 0.09544389614008718 (Δ = 7.286e-06)
- `es_avg_ch_dch_per_unit[0,2]`: A1 = 2.4142189550221365, A3 = 2.4144032551351238 (Δ = 1.843e-04)
- `es_soh_per_unit_cumul[0,2]`: A1 = 0.7510298862476065, A3 = 0.7510134995190723 (Δ = 1.639e-05)

A1 vs A4 agrees to ~1e-12 relative on every family (consistent with G5's own objective
finding, `|A1-A4|/|A1| = 5.37e-12` on the objective) — A4 (cold start) reproduces A1 (warm
start, IPOPT defaults) essentially exactly; only A3 (`mu_strategy=adaptive`) disagrees.

## 2. Mechanism consistent with the disagreement (reported, not asserted as proven)

`es_avg_ch_dch_per_unit[y_inv,y]` is defined (`shared_energy_storage_data.py:578-587`) as
a **weighted sum of `pch` and `pdch`** (`eff_ch*pch*dt + pdch*dt/eff_dch`, both added), not
of `pch - pdch`. `es_D_per_unit` is then an exact linear function of `avg_ch_dch`
(`D * (2*cl_eff*E_investment) == 365*num_years*avg_ch_dch`, confirmed by the identical
7.633e-05 relative diff on both D and avg_ch_dch above — an exact linear pass-through), and
`es_soh_per_unit_cumul` compounds `D` multiplicatively across cohort-years
(`soh[y] = soh[y-1] * exp(-D[y]) * phi_cal**num_years`).

G5 already found A3's `spurious_throughput_measured` (`= 2*sum(min(pch,pdch))`, the
barrier-induced complementarity leak) is **1.84x** A1's (0.004807 vs 0.002612). Since that
leak adds to **both** `pch` and `pdch` symmetrically, it cancels in the net-power
difference (which is identical, Section 1) but does **not** cancel in the `avg_ch_dch` sum
(which adds them). This is a plausible complete explanation of the SoH/D/avg_ch_dch
disagreement as a **propagation of the same already-diagnosed mu_strategy-dependent
leak**, through a different accounting path than the objective. I did **not** capture raw
per-period `pch`/`pdch` individually (only their difference), so I have not verified this
to floating-point precision the way Section 4 verifies the objective closure — this is a
plausible, structurally-supported hypothesis, not an independently confirmed identity.
Regardless of the mechanism, the disagreement is real and exceeds 1e-6 relative, so the
verdict in the header stands.

## 3. Slack barrier-identity arithmetic test (predeclared)

Predicted (per the task's formula `min(slack_up,slack_down) = mu_final/(2*s_obj*PENALTY_ESSO_SLACK)`,
`PENALTY_ESSO_SLACK = 1e3` confirmed in `definitions.py`):

| arm | mu_final | s_obj | **predicted** min(up,down) | **measured** mean min(up,down) | measured/predicted |
|---|---|---|---|---|---|
| A1 | 9.096027172233772e-10 | 0.1 | 4.548013586116886e-12 | **-9.990909100001491e-09** | **-2196.76** |
| A3 | 1.8704114605523745e-09 | 0.1 | 9.352057302761873e-12 | **-9.98326412415239e-09** | **-1067.49** |
| A4 | 9.096063001298259e-10 | 0.1 | 4.548031500649129e-12 | -9.990909100001493e-09 | -2196.75 |

**The predicted value is off by roughly three orders of magnitude, and the measured value
is negative** (the slack variables are `NonNegativeReals` — a value of ~-1e-8 is a bound
violation, not a barrier complementarity residual). The measured slack values are
essentially **constant across all 288 periods** (stdev 1.3e-23 for A1/A4, 3.4e-20 for A3 —
6-9 orders of magnitude below the mean), which is the signature of a **fixed bound
relaxation** (consistent with IPOPT's default `bound_relax_factor = 1e-8`, which shifts
every variable's effective lower bound by `~1e-8 * max(1,|bound|)`), not the
data-dependent, mu-scaled complementarity residual the pch/pdch pair exhibits. **The
predeclared slack barrier-identity test therefore FAILS as a predictive model for this
slack pair** — the slack pair does not exhibit the same barrier residual pch/pdch does; it
is dominated by a different, much larger, largely mu_strategy-independent numerical floor.

## 4. Does the slack term close the objective residual? Two answers, and why they differ

**Using the (failed) closed-form prediction of Section 3** (as the task's own worked
example does, with `N_periods = 288`, confirmed as the exact count of `(y,d,p)` triples
entering the objective's slack-penalty sum):

```
predicted_slack_term_diff = PENALTY_ESSO_SLACK * 2 * (min_A3 - min_A1) * 288
                           = 1e3 * 2 * (9.352057e-12 - 4.548014e-12) * 288
                           = 2.7671e-06
```
against the unexplained residual from the pch/pdch leak alone, **4.4035e-06** (matches the
task's own number exactly) — **ratio 1.591**, matching the task's predeclared "ratio 1.59"
exactly. Since Section 3 already shows the closed-form `predicted_min_up_down` does not
describe the measured slack behavior at all (wrong order of magnitude, wrong sign regime),
**this 1.59 ratio is not a meaningful "how much is left" number** — it is comparing the
unexplained residual against a prediction built from a formula that does not apply to this
slack pair.

**Using the measured slack values directly** (the actual quantities that enter
`model.objective = model.feasibility_penalty = PENALTY_ESSO_SLACK*sum(slack_up+slack_down)
+ EPS_ESSO_THROUGHPUT*sum(pch+pdch)`, confirmed as the ENTIRE objective — no other term is
present, `shared_energy_storage_data.py:757-763`), computed straight from the captured raw
slack arrays (both nodes, identical):

```
slack_penalty(A1) = PENALTY_ESSO_SLACK * sum(slack_up_A1 + slack_down_A1) = -0.005754763636363612
slack_penalty(A3) = PENALTY_ESSO_SLACK * sum(slack_up_A3 + slack_down_A3) = -0.0057503601249833
slack_penalty(A1) - slack_penalty(A3)                                    = -4.4035113803122736e-06

eps_term(A1) - eps_term(A3) = EPS_ESSO_THROUGHPUT * (spurious_A1 - spurious_A3)
                             = -2.1949951560620913e-06   [exact: net power is identical, Sec.1,
                               so throughput differs ONLY by the spurious/leak term]

predicted_total = (slack_penalty diff) + (eps_term diff) = -6.598506536374365e-06
observed objective diff (A1 - A3)                        = -6.5985065364586315e-06

residual = observed - predicted_total = -8.43e-17   (floating-point noise)
```

**The direct, exact accounting closes the ENTIRE A1-vs-A3 objective gap to floating-point
precision (residual 8.4e-17, i.e. exactly zero within solver arithmetic), on both nodes.**
This is expected once the two objective terms are both measured directly rather than one
of them (the slack term) approximated by a closed form that Section 3 shows does not
apply — the objective is an exact algebraic sum of exactly these two terms and nothing
else, so measuring both directly must close it; this is confirmation the objective formula
in the working tree matches what production builds, not new evidence of a mechanism beyond
that. I flag this because the summary JSON's `residual_closure_test_A1_vs_A3_by_node` field
computes the slack contribution via the (invalid) closed-form/mean-based route and has a
sign convention that does **not** match the `observed_diff` convention used there
(`min_A3 - min_A1` is in the A3-vs-A1 direction while `observed_diff` is A1-vs-A3) — do not
read `residual_after_pch_pdch_leak_and_slack_term_using_predicted_mins` in that file as a
closure result; the correct, exact closure is the one shown in this section, reproducible
directly from `g5b_physical_variable_capture_full.json`'s
`physical_raw_by_node_arm.{7,9}.{A1,A3}.slack_es_pnet_{up,down}` and
`arms_full.{7,9}.{A1,A3}.objective` / `.diagnostics.spurious_throughput_measured`.

## 5. Guard

`SolveProfileGuard` armed for the whole run, permitted call site
`shared_energy_storage_data.py:_run_solver_attempt` (identical to G5's own guard).
Declared in advance: **6** (2 nodes x 3 arms — identical solve profile to G5, since this
harness re-solves the same instance under the same three arms and adds no new solves).
Observed: `permitted_solve = 6`, `permitted_exec = 6`, `blocked_solve = 0`,
`blocked_exec = 0`, `verify([]) failures = []`. Exact match, no recovery retries fired on
any arm (`recovery_fired = False` for all 6), all six terminated `optimal`, and all six
objectives reproduced G5's own recorded values exactly (`0.02304784858129412` /
`0.02305444708783058` / `0.023047848581170288` for A1/A3/A4 on both nodes).

## Files inspected

- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_g5_reformulated_gate.py` (G5 harness, imported and reused unmodified)
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/shared_energy_storage_data.py` (variable declarations, objective construction, `_get_esso_complementarity_diagnostics`, `_parse_ipopt_barrier_terms`, `_complementarity_ratio_for_model`)
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/definitions.py` (`PENALTY_ESSO_SLACK = 1e3`, `EPS_ESSO_THROUGHPUT = 1e-3`)
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/helper_functions.py` (`fix_or_set`, `solver_result_succeeded`)
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p513_solve_profile_guard.py`
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_1_esso_reform_smoke.py`
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P5151/g5_gate_summary.json` (existing G5 artifact, read-only, confirmed untouched)

## Files modified

None (production code). New files only:

- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_g5b_physical_variable_capture.py` (new diagnostic harness)
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P5151/g5b_physical_variable_capture_summary.json` (new)
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P5151/g5b_physical_variable_capture_full.json` (new; raw per-arm physical variables + full arm diagnostics)
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P5151/g5b_gate_logs/node{7,9}_{A1,A3,A4}.txt` (new IPOPT logs)
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/WORKER_REPORT_G5B.md` (this report)

No file under `data/` that existed before this run was modified; `g5_gate_summary.json` and
`g5_gate_full.json` (G5's own artifacts) were asserted present and untouched by this
harness's preflight, and confirmed by MD5 to be unchanged after the run.

## Changes made

Wrote a new diagnostic harness, `p515_g5b_physical_variable_capture.py`, that:

1. Imports `p515_g5_reformulated_gate.py` (`G5`) and calls its
   `_build_fresh_node_models()` and `_run_arm()` **unmodified** — same fixture
   construction, same `ARM_SPECS`, same nodes, same option overrides. `p515_g5_reformulated_gate.py`
   itself was never edited.
2. Since `_run_arm` returns diagnostics but not the solved model object, and the model
   object is needed for physical-variable extraction, the harness temporarily wraps
   `SED._optimize` (the module attribute `_run_arm` already looks up at call time) with a
   thin recorder that calls straight through to the real, unmodified `SED._optimize` and
   stashes the model reference; restored via `finally`. No change to `_optimize`'s or
   `_run_arm`'s code, arguments, or options.
3. Redirects `_run_arm`'s IPOPT log output to a new directory (`g5b_gate_logs`) by
   overriding `G5.LOG_ROOT` before calling `_run_arm` — `_run_arm` reads that global at
   call time regardless, this only changes where its own output lands; restored in
   `finally`.
4. A preflight (`_assert_capture_paths_exist`) checks, before any solve: `SED.ESSO_TOL_OVERRIDES`,
   `SED.PENALTY_ESSO_SLACK == 1e3`, the mu_final/s_obj log parser on a pre-existing
   reference log, and — new for this task — that every required physical-variable Var
   (`es_soh_per_unit_cumul`, `es_D_per_unit`, `es_avg_ch_dch_per_unit`, `es_pch_per_unit`,
   `es_pdch_per_unit`, `es_pnet`, `slack_es_pnet_up`, `slack_es_pnet_down`) exists on the
   freshly built (unsolved) per-node model, and that none of this harness's own output
   paths pre-exist.
5. Arms a `SolveProfileGuard` (identical permitted call site to G5) with declared count 6,
   verified exactly.
6. Extracts, per arm/node, from the solved model: `es_soh_per_unit_cumul`,
   `es_D_per_unit`, `es_avg_ch_dch_per_unit` per `(y_inv, y)`; net power
   `es_pch_per_unit - es_pdch_per_unit` per `(y_inv, y, d, p)`; `es_pnet` per `(y, d, p)`;
   `slack_es_pnet_up`/`slack_es_pnet_down`/their per-period min, per `(y, d, p)`.
7. Computes cross-arm max-abs/max-relative-diff tables per variable family, the slack
   barrier-identity test, and (a since-flagged-as-flawed) closed-form residual-closure
   test — see Section 4 for the corrected, exact version derived from the same raw data.

## Commands / experiments run

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -m py_compile p515_g5b_physical_variable_capture.py
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B p515_g5b_physical_variable_capture.py
```
Exit code 0. Guard verify failures: `[]`. All 6 solves `optimal`, no recovery retries.

Post-hoc verification (read-only, no solve, from the already-written
`g5b_physical_variable_capture_full.json`): the exact slack-penalty/eps-throughput
closure arithmetic shown in Section 4, run via a one-off `python3 -c` snippet reading only
that JSON file (no re-solve, no guard implication).

## Results

See Sections 1-5 above. Summary: **I2 per the predeclared rule** (SoH/D/avg_ch_dch
disagree by up to 7.633e-05 relative, 76x the 1e-6 threshold, on both nodes); net power and
`es_pnet` agree to floating-point precision; the predeclared slack barrier-identity closed
form fails as a predictive model for the slack pair (off by ~3 orders of magnitude, wrong
sign regime, dominated by an apparent IPOPT bound-relaxation floor rather than a
mu-scaled complementarity residual); the FULL A1-vs-A3 objective gap is nonetheless closed
exactly (residual 8.4e-17) once the slack term is measured directly rather than predicted
by that closed form.

## Validation

- Guard declared/observed solve count matched exactly (6/6), zero blocked calls.
- All six re-solves reproduced G5's own recorded objectives exactly (bit-identical to the
  values in `g5_gate_summary.json`), confirming this harness solved the SAME instance under
  the SAME arms as G5, not a different one.
- `es_pnet` control confirmed exactly identical (0.0 abs/rel diff) across all three arms —
  the fixed-variable assumption the task's I1 hypothesis relies on is directly confirmed.
- The exact objective closure (Section 4) is an algebraic identity check on the objective's
  own two terms, verified to floating-point precision (8.4e-17) — this validates that the
  captured slack and throughput diagnostics are self-consistent with the recorded
  objectives, not merely plausible.
- G5's own artifacts (`g5_gate_summary.json`, `g5_gate_full.json`) confirmed present
  before the run (preflight) and MD5-unchanged after.
- Did NOT independently verify the Section 2 mechanism hypothesis (avg_ch_dch propagation)
  against raw per-period `pch`/`pdch` values — only the net (`pch - pdch`) was captured, so
  this is reported as a consistent, structurally-supported hypothesis, not a proven
  identity.

## Unexpected findings

- The predeclared slack barrier-identity arithmetic test (`min(slack_up,slack_down) =
  mu_final/(2*s_obj*PENALTY_ESSO_SLACK)`) **does not describe this slack pair's actual
  behavior**: measured `min(slack_up, slack_down)` is negative (~-1e-8, a bound violation
  since the domain is `NonNegativeReals`) and essentially constant across all 288 periods
  (stdev 6-9 orders of magnitude below the mean), consistent with a fixed IPOPT bound
  relaxation (default `bound_relax_factor = 1e-8`) rather than the mu-scaled,
  data-dependent complementarity residual pch/pdch exhibits. The task's own worked
  arithmetic (ratio 1.59) is reproduced exactly from the closed form, but Section 3/4 show
  that arithmetic is comparing against a formula that does not hold for this variable pair;
  the DIRECT measured slack values close the objective residual exactly instead.
- The physical-variable disagreement (SoH/D/avg_ch_dch) is entirely confined to variables
  built additively from `pch + pdch`; the variable that is subtractive (`pch - pdch`, the
  actual net dispatch) is identical to 1e-16 relative. This split is consistent with (not
  independently proven to be caused by) the objective-level complementarity leak G5 already
  identified.
- My own script's `residual_closure_test_A1_vs_A3_by_node` JSON field uses a sign
  convention (`min_A3 - min_A1`, per the task's literal formula) that does not match the
  `observed_diff` (A1-vs-A3) convention used elsewhere in the same record — flagged
  explicitly in Section 4 so it is not misread; the corrected exact closure is given there
  instead of being silently fixed in the JSON (no re-run was performed to avoid burning
  additional declared solves for a labeling issue in a derived field, given the correct
  numbers are already reported in this document and independently reproducible from the
  raw JSON already written).

## Remaining issues

- Whether the SoH/D/avg_ch_dch disagreement should be characterized as "genuine
  nonconvexity" (multiple local optima in the ESSO subproblem) or as "an artifact that
  happens to also perturb a reported-but-objective-irrelevant derived quantity" is a
  characterization question for the Planner; this report supplies the measurements
  (Sections 1-4) needed to make that call but does not make it.
- The Section 2 mechanism (avg_ch_dch propagation) was not verified against raw
  per-period `pch`/`pdch` values; if the Planner wants that confirmed exactly, a further
  capture (re-solving the same 6 arms with `pch`/`pdch` extracted individually, not just
  their difference) would be needed.

## Questions for Planner

None — the task was fully executable with the existing production code and G5 harness;
no production-code change was needed or made.
