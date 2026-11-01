# Worker Report — P5.15 Addendum 20 item (a): zero-solve balancing-rule replay

## Task received

Planner task "W2 item (a)": zero-solve replay of the residual-balancing rule on the saved
trajectories — would `rho_pf` have been lowered without the cycle-60 backstop, and when?
Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 20 and frozen spec v9
`data/SRP1/Results/P515S38/frozen_s38_pf_pace_spec_v9_7a2b4ab7.json` (both read-only, neither
staged). No production, case-file, or harness edits; no solves.

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` (Addenda 14–20, especially 19–20)
- `data/SRP1/Results/P515S38/frozen_s38_pf_pace_spec_v9_7a2b4ab7.json`
- `shared_resources_planning.py:6581-6923` (`_update_admm_penalties`, full docstring and body,
  `_get_admm_penalty_summary`, `_get_admm_gamma_summary`, `_init_admm_freeze_state`,
  `_admm_rho_at_clamp`, `_scale_admm_penalty`)
- `admm_parameters.py` (`ADMMParameters`, `read_parameters_from_file`)
- `data/SRP1/SRP1_params.json` (`admm` block)
- `data/SRP1/Results/P515S35_REF_run/g_baseline.json` (`cycle_trajectory`, 477 cycles)
- `data/SRP1/Results/P515S37_RHO0P01_run/g_s37_rho0p01.json` (150 cycles)
- `data/SRP1/Results/P515S37_RHO0P001_run/g_s37_rho0p001.json` (150 cycles)
- `p513_solve_profile_guard.py`
- `p515_s36_h1h2_dual_identifiability.py` (convention reference for a mirrored-logic replay;
  not needed here — the real function was drivable directly)

## Files modified / created

- Created `p515_s38_balancing_replay.py` (new script; committed before running,
  commit `07d973f2`, then amended once to fix a mislabeled counterfactual field before the
  final run — see "Unexpected findings").
- Created `data/SRP1/Results/P515S38/balancing_replay/validation.json`
- Created `data/SRP1/Results/P515S38/balancing_replay/counterfactual.json`
- Created `data/SRP1/Results/P515S38/balancing_replay/run.log`
- Created `data/SRP1/Results/P515S38/balancing_replay/evidence_manifest_sha256.json`
- This report.

No production file, case file, `p515_g_g1_g4_admm_gates.py`, or any s37/s38 evaluator/preflight
file was touched.

## Method: the real production function, not a reimplementation

`p515_s38_balancing_replay.py` imports and calls the actual
`shared_resources_planning._update_admm_penalties` (verbatim, unmodified) and
`_init_admm_freeze_state`. It is driven by:

- **Real `pyomo.environ.ConcreteModel` stand-ins** — one TSO model (`rho_v`, `rho_pf`, `rho_ess`,
  `prox_gamma_v/pf/ess`, all mutable `Param`s), one DSO model (`rho_v/pf/ess`), one ESSO model
  (`rho`). `_get_admm_penalty_summary`/`_get_admm_gamma_summary` only average across whatever
  models are supplied, so one of each, holding the trajectory's own recorded values, is
  sufficient and reproduces bit-identical output to N of each with identical rho.
- **Real `admm_parameters.ADMMParameters`**, built via its own `read_parameters_from_file` from a
  deep copy of `data/SRP1/SRP1_params.json`'s `admm` block, varying only
  `penalty_update.balancing_exempt_channels` (`[]` for run 1, `['ess']` for the two s37 arms —
  each cross-checked at cycle 1 against the trajectory's own `balancing_exempt_{v,pf,ess}`
  fields, which agreed) and, for the counterfactual only, `penalty_update.freeze_backstop_cycle`
  (60 → 200; `freeze_after_unchanged_cycles` stays at the case file's 10, already Addendum 20's
  value).
- **`boyd_metrics`/`residual_metrics` dicts rebuilt per cycle** directly from each run's own
  `cycle_trajectory` entries (`boyd_{group}_r/s/eps_pri/eps_dual/primal_ratio/dual_ratio/
  dual_ratio_balance`, `primal_{group}[_mean]`, `dual_{group}_mean`) — the same values the
  production ADMM loop computed and serialized.
- **Real `freeze_state`**, carried cycle-to-cycle exactly as production carries it.

No line of `_update_admm_penalties`'s decision logic is reimplemented anywhere in the replay
script; only the state it reads and writes is stood up around it. `SolveProfileGuard([])` is
installed for the whole script (empty permitted list) and `verify(expected_solves=0)` is asserted
before any output is written.

## Commands / experiments run

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s38_balancing_replay.py \
    > data/SRP1/Results/P515S38/balancing_replay/run.log 2>&1
```
Exit code 0 both times (initial run and the re-run after the field-label fix described below).

## Results

### 1. VALIDATION verdict — PASSED, all three runs, every cycle, every channel

| run | cycles | n_mismatches | passed |
|---|---|---|---|
| run1_s35ref | 477 | 0 | True |
| s37_rho0p01 | 150 | 0 | True |
| s37_rho0p001 | 150 | 0 | True |

For every cycle and every channel (`v`, `pf`, `ess`) of all three trajectories, replaying the
real `_update_admm_penalties` against the trajectory's own recorded `rho_{c}_before`/
`gamma_{c}_before` reproduced the trajectory's own recorded `rho_{c}_action`, `rho_{c}_after`,
`gamma_{c}_after`, `rho_frozen_{c}`, `rho_unchanged_streak_{c}` and `rho_at_clamp_{c}` exactly
(rho/gamma compared with `rel_tol=1e-9, abs_tol=1e-12`; everything else by exact equality). Zero
mismatches (`data/SRP1/Results/P515S38/balancing_replay/validation.json`,
`overall_passed: true`). The counterfactual below is therefore **trusted**.

### 2. COUNTERFACTUAL first-action table (Addendum 20 freeze rule: unchanged-10 + absolute-200)

PF channel (the question asked):

| run | first PF action | cycle | rho before → after | primal_ratio | dual_ratio_balance | ratio (dual/primal) |
|---|---|---|---|---|---|---|
| run1_s35ref | decreased | **69** | 0.198 → 0.132 | 1.4739 | 5.0589 | 3.432 |
| s37_rho0p01 | decreased | **79** | 0.198 → 0.132 | 0.9403 | 4.4512 | 4.734 |
| s37_rho0p001 | decreased | **103** | 0.198 → 0.132 | 0.8849 | 3.1689 | 3.581 |

The first PF action in every run is a **decrease** (there is no prior increase to report — PF's
`primal_ratio` never exceeds `5×dual_ratio_balance` in these trajectories at any cycle checked).
All three fire the PF decrease well before run 1's real PF first-pass cycle (226) and well inside
the cap-300 budget. This directly answers the task question: **yes, `rho_pf` would have been
lowered without the cycle-60 backstop**, at cycle 69 (run 1), 79 (s37 rho_ess=0.01) and 103 (s37
rho_ess=0.001) respectively — 61–164 cycles earlier than run 1's actual PF first-pass.

V and ESS channels (for completeness — these are the "not-a-question but reported for context"
channels; ESS is exempt in the s37 arms):

| run | V first action | cycle | ESS first action | cycle |
|---|---|---|---|---|
| run1_s35ref | increased | 1 | increased | 1 |
| s37_rho0p01 | increased | 1 | exempt (fixed) — n/a | — |
| s37_rho0p001 | increased | 1 | exempt (fixed) — n/a | — |

Freeze state at the end of each replayed trajectory (open-loop; PF channel):

| run | terminal cycle used | PF frozen at that cycle | PF action at that cycle | PF rho at that cycle | cycle 200 reached by this trajectory? | PF frozen at cycle 200 |
|---|---|---|---|---|---|---|
| run1_s35ref | 477 | True | `held (frozen after 10 unchanged cycles)` | 0.0001 (clamp) | yes | **True** |
| s37_rho0p01 | 150 | False | `decreased` (still active) | 0.0001 (clamp) | **no** | not verifiable — see caveat |
| s37_rho0p001 | 150 | True | `held (frozen after 10 unchanged cycles)` | 0.0391 | **no** | not verifiable — see caveat |

**Scoped claim:** the two s37 trajectories only run to cycle 150, so "frozen at cycle 200" is
**not observable** from them — the script reports `cycle_200_reached_by_this_trajectory: false`
and `frozen_at_cycle_200_open_loop: null` for both, rather than extrapolating. Within the
available 150 cycles, run 1's PF channel reaches the rho floor (1e-4) and freezes by cycle 477;
s37_rho0p001's PF channel also reaches freeze (at a higher rho, 0.0391) by cycle 150; s37_rho0p01's
PF channel is still actively decreasing (already at the 1e-4 floor, still issuing `decreased`
actions cycle to cycle since the floor clip does not itself count as "unchanged" — see limitation
below) and has not frozen within the observed window.

Subsequent open-loop firings after the first PF decrease (NOT predictive — see caveat in
`counterfactual.json`): run1_s35ref 36, s37_rho0p01 59, s37_rho0p001 3.

### 3. PF ratio series (`dual_ratio_balance / primal_ratio`) at matched cycles

Full table for all three runs is in `counterfactual.json['pf_ratio_series']`. Run 1 (representative):

| cycle | primal_ratio | dual_ratio_balance | ratio |
|---|---|---|---|
| 1 | 2513.82 | 1966.04 | 0.782 |
| 2 | 1980.31 | 1077.44 | 0.544 |
| 5 | 338.39 | 491.06 | 1.451 |
| 10 | 132.66 | 116.11 | 0.875 |
| 20 | 31.27 | 31.58 | 1.010 |
| 30 | 10.44 | 17.66 | 1.692 |
| 50 | 6.56 | 7.91 | 1.207 |
| 60 | 3.13 | 6.31 | 2.015 |
| 75 | 1.43 | 4.63 | 3.230 |
| 100 | 0.632 | 3.473 | 5.498 |
| 125 | 0.582 | 2.491 | 4.282 |
| 150 | 0.511 | 1.929 | 3.775 |
| 200 | 0.216 | 0.887 | 4.111 |
| 226 | 0.129 | 0.704 | 5.445 |
| 300 | 0.744 | 0.365 | 0.491 |
| 400 | 0.285 | 0.118 | 0.414 |
| 477 | 0.004 | 0.064 | 15.789 |

First cycle the ratio exceeds 3.0 (PF decrease threshold): **cycle 69** (ratio 3.432) — consistent
with, and the direct cause of, the counterfactual's first PF decrease at cycle 69. First cycle it
exceeds 5.0: **cycle 88** (ratio 5.554). The s37 arms show the same qualitative shape shifted
later (first-exceeds-3.0 at cycle 79 and 103 respectively, matching their first-decrease cycles
exactly, by construction — the decrease fires the instant the ratio crosses the threshold).

## Validation

- Method-level: validated the replay against the real trajectories under the actual
  configuration before running any counterfactual, per the task's mandatory ordering. Zero
  mismatches across 477 + 150 + 150 = 777 cycles × 3 channels = 2,331 channel-cycles.
- `SolveProfileGuard([]).verify(expected_solves=0)` returned no failures (`permitted_solve=0`,
  `permitted_exec=0`, `blocked_solve=0`, `blocked_exec=0`) — printed at the end of `run.log`:
  `SolveProfileGuard: 0 solves, 0 execs -- verified.`
- Internal cross-check: the counterfactual's `pf_first_decrease` cycle in each run coincides
  exactly with the PF ratio series' `first_cycle_ratio_exceeds_3p0` cycle (69/79/103 in both), as
  it must given the production decision rule (`boyd_dual_ratio_balance > decrease_balance_ratio *
  boyd_primal_ratio`, i.e. `ratio > 3.0` for PF).
- Distinguishing what was established: (a) the replay code executes correctly and reproduces the
  real function's output exactly on the actual configuration (validation PASSED); (b) the
  counterfactual is a legitimate deterministic replay of the real decision rule under a changed
  freeze parameter, given the OBSERVED (not re-solved) residual ratios — it is not, and cannot be,
  a claim about what a real re-run with an earlier-firing balancer would produce, because the
  residual ratios themselves would differ once rho differs (open-loop limitation, stated
  explicitly in every output and below).

## Unexpected findings

- First draft of the counterfactual computed `frozen_by_200_open_loop` by indexing
  `per_cycle_records[min(199, len-1)]`, which silently fell back to the **last available cycle**
  for the two 150-cycle s37 trajectories and mislabeled it as "by 200" — a scoped-claim violation
  (CLAUDE.md rule seven) caught before committing any output. Fixed to report
  `cycle_200_reached_by_this_trajectory` explicitly and return `None` (not a guess) when cycle 200
  is outside the trajectory; the stale run.log/validation.json/counterfactual.json from the first
  run were deleted (never committed, never cited by any report) and regenerated. The fix and the
  final run are what is committed here.
- s37_rho0p01's PF channel is still issuing `decreased` actions at cycle 150 even though rho is
  already at the configured floor (`penalty_update['min'] = 1e-4`) — `_scale_admm_penalty` clips
  the value but the caller still records the action label as `'decreased'` (a real balancing
  decision was taken; the clip is a separate, silent floor). This means the per-channel
  "unchanged streak" freeze trigger never engages at the floor for this arm within the observed
  window (streak keeps resetting to 0 on every `'decreased'` action, floor-clipped or not) — worth
  flagging to the Planner as a possible open item for whoever designs a production freeze rule
  meant to also catch "the ratio is still nominally imbalanced but rho has nowhere further to go".
  Not fixed here (out of scope: diagnostic-only zero-solve replay).

## Remaining issues

- None affecting the validated claims. The counterfactual's "subsequent open-loop firings" are, by
  design and by the task's own instruction, not predictive and are reported only as such.
- Whether an arm run with the Addendum 20 freeze rule live from cycle 1 (not a replay) would reach
  PF first-pass materially before cycle 180 (the frozen-spec-v9 "helps" criterion) is a question
  for the actual arm runs (s38_A_tau0 / s38_B_pfbal), not for this zero-solve replay — this replay
  only establishes that the balancing rule itself would have fired far earlier than cycle 226 had
  the backstop not suppressed it, which is a necessary but not sufficient condition for an arm to
  help.

## Questions for Planner

None — the task's method, validation and counterfactual specification were unambiguous and are
fully answered by the results above.

## Evidence

- Script: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_s38_balancing_replay.py`
- Outputs: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515S38/balancing_replay/{validation.json,counterfactual.json,run.log,evidence_manifest_sha256.json}`
- This report: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/WORKER_REPORT_S38_REPLAY.md`
