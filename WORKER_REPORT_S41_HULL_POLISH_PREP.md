# Worker Report — P5.15 Addendum 23, item (2): interval-hull polish harness, checks, smoke test

**Continues** `WORKER_REPORT_S41_POLISH_PREREQ.md` (Part 1, committed `f5dfcb3f`). Authority:
`PLANNER_BRIEF_2026-09-13.md` Addendum 23, item `item2_hull_polish` of frozen spec v12
`data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json` (`a099c7c5`). The Worker builds
and smoke-tests the harness; the Planner launches the full run (NOT run here, per instruction).

---

## Files inspected (beyond Part 1's list)

- `p56a_oracle.py` full read of `apply_common_values`/`_interface_expression` (`:390-458`) against
  `model_construction_helpers.py:1167-1219` (`interface_pf_p/q_transmission_def`,
  `interface_pf_p/q_distribution_def`) and `shared_resources_planning.py:3560-3628`
  (`create_transmission_network_model`'s interface-fixing block) — the source of the
  `interface_delta_p` finding (§2 below).
- `network.py:285` (`vmag_sqr` Var), `:398-440` (`shared_es_pnet`/`qnet`, `pc_adn`/`qc_adn`/`pg_adn`/
  `qg_adn` Expressions), `:540-608` (`_create_smopf_solver`/`_run_smopf_solver_attempt`),
  `:638-657` (`_snapshot/_clear/_restore_multiplier_suffixes`), `:746-840` (`_run_smopf`, tier-1/
  tier-2 recovery).
- `shared_energy_storage_data.py:36-60` (`SharedEnergyStorageData.years`/`.days`/`build_subproblem`),
  `:439-440` (`es_pnet`/`es_qnet`).
- `shared_resources_planning.py:4264-4280` (the production label→index mapping convention for the
  ESSO subproblem's `years`/`days` Sets — `years = list(shared_ess_data.years); year = years[y]`).
- `p515_s32_zero_solve_checks.py:102-154` (`_build_admm_ready_state`, reused by import).
- `p513_solve_profile_guard.py` (full, `SolveProfileGuard` API).
- `data/SRP1/case9/case9_params.json:33-53` (`solver.options`, confirming `bound_push` etc. `=1e-6`).

## Files created

- `p515_s41_hull_polish.py` — the hull-polish harness.
- `p515_s41_hull_polish_checks.py` — zero-solve checks.
- `data/SRP1/Results/P515S41/hull_polish_checks/results.json` — checks output.
- `data/SRP1/Results/P515S41/hull_polish_smoke/` — smoke-test evidence (48-file manifest).
- This report.

No file in the not-permitted list (`shared_resources_planning.py`, `network.py`, `network_data.py`,
`shared_energy_storage_data.py`, `admm_parameters.py`, `admm_persistent_workers.py`) or
`p515_s40_polish_gap.py` was modified.

---

## 1. Design summary

`p515_s41_hull_polish.py` runs the case-file configuration to certification in-process
(`p515_g_g1_g4_admm_gates.run_admm_arm`, identical invocation to `p515_s40_polish_gap.py`), requires
D's bitwise reproduction via `p515_s40_polish_gap._reproduction_check` (imported unchanged, per the
spec — "reuse its `_reproduction_check`, with the `rule_eleven_checklist` subtree non-gating"), then:

- **Hull construction** (`hull_entries_with_esso`): `p56a_oracle.common_coordinated_values`
  (imported unchanged) gives the TSO/DSO achieved values per (node, year, day, period); this function
  adds the ESSO's own achieved `es_pnet`/`es_qnet` as the third ESS endpoint.
- **Hull application** (`apply_hull_bounds`): for every entry, constrains the corresponding Pyomo
  quantity to `[min, max]` (or fixes it, if degenerate) — see the harness's own docstring for exactly
  which Var/Expression carries each family, with file:line citations. Reused unchanged where a Var
  exists (`vmag_sqr`, `shared_es_pnet`/`qnet`); a **new** `ConstraintList` range row where the
  coordinated quantity is a Pyomo `Expression` (`pc_adn`/`qc_adn`/`pg_adn`/`qg_adn`).
- **Objective switching**: `p515_s40_polish_gap._switch_to_base_objective` (imported unchanged).
- **Solve**: `network.run_smopf(model, params, from_warm_start=False)` (automatic primal warm start
  from the certified block solution; no code resets a Var's `.value`), with
  `bound_push`/`bound_frac`/`slack_bound_push`/`slack_bound_frac` overridden to IPOPT's compiled
  default (`0.01`) on a save/restore-scoped copy of `params.solver_params.options`, and
  `network._clear_multiplier_suffixes` called first (belt-and-suspenders — `from_warm_start=False`
  already prevents IPOPT from being told to import multipliers). Production recovery policy
  (tier-1 cold, tier-2 cold+adaptive) is inherited unchanged from `network._run_smopf`.
- **Gate**: `|Delta|/recourse_before < 0.1%`, `Delta = Σ_i [f_i(polished) − f_i(certified)]` on the
  weighted base objective, only evaluated when all 48 blocks solve. Reports total Delta, TSO/DSO
  split, max `|Delta_i|` and its block, per-channel hull-bound-active counts (within `1e-9` relative
  of an interval end), the per-block table, and flagged blocks (`Delta_i > 1e-6·certified_cost ≈
  650.97`).

---

## 2. A real defect found and avoided: `p56a_oracle._interface_expression` is stale

While designing which Pyomo quantity to bound for interface P/Q, I found that `p56a_oracle.
apply_common_values` (the v11 exact-fix harness's own mechanism — the only prior art for "how to fix
a coupling entry" this repository has) pins the TSO's total interface power via
`_interface_expression(t_model, adn_load, p, 'p') == common_p` (`p56a_oracle.py:440-441`), where
`_interface_expression` (`p56a_oracle.py:390-399`) computes
`t_model.pc[adn_load, 0, 0, p] + flex_p_up[...] - flex_p_down[...]`.

Reading `create_transmission_network_model` (`shared_resources_planning.py:3580-3607`, "P5.15 Step
3.1-C / Addendum 12 item 2") shows that in the **current** ADMM path: `pc` is fixed **once**, at
construction, to the DSO's cycle-0 consensus interface power (the anchor — independently confirmed
in `WORKER_REPORT_S36_CLONE_CAPTURE.md:156-165`); `flex_p_up`/`flex_p_down`/`flex_q_up`/`flex_q_down`
are **also** fixed at `0.00`; and interface flexibility is instead carried **entirely** by
`interface_delta_p`/`interface_delta_q` (freed, bounded `±interface_transf_rating`). The model's own
`pc_adn` Expression (`network.py:434`, `interface_pf_p_transmission_def`,
`model_construction_helpers.py:1167-1185`) correctly includes `interface_delta_p`;
`p56a_oracle._interface_expression` does **not** — it predates Addendum 12's reparametrization and
was never updated.

So in the certified model, `_interface_expression(...) == pc(FIXED) + 0 − 0 == pc`, a **constant**,
while the real achieved interface power is `pc + interface_delta_p(achieved)` — a **different**
constant whenever `interface_delta_p` is materially nonzero, which it generically is (it is the
*sole* carrier of interface deviation in the ADMM path). `apply_common_values`'s row therefore
equates two constants that generically differ: a **zero-gradient, unconditionally infeasible row**,
independent of any active rating or voltage bound — no free variable in the model appears in it at
all.

**This is consistent with, and plausibly explains, Part 1's "frozen from iteration 0" finding** on
all six read TSO blocks (constraint violation identical to 16 significant figures from iteration 0
to termination, across warm/cold/cold+adaptive attempts) — and plausibly the other six TSO failures
the v11 exact-fix run reported (not independently re-read; that run is not re-run, per instruction).

**Scope decision.** This is a defect in `p56a_oracle.py`, not in `p515_s40_polish_gap.py` (whose own
two checked defects — starting point, consensus re-fix — were both absent, Part 1 §§2-3).
`p56a_oracle.py` is not in this task's forbidden-file list, but it is shared code several other
committed P5.15 stages depend on (P5.5-D1, P5.7, the s31/s32/s34/s35/s38/s39/s40 arms all import
`per_block_base_objectives`/`_tagged_holders`/etc. from it), so **I did not edit it** — out of this
task's authorized scope (small, localized, necessary changes only). Instead, `apply_hull_bounds`
bounds the model's own `pc_adn`/`qc_adn`/`pg_adn`/`qg_adn` Expressions directly (the same quantities
`common_coordinated_values` already reads to obtain `tso_p`/`dso_p`/`tso_q`/`dso_q`), sidestepping
the defect entirely. **Reported to the Planner as the primary unexpected finding of this task** —
see the final message; the Planner should decide whether `p56a_oracle.py` needs fixing and whether
the v11 exact-fix evidence needs re-reading in this light (not re-running).

---

## 3. Two further defects found and fixed while building the zero-solve checks

1. **ESSO subproblem indexing.** `models['esso'][node].es_pnet`/`.es_qnet` are indexed by
   **position** into `shared_ess_data.years`/`.days` (`range(len(years))`/`range(len(days))`,
   `shared_energy_storage_data.py:36-37`), **not** by the `(year, day)` label the TSO/DSO network
   models use — confirmed against production's own mapping convention at
   `shared_resources_planning.py:4264-4280`. `hull_entries_with_esso` maps the label to its index
   (`esso_years.index(year)`, `esso_days.index(day)`) before reading. Found because the FIRST
   version of the harness raised `KeyError: "Index '(2025, 'Spring', 0)' is not valid..."` when the
   checks script exercised it (never reached the full-scale ADMM run, since the checks build a
   real model and set values directly — caught before any solve).
2. **`_bound_var` must explicitly unfix.** A shared-ESS index whose installed capacity is `0` at
   model-construction time (`configure_shared_ess_operational_state`) is left `.fix()`-ed at `0`
   with bounds `(0, 0)`. Calling only `setlb`/`setub` on an already-fixed Var leaves it fixed — IPOPT
   and the NL writer treat a `.fixed` Var as a constant **regardless of its bounds**, silently
   defeating the interval. `_bound_var` now calls `.unfix()` first in the non-degenerate branch. At
   the certified C\* point every active shared-ESS index has nonzero installed capacity and so is
   genuinely unfixed by the time of polishing, but the guard is defensive and correct regardless of
   that circumstance — found via the checks script's synthetic (zero-investment) probe state, not
   via the real C\* run.

Both are documented in code comments at their fix sites (`p515_s41_hull_polish.py`).

---

## 4. Zero-solve checks — all pass

`p515_s41_hull_polish_checks.py`, `SolveProfileGuard(permitted=())` armed for the whole script,
`verify(0)` checked before writing output:

| check | result |
|---|---|
| interface P/Q distinct across TSO/DSO (2-agent, non-degenerate) | pass |
| exactly the intended number of `ConstraintList` rows added (TSO: `n_nodes × n_periods`, since the TSO model is shared across all its DSO interfaces; DSO: `n_periods`, its own model only) | pass |
| voltage bound set to `[min(v)², max(v)²]` on both `vmag_sqr` copies, not fixed | pass |
| voltage degenerate case: both copies `.fix()`-ed at the shared value | pass |
| interface-P row (`ConstraintList[1]`, the first row added, corresponding to the first node/period `apply_hull_bounds` processes) carries exactly `(lo, expr, hi)` | pass |
| shared-ESS P bound set to `[min(tso,dso,esso), max(...)]` (3-agent), not fixed, ESSO's own Var untouched | pass |
| shared-ESS P degenerate (all three agents equal): both TSO/DSO copies `.fix()`-ed | pass |
| every one of the 8,640 hull descriptors produced on the probe state contains its own achieved point (`lo ≤ value ≤ hi`) | pass — 0 failures |
| objective switching (`_switch_to_base_objective`) changes only the active-`Objective` set; every Var value/bound/fixed-flag and every Constraint active-flag is bit-identical before/after | pass |
| `NET._clear_multiplier_suffixes` empties `ipopt_zL_in`/`ipopt_zU_in`/`dual` | pass |
| `network._create_smopf_solver`'s `from_warm_start` gate is present in source (static confirmation that `from_warm_start=False` skips multiplier import) | pass |

`data/SRP1/Results/P515S41/hull_polish_checks/results.json`, guard counts
`{'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}`, `verify_failures: []`.

---

## 5. Smoke test — `--smoke-cycles 2`

Command run (attached, both streams captured, no `&`/`screen`/`nohup`):

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s41_hull_polish.py \
    --smoke-cycles 2 > data/SRP1/Results/P515S41/hull_polish_smoke_launch.log 2>&1
```

Preconditions checked before the run (per `_check_preconditions`): no `.p515_g_gate.lock`, no
forbidden live process, fresh output dir, production files (`shared_resources_planning.py`,
`network.py`, `network_data.py`, `shared_energy_storage_data.py`, `admm_parameters.py`,
`p515_g_g1_g4_admm_gates.py`, `data/SRP1/SRP1_params.json`) clean in git, D's committed reference
report present — all passed.

**Reproduction (against D's first 2 cycles): passed, 0 diffs, 0 provenance diffs.**

**Hull polish of all 48 blocks: all 48 solved. Zero failed blocks, zero flagged blocks.**

```
solve_profile: {'permitted_solve': 48, 'permitted_exec': 48, 'blocked_solve': 0, 'blocked_exec': 0}
```

Gate (meaningless at 2 cycles, per spec — reported, not a verdict): `relative=1.455986%` (threshold
`0.1%`), `PASS=False` — expected and documented; the harness's own `main()` therefore exits with
code 1 (confirmed: `EXIT_CODE=1` captured from the run), which is the harness working as designed,
not a failure of the smoke test. **The point of the smoke test is mechanics, not the gate value**:
every block whose certified point lies inside its own hull solved successfully — none was infeasible.

Per-block deltas at this 2-cycle point (top 10 by `|Δ_i|`, all TSO, all negative — expected, since
the ADMM hasn't converged and the base objective at 2 cycles is far from any stationary point):
`TSO|2025|Winter −22,094`; `TSO|2025|Autumn −20,222`; `TSO|2035|Summer −16,854`; `TSO|2025|Summer
−16,135`; `TSO|2025|Spring −12,916`; `TSO|2030|Summer −11,741`; `TSO|2035|Spring −10,899`;
`TSO|2035|Winter −9,817`; `TSO|2030|Winter −9,037`; `TSO|2030|Spring −7,659`. Total `Δ = −10,785,116`
(TSO `−148,490`, DSO `−7,274`; note the reconciliation residual `Δ_total − (Δ_TSO+Δ_DSO) =
−10,629,352` is **not** near zero at 2 cycles — expected, since `recourse_before`/`recourse_after`
are `gross_operational_cost` figures that include components beyond the 48 blocks' own weighted base
objectives, e.g. investment cost and any settlement terms; the per-block table is the primary
evidence, the top-level Delta is the gate value).

Hull-bound-active counts at the polished point (informational at 2 cycles): `ESS_P: 87`,
`PF_Q: 37`, `PF_P: 12` (of 1,728 descriptors each channel; `V` and `ESS_Q` show 0 active — no
surprise this early).

Runtime: ADMM (2 cycles) 95 s; polish (48 solves) 27 s; total wall 135 s.

Evidence: `data/SRP1/Results/P515S41/hull_polish_smoke/` (28-file `manifest_sha256.json`, including
`hull_polish_results.json`, `hull_bound_detail.json` — full per-descriptor detail, 8,640 entries,
`g_s39_D.json` — the ADMM report, `boyd_terminal.json`, `component_levels_terminal.json`, the ESSO
pickle, per-cycle sidecars).

**No errors, no tracebacks.** The only `WARNING` line in the full launch log is the expected
`ADMM did NOT converge in 2 iterations!` (a 2-cycle smoke run is never certified).

---

## 6. Exact full-run command (Planner launches; NOT run here)

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s41_hull_polish.py \
    > data/SRP1/Results/P515S41/hull_polish_launch.log 2>&1
```

Same preconditions as the smoke test, checked automatically by the script before it writes anything;
writes to `data/SRP1/Results/P515S41/hull_polish/` (a fresh, currently-nonexistent directory —
write-once, refuses to overwrite). Expect ~139 ADMM cycles (matching D's certification) plus the
48-block polish; based on the smoke test's per-cycle cost (~35-47 s/cycle observed in the log) and
prior S40 timings, a rough order-of-magnitude estimate is 60-90 minutes, not separately benchmarked
here.

---

## Validation

- The harness imports cleanly (`import p515_s41_hull_polish` succeeds against the canonical
  interpreter).
- Zero-solve checks pass with an armed, verified `SolveProfileGuard(0)`.
- The smoke test reproduces D's first 2 cycles bitwise and hull-polishes all 48 blocks
  successfully — the mechanics claim ("a block whose certified point is inside its hull should not
  be infeasible") is directly demonstrated, not merely argued.
- I distinguish: the checks script *proves* `apply_hull_bounds`/`hull_entries_with_esso` do what
  their docstrings claim on a synthetic state; the smoke test *demonstrates* the same mechanics
  survive contact with a real (if short) ADMM trajectory; neither substitutes for the full-length
  gate, which only the Planner's authorized run can evaluate.

## Unexpected findings

- **§2 above (`p56a_oracle._interface_expression` staleness) is the headline finding of this task.**
  It very plausibly reframes Part 1's "six unexplained TSO failures" and quite possibly the other six
  as well — not a fundamentally ill-posed test at active coupling constraints (Addendum 23's
  diagnosis), but a stale helper function computing the wrong achieved value for one of the two
  interface families. This bears directly on how the Planner reads the v11 exact-fix outcome
  (17/48 infeasible) and on whether `p56a_oracle.py` should be corrected for its other consumers.
- The ESSO-indexing and `_bound_var`-unfix defects (§3) are narrower, but the first would have
  crashed the full run outright, and the second could have silently produced a **non-hull** (over-
  tight, fixed-at-zero) constraint at any shared-ESS index with zero installed capacity — not a risk
  at C\* (all three nodes have installed capacity), but worth recording as a defensive fix.

## Remaining issues

- The full-length run's actual runtime is not measured (only estimated from the smoke test and
  prior S40 timings). The Planner's launch will produce the real figure.
- The `interface_delta_p` finding is reported, not fixed in `p56a_oracle.py` — a Planner decision is
  needed on whether/how to correct it there, independent of whether this harness (already immune to
  it) proceeds.

## Questions for Planner

1. Does the `interface_delta_p` finding change how you want the v11 exact-fix outcome (17/48
   infeasible) reported in the manuscript, given it is now attributable in large part to a stale
   helper rather than to "exact fixing is ill-posed at active coupling constraints" per se?
2. Should `p56a_oracle._interface_expression`/`apply_common_values` be corrected for its other
   consumers (P5.5-D1, P5.7, s31/s32/s34/s35/s38/s39/s40)? Out of this task's scope; flagged for a
   separate authorization if wanted.
3. Ready to authorize the full-length hull-polish run per the command in §6?

---

## Evidence index

| item | artifact | commit |
|---|---|---|
| Part 1 report | `WORKER_REPORT_S41_POLISH_PREREQ.md` | `f5dfcb3f` |
| harness + checks | `p515_s41_hull_polish.py`, `p515_s41_hull_polish_checks.py`, `data/SRP1/Results/P515S41/hull_polish_checks/results.json` | `5c460c2c` |
| smoke test + this report | `data/SRP1/Results/P515S41/hull_polish_smoke/`, `data/SRP1/Results/P515S41/hull_polish_smoke_launch.log`, `WORKER_REPORT_S41_HULL_POLISH_PREP.md` | (this commit) |

Zero-solve claims are backed by armed `SolveProfileGuard`s (`verify(0)` for the checks script; the
smoke/full run's own `48`-permitted-solve guard for the polish window, `153`-solve guard for the
2-cycle ADMM window, both reported in `hull_polish_results.json`).
