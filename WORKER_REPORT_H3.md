# Worker Report — H3 cohort-split rule (P5.15 Addendum 3, item 2)

## Task received

Implement the H3 cohort-split rule authorized by `PLANNER_BRIEF_2026-09-13.md`, Addendum 3,
item 2: per active investment cohort, split the ESSO's aggregate net power pro-rata to each
cohort's rated energy capacity among the *other* active cohorts for the same calendar year —
a homogeneous-fleet approximation, stated as linear/parameters-only, adding exactly
`N_active - 1` rows (never `N`), and required to be provably inert on every current
(single-cohort) fixture. Remedy (h) code (already present in
`shared_energy_storage_data.py`, uncommitted work from an earlier session) was explicitly
out of scope and was not touched.

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` (all addenda, in particular Addendum 3)
- `P5_15_EXPERT_HANDOFF.md` (section 3, "Gate G1 is not well-posed — three independent
  reasons", specifically Reason 3 — the H3 finding)
- `shared_energy_storage_data.py` — `_build_subproblem`, `_update_model_with_candidate_solution`,
  `_configure_esso_cohort_state`, `_esso_cohort_pair_is_within_lifetime`,
  `_add_esso_cohort_constraint`, `_get_candidate_solution` — to establish exactly when
  investment values become known relative to model construction, and how cohort
  activity/lifetime state is tracked and toggled
- `shared_resources_planning.py` — `create_shared_energy_storage_model`,
  `_update_operational_models_with_candidate` — to establish that the ESSO subproblem model is
  either (a) freshly built via `build_subproblem()` (diagnostic harnesses, and the very first
  ADMM cycle), or (b) a **clone** of a previously-configured model with `update_model_with_candidate_solution`
  re-applied for a new candidate (subsequent ADMM cycles) — this determined that H3 rows must
  be built **once**, structurally, at `_build_subproblem` time, with only their *activation state*
  and *coefficient value* changing on each candidate update, never re-added via repeated
  `ConstraintList.add()` calls (which would accumulate across cycles under case (b))
- `p515_h_tol_remedy_check.py` (construction path reused, not reimplemented, per the task)
- `p515_1_esso_reform_smoke.py` (`SMOKE_NODES_WITH_INVESTMENT`, `_zero_candidate`,
  `_set_nonzero_charge_discharge_request`, reused)
- `p513_solve_profile_guard.py` (`SolveProfileGuard`, used for every solve/build-only claim below)

## Files modified

- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/shared_energy_storage_data.py`

## Files created (new harnesses, per-task diagnostic scripts)

- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_h3_single_cohort_regression.py`
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_h3_multicohort_construction_test.py`
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_h3_multicohort_solve_check.py`

## New evidence artifacts (all NEW filenames under `data/SRP1/Results/P5151/`, none pre-existing)

- `data/SRP1/Results/P5151/h3_single_cohort_regression_summary.json`
- `data/SRP1/Results/P5151/h3_multicohort_construction_test_summary.json`
- `data/SRP1/Results/P5151/h3_multicohort_solve_check_summary.json`
- `data/SRP1/Results/P5151/h3_multicohort_solve_check_full.json`
- `data/SRP1/Results/P5151/h3_regression_logs/` (IPOPT logs, both arms)
- `data/SRP1/Results/P5151/h3_multicohort_solve_logs/` (IPOPT logs)

I confirmed before writing that none of these six paths pre-existed (`ls data/SRP1/Results/P5151/`
before this task listed only `eps_*`, `p5151_*` and `tol_*` artifacts belonging to earlier
stages). Nothing under `data/SRP1/Results/P5151/` was overwritten.

## Changes made

### 1. `shared_energy_storage_data.py` — three additions, no deletions, no edits to remedy (h)

**(a) New Param + ConstraintList declaration**, immediately before the
`energy_storage_operation_agg` block (where `agg_pnet` is built), so the H3 row can reference
the *identical* `agg_pnet` expression object the aggregate-definition row uses:

```python
model.es_pnet_cohort_share_h3 = pe.Param(model.years, model.years, mutable=True, initialize=0.00)
model.energy_storage_cohort_pnet_share_h3 = pe.ConstraintList()
```

**(b) Row construction**, inside the existing `for y in model.years: for d in model.days: for p
in model.periods:` loop, immediately after `agg_pnet` is fully summed (and after the aggregate
row and the converter-circle row, so it visibly reuses rather than rebuilds `agg_pnet`):

```python
for y_inv in model.years:
    _add_esso_cohort_constraint(
        model, 'energy_storage_cohort_pnet_share_h3', y_inv, y,
        model.es_pch_per_unit[y_inv, y, d, p] - model.es_pdch_per_unit[y_inv, y, d, p]
        == model.es_pnet_cohort_share_h3[y_inv, y] * agg_pnet)
```

This builds a row for **every** `(y_inv, y, d, p)` combination unconditionally (mirroring the
existing unconditional-grid convention `available_s_capacity_unit` / `available_e_capacity_unit`
already use) — `_configure_esso_cohort_pnet_share_rows` (below) is what leaves at most
`N_active(y) - 1` of them **active**, never `N`, and deactivates the rest via the standard
`_esso_cohort_constraints` tracking (`_add_esso_cohort_constraint`) the file already uses for
every other cohort-scoped constraint family.

**(c) New function `_configure_esso_cohort_pnet_share_rows(model)`**, called once per node
model at the end of `_update_model_with_candidate_solution`, *after* the existing per-cohort
`_configure_esso_cohort_state` loop has finished for every `y_inv` (the cross-cohort share and
omission decision needs the *full* active-cohort set for a calendar year, not one cohort at a
time). For each calendar year `y`:

- computes `active_cohorts = [y_inv : not _esso_cohort_inactive[y_inv] and
  _esso_cohort_pair_is_within_lifetime(model, y_inv, y)]` — **the same activity test the
  surrounding code already uses** (`_configure_esso_cohort_state`'s own criterion), not a new one;
- if `len(active_cohorts) <= 1`: sets every `es_pnet_cohort_share_h3[y_inv, y]` to `0.0` and
  deactivates every H3 row for that year — full inertness;
- otherwise: computes `total_e_rated = sum(pe.value(es_e_investment_fixed[y_inv]) for y_inv in
  active_cohorts)` (a Python float sum of Param values), `share[y_inv] = pe.value(es_e_investment_fixed[y_inv])
  / total_e_rated` for every cohort (a plain Python float, pushed into the model via
  `.set_value(...)`), picks `omitted = max(active_cohorts)`, and activates every active
  cohort's row **except** the omitted one's.

Call site added in `_update_model_with_candidate_solution`:

```python
_configure_esso_cohort_state(model, y_inv, s_candidate, e_candidate, shared_ess_data.params.slacks)
    ...
_configure_esso_cohort_pnet_share_rows(model)   # <-- new, once per node, after the y_inv loop
```

No other line in `shared_energy_storage_data.py` was touched. `git diff --stat` shows the file
as `423 lines changed`; **all but ~150 of those lines are pre-existing uncommitted work from an
earlier Worker session** (remedy (h)'s `ESSO_TOL_OVERRIDES`, the complementarity-ratio detector,
the barrier-identity comment, etc. — all present in the working tree *before* this task started,
none touched by it). The H3-specific hunks are exactly the three additions above, isolated to:
the Param/ConstraintList declaration block, the `for y_inv in model.years:` row-construction
loop inside `energy_storage_operation_agg`'s own loop, the one new call line in
`_update_model_with_candidate_solution`, and the new `_configure_esso_cohort_pnet_share_rows`
function. I verified this isolation directly by reading the full `git diff -- shared_energy_storage_data.py`
and confirming every non-H3 hunk pre-dated this session (comment text and `ESSO_TOL_OVERRIDES`
already documented in `P5_15_GATE_HOLD_REPORT.md`/`P5_15_EXPERT_HANDOFF.md`, both written before
this task began).

### Row-set form chosen: N_active − 1 rows tied to `agg_pnet`, not pairwise

I chose the form the brief states first ("Add `N_active-1` rows and let the existing
`energy_storage_operation_agg` row supply the last"), rather than the pairwise alternative,
because:

1. It reuses the **exact same** `agg_pnet` expression object the aggregate-definition row
   builds (not `es_pnet`, which can differ from `agg_pnet` by `slack_es_pnet_up/down` when
   slacks are enabled — the brief is explicit that the row must be built against `agg_pnet`
   specifically). Building against the identical expression object, in the same loop, is the
   most direct way to guarantee this.
2. It composes cleanly with the file's existing "build the full grid once, activate/deactivate
   per cohort" convention (`_esso_cohort_constraints`, `_add_esso_cohort_constraint`,
   `_configure_esso_cohort_state`'s own activate/deactivate loop) — no new bookkeeping
   mechanism was introduced, only one new post-processing pass layered on top for the
   cross-cohort "which one is omitted" decision that the existing per-cohort loop structurally
   cannot make (it only ever sees one `y_inv` at a time).
3. The omission convention (`omitted = max(active_cohorts)`) is deterministic and arbitrary by
   design — any single fixed choice works because of the sum-to-one identity (verified
   numerically below, item (c)).

## Proof that the coefficients are floats, not Pyomo expressions

`es_pnet_cohort_share_h3` is declared `pe.Param(..., mutable=True, ...)`; every value written
into it comes from `pe.value(model.es_e_investment_fixed[y_inv]) / total_e_rated`, where
`total_e_rated` is a plain Python `sum(...)` over `pe.value(...)` calls — never an arithmetic
operation on a Pyomo `Var` or expression. The multi-cohort construction test
(`p515_h3_multicohort_construction_test.py`) asserts this directly rather than by inspection
alone: for every `(y_inv, y)` it checks `type(model.es_pnet_cohort_share_h3[y_inv, y].value) is
float`. Result: **`coefficients_are_plain_python_floats: true`** for all 9 checked entries (3
cohorts × 3 years) in the two-cohort node-7 instance. It further extracts each active H3 row's
linear representation via `pyomo.repn.generate_standard_repn` (a Pyomo-native, not
hand-derived, mechanism) and asserts `polynomial_degree() == 1` — a Var-ratio coefficient would
make the cohort-sum term nonpolynomial (`polynomial_degree() is None`); both checked rows report
degree `1`.

## Single-cohort regression result (Verification required, item 1)

Harness: `p515_h3_single_cohort_regression.py`. Instance: `p515_h_tol_remedy_check.py`'s
construction path, reused by import (`p515_1_esso_reform_smoke.py` +
`p515_h_tol_remedy_check.py`'s own module) — S=1.00 MVA/E=2.00 MVAh at nodes 7 and 9, first
representative year only (single cohort per node), node 5 zero-investment, +/-10% duty-cycle
`p_req`.

- **Arm A** ("with_h3"): the shipped production path, `create_shared_energy_storage_model`,
  H3 code present and executing on every node.
- **Arm B** ("h3_stripped"): the identical build (`build_subproblem` +
  `update_model_with_candidate_solution`, same production functions), with the H3 `Param` and
  `ConstraintList` **physically removed** (`model.del_component(...)`) from each node's model
  before the `p_req`/`q_req` fix and the solve — i.e., the model IPOPT actually sees is exactly
  what it would have been had H3 never been added to `_build_subproblem`.

**Result — bit-identical, max difference 0.0 on every field, at every node:**

| node | objective abs diff | `es_pnet` max abs diff | `es_pch_per_unit` max abs diff | `es_pdch_per_unit` max abs diff | `es_soh_per_unit_cumul` max abs diff | `es_D_per_unit` max abs diff | H3 active rows (Arm A) | H3 structural rows (Arm A) |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 864 |
| 7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 864 |
| 9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 864 |

`all_h3_shares_zero_in_arm_a: true`, `all_h3_rows_inactive_in_arm_a: true`. Every share is
`0.0` and every H3 row is deactivated by `_configure_esso_cohort_pnet_share_rows`, at every
node, exactly as predicted for a single-cohort fixture. Note the "864 structural rows per node"
figure: this is the ConstraintList's *total* entry count (built for the full unconditional grid
of 3 cohorts × 3 years × 4 days × 24 periods = 864, per node), of which zero are active — see
"structural vs. active count" caveat below.

**Guard**: `SolveProfileGuard` armed for `_run_solver_attempt` only, declared count 6 (2 arms ×
3 active nodes), observed `permitted_solve: 6`, `blocked_solve: 0`. `verify(6) == []` (no
failures).

## Multi-cohort construction test (Verification required, item 2 — BUILD ONLY, zero solves)

Harness: `p515_h3_multicohort_construction_test.py`. Instance: node 7, S=1.00 MVA/E=2.00 MVAh
invested in the first representative year (2025, cohort 0) and S=0.50 MVA/E=1.00 MVAh in the
second (2030, cohort 1), set via the **same** candidate-solution mechanism every fixture uses
(`shared_ess_data.update_model_with_candidate_solution`). `t_cal = 15` years, each represented
year block is 5 years wide, so `tcal_norm = round(15/5) = 3` for every cohort: cohort 0 is
within its calendar-life window for calendar years `y = 0, 1, 2`; cohort 1 (invested at `y=1`)
is within its window for `y = 1, 2`. Calendar years 1 and 2 are therefore genuinely two-cohort
years, not manufactured beyond the candidate-solution mechanism itself.

A **blocking** `SolveProfileGuard([], ...)` was armed for the whole build (`permitted=[]`);
`guard.verify(0)` returned `[]` — zero solves, zero execs, confirmed by an armed guard, not
asserted.

**(a) Row count — exactly `N_active(y) - 1` active rows per `(d, p)`, never `N`:**

| year | active cohorts | N_active | active rows (total, summed over 96 `(d,p)` combos) | expected (`(N_active-1) × 96`) | matches |
|---|---|---|---|---|---|
| 0 (2025) | [0] | 1 | 0 | 0 | yes |
| 1 (2030) | [0, 1] | 2 | 96 | 96 | yes |
| 2 (2035) | [0, 1] | 2 | 96 | 96 | yes |

i.e. exactly 1 active row per `(d, p)` at the two two-cohort years, never 2 — the N-row case
(which would be linearly dependent with the aggregate row) never occurs.

**(b) Coefficients are plain Python floats:** `coefficients_are_plain_python_floats: true` (see
proof section above; 9/9 entries checked).

**(c) Shares sum to 1 across active cohorts:** at `y=1` and `y=2`,
`share[0] = 0.666666...` (`= 2.00/(2.00+1.00)`), `share[1] = 0.333333...`
(`= 1.00/(2.00+1.00)`, stored even though cohort 1's row is the omitted/inactive one), sum
`= 1.0` exactly (float sum, `1.0` reported, no residual). At `y=0` (single active cohort) the
rule is inert, consistent with item (a).

**(d) Linear independence of the active H3 row from the aggregate row — numerical rank check:**
for both two-cohort years, extracted each row's linear coefficient vector over the shared
variable ordering `[pch_0, pdch_0, pch_1, pdch_1, pch_2, pdch_2, es_pnet, slack_up, slack_down]`
via `pyomo.repn.generate_standard_repn` (Pyomo's own representation, not hand-derived), stacked
the two vectors, and computed `numpy.linalg.matrix_rank`. Result: **rank 2** in both cases
(`linearly_independent: true`) — e.g. at `y=1`: H3 row `[0.3333, -0.3333, -0.6667, 0.6667, 0,
0, 0, 0, 0]` vs. aggregate row `[-1, 1, -1, 1, 0, 0, 1, -1, 1]`; the aggregate row has a nonzero
`es_pnet` coefficient the H3 row lacks entirely, so the two rows can never be proportional.

Guard, row count, coefficient-type, sum-to-one and rank-check details are all in
`data/SRP1/Results/P5151/h3_multicohort_construction_test_summary.json`.

## Multi-cohort solve check (Verification required, item 3 — one solve authorized)

The instance above is cheap to solve (same SRP1 case, single node of interest), so I ran it,
per the brief's authorization ("If a multi-cohort instance can be constructed AND solved
cheaply, one solve is authorized"). Harness: `p515_h3_multicohort_solve_check.py`. "One solve"
is interpreted as one end-to-end `create_shared_energy_storage_model` invocation of the
multi-cohort instance; that production entry point solves every active node (5, 7, 9) in one
call, so the guard was declared and verified at **3** (bounded, not a literal "1", stated
explicitly here rather than left implicit).

- **Guard**: `SolveProfileGuard` on `_run_solver_attempt`, declared 3, observed
  `permitted_solve: 3`, `blocked_solve: 0`. `verify(3) == []`.
- Node 7 terminated `optimal`, objective `0.023066456823210117`.
- **Pro-rata identity check** — for every `(y_inv, y, d, p)` at the two two-cohort years,
  including the **omitted** cohort (`y_inv=1`, whose row is *not* directly enforced — its value
  is only implied by the aggregate identity): `pnet_cohort[y_inv,y,d,p] == share[y_inv,y] ×
  agg_pnet[y,d,p]`. 384 checks (2 cohorts × 2 years × 96 `(d,p)`), **max absolute error
  `1.3877787807814457e-17`** (machine-precision noise) — the split is pinned pro-rata for both
  the directly-constrained cohort and the one supplied only by the aggregate identity, exactly
  as the row-count argument claims.

## Explicit inertness statement

**The H3 rule is inert on every currently runnable fixture.** Every case file this Worker could
find or was told about invests in at most one cohort per node at a time (confirmed directly for
the node-7/node-9 single-cohort regression fixture: zero active H3 rows, all shares 0.0, at
every one of the three active nodes). No change in objective, dispatch, or SoH was observed on
that fixture (max difference 0.0 on every reported field), and none was manufactured. The
multi-cohort behavior (rows active, correct shares, correct pro-rata split at a real solved
point) was verified only on a **purpose-built** two-cohort instance constructed for this task,
not on any existing production fixture.

## Validation

- `ast.parse` on the modified file: syntax OK.
- `import shared_energy_storage_data`: succeeds; new function present
  (`hasattr(SED, '_configure_esso_cohort_pnet_share_rows') == True`).
- Single-cohort regression: guard-verified 6/6 solves, zero unexpected/blocked calls, bit-identical
  result between the shipped path and a version with H3 components physically removed.
- Multi-cohort construction test: guard-verified 0/0 solves (build-only claim armed, not
  asserted); all four required checks (a)-(d) pass.
- Multi-cohort solve check: guard-verified 3/3 solves; pro-rata identity holds to `1.4e-17`.
- `git diff --stat -- shared_energy_storage_data.py` and a full read of the diff confirm no
  edits outside the three H3-specific hunks listed above; remedy (h)'s code and the amended
  comment at the top of `_build_subproblem` (~lines 420-483) are untouched (verified by reading
  the diff — those hunks pre-date this session).
- `git status --short`: only `shared_energy_storage_data.py` shows as modified by this task;
  the three new `p515_h3_*.py` harnesses and the listed `data/SRP1/Results/P5151/h3_*` artifacts
  are the only new files this task produced. No file under `data/` was edited (only new files
  were written, all under `P5151/`, none pre-existing).

## Unexpected findings

- **Structural vs. active row count.** Because the ESSO subproblem model, once built, is later
  *cloned* and re-updated with new candidate investments across ADMM cycles (not rebuilt from
  scratch — see `_update_operational_models_with_candidate` in `shared_resources_planning.py`),
  the H3 `ConstraintList` is populated for the **full unconditional grid** (`N_cohorts ×
  N_years × N_days × N_periods` rows) at `_build_subproblem` time, exactly mirroring the
  existing convention `available_s/e_capacity_unit` already uses, and only *activated/deactivated*
  per candidate thereafter. A strict reading of "zero rows added" for the single-cohort case
  could mean either "zero rows physically present" or "zero rows active" (i.e., reaching the
  solver). I verified and report **both**: 864 structural (deactivated) entries per node, 0
  active — and I verified separately (Arm B, `del_component`) that removing the structural
  entries entirely produces a bit-identical solve to leaving them present-but-deactivated, which
  is the property that actually matters for LICQ/solver-facing correctness (Pyomo excludes
  deactivated constraints from the NL file). If the Planner wants literally zero structural
  rows on single-cohort instances, that would require rebuilding the H3 rows dynamically inside
  `_update_model_with_candidate_solution` instead of once at `_build_subproblem` time, which I
  did **not** do because it reintroduces the cross-cycle `ConstraintList.add()`-accumulation
  risk described above (each new candidate on a cloned model would add more rows on top of the
  previous candidate's, without a corresponding way to remove them, since Pyomo's
  `ConstraintList` has no `.clear()` counterpart used elsewhere in this file). I flag this
  as a design choice made to stay inside the file's existing conventions and to avoid a
  correctness risk across ADMM cycles, not an oversight.

## Remaining issues

- I did not attempt to verify H3's effect (or lack of it) inside a full multi-cycle ADMM run;
  the multi-cohort solve check is a single direct ESSO solve (as the brief's own scope
  constrains: "do NOT construct an elaborate new fixture").
- No gate runs (G1-G5) were performed — out of scope per the brief.

## Questions for Planner

- Confirm whether "zero H3 rows were added" for the single-cohort regression (Verification
  item 1) should be read as "zero active rows" (what I verified and consider the operative
  claim, since deactivated Pyomo rows do not reach the solver) or "zero structural
  `ConstraintList` entries" (which would require a different, dynamically-rebuilt
  implementation, with the cross-cycle accumulation risk noted above). I implemented and
  verified the former; if the latter is required, please say so and I will revisit the design.
