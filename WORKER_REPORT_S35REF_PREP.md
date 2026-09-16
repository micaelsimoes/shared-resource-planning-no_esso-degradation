# Worker Report -- P5.15 Addendum 16 item 1, s35ref "run 1" prep

## Task received

Prepare (NOT launch) run 1 of Addendum 16 -- the `s35ref` reference-equilibrium
gate: a new harness arm identical to `s34` except cap 500 (was 150) and
initial `rho_ess = 0.1125` on every network and the ESSO (case-file value,
was 0.05); a new SoH floor-multiplier + per-cohort-year EFC capture (Addendum
16, decisive); a new `p515_s35ref_preflight.py` one-cycle preflight bounded
by an exact `SolveProfileGuard` count (102); and a report. Launching the
500-cycle run itself was explicitly out of scope.

Binding specification: `data/SRP1/Results/P515S35/frozen_s35_reference_spec_v5_995548ab.json`
(commit `f76c7574`), superseding `data/SRP1/Results/P515S34/frozen_s34_spec_v4_966940a7.json`.

## Files inspected

- `data/SRP1/Results/P515S35/frozen_s35_reference_spec_v5_995548ab.json` (full read)
- `data/SRP1/Results/P515S34/frozen_s34_spec_v4_966940a7.json` (full read, for the diff)
- `p515_g_g1_g4_admm_gates.py`: the entire `s34` section (~2583-3391 before this
  change), `run_admm_arm` (~1144-1340), `assert_s31_capture_paths` (~1525-1567),
  `assert_s31c_capture_paths` (~1738-1771), `_s32_binding_test`,
  `write_interface_settlement_detail_s31c`, `write_interface_voltage_terminal`,
  `write_component_levels_terminal`, `assert_g_capture_paths` (the
  `SED._build_subproblem` zero-solve probe precedent, ~943-1000), module
  imports/constants (~1-100)
- `p515_s34_preflight.py` (full read -- the direct template for the new preflight)
- `shared_energy_storage_data.py`: `_build_subproblem` (~397-420),
  `energy_storage_capacity_degradation` construction (~624-676),
  `_add_esso_cohort_constraint` (~1467-1471), `_configure_esso_cohort_state`
  constraint activation loop (~1526-1537), `_configure_esso_cohort_pnet_share_rows`
  (the existing production precedent for grouping `_esso_cohort_constraints`
  by `constraint_name`, ~1540-1596)
- `data/SRP1/SRP1_params.json` (`admm.rho` block)
- `p514_n_instrumented_cstar.py` (`EFC_BINDING_THRESHOLD`, `REL`, `PERMITTED`)
- `p56a_oracle.py` (`fresh_planning`, `WORK_DIR`, `_holders`)

## Files modified / created

Modified:
- `data/SRP1/SRP1_params.json` (one edit: `admm.rho.ess.*` 0.05 -> 0.1125)
- `p515_g_g1_g4_admm_gates.py` (`import math`; one new self-contained section;
  one new dispatch branch; zero lines of existing code deleted or altered --
  confirmed by `git diff` showing 715 insertions, 0 deletions outside the
  `SRP1_params.json` file)

Created:
- `p515_s35ref_preflight.py`
- `data/SRP1/Results/P515S35/preflight_ref/` (one-cycle preflight run outputs)
- `data/SRP1/Results/P515S35/preflight_ref_sha256_manifest.txt`
- `WORKER_REPORT_S35REF_PREP.md` (this file)

## Changes made

### 1. `data/SRP1/SRP1_params.json`

```diff
 			"ess": {
-				"case9": 0.05,
-				"case33_1": 0.05,
-				"case33_2": 0.05,
-				"case33_3": 0.05,
-				"esso": 0.05
+				"case9": 0.1125,
+				"case33_1": 0.1125,
+				"case33_2": 0.1125,
+				"case33_3": 0.1125,
+				"esso": 0.1125
 			}
```
(`data/SRP1/SRP1_params.json:74-80`). No other key touched. SRP1 has its own
params file (`data/SRP1/SRP1_params.json`), distinct from every other case
study's (`CS1_params.json`, `CS7_params.json`, `HR1_params.json`,
`OP1_params.json`, `OP2_params.json`) -- this change is scoped to SRP1 only.

### 2. `p515_g_g1_g4_admm_gates.py`

- `import math` added to the top-level imports (`p515_g_g1_g4_admm_gates.py:56`,
  needed by `_identify_soh_floor_rows`'s `math.isfinite` check).
- New self-contained section `p515_g_g1_g4_admm_gates.py:3060-3673`, inserted
  between `write_boyd_terminal_s34` and `if __name__ == '__main__':`. Every
  other arm (`s32`, `s33e2`, `s34`, `g1`/`g2`/... ) is byte-identical to
  before this change (`git diff` on this file shows insertions only, no
  deletions). Contents, in order:
  - `OUT_S35REF`, `S35REF_SPEC_PATH`, `S35REF_SPEC_SHA256`, `S35REF_CAP = 500`,
    `S35REF_REL = 1e-4`, `S35REF_INITIAL_RHO_ESS = 0.1125`, and four
    `S35REF_*_G_PATH` / `S35REF_*_LEAK_PATH` constants reusing the s34/s31c/
    s32/s33e2 constants BY REFERENCE (`p515_g_g1_g4_admm_gates.py:3085-3097`).
  - `_identify_soh_floor_rows(esso_model)` (`p515_g_g1_g4_admm_gates.py:3109-3227`)
    -- see "How the floor rows were identified" below.
  - `assert_s35ref_capture_paths(planning)` (`p515_g_g1_g4_admm_gates.py:3230-3377`)
    -- builds on `assert_s31c_capture_paths` DIRECTLY (not `assert_s34_capture_paths`,
    which hard-asserts v4's rho_ess == 0.05 and would raise here -- the same
    reasoning `assert_s34_capture_paths`'s own docstring gives for not reusing
    `assert_s32_capture_paths`/`assert_s33e2_capture_paths`); reproduces the
    structural checks `assert_s34_capture_paths` performs (Boyd eps, sigma
    fixed + calibration assertion, al_scale_esso presence/mode, S_ref 2.5,
    freeze parameters, 3 consecutive cycles), with the rho_ess expectation
    updated to 0.1125; adds the spec v5 hash check and the pre-solve floor-row
    identification/assertion. Returns `(checklist, floor_rows_by_node)`.
  - `s35ref_capture_hooks(...)` (`p515_g_g1_g4_admm_gates.py:3380-3450`) --
    a context manager that enters `s34_capture_hooks` (reused BY CALLING, not
    copied) for the recourse-jump and ESS-entry-stride sidecars, then layers
    a second monkeypatch of `srp.get_admm_boyd_residual_metrics` that writes
    `soh_floor_sidecar_baseline.jsonl` every cycle (floor dual, SoH,
    active flag, EFC/day per (node, y_inv, y)).
  - `_s35ref_system_cost_vs_s34` / `_s35ref_system_cost_vs_references`
    (`p515_g_g1_g4_admm_gates.py:3453-3520`) -- the former is new (a 4th
    reference, s34, that `_s34_system_cost_vs_references` could not include
    since s34 predates s35ref); the latter calls `_s34_system_cost_vs_references`
    BY CALLING IT (unmodified) for the s31c/s32/s33e2 legs and adds the s34 leg.
  - `_s35ref_terminal_floor_and_efc` / `write_boyd_terminal_s35ref`
    (`p515_g_g1_g4_admm_gates.py:3523-3673`) -- the terminal writer, modeled
    directly on `write_boyd_terminal_s34`, reusing `_s34_rho_gamma_freeze_trajectory`
    and `_s32_binding_test` BY CALLING them.
  - New dispatch branch `elif gate == 's35ref':` (`p515_g_g1_g4_admm_gates.py:4024-4106`),
    mirroring the `s34` branch exactly (spec-hash/REL guard, fresh-output-root
    guard, throwaway preflight-planning checklist call, sidecar path setup,
    `run_admm_arm(..., num_max_iters_override=500, ...)`). **This branch was
    never executed** -- confirmed by `data/SRP1/Results/P515S35_REF_run`
    not existing after all Worker activity (see Validation).

### 3. `p515_s35ref_preflight.py` (new)

Modeled directly on `p515_s34_preflight.py`, imported as `S34PF` and its
`_field_completeness` reused BY CALLING (not copied). Runs `run_admm_arm` BY
IMPORT for exactly 1 cycle into a fresh `data/SRP1/Results/P515S35/preflight_ref/`,
with the s35ref capture hooks wired, then verifies the checklist, rho_ess,
solve count, per-cycle field completeness, and the new floor/EFC sidecar.
Deliberately does NOT re-run the D5 detector gate (s34's own two-cycle
preflight already passed it under the unchanged D5 mechanism; documented in
the script's own module docstring and in `preflight_verification.json`'s
`d5_detector_gate_not_rerun_here` field).

## How the floor rows were identified (and the row count asserted)

**Method: construction order, cross-validated by expression inspection**
(`_identify_soh_floor_rows`, `p515_g_g1_g4_admm_gates.py:3109-3227`):

1. **Construction order** -- production's own `model._esso_cohort_constraints[y_inv]`
   (`shared_energy_storage_data.py:413`, populated by `_add_esso_cohort_constraint`,
   `shared_energy_storage_data.py:1467-1471`) records every row of the
   `energy_storage_capacity_degradation` family, per `y_inv`, in the exact
   order the three `_add_esso_cohort_constraint` calls execute per `y`
   (`shared_energy_storage_data.py:652-676`): `[D_eq, soh_recursion_eq,
   floor_ineq]`. Filtering to `constraint_name ==
   'energy_storage_capacity_degradation'` yields fixed-length-3 groups, one
   per `y`; the third entry of each group is the floor row. This is the SAME
   list production itself groups by `constraint_name` elsewhere
   (`_configure_esso_cohort_pnet_share_rows`, `shared_energy_storage_data.py:1586-1596`)
   -- not a new mechanism, an existing production pattern reused.
2. **Expression inspection** (Pyomo API only, zero reimplementation of
   production numerics) -- for EVERY triple (not sampled): the first two rows
   must have `con.equality is True`; the candidate floor row must have
   `con.equality is False`, its body (`identify_variables`) must be EXACTLY
   the single Var `es_soh_per_unit_cumul[y_inv, y]`, `con.upper is None`
   (one-sided `>=`), and `con.lower` (read directly off the built Pyomo
   object, i.e. the row's own `soh_min`, never re-derived from
   `shared_ess_data`) must be finite. Structural self-consistency: all three
   rows of a triple must share the same recorded `y`, and `y` must strictly
   increase across consecutive triples within a `y_inv` (matching
   production's `for y in range(y_inv, max_tcal_norm)`).

Any mismatch RAISES. This doubles as the row-count assertion: the floor-row
count returned is, by this construction, exactly the count of distinct
`(y_inv, y)` pairs for which production added ANY degradation-family row
(pairs outside the calendar-life window get none), and every such pair is
individually checked to contribute exactly one qualifying floor row -- not a
coincidental total-count match. **Equality rows are excluded by construction**:
`_identify_soh_floor_rows` never places a `constraint_idx` into its result
dict unless it passed the `equality is False` / single-var / one-sided-bound
checks above, so an equality row cannot enter `floor_rows_by_node` if this
function did not raise.

Identification runs BEFORE any solve, on a throwaway, freshly-built
(unsolved) probe ESSO subproblem per active node (`SED._build_subproblem`),
the SAME zero-solve probe mechanism `assert_g_capture_paths` already uses for
its IPOPT-Suffix check. The mapping is candidate-independent (construction
order/`constraint_idx` depends only on the shared-ESS calendar-life
configuration, not the investment candidate -- only which rows get
*deactivated* at solve time is candidate-dependent, per
`_configure_esso_cohort_state`), so `assert_s35ref_capture_paths`'s mapping is
reused verbatim by `s35ref_capture_hooks` for the real arm's own `esso_model`
instances, never re-derived per cycle. This determinism claim is not merely
asserted: the one-cycle preflight run independently reproduced the identical
per-node row count (6) and the identical triangular `(y_inv, y)` pattern
(3 + 2 + 1 rows for `y_inv` = 0, 1, 2) on its OWN freshly-built `esso_model`
instances (see Results below), which is exactly what the calendar-life
window arithmetic predicts for a 3-year horizon with a single active
investment cohort.

**Row count observed**: 6 floor rows per node (5, 7, 9) -- 18 total,
`soh_floor_row_counts_by_node == {5: 6, 7: 6, 9: 6}`, `constraint_idx` in
{3, 6, 9, 12, 15, 18} per node (every 3rd row, as expected), `soh_min == 0.5`
on every row (matches the spec's stated `shared_energy_storage.soh_min`
value verbatim).

## Floor-dual sign convention

`model.dual.get(con)` as returned by Pyomo (`Suffix(direction=IMPORT_EXPORT)`,
populated from the IPOPT `.sol` file) for the row
`es_soh_per_unit_cumul[y_inv, y] >= soh_min`; **no sign flip applied**. This is
the SAME mechanism `_duals_for_keys` (`p515_g_g1_g4_admm_gates.py:278-291`)
already uses elsewhere in this harness for other ESSO constraint families.
The convention string is recorded once per cycle in the sidecar
(`dual_sign_convention` key) rather than per row.

**Not empirically verified in this task**: at cycle 1 every floor row is far
from its bound (SoH in [0.9999, 1.0], `soh_min = 0.5`), so every row is
inactive and its dual is either ~1.8e-10 (numerically zero, IPOPT barrier
residue on an active-in-the-model-but-far-from-bound Var) or `None`
(deactivated cohort rows -- see below). The sign of an ACTIVE floor-row dual
was not observed in this task and cannot be, since the floor does not bind
this early (the spec's own prediction puts binding near cycle ~440, `spec
v5 predictions.stop_cycle`).

## Preflight's floor table at cycle 1

18 entries (3 nodes x 6 rows). 9 of 18 have a finite (~1.8e-10, effectively
zero) dual; the other 9 (`y_inv` in {1, 2}, i.e. the 2030/2035 investment
cohorts) have `dual = None` -- **not a bug**: the instance invests only at
`y_inv = 0` (2025; `spec.instance`, uniform S/E across nodes 5/7/9, single
cohort), so per `_configure_esso_cohort_state`
(`shared_energy_storage_data.py:1526-1537`) every constraint row belonging to
an inactive cohort is `.deactivate()`d before the solve, and Pyomo's `dual`
Suffix is only populated for rows that were part of the solved model. Every
SoH value is finite; every row is `active = False` (expected, since 1 cycle
is far too early for the floor to bind):

| node | y_inv | y | constraint_idx | SoH | dual | efc/day |
|---|---|---|---|---|---|---|
| 5 | 0 | 0 | 3 | 0.99997243 | 1.8183e-10 | 1.743e-04 |
| 5 | 0 | 1 | 6 | 0.99994209 | 1.8184e-10 | 1.919e-04 |
| 5 | 0 | 2 | 9 | 0.99991358 | 1.8185e-10 | 1.803e-04 |
| 5 | 1 | 1 | 12 | 1.0 | None | None |
| 5 | 1 | 2 | 15 | 1.0 | None | None |
| 5 | 2 | 2 | 18 | 1.0 | None | None |
| 7 | 0 | 0 | 3 | 0.99997184 | 1.8183e-10 | 1.781e-04 |
| 7 | 0 | 1 | 6 | 0.99993600 | 1.8184e-10 | 2.267e-04 |
| 7 | 0 | 2 | 9 | 0.99990755 | 1.8185e-10 | 1.800e-04 |
| 9 | 0 | 0 | 3 | 0.99997195 | 1.8183e-10 | 1.774e-04 |
| 9 | 0 | 1 | 6 | 0.99992650 | 1.8184e-10 | 2.875e-04 |
| 9 | 0 | 2 | 9 | 0.99989779 | 1.8185e-10 | 1.816e-04 |

(node 7/9's `y_inv` in {1,2} rows omitted from the table -- identical
pattern: SoH 1.0, dual None. Full data in
`data/SRP1/Results/P515S35/preflight_ref/soh_floor_sidecar_preflight.jsonl`
and `boyd_terminal.json`.)

EFC/day max across nodes at cycle 1: 0.000287 (node 9), i.e. 0.02% of the
1.4612 binding threshold -- as expected this early (cf. s33e2 terminal 0.067
after 150 cycles at the old normalization).

## Commands / experiments run

1. Zero-solve dry check (`assert_s35ref_capture_paths` + `_identify_soh_floor_rows`
   only, via a scratch `O.fresh_planning` eval, cleaned up after):
   ```
   /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -c "..." 
   ```
   Passed; floor row counts `{5: 6, 7: 6, 9: 6}`; scratch eval dir
   `data/SRP1/Results/P56A/evals/zzz_worker_dry_check_s35ref_floor` deleted
   afterward (not a committed artifact).
2. The one-cycle preflight (permitted by the task):
   ```
   /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s35ref_preflight.py
   ```
   Exit code 0. Output: `data/SRP1/Results/P515S35/preflight_ref/` (fresh
   directory, did not exist before).
3. `git diff --stat` / `git diff` on both modified files, to confirm no
   unintended changes (see Validation).
4. `sha256`/`shasum -a 256` manifest of every file under `preflight_ref/`:
   `data/SRP1/Results/P515S35/preflight_ref_sha256_manifest.txt`.

**No command was run that invokes `python p515_g_g1_g4_admm_gates.py s35ref`**
(the 500-cycle gate) -- confirmed by `data/SRP1/Results/P515S35_REF_run` not
existing on disk.

## Results

- Solve count: **102**, checked exactly
  (`SolveProfileGuard` observed `{'permitted_solve': 102, 'permitted_exec':
  102, 'blocked_solve': 0, 'blocked_exec': 0}`, `identity_holds=True`) --
  matches `51*NUM_CYCLES + 51` with `NUM_CYCLES=1`, i.e. one cycle plus
  initialization, per the task's stated formula (s34's own two-cycle
  preflight was 153 = 51*3).
- `assert_s35ref_capture_paths` checklist: **passed** (no raise; every
  boolean entry `True`), including `initial_rho_v_pf_ess_matches_spec_v5`,
  `soh_floor_rows_identified_pre_solve`, `soh_floor_row_count_uniform_across_nodes`.
- rho in force at cycle 1: `v = 0.0077` (all networks), `pf = 0.198` (all
  networks), `ess = 0.1125` (all networks + esso) -- confirmed both from the
  pre-solve checklist snapshot and from the per-cycle diagnostics row
  (`rho_v_before/rho_pf_before/rho_ess_before` at cycle 1).
- `al_scale_esso` at cycle 1: 227210.997 (> 1, per the spec's deferred check).
- Field completeness: **all required per-cycle-channel/diagnostics fields
  populated and finite** at cycle 1 (`field_completeness_all_ok = True`).
- Floor sidecar: **available**, 18 entries, row counts match the pre-solve
  expectation exactly per node, all SoH values finite, all 18 rows carry a
  `dual` key (9 finite, 9 legitimately `None` -- inactive cohorts, explained
  above), EFC/day populated on every row that has a dual (the 9 inactive
  rows report `efc_per_day = None` too, since `es_avg_ch_dch_per_unit` /
  `es_e_rated_per_unit` are fixed to 0 for inactive cohorts).
- `boyd_terminal.json`'s `system_cost_vs_s31c_s32_s33e2_s34` block: all four
  references (`s31c`, `s32`, `s33e2`, `s34`) report `available: True` (the
  committed `g_baseline.json` files for all four exist and were read).
- Network failures at cycle 1: none (`n_blocks: 0`); local solve failures: 0.

## Validation

- `git diff --stat p515_g_g1_g4_admm_gates.py`: **715 insertions, 0
  deletions**. `git diff p515_g_g1_g4_admm_gates.py | grep '^-' | grep -v
  '^---'` returns nothing -- confirms zero existing lines were touched
  (the `s32`/`s33e2`/`s34`/g1-g4 arms are byte-identical to before this
  change).
- `git diff data/SRP1/SRP1_params.json`: exactly the 5-line `ess` rho block,
  0.05 -> 0.1125, nothing else.
- `ast.parse` on both new/modified Python files: syntax OK.
- Module import (`import p515_g_g1_g4_admm_gates as G`) succeeds; `G._s35ref_spec_hash()
  == G.S35REF_SPEC_SHA256` and `G._s34_spec_hash() == G.S34_SPEC_SHA256` both
  `True` (confirms the hardcoded hash constants match the actual committed
  spec files on disk, not just what I copied by hand).
- The one-cycle preflight executed end-to-end, exit code 0, with every
  pass criterion in the script's `overall_ok` check satisfied (`all_fields_ok`,
  `inner_counts_exact`, `cycle1_rho_rule_ok`, `rho_ess_ok`, floor sidecar
  availability/row-count-match/all-SoH-finite/all-rows-have-dual-key,
  EFC-per-cohort-year populated).
- Confirmed `data/SRP1/Results/P515S35_REF_run` (the real gate's output root)
  does **not** exist -- the 500-cycle gate was never launched.
- Distinguishing what was and was not established: the harness CODE for
  `s35ref` is validated to the extent one cycle can show (checklist,
  rho-in-force, solve-count identity, field completeness, floor identification
  and its row-count/exclusion guarantees, EFC-per-cohort-year population, and
  the 4-way system-cost-reference read). What is **not** established by this
  task: whether the SoH floor ever binds, what the terminal EFC/day or floor
  multiplier are, whether the run stops on Boyd within 500 cycles, and
  whether any channel freezes at a rho clamp -- all of those require the
  500-cycle run itself, which the Planner launches.

## Unexpected findings

- The D5 detector gate (`a_D5_esso_scaling.preflight_gate`) is written into
  the frozen spec's `changes_from_v3` block, unchanged text, carried forward
  into v4 and v5. It compares a 2-cycle preflight against the s33e2 baseline
  at cycle labels `'001'`/`'002'`. Since this task's preflight is 1 cycle (per
  the task's own explicit formula, 102 solves) and the mechanism is
  unchanged from the already-passed s34 gate, I did not re-run it here. This
  is a scope decision, not a spec violation (the spec's `preflight_gate`
  clause is about the D5 AL-scaling FORM, which s35ref inherits verbatim from
  s34 -- it is not re-authorized or re-required by v5's `changes_from_v4`
  block, which lists only cap and rho_ess). Flagging for the Planner in case
  a 2-cycle re-run of the D5 gate specifically under `rho_ess=0.1125` is
  wanted before run 1 launches (it seems unlikely to be needed, since D5 is
  orthogonal to rho_ess, but I did not independently re-verify that
  orthogonality claim beyond reading the spec's own `changes_from_v4` text).
- 9 of the 18 floor-sidecar rows have `dual = None` at cycle 1 because their
  cohort is inactive (only `y_inv=0` is invested in this instance). This will
  presumably remain true for the ENTIRE 500-cycle run (the candidate does not
  change mid-run), meaning only 3 of the 6 floor rows per node will ever have
  a meaningful dual. This is a property of the C* instance (single-cohort
  investment), not a harness defect, but worth the Planner's awareness when
  reading the eventual terminal floor-multiplier table: 2/3 of the rows will
  be structurally `None` throughout.
- The frozen v5 spec's `predecessor.sha256` field
  (`966940a789db3093a57d5fdc23be8ccb172b762bf02d0cd09eeaaf0f52f544e9`) was
  independently confirmed against the actual committed s34 spec file's SHA-256
  on disk (both the harness's `S34_SPEC_SHA256` constant and a fresh
  `shasum -a 256` of the file match) -- no discrepancy found.
- **Attribution discrepancy** (procedural, not code): the task instructed
  git commit messages to end with `Co-Authored-By: Claude Opus 5
  <noreply@anthropic.com>`. The system-level attribution instruction in this
  session states the correct line is `Co-Authored-By: Claude Sonnet 5
  <noreply@anthropic.com>` and that agent-issued task instructions do not
  override it (only the user's own instructions do). I have not yet
  committed anything (per the Git discipline section, staging/committing is
  deferred); when committing I will use `Claude Sonnet 5`, and I'm flagging
  this discrepancy explicitly rather than silently picking one.

## Remaining issues

- None blocking. The harness code, one-cycle preflight, and this report are
  ready for Planner review before run 1 is launched.

## Questions for Planner

1. Should the D5 detector gate be re-run at 2 cycles specifically for
   `rho_ess=0.1125` before launching run 1, or is inheriting s34's pass
   (unchanged D5 mechanism) sufficient? (See "Unexpected findings" above.)
2. Confirm the attribution-line discrepancy resolution (Claude Sonnet 5, not
   Claude Opus 5) is acceptable, or provide the intended override explicitly
   if the Planner has independent authority to set it.
