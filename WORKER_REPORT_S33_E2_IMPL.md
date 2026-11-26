# Worker Report -- P5.15 Step 3.2 E2 implementation (s33e2)

## Task received

Implement frozen spec v3
(`data/SRP1/Results/P515S33/frozen_s33_e2_spec_v3_825f1f02.json`, sha256
`825f1f02d5319137e0248fc529b9aefdca0251272529ee140e09af6e73a04780`,
supersedes v2 `data/SRP1/Results/P515S32/frozen_s32_spec_v2_516bd749.json`):
gamma tied to rho (gamma_c = tau*rho_c per channel, tau=1), rho/gamma
adaptation frozen after cycle 30, `minimum_consecutive_converged_cycles=3`.
Add the `s33e2` harness arm. Run zero-solve checks plus a two-cycle
preflight. **Do NOT launch the 150-cycle gate.**

## Files inspected

- `data/SRP1/Results/P515S33/frozen_s33_e2_spec_v3_825f1f02.json` (hash
  verified against its own filename and against `predecessor.sha256`,
  which matches `data/SRP1/Results/P515S32/frozen_s32_spec_v2_516bd749.json`'s
  own hash).
- `PLANNER_BRIEF_2026-09-13.md` Addendum 14.
- `P5_15_S32_BOYD_GATE_REPORT.md`, `WORKER_REPORT_S32V2_IMPL.md`.
- `shared_resources_planning.py`: `update_transmission_model_to_admm`
  (:3911-4046), `get_admm_boyd_residual_metrics` (:5539-5775),
  `_get_admm_penalty_summary`/`_update_admm_penalties` (:6121-6403), the
  ADMM cycle loop and `admm_diagnostics` dict literal in
  `_run_operational_planning` (:2449-2899).
- `admm_parameters.py` (`__init__`, `_read_parameters_from_file`).
- `p515_g_g1_g4_admm_gates.py`: the `s32` arm section in full
  (:1925-2417) as the template for `s33e2`.
- `p515_s32_zero_solve_checks.py`, `p515_s32v2_zero_solve_checks.py`,
  `p515_s32_preflight.py`, `p515_s32v2_preflight.py` (precedents, helpers
  reused by import).
- `data/SRP1/SRP1_params.json` `admm` block (confirmed pre-edit state:
  `rho`=1.0 on every network/channel, `proximal_regularization.tso.gamma`
  =1.0 on v/pf/ess, `minimum_consecutive_converged_cycles`=1, no
  `penalty_update.freeze_after_cycle` key).
- `data/SRP1/Results/P515S32_run/g_baseline.json` (the committed s32 150-cycle
  result -- used READ ONLY as the identity-cycles-1-2 reference and as an
  `s33e2` system-cost comparison target; never re-run/overwritten).
- `data/SRP1/Results/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl`
  and `matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl` (confirmed,
  by direct inspection, `prox_gamma_v.mutable == False`, value `1.0` --
  the "s32 FrozenSMOPF pickles with immutable gamma" the task refers to).
- Commits `d06f3636` (pre-v3 Boyd/balancing production code) and `0bde2c25`
  (pre-v3 harness/preflight), loaded via `git show` as separate modules for
  the zero-solve bit-identical/numeric comparisons.

## Files modified

Production (commit `a00cbb29`):
- `admm_parameters.py`
- `shared_resources_planning.py`
- `data/SRP1/SRP1_params.json`
- New: `p515_s33e2_zero_solve_checks.py`
- New (results): `data/SRP1/Results/P515S33/zero_solve_checks_e2/zero_solve_checks_e2.json`,
  `manifest_sha256.json`

Harness (this commit):
- `p515_g_g1_g4_admm_gates.py` (new `s33e2` section only; every other arm,
  including `s32`, unchanged in behaviour -- see Validation).
- New: `p515_s33e2_preflight.py`
- New (results): `data/SRP1/Results/P515S33/preflight_e2/*` (22 files) +
  `manifest_sha256.json`.
- This report.

## Changes made

### 1. `admm_parameters.py`

- `__init__`: added `penalty_update['freeze_after_cycle'] = None` (with a
  comment); added `proximal_regularization['tso']['gamma_policy'] = 'fixed'`
  and `['tau'] = 1.0`. `dso` is untouched (no `gamma_policy`/`tau` key --
  the task scoped this to `tso` only).
- `_read_parameters_from_file`: `freeze_after_cycle` is excluded from the
  generic float-coercion loop over `penalty_update` (it is an optional int
  or `None`, not a float penalty coefficient) and parsed separately, with a
  non-negativity check. `gamma_policy` (validated against
  `('fixed', 'tied_to_rho')`) and `tau` (validated `> 0`) are read from
  `proximal_data['tso']` only, defaulting to the `__init__` values when
  absent -- **every other case study's params file loads unchanged**
  (confirmed: CS1/CS7/HR1/OP1/OP2 all report `gamma_policy='fixed'`,
  `tau=1.0`, `freeze_after_cycle=None`).

### 2. `data/SRP1/SRP1_params.json`

Exactly the four edits authorized:
`admm.proximal_regularization.tso.gamma_policy = "tied_to_rho"`,
`.tau = 1.0`, `admm.penalty_update.freeze_after_cycle = 30`,
`admm.minimum_consecutive_converged_cycles = 3`. `git diff --stat`:
`+7/-2` (2 hunks). No other key touched -- `rho`, `gamma` (the v/pf/ess
values, still 1.0), `num_max_iters` (harness overrides this to 150 via
`num_max_iters_override`, per the s32 precedent), tolerances, etc. are
byte-identical.

### 3. `shared_resources_planning.py` (diff hunks, file:line as of commit `a00cbb29`)

1. `update_transmission_model_to_admm` (~3916-3960): reads
   `tso_gamma_policy`/`tso_gamma_tau` once from `tso_proximal_cfg`; the
   `prox_gamma_v/pf/ess` Params are now `pe.Param(mutable=True, ...)`
   (were immutable); under `'tied_to_rho'` each is initialized to
   `tau * (that TSO model's initial rho for the channel)` (the SAME
   `params.rho[...][transmission_network.name]` value used to initialize
   `rho_v/pf/ess` just below); under `'fixed'` (default) initialization is
   unchanged (the configured `tso_proximal_cfg['gamma'][...]`).
2. `get_admm_boyd_residual_metrics` docstring (~5544-5565): cites spec v3,
   documents that gamma is now read from the TSO model's Params in force.
   `gamma_tso = proximal_cfg['tso']['gamma']` (the case-file-constant
   read) removed; the per-`(year, day)` `gamma_v/pf/ess` assignment
   (~5657-5662) now reads `pe.value(tso_model[year][day].prox_gamma_v)`
   etc. (0.0 when the TSO proximal term is disabled, unchanged).
3. New `_get_admm_gamma_summary(tso_model)` (~6121-6139), mirroring
   `_get_admm_penalty_summary`'s TSO iteration/averaging convention but
   TSO-only (gamma has no DSO/ESSO Param); returns `None` per group when
   disabled.
4. `_update_admm_penalties` (~6179-6403): new `iter=None` parameter
   (keyword-compatible with the pre-v3 3-positional-plus-kwargs call
   signature -- old call sites that never pass `iter` get `frozen=False`,
   i.e. unchanged behaviour); `frozen = freeze_after_cycle is not None and
   iter is not None and iter > freeze_after_cycle`, computed once at the
   top from `params.penalty_update.get('freeze_after_cycle')`. Per-group
   action selection: `if frozen: action = f'held (frozen after cycle
   {freeze_after_cycle})'` is now the FIRST branch (highest precedence),
   ahead of `not params.adaptive_penalty` / `not allow_update` /
   increase-decrease. The rho-scaling block's guard is now
   `params.adaptive_penalty and allow_update and not frozen` (was `...and
   allow_update`); when `gamma_policy == 'tied_to_rho'`, immediately after
   the rho-scaling loops (same guarded block), every TSO model's
   `prox_gamma_c` is reset to `tau * (that model's just-scaled rho_c)`.
   Returns a 6-tuple `(actions, before, after, before_gamma, after_gamma,
   frozen)` (was 3-tuple `(actions, before, after)`); the two pre-v3
   zero-solve check scripts that unpack a 3-tuple (`p515_s32_zero_solve_checks.py`,
   `p515_s32v2_zero_solve_checks.py`) would raise `ValueError` if literally
   re-run now -- their committed JSON outputs remain valid, frozen evidence
   for the code as it existed when they ran (same precedent as
   `WORKER_REPORT_S32V2_IMPL.md`'s "Unexpected findings").
5. `[ADMM RHO BOYD]` print (~6386-6402): added `gamma_before=... |
   gamma_after=...` (formatted `N/A` when `None`).
6. `_run_operational_planning`'s cycle loop (~2696): call site updated to
   the 6-tuple and passes `iter=iter`.
7. `admm_diagnostics` dict literal (~2793-2805): added
   `gamma_v_before/after`, `gamma_pf_before/after`, `gamma_ess_before/after`,
   `rho_freeze_active`, `freeze_after_cycle`, `gamma_policy`, `gamma_tau`.

**Freeze / failure-hold precedence implemented (stated explicitly in the
`_update_admm_penalties` docstring and verified by check (d)):** frozen
(highest) > `not params.adaptive_penalty` ('fixed') > `not allow_update`
('held after solver failure', own label, but ONLY reached when NOT frozen)
> ordinary increase/decrease/dead-band. Frozen overrides a simultaneous
failure hold (verified: iter=31 with `allow_update=False` still yields the
frozen label, not the failure-hold label); the failure hold keeps its own
label at iter=29 (`allow_update=False`, not frozen).

`git diff --stat -- shared_resources_planning.py`: `173 ++++/-23-` across
15 hunks, all confined to the four permitted sites (`_run_operational_planning`'s
call site + diagnostics dict, `update_transmission_model_to_admm`,
`get_admm_boyd_residual_metrics`, `_get_admm_penalty_summary`/`_update_admm_penalties`)
-- confirmed by direct inspection of every `@@` hunk header.

**Not touched:** the Boyd r/s/eps formulas beyond the gamma source (spec
v2's `dual_ratio_balance` mechanism is untouched), the balancing ratio
definitions/dead band/factors/clamp, the multiplier updates (including the
pre-loop lambda update), normalization, sigma, ESSO scaling, objectives,
settlement, `_run_operational_planning_hierarchical`,
`_run_operational_planning_without_coordination` (confirmed byte-identical
to HEAD by zero-solve check (j)).

### 4. `p515_g_g1_g4_admm_gates.py`: new arm `s33e2`

Same machinery as `s32` (module-level spec path/hash constants, rule-eleven
checklist function, per-cycle rho/gamma trajectory helpers, system-cost-vs-
reference helper (extended to compare against BOTH s31c and s32), the
`boyd_terminal.json` writer, and the CLI dispatch branch). New:
`assert_s33e2_capture_paths` (spec v3 hash, `gamma_policy`, `tau`,
`freeze_after_cycle`, `minimum_consecutive_converged_cycles`, the new
diagnostics keys, and a structural check that `prox_gamma_v` is
constructed with `mutable=True`); `_s33e2_gamma_trajectory`;
`_s33e2_system_cost_vs_references` (s31c AND s32, at the spec's matched
cycles); `_interface_voltage_detail`/`write_interface_voltage_terminal`
(new writer, top-level `{"entries": [...], "summary": {...}}`, metadata
folded into `summary`); `write_boyd_terminal_s33e2` (calls
`write_interface_settlement_detail_s31c` -- which itself calls
`write_component_levels_terminal` -- FIRST, exactly as `s32` does, then
`write_interface_voltage_terminal`, then assembles `boyd_terminal.json`
with the spec v3 `report_terminal` fields, including the E4 noise-floor
`delta_c` values alongside the terminal per-channel Boyd `r`/`s`). Output
root `data/SRP1/Results/P515S33_E2_run` (refuses if it exists); label
`baseline`. No other arm's code was touched (verified: `git diff` shows a
single new, self-contained section plus one new `elif gate == 's33e2':`
branch; the `s32` section above it is byte-for-byte what commit `0bde2c25`
left it as -- confirmed by `git diff HEAD~1 -- p515_g_g1_g4_admm_gates.py`
touching only new lines, no deletions inside the pre-existing arms).

### 5. `p515_s33e2_zero_solve_checks.py` (new)

Ten checks (a)-(j), `SolveProfileGuard(permitted=())` for the whole run.
Output: `data/SRP1/Results/P515S33/zero_solve_checks_e2/` (new directory;
refused if it existed).

### 6. `p515_s33e2_preflight.py` (new)

Runs `run_admm_arm` (by import) for exactly 2 cycles through the `s33e2`
machinery into `data/SRP1/Results/P515S33/preflight_e2/` (fresh root).
Checks the field-completeness of every `report_per_cycle_channel` field
(now including gamma/freeze), the `identity_cycles_1_2` prediction against
the committed `data/SRP1/Results/P515S32_run/g_baseline.json` (read only),
the cycle-2 expectation (V decreased, `rho_v_after == gamma_v_after ==
0.6667`), and that `interface_voltage_terminal.json` is written and sane.

## Commands / experiments run

1. `python -c "import ast; ast.parse(...)"` on every modified/new file --
   syntax OK after each edit.
2. `python -c "..."` loading `ADMMParameters` directly against
   `data/SRP1/SRP1_params.json` and against CS1/CS7/HR1/OP1/OP2's own
   params files -- confirmed the four case-file edits load as intended and
   every OTHER case study loads with the unchanged defaults.
3. Standalone `assert_s33e2_capture_paths(planning)` call, under a
   zero-permitted `SolveProfileGuard`, on a freshly built (never solved)
   planning object -- passed; scratch eval dir removed afterward.
4. `python p515_s33e2_zero_solve_checks.py` -- 10/10 checks pass,
   `SolveProfileGuard` counts all zero, `verify_failures=[]`. (One
   iteration: check (g)'s structural-diff keyword filter initially flagged
   a bare `else:` line as "offending"; widened the allow-list to admit
   that one, unambiguous, control-flow-only line rather than the keyword
   set generally -- re-run confirmed `structural_diff_confined_to_gamma:
   true`, `objective_values_match: true`.)
5. `python p515_s33e2_preflight.py` (real IPOPT solves, 2 cycles, 83 s wall
   time) -- see Results below.
6. `sha256sum`-equivalent manifest generation (Python `hashlib`) for
   `zero_solve_checks_e2/` and `preflight_e2/`.
7. Scratch eval directories under `data/SRP1/Results/P56A/evals/` created
   by (3), (4) and (5)'s precheck calls were removed after each run;
   `data/SRP1/Results/P515S33_E2_run/` (the real gate's output root) was
   **never created** -- confirmed by `ls`.

Interpreter used throughout: `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`.
Solver: production resolves `NLP_SOLVER_PATH` from `.env`
(`/usr/local/bin/ipopt`), unchanged by this task. Neither the 150-cycle
gate command nor any run longer than the 2-cycle preflight was executed.

## Results

### Zero-solve checks (`p515_s33e2_zero_solve_checks.py`)

All 10 checks (a)-(j) pass; `solve_profile_guard.counts = {permitted_solve:
0, permitted_exec: 0, blocked_solve: 0, blocked_exec: 0}`,
`verify_failures = []`.

| check | result | key evidence |
|---|---|---|
| (a) tied gamma tracks rho | **pass** | fresh build: gamma_v/pf/ess == tau*rho_v/pf/ess on every TSO model immediately after construction; synthetic decrease (all channels, `dual_ratio_balance`-driven) -> rho 1.0->0.6667, gamma 1.0->0.6667 on every channel/model; synthetic increase -> rho 0.6667->1.0, gamma 0.6667->1.0. |
| (b) fixed gamma | **pass** | explicit `gamma_policy='fixed'` override: gamma stays at the configured 1.0 across a decrease+increase synthetic update; legacy metrics, every Boyd field (per channel), and the update's actions/rho/before/after are numerically identical to commit `d06f3636` on a fresh single-entry-perturbed snapshot. |
| (c) Boyd uses model gamma | **pass** | `prox_gamma_v` set to `case_file+7.0` (8.0) on the model; `s`/`s_proximal_part` follow 8.0, not the unchanged case-file constant 1.0 (`follows_model_value=True`, `does_not_follow_case_file_constant=True`). |
| (d) freeze | **pass** | iter=30 (not frozen, 30>30 False): provocative ratio acts (`increased`, sanity check `acted_at_30=True`); iter=31 (frozen): `action == 'held (frozen after cycle 30)'` on every channel, `after == before` exactly, gamma unchanged; iter=31 + `allow_update=False`: STILL the frozen label (frozen precedence over failure hold); iter=29 + `allow_update=False`: `'held after solver failure'` (failure hold's own label, not frozen). |
| (e) stop rule | **pass** | source-text confirms the exact `+= 1`/`= 0`/`>=` lines and that `cycle_convergence = boyd_all_pass and local_solves_ok` (no `objective_convergence`); mechanical simulation of that quoted formula over `[T,T,F,T,T,T]` with minimum=3: resets to 0 at the F (cycle 3), stops (first `convergence=True`) at cycle 6, i.e. the THIRD consecutive pass after the reset. |
| (f) update isolation | **pass** | `consensus_vars`/`dual_vars` bit-identical before/after (dict equality); rho_v 1.0->1.5 (`increased`), gamma_v tracks (1.5 == 1.0*1.5). |
| (g) objective structure | **pass** | (g1) diff of `update_transmission_model_to_admm` vs `d06f3636`: 2 hunks / 23 changed lines, every one confined to gamma/tau/policy/mutable keywords or the bare `else:` introducing the fixed branch (`offending_lines=[]`); (g2) with gamma=1 on both sides (current code, `'fixed'` override, vs `d06f3636`, always-fixed), the built `admm_objective`, evaluated at a hand-completed (zero-valued `*_req`/`dual_*` Params) zero-solve state, is the SAME number on both sides: `36.0 == 36.0`. |
| (h) other cases | **pass** | CS1/CS7/HR1/OP1/OP2 all load `gamma_policy='fixed'`, `tau=1.0`, `freeze_after_cycle=None`; SRP1 loads `gamma_policy='tied_to_rho'`, `tau=1.0`, `freeze_after_cycle=30`, `minimum_consecutive_converged_cycles=3`. |
| (i) fixtures | **pass** | 3/3 S31C fixtures unpickle; the two s32 FrozenSMOPF pickles unpickle, with `prox_gamma_v.mutable == False` confirmed on both (pre-v3 immutable Params, unaffected by this change). |
| (j) untouched paths | **pass** | `_run_operational_planning_hierarchical`/`_run_operational_planning_without_coordination` byte-identical to HEAD; `diff_hunk_count=15`, every hunk inside the four permitted sites. |

### Preflight (`p515_s33e2_preflight.py`, two real ADMM cycles, C\* candidate)

`cycles_run=2`, `local_solve_failures=0`. Solve counts checked **exactly**
against 153 (`51*2+51`): `{'permitted_solve': 153, 'permitted_exec': 153,
'blocked_solve': 0, 'blocked_exec': 0}` (`identity_holds=True`).
`network_failures.n_blocks=0`. Every required `report_per_cycle_channel`
field (Boyd fields, rho fields, AND the new `gamma_{v,pf,ess}_before/after`,
`rho_freeze_active`, `freeze_after_cycle`, `gamma_policy`, `gamma_tau`)
populated and finite on both cycles (`field_completeness_all_ok=True`).

**`identity_cycles_1_2` HOLDS EXACTLY.** Comparing cycles 1 and 2 against
the committed `data/SRP1/Results/P515S32_run/g_baseline.json` (read only,
never re-run) on `gross_operational_cost`, every `rho_{v,pf,ess}_before/
after/action`, and every `boyd_{v,pf,ess}_*` field (43 numeric fields
total, per cycle): **max absolute difference across every field and both
cycles = 0.0** (`identity_holds=True`, `first_divergence=None`). No
difference occurred before, at, or through cycle 2's update -- the run was
not stopped.

**Cycle-2 expectation confirmed exactly, matching s32:** `rho_v_action=
'decreased'`, `rho_v_after=0.6666666666666671`, and (new under the tied
policy) `gamma_v_after=0.6666666666666666` -- `gamma_tracks_rho_at_c2=True`
(difference `< 1e-9`). `cycle2_expectation.pass = True`.

**Cycle 1** (gamma=rho=1 everywhere, held on every channel): matches the
s32 v2 preflight's own cycle-1 values exactly (V `r=1.011663e-01`,
`s=5.302716e-02`; PF `r=3.640737e+00`, `s=7.127406e+00`; ESS
`r=5.287717e-03`, `s=5.875285e-03` -- identical to
`WORKER_REPORT_S32V2_IMPL.md`'s reported cycle-1 table).

`gross_operational_cost`: cycle 1 = 968,494,925.0938367 (identical to s32),
cycle 2 = 1,219,820,650.0962644 (identical to s32).

**`interface_voltage_terminal.json`:** written, top-level shape exactly
`{"entries": [...], "summary": {...}}`; 864 entries (3 nodes x 4 years x 4
days x 18 periods); `min_distance_to_bound_pu_overall = 0.0905` (node 5),
per-node minima 0.0905/0.0964/0.0907 (nodes 5/7/9); `n_entries_at_bound_
within_1e-6_pu = 0`, `n_entries_within_0p005_pu = 0` -- no interface
voltage is close to a TSO node bound at cycle 2 of this candidate. `sane`
check (top-level key set, entry count matches `n_entries`, every entry's
`tso_pu` on the correct side of at least one bound) passes.

## Validation

- Code executes correctly: confirmed (zero-solve checks + a real two-cycle
  preflight, both exited 0 and produced the expected JSON outputs).
- Requested diagnostic works: confirmed -- every new capture field
  (gamma/freeze/policy/tau) is populated on real cycles; `boyd_terminal.json`
  and `interface_voltage_terminal.json` are written with the spec-required
  shape; `assert_s33e2_capture_paths` passes on a fresh, never-solved
  planning object.
- Underlying numerical question (does the E2 stabiliser change the 150-cycle
  trajectory, and does it converge under the Boyd rule within 150 cycles)
  is **NOT** assessed here -- that requires the full gate, which this task
  explicitly does not launch.
- `git diff` hunk inspection (15 hunks in `shared_resources_planning.py`,
  2 in `data/SRP1/SRP1_params.json`, all confined to the permitted sites)
  plus the zero-solve checks' independent source-text/byte-identity/
  numeric-value assertions confirm no unintended production modification.
- The exact gate command
  (`python -u p515_g_g1_g4_admm_gates.py s33e2 > data/SRP1/Results/P515S33_E2_launch.log 2>&1`)
  was **not** executed; `data/SRP1/Results/P515S33_E2_run/` does not exist.

## Freeze and failure-hold precedence (as implemented)

Frozen (`freeze_after_cycle is not None and iter > freeze_after_cycle`) is
the FIRST, highest-precedence branch: when true, every channel's action is
`'held (frozen after cycle N)'` with no rho or gamma change, REGARDLESS of
`allow_update`/a failed local solve. The failure hold (`'held after solver
failure'`) keeps its own label and only applies when NOT frozen. Verified
in zero-solve check (d) at the exact boundary (iter=30 not frozen and
acting; iter=31 frozen and held; iter=31 with a simultaneous failure held
under the frozen label; iter=29 with a failure held under its own label).

## Preflight identity comparison (max difference per field class, cycles 1-2)

| field class | max abs diff (cycles 1-2, vs committed s32 g_baseline.json) |
|---|---|
| `gross_operational_cost` | 0.0 |
| `rho_{v,pf,ess}_before/after` (6 fields) | 0.0 |
| `boyd_{v,pf,ess}_{r,s,s_rho_part,s_proximal_part,eps_pri,eps_dual,norm_x,norm_z,norm_y,primal_ratio,dual_ratio,dual_ratio_balance}` (36 numeric fields) | 0.0 |
| `rho_{v,pf,ess}_action` (categorical) | exact string match |

43/43 numeric fields compared, every one 0.0 exactly (double-precision
floats produced by the identical code path up to the point gamma/rho would
first diverge, which is cycle 3 per the spec's own prediction -- not
reached by this 2-cycle preflight).

## Cycle-2 rho and gamma after

`rho_v_after = gamma_v_after = 0.6666666666666671/0.6666666666666666`
(V decreased at end of cycle 2, matching s32; gamma newly tracks it under
the tied policy). PF and ESS: `rho_after = gamma_after = 1.0` (held, both
channels, both quantities unchanged).

## Solve count

153 permitted solves for the 2-cycle preflight (`51*2+51`, exact match to
the arm's own internal identity check); 0 permitted/blocked solves for
either zero-solve check run (10 checks total, `p515_s33e2_zero_solve_checks.py`,
under a `permitted=()` blocking guard).

## Summary of `interface_voltage_terminal.json`

864 per-entry records (node x year x day x period, TSO and DSO copies in
pu, plus the TSO node's own `[v_min_pu, v_max_pu]` and the signed distance
to the nearer bound). At this candidate and cycle (2 of the preflight),
every interface voltage sits comfortably inside its bounds: minimum
distance to a bound is 0.0905 pu (node 5), and no entry is within 0.005 pu
of either bound. This is a 2-cycle, non-terminal snapshot -- the terminal
(150-cycle or Boyd-stop) picture is a matter for the full gate.

## Deviation from the prompt / spec

None found. Where the prompt and the spec overlapped, they agreed; no
discrepancy between `frozen_s33_e2_spec_v3_825f1f02.json` and the prompt's
"Verified code facts" / "Changes PERMITTED" sections was found during
implementation.

## Unexpected findings

- None beyond the expected, precedent-consistent breakage of the v1/v2
  zero-solve check scripts' fake-boyd-metrics helpers if literally re-run
  against the new `_update_admm_penalties` signature/return arity (same
  category as the "Unexpected findings" entry in `WORKER_REPORT_S32V2_IMPL.md`)
  -- not treated as a defect; their committed JSON outputs remain valid,
  frozen evidence for the code as it existed when they ran, and neither was
  re-run onto its own committed path by this task.

## Remaining issues

- The `s33e2` 150-cycle gate itself has not been run; only the Planner
  launches it. The exact command
  (`python -u p515_g_g1_g4_admm_gates.py s33e2 > data/SRP1/Results/P515S33_E2_launch.log 2>&1`)
  is wired, and its rule-eleven checklist (spec v3 hash, `gamma_policy`,
  `tau`, `freeze_after_cycle`, `minimum_consecutive_converged_cycles`,
  initial rho, per-cycle-channel/diagnostics-key capture paths) is verified
  to pass, without running the campaign.
- `check_g_objective_structure`'s "structural confinement" (g1) uses a
  keyword/allow-list filter over the unified diff against commit
  `d06f3636`, plus one explicit allowance for a bare `else:` line -- this
  is a heuristic, not a formal proof that no other change occurred; the
  numeric objective-value equality (g2) at gamma=1 is the stronger,
  independent check.

## Questions for Planner

- None blocking.
