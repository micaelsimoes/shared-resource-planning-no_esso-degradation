# Worker Report -- P5.15 Step 3.2 + 3.3(a) frozen spec v2 implementation (s32)

## Task received

Apply frozen spec v2 (`data/SRP1/Results/P515S32/frozen_s32_spec_v2_516bd749.json`,
supersedes v1 `frozen_s32_spec_v1_14a18674.json`) to the s32 residual
balancing: the balancing decision's dual ratio must use
`s_rho_part / eps_dual` (`dual_ratio_balance`) instead of the full
`s / eps_dual` (`dual_ratio`), for the increase/decrease thresholds ONLY.
The Boyd stopping test is unchanged and still uses the full `s`. **Not**
authorized to launch the 150-cycle s32 gate (`p515_g_g1_g4_admm_gates.py
s32`); that remains the Planner's call.

## Files inspected

- `data/SRP1/Results/P515S32/frozen_s32_spec_v2_516bd749.json` (hash
  verified: `516bd749c5eff6c5f26c0dac570d10964ca989b2c64394ac1ba5deca32356b85`,
  matches filename and `sha256sum`).
- `data/SRP1/Results/P515S32/frozen_s32_spec_v1_14a18674.json` (hash
  verified: `14a18674c2f2f7119781fafb72c569a3a4788b8edbfc821a9ffc2921934e7356`,
  matches `predecessor.sha256` recorded in the v2 spec).
- `WORKER_REPORT_S32_IMPL.md`, `WORKER_REPORT_S32_MAPPING_DIAG.md`
  (background: v1 implementation, mapping diagnostic).
- `shared_resources_planning.py`: `get_admm_boyd_residual_metrics`
  (:5519-5750), `_update_admm_penalties` (:6093-6234), the `admm_diagnostics`
  dict literal in `_run_operational_planning` (:~2780-2868).
- `p515_g_g1_g4_admm_gates.py`: the s32 arm section (:1925-2410) --
  `S32_SPEC_PATH`/`S32_SPEC_SHA256`, `S32_REPORT_PER_CYCLE_CHANNEL_FIELDS`,
  `S32_ADMM_DIAGNOSTICS_KEYS`, `assert_s32_capture_paths`,
  `_s32_binding_test`, `write_boyd_terminal_s32`, the `s32` CLI dispatch.
- `p515_s32_zero_solve_checks.py`, `p515_s32_preflight.py` (v1 precedents,
  reused/copied per the task's explicit permission).
- `admm_parameters.py` (`proximal_regularization`, `penalty_update`
  defaults) and `data/SRP1/SRP1_params.json` `admm` block (confirmed
  `rho`=1.0, `proximal_regularization.tso.gamma`=1.0 on v/pf/ess, matching
  the spec's cycle-1 rho=gamma=1 assumption).

## Files modified

- `shared_resources_planning.py` (production).
- `p515_g_g1_g4_admm_gates.py` (harness, s32 arm section only).
- New: `p515_s32v2_zero_solve_checks.py`.
- New: `p515_s32v2_preflight.py`.
- New (results): `data/SRP1/Results/P515S32/zero_solve_checks_v2/zero_solve_checks_v2.json`,
  `manifest_sha256.json` (`synthetic_admm_snapshot_v2.pkl`, ~104 MB,
  hash-recorded, NOT committed).
- New (results): `data/SRP1/Results/P515S32/preflight_v2/*` (18 files) +
  `manifest_sha256.json`.

## Changes made

### `shared_resources_planning.py` (diff hunks, file:line as of this commit)

1. `get_admm_boyd_residual_metrics` docstring (~:5519-5535): cites spec v2,
   documents that `dual_ratio_balance` feeds ONLY the balancing decision and
   `dual_ratio` (full `s`) still feeds the stopping test.
2. `get_admm_boyd_residual_metrics`, per-channel loop (~:5726-5731): added
   `dual_ratio_balance = (s_rho_part / eps_dual) if eps_dual > 0.0 else
   float('inf')` (same inf guard as `dual_ratio`), added to `channel_entry`
   next to `dual_ratio`.
3. `_update_admm_penalties` docstring (~:6111-6127): cites spec v2, states
   the balancing decision now reads `boyd_metrics[group]['dual_ratio_balance']`
   instead of `['dual_ratio']`.
4. `_update_admm_penalties`, ratio read (~:6168-6169): `boyd_dual_ratio` ->
   `boyd_dual_ratio_balance = boyd_metrics[group]['dual_ratio_balance']`.
5. `_update_admm_penalties`, increase/decrease tests (~:6190-6193): both
   comparisons now use `boyd_dual_ratio_balance` in place of `boyd_dual_ratio`.
6. `_update_admm_penalties`, `[ADMM RHO BOYD]` print (~:6255): added
   `dual_ratio_balance={...:.6e}`.
7-9. `admm_diagnostics` dict literal (~:2812, :2828, :2847): added
   `'boyd_v_dual_ratio_balance'`, `'boyd_pf_dual_ratio_balance'`,
   `'boyd_ess_dual_ratio_balance'`.

`git diff --stat`: `shared_resources_planning.py | 47 ++++++++++++++++++++++++++++++++++----------`
(10 hunks total, all confined to the three permitted sites -- verified by
`p515_s32v2_zero_solve_checks.py` check (h), `diff_hunk_count=10`, all hunk
headers inside `_run_operational_planning`, `get_admm_boyd_residual_metrics`
or `_update_admm_penalties`).

### `p515_g_g1_g4_admm_gates.py` (s32 arm only)

1. Module header comment + `S32_SPEC_PATH`/`S32_SPEC_SHA256` (:1925-1935):
   now point at spec v2 (`frozen_s32_spec_v2_516bd749.json`,
   `516bd749...`), with an explicit "supersedes v1" note. `write_boyd_terminal_s32`
   reads these same module-level constants, so `boyd_terminal.json`'s
   `spec_file`/`spec_file_sha256` fields automatically record v2 -- no
   change needed there.
2. `S32_REPORT_PER_CYCLE_CHANNEL_FIELDS` (:1946): added `'dual_ratio_balance'`.
3. `S32_ADMM_DIAGNOSTICS_KEYS` (:1951, :1954, :1958): added
   `'boyd_v_dual_ratio_balance'`, `'boyd_pf_dual_ratio_balance'`,
   `'boyd_ess_dual_ratio_balance'` (checked, via `assert_s32_capture_paths`,
   against the literal presence of these keys in `shared_resources_planning.py`'s
   source -- structurally confirms the production change before any solve).
4. `_s32_binding_test` (:2070): added `'dual_ratio_balance':
   last_row.get(f'boyd_{group}_dual_ratio_balance')` to the per-channel
   binding-test dict (`report_terminal`'s "binding test per channel and
   ratio to threshold").
5. `s32` CLI dispatch comment (:2369-2374): updated to reference spec v2.

No other production hunks in this file (`p515_s32v2_zero_solve_checks.py`
check (h) confirms `_run_operational_planning_hierarchical` and
`_run_operational_planning_without_coordination` in `shared_resources_planning.py`
are byte-identical to HEAD; the other g1-g4/s30/s31/s31c/ablation arms in
`p515_g_g1_g4_admm_gates.py` were not touched, confirmed by inspection --
the s32 section is a self-contained block at the end of the file, and `git
diff` shows every hunk inside it).

## Commands / experiments run

1. `python -c "import ast; ast.parse(...)"` on all four modified/new files
   -- syntax OK, run after each edit.
2. `python p515_s32v2_zero_solve_checks.py` -- zero-solve verification, 8
   checks (a)-(h), `SolveProfileGuard(permitted=())`.
3. `python p515_s32v2_preflight.py` -- one real ADMM cycle through the (now
   v2) s32 arm machinery.
4. A standalone dry-run replication of the `s32` CLI gate's pre-solve
   checklist (per the task: "the checklist passes up to the point of the
   first solve, without running the campaign... a dry capture-path assertion
   call as before") -- imports `p515_g_g1_g4_admm_gates`, checks
   `N.REL == S32_REL`, checks the spec-v2 hash, builds a fresh planning
   object under a zero-permitted `SolveProfileGuard`, calls
   `assert_s32_capture_paths(planning)`. **Did NOT** invoke
   `python p515_g_g1_g4_admm_gates.py s32` itself (that call proceeds
   straight into `run_admm_arm`, i.e. the real 150-cycle campaign, which
   this task must not launch). Scratch eval directories from this dry run
   were removed afterward (`data/SRP1/Results/P56A/evals/p515s32v2_gate_dry_capture_check*`);
   `data/SRP1/Results/P515S32_run/` (the real gate's output root) was never
   created -- confirmed by `ls`.
5. `sha256sum`-based manifests written for `zero_solve_checks_v2/` and
   `preflight_v2/`.

Interpreter used throughout: `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`.
Solver: production resolves `NLP_SOLVER_PATH` from `.env`
(`/usr/local/bin/ipopt`), unchanged by this task.

**Note on a concurrent commit.** Partway through this task, commit
`1917924f` ("P5.15 Step 3.2/3.3(a): s32 gate evaluation script (committed
before the gate)") landed on this branch from outside this worker session
(a new file, `p515_s32_evaluate.py`, 185 lines). Checked:
`git diff 9603eee2 1917924f -- shared_resources_planning.py
p515_g_g1_g4_admm_gates.py` is empty -- that commit did not touch either
file this task modifies, so it does not conflict with or invalidate any
check run in this session.

## Results

### Zero-solve checks (`p515_s32v2_zero_solve_checks.py`)

All 8 checks pass; `solve_profile_guard.counts = {permitted_solve: 0,
permitted_exec: 0, blocked_solve: 0, blocked_exec: 0}`, `verify_failures = []`.

| check | result | key evidence |
|---|---|---|
| (a) balancing unit test | **pass** | increase/decrease/dead-band/no-freeze/failure-hold (5 sub-cases, same structure as v1) all pass under the `dual_ratio_balance`-gated code path, PLUS a 6th sub-case built from the spec's own numbers: V `dual_ratio`=178.13 (v1 would decrease, 178.13 > 5*31.34=156.7) vs `dual_ratio_balance`=125.96 (v2 holds, 125.96 <= 156.7); PF `dual_ratio`=7734.2 (v1 would decrease, > 3*2541.6=7624.8) vs `dual_ratio_balance`=5468.9 (v2 holds, <= 7624.8). Observed: `v1_rule_would_decrease_v=True`, `v1_rule_would_decrease_pf=True`, `v2_rule_decides_decrease_v=False`, `v2_rule_decides_decrease_pf=False`, `v2_action_v='held'`, `v2_action_pf='held'`. |
| (b) stop-rule uses full `s` | **pass** | real `get_admm_boyd_residual_metrics` call, single-entry V perturbation, rho=1.0/gamma=1.0 (case-file default), `dz_norm=0.02` -> `s_rho_part=0.02`, `s=0.028284`; `eps_dual` set to `0.024142` (strictly between). Observed: `s_rho_part=0.02 <= eps_dual=0.024142` (`dual_ratio_balance=0.828 <= 1`) **and** `s=0.028284 > eps_dual` (`dual_ratio=1.172 > 1`, `dual_pass=False`, `channel_pass=False`) -- the stopping test correctly blocks on the full `s` even though the balancing ratio alone would look "converged". |
| (c) penalties isolated | **pass** | `consensus_vars_unchanged=True`, `dual_vars_unchanged=True`, rho changed as expected. |
| (d) legacy + Boyd bit-identical to commit `2d765573` | **pass** | on a pickled-and-reloaded synthetic snapshot (`synthetic_admm_snapshot_v2.pkl`, includes `dual_vars`, unlike v1's fixture which omitted them), every legacy metric and every pre-existing Boyd field (r/s/s_rho_part/s_proximal_part/eps_pri/eps_dual/norm_x/norm_z/norm_y/primal_ratio/dual_ratio/passes/n_entries, per channel, plus top-level eps_abs/eps_rel/boyd_eps_source/all_boyd_pass) match exactly; `new_only_keys == {'dual_ratio_balance'}` on every channel. |
| (e) AL objective / proximal-centre unchanged vs `2d765573` | **pass** | all 6 functions byte-identical (sha256 match). |
| (f) other case params load | **pass** | CS1 `boyd_eps_source='default'`, SRP1 `='case_file'`, both with `{eps_abs: 1e-5, eps_rel: 1e-4}`. |
| (g) preserved fixtures unpickle | **pass** | 3/3 S31C fixtures load. |
| (h) hierarchical/uncoordinated untouched | **pass** | both functions byte-identical to HEAD; `diff_hunk_count=10`, all 10 hunks confined to `_run_operational_planning`'s `admm_diagnostics` dict, `get_admm_boyd_residual_metrics`, and `_update_admm_penalties`. |

### Preflight (`p515_s32v2_preflight.py`, one real ADMM cycle, C\* candidate)

`cycles_run=1`, `local_solve_failures=0`, solve counts checked **exactly**
against 102: `{'permitted_solve': 102, 'permitted_exec': 102,
'blocked_solve': 0, 'blocked_exec': 0}` (`identity_holds=True`,
`51*1+51=102`), `network_failures.n_blocks=0`. Every required
`report_per_cycle_channel` field (now including `dual_ratio_balance`)
populated and finite; `gap_proxy_G`/`gap_proxy_G_over_Q` are `None` with the
stated reason (unchanged, as designed). `recourse = gross_operational_cost
= 968,494,925.09` -- **identical** to the v1 preflight's value (see residual
diff below).

| channel | r | s | s_rho_part | eps_pri | eps_dual | primal_ratio | dual_ratio | dual_ratio_balance | action | rho before -> after |
|---|---|---|---|---|---|---|---|---|---|---|
| V | 1.011663e-01 | 5.302716e-02 | 3.749586e-02 | 3.228219e-03 | 2.976884e-04 | 31.338 | 178.130 | **125.957** | held | 1.0 -> 1.0 |
| PF | 3.640737e+00 | 7.127406e+00 | 5.039837e+00 | 1.432453e-03 | 9.215441e-04 | 2541.610 | 7734.200 | **5468.905** | held | 1.0 -> 1.0 |
| ESS | 5.287717e-03 | 5.875285e-03 | 1.840369e-03 | 7.300121e-04 | 7.205288e-04 | 7.243 | 8.154 | **2.554** | held | 1.0 -> 1.0 |

**Planner's prediction: HOLDS exactly.** rho held on V, PF and ESS at cycle
1 (`planner_prediction_rho_held_v_pf_ess.holds=True`,
`per_channel={'v': True, 'pf': True, 'ess': True}`). The observed
`dual_ratio_balance` values (V 125.957, PF 5468.905, ESS 2.554) match the
Planner's stated figures (125.96, 5468.9, 2.55) to the precision given; V's
decrease threshold `5*31.338=156.69` and PF's `3*2541.610=7624.83` both
exceed the respective `dual_ratio_balance`, so neither channel crosses the
decrease test; ESS's `dual_ratio_balance=2.554` is inside the band both
ways (`primal_ratio=7.243` does not exceed `5*2.554=12.77`;
`dual_ratio_balance=2.554` does not exceed `5*7.243=36.2`) -- matches
"inside the band both ways" exactly.

**Cycle-1 residuals identical to the v1 preflight.** `v1_vs_v2_residuals_identical=True`
-- every one of `r, s, s_rho_part, s_proximal_part, eps_pri, eps_dual,
norm_x, norm_z, norm_y, primal_ratio, dual_ratio` matches the v1 preflight's
committed `data/SRP1/Results/P515S32/preflight/preflight_verification.json`
(read only, never re-run) to within `1e-9` absolute difference on every
channel -- confirms balancing acts only at the END of the cycle and does
not perturb the cycle's own residual computation, as expected (v1's PF
`s`=7.127406e+00 vs v2's 7.127406e+00, etc. -- exact match to the digits
shown).

`boyd_all_pass=False`, `stopped_by='cap'` (1/1 cycles, one-cycle smoke
test, as expected).

## Validation

- Code executes correctly: confirmed (zero-solve checks + one real-solve
  preflight cycle, both produced the expected JSON output; preflight's own
  internal exit condition -- `all_fields_ok and balancing_all_ok and
  inner_counts_exact` -- was `True` on all three, matching a clean process
  exit).
- Requested diagnostic works: confirmed -- every `report_per_cycle_channel`
  field (including the new `dual_ratio_balance`) is populated on a real
  cycle; `boyd_terminal.json` records the v2 spec path/hash;
  `assert_s32_capture_paths` (now checking the `dual_ratio_balance` keys
  too) passes on a fresh, never-solved planning object.
- Underlying numerical problem (ADMM convergence under spec v2) is **not**
  assessed here -- that requires the full 150-cycle campaign, which this
  task explicitly does not launch, and which is out of scope for a
  one-cycle preflight regardless.
- `git diff` hunk inspection (10 hunks total, all inside the three
  permitted production sites) plus the zero-solve checks' independent
  source-text/byte-identity assertions confirm no unintended modification.
- The exact gate command (`python -u p515_g_g1_g4_admm_gates.py s32 >
  data/SRP1/Results/P515S32_launch.log 2>&1`) was **not** executed as a
  literal shell command (that would launch the real 150-cycle campaign);
  its pre-solve checklist (spec-v2 hash match, `N.REL==S32_REL`,
  `assert_s32_capture_paths`) was instead replicated standalone under a
  zero-permitted `SolveProfileGuard`, per the task's explicit instruction,
  and passed (`checklist all True: True`, `solve_guard counts: {all zero}`).

## Unexpected findings

- The v1 zero-solve check script (`p515_s32_zero_solve_checks.py`), if
  literally re-run now, would fail: its `_fake_boyd_metrics` helper does
  not set `dual_ratio_balance`, and the new `_update_admm_penalties` now
  requires `boyd_metrics[group]['dual_ratio_balance']` unconditionally
  (`KeyError` on lookup). This is expected and not a defect -- the task
  explicitly forbids re-running that script onto its committed directory,
  and its committed `zero_solve_checks.json` remains valid, frozen evidence
  for the v1 production code as it existed when that script ran. Flagged
  for completeness, not treated as something to fix.
- No other production or harness hunks were introduced outside the
  explicitly permitted sites; nothing else needed the "Unexpected findings"
  treatment.

## Remaining issues

- The s32 gate campaign itself (150 cycles, under spec v2) has **not** been
  run; only the Planner launches it. The exact command
  (`python -u p515_g_g1_g4_admm_gates.py s32 > data/SRP1/Results/P515S32_launch.log 2>&1`)
  is wired and its pre-solve checklist verified to pass (spec-v2 hash,
  `N.REL`, capture paths) without running the campaign.
- `data/SRP1/Results/P515S32/preflight_v2/preflight_v2_verification.json`'s
  `v1_vs_v2_cycle1_residual_diff` and `v1_vs_v2_residuals_identical` fields
  are only meaningful given the v1 preflight's committed values, which this
  task read but did not re-generate (per the "never re-run onto a committed
  artifact" rule) -- if the v1 preflight is ever regenerated under a
  different candidate/config, this comparison would need to be re-derived
  from the new file, not silently reused.

## Questions for Planner

- None blocking. The commit `1917924f` noted above (a new,
  unauthorized-by-this-task file `p515_s32_evaluate.py`) landed on this
  branch during this session from outside this worker; it does not conflict
  with anything in this task, but the Planner may want to confirm who
  authored it and whether it was an intended concurrent task.

## Commit hashes

Recorded after the two commits below are created (see final reply).
