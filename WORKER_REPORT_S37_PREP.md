# Worker Report -- P5.15 Addendum 19, s37 preparation (the rho_ess experiment)

## Task received

Prepare (never launch) the two ρ_ess arms (`s37_rho0p01`, `s37_rho0p001`) authorized by
`PLANNER_BRIEF_2026-09-13.md` Addendum 19, per binding specification
`data/SRP1/Results/P515S37/frozen_s37_rho_ess_spec_v8_f91de983.json` (frozen spec v8, sha256
`f91de9836d279a08767b7d84a5ca7e37cdc7e7a3fa05a240be6e713910714e72`):

1. Production: an optional per-channel ADMM balancing exemption (`admm.penalty_update.
   balancing_exempt_channels`, default `[]`), plus zero-solve checks.
2. Harness: both arms in `p515_g_g1_g4_admm_gates.py`, sharing one implementation.
3. A new evaluator, `p515_s37_evaluate.py`.
4. A new preflight, `p515_s37_preflight.py`, run for real (2 cycles, `s37_rho0p01` only).

The Planner launches the two 150-cycle arms; this task explicitly forbids doing so.

## Files inspected

- `data/SRP1/Results/P515S37/frozen_s37_rho_ess_spec_v8_f91de983.json` (binding spec)
- `PLANNER_BRIEF_2026-09-13.md` Addendum 19 (lines 798-832)
- `admm_parameters.py` (full)
- `shared_resources_planning.py`: `_init_admm_freeze_state`, `_update_admm_penalties`
  (~6537-6875), the per-cycle `admm_diagnostics` dict (~2748-2900)
- `p515_g_g1_g4_admm_gates.py` (full read of the s31c/s32/s33e2/s34/s35ref/s35pt/
  s35ref_replay sections, `run_admm_arm`, `_construct_arm_planning`)
- `p59_rho.py` (`apply_rho_to_params`, `set_adaptive_penalty`)
- `p515_s35ref_evaluate.py`, `p515_s35pt_evaluate.py` (evaluator style/pattern)
- `p515_s34_zero_solve_checks.py`, `p515_s32_zero_solve_checks.py`, `p515_s32v2_zero_solve_checks.py`
  (zero-solve check pattern, reused helpers)
- `p515_s35ref_preflight.py` (preflight pattern, reused structurally)
- `data/SRP1/SRP1_params.json` (`admm.rho`, `admm.penalty_update`, `admm.
  shared_ess_initialization` -- confirmed case-file defaults: rho v/pf/ess =
  0.0077/0.198/0.1125 including an `esso` key in `rho.ess`; `shared_ess_initialization =
  "price_taker"`; no `balancing_exempt_channels` key)

## Files modified / created

**Commit 1** (`dee8735f`):
- `admm_parameters.py` (modified)
- `shared_resources_planning.py` (modified)
- `p515_s37_zero_solve_checks.py` (new)
- `data/SRP1/Results/P515S37/zero_solve_checks/zero_solve_checks.json` (new)
- `data/SRP1/Results/P515S37/zero_solve_checks/manifest_sha256.json` (new)

**Commit 2** (this commit):
- `p515_g_g1_g4_admm_gates.py` (modified -- new s37 section + two dispatch branches)
- `p515_s37_evaluate.py` (new)
- `p515_s37_preflight.py` (new)
- `data/SRP1/Results/P515S37/preflight_rho0p01/` (new -- real 2-cycle run output, 25 files)
- `data/SRP1/Results/P515S37/preflight_rho0p01/manifest_sha256.json` (new)
- `WORKER_REPORT_S37_PREP.md` (this file)

## Changes made

### 1. Production (commit `dee8735f`)

`admm_parameters.py`:
- `ADMMParameters.__init__` (line ~57): `penalty_update['balancing_exempt_channels'] = []`
  (default), plus `self.balancing_exempt_channels_source = 'default'`.
- `_read_parameters_from_file` (line ~198-208): excludes `balancing_exempt_channels` from
  the float-cast loop (it is a list, not a coefficient).
- `_read_parameters_from_file` (line ~244-264): validates and assigns the key -- must be a
  list drawn from `{'v', 'pf', 'ess'}`; raises `ValueError` otherwise; records
  `balancing_exempt_channels_source` as `'case_file'`/`'default'`.

`shared_resources_planning.py`:
- `_init_admm_freeze_state` (line ~6537): adds `'exempt': False` to the per-channel state
  dict.
- `_update_admm_penalties` (line ~6568): channels in `balancing_exempt_channels` are marked
  `exempt=True`/`frozen=True`/`reason='exempt'`/`at_clamp=False` on the FIRST call that sees
  them (sticky); the action-precedence chain now checks `group_state['exempt']` FIRST (ahead
  of the legacy/backstop/streak freeze branch), recording `'exempt (fixed)'` and leaving
  `factor=1.0` -- the existing group-wise scaling loop (unmodified) then applies a
  mathematically exact no-op (`value * 1.0`, then clamp of an already-in-bounds value) to
  that channel's ρ (and, under `gamma_policy=='tied_to_rho'`, γ recomputes to the SAME
  `tau*rho` every cycle since ρ itself never changes) -- verified bit-identical, not merely
  argued, by zero-solve check (b) below.
- Per-cycle `admm_diagnostics` dict (line ~2877-2885): three new keys,
  `balancing_exempt_v`/`_pf`/`_ess` (bool).

Non-exempt channels and every case study/arm that does not set the key are provably
unaffected (zero-solve check (a)).

### 2. `p515_s37_zero_solve_checks.py` (commit `dee8735f`)

Six checks, `SolveProfileGuard(permitted=(), ...)` armed for the whole script:
- (a) absent key -> current code bit-identical to pre-change code (commit `7323b2fb`, this
  worker's own starting HEAD) on a 20-cycle synthetic Boyd-metric replay exercising
  increase/decrease/hold on every channel.
- (b) `balancing_exempt_channels=['ess']`, 60 synthetic cycles, strongly imbalanced ESS
  ratios (primal_ratio=1e6, dual_ratio_balance=1e-6 -- would force `'increased'` every cycle
  absent the exemption, confirmed via a parallel no-exemption control run on a SEPARATE
  model set): ESS ρ/γ never change; action always `'exempt (fixed)'`; frozen from cycle 1;
  never at clamp, including across SRP1's own case-file global backstop at cycle 60 (the
  exempt run's ESS channel stays `'exempt (fixed)'` at cycle 60 while the CONTROL run's
  non-exempt ESS is legitimately frozen by the backstop at that same cycle -- both asserted,
  confirming the synthetic scenario is genuinely provocative and the exemption survives the
  backstop specifically); V/PF bit-identical between the exempt and control runs.
- (c) invalid channel names rejected (`ValueError`, bogus entries named in the message);
  non-list values rejected; valid single/all-channel lists accepted.
- (d) five other case studies (CS1, CS7, HR1, OP1, OP2) plus SRP1 itself load with
  `balancing_exempt_channels == []`, source `'default'`.
- (e) the `S31C_FIXTURES` set (including `P512R/cycle21_pre_setup/snapshot.pkl`) plus the two
  `FrozenSMOPF` comparator pickles all unpickle.
- (f) `_run_operational_planning_hierarchical`/`_run_operational_planning_without_
  coordination` source-text-identical to commit `7323b2fb`; diff hunk headers recorded.

All six `pass: true`; `all_checks_pass: true`; guard `verify_failures: []`.

### 3. Harness: `p515_g_g1_g4_admm_gates.py` (commit 2)

New section (lines 4536-4887) plus two dispatch branches (lines 5517-5538):

- `S37_SPEC_PATH`/`S37_SPEC_SHA256`/`S37_ARMS` (line 4569): both arms' configuration
  (`rho_ess`, `out_dir`, `eval_id`, `preflight_eval_id`, `launch_condition`).
- `assert_s37_capture_paths(planning, rho_ess_value)` (line 4610): rule-eleven checklist,
  built on `assert_s31c_capture_paths` DIRECTLY (not `assert_s35ref_capture_paths`, whose own
  `initial_rho_v_pf_ess_matches_spec_v5` hard-asserts ρ_ess==0.1125 and would raise here) --
  asserts every spec v8 configuration value (ρ_v=0.0077, ρ_pf=0.198, ρ_ess=arm value on every
  network AND the esso, `balancing_exempt_channels==['ess']`, standalone init, tied-gamma,
  freeze policy 10/60, boyd eps, sigma, S_ref, AL scale) and every capture path (per-entry
  ESS x/z at stride 1, EFC/day per node, ESS action label, SoH floor identification
  pre-solve), raising if any check fails.
- `_assert_s37_overrides_in_force` (line 4776) / `_s37_configure_hook` (line 4814): the
  `pre_solve_hook` that deep-copies `planning.params`, applies the standalone init override,
  ρ_v/ρ_pf/ρ_ess via `p59_rho.apply_rho_to_params`, and `balancing_exempt_channels=['ess']`.
  See "Unexpected findings" for why the hook runs a LIGHTWEIGHT override check rather than
  re-running the full `assert_s37_capture_paths`.
- `run_s37_arm(arm_key, num_max_iters_override=None, output_root_override=None)` (line 4843):
  shared implementation for both arms (only `rho_ess` and the output root differ); throwaway
  precheck planning object (full checklist, before any solve) -> `s35ref_capture_hooks`
  (per-entry ESS x/z stride 1, EFC/day/node, SoH-floor sidecar, recourse-jump sidecar) ->
  `run_admm_arm` with `apply_rho=False`, `full_diagnostics_in_rows=True`,
  `post_run_hook=write_boyd_terminal_s35ref` (run 1's own terminal writer, reused verbatim).
  Returns `(report, report_path)`.
- Dispatch: `elif gate == 's37_rho0p01': run_s37_arm('s37_rho0p01')` /
  `elif gate == 's37_rho0p001': run_s37_arm('s37_rho0p001')`.

Exact commands (unchanged from the task):
```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_g_g1_g4_admm_gates.py s37_rho0p01 > data/SRP1/Results/P515S37_RHO0P01_launch.log 2>&1
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_g_g1_g4_admm_gates.py s37_rho0p001 > data/SRP1/Results/P515S37_RHO0P001_launch.log 2>&1
```

### 4. Evaluator: `p515_s37_evaluate.py`

Zero solves, `SolveProfileGuard(permitted=())` armed. `main([RUN_DIR] [--dry-run])`; `RUN_DIR`
defaults to the `s37_rho0p01` output root; identifies the arm by exact path match against the
two canonical output roots (raises otherwise). Computes, per arm:
- **CERTIFICATION** (spec v8 `certified_iff`, all four): trajectory-derived Boyd stop within
  150 (`_stop_from_trajectory`, the v2 method s35ref/s35pt's own evaluators established --
  NEVER `boyd_terminal.stopped_by`); no V/PF clamp (ESS clamp reported, not gated, per the
  spec's own wording); storage terminal ratio `max(boyd_ess_primal_ratio,
  boyd_ess_dual_ratio) < 0.9`; cost within the rule-nine bar of run 1, run 1's terminal
  cost/step read from its committed `g_baseline.json` BY PATH with sha256 recorded.
- **PREDICTIONS**: per-entry RMS storage-consensus (P only) step over cycles 2-20
  (`_load_ess_stride_p_vectors`/`_rms_over_window`), ratio vs run 1 (predicted ~11x); the
  0.001-vs-0.01 ratio once both arms exist (predicted ~10x); EFC/day >= 1.06 by cycle 50
  (0.01 arm only); "at least one arm certifies".
- **Cross-arm ADOPTION**: PENDING with only one arm present (reports which); once both exist,
  larger ρ_ess if both certify, the certifying one if exactly one does, else "stop for
  review".
- **Monitoring**: local-solve/network failures by tier; per-cycle ESS sign-change fraction
  and consecutive-step cosine (exact spec v8 definition: step_c = z_c - z_(c-1), entries with
  `|step_c| < 1e-9` MW excluded from the fraction; cosine unfiltered) with the OSCILLATION
  FLAG (>0.2 sign-change fraction for 10 consecutive cycles, or cosine < 0 for 5); EFC/day,
  step-size, ρ/γ and all channel-ratio trajectories, sampled at matched cycles.
- **LEAD**: arm, ρ_ess, certification verdict, cycle count, EFC trajectory (first fields in
  the output).

### 5. Preflight: `p515_s37_preflight.py`

Runs `run_s37_arm('s37_rho0p01', num_max_iters_override=2, output_root_override=<NEW dir>)`
for REAL (genuine IPOPT solves; the task explicitly authorizes this one 2-cycle smoke test).
Verifies the bounded solve-profile identity (base `51*(2+1)=153` + recovery-retry overhead,
tier-1 +1/tier-2 +2, read from THIS run's own `network_failures_summary.classes`), ρ_ess
constancy/label/freeze/clamp, V/PF normal balancing, every capture path populated, and runs
`p515_s37_evaluate.main(['--dry-run'])` in-process against the preflight's own output
(`ARM_ROOTS['s37_rho0p01']` temporarily repointed at the preflight directory, restored in a
`finally`).

## Commands / experiments run

1. `admm_parameters.py`/`shared_resources_planning.py` sanity (`ADMMParameters()` default,
   case-file load, invalid-channel `ValueError`, `_init_admm_freeze_state()` shape) -- ad hoc,
   interactive, no files written.
2. `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s37_zero_solve_checks.py`
   -- twice (first run found the check-(a)/(b) test-harness gamma-reset bug below and the
   backstop-collision sanity-check bug below; both fixed in the SCRIPT, not production; the
   third run passed all six checks).
3. Dry (zero-solve) `assert_s37_capture_paths` checklist for BOTH arms, applying the exact
   override recipe `_s37_configure_hook` applies, on throwaway `O.fresh_planning` objects --
   twice (once before, once after the preflight fix below) -- both times `missing: []` for
   both arms.
4. `p515_s37_evaluate.main` exercised on synthetic trajectory + stride-sidecar data (built in
   the session scratchpad, never under `data/`) to validate the certification/prediction/
   monitoring computations, including a targeted oscillation-flag unit check (forced
   sign-alternating synthetic vectors) -- all logic paths exercised correctly.
5. `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s37_preflight.py`
   -- three times: run 1 failed on the `pre_solve_hook`/monkeypatch collision below (raised
   before any solve, i.e. the guard/checklist mechanism itself caught it); run 2 (after that
   fix) completed all real solves but failed the `rho_ess constant` check on a floating-point
   test-harness artifact (below); run 3 (after that fix) passed with exit code 0.

## Results

### Zero-solve checks (`p515_s37_zero_solve_checks.py`, third/final run)

| Check | Result |
|---|---|
| (a) absent key bit-identical vs pre-change (`7323b2fb`), 20 cycles | PASS |
| (b) `['ess']` exemption, 60 cycles, strongly imbalanced ratios | PASS |
| (c) invalid channel names / non-list rejected | PASS |
| (d) other case studies default to `[]` | PASS |
| (e) preserved fixtures unpickle | PASS |
| (f) hierarchical/uncoordinated paths untouched | PASS |

`all_checks_pass: true`; guard `{'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve':
0, 'blocked_exec': 0}`, `verify_failures: []`.
Output: `data/SRP1/Results/P515S37/zero_solve_checks/zero_solve_checks.json` (sha256
`0bd6a6fd1626643a05f9417ae3d35d177eb416b2f00059bcc941ee80415d597f`).

### Preflight (`s37_rho0p01`, 2 real cycles, third/final run)

| Item | Value |
|---|---|
| Solves permitted (exact) | base `51*(2+1)=153` + retry overhead `0*1+0*2=0` = **153** |
| Solves observed | **153** (`identity_holds: true`) |
| ρ_ess (before/after, both cycles) | constant at `0.01` (float64 average-of-models noise
  `0.010000000000000004`, `math.isclose` rel_tol 1e-9 -- see "Unexpected findings") |
| ESS action label, both cycles | `'exempt (fixed)'` |
| ESS frozen every cycle | `true` |
| ESS at rho clamp | `false`, every cycle |
| V/PF balancing actions | cycle 1: v=`increased`, pf=`held`; cycle 2: v=`held`, pf=`held`
  (`balancing_exempt_{v,pf}` = `false` every cycle) |
| Capture paths populated (9/9) | recourse-jump, ess-entry-stride, SoH-floor sidecars;
  `boyd_terminal.json`; `g_s37_rho0p01.json`; interface voltage/settlement terminal;
  component levels terminal; ESSO models pickle -- all present, all non-empty |
| Evaluator dry-run on preflight output | return code `0` (certification correctly `False`
  at 2 cycles -- expected, not a failure; predictions/monitoring all computed without error) |
| Cycle 1 EFC/day max | `0.0002872608764361283` |
| Cycle 2 EFC/day max | `0.3291147647062786` |
| Cycle 1 step size | `None` (no preceding captured cycle within this 2-cycle preflight --
  expected) |
| Cycle 2 step size | RMS `0.09492027603181556`, max\|·\| `0.2338802703248879` |
| Cycle 1/2 channel ratios (primal, dual) | v: (76.29, 6.79) / (30.23, 10.88); pf: (2515.5,
  2780.5) / (1980.1, 1523.8); ess: (1114.7, 18.2) / (509.2, 13.8) |
| Cycle 1/2 sign-change fraction / cosine | `None` both cycles (needs 3 consecutive captured
  cycles for a step-vs-previous-step comparison; only 2 cycles exist here -- expected) |

Overall `overall_ok: true` (all seven mechanism gates pass); `sys.exit` not called.
Output root: `data/SRP1/Results/P515S37/preflight_rho0p01/` (25 files, 7.6 MB; manifest
`data/SRP1/Results/P515S37/preflight_rho0p01/manifest_sha256.json`).

### Both arms' dry checklists

`assert_s37_capture_paths` on a throwaway planning object with the EXACT override recipe
(`_s37_configure_hook`) applied: `s37_rho0p01` -> `missing: []`, `rho_ess_value_in_force:
0.01`; `s37_rho0p001` -> `missing: []`, `rho_ess_value_in_force: 0.001`. Both pass, zero
solves.

## Validation

- Code executes correctly: confirmed (module imports cleanly; `ast.parse` on the full
  harness file; the real 2-cycle preflight ran to completion through production's actual
  IPOPT path with 153/153 solves and zero local-solve failures).
- Tests pass: confirmed (all six zero-solve checks; all seven preflight mechanism gates;
  both arms' dry checklists).
- The requested diagnostic (rule-eleven checklist, solve-profile identity, capture-path
  population, evaluator dry-run) works: confirmed, on REAL production data (the 2-cycle
  preflight), not only on synthetic fixtures.
- The underlying numerical question (does ρ_ess=0.01/0.001 with ESS balancing exempted
  actually resolve the primal-walk-speed bottleneck over 150 cycles) is NOT answered by this
  task and was not attempted -- that is exactly what the Planner's authorized 150-cycle
  launches will determine. The 2-cycle preflight's own EFC/step/ratio numbers are FAR from
  settled (as expected at 2 of 150 cycles) and are reported as mechanism evidence only, never
  as a preview of the real result.

## Unexpected findings

1. **Zero-solve-check test-harness bug (script-only, not production)**: the first
   `p515_s37_zero_solve_checks.py` run failed check (a) (spurious `gamma_before` mismatch at
   cycle 1) and check (b) (spurious `ess_gamma_constant=False`, and a `plain_run_ess_always_
   increased` sanity check failure at cycle 60). Root cause: `p515_s32_zero_solve_checks.
   _reset_rho` resets ONLY the ρ Params, never `prox_gamma_*`; replaying two trajectories
   back-to-back on the SAME model set (or starting a fresh model set with ρ artificially
   reset to a test value) leaves γ at a stale, inconsistent value relative to the reset ρ.
   Fixed by adding `_reset_gamma_tied` (resets γ to match the reset ρ) and, for check (b),
   by testing the backstop interaction explicitly instead of asserting a "never
   backstop-frozen" sanity check that SRP1's own case-file `freeze_backstop_cycle=60`
   contradicts by design. Neither fix touched production code.
2. **Pre-solve-hook / monkeypatch call-order collision (harness bug I introduced and then
   fixed, `p515_g_g1_g4_admm_gates.py`)**: my first `_s37_configure_hook` re-ran the FULL
   `assert_s37_capture_paths` (including its `inspect.getsource(srp.get_admm_
   boyd_residual_metrics)` module-source checks) from INSIDE `run_s37_arm`'s `pre_solve_hook`
   -- but `pre_solve_hook` fires AFTER `s35ref_capture_hooks`'s `with` block has already
   monkeypatched that exact function, so `inspect.getsource` read the WRAPPER's source, not
   the original, and the `boyd_field_*_in_source` checks spuriously failed (raised, caught by
   the guard mechanism itself, before any solve -- the safety net worked exactly as intended,
   just against my own bug). Fixed by replacing the second call with a NEW, narrower
   `_assert_s37_overrides_in_force` that checks only the override-application facts (ρ
   values, exemption, standalone init) and does not inspect any monkeypatchable function's
   source; the FULL structural checklist still runs to completion, genuinely before any
   solve, via the throwaway precheck object in `run_s37_arm` (unaffected, since it runs
   BEFORE the `with s35ref_capture_hooks(...)` block is entered). I note for the Planner's
   awareness that `s35ref_replay`'s own `_s35ref_replay_force_standalone_hook` has the
   IDENTICAL structural risk (it also re-runs `assert_s35ref_capture_paths` from inside a
   `pre_solve_hook` wrapped by the same `s35ref_capture_hooks` context manager) -- it has
   never actually been exercised end-to-end (per its own docstring, "PREPARED, NOT
   LAUNCHED"), so this has not yet surfaced there, but WOULD surface identically if that arm
   were ever run. This is a scoped observation for the record, not a fix I made (out of
   scope for this task; that arm's code is explicitly "UNCHANGED" per this task's
   instructions).
3. **Floating-point averaging noise in `rho_ess_before`/`_after` (production behaviour,
   correctly understood, not a bug)**: `_get_admm_penalty_summary`'s own documented
   "averaging convention" over every TSO/DSO/ESSO model returns `0.010000000000000004`
   instead of exactly `0.01` even though every underlying Param is bit-identical (0.01 is not
   exactly representable in float64, and averaging several models' Params accumulates ~1e-17
   relative rounding). My preflight's exact-equality check was too strict; fixed with
   `math.isclose(v, 0.01, rel_tol=1e-9)`.

## Remaining issues

- The two 150-cycle arms have not been run (by design -- the Planner launches them).
- Once launched, `p515_s37_evaluate.py` should be re-run (write mode, not `--dry-run`) on
  each arm's own output root to produce the certification/adoption verdicts this task
  prepares but cannot produce without the real data.
- Item 2 under "Unexpected findings" (the `s35ref_replay` latent call-order risk) is flagged
  for the Planner's awareness, not fixed, per scope.

## Questions for Planner

None -- the task's instructions were unambiguous and the spec v8 configuration matched the
prompt's summary exactly (no discrepancy found between this prompt and the frozen spec file).

## Deviations from spec v8

None identified. Every `configuration_common` field, `capture_requirements` item, and
`certification_per_arm` criterion in the frozen spec is implemented as specified;
`predictions_recorded_in_advance` and `monitoring_reported_not_gated` are both implemented in
the evaluator using the spec's own exact definitions and thresholds.
