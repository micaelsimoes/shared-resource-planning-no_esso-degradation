# Worker Report -- P5.15 Addendum 20, W1: s38 PF-pace arms preparation

## Task received

Prepare (never launch) the four s38 arms (`s38_A_tau0`, `s38_A_tau0p25_fallback`,
`s38_B_pfbal`, `s38_C_combined`) authorized by `PLANNER_BRIEF_2026-09-13.md`
Addendum 20, per binding specification `data/SRP1/Results/P515S38/
frozen_s38_pf_pace_spec_v9_7a2b4ab7.json` (frozen spec v9, sha256
`7a2b4ab7a98ae0145d647a977aa1d2c63a88b0e169a4c6089cdf96543e409dbd`):

1. Production: relax `admm_parameters.py`'s TSO `tau` validation (`tau <= 0 -> error`
   -> `tau < 0 -> error`), gated by a repo-wide grep for any division by tau/gamma.
2. Harness: four arms in `p515_g_g1_g4_admm_gates.py`, per-entry PF capture, the
   Addendum 20 freeze policy override (10-unchanged + absolute freeze at 200).
3. A new evaluator, `p515_s38_evaluate.py`.
4. Zero-solve checks, `p515_s38_zero_solve_checks.py`.
5. A new preflight, `p515_s38_preflight.py`, run for real (2 cycles each, arms A and B).

The Planner launches the 300-cycle arms; this task explicitly forbids doing so
(`s38_C_combined` additionally refuses to launch structurally, per spec v9's own
`order` field).

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` Addendum 20 (lines 828-857) and its predecessor
  Addendum 19 (lines 798-826), read for the mechanism reading (HP1+HP2) this stage
  tests.
- `data/SRP1/Results/P515S38/frozen_s38_pf_pace_spec_v9_7a2b4ab7.json` (binding
  spec v9, read in full).
- `admm_parameters.py` (full; tau validation at the then-line ~321-324).
- `shared_resources_planning.py`: `get_admm_boyd_residual_metrics` (5894-6154, the
  PF block at 6038-6062 reproduced by the new capture hook), `_init_admm_freeze_state`
  (6543-6571), `_update_admm_penalties` (6581-6884, the freeze precedence chain and
  the `tied_to_rho` gamma recompute), `update_transmission_model_to_admm` (4237-4388,
  the gamma-Param initialization and the proximal objective terms),
  `create_transmission_network_model` (3446-3553, the ONE `.optimize()` call a
  zero-solve probe TSO build must stub).
- `p515_g_g1_g4_admm_gates.py` (full read of the s37 section -- `S37_ARMS`,
  `assert_s37_capture_paths`, `run_s37_arm`, `_s37_configure_hook`,
  `s35ref_capture_hooks`/`s34_capture_hooks`, `run_admm_arm`, `_construct_arm_planning`,
  `assert_s31c_capture_paths`, `_acquire_exclusive_run_lock`, `_require_fresh_output_root`,
  `_refuse_overwrite`).
- `p515_s37_evaluate.py`, `p515_s37_preflight.py`, `p515_s37_zero_solve_checks.py`
  (templates mirrored, per task instruction).
- `p515_s32_zero_solve_checks.py` (`_build_admm_ready_state`, `_reset_rho`,
  `_dummy_residual_metrics`, `_stub_optimize` -- the zero-solve construction pattern
  `_s38_build_probe_tso_model` and the zero-solve check script both reuse).
- `p59_rho.py` (`apply_rho_to_params`).
- `WORKER_REPORT_S37_PREP.md` (the two bugs it recorded -- the pre-solve-hook/
  monkeypatch call-order collision and the `rho_ess` float-averaging tolerance --
  both informed this task's design and its own bugfix, see "Unexpected findings").
- `data/SRP1/SRP1_params.json` (`admm.rho`, `admm.penalty_update`,
  `admm.proximal_regularization` -- confirmed case-file defaults: rho v/pf/ess
  0.0077/0.198/0.1125; `freeze_after_unchanged_cycles=10`, `freeze_backstop_cycle=60`;
  `proximal_regularization.tso.tau=1.0`, `gamma_policy=tied_to_rho`).

## Files modified / created

**Commit 1** (`c315d39d`):
- `admm_parameters.py` (modified)
- `p515_s38_zero_solve_checks.py` (new)
- `data/SRP1/Results/P515S38/zero_solve_checks/zero_solve_checks.json` (new)
- `data/SRP1/Results/P515S38/zero_solve_checks/manifest_sha256.json` (new)

**Commit 2** (`11332cca`):
- `p515_g_g1_g4_admm_gates.py` (modified -- new s38 section + four dispatch branches)
- `p515_s38_evaluate.py` (new)
- `p515_s38_preflight.py` (new)

**Commit 3** (`1ef32245`, bugfix found by actually running the preflights):
- `p515_g_g1_g4_admm_gates.py` (modified -- eval-id collision fix in `run_s38_arm`)
- `p515_s38_preflight.py` (modified -- `math.isclose` gamma comparison)

**Commit 4** (this commit):
- `data/SRP1/Results/P515S38/preflight_A/` (new -- real 2-cycle run output, 27 files)
- `data/SRP1/Results/P515S38/preflight_A/manifest_sha256.json` (new)
- `data/SRP1/Results/P515S38/preflight_A_launch.log` (new)
- `data/SRP1/Results/P515S38/preflight_B/` (new -- real 2-cycle run output, 27 files)
- `data/SRP1/Results/P515S38/preflight_B/manifest_sha256.json` (new)
- `data/SRP1/Results/P515S38/preflight_B_launch.log` (new)
- `WORKER_REPORT_S38_PREP.md` (this file)

## Changes made

### 1. Production tau-validation relaxation (commit `c315d39d`)

**Grep results (before changing anything), the task's precondition:**

```
grep -rn "\btau\b" -- non-p5* production files: only admm_parameters.py:321-324
(the validation itself) and shared_resources_planning.py:2851 (a comment).

grep -rn "/ *tau\|/tau\b|/ *gamma\|/gamma\b|log(.*gamma|log(.*tau" \
  shared_resources_planning.py network.py shared_energy_storage_data.py \
  admm_parameters.py model_construction_helpers.py
-> NO HITS (zero division-by-tau/gamma or log-of-tau/gamma anywhere in production).

grep -n "prox_gamma" shared_resources_planning.py network.py \
  shared_energy_storage_data.py model_construction_helpers.py
-> every hit is a MULTIPLICATION: `(model[year][day].prox_gamma_v / 2) * proximal_v ** 2`
   (shared_resources_planning.py:4352, and the pf_p/pf_q/ess_p/ess_q siblings at
   4358-4359, 4383-4384) or a read (`pe.value(...)`, `.set_value(...)`). The ONE
   division found anywhere near gamma is `proximal_share = s_proximal_part / s` in
   `get_admm_boyd_residual_metrics` (shared_resources_planning.py:6122) -- division
   by the Boyd residual norm `s`, NOT by gamma or tau, and already guarded
   `if s > 0.0 else 0.0`.
```

No hit anywhere divides by tau or gamma. Cleared to change.

**Diff** (`admm_parameters.py`, `_read_parameters_from_file`, was lines 321-324):

```python
-            tau = float(agent_data.get('tau', admm_params.proximal_regularization['tso']['tau']))
-            if tau <= 0.0:
-                raise ValueError('ADMM proximal_regularization.tso.tau must be positive.')
-            admm_params.proximal_regularization['tso']['tau'] = tau
+            tau = float(agent_data.get('tau', admm_params.proximal_regularization['tso']['tau']))
+            # P5.15 Addendum 20 / frozen spec v9 ... [comment, see file]
+            if tau < 0.0:
+                raise ValueError('ADMM proximal_regularization.tso.tau must be non-negative.')
+            admm_params.proximal_regularization['tso']['tau'] = tau
```

Now (final line numbers) `admm_parameters.py:335-336`.

### 2. `p515_s38_zero_solve_checks.py` (commit `c315d39d`)

Four checks, `SolveProfileGuard(permitted=(), ...)` armed for the whole script:

- **(i) tau validation**: `ADMMParameters().read_parameters_from_file` with
  `tau=0.0` (loads, stored `0.0`), `tau=-0.1` (raises, message contains
  "non-negative"), `tau=0.25` (loads, stored `0.25`).
- **(ii) built-TSO-model gamma Params + proximal term**: a FRESHLY BUILT (zero-solve)
  TSO ADMM model per arm's tau, via a new harness helper
  `p515_g_g1_g4_admm_gates._s38_build_probe_tso_model` (stubs
  `transmission_network.optimize` -- the ONE solve call inside
  `create_transmission_network_model`, line ~3529 -- and overrides rho v/pf/ess to
  the spec v9 base BEFORE the build, since the case file's own rho_ess is 0.1125, not
  0.01). Under arm A's tau=0.0: `prox_gamma_v/pf/ess` == exactly `0.0` on every
  channel. Under arm B's tau=1.0: `prox_gamma_v/pf/ess` == that model's own
  `rho_v/pf/ess` exactly. The proximal objective TERM
  (`(gamma_c/2)*(z-z_prev)**2`), evaluated on the ACTUAL built model with arbitrary
  NONZERO Var/Param values set on both sides of the difference (z=7.3 vs
  z_prev=5.1 for V; similarly nonzero, unequal pairs for pf_p/pf_q/ess_p/ess_q), is
  EXACTLY `0.0` under A on all five legs.
- **(iii) freeze-policy replay** (reuses `p515_s37_zero_solve_checks._run_trajectory`/
  `_synthetic_boyd_sequence`/`_reset_gamma_tied` UNMODIFIED, `freeze_backstop_cycle`
  overridden to 200): no channel frozen at cycle 60 with an "always acting"
  increase/decrease pattern; every non-exempt channel freezes at cycle 200
  (`reason='backstop'`, action label `'held (frozen backstop cycle 200)'`), and NOT
  before; a channel that never acts (`ever_acted=False` throughout) NEVER freezes via
  `'streak'`, only via the cycle-200 backstop; a channel that acts once then holds
  freezes via `'streak'` at cycle 11 (act at cycle 1, held cycles 2-11 -- the streak
  counter reaches 10 at the END of cycle 11's OWN `_update_admm_penalties` call, so
  `frozen=True` is already recorded in the row FOR cycle 11, not cycle 12 -- see
  "Unexpected findings" #1); a PF-exempt channel under arm A's own list
  (`['ess','pf']`) never changes rho or gamma across 210 synthetic cycles, even under
  a strongly imbalanced ratio pattern (`primal_ratio=1e6, dual_ratio_balance=1e-6`)
  that would otherwise force `'increased'` every cycle, and survives the cycle-200
  backstop (`action` stays `'exempt (fixed)'`).
- **(iv) PF capture reconstruction identity**, exercised FOR REAL (not deferred to
  the preflight): a zero-solve `p515_s32_zero_solve_checks._build_admm_ready_state`
  model set, a MULTI-ENTRY (216 randomized inputs across 2 nodes x 5 years x days x
  2 power types x 3 periods) synthetic PF perturbation, the new
  `p515_g_g1_g4_admm_gates.s38_pf_capture_hooks` context manager wrapped around one
  real call to `srp.get_admm_boyd_residual_metrics`: reconstructed `r`/`s` reproduce
  production's own `result['pf']['r']`/`['s']` to relative error `0.0`/`0.0` (well
  under the 1e-9 bound), the wrapped function is confirmed restored on exit, and the
  full 1728-entry capture (every node/year/day/power_type/period, not just the 216
  perturbed inputs) is confirmed non-empty.

All four `pass: true`; `all_checks_pass: true`; guard
`{'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}`,
`verify_failures: []`.
Output: `data/SRP1/Results/P515S38/zero_solve_checks/zero_solve_checks.json`
(sha256 `3fbb7e10d02a1476fcb77c2bedef3f7914ba7c389d4cf25f23c2e71e2e97c962`).

### 3. Harness: `p515_g_g1_g4_admm_gates.py` (commits `11332cca`, `1ef32245`)

New section (approx. lines 4990-5605 after the s37 section) plus four dispatch
branches (approx. lines 6235-6272):

- `S38_SPEC_PATH`/`S38_SPEC_SHA256`/`S38_ARMS` (line ~4994): all four arms'
  configuration (`tau`, `exempt_channels`, `out_dir`, `eval_id`, `preflight_eval_id`,
  `launch_condition`, `order`).
- `_s38_stub_tso_optimize`/`_s38_build_probe_tso_model` (line 5068): zero-solve
  construction of a REAL TSO ADMM model under a given tau, used by the rule-eleven
  checklist to assert tau in force on the BUILT model's own `prox_gamma_*` Params,
  not merely on the params dict.
- `s38_pf_capture_hooks` (line 5120, `@contextmanager`, layered on top of
  `s35ref_capture_hooks`): the THIRD monkeypatch of `get_admm_boyd_residual_metrics`.
  Per cycle, for every PF entry (node, year, day, power_type, period): x_DSO,
  z_TSO current/prev, lambda_DSO, interface_rating, s_base_dso, rho_pf, gamma_pf in
  force, and the per-entry r/s contributions reproduced exactly as production's PF
  block (shared_resources_planning.py 6038-6062) computes them. ASSERTS every cycle
  (raises `RuntimeError` before writing the sidecar line on failure) that the
  per-entry sums reproduce that cycle's own `result['pf']['r']`/`['s']` to relative
  1e-9. Sidecar: `pf_entry_stride_<label>.jsonl`, write-once (`_refuse_overwrite`).
- `assert_s38_capture_paths(planning, arm_key)` (line 5230): rule-eleven checklist,
  built on `assert_s31c_capture_paths` DIRECTLY (not `assert_s37_capture_paths`,
  whose hard-asserted `freeze_backstop_cycle==60` and single arm-independent
  `balancing_exempt_channels==['ess']` would raise here) -- asserts every spec v9
  configuration value (rho v/pf/ess fixed at the spec v9 base, the ARM's OWN
  exemption list and tau, `freeze_after_unchanged_cycles=10`,
  `freeze_backstop_cycle=200`, standalone init, tied-gamma, boyd eps, sigma, S_ref,
  AL scale, cap=300) and every capture path, PLUS the Addendum 20 addition: tau in
  force checked BOTH on `planning.params.admm` AND, independently, on a freshly
  built (zero-solve) TSO model's own `prox_gamma_v/pf/ess` Params via
  `_s38_build_probe_tso_model` -- 0 for A/C, 0.25*rho for the fallback, rho for B.
- `_assert_s38_overrides_in_force`/`_s38_configure_hook` (lines 5427, 5464): the
  `pre_solve_hook` that deep-copies `planning.params` and applies standalone init,
  rho v/pf/ess, the arm's `balancing_exempt_channels`, the Addendum 20 freeze policy
  override (`freeze_after_unchanged_cycles=10`, `freeze_backstop_cycle=200`,
  OVERRIDING the case file's 60), and the arm's tau -- mirroring s37's
  lightweight-hook-check pattern for the same call-order reason
  (`s38_pf_capture_hooks`'s `with` block has already monkeypatched
  `get_admm_boyd_residual_metrics` by the time this hook runs).
- `run_s38_arm(arm_key, num_max_iters_override=None, output_root_override=None, _confirm_c_authorized=False)`
  (line 5495): shared implementation for all four arms. `s38_C_combined` is refused
  UNCONDITIONALLY unless `_confirm_c_authorized=True` (never passed by the CLI
  dispatch), per spec v9's own `order` field ("3 (NOT authorized to launch by this
  spec)"). Throwaway precheck planning object (full checklist, before any solve) ->
  `s38_pf_capture_hooks` -> `run_admm_arm` with `apply_rho=False`,
  `full_diagnostics_in_rows=True`, `post_run_hook=_s38_hook` (writes the PF-stride
  sidecar path into the report, then calls `write_boyd_terminal_s35ref` unchanged).
  Returns `(report, report_path)`.
- Dispatch: `elif gate == 's38_a_tau0': run_s38_arm('s38_A_tau0')` (and the analogous
  three branches for the fallback, B, and C -- C's branch calls `run_s38_arm`
  WITHOUT the confirm flag, so invoking it raises cleanly, citing spec v9).

**Bugfix commit `1ef32245`** (found by actually running the preflights, not by the
zero-solve checklist -- see "Unexpected findings" #2 and #3 below): `run_s38_arm` now
derives a THIRD, distinct eval id (`preflight_eval_id + '_run'`) automatically for
the REAL `run_admm_arm` call whenever `output_root_override` is given, instead of
always reusing `arm_cfg['eval_id']`. The real (non-preflight) launch path is
unaffected.

**Exact launch commands** (unchanged form from s37, task's own convention):

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_g_g1_g4_admm_gates.py s38_a_tau0 \
  > data/SRP1/Results/P515S38_A_TAU0_launch.log 2>&1

/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_g_g1_g4_admm_gates.py s38_a_tau0p25_fallback \
  > data/SRP1/Results/P515S38_A_TAU0P25_launch.log 2>&1
  # only if s38_A_tau0's evaluation (p515_s38_evaluate.py) reports TSO_INSTABILITY_TRIGGER.TRIGGERED == true

/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_g_g1_g4_admm_gates.py s38_b_pfbal \
  > data/SRP1/Results/P515S38_B_PFBAL_launch.log 2>&1
  # only after arm A (and its fallback, if triggered) has exited; never concurrently

# s38_c_combined is NOT authorized to launch by spec v9 -- invoking it raises a
# RuntimeError citing the spec's own "order" field; there is no launch command for it.
```

Precondition-check form used for every launch (mirrors s37; the harness's OWN
`_require_fresh_output_root`/`_acquire_exclusive_run_lock`/preflight-eval-id-exists
checks are the enforcement, this is the Planner-facing summary): before each launch,
confirm (a) `data/SRP1/Results/P515S38_<ARM>_run/` does not exist (or the fallback's
`P515S38_A_TAU0P25_run/`), (b) `.p515_g_gate.lock` does not exist, (c) no
`p515_g_g1_g4_admm_gates.py` process is running (`ps aux | grep p515_g_g1_g4`),
(d) `data/SRP1/Results/P56A/evals/p515s38_<arm>_arm` and
`p515s38_<arm>_preflight_capture_check` do not already exist (both are checked, and
the run is refused, by `run_s38_arm` itself -- restated here as the Planner-facing
precondition, per s37's own convention).

### 4. Evaluator: `p515_s38_evaluate.py`

Zero solves, `SolveProfileGuard(permitted=())` armed. `main([RUN_DIR] [--dry-run])`;
`RUN_DIR` defaults to `s38_A_tau0`'s output root; identifies the arm by exact path
match against the four canonical output roots. Computes, per arm:

- **CERTIFICATION** (spec v9 `certified_iff`, all four, each attributed): Boyd stop
  on all three channels within 300, 3 consecutive, derived from the trajectory
  (`p515_s37_evaluate._stop_from_trajectory`, reused with `CAP=300`); no V or PF
  channel EVER frozen at a rho clamp (ESS reported, not gated, exactly as spec v9's
  own wording, which names PF explicitly even though PF is ALSO exempt in A -- an
  exempt channel's `at_clamp` is production-guaranteed False, so this is expected to
  pass trivially for PF in A/fallback, and is reported as such, not silently
  assumed); storage terminal ratio < 0.9; cost within the rule-nine bar of run 1
  (read by path from `data/SRP1/Results/P515S35_REF_run/g_baseline.json`, sha256
  recorded).
- **FIRST_PASS_CYCLES** (PF, V, ESS): first cycle where `boyd_<channel>_channel_pass`
  is True.
- **HELPS**: PF first-pass <= 180.
- **TSO_INSTABILITY_TRIGGER** (arm A only, spec v9's four criteria, each computed
  independently): (i) any unrecovered TSO/case9 network failure (filtered by
  `agent=='TSO'` in `network_failures_*.jsonl`); (ii) TSO failure rate > 2x run 1's
  (0.1216/cycle, bound 0.2432/cycle); (iii) PF oscillation flag (ALL p/q entries,
  `p515_s37_evaluate._step_series`/`_oscillation_series`/`_oscillation_flag` reused
  unmodified on the PF z-vector series this module builds from the new
  `pf_entry_stride` sidecar); (iv) ESS oscillation flag (v8/s37 definition, P-only,
  same reused functions).
- **PF_DECOMPOSITION**: at matched cycles (1,2,5,...,300, capped at the arm's own
  cycle count), share of total sum(s^2)/sum(r^2) by node_id/year/day/power_type/
  period, plus the top-10 entries by s^2; a late-phase per-group decay rate (defined
  in the module docstring BEFORE computing it, per the CLAUDE.md evidence rule:
  geometric mean of the ratio group_s(cycle_{i+1})/group_s(cycle_i) over consecutive
  matched-cycle pairs inside [100,200], falling back to the latest available pair if
  fewer than two matched cycles fall inside that window -- the fallback is recorded
  explicitly, `window_used_note`).
- **PREDICTIONS**: both spec v9 predictions, scored once the relevant arm(s) exist
  (using the fallback in place of A if A's own trigger fired), PENDING otherwise.
- **cross_arm.ADOPTION**: spec v9's adoption rule (larger-rho-style logic replaced
  by: both help -> combined run authorized; exactly one helps -> candidate oracle;
  neither -> stop for review), PENDING until both an A-class run and B exist.

### 5. Preflight: `p515_s38_preflight.py`

`python p515_s38_preflight.py {A|B}`. Acquires
`p515_g_g1_g4_admm_gates._acquire_exclusive_run_lock()` FIRST (refusing if
`.p515_g_gate.lock` exists or is uncreatable -- the harness's own mechanism, which
also structurally prevents any concurrent `p515_g_g1_g4_admm_gates.py` process,
since every gate/preflight acquires the SAME lock file). Runs
`run_s38_arm(<arm_key>, num_max_iters_override=2, output_root_override=<preflight dir>)`
for REAL. Verifies: bounded solve-profile identity (base `51*3=153` + recovery-retry
overhead); tau/gamma in force per cycle (`gamma_after == tau*rho_after`,
`math.isclose` after the bugfix); PF/ESS action labels ('exempt (fixed)' when
exempt, live labels otherwise); the PF capture identity holds every cycle (re-read
from the sidecar, independent of the run's own in-flight assertion); every capture
path populated (10 paths, including the NEW `pf_entry_stride` sidecar); the
evaluator's dry-run exit code.

## Commands / experiments run

1. `admm_parameters.py` sanity (ad hoc, interactive, no files written): default load,
   case-file load, `tau=0`/`tau=-0.1`/`tau=0.25` load/reject.
2. Zero-solve smoke tests of `_s38_build_probe_tso_model` (three tau values, guard
   armed at 0 solves) -- found and fixed the rho-override bug (see "Unexpected
   findings" #4) before formalizing into the zero-solve check script.
3. Zero-solve smoke test of `assert_s38_capture_paths` on all four arms (guard armed
   at 0 solves) -- `missing: []` for all four.
4. Zero-solve smoke test of `run_s38_arm('s38_C_combined')` -- confirmed it raises
   without `_confirm_c_authorized`.
5. Zero-solve smoke test of `s38_pf_capture_hooks`'s identity on a randomized
   multi-entry perturbation, on `p515_s32_zero_solve_checks._build_admm_ready_state`
   -- found the missing `@contextmanager` decorator (see "Unexpected findings" #5)
   before it could be exercised at all; fixed, then confirmed rel_err ~1e-16.
6. `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s38_zero_solve_checks.py`
   -- four runs: the first two failed on check-(iii)'s streak-freeze-cycle
   off-by-one and check-(iv)'s spurious entry-count equality (both test-script bugs,
   fixed in the SCRIPT, not production; see "Unexpected findings" #1/#6); the third
   run passed all four checks but left stray eval-log directories under
   `data/SRP1/Results/P56A/evals/`; the fourth (final) run, after redirecting
   check-(iv)'s sidecar paths to a genuine `tempfile.mkdtemp()` dir instead of a
   named subdirectory of the committed `OUT_DIR`, passed cleanly with only
   `zero_solve_checks.json` written under the committed directory.
7. Synthetic end-to-end smoke tests of `p515_s38_evaluate.main` (built in the session
   scratchpad, never under `data/`): single-arm PENDING-adoption path, two-arm
   SCORED/PENDING-adoption path (A synthetically triggered by construction, so
   predictions correctly fall back to B alone), PF decomposition and late-phase
   decay-rate output structure -- all logic paths exercised correctly.
8. `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s38_preflight.py A`
   -- two real runs: the first (before the eval-id/gamma-tolerance bugfix) reported
   `overall_ok=True` for A (A's own gamma checks happened to pass with exact equality
   at tau=0, since 0.0 averages to exactly 0.0) but had ALREADY consumed the real
   launch's `eval_id` (see "Unexpected findings" #2) -- discarded (not committed,
   not cited by anything) and re-run after the fix; the second run is the one
   committed.
9. `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s38_preflight.py B`
   -- two real runs: the first FAILED on the gamma exact-equality bug (see
   "Unexpected findings" #3, tau=1.0 makes the averaging noise visible) -- discarded
   (not committed, not cited by anything, per the same reasoning as #8) and re-run
   after the fix; the second run is the one committed. A third, transient
   `RuntimeError` (preflight eval dir already existed, from the first B attempt's own
   precheck object) was hit and resolved by a full clean sweep of every s38-prefixed
   eval-log directory before the final, clean pair of A/B runs.
10. sha256 manifests generated for both committed preflight directories (Python,
    `hashlib.sha256`, walked every file including the `esso_capture`/`results`
    subdirectories -- 26 hashed files + the manifest itself = 27 per arm).

## Results

### Zero-solve checks (`p515_s38_zero_solve_checks.py`, final run)

| Check | Result |
|---|---|
| (i) tau=0 loads, tau=-0.1 rejected, tau=0.25 loads | PASS |
| (ii) built-model gamma Params (A: all 0; B: == rho) + proximal term (A: exactly 0 on all 5 legs) | PASS |
| (iii) freeze-policy replay (no freeze@60; freeze@200 non-exempt; streak only after prior action; PF-exempt never changes) | PASS |
| (iv) PF capture identity, multi-entry, zero-solve | PASS |

`all_checks_pass: true`; guard `{'permitted_solve': 0, 'permitted_exec': 0,
'blocked_solve': 0, 'blocked_exec': 0}`, `verify_failures: []`.
Output: `data/SRP1/Results/P515S38/zero_solve_checks/zero_solve_checks.json`
(sha256 `3fbb7e10d02a1476fcb77c2bedef3f7914ba7c389d4cf25f23c2e71e2e97c962`).

### Rule-eleven checklist contents (both arms' dry runs, `assert_s38_capture_paths`)

Every boolean key evaluated `True` for all four arms (`s38_A_tau0`,
`s38_A_tau0p25_fallback`, `s38_B_pfbal`, `s38_C_combined`); representative keys and
their values for `s38_A_tau0` (tau=0.0):

| key | value |
|---|---|
| `s38_spec_file_hash_matches` | `True` |
| `rho_v_pf_ess_matches_spec_v9_base` | `True` (v=0.0077, pf=0.198, ess=0.01 on every network + esso) |
| `balancing_exempt_channels_matches_arm` | `True` (`['ess', 'pf']`) |
| `freeze_after_unchanged_cycles_is_10` | `True` |
| `freeze_backstop_cycle_is_200` | `True` |
| `shared_ess_initialization_is_standalone` | `True` |
| `gamma_policy_is_tied_to_rho` | `True` |
| `tau_in_force_matches_arm` | `True` (`tau_value_expected: 0.0`) |
| `prox_gamma_v_pf_ess_in_built_tso_model_matches_expected` | `True` (`prox_gamma_expected: {'v': 0.0, 'pf': 0.0, 'ess': 0.0}`) |
| `s38_pf_capture_hooks_callable` / `pf_capture_writes_*_in_source` / `pf_capture_asserts_identity_raises_on_failure` | all `True` |
| `soh_floor_rows_identified_pre_solve` | `True` (`soh_floor_row_counts_by_node: {5: 6, 7: 6, 9: 6}`) |
| `cap_is_300` | `True` |

For `s38_B_pfbal` (tau=1.0), the ONE structurally different values are
`balancing_exempt_channels_matches_arm` checked against `['ess']`,
`tau_value_expected: 1.0`, and `prox_gamma_expected: {'v': 0.0077, 'pf': 0.198,
'ess': 0.01}` (== that arm's own rho) -- both `True`.

Full raw checklist dicts are in `data/SRP1/Results/P515S38/preflight_A_launch.log`
(line 2) and `.../preflight_B_launch.log` (line 2).

### Preflight A (`s38_A_tau0`, 2 real cycles, final run)

| Item | Value |
|---|---|
| Solves permitted (exact) | base `51*(2+1)=153` + retry overhead `0` = **153** |
| Solves observed | **153** (`exact: true`) |
| tau in force | `0.0` (every cycle) |
| gamma_v/pf/ess (cycle 1, then cycle 2) | `0.0, 0.0, 0.0` both cycles (`gamma_ok: true` every channel, every cycle) |
| rho_v (cycle 1 -> 2) | `0.011550 -> 0.017325` (live balancing, `action: increased` both cycles) |
| rho_pf (cycle 1, 2) | `0.198000` constant (`action: 'exempt (fixed)'` both cycles) |
| rho_ess (cycle 1, 2) | `0.010000` constant (`action: 'exempt (fixed)'` both cycles) |
| PF capture identity, cycle 1 / cycle 2 | `rel_err_r = 2.67e-16 / 0.0`, `rel_err_s = 0.0 / 0.0` (both `identity_holds: true`, 1728 entries/cycle) |
| Capture paths populated (10/10) | recourse-jump, ess-entry-stride, SoH-floor, PF-entry-stride sidecars; `boyd_terminal.json`; `g_s38_A_tau0.json`; interface voltage/settlement terminal; component levels terminal; ESSO models pickle -- all present, non-empty |
| Evaluator dry-run | return code `0` (certification correctly `False` at 2 cycles, `TSO_INSTABILITY_TRIGGER.TRIGGERED: false`) |
| Wall time | 94.4 s |
| Local solve failures / network failures | 0 / 0 |

`overall_ok: true`. Output root: `data/SRP1/Results/P515S38/preflight_A/` (27
files); manifest `data/SRP1/Results/P515S38/preflight_A/manifest_sha256.json`.

### Preflight B (`s38_B_pfbal`, 2 real cycles, final run)

| Item | Value |
|---|---|
| Solves permitted (exact) | base `51*(2+1)=153` + retry overhead `0` = **153** |
| Solves observed | **153** (`exact: true`) |
| tau in force | `1.0` (every cycle) |
| gamma_v (cycle 1, 2) | `0.011550 (== rho_v)` both cycles, `gamma_ok: true` |
| gamma_pf (cycle 1, 2) | `0.198000 (== rho_pf)` both cycles, `gamma_ok: true` |
| gamma_ess (cycle 1, 2) | `0.010000 (== rho_ess)` both cycles, `gamma_ok: true` |
| rho_v action (cycle 1, 2) | `increased`, `held` (live balancing) |
| rho_pf action (cycle 1, 2) | `held`, `held` (live balancing; PF NOT exempt in B) |
| rho_ess action (cycle 1, 2) | `'exempt (fixed)'`, `'exempt (fixed)'` |
| PF capture identity, cycle 1 / cycle 2 | `rel_err_r = 0.0 / 0.0`, `rel_err_s = 0.0 / 1.56e-16` (both `identity_holds: true`, 1728 entries/cycle) |
| Capture paths populated (10/10) | all present, non-empty (same list as A) |
| Evaluator dry-run | return code `0` (certification correctly `False` at 2 cycles) |
| Wall time | 92.1 s |
| Local solve failures / network failures | 0 / 0 |

`overall_ok: true`. Output root: `data/SRP1/Results/P515S38/preflight_B/` (27
files); manifest `data/SRP1/Results/P515S38/preflight_B/manifest_sha256.json`.

## Validation

- Code executes correctly: confirmed (module imports cleanly; `ast.parse` on both
  the harness and every new script; both preflights ran to completion through
  production's actual IPOPT path with 153/153 solves and zero local-solve failures).
- Tests pass: confirmed (all four zero-solve checks; both preflights' six mechanism
  gates each; both arms' dry checklists for all four arm configurations).
- The requested diagnostics (rule-eleven checklist including the built-model gamma
  assertion, solve-profile identity, PF capture identity, capture-path population,
  evaluator dry-run) work: confirmed, on REAL production data (the two 2-cycle
  preflights), not only on synthetic fixtures.
- The underlying numerical question (does tau=0 with rho_pf fixed, or PF balancing
  live with the new freeze policy, actually shorten the PF-pace bottleneck over 300
  cycles) is NOT answered by this task and was not attempted -- that is exactly what
  the Planner's authorized 300-cycle launches will determine. Both preflights' own
  EFC/step/ratio numbers are FAR from settled (as expected at 2 of 300 cycles) and
  are reported as mechanism evidence only, never as a preview of the real result.

## Unexpected findings

1. **Streak-freeze-cycle off-by-one in my OWN zero-solve check (script bug, not
   production)**: my first draft of check (iii) expected the streak-triggered freeze
   to land at cycle 12 (one act at cycle 1, then 10 held cycles). The observed
   (correct) production behaviour freezes at cycle 11: the
   `unchanged_streak >= freeze_after_unchanged_cycles` test runs INSIDE the SAME
   `_update_admm_penalties` call that just incremented the streak to 10, so
   `frozen=True` is already recorded in the row for cycle 11 itself. Fixed the
   check's expectation (and documented the reasoning inline), not production.
2. **Preflight-A/B eval-id collision (harness bug I introduced, then fixed,
   `p515_g_g1_g4_admm_gates.py`)**: my first `run_s38_arm` implementation passed
   `eval_id=arm_cfg['eval_id']` to `run_admm_arm` UNCONDITIONALLY -- the exact defect
   the task instructions named explicitly ("the s37 arm-1 launch was refused because
   the preflight and the arm shared a working directory -- do not repeat this"). The
   first preflight-A run (discarded, never committed) consumed
   `data/SRP1/Results/P56A/evals/p515s38_a_tau0_arm` -- the SAME id the real
   300-cycle launch needs. Caught by inspection (checking `ls .../evals/ | grep s38`
   before considering the task complete, not by any automated check), fixed by
   deriving a distinct THIRD id automatically inside `run_s38_arm` whenever
   `output_root_override` is given, and both preflights were re-run clean after a
   full sweep of the stale eval directories.
3. **Gamma float-averaging noise (production behaviour, correctly understood, not a
   bug -- the SAME class of defect s37's own preflight hit for `rho_ess`)**: my
   preflight's gamma-in-force check compared `gamma_after` to `tau * rho_after` with
   exact equality; `gamma_after` is `_get_admm_gamma_summary`'s average over 12 TSO
   year/day models, which is not bit-identical to a directly computed
   `tau * rho_after` at the ~1e-17 relative level even when every underlying Param is
   individually exact. This was invisible at `tau=0.0` (arm A: `0.0 * anything ==
   0.0` exactly, no rounding) and only surfaced at `tau=1.0` (arm B). Fixed with
   `math.isclose(rel_tol=1e-9)`, mirroring s37's own documented fix for the same
   class of issue.
4. **Probe TSO model rho mismatch (test-development bug, caught before it reached any
   committed check)**: my first version of `_s38_build_probe_tso_model` overrode only
   `tau`, not rho -- the probe TSO model was therefore built against the CASE FILE's
   `rho_ess=0.1125`, not the spec v9 base `0.01`, so the checklist's
   `expected_gamma['ess'] = tau * 0.01` comparison would have spuriously failed for
   every nonzero tau. Caught by an ad hoc smoke test (step 2 above) before the
   function was ever used inside `assert_s38_capture_paths`; fixed by adding
   `RH.apply_rho_to_params(...)` to the probe builder, with the reasoning documented
   in the function's own docstring.
5. **Missing `@contextmanager` decorator (my own bug, caught immediately)**: my first
   draft of `s38_pf_capture_hooks` omitted `@contextmanager`, so `with
   G.s38_pf_capture_hooks(...)` raised `TypeError: 'generator' object does not
   support the context manager protocol` on the very first smoke test. Fixed before
   any further testing.
6. **Spurious entry-count equality in my OWN zero-solve check (iv) (script bug, not
   production)**: my first draft asserted
   `len(line['entries']) == n_perturbed_inputs`; the sidecar correctly captures EVERY
   PF entry the model has (1728, every period x power_type x node/year/day), not just
   the 216 randomly perturbed ones -- the non-perturbed entries contribute their
   build-default values, which is expected and correct. Fixed the assertion to
   `>=`, with the reasoning documented inline; the genuine pass/fail criterion
   (the r/s identity) was unaffected by this bug.

## Remaining issues

- The four 300-cycle arms have not been run (by design -- the Planner launches A,
  the fallback if triggered, and B; `s38_C_combined` additionally cannot be launched
  by this code until the Planner explicitly re-authorizes it, per spec v9).
- Once A (and, if triggered, its fallback) and B are launched, `p515_s38_evaluate.py`
  should be re-run (write mode, not `--dry-run`) on each arm's own output root to
  produce the certification/helps/trigger/adoption verdicts this task prepares but
  cannot produce without the real 300-cycle data.
- The zero-solve balancing-rule replay item (a) in spec v9's `sequencing` (whether
  rho_pf would have been lowered on the saved run-1/s37 trajectories, and when) is a
  SEPARATE authorized item, not part of this Worker task's scope, and was not
  attempted here.
- `s38_C_combined`'s own preflight was NOT run (the task specifies preflights for A
  and B only); its checklist was verified zero-solve (dry, all four arms) but its
  real-solve mechanics are therefore verified only by structural similarity to A
  (same tau=0.0) and B (same `balancing_exempt_channels=['ess']`), not independently.

## Questions for Planner

None -- the task's instructions were unambiguous, and the frozen spec v9
configuration matched the prompt's summary exactly. One judgment call I made, stated
for the record rather than as a question: `assert_s38_capture_paths`'s per-arm
`prox_gamma_*_in_built_tso_model_matches_expected` check builds a NEW probe TSO model
per checklist invocation (its own fresh `O.fresh_planning` call, own eval id) rather
than reusing the throwaway `preflight_planning` object the rest of the checklist
already built -- this costs one extra (zero-solve) planning construction per
checklist call but keeps the TSO-model-build code path (which needs a real
`transmission_network.optimize` stub) fully isolated from the rest of the checklist,
which never touches model construction. I judged this the safer choice given the
call-order fragility s37's own worker report already flagged once (see
`WORKER_REPORT_S37_PREP.md` "Unexpected findings" #2); happy to simplify if the
Planner prefers a single shared probe object.

## Deviations from spec v9

None identified in the FINAL committed state. Two deviations existed transiently
during development (the eval-id collision and the gamma-tolerance bug, both in this
Worker's OWN harness/preflight code, not in the frozen spec's configuration values)
and were caught and fixed before anything was committed to reference them -- see
"Unexpected findings" #2/#3. Every `configuration_common` field, `capture_requirements`
item, and `certification_per_arm` criterion in the frozen spec is implemented as
specified in the final, committed code.

## Commit hashes

1. `c315d39d` -- production tau relaxation + zero-solve checks script + outputs.
2. `11332cca` -- harness s38 arms, PF capture, evaluator, preflight script.
3. `1ef32245` -- eval-id collision fix + gamma-tolerance fix (found by running the
   preflights).
4. (this commit) -- preflight A/B outputs with sha256 manifests + this report.
