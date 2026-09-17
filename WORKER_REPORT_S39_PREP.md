# Worker Report — P5.15 Addendum 21, task W1 (s39 preparation)

## Task received

Prepare arms `s39_C`/`s39_D` (production two-phase ESS policy, harness arms, evaluator
under the new certification bar, zero-solve checks, a zero-solve replay of D's policy,
and two-cycle preflights). The 300-cycle arms are NOT launched by this task.

## Lead finding — the bitwise premise is falsified (report, do not rationalize)

**The frozen spec v10 preflight premise "A differs only in PF exemption and the
3-consecutive stop, neither of which can act in cycles 1-2" is FALSE.** PF balancing
genuinely fires at cycle 2 in both `s39_C` and `s39_D` (PF is live in both, unlike v9
arm A, which exempts PF): `rho_pf` decreases from 0.198 to 0.132 at cycle 2 in both
preflights, while v9 arm A holds `rho_pf` fixed at 0.198 (exempt). The BITWISE check
(item 8 of the preflight script) therefore correctly reports a **MISMATCH**, not a bug:

| cycle | field | s39_C / s39_D | v9 arm A | cause |
|---|---|---|---|---|
| 1 | `rho_frozen_pf` | `False` | `True` | A's static PF exemption sets `freeze_state['pf']['frozen']=True` immediately; C/D's live PF does not |
| 1 | `rho_unchanged_streak_pf` | `1` | `0` | downstream of the above |
| 2 | `rho_pf_after` | `0.132` | `0.198` | PF balancing genuinely **decreases** rho_pf at cycle 2 when PF is not exempt |
| 1 (D only) | `rho_frozen_ess` | `False` | `True` | D's NEW conditional ESS exemption (`balancing_exempt_until`) does not set `freeze_state['ess']['frozen']`; only the static mechanism (A, C) does |

Every other cycle-1/2 value (V channel, ESS `rho`/action, `gross_operational_cost`,
`recourse`, all `boyd_*_r/s`, `primal_*`, `dual_*`) is bitwise IDENTICAL between C, D
and v9 arm A — confirmed by the same comparator (192 common fields checked per cycle,
only the fields above differ). Both preflights otherwise pass every other verification
(solve-profile identity, tau/gamma=0, action labels, min-consecutive=10, PF capture
identity, all capture paths populated, evaluator dry-run, D's exempt-until sidecar).
This is reported exactly per the task's own instruction ("if they differ, report which
fields and stop, do not rationalize"); no code was changed to make the comparison pass,
and neither preflight's `overall_ok` was forced to True.

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` (Addendum 21, full history back through Addendum 1)
- `data/SRP1/Results/P515S39/frozen_s39_oracle_spec_v10_f1b2b999.json` (binding, read-only)
- `p515_g_g1_g4_admm_gates.py` (s38 section, `run_admm_arm`, `_construct_arm_planning`,
  `_acquire_exclusive_run_lock`, `_require_fresh_output_root`, `assert_s31c_capture_paths`)
- `p515_s38_evaluate.py`, `p515_s37_evaluate.py`, `p515_s38_preflight.py`,
  `p515_s38_zero_solve_checks.py`, `p515_s38_balancing_replay.py`, `p515_s37_zero_solve_checks.py`,
  `p515_s32_zero_solve_checks.py`, `p515_s32v2_zero_solve_checks.py`
- `admm_parameters.py` (full), `shared_resources_planning.py` (`_init_admm_freeze_state`,
  `_update_admm_penalties`, the ADMM row-builder at ~2748-2900 — read only, NOT edited,
  per the task's scope restriction to `_update_admm_penalties` only)
- `data/SRP1/SRP1_params.json` (`admm` block, read-only)
- Committed s38 arm trajectories: `data/SRP1/Results/P515S38_A_TAU0_run/g_s38_A_tau0.json`,
  `data/SRP1/Results/P515S38_B_PFBAL_run/g_s38_B_pfbal.json`
- `p515_s36_step36_timing_run.py` (READ ONLY, for the `ps aux` precondition-check
  pattern — never touched, never imported; my own precondition check is a fresh,
  self-contained implementation)

## Files modified / created

Production:
- `admm_parameters.py` (+68/-1 lines)
- `shared_resources_planning.py` (+123/-4 lines, `_init_admm_freeze_state` and
  `_update_admm_penalties` ONLY)

New scripts:
- `p515_s39_zero_solve_checks.py`
- `p515_s39_d_policy_replay.py`
- `p515_s39_evaluate.py`
- `p515_s39_preflight.py`

Harness (extended, s31c…s38 sections untouched):
- `p515_g_g1_g4_admm_gates.py` (+618 lines: s39 section — constants, `S39_ARMS`,
  `_s39_spec_hash`, `_s39_working_dir_ids`, `s39_exempt_until_capture_hooks`,
  `assert_s39_capture_paths`, `_assert_s39_overrides_in_force`, `_s39_configure_hook`,
  `run_s39_arm`, CLI branches `s39_c`/`s39_d`)

New evidence (all under `data/SRP1/Results/P515S39/`, write-once, never overwritten):
- `zero_solve_checks/zero_solve_checks.json` + `evidence_manifest_sha256.json`
- `d_policy_replay_on_s38/d_policy_replay_on_s38_A_tau0.json`,
  `d_policy_replay_on_s38_B_pfbal.json`, `replay_A_run.log`, `replay_B_run.log`,
  `evidence_manifest_sha256.json`
- `preflight_C/` (27 files incl. `preflight_verification.json`, `g_s39_C.json`,
  `pf_entry_stride_s39_C.jsonl`, `ess_exempt_until_state_s39_C.jsonl`,
  `evidence_manifest_sha256.json`), `preflight_C_launch.log`
- `preflight_D/` (27 files, same structure + `ess_exempt_until_state_s39_D.jsonl`),
  `preflight_D_launch.log`

## Changes made

### 1. Production: `admm_parameters.py`

- `ADMMParameters.__init__`: new `penalty_update['balancing_exempt_until']` key,
  default `{}`; new `self.balancing_exempt_until_source = 'default'`.
- `_read_parameters_from_file`: `'balancing_exempt_until'` added to
  `_non_float_penalty_update_keys`; new validation block (after the existing
  `balancing_exempt_channels` block) — must be a dict; keys drawn from
  `{'v','pf','ess'}`; each value must set `dual_ratio_below` (float > 0) and
  `consecutive_cycles` (int >= 1); a channel may not be in BOTH
  `balancing_exempt_channels` and `balancing_exempt_until` (raises `ValueError`
  naming the overlap). Absent/empty key → `{}`, source `'default'`.

### 2. Production: `shared_resources_planning.py`

- `_init_admm_freeze_state`: three new per-channel keys, all inert defaults —
  `exempt_until_streak` (0), `exempt_until_lifted` (False),
  `exempt_until_lift_cycle` (None).
- `_update_admm_penalties`: new block (Addendum 21 addition, cited in the function's
  own docstring) computing, per channel, whether it is configured in
  `balancing_exempt_until` and not yet lifted; if so, its Boyd `dual_ratio` (full
  s/eps_dual, the SAME field the Boyd stop test uses) is compared against the
  configured threshold every call; a streak of `consecutive_cycles` below-threshold
  calls lifts the exemption ONE-WAY on the call that completes it, in the SAME call
  falling through to the standard rule (so the lift cycle may genuinely balance);
  `action = 'exempt (fixed)'` (same label, same precedence rank as the existing
  unconditional exemption) while pending; the lift cycle's own action is suffixed
  `' (exemption lifted)'` when it is NOT `'increased'`/`'decreased'`; the
  unchanged-streak bookkeeping is skipped while pending (so it counts fresh from the
  lift); `at_clamp` is never set while pending. Absent/empty `balancing_exempt_until`
  is verified a byte-identical no-op (see zero-solve check (i) below — 0 mismatches
  on 5 already-committed trajectories, none of which ever set this key).

Design note (reported, not asked): I deliberately did **not** reuse the existing
`freeze_state[channel]['exempt']` field for the pending phase (kept three new,
dedicated fields instead), because that field is documented and consumed elsewhere as
*permanently* sticky (the unconditional exemption's own semantics), and D's
conditional exemption is one-way in the OPPOSITE direction (starts exempt, becomes
non-exempt). This is why `balancing_exempt_ess`/`rho_frozen_ess` read `False` for arm D
even while its ESS channel is genuinely exempt — see the lead finding table above.

Sanity-tested before writing any harness code (4-then-break-then-5 sequence on a real
zero-solve model set): lift correctly occurs on the cycle completing the 5th
CONSECUTIVE below-threshold call (cycle 10, not 9), the lift cycle's own action is
labelled `'held (exemption lifted)'` when the standard rule holds, and it is a plain
`'increased'`/`'decreased'` when the standard rule genuinely fires.

### 3. Harness: `p515_g_g1_g4_admm_gates.py` — s39 section

Mirrors the s38 section's structure (arms dict, spec-hash guard, capture-hooks
context manager, structural checklist, lightweight override check, configure hook,
`run_s39_arm`, CLI branches), with:

- `S39_ARMS['s39_C']`: `exempt_channels=['ess']`, `exempt_until={}`.
- `S39_ARMS['s39_D']`: `exempt_channels=[]`,
  `exempt_until={'ess': {'dual_ratio_below': 1.0, 'consecutive_cycles': 5}}`.
- Common: rho v/pf/ess = 0.0077/0.198/0.01 (fixed, unchanged from s38 base), tau=0.0
  (both arms, unlike s38 which varied tau per arm), `minimum_consecutive_converged_cycles
  = 10` (override, spec v10 — s38 used 3), freeze after 10 unchanged + absolute
  backstop 200 (unchanged from Addendum 20), cap 300.
- **New capture**: `s39_exempt_until_capture_hooks`, monkeypatching
  `srp._update_admm_penalties` (same technique `s38_pf_capture_hooks` uses on
  `srp.get_admm_boyd_residual_metrics`), writing
  `ess_exempt_until_state_<arm>.jsonl` per cycle from the REAL returned
  `freeze_state` — empty content for arm C (whose `balancing_exempt_until` is `{}`),
  populated for arm D.
- **Structural fix** (mandatory per task): `run_s39_arm` derives ALL THREE working-dir
  ids (pre-check `precheck_id_stub`, probe `p515s39_<arm>_tso_probe_checklist`, run
  `run_id_stub`) from a `mode` suffix (`'real'` when `output_root_override is None`,
  `'preflight'` otherwise) via the new `_s39_working_dir_ids(arm_key)` helper — so a
  preflight run and the later real launch can never collide, in either invocation
  order. Verified zero-solve in `assert_s39_capture_paths` (disjointness check) AND
  independently in `p515_s39_zero_solve_checks.py` check (iv), AND empirically after
  running both preflights (see Results below — all six `real`-mode ids absent, all
  six `preflight`-mode ids present).
- `s38_C_combined` and every s31c…s38 CLI branch/function: untouched (confirmed by
  `git diff --stat` showing only new lines added after the s38 section).

### 4. Evaluator: `p515_s39_evaluate.py`

Certification: all three channels' Boyd primal/dual ratios AND local-solve success,
10 CONSECUTIVE cycles, derived from the trajectory's own `cycle_convergence`/
`consecutive_converged_cycles` per-row fields (production's own bookkeeping,
`shared_resources_planning.py` ~2735-2741) — never from `boyd_terminal.json`'s
`stopped_by` field. Reported-not-gated: per-channel terminal ratios, rule-ten ratio,
clamp flags, cost-vs-run-1 under the new bar (`arm_max_step_last10 + 256.2581009864807`,
the latter read from the frozen spec v10 JSON by path with its own sha256 recorded, next
to `run1_cost = 651039166.0347285`). Predictions scored strictly PER ARM (C's two
predictions from C's own evaluation only; D's from D's own evaluation only) — `PENDING`
when that arm's evaluation is absent — this design makes the s38 evaluator's "SCORED
with an arm missing" defect structurally impossible to repeat, since neither arm's
prediction ever reads the other arm's dict. Adoption: `PENDING` unless both C and D
evaluations exist. Late-phase PF decay-rate formula **fixed**: the true per-cycle
geometric rate `(group_s(c1)/group_s(c0))**(1/(c1-c0))`, not s38's raw pairwise ratio
(wrong whenever matched cycles are unevenly spaced, which s39's own matched-cycle list
is: `...,125,131,140,145,150,...`). TSO-instability trigger, PF decomposition, PF
stride loader and `_rows`/oscillation/step-series helpers reused **by import**
(`p515_s37_evaluate`, `p515_s38_evaluate`) — confirmed fully generic (operate only on
`run_dir`/`rows`/`entries`, no s38-arm-specific globals) before reuse.

Not yet exercised against a real evaluation output (no 300-cycle run exists); exercised
successfully in `--dry-run` mode against both 2-cycle preflight outputs (see Results).

### 5. Zero-solve checks: `p515_s39_zero_solve_checks.py`

- **(i) default-off identity**: extends `p515_s38_balancing_replay.py`'s own
  methodology (reused model builders/metric adapters BY IMPORT) to FIVE already-
  completed trajectories — run 1 (s35ref), both s37 arms, AND (newly, not previously
  validated by any script) both s38 arms (A, B) — with each run's own config
  (exempt channels, freeze backstop/unchanged-cycles, tau) INFERRED from its own
  first recorded cycle, never hand-declared. **0 mismatches on all 5 runs.**
- **(ii) synthetic sequences**, all on a fresh zero-solve model set, replayed through
  the real `_update_admm_penalties` via `p515_s37_zero_solve_checks._run_trajectory`
  (reused unmodified): 4-then-break-then-5 lifts at exactly cycle 10; one-way (never
  re-exempted after forcing the ratio back up for 10 cycles); the lift cycle
  genuinely balances (forced `'increased'`) when the ratio test says so; the
  unchanged-streak freeze fires at lift_cycle+10 (=15), not earlier, and only counts
  from a post-lift action; the absolute backstop (200) freezes a channel that lifted
  early and keeps acting (isolated from `'streak'` by construction — alternating
  increase/decrease every cycle 6-199) but does NOT interrupt a channel still
  pending at cycle 250; `read_parameters_from_file` rejects every documented bad
  config (unknown channel, non-positive threshold, non-positive `consecutive_cycles`,
  overlap with `balancing_exempt_channels`).
- **(iii)** tau=0 and `minimum_consecutive_converged_cycles=10` load via
  `read_parameters_from_file` AND are in force on a freshly built, UNSOLVED
  `V1._build_admm_ready_state` planning object.
- **(iv)** the structural working-dir-id fix: for each arm, `real` ∩ `preflight` = ∅;
  across both arms and both modes, all twelve derived ids are pairwise distinct.

All four checks pass; `SolveProfileGuard(permitted=())` verified 0 solves for the
whole script.

### 6. D-policy replay: `p515_s39_d_policy_replay.py`

Extends `p515_s38_balancing_replay.py`'s counterfactual methodology (reused model
builders/metric adapters BY IMPORT) into a standalone, general-purpose script: takes
`<trajectory_path> <output_dir>` on the CLI, applies arm D's OWN configuration
(`balancing_exempt_channels=[]`, `balancing_exempt_until={'ess': {'dual_ratio_below':
1.0, 'consecutive_cycles': 5}}`, freeze policy, tau=0.0) via the real
`ADMMParameters.read_parameters_from_file`, and open-loop replays the REAL
`_update_admm_penalties` on the donor trajectory's OWN observed boyd ratios (never
re-solved), reporting the predicted ESS lift cycle, the first ESS balancing action
strictly after the lift (cycle, direction, rho before/after), and every subsequent
action labelled non-predictive. Output file name is derived from the trajectory's own
basename (`d_policy_replay_on_<label>.json`), so repeated calls into the SAME
`output_dir` with different trajectories never collide — ready for the Planner's later
call on arm C's own completed trajectory without any script change.

## Commands / experiments run

```
# production sanity test (ad hoc, scratch eval dir cleaned up afterward)
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -c "<inline lift-sequence test>"

# zero-solve checks
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s39_zero_solve_checks.py

# D-policy replay, both donor trajectories
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s39_d_policy_replay.py \
    data/SRP1/Results/P515S38_A_TAU0_run/g_s38_A_tau0.json \
    data/SRP1/Results/P515S39/d_policy_replay_on_s38 \
    > data/SRP1/Results/P515S39/d_policy_replay_on_s38/replay_A_run.log 2>&1

/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s39_d_policy_replay.py \
    data/SRP1/Results/P515S38_B_PFBAL_run/g_s38_B_pfbal.json \
    data/SRP1/Results/P515S39/d_policy_replay_on_s38 \
    > data/SRP1/Results/P515S39/d_policy_replay_on_s38/replay_B_run.log 2>&1

# preflights (exact launch commands, full precondition checks documented below)
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s39_preflight.py C \
    > data/SRP1/Results/P515S39/preflight_C_launch.log 2>&1

/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s39_preflight.py D \
    > data/SRP1/Results/P515S39/preflight_D_launch.log 2>&1
```

**Precondition checks performed before each preflight launch** (both, independently):
`.p515_g_gate.lock` absent (`ls` confirmed no such file); no forbidden process alive
(`p515_s39_preflight.py`'s own `_check_preconditions`, `ps aux` scan excluding this
process's own ancestor chain — see "Unexpected findings" for why the ancestor-exclusion
was needed); production files were git-clean at the time of each launch (`git status`
showed only the already-declared modifications to `admm_parameters.py`,
`shared_resources_planning.py`, `p515_g_g1_g4_admm_gates.py`, committed in the prior
step of this same task — no uncommitted DRIFT introduced between commits); output roots
`data/SRP1/Results/P515S39/preflight_C/` and `preflight_D/` absent before each run;
ALL SIX working-dir ids (both modes, both arms) checked absent immediately before the
first preflight, confirmed via `_s39_working_dir_ids` + `os.path.exists` against
`O.WORK_DIR` (see Results — before either preflight ran, `real` AND `preflight` ids
were both absent for both arms).

## Results

### Zero-solve checks

`all_checks_pass: True`. Per-check: `default_off_identity` (5/5 runs, 0 mismatches
each — run1_s35ref, s37_rho0p01, s37_rho0p001, s38_A_tau0, s38_B_pfbal, inferred
configs printed and match each run's own known configuration exactly);
`synthetic_sequences` (all 6 sub-checks pass); `tau0_and_min_consecutive_10` (pass);
`working_dir_ids_disjoint` (pass, 12/12 ids pairwise distinct).
`solve_profile_guard.verify_failures = []`.

### D-policy replay

| donor trajectory | predicted ESS lift cycle | first ESS action after lift |
|---|---|---|
| s38 arm A (`g_s38_A_tau0.json`, 133 cycles) | **35** | cycle 35, `increased`, rho 0.01 → 0.015 (the lift cycle itself) |
| s38 arm B (`g_s38_B_pfbal.json`, 151 cycles) | **35** | cycle 35, `increased`, rho 0.01 → 0.015 (the lift cycle itself) |

Both donors predict the SAME lift cycle (35) — expected, since neither donor's own
config differs from D in early-cycle ESS behaviour before their PF-specific paths
diverge. On BOTH open-loop replays, `rho_ess` is subsequently walked all the way to
the clamp (`1.0e4`) by the later cycles of the donor trajectory — reported honestly as
non-predictive open-loop drift (the real trajectory would differ once rho itself
diverges from the donor's observed path), per both scripts' own explicit caveat.
`SolveProfileGuard` verified 0 solves for both runs.

### Preflights — checklist / configuration in force (both arms, both cycles)

- Solve-profile identity: exact (153 = 51×3, 0 retries) for both C and D.
- `gamma_{v,pf,ess}_after == 0.0` every cycle, both arms — confirmed.
- V, PF: live labels (`increased`/`decreased`/`held`) every cycle, both arms —
  confirmed.
- ESS: `'exempt (fixed)'` every cycle, both arms — confirmed. D's own
  `ess_exempt_until_state_s39_D.jsonl` sidecar: cycle 2 `{'streak': 0, 'lifted':
  False, 'lift_cycle': None}` — not lifted, as required (2 cycles cannot complete a
  5-consecutive streak).
- `required_consecutive_cycles == 10` every cycle, both arms — confirmed.
- PF capture identity: holds every cycle, both arms (`rel_err_r`/`rel_err_s` ≤ 1e-9).
- All capture paths populated, both arms (recourse-jump, ess-entry-stride, SoH-floor,
  PF-entry-stride, ess-exempt-until-state sidecars; `boyd_terminal.json`;
  `g_s39_<C|D>.json`; interface voltage/settlement terminal; component levels; ESSO
  models pickle).
- Evaluator dry-run: return code 0, both arms (ran to completion, wrote a well-formed
  in-memory report, did not write `s39_evaluation.json` per `--dry-run`).

### Preflights — BITWISE vs v9 arm A: **FAILS**, exact fields identified (see Lead finding)

Both `preflight_C` and `preflight_D` report `overall_ok=False` **solely** because of
the bitwise-comparison mismatch above; every other verification passes. This is
reported as designed evidence, not a script defect.

### Working dirs created (post-preflight, verified)

```
data/SRP1/Results/P515S39/preflight_C/   (27 files + evidence_manifest_sha256.json)
data/SRP1/Results/P515S39/preflight_D/   (27 files + evidence_manifest_sha256.json)
data/SRP1/Results/P515S39/preflight_C_launch.log
data/SRP1/Results/P515S39/preflight_D_launch.log
```

Working-dir ids under `O.WORK_DIR` (`data/SRP1/Results/P56A/evals/`), checked AFTER
both preflights:

| arm | mode | precheck | probe | run | exists? |
|---|---|---|---|---|---|
| s39_C | real | `p515s39_c_precheck_real` | `p515s39_s39_C_tso_probe_checklist_real` | `p515s39_c_arm_real` | **absent** (all 3) |
| s39_C | preflight | `p515s39_c_precheck_preflight` | `p515s39_s39_C_tso_probe_checklist_preflight` | `p515s39_c_arm_preflight` | **exists** (all 3) |
| s39_D | real | `p515s39_d_precheck_real` | `p515s39_s39_D_tso_probe_checklist_real` | `p515s39_d_arm_real` | **absent** (all 3) |
| s39_D | preflight | `p515s39_d_precheck_preflight` | `p515s39_s39_D_tso_probe_checklist_preflight` | `p515s39_d_arm_preflight` | **exists** (all 3) |

Confirms the structural fix: none of the ids the REAL 300-cycle launch will use exist
after both preflights ran — the launch can proceed without the s38-style dir-collision
incident.

## Validation

- Production change: syntax-checked (`ast.parse`), sanity-tested on a real zero-solve
  model set BEFORE any harness code was written, and fully zero-solve-checked (4
  independent check families, all pass). Default-off byte-identity independently
  confirmed against 5 already-committed real trajectories (0 mismatches).
- Harness: imports cleanly; `assert_s39_capture_paths` executed standalone against a
  freshly built (unsolved) planning object under D's full configuration before ever
  wiring it into `run_s39_arm` — all checks passed, zero solves.
- Preflights: both ran end-to-end through the REAL `run_s39_arm`/`run_admm_arm`
  machinery (2 real ADMM cycles each, real IPOPT solves — this is NOT a zero-solve
  check), produced fully populated output directories, and their own
  `preflight_verification.json` records every individual check's pass/fail
  transparently (nothing was suppressed to force `overall_ok=True`).
- Evaluator: exercised successfully in `--dry-run` mode by both preflights (return
  code 0) — this is the only test possible before a real 300-cycle run exists.
- Every new eval working directory this Worker created as scratch/sanity-test residue
  (`sanity_check_s39_prod`, `p515s39_sanity_check_precheck_test`,
  `p515s39_s39_D_tso_probe_checklist_preflight` from the earlier ad hoc test) was
  removed before the real preflight runs, so it could not be mistaken for real-launch
  evidence. The zero-solve-check script's own eval dirs (`p515s39zsc_check_ii`,
  `p515s39zsc_check_iii`) were left in place, matching the precedent s38's own
  zero-solve-check script set (`p515s38zsc_check_iii`, never cleaned up) — inert
  network-log residue, not evidence.

## Unexpected findings

1. **The bitwise premise is falsified** — see Lead finding above. PF balancing DOES
   act within cycles 1-2 when PF is not exempt (unlike v9 arm A). This is a real
   mechanism difference, correctly detected, not a code defect.
2. **The `ps aux` precondition check needed an ancestor-pid exclusion, not just
   self-pid.** A naive `pid == os.getpid()` exclusion (the pattern used by the
   read-only-inspected `p515_s36_step36_timing_run.py`) produces a FALSE POSITIVE
   under this Bash tool: the shell process that launches
   `python p515_s39_preflight.py C` necessarily has the full command text (including
   this script's own filename) in its own `ps aux` line, and is a DIFFERENT pid than
   the python process itself. Fixed by walking the parent-pid chain
   (`_ancestor_pids`) and excluding every ancestor, not just `os.getpid()`. Verified:
   before the fix, a bare `ps aux | grep <these five filenames>` matched 2 unrelated
   lines (the grep process itself and its own launching shell) purely from the
   grep pattern's own text; after the fix, `_check_preconditions` correctly returned
   `[]` with no forbidden process detected. Reported as a design correction made
   during this task, not a Planner instruction.
3. **D's `rho_frozen_ess`/`balancing_exempt_ess` diagnostic fields read `False`**, even
   though D's ESS channel IS exempt at cycles 1-2 (action = `'exempt (fixed)'`) — a
   direct, intended consequence of using the new `exempt_until_*` fields rather than
   the pre-existing `exempt`/`frozen` fields (design note in Changes §2). Anyone
   reading `balancing_exempt_ess`/`rho_frozen_ess` alone from a D trajectory would
   need to also check `rho_ess_action` (or the new `ess_exempt_until_state_*.jsonl`
   sidecar) to see that ESS is in fact exempt during the pending phase.

## Remaining issues

- The evaluator (`p515_s39_evaluate.py`) has not been exercised against a real
  (non-2-cycle) evaluation with a genuine certification, since no 300-cycle run
  exists yet — only `--dry-run` against the preflight outputs.
- The Planner's own reading of the bitwise-comparison failure (whether the spec's
  premise needs correcting, or whether cycles 1-2 were never actually expected to be
  identical once the mechanism is examined closely) is left to the Planner, per scope
  discipline.
- `p515_s39_d_policy_replay.py` has not yet been run on arm C's own trajectory (not
  authorized by this task — "the Planner will later ask for the same on arm C's
  trajectory"); the script is ready to do so without modification.

## Questions for Planner

1. Given the bitwise-comparison failure, should the s39 preflight's `overall_ok`
   continue to gate on item 8 (bitwise match), or should item 8 be downgraded to
   "reported, not gated" now that its own premise is shown incorrect? I left both
   preflights reporting `overall_ok=False` rather than deciding this myself.
2. Is the design choice in §2 (new dedicated `exempt_until_*` state fields, rather
   than reusing `exempt`/`frozen`) the one you want, given its consequence for
   `balancing_exempt_ess`/`rho_frozen_ess` readability on arm D? An alternative
   would set `freeze_state['ess']['frozen'] = True` (but NOT `['exempt']`) for the
   pending phase, which would make `rho_frozen_ess` read `True` (matching A/C) while
   keeping `balancing_exempt_ess` (`['exempt']`) correctly `False` (since D's
   mechanism is genuinely different) — I did not make this change without
   authorization, since it touches the same production function this task already
   scoped tightly.

## Commit hashes

Recorded after each commit below (order: production+checks script+output →
replay script+output → harness+evaluator+preflight script → preflight evidence+report).
