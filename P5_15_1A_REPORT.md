# P5.15-1a — Warm-start policy production fix + recovery policy (Step 1a)

Worker report. Authorized by `PLANNER_BRIEF_2026-09-13.md`, Addendum 1, "Step 1 —
amendments / Step 1a". Context: `P5_15_0_REPORT.md` (Step 0 evidence) and
`data/SRP1/Results/P5150/p5150_report.json`. Reused Step 0's harness
`p515_0_esso_warmstart_ab.py` (its loading paths, `ESSO_PICKLE`, `esso_preflight`,
`esso_clobber_probe`, `parse_ipopt_log_tail`, `max_constraint_violation`) via import,
unmodified.

Repository provenance at run time: HEAD `a858fbf6037e56ce00fc0baeaa0fc7d4e4e02ac8`,
branch `feature/derivative-free-planning`. Interpreter
`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`. IPOPT
`/usr/local/bin/ipopt`, version `3.14.18 (aarch64-apple-darwin24.5.0), ASL(20241111)`.
Diagnostic/implementation stage; frozen-artifact ceremony not invoked (per the
brief, not required for Steps 0-2).

New harness: `p515_1a_gate_check.py` (only new file this task created, besides
the two production files and this report). Evidence:
`data/SRP1/Results/P5151A/p5151a_gate_report.json`,
`data/SRP1/Results/P5151A/logs/*.txt`.

Nothing under `data/` was edited. Nothing committed (commits are the Planner's
responsibility per instructions). `git add .` / `git add -A` not used at any point.

---

## 1. Files modified

- `shared_energy_storage_data.py` — `_create_solver`, `_is_recoverable_shared_ess_failure`,
  `_optimize`.
- `network.py` — `_create_smopf_solver`, `_is_recoverable_network_failure`, `_run_smopf`.

No other production file was touched. `_build_subproblem` was not touched (out of
scope for this task, reserved for the ESSO reformulation step). No tolerance, rho,
or calibration constant was changed. No `data/` file was edited.

---

## 2. Diff summary per file

### 2.1 `shared_energy_storage_data.py::_create_solver`

**Item 1 (clobbering fix).** Deleted the five unconditional post-merge assignments
(`solver.options['warm_start_bound_push'] = 1e-9`, etc., previously at lines
907-911, run *after* `option_overrides` had already been merged into `options` and
applied to `solver.options`). Nothing replaces them: if the case file
(`params.options`) or the caller's `option_overrides` set a `warm_start_*` key, it
is already on `solver.options` from the generic merge/apply loop that runs earlier
in the same function (`options.update(...)` then `for key, value in
options.items(): solver.options[key] = value`); if neither sets it, the key is
simply never assigned and IPOPT's own compiled default (1e-3) governs. This is the
"apply the warm-start block before the merge" resolution the brief offered as one
of two equivalent options — here realized by removing the post-merge overwrite
entirely rather than reordering it, since the new policy requires "unset unless
explicit" rather than "some other hardcoded default". `warm_start_init_point =
'yes'` and the two `replace_warm_start_suffix` calls are unchanged.

**Item 3 (`max_iter`).** Added `options['max_iter'] = 500` to the base options
dict, in the same `if params.solver.lower() == 'ipopt':` block that pins
`fixed_variable_treatment`, i.e. *before* `params.options` / `option_overrides`
are merged in — so a case file or caller can still override it (none currently
does).

### 2.2 `shared_energy_storage_data.py::_is_recoverable_shared_ess_failure`

**Item 4 (fire conditions).** Changed the single equality check
`result.solver.termination_condition == po.TerminationCondition.internalSolverError`
to membership in `(internalSolverError, maxIterations, infeasible)`. The
`recovery_options` truthiness guard and the `node_id`/`solver`-name guards are
unchanged.

### 2.3 `shared_energy_storage_data.py::_optimize`

**Item 4 (recovery = cold start, same primary options, exact Hessian, drop
`hessian_approximation`).**
- `recovery_options` is now built as `params.recovery_options` filtered to drop the
  `hessian_approximation` key, then `warm_start_init_point = 'no'` is added.
- The recovery `_run_solver_attempt` call now passes `from_warm_start=False`
  (previously `from_warm_start=from_warm_start`, i.e. it inherited the primary's
  warm-start flag — a bug relative to the intended "one change: cold start"
  policy, since a warm-started primary's failed recovery attempt was itself
  warm-started).
- Before the recovery call, `model.ipopt_zL_in.clear()`, `model.ipopt_zU_in.clear()`
  and `model.dual.clear()` are called (belt-and-braces; see §4 — not load-bearing
  for this model family, per the Planner's established P5.12-R `.nl` finding).
- The `diagnostic_sink` entry's `'recovery_options'` field now reports the
  effective (filtered + cold-start) dict actually used, not the raw case-file
  dict, so the log is not misleading about what was applied.

### 2.4 `network.py::_create_smopf_solver`

**Item 3 (primary policy).**
- Added `options['max_iter'] = 500` next to the `fixed_variable_treatment` pin
  (same before-merge placement as the ESSO side).
- Deleted the `if network.is_transmission: solver.options['acceptable_iter'] = 0;
  solver.options['acceptable_tol'] = ...` block entirely (the TSO-only override
  the brief asked to remove).
- Deleted the five `solver.options['warm_start_*'] = options.get('warm_start_*',
  options.get('<generic>', 1e-6))` lines (bound_push, bound_frac,
  slack_bound_frac, slack_bound_push, mult_bound_push). As on the ESSO side,
  nothing replaces them: any explicit `warm_start_*` key in `solver_params.options`
  or `option_overrides` is already applied via the earlier generic merge/apply
  loop (lines ~496-519, unchanged); if absent, IPOPT's compiled default (1e-3)
  applies. This removes **all five** generic-key fallbacks (`bound_push`,
  `bound_frac`, `slack_bound_frac`, `slack_bound_push` were also derived that way,
  not only `warm_start_mult_bound_push` at the line the brief named specifically),
  because leaving any of the other four fallbacks in place would have kept the
  exact mechanism that produced the DSO cycle-21 failure in Step 0 (case file sets
  the generic `bound_push=1e-5`, which silently drove `warm_start_bound_push` to
  the same value). `warm_start_init_point = 'yes'` and the two
  `replace_warm_start_suffix` calls are unchanged.

### 2.5 `network.py::_is_recoverable_network_failure`

Same change as the ESSO side: fires on `(internalSolverError, maxIterations,
infeasible)` instead of `internalSolverError` only.

### 2.6 `network.py::_run_smopf`

**Item 4.** The recovery branch already forced `from_warm_start=False` and
already cleared `ipopt_zL_in`/`ipopt_zU_in`/`dual` via the pre-existing
`_clear_multiplier_suffixes` (with a snapshot/restore-on-failure this file already
had and that this task did not change). Added: `recovery_options` is now built as
`params.solver_params.recovery_options` filtered to drop `hessian_approximation`,
plus an explicit `recovery_options['warm_start_init_point'] = 'no'` (previously
implicit via IPOPT's own cold-start default, now explicit per the brief's
wording). The log line was changed from "without multiplier warm start" to "cold
start" to match.

---

## 3. The dead `recovery_options` case-file entries

Per the brief's explicit instruction, `data/SRP1/SharedESS/SRP1_ESS_Params.json`
was **not edited**. Its `recovery_options.hessian_approximation: "limited-memory"`
entry is now dead: the code filters it out before applying `recovery_options` as
`option_overrides` on the cold retry (§2.3). Verified live (see §5.3): a
synthetic recovery-path solve's logged effective options were
`acceptable_iter=1, acceptable_tol=0.0001, warm_start_init_point=no` — no
`hessian_approximation` — confirming the filter is active.

**The same is now true, and was not explicitly named in the brief's file list, for
all three network case files**, since item 4 was written for both families:
`data/SRP1/case33_2/case33_2_params.json`,
`data/SRP1/case33_3/case33_3_params.json` and `data/SRP1/case9/case9_params.json`
each have `recovery_options.hessian_approximation: "limited-memory"`, and none of
them were edited; the network recovery path (§2.6) now filters this key out the
same way. **Flagging this for the Planner to raise with the author alongside the
ESSO one**, since it was not previously called out per-file.

---

## 4. Pre-Step-1 zero-solve check (`model.dual` belt-and-braces, not load-bearing)

Per the Planner's finding stated in the task (not re-derived here): the preserved
P5.12-R `.nl` files (`data/SRP1/Results/P512R/cycle21_prepared/original.nl`,
`cycle21_pre_setup/original.nl`) contain exactly two suffix segments (`S4 7492
ipopt_zL_in`, `S4 7320 ipopt_zU_in`) and the string `dual` appears zero times —
constraint multipliers are not exported for this model family. `model.dual.clear()`
was added to both recovery paths anyway (§2.3, and confirmed already present in
`network.py`'s `_clear_multiplier_suffixes`, §2.6) as instructed, and is recorded
here as **not load-bearing**: it clears a suffix that carries no exported content
for the ESSO in production. Step 0's A4 arm therefore stands unmodified; it was
not re-run (the Planner's finding states this explicitly and this task did not
re-derive it).

---

## 5. Gate results

### 5.1 Gate — `clobber_probe` overrides respected

Reused `p515_0_esso_warmstart_ab.esso_clobber_probe` unmodified, called against
the now-patched `shared_energy_storage_data._create_solver` with the same
request Step 0 used (`option_overrides={'warm_start_mult_bound_push': 1e-3,
'warm_start_bound_push': 1e-3, 'warm_start_slack_bound_push': 1e-3}`,
`from_warm_start=True`).

```
option_overrides_requested            : {'warm_start_mult_bound_push': 0.001, 'warm_start_bound_push': 0.001, 'warm_start_slack_bound_push': 0.001}
solver_options_observed_after_create  : {'warm_start_mult_bound_push': 0.001, 'warm_start_bound_push': 0.001,
                                          'warm_start_slack_bound_push': 0.001, 'warm_start_init_point': 'yes'}
clobbered                             : False
```

**PASS.** Previously (Step 0) the observed values were all `1e-9` regardless of
the request; now the requested values are exactly what lands on the solver, and
no frac key silently reappears (previously the two frac keys reappeared at `1e-9`
even though not requested — they are now simply absent, matching "leave at
compiled default unless explicit").

### 5.2 Gates — old policy still reproduces `maxIterations`; A1 policy converges fast; repeated on node 9

Instance: `data/SRP1/Results/P514N/esso_models_k10000.pkl` (same pickle Step 0
used), independent `copy.deepcopy` per arm, production `_create_solver` +
`solver.solve(...)`, `params.solver_params` built the same way Step 0 built it
(`p56a_oracle.fresh_planning`, read-only). Old-policy override =
`{'warm_start_mult_bound_push': 1e-9, 'warm_start_bound_push': 1e-9,
'warm_start_bound_frac': 1e-9, 'warm_start_slack_bound_push': 1e-9,
'warm_start_slack_bound_frac': 1e-9}` (all five, matching the deleted hardcoded
block exactly). A1 override = the same five keys at `1e-3`. Both passed via
`option_overrides` alone (no post-creation `solver.options` editing — unlike
Step 0, this is no longer necessary, since the clobbering that forced Step 0 to
use that technique is fixed).

| node | arm | termination | iterations | `admm_objective` | max constraint violation |
|---|---|---|---|---|---|
| 7 | old_policy (1e-9, via `option_overrides`) | `maxIterations` | 500 | 1.366326462684692 | 6.454e-06 |
| 7 | A1 policy (1e-3, via `option_overrides`) | `optimal` | 30 | -0.01198375471524498 | 2.916e-11 |
| 9 | old_policy (1e-9, via `option_overrides`) | `maxIterations` | 500 | 1.3721959549978053 | 1.099e-05 |
| 9 | A1 policy (1e-3, via `option_overrides`) | `optimal` | 23 | -0.008104095649293843 | 1.075e-12 |

**All four PASS** against the gate's stated criteria (old policy →
`maxIterations`; A1 policy → converged, "~30 iterations"). Full evidence:
`data/SRP1/Results/P5151A/p5151a_gate_report.json`,
`data/SRP1/Results/P5151A/logs/node{7,9}_{old_policy,a1_policy}.txt`.

Notes on the numbers, for the record:
- Node 7's old-policy run stops at exactly **500** iterations, not Step 0's
  3000, because `max_iter = 500` is now a production default (item 3) and this
  arm supplies no override for it; the termination condition is still
  `maxIterations` as the gate requires. The objective at 500 iterations
  (`1.366326462684692`) matches Step 0's A5 arm ("A0 + `max_iter=500`",
  `1.366326462684692`) to the last printed digit — consistent, since A5 was
  exactly this configuration.
- Node 7's A1-policy run matches Step 0's A1 arm exactly: 30 iterations,
  objective `-0.01198375471524498` to the last printed digit, confirming the new
  override path reaches the identical numerical result Step 0 obtained via the
  post-creation-edit technique.
- Node 9 was not part of Step 0's A1/A5 arms (Step 0 only ran A0/A4 on node 9);
  these are new numbers for this task, added per the brief's instruction to add
  one check of the Worker's own on node 9. Node 9's A1-policy objective
  (`-0.008104095649293843`) agrees with Step 0's node-9 A4 (cold start,
  `-0.008104095650270938`) to 9 significant figures, consistent with H-WS's
  conclusion that A1 and A4 reach the same point.

**`all_gates_pass = True`** (recorded in
`data/SRP1/Results/P5151A/p5151a_gate_report.json`).

### 5.3 Additional validation (not part of the mandated gate, run for corroboration)

Two extra checks through the full production `_optimize` entry point (not just
`_create_solver`), to confirm the fix behaves correctly end-to-end, not only at
the option-merge level:

**(a) Default policy, no overrides at all.** `SED._optimize(model, params,
from_warm_start=True, node_id=node_id)` with the unmodified production
`params` (no `option_overrides`) on both previously-failing nodes:

```
node 7: termination = optimal
node 9: termination = optimal
```

Both converge on the **primary** attempt (no recovery needed) with the new
"leave at compiled default" policy alone — i.e. the fix does not merely make
`option_overrides` respected, it also fixes the failure under the actual default
production configuration used by `optimize()`/`optimize_master_problem()`.

**(b) Recovery path, synthetic trigger.** To exercise the new
`_is_recoverable_shared_ess_failure`/`_optimize` recovery branch itself (the
default policy in (a) no longer fails, so it cannot trigger recovery on its
own), `params.options` was temporarily monkeypatched in a throwaway copy (not
the case file) to reinstate the old five `1e-9` keys, forcing a primary
`maxIterations`:

```
[WARNING] Shared ESS primary solve did not converge for ESS node=7: status=warning, termination=maxIterations, ...
[INFO] Retrying Shared ESS solve once for ESS node=7, cold start, with acceptable_iter=1, acceptable_tol=0.0001, warm_start_init_point=no.
[INFO] Shared ESS recovery solve succeeded for ESS node=7.
final termination: optimal
diagnostic_sink recovery_options: 'acceptable_iter=1, acceptable_tol=0.0001, warm_start_init_point=no'
```

Confirms: recovery fires on `maxIterations`; the applied recovery options are
exactly `acceptable_iter`/`acceptable_tol` (from the case file) plus
`warm_start_init_point=no` (added by the code) — **no `hessian_approximation`**,
i.e. the filter in §2.3/§3 is live; the recovery solve reaches `optimal`
("Solved To Acceptable Level").

This synthetic check is not persisted as a JSON artifact (it used a
monkeypatched, non-case-file `params` object purely to exercise the recovery
branch) — reported here as console evidence only, per the "not a number that
will be reused" carve-out; the numbers that are reused (§5.2, §5.1) are in the
JSON report.

---

## 6. Validation

- `py_compile` clean on both modified files.
- Module import smoke test: `import network`, `import shared_energy_storage_data`
  succeed; the four target functions resolve.
- `grep` across the repository for the deleted hardcoded literal
  (`warm_start_bound_push.*1e-9`) and the removed TSO override
  (`acceptable_iter.*= 0`) found no other production or harness file depending on
  either (only this task's own new harness, which intentionally passes `1e-9` as
  a test override value).
- `network.is_transmission` (the attribute the deleted block used) remains
  defined and used elsewhere in `network.py` for unrelated purposes (voltage/PQ
  handling) — removing its one warm-start use here does not orphan the
  attribute.
- Confirmed via `git status` that only `network.py` and
  `shared_energy_storage_data.py` (tracked) were modified, plus the new untracked
  harness `p515_1a_gate_check.py` and the untracked evidence directory
  `data/SRP1/Results/P5151A/`. No `data/` file was edited. No `git add` was run
  (nothing staged, nothing committed).
- One stray root-level log file created by an ad hoc interactive sanity check
  (`optim_log_node_7_recovery.txt`, a byproduct of §5.3(b), which used the
  production default output-file naming rather than the harness's redirected
  logs) was deleted after inspection; the pre-existing, gitignored
  `optim_log_node_7.txt` (append-only, `.gitignore` line 25) received a few
  additional appended solves from the same checks (harmless, consistent with
  how this file has always accumulated across runs — Step 0 flagged this as a
  pre-existing, out-of-scope condition, not something this task introduced).

**Distinguishing what was actually shown, per the reporting convention:**
- *Code executes correctly*: yes (compiles, imports, gate solves ran to
  completion with the expected termination conditions and matching numbers).
- *Gate passes*: yes, all four required checks plus the added node-9 check
  (`all_gates_pass = True` in the JSON report).
- *Underlying numerical problem is solved*: partially and narrowly — this task
  fixes the warm-start-policy mechanism identified in Step 0 (H-WS) and confirms
  it end-to-end on the two known-failing ESSO instances via both the raw
  override path (§5.2) and the full `_optimize`/default-policy path (§5.3a). It
  does **not** by itself establish the ESSO reformulation's correctness (Step 1
  proper, not authorized as part of this task) or run the DSO/network family
  gate (not required by Step 1a's gate; the network-side production changes were
  made per the brief but only compile/import-checked here, not solved against
  the cycle-21 fixture in this task).

---

## 7. Unexpected findings

1. **The ESSO recovery call previously inherited the primary's `from_warm_start`
   flag** (`_run_solver_attempt(..., from_warm_start=from_warm_start, ...)`
   inside `_optimize`'s recovery branch) rather than forcing a cold retry. Under
   the old policy this was largely moot (the hardcoded `1e-9` pushes made
   "recovery" ineffective anyway, since `internalSolverError` was the only
   trigger and warm/cold made little practical difference to that failure mode),
   but it was inconsistent with `network.py`'s equivalent branch, which already
   forced `from_warm_start=False`. Fixed as part of item 4 (§2.3).
2. **The network case files' `recovery_options.hessian_approximation` entries
   also become dead** by this change (§3), not only the ESSO one the brief named
   explicitly by path. Flagging for the Planner.
3. Passing the OLD policy through `option_overrides` under the *new* production
   code now hits `maxIterations` at **500** iterations rather than Step 0's 3000,
   because `max_iter = 500` is now unconditionally in effect unless overridden —
   this is expected (item 3), not a regression, and the gate wording ("still
   reproduces `maxIterations`") is satisfied by the termination condition, not a
   specific iteration count.

## 8. Remaining issues

- The network-family production changes (§2.4-2.6) were validated by
  compile/import checks and by grep for stale dependents, but **not** by an
  actual solve against the preserved cycle-21 DSO fixture in this task — Step
  1a's mandated gate is ESSO-only (node 7 and node 9). The Planner may want a
  DSO-side re-run of Step 0's A0 arm through the new code before relying on the
  network-family change in a gate run, though the brief's own Step 1a gate does
  not require it.
- `_build_subproblem`, the master-problem Benders-cut retirement, and the
  ESSO reformulation itself are explicitly out of scope for this task and were
  not touched.

## 9. Questions for Planner

1. Should a DSO-side re-run of Step 0's A0/A1/A4 arms through the now-fixed
   `network.py` code be scheduled before Step 1's gate runs, given Step 1a's
   gate did not require it (see §8)?
2. Confirming §3: should the Planner raise the now-dead
   `hessian_approximation: "limited-memory"` entries in all four
   `recovery_options` blocks (`SRP1_ESS_Params.json`, `case33_2_params.json`,
   `case33_3_params.json`, `case9_params.json`) with the case-file author in one
   pass, or only the ESSO one as originally named?

---

## 10. Addendum — Part A of the Step 1 task (2026-09-13): DSO solve validation, STOPPED

Per `PLANNER_BRIEF_2026-09-13.md`'s Step-1 task instructions (Part A, a
pre-flight gate ahead of the ESSO reformulation, Part B): "validate the Step
1a network change by solving (it was only compile/grep-checked)" on the
preserved cycle-21 DSO fixture, three arms (default/no-overrides,
old-policy-via-`option_overrides` at "all five `warm_start_*` at 1e-6", cold),
through the production `network.py` path, plus an inspection that an override
setting only `bound_push` leaves `warm_start_mult_bound_push` unset.

New harness: `p515_1_partA_dso_check.py` (imports and reuses
`p515_0_esso_warmstart_ab.parse_ipopt_log_tail` /
`.max_constraint_violation` / `.active_objective` / `.dso_preflight`
unmodified). Instance: `data/SRP1/Results/P512R/cycle21_pre_setup/snapshot.pkl`,
target `DSO:case33_3|2025|Spring` (preflight `pass: True`, unchanged
verification method from §5 above). Evidence:
`data/SRP1/Results/P5151A/partA/p5151_partA_report.json`,
`data/SRP1/Results/P5151A/partA/partA_logs/*.log`.

### 10.1 Results

| arm | option_overrides | termination | iterations | objective | max constraint violation |
|---|---|---|---|---|---|
| default | none | `optimal` | 85 | 1293.666817122966 | 7.135e-09 |
| old_policy (literal, 1e-6) | all five `warm_start_*` = 1e-6 | **`optimal`** | 96 | 1293.6668155201835 | 7.135e-09 |
| cold | `warm_start_init_point='no'`, suffixes (`ipopt_zL_in`, `ipopt_zU_in`, `dual`) cleared | `optimal` | 77 | 1293.6668125362894 | 7.135e-09 |
| old_policy_1e5 (diagnostic, not a brief arm) | all five `warm_start_*` = 1e-5 | `maxIterations` | 500 | 1302.2785427050633 | 0.0709 |

- **`default` matches the brief's stated expectation**: "Step 0's A1 converged
  in ~85 iterations" — the new default-policy arm converges in exactly 85
  iterations here (the brief called this an expectation, not a requirement;
  it is met).
- **`default` and `cold` agree on the objective to 1e-6 relative**: PASS.
  `|1293.666817122966 − 1293.6668125362894| / 1293.6668… = 3.545e-09`.
- **The `derivation_gone` inspection PASSES**: `option_overrides={'bound_push':
  1e-5}` alone (no `warm_start_*` key) leaves all five `warm_start_*` keys
  (`warm_start_mult_bound_push`, `warm_start_bound_push`,
  `warm_start_bound_frac`, `warm_start_slack_bound_push`,
  `warm_start_slack_bound_frac`) **absent** from the assembled
  `solver.options` — confirmed by direct inspection of the dict returned
  from `_create_smopf_solver`, not inferred. The former derivation
  (`warm_start_mult_bound_push <- bound_push`, etc.) is gone, matching the
  Step 1a diff (§2.4 above).
- **The literal `old_policy` arm, as specified in the task ("all five
  `warm_start_*` at 1e-6"), does NOT reproduce `maxIterations`.** It converges
  to `optimal` in 96 iterations, at an objective (1293.6668155201835) that
  agrees with `default` and `cold` to ~1.2e-9 relative — i.e. 1e-6 is simply a
  mild, still-well-conditioned warm start on this fixture, not a
  failure-inducing one.

### 10.2 Root cause of the mismatch (diagnostic, not a brief-required arm)

Inspecting the archived Step 0 evidence
(`data/SRP1/Results/P5150/p5150_report.json`, `dso.arms.A0.pre_adjust_options`)
shows the **pre-1a** production code's actual derived values on this fixture
were **1e-5**, not 1e-6: `case33_3_params.json` sets the generic
`bound_push`, `bound_frac`, `slack_bound_push`, `slack_bound_frac` keys to
`1e-5` each, and the old derivation block (`network.py`, removed in Step 1a,
§2.4 above) read `options.get('warm_start_bound_push', options.get(
'bound_push', 1e-6))` — i.e. it used the case file's `1e-5` value, falling
back to the hardcoded `1e-6` only when the case file left the generic key
unset. The `1e-6` value named in the Step-1-task text is the **code's own
compiled fallback default**, not the value that actually drove this
fixture's `maxIterations` failure in Step 0/P5.12-P.

Confirmed directly: re-running the identical arm with all five
`warm_start_*` keys at **1e-5** (`old_policy_1e5_diagnostic` in the table
above) reproduces `maxIterations` at exactly 500 iterations (the new
`max_iter` cap), with an unconverged objective (1302.2785427050633, max
constraint violation 0.0709) — consistent with Step 0's A0 (`maxIterations`
at 3000 iterations under the old `max_iter`, objective 1302.2135784698214,
same order of unconverged-objective mismatch against the converged
~1293.67). This is offered as evidence for the Planner's use, not as a
substitute for the literal arm the task specified.

### 10.3 Gate verdict and reason for stopping before Part B

Per the task's own instruction — "If any of this fails, STOP and report
without starting Part B" — **Part A's gate is not fully satisfied**:

- PASS: `default` vs `cold` objective agreement (1e-6 relative).
- PASS: the `bound_push`-only-override inspection (`warm_start_mult_bound_push`
  and siblings left unset).
- **FAIL**: the specified `old_policy` arm ("all five `warm_start_*` at
  1e-6") does not reproduce `maxIterations` on this fixture; it converges to
  `optimal`. The mismatch is fully explained (§10.2): the code's fallback
  default (1e-6) and the fixture's actual historical value (1e-5, from the
  case file) are different numbers, and the task's arm-2 specification names
  the former, not the latter.

**Consequently, per the task's explicit pre-flight-gate instruction, Part B
(the ESSO subproblem reformulation) was NOT started.** This report and its
evidence directory (`data/SRP1/Results/P5151A/partA/`) are being appended per
the task's instruction to "Append the results to `P5_15_1A_REPORT.md` and its
evidence dir" even though the gate did not fully pass, so the Planner has the
complete evidence to decide how to proceed (e.g., re-issue Part A's arm 2
at `1e-5` instead of `1e-6`, or accept the `1e-5` diagnostic result as
satisfying the intent of the gate and authorize Part B on that basis).

No production file was modified in this addendum. Files touched: the new
harness `p515_1_partA_dso_check.py` (repo root, new), and its evidence under
`data/SRP1/Results/P5151A/partA/` (new). Nothing under `data/` was edited
apart from adding this new evidence directory (result data, not case
data). Not committed; `git add .`/`git add -A` not used.
