# Worker Report — ESSO log-handling fix (P5.15-F, 2026-09-14)

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 5 (ESSO log-handling fix, four parts) and
`P5_15_G1_G4_BLOCKED.md`. One production change, one commit's worth, **not committed by the
Worker** (per instruction).

---

## 1. Files modified

- `shared_energy_storage_data.py`
- `shared_resources_planning.py`

No other file was edited. `git status --porcelain -- '*.py' | grep -v '^??'` shows only these
two. `.p515_g_gate.lock` was not touched; `p515_g_g1_g4_admm_gates.py` was never run (verified:
no matching process at any point in this session).

---

## 2. Diff summary, per file

### `shared_energy_storage_data.py` (+118/−… lines; full diff inspected, reproduced in essence below)

1. **`SharedEnergyStorageData.__init__`**: added `self.logs_dir = str()` (empty default —
   documented backward-compatible fallback for callers that never set it).
2. **`optimize_master_problem`**: now passes `logs_dir=self.logs_dir` to `_optimize` (inert for
   the master problem today, since it uses the non-IPOPT LP solver and the `output_file`
   branch is gated on `params.solver.lower() == 'ipopt'`).
3. **`optimize` (ESSO subproblem entry point)**: new `cycle=None` parameter, threaded to
   `_optimize(..., cycle=cycle, logs_dir=self.logs_dir)`. `cycle=None` is the documented
   fallback (see stamp scheme below).
4. **`_create_solver`**: new `cycle=None, logs_dir=None` parameters.
   - The default `output_file` (when the caller did not supply one) is now
     `optim_log_esso_node{node_id}_{stamp}.txt` (or `optim_log_esso_master_{stamp}.txt` if
     `node_id is None`), where `stamp = f'cycle{cycle:03d}'` for an integer `cycle`, else
     `'init'`.
   - `output_file` is resolved against `logs_dir` (`os.path.join`, then `os.makedirs(logs_dir,
     exist_ok=True)`) exactly the way `network.py:520-521` already resolves the TSO/DSO
     `output_file` against `network.logs_dir`. If `logs_dir` is falsy (empty string —
     `SharedEnergyStorageData`'s default, or a caller that never set it), the fallback is the
     pre-fix behaviour: resolve the bare filename against the process cwd
     (`os.path.abspath`). An already-absolute caller-supplied `output_file` (some harnesses set
     one via `option_overrides`) is unaffected — `os.path.join` with an absolute second
     argument discards `logs_dir`.
   - **Collision handling**: if the resolved path already exists, the file is **not** appended
     to. A `_dup2`, `_dup3`, ... suffix is chosen (first non-colliding), and a `[WARNING]` is
     printed stating the original and the chosen path.
   - `options['file_append']` changed from the previous unconditional `'yes'` to `'no'` — one
     fresh file per solve is now the mechanism for isolation, not `file_append`.
5. **`_run_solver_attempt`**: new `cycle=None, logs_dir=None` parameters, forwarded to
   `_create_solver`.
6. **`_optimize`**: new `cycle=None, logs_dir=None` parameters, forwarded to **both**
   `_run_solver_attempt` calls (primary and the cold-start recovery retry), so a recovery log
   gets the same node/cycle stamp plus the existing `_recovery` suffix (unchanged mechanism).
7. **`_parse_ipopt_barrier_terms`**: changed from `re.search` (first match) to
   `_REGEX.finditer(text)` and taking the **last** match (`[-1]`) for both the `Objective` and
   `Complementarity` lines. Return contract (`mu_final, s_obj, reason`) unchanged. Docstring
   amended to explain why (the `df46f118` bug) and to note this is now normally a no-op (one
   solve per file) except for a log that was NOT freshly isolated (e.g. a caller-forced
   `file_append` via `option_overrides`, or a pre-fix log).

### `shared_resources_planning.py` (+156/−… lines)

1. **`SharedResourcesPlanning.__init__`**: `self.results_dir`, `self.diagrams_dir`,
   `self.logs_dir` now wrapped in `os.path.abspath(...)`, resolved at construction time
   (before any caller can `os.chdir`). This is the single point every downstream
   `results_dir`/`logs_dir` (TSO, DSO, ESSO — all assigned `= planning_problem.results_dir` /
   `.logs_dir` in `_read_planning_problem`) inherits from.
2. **`_read_planning_problem`**: added `shared_ess_data.logs_dir = planning_problem.logs_dir`,
   at the same site `distribution_network.logs_dir`/`transmission_network.logs_dir` are already
   pushed (lines ~6044, ~6088 pre-diff). This is also what makes
   `p56a_oracle.fresh_planning`'s existing `hasattr(planning.shared_ess_data, 'logs_dir')` guard
   now fire (it previously never did, since the attribute did not exist).
3. **`_run_operational_planning`**: the per-cycle ESSO call now passes `cycle=iter`:
   `update_shared_energy_storages_coordination_model_and_solve(..., cycle=iter)`.
4. **`update_shared_energy_storages_coordination_model_and_solve`**: new `cycle=None`
   parameter, forwarded to `shared_ess_data.optimize(models, from_warm_start=..., cycle=cycle)`.
5. **`create_shared_energy_storage_model`** (the initialization call site, ~line 3192): **not
   modified** — its call `shared_ess_data.optimize(esso_model)` still omits `cycle`, which is
   exactly the documented fallback path that stamps `'init'`.
6. **`_save_frozen_smopf_block`** and **`_save_frozen_network_block`**: both now wrap their
   entire body in `try/except Exception`. On any exception, they print a `[WARNING][FROZEN
   SMOPF] ...` with the exception (`error!r`) and the full context (agent/node/network/year/
   day/cycle/save_dir), and **return `None`** instead of propagating. Neither function's success
   path changed. Confirmed both callers (`network_data.py:NetworkData.optimize`, the only
   caller of the `failure_snapshot_callback`/`pre_solve_snapshot_callback` contract) never
   dereference the return value, so returning `None` on failure changes nothing about control
   flow downstream — the failure itself is still recorded by the caller's own
   `_solver_result_succeeded` check, independent of whether the snapshot saved.

**Full diff** (`git diff -- shared_energy_storage_data.py shared_resources_planning.py`) was
inspected in full before running any test; no unintended changes (no algorithm, tolerance, rho,
budget, warm-start, or recovery-policy edits — verified by re-reading the diff hunks: only the
lines quoted above changed).

---

## 3. The stamp scheme (item 2)

| call site | `cycle` passed | stamp | example filename |
|---|---|---|---|
| `create_shared_energy_storage_model` (init, before ADMM starts) | not passed (`None`) | `init` | `optim_log_esso_node7_init.txt` |
| `_run_operational_planning`'s ADMM loop, iteration `iter` | `iter` (int) | `cycle{iter:03d}` | `optim_log_esso_node7_cycle002.txt` |
| any recovery retry (either site) | same `cycle` as its primary attempt | + `_recovery` suffix (existing `log_suffix` mechanism) | `optim_log_esso_node7_cycle002_recovery.txt` |
| collision (name already exists) | — | `_dup2`, `_dup3`, ... appended, with a `[WARNING]` | `optim_log_esso_node7_init_dup2.txt` |

All ESSO logs resolve against `shared_ess_data.logs_dir` (falsy → falls back to the pre-fix
cwd-relative behaviour, for backward compatibility with any caller that never sets it).
`file_append` is `'no'` unconditionally now — isolation is by filename, not by append-then-parse.

---

## 4. Commands / experiments run

All four tests below were run **in the foreground** (no `screen`/`nohup`/`&`), stderr captured
into the same log as stdout (`2>&1`), one at a time, with `guard` (`SolveProfileGuard`) armed
for every test that solves. One correction mid-task: my first T2 invocation was accidentally
backgrounded with a trailing `&` (a mistake on my part, against the stated execution rules); it
was killed **before it produced any output** (confirmed empty log) and re-run correctly in the
foreground with an explicit extended `timeout`. No other command was backgrounded; T1's
first attempt exceeded the tool's default 120s auto-background threshold (not a `screen`/
`nohup`/`&` on my part) — I did not treat that as a violation, but subsequent commands used an
explicit longer `timeout` parameter to stay genuinely foregrounded for their actual (always
<9 min) duration.

Test harnesses are diagnostic-only Worker scripts, written to the session scratchpad (not
committed to the repo), importing real production functions/constants throughout (per
CLAUDE.md): `p56a_oracle.fresh_planning`, `p514_n_instrumented_cstar` constants (`S_INV, E_INV,
INVEST_YEAR, BUDGET, RHO, REL`), `p59_rho.apply_rho_to_params/set_adaptive_penalty`,
`p58_rescale.patched_admm_objectives`, `shared_resources_planning._rebuild_candidate_total_
capacities`, `p515_h_tol_remedy_check.run_arm`, `p514_l_capacity_ladder.main`,
`shared_energy_storage_data._parse_ipopt_barrier_terms` / `._get_esso_complementarity_
diagnostics`, and `p513_solve_profile_guard.SolveProfileGuard`.

Interpreter used throughout: `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`
(canonical, per `CLAUDE.local.md`).

### T1 — two-cycle probe

```
cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python t1_two_cycle_probe.py \
    > t1_stdout.log 2>&1
```
C* control configuration (imported from `p514_n_instrumented_cstar.py`): `S_INV=0.96875`,
`E_INV=3.875`, `INVEST_YEAR=2025`, `BUDGET=5.0e6`, `REL=1e-4`, `RHO={'v':1.5,'pf':300.0,
'ess':1.0}`, adaptive penalty on. `num_max_iters` set to **2** (only deviation from C*). Ran via
`O.fresh_planning('p515f_t1_two_cycle')`, `run_operational_planning(type='distributed',
return_state=True)`. Guard permitted sites: `network.py:_run_smopf_solver_attempt`,
`shared_energy_storage_data.py:_run_solver_attempt`.

### T2 — deliberately triggered network failure + no-abort unit test

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python t2_failure_probe.py \
    > t2_stdout.log 2>&1
```
Part A: same C* configuration, `num_max_iters=1`. A **test-harness-only** monkeypatch of
`network._run_smopf_solver_attempt` forces `max_iter=1` on exactly one TSO block (the first
`(year, day)` reached inside `update_transmission_coordination_model_and_solve` — i.e. the
per-cycle path, not the pre-ADMM initialization path, which has no
`failure_snapshot_callback` wired at all and so would not exercise the code this fix changes);
applied to both the primary and the recovery attempt so it is not rescued. The planning object
is constructed/loaded (`O.fresh_planning`) **at the repo root**, then `os.chdir` to a fresh
`tempfile.mkdtemp()` **after** construction, reproducing the exact sequence that crashed the
original campaign (construct with a then-relative `results_dir` → chdir → local failure →
`_save_frozen_network_block`'s `os.makedirs` resolves against the wrong cwd). `FrozenSMOPF`
output redirected to `data/SRP1/Results/P515F/t2_results` (new path; production's own
`data/SRP1/Results/FrozenSMOPF` tree is never touched). Part B calls
`shared_resources_planning._save_frozen_network_block` directly with a `save_dir` that walks
through a plain file (guaranteeing `NotADirectoryError` from `os.makedirs`, portable and
deterministic — no reliance on filesystem permission semantics).

### T3 — single-solve detector values, re-run by import

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python t3_tol_remedy_recheck.py \
    > t3_stdout.log 2>&1
```
Imports `p515_h_tol_remedy_check` and calls its `run_arm(tag, tol_overrides)` directly for both
arms (`control`: tol=1e-6/acceptable_tol=1e-5; `remedy_h`: tol=1e-8/acceptable_tol=1e-7) — the
module's `main()` (which writes onto the committed
`data/SRP1/Results/P5151/tol_remedy_check_summary.json`) was **never called**. The module's
`LOG_ROOT` global was monkeypatched to `data/SRP1/Results/P515F/t3_tol_check_logs` before the
calls and restored after, so nothing under `data/SRP1/Results/P5151/` was written to. Guard
permitted only `shared_energy_storage_data.py:_run_solver_attempt`, declared count 6 (2 arms ×
3 nodes), verified exactly with `guard.verify(6)`.

### T4 — capacity-ladder initialization + zero-solve check

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python t4_ladder_zero_solve_check.py \
    > t4_stdout.log 2>&1
```
Imports `p514_l_capacity_ladder` and calls `L.main(1.00)` (1.00 MVA / 4.00 MWh, `RATIO=4.0`
hard-coded in that module, node 7 the paper's focus — the module applies the same investment to
all active nodes). The module's `OUT` global (report-JSON output directory) was monkeypatched
to `data/SRP1/Results/P515F/t4_ladder` before the call and restored after; `data/SRP1/Results/
P514L/ladder_s1.json` (the committed artifact) was **not** written to (verified below).

---

## 5. Results

### T1 — two-cycle probe

Wall clock: **131.6 s** for 2 cycles (well under the ~9 min stop threshold).

Nine ESSO log files, one per (node, stamp), each containing **exactly one** IPOPT run
(`grep -c "This is Ipopt" <file>` = 1 for all nine, checked directly):

```
optim_log_esso_node5_cycle001.txt   optim_log_esso_node5_cycle002.txt   optim_log_esso_node5_init.txt
optim_log_esso_node7_cycle001.txt   optim_log_esso_node7_cycle002.txt   optim_log_esso_node7_init.txt
optim_log_esso_node9_cycle001.txt   optim_log_esso_node9_cycle002.txt   optim_log_esso_node9_init.txt
```

`esso_complementarity_diagnostics` sink, 9 entries, `mu_final` per node/cycle (parsed from that
solve's own file):

| node | init | cycle 1 | cycle 2 |
|---|---|---|---|
| 5 | 3.1316e-09 | 2.9927e-09 | 4.2524e-09 |
| 7 | 3.1527e-09 | 3.0710e-09 | 4.3232e-09 |
| 9 | 3.1275e-09 | 2.9795e-09 | 4.2496e-09 |

Cycle-1 and cycle-2 values are **not all identical** (required). `logs_dir` in force:
`planning.logs_dir`, `shared_ess_data.logs_dir`, and `transmission_network.logs_dir` were all
the same absolute `.../P56A/evals/p515f_t1_two_cycle/logs` directory (confirming item 1's
propagation matches the network family's). Guard: `permitted_solve=154, blocked_solve=0`.

### T2 — failure probe

**Part A.** Target block forced: `(2025, 'Spring')` (TSO, `case9`). `forced_solver_attempt_calls
= 2` (primary + recovery, both forced, confirming the failure was not rescued).
`tso_results_dir_before_chdir` = `.../data/SRP1/Results/P515F/t2_results` — **absolute**
(`tso_results_dir_is_absolute = True`), set before the `os.chdir`. `completed_without_exception
= True`, `error = None`. Wall clock: **78.6 s**. Guard: `permitted_solve=104, blocked_solve=0`.
A `FrozenSMOPF` pickle was written under the absolute results dir:
`data/SRP1/Results/P515F/t2_results/FrozenSMOPF/failure_TSO_case9_2025_Spring_cycle1.pkl`.

**Part B.** `_save_frozen_network_block(..., save_dir=<path through a plain file>)` printed
`[WARNING][FROZEN SMOPF] Could not save failure pre-solve block for agent=TSO, node_id=None,
network=unit_test_network, year=2025, day=UnitTest, cycle=1 to save_dir=...: NotADirectoryError
(20, 'Not a directory')` and **returned `None`** — no exception propagated to the caller.

### T3 — single-solve detector re-check

Guard: `permitted_solve=6, blocked_solve=0`, `guard.verify(6)` returned **no failures**
(exact declared count matched).

For every (arm, node) pair, `mu_final`, `s_obj`, `complementarity_ratio_max`,
`spurious_throughput_measured`, `spurious_throughput_bound` from the **production sink**
(`shared_ess_data.esso_complementarity_diagnostics`, populated with the currently-in-force,
correct log path) matched the committed `data/SRP1/Results/P5151/tol_remedy_check_summary.json`
values **exactly**:

**`MAX_RELATIVE_DIFFERENCE = 0.0`** (bit-identical across all 6 node/arm combinations and all 5
fields — well inside the ≤1e-12 requirement).

**Collateral finding (reported, not fixed — out of this task's authorized scope):** the
harness's own secondary, manually-reconstructed path guess
(`p515_h_tol_remedy_check.py:182`, `primary_log = os.path.join(arm_dir,
f'optim_log_node_{node_id}.txt')`) no longer resolves to a real file, because production now
writes `optim_log_esso_node{node_id}_init.txt` under `shared_ess_data.logs_dir` (the oracle's
own per-eval logs directory) instead of the old bare filename resolved against the harness's
`os.chdir`'d `arm_dir`. All 6 of these harness-local re-parses now report `parse_reason=
'IPOPT log not found: ...'`. This is exactly the class of pre-existing-harness breakage
`REVISION_CONTEXT.md`/`CLAUDE.md` treat as acceptable collateral from an authorized production
change ("historical harnesses may break and that is acceptable, list them"); the actual
production computation is unaffected (confirmed via the sink, above). Full list of the 6 broken
harness-local reconstructions is in
`data/SRP1/Results/P515F/t3_tol_remedy_recheck.json` →
`harness_own_path_reconstruction_broken_by_this_fix`.

### T4 — capacity ladder + zero-solve check

`p514_l_capacity_ladder.main(1.00)`: `all_ok=True, failed=[], solves=51 (0 blocked), wall=39s`
— reproduces the committed G3-init PASS at this rung.

For the first time, this eval directory (`data/SRP1/Results/P56A/evals/ladder_s1/logs/`)
contains per-node ESSO **init** logs at all — confirmed by directory listing **before** this
run: it held 48 network SMOPF `.log` files and **zero** `optim_log_node_*`/`optim_log_esso_*`
files. Under the pre-fix code the ESSO never had `logs_dir` awareness, so its bare relative
`output_file` resolved against whatever the process cwd happened to be at solve time — never
into this structured per-eval location. This is itself evidence for the defect Addendum 5
diagnosed.

`EPS_ESSO_THROUGHPUT` in force: `0.001` (case default, per Addendum 2/3 gate — unchanged).

| node | `mu_final` | `s_obj` | predicted = `mu_final/(2·s_obj·ε)` |
|---|---|---|---|
| 5 | 3.1489e-09 | 0.1 | 1.5745e-05 |
| 7 | 3.1555e-09 | 0.1 | 1.5778e-05 |
| 9 | 3.1304e-09 | 0.1 | 1.5652e-05 |

Measured reference (independently re-verified on disk, `data/SRP1/Results/P515G/ladder/
ladder_s1.json` → `complementarity_detector_global.absolute_violation` =
**2.4031141706579106e-05**, matching the `2.4031e-05` cited in `P5_15_EXPERT_HANDOFF_3.md` /
`WORKER_REPORT_G1_G4.md`; Addendum 5's own ratio approximation of that: `≈2.5e-05`).

`predicted / measured_ratio_reference (2.5e-05)` ≈ **0.626–0.631** across the three nodes — i.e.
the identity's prediction is now within a factor of **~1.6×** of the measured reference, a large
narrowing of the previously reported **~5×** gap (`P5_15_EXPERT_HANDOFF_3.md` compared the
ladder's ~2.4e-05 against a *different* instance's 4.5e-06, not this call site's own
`mu_final`/`s_obj`; this is the first time `mu_final`/`s_obj` have been parsed from this
call site's own logs at all).

**`tol` actually in force.** The log text itself does **not** literally print `tol` (or any
option value) at production's default `print_level` — confirmed by `grep -i tol` on the raw log
file (no match) and by inspecting the header (option summary block is not part of IPOPT's
default-verbosity output; only the NLP-size summary and the iteration table are). So "read tol
... from the log header/options" cannot be satisfied by a literal text match under current
verbosity, and I did not change `print_level` (out of scope — "no change to ... solver
options"). `tol=1e-8` is nonetheless established as in force on this path by two independent,
non-text-literal pieces of evidence: (a) T3's bit-identical, guard-verified confirmation that
`SED.ESSO_TOL_OVERRIDES = {'tol': 1e-8, 'acceptable_tol': 1e-7}` is applied via
`option_overrides` at exactly this `optimize()` entry point (`create_shared_energy_storage_
model` → `shared_ess_data.optimize(esso_model)`, the same call `p514_l_capacity_ladder.main`
uses); (b) the log's own terminal barrier value, `lg(mu) = -8.6` at the last iteration
(`iter 60`, node 7 shown), i.e. `mu ≈ 2.5e-9` — consistent with a `tol=1e-8` target, not the
case-file default `tol=1e-6`. This resolves Addendum 5's explicit "not verified from the logs"
note for this path, with the caveat stated precisely (inferred, not literally read).

---

## 6. Validation

- `ast.parse` on both modified files: OK (syntax valid) before any test was run.
- Full `git diff` on both files read line-by-line before running anything; confirmed no changes
  outside the four authorized parts (no formulation, tol/eps/penalty/rho/budget/warm-start/
  recovery-policy edits).
- `git status --porcelain -- '*.py' | grep -v '^??'` → only the two intended files modified.
- Committed evidence files `data/SRP1/Results/P5151/tol_remedy_check_summary.json` and
  `data/SRP1/Results/P514L/ladder_s1.json` independently re-checked (timestamp + md5) to have
  **not** been modified by this task's runs (both still dated 2026-09-13, pre-dating this
  session).
- `.p515_g_gate.lock` present and unmodified; `p515_g_g1_g4_admm_gates.py` never invoked.
- All four tests ran to completion with exit code 0, guard-armed where solves occurred, guard
  `blocked_solve/blocked_exec = 0` in every test, and (T3) `guard.verify(EXPECTED_SOLVES)`
  returned no failures against a count declared in advance.
- Distinguishing what was actually established: **code executes correctly** (all four tests ran
  to completion) — confirmed. **Requested diagnostic works** (per-solve log isolation, per-cycle
  detector trajectory as a directory listing, no-abort failure snapshot, last-match parsing,
  absolute results_dir) — confirmed by T1/T2. **Single-solve values unchanged** — confirmed
  bit-identical by T3. **The underlying `tol=1e-8`-in-force question and the ladder's ~5× gap**
  — narrowed and mostly explained (T4), not fully closed (the remaining ~1.6× residual between
  predicted and measured is not diagnosed further here — out of this task's scope, which was
  the log-handling defect, not the barrier-identity residual itself).

---

## 7. Unexpected findings

1. **The pre-fix ESSO never wrote logs into the oracle's structured per-eval directory at all**
   (T4, §5) — not merely "unisolated," but effectively uncaptured for any harness that used
   `p56a_oracle.fresh_planning` without its own explicit `os.chdir` workaround. This is a wider
   capture gap than "cycle 1's numbers repeat for every later cycle"; for `p514_l_capacity_
   ladder.py` specifically, ESSO per-solve logs did not exist anywhere in the structured
   evidence tree before this fix.
2. **`p515_h_tol_remedy_check.py`'s own secondary path-reconstruction (not its use of production
   functions) is now stale** (T3, §5) — reported as a broken-historical harness detail, not
   fixed (out of scope; the harness's *primary* diagnostic path, `esso_complementarity_
   diagnostics_sink`, is unaffected since it is populated by production itself).
3. The `P56A/evals/ladder_s1/logs/` directory is reused verbatim by `p514_l_capacity_ladder.py`
   (hard-coded eval id `f'ladder_s{s_mva:g}'`), so T4's run appended (via the *network* family's
   still-unmodified `file_append='yes'`, `network.py`, out of this task's scope) onto the 48
   pre-existing network SMOPF `.log` files there. This is non-destructive (append-only) and
   matches the precedent already set by a prior Worker task (`WORKER_REPORT_G1_G4.md`'s G3-init
   rerun), which reused the same eval id without flagging it. No ESSO log was appended to (all
   nine — three nodes × {init} — were newly created, since none existed there before, per
   finding 1).
4. My own process error, corrected: the first T2 launch was mistakenly appended with `&`, which
   the task's execution rules explicitly forbid. It was killed immediately, before producing any
   output, and re-run correctly. Recorded here per the instruction to not hide failures/mistakes.

---

## 8. Remaining issues

- The ~1.6× residual between the barrier identity's prediction and the measured detector at the
  capacity-ladder initialization call site (T4) is narrowed from the previously-reported ~5× but
  not closed. Not diagnosed further — outside this task's authorized scope.
- `p515_h_tol_remedy_check.py`'s own manual log-path reconstruction (lines ~182-190) is stale
  and will report `parse_reason='IPOPT log not found'` for all 6 node/arm entries on any future
  re-run of that harness's `main()`; not repaired (Addendum 2 explicitly limits harness repair to
  `p514_n_instrumented_cstar.py` and `p514_l_capacity_ladder.py`).
- G1/G2/G3-full/G4 themselves were **not** re-attempted in this task (out of scope — this task
  was the log-handling fix and its four required tests only, per the Planner brief's explicit
  dispatch order: "stage this round by name → fix + tests → zero-solve check → G1 → G2 →
  G3-full → G4, sequentially").

---

## 9. Questions for Planner

None — the four authorized change items and all four required tests are complete, with results
matching or exceeding the stated acceptance criteria (T1 distinct per-cycle mu_final; T2 no
exception + snapshot written + no-abort unit test both pass; T3 bit-identical, 0.0 relative
difference; T4 tol-in-force established via non-text-literal evidence, with the residual gap
narrowed and reported honestly). Ready for the Planner's zero-solve check sign-off and the
subsequent G1 dispatch.
