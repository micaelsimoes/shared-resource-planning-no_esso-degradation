# Worker Report — P5.15 Step 3.6, Worker task W3: per-phase timing instrumentation (harness-side)

**Task**: implement per-phase, per-block cycle-time instrumentation WITHOUT editing production
code, plus its zero-solve checks; write (but do NOT run) the measurement entry point.
Authority: `PLANNER_BRIEF_2026-09-13.md` Addenda 18/19/20; design
`P5_15_STEP36_TIMING_DESIGN.md` (commit `2d9ebfc1`); prior audit
`WORKER_REPORT_S36_PARALLEL_AUDIT.md`. **No production file was edited. No solve was executed
by this Worker.**

---

## Task received

Implement, harness-side (monkeypatching, restored in `finally`, never a `timing_recorder=None`
kwarg threaded through production — Planner's explicit deviation from the design):

1. `p515_s36_step36_timing.py` — the recorder, an installer context manager, a sidecar writer,
   and an analysis function (design §5 table, `overhead_local`/`overhead_serial`, X=70%
   verdict, Amdahl 8-worker projection).
2. `p515_s36_step36_timing_run.py` — the (not-to-be-run) measurement entry point, reusing
   `p515_g_g1_g4_admm_gates.py` read-only.
3. `p515_s36_step36_timing_checks.py` — zero-solve checks (`SolveProfileGuard` armed, 0 solves).
4. This report.

Explicitly out of scope: editing `shared_resources_planning.py`, `network.py`,
`network_data.py`, `shared_energy_storage_data.py`, `admm_parameters.py`, the case file, or
`p515_g_g1_g4_admm_gates.py`; any solve; launching the measurement; the two "cheap wins"
(dropped per Addendum 20).

---

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` Addenda 18 (lines 770–795), 19 (798–825), 20 (828–856).
- `P5_15_STEP36_TIMING_DESIGN.md` (full file, commit `2d9ebfc1`) — §1 call-path map, §2 data
  model/instrumentation design, §5 measurement plan and decision rule.
- `WORKER_REPORT_S36_PARALLEL_AUDIT.md` (full file) — prior zero-solve audit, its 21.488 s
  median-overhead / 12.990 s median-IPOPT / 33.725 s median-wall table, its 8-worker LPT
  methodology (reused, cited, in `p515_s36_step36_timing.lpt_partition`), its
  `SolveProfileGuard`-across-process-boundary finding (composability note below).
- `p513_solve_profile_guard.py` (full file) — reused directly (`SolveProfileGuard`) and as the
  precedent for the frame-walk attribution technique (`_matching_frame`, generalized here to
  `_find_ancestor_frame`).
- `shared_resources_planning.py`: `_run_operational_planning` (:2310, loop :2501–2758+, DSO
  dispatch call :2511–2521, `update_and_check_convergence` calls :2524–2531/:2550–2557/
  :2568–2579, TSO dispatch call :2535–2544, `_update_tso_proximal_centres_after_solve` call
  :2547, ESSO dispatch call :2561–2565, `get_admm_residual_metrics`/`get_admm_boyd_residual_metrics`
  calls :2585–2586, `_update_admm_penalties` call :2748–2751); dispatcher/sequential/parallel DSO
  functions (:5317–5432, node-7 clone gating :5393/:5411); TSO update (:5130–5223, unconditional
  clone gating :5209/:5215); ESSO update (:5508–5545); `update_and_check_convergence` (:3082–3113);
  `get_admm_residual_metrics` (:5579), `get_admm_boyd_residual_metrics` (:5894);
  `_update_tso_proximal_centres_after_solve` (:4604); `_update_admm_penalties` (:6581).
- `network.py`: `Network.run_smopf` (:59–62), `_create_smopf_solver` (:540–601),
  `_run_smopf_solver_attempt` (:604–611, `solver.solve` at :608), `_run_smopf` (:673–777,
  `model.solutions.load_from` at :738).
- `network_data.py`: `NetworkData.optimize` (:54–68, `.clone()` at :61).
- `shared_energy_storage_data.py`: `SharedEnergyStorageData.optimize` (:67–95, `_optimize` call
  at :78–94); `_create_solver` (:1023–1132); `_run_solver_attempt` (:1135–1153, `solver.solve` at
  :1150); `_optimize` (:1177–1330s, `model.solutions.load_from` at :1296, diagnostics call at
  :1316–1318); `_get_esso_complementarity_diagnostics` (:1994–2028).
- `p515_g_g1_g4_admm_gates.py` (read-only; not edited): `run_admm_arm` (:1146–1370),
  `_construct_arm_planning` (:1058–1143), `_acquire_exclusive_run_lock` (:1388–1418), the
  `elif gate == 's35ref':` CLI branch (:5261–5333), `assert_s35ref_capture_paths` (:3253–3399),
  `s35ref_capture_hooks` (:3403+), `write_boyd_terminal_s35ref` (:3603+), `tee_stdout` (:237–242),
  module-level import list (:79–90, confirms `srp`/`SED`/`O`/`N`/`A`/`R`/`RH` aliases), `if
  __name__ == '__main__':` guard at :4932 (confirms importing the module as a library performs
  no solve — every gate-dispatch branch is below this line).
- `p56a_oracle.py`: `fresh_planning`/`WORK_DIR`/`_BASELINE` (:56–150).
- Installed Pyomo 6.9.5: `pyomo.opt.base.solvers.OptSolver.solve`;
  `pyomo.opt.solver.shellcmd.SystemCallSolver._execute_command`;
  `pyomo.core.base.block.BlockData.clone`; `pyomo.core.base.PyomoModel.ModelSolutions.load_from`
  — all four identity-checked live (see Check 2 below), not only read as source.

---

## Files modified

None among the protected list. **No production file was edited.**

## Files created

- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_s36_step36_timing.py`
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_s36_step36_timing_checks.py`
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_s36_step36_timing_run.py`
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515S36/step36_timing/zero_solve_checks/results.json`
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515S36/step36_timing/zero_solve_checks/manifest_sha256.json`
- This file.

`git status --porcelain` confirms `shared_resources_planning.py`, `network.py`,
`network_data.py`, `shared_energy_storage_data.py` are unmodified by this task.
(`admm_parameters.py` and `p515_g_g1_g4_admm_gates.py` show as modified in the working tree —
**pre-existing, from other concurrent Worker/Planner activity**, confirmed by `git diff --stat`
before this task began any edit; this Worker never opened either with `Edit`/`Write`.)

---

## Changes made

### 1. `p515_s36_step36_timing.py` — the recorder

**Data model**: `PhaseTimingRecorder.records`, a list of
`{seq, cycle, agent, block, phase, attempt, start_perf, end_perf, elapsed_s}` dicts, appended
under a lock from `try/finally` blocks only (a recording call never changes what exception, if
any, propagates). `PHASES = ('param_update', 'clone', 'solve_bundle', 'load_solution',
'bookkeeping', 'diagnostics_parse', 'admm_global', 'unattributed')` — the eight categories the
task asks for; two more internal-only tags (`stage_total`, `block_total`) exist purely as
derivation inputs, never reported as a top-level category.

**Wrap-point table** (every target verified live — see Checks 1/2 below):

| Wrap point | File:line (real caller) | Resolution mechanism | Attribution source |
|---|---|---|---|
| `update_distribution_coordination_models_and_solve` | `shared_resources_planning.py:2511` calls `:5317` | module attribute, unqualified `LOAD_GLOBAL` in `_run_operational_planning`'s own bytecode (Check 1) | its own `cycle` kwarg (`inspect.signature(...).bind`) |
| `update_transmission_coordination_model_and_solve` | `:2535` calls `:5130` | same | its own `cycle` kwarg |
| `update_shared_energy_storages_coordination_model_and_solve` | `:2561` calls `:5508` | same | its own `cycle` kwarg |
| `update_and_check_convergence` | `:2524`/`:2550`/`:2568` call `:3082` | same | recorder's `current_cycle` (no `cycle` param exists) |
| `get_admm_residual_metrics` | `:2585` calls `:5579` | same | recorder's `current_cycle` |
| `get_admm_boyd_residual_metrics` | `:2586` calls `:5894` | same | recorder's `current_cycle` |
| `_update_tso_proximal_centres_after_solve` | `:2547` calls `:4604` | same | its own `cycle=` kwarg |
| `_update_admm_penalties` | `:2748` calls `:6581` | same | its own `iter=` kwarg (different name, same value) |
| `Network.run_smopf` | `network_data.py:62` calls `network.py:59` | class attribute; instance-method dispatch (`type(instance).run_smopf`) — verified live via a real, zero-solve `Network()` instantiation (Check 2) | `self.name`/`.year`/`.day`/`.is_transmission` (bound `self` argument — no frame walk needed) |
| `SharedEnergyStorageData._optimize` (module fn) | `shared_energy_storage_data.py:78` calls `:1177` | module attribute, `LOAD_GLOBAL` in `SharedEnergyStorageData.optimize`'s bytecode (Check 1) | its own `node_id=`/`cycle=` kwargs |
| `BlockData.clone` | `network_data.py:61` | class attribute; verified live via a real `pe.ConcreteModel()` (Check 2) — **timed only when the immediate caller frame is `network_data.py`+`optimize`** (i.e. exactly the TSO-unconditional / DSO-node-7 clone the design's §1.1–1.2 identifies); every other `.clone()` call anywhere in the process is passed through **untimed** | caller frame's `self`/`year`/`day` locals |
| `OptSolver.solve` | `network.py:608` / `shared_energy_storage_data.py:1150` | class attribute; verified live via a real `SolverFactory('ipopt', ...)` instance (Check 2) — **same identity precedent `SolveProfileGuard` already relies on** | frame walk to `_run_smopf_solver_attempt` (network.py) or `_run_solver_attempt` (ESSO); `network`/`node_id` + `log_suffix` locals in that frame |
| `ModelSolutions.load_from` | `network.py:738` / `shared_energy_storage_data.py:1296` | class attribute; verified live via a real `ConcreteModel()` (Check 2) | frame walk to `_run_smopf` (network.py) or `_optimize` (ESSO); `network`/`node_id` + `recovery_attempted`/`tier2_attempted` locals |
| `_get_esso_complementarity_diagnostics` | `shared_energy_storage_data.py:1316` calls `:1994` | module attribute, `LOAD_GLOBAL` in `_optimize`'s bytecode (Check 1) | direct `node_id` arg; `cycle` from the immediate caller frame's local |

**Derivations, not live wraps** (per the task's own instruction: "param_update per block/stage =
stage total minus the enclosed optimize time"):

- **`param_update`** — `derive_param_update_and_bookkeeping`. **DSO is per-node** (the
  set_value loop and `distribution_network.optimize(...)` are *interleaved* per node,
  `shared_resources_planning.py:5329–5420`): computed from the gap between successive
  `Network.run_smopf` "block_total" events grouped by `Network.name`, and the DSO stage-wrap's
  own entry for the first node. **TSO and ESSO are stage-level only** — both do the ENTIRE
  set_value loop for every block *before* calling `.optimize()`/`shared_ess_data.optimize()`
  once (not interleaved), so their param_update is `stage_total − Σ(block_total)`, one number
  per cycle, not resolvable per block without editing production. This asymmetry is a **finding**
  (unrequested, discovered while tracing the exact interleaving), not a workaround.
- **`bookkeeping`** — per `(cycle, agent, block)`: `block_total − Σ(clone) − Σ(solve_bundle,
  all attempts) − Σ(load_solution, all attempts) − Σ(diagnostics_parse)`. Left unclamped if
  negative (would be a correctness signal, not expected given the source's structural nesting).
- **`unattributed`** — `analyze_phase_timing`'s per-cycle residual between production's own
  `"[INFO] \t - Iteration {iter}: {X:.2f} s"` print (`shared_resources_planning.py:3007`,
  parsed from captured stdout by the caller) and this recorder's own reconstructed per-cycle
  span (first event start to last event end). Captures whatever precedes the first wrap
  (nothing, in practice — DSO fires right after `iter_start`) and whatever follows the last
  (the worst-primal/worst-pf diagnostic prints and the iteration-time print itself).

**Analysis function** (`analyze_phase_timing`) — every formula reproduced from design §5 in its
own docstring: per-phase-by-agent median/mean/max/sum **per sampled cycle individually** (design
§5: "report both cycles individually rather than a median, since 2 points do not support a
robust median") plus an aggregate; `overhead_local_total`/`overhead_serial_total`;
`verdict_ratio = overhead_local/(overhead_local+overhead_serial)` against `x_threshold` (default
0.70, Addendum 20's interim X); `lpt_partition` (reused LPT-greedy heuristic, generalized to `W`
workers, cited from `WORKER_REPORT_S36_PARALLEL_AUDIT.md`'s own methodology); Amdahl
`f_serial`/`speedup_at_{W}_workers`.

**One documented approximation**: the NL-write (b) share of `solve_bundle` (b+c+d1 combined) is
**not separable** from the manual `perf_counter()` wrap alone (design §2.5) — it requires
Pyomo's own `report_timing=True` print output, which `recorder_installed(...,
inject_report_timing=True)` can inject into the wrapped `solver.solve(...)` call for exactly
the one designated cross-check run (design §5). When that data is absent, `analyze_phase_timing`
reports the **full** `solve_bundle` as an explicit **upper bound** on the NL-write share
(`nl_write_share_is_upper_bound: True` in its output), never silently substituting a guessed
number.

### 2. `p515_s36_step36_timing_run.py` — measurement entry point (NOT executed)

Reuses `p515_g_g1_g4_admm_gates.py` (imported as `G`, read-only) function-for-function: the
same sequence the committed `elif gate == 's35ref':` branch runs (`G.O.fresh_planning` preflight
→ `G.assert_s35ref_capture_paths` → `G.s35ref_capture_hooks` → `G.run_admm_arm(...,
apply_rho=False, full_diagnostics_in_rows=True, k_override=None, investment_map=None,
post_run_hook=...)` → `G.write_boyd_terminal_s35ref`), with the **only** deliberate differences:
`num_max_iters_override=2` (design §5's two-cycle preflight, not `G.S35REF_CAP`=500) and fresh
`eval_id`/`out_dir` pairs (`p515s36_timing_off`/`_on`, `data/SRP1/Results/P515S36/step36_timing/
{off,on}/`) so nothing under the committed `P515S35_REF_run` is ever touched.

- **Lock**: calls `G._acquire_exclusive_run_lock()` verbatim (same `.p515_g_gate.lock`,
  `O_CREAT|O_EXCL` semantics, same failure message) — reused, not reimplemented.
- **Precondition checks** (`_check_preconditions`, run before the lock is even requested):
  lock absent; no other `p515_g_g1_g4_admm_gates.py` process alive (`ps aux` scan, this
  script's own PID excluded); neither output directory exists yet (write-once); the four
  production files this instrumentation reads are clean in git
  (`git status --porcelain -- shared_resources_planning.py network.py network_data.py
  shared_energy_storage_data.py`).
- **Both streams captured**: `G.tee_stdout` (reused, unedited) covers stdout per-run;
  `tee_stderr` (new, local to this file — `G` has no stderr-tee) covers stderr for the
  **whole** script. No `screen`/`nohup`/backgrounding.
- **Composability with `SolveProfileGuard`**: `T.recorder_installed(...)` is opened **outside**
  the `G.run_admm_arm(...)` call, so `run_admm_arm`'s own internal
  `SolveProfileGuard(...).install()`/`.uninstall()` nests **inside** it (LIFO) — each of our
  wrappers captures "whatever is currently at the attribute" as `original` at install time and
  restores exactly that at uninstall time, so composing with a guard installed afterward, on
  top, is safe by construction (verified structurally; not itself part of the zero-solve checks,
  since the checks script does not invoke `run_admm_arm`).
- **Bitwise diff** (`_bitwise_diff`): `cycle_trajectory` (every `A.cycle_row` field plus every
  raw `admm_diagnostics` field, since `full_diagnostics_in_rows=True` — Boyd residuals/ratios,
  rho/gamma before/after/action, recourse, EFC/day), `esso_complementarity_diagnostics_by_round`
  (the detector), and the SoH floor-multiplier sidecar (`s35ref_capture_hooks`' own per-run
  JSONL, read and compared line-by-line — the SoH trajectory).
- **Exact launch command and expected wall time**: in the file's module docstring (reproduced
  below).

### 3. `p515_s36_step36_timing_checks.py` — zero-solve checks (executed)

`SolveProfileGuard(permitted=())` armed for the **whole** script; `verify(expected_solves=0,
expected_execs=0)` at the end. **34/34 checks passed** (full detail:
`data/SRP1/Results/P515S36/step36_timing/zero_solve_checks/results.json`).

- **Check 1** (resolution — module-level): `dis`-scans `_run_operational_planning`'s own
  bytecode and confirms all 8 `shared_resources_planning`-level wrap names appear as
  `LOAD_GLOBAL` argvals; scans `SharedEnergyStorageData.optimize` for `LOAD_GLOBAL '_optimize'`
  and `_optimize` for `LOAD_GLOBAL '_get_esso_complementarity_diagnostics'`.
- **Check 2** (resolution — class-level): real, zero-solve instantiation of `Network()`,
  `pe.ConcreteModel()`, and `po.SolverFactory('ipopt', executable='/usr/local/bin/ipopt')`
  (canonical IPOPT path, per `CLAUDE.local.md` — **never invoked**, only used to construct a
  real solver plugin object so its MRO can be inspected); confirms `type(instance).<method> is
  <OwningClass>.<method>` for `run_smopf`, `clone`, `solutions.load_from`, `solve`,
  `_execute_command` (the last two reusing the exact identity precedent
  `p513_solve_profile_guard.py` already relies on).
- **Check 3** (install/uninstall identity): snapshots all 14 wrap-target attributes before,
  during, and after a `T.recorder_installed(...)` block — every target differs while installed,
  every target is restored to **exact** (`is`) identity afterward.
- **Check 4** (behaviour preservation, trivial stand-ins): for every wrap point, a stand-in
  (never a real Pyomo solve/clone/parse) is installed **before** `recorder_installed`, so the
  recorder's own wrapper's `original` is the stand-in; verifies return-value passthrough,
  exception-type-and-message passthrough (a distinct `_BoomError`), and that exactly one record
  is appended in both the success and the exception case. The six frame-walk-dependent wrap
  points (`solve_bundle` ×2 ancestries, `load_solution` ×1, `diagnostics_parse` ×1, `clone` ×1,
  plus `clone`'s negative control) are exercised against **synthesized ancestor frames** —
  functions `compile()`d with `co_filename`/`co_name` set to the exact real call-site (file,
  function) pair and the exact real local-variable names, **without executing any real
  production code** — proving the frame-walk attribution logic in isolation. `clone`'s negative
  control confirms a `.clone()` call with no matching ancestor is passed through **untimed** (no
  record appended at all), matching the module docstring's stated scope.
- **Check 5** (analysis correctness): a 6-record hand-computed toy example (two DSO nodes, one
  cycle) — `derive_param_update_and_bookkeeping`'s `param_update`/`bookkeeping` outputs and
  `analyze_phase_timing`'s `overhead_local_total`/`overhead_serial_total`/`verdict_pass`/
  `unattributed` all match the hand computation exactly (see the script's inline arithmetic).

---

## Commands / experiments run

- `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s36_step36_timing_checks.py`
  — exit clean, `34/34 checks passed`, `SolveProfileGuard verified: 0 permitted solves, 0
  blocked solves`.
- Read-only, zero-solve sanity import: `SolveProfileGuard(permitted=()).install()` around
  `import p515_g_g1_g4_admm_gates as G` — confirmed the import performs 0 solves and every
  function this entry point reuses (`run_admm_arm`, `_acquire_exclusive_run_lock`,
  `assert_s35ref_capture_paths`, `s35ref_capture_hooks`, `write_boyd_terminal_s35ref`,
  `O.WORK_DIR`) resolves and is callable/present.
- `ast.parse(...)` on all three new `.py` files — syntax-checked, never executed as
  `__main__` (in particular, `p515_s36_step36_timing_run.py` was **never run**, not even its
  `main()` fail-safe stub, per the task's explicit "DO NOT RUN IT").
- `git status --porcelain -- shared_resources_planning.py network.py network_data.py
  shared_energy_storage_data.py p515_s36_step36_timing.py p515_s36_step36_timing_checks.py
  p515_s36_step36_timing_run.py data/SRP1/Results/P515S36/step36_timing` — confirmed the four
  production files are unmodified.
- `git diff --stat -- admm_parameters.py p515_g_g1_g4_admm_gates.py` — confirmed their observed
  working-tree modifications pre-date and are unrelated to this task (17 lines / 710 lines,
  consistent with "another Worker is editing p515_g_g1_g4_admm_gates.py now").

---

## Results

- 34/34 zero-solve checks pass. Full detail:
  `data/SRP1/Results/P515S36/step36_timing/zero_solve_checks/results.json` (sha256-manifested:
  `manifest_sha256.json` in the same directory).
- Every wrap target's resolution mechanism is verified live (not only cited from source):
  module-level via `dis`-scanned `LOAD_GLOBAL`, class-level via real (zero-solve) instantiation
  and MRO identity check.
- Install/uninstall is exact-identity reversible for all 14 wrap targets.
- Every wrapper preserves return values and exceptions, including under the frame-walk
  attribution logic, verified against synthesized ancestor frames without executing any
  production code.
- `analyze_phase_timing`/`derive_param_update_and_bookkeeping` reproduce a hand-computed toy
  example exactly.

---

## Validation

Distinguishing what was actually established from what remains to be measured:

- **Code executes correctly**: yes — 34/34 zero-solve checks pass, including live identity
  checks against real (never-solved) Pyomo/production objects.
- **Test passes**: yes — `p515_s36_step36_timing_checks.py` exits 0.
- **Requested diagnostic works**: the recorder, installer, and analysis function are
  demonstrated correct on synthetic data and on real-but-unsolved production objects; whether
  they behave identically when wrapped around an ACTUAL 2-cycle ADMM run (the bitwise-identity
  gate `p515_s36_step36_timing_run.py` is built to perform) is **not yet established** — that
  requires running the measurement, which this task explicitly forbids.
- **Underlying numerical/performance question (is `overhead_local/overhead_total ≥ 70%`?)**:
  **not addressed by this task** — no measurement was taken. The design's own X=70% derivation
  (design §5) used the PRIOR audit's 477-cycle numbers as a screening threshold; this task
  produces the INSTRUMENT to measure it fresh on a cold 2-cycle run at the s35ref configuration
  class, not the measurement itself.

---

## Unexpected findings

- **DSO's `param_update` is resolvable per-node; TSO's and ESSO's are not** — a structural
  asymmetry in how the three stage-update functions interleave their `set_value` loops with
  their `.optimize()` calls (DSO: per-node, interleaved; TSO/ESSO: one shot for the whole
  stage, not interleaved). Not mentioned in the design; found while implementing the
  gap-based derivation.
- Confirmed, mechanically (via `dis`), every claim the design made by source-reading about
  which calls are unqualified module-level globals — no drift found despite the design's own
  disclaimer that `admm_parameters.py`/`shared_resources_planning.py` "carry uncommitted
  modifications from other in-progress work"; the specific line ranges this task's wrap points
  depend on were re-checked directly against the current on-disk state and matched the design's
  citations exactly (no line-number drift observed in the four files this instrumentation
  touches).

---

## Remaining issues

- The measurement itself (two-cycle OFF vs ON, bitwise diff, phase-timing table, X=70%
  verdict, 8-worker Amdahl projection) has **not** been run — by task design, the Planner
  schedules it.
- `nl_write_share_is_upper_bound` will be `True` unless the measurement run's
  `inject_report_timing=True` cross-check successfully parses at least one "seconds required to
  write file" line from Pyomo's own `report_timing` output; `p515_s36_step36_timing_run.py`
  parses this via a regex against captured stdout (`parse_report_timing_nl_write_seconds`),
  not independently verified against a real run (would require executing it).
- `analyze_phase_timing`'s Amdahl LPT partition, when `nl_write_seconds` is present, divides the
  aggregate NL-write time evenly across an agent's per-block `solve_bundle` records
  (`sum(nl_write_seconds.values()) / max(len(durations), 1)`) rather than a true per-block
  NL-write figure, because `report_timing`'s print output is not tagged with a (cycle, agent,
  block) key (design §2.5, noted there too) — documented as an approximation in the function's
  own docstring, not silently assumed exact.

---

## Exact launch command (for the Planner)

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \
    p515_s36_step36_timing_run.py \
    > data/SRP1/Results/P515S36_STEP36_TIMING_launch.log 2>&1
```

Preconditions the script itself checks before writing anything (`_check_preconditions`, all
must pass or it exits 1 before acquiring the lock):

1. `.p515_g_gate.lock` does not already exist.
2. No other `p515_g_g1_g4_admm_gates.py` process is alive (`ps aux` scan).
3. Neither `data/SRP1/Results/P515S36/step36_timing/off/` nor `.../on/` exists yet.
4. `git status --porcelain -- shared_resources_planning.py network.py network_data.py
   shared_energy_storage_data.py` is empty.

Note: at the time of this report, precondition 4 would currently **pass** (those four files are
clean); `admm_parameters.py`/`p515_g_g1_g4_admm_gates.py` are dirty from other concurrent work
but are **not** in the checked list (this instrumentation's wrap points never read
`admm_parameters.py` directly, and `p515_g_g1_g4_admm_gates.py` is explicitly reused read-only,
not a wrap target) — the Planner may want to widen the precondition list before authorizing the
run if a stricter "whole tree clean" bar is wanted; not done here since it was not asked for and
would make the check fail on unrelated, already-authorized concurrent work.

**Safety note**: `p515_s36_step36_timing_run.py`'s `main()` (the `if __name__ == '__main__':`
entry point) is a deliberate fail-safe — it raises `SystemExit` unconditionally and does
**not** run the measurement; the real logic is in `main_()` (trailing underscore), which the
Planner must invoke explicitly (edit the guard, or call `main_()` from a fresh process) — an
auditable, non-accidental step, so `python p515_s36_step36_timing_run.py` alone can never
launch the measurement by mistake.

**Expected wall time**: approximately 1–2 minutes for the two 2-cycle ADMM runs themselves
(`WORKER_REPORT_S36_PARALLEL_AUDIT.md` median cycle wall 33.7 s, cold cycle 1 alone 30.3 s,
×2 cycles ×2 runs), plus each run's own model-construction/initialization
(tens of seconds, `P515S35_REF_run` evidence) — a few minutes end to end, not the ~4–5 hour
scale of a capped-500 certification run.

---

## Questions for Planner

1. Is the precondition file list for "production files clean in git" (the four files this
   instrumentation's wrap points read) the right scope, or should it be widened to the whole
   tree (which would currently fail on `admm_parameters.py`/`p515_g_g1_g4_admm_gates.py`'s
   unrelated, already-authorized concurrent edits)?
2. `nl_write_share_is_upper_bound` degrades gracefully (full `solve_bundle` reported as the
   upper bound) if `report_timing`'s stdout parse finds nothing — is a hard failure preferred
   instead, so a silently-degraded measurement can never be mistaken for the cross-checked one?
3. This task's `S = 3` / `X = 70%` screening threshold is Addendum 20's *interim* figure
   ("the measurement sets the real one") — should `analyze_phase_timing`'s `x_threshold`
   default be changed once the Planner has a firmer bar, or is passing it explicitly at call
   time (as `p515_s36_step36_timing_run.py` already does) sufficient?

---

## Addendum: follow-up Worker task (2026-09-17) — Planner decisions on Questions 1–3, plus Q4 (main()/main_() split)

**Context**: this addendum was implemented by a follow-up Worker while a 300-cycle numerical
arm (`s38_A_tau0`, PID confirmed alive throughout via `ps aux`) was running in the same working
tree. No solve was executed by this addendum; `p515_g_g1_g4_admm_gates.py`, every production
file, the case file, and everything under `data/SRP1/Results/P515S38*` were untouched (not
opened with `Edit`/`Write`).

### Task received

The Planner's decisions on the three prior Worker's questions, plus a fourth item:

1. Widen the precondition clean-file check in `p515_s36_step36_timing_run.py` to also cover
   `admm_parameters.py`, `p515_g_g1_g4_admm_gates.py`, `data/SRP1/SRP1_params.json`, and the
   three `p515_s36_step36_timing*.py` files (all committed) — plus refuse if any process
   matching `p515_g_g1_g4_admm_gates.py` or `p515_s38_` is alive, or `.p515_g_gate.lock` exists
   (existing checks kept).
2. Degraded mode (no NL-write sub-times from the `report_timing` cross-check) becomes a hard
   non-verdict: `analyze_phase_timing` still writes every measured table, but sets
   `verdict_pass` to the literal string `'INDETERMINATE (NL-write share not separated)'` —
   never `True`/`False` — and the entry point exits non-zero after writing everything.
3. `analyze_phase_timing`'s `x_threshold` gets no default (required argument); the entry
   point passes `0.70` explicitly, once, as a module constant.
4. Replace the `main()`/`main_()` fail-safe split with a single `main()` under
   `if __name__ == '__main__':` — the precondition checks are now the sole safety mechanism.
   Update the launch command in this report if it changed (it did not).
5. Re-run `p515_s36_step36_timing_checks.py` (zero solves, `SolveProfileGuard` armed) into a
   **new** output directory (`zero_solve_checks_v2/`, never overwriting the committed
   `zero_solve_checks/`), adding checks for items 2 and 3 on the toy example, plus a check
   that item 1's precondition function refuses when a listed file is dirty (simulated via a
   stub of the git-status subprocess call, never by dirtying a real file).

### Files inspected

`CLAUDE.md`; this report (as it stood before this addendum); `p515_s36_step36_timing.py`;
`p515_s36_step36_timing_run.py`; `p515_s36_step36_timing_checks.py`; `p513_solve_profile_guard.py`
(`SolveProfileGuard` API, reused unmodified). `git status --porcelain` / `ps aux` / the
`.p515_g_gate.lock` file, to confirm the live numerical arm's state before and after every
change.

### Files modified

- `p515_s36_step36_timing.py` — `analyze_phase_timing`: `x_threshold` moved to a required
  (no-default) parameter, positioned right after `records`; added module-level
  `_DEGRADED_VERDICT = 'INDETERMINATE (NL-write share not separated)'`; `verdict_pass` is now
  that string whenever `nl_write_share_is_upper_bound` is `True`, otherwise unchanged
  (`bool`). `verdict_ratio` is still always computed and reported, even in degraded mode —
  only `verdict_pass` is downgraded. Docstring updated in place.
- `p515_s36_step36_timing_run.py` — `_PRODUCTION_FILES_TO_CHECK_CLEAN` widened to 10 entries
  (the original 4 production files, plus `admm_parameters.py`,
  `p515_g_g1_g4_admm_gates.py`, `data/SRP1/SRP1_params.json`, and the three
  `p515_s36_step36_timing*.py` files); new `_FORBIDDEN_LIVE_PROCESS_SUBSTRINGS = ('p515_g_g1_g4_admm_gates.py', 'p515_s38_')`
  used by the `ps aux` scan (was hardcoded to the single harness-name substring before); new
  module constant `X_THRESHOLD = 0.70`, passed explicitly to `analyze_phase_timing`; new
  `verdict_is_indeterminate(analysis)` helper (module-level, unit-testable without a run);
  `main()` now performs the real measurement directly (the previous `main()` fail-safe stub
  and the `main_()` real-logic function are merged into one `main()`); after writing every
  artifact (recorder JSONL, bitwise-diff JSON, analysis JSON, sha256 manifest), `main()` calls
  `sys.exit(1)` if `verdict_is_indeterminate(analysis)`. Module docstring's precondition list
  (item d) and the "1. Precondition checks" section updated to describe the widened file list
  and process substrings; the "EXACT LAUNCH COMMAND" section is **unchanged** (still the
  single command already given in the base report — the merge into one `main()` does not
  change how the script is invoked).
- `p515_s36_step36_timing_checks.py` — `OUT_DIR` now reads
  `os.environ.get('P515_S36_CHECKS_OUT_DIR', <original default path>)`, so the default
  (unset-env-var) behaviour is byte-identical to before and still write-once-protected
  against the committed `zero_solve_checks/`; added `import p515_s36_step36_timing_run as R`.
  Existing Check 5's toy-example assertion `C5_verdict_pass_at_x_0p70` (expected `True`)
  replaced by two assertions matching the new degraded-mode behaviour: the ratio itself still
  clears 0.70 (`C5_verdict_ratio_clears_x_0p70_bar`), but `verdict_pass` is now the
  indeterminate string, not `True` (`C5_verdict_pass_is_degraded_nonverdict_not_bool_true`) —
  the toy example's `nl_write_seconds` was never supplied, so it was always in degraded mode;
  only the check's expectation was stale. Added Check 6 (item 3: `x_threshold` has no default,
  `TypeError` when omitted, `R.X_THRESHOLD == 0.70`), Check 7 (item 2: the same toy records
  analyzed once with `nl_write_seconds=None` — degraded, string verdict — and once with a
  non-empty `nl_write_seconds` — determined, bool verdict — plus direct unit tests of
  `R.verdict_is_indeterminate` on synthetic dicts), and Check 8 (item 1: `R._check_preconditions()`
  called with `R.subprocess.run` monkeypatched so a `['git', 'status', ...]` call returns a
  fabricated ` M admm_parameters.py` line — never touching a real file — verifying the
  returned failure list contains a "production files are not clean in git" entry naming
  `admm_parameters.py`; `subprocess.run` identity is confirmed restored afterward).

### Commands / experiments run

1. `python3 -c "import ast; ast.parse(...)"` on all three files after each edit — syntax
   checked before any execution.
2. `git status --porcelain -- shared_resources_planning.py network.py network_data.py
   shared_energy_storage_data.py admm_parameters.py p515_g_g1_g4_admm_gates.py
   data/SRP1/SRP1_params.json p515_s36_step36_timing.py p515_s36_step36_timing_run.py
   p515_s36_step36_timing_checks.py` — empty (all ten clean) before any edit, confirming the
   starting state matched what the widened precondition list will require of a real run.
3. `ps aux | grep -i "p515_g_g1_g4_admm_gates.py\|p515_s38_"` and `ls -la .p515_g_gate.lock` —
   run before, during, and after this addendum's edits and the checks re-run: PID 65458
   (`p515_g_g1_g4_admm_gates.py s38_a_tau0`) alive throughout, `.p515_g_gate.lock` present
   throughout — confirmed the numerical arm was never touched.
4. `git add -- p515_s36_step36_timing.py p515_s36_step36_timing_run.py
   p515_s36_step36_timing_checks.py` then
   `git commit --only -m "..." -- <same three files>` — commit `e22c6975` (script changes,
   before the checks re-run, per the Planner's ordering instruction).
5. `P515_S36_CHECKS_OUT_DIR=data/SRP1/Results/P515S36/step36_timing/zero_solve_checks_v2 \
   /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s36_step36_timing_checks.py`
   — the exact re-run command (env-var override, no other change).
6. `ls -la data/SRP1/Results/P515S36/step36_timing/zero_solve_checks/
   data/SRP1/Results/P515S36/step36_timing/zero_solve_checks_v2/` — confirmed the original
   directory's file mtimes (14:04) predate this addendum's run (14:57) and are unchanged; the
   v2 directory is new.
7. `shasum -a 256 data/SRP1/Results/P515S36/step36_timing/zero_solve_checks_v2/results.json`
   — matched the `manifest_sha256.json` the script itself wrote.

### Results

- **48/48 checks passed** in the re-run (34 original + 14 new: 3 in Check 6, 8 in Check 7, 2
  in Check 8, plus one extra ratio-check split out of the old Check 5). `FAILURES: []`.
  Evidence: `data/SRP1/Results/P515S36/step36_timing/zero_solve_checks_v2/results.json`
  (sha256 `e8f3180d0fc074be45517f461cbc739866806ebd37262d7818f56268b8205f1d`, matching its own
  `manifest_sha256.json`).
- `SolveProfileGuard(permitted=())` verified `0` permitted solves, `0` blocked solves for the
  whole re-run (same guard discipline as the original run).
- Check 8's real (unstubbed) `ps aux`/lock-file half of `_check_preconditions()` independently
  surfaced the **live** `s38_a_tau0` process and the live `.p515_g_gate.lock` as real failures
  (visible in the check's own recorded detail, alongside the simulated dirty-file failure) —
  incidental confirmation, from real process state, that the widened precondition function
  behaves correctly under the exact conditions this task required it not to disturb.

### Validation

- **Code executes correctly**: yes — all three files parse and the checks script runs clean.
- **Test passes**: yes — 48/48, `SolveProfileGuard` verified at 0 solves.
- **Requested diagnostic works**: the three Planner decisions (widened preconditions, hard
  non-verdict, required `x_threshold`) and the `main()`/`main_()` merge are implemented and
  unit-tested on synthetic/toy data; whether they behave identically inside an actual 2-cycle
  ADMM run is still **not established** — this addendum, like the base task, never ran
  `p515_s36_step36_timing_run.py`'s `main()`.
- **Underlying numerical/performance question**: still not addressed — no measurement was
  taken by this addendum either.

### Unexpected findings

None beyond what Check 8 incidentally confirmed (above) — no drift, no surprising interaction
with the live numerical arm.

### Remaining issues

Same as the base report's "Remaining issues" — the measurement itself has still not been run.

### Questions for Planner

None — all four items were fully specified by the Planner's decisions; no ambiguity was
encountered during implementation.

### Commit hashes

- `e22c6975` — script changes (`p515_s36_step36_timing.py`, `p515_s36_step36_timing_run.py`,
  `p515_s36_step36_timing_checks.py`), committed **before** the checks re-run.
- (next commit, this report + `zero_solve_checks_v2/` outputs + their sha256 manifest) — see
  the commit that follows this addendum in the repository history.

### Exact commands (for reproducibility)

```
# Re-run the (updated) zero-solve checks into a NEW directory, never touching the
# committed zero_solve_checks/:
P515_S36_CHECKS_OUT_DIR=data/SRP1/Results/P515S36/step36_timing/zero_solve_checks_v2 \
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s36_step36_timing_checks.py
```

The measurement entry point's own launch command is **unchanged** from the base report (single
`main()` now, but same invocation):

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \
    p515_s36_step36_timing_run.py \
    > data/SRP1/Results/P515S36_STEP36_TIMING_launch.log 2>&1
```
