# P5.15 Step 3.6 — Per-phase timing instrumentation design, and the two cheap wins

**Task type**: bounded READ-ONLY design. `PLANNER_BRIEF_2026-09-13.md` Addendum 19, Step 3.6
bullet ("first, per-phase timing per block... Cheap wins to test: the Pyomo NL writer v2, and
no symbolic labels in the `.sol` round trip. Preflight and gate after the ρ_ess arms."), and
Addendum 18 (persistent-worker design this instrumentation is meant to size). No `.py` file, no
case file was edited to produce this document. No solve was executed — every claim below is
either a citation of source (production or the installed Pyomo package) or a citation of
`WORKER_REPORT_S36_PARALLEL_AUDIT.md` (commit `710af0de`), the prior zero-solve audit this
document extends.

**Interpreter**: `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`. **Pyomo
version installed there**: **6.9.5** (`import pyomo; pyomo.__version__`, read-only check, no
solve). All Pyomo file:line citations below are against
`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/lib/python3.11/site-packages/pyomo/...`.
Production repository file:line citations are against the working tree as read on 2026-09-17
(two files, `admm_parameters.py` and `shared_resources_planning.py`, carry uncommitted
modifications from other in-progress work at the time of reading; the line numbers below are
current against that on-disk state, which is what this document's citations must match, and were
not caused by this task).

---

## 1. Call-path map for one network block solve

Three agent families share one core solve primitive (`solver.solve(model, ...,
load_solutions=False)` followed by an explicit `model.solutions.load_from(result)`), reached
through three different per-cycle update functions. All three are driven from one ADMM loop.

### 1.0 Entry point

`_run_operational_planning` (`shared_resources_planning.py:2310`) contains the ADMM main loop
(`for iter in range(1, ...)`, loop body `shared_resources_planning.py:2501-3007`). Per cycle, in
strict order (each stage's consensus/dual update must complete before the next stage's parameter
update reads it):

1. DSO solve — `update_distribution_coordination_models_and_solve(...)` (`:2511-2521`), which
   dispatches to `_sequential` (`:5318`) unless `planning_problem.parallel_execution` (`:5312-5315`).
2. `update_and_check_convergence(..., update_flags={"update_dns": True, ...})` (`:2524-2531`).
3. TSO solve — `update_transmission_coordination_model_and_solve(...)` (`:2535-2544`).
4. `_update_tso_proximal_centres_after_solve(...)` (`:2547`) and
   `update_and_check_convergence(..., update_flags={"update_tn": True, ...})` (`:2550-2557`).
5. ESSO solve — `update_shared_energy_storages_coordination_model_and_solve(...)` (`:2561-2565`).
6. `update_and_check_convergence(..., update_flags={"update_sess": True})` (`:2568-2579`).
7. `get_admm_residual_metrics(...)` (`:2585`) and `get_admm_boyd_residual_metrics(...)` (`:2586`).
8. `iter_end = time.time(); print(f"[INFO] \t - Iteration {iter}: {iter_end - iter_start:.2f} s")`
   (`:3006-3007`) — the **only** per-cycle timing signal currently in production, wrapping steps
   1-7 as a single undifferentiated number. `iter_start = time.time()` is at `:2507`.

No `cycle_callback` / heartbeat hook exists inside this loop (grep on
`shared_resources_planning.py` for `cycle_callback`/`heartbeat` is empty — confirms
`WORKER_REPORT_S36_PARALLEL_AUDIT.md`'s finding, re-checked here). Harness-side capture (e.g. the
heartbeat file written by the campaign harness, `p515_g_g1_g4_admm_gates.py`, the only production
script that greps for `heartbeat`) happens in the driver process **around** the call into this
loop, not inside it — it is unaffected by anything proposed here.

### 1.1 DSO — `update_distribution_coordination_models_and_solve_sequential`

`shared_resources_planning.py:5318-5426`.

- **(a) parameter/model update**: `:5323-5361`, one `for node_id` loop, per `(year, day, period)`
  `set_value(...)` calls on `shared_es_s_rated_fixed`, `shared_es_e_rated_fixed`,
  `dual_vmag_req`, `vmag_req`, `dual_pf_p_req`/`q_req`, `p_pf_req`/`q_pf_req`, `dual_ess_p_req`/
  `q_req`, `p_ess_req`/`q_ess_req`, and (conditionally) the `*_prev` consensus copies, plus
  `configure_shared_ess_operational_state(...)` (`:5340`).
- Two FrozenSMOPF snapshot closures are defined but only *wired* conditionally: the failure
  callback only for `node_id == 7` (`:5365-5387`), the success comparator only for `node_id == 7
  and cycle == 7` (`:5389-5405`) — so the extra `model.clone()` cost noted below (§1.2) is **not**
  paid by the DSO on every cycle, only at node 7.
- **Solve dispatch**: `distribution_network.optimize(model, from_warm_start=..., 
  failure_snapshot_callback=snapshot_callback, pre_solve_snapshot_callback=success_snapshot_callback)`
  (`:5409-5414`) → `NetworkData.optimize` (`network_data.py:54-68`): loops `year`/`day`;
  `pre_solve_model = model[year][day].clone()` (`network_data.py:61`) **only if** either callback
  is non-`None` for that node; then `self.network[year][day].run_smopf(model[year][day], ...)`
  (`network_data.py:63`) → `Network.run_smopf` (`network.py:59-61`) →
  `_run_smopf(network, model, params, from_warm_start)` (`network.py:673-777`).
- Inside `_run_smopf` (shared by DSO and TSO — this is the single core solve primitive):
  - `_run_smopf_solver_attempt` (`network.py:604-611`) calls `_create_smopf_solver` (`:540-601`,
    builds solver options `:545-558`, resolves the IPOPT `output_file` log path against
    `network.logs_dir` `:560-576`, applies the warm-start suffix replace and
    `warm_start_init_point='yes'` `:580-599` when `from_warm_start`), then
    **`result = solver.solve(model, tee=params.solver_params.verbose, load_solutions=False)`**
    (`:608`) — this single call is where Pyomo internally performs **(b) NL write**, **(c) the
    IPOPT subprocess**, and **(d1) `.sol` parse into a `SolverResults` object** (see §2 for how
    Pyomo itself exposes the split). Because `load_solutions=False` is passed, the model's
    `Var`/`Suffix` objects are **not yet updated** by this call.
  - **(d2) solution load into the model**: `model.solutions.load_from(result)`
    (`network.py:738`), inside the `if solver_result_succeeded(result):` branch (`:736-750`) —
    a separate, explicit call, distinct from (d1) (see §2, this is the key finding for the
    instrumentation design).
  - **(e) post-solve bookkeeping**: `_is_recoverable_network_failure` (`:614-631`) decides
    whether to retry; if so, tier-1 cold retry (`:689-711`, itself a full second (a')-(d) unit via
    another `_run_smopf_solver_attempt` call with `option_overrides`) and, on tier-1 failure, an
    optional tier-2 cold+adaptive retry (`:715-734`, a third such unit); failure printing
    (`_print_network_failure_context`, `:660-671`, called at `:690` and `:724` and on final
    failure at `:750`); the two FrozenSMOPF callbacks, when wired, fire at
    `network_data.py:63-66`, dispatching to `_save_frozen_smopf_block`
    (`shared_resources_planning.py:5230-5270`) or `_save_frozen_network_block`
    (`shared_resources_planning.py:5272-5309`).
  - Back in `update_distribution_coordination_models_and_solve_sequential`, a final
    per-`(year,day)` convergence-warning print (`:5415-5424`) — cheap, still (e).

### 1.2 TSO — `update_transmission_coordination_model_and_solve`

`shared_resources_planning.py:5124-5222`. Same `_run_smopf` chain as §1.1 (shared production
code — `network.py` does not distinguish DSO/TSO callers). Two differences worth flagging for
the timing design:

- **(a)**: `:5128-5170`, structurally identical to the DSO's, but indexed by `dn` (active
  distribution-network index) as well as `(year, day, period)`.
- **Unconditional clone**: `res = transmission_network.optimize(model, from_warm_start=...,
  failure_snapshot_callback=save_failed_tso_block, pre_solve_snapshot_callback=
  save_selected_tso_comparator)` (`:5206-5211`) passes `failure_snapshot_callback` **unconditionally**
  (not gated on cycle, unlike the DSO's node-7-only gating at `:5387`) — so
  `NetworkData.optimize`'s `pre_solve_model = model[year][day].clone()` (`network_data.py:61`)
  fires on **every** TSO cycle, for all 12 `(year, day)` TSO blocks, regardless of whether a
  failure occurs. This is a real, currently-unmeasured (a)/(e)-adjacent cost specific to the TSO
  path and should get its own phase tag (`clone`) in the instrumentation, not be folded into a
  generic "(e) bookkeeping" bucket that would then look artificially larger for TSO than DSO for
  reasons unrelated to solver work.

### 1.3 ESSO — `update_shared_energy_storages_coordination_model_and_solve`

`shared_resources_planning.py:5502-5544`.

- **(a)**: `:5509-5531`, per `(node_id, year, day, period)` `set_value(...)` on `p_req`, `q_req`,
  `dual_p_req`, `dual_q_req`.
- **Solve dispatch**: `shared_ess_data.optimize(models, from_warm_start=..., cycle=cycle)`
  (`:5536`) → `SharedEnergyStorageData.optimize` (`shared_energy_storage_data.py:67-95`), one
  `_optimize` call per node (`:78-94`, threading `option_overrides=ESSO_TOL_OVERRIDES` and
  `cycle=cycle` for the per-solve log stamp).
- Inside `_optimize` (`shared_energy_storage_data.py:1177-...`), the same core structure as
  `_run_smopf`:
  - `_run_solver_attempt` (`:1135-1153`) → `_create_solver` (`:1023-1132`) builds options
    (`:1027-1043`) and, distinctively, a **per-solve unique log filename** stamped by
    `node_id`/`cycle` (`:1046-1107`) — resolved against `shared_ess_data.logs_dir`, with an
    explicit `os.path.exists()` check and a `_dupN` rename on collision (`:1093-1103`) rather
    than the DSO/TSO's implicit `file_append='yes'` tolerance (`network.py:576`). This log-path
    construction, including the existence check, is itself a small (a)/(e)-adjacent cost unique
    to the ESSO path.
  - **`result = solver.solve(model, tee=params.verbose, load_solutions=False)`** (`:1150`) —
    same (b)+(c)+(d1) bundling as network.py.
  - **(d2)**: `model.solutions.load_from(result)` (`:1296`).
  - **(e)**: recovery/tier-2 retries (`:1206-1292`, structurally identical to `_run_smopf`'s), and
    **an ESSO-specific extra bookkeeping step absent from the DSO/TSO path**:
    `_get_esso_complementarity_diagnostics(model, node_id, diagnostics_log_path)`
    (`:1309-1322`, function body `:1994-2028`) — on every successful solve, this **re-opens and
    parses the just-written IPOPT log file** (`_parse_ipopt_barrier_terms`, called at `:2011`) to
    compute the complementarity-leak detector and the closed-form spurious-throughput bound. This
    is a per-solve disk-read + text-parse cost, currently unmeasured, layered on top of the solve
    itself; it should get its own phase tag (`diagnostics_parse`) distinct from generic (e), since
    it is a fixed per-solve cost (proportional to log length, not to IPOPT iteration count in any
    simple way) that a persistent worker would need to absorb along with the solve if the ESSO is
    ever parallelized (it is not today — confirmed by `WORKER_REPORT_S36_PARALLEL_AUDIT.md`, no
    `_parallel` function exists for the ESSO or the TSO, DSO-only).

### 1.4 Phase (f): per-cycle global ADMM work outside blocks

- `update_and_check_convergence` (`shared_resources_planning.py:3076-...`), called three times
  per cycle (`:2524`, `:2550`, `:2568`) — consensus/dual updates, gated by `update_flags` per call.
- `_update_tso_proximal_centres_after_solve` (`:4598-...`), called once, after the TSO solve
  (`:2547`).
- `get_admm_residual_metrics` (`:5573-5888`, ~300 lines) and `get_admm_boyd_residual_metrics`
  (`:5888-...`), called once per cycle, **after** all three block types have solved (`:2585-2586`)
  — this needs all 51 blocks' results before it can run, so it is inherently serial regardless of
  how the block solves themselves are parallelized (already flagged in
  `WORKER_REPORT_S36_PARALLEL_AUDIT.md`'s verdict; confirmed here by reading the function
  signature and call site, not re-derived).
- Harness capture hooks: not present inside this loop (see §1.0); measured separately by the
  harness driver, outside the scope of what this instrumentation needs to touch.

---

## 2. Instrumentation design

### 2.1 Data model

A recorder keyed by `(cycle, agent, block, phase)` → elapsed seconds, `agent ∈ {dso, tso, esso}`,
`block` = `node_id` for DSO/ESSO or `(year, day)` (or `(dn, year, day)`) for TSO/DSO within a
node, `phase ∈ {param_update, clone, solve_bundle, load_solution, bookkeeping,
diagnostics_parse, admm_global}` (the last covering §1.4). `time.perf_counter()` pairs, recorded
with `try/finally` so a recording block never changes what exception (if any) propagates from the
wrapped call.

### 2.2 How it is switched on

An explicit, optional keyword argument threaded down the same call chains already documented in
§1, following the repository's existing convention for optional per-call parameters (`cycle=None`
already on every function in §1.1-1.3; `option_overrides=None`, `logs_dir=None`,
`log_suffix=None` on `_create_solver`/`_run_solver_attempt` in both `network.py` and
`shared_energy_storage_data.py`): a `timing_recorder=None` parameter, added at the same call
sites that already carry `cycle=None`, defaulting to `None` everywhere so an unmodified call
(every existing harness and every production run today) is byte-for-byte the same code path with
the recorder branch never taken. The top-level driver (`_run_operational_planning` or its caller)
would own the single recorder instance and pass it down; a harness sets
`timing_recorder=PhaseTimingRecorder()` to turn it on, leaves it `None` (the default) to turn it
off — no module-level global state (the audit already flagged module-level globals as a
multiprocessing hazard; this design avoids that class of defect from the start).

### 2.3 Where records go

The recorder accumulates in memory (a list of small tuples/dicts); the **harness** (not
production) is responsible for flushing it to a sidecar JSONL file, e.g. once per cycle or once
at the end of the run — keeping production code I/O-free when the recorder is off and avoiding
adding disk-write timing noise to the very quantity being measured. Under a persistent-worker
design (later, not this task), each worker's recorder would need its own file (or return its
records to the parent for aggregation) for exactly the reason `WORKER_REPORT_S36_PARALLEL_AUDIT.md`
already gives for `SolveProfileGuard` and for per-worker log paths: `ProcessPoolExecutor` on
macOS uses `spawn`, so no in-process object (recorder included) survives into a child.

### 2.4 Proving it changes nothing

A `time.perf_counter()` call is a monotonic clock read with no side effect on any Pyomo object,
solver option, or file; appending a float to a Python list has no numeric consequence for the
NLP. The prescribed proof is nonetheless the CLAUDE.md-mandated one: a **two-cycle run, recorder
off vs. recorder on**, both cold-start, same case files and seeds, diffed **bitwise** on every
numeric artifact already used as the determinism reference elsewhere in this programme (per-cycle
recourse, primal/dual residuals, SoH trajectory, the ESSO complementarity detector) — the same
class of gate already used for Step 3.0's determinism baseline and the Addendum 18 serial-vs-
parallel preflight. This is a **verification**, not a discovery step: if the diff is non-bitwise
it means the instrumentation was wired incorrectly (e.g. accidentally mutating a `Suffix` while
building the block key), not that timing measurement is inherently risky.

### 2.5 Separating (b), (c), (d1) — and the (d1)/(d2) split the brief's phase (d) actually hides

Pyomo already ships a zero-code-change way to split what happens **inside** `solver.solve(...)`
into exactly the three buckets the brief asks about, via the `report_timing=True` keyword (a
`solve()` kwarg popped and stored at `pyomo/opt/base/solvers.py:720`, purely gating `print()`
calls — no numeric or control-flow effect, confirmed by reading `OptSolver.solve`,
`pyomo/opt/base/solvers.py:557-712`, and `SystemCallSolver`'s override chain,
`pyomo/opt/solver/shellcmd.py`):

1. **(b) NL write**: `OptSolver._presolve` (`pyomo/opt/base/solvers.py:714-736`) times
   `self._convert_problem(...)` (`:728-734`, the call that reaches the NL writer — see §3) and
   prints `"%6.2f seconds required to write file"` when `report_timing`.
2. **(c) IPOPT subprocess**: back in `OptSolver.solve`, the call to `self._apply_solver()`
   (`:636`) is timed around (`:626-660`, prints `"seconds required for solver"`).
   `SystemCallSolver._apply_solver` (`pyomo/opt/solver/shellcmd.py`, `_apply_solver`/
   `_execute_command`) launches `subprocess.run(command.cmd, ...)` — the IPOPT process itself —
   and independently records `self._last_solve_time = time.time() - start_time` around it (own
   attribute, always populated, not gated by `report_timing`). This can be cross-checked against
   the IPOPT log's own `"Total seconds in IPOPT"` line, which `WORKER_REPORT_S36_PARALLEL_AUDIT.md`
   already parses from every production log file — the two should agree up to subprocess launch
   overhead (process fork/exec, environment setup), and any gap between them is a real, separate
   number worth recording (subprocess launch overhead, not IPOPT compute).
3. **(d1) `.sol` parse**: `self._postsolve()` (`solve()` line `:662`, timed `:655-704`, prints
   `"seconds required for postsolve"`) → `SystemCallSolver._postsolve` writes the raw log text,
   then (when `self._results_format is not None`) calls `self.process_output(self._rc)` →
   `process_logfile()` + `process_soln_file()` (or the `_results_reader` branch) — this is where
   the `.sol` file is actually opened and parsed into a `SolverResults`/`Solution` object.
   `process_output` has its **own** finer `report_timing` prints (`"seconds required to read
   logfile"` / `"...to read solution file"`, `pyomo/opt/solver/shellcmd.py`, inside
   `process_output`), so `report_timing=True` in fact separates (d1) into log-read and
   solution-file-read sub-buckets for free.
4. **(d2) load into the model — NOT covered by `report_timing` at all.** Back in
   `OptSolver.solve` (`pyomo/opt/base/solvers.py:684-697`), the parsed result is loaded into the
   model **only if `self._load_solutions` is True** (the `load_solutions` kwarg, default `True`).
   Both production call sites pass **`load_solutions=False`** explicitly
   (`network.py:608`; `shared_energy_storage_data.py:1150`) — so this in-`solve()` load branch is
   **never taken** in production. The actual load of parsed values into the model's `Var`/`Suffix`
   objects happens later, via the explicit, separate call `model.solutions.load_from(result)`
   (`network.py:738`; `shared_energy_storage_data.py:1296`), which `report_timing` knows nothing
   about (it is outside `solve()` entirely). **This is the one finding that changes how the brief's
   phase (d) should be instrumented**: it is two sub-phases, not one — (d1) `.sol` parse, inside
   `solve()`, covered by `report_timing`; (d2) load into the model, outside `solve()`, requiring
   its own manual `perf_counter()` wrap (already planned as one of the two production wrap points
   in §2.1-2.2).

**Consequence for the design**: only **two** manual `perf_counter()` wraps are needed to fully
separate (a)-(e) at each of the six call sites in §1.1-1.3 (`network.py:608`+`:738`;
`shared_energy_storage_data.py:1150`+`:1296`, each reached once per primary attempt and once per
retry): one around the `solver.solve(...)` call (giving `solve_bundle` = b+c+d1 combined) and one
around the following `model.solutions.load_from(result)` call (giving `load_solution` = d2 alone).
`report_timing=True` is not needed as the permanent mechanism (it prints unstructured text to
stdout, not the JSONL the design wants, and would have to be captured from `tee`d IPOPT output
which already exists as a large per-run text stream) — it is proposed **only** as a one-off,
zero-risk cross-check in the Step 5 preflight, to validate that the manual `solve_bundle` wrapper
agrees with Pyomo's own three-way split (and, transitively, that the IPOPT log's
`"Total seconds in IPOPT"` figure the existing audit already extracts is consistent with
`_last_solve_time`) before trusting the manual numbers for the persistent-worker sizing decision.

`keepfiles=True` (retains the `.nl`/`.sol` scratch files instead of deleting them,
`pyomo/opt/solver/shellcmd.py:_postsolve`, `:pop("keepfiles", False)` at `shellcmd.py:_presolve`)
is not needed for the timing split itself — it only affects whether the files survive after the
solve, not how long each step takes — but is useful, separately, as the zero-solve read for §4's
question (confirming numeric vs. symbolic tokens in an already-existing `.nl` file rather than
generating a new one).

---

## 3. Cheap win 1 — NL writer v2

**Already the production writer; there is no change to adopt.**

Chain of evidence, all read-only:

- `pyomo/repn/plugins/__init__.py:29-31` — the bare `'nl'` format is registered **as an alias
  for** `nl_v2`: `WriterFactory.register('nl', ...)( WriterFactory.get_class('nl_v2') )`.
- `pyomo/repn/plugins/nl_writer.py:160` — `@WriterFactory.register('nl_v2', ...) class NLWriter`
  is the class this resolves to.
- `pyomo/repn/plugins/ampl/ampl_.py:322` — `@WriterFactory.register('nl_v1', ...) class
  ProblemWriter_nl` is the legacy writer, a **different class**, reached only if something
  explicitly asks for `'nl_v1'` or calls the documented debugging tool
  `pyomo.repn.plugins.activate_writer_version('nl', 1)` (`pyomo/repn/plugins/__init__.py:38-43`,
  labelled `"""DEBUGGING TOOL to switch the 'default' writer implementation"""` in its own
  docstring — a **global** WriterFactory-registry mutation, not a per-call option).
- Call path from production: `network.py:608` (`solver.solve(model, ...)`) → `OptSolver.solve`
  (`pyomo/opt/base/solvers.py:557`) → `self._presolve` → `OptSolver._presolve` (`:714-736`) →
  `self._convert_problem(...)` (`:730`) → `convert_problem()` (`pyomo/opt/base/convert.py:25`) →
  `PyomoMIPConverter.apply()` (`pyomo/solvers/plugins/converter/model.py:43`, the `nl`/`mps`
  branch at `:167-183`) → `instance.write(filename=..., format=ProblemFormat.nl,
  io_options=io_options)` → `Block.write()` (`pyomo/core/base/block.py:1955-2032`) →
  `problem_writer = WriterFactory(format)` (`:1997`) — resolves to the registered `nl_v2` class,
  never `nl_v1` — → `problem_writer(self, filename, solver_capability, io_options)` (`:2009`),
  which invokes `NLWriter.__call__` (`nl_writer.py:288`).
- No case file (`data/SRP1/**/*.json`, grep clean) or production code passes any writer-selecting
  `io_options` key; the only two files in the repository that pass `io_options` to a `.write()`
  call at all are diagnostic, zero-solve harnesses (`p513_e_gated_capture.py:100`,
  `p513_c_param_move_gate.py:171`), and both only set `symbolic_solver_labels` (see §4), never a
  writer version.

**What could change numerically, and why it doesn't here.** `nl_v2`'s two headline optional
features relative to `nl_v1` are `scale_model` (write variables/constraints in scaled space using
a `scaling_factor` `Suffix`) and `linear_presolve` (variable elimination without fill-in) —
`nl_writer.py:208-219` and `:271-281`. Both default `True` at the `NLWriter.CONFIG` class level
(`:211`, `:274`), but **`NLWriter.__call__` — the exact entry point `Block.write()` uses —
force-disables both** (`nl_writer.py:296-303`): `config.scale_model = False; config.linear_presolve
= False`, with the comment *"There is no (convenient) way to pass the scaling factors or
information about presolved variables back to the solver through the old ('call') interface...
We will play it safe and disable scaling / presolve when called through this API."* So, on this
call path, `nl_v2`'s only live difference from `nl_v1` is internal representation and constraint/
variable ordering (`file_determinism`, default `ORDERED` — declaration order, `nl_writer.py:182-198`),
not scaling or presolve. This closes the main numerical-risk vector the brief asked about.

**Test.** Since there is no live change to test (nl_v2 is not a candidate, it is the status quo),
no solve-based equivalence gate is applicable or needed. If the Planner wants a documented
negative control (e.g. for the manuscript, or as a future regression check that a Pyomo upgrade
hasn't silently flipped the registered default), the realistic test is a **zero-solve** one:
`model.write(path, format=ProblemFormat.nl_v1, io_options={'symbolic_solver_labels': False})`
against the existing `nl_v2` write on one preserved fixture (e.g. a P5.12-R snapshot), diffing
`.nl` file size and the row/col ordering recorded in `NLWriterInfo` — **not** an IPOPT-iterate
equivalence gate, since `nl_v1` uses an entirely different internal representation (`ampl_repn`)
and is not expected to produce byte-identical NL text even when mathematically equivalent; an
iterate-equivalence gate would only be meaningful if switching writers were actually proposed,
which it is not. **Recommendation: close Cheap Win 1 as already-adopted; no gate required.**

---

## 4. Cheap win 2 — no symbolic labels in the `.sol` round trip

**Already off (`False`) for both the network solves and the ESSO solves; there is no change to
adopt, and the two paths do not differ.**

- Pyomo default: `NLWriter.CONFIG.declare('symbolic_solver_labels', ConfigValue(default=False,
  ...))` (`nl_writer.py:200-207`). `NLWriter.__call__` reads `config = self.config(io_options)`
  (`:295`) — `io_options` is whatever extra keyword arguments reached `.write()`, which trace back
  to whatever extra keywords were passed to `solver.solve(...)`.
- Production calls: `network.py:608` — `solver.solve(model, tee=params.solver_params.verbose,
  load_solutions=False)`; `shared_energy_storage_data.py:1150` — `solver.solve(model,
  tee=params.verbose, load_solutions=False)`. **Neither passes `symbolic_solver_labels` or any
  other `io_options`-bound keyword.** No case file under `data/SRP1/` sets it either (grep across
  every `*.json` under `data/SRP1/` is empty). This directly answers the brief's "which may
  differ" question for the ESSO: **it does not differ** — both paths get the Pyomo default,
  `False`.
- The only two places in the whole repository that ever set this key both set it to **`False`**
  explicitly (redundant with the default, but present, and both are zero-solve diagnostic
  harnesses, not the ADMM path): `p513_e_gated_capture.py:100`, `p513_c_param_move_gate.py:171`.

**What switching it off would save**: nothing further — it is already off; no `.row`/`.col`
sidecar files are written today, and the NL/`.sol` round trip already uses bare numeric row/column
indices exclusively.

**What would break if it were ever turned on** (recorded for completeness, since the brief asks
"what would break", not only "is it on"): by inspection of `SystemCallSolver.process_output` /
`process_soln_file` (`pyomo/opt/solver/shellcmd.py`), the `.sol` file is always parsed by numeric
position — the flag controls only whether `.row`/`.col` name-lookup sidecar files are written and
whether variable/constraint *names* appear in solver log text, not how the `.sol` file itself is
read back. The warm-start suffix mapping (`ipopt_zL_in`/`zU_in`/`dual`, replaced via
`replace_warm_start_suffix` at `network.py:583-584` and `shared_energy_storage_data.py:1118-1119`)
is keyed through Pyomo's `ComponentMap`/`SymbolMap` machinery inside `model.solutions.load_from`,
not through the `.row`/`.col` files, so by inspection it would not break either — but this is a
claim about Pyomo's internals this READ-ONLY task cannot confirm by solving, and is flagged as
such rather than asserted.

**Test.** Same posture as Cheap Win 1 — no gate is needed to adopt this win, because it is already
adopted. If positive confirmation beyond code reading is wanted, the test is zero-solve: read an
existing `.nl`/`.sol` artifact pair already on disk from a `keepfiles=True` capture (e.g.
`data/SRP1/Results/P512ArmA/used_tmp27ntrqce.pyomo.nl` / the paired `.sol`) and confirm variable/
constraint tokens are bare numeric (`v0`, `v1`, ... / row indices), not names — a read of an
artifact that already exists, no new solve required. **Recommendation: close Cheap Win 2 as
already-adopted; no gate required.**

---

## 5. Measurement plan for the preflight

**Timing**: after the ρ_ess arms complete, per Addendum 19's stated ordering ("Step 3.6 proceeds,
decoupled... Preflight and gate after the ρ_ess arms").

**Run**: one two-cycle production run (cold start, C\* candidate — the same configuration class
already used for the Step 3.4/3.6 gates), executed once with the `timing_recorder` (§2.2) **on**
and `report_timing=True` also on for that one run only (the §2.5 cross-check, not the permanent
mechanism), producing the JSONL sidecar plus the ordinary production log/IPOPT-log artifacts. This
single run also serves as one side of the §2.4 bitwise-identity pair (the other side being the
same two cycles with the recorder off, already the production default).

**Per-block-type breakdown**: extend `WORKER_REPORT_S36_PARALLEL_AUDIT.md`'s existing
`cycle | IPOPT DSO | IPOPT TSO | IPOPT ESSO | IPOPT total | wall | overhead | overhead %` table
format (its Part 2 table, reused rather than redesigned) with the phase split from §1/§2:
`param_update` (a), `clone` (TSO-unconditional + DSO node-7-only, §1.2), `solve_bundle` (b+c+d1,
cross-checked against `report_timing`'s three sub-buckets for this one run), `load_solution` (d2),
`bookkeeping` (e, network path), `diagnostics_parse` (e, ESSO-only, §1.3), `admm_global` (f, §1.4)
— median/mean/max per block type (DSO, TSO, ESSO), matching the existing report's per-cycle
median-over-12-cycles convention (this run only has 2 cycles; report both cycles individually
rather than a median, since 2 points do not support a robust median).

**Decision rule (pre-registered before the preflight is run):**

Define `overhead_local` = the sum of phases a persistent worker, owning one block and running the
model-update-through-postsolve sequence entirely inside its own process, can absorb without any
cross-process communication until results are returned: `param_update` (a) + `clone` (only where
it currently fires) + the NL-write share of `solve_bundle` (b) + `load_solution` (d2) +
`bookkeeping`/`diagnostics_parse` (e). Define `overhead_serial` = whatever of the measured ~21.5 s
median overhead (`WORKER_REPORT_S36_PARALLEL_AUDIT.md` Part 2) is **not** `overhead_local` — this
is dominated by `admm_global` (f), which is architecturally serial regardless of the redesign
(§1.4), plus any Python glue between stages that this preflight's phase split does not attribute
to a specific block.

**Build the persistent-worker path only if `overhead_local / (overhead_local + overhead_serial) ≥
X = 70%`** of the measured overhead.

**Reasoning for X = 70%** (an Amdahl-law derivation from numbers already measured by
`WORKER_REPORT_S36_PARALLEL_AUDIT.md`, not a round-number guess): that audit's own pessimistic,
solve-only-parallelism bound already reaches only **1.483×** at 8 workers (its Part 2 table),
because it (correctly, for that audit's scope) treats the full 21.488 s median overhead as
unchanged and fully serial. Persistent workers are worth building only if they move the achievable
speed-up decisively past that number — set the bar at **3×** at 8 workers (roughly double the
existing pessimistic bound; still well short of the 4-5× / 8-10 s aspirational target in Addendum
18, and that gap should be stated plainly, not implied away). Amdahl's law,
`speedup(W) = 1 / (f_serial + (1 - f_serial)/W)`, inverted for the serial fraction *of the whole
cycle wall time* needed to reach speedup `S` at `W` workers:
`f_serial ≤ (1/S - 1/W) / (1 - 1/W)`. At `S = 3`, `W = 8`: `f_serial ≤ (1/3 - 1/8)/(1 - 1/8) =
0.2083/0.875 ≈ 0.238`, i.e. no more than **≈23.8%** of the 33.725 s median cycle wall (≈8.03 s)
may remain serial. In the persistent-worker design, the 12.990 s median IPOPT time is itself
parallelized (down to the already-computed 8-worker LPT loads: DSO 2.113 s + TSO 0.113 s + ESSO
0.014 s ≈ 2.24 s, summed because the three stages are still sequential per §1.0 — the persistent-
worker redesign changes concurrency *within* a stage, not the DSO→TSO→ESSO ordering), so the
remaining serial-time budget for `overhead_serial` is `8.03 - 2.24 ≈ 5.8 s`, i.e.
`overhead_local ≥ 21.488 - 5.8 ≈ 15.7 s`, i.e. `overhead_local/overhead_total ≥ 15.7/21.488 ≈
73%`. **X = 70%** is this figure rounded down slightly as a conservative screening threshold,
given the acknowledged imprecision already flagged upstream (the LPT partition is a greedy
heuristic, not exact; the `overhead_serial` figure this preflight will measure is itself only as
accurate as the phase split's own boundaries). This is a **screening rule**, not a speed-up
guarantee: passing it licenses building the redesign as worth trying; it does not certify that 3×
will in fact be reached (a further, later gate — the two-cycle bitwise-identity preflight already
specified in Addendum 18 — is what certifies correctness, not speed).

**Projected 8-worker speed-up to report alongside the pass/fail**: once the preflight's actual
`overhead_local`/`overhead_serial` split is measured, compute `f_serial = (overhead_serial + Σ
per-stage-8-worker-residual-IPOPT-time) / wall` and report `speedup(8) = 1/(f_serial +
(1-f_serial)/8)` directly from Amdahl's law using the measured numbers — this document
pre-registers the **rule and its derivation**, not the resulting number, since the number depends
on data this preflight has not yet collected.

---

## 6. Risks

- **Determinism.** The instrumentation itself is inert by construction (§2.4), but any future
  persistent-worker redesign this preflight informs inherits every risk
  `WORKER_REPORT_S36_PARALLEL_AUDIT.md`'s "Concrete list" already itemizes (its items 1-8): no
  `OMP_NUM_THREADS`/single-linear-solver-thread cap exists anywhere today (its Determinism item,
  #7) — the bitwise-identity gate in Addendum 18 is meaningless without it, and this timing
  instrumentation does not change that gap.
- **Log and temp-file collisions.** The DSO/TSO log path (`network.py:560-576`) still relies on
  `file_append='yes'` and a shared `logs_dir` object across all DSO nodes and the TSO
  (`shared_resources_planning.py:7348`, `:7392` in the audit's citation — re-confirmed structurally
  by reading `network.py:560-576` here) — a collision hazard this instrumentation does not touch,
  but the timing sidecar itself must not repeat the same mistake: per §2.3, each worker's JSONL
  (once workers exist) must be its own file, aggregated by the parent, exactly as recommended for
  `SolveProfileGuard` counts (item 3 below) and for logs (already flagged, item 2, in the prior
  audit's concrete list).
- **Guard accounting across processes.** `SolveProfileGuard.install()` monkeypatches
  `OptSolver.solve`/`SystemCallSolver._execute_command` at the class level in the current
  process's in-memory `pyomo` copy (`p513_solve_profile_guard.py:50-78`); under `spawn` (macOS
  default), a `ProcessPoolExecutor` worker never inherits this patch
  (`WORKER_REPORT_S36_PARALLEL_AUDIT.md`, "SolveProfileGuard across the process boundary"). Any
  future zero-solve or bounded-solve claim made about a persistent-worker run must install the
  guard **inside each worker** (e.g. via `ProcessPoolExecutor(initializer=...)`) and sum counts
  back to the parent — this instrumentation task does not add or remove solves, so it does not by
  itself change this risk, but a future preflight/gate harness that arms a guard around the
  two-cycle timing run must account for it.
- **Recovery retries.** Each tier-1/tier-2 retry (`network.py:689-734`;
  `shared_energy_storage_data.py:1206-1292`) is itself a full (a')-(d) solve unit reusing the same
  `solve_bundle`/`load_solution` wrap points (§2.5) — the phase-timing design must tag each retry
  attempt with its own `(cycle, agent, block, phase, attempt_label)` record, not fold a retry's
  time into the primary attempt's bucket, or the per-block totals for cycles with recovery
  activity (298 of them in the `P515S35_REF_run` reference, per the prior audit) would silently
  overstate a single attempt's cost.
- **FrozenSMOPF snapshots.** `model[year][day].clone()` (`network_data.py:61`) is a real,
  currently-untimed cost, unconditionally paid every TSO cycle and conditionally paid for DSO
  node 7 (§1.1-1.2) — it must be its own phase tag (`clone`), not merged into `bookkeeping`,
  because it is a snapshot/debug cost orthogonal to the solve itself and would otherwise
  contaminate the `overhead_local` figure the §5 decision rule depends on (a persistent worker
  would still need to decide whether to keep paying it, a question this preflight can answer once
  `clone` is measured separately, but cannot answer today).

---

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` (Addendum 18 full text; Addendum 19 full text, especially the
  Step 3.6 bullet).
- `WORKER_REPORT_S36_PARALLEL_AUDIT.md` (full file, commit `710af0de`).
- `shared_resources_planning.py`: ADMM loop and entry (`:2310`, `:2495-3007`); DSO sequential
  update (`:5311-5426`); DSO parallel update and `update_and_solve_dso`
  (`:5429-5499`, read for cross-reference to the audit, not re-derived); TSO update
  (`:5124-5222`); ESSO update (`:5502-5544`); FrozenSMOPF savers (`:5230-5309`);
  `get_admm_residual_metrics`/`get_admm_boyd_residual_metrics` signatures (`:5573`, `:5888`);
  `update_and_check_convergence` signature (`:3076`); `_update_tso_proximal_centres_after_solve`
  signature (`:4598`); `logs_dir` wiring (`:7330-7350`, `:7385-7396`, `:7438-7448`).
- `network.py`: full `_create_smopf_solver`/`_run_smopf_solver_attempt`/
  `_is_recoverable_network_failure`/`_run_smopf` block (`:540-777`); `Network.run_smopf`
  (`:59-61`); model suffix declarations (`:530-537`).
- `network_data.py`: `NetworkData.optimize` (`:54-68`).
- `shared_energy_storage_data.py`: `SharedEnergyStorageData.optimize` (`:67-95`); `_create_solver`
  (`:1023-1132`); `_run_solver_attempt` (`:1135-1153`); `_is_recoverable_shared_ess_failure`
  (`:1156-1170`); `_optimize` (`:1177-...`, through the success/failure branches to `:1330`s);
  `_get_esso_complementarity_diagnostics` (`:1994-2028`); `ESSO_TOL_OVERRIDES` (`:1020`).
- `p513_solve_profile_guard.py` (full file — `install`/`verify` mechanism, cited in §6).
- Installed Pyomo 6.9.5 (`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/lib/python3.11/site-packages/pyomo/`):
  `opt/base/solvers.py` (`OptSolver.solve` `:557-712`; `_presolve` `:714-...`; `_convert_problem`
  `:781-784`); `opt/solver/shellcmd.py` (`SystemCallSolver._presolve`/`_apply_solver`/
  `_postsolve`/`_execute_command`/`process_output`, full file read); `opt/base/convert.py`
  (`convert_problem`, full file); `solvers/plugins/solvers/IPOPT.py` (full file);
  `solvers/plugins/solvers/ASL.py` (full file); `solvers/plugins/converter/model.py`
  (`PyomoMIPConverter.apply`, `:22-183`); `core/base/block.py` (`Block.write`, `:1955-2032`);
  `repn/plugins/__init__.py` (full file); `repn/plugins/nl_writer.py` (`:1-330` — `NLWriter.CONFIG`
  declarations and `__call__`); `repn/plugins/ampl/ampl_.py` (`:300-330` — `nl_v1` registration);
  `common/timing.py` (class/def listing only, to confirm no separate global timing mechanism
  applies here).
- `data/SRP1/**/*.json` (grepped for `symbolic_solver_labels`, `keepfiles`; both empty).
- `p513_e_gated_capture.py:100`, `p513_c_param_move_gate.py:171` (the only two repository files
  that set `symbolic_solver_labels`, both `False`, both zero-solve harnesses).

## Files created

- `P5_15_STEP36_TIMING_DESIGN.md` (this file). No other file was created or modified.

## Commands run

- `import pyomo; pyomo.__version__` (read-only version check, no model, no solve).
- `grep`/`find`/`wc -l`/`sed -n` read-only inspection of the files listed above. No `solver.solve`,
  no `ipopt` invocation, no production entry point was executed.

## Validation

- No `.py` file, case file, or other tracked source file was edited.
- No solve was run; every quantitative claim about Pyomo's runtime behavior is a citation of the
  installed package's source, not a measurement — where source alone cannot settle a claim (e.g.
  whether disabling symbolic labels could break warm-start suffix mapping in practice), this
  document says so explicitly (§4) rather than asserting the untested inference as fact.
- Every WORKER_REPORT_S36_PARALLEL_AUDIT.md number reused here (median overhead 21.488 s, median
  IPOPT total 12.990 s, median wall 33.725 s, the 8-worker LPT loads, the 1.483× pessimistic
  bound) is cited from that report's own tables, not re-derived from raw logs by this task.

## Unexpected findings

- **Both proposed "cheap wins" are already in force in production** — the NL writer is already
  `nl_v2` (§3) and `symbolic_solver_labels` is already `False` on both the network and ESSO paths
  (§4, closing the brief's open "may differ" question with "it does not"). Neither is a change
  the Planner can authorize or test in the ordinary sense; both close as already-adopted, with a
  zero-solve negative-control test offered in case documentation is wanted.
- **The brief's phase (d) is two sub-phases, not one**, because production universally solves
  with `load_solutions=False` (§2.5) — `.sol` parse (d1, inside `solve()`, covered by
  `report_timing`) and model load (d2, outside `solve()`, requires its own manual wrap). This
  changes the instrumentation design's wrap points from what a literal reading of the brief's
  phase list would suggest, but does not add extra wrap points beyond the two already planned in
  §2.1 (`solve_bundle`, `load_solution`).
- **The TSO pays an unconditional per-cycle `model.clone()`** (§1.2) that the DSO only pays at
  node 7 — an asymmetry not mentioned in the brief or the prior audit, found while tracing the
  callback wiring for the FrozenSMOPF phase-map entry.
- **The ESSO carries a per-solve log re-parse** (`_get_esso_complementarity_diagnostics`, §1.3)
  that the DSO/TSO path has no equivalent of — a small but real, currently-unmeasured, fixed cost
  a future ESSO-parallelizing worker would need to absorb.

## Remaining issues

- This document proposes wrap points and a decision rule; it does not implement them (out of
  scope) or measure `overhead_local`/`overhead_serial` (requires the Step 5 preflight run, itself
  gated to occur after the ρ_ess arms, per Addendum 19).
- The `X = 70%` threshold's derivation (§5) rests on the existing audit's LPT-heuristic 8-worker
  loads and on a chosen interim target (`S = 3`, not the aspirational `4-5×`) — both are stated
  assumptions, not measurements; the Planner may want a different `S` before the preflight runs,
  which would change `X` via the same formula.
- Whether disabling `symbolic_solver_labels` could ever affect warm-start suffix mapping (§4) is
  answered by code inspection only, not by a solve; flagged, not resolved, since resolving it
  would require running a solve, which this task's scope forbids.

## Questions for Planner

- Both cheap wins are closed as already-adopted (§3, §4) rather than as pending changes — is a
  zero-solve negative-control test (offered in both sections) wanted for the manuscript/record, or
  is "already the default, closes with this reading" sufficient documentation on its own?
- Is `S = 3` (the interim 8-worker speed-up bar behind `X = 70%`, §5) the right target, or should
  the threshold be re-derived against a different bar (e.g. `S = 2` for a lower bar to clear, or
  the full aspirational `4-5×` for a stricter one) before the Step 5 preflight is authorized to
  run?
