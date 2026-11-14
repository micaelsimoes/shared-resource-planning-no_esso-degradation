# Worker Report — P5.15 Step 3.6 (Addendum 18): parallel-path audit and cycle-time attribution

**Task**: bounded READ-ONLY audit of the existing `*_parallel` solve path and of where the
~39 s ADMM cycle wall time goes. Zero solves. No edits to production, case files or the
harness. `PLANNER_BRIEF_2026-09-13.md` Addendum 18.

**Script**: `p515_s36_parallel_audit.py` (repo root). **Output**:
`data/SRP1/Results/P515S36/parallel_audit/` (new directory; the script refuses to run if it
already exists). **Zero solves**: `p513_solve_profile_guard.SolveProfileGuard` armed with an
empty permitted list for the whole run, verified `expected_solves=0, expected_execs=0` at the
end — `[OK] SolveProfileGuard verified: 0 permitted solves, 0 blocked solves.`

Evidence used: `data/SRP1/Results/P515S35_REF_run/` (477-cycle production run,
`parallel_execution=False`), its per-solve IPOPT logs under
`data/SRP1/Results/P56A/evals/p515s35ref_baseline/logs/`, its `stdout_baseline.log`, and its
`network_failures_baseline.jsonl` (298 recovered failures, 0 unrecovered).

---

## PART 1 — Code audit of the existing parallel path

### Dispatch mechanism

`concurrent.futures.ProcessPoolExecutor` (not threads, not joblib) —
`shared_resources_planning.py:6`. A **new** executor is constructed inside each parallel
function and torn down at the end of its `with` block, so worker processes are **spawned per
call** (i.e. every cycle for the DSO update step), **not persistent**:

- `create_distribution_networks_models_parallel` — `shared_resources_planning.py:3625-3644`,
  `max_workers = min(os.cpu_count() // 2, len(distribution_networks))`.
- `update_distribution_coordination_models_and_solve_parallel` —
  `shared_resources_planning.py:5429-5449`, `max_workers = os.cpu_count() // 2`.

Every argument, **including the full per-node Pyomo model dict**, is pickled to the worker
(`shared_resources_planning.py:5439-5442`, `executor.submit(update_and_solve_dso, node_id,
distribution_networks[node_id], models[node_id], ...)`), and the updated model is pickled back.
Models are **not rebuilt** each cycle (they persist across cycles in the parent as plain Python
objects between calls), but they **are re-pickled to/from a worker every single cycle** under
this design — an overhead cost of its own that this audit did not attempt to measure.

### What is sent to / returned from workers, and result assembly

Sent: `node_id`, the `DistributionNetwork`/`NetworkData` object, the full per-`(year,day)`
Pyomo model dict, the consensus/dual dicts (`vmag_req`, `dual_vmag`, `pf_req`, `dual_pf`,
`ess_req`, `dual_ess`), the ADMM params object, the node's estimated SESS capacity. Returned:
`(node_id, results_dict, updated_model)` (`shared_resources_planning.py:5499`). Results are
assembled by **completion order** via `as_completed()` and written into `node_id`-keyed dicts
(`shared_resources_planning.py:5444-5447`), so the final structures are order-independent —
this part is safe.

### Solver invocation per worker

- Executable: `po.SolverFactory(solver_params.solver, executable=solver_params.solver_path)`
  (`network.py:542`) — the same `NLP_SOLVER_PATH`-configured executable production always uses;
  no per-worker override.
- Linear solver / threads: from the case-file `solver.options` only (e.g. `linear_solver: ma97`
  in `case33_1/2/3_params.json`, `case9_params.json`). **No `OMP_NUM_THREADS` or any per-worker
  thread cap exists anywhere** in `shared_resources_planning.py`, `network.py`,
  `shared_energy_storage_data.py` or `network_parameters.py` (targeted grep, all four files).
  `W` workers today would mean `W` simultaneous, unthrottled, potentially multi-threaded MA97
  processes.
- Pyomo `TempfileManager`: never explicitly configured (no `tmpdir=`/`keepfiles=True` passed to
  `solver.solve` anywhere) — `.nl`/`.sol` scratch files use Pyomo's default unique-per-call
  names, which is **not** a collision risk under multiprocessing.
- **IPOPT log path**: built deterministically in `_create_smopf_solver`
  (`network.py:540-602`) from `(case name, year, day[, log_suffix])` — **no node_id, worker id,
  or PID component** — resolved against `network.logs_dir`, which is the **same** directory
  object for every DSO node, the TSO, and the ESSO
  (`shared_resources_planning.py:7348`, `:7392`, `:7444`: `distribution_network.logs_dir =
  planning_problem.logs_dir` / `transmission_network.logs_dir = ...` / `shared_ess_data.logs_dir
  = ...`). `file_append = 'yes'` is set unconditionally (`network.py:573-574`), so a collision
  would **silently interleave/append rather than error**. In the current 3-node SRP1 case
  (`case33_1`/`case33_2`/`case33_3` at nodes 5/7/9) this happens not to collide, because each
  node already has a distinct case name — but nothing in the code prevents a collision if two
  blocks with the same case name + year + day were ever solved by two workers at once.
- By contrast, the **ESSO's** `_create_solver` (`shared_energy_storage_data.py:1023-1132`,
  Addendum 5 / P5.15-F) **does** stamp the log by node and cycle, resolves it against its own
  `logs_dir`, and checks `os.path.exists()` before writing — renaming to a `_dupN` suffix rather
  than silently appending (`:1058-1107`). This asymmetry (the ESSO was hardened after a
  documented defect; the DSO/TSO path was not) is direct evidence of where the next hardening
  effort belongs.

### TSO / ESSO coverage

**DSO only.** No `update_transmission_..._parallel` or
`update_shared_energy_storages_..._parallel` function exists anywhere in the repository (grep
for `_parallel` across `shared_resources_planning.py`, `network.py`,
`shared_energy_storage_data.py` finds only the two DSO functions and their model-construction
counterpart, `create_distribution_networks_models_parallel`). The TSO and ESSO solves always run
sequentially in the parent process, in every configuration.

### Block-level parallelism actually achieved today

Even where the parallel path exists, it parallelizes over **DSO nodes** (3 for SRP1), not over
the 36 independent `(node, year, day)` blocks. `update_and_solve_dso`
(`shared_resources_planning.py:5452-5499`) is submitted **once per node**, and internally loops
over all 12 `(year, day)` pairs **sequentially** via a single call to
`distribution_network.optimize(model, ...)` (`network_data.py:54-68`, which itself loops over
`self.years`/`self.days` sequentially). So today's parallel path gives at most **3-way**
concurrency for SRP1's DSO step (further capped by `os.cpu_count() // 2`), never the 36-way (or
51-way DSO+TSO+ESSO) concurrency Addendum 18 targets. Reaching that requires restructuring the
unit of work from "one node's whole year/day loop" to "one `(node, year, day)` block" — exactly
the persistent-worker redesign the brief proposes.

### ADMM update order / recovery / snapshots / guard across the process boundary

- **Update order**: preserved for the DSO step's own bookkeeping (node-keyed dicts), but the
  parallel path **drops the `cycle` argument entirely**
  (`update_distribution_coordination_models_and_solve_parallel` signature at
  `shared_resources_planning.py:5429` has no `cycle` parameter, unlike the sequential twin at
  `:5318`) and never threads `failure_snapshot_callback` / `pre_solve_snapshot_callback` through
  (`shared_resources_planning.py:5489`: `res = distribution_network.optimize(model,
  from_warm_start=from_warm_start)` — contrast the sequential call at `:5407-5413`, which passes
  both callbacks).
- **Consequence**: **FrozenSMOPF snapshot writing is entirely un-wired in the parallel path.** A
  node-7 failure under `parallel_execution=True` would never produce the frozen pre-solve
  snapshot the sequential path captures today.
- **Recovery (tier-1/tier-2)**: `network.py:_run_smopf` (`:673-763`) is pure and function-local
  — it runs identically inside a worker, since `update_and_solve_dso` calls the same
  `distribution_network.optimize()` → `network.run_smopf()` → `_run_smopf()` chain. **Recovery
  retries do work across the process boundary today.** What would not: any diagnostics **sink**
  object mutated inside a worker (the ESSO's `self.solver_recovery_diagnostics`,
  `shared_energy_storage_data.py:67-95`) is lost across a process boundary, because only
  `(node_id, res, model)` is returned — not the mutated network/diagnostics object. This is not
  currently exercised for the DSO (no equivalent sink parameter exists on that path today), but
  it is the exact failure mode a future DSO diagnostics sink would hit if added without also
  being returned from the worker.
- **`SolveProfileGuard` across the process boundary**: `install()`
  (`p513_solve_profile_guard.py:50-78`) monkeypatches `OptSolver.solve` and
  `SystemCallSolver._execute_command` **at the class-object level in the current process's
  in-memory copy of `pyomo`**. `ProcessPoolExecutor` on macOS/Darwin uses the **`spawn`** start
  method by default (not `fork`, since Python 3.8) — every worker re-executes the interpreter
  and re-imports `pyomo` fresh, and **never inherits the parent's monkeypatch**, regardless of
  when the guard was installed. A solve executed inside a worker therefore calls the
  **unpatched** `OptSolver.solve`: it is invisible to the guard — neither `permitted_solve` nor
  `blocked_solve`. Consequences: (a) a guard armed for **0 solves** in the parent would
  **silently pass** even if 36 DSO blocks solved via the parallel path in child processes — a
  false "no-solve" claim; (b) a guard armed for the full 51-solve profile would **undercount**
  by exactly the DSO solves routed through the parallel path, and `verify()` would report "too
  few" — correctly flagging a discrepancy, but for the wrong structural reason (an invisible
  call site, not a code path that failed to run), which would misdiagnose the cause unless the
  reader already knows the parallel path exists. **Any future guard covering a persistent-worker
  design must install the same monkeypatch inside every worker process (e.g. at a
  worker-initializer) and aggregate each worker's own counts back to the parent.**

### P5.6-B concurrency crashes — scoped search and attribution

**Scope of search**: `REVISION_CONTEXT.md` (all `P5.6-B` occurrences),
`P5_6_NONLINEAR_DERIVATIVE_FREE_PLANNING_REPORT.md` in full, the one `P5_11...` reference to
`P5.6-B7`, and every `p56*.py` script in the repository root (grep for `ProcessPoolExecutor`,
`multiprocessing`, `concurrent`, `contention`, `SOLVER_CRASH`). Did **not** search git history or
other branches.

**Finding**: the P5.6-B crash is **not the same code path** as this audit's subject. It is
**candidate-evaluation-level** concurrency in the derivative-free search harness
(`p56a_oracle.py` and siblings): multiple **full**, independent 48-block sequential ADMM
evaluations (one whole SRP1 planning run per worker) running concurrently, one worker per
candidate — an outer layer above the within-cycle block solves audited here. The report records:
*"Resource contention is an observed risk... `W` workers means `W` simultaneous IPOPT processes
on top of whatever threading MA97 uses. On this 8-core machine, a `SOLVER_CRASH`
(`ApplicationError: Solver (ipopt) did not exit normally`) was actually observed during P5.6-A
while several heavy processes ran concurrently"* (report, section B7). **No script in the
repository actually orchestrates that concurrent launch** via `ProcessPoolExecutor`/
`multiprocessing` — grep across all `p56*.py` files returns nothing — so the concurrent load
described was very likely produced by manually/shell-launched separate Python processes, not a
harness this audit can re-inspect as code. **The report does not attribute the crash to a
specific mechanism** — no mention of shared temp files, shared log paths, or `.nl`/`.sol`
collisions anywhere in the searched material in connection with this crash; it is described only
as generic IPOPT/MA97 resource contention on an 8-core machine running several full evaluations
at once. **The crash was subsequently found not reproducible**: *"`se|ALL|x19` crashed IPOPT
under both starts... **Withdrawn by P5.6-C.** That crash was not reproducible... reclassified as
a transient solver failure"* (same report).

Direct, contemporaneous evidence that this class of harness *did* worry about log-file collision
under concurrency: `p56a_oracle.py`'s `fresh_planning(eval_id)` docstring states explicitly that
*"IPOPT is configured with `file_append='yes'` and would otherwise append every evaluation's log
into the same file"*, and mitigates it by giving every `eval_id` its own private `logs_dir` — a
real, named log-collision hazard from **the exact same `file_append='yes'` + shared-`logs_dir`
mechanism** this audit's Part 1 flags in `network.py`, but at the evaluation layer, not the block
layer, and **pre-empted rather than diagnosed as the `SOLVER_CRASH`'s cause**.

### Global/module-level mutable state

No hazardous module-level mutable dict/list/counter touched by the DSO solve path was found in
`shared_resources_planning.py`, `network.py` or `shared_energy_storage_data.py` (targeted grep
plus manual reading of the parallel functions). No production `os.chdir()` calls exist (only
comments describing a historical harness-level `os.chdir` defect that P5.15-F already removed
the need for — `shared_energy_storage_data.py:1072`, `shared_resources_planning.py:47`).
`p56a_oracle.py` keeps one process-level cache, `_BASELINE` (module-level global, lazily
populated, documented read-only after first population) — safe under `ProcessPoolExecutor`'s
`spawn` start method because each spawned process gets its own fresh, empty `_BASELINE`, but
would be a hazard under a `fork`-based or thread-based pool. No other global cache of this kind
was found in the audited production files.

---

## PART 2 — Where the ~39 s cycle goes (zero solves)

**Run**: `data/SRP1/Results/P515S35_REF_run/` (477 cycles, `parallel_execution=False`).
**Sampled cycles** (12, spread across the run): 1, 25, 50, 100, 150, 200, 250, 300, 350, 400,
450, 477 — all 12 had complete, unambiguous log data (`missing_data_notes.json` is empty).

**Mapping log files to cycles**: every `(case name, year, day)` primary IPOPT log
(`optim_log_{case}_{year}_{day}.log`) is appended across the whole run (`file_append='yes'`) and
contains **exactly 478** `"Total seconds in IPOPT"` occurrences for **every one of the 48
DSO+TSO blocks** — verified by direct count on all 48 files. This equals **1 pre-loop
initialization solve + 477 ADMM-cycle solves**, so occurrence index `c` (1-based) in file order
is cycle `c`'s primary solve. The ESSO logs are one file per `(node, cycle)`
(`optim_log_esso_node{node}_cycle{cycle:03d}.txt`, plus one `_init.txt` per node — 3×478 = 1434
files total, matching the heartbeat's `esso_solves_so_far: 1434` at cycle 477), so no
occurrence-counting is needed there. Recovery/tier-2 retries are cross-checked against
`network_failures_baseline.jsonl` (298 records, all `class` ∈ {`recovered_tier1`,
`recovered_tier2`}, 0 unrecovered): for each `(agent, node_id, network_name, year, day)` key, the
records are sorted by cycle and their position gives the occurrence index into that key's
`_recovery.log` / `_recovery_tier2.log` file (14 of the 298 were tier-2). No `"Total CPU secs in
IPOPT (w/o function evaluations)"` / `"in NLP function evaluations"` lines are present in this
IPOPT 3.14.18 build's summary — only the aggregate `"Total seconds in IPOPT"` line, used
throughout.

### Per-sampled-cycle table (seconds)

| cycle | IPOPT DSO | IPOPT TSO | IPOPT ESSO | IPOPT total | wall (stdout) | overhead | overhead % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1   | 11.912 | 0.689 | 0.030 | 12.631 | 30.32 | 17.689 | 58.3% |
| 25  | 15.256 | 1.865 | 0.036 | 17.157 | 40.24 | 23.083 | 57.4% |
| 50  | 15.191 | 0.734 | 0.034 | 15.959 | 38.75 | 22.791 | 58.8% |
| 100 | 16.438 | 0.965 | 0.031 | 17.434 | 37.57 | 20.136 | 53.6% |
| 150 | 11.366 | 1.010 | 0.038 | 12.414 | 33.92 | 21.506 | 63.4% |
| 200 | 19.357 | 0.706 | 0.040 | 20.103 | 41.92 | 21.817 | 52.0% |
| 250 | 18.900 | 0.725 | 0.042 | 19.667 | 41.52 | 21.853 | 52.6% |
| 300 | 11.075 | 0.699 | 0.042 | 11.816 | 31.49 | 19.674 | 62.5% |
| 350 | 10.929 | 0.710 | 0.042 | 11.681 | 33.15 | 21.469 | 64.8% |
| 400 | 10.389 | 1.233 | 0.042 | 11.664 | 31.35 | 19.686 | 62.8% |
| 450 | 12.530 | 0.778 | 0.041 | 13.349 | 33.53 | 20.181 | 60.2% |
| 477 | 10.945 | 0.654 | 0.041 | 11.640 | 33.50 | 21.860 | 65.3% |
| **median** | 12.221 | 0.730 | 0.041 | 12.990 | 33.73 | 21.488 | **59.5%** |

Full per-block breakdown for every one of the 51 blocks × 12 cycles is in
`per_cycle_table.json`; the table above is `per_cycle_table_compact.json`.

**Overhead is large, dominant, and remarkably stable in absolute terms** (19.7–23.1 s across all
12 cycles, median 21.5 s) **even though IPOPT time itself varies 2×** (11.6–20.1 s) with the
recovery/tier-2 retry load of that particular cycle. This is consistent with the overhead being
mostly a roughly fixed per-cycle cost (NL write + `.sol` read + Python parameter updates +
ADMM residual/consensus computation across 51 blocks) rather than something that scales with
solver difficulty.

**What the harness captures**: none of `P515S35_REF_run`'s artifacts (`heartbeat_baseline.json`,
the JSONL sidecars) carry a per-cycle timing breakdown finer than the single `wall_s` in the
heartbeat and the `"Iteration N: X s"` line already used above. That line comes from the
**unmodified production print statement** `shared_resources_planning.py:3007`
(`iter_start = time.time()` at `:2507`, right before the DSO solve step, through `iter_end =
time.time()` at the very end of the cycle body, after DSO+TSO+ESSO solves, ADMM updates and
convergence checks) — no `cycle_callback`/heartbeat hook was found inside the production ADMM
loop (grep for `cycle_callback`/`heartbeat` in `shared_resources_planning.py` returns nothing),
so the printed wall time is pure production execution time, unaffected by the harness's own
capture (which happens in the harness's driver process around the call, not inside it).

**Instrumentation that would separate NL write/read from Python ADMM updates (not added)**: wrap
`solver.solve(...)` in `network.py:_run_smopf_solver_attempt` (`:604-611`) and
`shared_energy_storage_data.py:_run_solver_attempt` (`:1135-1153`) with `time.perf_counter()`
immediately before and after the call, and separately time the `model.solutions.load_from(result)`
call (`network.py:738`) — the gap between "wrapper-measured wall time for one `solver.solve()`
call" and "IPOPT's own reported `Total seconds in IPOPT`" isolates Pyomo's NL-write (which
happens inside `solver.solve()`, before the IPOPT subprocess starts) plus process-launch
overhead; the gap between that and the surrounding `fix_or_set`/`set_value` update loops
(`shared_resources_planning.py:5462-5486`) isolates the pure-Python ADMM parameter-update cost.
This was not added, per the zero-solve/no-harness-edit scope of this task.

### Verdict — does Pyomo NL write/read overhead plausibly dominate?

**Plausibly yes, but it cannot be cleanly separated from Python ADMM bookkeeping with the
existing artifacts.** The measured `overhead` (median 21.5 s, 59.5% of the 33.7 s median cycle
wall) is **everything that is not inside IPOPT's own reported time**: Pyomo's NL write (which
happens inside `solver.solve()`, before IPOPT starts, and is therefore never counted in "Total
seconds in IPOPT"), `.sol` read/`model.solutions.load_from`, the per-block parameter-update loops
(`fix_or_set`/`set_value`, 51 blocks × several periods × several variables each cycle), and the
ADMM residual/consensus computation (`get_admm_residual_metrics` and friends, which needs
**all** 51 blocks' results before it can run and is therefore inherently serial regardless of
how the solves themselves are parallelized). Recording "in-memory NLP interface
(PyNumero/cyipopt)" as a later, separate item (with a numerical-equivalence gate) is
**supported** by this evidence — the overhead is large enough, and stable enough across
cycles of varying solver difficulty, to be a real and addressable target — but the specific
claim "NL write/read alone accounts for X% of the 59.5%" is **not** established here and would
require the instrumentation described above (not added).

### Theoretical parallel speed-up (from block-type timings; LPT-greedy worker partition)

For each sampled cycle, the 36 DSO / 12 TSO / 3 ESSO per-block IPOPT times were partitioned
across `W` workers by longest-processing-time-first greedy bin-packing (a standard, easily
reproducible heuristic, not an optimal solve), giving
`block-type parallel wall ≈ max load over the W workers` for that type; DSO→TSO→ESSO ordering is
preserved (summed, since each type must fully finish before the next begins under the
unmodified ADMM dependency), and the **measured overhead is conservatively assumed unchanged and
fully serial** (a pessimistic assumption — see caveat below).

| workers | median DSO wall | median TSO wall | median ESSO wall | + overhead (unchanged) | projected cycle wall | **projected speed-up** | Amdahl bound* |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.221 | 0.730 | 0.041 | 21.488 | 33.725 | 1.000× | 1.000× |
| 3 | 4.193  | 0.245 | 0.014 | 21.488 | 25.755 | **1.360×** | 1.370× |
| 4 | 3.131  | 0.191 | 0.014 | 21.488 | 24.689 | **1.428×** | 1.436× |
| 8 | 2.113  | 0.113 | 0.014 | 21.488 | 23.408 | **1.483×** | 1.549× |

(exact medians over the 12 sampled cycles: `summary.json`; per-cycle detail:
`speedup_projection.json`. \*Amdahl bound uses `f_serial = overhead / wall` measured per cycle,
`speedup ≤ 1 / (f_serial + (1-f_serial)/W)`.)

**This bound is pessimistic by construction and falls well short of the Addendum 18 target
(~39 s → 8–10 s, i.e. ~4–5×).** It assumes only the IPOPT calls themselves are parallelized and
that the ~21.5 s of overhead stays exactly as measured and fully serial. In the persistent-worker
design the Planner has proposed, each worker performs its own block's parameter update + NL
write + solve + `.sol` read **inside the worker**, so a substantial part of that 21.5 s (the
per-block update/NL-write/read components) would plausibly parallelize along with the solves
themselves — only the cross-block ADMM residual/consensus step and incidental Python bookkeeping
would remain inherently serial. **This audit cannot size that split** without the instrumentation
described above (not added, per scope), so the true achievable speed-up under a persistent-worker
redesign lies somewhere between this pessimistic ~1.5× (8 workers) and a more optimistic figure
bounded above by `wall / (residual_computation_time + incidental_overhead)`, which is currently
unmeasured. **Recommendation to the Planner: instrument the split described above (as a
follow-up, not as part of this READ-ONLY audit) before sizing the persistent-worker design's
expected win**, since the current evidence only supports "overhead dominates" and "solve-only
parallelism under-delivers against the 8–10 s target," not a specific achievable number.

---

## Concrete list for a persistent-worker implementation

1. **Temp directory**: no change forced by current evidence (Pyomo's default `TempfileManager`
   naming is already per-call-unique and safe), but a private tmp dir per worker is still
   prudent given the design brief asks for it explicitly, and removes even a latent risk under
   any future `keepfiles=True`/`tmpdir=` change.
2. **Logs**: `network.py:_create_smopf_solver` must gain the same isolation `_create_solver`
   already has for the ESSO (`shared_energy_storage_data.py:1058-1107`) — stamp by worker/node
   and cycle, resolve against a private or at least existence-checked path, and stop relying on
   `file_append='yes'` as an implicit collision tolerance mechanism.
3. **Guard accounting**: `SolveProfileGuard` must be installed **inside every worker process**
   (e.g. via a `ProcessPoolExecutor(initializer=...)`), with each worker's counts returned to and
   summed by the parent; the current class-level monkeypatch in the parent is invisible to
   `spawn`-started children and would silently misreport both zero-solve and bounded-solve
   claims.
4. **Recovery**: already process-safe (pure function of `(network, model, params)`) — no change
   forced, but any future diagnostics-sink parameter added to the DSO recovery path (mirroring
   the ESSO's `solver_recovery_diagnostics`) must be returned from the worker explicitly, not
   assumed mutated-in-place, exactly as the ESSO's list would be lost today if the ESSO were ever
   parallelized.
5. **Snapshots**: FrozenSMOPF snapshot writing (`failure_snapshot_callback`,
   `pre_solve_snapshot_callback`) must be threaded through the persistent-worker path; it is
   currently entirely absent from `update_and_solve_dso`.
6. **Result ordering**: already safe (node/block-keyed dicts assembled by `as_completed()`, not
   append order) — preserve this property when extending to full `(node, year, day)` blocks.
7. **Determinism**: single-threaded linear solver per worker (`OMP_NUM_THREADS=1`, MA57 or
   single-thread MA97) is not currently enforced anywhere — this audit found no thread cap at
   all, consistent with the P5.6-B7 report's own observation that "`W` workers means `W`
   simultaneous IPOPT processes on top of whatever threading MA97 uses." This must be added
   explicitly for the two-cycle bitwise-identical gate to be meaningful.
8. **Sizing the win**: per the Part 2 verdict, do not assume the persistent-worker redesign
   reaches 8–10 s from the ~1.5× solve-only bound computed here; measure the NL-write/`.sol`-read
   vs. residual-computation split first (instrumentation described above) to bound the
   achievable speed-up before committing to the 8–10 s target number.

---

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` (Addendum 18, lines 770-802)
- `shared_resources_planning.py` (parallel functions ~3550-3720 and ~5311-5500; log-dir wiring
  ~7330-7444; ADMM loop timing ~2495-3010)
- `network.py` (`_create_smopf_solver`, `_run_smopf`, recovery tiers, ~480-940)
- `network_data.py` (`NetworkData.optimize`, ~54-68)
- `shared_energy_storage_data.py` (`_create_solver`, `optimize`, ~40-190, ~1000-1160)
- `p513_solve_profile_guard.py` (full file)
- `p56a_oracle.py` (`fresh_planning`, `load_baseline`, `_BASELINE`, ~90-150)
- `P5_6_NONLINEAR_DERIVATIVE_FREE_PLANNING_REPORT.md` (B7 section, Verdict section)
- `P5_11_STABILIZED_ORACLE_CONSOLIDATION_REPORT.md` (P5.6-B7 reference)
- `REVISION_CONTEXT.md` (all `P5.6-B` occurrences)
- `data/SRP1/SRP1.json`, `data/SRP1/case33_{1,2,3}/case33_{1,2,3}_params.json`,
  `data/SRP1/case9/case9_params.json` (topology, `output_file`/`linear_solver` options)
- `data/SRP1/Results/P515S35_REF_run/` (`stdout_baseline.log`, `heartbeat_baseline.json`,
  `network_failures_baseline.jsonl`)
- `data/SRP1/Results/P56A/evals/p515s35ref_baseline/logs/` (all DSO/TSO/ESSO per-solve IPOPT logs)

## Files created

- `p515_s36_parallel_audit.py`
- `data/SRP1/Results/P515S36/parallel_audit/provenance.json`
- `data/SRP1/Results/P515S36/parallel_audit/per_cycle_table.json`
- `data/SRP1/Results/P515S36/parallel_audit/per_cycle_table_compact.json`
- `data/SRP1/Results/P515S36/parallel_audit/speedup_projection.json`
- `data/SRP1/Results/P515S36/parallel_audit/summary.json`
- `data/SRP1/Results/P515S36/parallel_audit/code_audit_findings.json`
- `data/SRP1/Results/P515S36/parallel_audit/missing_data_notes.json` (empty — no gaps)
- `data/SRP1/Results/P515S36/parallel_audit/manifest_sha256.json`
- `WORKER_REPORT_S36_PARALLEL_AUDIT.md` (this file)

## Validation

- Script executed once (`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python
  p515_s36_parallel_audit.py`), exit clean, no missing-data notes for any of the 12 sampled
  cycles.
- `SolveProfileGuard.verify(expected_solves=0, expected_execs=0)` returned no failures — the
  audit performed exactly zero solves, enforced (not asserted).
- Cross-checked primary-log occurrence counts: all 48 DSO/TSO logs have exactly 478 occurrences
  (1 init + 477 cycles); ESSO file count (1434 = 3 × 478) matches
  `heartbeat_baseline.json`'s `esso_solves_so_far: 1434` at `cycle: 477`.
- Cross-checked failure/recovery reconciliation against `network_failures_baseline.jsonl`
  (298 records, 14 tier-2) — no unmatched recovery/tier-2 occurrence in any sampled cycle.
- Did not re-run any harness onto cited evidence; only read existing files under
  `P515S35_REF_run` and `P56A/evals/p515s35ref_baseline`.

## Unexpected findings

- **Overhead dominates the cycle wall time (median 59.5%, ~21.5 s of ~33.7 s)** and is
  remarkably stable across cycles regardless of how much IPOPT solve time or how many recovery
  retries that cycle needed — this was not something the audit set out to look for but is the
  single largest number in this report, and directly bears on whether the 8–10 s target is
  reachable by parallelizing solves alone (it is not, per the pessimistic-bound calculation
  above).
- The ESSO's log-isolation code (`shared_energy_storage_data.py:1058-1107`) is materially more
  defensive than the DSO/TSO path it was modeled after avoiding, which is worth noting as a
  precedent to copy rather than re-derive.
- The P5.6-B `SOLVER_CRASH` this task asked to investigate turned out to be at a different layer
  (candidate-evaluation, not within-cycle block) than the code this audit was asked to inspect,
  and was itself later withdrawn as non-reproducible — so it is weak evidence for anything about
  the Addendum 18 design, beyond the generic "unthrottled concurrent IPOPT/MA97 processes on a
  fixed core count is a real, previously-observed risk class" it does support.

## Remaining issues

- The NL-write/`.sol`-read vs. Python-ADMM-bookkeeping split within the ~21.5 s overhead is
  unmeasured; the instrumentation to separate them is described above but was not added (scope).
- Only the REF run was analyzed in full; the PT run (`P515S35_PT_run`, 150 cycles) was inspected
  only enough to confirm it has the same log/heartbeat structure, as a cross-check that the REF
  run's pattern is not an artifact of one particular run.
- The theoretical speed-up table uses a greedy LPT heuristic for the worker partition, not an
  optimal bin-packing solve; with only 36/12/3 items per type this is expected to be very close
  to optimal, but was not verified against an exact partition.

## Questions for Planner

- Should the follow-up NL-write/`.sol`-read instrumentation (Part 2's "not added" item) be
  authorized as a small, separate zero-solve-adjacent task before sizing the persistent-worker
  redesign's expected speed-up, given the ~1.5×-at-8-workers pessimistic bound falls well short
  of the 8–10 s target if only solves are parallelized?
- Should the DSO/TSO log-path hardening (item 2 in the concrete list) be treated as a
  prerequisite fix (independent of the broader persistent-worker redesign), given it is a
  one-file, localized change modeled directly on the ESSO's already-accepted pattern?
