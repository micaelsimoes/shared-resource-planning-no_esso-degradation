# Worker Report — Recovery policy (Part 1) + regression (Part 2) + cycle-38 block (Part 3)

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 7, items 1 and 3.
Task HEAD stated by the Planner: `c3bbbebb`. Actual HEAD at task start: `76ab3f99`
(one intervening commit, `P5.15: freeze ablation A/B criteria before any ablation
run` — a frozen-spec JSON only, no code files; does not conflict with anything
below; noted for the record, not treated as a problem).

No ADMM campaign was run. `p515_g_g1_g4_admm_gates.py`'s `__main__` was never
invoked (only its functions, by import). No file under `data/` was edited except
new files under the new directory `data/SRP1/Results/P515R/` — **with one
disclosed exception**, see "Deviations from the data/ restriction" below.
`.p515_g_gate.lock` was not created. Nothing was committed.

## Task received

Implement an explicit, production `recovery.enabled` / `recovery.tier2_enabled`
policy for the network and ESSO solve paths (independent of whether
`recovery_options` is populated), add a tier-2 retry (cold + `mu_strategy=adaptive`)
after a failed tier-1 cold retry, extend the G1–G4 harness's failure parser to
classify `recovered_tier1` / `recovered_tier2` / `unrecovered` / `not_attempted`
while keeping `recovered` = tier1+tier2, and re-validate against G1's committed
stdout. Provide a harness-side way to set the policy per network without editing
case files. Run a bitwise regression (R1, R2) against pre-existing committed
evidence. Run five single-solve arms (P0–P4) on the preserved cycle-38 DSO
pre-solve fixture and classify intrinsic / path-dependent / tolerance-limited.
Report.

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` (Addendum 7 and the preceding addenda for context)
- `network.py` (`_is_recoverable_network_failure`, `_create_smopf_solver`,
  `_run_smopf_solver_attempt`, `_run_smopf`, `_print_network_failure_context`)
- `shared_energy_storage_data.py` (`_is_recoverable_shared_ess_failure`,
  `_create_solver`, `_run_solver_attempt`, `_optimize`)
- `solver_parameters.py` (`SolverParameters`, `_read_solver_parameters`)
- `network_parameters.py`, `shared_energy_storage_parameters.py` (where
  `SolverParameters` is instantiated / `read_solver_parameters` is called)
- `shared_resources_planning.py` (`_write_solver_recovery_diagnostics_to_excel`)
- `p515_g_g1_g4_admm_gates.py` (`_scan_network_failures` and its regexes,
  `_new_network_event`, `run_admm_arm`, `_construct_arm_planning`,
  `run_ladder_init`/`__main__` — read only, never executed)
- `p59_rho.py` (`apply_rho_to_params`, `set_adaptive_penalty` — the existing
  "set X on a deep-copy planning object" convention `set_recovery_policy` follows)
- `p512_x_comparator_replay.py` (fixture-load / net-params-reconstruction pattern
  reused for Part 3)
- `p515_h_tol_remedy_check.py`, `p515f_t3_tol_remedy_recheck.py` (reused by
  import for R1)
- Case files: `data/SRP1/case33_1/case33_1_params.json`,
  `case33_2/case33_2_params.json`, `case33_3/case33_3_params.json`,
  `case9/case9_params.json`, `SharedESS/SRP1_ESS_Params.json`,
  `data/SRP1/SRP1.json` (node↔case-name map: `case33_1`→node 5, `case33_2`→node 7,
  `case33_3`→node 9)
- Committed evidence read (not modified): `data/SRP1/Results/P515G1/stdout_control.log`,
  `g_control.json`, `leak_classification_control.jsonl`, `network_failures_control.jsonl`;
  `data/SRP1/Results/P5151/tol_remedy_check_summary.json`;
  `data/SRP1/Results/P515G3F_r2/results/FrozenSMOPF/frozen_DSO_node7_case33_2_2035_Autumn_cycle38.pkl`
  and `data/SRP1/Results/P515G3F_r2/network_failures_g3_full_node7.jsonl`

## Files modified (production + harness)

- `solver_parameters.py`
- `network.py`
- `shared_energy_storage_data.py`
- `shared_resources_planning.py`
- `p515_g_g1_g4_admm_gates.py`
- `p59_rho.py`

No case file under `data/SRP1/` was edited (verified: `git status` shows none of
`case33_1/2/3_params.json`, `case9_params.json`, `SRP1_ESS_Params.json` touched).
The `limited-memory` `hessian_approximation` entries in `recovery_options` were
**not** removed from any case file (Addendum 7: only after this change lands and
is accepted — out of scope for this task).

## Changes made

### Part 1 item 1 — explicit `recovery.enabled`

`solver_parameters.py`: `SolverParameters.__init__` gets
`self.recovery_enabled = True`, `self.recovery_tier2_enabled = True` (so the
defaults hold even for a `SolverParameters` instance that never calls
`read_solver_parameters`). `_read_solver_parameters` reads an optional nested
`recovery` block (`solver_data.get('recovery') or {}`) with keys `enabled` /
`tier2_enabled`, defaulting to `True`/`True` when absent — no case file needs to
declare it.

`network.py:_is_recoverable_network_failure` and
`shared_energy_storage_data.py:_is_recoverable_shared_ess_failure`: the
`if not solver_params.recovery_options: return False` guard (which made
eligibility depend on `recovery_options` being non-empty) is replaced by
`if not getattr(solver_params, 'recovery_enabled', True): return False` — the
termination-condition check is unchanged. `case33_1` (no `recovery_options` in
its case file) is now eligible by default.

The two retry-options builders (`{k: v for k, v in params...recovery_options.items() if k != 'hessian_approximation'}`)
now guard against `recovery_options` being `None`:
`(params.solver_params.recovery_options or {}).items()` / `(params.recovery_options or {}).items()`
— required once eligibility no longer implies `recovery_options` is truthy.

### Part 1 item 2 — tier-2 retry

`network.py:_run_smopf` and `shared_energy_storage_data.py:_optimize`: after a
failed tier-1 cold retry, if `recovery.tier2_enabled` is True **and** the tier-1
failure is itself of a recoverable termination class
(`_is_recoverable_network_failure`/`_is_recoverable_shared_ess_failure` re-applied
to the tier-1 result), one further solve: cold start (suffixes cleared) +
`mu_strategy = adaptive`, same other recovery options (tier-1's `recovery_options`
dict, i.e. case-file `recovery_options` minus `hessian_approximation`, plus
`warm_start_init_point = 'no'`). Log suffix `recovery_tier2`.

New, tier-1-distinct print strings (verified byte-identical to the existing
tier-1 strings where tier-1 alone fires — see Validation):
- `[INFO] Retrying network solve (tier 2: cold, mu_strategy=adaptive) for <ctx>, with <opts>.`
- `[INFO] Network tier-2 recovery solve succeeded for <ctx>.`
- (failure) `[WARNING] Network tier-2 recovery solve did not converge for <ctx>: ...`
  (via `_print_network_failure_context(..., attempt_label='tier-2 recovery solve')`)
- ESSO: `[INFO] Retrying Shared ESS solve (tier 2: cold, mu_strategy=adaptive) for <ctx>, with <opts>.`,
  `[INFO] Shared ESS tier-2 recovery solve succeeded for <ctx>.`,
  `[WARNING] Shared ESS tier-2 recovery did not converge for <ctx>: ...`

### Part 1 item 3 — counting

`shared_energy_storage_data.py:_optimize`'s `diagnostic_sink.append({...})` now
records `'tier': 'tier2' if tier2_attempted else 'tier1'`, plus
`tier2_attempted`, `tier2_result`, `tier2_options`, `tier2_log`.
`shared_resources_planning.py:_write_solver_recovery_diagnostics_to_excel` gets
four new columns (`Recovery Tier`, `Tier-2 Attempted`, `Tier-2 Result`, `Tier-2 Log`).

`p515_g_g1_g4_admm_gates.py:_scan_network_failures`: two new regexes
(`_NET_RETRY_TIER2_RE`, `_NET_RECOVER_TIER2_OK_RE`); the `'recovery solve'` /
`'recovery solve'`-success branches now set class `recovered_tier1` (was
`recovered`); a new `'tier-2 recovery solve'` branch (fail) and a new
`_NET_RECOVER_TIER2_OK_RE` branch (success) set `recovered_tier2` /
(`unrecovered` if tier-2 also fails, on the same already-open/-appended event
object — never a second row for the same key). `_NET_LOG_RE`'s dispatch checks
`'tier-2' in label` **before** `'recovery' in label` (the label `'tier-2 recovery
solve'` contains the substring `'recovery'`, so order matters). `run_admm_arm`'s
summary now reports `recovered_tier1` / `recovered_tier2` / `unrecovered` /
`not_attempted` / `indeterminate`, plus `recovered = recovered_tier1 + recovered_tier2`
kept for backward comparison.

### Part 1 item 4 — harness-side policy setter (no case-file edit)

`p59_rho.py:set_recovery_policy(planning, enabled=True, tier2_enabled=True, node_overrides=None, include_tso=True, include_esso=True)`
— mirrors `apply_rho_to_params`/`set_adaptive_penalty` in the same file. Sets
`recovery_enabled`/`recovery_tier2_enabled` on
`planning.transmission_network.params.solver_params`,
`planning.distribution_networks[node_id].params.solver_params` for every node,
and `planning.shared_ess_data.params.solver_params`, with `node_overrides`
(keyed by `node_id`, and `'TSO'`/`'ESSO'` for the other two) for per-network
exceptions. Returns the previous values for restoration. Documented call for the
G1-equivalent policy used in R2:
```python
RH.set_recovery_policy(planning, enabled=True, tier2_enabled=False,
                        node_overrides={5: {'enabled': False}})
```
(`5` = `case33_1`'s `connection_node_id`, `data/SRP1/SRP1.json`.)

## Commands / experiments run

All via the canonical interpreter
`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`, foreground,
stderr captured, one at a time.

1. **Parser re-validation** — `p515_g_g1_g4_admm_gates._scan_network_failures`
   applied directly to the committed `data/SRP1/Results/P515G1/stdout_control.log`
   (read-only; output written only to
   `data/SRP1/Results/P515R/g1_parser_revalidation.json`).
2. **Part 1 unit checks** — zero-solve checks of `SolverParameters` defaults/JSON
   parsing, `_is_recoverable_network_failure`/`_is_recoverable_shared_ess_failure`
   eligibility with mock `result`/`params` objects, and `set_recovery_policy` on
   mock planning objects (no real planning construction, no data/ writes outside
   `data/SRP1/Results/P515R/part1_policy_unit_checks.json`).
3. **R1** — re-ran `p515_h_tol_remedy_check.ARMS`/`run_arm` (the same two arms
   `p515f_t3_tol_remedy_recheck.py` re-runs) by import, guarded
   (`SolveProfileGuard`, permitted site `shared_energy_storage_data.py:_run_solver_attempt`,
   declared count 6), log root and report redirected to
   `data/SRP1/Results/P515R/r1_t3_recheck/` (P515F, which a committed report
   cites, was never touched).
4. **R2** — 2-cycle C* control run through `run_admm_arm('r2_g1policy', ...,
   eval_id='p515r_r2_g1policy', num_max_iters_override=2)`, with G1-equivalent
   recovery policy applied via a process-local monkeypatch of
   `p515_g_g1_g4_admm_gates._construct_arm_planning` calling
   `RH.set_recovery_policy` right after construction, restored in `finally`.
   Fresh root `data/SRP1/Results/P515R/r2_g1policy/`, fresh eval id.
5. **Part 3** — five single-solve arms (P0–P4) on the cycle-38 fixture, each from
   an independent `deepcopy` of the pristine fixture model, each in its own fresh
   `O.fresh_planning` eval (`p515r_c38_p0`…`p4`) with `net.logs_dir` redirected
   into `data/SRP1/Results/P515R/part3_cycle38/logs/<eval>/`. P0/P4 through
   `Network.run_smopf` (full production orchestration); P1/P2/P3 through
   `network._run_smopf_solver_attempt` (single attempt, `option_overrides`).
   Guarded per arm (`SolveProfileGuard`, permitted site
   `network.py:_run_smopf_solver_attempt`).
6. Evidence manifest: sha256 inventory of every file under
   `data/SRP1/Results/P515R/` written to
   `data/SRP1/Results/P515R/evidence_manifest_sha256.json`.

## Results

### Parser re-validation (G1's committed stdout, current/fixed scanner)

`recovered_tier1=12, recovered_tier2=0, unrecovered=0, not_attempted=6, indeterminate=0`
→ `recovered=12`. not_attempted cycles: `{7, 39, 43, 46, 48, 49}`. **Matches the
required 12/0/6 and cycle set exactly.** (`recovered_tier2=0` because the
scanned text predates tier 2 entirely — expected, not evidence tier 2 works;
that evidence is Part 3.)

Note: `g_control.json`'s own baked-in `network_failures_summary` field reads
`{'recovered': 12, 'unrecovered': 4, 'not_attempted': 0}` — this is the
pre-existing, already-documented "second G1 bug" (6 failed cycles collapsed into
4 rows by a since-fixed keying bug; the docstring at
`p515_g_g1_g4_admm_gates.py:539-599` records this explicitly). Not something
this task fixed or needed to fix; the stdout re-parse (above) is the corrected,
authoritative classification and is what R2 is compared against.

### Part 1 unit checks (zero solves)

- `SolverParameters(require_path=False)` defaults: `recovery_enabled=True`,
  `recovery_tier2_enabled=True`.
- Reading a `case33_1`-like solver block (no `recovery_options`, no `recovery`
  key): `recovery_options=None`, `recovery_enabled=True`, `recovery_tier2_enabled=True`
  — i.e. now eligible (previously ineligible because `recovery_options` was falsy).
- Reading an explicit `{'recovery': {'enabled': False, 'tier2_enabled': False}}`
  block: both flags read back `False`.
- `network._is_recoverable_network_failure` with a `case33_1`-like
  `SolverParameters` (`recovery_options=None`, `recovery_enabled=True`) and a
  `maxIterations` fake result: **`True`** (was `False` before this change).
  With `recovery_enabled=False`: `False`. With `optimal` termination: `False`.
- `shared_energy_storage_data._is_recoverable_shared_ess_failure`: eligible with
  no `recovery_options` and `node_id` set; never eligible for the master problem
  (`node_id=None`).
- `p59_rho.set_recovery_policy` on a mock planning object with
  `enabled=True, tier2_enabled=False, node_overrides={5: {'enabled': False}}`:
  TSO/DSO7/DSO9/ESSO → `(True, False)`; DSO5 (case33_1) → `(False, False)`.
  Previous values correctly captured as `(True, True)` everywhere.

Full evidence: `data/SRP1/Results/P515R/part1_policy_unit_checks.json`.

### R1 — single-solve bitwise regression

`GUARD_COUNTS {'permitted_solve': 6, 'permitted_exec': 6, 'blocked_solve': 0, 'blocked_exec': 0}`,
`GUARD_VERIFY_FAILURES []`, **`max_relative_difference = 0.0`** across both arms
(`control`, `remedy_h`) × all three active nodes × the five fields
(`mu_final`, `s_obj`, `complementarity_ratio_max`, `spurious_throughput_measured`,
`spurious_throughput_bound`). **Bitwise pass.**
Evidence: `data/SRP1/Results/P515R/r1_t3_recheck/r1_t3_recheck_report.json`.

### R2 — two-cycle, G1-equivalent-policy regression

Policy applied (verified before/after):
TSO/DSO7(case33_2)/DSO9(case33_3)/ESSO → `(enabled=True, tier2=True)` → `(True, False)`;
DSO5(case33_1) → `(True, True)` → `(False, False)`.

Run summary (from production's own print, `run_admm_arm`):
`recourse=1832134032.524669 cycles=2 solves=154 local_failures=0 wall=135s`;
one network block: `{'recovered_tier1': 1, 'recovered_tier2': 0, 'unrecovered': 0,
'not_attempted': 0}` — the case9 2035 Winter TSO recovery at initialization the
brief named, now classified `recovered_tier1` (a tier-1 recovery fired and
succeeded, as G1 had). `case33_1` had no failures in cycles 1–2 (its own known
failures are at cycles 7/39/43/46/48/49, outside this 2-cycle window), so the
G1-equivalent ineligibility on node 5 was never exercised by this window — it is
present in the policy but not load-bearing for this particular comparison.

Comparison against `data/SRP1/Results/P515G1/g_control.json` /
`leak_classification_control.jsonl`:
- **Cycle-trajectory rows 1 and 2, every field** (recourse, objective_change_abs,
  objective_tolerance, all primal/dual residuals and ratios, slack/state-step
  norms, recourse_change, nonfinite flag): **0 mismatches.**
- **ESSO complementarity detector, `init`/`001`/`002`, all 3 nodes, 9 fields each**
  (`mu_scaled`, `mu_unscaled`, `obj_scaled`, `obj_unscaled`,
  `complementarity_ratio_max`, `spurious_throughput_measured`, `lg_mu_terminal`,
  `mu_barrier_scaled`, `s_obj`): **0 mismatches.**
- **`n_mismatches = 0` overall. `bitwise_pass = True`.**

`guard.counts['permitted_solve'] = 154` against the harness's own
`51 * cycles + 51` identity check (153) — `identity_holds=False`, i.e. exactly
**one extra solve**, which is precisely the tier-1 recovery cold retry for
case9/2035/Winter (expected and consistent with the network-failures summary
above; not a discrepancy).

Full evidence: `data/SRP1/Results/P515R/r2_g1policy/g_r2_g1policy.json`,
`r2_comparison_vs_g1.json`, `recovery_policy_applied.json`,
`leak_classification_r2_g1policy.jsonl`, `network_failures_r2_g1policy.jsonl`,
`stdout_r2_g1policy.log`.

### Part 3 — cycle-38 block

Fixture `frozen_DSO_node7_case33_2_2035_Autumn_cycle38.pkl`:
sha256 **before = after** (`4499cf99a4341eb99929f53bdc7cd8ef7bfd2e6edad1dcb9eab93792d9c6fe47`,
verified in the same script, load-only, never written back). `tol` in force for
`case33_2`: **1e-5** (read from the case file); P3 used **1e-4**.

| arm | description | termination | iters (primary/tier1/tier2) | max constr. viol. (unscaled, final) | lg(mu) (final) | wall (s) | solves |
|---|---|---|---|---|---|---|---|
| P0 | warm primary → tier-1 cold (tier2 OFF) — reproduction of G3-full | **maxIterations** (both) | 500 / 500 | 6.20e-05 / 5.37e-05 | -8.0 / -8.0 | 5.42 | 2 |
| P1 | cold only | **maxIterations** | 500 | 5.37e-05 | -8.0 | 2.77 | 1 |
| P2 | cold + `mu_strategy=adaptive` | **optimal** | 66 | 1.31e-07 | -7.2 | 0.59 | 1 |
| P3 | cold + `tol` one decade looser (1e-4) | **maxIterations** | 500 | 5.37e-05 | -8.0 | 2.73 | 1 |
| P4 | warm primary → tier 1 → tier 2 (new policy) | **optimal** (tier 2) | 500 / 500 / 66 | 6.20e-05 / 5.37e-05 / 1.31e-07 | -8.0 / -8.0 / -7.2 | 6.00 | 3 |

**Reproduction gate (P0): PASSED** — primary `maxIterations` (500 iters),
tier-1 recovery `maxIterations` (500 iters), matching
`data/SRP1/Results/P515G3F_r2/network_failures_g3_full_node7.jsonl`'s recorded
`primary_termination: maxIterations`, `recovery_termination: maxIterations`.

**Classification (predeclared): `path-dependent`** — the warm primary fails and
a cold arm (P2) succeeds; not `intrinsic` (P2 succeeds, so not all of P1/P2/P3
fail); not `tolerance-limited` (P3 alone did not succeed — P3 failed and P2 did).
Note the nuance: it is specifically `mu_strategy=adaptive` that recovers this
instance — plain cold (P1) and cold+looser-tol (P3) both still fail at
`maxIterations`, so "cold" alone is not what recovers it; the adaptive
mu-strategy is.

**Tier 2 (P4) recovers it: `True`.** P4's tier-2 sub-solve is numerically
consistent with P2's standalone solve (66 iterations, matching objective
0.8553691009125476 / constraint violation 1.31e-07 / `lg(mu)=-7.2`), as expected
since tier 2 applies exactly P2's recipe as its own third attempt.

Guard counts per arm confirm the exact declared solve structure with zero
blocked calls: P0=2, P1=1, P2=1, P3=1, P4=3 (`blocked_solve=0`,
`blocked_exec=0` in every arm).

Full evidence: `data/SRP1/Results/P515R/part3_cycle38/part3_cycle38_report.json`
(per-arm logs under `data/SRP1/Results/P515R/part3_cycle38/logs/`).

## Validation

- All six modified files parse (`ast.parse`) and import cleanly together
  (`import network, shared_energy_storage_data, shared_resources_planning, solver_parameters, p59_rho, p515_g_g1_g4_admm_gates`).
- `git diff` for every modified file reviewed in full; no unintended
  modifications found (diffs shown to the Planner on request; summarized above).
- Production code executes correctly: confirmed live in R2 (tier-1 recovery
  fired and succeeded for case9/2035/Winter under the new eligibility path) and
  in Part 3 (tier-2 actually fired, printed its distinct messages, and
  recovered a real `maxIterations` failure end-to-end).
- Tests pass: R1 bitwise (max rel diff 0.0), R2 bitwise (0/0 mismatches across
  two independent comparison sets), parser re-validation exact (12/0/6, correct
  cycle set).
- Diagnostic works as specified: Part 3's guard counts, sha256 fixture
  invariance, and per-arm parsed summaries (iterations, objective, constraint
  violation, terminal `lg(mu)`, wall time, log path) are all present and
  internally consistent (P4's tier-2 numbers reproduce P2's standalone numbers).
- **Distinguishing what is and is not shown:** R1/R2 show the refactor is a
  behavioral no-op under G1-equivalent policy (tier 2 off, case33_1 disabled) —
  this is a regression result, not evidence tier 2 helps. Part 3 is the evidence
  that tier 2 helps on a real, previously-unrecovered failure (G2 will need its
  own re-run, per Addendum 7 item 1, to show the effect at scale — not run here,
  not authorized here).
- Excel writer change (`_write_solver_recovery_diagnostics_to_excel`) was
  reviewed for correctness (columns list, `value is None` skip logic) but not
  exercised end-to-end (no full operational-planning Excel export was run in
  this task — out of scope; it is a superset-columns change, backward compatible
  with existing readers keyed by column label).

## Unexpected findings

- **G1's own `g_control.json`/`network_failures_control.jsonl` bake in a stale
  classification** (`recovered=12, unrecovered=4, not_attempted=0`) that
  disagrees with the corrected scanner's re-parse of the same run's raw stdout
  (`recovered_tier1=12, unrecovered=0, not_attempted=6`). This is not new — it
  is the exact, already-documented "second G1 bug" in
  `_scan_network_failures`'s own docstring (a since-fixed event-keying bug from
  an earlier iteration of this harness) — but it means anyone reading
  `g_control.json`'s `network_failures_summary` field directly (rather than
  re-parsing `stdout_control.log` with the current scanner) will see the wrong
  historical numbers. Flagged for the Planner's awareness; not touched (would
  require rewriting a committed artifact, out of scope).
- **`Network.run_smopf` prints the same `[WARNING] Network recovery solve did
  not converge` text twice** in a genuine two-failure trace (once as the
  tier-1→tier-2 transition message, once — with a different `attempt_label`
  after this task's change — from the final failure branch) is NOT what
  happens: verified live in Part 3 that only ONE "recovery solve did not
  converge" line prints before "Retrying ... tier 2", and if tier 2 also fails
  the final line reads "tier-2 recovery solve did not converge" (distinct
  text). No double-print observed in the R2 or Part-3 stdout captures.

## Deviations from the data/ restriction (disclosed)

The instruction was "No edits under `data/` except NEW files in a NEW directory
`data/SRP1/Results/P515R/`." Every **report/result artifact** I wrote is under
`P515R/`. Two categories of unavoidable side effect fall outside it, both from
reused harness/production infrastructure I did not modify:

1. **`O.fresh_planning` (`p56a_oracle.py`) always sets `logs_dir` under
   `data/SRP1/Results/P56A/evals/<eval_id>/logs`,** independent of any
   `results_dir` redirection (`run_admm_arm`'s Fix-1 redirection covers
   `results_dir`/FrozenSMOPF, not `logs_dir`). R2 (`run_admm_arm`, unmodified)
   and Part 3 (where I did not override `net.logs_dir` before construction, only
   after) therefore created **new**, uniquely-named eval directories under
   `P56A/evals/` (`p515r_r2_g1policy`, `p515r_c38_p0`…`p515r_c38_p4`). For Part
   3 I additionally set `net.logs_dir` explicitly to a path under `P515R/`
   before any solve, so those five `P56A/evals/p515r_c38_p*` directories are
   confirmed **empty** (verified: `find ... -type f` returns nothing) — an
   artifact of `fresh_planning`'s own `os.makedirs`, not of anything I wrote.
   R2's `P56A/evals/p515r_r2_g1policy/logs/` **does** contain real IPOPT log
   files (54 files, network + ESSO), since `run_admm_arm` is reused unmodified
   per the brief's own instruction ("`run_admm_arm` by import with its
   smoke-only cycle override") and does not expose a `logs_dir` override.
2. **R1's reused `p515_h_tol_remedy_check.run_arm`** hardcodes
   `oracle.fresh_planning(f'p515h_tol_{tag}')`, which resolves ESSO log output
   against that same `P56A/evals/p515h_tol_{tag}/logs` path (not under my
   `H.LOG_ROOT` override, which only affects the working directory / arm-report
   JSON). Re-running it created 3 new `*_dup2.txt` log files per arm (6 total)
   in the **pre-existing** `p515h_tol_control` / `p515h_tol_remedy_h` eval
   directories — production's own "don't append, choose a non-colliding name"
   logic (Addendum 5 item 2), triggered because those directories already held
   the original stage's logs from an earlier date. Verified: the **original**
   files (10:18 today, from an earlier, already-existing re-run this task did
   not perform) are byte-for-byte unchanged (same size, same mtime); only the
   `_dup2` files are new, and their contents are numerically identical to the
   originals (consistent with R1's bitwise pass).

None of this touched, appended to, or overwrote any pre-existing file; all of it
is new-file-only, all of it traces to production/harness functions this task
was explicitly told to reuse rather than reimplement (CLAUDE.md), and none of
it is under a `P515*`-numbered evidence directory a committed report cites.
Flagged rather than silently accepted, per the Planner's own evidence rules.

## Remaining issues

- G2 re-run (Addendum 7 item 1, "Then re-run G2, capturing ESSO slack values as
  well as duals") is **not** part of this task and was **not** run.
- Ablation A/B (Addendum 7 item 2, frozen at `76ab3f99` concurrently with this
  task) is **not** part of this task and was **not** run.
- The Excel diagnostics export with the four new columns was not exercised
  end-to-end (see Validation).
- `g_control.json`'s stale `network_failures_summary` (Unexpected findings,
  above) was left as-is; the Planner may want to note this when citing that
  file directly in future reports.

## Questions for Planner

1. The two disclosed `P56A/evals/` side effects (R2's real log files under a
   new, non-colliding eval id; R1's 6 new `_dup2` log files in two pre-existing
   eval directories) — is this an acceptable reading of "no edits under `data/`
   except new files in `P515R/`" given they trace to unmodified,
   explicitly-reused harness functions, or should a follow-up task redirect
   `logs_dir` too (would require touching `p56a_oracle.py` or the two reused
   harness files, both currently out of this task's scope)?
2. `g_control.json`'s stale `network_failures_summary` field (Unexpected
   findings) — leave as committed history, or should a follow-up correct it
   given the current scanner disagrees with it?
