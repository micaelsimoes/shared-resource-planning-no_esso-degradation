# Worker Report — P5.15 G2PREP (harness-only fix, zero-solve validation)

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 6, Planner task "Authorized
HARNESS-only change with ZERO-SOLVE validation" (2026-09-14). HEAD at start: `4b56c835`.
File changed: `p515_g_g1_g4_admm_gates.py` only. No solve was performed at any point in
this task (verified below with an armed `SolveProfileGuard([], ...)`, permitted count 0).

## Task received

Three fixes to `p515_g_g1_g4_admm_gates.py`:

1. Every campaign's `results_dir` must be redirected to its own arm root (G1 overwrote
   preserved `FrozenSMOPF` comparators because `results_dir` stayed pointed at the
   shared tree).
2. Every remaining CLI gate (`g2`, `g3_full`, `g4b`) gets its own fresh output root and
   fresh eval id, refusing to start if either already exists.
3. `_scan_network_failures` rewritten to classify network failures from the tee'd
   stdout only (not from the cumulative, misleading per-network IPOPT log files), with
   correct per-cycle event granularity, validated against G1's committed stdout to an
   exact, pre-declared expectation.

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` (Addenda 1–6, governing decisions and gate
  specifications).
- `p515_g_g1_g4_admm_gates.py` (full file, before and after every edit).
- `p56a_oracle.py` — `fresh_planning`, `_holders`, `WORK_DIR`/`OUT_DIR`.
- `shared_resources_planning.py` — `SharedResourcesPlanning.__init__` (results_dir
  construction, line 56), `save_failed_tso_block`/`save_selected_tso_comparator`
  (4460–4487), `_save_frozen_smopf_block` (4510–4549), `_save_frozen_network_block`
  (4552–4594), `save_failed_dso_block`/`save_selected_dso_comparator` (4651–4691),
  the sequential and parallel DSO "did not converge" final prints (4701–4712,
  4774–4785), the ADMM cycle loop and its `[INFO] \t - ADMM Iteration {iter}` print
  (2219–2225), `update_shared_energy_storages_coordination_model_and_solve` (4788+).
- `network.py` — `_create_smopf_solver`, `_run_smopf_solver_attempt`,
  `_is_recoverable_network_failure`, `_print_network_failure_context`, `_run_smopf`
  (≈495–685: the exact print sequence for primary/retry/recovery/not-attempted).
- `network_data.py` — `network_planning.network[year][day].results_dir` assignment at
  construction time (line 183), confirming it is frozen from the OLD, shared
  `results_dir` before `fresh_planning` ever runs.
- `helper_functions.py` — `solver_result_summary`, `solver_result_succeeded`.
- `p513_solve_profile_guard.py` — `SolveProfileGuard` API (`install`/`uninstall`,
  empty `permitted` list blocks every solve call).
- `p515f_t2_failure_probe.py` — prior precedent for redirecting `results_dir` after
  construction, at the repo root, before any chdir.
- `data/SRP1/Results/P515G1/network_failures_control.jsonl`,
  `data/SRP1/Results/P515G1/g_control.json`,
  `data/SRP1/Results/P515G1/stdout_control.log` — read-only, for the required
  validation (never modified — see "no P515G1 writes" check below).
- `data/SRP1/case33_1/case33_1_params.json` vs `case33_2`/`case33_3` — confirmed
  `case33_1` (node 5) has no `recovery_options` key at all (see Unexpected findings).

## Files modified

- `p515_g_g1_g4_admm_gates.py` (the only file changed; 406 insertions / 119 deletions,
  confirmed with `git diff --stat`).

New files written (all under the one authorized location):

- `data/SRP1/Results/P515G1_parser_validation/network_failures_control_reparsed.jsonl`
- `data/SRP1/Results/P515G1_parser_validation/network_failures_control_reparsed_summary.json`
- `data/SRP1/Results/P515G1_parser_validation/zero_solve_probe/zero_solve_probe_report.json`

One exception, disclosed: the Fix‑1 zero-solve verification, run exactly as the task
specifies ("construct the planning object for a dummy arm exactly as `run_admm_arm`
does"), necessarily calls production's own `O.fresh_planning`, which always creates
`data/SRP1/Results/P56A/evals/<eval_id>/logs` regardless of the harness's own output
root. This produced one new, empty directory:
`data/SRP1/Results/P56A/evals/p515g2prep_zero_solve_probe/logs/` (no files inside — no
solve occurred, so nothing was ever logged there; it does not appear in `git status`
because git does not track empty directories). This is the same, unavoidable side
effect every prior arm in this campaign has had (`g1`, and the `p515f_t2_failure_probe`
precedent) and is disclosed here rather than worked around, per the instruction to
perform this exact construction. Nothing else outside
`data/SRP1/Results/P515G1_parser_validation/` was written.

## Changes made

### Fix 1 — `results_dir` redirection

- `_set_results_dir_for_arm(planning, results_dir)` (new): redirects `planning`,
  every holder `p56a_oracle._holders()` walks for `logs_dir` (transmission_network,
  each distribution_network, each holder's `network[year][day]` object), and
  `shared_ess_data`, mirroring `fresh_planning`'s own traversal exactly, but for
  `results_dir` instead of `logs_dir`.
- Callback save-path table (every place a callback resolves a save directory from a
  `results_dir` attribute — verified by reading production, not assumed):

  | callback | file:line | reads |
  |---|---|---|
  | `save_failed_tso_block` | `shared_resources_planning.py:4460-4472` | `transmission_network.results_dir` (line 4463) |
  | `save_selected_tso_comparator` | `shared_resources_planning.py:4474-4487` | `transmission_network.results_dir` (line 4478) |
  | `save_failed_dso_block` | `shared_resources_planning.py:4651-4671` | `distribution_network.results_dir` (line 4661) |
  | `save_selected_dso_comparator` | `shared_resources_planning.py:4675-4689` | `distribution_network.results_dir` (line 4679) |
  | `_save_frozen_network_block` | `shared_resources_planning.py:4552-4594` | takes `save_dir` as an argument (does not read `results_dir` itself — every caller above passes `os.path.join(<holder>.results_dir, 'FrozenSMOPF')`) |
  | `_save_frozen_smopf_block` | `shared_resources_planning.py:4510-4549` | same — argument, not attribute read |

  `network[year][day].results_dir` is also redirected (no current callback reads it,
  but `network_data.py:183` freezes it from the OLD shared `results_dir` at
  construction time, and `network_data.py:143` / `shared_energy_storage_data.py:220`
  do read a per-object `.results_dir` for their own, unrelated Excel writers) — this
  matches the task's instruction to mirror the `logs_dir` traversal exactly.
- `_hash_dir_pkls(dir_path)` (new): sha256 of every non-recursive `*.pkl` in a
  directory, used to prove (not assert) the shared `FrozenSMOPF` tree is untouched.
- `run_admm_arm` now hashes `SHARED_FROZEN_SMOPF_DIR` immediately on entry and again
  after the `finally` block, brackets the *whole* arm (not just the solve window),
  and records `shared_frozen_smopf_modified` / `shared_frozen_smopf_new_files` in the
  report JSON, printing `[ERROR] shared FrozenSMOPF directory modified during arm
  {label}: ...` if either is non-empty.
- `assert_g_capture_paths` (rule eleven) extended: every holder's `results_dir` must
  be absolute AND under the arm's own `out_dir` (`planning`, `shared_ess_data`,
  `transmission_network`, every `distribution_networks[node_id]`) — checked before
  any solve is attempted, same as the existing `logs_dir` checks.
- `run_admm_arm`'s construction logic (eval-dir freshness check, `fresh_planning`,
  `results_dir` redirection, ADMM/budget/rho params, `k_override`, investment
  candidate) was factored out into `_construct_arm_planning(...)` so the zero-solve
  verification below calls the *same* code `run_admm_arm` calls, not a
  reimplementation.

### Fix 2 — fresh roots and eval ids

- `OUT_G2 = data/SRP1/Results/P515G2/`, `OUT_G3F = data/SRP1/Results/P515G3F/`,
  `OUT_G4 = data/SRP1/Results/P515G4/` (new module-level constants, alongside the
  existing `OUT_G1`).
- `g2`: `_require_fresh_output_root(OUT_G2)`, then
  `run_admm_arm('k10000', OUT_G2, k_override=10000.0, eval_id='p515g2_k10000')`.
- `g4b`: `_require_fresh_output_root(OUT_G4)`, then
  `run_admm_arm('control_rep2', OUT_G4, k_override=None, eval_id='p515g4_control_rep2')`.
- `g3_full`: `_require_fresh_output_root(OUT_G3F)`; the probe (reads
  `active_distribution_network_nodes`, no solve, no candidate) now uses its own fresh,
  freshness-checked eval id `p515g3f_probe` (previously `p515g_g3_full_probe`, with no
  freshness check at all); the main arm uses `eval_id='p515g3f_node7'` (previously
  none — fell back to the shared `p515g_{label}` naming under the shared `OUT`/`P56A`
  eval tree).
- `g3_init` (ladder initialization stage) is unchanged — out of Fix 2's scope (the
  Planner's task text lists only `g2`, `g3_full`, `g4b`; `g3_init` already passed
  under Addendum 5 and uses a different code path, `run_ladder_init`, that this task
  did not authorize touching).

### Fix 3 — network-failure classification rewrite

- `_scan_network_failures(stdout_path, name_to_agent)` rewritten (signature changed
  from `(stdout_path, planning_problem)` to take `name_to_agent` directly — see
  "Files inspected"/validation below for why: it lets the offline reparse validation
  call the exact same function without constructing a planning object). New helper
  `_build_name_to_agent(planning_problem)` extracts the old inline logic unchanged;
  `_scan_and_write_network_failures` (both call sites: per-cycle, via the ESSO
  wrapper hook, and once more at end-of-run) now calls
  `_build_name_to_agent(planning_problem)` then the new `_scan_network_failures`.
- Classification now reads `network.py`'s own three-way print (previously the
  `label` group of `_NET_FAIL_RE` was captured but discarded):
  - `attempt_label='solver'` (recovery not eligible — e.g. no `recovery_options` in
    that network's case file, or a termination condition outside
    `{internalSolverError, maxIterations, infeasible}`) → **not_attempted**, from a
    single print, no retry ever happens.
  - `attempt_label='primary solve'` then `[INFO] Retrying ...` then either
    `[INFO] Network recovery solve succeeded for ...` → **recovered**, or
    `attempt_label='recovery solve'` (`[WARNING] Network recovery solve did not
    converge for ...`) → **unrecovered**.
  - `[ERROR] Transmission network ... did not converge` /
    `[WARNING] Distribution network node=N, ... did not converge` (the caller-side
    summary in `shared_resources_planning.py`) are kept only as
    `final_summary_crosscheck`, never as the classification signal.
- **Block keying fixed**: events are now keyed by `(network_name, year, day, cycle)`,
  not `(network_name, year, day)` alone. The old keying silently merged every cycle a
  given (network, year, day) triple recurred across into ONE block, overwriting
  earlier cycles' outcomes with the latest — this was the actual cause of G1's wrong
  "12 recovered / 4 unrecovered / 0 not_attempted" (4 merged rows) against a
  6-not_attempted ground truth.
- **Cycle attribution** (stated explicitly, as required): `[INFO] \t - ADMM Iteration
  N` (`shared_resources_planning.py` line ~2225) prints at the very START of cycle
  N's body, strictly before that cycle's DSO/TSO/ESSO solves run (single-threaded,
  sequential execution) and strictly before the next `ADMM Iteration N+1` line.
  Every stdout line between one such marker and the next therefore belongs
  unambiguously to cycle N; lines before the first marker are cycle `'init'`. This
  rule was already in force in the pre-fix harness (only the block-merging bug
  corrupted its effect) — it is validated below against `g_control.json`'s own
  `cycle_trajectory`, not merely asserted.
- Dropped: `_last_exit_and_iterations` and its two call sites (`primary_exit`,
  `primary_iterations`, `recovery_exit`, `recovery_iterations` fields). These
  re-parsed each network's IPOPT log file, which is **cumulative** across the whole
  run (`file_append='yes'`, `network.py`, unaffected by the P5.15-F ESSO-only
  per-solve-log fix) — its last `EXIT:` line reflects whichever cycle last touched
  that (network, year, day) combination, not the cycle being scanned. This was the
  cause of G1's `primary_exit: "Optimal Solution Found"` on genuinely failing rows.
  Termination condition is now taken directly from the stdout `summary` text
  (`status=..., termination=T`, captured verbatim from
  `helper_functions.solver_result_summary`) via a new `_extract_termination` regex.
  `primary_log`/`recovery_log` **paths** are still captured (from
  `[WARNING] IPOPT {label} log for ...` lines) — only the log-file re-parse for
  classification is removed.
- `network_failures_summary.classes` now also counts `'indeterminate'` (malformed
  capture: a resolving line missing, or an unrecognized `attempt_label`) so a
  parser failure cannot silently disappear into a zero count.

## Commands / experiments run

All run with the canonical interpreter
(`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`), foreground, stderr
captured (`2>&1`), no `screen`/`nohup`/`&`. No harness `__main__` was ever invoked; no
`.p515_g_gate.lock` was created or touched.

1. `python -m py_compile p515_g_g1_g4_admm_gates.py` — after every edit round.
2. Zero-solve verification (Fix 1), scratchpad script
   `zero_solve_probe.py` (not committed to the repo), guarding with
   `SolveProfileGuard([], label='G2PREP zero-solve probe')` (permitted count 0, i.e.
   ANY solve call raises immediately), calling
   `G._construct_arm_planning('zero_solve_probe', PROBE_OUT, report, k_override=None,
   investment_map=None, eval_id='p515g2prep_zero_solve_probe')` — the exact
   construction path `run_admm_arm` uses, up to but not including
   `run_operational_planning`.
3. Reparse validation (Fix 3), scratchpad script `reparse_g1.py` (not committed),
   calling `G._scan_network_failures` directly on the committed
   `data/SRP1/Results/P515G1/stdout_control.log`, with `name_to_agent` sourced from
   the pre-existing, already-committed `network_failures_control.jsonl` (read-only;
   not fabricated — every `(name, agent, node_id)` triple is read out of that file,
   not hand-typed), and cross-checked against `data/SRP1/Results/P515G1/g_control.json`
   `cycle_trajectory`.
4. `git status`, `find ... -newer`, `sha256sum`/`shasum -a 256` on
   `data/SRP1/Results/FrozenSMOPF/*.pkl`, existence checks on the four fresh roots and
   four fresh eval ids — all as evidence for this report.

## Results

### Zero-solve verification (Fix 1)

Guard counts after construction, before any solve was ever attempted:

```
{"permitted_solve": 0, "permitted_exec": 0, "blocked_solve": 0, "blocked_exec": 0}
```

Every holder's `results_dir` (printed for `planning`, `transmission_network`, every
`distribution_networks[5|7|9]`, `shared_ess_data`, and every `network[year][day]`
object under all of those — 96 per-year/day prints, all identical) resolved to:

```
/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515G1_parser_validation/zero_solve_probe/results
```

— absolute, and under the probe's own `out_dir`
(`ALL_UNDER_ARM_OUT_DIR = True`). `logs_dir` remained correctly isolated per eval at
`data/SRP1/Results/P56A/evals/p515g2prep_zero_solve_probe/logs` (unchanged behavior —
`fresh_planning`'s own mechanism, not touched by this fix). The shared
`data/SRP1/Results/FrozenSMOPF/` still held exactly 2 files
(the two pre-existing `matched_success_*` comparators) before and after construction.

Full printout preserved in
`data/SRP1/Results/P515G1_parser_validation/zero_solve_probe/zero_solve_probe_report.json`.

### Fresh-root / fresh-eval-id existence checks (Fix 2)

Before this task, all four confirmed absent:

| gate | root | eval id(s) |
|---|---|---|
| g2 | `data/SRP1/Results/P515G2/` (absent) | `p515g2_k10000` (absent under `P56A/evals/`) |
| g3_full | `data/SRP1/Results/P515G3F/` (absent) | `p515g3f_probe`, `p515g3f_node7` (both absent) |
| g4b | `data/SRP1/Results/P515G4/` (absent) | `p515g4_control_rep2` (absent) |

(g1's own root/eval, `P515G1/` / `p515g1_control`, already exist from the completed
G1 run and are untouched by this task.)

### Re-parsed G1 classification (Fix 3)

Re-running the new `_scan_network_failures` on the committed
`data/SRP1/Results/P515G1/stdout_control.log` (read-only; original file's mtime
unchanged, confirmed with `find -newer`) produced **exactly** the Planner's declared
expectation:

- **18 total blocks**: 12 recovered, 0 unrecovered, 6 not_attempted.
- **3 TSO blocks, all recovered**: `case9` 2025 Autumn (cycle 5), 2030 Winter
  (cycle 40), 2035 Winter (cycle 1) — the same set the Planner named (`2035 Winter,
  2025 Autumn, 2030 Winter`).
- **6 not_attempted, all DSO node 5 `case33_1`, termination `maxIterations`**,
  at cycles **exactly {7, 39, 43, 46, 48, 49}**.
- Cross-check: `g_control.json`'s own `cycle_trajectory` rows with
  `local_solves_ok = False` are at cycles `[7, 39, 43, 46, 48, 49]` — an exact match
  (`not_attempted_cycles_match_g_control_trajectory: true`).

Full table (agent, node, network, year, day, cycle, termination, class):

| agent | node | network | year | day | cycle | class | termination |
|---|---|---|---|---|---|---|---|
| TSO | — | case9 | 2035 | Winter | 1 | recovered | (recovered) |
| TSO | — | case9 | 2025 | Autumn | 5 | recovered | (recovered) |
| DSO | 5 | case33_1 | 2035 | Winter | 7 | not_attempted | maxIterations |
| DSO | 7 | case33_2 | 2025 | Spring | 7 | recovered | (recovered) |
| DSO | 9 | case33_3 | 2025 | Summer | 19 | recovered | (recovered) |
| DSO | 9 | case33_3 | 2025 | Autumn | 27 | recovered | (recovered) |
| DSO | 9 | case33_3 | 2035 | Autumn | 30 | recovered | (recovered) |
| DSO | 7 | case33_2 | 2025 | Summer | 33 | recovered | (recovered) |
| DSO | 5 | case33_1 | 2035 | Autumn | 39 | not_attempted | maxIterations |
| TSO | — | case9 | 2030 | Winter | 40 | recovered | (recovered) |
| DSO | 5 | case33_1 | 2025 | Spring | 43 | not_attempted | maxIterations |
| DSO | 7 | case33_2 | 2035 | Winter | 45 | recovered | (recovered) |
| DSO | 5 | case33_1 | 2035 | Autumn | 46 | not_attempted | maxIterations |
| DSO | 5 | case33_1 | 2035 | Spring | 48 | not_attempted | maxIterations |
| DSO | 5 | case33_1 | 2035 | Spring | 49 | not_attempted | maxIterations |
| DSO | 9 | case33_3 | 2025 | Winter | 51 | recovered | (recovered) |
| DSO | 7 | case33_2 | 2035 | Spring | 59 | recovered | (recovered) |
| DSO | 7 | case33_2 | 2035 | Autumn | 62 | recovered | (recovered) |

No `note` field was set on any of the 18 events (no malformed-capture / unresolved
event detected). Full JSONL and summary at
`data/SRP1/Results/P515G1_parser_validation/network_failures_control_reparsed.jsonl`
and `..._summary.json`.

### FrozenSMOPF / production untouched

```
$ git status --porcelain
 M .claude/agents/planner.md      <- pre-existing at session start, not touched by me
 M .claude/agents/worker.md       <- pre-existing at session start, not touched by me
 M p515_g_g1_g4_admm_gates.py     <- the one authorized change
?? data/SRP1/Results/P515G1_parser_validation/   <- the one authorized new directory
(plus the pre-existing, unrelated untracked residue already present at session start)
```

`data/SRP1/Results/FrozenSMOPF/matched_success_*.pkl` sha256 unchanged across the
whole task (verified before/after all scratchpad runs):

```
7e5aa39d388046c5fbad5d6a983c9f58872e7dd547ec74fdffe2dbbcbea34a03  matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl
fbfaa6b1291373302b61bf0b61d092d232d7965d80580d1abe7c8b7947b1a46c  matched_success_TSO_case9_2025_Summer_cycle7.pkl
```

`find data/SRP1/Results/P515G1 -newer p515_g_g1_g4_admm_gates.py -type f` returned
nothing — no file under `P515G1/` was modified during this session. No
`.p515_g_gate.lock` was created.

## Validation

- **Code executes correctly**: `py_compile` clean after every edit; the zero-solve
  probe and the reparse script both ran to completion without exception.
- **Zero-solve claim enforced, not asserted**: `SolveProfileGuard([], ...)` (permitted
  count 0) was armed around the entire construction path; `guard.counts` confirms
  zero solves, zero execs, zero blocked calls (there was nothing to block — the
  construction path genuinely never reaches a solver).
- **Fix 1 diagnostic works**: every holder's `results_dir` is absolute and under the
  arm's own root, verified by direct printout, not by re-reading the source and
  reasoning about it.
- **Fix 3 diagnostic works and matches an independently-declared ground truth
  exactly**: 12/0/6, the named TSO set, and the named cycle set {7,39,43,46,48,49}
  all matched on the first run, with no tuning of the parser to the answer (the
  regex/keying logic was written from reading `network.py`'s print statements
  directly, before running the validation).
- **Underlying numerical problem**: not addressed by this task and not claimed to
  be — this is a harness/evidence-capture fix only. The actual G2/G3-full/G4 runs
  have not been executed; Fix 2 only prepares fresh, collision-free launch points for
  them.

## Unexpected findings

- **`case33_1` (node 5) has no `recovery_options` key at all** in
  `data/SRP1/case33_1/case33_1_params.json` (confirmed by direct read), while
  `case33_2`/`case33_3` do. This independently explains *why* node 5's six failures
  are uniformly `not_attempted` (`_is_recoverable_network_failure` returns `False`
  whenever `not solver_params.recovery_options`, regardless of termination
  condition) — it is a configuration asymmetry between node 5 and nodes 7/9, not an
  ADMM-path or harness artifact. This is outside this task's scope (no case file was
  touched); flagging it for the Planner's attention since it may be relevant to
  interpreting G2/G3/G4's own not_attempted counts once they run.
- The `git status` "collapse" of `data/SRP1/Results/P515G1/` to
  `data/SRP1/Results/P515G1/esso_capture/` (rather than listing `P515G1/` as one
  line) is a pre-existing git porcelain-output artifact unrelated to this task —
  confirmed via `find -newer` that nothing under `P515G1/` changed. Documented above
  so it isn't mistaken for evidence of an unintended write.

## Remaining issues

- G2, G3-full, and G4 have not been run. The exact commands (unchanged from the
  Planner's text, now pointing at fresh, verified-absent roots) are:

  ```
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_g_g1_g4_admm_gates.py g2 > data/SRP1/Results/P515G2_launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_g_g1_g4_admm_gates.py g3_full > data/SRP1/Results/P515G3F_launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_g_g1_g4_admm_gates.py g4b > data/SRP1/Results/P515G4_launch.log 2>&1
  ```

  Each writes its per-arm artifacts under its own fresh root
  (`data/SRP1/Results/P515G2/`, `P515G3F/`, `P515G4/` respectively — heartbeat,
  stdout, leak classification, ESSO capture, network failures, `results/` — the last
  being the Fix‑1 redirect target, confirmed never resolving to the shared
  `data/SRP1/Results/FrozenSMOPF/`), and each refuses to start if its root or eval id
  already exists (verified above, before any of the three has been launched). These
  three gates run **sequentially**, one Worker task per gate per the Planner's brief
  and the exclusive-run-lock the harness itself enforces (`.p515_g_gate.lock`,
  untouched by this task).
- The `not_attempted` mechanism (missing `recovery_options` for `case33_1`) is
  unresolved as a modeling question — left for the Planner per "Unexpected findings".

## Questions for Planner

- None blocking. If the `case33_1` missing-`recovery_options` asymmetry (Unexpected
  findings) should be corrected before G2/G3-full/G4 run, that is a production-config
  change outside this task's authorization and would need its own instruction.
