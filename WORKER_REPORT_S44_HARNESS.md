# Worker Report — P5.15 Addendum 25 item 1: per-node zero check, campaign harness, s44_gate

## Task received

Addendum 25 item 1 (frozen spec v14 `frozen_s44_selection_spec_v14_e4500e27.json`, `item1_harness`;
Addendum 26; `STEP4_DFO_METHOD.md` §2). The alias tie-break fix was already done (`8682cfdd`). This task
covered:

- (A) a zero-solve check that per-node zero storage is evaluable;
- (B) the campaign harness and its zero-solve checks;
- (C) the `s44_gate` campaign: C\*, paper plan and node-7-empty; cap 500; 3 concurrent; D configuration.
  The gate is C\* bitwise against D. The two companions are reported with their outcomes.

## Files inspected

- Authority: `PLANNER_BRIEF_2026-09-13.md` (Addenda 25/26), `STEP4_DFO_METHOD.md`, spec v14.
- Harness code: `p515_g_g1_g4_admm_gates.py`. Functions read: `run_admm_arm`, `_construct_arm_planning`,
  `_acquire_exclusive_run_lock`, `s34/s35ref/s38/s39` capture hooks, `_identify_soh_floor_rows`,
  `write_boyd_terminal_s35ref`, `run_s39_arm`.
- Oracle, capture and comparison scripts: `p56a_oracle.py` (`fresh_planning`), `p514_n_instrumented_cstar.py`,
  `p515_s40_clone_capture_preflight.py` (comparator, `8f5cff48`), `p515_s40_polish_gap.py`,
  `p515_s40_case_file_repro.py`, `p515_s43_aa_run.py`, `p515_s43_aa_flagoff_gate.py` (classifier),
  `p515_s39_evaluate.py`, `p513_solve_profile_guard.py`, `p515_s32_zero_solve_checks.py`.
- Production code: `shared_resources_planning.py` (ADMM init, S_ref helpers, residual metrics, EFC diagnostic),
  `shared_energy_storage_data.py` (`_build_subproblem`, cohort gating, `get_updated_capacities`),
  `model_construction_helpers.py` (`configure_shared_ess_operational_state`), `admm_persistent_workers.py`
  (thread-cap precedent).
- D reference: `data/SRP1/Results/P515S39_D_run/`, `WORKER_REPORT_S40_CASE_FILE.md`, `P5_15_G3F_REPORT.md`.

## Files modified

None in production. No change to ADMM math, tolerances, penalty policy, AA code or the case file.

New files, committed:
- `p515_s44_per_node_zero_check.py` (Task A)
- `p515_s44_campaign_harness.py` (harness library + child entry point)
- `p515_s44_campaign_harness_checks.py` (zero-solve checks)
- `p515_s44_gate.py` (the gate campaign)
- `p515_s44_gate_tie_analysis.py` (a zero-solve analysis after the gate)
- evidence under `data/SRP1/Results/P515S44/`
- this report

## Changes made — harness design

**Process model.** `evaluate(batch, ctx) -> records` (§2.7).
- Each candidate runs in its own fresh interpreter: `python -u p515_s44_campaign_harness.py --child ...`,
  launched with `subprocess.Popen`, never forked.
- The child environment sets `OMP/MKL/OPENBLAS/VECLIB/NUMEXPR_NUM_THREADS=1`. The IPOPT executables inherit it.
- There are `concurrency` slots, filled in batch order. The parent reaps each child with non-blocking
  `os.wait4`, so it gets that child's own rusage.
- Records come back in batch order. If a child leaves no record, the parent synthesizes a barrier record with
  the exit code and the stderr tail.

**The parent never solves.**
- `p515_s44_gate.py` installs `SolveProfileGuard(permitted=())` before any model module is imported.
- At the end it verified 0 solves and 0 executions.

**Locks.**
- There is one new campaign lock, `.p515_s44_campaign.lock`. It is created O_EXCL and holds
  `{pid, campaign_id, spec sha256}`.
- The parent refuses to start if this lock or the legacy `.p515_g_gate.lock` exists.
- Children never take the legacy lock. `run_admm_arm` never takes it either; only the legacy harnesses' own
  `main`s do.
- Each child refuses to start unless the campaign lock exists, names its parent pid (`os.getppid()`) and names
  the same spec sha256.

**Frozen campaign spec.**
- The spec is written once to `campaign_spec_<id>_<sha8>.json`. It holds:
  - the candidates (canonical form and sha256 key);
  - the configuration: case file path, sha256 and last commit, arm label, overrides;
  - cap, concurrency, required consecutive cycles, thread caps;
  - interpreter, resolved `NLP_SOLVER_PATH`, git HEAD, and the harness sha256.
- The child checks the spec's sha256 against the file. That sha256 is written into every record.

**Per evaluation** (write-once dir `<campaign_root>/evals/<key16>_<label>/`).
- Written by the parent: `launch.json`, `child_stdout.log`, `child_stderr.log`, `exit_code.txt`,
  `wait4_rusage.json`.
- Written by the child:
  - every `run_admm_arm` artifact: `g_s39_D.json` with the full trajectory, `heartbeat_s39_D.json` (updated
    every ESSO solve), stdout, `esso_capture/`, the leak / network-failure / recovery / frozen sidecars, the
    ESSO pickle, and `results/FrozenSMOPF`;
  - the five per-cycle capture sidecars, appended every cycle so they survive a crash;
  - the four terminal artifacts;
  - `per_cycle_record.jsonl`;
  - `evaluation_record.json` (the §2.5 record);
  - `child_manifest_sha256.json`.
- Working-dir ids are `p515s44_<campaign>_<key16>_{run,precheck}`. The child refuses to start if either id
  already exists.

**The §2.5 record** (`build_evaluation_record`, a pure function) contains:
- status, the barrier flag and its cause;
- the certified cost (`gross_operational_cost`, settlement excluded; the convention is stated in the record);
- the bar: the maximum `objective_change_abs` over the last 10 cycles, with the window listed;
- the certification (via `p515_s39_evaluate` helpers, imported) and the first pass per channel;
- terminal primal/dual ratios per channel;
- rule ten (the terminal objective step over its threshold, plus the Boyd maxima), reported and not gated;
- the component decomposition (`totals_weighted`) and `recourse_components`;
- the settlement remainder (`interface_settlement_total`);
- per node: s, e, `has_storage`, EFC/day (max and per cohort-year), terminal SoH per active cohort-year and
  its minimum, active SoH-floor rows, and the terminal published available capacity;
- wall time and peak RSS.

**Configuration.**
- The configuration is the case file alone. `num_max_iters` is set to the spec cap, which
  `_construct_arm_planning` always does.
- A `pre_solve_hook` checks that the case file carries D (the `p515_s43_aa_run` checks, plus cap,
  persistent workers off, parallel off and AA off).
- It then applies only the spec overrides, restricted to `anderson_acceleration`. There are none for D.
- Its results are recorded in `rule_eleven_checklist`, which is provenance.

**Candidates.**
- A candidate is a per-node `(s_mva, e_mwh)` map at year 2025. Every active node must be given, and
  s = 0 ⇔ e = 0.
- It is passed as `run_admm_arm(investment_map=...)`.

**Rule eleven.** `assert_record_capture_paths()` runs in every child before its run.

## Commands / experiments run

All runs used the canonical interpreter, were attached (the gate through the tool's background runner, like
the G3F precedent), captured both streams, and ran alone with the preconditions checked.

| # | command (repo root) | output |
|---|---|---|
| 1 | `python -u p515_s44_per_node_zero_check.py > .../P515S44/per_node_zero_check_launch.log 2>&1` | `per_node_zero_check/` |
| 2 | `python -u p515_s44_campaign_harness_checks.py > .../harness_checks_launch.log 2>&1` | `harness_checks/` (C6 FAIL) |
| 3 | `python -u p515_s44_campaign_harness_checks.py r2 > .../harness_checks_r2_launch.log 2>&1` | `harness_checks_r2/` |
| 4 | `python -u p515_s44_gate.py > .../campaign_s44_gate_launch.log 2>&1` | `campaign_s44_gate/`, exit 1 |
| 5 | `python -u p515_s44_gate_tie_analysis.py > .../gate_tie_order_analysis_launch.log 2>&1` | `gate_tie_order_analysis/` |

Gate campaign configuration:
- Spec: `campaign_spec_s44_gate_4047b4e3.json`, sha256 `4047b4e3f1852c36…7f33`.
- Case file: `fb3de341`. Cap 500, concurrency 3, 10 consecutive cycles, no overrides.
- Git HEAD at launch: `710dae7d`.
- Started 09:35:30, ended 10:53:41; 3 slots observed concurrently.

## Results

### Task A — per-node zero storage: structurally evaluable (0 solves)

The check ran for C\*, the paper plan (nodes 5 and 9 zero) and node-7-empty (node 7 zero). All items pass.
- **Candidate map.** The map is honoured. The C\* map-path candidate equals the uniform-path candidate exactly.
- **Network gating.** Zero nodes have zero `total_capacity` in every year. Production gates their shared ESS
  inactive in all 12 TSO and 12 DSO blocks.
- **ESSO rows.** At a zero node, production marks every ESSO cohort inactive. That leaves 0 active
  charging/degradation/limit rows at the node, against 3/9/576 at a positive node.
- **S_ref normalization.** Every normalization call-site argument returns 2.5 MVA. This includes zero ratings,
  which already occur at C\*: the ESSO's per-year `.s` is 0 in 2030 and 2035.
- **Consensus entries.** All consensus and dual entries are present (576 per agent per node) and finite.
- **Per-cycle evaluators.**
  - `get_admm_residual_metrics` and `get_admm_boyd_residual_metrics` run inside the oracle's own capture
    wrappers, with finite output.
  - All five sidecars are written.
  - `capture_esso` gives EFC = None at zero nodes, not an error.
- **Terminal evaluators.** `write_boyd_terminal_s35ref` and the three writers it calls run with finite output.

A zero-solve check cannot settle two numerical questions; both were then observed in the run:
- The zero node's ESSO converter circle has a zero right-hand side, a degenerate feasible set. Both
  companions certified with 0 local solve failures and 0 unrecovered failures.
- The zero node's published capacity is a solved value, gated at 1e-10. At the terminal cycle it was exactly
  0.0 in every year at nodes 5 and 9 (paper plan) and at node 7 (node-7-empty).

The Planner's instructions did not trigger a stop.

### Harness checks — r2 all pass (0 solves)

**First attempt** (`harness_checks/`, at `41758012`).
- C1–C5 and C7 passed.
- **C6 failed.** My `assert_record_capture_paths` looked for `'interface_settlement_total'` in the gates module.
  The key is produced by production `_get_operational_recourse_components`.
- The defect was in the assertion itself, which each child also runs, so it would have failed every real
  evaluation before any solve.
- Fixed in `10e49f2d`. The failed attempt is committed as evidence.

**r2** (`harness_checks_r2/`, at `10e49f2d`). All seven checks pass:
- C1 canonical keys: order- and spelling-independent; ±0 folded; invalid candidates rejected.
- C2 spec: the sha8 in the file name equals the file hash; loading by hash works; write-once, unsupported
  overrides and duplicates are refused.
- C3 lock: exclusive; the legacy lock blocks; release by a wrong pid is refused.
- C4 stub spawn: 4 fresh interpreters with 2 slots.
  - Observed peak concurrency is 2; wall time 15.0 s against ≥16 s if run serially.
  - The thread caps are present in every child. Each child saw the lock naming its parent.
  - The spec sha256 is in every record. The per-evaluation files are present. `wait4` captured about 120 MB
    of RSS for each 96 MB allocation.
  - Re-evaluating the same candidate is refused. A failing child produces a parent barrier record with its
    stderr tail.
  - A child refuses a lock that names another pid, and refuses stub mode under a non-test spec.
- C5 record built from D's committed artifacts:
  - certified at cycle 139, cost 650,966,975.2943751;
  - bar 7,898.63; pf first pass 125 and ESS terminal ratio 0.78620, both equal to the committed
    `s39_evaluation.json`;
  - the barrier path works on a truncated trajectory.
- C6 capture paths pass.
- C7 the configuration hook, run on a real unsolved C\* planning with cap 500: all D checks pass and no
  override is applied. The AA override applies on a separate object.

### Gate — as run: FAIL under the committed classifier; every difference is tie-order in one diagnostic list

`gate_results.json` gives `gate_pass: false` and exit code 1. I have not re-classified it.

C\* against `P515S39_D_run` (comparator `8f5cff48`, classifier from `p515_s43_aa_flagoff_gate`):

| artifact | genuine | non-gating classes |
|---|---|---|
| `g_s39_D.json` (full report, 139-row trajectory) | **0** | provenance 3 (`rule_eleven_checklist`: D's s39 override block and the harness's 2 config blocks); aa_new_field 1,668 (= 12 fields × 139 rows, flag-off values); candidate-specification 4 (`instance`: uniform vs per-node map, verified equivalent) |
| `boyd_terminal.json`, `component_levels_terminal.json`, `interface_settlement_detail_s31c.json`, `interface_voltage_terminal.json` | **0** | none |
| `ess_entry_stride_baseline.jsonl`, `soh_floor_sidecar_baseline.jsonl`, `pf_entry_stride_s39_D.jsonl`, `ess_exempt_until_state_s39_D.jsonl` | **0** | none |
| `recourse_jump_sidecar_baseline.jsonl` | **43** | known tie-break 208 |

**Scalar checks.**
- `cycles_run` is 139 on both sides.
- `gross_operational_cost` is 650966975.2943751 on both sides (exact).
- C\* certified at cycle 139. Its solve count is 7,179, the same as D.

**Supplementary JSONL** (not part of the conventions):
- `leak_classification`: 0 diffs (420 rows).
- `esso_recovery_events`: 0 rows on both sides.
- `network_failures`: 39 diffs, all `primary_log`/`recovery_log` paths.
- `frozen_snapshots`: 4 diffs, `path`/`mtime` only.

**The 43 unclassified diffs.** All sit in `objective_component_block_deltas`, the diagnostic top-10 list of the
recourse-jump sidecar. Examples:
- `[59].[4].component`: `flexibility_cost` → `economic_market_cost`.
- `[76].[5]`: `economic_market_cost` → `classified_total`, with its `previous`/`current` swapping in from `[7]`.

`p515_s44_gate_tie_analysis.py` (zero solves, committed before it ran) tests one hypothesis: every
difference is the `8682cfdd` sort-key change acting on exact `abs_delta` ties. **It holds for 139/139 rows:**
- 50 rows are identical;
- 78 rows are reproduced exactly by re-sorting D's committed list with the new key (every field of every entry);
- 11 rows are cut straddles: a tie group crosses position 10, so D's hash-seed order kept a different member of
  the same tie group. The prefix is identical after re-sorting. At the cut, `abs_delta` and `block_key` are
  identical, and every entry present on both sides is identical.

The tie groups observed:
- {economic_market_cost, generation_cost}: aliases with identical values; the classifier knows this pair.
- {economic_market_cost, flexibility_cost}: aliases in DSO blocks with identical values; not in the
  classifier; 100 rows.
- {classified_total, economic_market_cost[, generation_cost]}: not aliases; different values but an exactly
  equal `abs_delta`.

**Reading.**
- Every numeric trajectory field and the cost are bitwise identical to D.
- Every other artifact and sidecar is identical.
- The only differences are the order of exactly-tied entries (and, at 11 cuts, which tied member is kept) in
  one diagnostic list. The task note anticipated this for "the tied aliases".
- The committed classifier recognizes only one alias pair, so the strict gate reads FAIL.
- Whether this satisfies the gate is the Planner's decision. The companions count for item 3 only if it does.

### Companions (D configuration, cap 500): both certify; the spec v14 stop rule is not triggered

Objective convention: `gross_operational_cost`, settlement excluded. Net equals gross here: the terminal
salvage is at most 1.4e-50. Candidates are identified by their sha256 keys.

| | C\* `578636da…` | paper plan `d1a02e67…` | node-7-empty `e30704e6…` |
|---|---|---|---|
| certified / cycle | yes / 139 | yes / **139** | yes / **136** |
| certified cost (gross) | 650,966,975.2943751 | **653,029,766.9858261** | **651,900,014.159667** |
| bar (max step, last 10 cycles) | 7,898.63 | 661.58 | 9,104.68 |
| first pass v / pf / ess | 48 / 125 / 75 | 47 / 119 / 69 | 48 / 127 / 67 |
| terminal ratio max v / pf / ess | 0.186 / 0.486 / 0.786 | 0.189 / 0.795 / 0.530 | 0.198 / 0.752 / 0.886 |
| rule ten (terminal step / objective threshold) | 0.0481 | 0.00798 | 0.00582 |
| settlement remainder | 36,679.08 | −1,907.42 | 58,639.34 |
| generation / internal flexibility | 443,877,068 / 207,102,708 | 445,142,652 / 207,899,878 | 444,460,026 / 207,452,771 |
| EFC/day max per node (5 / 7 / 9) | 1.1298 / 1.1335 / 1.1319 | — / 1.2361 / — | 1.1291 / — / 1.1311 |
| terminal SoH min per node | 0.6346 / 0.6335 / 0.6341 | — / 0.6186 / — | 0.6338 / — / 0.6333 |
| SoH floor rows active at terminal | none | none | none |
| local failures; network failure blocks (all recovered) | 0; 38 | 0; 32 | 0; 25 |
| solves | 7,179 | 7,173 | 7,012 |
| **peak RSS** (child process) | **2,345,844,736 B (2.18 GiB)** | **2,376,548,352 B (2.21 GiB)** | **2,383,642,624 B (2.22 GiB)** |
| IPOPT subprocess max RSS | 52.8 MB | 55.8 MB | 57.4 MB |
| wall (parent view) | 4,685.5 s | 4,149.9 s | 4,305.1 s |

The cost differences from C\* are:

| | difference from C\* | sum of the two bars | ratio |
|---|---|---|---|
| paper plan | +2,062,791.69 | 8,560.2 | 241× |
| node-7-empty | +933,038.87 | 17,003.3 | 55× |

In the sense of CLAUDE.md rule nine, neither difference is explained by stopping slack. That says nothing
about path divergence (rule ten); every cell stopped at 0.6–4.8 % of its objective threshold.

Wall time with 3 concurrent was 4,150–4,686 s each, against 4,893 s for D alone. The `wait4` RSS equals the
child's own RUSAGE_SELF in every case.

## Validation

- **Code executes.** Task A, the harness checks (r2) and the gate all ran to completion. Guards are at 0 in
  every zero-solve script and in the gate parent.
- **Requested diagnostics work.**
  - Every evaluation wrote its full record set.
  - The spec sha256 appears in all 3 records.
  - The campaign manifest covers 1,335 files, all re-verified against disk before commit with 0 mismatches.
  - The lock was released.
- **Test result.** Strict gate FAIL (43 tie-order diffs). The tie-order hypothesis holds for 139/139 rows.
- **Underlying question.**
  - Concurrent, capped, map-built C\* evaluation reproduces D's numerics bitwise: trajectory, cost, and all
    numeric sidecars and artifacts.
  - The single-thread caps (HSL MA97 with Accelerate) did not change any number.
- **Diff check.** `git diff` was inspected before each commit. Only new files were added. The index was checked
  empty before each staging.

## Unexpected findings

1. **The tie-break fix is not complete as a determinism guarantee against earlier artifacts, and the classifier
   is too narrow.**
   - Beyond {economic_market_cost, generation_cost}, the top-10 list has a second alias pair,
     {economic_market_cost, flexibility_cost}, in DSO blocks.
   - It also has non-alias exact ties with `classified_total`.
   - `p515_s43_aa_flagoff_gate.KNOWN_TIE_BREAK_ALIAS_PAIRS` lists only the first pair. Any future bitwise gate
     against a pre-`8682cfdd` sidecar will hit the same 43.
   - From `8682cfdd` on, the order is a total order, so harness-vs-harness comparisons are unaffected.
2. **Hazard, not observed.** The sibling list `block_deltas` in the same hook still sorts by `abs_delta` alone,
   over `set(...)` (`p515_g_g1_g4_admm_gates.py`, `s34_capture_hooks`). No diff was observed in it here.
3. **Zero ESSO ratings already occur at C\*.** The ESSO's per-year `.s` is 0 in 2030 and 2035 (investment only
   in 2025). A zero rating argument is therefore not new with per-node-zero candidates; S_ref covers both.
4. **An eval dir I did not create.** `data/SRP1/Results/P56A/evals/p515s44_zero_storage_probe` already exists.
   It is from the previous attempt or the other Worker; I left it untouched.
5. **Supplementary JSONL files embed absolute paths.** `network_failures_*.jsonl` embeds absolute log paths,
   which the `8f5cff48` exclusions do not cover. It is outside the gate's artifact lists, so it is reported
   only.

## Remaining issues

- **The gate verdict needs a Planner ruling.** The strict result is FAIL, and the evidence shows tie-order only.
  The companions count for item 3 only if the gate passes (spec v14).
- **The legacy one-run lock gap.** Legacy harnesses do not read the campaign lock. A legacy one-run harness
  started during a campaign would not be blocked by the lock, though several of them scan for `p515_s4`
  processes. I did not close this; the option is noted below.
- **Evidence I did not commit.** These are hash-recorded in `campaign_manifest_sha256.json` (committed), about
  634 MB in total:
  - `esso_capture/`
  - `results/FrozenSMOPF`
  - `ess_entry_stride_baseline.jsonl` and `pf_entry_stride_s39_D.jsonl` (per-entry strides)

  The ESSO pickles (about 2.7 MB each) are committed, as D's was.
- **Hash seed.** `PYTHONHASHSEED` is inherited (unset) in children, the same as in D's run. It is recorded in
  each record. I did not pin it.

## Questions for Planner

1. Does the gate pass? The facts:
   - every numeric trajectory field, the cost and all 8 other artifacts/sidecars are bitwise identical;
   - the 43 remaining diffs are proven tie-order, from `8682cfdd`, in one diagnostic top-10 list, including
     11 cut straddles and 2 tie groups the classifier does not list.
2. If it does pass: should the classifier's `KNOWN_TIE_BREAK_ALIAS_PAIRS` be widened, or should the tie
   analysis's re-sort/straddle criterion be adopted for later gates against pre-`8682cfdd` sidecars? The same
   question applies to fixing `block_deltas`' sort key the same way as `8682cfdd`, which needs your
   authorization.
3. Should the campaign parent also hold the legacy one-run lock? That would block legacy harnesses during a
   campaign at no cost to concurrency. Alternatively, should `_acquire_exclusive_run_lock` refuse when the
   campaign lock exists? That would be a harness-file edit.

## Commits

- `a60ea791` Task A script
- `d8df67eb` Task A evidence
- `41758012` harness + checks + gate script
- `10e49f2d` C6 assertion fix
- `710dae7d` checks evidence (both attempts)
- `7ac75e9f` tie analysis script
- the evidence commit carrying this report (gate campaign files, tie analysis output, launch logs)
