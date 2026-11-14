# Worker Report — P5.15 Step 3.6: corrected per-phase timing ANALYSIS (zero-solve re-analysis)

**Task**: correct the per-phase timing analysis code and re-analyze the already-captured
Step 3.6 measurement (`data/SRP1/Results/P515S36/step36_timing/{off,on}/`, produced by
`p515_s36_step36_timing_run.py` at commit `1e7b0a69`, launch log exit `0`). **Zero solves.**
No production file, case file, `p515_g_g1_g4_admm_gates.py`, or anything under
`data/SRP1/Results/P515S38*` was touched. The `s38_b_pfbal` numerical arm (PID `74523`,
`ps aux` confirmed alive throughout, before and after this task) was never disturbed.

---

## Task received

Verify (not assume) four defects the Planner found in the v1 analysis, fix the analysis code,
build a zero-solve re-analysis script (`SolveProfileGuard` armed), and write the corrected
per-cycle table, verdict, identity check, manifest, and this report. Full task text: see the
Planner brief this report answers (reproduced in the conversation this Worker was launched
with; not re-typed here).

---

## Files inspected

- `P5_15_STEP36_TIMING_DESIGN.md` (full file) — §1 call-path map, §2 instrumentation design
  (especially §2.5's b/c/d1 split via `report_timing=True`), §5 measurement plan, the X=70%/
  Amdahl derivation.
- `WORKER_REPORT_S36_TIMING.md` (full file, including its 2026-09-17 addendum) — the recorder's
  wrap-point table, the derivation contract ("param_update per block/stage = stage total minus
  the enclosed optimize time"), the Addendum-20 degraded-verdict rule.
- `p515_s36_step36_timing.py` (full file, before and after edits) — `PhaseTimingRecorder`,
  all 14 wrap installers, `derive_param_update_and_bookkeeping`, `analyze_phase_timing`.
- `p515_s36_step36_timing_run.py` (full file, read-only — NOT edited, NOT executed) — the
  measurement entry point that produced the v1 artifacts; its `parse_cycle_wall_times` /
  `parse_report_timing_nl_write_seconds` patterns, reused (reproduced, not imported) in the
  new re-analysis script.
- `p515_s36_step36_timing_checks.py` (full file, before and after edits).
- `network_data.py` (`NetworkData.optimize`, `:54-68`) and `network.py` (`_run_smopf`,
  `:673-777`, `_run_smopf_solver_attempt`, `:604-611`) — re-read to settle D1 by checking the
  ACTUAL nesting, not assuming it from phase names.
- `shared_energy_storage_data.py` (`_optimize`, `:1177-1330s`) — confirmed the ESSO path has
  no `clone()` call at all (no equivalent defect there).
- Data artifacts: `data/SRP1/Results/P515S36/step36_timing/{off,on}/*` (all files),
  `phase_timing_analysis.json` (v1), `bitwise_identity_check.json` (v1),
  `data/SRP1/Results/P56A/evals/p515s36_timing_{off,on}/logs/*` (per-block IPOPT logs written
  by the ORIGINAL measurement run — read-only, for the D3 IPOPT-log cross-check; not re-run),
  `data/SRP1/Results/P515S36_STEP36_TIMING_launch.log` (exit `0`),
  `data/SRP1/Results/P515S36_STEP36_TIMING_launch_attempt1_self_match_20260917T150548Z.log`
  and its exit-code file (`...exit_code_attempt1_self_match_...txt`, contents `1`).

---

## Files modified

- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_s36_step36_timing.py`
  — `derive_param_update_and_bookkeeping` (D1 fix + raise-on-negative contract);
  `analyze_phase_timing` (D2 cycle-scoping, D3 overhead reclassification, new
  `solve_bundle_subtimes` parameter, `initialization_totals`/`recovery_totals` outputs).
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_s36_step36_timing_checks.py`
  — added Checks 9, 10, 11 (D1/D2/D3 unit coverage on toy examples); corrected Check 7's
  stale "aggregate NL-write alone is determinate" expectation (that expectation was itself
  part of the pre-D3-fix picture).

## Files created

- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_s36_step36_timing_reanalyze.py`
  — the zero-solve re-analysis entry point (`SolveProfileGuard(permitted=())` armed for the
  whole script; `verify(expected_solves=0, expected_execs=0)` at the end).
- `data/SRP1/Results/P515S36/step36_timing/phase_timing_analysis_v2.json`
- `data/SRP1/Results/P515S36/step36_timing/bitwise_identity_check_v2.json`
- `data/SRP1/Results/P515S36/step36_timing/manifest_sha256_v2.json`
- `data/SRP1/Results/P515S36/step36_timing/zero_solve_checks_v3/{results.json,manifest_sha256.json}`
- This file.

`git status --porcelain -- shared_resources_planning.py network.py network_data.py
shared_energy_storage_data.py admm_parameters.py p515_g_g1_g4_admm_gates.py
data/SRP1/SRP1_params.json` — empty throughout this task; none of these were opened with
`Edit`/`Write`.

---

## Defects: confirmed / refuted

### D1 — bookkeeping went negative (CONFIRMED, root cause identified and fixed)

Re-read `network_data.py:54-68` (`NetworkData.optimize`), the function BOTH the DSO and the
TSO paths call into (confirmed both use the same `NetworkData` class,
`shared_resources_planning.py:72`, `grep` for `TransmissionNetwork(`/`DistributionNetwork(`
found no such subclasses):

```python
def optimize(self, model, from_warm_start=False, ...):
    ...
    pre_solve_model = model[year][day].clone()                              # :61
    results[year][day] = self.network[year][day].run_smopf(model[year][day], ...)  # :62-63
```

`clone()` (:61) is a **sibling** call that runs **before** `run_smopf()` (:62-63 — the exact
function `p515_s36_step36_timing.py` wraps as `block_total`), **not** nested inside it.
`block_total`'s measured span (the `Network.run_smopf` wrap) therefore never includes any
`clone()` time at all. v1's `derive_param_update_and_bookkeeping` subtracted `clone` from
`block_total` anyway (`bookkeeping = block_total - clone - solve_bundle - load_solution -
diagnostics_parse`), subtracting time that was never part of that span — exactly why
`overhead_local_components.bookkeeping` reached **-8.37 s** in v1's aggregate.

**Fix**: `clone` removed from the bookkeeping subtraction (kept as its own separately-reported
phase, still summed into `overhead_local_components.clone`). **Verified against the real
captured data**, not only the toy checks:

```
bookkeeping min/max/sum (v1 formula, real ON-run records): -0.1087 / 0.0081 / -5.80  (dso: -5.80 sum; negative)
bookkeeping min/max/sum (fixed formula, real ON-run records): 0.00035 / 0.0081 / 0.1128  (ALL non-negative)
```

(`tso`'s v1 bookkeeping was also negative: median -0.1047 s per solved block — same defect,
confirmed fixed the same way.)

**Deliverable requirement — raise, never silently report a negative duration**: implemented.
`_raise_if_negative_duration` (new) raises `ValueError` whenever a derived `param_update` or
`bookkeeping` record's `elapsed_s` is more negative than a `1e-6` floating-point-noise
tolerance. Verified two ways:
1. A synthetic nesting-defect toy (`block_total` shorter than the `solve_bundle` it is
   supposed to enclose) raises `ValueError` (Check 9a).
2. Applying `derive_param_update_and_bookkeeping` to the v1 formula on the REAL captured
   records (i.e. reproducing D1's own defect) — confirmed by hand outside the committed
   checks script, not shipped as production behavior — would have raised under the NEW
   contract, since the v1 formula's outputs are more negative than the tolerance. The FIXED
   formula, run on the same real records, produces zero raises (all non-negative), confirming
   the fix resolves the actual measured defect, not only a synthetic one.

A regression guard (Check 9b) confirms `clone` at the same block as a *correctly*-nested
`block_total` is never subtracted (bookkeeping = block_total − solve_bundle − load_solution
only; if `clone` were still subtracted, the same toy would go negative and raise, per Check
9a's own contract).

### D2 — window inconsistency (CONFIRMED: 156 pre-loop initialization records, not a scope leak elsewhere)

`data/SRP1/Results/P515S36/step36_timing/on/phase_timing_records.jsonl` has 537 raw records;
**156 of them carry `cycle: null`** (seq range 1–156, strictly BEFORE cycle 1's first record at
seq 157) — the pre-ADMM-loop initialization solves (`create_distribution_networks_models` /
`create_transmission_network_model` / `create_shared_energy_storage_model`), which go through
the SAME wrapped call sites (`Network.run_smopf`, `SED._optimize`, `OptSolver.solve`,
`ModelSolutions.load_from`) but outside any `update_..._and_solve(cycle=...)` stage wrap:

```
cycle=None breakdown: block_total  dso=36  tso=12  esso=3
                      solve_bundle dso=36  tso=12  esso=3
                      load_solution dso=36 tso=12  esso=3
                      diagnostics_parse esso=3
```

36 DSO blocks = 3 nodes × 12 (year, day); 12 TSO blocks; 3 ESSO nodes — exactly the full
initialization sweep. v1's `analyze_phase_timing` summed `records` directly (no cycle filter),
so these 156 records were silently folded into the "2-cycle" totals — explaining v1's DSO
`solve_bundle` count of **109** (36 init + 36 cycle 1 + 37 cycle 2, the +1 being the one
recovery re-solve) rather than 73 (the correct 2-cycle count), and inflating every other
DSO/TSO/ESSO total by the same initialization contribution.

**Fix**: `analyze_phase_timing` now restricts every total/table to `cycle in {sampled cycles}`
(`in_scope` records); the excluded `cycle is None` records are reported separately under
`initialization_totals` (never dropped, never mixed in). The one recovery re-solve (D2's
second part) — confirmed from `on/network_failures_on.jsonl`: DSO node 5, `case33_1`,
2030/Winter, **cycle 2**, tier-1 recovery, succeeded — is reported separately under
`recovery_totals` (1 DSO `solve_bundle` + 1 DSO `load_solution`, both within cycle 2, still
also counted once in the main cycle-2 totals since it IS part of that cycle's real work).

Verified against real data: `per_phase_by_agent.dso.solve_bundle.count` in v2 = **73**
(36 + 37); `initialization_totals.dso.solve_bundle.count` = **36**, sum = **17.59 s**.

### D3 — classification error (CONFIRMED, and materially changes the verdict)

v1 put `solve_bundle_remainder_(ipopt_plus_sol_parse)` (**61.9 s**) entirely into
`overhead_SERIAL`. This bucket bundles the real IPOPT subprocess compute time (parallelizable
solve work, already handled separately by the LPT 8-worker projection) together with the
`.sol`-parse remainder (genuinely local, per-block overhead) — classifying ALL of it as serial
overhead both double-penalizes the IPOPT time (once by exclusion from `overhead_local`, a
second time by inflating `overhead_serial`) and misclassifies the `.sol`-parse share.

**Fix**: this task's `on/stdout_on.log` (captured with `inject_report_timing=True`, per design
§2.5's one-off cross-check — already the ON run's own configuration, not a re-run) contains
**924** `report_timing` lines = exactly **154** six-line groups (`seconds required to write
file` / `for presolve` / `for solver` / `to read logfile` / `to read solution file` / `for
postsolve`), one per `solve_bundle` event, in call order — matching the recorder's own 154
`solve_bundle` records **exactly** (count match asserted in code; the re-analysis script
raises if they do not match, rather than zipping mismatched lists). Glue (measured
`solve_bundle.elapsed_s` minus the three measured sub-times) is small: max +0.076 s, min
-0.0084 s (2-decimal-rounding noise, not a misalignment signal), mean +0.0043 s — consistent
with genuine 1:1 ordering, not a coincidence.

`analyze_phase_timing` now takes `solve_bundle_subtimes` (`{seq: {nl_write, ipopt,
sol_parse}}`) and, when it fully covers every in-scope `solve_bundle` record, computes:

```
overhead_total  = wall_total - ipopt_total                (cross-checked, not assumed)
overhead_local  = param_update + clone + nl_write(b) + load_solution(d2)
                 + bookkeeping + diagnostics_parse(e) + sol_parse(d1) + solve_bundle_glue
overhead_serial = admm_global(f) + unattributed
verdict_ratio   = overhead_local / overhead_total
```

Partial coverage (fewer than all in-scope `solve_bundle` records have subtimes) falls back to
the OLD aggregate-only classification and is a **hard non-verdict** — Addendum 20's "NL-write
not separated" rule is extended, since D3 showed that knowing only the NL-write share is
*also* insufficient (the true defect was guessing the ipopt/sol-parse split, not only the
NL-write share). Verified: `solve_bundle_subtimes_fully_covered` is `True` for this
measurement's real data.

**Cross-check (design §2.5's second half)**: summing `report_timing`'s "seconds required for
solver" across ALL 154 solves gives **53.82 s**; summing the per-block IPOPT logs' own "Total
seconds in IPOPT" line (`data/SRP1/Results/P56A/evals/p515s36_timing_on/logs/*`, 154
occurrences — read-only, these logs were written by the ORIGINAL measurement run, not
regenerated) gives **45.70 s**. The **8.12 s** gap matches design §2.5's own prediction
("subprocess launch overhead, not IPOPT compute") — both figures are over all 154 solves
(init + cycle 1 + cycle 2 combined), since the DSO/TSO IPOPT logs commingle cycles via
`file_append='yes'` and cannot be reliably split by cycle from the log files alone.

### D4 — bitwise identity (CONFIRMED: identical once `log_path` is excluded)

v1's single diff was in `esso_complementarity_diagnostics_by_round`; re-inspecting it
(`bitwise_identity_check.json`) shows the ENTIRE diff is the `log_path` field inside each
`per_round[*].entries[*]` dict — necessarily different strings (`.../p515s36_timing_off/logs/
optim_log_esso_node5_init.txt` vs `.../p515s36_timing_on/logs/optim_log_esso_node5_init.txt`,
differing only in the `off`/`on` output-directory component). Every OTHER field in every
entry (`complementarity_ratio_max`, `mu_final`, `spurious_throughput_bound`, etc.) is
identical.

**Fix**: `bitwise_diff_v2` (in `p515_s36_step36_timing_reanalyze.py`) recursively strips
`IDENTITY_EXCLUDED_FIELDS = ('log_path',)` — this single, explicitly-named field, never a
wildcard/substring match — from both sides before comparing, re-derived from the SAME saved
`off/g_off.json` / `on/g_on.json` + `soh_floor_sidecar_{off,on}.jsonl` artifacts the v1
`_bitwise_diff` used (never re-running the harness — doing so would overwrite the v1 evidence
this report cites). Result: **`identical: true`, `n_diffs: 0`**
(`bitwise_identity_check_v2.json`). `cycle_trajectory` and the SoH floor sidecar had zero
diffs even before exclusion (no path/label fields in either).

---

## Corrected table

**Aggregate (both sampled cycles, cycle-scoped per D2)**

| Quantity | v1 (defective) | v2 (corrected) |
|---|---:|---:|
| DSO `solve_bundle` count (2-cycle) | 109 (incl. 36 init) | 73 |
| `overhead_local_components.bookkeeping` | **-8.37 s** | **+0.087 s** (dso 0.073 + tso 0.014 + esso 0.003) |
| `overhead_local_total` | 25.25 s | 33.93 s |
| `overhead_serial_total` | 62.76 s | 0.89 s |
| `overhead_total` | 88.01 s (> wall!) | 34.82 s (cross-check: wall−IPOPT = 34.02 s) |
| `ipopt_total` (cycle-scoped) | not computed | 39.85 s |
| `verdict_ratio` (X=70%) | 0.287 | **0.974** |
| `verdict_pass` | **False** | **True** |
| `f_serial` | 0.958 | 0.082 |
| `speedup_at_8_workers` | **1.038×** | **5.078×** |

**Per cycle (v2, corrected)**

| Quantity | Cycle 1 | Cycle 2 |
|---|---:|---:|
| wall (production's own `Iteration N: X s` print) | 35.14 s | 38.73 s |
| IPOPT total — DSO | 16.61 s | 20.41 s |
| IPOPT total — TSO | 1.38 s | 1.21 s |
| IPOPT total — ESSO | 0.12 s | 0.12 s |
| IPOPT total (all agents) | 18.11 s | 21.74 s |
| overhead_local (param_update+clone+nl_write+load_solution+bookkeeping+diag+sol_parse+glue) | 16.95 s | 16.98 s |
| overhead_serial (admm_global+unattributed) | 0.451 s | 0.443 s |
| overhead_total = overhead_local+serial | 17.40 s | 17.42 s |
| overhead_total cross-check (wall−IPOPT) | 17.03 s | 16.99 s |
| unattributed residual | -0.0020 s | -0.0021 s |
| X ratio (overhead_local/overhead_total) | **0.9741** | **0.9746** |
| verdict at x_threshold=0.70 | **True (PASS)** | **True (PASS)** |
| f_serial | 0.0843 | 0.1336 |
| speedup_at_8_workers | **5.032×** | **4.134×** |
| recovery event in this cycle | none | DSO node 5, case33_1, 2030/Winter, tier-1 recovery, succeeded |

`overhead_total` vs its `wall − IPOPT` cross-check differ by ~0.37–0.43 s (~2.1–2.5%) per
cycle — attributable to independent measurement precision (perf_counter sub-millisecond
resolution for the recorder vs `report_timing`'s 2-decimal-rounded prints vs production's own
2-decimal `Iteration N: X s` print), not a defect; both figures are reported rather than one
being silently preferred.

**Initialization (cycle=None, reported separately, never mixed into the above)**

| Agent | solve_bundle count | solve_bundle sum |
|---|---:|---:|
| DSO | 36 | 17.59 s |
| TSO | 12 | 1.84 s |
| ESSO | 3 | 0.156 s |

**Comparison to `WORKER_REPORT_S36_PARALLEL_AUDIT.md`**: that report's medians (IPOPT 12.990 s,
overhead 21.488 s, wall 33.725 s) are over 477 cycles of a DIFFERENT 500-cycle-capped
reference run; this measurement is 2 cold cycles of a fresh run at the same s35ref
configuration class. Cycle-1 wall here (35.14 s) is close to that audit's median (33.725 s);
this measurement's cycle-scoped IPOPT totals (18.11 s / 21.74 s) run somewhat higher than the
audit's 12.990 s median — plausibly cold-cycle variance (cycle 1 here includes a cold start,
and cycle 2 includes the one recovery re-solve) rather than a discrepancy needing
reconciliation; stated directionally, not resolved further (out of this task's scope, which is
re-analysis of the ALREADY-captured measurement, not a new comparison campaign).

---

## Commands / experiments run

- Manual, zero-solve sanity checks of `derive_param_update_and_bookkeeping` and
  `analyze_phase_timing` against hand-computed toy examples (via `python -c`, no file writes) —
  confirmed the D1/D2/D3 fixes before wiring them into the committed checks script.
- `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s36_step36_timing_reanalyze.py`
  — `SolveProfileGuard(permitted=())` verified 0 permitted / 0 blocked solves. Produced
  `phase_timing_analysis_v2.json`, `bitwise_identity_check_v2.json`, `manifest_sha256_v2.json`
  (write-once; refuses if any already exists).
- `P515_S36_CHECKS_OUT_DIR=data/SRP1/Results/P515S36/step36_timing/zero_solve_checks_v3
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s36_step36_timing_checks.py`
  — **61/61 checks passed** (50 original/follow-up + 11 new: Check 9 ×2, Check 10 ×3,
  Check 11 ×6), `SolveProfileGuard` verified 0/0 solves.
- `ps aux | grep -i "p515_g_g1_g4_admm_gates.py\|p515_s38_"` — run before, during, and after
  every edit and every script execution: PID `74523` (`p515_g_g1_g4_admm_gates.py s38_b_pfbal`)
  alive throughout; never touched.
- `git status --porcelain -- shared_resources_planning.py network.py network_data.py
  shared_energy_storage_data.py admm_parameters.py p515_g_g1_g4_admm_gates.py
  data/SRP1/SRP1_params.json` — empty throughout.

---

## Results

- D1 confirmed and fixed: bookkeeping's negative values (v1 aggregate -8.37 s) traced to an
  incorrect `clone` subtraction; fixed formula produces zero negative values on the real
  captured data (min 0.00035 s).
- D2 confirmed and fixed: 156 pre-loop initialization records (cycle=None) were being summed
  into the "2-cycle" totals; now excluded and reported separately (`initialization_totals`);
  the one recovery re-solve reported separately too (`recovery_totals`).
- D3 confirmed and fixed: the full solve_bundle-minus-NL-write remainder (61.9 s in v1) was
  being classified as 100% serial overhead, including real IPOPT compute time; the corrected
  classification (via a per-solve `report_timing` b/c/d1 split) changes the verdict from
  **False (ratio 0.287)** to **True (ratio 0.974)**, and the projected 8-worker speedup from
  **1.038×** to **5.078×** (aggregate), **5.03×**/**4.13×** per cycle.
- D4 confirmed and fixed: the sole v1 diff (`log_path`) is a path field that necessarily
  differs between the OFF/ON output directories; excluding it (explicitly, by name) yields
  **0 diffs** — the instrumentation is bitwise-identical to production with the recorder off.

---

## Validation

- **Code executes correctly**: yes — `p515_s36_step36_timing.py` imports and runs cleanly;
  `p515_s36_step36_timing_reanalyze.py` completes with exit 0 and 0 solves (guard-verified).
- **Test passes**: yes — 61/61 zero-solve checks pass, including 11 new checks directly
  targeting D1/D2/D3 on toy examples (both the defect-reproduction case and the fix).
- **Requested diagnostic works**: the corrected `analyze_phase_timing` reproduces its own
  hand-computed toy examples exactly (Checks 5, 10, 11) and, applied to the REAL captured
  records, eliminates the negative-bookkeeping defect and produces a self-consistent
  `overhead_total` (cross-checked two independent ways, agreeing to ~2%).
- **Underlying performance question (persistent-worker redesign screening, X≥70%)**: this
  measurement (2 cold cycles) now clears the bar decisively (ratio 0.974 aggregate, 0.974/0.975
  per cycle) — a CORRECTED, not merely re-computed, verdict. This is one 2-cycle cold-start
  measurement; it is NOT a claim that the persistent-worker redesign will in fact deliver
  ~5× (design §5's own caveat: "a screening rule... it does not certify that 3× will in fact
  be reached").

---

## Unexpected findings

- The verdict reversal (False→True) is large and entirely attributable to a classification
  bug, not new data — the underlying measurement (which records exist, their durations) is
  unchanged between v1 and v2; only how they are SUMMED into `overhead_local` vs
  `overhead_serial` changed. This is exactly the class of defect CLAUDE.md's evidence rules
  warn about (a derived quantity's formula, not only its inputs, must be preserved and
  correct).
- The `report_timing` capture already present in `on/stdout_on.log` (from the ORIGINAL
  measurement run's `inject_report_timing=True`) was sufficient to fully resolve D3 with NO
  new solve — the one-off cross-check design §5 specified was captured correctly the first
  time; only the analysis code failed to use it correctly (it used only the aggregate
  NL-write sum, not the full per-solve b/c/d1 split the same stdout already contained).
- The per-block IPOPT logs under `data/SRP1/Results/P56A/evals/p515s36_timing_on/logs/`
  commingle cycles for DSO/TSO (`file_append='yes'`, confirmed: one log file contains 3
  "Total seconds in IPOPT" occurrences — init, cycle 1, cycle 2 — in that order), so they
  could only be used as an aggregate cross-check, not to independently re-derive a per-cycle
  IPOPT split (the `report_timing`-vs-`solve_bundle` seq alignment already provides that, more
  reliably).

---

## Remaining issues

- This is a 2-cycle, cold-start, single-instance measurement (design §5's own scope) — the
  X≥70% verdict and the 5.03×/4.13× per-cycle speedup projections are not a certified
  multi-cycle result; a warm/longer-run remeasurement was explicitly out of this task's
  scope (zero-solve re-analysis only).
- One recovery event occurred (DSO node 5, cycle 2, tier-1, succeeded) — reported separately
  (`recovery_totals`), but this means cycle 2's numbers are not a "clean" cycle in the sense
  of having zero retries; stated as a limitation, not smoothed over.
- The refused launch attempt (`P515S36_STEP36_TIMING_launch_attempt1_self_match_...log`,
  exit code `1`) is a genuine limitation of substring-based process scanning: the precondition
  check's `ps aux` scan matched the LAUNCHING SHELL's own command text (which contained the
  literal string `p515_g_g1_g4_admm_gates.py` inside a quoted shell-eval string), not an
  actual running instance of that harness — a false positive, nothing ran, preserved as
  evidence of the limitation rather than deleted.
- `overhead_total` and its `wall − IPOPT` cross-check differ by ~2–2.5% per cycle (measurement
  precision across three independently-clocked sources, not a formula defect) — reported as
  two numbers, not reconciled into one, per CLAUDE.md's "report a difference with its
  resolution" rule.
- The P56A/evals per-block IPOPT logs used for the D3 aggregate cross-check
  (`data/SRP1/Results/P56A/evals/p515s36_timing_{off,on}/logs/`) are NOT part of this task's
  committed evidence set (outside the `off/`+`on/` directories the Planner's deliverable list
  names) — they are cited by aggregate figure only; the underlying log files remain on disk,
  uncommitted, exactly as the original measurement run left them.

---

## Questions for Planner

- The per-cycle table (`per_cycle_analysis` in `phase_timing_analysis_v2.json`) re-runs
  `derive_param_update_and_bookkeeping`/`analyze_phase_timing` on each cycle's record subset
  independently (rather than slicing the aggregate result) — confirmed this reproduces the
  same aggregate when summed (spot-checked: aggregate `ipopt_total` 39.85 s = cycle-1 18.11 s
  + cycle-2 21.74 s exactly). Is this per-cycle breakdown format acceptable, or would the
  Planner prefer per-cycle figures embedded directly in `per_phase_by_agent`'s existing
  `per_cycle` sub-dicts instead of a separate top-level key?
- Should the P56A/evals per-block IPOPT logs (used only for the D3 aggregate cross-check) be
  brought into the committed evidence set in a follow-up task, or is the aggregate figure
  (cited, not hashed) sufficient documentation?
