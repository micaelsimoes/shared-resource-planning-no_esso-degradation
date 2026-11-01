# Worker Report — P5.15 Addendum 22 item (2), timing half: two defects in the 10-cycle re-measurement, fixed and re-run

## Task received

Diagnose and fix two defects the Planner found in the 10-cycle re-measurement
(`data/SRP1/Results/P515S36/step36_timing_10cyc/`, produced by
`p515_s36_step36_timing_run.py 10`, commit `ab27d065`'s parameterization):

1. **Defect 1 (clone attribution)** — the run showed 12 `clone` events/cycle
   attributed to `dso`, 0 to `tso`, apparently contradicting
   `p515_s40_clone_capture_preflight.py`'s "legacy 12/cycle → lightweight
   0/cycle" oracle-configuration finding. Determine the real per-agent,
   per-mode clone picture and fix the harness attribution if wrong.
2. **Defect 2 (sub-time coverage / INDETERMINATE verdict)** — at 10 cycles
   `solve_bundle_subtimes_fully_covered` was `false` and `per_cycle_analysis`
   was absent, degrading the verdict to `INDETERMINATE (NL-write share not
   separated)` (ratio 0.374) versus the 2-cycle measurement's determinate
   0.974. Find why and fix the parser (or report the shortfall per record).
3. **"Also"** — the run's own off-vs-on bitwise identity check reported 1
   diff (`esso_complementarity_diagnostics_by_round[...].log_path`); apply
   the same key-name exclusion already used in
   `p515_s40_clone_capture_preflight.py` (commit `8f5cff48`).
4. **Then re-run** the 10-cycle measurement into a **new** output directory
   (never overwriting `step36_timing_10cyc/`), attached, both streams
   captured, machine idle, and report the full table.

No edits permitted to `shared_resources_planning.py`, `network.py`,
`network_data.py`, `shared_energy_storage_data.py`, `admm_parameters.py`,
the case file, or `p515_g_g1_g4_admm_gates.py`.

## Files inspected

- `P5_15_STEP36_TIMING_DESIGN.md` (full) — call-path map, wrap-point design,
  §5 decision rule.
- `p515_s36_step36_timing.py` (full) — the recorder, wrap installers
  (`_install_clone`, `_install_solve_bundle`, `_install_load_solution`, ...),
  `derive_param_update_and_bookkeeping`, `analyze_phase_timing`.
- `p515_s36_step36_timing_run.py` (full, before and after edits) — the
  measurement entry point.
- `p515_s36_step36_timing_reanalyze.py` (full) — the D1–D4-fixed 2-cycle-only
  re-analysis this task's Defect-2 fix generalizes.
- `p515_s40_clone_capture_preflight.py` (full) — the oracle-config clone
  counter and its `log_path` exclusion precedent (D4).
- `WORKER_REPORT_S40_CLONE_PREFLIGHT.md` — confirms the preflight's clone
  counter is scoped to the TSO update function only.
- `shared_resources_planning.py`: `_run_operational_planning`'s
  `tso_pristine_base` construction (~`:2507-2515`);
  `update_transmission_coordination_model_and_solve` (~`:5150-5390`, both the
  `legacy_clone` branch and the lightweight `capture_block_mutable_state` /
  `apply_block_mutable_state` branch); `update_distribution_coordination_
  models_and_solve_sequential` (~`:5395-5495`, the node-7-only
  `snapshot_callback` wiring). Read-only.
- `network_data.py`: `NetworkData.optimize` (`:54-68`, the single clone call
  site, `:61`). Read-only.
- `admm_parameters.py:193` (`tso_snapshot_capture_mode = 'lightweight'`
  default). Read-only.
- Already-captured evidence read, zero-solve: `data/SRP1/Results/P515S36/
  step36_timing_10cyc/{off,on}/{phase_timing_records.jsonl,stdout_on.log,
  g_off.json,g_on.json,soh_floor_sidecar_*.jsonl}`,
  `phase_timing_analysis.json`, `bitwise_identity_check.json`.

## Files modified

- `p515_s36_step36_timing_run.py` — the harness entry point (diagnostic
  script, not production). No other file was edited.

## Changes made

All changes are in `p515_s36_step36_timing_run.py` only; `p515_s36_step36_
timing.py` was **not** modified (see Defect 1 finding — no code defect found
there).

1. **Defect 2 fix (root cause, not a parser bug).** `main()` previously
   called `analyze_phase_timing(...)` **without ever passing
   `solve_bundle_subtimes`** — only the aggregate `nl_write_seconds` figure.
   Since `analyze_phase_timing` defaults `solve_bundle_subtimes` to `{}` when
   omitted, `fully_covered` (`sb_seqs.issubset(solve_bundle_subtimes.keys())`)
   was `False` **by construction**, for every invocation of this script,
   2-cycle default included — the real per-solve `{nl_write, ipopt,
   sol_parse}` split (`build_solve_bundle_subtimes`, the D3 fix) only ever
   existed in `p515_s36_step36_timing_reanalyze.py`, a **separate** script
   hardcoded to the committed 2-cycle `step36_timing/` root, never run and
   not pointable at `step36_timing_10cyc/`. Added
   `parse_report_timing_groups` and `build_solve_bundle_subtimes_with_
   coverage` to `p515_s36_step36_timing_run.py` (same regexes/grouping logic
   as the reanalyze script, reproduced not imported, per this codebase's own
   convention for these sibling diagnostic scripts) and wired the result into
   `main()`'s `analyze_phase_timing` call. Also added the `per_cycle_analysis`
   loop (same pattern as the reanalyze script), which `main()` never computed
   before at any cycle count.

   Per the task's "fix the parser so coverage is exact or the shortfall is
   reported per record" instruction, `build_solve_bundle_subtimes_with_
   coverage` never raises and never silently drops a record: it reports one
   `{seq, cycle, agent, block, attempt, covered}` entry per raw `solve_bundle`
   record (`analysis['solve_bundle_subtime_coverage']['per_record']`), and
   only trusts the positional stdout-order↔call-order match when the raw
   `solve_bundle` record count and the complete `report_timing` group count
   are **exactly** equal (a partial/subset match after a count mismatch is
   not assumed safe, since the mismatch could occur anywhere in the
   sequence).

2. **"Also" fix.** Added `IDENTITY_EXCLUDED_FIELDS = ('log_path',)` and
   `_strip_excluded_fields` (recursive key-name stripping at any depth) to
   `_bitwise_diff`, exactly the technique already used in
   `p515_s40_clone_capture_preflight.py` (commit `8f5cff48`).

3. **Collision-avoidance for the re-run.** Added an optional third CLI
   argument (`argv[2]`, a free-text run-suffix label) so this corrected
   measurement could write to `step36_timing_10cyc_defectfix/` without ever
   touching `step36_timing_10cyc/` (the pre-fix evidence this report cites as
   superseded). Default behaviour (no suffix) is unchanged for every prior
   invocation shape.

4. **Defect 1: no code change** (see Finding below — the recorder's
   attribution was already correct).

## Commands / experiments run

1. Read-only validation of the fix against the **already-committed, pre-fix**
   10-cycle evidence, zero-solve, before touching anything:
   ```
   /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python3 -c "..."
   ```
   confirmed `build_solve_bundle_subtimes_with_coverage` gives `status='exact'`
   (564/564, 0 incomplete groups) and `analyze_phase_timing` with that input
   flips the verdict from `INDETERMINATE` (ratio 0.374) to `True` (ratio
   0.9705), and `_bitwise_diff` with `log_path` excluded gives
   `identical=True, n_diffs=0` — all read-only, no solve, no write.
2. `git add -- data/SRP1/Results/P515S36/step36_timing_10cyc
   data/SRP1/Results/P515S36_STEP36_TIMING_10CYC_launch.log` +
   `git commit --only ...` — committed the pre-fix 10-cycle run as superseded
   evidence (commit `20b6fd23`).
3. `git commit --only -- p515_s36_step36_timing_run.py` — committed the two
   code fixes (commit `9428c0b7`).
4. Preconditions re-checked immediately before launch: `.p515_g_gate.lock`
   absent; no `p515_g_g1_g4_admm_gates.py` / `p515_s38_` /
   `p515_s36_step36_timing_run.py` process alive; the full
   `_PRODUCTION_FILES_TO_CHECK_CLEAN` list clean in git (confirmed via
   `git status --porcelain`).
5. **The re-run** (attached, both streams captured, no `screen`/`nohup`/
   backgrounding — the shell command itself was run directly, `&`-free; the
   agent harness's own tool infrastructure moved the *monitoring* of the
   already-launched, already-attached process to a background poll after its
   2-minute synchronous window elapsed, which does not change how the
   `python` process itself was started):
   ```
   /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \
       p515_s36_step36_timing_run.py 10 defectfix \
       > data/SRP1/Results/P515S36_STEP36_TIMING_10CYC_DEFECTFIX_launch.log 2>&1
   ```
   Wall time: 789.3 s (~13.2 min) for both OFF and ON 10-cycle runs plus
   initialization, matching the script's own docstring estimate ("order
   10-20 minutes total").
6. Post-run inspection of `data/SRP1/Results/P515S36/step36_timing_10cyc_defectfix/
   phase_timing_analysis.json` and `bitwise_identity_check.json` (read-only).
7. `git add -- data/SRP1/Results/P515S36/step36_timing_10cyc_defectfix
   data/SRP1/Results/P515S36_STEP36_TIMING_10CYC_DEFECTFIX_launch.log
   WORKER_REPORT_S36_TIMING_10CYC.md` + `git commit --only ...` — committed
   the corrected evidence and this report together.

## Results

### Defect 1 — clone attribution: no harness bug found; corrected picture

**Call sites that actually clone**, by production code (read, not inferred):

| Call site | Agent | Gating | Mode dependence |
|---|---|---|---|
| `network_data.py:61`, `NetworkData.optimize` | DSO node 7 only | `failure_snapshot_callback` wired **unconditionally** for `node_id == 7` in `update_distribution_coordination_models_and_solve_sequential` (`shared_resources_planning.py`) — fires **every cycle**, all 12 `(year, day)` blocks of node 7's network | Independent of `tso_snapshot_capture_mode`; untouched by Step 3.6 |
| `network_data.py:61`, `NetworkData.optimize` | TSO (all 12 `(year,day)` blocks) | `failure_snapshot_callback` wired **unconditionally** in `update_transmission_coordination_model_and_solve`'s `else:` branch | **Only** when `admm_parameters.tso_snapshot_capture_mode == 'legacy_clone'`. Production default is `'lightweight'` (`admm_parameters.py:193`), which takes the `if tso_pristine_base is not None:` branch and calls `transmission_network.optimize(model, from_warm_start=...)` with **no callbacks at all** — `NetworkData.optimize`'s clone-firing condition is never true, so **0** clones here in the default mode |
| `shared_resources_planning.py` (~`:5264`, inside `update_transmission_coordination_model_and_solve`'s lightweight branch) | TSO, rare | `tso_pristine_base[year][day].clone()` fires **only** when `needs_failure_snapshot` (a network failure) or `needs_comparator_snapshot` (`cycle == 7`, 2025 Summer, success) | Lightweight-mode only. **Not wrapped by the recorder** (see below) — its own call frame is `update_transmission_coordination_model_and_solve`, not `NetworkData.optimize`, so `_install_clone`'s frame-walk (`_find_ancestor_frame('network_data.py', 'optimize')`) does not find it and passes it through untimed, exactly as `p515_s36_step36_timing.py`'s own module docstring says every out-of-scope clone is handled ("passed through UNTIMED, not folded into any bucket") |
| `shared_resources_planning.py` (~`:2513`), inside `_run_operational_planning` | TSO, once per run | `tso_pristine_base = {year: {day: tso_model[year][day].clone() ...}}` built **once**, before the ADMM loop starts | Lightweight-mode only; also not wrapped (out of the recorder's declared scope, same reasoning) |

**Recorder attribution (`p515_s36_step36_timing.py::_install_clone`) is
correct.** It attributes every `BlockData.clone()` call reached through
`NetworkData.optimize` by inspecting `frame.f_locals['self'].is_transmission`
on that ancestor frame — i.e. by which `NetworkData` instance's `.optimize()`
is on the call stack, exactly the two clone call sites that route through
`network_data.py:61`. This is by construction correct for both of those
sites; it simply does not (and, per its own documented scope, does not try
to) time the two additional clone call sites listed above that bypass
`NetworkData.optimize` entirely. **No code change was made to
`p515_s36_step36_timing.py`.**

**Why the oracle preflight (`p515_s40_clone_capture_preflight.py`) reported
"legacy 12/cycle → lightweight 0/cycle" while the timing run shows 12 DSO
clones/cycle in the default (lightweight) mode: this is not a
contradiction.** The preflight's clone counter is explicitly scoped — by its
own docstring and by inspection of `_clone_count_hooks` — to calls made
**while `srp.update_transmission_coordination_model_and_solve` is on the
stack**, i.e. it only ever counts TSO-path clones. It never wraps
`update_distribution_coordination_models_and_solve*` and therefore cannot
see, and never made any claim about, DSO clones.
`WORKER_REPORT_S40_CLONE_PREFLIGHT.md` states this explicitly ("legacy pays
one `BlockData.clone()` per [TSO] ... case A/B") — confirming the preflight's
"0/cycle lightweight" finding is a **TSO-only** measurement, fully consistent
with this timing run's "0 TSO clones" finding. The two measurements agree on
the TSO; the preflight simply has nothing to say about the DSO.

**Does the TSO clone removal save time in the oracle (s39_D) configuration
that actually runs campaigns?** Yes. `tso_snapshot_capture_mode` defaults to
`'lightweight'` at the `AdmmParameters` class level (`admm_parameters.py:193`)
— it is not gated by which arm/config (`s35ref`, `s39_D`, ...) is selected,
only by that one field, which no s39/s40 script overrides to `'legacy_clone'`
except inside the preflight's own deliberate A/B monkeypatch. So the
production TSO clone removal is in force for the oracle exactly as it is for
this timing harness's `s35ref`-class configuration; both measurements
(`p515_s40_clone_capture_preflight.py`'s TSO-scoped counter and this timing
run's full phase breakdown) are measuring the same code path.

**Is any remaining clone cost DSO-side and structurally removable by the same
technique?** As a finding (no fix proposed, per the task's instruction):
**yes** — the DSO node-7 `failure_snapshot_callback` wiring
(`update_distribution_coordination_models_and_solve_sequential`) is
structurally the same unconditional-callback pattern the TSO used to have
before Step 3.6's `capture_block_mutable_state`/`apply_block_mutable_state`
replacement, and it measures at **~3.1–3.2 s/cycle** across 12
`(year, day)` blocks (12 clone events/cycle, mean ≈0.26 s each; see the
per-cycle table below) — roughly **9%** of `overhead_local` and about **0.8%**
of total cycle wall time in this measurement. Stated as an observation only.

### Defect 2 — root cause and fix

**Confirmed root cause: a harness wiring gap, not a genuine report_timing
coverage failure at 10-cycle scale.** Checked directly against the
already-captured (pre-fix) 10-cycle `on/` run: 564 raw `solve_bundle` records
(51/cycle × 10 cycles + 3 tier-1 recovery retries + 51 pre-loop
initialization solves) against **564 complete** `report_timing` groups parsed
from that same run's own `stdout_on.log` — an **exact** match, **0**
incomplete groups. Retries, interleaved stdout, and the larger record count
did **not** break the parse at 10 cycles. The prior `INDETERMINATE` verdict
was produced because `p515_s36_step36_timing_run.py`'s `main()` never called
any function that builds the per-solve `{nl_write, ipopt, sol_parse}` split
in the first place — it only ever passed the aggregate `nl_write_seconds`
figure, so `analyze_phase_timing`'s `solve_bundle_subtimes` parameter
defaulted to `{}` and `fully_covered` was `False` by construction, for
**every** invocation of this script (the 2-cycle default included — the
0.974 ratio the Planner cites for the 2-cycle run came from the **separate**
`p515_s36_step36_timing_reanalyze.py` script, never from this entry point's
own output).

### The corrected 10-cycle re-run (`step36_timing_10cyc_defectfix/`)

Config: identical to the pre-fix run — `s35ref` configuration class
(`k_override=None`, `apply_rho=False`, `full_diagnostics_in_rows=True`,
`investment_map=None`), 10 ADMM cycles, cold start, OFF then ON,
`inject_report_timing=True` on the ON run only. `X_THRESHOLD = 0.70`
(unchanged, Addendum 20 interim figure).

**Solve-bundle coverage**: exact, 564/564, 0 incomplete groups
(`solve_bundle_subtime_coverage.status == 'exact'`,
`solve_bundle_subtimes_fully_covered: true`).

**Bitwise identity (OFF vs ON, `log_path` excluded)**: **identical = True, 0
diffs** (down from 1 in the pre-fix run — that 1 diff was entirely the
excluded `log_path` field, as directly verified by flattening the pre-fix
run's own diff payload: all 66 leaf-level differences under it were
`....log_path`, nothing else).

**Per-cycle wall / IPOPT / verdict:**

| cycle | wall (s) | IPOPT total (s) | verdict ratio | verdict |
|---|---|---|---|---|
| 1 | 32.49 | 17.35 | 0.9682 | PASS |
| 2 | 36.70 | 20.85 | 0.9699 | PASS |
| 3 | 30.79 | 16.56 | 0.9657 | PASS |
| 4 | 36.47 | 20.22 | 0.9703 | PASS |
| 5 | 37.78 | 21.29 | 0.9704 | PASS |
| 6 | 33.63 | 16.83 | 0.9712 | PASS |
| 7 | 36.73 | 19.32 | 0.9722 | PASS |
| 8 | 35.29 | 18.28 | 0.9717 | PASS |
| 9 | 37.41 | 20.32 | 0.9720 | PASS |
| 10 | 41.51 | 24.30 | 0.9712 | PASS |

**Aggregate (10 cycles, cycle-scoped, initialization excluded):**

- `wall_total` = 358.80 s, `ipopt_total` = 195.32 s (cross-check:
  `wall_total − ipopt_total` = 163.48 s vs. `overhead_total` = 153.94 s —
  the ≈9.5 s gap is the LPT-partitioned-vs-serial-sum difference already
  documented in `analyze_phase_timing`'s own derivation, not a new
  discrepancy).
- `overhead_local_total` = 149.38 s; `overhead_serial_total` = 4.56 s;
  `overhead_total` = 153.94 s.
- **Verdict ratio = 0.9704** (`overhead_local / overhead_total`), **PASS**
  against `X = 0.70`.
- `overhead_local_components`: `param_update` 4.40 s, `clone` 31.49 s (all
  DSO, 120 events = 12/cycle × 10), `nl_write_share_of_solve_bundle` 69.64 s,
  `sol_parse_share_of_solve_bundle` 22.87 s, `load_solution` 18.58 s,
  `bookkeeping` 0.62 s, `diagnostics_parse` 0.013 s,
  `solve_bundle_glue` 1.77 s.
- `overhead_serial_components`: `admm_global` 4.548 s, `unattributed`
  0.011 s.
- `f_serial` = 0.0816; **projected 8-worker speed-up = 5.092×** — well past
  the `S = 3` interim bar the design's §5 X-threshold derivation was built
  against, and closer to (though still short of) the 4–5× aspirational
  Addendum-18 target.
- **Every substantive `clone` cost is DSO-side** (120 events, 31.49 s total,
  0 TSO events) — matching Defect 1's finding above exactly.
- `per_phase_by_agent['tso']['clone']` = 0 events (`sum: None`); `dso` = 120
  events, `esso` = 0 events (ESSO has no clone call site in production at
  all — not touched by Step 3.6, no finding here).

**Comparison to the pre-fix (superseded) 10-cycle run**: wall/IPOPT/clone/
per-phase figures are numerically close but not identical between the two
runs (e.g. `wall_total` 353.01 s pre-fix vs. 358.80 s here; `ipopt_total`
degraded-mode `None` pre-fix vs. 195.32 s here) because these are two
**independent** cold-start ADMM runs of a stochastic-timing (wall-clock)
process, not a re-analysis of the same run — the pre-fix run's own
`bitwise_identity_check.json` and this run's both separately confirm OFF vs
ON determinism (the *ADMM trajectory* is bitwise reproducible); *wall-clock*
timing across two separate process launches is not expected to be, and was
never claimed to be, bitwise identical.

## Validation

- Fix validated **read-only**, before any write, against the already-
  committed pre-fix evidence (item 1 under Commands): confirms the fix logic
  itself (not just the fresh run) resolves both defects.
- Fresh, corrected 10-cycle run reproduces the same qualitative result
  (exact coverage, PASS verdict, clean identity check) as the read-only
  validation, on an independent process launch — code executes correctly,
  the diagnostic (`analyze_phase_timing`) now works as designed, and its
  output is now determinate. This validates the **harness fix**; it says
  nothing new about the underlying persistent-worker design question beyond
  what the corrected numbers report (8-worker projected speed-up 5.09×, up
  from the pre-fix run's spurious 1.39× INDETERMINATE-mode figure, which was
  never a real projection to begin with).
- `solve_bundle_subtime_coverage` in the written `phase_timing_analysis.json`
  carries the full per-record coverage list (564 entries, `covered: true`
  each) — inspectable directly, not just asserted by this report.
- Preconditions (lock, forbidden processes, clean production files, fresh
  output dirs) were checked and passed immediately before the launch; the
  run was launched attached (direct shell invocation, no `&`, no `nohup`, no
  `screen`) with both stdout and stderr captured (`2>&1` into the launch log,
  plus the script's own `tee_stderr` for the stderr half); no other
  `p515_g_g1_g4_admm_gates.py` / `p515_s38_` / `p515_s36_step36_timing_run.py`
  process was alive at launch time (checked via `ps aux`).
- The prior committed evidence (`step36_timing_10cyc/`) was **not**
  re-run onto — the corrected measurement was written to a new root
  (`step36_timing_10cyc_defectfix/`) via the added run-suffix argument, and
  the original directory's contents are unchanged (confirmed: this task
  never opened any file under `step36_timing_10cyc/` for writing after its
  own commit).

## Unexpected findings

- The 2-cycle `step36_timing/` measurement's own `phase_timing_analysis.json`
  (v1, produced by this same `_run.py` entry point) has **the same
  wiring gap** as the pre-fix 10-cycle run: `verdict_pass: False,
  verdict_ratio: 0.287, solve_bundle_subtimes_fully_covered: None`. The
  0.974-ratio, `PASS`, fully-covered result the design/Addendum-19 record
  cites for the 2-cycle measurement is entirely from the **separate**
  `phase_timing_analysis_v2.json` produced by
  `p515_s36_step36_timing_reanalyze.py`, not from `_run.py`'s own output.
  This task's fix makes `_run.py`'s own output match what the reanalyze
  script already established for 2 cycles, so this is now consistent across
  both cycle counts going forward, but is flagged as a pre-existing gap in
  the 2-cycle artifact set, not something this task altered (v1's
  `phase_timing_analysis.json` under `step36_timing/` was left untouched,
  per the "never re-run onto a cited artifact" rule).
- Two clone call sites (`tso_pristine_base[year][day].clone()` inside
  `update_transmission_coordination_model_and_solve`'s lightweight branch,
  and the one-time pristine-base construction in `_run_operational_planning`)
  are real production clone costs in the lightweight (default) mode that the
  recorder's `_install_clone` wrap point does **not** time, by explicit,
  documented design (its frame-walk only recognizes clones reached via
  `NetworkData.optimize`). The per-run one-time construction (12 clones,
  once) is negligible; the per-cycle rare-rebuild clone fires at most once
  per network failure or at the cycle-7 comparator — in this 10-cycle run,
  3 network failures (all tier-1-recovered) occurred, so up to 3-4 such
  untimed clones may have fired, contributing an unknown, uncounted, but
  bounded (single-digit-count, sub-second-each by analogy to the measured
  DSO clone cost) amount to whatever `analyze_phase_timing` reports as
  `unattributed`/`bookkeeping` residual. Not fixed (out of this task's scope
  — reported as a finding, no fix proposed, per the task's own instruction).

## Remaining issues

- The DSO node-7 clone (~3.1–3.2 s/cycle, ~9% of `overhead_local`) remains
  unaddressed — structurally the same pattern the TSO's Step 3.6 change
  already eliminated, but this task was scoped to diagnosis/reporting, not
  a further production change, per its explicit "state it as a finding;
  propose nothing" instruction.
- The pre-fix 2-cycle `phase_timing_analysis.json` (v1, under
  `step36_timing/`, not `_v2.json`) still reports the same wiring-gap
  `INDETERMINATE`-equivalent (`False`/0.287) result this task's fix would
  now avoid for any NEW run at that cycle count; it was not re-run or
  corrected in place (correctly, per the evidence-preservation rule), so a
  reader consulting `step36_timing/phase_timing_analysis.json` directly
  (rather than `_v2.json`) would still see the stale, uncorrected figure.
  Flagged for the Planner's awareness, not acted on.

## Questions for Planner

- Should `p515_s36_step36_timing_reanalyze.py` (the 2-cycle-only D1–D4
  reanalysis script) now be retired/superseded in favor of
  `p515_s36_step36_timing_run.py`'s own corrected `main()`, or kept as a
  standalone artifact documenting the original defect-discovery process? No
  change was made to it in this task (out of scope), but its logic is now
  duplicated (by design, per the sibling-script convention already in use)
  rather than shared.
- Is the DSO node-7 clone (Defect 1 finding, ~3.1–3.2 s/cycle) worth a future
  bounded task applying the same `capture_block_mutable_state`/
  `apply_block_mutable_state` technique Step 3.6 already applied to the TSO?
  This report states the finding only, per instruction, and takes no
  position on whether it is worth pursuing.
