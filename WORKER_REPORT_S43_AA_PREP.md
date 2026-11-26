# Worker Report — P5.15 Step 3.7 integration: flag-off gate (run) + flag-on run harness (build, smoke only)

## Task received

Build and RUN the flag-off two-cycle bitwise gate for Anderson acceleration
(AA); build the flag-ON run harness and run only a `--smoke-cycles 3` smoke
test of it (the Planner launches the full 300-cycle run). Per
`PLANNER_BRIEF_2026-09-13.md` Addendum 23 item 3.7 amendment (v), Addendum 24
("Step 3.7 code choices accepted"), frozen spec v12
`data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json`
(`item4_step_3_7_anderson`) and frozen spec v13
`data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json`.

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` Addenda 22–24 (read-only).
- `WORKER_REPORT_S41_AA.md` (AA design, w/F/g mapping, safeguard, per-cycle
  `aa_*` fields, `peak_rss_*`, "What the harness must add" section).
- Frozen specs v12 / v13 (full).
- `admm_anderson_acceleration.py` (full — module docstring, `build_iterate_
  layout`, `collect_w`/`write_back_w`, `AndersonAccelerationState`,
  `combined_scaled_residual`).
- `admm_parameters.py` (`self.anderson_acceleration` field, default off).
- `shared_resources_planning.py` (the `aa_*` `admm_diagnostics` dict literal
  ~3147–3158; `peak_rss_ru_maxrss`/`peak_rss_platform_units` ~3248–3271, in
  `state`, NOT copied into `report`; `_get_admm_rho_channel_scalars_for_aa`,
  `_anderson_acceleration_cycle_step`).
- `p515_g_g1_g4_admm_gates.py` (`run_admm_arm`, `_construct_arm_planning`,
  `S39_ARMS`/`_s39_ids_for_mode`/`_s39_working_dir_ids`/`run_s39_arm`,
  `_derive_stopped_by_from_trajectory`, `write_boyd_terminal_s35ref` →
  `write_interface_settlement_detail_s31c` → `write_component_levels_
  terminal`, `_acquire_exclusive_run_lock`, S39 module-level constants).
- `p515_s40_clone_capture_preflight.py` (full — the bitwise comparator
  `_diff`/`EXCLUDE_KEY_NAMES`/`EXCLUDE_DOTTED_SUFFIXES`/
  `INTENTIONAL_DIFF_SUFFIXES`, as fixed in `8f5cff48`; precondition
  conventions).
- `p515_s40_polish_gap.py` (`_reproduction_check`, `_derive_top_level_from_
  rows`, `ARM_LABEL`, `FULL_NUM_MAX_ITERS`, `D_REFERENCE_PATH`,
  `D_CERTIFICATION_CYCLE`, `D_CERTIFIED_COST`, `_build_floor_rows`,
  `_switch_to_base_objective`).
- `p515_s41_hull_polish.py` (full — `hull_entries_with_esso`, `apply_hull_
  bounds`, `_polish_all_blocks_hull`, `_hull_bounds_active`, gate
  construction, IPOPT-default bound-push override).
- `p515_s40_cost_decomposition.py` (full — `RUNS`, `PRICED_COMPONENT_KEYS`,
  `DETECTOR_COMPONENT_KEYS`, `PURE_REPORTING_KEYS`, the dominant-two-
  components reconciliation method, section 4).
- `p515_s42_exact_fix_rerun.py` (`_persist_certified_models`,
  `--persist-certified-models` CLI convention).
- `data/SRP1/Results/P515S39_D_run/g_s39_D.json` (D's committed 139-cycle
  trajectory) and `data/SRP1/Results/P515S40/clone_preflight_v2/lightweight/`
  (2-cycle reference, commit `8214be0d`).
- `data/SRP1/SRP1_params.json` (`admm` block — confirmed it already encodes
  every D-oracle override).
- `git log` for the commits touching `shared_resources_planning.py`/
  `network.py`/`admm_parameters.py` since `clone_preflight_v2` was captured
  (`8214be0d`), to scope what could legitimately differ against that
  reference (`16a19456` bound-restore fix, `9a965494`/`9ced0ad4`/`84257415`
  AA integration).

## Files modified

None (production code untouched).

## Files added

- `p515_s43_aa_flagoff_gate.py` — Task 1 harness (run).
- `p515_s43_aa_run.py` — Task 2 harness (built; smoke run only).
- `WORKER_REPORT_S43_AA_PREP.md` — this report.

## Changes made

### Task 1 — `p515_s43_aa_flagoff_gate.py`

Runs `run_s39_arm('s39_D', num_max_iters_override=2, output_root_override=...,
mode_label_override='preflight_s43flagoff')` with `admm_parameters.anderson_
acceleration['enabled']` explicitly forced to `False` (via the same
`_s39_configure_hook`-wrapping technique `p515_s40_clone_capture_
preflight._capture_mode_hook_override` already uses), instrumented with
pass-through call counters on every `admm_anderson_acceleration` function/
method that reads or writes the iterate plus the two `shared_resources_
planning.py` orchestration helpers (`_get_admm_rho_channel_scalars_for_aa`,
`_anderson_acceleration_cycle_step`) — asserts the total is exactly 0.
`anderson_acceleration_enabled` (the flag read) is counted separately,
informationally, not gated.

Compares the result bitwise against:
1. D's committed first two cycles, reusing `p515_s40_polish_gap.
   _reproduction_check` (truncated-mode) BY IMPORT.
2. `data/SRP1/Results/P515S40/clone_preflight_v2/lightweight/`, using
   `p515_s40_clone_capture_preflight._diff`/exclude-lists BY IMPORT.

Twelve new `aa_*` fields are added unconditionally to every
`admm_diagnostics` row by the AA integration commit, so both pre-AA
references necessarily lack them — `_classify_diffs` splits every raw diff
into `provenance_diffs` (`rule_eleven_checklist` subtree — non-gating, per
this task's own instruction), `aa_new_field_diffs` (reference `<MISSING>`
AND the new value matches the documented flag-off constant — non-gating),
`known_tie_break_diffs` (see Unexpected Findings — non-gating, evidenced),
and `genuine_diffs` (gates). Also wraps `shared_resources_planning.
_run_operational_planning` to capture the returned `state` dict and
recursively scans every compared JSON artifact for a `peak_rss` key.

### Task 2 — `p515_s43_aa_run.py`

The case file (`data/SRP1/SRP1_params.json`) already encodes D's full oracle
configuration (Step 3 closure) — verified directly (see Results). The
harness therefore runs the SAME "case-file-alone" invocation `p515_s41_
hull_polish.py`/`p515_s42_exact_fix_rerun.py` already use
(`run_admm_arm(label='s39_D', apply_rho=False, num_max_iters_override=300,
full_diagnostics_in_rows=True)`), with a `pre_solve_hook` whose ONLY action
is verifying the case file's D-oracle fields and setting `anderson_
acceleration['enabled'] = True` (memory 5 / regularization 1e-10 already
the `ADMMParameters` defaults, verified not assumed).

Standard s39-style captures (`s38_pf_capture_hooks`, `s39_exempt_until_
capture_hooks`, `write_boyd_terminal_s35ref` → `write_interface_settlement_
detail_s31c` → `write_component_levels_terminal` → `write_interface_voltage_
terminal`/`boyd_terminal.json`) are unchanged, PLUS: `aa_per_cycle.jsonl`
(derived post-hoc from the SAME `aa_*` fields already in `cycle_trajectory`
— no new computation) and `peak_rss_ru_maxrss`/`peak_rss_platform_units`
from the returned `state` dict.

IN-PROCESS, only if the run certifies (`stopped_by == 'boyd'` via
`p515_g_g1_g4_admm_gates._derive_stopped_by_from_trajectory`, BY IMPORT,
unchanged — corrects `write_boyd_terminal_s35ref`'s own naive `stopped_by`)
AND it is not a `--smoke-cycles` run, evaluates:
- (a) certification cycle (`report['cycles_run']`) vs ≤ 80.
- (b) `|Q − 650966975.2943751| ≤ 1.5e-4 × D` (= 97645.05).
- (c) cost decomposition vs D — re-applies `p515_s40_cost_decomposition.py`'s
  own method/constants (`PRICED_COMPONENT_KEYS`/`DETECTOR_COMPONENT_KEYS`,
  imported) to the (D, AA) pair; that script's `main()` is hardcoded to four
  specific committed runs and is not factored into a callable two-run
  function, so the METHOD is re-applied, not the whole script imported —
  every key list is the imported module constant. A reconciliation
  tolerance (5% of the headline diff, or an absolute $1,000 floor,
  whichever is larger) is used to report `reconciles_informal`; this
  tolerance is HARNESS-CHOSEN (the frozen spec's gate text states no
  number) and flagged as such in the payload for Planner confirmation.
- (d) the interval-hull polish gap, reusing `p515_s41_hull_polish.
  _polish_all_blocks_hull` (which itself calls `hull_entries_with_esso`/
  `apply_hull_bounds`/`_hull_bounds_active`) BY IMPORT, UNCHANGED.
- (e) optional (`--persist-certified-models`) certified-model persistence,
  reusing `p515_s42_exact_fix_rerun._persist_certified_models` BY IMPORT,
  UNCHANGED, called BEFORE the hull polish mutates `models`.

No reproduction-vs-D check (an AA-on run is expected to differ); the full
trajectory is recorded instead. A zero-solve precheck (armed
`SolveProfileGuard(permitted=())`) runs before any solve: applies the
pre_solve_hook to a freshly built (unsolved) planning object, builds `admm_
anderson_acceleration.build_iterate_layout` on it, and asserts every `aa_*`
field name is present in `shared_resources_planning.py`'s own source.

## Commands / experiments run

All via the canonical interpreter
(`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`).

1. `python -m py_compile p515_s43_aa_flagoff_gate.py` / `p515_s43_aa_run.py`
   — clean.
2. `python -c "import p515_s43_aa_run as R; ..."` — import-only sanity
   check (no side effects at import time).
3. `python -u p515_s43_aa_flagoff_gate.py > data/SRP1/Results/P515S43/
   flagoff_gate_launch.log 2>&1` (attached, alone) — run TWICE: the first
   run (committed as `dbfd204a`'s code) surfaced two classification gaps in
   my OWN comparator (missing `rule_eleven_checklist` provenance filter;
   an unclassified pre-existing tie-break diff), fixed in `c8e642c2`
   (`p515_s39_d_arm_preflight_s43flagoff`/`p515s39_d_precheck_preflight_
   s43flagoff` eval working dirs from the first attempt were removed before
   the corrected re-run, since neither the code fix nor the first attempt's
   output had been committed as evidence at that point). Final run:
   exit code 0, `GATE_PASS=True`.
4. `python -u p515_s43_aa_run.py --smoke-cycles 3 > data/SRP1/Results/
   P515S43/aa_run_smoke_launch.log 2>&1` (attached, alone) — exit code 0.

No `screen`/`nohup`/`&` used at any point; both streams captured via shell
redirection in every invocation.

## Results

### Task 1 — flag-off gate: GATE PASSES

`data/SRP1/Results/P515S43/flagoff_gate/flagoff_gate_results.json`:
`gate_pass: true`.
- `zero_aa_calls: true` — every instrumented AA-module function/method and
  both orchestration helpers: 0 calls. `anderson_acceleration_enabled`
  (informational): 1 call (the single flag read at the top of
  `_run_operational_planning`), as expected.
- vs D committed (truncated, 2 rows): `n_diffs_raw=24`, all 24 classified as
  `aa_new_field_diffs` (the 12 new `aa_*` fields × 2 cycles, each
  `<MISSING>` in D's reference and matching the documented flag-off
  constant — `aa_enabled=False`, everything else `None`); `genuine_diffs=0`.
- vs `clone_preflight_v2/lightweight` (full artifact/sidecar comparison):
  `n_diffs_raw=27` → 3 `provenance_diffs` (`rule_eleven_checklist` subtree:
  the OLD run's `s40_tso_snapshot_capture_mode_override` marker vs this
  run's own new `s43_anderson_acceleration_*` markers — configuration
  bookkeeping, not behaviour), duplicated once more inside the
  `g_s39_D.json` artifact comparison (same 3, read from disk) = 3;
  24 `aa_new_field_diffs`; 4 `known_tie_break_diffs` (see Unexpected
  Findings); `genuine_diffs=0`.
- `aa_field_pattern_ok: true` — every row's `aa_enabled` is exactly `False`
  and every other `aa_*` field is exactly `None`, in every artifact.
- `peak_rss`: `appears_in_any_compared_artifact: false` —
  `peak_rss_ru_maxrss`/`peak_rss_platform_units` exist only in the returned
  `state` dict (captured directly: `state['peak_rss_ru_maxrss'] =
  2232647680`), never copied into `report`/`g_s39_D.json` or any other
  compared JSON artifact — confirmed by a recursive key scan, not by
  static reading of the source alone.

### Task 2 — flag-on run harness: smoke PASSES

`data/SRP1/Results/P515S43/aa_run_smoke/aa_run_results.json`:
- Zero-solve precheck: all checks pass (`aa_enabled_after_hook: true`,
  12,096 AA iterate entries, `S_ref=2.5`, every `aa_*` field name present
  in `shared_resources_planning.py`'s source, `solve_profile_guard_
  verify_0_ok: true`).
- 3-cycle run: `cycles_run=3`, `stopped_by='cap'`, `certified=False` (3
  cycles cannot reach the 10-consecutive-cycle bar — correct).
- `post_certification_items_evaluated: false`,
  `post_certification_skip_reason: "smoke test (post-certification items
  intentionally not evaluated)"`; `gate_a_certification_cycle`,
  `gate_b_cost_vs_d`, `gate_c_cost_decomposition`, `gate_d_hull_polish`,
  `persisted_models` all `None` — skipped cleanly, not attempted on an
  uncertified point.
- `aa_per_cycle.jsonl` (3 rows) — sensible: cycle 1 `aa_action="insufficient
  memory (m_k=0)"` with `aa_rho_changed_channels=["v"]` (a genuine rho
  change clears memory at the end of cycle 1); cycle 2 same action with
  `aa_rho_changed_channels=["pf","v"]`; cycle 3 still `m_k=0`
  (`aa_rho_changed_channels=[]`, but memory was cleared by cycle 2's rho
  change) — consistent with D's own known rho schedule ("V/PF ρ change at
  cycles 1–2 only", `WORKER_REPORT_S41_AA.md`). No accept/reject events yet
  at 3 cycles (expected — the first candidate is only possible once m_k≥1,
  i.e. cycle 4 onward with no intervening rho change).
- `peak_rss_ru_maxrss=2269413376` (bytes, macOS).

## Validation

- Code executes correctly: yes (both scripts, `py_compile` clean, real runs
  complete with exit code 0).
- Task 1 diagnostic (flag-off bitwise identity + zero AA calls): PASSES,
  with an armed instrumentation (not merely asserted).
- Task 2 harness: the SMOKE test validates the harness executes correctly,
  the AA sidecar is populated sensibly, and the post-certification gating
  logic skips cleanly on an uncertified point. It does NOT validate the
  gate items (a)–(e) themselves (require a certified point — only the full
  ≥ ~90-cycle run reaches that, which the Planner launches).
- The underlying claim "AA-on eventually certifies within the frozen spec's
  gate" is NOT evaluated by this task (explicitly out of scope — Worker
  runs neither the full flag-on campaign nor asserts its outcome).

## Unexpected findings

1. **A pre-existing, non-AA, harness-level tie-break non-determinism** in
   `p515_g_g1_g4_admm_gates.py`'s recourse-jump capture hook: it builds
   `objective_component_block_deltas` by iterating `set(flat_current) |
   set(flat_previous)` (tuples of `(block_key, component_name)` strings)
   then `list.sort(key=abs_delta, reverse=True)` — a STABLE sort.
   `'economic_market_cost'` and `'generation_cost'` are two always-
   numerically-identical aliases of the same block objective component
   (`shared_resources_planning.py:5158`), so their `abs_delta` ties
   EXACTLY, and the tie is broken by `set` iteration order, which depends
   on Python's per-process string-hash randomization (`PYTHONHASHSEED`,
   unset here). Reproduced directly: six independent `python3 -c
   "print(list({('k','economic_market_cost'),('k','generation_cost')}))"`
   invocations in the same shell produced BOTH possible orderings across
   the six separate processes. This is a diagnostic-only (never gates
   production cost) sidecar artifact, unrelated to AA or to this task's
   changes; reported to the Planner rather than fixed (out of this task's
   scope — no production file was touched).
2. **My own first flag-off-gate run had two classification gaps**
   (missing `rule_eleven_checklist` provenance filter for the lightweight
   comparison; the tie-break diffs above were initially unclassified,
   producing a spurious `GATE FAILED`). Both were fixed in the harness
   (not by adjusting the comparator's underlying exclude lists — a new,
   narrowly-scoped, evidenced classification bucket was added instead) and
   the corrected run passes; the first (failing) attempt's own output was
   never committed as evidence, so nothing here overwrites cited data.
3. **`run_s39_arm` has no `post_run_hook`/`state` extension point** — its
   own post-run capture (`_s39_hook`) is a local closure. Task 2 therefore
   calls `run_admm_arm` directly (mirroring `p515_s41_hull_polish.py`/
   `p515_s42_exact_fix_rerun.py`'s own established pattern) rather than
   `run_s39_arm`, which is possible only because the case file itself now
   encodes D's full configuration (confirmed directly against `data/SRP1/
   SRP1_params.json`) — the s39-style override hooks are no longer needed.

## Remaining issues

- The reconciliation tolerance in gate item (c) (cost decomposition vs D)
  is a WORKER CHOICE (5% of the headline diff or an absolute $1,000 floor),
  since the frozen spec's gate text ("reconciles to generation + internal
  flexibility cost") states no number. Flagged in the payload
  (`reconciliation_tolerance_is_harness_chosen_not_spec_specified: true`)
  for Planner confirmation or a different number.
- Items (a)–(e) of the frozen spec's gate are UNVALIDATED by this task
  (require the full run to reach a certified point).
- The pre-existing tie-break non-determinism (Unexpected Finding 1) means
  `recourse_jump_sidecar_baseline.jsonl`'s top-10 list can non-
  deterministically re-order tied-value entries between separate process
  invocations of ANY harness in this family, not just this one — flagged
  for the Planner's awareness, not fixed here (out of scope; no production
  file touched).

## Questions for Planner

1. Is the reconciliation tolerance chosen for gate item (c) (5% of the
   headline diff, or $1,000 absolute, whichever is larger) acceptable, or
   should a different number be specified?
2. Should the pre-existing recourse-jump-sidecar tie-break non-determinism
   (Unexpected Finding 1) be fixed (e.g. a deterministic secondary sort key
   such as `component` name) in a future bounded task, given it affects
   every harness in this family that reports a top-10 block-delta list?
3. Confirm the exact full-run command below before launching.

## Exact full-run command (Planner launches; Worker does NOT)

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \
    p515_s43_aa_run.py --persist-certified-models \
    > data/SRP1/Results/P515S43/aa_run_launch.log 2>&1
```
