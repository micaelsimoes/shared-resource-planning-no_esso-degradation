# Worker Report — P5.15 Addendum 24 item 1: `p56a_oracle._interface_expression`
# fix, zero-solve regression check, exact-fix re-run harness

**Task received from the Planner:** `PLANNER_BRIEF_2026-09-13.md` Addendum 24 and frozen spec v13
`data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json` (commit `cbdedbb4`), item
`item1_helper_fix` and `reporting_obligations`. Fix the stale interface helper; prove the fix with
a zero-solve regression check; prepare and smoke-test the exact-fix polish re-run at D's certified
point; report non-degenerate hull-bound counts. The Worker does **not** run the full-length
re-run — the Planner launches it.

---

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` Addendum 24 (task authority) and the relevant history: Step 3.5's
  restatement (Addendum 23), row 3′/Addendum 12 (the signed interface reparametrization the stale
  helper predates).
- `data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json` (binding spec).
- `WORKER_REPORT_S41_HULL_POLISH_PREP.md` §2 (the finding), `WORKER_REPORT_S41_POLISH_PREREQ.md`
  (the six-TSO/five-DSO failure classification the exact-fix rerun's predictions build on),
  `P5_15_S41_STEP3_CLOSED_REPORT.md` §3.
- `p56a_oracle.py`, full read; `_interface_expression`/`apply_common_values`/
  `common_coordinated_values` (~pre-fix lines 233-458).
- `model_construction_helpers.py:1160-1229` (`interface_pf_p/q_transmission_def`,
  `interface_pf_p/q_distribution_def`).
- `network.py:395-440` (`interface_delta_p/q` declaration and fix-at-0 default,
  `pc_adn`/`qc_adn`/`pg_adn`/`qg_adn` Expressions), `network.py:540-850`
  (`_create_smopf_solver`, `_run_smopf_solver_attempt`, `_run_smopf`, log-path/`output_file`
  construction, recovery/tier-2 control flow).
- `shared_resources_planning.py:3544-3628` (`create_transmission_network_model`'s interface-fixing
  block — the ADMM path's `pc`/`qc` fix-once, flex-legs-fixed-at-0, `interface_delta_p/q`-freed
  construction).
- `helper_functions.py` (`solver_result_succeeded`, `solver_result_summary`).
- `p515_s40_polish_gap.py` (full — the v11/VOID exact-fix harness whose semantics this task's
  Task 3 harness reuses; not modified), `p515_s41_hull_polish.py`/`p515_s41_hull_polish_checks.py`
  (the hull-polish harness and its checks — conventions reused: preconditions, lock, manifest,
  `_build_admm_ready_state` zero-solve pattern, `_hull_bounds_active`'s degenerate-inclusive
  convention).
- `p515_s31c_zero_solve_checks.py` (confirmed it matches only by name — its own
  `check_interface_expression_contains_pc_delta_no_legs` tests the production `pc_adn` Expression
  directly, never `p56a_oracle._interface_expression`).
- `p513_solve_profile_guard.py` (`SolveProfileGuard` API).
- `data/SRP1/case9/case9_params.json` (solver options, `output_file`/log-path convention).
- `WORKER_REPORT_S36_CLONE_CAPTURE.md:250-290` (the 7 preserved-fixture unpickle list, re-verified
  below).
- `data/SRP1/Results/P515S41/hull_polish/hull_bound_detail.json` and `hull_polish_results.json`
  (committed, commit `2e6c5570` — the source for the non-degenerate hull counts).

## Files modified

- `p56a_oracle.py` (the only production file touched, as authorized — "a shared oracle module,
  authorized by Addendum 24").

## Files created

- `p515_s42_interface_helper_checks.py` — Task 2's zero-solve regression check.
- `p515_s42_exact_fix_rerun.py` — Task 3's exact-fix re-run harness (new file; `p515_s40_polish_gap.py`
  is untouched).
- `p515_s42_exact_fix_rerun_checks.py` — Task 3's zero-solve checks for the harness's own helper
  logic (log parsing, attempt classification, interface diagnostics, prediction scoring, hash
  round-trip).
- `p515_s42_hull_counts.py` — Task 4's zero-solve hull-count re-derivation.
- This report.

---

## Task 1 — the helper fix

**Fix:** `p56a_oracle.py:394-395` (`_interface_expression`) now returns the model's OWN
`pc_adn`/`qc_adn` Expression directly:

```python
def _interface_expression(t_model, dn, p, kind):
    return t_model.pc_adn[dn, 0, 0, p] if kind == 'p' else t_model.qc_adn[dn, 0, 0, p]
```

`pc_adn`/`qc_adn` are built at `network.py:435-436` from `interface_pf_p_transmission_def`/
`interface_pf_q_transmission_def` (`model_construction_helpers.py:1167-1199`):
`pc[adn_load] + interface_delta_p[dn] + (fl_reg ? flex_p_up − flex_p_down : 0)` — the production
expression, including `interface_delta_p/q` (P5.15 Step 3.1-C / Addendum 12 item 2), which the old
hand-built form omitted. `apply_common_values` (`p56a_oracle.py:452-490` post-fix) now computes
`dn` (the ADN index, the same index `common_coordinated_values` already uses to read `pc_adn`) and
passes it instead of the ADN-load index the old form needed.

**Deactivate-and-unwire:** the old form is kept, callable, as
`_interface_expression_legacy_pre_addendum12` (`p56a_oracle.py:421-441`), documented as unwired
and not referenced anywhere in this module or (per the consumer scan below) elsewhere. It is *not*
embedded via `functools.partial` in any pickled model (`p56a_oracle.py` functions are never
imported by `model_construction_helpers.py`/`network.py`/`shared_resources_planning.py` —
confirmed by `grep`, the only two hits are comments), so no preserved fixture resolves it by name;
nothing was deleted regardless, per the rule.

**Verified every preserved fixture still unpickles** (the 7 listed in
`WORKER_REPORT_S36_CLONE_CAPTURE.md:275-282`), after the `p56a_oracle.py` edit:

| fixture | unpickled | `n_vars` |
|---|---|---:|
| `P512R/cycle21_pre_setup/snapshot.pkl` | True | 10,756 |
| `P512R/cycle21_prepared/snapshot.pkl` | True | 10,756 |
| `P512R/production_snapshots/FrozenSMOPF/matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl` | True | 10,756 |
| `P512R/production_snapshots/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl` | True | 3,756 |
| `FrozenSMOPF/matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl` | True | 10,882 |
| `FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl` | True | 3,762 |
| `P515F/t2_results/FrozenSMOPF/failure_TSO_case9_2025_Spring_cycle1.pkl` | True | 3,762 |

All 7 match the `n_vars` counts already on record — every count identical to
`WORKER_REPORT_S36_CLONE_CAPTURE.md`'s own table.

**`ORACLE_VERSION` bumped** `p56a.2 → p56a.3` (`p56a_oracle.py:68-73`) — the module's own
convention ("bumped whenever anything in this module could change a returned objective"); this
invalidates any cache entry (`p56a_cache.json`, untracked, local) produced under the defective
helper.

**Consumer list** (every caller of `apply_common_values`/`_interface_expression`/
`common_coordinated_values`, `grep`-confirmed repo-wide):

| consumer | uses | relative to Addendum 12 (commit `2783c223`, 2026-09-15) |
|---|---|---|
| `p58_eval.py`, `p512_a_cold_rescaled_convergence.py`, `p55d_d1_polish.py` (own local reimplementation, not `p56a_oracle`'s), `p56b_policy.py`, `p57_eval.py`, `p56a_a2_branch.py`, `p56b_b1_template.py`, `p57_hypotheses.py` | `common_coordinated_values`/`apply_common_values` | **predate** Addendum 12 (all last touched 2026-09-10 or earlier, confirmed via `git log`) — the Planner's stated scoping is confirmed correct, with one nuance recorded below |
| `p515_s40_polish_gap.py`, `p515_s40_polish_gap_checks.py` | same | **postdate** Addendum 12; this is the v11 exact-fix run, already recorded VOID by Addendum 24 and not re-run |
| `p515_s41_hull_polish.py` | imports `common_coordinated_values` only (read-only; bypasses `apply_common_values`/`_interface_expression` entirely, by design — see its own §2 finding) | unaffected either way |
| `p515_s31c_zero_solve_checks.py` | its own local `check_interface_expression_contains_pc_delta_no_legs`, which tests the production `pc_adn` Expression directly, **never** `p56a_oracle._interface_expression` | matches only by name, confirmed |

**Nuance on the "predates Addendum 12" scoping:** those P5.5–P5.8 harness *files* were last
committed before Addendum 12, so their own *committed, historical* evidence is unaffected by this
fix (it was produced under a model construction that did not yet have `interface_delta_p/q` at
all). But every one of those files calls the *shared production* `run_operational_planning`/
`create_transmission_network_model`, which today *does* include the Addendum-12 reparametrization
— so if any of them were **re-run now** (not requested; none was run for this task), it would
exercise the current ADMM path and would now benefit from the same fix, exactly as
`p515_s42_exact_fix_rerun.py` does. This is stated for completeness, not acted on.

---

## Task 2 — zero-solve regression check

`p515_s42_interface_helper_checks.py`: `SolveProfileGuard(permitted=())` armed for the whole
script, `verify(0)` checked before writing output. Builds a fresh, production-constructed,
never-solved TSO/DSO state (`p515_s32_zero_solve_checks._build_admm_ready_state`, BY IMPORT,
unchanged — the same pattern `p515_s41_hull_polish_checks.py` uses), sets **nonzero**
`interface_delta_p/q` at every (ADN node, year, day, period) (864 entries × 2 kinds = 1,728
checked), leaving the `flex_p/q_up/down` legs at production's own fixed-at-0 default.

**Note on "at the certified snapshots available without a new run":** no committed pickle carries
a *live* TSO model built on the Addendum-12 ADMM path with a materially nonzero
`interface_delta` in a still-solvable state (documented in the check's own output under
`note_on_preexisting_fixtures`) — the freshly-built state is the evidence base, exactly as the
spec's own parenthetical ("on a freshly built TSO block with nonzero interface_delta values")
permits.

**Results** (`data/SRP1/Results/P515S42/interface_helper_checks/results.json`):

| check | result |
|---|---|
| fixed helper equals the model's own `pc_adn`/`qc_adn`, every (node, year, day, period, p/q) | **exact** (max abs diff `0.0`), 1,728/1,728 — not merely round-off: the fixed helper literally returns the same Pyomo object |
| old helper's defect equals `interface_delta_p/q` exactly | **exact** (max abs diff `0.0`), 1,728/1,728 |
| free-variable content (representative entry) | fixed helper's expression depends on `interface_delta_p` **only** (`pc` and the legs fixed); the legacy helper's expression is a **constant** (zero free variables) |
| `apply_common_values`'s new row is feasible (required `interface_delta_p/q` value lies within its own bound) at synthetic certified-like values | **pass** — `required_delta_p=0.125` within `[-2.0, 2.0]`; the legacy construction (`pc == common_p`, `0.0 == 0.125`) would **not** have been feasible |

`solve_profile_guard`: `{'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}`,
`verify_failures: []`.

**Old helper's measured defect:** identically equal to `interface_delta_p`/`interface_delta_q` at
every one of the 1,728 checked entries — i.e. the defect is exactly what the analysis predicted,
not merely "close".

---

## Task 3 — exact-fix re-run harness

`p515_s42_exact_fix_rerun.py` (new file; `p515_s40_polish_gap.py` untouched, its committed v11
evidence stays VOID). Reproduces D bitwise in-process
(`p515_g_g1_g4_admm_gates.run_admm_arm(label='s39_D', apply_rho=False, num_max_iters_override=300)`,
`p515_s40_polish_gap._reproduction_check` reused BY IMPORT with the `rule_eleven_checklist`
subtree non-gating), then fixes coordination at the midpoint via `p56a_oracle.
common_coordinated_values`/`apply_common_values` — BY IMPORT, unchanged, so it automatically picks
up the Task 1 fix — with `p515_s40_polish_gap._switch_to_base_objective` (BY IMPORT, unchanged,
the objective-switch fix that stage already made) and production's own solver options (case-file
bound push — **not** overridden to IPOPT's default, since that is `p515_s41_hull_polish.py`'s own
v12-specific departure, not part of "the same exact-fix (midpoint) semantics as
`p515_s40_polish_gap.py`").

**The ESSO position-indexing fix is confirmed NOT relevant here:** `common_coordinated_values`/
`apply_common_values` never read `models['esso']` anywhere (grep-confirmed and documented in the
harness's own module docstring) — unlike `p515_s41_hull_polish.hull_entries_with_esso`, which adds
the ESSO as a third hull endpoint and therefore needed that fix.

**Per-block instrumentation, beyond `p515_s40_polish_gap.py`:**
- Final-attempt classification (`primary`/`recovery`/`recovery_tier2`) from `network.py`'s own
  always-printed control-flow substrings (`network.py:762-807`), captured per block via a
  per-block `redirect_stdout`, not a blanket one.
- IPOPT log-path reconstruction (`_expected_log_path`, read-only replica of
  `network.py:560-578`'s `output_file` → path formula; `_run_smopf`/`network.run_smopf` return no
  path themselves).
- Log-tail parsing (`_parse_ipopt_log_tail`): iterations, exit message, and the **unscaled**
  constraint violation — the second number on the `Constraint violation....:` line printed
  immediately after `Number of Iterations....: N` in the terminal summary block — the same
  convention `WORKER_REPORT_S41_POLISH_PREREQ.md` §1 already read by hand; validated
  programmatically in Task 3's checks against verbatim excerpts of that report's own log.
- **MVA conversion:** `MVA = unscaled_pu_value × network.baseMVA` (100 for case9/TSO — IPOPT's
  "unscaled" figure is already in the Pyomo model's own per-unit convention; no other scaling
  applies). Stated once in the harness (`MVA_CONVERSION_NOTE`) and reused in every report.
- Exact per-(ADN node, period) interface-diagnostics (`_tso_interface_diagnostics`): the required
  `interface_delta_p/q` value, its declared bound, and the signed excess — computed directly from
  the model's fixed `pc`/`qc` values and the `common` dict, not read off a generic constraint
  scan.
- `largest_violated_rows` (via `p56a_oracle.scan_constraints`) for any failed block.
- **Prediction scoring** against spec v13's `predictions_recorded_in_advance.exact_fix_rerun`
  (six spring/summer TSO blocks predicted locally infeasible, violation ≤ 5e-4 MVA; six
  autumn/winter TSO blocks predicted to solve; five DSO-2025 blocks recorded without prediction;
  the `all_12_tso_solve` branch re-examines the rating-midpoint check per spec if triggered).
- **Optional `--persist-certified-models`:** pickles `{'tso': models['tso'], 'dso': models['dso']}`
  before polishing, sha256- and size-recorded.

### Zero-solve checks for the harness

`p515_s42_exact_fix_rerun_checks.py` (`SolveProfileGuard(permitted=())` armed, `verify(0)`):
validates `_parse_ipopt_log_tail` against synthetic text built from the *exact* excerpts
`WORKER_REPORT_S41_POLISH_PREREQ.md` already read by hand (both a failing block's terminal summary
— iterations 50, violation `3.2299223958199708e-01`, exit "local infeasibility" — and a successful
one — iterations 42, violation `1.4589942587406313e-09`, exit "Optimal Solution Found."); the
final-attempt classifier against all four `network.py` control-flow shapes (no retry; tier-1
succeeded/failed; tier-1-then-tier-2 succeeded/failed); the log-path reconstruction and the
interface-diagnostics arithmetic against a freshly built production TSO/DSO block (hand-computed
cross-check, exact match); the prediction-scoring logic against two synthetic per-block tables
(one matching every prediction — 12/12 — one where all 12 TSO blocks solve, exercising the
`all_12_tso_solve` branch); and the persisted-model hash/size round trip. **All checks pass.**

### Smoke test (`--smoke-cycles 2 --persist-certified-models`)

Command run (attached, both streams captured, no `&`/`screen`/`nohup`; the shell tool moved the
process to background polling after its 120 s default timeout, but the process itself stayed
attached via the `>`/`2>&1` redirection the whole time — confirmed with `ps` mid-run):

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s42_exact_fix_rerun.py \
    --smoke-cycles 2 --persist-certified-models \
    > data/SRP1/Results/P515S42/exact_fix_rerun_smoke_launch.log 2>&1
```

Preconditions passed (no lock, no forbidden process, fresh output dir, production files clean in
git, D's committed reference report present). **Reproduction (against D's first 2 cycles): 0
diffs.** Certified models persisted before polishing: `165,541,141` bytes,
sha256 `e59d2c2678d8ea5cb645681562c5d614bc35a7a7f2e08cce5a688470cae8ab8e` (recorded in
`exact_fix_rerun_results.json`/`manifest_sha256.json`; **the 161 MB pickle itself is not
committed to git** — disposable 2-cycle smoke byproduct, not the full run's certified point; its
hash/size are the settling record, per CLAUDE.md's evidence rule, and it remains on local disk).

**Polish (48 blocks): all 12 TSO blocks solved; one DSO block failed** (`DSO5|2025|Spring`,
expected this early — far from a converged point). Per-block IPOPT diagnostics populated correctly
for every solved TSO block (23–28 iterations, "Optimal Solution Found.", violations
`1e-6`–`1e-9` MVA). **Prediction score at 2 cycles: 6/12 matched** — as documented, this is *not*
the prediction test (interface flows are near zero this early, nowhere near node 7's rating, so
none of the six spring/summer blocks is predicted-and-observed infeasible yet); the point of the
smoke test is mechanics, and every code path (reproduction gate, hook wiring, `apply_common_values`
with the fixed helper, per-block log parsing, prediction scoring, certified-model persistence) ran
without error. `main()` exited 0 (smoke mode does not treat an unmatched prediction score as a
failure).

---

## Task 4 — hull-bound active counts, excluding degenerate intervals

`p515_s42_hull_counts.py` (`SolveProfileGuard(permitted=())` armed, `verify(0)`): reads the
already-committed `data/SRP1/Results/P515S41/hull_polish/hull_bound_detail.json` (commit
`2e6c5570`, 8,640 per-descriptor entries, each carrying `lo`/`hi`/`degenerate`/`active`). The
committed `hull_polish_results.json`'s own `hull_bounds_active_by_channel` counts a degenerate
interval as active **by definition** (`p515_s41_hull_polish._hull_bounds_active`'s own documented
convention); this script filters degenerate entries out of both the numerator and the denominator.

**Result: zero degenerate intervals at D's certified point, on every channel.** The
excluding-degenerate counts are therefore identical to the committed including-degenerate ones:

| channel | active (excl. degenerate) | of non-degenerate | n degenerate |
|---|---:|---:|---:|
| V | 564 | 1,728 | 0 |
| PF_P | 458 | 1,728 | 0 |
| PF_Q | 197 | 1,728 | 0 |
| ESS_P | 27 | 1,728 | 0 |
| ESS_Q | 1,267 | 1,728 | 0 |

Cross-check against the committed `hull_polish_results.json`'s own top-level counts: **matches**
exactly (both the active and total counts, per channel).

---

## Commands / experiments run

1. `python p515_s42_interface_helper_checks.py` — Task 2 regression check. Exit 0, ALL PASS = True.
2. `python p515_s42_exact_fix_rerun_checks.py` — Task 3 harness zero-solve checks. Exit 0, ALL PASS = True.
3. `python p515_s42_exact_fix_rerun.py --smoke-cycles 2 --persist-certified-models` — Task 3 smoke
   test. Exit 0.
4. `python p515_s42_hull_counts.py` — Task 4. Exit 0.
5. Ad hoc (scratchpad, not committed): re-verified the 7 preserved fixtures still unpickle after
   the `p56a_oracle.py` edit.

## Results

Summarized inline above (Tasks 1–4). Headline: the fix is exact and its regression check is
exact (`0.0` max abs diff on both the fixed-helper-equals-model-expression check and the
old-helper-defect-equals-`interface_delta` check); the exact-fix rerun harness is built, its own
zero-solve checks pass, and its 2-cycle smoke test demonstrates every code path end to end without
error; the hull-bound counts, once degenerate intervals are excluded, are unchanged from the
committed figures because there were none to exclude at D's certified point.

## Validation

- Production diff is minimal and localized: `p56a_oracle.py` only — `_interface_expression`
  rewritten, the pre-fix form preserved unwired under an explicit legacy name, `apply_common_values`
  updated to pass `dn` instead of `adn_load`, `ORACLE_VERSION` bumped. No other production file
  touched.
- Every zero-solve check is backed by an armed `SolveProfileGuard`, `verify(0)` checked before any
  output is written, per CLAUDE.md's sixth evidence rule.
- The regression check's "exact" claims are numerically confirmed (`max_abs_diff: 0.0`), not
  asserted from the code's structure alone.
- The 7 preserved fixtures were re-verified to unpickle, not merely argued to be unaffected.
- The smoke test's reproduction check passed with 0 diffs against D's first 2 committed cycles,
  confirming the harness's in-process reproduction machinery works correctly before any polish
  solve runs.
- The hull-count re-derivation cross-checks exactly against the already-committed
  `hull_polish_results.json`, a second independent read of the same underlying per-descriptor
  data.

## Unexpected findings

- **Zero degenerate hull intervals at D's certified point, on every channel.** At C\*, no
  coordinated quantity's TSO/DSO(/ESSO) achieved values ever land on the same floating-point value
  — every one of the 8,640 descriptors is genuinely non-degenerate. This was not previously stated
  explicitly (the committed report only gave the degenerate-inclusive active counts); it means the
  Addendum-24 "excluding degenerate" instruction, while correctly executed, changes nothing
  numerically for this specific committed run — worth recording as a fact about D's certified
  point, not a defect in the original hull-polish reporting.
- The 161 MB size of the persisted certified-models pickle (TSO + DSO models, all 48 blocks) is
  larger than any previously committed pickle in this programme (the next largest,
  `esso_models_s39_D.pkl`, is 2.8 MB) — a practical consideration for the Planner if
  `--persist-certified-models` is used on the full run: the resulting pickle should probably be
  hash-recorded and kept off git (as done here for the smoke test), or moved outside the repo's
  git-tracked evidence tree entirely, depending on how the Planner wants to balance "no
  80-minute re-reproduction" against repository size.

## Remaining issues

- The full-length exact-fix re-run (48 real polish solves at D's actual certified point) has not
  been run — per instruction, the Planner launches it. The exact command (with and without
  `--persist-certified-models`) is stated in the harness's own module docstring and reproduced
  below.
- The smoke test's 6/12 prediction match is explicitly not informative about the real prediction
  (documented in both the harness's printed output and this report) — the full run is needed to
  score the actual predictions.

## Questions for Planner

1. Ready to authorize the full-length exact-fix re-run per the command below?
2. Persist the certified models (`--persist-certified-models`) on the full run? Given the 161 MB
   smoke-test pickle size, do you want it committed to git, hash-recorded only (as done here), or
   written outside the repo's tracked evidence tree?

---

## Exact full-run command (Planner launches; NOT run here)

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \
    p515_s42_exact_fix_rerun.py --persist-certified-models \
    > data/SRP1/Results/P515S42/exact_fix_rerun_launch.log 2>&1
```

(drop `--persist-certified-models` if not wanted; both variants write to the same fresh,
write-once `data/SRP1/Results/P515S42/exact_fix_rerun/` — run only one.) Preconditions identical
in kind to `p515_s40_polish_gap.py`/`p515_s41_hull_polish.py`; run attached, alone, both streams
captured; never `screen`/`nohup`/`&`. Expect ~139 ADMM cycles (matching D's certification) plus 48
polish solves; based on the 2-cycle smoke test's timing (~50 s/cycle including capture overhead)
and prior S40/S41 timings, order-of-magnitude 60–90 minutes, not separately benchmarked here.

---

## Evidence index

| item | artifact | commit |
|---|---|---|
| helper fix + regression-check script | `p56a_oracle.py`, `p515_s42_interface_helper_checks.py` | `2051309c` |
| regression-check evidence | `data/SRP1/Results/P515S42/interface_helper_checks/{results.json,manifest_sha256.json}` | `5c537a9c` |
| exact-fix rerun harness + its own checks | `p515_s42_exact_fix_rerun.py`, `p515_s42_exact_fix_rerun_checks.py`, `data/SRP1/Results/P515S42/exact_fix_rerun_checks/` | `7a9d6d7f` |
| smoke-test evidence | `data/SRP1/Results/P515S42/exact_fix_rerun_smoke/` (minus the 161 MB `certified_models.pkl`, hash-recorded not committed), `exact_fix_rerun_smoke_launch.log` | `52b31276` |
| hull-bound counts excluding degenerate intervals | `p515_s42_hull_counts.py`, `data/SRP1/Results/P515S42/hull_counts/` | `0abc08f2` |
| this report | `WORKER_REPORT_S42_HELPER.md` | (this commit) |

Zero-solve claims are backed by armed `SolveProfileGuard`s in every checks script
(`p515_s42_interface_helper_checks.py`, `p515_s42_exact_fix_rerun_checks.py`,
`p515_s42_hull_counts.py`), each with `verify(0)` checked before any output was written.
