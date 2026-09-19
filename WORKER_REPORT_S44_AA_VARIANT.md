# Worker Report — P5.15 Addendum 25 item 2: AA keep_memory variant, harness post-certification step, gate-ruling follow-ups

## Task received

The authority is Addendum 25 and spec v14 (`item2_aa_variant`, `item3_selection_run`). The "Deviation" and
"Follow-ups" sections of `P5_15_S44_GATE_RULING.md` (`8ce0b872`) are binding. The task had four parts:

1. Follow-ups:
   - (a) give `block_deltas` the `8682cfdd` total-order sort key;
   - (b) make the legacy one-run lock refuse while a campaign lock exists;
   - (c) turn the tie analysis's re-sort-and-straddle test into a reusable classifier.
2. Add the AA `reject_policy` sub-option. The default is the Step 3.7 behaviour; the other value is
   `keep_memory`. Deliver zero-solve checks and a two-cycle flag-off bitwise gate in a new directory.
3. Add an optional post-certification step to the campaign harness:
   - persisted models;
   - hull polish;
   - gates (b)/(c) against a named D reference;
   - the AA sidecar and peak RSS.

   The only overrides allowed are the AA flag and its sub-option.
4. Run a smoke campaign: two concurrent 3-cycle evaluations. Then state the next-campaign command without
   running it.

## Files inspected

- Authority: `PLANNER_BRIEF_2026-09-13.md` (Addenda 25/26, read only), spec v14, `P5_15_S44_GATE_RULING.md`.
- Code:
  - `admm_anderson_acceleration.py` and its wiring in `shared_resources_planning.py` (`_run_operational_planning`,
    `_anderson_acceleration_cycle_step`, `_update_admm_penalties`);
  - `admm_parameters.py`;
  - `p515_s41_aa_checks.py`, `p515_s43_aa_run.py`, `p515_s43_aa_flagoff_gate.py`;
  - `p515_s44_campaign_harness.py` and its checks, `p515_s44_gate.py`;
  - `p515_s41_hull_polish.py`, `p515_s42_exact_fix_rerun.py`, `p515_s42_hull_counts.py`;
  - `p515_s40_cost_decomposition.py`, `p515_s40_polish_gap.py`, `p515_s40_clone_capture_preflight.py`;
  - `p515_s44_gate_tie_analysis.py`, `p515_s44_alias_fix_check.py`;
  - `p515_g_g1_g4_admm_gates.py` (`s34_capture_hooks`, `run_admm_arm`, `run_s39_arm`,
    `_acquire_exclusive_run_lock`).
- Reports: `WORKER_REPORT_S44_HARNESS.md`.
- Committed evidence: `P515S43/aa_run*`, `P515S43/flagoff_gate`, `P515S41/aa_checks`, `P515S42/hull_counts`,
  `campaign_s44_gate`.

## Files modified

**Production.** Default behaviour is unchanged; each of these was verified.

- `admm_anderson_acceleration.py`: the sub-option.
- `shared_resources_planning.py`: one keyword argument, inside `if aa_enabled:`.
- `admm_parameters.py`: comment only.

**Harness and diagnostic files**
- `p515_g_g1_g4_admm_gates.py`: follow-ups (a) and (b).
- `p515_s44_campaign_harness.py`: follow-up (b) mirror, plus Task 3.
- `p515_s44_campaign_harness_checks.py`: C8–C12.
- `p515_s43_aa_run.py`: `_cost_decomposition_vs_d` gains an optional `reference_dir`. The default `None` is the
  old behaviour.

**New files**
- Code: `p515_s44_tie_classifier.py`, `p515_s44_followups_check.py`, `p515_s44_aa_variant_checks.py`,
  `p515_s44_aa_variant_flagoff_gate.py`, `p515_s44_aa_variant_campaign.py`.
- Evidence: under `data/SRP1/Results/P515S44/`.
- This report.

## Changes made

### Task 1 — follow-ups (`631d4183`)

**(a) `block_deltas` sort key.** `p515_g_g1_g4_admm_gates.py:2868` now sorts by
`(-abs_delta, str(agent), str(node_id), year, day)`.

**(b) Legacy lock.**
- `_acquire_exclusive_run_lock` (`:1396`) refuses while `.p515_s44_campaign.lock` exists (`CAMPAIGN_LOCK_PATH`,
  `:1393`).
- It checks the campaign lock again after its own O_EXCL create. If the campaign lock has appeared, it removes
  its own lock and refuses.
- The harness's `acquire_campaign_lock` does the mirror image.
- Optional path parameters exist only for the checks.
- Campaign children still never reference the legacy lock function.

**(c) Classifier.** `p515_s44_tie_classifier.py` provides `classify_top_k_list`, `classify_row` and
`reclassify_sidecar_diffs`, covering both top-k lists. `p515_s44_gate_tie_analysis.py` is not modified; it is
the settling artifact of a ruling.

### Task 2 — the variant (`42f48b92`)

**Semantics** (`admm_anderson_acceleration.py`):
- `REJECT_POLICIES = ('clear_memory', 'keep_memory')` and `DEFAULT_REJECT_POLICY = 'clear_memory'` (`:179-180`).
- The constructor takes `reject_policy`, validated (`:378-386`).
- `step()` appends `(w_k, g_k)` before deciding, unchanged (`:485`).
- On a safeguard rejection under `keep_memory` (`:519-530`):
  - the plain iterate `w_k + g_k` is returned;
  - `last_accepted_residual` (the mark) is unchanged;
  - no pair is removed. The memory is the post-append size, so this cycle's pair is the newest. At full memory
    the deque's `maxlen` drops the oldest pair, exactly as on an accept.
  - The record reads `action='rejected (safeguard; memory retained)'`, `reset=False`,
    `memory_size_after = m_k`.
- `clear_memory` keeps the old branch verbatim (`:532`).
- Memory is still cleared under both policies:
  - on a rho change: `clear_for_rho_change` (`:398`), called at `shared_resources_planning.py:2923`;
  - on a failure cycle: `skip_on_failure` (`:416`), called at `:2790`.

**Wiring.** `shared_resources_planning.py:2582` passes
`reject_policy=aa_settings.get('reject_policy', admm_anderson_acceleration.DEFAULT_REJECT_POLICY)`. The key is
not added to `ADMMParameters`' default dict, so every existing settings dict and committed run keeps meaning
Step 3.7.

### Task 3 — harness (`8f2b2736`)

**Evaluations and configuration**
- A spec entry is now one evaluation (candidate × configuration). An entry may carry its own `overrides`, which
  replace the campaign-level ones.
- `validate_overrides` accepts only `anderson_acceleration.{enabled: bool, reject_policy}`. After the override,
  the config hook re-checks that memory is 5 and regularization is 1e-10.
- `eval_key` is the candidate key under the case file, otherwise sha256{candidate_key, overrides}. It names the
  eval directory and working-dir ids, so C\* can run under D and AA in one campaign. The D directory name is
  unchanged: `578636daa6d6360d_<label>`.
- `evaluate` accepts labels. A candidate listed twice must be given by label.

**`post_certification` request**
- The request has three fields: `persist_certified_models`, `hull_polish`, and a `reference`.
- The reference is resolved at freeze time. It must be:
  - a certified evaluation;
  - of the same candidate key;
  - with no overrides;
  - with a component-levels gross cost equal to the record's certified cost.
- The sha256 of its record and of its component levels go into the spec. The child re-verifies both before the
  run and before use.

**`run_post_certification`** (`p515_s44_campaign_harness.py:1214`) runs in the child, inside `run_admm_arm`'s
post_run_hook, on the same live models and state.
- It first requires certification under the spec's own bar (`p515_s39_evaluate._certification_from_trajectory`).
  Otherwise it records `status: skipped` with the reason.
- **(b)** `|Q − Q_ref| ≤ 1.5e-4·Q_ref` (`S43.COST_RELATIVE_TOLERANCE`).
- **(c)** `S43._cost_decomposition_vs_d(..., reference_dir=...)`: residual ≤ 1.0, and every other priced
  component identically 0.
- **Persist**: `EF._persist_certified_models`, run before the polish.
- **(d) Hull polish**: `HP._polish_all_blocks_hull`, reused verbatim.
  - Pass requires every block solved and the gate on the sum of per-block changes.
  - Reported alongside: the settlement-excluded change, the settlement remainder before and after, and per
    channel the non-degenerate totals and active counts (`non_degenerate_hull_counts`).
  - `hull_bound_detail.json` is written.
- An exception in this step is recorded as `status: error` with its traceback. The child exits 2 after writing
  its record.

**AA-on evaluations** also write `aa_per_cycle.jsonl` (`S43._build_aa_per_cycle_sidecar`) and an action
summary, including rejections with memory retained.

**Peak RSS** keeps both measures. The child process RSS includes the step; the production state RSS covers the
ADMM run alone.

### Task 4 — campaign script (`92ddc722`)

`p515_s44_aa_variant_campaign.py`: the full campaign by default, and the smoke with `--smoke`.

## Commands / experiments run

All runs used the canonical interpreter, repo-root cwd, attached, with both streams captured. They ran one at a
time, with no lock present and no other `p515_s4*` process.

| # | command | output | exit |
|---|---|---|---|
| 1 | `python -u p515_s44_followups_check.py > .../followups_check_launch.log 2>&1` | `P515S44/followups_check/` | 0 |
| 2 | `python -u p515_s44_aa_variant_checks.py > .../aa_variant_checks_launch.log 2>&1` | `P515S44/aa_variant_checks/` | 0 |
| 3 | `python -u p515_s44_aa_variant_flagoff_gate.py > .../aa_variant_flagoff_gate_launch.log 2>&1` (HEAD `42f48b92`) | `P515S44/aa_variant_flagoff_gate/` | 0 |
| 4 | `python -u p515_s44_campaign_harness_checks.py r3 > .../harness_checks_r3_launch.log 2>&1` | `P515S44/harness_checks_r3/` | 0 |
| 5 | `python -u p515_s44_aa_variant_campaign.py --smoke > .../campaign_s44_aa_variant_smoke_launch.log 2>&1` (tool background runner, as for s44_gate; HEAD `92ddc722`) | `P515S44/campaign_s44_aa_variant_smoke/` | 0 |

Runs 1, 2 and 4 are zero-solve, with an armed `SolveProfileGuard(permitted=())` verified at 0. The parent of
run 5 is also guarded at 0.

I pre-exercised the check functions once against a scratch directory outside the repository before each
recorded run. No recorded output was overwritten or re-run.

## Results

### Follow-ups (F1–F3, all pass)

**F1: `block_deltas` sort key.**
- The real `s34_capture_hooks` ran on synthetic ties that straddle the top-10 cut, one child per
  `PYTHONHASHSEED` value (9 values).
- `block_deltas`: 1 order across the 9 seeds, equal to the name order.
- `objective_component_block_deltas`: 1 order.
- Sensitivity: the pre-fix key gives **9 distinct orders** over the same seeds.

**F2: locks.**
- The legacy lock refuses while a campaign lock exists, and leaves no lock behind.
- It acquires when no campaign lock exists, and stays exclusive.
- A simulated race (the campaign lock appears between the check and the create) backs out and refuses. The
  mirror race on the campaign side does the same.
- Both modules name the same campaign-lock path.
- By ast inspection, neither the harness nor `run_admm_arm` references `_acquire_exclusive_run_lock`.
- The real repository locks were untouched.

**F3: classifier.**
- It reproduces the committed tie analysis row for row: identical 50, resort 78, straddle 11.
- Recomputing the s44_gate's 43 genuine sidecar diffs with the gate's own comparator and classifier, all 43 are
  explained as tie order and 0 remain genuine.
- Synthetic cases: changed values, a changed prefix and changed non-list fields all come out UNEXPLAINED.

### Variant checks (V0–V3, all pass)

- **V0:** the committed Step 3.7 checks A, B and C pass. The outputs of B and C equal the committed
  `P515S41/aa_checks/aa_checks.json` exactly.
- **V1: default path bitwise.**
  - Setup: the `9a965494` module (loaded with `git show`) against the new module, both with the default
    constructor and with an explicit `'clear_memory'`. 24 randomized sequences of 200 cycles, 0 mismatches in
    iterates or records.
  - The sequences contain:

    | event | count |
    |---|---|
    | accepts | 1,947 |
    | rejections | 1,108 |
    | rho clears | 241 |
    | failure skips | 130 |
    | off cycles | 284 |
  - Sensitivity: `keep_memory` is detected as different in 24 of 24 sequences.
- **V2: `keep_memory` on an 8-D contraction.**
  - A rejection with 3 columns gives the plain iterate, the mark unchanged, memory 3→4, and this cycle's
    `(w, g)` as the newest pair (checked by object identity).
  - The next attempt is accepted with 5 columns.
  - At full memory, a rejection keeps 5 columns and the oldest pair drops, as on an accept.
  - A rho change still clears the memory, and the next step reports insufficient memory. A failure cycle still
    clears it (3→0).
  - Against `clear_memory` on the same inputs:
    - records are identical for cycles 1–5;
    - the cycle-6 rejection yields the same iterate;
    - that record differs only in `action`, `memory_size_after`, `reset` and `reset_reason`;
    - the iterates first differ at cycle 7, where keep is accepted and clear reports insufficient memory.
  - An unknown policy is refused.
- **V3:** there is exactly one construction site, with the exact `reject_policy=` expression, guarded by
  `aa_enabled`. There is no new default key.

### Flag-off two-cycle bitwise gate — PASS (`1fdf3142`)

- **AA calls:** 0 into any AA iterate function or method. The flag was read once (informational).
- **Against D's committed first 2 cycles:** 0 genuine diffs, 0 provenance diffs, 24 `aa_*` new-field diffs at
  the flag-off values.
- **Against `clone_preflight_v2/lightweight`:** 0 genuine diffs.
  - The report has 27 raw diffs: 24 `aa_*` diffs plus 3 non-gating provenance diffs.
  - The 4 terminal artifacts and 4 of the 5 sidecars have 0 raw diffs.
  - The recourse-jump sidecar has 2 raw diffs. Both are the known alias-pair labels at cycle 2,
    `objective_component_block_deltas[8..9].component`. The new classifier had nothing left to decide.
  - `block_deltas`: 0 diffs.
- **Other checks:** the `aa_*` fields are at their flag-off values, and the written report has no `peak_rss`
  key.

### Harness checks r3 — all 12 pass (0 solves)

- **C1–C7:** regression checks on the extended harness.
- **C8:** per-evaluation spec.
  - The D evaluation key equals the candidate key, and its directory name is as in s44_gate.
  - The reference is hash-recorded: certified cost 650,966,975.2943751 at cycle 139.
  - All 13 refusals raise.
- **C9:** the same candidate under two configurations spawned concurrently.
- **C10:** the real `run_post_certification`, with fakes only for the persist and polish calls.
  - Skip path on the committed 3-cycle AA smoke: no calls and no files.
  - Certified path, with the committed Step 3.7 AA run as "this evaluation" and the s44_gate D C\* evaluation
    as the reference:
    - the real decomposition reproduces the committed S43 gate (c) numbers exactly (reconciles);
    - gate (b) reproduces the committed abs_diff of 7,456.2266 and its tolerance of 97,645.05;
    - persist runs before polish;
    - the gate (d) summary reproduces the committed S43 relative of 0.000347 % with 48/48 blocks;
    - the non-degenerate counts reproduce the committed `P515S42/hull_counts` exactly.
- **C11:** the AA sidecar builder reproduces the committed `P515S43/aa_run/aa_per_cycle.jsonl` byte for byte.
  The summary gives 50 accepted, 22 rejected, first accept at 4, first reject at 14.
- **C12:** keep_memory override on a real, unsolved C\* planning.
  - Settings in force: `{enabled: True, memory: 5, regularization: 1e-10, reject_policy: 'keep_memory'}`.
  - The production construction expression gives `keep_memory`.
  - The AA layout builds.
  - A `memory` override is refused.
  - The post-certification capture paths pass.

### Smoke — two concurrent 3-cycle evaluations (spec `e3f83174…`, 155 s, both exit 0, empty stderr)

| | c_star_aa_keep_memory (`837fc982…`) | c_star_d (`578636da…`) |
|---|---|---|
| status | not_certified (3/3) | not_certified (3/3) |
| settings in force | AA `{True, 5, 1e-10, keep_memory}` | overrides `{}` |
| post-certification | **skipped**: "trajectory not certified under the spec's bar (cycles_run=3, cap=3, required_consecutive=10, terminal_consecutive_converged_cycles=0, stopped_by='cap')" | not requested |
| `post_certification.json` | written | — |
| persisted models / hull detail | none written | — |
| post-cert rule eleven before the run | all pass; reference hashes verified | — |
| AA sidecar | 3 rows, all `insufficient memory (m_k=0)` | none |
| solves | 205 | 205 |
| peak RSS (child / production / IPOPT) | 2.266 GB / 2.257 GB / 51.9 MB | 2.260 GB / 2.252 GB / 51.5 MB |
| first 3 rows vs reference | identical to the committed Step 3.7 AA run | 0 genuine diffs vs D's committed rows |

- The 205 solves equal the committed S43 3-cycle smoke. `identity_holds: False` there as well: one extra solve.
- All 10 smoke checks in the script pass. The spec sha256 is in both records. The parent guard is at 0. 2 slots
  ran concurrently.

**The Task 4 requirement "the AA sidecar must show accept/reject records and, on a rejection, a retained memory"
is NOT met at 3 cycles, and cannot be.**
- Rho changes at cycles 1 (`v`) and 2 (`pf`, `v`) clear the memory, so cycles 1–3 are
  `insufficient memory (m_k=0)`.
- This matches the committed Step 3.7 sidecar, which the smoke reproduces bitwise for these rows.
- On that trajectory the first accept is at cycle 4 and the first rejection at cycle 14.
- The two policies cannot differ before the first rejection. The earliest cycle at which a real run can show a
  retained memory on rejection is 14, and 15 for its use.
- The retained-memory semantics are therefore evidenced only by the zero-solve V2 checks, the C12 settings
  check, and the smoke's settings-in-force record.

## Validation

**What works**
- The code executes.
- Every check run passes:
  - follow-ups;
  - variant V0–V3;
  - flag-off gate;
  - harness r3 C1–C12;
  - the smoke's own 10 checks.

**What is and is not established**
- The default (`clear_memory`) path is bitwise identical to Step 3.7: synthetic sequences, the committed check
  outputs, and the flag-off two-cycle gate.
- The post-certification step's skip path ran for real.
- The certified path is verified zero-solve, with fakes only for the persist and polish calls. It has not run
  end to end on a live certified model inside the harness; the full campaign is its first real execution. Its
  components (`HP._polish_all_blocks_hull`, `EF._persist_certified_models`) are the ones that ran in S42 and S43.
- The variant's effect (cycles, gates (b)–(d)) is unmeasured; that is the Planner's run.

**Diff checks**
- `git diff` was reviewed before each commit.
- The index was empty before every staging.
- Files were staged by name with `git add -- …` and committed with `git commit --only`.
- `PLANNER_BRIEF`, `.env`, `.claude/`, `CLAUDE.local.md` and `STEP4_DFO_METHOD.md` were never staged.

## Unexpected findings

1. **The 3-cycle smoke cannot show accept/reject** (see the smoke section above). The Task 4 acceptance criterion
   conflicts with the 3-cycle length.
2. **"rho change (… exemption lift, freeze)".**
   - The committed detector clears the memory when a channel's rho value changes (`penalties_after[g] !=
     penalties_before[g]`, `shared_resources_planning.py:2921`).
   - A freeze, or an exemption lift, that does not itself change rho does not clear the memory. This is the
     Step 3.7 choice recorded in the module docstring.
   - The instruction said the clear-on-rho-change rule is unchanged ("still cleared"), so I kept it. Addendum 25's
     parenthetical could be read as asking for a clear on freeze and lift events as well.
3. **`p515_s41_hull_polish.FLAG_ABS` is a C\*-specific literal:** `1e-6 × 650,966,975.29`.
   - It sets `flagged_blocks` only; that output is reported and does not gate.
   - At 2×C\* it will still use C\*'s D cost.
   - It was reused unchanged, as required.
4. **Importing `p515_s43_aa_flagoff_gate` sets its module `OUT_DIR` suffix from the importer's argv.** This happens
   in the child and in my scripts. The harness never uses that constant. It is harmless, but a latent hazard.
5. **The flag-off gate's lightweight reference diff counts the report twice** (as the in-memory report and as
   `g_s39_D.json`). This is inherited from the S43 method and does not affect the verdict.

## Remaining issues

- **The certified path of the post-certification step has not run on a live certified evaluation through the
  harness.** Mitigation: it is the S43/S42 functions called in the same order, and C10 verifies the glue. If it
  errors, the record, the evaluation and the traceback survive, and the child exits 2.
- **Evidence left uncommitted:**
  - the Step 3.7-style `certified_models.pkl` (about 165 MB each) will be hash-recorded in the manifest in the
    full run, not committed;
  - the smoke root is committed in full (22 MB).
- **`PYTHONHASHSEED`** is still inherited, unset, in children. With (a) this matters only for comparisons against
  pre-fix sidecars, which the classifier now handles.

## Questions for Planner

1. Do you want a longer smoke to see a real retained-memory rejection before the full run? About 16 cycles,
   roughly 10 minutes.
   - Command:
     `python -u p515_s44_aa_variant_campaign.py --smoke --suffix c16 > data/SRP1/Results/P515S44/campaign_s44_aa_variant_smoke_c16_launch.log 2>&1`
   - **Caveat:** the cap is fixed at 3 in `_plan`. A 16-cycle smoke would need a one-line change adding a
     `--smoke-cap` argument, which I have not made.
   - Or do the zero-solve V2/C12 evidence and the settings-in-force record suffice?
2. Should a freeze or exemption lift that leaves rho unchanged also clear the memory? (Unexpected finding 2.)
   The current and committed behaviour is no.
3. The next campaign's result will report the first cycle at which the AA trajectory departs from the Step 3.7
   run. My expectation: cycle 14 differs in `aa_*` fields only, and numeric fields differ from cycle 15 or 16. Do
   you want this reported as a check, or left informational (as now)?

## Next-campaign command (NOT run)

Run from the repository root, attached, alone, with both streams captured:

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s44_aa_variant_campaign.py \
    > data/SRP1/Results/P515S44/campaign_s44_aa_variant_launch.log 2>&1
```

- **Campaign:** `s44_aa_variant`, root `data/SRP1/Results/P515S44/campaign_s44_aa_variant/`. Cap 500,
  concurrency 2, 10 consecutive cycles.
- **Evaluation (i):** `c_star_aa_keep_memory`, AA `{enabled: True, reject_policy: 'keep_memory'}`. Post-certification:
  - (b)/(c) against `data/SRP1/Results/P515S44/campaign_s44_gate/evals/578636daa6d6360d_c_star/`;
  - persisted models;
  - hull polish (d).
- **Evaluation (ii):** `two_c_star_d`, 1.9375 MVA / 7.75 MWh at nodes 5, 7, 9, under D. Post-certification:
  persisted models and hull polish.
- **Preconditions it enforces:**
  - no legacy or campaign lock;
  - no live `p515_g_g1_g4_admm_gates.py` or `p515_s4*` process;
  - a fresh root;
  - production, harness and case-file clean in git;
  - the references present.
- **Output:** `campaign_results.json`, which carries:
  - per-evaluation summaries;
  - the post-certification gates;
  - the AA summary;
  - the divergence from Step 3.7;
  - the spec v14 item-2 choice-rule inputs;
  - the item-3 stop-rule flag for 2×C\*.

  Plus `campaign_manifest_sha256.json`.

## Commits

- `631d4183`: Task 1 follow-ups, code and evidence.
- `42f48b92`: Task 2 variant code, checks and evidence, and the flag-off gate script.
- `1fdf3142`: Task 2 flag-off gate evidence (PASS).
- `8f2b2736`: Task 3 harness post-certification step, checks and r3 evidence.
- `92ddc722`: Task 4 campaign script.
- The commit carrying this report: the smoke evidence and this report.
