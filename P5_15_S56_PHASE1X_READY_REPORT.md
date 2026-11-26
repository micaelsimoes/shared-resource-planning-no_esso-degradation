# P5.15 S56 — EV 1× phase built, tested, frozen, preflight and resume check passed (Worker report, 2026-11-26)

Authority: `TASKS.md` step 5d, the author's decisions of 2026-11-26 on the 2× report (1× phase authorized, 60 h
ceiling, execution as at 2×, budget B = 1,000,000 € kept), and the Planner's 5d task with decisions D1 and D2.
Read: `CLAUDE.md`, `TASKS.md` 5a–5f, `P5_15_S56_PHASE2X_REPORT.md` (§7b, §8), the v2/v3 ready reports, the driver
section "THE 2x PHASE", `P.phase_step`, `p515_s56_phase_tests.py`, `s56_phase_check.py`, the harness's flex-price
handling.

**Solve claim.** Nothing was solved and no evaluation was launched. The tests, the gate, the freeze, the
preflight-only run and the resume check all ran under an armed `SolveProfileGuard(permitted=())`, verified at 0.
Pickle load/loads were blocked and counted 0 times. The 1× root holds only its spec and freeze logs.
`phase2x_ev_v1/`, `_v2/`, `_v3/` and `pilot_ev_v1/` were only read. `TASKS.md` is untouched.

**Objective convention.** F = I(x) + gross_operational_cost − TV credit, in € of total discounted planning cost
(oracle mapping v2, cost-breakdown schema v2; gross is settlement-excluded). dF_exit = F_exit(x) − F_exit(x0). No
system value is computed in this task. All test values are SYNTHETIC: they come from the landscape or from stub
children.

## Blocked on Planner

Nothing blocks launch 1. The spec is frozen, the preflight passes, the resume check holds 21/21, and the commands
are in §6. Five readings need your confirmation or a ruling. Changing any of them before launch 1 costs a re-freeze
into a new root `phase1x_ev_v2/` (zero solves, about 15 min). After launch 1 it needs a continuation.

1. **Exit-view check mode at 1×: I froze `stop`, not `record`.**
   - The R4 path sets mode `record` whenever `pilot_exit_views` is False. That rule was written for a continuation
     that had already found a mismatch.
   - Q7 as you stated it reads "as at 2× … automatic exit 5 on failure", and at 2× the mode was `stop`.
   - At 1× no exit view is served, so the check compares each candidate's exit run with the exit view of its own
     settled run, exactly as at 2×.
   - Rule now in the code: mode `stop` for any non-continuation phase; `record` only for a continuation that serves
     no exit view.
   - Option: `record`. This needs a re-freeze.
2. **Disk margin is thinner than at 2×.**
   - Free space now: **100.98 GiB**. At the 2× R5 reading it was 187 GiB.
   - The R5 figure (~300 evaluations × ~391 MB = 117 GB) is **not** covered.
   - I checked the 1× phase against what its own ceiling allows. The 2× eval dirs measure: exit max 235 MiB (mean
     176), settled max 490 MiB (mean 416). At the frozen block projections, 60 h admits 46 exit blocks
     (230 evaluations), or about **52.7 GiB** at the exit maximum. All settled, it is 19 blocks, 45.4 GiB.
   - The check passes on that bound. The bound is my reading, not a ruled rule. Freeing disk is the author's call.
3. **Two projections in the spec disagree; the restated one is the Planner's.**
   - `projected_hours` is computed as at 2×. Its assumptions treat x0, the unit and the staged plan as cached.
     It gives 32.8 h (exit-then-confirm, no move, q = 0) and **65.2 h settled-direct, no move, which is above the
     60 h ceiling**.
   - `phase.ceiling.projection_restated_1x` holds your restated 16.7–32.8 h / 7.8–22.0 h / +8–22 h per move.
   - The computed field is the v2 formula that over-projected 2× (65.2 h projected, 25.4 h measured). I kept it
     unchanged, so that the 2× spec builds identically.
   - If the settled-direct branch is taken and the search moves, the 60 h ceiling may stop the phase. In that case
     the outcome is exit 6 with an hours summary, as frozen.
4. **The existing test T12 fails one check, as it now does at `8fe07ca5`.**
   - The check is the REAL-artefacts collision scan for the v2 declaration. Its cause is v3's tracked launch
     artefacts (7 files: launch records 1–3 and their campaign specs), which carry 2× keys. They were committed by
     `690d4737` and `50d7dc67`, after tests v4 (69/69).
   - New check T17 runs T12's exact call under both drivers. The outcome is identical, so the failure is not caused
     by this change.
   - I did not re-scope or weaken T12. Re-scoping it, for example by declaring v3 or by scanning a fixed file list,
     is your call.
5. **Readings in the predictions file (`evaluated_as`).**
   - Q1: a screen row without a numeric dF_exit counts as "not dF_exit < 0". The number of such rows is reported
     beside it.
   - Q8: the mean is taken over the labels with a numeric dF_exit at both levels, and n is reported. The
     statement's 18 holds iff n = 18; all 18 are certified at 2×.
   - Q9: a settled summary without a numeric leak is listed. It is not counted as holding.

## Changed

| commit | content |
|---|---|
| `105ae315` | code, tests, predictions file, tests v5 and gate v5 evidence (before the freeze) |
| this commit | the frozen root, preflight, resume check, evidence manifest, this report |

The commit message of `105ae315` gives the per-test counts as T18 9/9 and T19 7/7. The correct counts are
**T18 10/10 and T19 6/6**. The total of 23 new checks is right.

**`p515_s56_search_driver.py`** (phase section, plus one docstring paragraph at the top):
- **Price level.** `phase_spec(..., flex_price=2.0)`, CLI `--phase-flex` (default 2.0).
  - Allowed values: {2.0, 1.0}. Anything else is refused: 1.5, 0, a bool, a string.
  - Default 2.0 builds the 2× spec field for field as before (T17).
- **1× (`flex_price` 1.0):**
  - arm `ev_1x`;
  - ceiling `PHASE_CEILING_1X_H` = 60 h (`--ceiling-h` stays continuation-only);
  - `PHASE_AUTHORITY_1X`;
  - predictions Q1–Q9 required (`PHASE_PREDICTION_IDS_1X`);
  - `phase.price_level` record (D1, D2, the check mode);
  - `phase.ceiling.projection_restated_1x`.
- **Harness configuration.** `phase_harness_config(data_root, flex)` keeps the pilot x0_i1 check unchanged. It then
  sets `flex_price_multiplier` 1.0 at both depths and nothing else. The source carries a `price_level` note.
- **No pilot record.**
  - A non-continuation 1× spec must be frozen with `--no-pilot-exit-views`; 1× with exit views on is refused.
  - **No cache source** is declared (choice below).
  - The synthetic/stub test path precaches nothing at 1×.
- **Capture checklist.** `phase_capture_checklist(H, price_level_1x)` adds 5 items at 1× (17 in all):
  - leak priced as conversion loss in every settled summary;
  - Q9 inputs;
  - the ESSO recovery reconciliation (`4030508f`) in the guards;
  - the price level applied, read back and recorded in the child;
  - screen dF_exit per row.
- **Prediction inputs.** `phase_prediction_inputs` adds Q8 (screen dF_exit at 1×) and Q9 (leak over every settled
  summary) for a 1× spec only.
- **Continuations.** `continuation_source` refuses a continuation whose predecessor runs at another price level.
  Before this change, a 1× continuation of v2 would have indexed nothing and passed silently.
- **Ceiling-stop note.** It now names the spec's ceiling, not "120 h". This is run output; no spec field changes.
- `validate_phase_predictions(pred, ids)`; helpers `phase_flex_level`, `phase_flex_of`.

**`p515_s56_phase_tests.py`:**
- `freeze(..., extra=())` and `dummy_predictions(ids)` were extended. Defaults are unchanged.
- T17, T18 and T19 were added (Found §1).

**New files:**
- `p515_s56_phase1x_resume_check.py`: zero-solve resume check of the frozen root;
- `data/SRP1/Results/P515S56/phase1x_ev_v1_predictions.json` (sha256 `de8313a4…`);
- `phase1x_ev_v1/` (spec + freeze logs);
- `phase1x_ev_v1_preflight/` + sibling logs;
- `resume_check_s56_1x_v1/` + sibling logs;
- `phase1x_tests_v2/` + logs;
- `search_driver_v5/` + logs;
- `phase1x_ready_evidence_manifest_sha256.json`.

**Spec.** `data/SRP1/Results/P515S56/phase1x_ev_v1/phase_spec_feb487b7.json`
- sha256 `feb487b788c9a769113a1e24e32c4db8e8c296375553168aa12bcee118d8c0ac`;
- phase id `s56phase1x_ev_v1`, git_head `105ae315`;
- configuration = v3 `phase_spec_600bbceb` with `flex_price_multiplier` 1.0 at both depths;
- R 10,000 €, e_EV 17,494.05 €, K_EV 300, caps 300/300, block 5, cap 80, concurrency 5, 3.29 GiB per slot;
- efficiency rule P1, mapping v2 parameters;
- no cache source, `pilot_exit_views` false, exit-view check `stop`;
- ceiling 60 h, carried 0;
- predictions Q1–Q9 embedded (source sha256 `de8313a4…`).

**Planner decisions as implemented:**
- **D1:** e_EV and K_EV are read from the pilot as at 2×. The `e_EV != 17,494` and `K_EV != 300` refusals are
  unchanged. There is no 1× pilot. The offset review stop and Q7 (mode `stop`) guard the carry-over.
- **D2:** no record carries over. The existing R4 path (`pilot_exit_views=False`) is used, with no second path. The
  policy then requests:
  - x0's exit run first, alone;
  - x0's settled run in the candidate confirmation;
  - the unit's and the staged plan's settled runs in the candidate confirmation or the pool.

  This is shown on the frozen configuration in the resume check, part D. The change beyond the R4 path is a few
  lines: allow `pilot_exit_views=False` without a continuation at 1×, and skip the pilot cache index.
- **Cache-source choice: the pilot root is dropped, not declared.** It takes the same amount of code either way: the
  pilot-index block is skipped. Undeclared, the pilot's tracked files stay **inside** the collision scan, which
  finds none of the 1× keys there. A declared root would have been excluded from it.

## Found

### 1. Tests: v5 run `phase1x_tests_v2` 91/92; gate v5 `search_driver_v5` 167/167

Tests: `python -u p515_s56_phase_tests.py --out data/SRP1/Results/P515S56/phase1x_tests_v2`, 1,303 s, stderr empty.
- **Existing tests: 68/69 against 69/69 at v4.** The only failure is T12's real-artefacts check (Blocked 4).
  T1–T11 and T13–T16 all hold. That includes T13: continuations on stubs still work with the price-level refusal
  added.
- **T17, the 2× option-off identity: 7/7.** The driver at `8fe07ca5` was loaded beside the working tree. The
  following built by both are equal in every field except `frozen_utc`, `code_sha256` and `git_head`:
  - the real 2× spec as v2 (non-continuation);
  - the real 2× spec as v3 (continuation of v2 at `18a224d1`, v1 `de4e947e` un-launched, the real v2 predictions
    file);
  - a stub spec;
  - a synthetic spec.

  The freeze collision-scan outcome is equal under both drivers, and so is T12's exact call. The comparison is not
  vacuous: the two driver files differ. As information only: the v3-as-built spec equals the **committed**
  `phase_spec_600bbceb.json` in every field except those three and `collision_scan`, with 0 differing paths.
- **T18, the real 1× spec: 10/10.**
  - Fields as frozen.
  - The flat path diff against the 2× spec lies only inside the declared fields. R, e_EV, K_EV, caps, efficiency,
    screen, ladder, mapping and `projected_hours` are identical.
  - Harness configuration: equal to 2× except `flex_price_multiplier` at both depths.
  - Capture checklist: all 12 2× items equal, plus 5 true.
  - Cache-root audit: ok with 0 files.
  - Eval keys: every one of the **74,084** is composed with 1.0 and equals the multiplier-absent key. That is the
    harness rule: 1.0 is today's price and is not keyed. **0 of them are shared with the 74,084 2× keys.**
  - `H.freeze_campaign_spec` freezes the real 1× configuration, x0 and the unit at both depths, in a scratch dir
    outside the repository that is removed afterwards. **The harness does not refuse 1.0 on S56 EV entries.** Keys
    recompute; entries carry `flex_price_multiplier` 1.0 and no flex label.
  - Collisions: **0** over the 361 tracked artefacts, and **0** over the 108 spec / launch-record /
    evaluation-record files on disk under `phase2x_ev_v1–v3` and `pilot_ev_v1`.
  - Refused:
    - `--ceiling-h` 80 and 60 without a continuation;
    - price levels 1.5, 0, bool, string;
    - 1× serving pilot exit views;
    - **a 1× continuation of the 2× root v2, refused with the price-level reason**;
    - predictions without Q8/Q9;
    - unchanged: the 2× non-continuation `--no-pilot-exit-views`;
    - the CLI `--phase-flex 1.5`: exit 2, nothing written.
- **T19, the 1× flow: 6/6.**
  - Synthetic 1× phase through the CLI: exit 0, state byte-equal to the in-process phase.
  - Launch 1 is x0's exit run alone; launch 2 is the first screen block of 5.
  - x0 runs at both depths; the unit and the staged plan run settled; no record is served from a cache.
  - Stub-children 1× phase: exit 0, byte-equal, collision scan clean, no cache source.
  - The test ceiling defaults to 60.
  - The real 1× commands carry no hazard string.
- Guards: verify(0) == []; pickle 0.
- A first run, `phase1x_tests_v1` (90/91, the same T12 failure), predates the added T17 check. It is left
  **uncommitted** and is superseded by v2.

Gate: `python -u p515_s56_search_gate.py --out data/SRP1/Results/P515S56/search_driver_v5`, 1,253 s, stderr empty.
**167/167**, unchanged from v4.

### 2. Freeze (zero solves), from `105ae315` with clean code files

```bash
mkdir data/SRP1/Results/P515S56/phase1x_ev_v1 && NLP_SOLVER_PATH=/usr/local/bin/ipopt /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_search_driver.py --phase-freeze --root data/SRP1/Results/P515S56/phase1x_ev_v1 --phase-id s56phase1x_ev_v1 --oracle harness --predictions data/SRP1/Results/P515S56/phase1x_ev_v1_predictions.json --phase-flex 1.0 --no-pilot-exit-views > data/SRP1/Results/P515S56/phase1x_ev_v1/freeze_stdout.log 2> data/SRP1/Results/P515S56/phase1x_ev_v1/freeze_stderr.log
```

- Exit 0; stderr empty.
- Collision scan: 74,084 keys against 361 tracked artefacts gives **0 collisions**. Excluded: 0 cache-root files,
  0 own-root files.
- Cache-root key audit: ok, 0 files.
- Domain keys distinct: true.
- The 2× report §8 simulated the 1× freeze at 0 collisions over 360 files. The real freeze scanned 361 and found 0.

### 3. Preflight-only (zero solves; `NLP_SOLVER_PATH=/usr/local/bin/ipopt`) — PASS

Run: `--phase-run --root …/phase1x_ev_v1 --preflight-only --out …/phase1x_ev_v1_preflight`. Exit 0. Record
`phase_preflight.json` `89aa98dd…`; logs `phase1x_ev_v1_preflight_stdout.log` / `_stderr.log` (empty).
- **Next action: launch 1, stage screen, purpose `x0_exit`, exit depth, 1 plan (x0).**
- Cache as loaded: empty.
- Ceiling: 0 + 1.29 h ≤ 60 h.
- Harness preconditions `[]`; refusals `[]`.
- Memory for this block (1 slot): 19.81 GiB available ≥ 3.29.
- Lock acquired as `s56phase1x_ev_v1_l001` and released; no launch dir; nothing written under the root.

### 4. Resume check (zero solves) — 21/21

`p515_s56_phase1x_resume_check.py --root …/phase1x_ev_v1 --out …/resume_check_s56_1x_v1`. Exit 0. Record
`f74d41b8…`; stderr empty.
- **A, the spec:** hash-named; arm ev_1x; 1.0 at both depths with D1/D2; 60 h, carried 0; R, e_EV, K_EV, caps,
  block, cap, concurrency and memory as above; no cache source, no exit view, mode stop; Q1–Q9 with the source
  sha; 17 checklist items true; 0 collisions; audit ok; **no code drift**; no predecessor.
- **B, the root:** only the spec and freeze logs, no launch; unchanged by the check.
- **C, the resume point:**
  - the driver's cache is **empty**;
  - the replay of the empty chain gives the initial state;
  - **first action = x0's exit run alone**, eval key `9a4396f604a8a3af…`, label `x0__exit__ev_1x`;
  - ceiling passes;
  - **memory for a full block: 19.82 GiB available ≥ 5 × 3.29 = 16.45 GiB**, read with the Claude app open;
  - disk passes on the ceiling bound (Blocked 2).
- **D, the order** (synthetic values, frozen configuration):
  - x0 exit, then the 18 screen plans in spec order in blocks of ≤ 5 (the unit's row among them, as an exit run);
  - then the ladder, candidate confirmation (x0 settled + the candidates), candidate polls, pool and search;
  - x0, the unit and the staged plan each run settled by the phase;
  - no (plan, depth) is requested twice.

### 5. Recorded predictions (embedded in the spec, sha256 `de8313a4…`)

Statements are the Planner's, verbatim. `evaluated_as` names the phase_results fields.

| id | statement (short) | stops? |
|---|---|---|
| Q1 | ≤ 9 of 18 screen plans dF_exit < 0 vs 1× x0 | no (recorded; 2× R1 reasoning) |
| Q2 | 0 harness_error, 0 local-solve failures | as at 2× |
| Q3 | best start is x0 | no; the search starts from the rule's start |
| Q4 | exit 0 inside 60 h | exit 6 → author; exit 5 → Planner |
| Q5 | peak child RSS ≤ 3.29 GiB | as at 2× |
| Q6 | efficiency fraction recorded | — |
| Q7 | exit run == exit view of the candidate's settled summary, bitwise | **automatic exit 5** (mode stop) |
| Q8 | mean[dF_exit(1×) − dF_exit(2×)] over 18 screen plans > 0 | no; computed after the phase at zero solves |
| Q9 | loss-priced leak ≤ 25 € on every settled evaluation | no; named in report (ii) for the Expert |

- **Q8 reference:** `phase2x_ev_v3/phase_results.json`, sha256 `2f8778b5ed573492…`. Its 18 screen dF_exit values
  are copied into the predictions file (`reference_2x`).
- **Projection (restated, not a prediction):**
  - 16.7–32.8 h with two candidates and no search move;
  - 7.8–22.0 h if no screen row is a confirm-candidate;
  - each move + 8–22 h.

### 6. The commands (author, Terminal, Claude app closed)

None of the lines contains the harness hazard string (T19).

**(a) Launch 1 only (the first-block hold).** It refuses to run if `launch_001` logs exist. It runs x0's exit run
alone: one evaluation, about 1.0–1.3 h. Expected output: `<UTC> launch 001 exit 4`.

```bash
cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && caffeinate -i zsh -c 'R=data/SRP1/Results/P515S56/phase1x_ev_v1; n=001; if [ -e "$R/launch_${n}_stdout.log" ]; then echo "launch_${n} logs exist: not re-run"; else NLP_SOLVER_PATH=/usr/local/bin/ipopt /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_search_driver.py --phase-run --root "$R" --take-over-stale-lock > "$R/launch_${n}_stdout.log" 2> "$R/launch_${n}_stderr.log"; rc=$?; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launch $n exit $rc"; fi'
```

**The check of launch 1** (read-only; 0 all PASS, 1 any FAIL, 2 record missing):

```bash
cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python s56_phase_check.py data/SRP1/Results/P515S56/phase1x_ev_v1 --launch 1
```

**(b) The loop, after the Planner has checked `launch_001.json`.** It continues the numbering at 002 and re-issues
the launch while the exit code is 4. It stops at 0 (complete), 5 (review, to the Planner), 6 (ceiling, hours summary
to the author) or any other code (fault):

```bash
cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && caffeinate -i zsh -c 'R=data/SRP1/Results/P515S56/phase1x_ev_v1; i=1; while [ -e "$R/launch_$(printf %03d $i)_stdout.log" ]; do i=$((i+1)); done; while true; do n=$(printf %03d $i); NLP_SOLVER_PATH=/usr/local/bin/ipopt /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_search_driver.py --phase-run --root "$R" --take-over-stale-lock > "$R/launch_${n}_stdout.log" 2> "$R/launch_${n}_stderr.log"; rc=$?; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launch $n exit $rc"; if [ $rc -ne 4 ]; then echo "loop stopped: exit $rc (0 complete, 5 review, 6 ceiling, other fault)"; break; fi; i=$((i+1)); done'
```

## Not confirmed

- **No 1× evaluation exists.** Walls, RSS, the efficiency branch and the leak at 1× are unmeasured. The child's
  application and read-back of the 1.0 multiplier on the real (non-stub) path was not exercised here: only the
  harness freeze accepted it, in T18.
- **Collision scope.** Searched: the tracked artefacts in the driver's scope (`git ls-files data/SRP1/Results`:
  campaign/search/phase specs, launch records; 361 files), and every spec / launch record / evaluation record on
  disk under `phase2x_ev_v1–v3` and `pilot_ev_v1` (108 files). Not searched: untracked artefacts elsewhere (other
  result dirs, worktrees, `data/SRP1/Results/P56A`).
- **360 vs 361 files.** The 2× report's simulated 1× scan counted 360 files; the real freeze scanned 361. I did not
  identify the extra file. It is a tracked file added since that simulation; the result is 0 collisions either way.
- **The disk bound** uses the 2× eval-dir sizes and the frozen block projections. If measured 1× walls are shorter
  than the projections, more blocks fit in 60 h and the bound is not a hard upper limit.
- **Memory** was read with the Claude app open (19.82 GiB available). The author runs with it closed.
