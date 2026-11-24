# P5.15 S56 — EV 2× phase spec v2: safety fixes built, tested, frozen, preflight passed (Worker report, 2026-11-24)

Authority: `TASKS.md` step 5a: "Advisor pre-launch review: GO-WITH-CHANGES" and Planner decisions R1, R4, R5, P1,
P2 (2026-11-24); the Planner's spec-v2 task of 2026-11-24. Read: `P5_15_S56_PHASE2X_READY_REPORT.md` (v1), the
driver section "THE 2x PHASE", `phase_step` in the policy, `p515_s56_phase_tests.py`.

**Solve claim.** Nothing was solved and nothing was launched. Every test, the gate, the freeze and the preflight
ran under an armed `SolveProfileGuard(permitted=())`, verified at 0; pickle load/loads were blocked and counted
0 times. The v2 root holds only its spec and freeze logs. The v1 root is untouched: 3 files, spec sha256
`de4e947e…`, no new file. `TASKS.md` is untouched.

**Objective convention (every € figure):** F = I(x) + gross_operational_cost − TV credit, in € of total
discounted planning cost (cost-breakdown schema v2; gross is settlement-excluded). dF_exit = F_exit(x) −
F_exit(x0). All test values are SYNTHETIC: they come from the landscape or from stub children.

## Blocked on Planner

Nothing blocks launch 1. The spec is frozen, the preflight passes, and the three commands are in Found §5.

1. **Decision: the efficiency threshold vs. the break-even of exit-then-confirm (change 9).** This decision is not
   needed before launch 1. It is needed before the efficiency rule applies, after the screen and ladder (about
   launch 6).
   - **Why the cost changes.** Under exit-then-confirm, every confirmation is one serial launch, projected at
     3.094 h. The final complete poll confirms every confirm-candidate in N(final incumbent). For the pilot
     staged plan, N(final incumbent) has 46 uncached neighbours.
   - **The confirm-candidate band is wide.** With e = e_EV = 17,494 € > R, the band is dF_exit ≤ −R + e =
     **+7,494 €**. A neighbour up to 7.5 k€ *worse* at exit is still confirmed.
   - **Break-even.** One complete poll (10 blocks) costs the same in both branches at **5.8 confirmations**.
     Above that, settled-direct is cheaper.
   - **The P1 threshold allows far more.** P1 selects exit-then-confirm whenever ≤ 1/2 of the screen + ladder
     rows are not worse against x0. That allows up to 23 confirmations in the final poll: 104.0 h with no move,
     against 65.2 h for settled-direct.
   - **A hypothesis, not measured.** The screen measures large moves away from x0. The final poll measures
     one-granule moves around the best plan, where |dF_exit| is likely small. The final poll's share q could
     therefore exceed the screen's. At q = 1 (46 confirmations), no-move projects to **175.1 h > 120 h**.
   - Options:
     (a) keep the rule as frozen in v2 (P1 as given);
     (b) freeze a v3 before the efficiency rule applies, with a threshold tied to the break-even. That changes an
     approved rule, so it is the Planner's or author's call;
     (c) keep v2 and rely on the ceiling, which suspends the phase at exit 6 with an hours summary.

   I have no basis to recommend between them: the confirm-candidate share of a final poll has not been measured.

2. **Worker readings, logged.** Each is frozen in spec `18a224d1`; changing one needs a v3 (zero solves, about
   15 min) before launch 1.
   - (a) **Efficiency denominator:** every screen and ladder row, uncertified ones included. Uncertified rows are
     neither worse nor not-worse (as in v1). A harness_error row cannot occur here: the phase stops first.
   - (b) **Exit-view check order:**
     - a harness_error stop on the first candidate in order comes first;
     - then exit_view_mismatch, over the candidates whose confirmation is not a harness error;
     - then the offset stop.
     A settled summary without an `exit_view` counts as a mismatch. The check covers the candidates only, as
     specified, not the search's confirmations.
   - (c) **Un-launched predecessor:**
     - Allowed contents are exactly one hash-named phase spec whose sha256 starts with the declared prefix,
       `freeze_stdout.log` and `freeze_stderr.log`. No directory of any kind; `.DS_Store` is ignored.
     - Verified at the freeze, re-verified by the collision scan, and re-verified at **every** `--phase-run`.
   - (d) **Continuation (v3):**
     - Allowed only if the predecessor stopped at exit 5 (results with a review termination) or exit 6 (a ceiling
       stop record after its last complete launch). A completed phase is refused.
     - Only records from complete, chain-verified launches are indexed; records moved aside after an
       interruption are never indexed.
     - The predecessor's measured hours are **carried into the continuation's ceiling**, so a raised ceiling is
       a total.
     - A raised ceiling is allowed only in a continuation, and only with `--ceiling-note`, which records the
       author's decision.
     - `--no-pilot-exit-views` (the R4 fallback) sets the exit-view check to `record`: it is recorded and does
       not stop the phase.
     - A continuation re-derives every decision from cached records, so **an exit-5 stop recurs unless the v3
       changes its cause**. The exit-view switch is the only cause-change built in. Resuming after harness_error,
       offset or not_settled needs a Planner ruling, and any code change behind it forces a new spec through the
       drift refusal.
   - (e) **Code-drift refusal:** runs before the lock, at every `--phase-run`, `--preflight-only` included.
   - (f) **Check script wall criterion:** the per-evaluation child wall (`summary.wall_s`). The launch wall is
     printed beside it.
   - (g) **Schemas:** the phase spec and phase state schemas are bumped to v2. `--phase-run` on v1's root now
     refuses on the schema, as well as on drift.

## Changed

| commit | content |
|---|---|
| `5c023309` | Build. Policy: efficiency measure P1 with per-class counts and the v1 band fraction as information; `exit_view_check` and the stop `exit_view_mismatch` (R4); state schema v2. Driver: spec schema v2; `--unlaunched-predecessor`, `--continuation-of`, `--ceiling-h` / `--ceiling-note`, `--no-pilot-exit-views`; code-drift refusal; `.DS_Store` ignored; Q4 projection with confirmations; Q7 inputs; `phase_single_launch_command`, `phase_check_command`. New `s56_phase_check.py`. Tests T10–T16 added; T1, T2 and T5 updated |
| `1e6ab91c` | `phase2x_ev_v2/phase_spec_18a224d1.json` with its freeze logs; `phase2x_ev_v2_predictions.json` |
| `7350e5da` | Test fix: T12's real-artefact scan uses the real v2 root as its own root (Found §1) |
| (this commit) | Test evidence v3 (68/69, failed as run) and v4 (69/69), gate v4 evidence, preflight record, hash inventory `phase2x_v2_ready_evidence_manifest_sha256.json` (`6f04b349e2d1…`, 28 files), this report |

**Spec v2** `phase_spec_18a224d1.json` (sha256 `18a224d13f9bc04e3c3177fb678ced009f5f845b77ab895817228b2ad28c1ce8`)
- Root `data/SRP1/Results/P515S56/phase2x_ev_v2`, phase id `s56phase2x_ev_v2`, frozen from `5c023309`.
- Predecessor: v1 `de4e947e…`, declared un-launched; recorded files: spec + 2 freeze logs.
- Code hashes: driver `18eca572…`, policy `27d93478…`, harness `05a9838c…` (unchanged from v1).
- **Unchanged from v1:** R = 10,000 €; e_EV = 17,494.05 €; K_EV = 300; exit cap 300; block 5; cap 80; ceiling
  120 h; 5 slots × 3.29 GiB; candidate, ladder, acceptance and pool rules; cache = the pilot's x0_i1, unit_i1 and
  staged_i1 at the settled key, exit views served.
- **New:**
  - `policy.exit_view_check = stop`;
  - `phase.efficiency.measure = not_worse_against_x0`;
  - `phase.pilot_exit_views = true`;
  - `phase.ceiling.carried_h = 0`;
  - `predecessors`;
  - the projection with confirmations;
  - the capture-path checklist, now 12 items, all true: the per-class counts, `exit_view` in the settled summary,
    and the per-candidate exit-view check were added.

**The nine changes, as built**
1. **P1: the efficiency rule.** fraction = #(rows whose class against x0 is confirm_candidate or unresolved) /
   #(screen + ladder rows). If it is > 1/2 (exact rational), the branch is settled-direct. The record holds
   `class_counts`, `n_not_worse`, the fraction, the branch, and `band_measure_information_only`: the v1 ±e_EV
   fraction and the branch it would have given. No decision reads that last item.
2. **R1: Q1.** Re-declared as recorded and non-stopping. The prediction text states that no phase decision reads
   its sign and that its threshold (dF_exit < 0) lies inside the e_EV band (Found §6).
3. **R4: held-out exit-view check, Q7.** At candidate confirmation, the exit run's (read_cycle, F, bar) is
   compared with the settled summary's `exit_view`. The comparison is bitwise: `float.hex` for floats, int by
   value, with the type. A mismatch stops the phase with exit 5, reason `exit_view_mismatch`, naming the
   candidates and fields. The fallback is in TASKS: a continuation with `--no-pilot-exit-views`.
4. **P2: collision scan.** It excludes a declared un-launched predecessor root (reading c) and nothing else.
5. **P2: continuation.** A v3 can declare a stopped predecessor with launches as a cache root (reading d). It is
   built and tested on stubs. No v3 is frozen.
6. **Code drift.** Every `--phase-run` recomputes the spec's `code_sha256` set. If any hash differs it refuses
   with exit 2 and logs `code drift: <file> sha256 now … frozen …`; nothing is written and no lock is taken.
7. **`.DS_Store`.** Files with exactly that name are ignored by `files_under` (manifests, replay, interruption
   records), the foreign-file check, the freeze's root check, the killed-evaluation listing and the predecessor
   verification.
8. **First-block hold.** The single-invocation command, `s56_phase_check.py`, and the loop that continues at 002
   (Found §5).
9. **Q4 basis.** Confirmations under exit-then-confirm are now counted: one launch each, including the final
   poll's (Found §4).

## Found

### 1. Tests (zero solves): v4 69/69

`p515_s56_phase_tests.py --out data/SRP1/Results/P515S56/phase2x_tests_v4` was run from `7350e5da`, with the code
files clean, attached and with both streams captured.
- **69/69**, 881 s, stderr empty, guard verify(0) `[]`, pickle 0.
- `phase_tests.json` `c8b6e2796def…`; `manifest_sha256.json` `f47c63d48386…` covers 8,867 files: the test trees
  are hash-recorded, not committed.
- `phase2x_tests_v4_stdout.log` `71afc2484e63…`.

**v3 is kept as run: 68/69, from `1e6ab91c`.** One check failed: T12, the real-artefact scan. Cause:
- after `1e6ab91c`, the v2 spec is itself a tracked artefact carrying the pilot keys;
- the test passed a scratch path as the scan's own root, so v2's own spec counted as foreign. The refusal named
  `phase2x_ev_v2/phase_spec_18a224d1.json`.

This is the CLAUDE.md W105 case (a pre-run scan must exclude the run's own root). The scan logic was right; the
test was wrong, and it had passed in scratch before the freeze. `7350e5da` passes the real v2 root, read only, and
also asserts that the undeclared refusal names v1's spec and nothing else. v3 record: `phase_tests.json`
`5cd29301b564…`, stdout `3a6ea47dd4db…`.

**v1/v2 tests re-run (T1–T9):** all pass. T1 still pins the default search path to `6b1d98b2`: byte-identical on
6 searches, and the plain landscape on 382 plans × 2 depths. T2 and T3 keep their branches under the new measure:
- T2: 10 of 22 rows not worse, so exit-then-confirm;
- T3: 20 of 22, so settled-direct.

| test | change | what | result |
|---|---|---|---|
| T10 | 1 (P1) | numerator = confirm_candidate + unresolved; uncertified in the denominator only; 11/22 gives exit-then-confirm, 12/22 settled-direct; per-class counts; **the measures disagree both ways**: 14/22 not worse with 0 in band gives settled-direct (v1 would give exit-then-confirm), and 3/22 with 12 in band gives exit-then-confirm (v1 would give settled-direct); an unknown class is refused; a row at dF = −20,000 € is a confirm-candidate outside the band | 5/5 |
| T11 | 3 (R4) | bitwise: one ulp in F, one ulp in bar, read_cycle + 1, read_cycle as a float, an absent view — each a mismatch naming the field. In the phase: mismatch on either candidate gives `exit_view_mismatch` at candidate_confirmation; harness_error precedes it; it precedes offset; offset alone still stops; mode `record` completes with the mismatch recorded; an unknown mode is refused. T5 adds the CLI case: exit 5, results written, byte-equal to the in-process run. T2: both candidates equal, mode stop, Q7 n_equal 2 | 3/3 (+1 in T5, +1 in T2) |
| T12 | 4 (P2) | scratch: a declared and verified predecessor (with a `.DS_Store`) is excluded and the scan passes; the same root undeclared is refused, by name. A declared root that later gains `launches/`, an eval dir, a state view or a launch log is refused at the scan; a wrong sha is refused. **Real artefacts:** the real v2 configuration (74,084 keys, 313 tracked files) with v1 declared passes: v1's spec excluded, v2's own spec excluded as own root. With v1 undeclared it is refused: "1 artefact(s) … phase2x_ev_v1/phase_spec_de4e947e.json" | 4/4 |
| T13 | 5 (P2) | stubs, 1× test configuration. (a) The predecessor stops at exit 6 (ceiling 12 h, 5 launches). The v3 continuation (`--ceiling-h 1000`, note) runs to exit 0, and its **decisions equal the uninterrupted stub phase's** (T8): table, efficiency, candidates, pool, incumbent path. The two phases' requests are disjoint and their union equals T8's. `carried_h` = 9.58 h = the predecessor's cumulative hours and enters every ceiling check. (b) The predecessor stops at exit 5, `exit_view_mismatch` (1 ulp in candidate 1's F). The continuation with `--no-pilot-exit-views` first runs x0 at exit depth, records the mismatch (mode record) and completes (exit 0). (c) Refusals: an incomplete predecessor launch; a completed predecessor; a wrong sha; a synthetic oracle; a raised ceiling or `--no-pilot-exit-views` without a continuation; an un-launched root declared as a continuation. (d) A predecessor that gains a launch after the v3 freeze makes every v3 launch refuse (exit 2, nothing written) | 5/5 |
| T14 | 6 | a spec frozen with a different policy hash: `--phase-run` and `--preflight-only` both exit 2 with "code drift: p515_s56_search_policy.py"; nothing written, no lock; the same spec without drift launches (4); `code_drift` lists exactly that file | 2/2 |
| T15 | 7 | `.DS_Store` planted at the top level, in `launches/`, in a launch dir and in its `evals/`: the next launch runs (4) and its manifest lists none. A killed launch holding only `.DS_Store` files is moved aside with 0 killed evaluations. The phase completes byte-equal to T2's. `files_under` skips them; a root holding only `.DS_Store` can be frozen | 2/2 |
| T16 | 8 | `s56_phase_check.py` on crafted records: all PASS gives 0 (walls 0.7 h and 1.6 h inclusive); uncertified, read_cycle 300, a guard false, no guards, 0.69 h and 1.61 h each give 1 with FAIL; a missing manifest or record gives 2. The **single invocation** on a test root runs launch 001 (exit 4) and refuses a second issue ("logs exist: not re-run"); **the loop then starts at 002** and ends at 0, byte-equal to the in-process run. No real command line contains the hazard string | 3/3 |

`.DS_Store` scan: no tracked file under `data/SRP1/Results` is named `.DS_Store`, and no driver-written manifest
lists one, so the filter changes no committed replay. Searched: `git ls-files`, and `git grep DS_Store` over the
manifests. Only two S46 harness manifests mention it, and the driver does not replay those.

### 2. Driver determinism gate v4 (re-run because shared driver code changed)

Not requested by the task. I ran it because the build touched driver code that the existing search shares:
- `files_under`;
- the interruption listing;
- `collision_scan` and `freeze_collision_scan`, both with new optional arguments;
- the synthetic landscape, behind a knob.

`p515_s56_search_gate.py --out data/SRP1/Results/P515S56/search_driver_v4`, from `7350e5da` (code hashes =
the spec's), attached, both streams captured: **167/167**, items 1–9, 1,261 s, stderr empty, guard verify(0) `[]`,
pickle 0.
- `search_gate.json` `d2507591620e…`; `manifest_sha256.json` `6a62fcabd6d1…`;
  `search_driver_v4_launch_stdout.log` `45428e3ebc88…`.
- The same 167 checks as gate v3 (`2aaffd52…`, 167/167).
- This includes item 4's 2× EV freeze: its collision scan is clean against the tracked artefacts, which now
  include the v1 and v2 phase specs. Its settled cap is the gate's test cap of 400, so its keys differ from the
  pilot's in any case.

### 3. Freeze, collision scan, preflight

Freeze, run from `5c023309` with clean code files; logs in the root, stderr empty:
`p515_s56_search_driver.py --phase-freeze --root data/SRP1/Results/P515S56/phase2x_ev_v2 --phase-id
s56phase2x_ev_v2 --oracle harness --predictions data/SRP1/Results/P515S56/phase2x_ev_v2_predictions.json
--unlaunched-predecessor data/SRP1/Results/P515S56/phase2x_ev_v1@de4e947e` → exit 0.
- **Collision scan:** 74,084 keys against 313 tracked artefacts gives **0 collisions**. Excluded: 6 files in the
  pilot cache root, and **1 predecessor file (`phase2x_ev_v1/phase_spec_de4e947e.json`)**, verified un-launched.
- **Cache-root key audit:** ok. The keys found are exactly the 3 indexed pilot keys; no exit-depth key was found.

Preflight-only, run from `7350e5da` (`NLP_SOLVER_PATH=/usr/local/bin/ipopt`):
`p515_s56_search_driver.py --phase-run --root data/SRP1/Results/P515S56/phase2x_ev_v2 --preflight-only --out
data/SRP1/Results/P515S56/phase2x_ev_v2_preflight` → **exit 0, pass true**. Logs were captured in the scratchpad
and copied in: `pf_stdout.log` `1fc4102094ee…`, `pf_stderr.log` empty. Record `phase_preflight.json`
`be2f5878ff48…`.
- Code drift: none. Un-launched predecessor: re-verified. Harness preconditions `[]`; refusals `[]`.
- **Memory: 16.73 GiB available vs 16.45 GiB required**, with the Claude app open. The margin is 0.28 GiB; the
  pilot measured 21.8 GiB with the app closed.
- **Next action: launch 1, screen, exit depth, 5 plans:** `n5_y2025_p0.25_e1`, `n5_y2025_p0.25_e0.5`,
  `n5_y2030_p0.25_e1`, `n5_y2030_p0.25_e0.5`, `n5_y2035_p0.25_e1`.
- **Ceiling:** carried 0 + 0 + 1.29 h ≤ 120 h.
- Cache as loaded: x0, the unit and the staged plan at both depths (exit views certified, settled records
  settled). Lock acquired as `s56phase2x_ev_v2_l001` and released; no launch dir; nothing written under the root.

### 4. Projected hours (PROJECTED, not measured; `projected_hours` in the spec)

Per block: exit 1.292 h, settled 3.094 h. Under exit-then-confirm, every confirmation is a separate launch at
3.094 h. q is the confirm-candidate share of the final complete poll's 46 uncached neighbours of the staged plan.
The candidate-poll bracket (small or large |N|) does not change any figure.

| search case | exit-then-confirm, q = 0 | q = 1/4 (12 conf.) | q = 1/2 (23 conf.) | settled-direct |
|---|---|---|---|---|
| no move (one complete poll) | 32.8 h | 69.9 h | 104.0 h | 65.2 h |
| 3 moves, improver in the first block, 1 confirmation each | 46.0 h | 83.1 h | 117.1 h | 74.5 h |
| 6 moves, improver last (max(2, final count) confirmations each) | 147.5 h | 370.2 h | 608.4 h | 250.9 h |

- The q = 0 column reproduces v1's figures exactly. v1's basis left out the final poll's confirmations.
- Break-even per complete poll: 5.8 confirmations. q = 1 in the no-move case gives 175.1 h (Blocked 1).
- Every launch is one block: about 1.3 h at exit depth, or about 2.4–3.1 h for a settled block or a single
  confirmation. All are under 4 h.

### 5. The three commands (author, Terminal, Claude app closed)

None of the three command lines contains the harness hazard string (T16).

**(a) Launch 1 only, without the loop.** It refuses to run if `launch_001` logs already exist:

```
cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && caffeinate -i zsh -c 'R=data/SRP1/Results/P515S56/phase2x_ev_v2; n=001; if [ -e "$R/launch_${n}_stdout.log" ]; then echo "launch_${n} logs exist: not re-run"; else NLP_SOLVER_PATH=/usr/local/bin/ipopt /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_search_driver.py --phase-run --root "$R" --take-over-stale-lock > "$R/launch_${n}_stdout.log" 2> "$R/launch_${n}_stderr.log"; rc=$?; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launch $n exit $rc"; fi'
```

Expected: one line `<UTC> launch 001 exit 4` after about 1.3 h. Any other exit code: stop and report it. Exit 2
means refused with nothing written (for example memory); the reason is in `launch_001_stdout.log`.

**(b) The check, run afterwards.** It reads `launches/launch_001.json` and prints the launch wall and whether the
manifest holds the record. Then, per evaluation, it prints status, read_cycle, the guards, the child wall, F and
bar, with PASS or FAIL against: certified; read_cycle < 300; all guards true; wall 0.7–1.6 h. It exits 0 if all
pass, 1 on any FAIL, 2 if the record is missing:

```
cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python s56_phase_check.py data/SRP1/Results/P515S56/phase2x_ev_v2 --launch 1
```

**(c) The loop for the remaining launches.** It starts its numbering after the logs already present, so after
(a) the first line is `launch 002`. It re-issues the launch while the exit code is 4:

```
cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && caffeinate -i zsh -c 'R=data/SRP1/Results/P515S56/phase2x_ev_v2; i=1; while [ -e "$R/launch_$(printf %03d $i)_stdout.log" ]; do i=$((i+1)); done; while true; do n=$(printf %03d $i); NLP_SOLVER_PATH=/usr/local/bin/ipopt /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_search_driver.py --phase-run --root "$R" --take-over-stale-lock > "$R/launch_${n}_stdout.log" 2> "$R/launch_${n}_stderr.log"; rc=$?; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launch $n exit $rc"; if [ $rc -ne 4 ]; then echo "loop stopped: exit $rc (0 complete, 5 review, 6 ceiling, other fault)"; break; fi; i=$((i+1)); done'
```

Exit codes:
- **0** complete.
- **5** review: harness_error, offset above e, not_settled trigger, start not certified, or
  **exit_view_mismatch**. The reason is in the stdout and in `phase_state.json`.
- **6** the ceiling (`ceiling/ceiling_stop_NNN.json`).
- **2** refused, nothing written. Code drift is now one cause, so **do not edit the driver, policy, harness,
  lattice, neighbourhood or `gate_result_io.py` while the phase runs**.
- **1** crash.

A log number counts invocations; a driver launch index counts complete launches. They differ only after a
refusal or a kill. The check script reads the driver index (`launches/launch_001.json`).

### 6. Recorded predictions (v2, embedded in the spec)

File `phase2x_ev_v2_predictions.json` (sha256 `5e660820d83c…`; predecessor v1 file `00d62e1c…`). Q2, Q3 and Q5 are
as in v1.

| # | change from v1 |
|---|---|
| Q1 | **recorded, non-stopping (R1).** `on_failure`: no phase decision reads sign(dF_exit) (screen, ladder, table, efficiency, candidates and pool read the exit class and settled values), and the threshold dF_exit < 0 lies inside the e_EV band (|dF_exit| ≤ 17,494 €). A failure is therefore not a determinate sign result and not a CLAUDE.md "failed sign prediction"; it is reported at the end. `decision_role`: none |
| Q4 | basis corrected: both branches with confirmations (§4) |
| Q6 | measure P1; evaluated as `report.efficiency`: n_not_worse, class_counts, fraction, branch, and `band_measure_information_only` |
| Q7 | **new (R4):** for each candidate, the exit run's (read_cycle, F, bar) equals the settled summary's `exit_view`, bitwise. It is checked automatically: mismatch gives exit 5, `exit_view_mismatch`. Fallback (TASKS): exit-depth runs of x0, the unit and the staged plan, then zero-solve reclassification, as a v3 continuation with `--no-pilot-exit-views`. Its freeze is a Planner decision. Inputs: `prediction_inputs.Q7` |

## Not confirmed

- **Exit view = exit run, on real records.** No EV exit-depth run at 2× exists. The bitwise check (Q7) is the
  first place it will be observed, at candidate confirmation, around launch 6. Until then the screen classifies
  candidates against x0's exit view. Searched: the v1 report's Not confirmed, and TASKS 2b (cycles 1–2 bitwise
  only).
- **The continuation on real records.** It is tested on stubs only: harness `evaluate` in stub mode, with the
  1× test configuration. The real path through `load_cache` → `map_record` on a continuation root has not run.
- **The real `HarnessOracle.evaluate` path with real children.** Unchanged from v1: it has never run with real
  children. The preflight stops before the children.
- **`s56_phase_check.py` on a real launch record.** It has run on crafted records only. The field names it
  reads (`summary.status`, `read_cycle`, `guards`, `wall_s`) are those `map_record` writes for exit runs; checked
  against the source, not against a real record.
- **The single-invocation command** was run with `zsh -c` from this agent shell on a test root (T16), not in
  Terminal.app with the Claude app closed.
- **The final poll's confirm-candidate share q** is unknown. The bracket 0–1/2 is tied to the P1 branch
  condition, not to any measurement (Blocked 1).
- **Memory headroom:** 0.28 GiB with the app open. The pilot's 21.8 GiB was measured with it closed.
