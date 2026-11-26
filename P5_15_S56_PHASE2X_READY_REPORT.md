# P5.15 S56 — EV 2× phase: built, tested, spec frozen, preflight passed (Worker report, 2026-11-24)

Authority: `TASKS.md`, "Author decisions on report (i)" (2026-11-24) and step 5a; the Planner's phase-build task of
2026-11-24. Read: `P5_15_S56_REPORT_i.md`, `P5_15_S56_PILOT_REPORT.md`, `S56_SEARCH_DRIVER_DESIGN.md` (all sections),
`S56_PLAN_v4_1.md` §3–4, the driver, policy, gate and oracle-test modules.

**Solve claim.** Nothing was solved. Every test, the gate, the freeze and the preflight ran under an armed
`SolveProfileGuard(permitted=())`, verified at 0 (pickle load/loads blocked, 0). No phase launch was made: the phase
root holds only its frozen spec and freeze logs.

**Objective convention (every € figure):** F = I(x) + gross_operational_cost − TV credit, € of total discounted
planning cost (cost-breakdown schema v2; gross settlement-excluded). dF_exit = F_exit(x) − F_exit(x0). All values in
the tests are SYNTHETIC (landscape or stub children); the € values quoted from the pilot are measured.

## Blocked on Planner

Nothing blocks the launch mechanically: the spec is frozen, the preflight passes and the command is below. The TASKS
step names an **Advisor review of the built phase** before the command goes to the author. These are the readings I
took where the task did not fix a value. Each is frozen in spec `de4e947e`; changing one needs a v2 spec before the
first launch (zero solves, about 15 min).

1. **Exit-depth cap = 300 (= K_EV).** No document I searched declares the search's exit cap (`S56_SEARCH_DRIVER_DESIGN.md`,
   `S56_PLAN_v4_1.md`, `P5_15_S56_REPORT_i.md`, the pilot report, the oracle tests use placeholder 500). I used 300, so an
   exit run differs from a pilot run only in the stop and the terminal capture keys. Consequence: an exit run that never
   certifies runs to 300 cycles (about 2.4 h), so that block lasts up to about 3 h against the ceiling's 1.29 h exit
   projection. The ceiling can then be overshot by at most that block's excess (≤ about 1.8 h).
2. **x0, the reference unit and the staged plan get their exit values from the exit views of the pilot's settled
   records.** Their exit-depth keys differ from the pilot keys (the pilot ran stop-disabled with pricing capture), so no
   exit-depth record matches exactly. The exit view is the value at the would-be certification cycle — the reading e_EV
   was calibrated on. Each such summary is tagged `served_as: exit_view_of_settled_cache_record` with its settled key.
   The alternative is one exit-depth run of x0 and the unit (≈ 1.3 h, one block) before the screen.
3. **All three review stops exit 5.** The task's exit-code table gives 5 for review; it also says "each has its own exit
   code". I read that as "a code distinct from complete, more work, ceiling and fault". The reason is printed on the
   launch's stdout and recorded in `phase_state.json` and `phase_results.json`. If distinct codes are wanted, that is a
   v2.
4. **Table class e.** I classify with `P.classify_exit(x, x0, R, e_floor = e_EV)`, the design's rule with e_EV as the
   floor: e = max(e_EV, bar_x + bar_x0). This equals e_EV = 17,494 € unless a bar sum exceeds it (the pilot's largest bar
   sum was 13,811 €). The efficiency band uses e_EV exactly.
5. **Efficiency fraction.** Denominator: every screen and ladder row, uncertified rows included (they count as outside
   the band), and the unit's cached row. "Exceeds 0.5" is strict: 11 of 22 → exit-then-confirm.
6. **Ladder.** Best duration = the duration of the best certified screen plan. The two best node-years are the two best
   certified screen plans at that duration. A point the lattice refuses is skipped, not replaced: **1.0 MVA × 4 h in 2025
   is over budget** (I = 1,271,828 € > 1,000,000 €). So if the best duration is 4 h and one best node-year is 2025, the
   ladder has 3 points.
7. **Candidates.** A candidate is "confirmed" if its settled run is `settled` with every gate passing; the plan §3 pair
   test against x0 is recorded beside it, and its held-out offset check applies (offset > e → review). An unconfirmed
   candidate is not polled. A candidate poll is a complete poll with no confirmation inside it (as in report (i)'s
   13 h per candidate).
8. **Settled-direct branch.** The first determinate improver (plan §3 on settled values) is accepted directly: there is
   no exit comparison, so no held-out check. Every settled classification counts as a confirmation for the not_settled
   trigger, which is applied per sub-search and phase-wide.
9. **No evaluation count.** No N_max was given. The sub-searches use N_max = the domain size (37,042, non-binding), so
   the author's 120 h ceiling is the only budget.
10. **Q1 is a sign prediction.** CLAUDE.md stops a campaign on a failed sign prediction. The driver evaluates no
    prediction mid-phase, so a Q1 failure would be read only at the end. If you want Q1 read after the screen (a manual
    stop between launches is possible: the loop can be interrupted between invocations), say so in the command note to
    the author.

## Changed

| commit | content |
|---|---|
| `637090ba` | Phase build: `p515_s56_search_policy.py` (settled-direct and poll-only modes; the pure phase machine `phase_step`), `p515_s56_search_driver.py` (`--phase-freeze`, `--phase-run`, ceiling, cache + exit views, preflight-only, results, author command), `p515_s56_search_gate.py` (item 6 re-scoped), `p515_s56_phase_tests.py` |
| `f2354a12` | `p515_s56_phase_tests.py` v2: T1's default-path identity pinned to `6b1d98b2` (Found §1) |
| `bd3a0534` | `phase2x_ev_v1/phase_spec_de4e947e.json` + freeze logs, `phase2x_ev_v1_predictions.json` |
| (this commit) | test evidence v1 + v2, gate v3 evidence, preflight record, the hash inventory `phase2x_ready_evidence_manifest_sha256.json` (`72edf0eda38d…`, 22 files), this report |

**Where it is built (Worker's call, logged).** The orchestration extends `p515_s56_search_driver.py` (a section "THE 2x
PHASE"), so the oracles, the lock, the interruption handling, the hash chain and the collision scan are the search's
own code. Every decision is in the pure policy module (`phase_step`), as the design requires.

**The phase, as frozen** (spec `de4e947e`; phase id `s56phase2x_ev_v1`; root
`data/SRP1/Results/P515S56/phase2x_ev_v1`):
- Grid 2025/2030/2035, EV twin, 2×, every S56 EV key as the pilot: the harness configuration is the pilot spec's
  `x0_i1` entry (ρ0 × 1). Settled depth: that entry's options (stop-disabled, pricing capture) at cap K_EV = 300; exit
  depth: the same without those two keys, at cap 300.
- R = 10,000 €; e_EV = 17,494.05 € (read from `pilot_ev_v1_result_r2/pilot_result.json`, asserted to round to 17,494);
  block 5; cap 80; same-sign diagonals; class rank default; 5 slots × 3.29 GiB.
- Screen: 18 single-unit plans (nodes 5, 7, 9 × 2025/2030/2035 × {0.25/1, 0.25/0.5}), each checked against the lattice
  (`reasons` empty) and canonicalised (`canonicalize` == `canonical_of_z`, `z_of_canonical` round-trips, key recorded).
  The 36 possible ladder points are checked the same way at freeze; 3 are inadmissible (1.0 MVA × 4 h in 2025 at each
  node).
- Cache: the pilot root, indexed by exact eval key → exactly x0_i1, unit_i1, staged_i1 at the settled key; i2/i3 are
  ignored (their ρ0 option enters the key). Exit views as in Blocked 2. No other settled record exists in the declared
  cache root, so the start pool has no historical members.
- Start pool: x0, the reference unit, the pilot staged plan, the confirmed candidates. Best start = lowest settled F among
  members whose settled summary is `settled` with gates passing (ties by label).
- Efficiency rule, ceiling, exit codes and stops as in the task (Blocked 3–5, 8, 9).

**Policy changes and the existing search.** Both new modes are read with `.get`, so a config without them runs exactly
as before. T1 v2 checks this against the pre-phase policy (`6b1d98b2`): state and terminal report byte-identical on 6
synthetic searches. The synthetic landscape gained optional knobs (test-only); without them it is byte-identical to
`6b1d98b2`'s on 382 plans × 2 depths. The collision scan's scope now includes `phase_spec_*` (they carry eval keys).

## Found

### 1. Tests (zero solves)

`p515_s56_phase_tests.py`, attached, both streams captured, stderr empty, guard verify(0) `[]`, pickle 0.

- **v2 (the evidence): 43/43**, 334 s, from `f2354a12` code (run at `bd3a0534`, code files clean).
  `phase2x_tests_v2/phase_tests.json` `1da30940638d…`, `manifest_sha256.json` `20b38704979…` (5,754 files; the test
  trees are hash-recorded, not committed: they are scratch phase roots), `phase2x_tests_v2_stdout.log` `296202ef8a2c…`.
- **v1: 41/41 from `637090ba`, but its T1 default-path checks were vacuous.** They compared the policy and the
  landscape against `HEAD`, which after the build commit is the new code itself. I caught it in the record (HEAD
  policy sha `8fdae9f4…` = the new policy). v2 pins the comparison to `6b1d98b2` and asserts the files differ. v1 is
  kept (`phase2x_tests_v1/`, `abf1d18cf131…`); every other v1 check passed as in v2.

| test | what | result (v2) |
|---|---|---|
| T1 | `classify_settled` boundaries (dF = −R exactly → improver, −R + 0.01 → unresolved, +R → worse; V_pair = R/4 and A = R/2 inclusive; gate failure, not_settled, harness_error, missing V; mixed depth and unusable incumbents refused); efficiency 11/22 → exit-then-confirm, 12/22 → settled-direct, uncertified rows in the denominator; band edge \|dF\| = e_EV in, + 0.01 out; candidate selection (same node or same year passed over; fewer logged); ladder (1.0 MVA × 4 h 2025 skipped for budget); **default search path byte-identical to `6b1d98b2`'s policy** on 6 synthetic searches (3 starts × 2 class ranks; 14–48 evaluations, 0–5 moves), non-vacuous; plain landscape byte-identical on 382 plans × 2 depths; capture checklist | 11/11 |
| T2 | whole phase through the launch CLI, exit-then-confirm: 31 launches (codes 4…4, 0), stages screen → ladder → candidate confirmation → candidate polls → search; 123 evaluations (screen 17, ladder 4, confirmations 2, candidate polls 35, search 61 exit + 4 settled); final state = results state = in-process phase, byte-equal; ≤ 5 per launch at one depth; pool members never requested, the unit's screen row served from the cache; hours = Σ launch walls; every launch passed the ceiling check | 7/7 |
| T3 | settled-direct branch (fraction 13/22): exit 0, byte-equal to in process; every candidate-poll and search evaluation at settled depth (44 + 75); the search moved; count identities hold | 3/3 |
| T4 | ceiling (test ceiling 12 h, simulated walls): after 5 launches, cumulative 9.58 h + projected settled block 3.09 h = 12.68 h > 12 → exit 6 before the block, nothing launched; arithmetic checked (projection = max(frozen, measured same depth)); re-issued → 6 again, no new launch, a second ceiling record | 1/1 |
| T5 | review stops, each exit 5 with its reason and stage: harness_error (screen); offset above e at a candidate confirmation; offset above e at a search confirmation; not_settled trigger (3 consecutive in a settled-direct candidate poll); start not certified (x0 exit uncertified) | 5/5 |
| T6 | kill (SIGKILL of the process group) and resume at 8 transitions, one or more per stage: after_lock (l1), after_launch_dir (l2, screen), during_evaluate (l5, ladder: 3 killed evaluations), after_evaluate (l6, confirmations), after_launch_record (l7, candidate poll), after_manifest (l15, search), after_state_view (l17, search confirmation), after_results (l31) → phase state and results state byte-equal to T2's run; killed evaluations uncounted; stale lock taken over | 8/8 |
| T7 | **the author's command** (the exact text of §5, with the test root and a scratch `--lock-path`): 31 invocations, each line printed, 4…4 then 0, "loop stopped: exit 0", one log pair per invocation, stderr empty; byte-equal to in process; re-issued on the finished phase: one invocation, exit 0, log number 032 | 2/2 |
| T8 | stub children (harness `evaluate`, stub mode, a 1× test configuration at caps 500/400): freeze with collision scan clean, 31 launches, exit 0, byte-equal to the in-process synthetic phase; one harness spec per launch, keys recomputed at replay | 2/2 |
| T9 | refusals (exit 2, nothing written): real freeze without predictions; synthetic phase not test-only; `--preflight-only` on a synthetic phase; a tampered record (replay refuses; restored → proceeds, exit 4); re-freezing an existing root | 2/2 |

### 2. Driver gate v3 (item 6 re-scoped)

`p515_s56_search_gate.py --out data/SRP1/Results/P515S56/search_driver_v3` from `637090ba`, attached, both streams
captured: **167/167**, 1,266 s, stderr empty, guard verify(0) `[]`, pickle 0. `search_gate.json` `2aaffd52dd71…`,
`manifest_sha256.json` `552ce0583b26…`, `search_driver_v3_launch_stdout.log` `91f456f6e1fe…`.

| item | checks | result | touches changed code |
|---|---|---|---|
| 1 N(x) | 11 | pass | driver (neighbourhood env unchanged) |
| 2 order determinism | 4 | pass | policy |
| 3 order vs values | 2 | pass | policy, driver launches |
| 4 stub replay + smoke alias absent | 8 | pass | policy `step`, driver launch, landscape |
| 5 kill and resume | 36 | pass | policy `step`, driver launch, `handle_interruptions` |
| 6 keys (re-scoped) | 28 | **pass** | collision-scan scope |
| 7 boundaries + not_settled | 45 | pass | policy |
| 8 arithmetic + not_settled in search | 25 | pass | policy `terminal_report`, `count_identity` |
| 9 published histories | 6 | pass | policy |

I re-ran the whole gate rather than a subset: every item runs the changed policy or driver.

**Item 6 re-scope verdict: holds.** Five checks were re-scoped to the cap-in-key rule, as ruled on the pilot-ready report
(Blocked 2):
- settled-key relation (EV and draw-0): exit = cap(b) and settled = cap(sk(b)) ≠ exit, with b the pre-change harness's
  (`b0391b80`) key of the same exit entry; candidate key unchanged; the entry field is True and absent on exit entries;
- option-off identity of the three S56 sets (`s56_ev_staged`, `s56_draw0_staged`, `calibration_key_coupled`): post
  eval key = cap(pre) on every entry; the specs otherwise byte-identical except the entry fields `eval_key`,
  `eval_dir`, `working_dir_ids` and `s56_cycle_cap_keyed` (the field set the v2 diagnosis found);
- the set without an S56 key (`srp1_single_year`) is still checked byte-identical, unchanged.

Against gate v2 (162/167, 5 item-6 failures): those 5 now pass under the re-scoped relations, and the other 162 pass
again.

### 3. Freeze, collision scan and cache-root audit

Command (from `f2354a12`, clean tree; logs `freeze_stdout.log` / `freeze_stderr.log` in the root, stderr empty):
`p515_s56_search_driver.py --phase-freeze --root data/SRP1/Results/P515S56/phase2x_ev_v1 --phase-id s56phase2x_ev_v1
--oracle harness --predictions data/SRP1/Results/P515S56/phase2x_ev_v1_predictions.json` → exit 0.

**Spec `phase_spec_de4e947e.json`** (sha256 `de4e947ec3d3720e7336c907ed0a683d9a3e079de40edb159f673fa5b2e6ac7e`),
committed in `bd3a0534` before any launch. Code hashes recorded: driver `6e1670e5…`, policy `8fdae9f4…`, harness
`05a9838c…` (the harness pin of gate F and the pilot).

- **Collision scan** (cache roots and own root excluded, the W105 rule): 74,084 keys (37,042 plans × 2 depths, all
  distinct) against 313 tracked campaign / search / phase specs and launch records → **0 collisions**. 6 files excluded
  as inside the declared cache root.
- **Cache-root key audit** (this covers the pilot-report item 4 check, since the pilot's launch records are tracked
  now): the phase's keys found inside the 6 tracked files of the pilot root are **exactly the 3 indexed cache keys**
  (`142a0289…` x0_i1, `fe407368…` unit_i1, `dddc5b83…` staged_i1, each in `launch_001.json` and its harness spec), and
  **no exit-depth key**. The i2/i3 records are ignored by the index ("another configuration": ρ0 enters the key).
- **Capture-path checklist** (rule eleven, before any run): all 9 true — the mapping's paths (F_exit, bar, settled F,
  V, gates), `local_solve_failures` and `peak_rss` in the record (Q2, Q5), the launch wall in the phase launch record
  (Q4, ceiling), the termination, efficiency record and best start in the phase state (Q3, Q4, Q6), the per-cycle
  trajectory by default (stage template).
- **Cache as loaded** (zero solves, checked in process from the scratch freeze of the same code):

| plan | depth | served as | status | F (€) | bar (€) | read cycle |
|---|---|---|---|---|---|---|
| x0 | exit | exit view of x0_i1 | certified | 812,009,208.71 | 4,271.36 | 120 |
| x0 | settled | cache record | settled | 812,007,610.45 | — | 300 |
| reference unit | exit | exit view of unit_i1 | certified | 811,988,275.92 | 1,406.11 | 126 |
| reference unit | settled | cache record | settled | 811,985,758.84 | — | 300 |
| pilot staged | exit | exit view of staged_i1 | certified | 811,976,794.78 | 2,490.03 | 127 |
| pilot staged | settled | cache record | settled | 811,973,592.57 | — | 300 |

- The 3 inadmissible possible ladder points are `n5_y2025_p1_e4`, `n7_y2025_p1_e4`, `n9_y2025_p1_e4` (budget).

### 4. Preflight-only (zero solves)

`p515_s56_search_driver.py --phase-run --root data/SRP1/Results/P515S56/phase2x_ev_v1 --preflight-only --out
data/SRP1/Results/P515S56/phase2x_ev_v1_preflight`, `NLP_SOLVER_PATH=/usr/local/bin/ipopt`, from `bd3a0534` → exit 0.
Logs captured in the scratchpad and copied in (`--out` must not exist before the run): `pf_stdout.log`
`07c8ae4e099b…`, `pf_stderr.log` empty. Record `phase_preflight.json` `c70aac5210e9…`.

- **pass: true.** Harness preconditions `[]` (no legacy lock, no campaign lock, no live `p515_s4` process, launch dir
  absent, production / harness / driver files clean in git); refusals `[]`; solver path as required.
- **Memory: 17.46 GiB available vs 16.45 GiB required** (5 × 3.29), measured with the Claude app open. The pilot
  measured 21.8 GiB with the app closed.
- **Next action: launch 1, stage screen, exit depth, 5 plans** (`n5_y2025_p0.25_e1`, `n5_y2025_p0.25_e0.5`,
  `n5_y2030_p0.25_e1`, `n5_y2030_p0.25_e0.5`, `n5_y2035_p0.25_e1`).
- **Ceiling check: 0.00 h + 1.29 h ≤ 120 h.**
- **Cache as loaded by the launch path:** x0, the unit and the staged plan at both depths (exit views certified, settled
  records settled).
- Lock acquired as `s56phase2x_ev_v1_l001` and released; no launch dir created; nothing written under the phase root.

### 5. The exact author command

Run once in Terminal, with the Claude app closed. It runs in the foreground under `caffeinate -i` and re-issues the
launch while the exit code is 4:

```
cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && caffeinate -i zsh -c 'R=data/SRP1/Results/P515S56/phase2x_ev_v1; i=1; while [ -e "$R/launch_$(printf %03d $i)_stdout.log" ]; do i=$((i+1)); done; while true; do n=$(printf %03d $i); NLP_SOLVER_PATH=/usr/local/bin/ipopt /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_search_driver.py --phase-run --root "$R" --take-over-stale-lock > "$R/launch_${n}_stdout.log" 2> "$R/launch_${n}_stderr.log"; rc=$?; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launch $n exit $rc"; if [ $rc -ne 4 ]; then echo "loop stopped: exit $rc (0 complete, 5 review, 6 ceiling, other fault)"; break; fi; i=$((i+1)); done'
```

- Each launch logs to `data/SRP1/Results/P515S56/phase2x_ev_v1/launch_NNN_stdout.log` and `launch_NNN_stderr.log`. NNN
  counts invocations; it continues from the logs already there, so after a stop the same command resumes.
- The loop prints one line per launch (`<UTC> launch NNN exit C`) and stops on anything but 4: **0** the phase is
  complete; **5** review (harness_error, offset above e, not_settled trigger, start not certified; the reason is in the
  launch's stdout and `phase_state.json`); **6** the 120 h ceiling (hours summary in `ceiling/ceiling_stop_NNN.json`);
  **2** refused, nothing written (for example the memory preflight); **1** crash.
- A reboot or a kill costs at most the block in flight: the next invocation moves the incomplete launch aside, takes
  over the dead lock (only if its pid is dead and it names this phase) and re-issues the same block.

### 6. Projected hours (PROJECTED, not measured)

Computed by `phase_hours_projection` and frozen in the spec (`projected_hours`). Per block (from the pilot's committed
records, `phase.ceiling.projection_inputs`):
- exit block = κ ψ t_exit = 1.18 × 1.0867 × 1.0079 h = **1.292 h**;
- settled block = κ ψ t_300 = 1.18 × 1.0867 × 2.4126 h = **3.094 h** (t_300 = mean pilot child wall to K = 300; the
  pilot's measured 5-way settled block was 2.63 h, so this is conservative).

Counts: screen 17 uncached (the unit is cached) → 4 blocks; ladder 1 block; candidate confirmations 1 settled block;
each candidate poll 18–19 uncached neighbours → 4 blocks; the search from the pilot staged plan, |N| = 46 (46
uncached).

| case | exit-then-confirm | settled-direct |
|---|---|---|
| search: no move (one complete poll) | **32.8 h** | **65.2 h** |
| search: 3 moves, improver in the first block, 1 confirmation each | **46.0 h** | **74.5 h** |
| search: 6 moves, improver last, 2 confirmations each | 147.5 h (ceiling stops it) | 250.9 h (ceiling stops it) |

- The candidate-poll bracket (smallest vs largest uncached |N|) does not change the totals: both give 4 blocks each.
- The ceiling binds only in the late-improver case. Report (i) projected 30–120 h.
- Every launch is one block: about 1.3 h (exit) or about 2.6–3.1 h (settled), all under 4 h.

### 7. Recorded predictions (frozen in the spec)

File `data/SRP1/Results/P515S56/phase2x_ev_v1_predictions.json` (sha256 `00d62e1c64d2…`), embedded verbatim in the
spec (`predictions_recorded_before_run`, with the file's sha256). Each has a statement, its operational evaluation
(fields of `phase_results.json`, which the driver writes at the end, including `prediction_inputs`), a threshold and
its failure consequence.

| # | prediction | evaluated as | my numbers (basis) |
|---|---|---|---|
| Q1 | every screen plan has F_exit below x0's F_exit | the 18 screen rows: certified and dF_exit < 0 (an uncertified row fails); the count below −R is recorded, not predicted | the only evaluated screen plan, the unit (I = 317,957 €): unit_i1 − x0_i1 dF_exit −20,933 €; 9 pairs −12,395 to −26,611 €; settled −21.9 to −22.2 k€. The other 17 are untested |
| Q2 | 0 harness_error, 0 local-solve failures | `prediction_inputs.Q2`: harness_error summaries = 0, Σ local_solve_failures over every launched record = 0 | pilot 7/7 0 failures over 300 cycles; step 4 4,974 solves 0 failures |
| Q3 | best start is not x0 | `pool.chosen_label` ≠ x0 | pilot settled F(300): x0 812,007,610.45; unit 811,985,758.84; staged 811,973,592.57 € — x0 is highest by 21.9 / 34.0 k€; Q3 fails only if no other member is eligible |
| Q4 | the search terminates (exit 0) inside 120 h | the loop ends 0, `report.termination` = phase_complete, `hours.cumulative_h` ≤ 120 | projected 32.8–46.0 h (exit-then-confirm), 65.2–74.5 h (settled-direct); 147.5 / 250.9 h in the 6-move late-improver case (§6) |
| Q5 | peak child RSS ≤ 3.29 GiB | max over launched records of `peak_rss.child_python_process_ru_maxrss` / 2³⁰ | pilot 2.77–2.86 GiB; 3.29 = 2.86 × 1.15 |
| Q6 | the efficiency fraction is recorded, no direction | `report.efficiency` (k, n, exact fraction, branch, rule) | — |

## Not confirmed

- **The real `HarnessOracle.evaluate` path has never run with real children.** The pilot ran through `_pilot_launch`
  (same lock format, `freeze_campaign_spec`, `CampaignContext`, `H.evaluate`); the search/phase path has run only with
  stub children (T8, gate items 4–5). The preflight exercises everything up to the launch, not the children. Searched:
  `P5_15_S56_PILOT_READY_REPORT.md`, `P5_15_S56_PILOT_REPORT.md`, the oracle tests' `--real-check` (it calls
  `H.evaluate` directly).
- **Exit view = exit-depth run.** That an exit-depth run (stop on, no pricing capture, cap 300) reproduces the
  stop-disabled run's trajectory up to c* is not verified: no EV exit-depth record at 2× exists. Evidence beside it: the
  integration real check (stop-disabled, pricing capture, cap 3) had cycles 1–2 bitwise equal to the smoke, which
  predates both keys (TASKS 2b) — two cycles only, not up to c*. Blocked 2 names the 1.3 h alternative.
- **The ceiling's exit projection** (1.29 h per exit block) assumes every run certifies; a non-certifying run to cap 300
  lengthens its block to about 2.5–3 h (Blocked 1). The projection adapts only after a longer exit block has been
  measured.
- **The author's loop** was tested with `zsh -c` from this agent shell (T7), not in Terminal.app with the Claude app
  closed; `caffeinate -i` was exercised in T7.
- **Memory on larger plans.** The 3.29 GiB per slot rests on the pilot's plans (≤ 4 cohorts); search plans may carry
  more cohorts (Q5).
- **Contention at tail depth** was measured only in the pilot's two blocks (ψ 1.087 / 1.083); κ = 1.18 is gate F's
  cycles 1–3 figure, applied on top (conservative).
