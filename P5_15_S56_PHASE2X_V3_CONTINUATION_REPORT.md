# P5.15 S56 — EV 2× phase: solve-profile guard extended; v3 continuation freeze REFUSED at the cache-root key audit (Worker report, 2026-11-25)

Authority: `TASKS.md` step 5b. Planner decision of 2026-11-25: "extend the reconciliation exactly (ESSO recovery solves
counted; every failure recovered) — harness accounting, not formulation", then a continuation spec v3 reusing all v2
records, with a ceiling of 120 h total.

**Solve claim.** Zero solves. Every test, re-mapping, simulation and freeze attempt ran under an armed
`SolveProfileGuard(permitted=())`, verified at 0, with pickle load/loads blocked (0). Nothing was launched.

**Untouched:** the v2 root (read only, no file written, nothing staged), `TASKS.md`, production code and the harness.

**Objective convention (every € figure):** F = I(x) + gross_operational_cost − TV credit, in € of total discounted
planning cost (oracle mapping; gross is settlement-excluded).
- dF_exit = F_exit(x) − F_exit(x0).
- In a poll, dF_set = F_set(x) − F_set(poll incumbent).

## Blocked on Planner

1. **The v3 freeze is refused, so steps 3–4 are not done: no v3 spec, no preflight.**
   - `--phase-freeze … --continuation-of phase2x_ev_v2@18a224d1` exited 2 at the **cache-root key audit**
     (`cache_root_key_audit`, `p515_s56_search_driver.py:2987`; rule at `:3011`:
     `ok = set(found) == expected …`).
   - The audit reads only **tracked** artefacts (`committed_spec_files`, via `git ls-files`). In the v2 root only
     `phase_spec_18a224d1.json` is tracked; its 10 launch records are not.
   - So the audit finds the 3 pilot keys and expects 3 + 41 = 44. The continuation can never pass while the
     predecessor's launch records are untracked.
   - This path had never run. T13 tested continuations on stub phases, and stub phases skip this audit (the v2 report's
     Not confirmed: "the real path … on a continuation root has not run").
   - Refusal log: `phase2x_ev_v3_freeze_refusal1_stdout.log` (sha256 `b1ae453f7224…`). I moved it out of the would-be
     root and removed that empty directory, so the root does not exist now.
   - **Options:**
     - **(A) Track v2's launch artefacts, then freeze. No code change.** Stage and commit the 30 files the audit's name
       patterns cover: `launches/launch_NNN.json`, `launches/launch_NNN_manifest_sha256.json`, and
       `launches/launch_NNN/campaign_spec_*.json`, for NNN = 001–010.
       - Nothing is written under `phase2x_ev_v2/` on disk, but the task said not to touch the v2 root, so this needs
         your authorization.
       - Precedent: the pilot root has exactly these files tracked, and more (heartbeats, init identities, 203
         eval-dir files), committed with the pilot result in `1198d570`. v2's evidence will need committing for the
         end-of-phase report in any case.
       - **Simulated at zero solves (Found §4): the audit passes, and the resume point and all decisions come out as
         required.**
       - Cost: about 15 min (stage and commit, freeze, preflight, a short addendum to this report).
     - **(B) Fix the audit for continuations. A code change.** The audit would also scan the predecessor's launch
       records and launch-dir campaign specs, which `continuation_source` already verifies by hash chain. The equality
       stays exact.
       - This changes a pre-run check that the task did not authorize. It also changes the driver hash, so the spec
         would record a new commit.
       - Cost: about 45 min (change, a test of the real path, gate re-run optional, freeze, preflight).
     - **Recommendation: (A).** It needs no code change, follows the pilot precedent, and is demonstrated by the
       simulation. After (A), the commands in Found §6 apply unchanged.
2. **"50 existing evaluations" is 47 cache records.** These are 41 v2 evaluations, plus the pilot's 3 settled records
   and their 3 exit views. All 47 are served from cache in the simulation (44 `cache_record` + 3
   `exit_view_of_settled_cache_record`). Nothing to decide unless 50 referred to something else.

## Changed

| commit | content |
|---|---|
| `4030508f` | `p515_s56_search_driver.py`: `solve_profile_reconciliation` (`:803`), read by `guards_exact` (`:864`); `read_eval_files` (`:694`) reads the ESSO events file; summaries carry `solve_profile_reconciliation`; `MAPPING_SCHEMA` = `p515_s56_oracle_mapping_v2` (`:653`); 3 new capture-path checks (`:1141` ff.). New `p515_s56_esso_guard_tests.py` and `p515_s56_phase2x_v3_continuation_check.py` |
| `c0134486` | Check script fixed (part D was over-strict on bookkeeping fields, Found §4) and part `freeze-sim` added |
| (this commit) | Evidence (15 files) and hash inventory `phase2x_v3_continuation_evidence_manifest_sha256.json` (`08afef4a3cb7…`); this report |

- **The harness is unchanged.** The reconciliation is computed in the driver from fields the harness already records:
  `solve_profile`, `network_failures_summary`, and `esso_recovery_events_<label>.jsonl` in the eval dir.
- The harness's own `identity_holds` therefore stays False on these records. Only the mapped
  `guards['solve_profile_identity_holds']` changes.
- **No v3 spec exists.** The driver hash at `c0134486` is `3371b9c238e8…`. Policy, harness, lattice, neighbourhood and
  `gate_result_io` are unchanged from v2's spec.

## Found

### 1. Diagnosis: one ESSO recovery event costs exactly 1 + [tier2_attempted] solves (code references)

- **Event recorded iff a recovery was attempted.** `shared_energy_storage_data._optimize` (`:1403`) appends the event to
  the diagnostic sink only when `recovery_attempted` is true (`:1569`). `recovery_attempted` (`:1427`) means the primary
  IPOPT solve ended `internalSolverError`, `maxIterations` or `infeasible` (`_is_recoverable_shared_ess_failure`,
  `:1382`).
- **What a recovery costs.** Exactly one tier-1 cold retry runs (`_run_solver_attempt(…, log_suffix='recovery')`,
  `:1476`), which is one `solver.solve` (`:1376`). One tier-2 retry follows iff `tier2_attempted` (`:1487`,
  `log_suffix='recovery_tier2'`, `:1514`).
- **What the guard counts.** It counts every `OptSolver.solve` entry, raising or not (`p513_solve_profile_guard.py:55`,
  `:62`).
- **The success flag.** The event's `recovery_succeeded` (`:1578`) is `solver_result_succeeded` of the solution
  finally loaded.
- **Where the events go.** The only ESSO caller passing a node id passes the sink (`:104`). The sink is reset at the top
  of `_run_operational_planning` (`shared_resources_planning.py:2827`), before the initialisation solves. The runner
  captures new sink entries after every round (`p515_g_g1_g4_admm_gates.py:925`), writes them to
  `esso_recovery_events_<label>.jsonl` (`:871`) and counts them (`:1322`).
- **Why the identity is unsupported.** The runner's identity credits only network retries (`:1395`) and declares itself
  unsupported when any ESSO event exists (`:1397`, comment `:1387`: "ESSO recovery events carry no attempt count").
  The event record carries that count (`tier2_attempted`).
- On all 4 records every event is tier 1 with `recovery_succeeded` True, so the per-event cost is 1. The cost was
  derived from the event records, not assumed.

### 2. The guard (mapping v2)

`guards['solve_profile_identity_holds']` holds iff **either**:
- the harness identity holds (unchanged path); **or**
- the harness declared the reconciliation unsupported, there is at least 1 ESSO event, **and all** of the following hold:
  - **(1)** observed == base_solves + retry_solves_credited + Σ_events (1 + [tier2_attempted]), exactly;
  - **(2)** every network failure is recovered: recovered_tier1 + recovered_tier2 == n_blocks; no unrecovered,
    not_attempted or indeterminate block; and retry_solves_credited == tier1 + 2 × tier2;
  - **(3)** every ESSO event has `recovery_succeeded is True` and a consistent tier record, and the events file holds
    exactly `n_esso_recovery_events` events.

Otherwise the guard fails, as before. Nothing else in the mapping changed.

The cross-check `credited == tier1 + 2 tier2` in (2) is an addition of mine. It holds on every v2 record. Drop it if you
prefer the three conditions literally.

### 3. Tests and re-mapping (zero solves)

**Guard tests:** `esso_guard_tests_v1` (sha256 `2db535053ec1…`), **42/42**, stderr empty.

| test | result |
|---|---|
| E1: capture paths, including the 3 new inputs | holds |
| E2: the 4 real records | pre-fix driver (`2d46d3c8`, asserted equal to v2's frozen driver hash): guard False, not_certified / not_settled. New: holds via the extended identity, observed = expected (6,481; 15,361; 15,353; 15,354), certified / settled |
| E3: synthetic failures, 15 cases, each fails and maps to not_settled / not_certified | an unrecovered ESSO event; `recovery_succeeded` absent; observed ± 1; an unpaid tier 2; events file absent; an event dropped; network block unrecovered / not attempted / indeterminate; credited off; declared supported; inconsistent tier; no ESSO event; wrong record type. Positive control: a paid tier 2 holds |
| E4: scan | pilot + v2: 48 eval dirs, 75 (record, depth) mappings. Every record that passed before still passes, with an identical summary apart from `mapping_schema` and the new field. Changed: exactly the 4 records. Under all of `P515S56`: 104 records with a solve profile, old pass → new pass |

**Oracle regression:** `search_oracle_v2/zero_solve_mapping_v2`, **119/119**, the same count as the committed run.

**Re-map of v2:** `phase2x_ev_v3_remap` (`f004aa39f3df…`), **10/10**.
- B: the chain with the stored (v1) summaries reproduces v2's 10 actions, and its terminal state **byte-equal**.
- C: with the re-mapped summaries, all 10 recorded actions are re-derived identically.

| item | v2 as run | under the corrected guard |
|---|---|---|
| screen `n9_y2025_p0.25_e0.5` | uncertified | certified, **unresolved**: dF_exit +11,484.61 €, e 17,494.05 (F_exit 812,020,693.32 vs x0 812,009,208.71) |
| table classes | 16 confirm-candidate, 5 unresolved, 1 uncertified | 16 confirm-candidate, 6 unresolved, 0 uncertified |
| efficiency | 21/22, settled_direct | 22/22, settled_direct (branch unchanged) |
| ladder, candidates | (5, 2030), (9, 2030); `n5_y2030_p1_e4`, `n7_y2025_p0.25_e1` | unchanged |
| cand. 1 poll, `n5_y2030_p1_e3.5 + n5_y2035_p0.25_e0.5` | not_settled | **worse**: dF_set +51,578.02 €, V_pair 830.22. Poll: 9 worse, 1 unresolved; still no improver |
| cand. 2 poll pos 0, `n7_y2025_p0.25_e0.5` | not_settled | **worse**: +36,228.71 € |
| cand. 2 poll pos 5, `n5_y2025_p0.25_e0.5 + n7_y2025_p0.25_e1` | not_settled (stop) | **unresolved**: +6,530.28 € (\|dF\| < R) |
| cand. 2 poll pos 6–9 (evaluated in launch 10, unclassified at the stop) | — | improver −29,465.44; improver −10,313.15; improver −45,564.50; worse +11,481.25 |
| not_settled review | cand. 1: 1/10; **cand. 2: 2/6, share rule → stop**; phase-wide 3/18 | cand. 1: 0/10; cand. 2: 0/10; phase-wide 0/22 |

**Candidate 2's poll would have continued.**
- Its next action is a settled block of 5: `n5_y2035_p0.25_e1 + n7_y2025_p0.25_e1`, `n7_y2025_p0.25_e1 +
  n7_y2030_p0.25_e0.5`, `n7_y2025_p0.25_e1 + n7_y2030_p0.25_e1`, `n7_y2025_p0.25_e1 + n7_y2035_p0.25_e0.5`,
  `n7_y2025_p0.25_e1 + n7_y2035_p0.25_e1`.
- 12 of its 22 neighbours remain.
- The poll so far holds 5 improvers against the unit: the two from launch 9 (−22,057.69 and −18,366.37) and the three
  above. Candidate polls do not accept moves (poll only).

### 4. Resume point (freeze simulation; nothing frozen in the repository)

`phase2x_ev_v3_freeze_sim` (`444a2fd76a61…`), **15/15**, from `c0134486`.

The freeze ran in process exactly as `--phase-freeze` would, with v1 declared as the un-launched predecessor and v2 as
the continuation, and with the v2 launch artefacts (30 files) added to the tracked list. That is option (A). It was
frozen into a scratch root, then part D ran on it.

- **Freeze checks:** cache-root audit ok (37 files); collision scan 0 collisions over 74,084 keys.
- **Ceiling:** 120 h, not raised, **carried 17.99 h**.
- **Unchanged from v2:** R 10,000; e_EV 17,494.05; K_EV 300; exit-view check `stop`; efficiency threshold 1/2; pilot
  exit views.
- **Simulated spec:** `phase_spec_91b9bdb4…`. It is hash-recorded only and **must never be tracked**: it carries v3's
  keys, so tracking it would collide with the real freeze.
- **First action:** the 5-plan settled block above, as candidate 2's poll (`cand_pos` 1, sub-search incumbent = the
  unit). None of the 5 is cached.
- **Cache:** all 41 v2 records indexed, none ignored. 47 cache records served; the v3 state has evaluated nothing.
- **Decisions equal remap C:** table, efficiency, candidates and confirmations, both polls, reviews.
- **Fields that differ by construction (reported, not compared):**
  - sub-search ids (`s56phase2x_ev_v3_cand*_poll`);
  - the sub-action `action_index`: 0 in v3, 2 in v2;
  - candidate 1's poll `budget_used`: 0 in v3, 9 in v2.

  Both counters count requests issued within the phase. N_max is the domain size, so they decide nothing.
- **My first part-D check failed** on exactly those fields, because it compared them. It is fixed in `c0134486`; the
  failing trial is in the scratchpad only.
- **Consequence for the end-of-phase report:** v3's evaluation accounting (`evaluations_requested`, `budget_used`,
  cache-hit flags) counts v3 launches only. v2's 41 appear as cache hits, while their hours are carried.

### 5. Hours (MEASURED for v2; PROJECTED for v3)

- **Used:** 17.99 h. v2 launches 1–10, completed walls, 0 interrupted:
  - exit launches: 1.09, 1.08, 1.09, 0.97, 1.39 h;
  - settled launches: 2.01, 2.48, 2.57, 2.58, 2.72 h.
- **Remaining under the 120 h total:** 102.01 h.
- **First v3 ceiling check:** 17.99 carried + 0 + 3.09 h (frozen settled projection; v3 has no measured launch yet) =
  21.08 h ≤ 120.
- **Rest of candidate 2's poll:** 12 plans = 3 launches (5, 5, 2). That is about 7.5–8.2 h at v2's measured settled
  walls, or 9.3 h at the frozen projection. Pool and search follow; I have not projected them.

### 6. Commands (valid only after the v3 freeze and its preflight pass, Blocked 1)

**Freeze** (Worker, after (A)):
```
cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && R=data/SRP1/Results/P515S56/phase2x_ev_v3; mkdir "$R" && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_search_driver.py --phase-freeze --root "$R" --phase-id s56phase2x_ev_v3 --oracle harness --predictions data/SRP1/Results/P515S56/phase2x_ev_v2_predictions.json --unlaunched-predecessor data/SRP1/Results/P515S56/phase2x_ev_v1@de4e947e --continuation-of data/SRP1/Results/P515S56/phase2x_ev_v2@18a224d1 > "$R/freeze_stdout.log" 2> "$R/freeze_stderr.log"
```
- The predictions file is v2's, unchanged (sha256 `5e660820d83c…`).
- v1 must be declared again: its spec carries the same keys.

**Preflight-only** (Worker):
```
cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && NLP_SOLVER_PATH=/usr/local/bin/ipopt /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_search_driver.py --phase-run --root data/SRP1/Results/P515S56/phase2x_ev_v3 --preflight-only --out data/SRP1/Results/P515S56/phase2x_ev_v3_preflight
```
Run the resume check after it, against the real root:
`p515_s56_phase2x_v3_continuation_check.py --part resume --v3-root data/SRP1/Results/P515S56/phase2x_ev_v3 --remap data/SRP1/Results/P515S56/phase2x_ev_v3_remap --out <new dir>`.

**Author loop** (Terminal, Claude app closed). The first line will be `launch 001`, and each launch is one settled block
of about 2.5–3.1 h:
```
cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && caffeinate -i zsh -c 'R=data/SRP1/Results/P515S56/phase2x_ev_v3; i=1; while [ -e "$R/launch_$(printf %03d $i)_stdout.log" ]; do i=$((i+1)); done; while true; do n=$(printf %03d $i); NLP_SOLVER_PATH=/usr/local/bin/ipopt /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_search_driver.py --phase-run --root "$R" --take-over-stale-lock > "$R/launch_${n}_stdout.log" 2> "$R/launch_${n}_stderr.log"; rc=$?; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launch $n exit $rc"; if [ $rc -ne 4 ]; then echo "loop stopped: exit $rc (0 complete, 5 review, 6 ceiling, other fault)"; break; fi; i=$((i+1)); done'
```
Exit codes are as for v2: 4 more work, 0 complete, 5 review, 6 ceiling, 2 refused, 1 crash.

**Check script** (optional):
`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python s56_phase_check.py data/SRP1/Results/P515S56/phase2x_ev_v3 --launch 1`.
- Its guard criterion now reads the corrected guard.
- Its wall criterion (0.7–1.6 h) was set for exit-depth launch 1. v2's settled children ran 2.6–2.7 h, so it will print
  **FAIL on wall for every settled evaluation by construction**. Read the other criteria only.
- None of these command lines contains the harness hazard string.

## Not confirmed

- **No real v3 spec, preflight or resume check exists.** The resume point is shown on a simulated freeze only. Its spec
  equals what the real freeze would build under (A), but `git_head` and timestamps differ.
- **The preflight's environment checks** (memory, harness preconditions, `NLP_SOLVER_PATH`, files clean in git) have not
  run for v3. The v2 preflight passed with a 0.28 GiB memory margin, measured with the Claude app open.
- **Per-event cost beyond tier 1.** It is 1 + [tier2_attempted] from the code. The real records contain tier-1 events
  only, so tier 2 is tested synthetically (E3), not on a real record.
- **Records outside the pilot and v2 roots** were checked only for "old pass → new pass" on the guard value (104
  records under `P515S56`), not at the summary level.
- **The phase tests (T1–T16) and the driver gate were not re-run.** The change is in `map_record` and
  `read_eval_files`, which stub phases do not call. The oracle suite (119/119), which maps real smoke records, was
  re-run.
- **A process I did not start:** a 2-day-old `caffeinate -ims` (pid 90407). It does not contain the hazard string, and
  I left it alone.
