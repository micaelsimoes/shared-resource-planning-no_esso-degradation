# P5.15 S56 critical path step 2: the EV search driver and its zero-solve determinism gate (Worker report, 2026-11-23)

**Authority:** `S56_SEARCH_DRIVER_DESIGN.md` (`2c15db47`), implemented as written; TASKS.md "Working mode and EV
priority" (critical path step 2) and "Search policy review (O1–O15)"; `S56_PLAN_v4_1.md` §3–4. This is an engineering
build: a test script with pass/fail checks, no prediction spec.

**Solve claim:** zero solves, enforced by an armed `SolveProfileGuard(permitted=())`.
- The guard is installed before any repository import in the gate and in every driver process. Each is verified at
  exactly 0.
- `pickle.load` / `pickle.loads` are blocked in the gate. The count is 0.
- No real evaluation, no search launch and no pilot were run.

**Values:** every objective value in the gate is **synthetic**: a hashed bowl landscape, attached by the driver to
synthetic or stub records. The exception is item 9, which reads the committed records of the published searches.
- No figure in this report is an evaluation of the system.

## Blocked on Planner

These items do not block this task. They block **using the driver with real evaluations** (the pilot and the searches).

1. **The real-record mapping is not built.** The `harness` oracle is refused in this build. Before any real launch you
   need to name four things:
   - **F.** The harness report (`b0391b80`) defines `total` = I(x) + net, with net = gross − TV credit.
   - **The bar to use with that F.** The record's `bar` is the maximum *gross*-cost step over the last 10 cycles. With
     TV in the objective it is not the step of `total`.
   - **The settled-run V components.** S, PR, G and A of plan v4.1 §3: which record field supplies each.
   - **The gate set behind `gates_pass`, and the status rule for a stop-disabled run.** A run to K_EV has no
     certificate, so "settled" needs a definition.

   The policy consumes only the summaries {status, F, bar} at exit and {status, F, V, gates_pass} at settled depth. The
   mapping is one driver function, `HarnessOracle.summary_from_dir`.
2. **The cap enters no eval key (the W98 rule), and this aliases a committed artefact today.** The harness smoke
   (`ev_harness_v1/smoke`, cap 2) evaluated x0 and the reference unit at 2× under the same configuration the EV 2×
   search would use, so its committed spec carries those exit keys.
   - The driver's freeze-time collision scan **refuses** that configuration and names the smoke spec. This is a gate
     check (item 4 control).
   - The same rule makes settled runs at different K_EV share keys across roots.
   - **Options:**
     - **(a) Key the cap on S56 EV-plan entries.** This is a harness change and separates exit, smoke and K_EV runs.
       Recommended.
     - **(b) Declare the smoke root an excluded, never-read alias.** This needs a driver option that does not exist yet.
   - Your TASKS decision (ii) (`28ba36b4`) keeps AA as the case file on EV, so no configuration difference removes
     the alias. One of (a) or (b) is needed before the 2× EV search is frozen.
3. **For information: the harness key I added.** The harness Worker did not deliver the keyed stop-disabled option
   (their Blocked item 2). Your TASKS decision (iv) assigns it to the driver Worker. I added `s56_settled_stop_disabled`
   (below), which enters the eval key. If you prefer another form, for example keying the existing calibration key on
   EV entries, say so before the pilot.

## Changed

| commit | content |
|---|---|
| `dcea9628` | `p515_s56_search_policy.py`, `p515_s56_search_driver.py`, `p515_s56_search_gate.py` (new); `p515_s44_campaign_harness.py` +64 lines, insertions only |
| (this commit) | gate evidence `data/SRP1/Results/P515S56/search_driver_v1/`, dry run `search_driver_v1_dry_run/`, this report |

**`p515_s56_search_policy.py`** (standard library only; every decision lives here):
- `order`: the O1–O15 key (−c(d), class rank, changed cohorts, label).
  - d = z − z_inc in granules.
  - c(d) = sign(d·m)(d·m)²/|d|², an exact `Fraction`, where m is the last accepted move.
  - With no last move, c = 0 for every point.
  - The key is asserted strictly total. More than 80 points is refused.
- `classify_exit`, `confirm`, `step`, `run_to_end`, `terminal_report`, `count_identity`, `reclassify`.

**`p515_s56_search_driver.py`:**
- **Modes:** `--freeze`, `--dry-run`, and `--run` (one launch = one action of `step`). The exit codes are 0 terminated,
  5 terminated for review, 4 more work, 2 refused.
- **Launch records:** write-once, hash-chained, each with a manifest.
- **Replay on every launch:** each action and state is re-derived, every record is re-read from its eval dir, and every
  eval key is recomputed. Any mismatch refuses the launch.
- **Derived view:** `search_state.json` is checked against the replay.
- **Interruptions:** an interrupted launch is recorded in `interrupted_<n>.json`, with hours, and moved aside, never
  deleted. Its evaluations are uncounted.
- **Stale lock:** takeover only with `--take-over-stale-lock`, only when the pid is dead and the lock names the search.
  The old lock is moved to `stale_locks/`.
- **Oracles:** `synthetic` (in process) and `stub_children` (one frozen harness spec per launch through `H.evaluate`;
  every frozen key recomputed). `harness` (real) is refused.
- **Freeze-time collision scan:** 74,084 keys (the domain at exit and settled depth) against the tracked campaign and
  search specs and launch records. The search's own root and its declared cache roots are excluded, and every refusal
  names the file.

**Harness key `s56_settled_stop_disabled`** (the design's "Harness requirement"):
- **What True does:** it installs the existing stop-disabling hooks (`assert_s56_certification_stop_path`,
  `s56_certification_stop_disabled_hooks`), so the run goes to the spec cap.
- **Where it is allowed:** only on entries declaring an S56 EV-plan key, without the coupling, without the calibration
  key, and under an enabled tail.
- **Keying:** True wraps the key as sha256{base_evaluation_key, key}. False is recorded but not keyed (the W33 rule).
- **Absent:** nothing changes.

**Decisions taken inside the design** (logged here; say if any should change):
- **The "node, year" key** is the tuple of changed cohorts in lattice order. Cosines are taken in granule space.
- **Move classes:**
  - R3 coordinate steps and the diagonals → resize;
  - R3 activation → activate;
  - R3 deactivation and class d → remove;
  - shifts → shift;
  - class f → redistribute.
- **Statuses:**
  - `harness_error`: status `harness_error`, an unknown status, or a certified summary with a missing or non-finite F
    or bar. A **missing bar stops** the search.
  - `uncertified`: `not_certified`.
- **Confirmations** run one candidate at a time, in order. One action is the candidate's settled run plus the
  incumbent's, if not stored, so at most 2 evaluations, each counted 1.
- **Held-out check:** an offset |dF_exit − dF_set| > e (the pair's exit e) stops the search. It is checked before
  acceptance.
- **Budget:**
  - an exit block is truncated to the remaining budget;
  - a confirmation that does not fit ends the search (`budget_exhausted`), and the candidate is listed as
    `confirm_candidate_unconfirmed`;
  - evaluations after a stop are tagged `speculative_post_termination`.
- **An interrupted launch never counts,** including the children that finished before the kill. The whole action
  re-issues.
- **One harness campaign root per launch.** The alternative, one spec holding the domain × 2 depths (74,084 entries), is
  of order 100 MB.

## Found

**The run.** One attached run of items 1–9 from committed code `dcea9628`, with both streams captured: 1,260 s. Guard
counts are all 0, and the pickle counts are 0.
- **Result: 143/143 checks pass.**
- A scratch rehearsal from the same code (except the count-identity reporting refinement) gave the same 143/143 and the
  same figures.
- **The run waited** for the five-block memory probe (critical path step 3) to finish. The probe's watchdog and
  evidence include system-wide swap and compressor counters, which concurrent load could perturb.

**Evidence:**
- `data/SRP1/Results/P515S56/search_driver_v1/search_gate.json`, sha256 `c3445d22…`.
- `manifest_sha256.json`, sha256 `3951318f…`. It hashes all 7,099 files of the run (80 MB) and the code files:
  policy `b37601c8`, driver `617ad8a7`, gate `dced0ee1`, harness `63257db1`.
- **Launch logs**, hashes recorded here because they were written during the run: `gate_launch_stdout.log`
  `11d60e47…`; `gate_launch_stderr.log` is empty (`e3b0c442…`).
- **Committed:** the gate JSON, the manifest, the launch logs and the three item 2 worker results.
- **Not committed, hash-recorded only:** the test trees under `item3_driver/`, `item4/`, `item5*/` and `item6_*/`.
  - They hold frozen campaign and search specs whose eval keys are those of the EV configurations.
  - Committing them would make the collision scan of a later real search hit them.

| item | result | evidence |
|---|---|---|
| 1. N(x) | **pass** (11 checks) | Domain re-enumerated by the lattice rule: 37,042 = the committed diagnostic-1 table (hash-checked). Driver \|N\| = table \|N\| on every plan; min / median / max 3 / 39 / 75 ≤ 80. 1,655 plans have \|N\| > 60 and are ordered. Named points 18 / 22 / 13 = `s56_neighbourhood` = committed keys. Every diagonal same-sign. 81 points refused. |
| 2. order determinism | **pass** (4) | Three fresh processes (PYTHONHASHSEED 0 / 4242 / random, input shuffled) and the in-process run. Each covers 37,042 plans × 3 orders: no last move, a last move, the reversed class rank. Digest identical: `86de1078…`. |
| 3. order vs values | **pass** (2) | 43 plans: the first-poll order is unchanged with every neighbour value cached and perturbed (2 landscapes), and at 1× vs 2×. Through the driver, the first two launch actions are identical at 1×, at 2× and under a perturbed oracle. |
| 4. stub-children replay | **pass** (8) | x0 at 1×, N_max 40, harness stub children with per-child random sleeps of 0.5–1.5 s. Serial and 5-way give the same history, evaluated set, budget (23) and full state as the in-process replay. Observed concurrency 1 and 5. **Control:** the 2× configuration is refused by the collision scan, which names the smoke spec (Blocked 2). |
| 5. kill and resume | **pass** (36) | Synthetic oracle, reference-unit start, 22 launches, budget 59. A kill at every transition (after lock, after launch dir, during evaluate, after evaluate, after launch record, after manifest, after state view, after results), at launches 1 / 2 / 3 / 22 where the transition exists (27 kills). After resume the state is byte-equal, the budget equal, killed evaluations uncounted (0–5 per kill), and the stale lock taken over. **Stub children:** the driver and its 5 (launch 2) or 2 (launch 3) running children killed by process group during `evaluate`; byte-equal after resume. **Refusals:** a live lock, a lock of another search, a stale lock without the flag, a tampered record (exit 2, nothing written) and a tampered view; a synthetic spec without `--test-only`. |
| 6. keys | **pass** (28) | 37,042 distinct `GeneralLattice.key`; labels distinct; the v2 round trip is exact. Staged key (harness) = `GeneralLattice.key(z)` on all plans, EV twin and draw-0 case. **1,778,016 distinct eval keys** over {1×, 2×} × {exit, settled} × {EV, draw-0} × 3 cohort formulations × TV on/off (48 configurations × 37,042). The driver's composition = what `freeze_campaign_spec` freezes (3 points × 48). The collision scan excludes its own root and cache roots, reports a foreign hit and names it; 308 committed specs and records scanned, none carries a lattice key. Settled key: 5 refusals, keyed on EV and draw-0, False not keyed. **Option-off identity vs `b0391b80`:** insertions only (64 lines); frozen specs byte-identical on 4 entry sets (SRP1 single-year, S56 EV staged, S56 draw-0 staged, coupled calibration key), volatile fields excluded. |
| 7. boundaries | **pass** (28) | dF = R + e exactly → worse; dF = −R + e exactly → confirm-candidate; one ulp inside → unresolved. e floor 15,000 vs bar sum. Missing bar (None / absent / NaN) → harness_error. Uncertified is never worse. Strict status. Mixed depth refused (4 forms). Confirmation boundaries hold: improvement = R, V_pair = R/4 and A = R/2 accept; offset = e does not stop, offset > e stops. Re-classification at e = 20,000 changes exactly one class. |
| 8. arithmetic, budget | **pass** (18) | Three starts, every N_max from 0 to the unlimited count (56 / 68 / 16). The partition equals \|N_final\|; roles = budget = evaluated set; the budget is never exceeded; each limited run is a prefix of the unlimited one; the count identity holds. Exercised: mid-poll exhaustion, truncated blocks, exhaustion before a block, and a confirmation that does not fit. Stops: a harness_error at position 0 (4 speculative), an uncertified start, and an offset > e. |
| 9. published histories | **pass** (6) | s47 Phase B 1×, s51 F2 Phase B 2× and s53 F2 certificate 2×, all committed and clean. I(x) of every incumbent is recomputed on the 3 × 3 lattice to 0.00 €. **Coverage of N(x) by the old caches:** x0 6/18; 2025 n7 0.75/3 4/10; the 2030 plan 5/13 (s51) and 7/13 (s53). **No history has full coverage**, so no full replay is possible. |

**Unexpected findings:**
- **`s56_neighbourhood(...)['points']` also lists the informational class `e_opp`.** N(x) is only the CLASS_ORDER
  points (`size` excludes `e_opp`).
  - The committed diagnostic-1 named-point lists carry them too: 2 on the 2030 plan.
  - The driver filters by class and asserts the count equals `size`. A consumer reading `points` directly over-counts.
- **Item 9, read under the new rule.** The old values are single-scenario draw 0 under the old formulation, so this is
  illustration only.
  - **s51 start:** the first confirm-candidate among the covered points is the shift n7 0.75/3 2025 → 2030, with
    dF_exit −41,458 €. That is the published move's direction.
  - **s53 (the 2030 plan):** n5 2030 0.25/1.0 + n7 2030 1/3 is a confirm-candidate at dF_exit −6,571 € (≤ −R + e =
    +5,000). The new policy would spend a confirmation there.
- **Under the interim e rule, −R + e ≥ +5,000 €.** Exit-"worse" candidates of up to 5 k€ are confirmed.
  - In the synthetic dry runs, confirmations were 11–13 of 64–76 evaluations.
  - The landscape is synthetic, so this is not a prediction.
- **`H.evaluate` polls children every 5 s (`POLL_S`).** A stub launch takes at least 5 s. This is negligible for real
  runs.

**Dry run** (`--dry-run`, synthetic oracle, exit 0, guard verified at 0): `search_driver_v1_dry_run/search_dry_run.json`
(`8c66442c…`) with its manifest (`c2210140…`). Launch log `search_driver_v1_dry_run_launch_stdout.log` (`350a399e…`);
the stderr log is empty.

| start | \|N\| | evaluations | actions | moves |
|---|---|---|---|---|
| x0 | 18 | 64 | 22 | 6 |
| reference unit | 22 | 76 | 26 | 8 |
| 2030 plan | 13 | 14 | 4 | 0 |

All three runs terminate as `complete_poll_no_accepted_candidate`.

## Not confirmed

- **No real harness record has been mapped or classified** (Blocked 1). Searched: none exists for the EV staged
  configuration beyond the 2-cycle smoke.
- **The settled-run key's child path has not run.** Only freeze, keys and the option-off identity were exercised. The
  child wiring reuses the calibration hooks unchanged and is guarded by `entry.get(key) is True`. NL identity was not
  re-run: no model-build line changed.
- **Gate F (5 concurrent children, bitwise) has not been re-run on the final harness.** The design requires it before
  the driver relies on concurrency. It solves, so it was not permitted here.
- **Not built:**
  - the cache loader for declared cache roots (the screen) and the start-pool rule: the driver takes a named start, and
    `cache_sources` is empty;
  - the memory preflight and `check_campaign_preconditions` for real launches;
  - the wall-time projection T ≈ κ·[…]: it needs the pilot's t_exit, t_set and ψ.
- **The hours of killed evaluations are lower bounds:** launch start to the last write in the eval dir.
- **Item 9** reports coverage and classes only. No published history covers N(x) completely; searched: s47, s51, s53,
  the only committed `campaign_results.json` files with a `poll_history`.
