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

## Integration (Worker, 2026-11-23): oracle mapping, cap in the key, cache roots, preflight, `--pilot`

**Authority:** `S56_SEARCH_DRIVER_DESIGN.md`, "Oracle mapping and key rules" (`acc80a2e`); the Planner's integration task.
Engineering build: one test script with pass/fail checks, no prediction spec. **Not launched:** the pilot and any search.

**Solve claim:** the zero-solve suite and every driver subprocess run with an armed `SolveProfileGuard(permitted=())`,
verified at 0; pickle load / loads blocked, 0. **One real evaluation** ran (the 3-cycle check): parent guard at 0, child
solves 204 = the declared ((1 + 3)·12 + 3)·(3 + 1), nothing blocked, identity held.

**Objective convention (every EUR figure):** F = I(x) + gross_operational_cost − TV credit, EUR of total discounted
planning cost; gross is settlement-excluded (contracted part and pin excluded; at one scenario that is the whole
settlement). Cost-breakdown schema v2.

### Blocked on Planner

These do not block the build. They block a **real** pilot or search: the real freeze and the real pilot freeze refuse
until items 1 and 3 are confirmed (refusals tested, T5).

1. **Two gate thresholds are not named anywhere: τ_cl and the ESSO slack gate.** Searched: `S56_PLAN_v4_1.md`,
   `S56_PLAN_v4.md`, `S56_DECOMPOSITION_NOTE.md`, `S56_SEARCH_DRIVER_DESIGN.md`, `P5_15_S56_EXPERT_HANDBACK*.md`,
   `model_construction_helpers.py`, `p515_s56_cohort_followup_tests.py`, the harness.
   - **Provisional values in the code:** both 1e-5, anchored to `definitions.EQUALITY_TOLERANCE`. That is the band
     production imposes on the closure when its slack is not built (`sess_soc_final_rule`).
     - τ_cl applies in p.u. of the block base, so 1e-3 MWh on base 100.
     - The ESSO slack gate is |s_up − s_down| ≤ 1e-5 MW / MVAr.
   - **Options:** confirm these values, or name others. `MAPPING_PARAMETERS['provisional']` then empties.
2. **s_share is not named in the design:** S's share of R/4, which the K_EV rule needs ("the smallest K at which S
   meets its share"). `--pilot-evaluate` refuses without `--s-share-eur`.
3. **The initialisations.** The harness has no initialisation variant as such.
   - **Smallest deterministic variation (proposed):** the existing, keyed harness option `rho0_scale` (W183a: ρ0 × s on
     every TSO / DSO channel and the ESSO ρ). It needs no production option.
   - **Proposed values:** s ∈ {1.0, 0.5, 2.0}. 1.0 is the default run.
   - **The staged plan** is `p515_s56_ev_harness_tests.STAGED[grid]`, the memory probe's mixed-duration plan.
   - Confirm or change both. The pilot spec records them as `PROPOSED`.
4. **Readings I took inside the decisions.** Each is stated in the driver's mapping comment; change any you disagree
   with.
   - **(a) "Would have certified each of the last W cycles"** means production's per-cycle test `cycle_convergence`
     (= boyd_all_pass and local_solves_ok, `shared_resources_planning.py:3469`) holds on each of them. With W = 10, the
     certificate length, that is "would have certified at K". The stronger reading, a complete certificate at each of
     the last W cycles (19 consecutive cycles), is not used.
   - **(b) PR.** Sign per ruling 3: PR = −price + dual.
     - The interface part uses the **TSO-side** multipliers × (TSO − DSO copy). This is the step 4 evaluator's form and
       production's `priced_residual_pr` conversion.
     - The ESS part is each of the three agents' own augmented-Lagrangian term, σ·λ_i·(x_i − z)/(2 S_i)
       (`_update_shared_energy_storage_variables`).
     - σ is the common objective scale: one scaled-objective unit of any agent = σ EUR. For the ESSO this follows from
       κ = a/σ.
   - **(c) G.**
     - The ESSO slack is priced σ·|λ_esso|·|s_up − s_down|/(2 S_esso).
     - The closure is priced σ·|row dual|·|s_up − s_down| (p.u.).
     - **Deactivated closure rows are excluded and counted:** 48 of 72 in the real check.
     - A closure row with a non-zero slack and no dual makes the summary incomplete (harness_error).
   - **(d) Pilot rules.**
     - K_EV must be ≥ every run's would-be certification cycle.
     - e_EV's pair set is the 9 x0/unit pairs plus the 6 staged pairs.
     - ψ = block makespan / mean t_exit of the block, max over blocks of ≥ 2 runs.
     - The pilot runs in concurrent blocks, so its t_exit already carries contention. κ = 1.18 in the projection may
       then double-count it.
5. **Policy gap (outside this task; not changed).** `P.confirm` maps a `not_settled` settled summary to
   `harness_error`, which **stops the search**. The design says "otherwise the class is uncertified". The mapping emits
   `not_settled`.
   - **Options:** (a) leave the policy as is: an unsettled confirmation stops for review; (b) a policy change: unsettled
     → "unresolved after confirmation", listed, the search continues.
6. **The driver gate's item 4 control now fails as written.** It asserted that the 2× configuration is **refused** (the
   smoke alias). The cap rule removes that alias (T3, T9). The gate was not re-run. Re-scope that control if the gate is
   re-run.
7. **A real launch needs `memory_per_slot_gib` declared** in the harness config or pilot spec, from the memory probe
   (`7974096f`: five-block exceeds 20.7 GiB; 2025/2030/2035 by the rule, awaiting the author). I did not read the
   probe's figures.

### Changed

| commit | content |
|---|---|
| `8948db26` | Harness **+328 lines, insertions only** (vs `acc80a2e`); driver integration; `p515_s56_search_oracle_tests.py` (new) |
| `370b7689` | Exit-view label fix (below); the tests re-map the real record |
| (this commit) | Evidence `data/SRP1/Results/P515S56/search_oracle_v1/` + manifest + this section |

**Harness (`p515_s44_campaign_harness.py`):**
- **Cap in the eval key (option (a)).** For every entry that declares an S56 EV-plan key, the eval key is wrapped
  **last** as sha256{base, `s56_cycle_cap`: cap}. The entry records `s56_cycle_cap_keyed`. For S56 entries this
  supersedes the W98 comment at `S56_SETTLED_STOP_DISABLED_KEY`; that comment is left in place, insertions only.
  Entries without S56 keys are not wrapped.
- **`s56_pricing_capture` (opt-in; allowed only on S56 EV-plan entries without the coupling).**
  - True is keyed (sha256{base, key}). False is recorded, not keyed. Absent changes nothing.
  - The capture path is asserted before any solve: 29 production-source checks.
  - Post-run, zero solves, it writes `s56_pricing_terminal.json` from production's returned terminal state
    (`consensus_vars` / `dual_vars` deep copies) and the run's terminal models:
    - σ;
    - the interface copies and multipliers;
    - the three ESS agents' x, λ, z and ratings;
    - the ESSO slacks;
    - the closure rows with their dual;
    - the settlement blocks;
    - the efficiencies.
  - A failed write exits the child with code 2.
- **The per-cycle TV credit needed no capture.** Production's trajectory row already carries `terminal_salvage_value`
  and `recourse` (= net) every cycle (`shared_resources_planning.py:3386–3388, 3589–3590`).

**Driver (`p515_s56_search_driver.py`):**
- **`map_record` / `HarnessOracle.summary_from_dir`:**
  - the mapping exactly as in the design;
  - `assert_mapping_capture_paths` runs at oracle construction, before any run;
  - every missing file or field gives `harness_error` with its reason.
- **`compose_eval_key`:** one composition for the search and the pilot, including the pricing and cap wraps.
- **Cache roots and start pool:**
  - `build_cache_index` (freeze) indexes only the records whose eval key is this search's key at a depth. Others are
    listed with the reason.
  - `load_cache` re-verifies the hash and the key and maps the records at every launch.
  - `best_start` returns the certified pool member with the lowest exit F, with the runner-up margin against e.
  - CLI: `--cache-root`, `--start best_of_pool`, `--start-pool`.
- **`real_launch_preflight`:**
  - the mapping parameters are final;
  - `check_campaign_preconditions` (live campaign children, legacy lock, write-once root, production / harness /
    driver files clean);
  - `NLP_SOLVER_PATH` = `/usr/local/bin/ipopt`;
  - the memory rule: E1's vm_stat rule, re-implemented because importing the E1 launcher would chdir and install a
    guard; the per-slot GiB is declared in the spec.
  - It runs before a launch dir is created.
- **`project_wall_time`** (T = κ[ψ·t_exit·Σblocks + t_set·batches]), **`count_projection`** (the design's N_search /
  E_j formula), `projection_from_state`. CLI `--project`.
- **`--pilot-freeze / --pilot-run / --pilot-evaluate`:**
  - 7 evaluations at 2×, `s56_settled_stop_disabled` + `s56_pricing_capture`, cap K_pilot = 300.
  - Blocks of 5 + 2, one attached launch per block, hash-chained launch records. Exit 4 while blocks remain.
  - `--grid srp1grid | five_block`.
  - `pilot_outputs` computes:
    - single offsets and their sign;
    - 15 pair offsets;
    - the dF_exit / dF_set spreads, with V_pair beside;
    - the bars, t_exit, t_set, ψ;
    - the leak at the settled point;
    - K_EV and e_EV by the frozen rule.
- **Settlement derivation (task item 1, PR)**, in `SETTLEMENT_DERIVATION`:
  - **The EV record's Q:** `gross_including_settlement − contracted settlement − pin` (`shared_resources_planning.py:1305–1307`).
  - **The in-full form applies only when the coupling is wired** (`:1316–1320`, `:4899–4900`), and it is never wired
    at one scenario (`:4406`).
  - **At one scenario the deviation part is 0 and contracted = total** (`:1274–1275`, `:826–827`).
  - **So the EV Q contains no settlement price part.** price = −Σ w_b·settlement_total.
  - Checked on every settled record: deviation = 0, contracted = total, and the blocks reconcile with
    `interface_settlement_total`. All three held on the real check.

### Found

**Zero-solve suite, from committed code `370b7689`: 103/103 pass.** Attached, both streams captured, 57 s, stderr
empty.

| item | result | evidence |
|---|---|---|
| T1 mapping on real records | **pass** (23) | Smoke x0 / unit: F = I + recourse(2) exactly = breakdown total (unit **873,899,483.26**, x0 873,896,422.55 — the harness report's values); recourse = gross − TV every cycle; bar_F = the record's net step (rel 1e-12) ≤ triangle bound; exit code / status error / missing files / no I(x) → harness_error; guards broken → not_certified. The real 3-cycle record re-mapped (exit read at terminal cycle 3; settled complete, not_settled) |
| T2 settled arithmetic | **pass** (21) | Real smoke files + a **synthetic** pricing fixture: PR, A, G = an independent hand computation (rel 1e-12); the PR sign; S; each settled condition alone → not_settled (named); a record short of K → harness_error; gate boundaries (dP = 12.5 kW passes, 12.6 fails; ESSO slack; τ_cl; D1); leak 25.01 € flagged, not gated; a closure row without dual, a non-zero deviation part → harness_error; `P.confirm` consumes the summaries |
| T3 keys | **pass** (11) | Composition = `freeze_campaign_spec` (3 points × 2 depths × caps 2/3/300/500), 24 distinct keys; settled K distinct; **alias gone:** the smoke's keys are hit by no 2× key at caps 3/300/500; **the freeze-time collision scan of the 2× configuration (74,084 keys, 308 files) finds nothing** — it named the smoke spec before; pricing key refusals (no S56 key, coupling, non-bool), True keyed, False not keyed |
| T4 option-off identity | **pass** (11) | Harness diff vs `acc80a2e`: 328 added, **0 deleted**. **308 committed specs, 7,293 entries: candidate keys reproduce; the 7,291 entries without S56 keys: full eval key post = pre on every one, and the 7,270 recorded eval keys all reproduce (0 mismatches).** The 2 entries with S56 keys (the smoke's x0 / unit) reproduce under pre and change under the cap rule, by design. Freeze pre = post byte-identical on 4 entry sets (SRP1 plain incl. ρ0 scale; pinned-capture AA; SP derived coupled + calibration key; SP derived plain); `build_evaluation_record` pre = post on 12 committed records |
| T5 preflight | **pass** (13) | Memory rule (pass / refuse / equality edge, undeclared per-slot, missing vm_stat line); `real_launch_preflight` refuses on the provisional mapping and on `NLP_SOLVER_PATH`, never creates the launch dir; the real freeze refuses (provisional; no `memory_per_slot_gib`); the real oracle refuses a settled depth without the pricing key; the mapping capture path passes on the EV configuration; the real pilot freeze refuses (provisional); unknown grid, stub pilot not test-only refused |
| T6 cache / start pool | **pass** (5) | **Synthetic** fixture records: 3 indexed; foreign key and the pre-rule smoke copy ignored with reason; mapped F = fixture F; best start = lowest certified exit F, runner-up margin 40,000 € vs e reported; a member without a record refused; a record changed after the freeze refuses |
| T7 pilot vs stubs | **pass** (11) | Freeze (SRP1 grid): 7 entries, 5 + 2, distinct keys, all stop-disabled + pricing at 300; three attached launches → exit **4, 0, 0**; hash-chained records, children exit 0; evaluate without s_share refuses; with a test s_share 500 € (**synthetic** traces): every output written, e_EV re-derived by the rule, K_EV meets its rule; unreachable share / no would-be certification → failure named, no value; five-block freeze: keys distinct from the SRP1 grid |
| T8 projection | **pass** (4) | Worked count example N 37 / blocks 7 / batches 6; T = 36.108 h on that example; unmeasured t_exit refused |
| T9 2× stub search | **pass** (2) | The 2× EV stub search freezes (it was refused before) and runs to termination by attached launches; final state = in-process replay |

**Real check:** `--real-check`, from committed code `8948db26`; 12/12 pass, attached, stderr empty.
- **Setup:** the reference unit, SRP1-grid EV twin, 2×, `s56_settled_stop_disabled` + `s56_pricing_capture`, cap 3.
- **Keys:** eval key `ae319309…`, candidate `0998d021…`. Spec `campaign_spec_s56orc_v1_real_check_6781f85a.json`.

| quantity | value |
|---|---|
| child exit / cycles / record status | 0 / 3 / not_certified |
| solves | 204 = declared; blocked 0; identity holds |
| stop disabling | length 10 → 10⁹ → restored 10; summary ok |
| pricing capture | written (632 kB): 36 interface, 36 ESS, 36 ESSO-slack blocks, 72 closure rows (24 with dual, 48 inactive) |
| F at K = 3 | **828,457,805.41** (I 317,957.01) = breakdown total |
| S (3-cycle window; K < W) | 936,978,515.20 |
| PR = −price + dual | −11,810,480.31 = −16,714,546.03 + 4,904,065.73 (interface P +4,920,672.90, Q −14,225.14, V −3,334.17; ESS P +955.23, Q −3.09) |
| A | 25,246,375.18 |
| G | 2,046.31 (closure 2,038.24; leak 8.07; ESSO slack 9.9e-12) |
| gates | **gates_pass False**: interface max \|dP\| 32.05 MW > 12.5 kW; ESSO slack, τ_cl (1.57e-8 MWh), D1 (η 0.97/0.96), status pass |
| settled | **not_settled**: K < W, last W not at tail depth, not certified on the last W |
| leak λL (loss) | 8.07 € (full-L 112.55 €), within 25 € — at 3 cycles, unconverged: a measurement, not a result |
| wall / peak RSS | 199.5 s child / 2.43 GiB (ru_maxrss) |

- **Cycles 1–2 are bitwise equal to the committed smoke unit** (gross and recourse float.hex). The settled key and
  the pricing capture leave the computation unchanged in those cycles.
- **Everything in the table is a 3-cycle unconverged point.** It shows that the mapping is complete. It is not an
  estimate.

**Unexpected findings:**
- **The real check found a label defect.** In a stop-disabled run without a would-be certification cycle, the exit
  view said "would-be certification cycle" although it read the terminal cycle. The status and value were right.
  - Fixed in `370b7689`.
  - `real_check.json` (written by `8948db26`) carries the old label. The suite's T1 re-maps the same record with the
    fix.
- **In the 3-cycle check the price part (−16.7 M€) dominates PR and A.** That is expected far from consensus, and it
  is why PR only withholds determinacy.

**Evidence** (`data/SRP1/Results/P515S56/search_oracle_v1/`):
- `zero_solve/zero_solve.json` `f118f2f3…`;
- `real_check/real_check.json` `252b2354…`;
- `manifest_sha256.json` `9e7904d7…`: 484 files hashed, including the uncommitted test trees;
- launch logs: `zero_solve_stdout.log` `06a81c5a…`, `real_check_stdout.log` `74fe7dff…`; stderr logs empty
  (`e3b0c442…`).

**Committed:**
- the JSONs, the manifest and the logs;
- the real-check campaign spec, heartbeat and init-identity record;
- its eval dir, without the ESSO pickle, `results/` (xlsx), `planning_read/` and `esso_capture/`, as for the smoke.

**Not committed, hash-recorded only:** `zero_solve/test_trees/`. It holds frozen campaign and pilot specs with
EV-configuration eval keys, which a later collision scan would hit.

### Not confirmed

- **No settled (K ≥ W, tail-depth) record has been mapped.** The settled branch is tested on a synthetic fixture (T2)
  and on the 3-cycle real record (not_settled). Searched: no committed EV record with the pricing capture beyond this
  check.
- **Every committed or new EV record has TV = 0.** The 2025 cohort has no remaining life, so the per-cycle TV term of F
  is exercised only at zero.
- **No would-be certification cycle has occurred in a real record.** The pilot's exit-view path (c* found) is tested
  on synthetic traces only.
- **The pricing conversion factors are derived from production source and asserted.** They were not compared against
  an independent dual-based estimate, which needs a converged run.
- **No real cache root exists.** T6 uses fixtures, and the screen has not run.
- **The memory preflight's per-slot figure is not taken from the memory probe** (Blocked 7).
- **Gate F (5 concurrent children, bitwise) was not re-run.** The driver gate was not re-run (Blocked 6).
