# S56 EV search driver: design (Advisor review 2026-11-23; Planner decisions logged)

Authority: `TASKS.md`, sections "Working mode and EV priority" and "Search policy review (O1–O15)", and
`S56_PLAN_v4_1.md` §3, §4 and §7. This note fixes how the driver is built. It changes no acceptance rule and no
formulation.

## Modules

**Reused by import:**
- `p516_lattice.py` (lattice, `GeneralLattice.key`, I(x)), 201–474.
- `p515_s56_neighbourhood.py` (`s56_neighbourhood`, N(x)), 161–227.
- `p515_s44_campaign_harness.py`: `evaluate`, campaign lock, write-once eval dirs, heartbeat tags and stub children
  (≈ 2853–2998, 2562–2612, 503–507, 2843–2850, 3343–3348).
- The resumable-launch pattern of `p515_s55_e1_launcher.py`: `search_launch`, `verify_own_records`, the memory
  preflight.

**Not reused:**
- E1's Halton poll and best-of-poll acceptance;
- `PB.classify` / `PB.resolution` (the σ_Q form);
- `resolution_by_bound`.

**New:**
1. `p515_s56_search_policy.py`: pure and stdlib-only.
   - `order(lattice, z_inc, last_move, class_rank)`: O1–O15 order. The sort keys are strictly total; cosines are
     compared in exact integer arithmetic; the case with no last move has a defined rule.
   - `classify_exit(...)`, `confirm(...)`, `step(state, records) -> (state, action)`, `terminal_report(state)`.
   - Every decision lives here.
2. `p515_s56_search_driver.py`: freeze, dry run, `--run` (one launch = one block), resume.
3. `p515_s56_search_gate.py`: the zero-solve determinism gate (below).

## Execution

- **Fixed blocks of 5.** The driver takes the next 5 uncached candidates in order, calls `evaluate` once, waits for
  all 5, and classifies them in order. The incumbent path equals the serial path. No rolling refill.
- **Budget.** Every completed evaluation counts 1 against N_max. This includes candidates ordered after an accepted
  one; they are tagged `speculative_post_acceptance` and are cache hits, counting 0, later.
  - **Planner decision:** settled confirmations also count 1 each.
  - An evaluation killed by an interruption does not count; its hours are logged.
  - The budget is checked before each block.
- **State.** The truth is an append-only, hash-chained set of launch records and manifests.
  - `search_state.json` is a derived view. On resume it is rebuilt by replaying through `step()` and must match.
  - Launch records are committed at review points; the hash chain covers the rest.
- **Resume after a kill.**
  - An `interrupted_<n>.json` record is written, and the orphan eval dirs are renamed aside. Nothing is deleted.
  - An explicit stale-lock takeover is allowed only when the pid is dead and the lock names this campaign. The old
    lock is moved aside.
- **Never detached.** Each launch is one block, under 4 h, attached, with both streams captured.
  - It exits with code 4 when more work remains.
  - For unattended progress the author runs a foreground re-issue loop in his own terminal under `caffeinate -i`.
- **One search at a time per checkout.** Gate F (5 concurrent children, bitwise) is re-run on the final harness before
  the driver relies on it.
- **Exit classes** (they partition |N| exactly):
  - confirm-candidate, worse or unresolved;
  - uncertified: `status != 'certified'`. It is never accepted and never "worse"; it is listed;
  - harness_error: stops the search.
- **Cap 80** is owned by the driver. It ignores `COMPLETION_CAP = 60` and `over_cap` in `p516_lattice`.
  Diagonal moves are same-sign only.

## Classification and confirmation

- **Interim exit error (Planner decision, logged).** e = max(15,000 €, bar_x + bar_inc), with the bars from
  `evaluation_record.bar`. Exit values are compared with exit values only; a mixed-depth comparison is refused.
  - worse: dF_exit ≥ R + e;
  - confirm-candidate: dF_exit ≤ −R + e;
  - otherwise unresolved.

  R = 10,000 €. Basis: single-scenario SRP1 production-exit bars, 3.6–59.9 k€ (median 9.7 k€), and an x0
  exit-to-settled offset of 14,970.62 € (`P5_15_W187_REPORT.md`). The EV arm earns no credit for being
  single-scenario.
- **Confirmation trigger (Planner decision, logged).** Candidates in confirm-candidate class are confirmed in order,
  and the first accepted one becomes the incumbent. Acceptance itself stays plan v4.1 §3: F_set(inc) − F_set(x′) ≥ R,
  V_pair ≤ R/4, and the guards and gates pass. Otherwise the candidate is "unresolved after confirmation".
- **Settled run.** The tail is on and the certification stop is disabled, to a pre-declared cap K_EV. There is no cap
  extension inside a search.
  - **Harness requirement:** the stop-disabled option must enter the eval key and must be allowed on single-scenario
    EV entries. Today `s56_certification_stop_disabled` does neither. Without that, exit and settled records of one
    candidate would alias.
- **Re-classification.** Classification is a pure function of stored values. When step 5 measures e, the whole history
  is re-classified at zero solves, and candidates the measured e would have confirmed are confirmed then.
- **A7 barrier stop:** dropped (Planner). It is not in plan v4.1.
- **Expert items:** none, unless V_pair ≤ R/4 proves unreachable for EV, or acceptance on exit values is proposed.

## Pilot protocol (critical-path step 4)

**Runs.** Seven evaluations at 2×, all stop-disabled to K_pilot:
- x0 and the reference unit, each from three initialisations;
- one staged plan.

The exit value is read at the would-be certification cycle, so each run yields both its exit and its settled value.

**Frozen before the run:**
- e_EV = max(2 × the largest pair offset |dF_exit − dF_set|, the dF_exit spread, the bar sum);
- K_EV;
- the measured t_exit, t_set and ψ.

**Reported:**
- single offsets and their sign;
- pair offsets over the 3 × 3 initialisation combinations, and for the staged plan against both;
- the dF_set spread beside V_pair;
- the bars.

**In the search,** every confirmation is a held-out check: an offset above the e in force stops the search for review.

## Determinism gate (zero solves; guard armed)

1. N(x) equals `s56_neighbourhood` on the named points (|N| 18 / 22 / 13), with max ≤ 80 over all 37,042 plans.
2. The order is byte-identical across fresh processes, with different `PYTHONHASHSEED` values and shuffled input.
3. The order is unchanged when every cached value is perturbed, and identical at 1× and 2×.
4. A synthetic-oracle replay through stub children gives an identical incumbent history, evaluated set and budget
   count, serial and 5-way, with randomised delays.
5. A kill at every transition, followed by a resume, gives a history byte-equal to the uninterrupted run, with killed
   evaluations uncounted.
6. Keys:
   - 37,042 distinct keys per configuration;
   - keys pairwise distinct across {1×, 2×} × {exit, settled} × {EV, draw-0} × cohort formulation × TV;
   - the staged key equals `GeneralLattice.key(z)`;
   - the collision scan excludes the campaign's own root.
7. Classification boundaries: dF = R + e exactly, a missing bar, uncertified. Mixed-depth input is refused.
8. Terminal arithmetic, and budget exhaustion mid-poll.
9. Published histories are replayed where the old cache covers N(x), with coverage reported.

## Count and time

For each accepted move j = 1..M:
- n_j = |N_j| − cache hits;
- p_j = the position of the first confirmed improver;
- B = 5;
- E_j = min(n_j, B·⌈p_j/B⌉);
- the final poll has E_f = n_f.

The evaluation counts are:
- N_search = S_0 + Σ_j (E_j + c_j) + E_f + c_f, where c counts confirmations;
- N_screen = |nodes|·|years|·2 + L + Σ_cand (1 + n_cand).

The wall time is T ≈ κ·[ψ·t_exit·Σ blocks + t_set·(confirmation batches)], with κ = 1.18 measured by gate F.

## Top three false-result risks and their checks

1. **Key aliasing** (exit/settled, 1×/2×, EV/draw-0, old/staged schema). Checks: gate item 6; every served key is
   recomputed in the loop; the child reads the configuration back.
2. **Mixed depth, or the wrong incumbent.** Checks: depth tags are enforced in `classify_exit`; a zero-solve verifier
   re-derives every decision from the records.
3. **Failures read as "worse" or dropped from |N|** (the D0 precedent). Checks: an exact class partition, a strict
   `status == 'certified'` test, and a stop on harness_error.

## Oracle mapping and key rules (Planner decisions, 2026-11-23; after the driver build `dcea9628` / `1d3b6cdb`)

These definitions follow `S56_PLAN_v4_1.md` (D3/D4 as accepted) and `S56_DECOMPOSITION_NOTE.md` ("The R/4 terms").
They are implemented in one function, `HarnessOracle.summary_from_dir`.

**F.** F = `total` = I(x) + gross − TV credit, in € of total discounted planning cost. The cost-breakdown schema v2
convention is stated on every record.

**Exit bar.** bar_F = max |F(k) − F(k−1)| over the last 10 cycles, from the per-cycle F series.
- If the records do not hold the per-cycle TV credit, add a capture for it (an opt-in harness key, keyed only on S56
  EV-plan entries).
- Until that capture exists, use bar_F = bar_gross + the range of the TV credit over the last 10 cycles (a triangle
  bound). The record's gross-only `bar` alone is not used.

**Settled run.**
- Run with `s56_settled_stop_disabled` True, to the declared cap K, with the tail on.
- "Settled" means all of the following:
  - the run reached K;
  - every one of the last W = 10 cycles is at tail depth;
  - the production Boyd test would have certified each of those cycles ("would-have-certified");
  - guards are exact and there are no local-solve failures.
- Otherwise the class is uncertified.

**V components** (€ of total discounted planning cost):
- **S(x):** the range of F over the last W = 10 cycles. The validated-extrapolation gap is not used in the EV arm until
  the pilot validates it.
- **PR(x):** the signed priced consensus residual over all channels: the interface P, Q and V expectation channels,
  and the three ESS channels, each times its consensus multiplier in physical units.
  - Sign convention per Expert ruling 3: the estimate is Q − (settlement price part contained in Q's convention) +
    dual.
  - Derive from the code which settlement price part the EV record's Q contains, and state it.
  - PR only withholds determinacy.
- **A(x):** the absolute form of PR. Guard: A(x) + A(x′) ≤ 5,000 €.
- **G(x):** three parts:
  - the ESSO P and Q slacks priced at the ESS multiplier (not the penalty);
  - the within-cohort leak priced as the conversion loss (TASKS `8f2a878a`);
  - the network closure slack priced at its row dual.

**Gates for `gates_pass`:**
- status ok and guards exact;
- the 12.5 kW interface ceiling;
- the ESSO slack gate;
- network closure slack ≤ τ_cl;
- the D1 efficiency-equality gate.

**The leak ceiling (λ·L ≤ 25 €)** is reported and flagged per evaluation. It is **not** in `gates_pass` until the pilot
measures it at the settled point (TASKS `28ba36b4`).

**Cap in the eval key.** For entries that carry any S56 EV-plan key, the cycle cap enters the eval key. This is a
harness change, option (a). Entries without S56 keys keep their keys bitwise. This removes the alias with the
`ev_harness_v1/smoke` spec and between settled runs with different K.

**K for the pilot:** K_pilot = 300 cycles, stop-disabled. The exit value is read at the would-be certification cycle.
K_EV for the search is set from the pilot as the smallest K at which S meets its share. K_EV is frozen with e_EV.

## Integration rulings (Planner, 2026-11-23; after `8948db26` / `370b7689` / `e1b87c27`)

1. **Gate thresholds.** Confirmed as built:
   - τ_cl = 1e-5 p.u. of the block base (1e-3 MWh on base 100);
   - ESSO slack gate |s_up − s_down| ≤ 1e-5 MW / MVAr per element.

   Both are backstops; the priced slacks enter G in any case.
2. **s_share** = 500 € per evaluation: S(x) ≤ 500 €, i.e. 1,000 € of the pair's 2,500 €. K_EV is the smallest K at which
   every pilot run's S ≤ 500 € and K ≥ every run's would-be certification cycle.
3. **Initialisation variant:** the keyed harness option `rho0_scale` ∈ {1.0, 0.5, 2.0}, where 1.0 is the default run.
   - Pilot staged plan: `n5_y2030_p0.5_e1__n7_y2025_p0.25_e1__n7_y2030_p0.25_e0.5__n9_y2035_p0.25_e0.5` (the memory
     probe's SRP1-grid plan).
4. **Readings accepted as built:**
   - "would have certified" = `cycle_convergence` holds on each of the last W cycles;
   - PR = −price + dual, with the interface terms on the TSO-side multipliers (λ_T = −λ_D bitwise, as step 4 showed)
     and the ESS terms on each agent's own multiplier;
   - G priced as built;
   - ψ = block makespan / mean t_exit;
   - e_EV over the 15 pairs.
5. **A not_settled confirmation is "unresolved after confirmation"** and the search continues (option b). **Review
   trigger:** three consecutive not_settled confirmations, or not_settled on more than 25 % of the confirmations so
   far (at least 4), stops the search for review.
6. **Driver gate item 4:** re-scoped to assert that the smoke alias is absent (the cap rule removed it). Re-run the
   whole gate on the final code.
7. **memory_per_slot_gib = 4.0 for EV evaluations.** Measured: 2.43 GiB at 3 cycles, and step 4 children ≤ 3.53 GiB
   over full runs. Five slots therefore need 20 GiB. The pilot measures the full-run peak; the search spec uses the
   pilot's measured peak × 1.15.
