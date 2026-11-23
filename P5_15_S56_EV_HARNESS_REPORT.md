# P5.15 S56 critical path step 1: EV-plan harness keys (Worker report, 2026-11-23)

Authority: TASKS.md, "Working mode and EV priority", critical path step 1 (author and Expert, 2026-11-23). These are
engineering tests: one script with pass/fail checks and no frozen prediction spec.

**Objective convention (every EUR figure):** EUR of total discounted cost.
- `gross` is production's `gross_operational_cost`, with the contracted settlement and the voltage pin excluded (the
  coupling option is off, see B3).
- `net` = gross − terminal-value credit.
- `total` = I(x) + net, where I(x) is the master's investment-cost expression (`GeneralLattice.investment_cost`).

## Blocked on Planner

1. **The leak ceiling is exceeded on the reference unit in the smoke.** Pricing follows your decision in `8f2a878a`:
   λ·L = conversion loss at the TN expected price. The harness has applied that rule since `12d3efc2`.
   - **Unit, cycle 2:** λ·L = **794.29 €** against the 25 € ceiling (31.8×). The full-L price, which does not decide, is
     11,083.12 €. L = 103.35 MWh, weighted.
   - **x0:** 0 €.
   - The largest within-cohort simultaneity is 2.49 kW. That is consistent with the complementarity row's designed slack
     at S = 0.25 MVA: √(1e-4) × 0.25 MVA = 2.5 kW. The mode `independent_units_all_node_years` now activates that row on
     single-cohort node-years as well.
   - This is a 2-cycle, unconverged point, so it is a measurement and not a result.
   - **If the row binds at convergence, the 25 € ceiling cannot hold for any storage plan.** You need to decide whether
     this is an outcome to report or a question for the Expert about the row tolerance.
   - The smoke's own files are schema v1, which read `within_ceiling` from the full-L price (False either way). The rule
     changed after the smoke ran.
2. **The search-driver design (`2c15db47`) needs a harness change that this task did not cover.**
   `s56_certification_stop_disabled` must enter the eval key and must be allowed on single-scenario EV entries. It is not
   implemented. It needs its own task.
3. **The V-1 coupling is inert on the EV twin.** Production activates it only with more than one scenario
   (`_per_scenario_interface_coupling_active`). The smoke therefore did not declare it. Two consequences:
   - The case-file AA (keep_memory, on) runs, because nothing forces AA off.
   - `priced_residual_pr` is absent on EV records. The label is carried with the value None.

   The EV pilot spec should state its AA setting explicitly.
4. **The reactive-flexibility line is −973.18 € on both x0 and the unit (2×).**
   - Likely cause (not verified per variable): `flex_q_down` is NonNegative, and its A5 availability is production's
     constant 1e-5 (`EQUALITY_TOLERANCE`). IPOPT's bound relaxation can leave values of about −1e-8, and those,
     multiplied by the 27,648 price × baseMVA × block-weight coefficients, are of this order. Variable values were not
     inspected: the models are not serialised.
   - The field reports what production's expression evaluates to, unclipped. You decide how it is presented.

## Changed

| commit | content |
|---|---|
| `a8889d4b` | `p515_s44_campaign_harness.py`: five entry keys and the cost breakdown. `p515_s56_ev_harness_tests.py` added. |
| `3e56b131` | Test script: `--smoke-recheck` (zero solves). |
| `54012647` | Test script: the manifest's commit-selection rule. |
| `0789f9da` | Evidence `ev_harness_v1/` + manifest `c8e77efd`. |
| `12d3efc2` | Leak ceiling on the conversion-loss price (your decision); cost-breakdown schema v2. Test script: tagged manifest. |
| (this commit) | Re-run evidence after `12d3efc2` (`nokey_r4`, `stub_run_srp1grid_r6`) + manifest v2 `c2139b56` (predecessor `c8e77efd`; 1,833 files hashed, 165 committed) + this report. |

**The keys.** Each is an entry option, absent by default. An entry without them has the same key, record and NL as
before.
- `s56_cohort_formulation`: sets production's `cohort_formulation` in memory. Keyed unless `legacy_h3`.
- `s56_terminal_value_in_objective`: sets `salvage_value.in_objective`. Keyed when True.
- `s56_esso_slack_penalty`: keyed unless 1e3.
- `s56_ev_profiles`: profiles path, sha256 and EV checksum. Always keyed. Requires a derived instance carrying that
  checksum.
- `s56_staged_investment`: a node-year plan, canonicalised by `p516_lattice` in schema v2. C5 admissibility is refused
  at freeze. The staged key equals `GeneralLattice.key(z)`.

**How keyed items enter the eval key.** As sha256{base_evaluation_key, s56_ev_plan}. `evaluation_key` and the
single-year `canonical_candidate` / `candidate_key` are untouched.

**Child order.** EV injection at the derived read, with the assertion run before the baseline is installed. Then, in the
configuration hook:
1. the EV assertion;
2. the ESSO settings, the staged write and the probe read-backs;
3. the 2× multiplier.

The multiplier refuses to run on an EV entry whose assertion has not run.

**Cost breakdown** (`s56_cost_breakdown.json`, written in the post-run hook; capture path asserted before any solve):
- I(x), gross, the TV credit, net and total;
- the flexibility cost split active / reactive, reconciled with `total_flex_cost`, with a per-family read-back of the
  multiplier;
- the ESSO P/Q slack totals;
- the leak detector;
- per-cohort throughput;
- the E6 certificate (per-cohort closure / balance / window / converter, the Q slack, the network closure slack and
  the `sess_comp` switch);
- κ;
- `priced_residual_pr`, labelled "mixed-convention reporting field, read by no acceptance test".

**Five-block EV twin built** (none existed in the injection-hook schema; Audit A's is the predecessor format):
- `ev_harness_v1/five_block_ev/`: case `61199eb0`, profiles `84491639`.
- EV checksum `d0d3a600`, the same as Audit A's five-block EV.

**`uncoordinated_benchmark.py` sync gap: logged, not fixed.** No import of it on the harness path. Scanned: the harness,
`p515_g_g1_g4_admm_gates`, `p56a_oracle`, `p515_s44_scale_measurement`, `shared_resources_planning`,
`p514_n_instrumented_cstar`.

## Found

All checks are zero-solve except the smoke. The final runs below are from committed code.

- **Five-block EV build — 27/27.**
  - Read-back is bitwise.
  - The assertion passes on EV and raises on an un-injected EV read, on SP, on SRP1 and on a wrong checksum.
  - The object walk against SP is identical.
  - The 8,920 injected means are bitwise equal to Audit A's. Audit A's 3,960 extra arrays are exactly the A5
    reactive-flex arrays.
- **Option-off identity — 14/14** (`nokey_r4`, pre = `aaaa71e2`).
  - The module diff deletes exactly the 9 declared lines.
  - Entries without the keys select nothing and add no record field.
  - `freeze_campaign_spec` gives the same spec before and after, on SRP1 and on a derived instance.
  - **All 307 committed campaign specs: 7,291 candidate keys and 7,270 eval keys reproduce their recorded values.**
  - `build_evaluation_record` gives the same record before and after on 12 committed evaluations.
- **NL identity — 5/5** (`nl_identity_r4`).
  - 102 NL files compared (TSO, DSO and ESSO, cycles 1–2, unit, 2×, option off): identical before and after.
  - Per-cycle diagnostics identical in float.hex.
- **Keys — 32/32** (`keys_r2`).
  - 66 settings give 42 distinct keys. The only shared keys are the declared-default merges, and 1e4 = 10000.
  - The key table is identical in a second process.
  - The staged form is invariant to permutation, int/str keys and explicit zeros.
  - Single-year keys are unchanged on 30 candidates.
  - 9 C5 refusals, each naming its class, and 10 freeze refusals, including the 'mid' ESS file.
- **All keys + EV + staged mixed-duration plan + 2× + V-1 — 30/30 on each grid** (`stub_run_srp1grid_r6`,
  `stub_run_five_block_r2`).
  - The hook order holds.
  - The EV assertion raises if it is run after the multiplier.
  - Multiplier read-back: flex_p_down and flex_q_down each 2,304/2,304 equal to 2 × the original.
  - TV:
    - κ is in the objective;
    - the credit is linear in the terminal-SoH Vars only (closed form ≤ 1e-12);
    - doubling the power cost leaves the credit bitwise unchanged (converter excluded);
    - doubling the energy cost doubles it;
    - 'mid' is refused in the child.
  - Arrangement-A rows are active on every node-year with at least one cohort:
    - SRP1 grid: 2 multi-cohort and 4 single-cohort node-years;
    - five-block: 3 multi-cohort and 7 single-cohort node-years.
  - Gross-flow flags are set exactly on the multi-cohort node-years, with their `sess_comp` inactive.
  - Every cost-breakdown field is captured.
  - The flex split reconciles to 2.4e-16.
- **Smoke — solved, 2 cycles, SRP1-grid EV twin, 2×, all five keys.**
  - **x0 and the unit each made 153 solves, equal to the declared ((1+3)·12+3)·3. 0 blocked, identity holds, the parent
    guard is at 0.** `smoke.json` records 2 FAILs from a check defect: it compared the `observed` dict with an integer.
    `smoke_recheck` applies the corrected comparison to the recorded artifacts: 15/15.

| | x0 | unit |
|---|---|---|
| child exit / cycles | 0 / 2 | 0 / 2 |
| peak RSS, child (GiB) | 2.37 | 2.40 |
| s/cycle (cycles 1, 2) | 26.1, 22.7 | 30.1, 23.8 |
| I(x) | 0 | 317,957.01 |
| gross (cycle 2) | 873,896,422.55 | 873,581,526.25 |
| TV credit | 0 | 0 (2025 cohort, no remaining life at 2040) |
| total | 873,896,422.55 | 873,899,483.26 |
| flex active-down (DSO) | 109,161,444.70 | 109,127,931.53 |
| flex reactive-down | −973.18 | −973.18 |
| ESSO P slack sum at 1e4 (MW) | −1.73e-5 | −1.73e-5 |
| ESSO Q slack sum at 1e4 (MVAr) | 0 | −5.76e-6 |
| leak λ·L, loss / full (€) | 0 / 0 | 794.29 / 11,083.12 |
| κ = a/σ | 2.4266e-3 | 2.4266e-3 |

- **The x0 / unit difference is indeterminate.** Cycle-2 objective change is 891.5 M€ on both.
- **Unit E6.** Closure residual 0; balance residual 1.3e-16 MWh; window margin 0.081 MWh; converter margin 0.083 MVA.
  The network closure slack max is reported as 0, but the actual entries are about −9.9e-7 MWh: the summary floors at
  0.
- **Unit throughput, cohort 2025.** EFC/day 0.588 / 0.553 / 0.500, and end-of-block SoH 0.900 / 0.811 / 0.733 (2025 /
  2030 / 2035).
- **Superseded attempts are kept,** each with its fault. Each fault was a test-script defect or a harness fix on the S56
  path only:
  - `nokey` (duplicate test key);
  - `nl_identity` r1–r3 (case-file AA undeclared; bool serialisation);
  - `stub_run_srp1grid` r1–r5 (the import scan matched its own text; floor rows were read after the candidate update;
    two test expectations: κ = a/σ with a = 227,211, and the absent PR);
  - `stub_run_five_block` r1;
  - `keys` r1.

## Not confirmed

- **No solved run with a nonzero TV credit.** The smoke's only cohort is 2025, whose credit is 0 by the
  remaining-life rule. The credit wiring is checked on built models only (stub runs).
- **R8 at 1e4 with positive controls on the final formulation:** not run. The smoke shows slacks at the bound-relaxation
  level over 2 cycles only.
- **No solved run on the five-block grid:** stub runs only.
- **Memory at 2 cycles is not the full-run peak.**
- **The diagnostic scripts that read the old ESSO penalty constant are untouched.** They are listed in plan v4.1 §6.
- **The Planner's T8 figure of 7.02 € was not reproduced.** It is on a different run and normalisation, and was not
  compared.
