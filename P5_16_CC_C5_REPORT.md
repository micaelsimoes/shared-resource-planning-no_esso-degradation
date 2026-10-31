# P5.16 convention change — C5 general lattice and gate G-CC-4 (Worker report, 2026-10-31)

Authority: `P5_16_CONVENTION_CHANGE_SPEC_v4.md` §1 C5 and §2 G-CC-4; `PLANNER_BRIEF_2026-09-13.md` Addendum 88 R3.
**Zero solves:** two blocking guards were armed for the whole run, the gate's own `SolveProfileGuard(permitted=())`
(armed before any project import) and `p515_s47_phase_b_record.PARENT_GUARD`. Both were verified at exactly 0/0.
**Objective convention:** this gate reports no Q or F values. Every euro figure below is I(x), the expected discounted
investment cost, which is the master's `investment_cost` expression.

## Blocked on Planner

Nothing blocks. Three items need a Planner decision:

1. **Commit scope.** The task said to stage three files: the two modules and this report. I staged only those.
   CLAUDE.md requires a report to be committed together with its evidence base and that evidence's manifest. The
   evidence is in `data/SRP1/Results/P516CC/g_cc4/`, and its manifest file has sha256 `5c99a3fb…`. If the commit
   should include the evidence, the files to add by name are:
   - `g_cc4_results.json`, `back_compat.json`, `oracle_points.json`, `neighbourhoods.json`;
   - `run.log`, `planning_read_stdout.log`, `manifest_sha256.json`;
   - the two launch logs, which sit next to the directory.

   `side_writes/` holds the 52 redirected Diagrams PDFs (1.1 MB). They are hashed in the manifest, and you can
   either commit them or leave them out.
2. **Finding F-1: the cumulative cap never binds at B = €1M.** With the committed cost file it does not bind on either
   instance (details under Found). Its effect on the admissible set appears only if a no-budget Phase B is run
   (STEP4 §1.2).
3. **Debug flag in the gate script.** `p516_lattice_gate.py` has a `--trial-out DIR` flag. It accepts only a directory
   outside the repository, and I used it twice under `/tmp` while debugging. Neither trial wrote into the repository.
   The evidence run did not use the flag. You can keep the flag or ask me to remove it.

## Changed

The two code files and this report are staged by filename and not committed. No production file was edited.

- **`p516_lattice.py`** (new; sha256 `ff5d63ef…`): a pure module that does not import Pyomo.
  - **Decision vector.** `LatticeInstance` and `GeneralLattice` define z ∈ ℤ^{2|N||Y|}. The order is node-major, then
    year, then (zP, zE), which is the order `p56a_oracle.vector_to_candidate` uses. The granules are 0.25 MVA and
    0.5 MWh, cited from `p515_s47_phase_b_record.py` l.186–187 and asserted equal in the gate.
  - **Loaders.** `load_instance(case_path)` and `load_instance_from_case_dict(case, data_dir)` read:
    - nodes from `DistributionNetworks[].connection_node_id`, and the Years map;
    - `DiscountFactor`;
    - the params file: budget, max_capacity, the E/P factors, `ageing.calendar_life_years`;
    - the cost file through openpyxl, the same cells production reads. `Scenarios!B1` is the scenario count, `C1…`
      are the ω_m, and the 'Investment Cost, Power' and 'Investment Cost, Energy' sheets have years in row 1 and
      scenario m in row 1 + m.

    Each file read is sha256-recorded in `sources`.
  - **Admissibility.**
    - Per cohort: the sign rule, P = 0 ⇔ E = 0, and the E/P window taken from the params file.
    - The cumulative energy rule per node and block uses production's cohort window: tcal_norm = round(t_cal/Years[y]).
    - The budget uses the master's own term order.

    `reasons()` keeps the old Lattice's short-circuit semantics. `family_violations()` checks each master-row family
    on its own.
  - **Keys and canonical form.** The canonical form follows key schema v2. The key reuses
    `p515_s44_campaign_harness.candidate_key`. The x = 0 alias is documented in the module. The module has
    `canonical_of_z`, `z_of_canonical` and `canonicalize`; `canonicalize` is idempotent and refuses off-lattice
    values and unknown nodes or years.
  - **Neighbourhood.** `neighbourhood()` follows R3 (a)/(b)/(c). The closed-form bound 2|N||Y| + 5A is asserted,
    which is 30 + 5A when |N||Y| = 15. Each point carries a move label, its key, its reasons, and its I(x).
    Inadmissible points are counted by reason class.
  - **Other pieces.** `resolution_by_bound()` is a pure function. `COMPLETION_CAP = 60` and `N_MAX = 120` are
    constants.
- **`p516_lattice_gate.py`** (new; sha256 `5942e7b2…`): the G-CC-4 gate, parts B1–B4, described in its docstring.
  Before any work it asserts that a capture path exists for 23 quantities. After building the master it asserts
  that the master has the 11 expected components.
- **Evidence:** `data/SRP1/Results/P516CC/g_cc4/`.
  - Run conditions: HEAD `0b92e24a`, wall time 31 s, stderr empty, all 30 checks passed.
  - Inputs: SRP1.json `61a794a7…`, SRP1_ESS.xlsx `e17bd588…`, SRP1_ESS_Params.json `39106f93…`. The W2 table
    `28152120…` matched its pin.
  - The launch command is the one in the gate's docstring. It was run attached, with stdout and stderr captured to
    `g_cc4_launch_std{out,err}.log`, and then `--manifest` was run.

## Found (predictions recorded in the script before the run)

| Prediction | Outcome |
|---|---|
| **L1** Old-domain round trip is exact and the counts are equal. | **Held.** The old `domain()` has 1,705 points. Every one maps to an admissible general point with one common year. An independent enumeration (124 options per node, 1.9 M combinations) gives 6,694 admissible general points with ≤ 1 nonzero year per node. Of these, 1,705 have a common year, and they map back onto exactly the old domain: both the key sets and the z⁷ sets are equal. The other 4,989 are new single-cohort-per-node plans whose nodes use different years. Reason classes agree on 2,000/2,000 seeded inadmissible raw points (seed 20261031, 2,106 draws, 254 of them budget-only). |
| **L2** The master-row oracle agrees on 200/200 points and I(x) is within ≤ 1e-12. | **Held.** All 200 points agree on the verdict (tolerance 0), on every family's violated (node, year) set, and with production's `_check_candidate_first_stage_feasibility`. The rated-equality residual is 0 on all of them. I(x) equals `pe.value(model.investment_cost)` bitwise (max relative difference 0.0); against `p56a_oracle.investment_cost` the difference is 4.0e-16. Points closest to the budget sit 1,963.93 € from it, so no verdict depends on rounding. |
| **L3** \|N(x=0)\| = 30 on the five-block years. | **Held: 30.** All 15 cohorts × 2 activation sizes fit the budget. The neighbourhood equals the independently built set of budget-fitting activations. On SRP1 the count is 18. |
| **L4** A staged 3 + 3 MWh plan at one node is rejected by the cumulative rule and by the master row. | **Held.** The plan is node 5, (1.0 MVA, 3 MWh) in 2025 and in 2030. The module gives "cumulative rated energy > 5 MWh in blocks [2030: 6, 2035: 6]". The master's `energy_storage_maximum_capacity` rows are violated at (5, 2030) and (5, 2035). The master's budget row is violated too, since I(x) = 1,774,050.44 €. |

**200-point oracle sample, by violated master families:**

| Violated families | Points |
|---|---|
| None (feasible) | 63 |
| Budget only | 67 |
| Max capacity + budget | 52 |
| Duration + budget | 12 |
| Duration only | 6 |

Row counts in the master: 9 maximum-capacity rows, 18 E/P rows, 1 budget row, 9 + 9 rated-equality rows, and no
Benders cuts.

**Neighbourhoods (Addendum 88 R3).** Every check passed for each of the five seeded incumbents on both instances:
- the raw count is within the bound, and on the five-block instance the bound equals 30 + 5A;
- every listed point is admissible and every dropped point is inadmissible;
- each shift has the expected z, leaves the other coordinates unchanged, and keeps the node's totals;
- running the neighbourhood twice gives the same result.

On SRP1, every listed and every dropped point was also put to the master rows, with no disagreement.

| Instance | Incumbent | A | Raw | Bound | Admissible | Dropped (class) | Shifts (merged) | B − I(x) € |
|---|---|---|---|---|---|---|---|---|
| five-block | x0 | 0 | 30 | 30 | 30 | — | 0 | 1,000,000 |
| five-block | single (n5 2028 0.75/2.5) | 1 | 34 | 35 | 30 | dur 1, bud 3 | 2 (0) | 304,884 |
| five-block | two-node | 2 | 40 | 40 | 35 | dur 5 | 4 (0) | 611,339 |
| five-block | staged (n5 2028 + 2031) | 2 | 38 | 40 | 8 | dur 3, bud 27 | 4 (2) | 100,331 |
| five-block | at budget (greedy, 6 cohorts) | 6 | 55 | 60 | 10 | dur 16, bud 29 | 8 (2) | 9,005 |
| SRP1 | x0 / single / two-node / staged / at budget | 0/1/2/2/4 | 18/22/26/27/36 | 18/23/28/28/38 | 18/19/21/22/11 | see json | 0/1/2/3/6 | — |

The "dur" drops in the table are the −1 coordinate steps that take a cohort to P = 0 with E > 0, or to E < P. These
are expected: removing a smallest cohort is the separate deactivation move.

`resolution_by_bound` was tested on synthetic numbers only, since no lower bound exists yet. It resolves exactly the
points with F_LB − margin ≥ UB, accepts a scalar margin, a per-key margin map or a callable, and resolves nothing when
no bound is supplied.

**Finding F-1 (informational, not a gate check).** Under B = €1M and the committed cost file, every plan that
violates the 5 MWh cumulative cap is also over budget.
- The cheapest over-cap plan costs 1,108,893.20 € on SRP1 (1.5 MVA / 5.5 MWh in 2035) and 1,031,559.51 € on the
  five-block instance (the same plan in 2037).
- This was enumerated for one node with ≤ 2 cohorts, zE ≤ 11 per cohort: 4,266 and 14,190 over-cap plans
  respectively. Unit costs do not depend on the node.
- For more than 2 cohorts it follows from an argument rather than the enumeration. Each cohort needs zP ≥ zE/2, and
  the discounted unit costs are lowest in the last year. So no split beats one cohort in the last year.

Consequence: at B = €1M the cumulative rule does not change the admissible set; the budget dominates it. It acts
only in a no-budget Phase B. L4's plan is rejected by both rules. `reasons()` reports only the cumulative one,
because, as in the old Lattice, the budget is evaluated only when no structural reason exists. `family_violations()`
reports both.

## Not confirmed

- **Five-block instance against the master.** No five-block case file exists yet (C4 / G-CC-5 is a separate task), so
  the five-block neighbourhoods were checked against the module only. The years come from `derive_case('paper', 1 × 1)`.
  The master-row oracle and the bitwise loader check (that the module reads exactly what production reads) were run
  on SRP1 only.
- **Calendar window.** The general cohort window is implemented, but no instance tests it below the horizon end: with
  15 y of life, every cohort on both instances stays alive to the horizon end.
- **Cap–budget dominance for > 2 cohorts.** This rests on the argument above, not on enumeration.
- **The other Workers' roots.** I did not read or touch `P516D0/t01b`, `P516TPERF/azero`, `p516_d0_*` or
  `p516_tperf_azero.py`. HEAD advanced to `0b92e24a` (D0 T0.1b) during my work, and that is the HEAD the run recorded.
