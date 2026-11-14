# Request for rulings — Addendum 88 (author → expert, 2026-10-31)

Context: Addendum 87 (adopted 2026-10-31) realigns the revision with the submitted paper's intent under valid
mathematics: a convex (SOC, W-space) relaxation of the production recourse supplies a lower bound and the Benders
master over the full (node, year) lattice; the certified full-AC ADMM point supplies the upper bound and every
reported physical quantity; MADS supplies the mesh-local certificate around the master's plan. One re-evaluation
campaign carries a convention change: salvage in the planning objective, mid-block SoH, five 3-year blocks
2025–2037, lattice unchanged. Full reasoning: `REVISION_REALIGNMENT_ANALYSIS_2026-10-30.md` (v2); specs
`P5_16_D0_SPEC_v2.md`, `P5_16_CONVENTION_CHANGE_SPEC_v2.md`, `P5_16_TPERF_SPEC_v1.md`. Six rulings are requested;
each states the options, the cost and the Planner's recommendation. Code facts cited were verified against the
repository (file:line in the specs).

## R1 — Salvage mechanism (formulation; supersedes the governing decision "no explicit degradation cost")

Options. (a) A terminal salvage term in the ESSO objective, `objective = feasibility_penalty + κ·salvage_credit`,
κ = `admm_esso_al_scale / admm_common_objective_scale` (= 1/w_med; the ESSO's AL terms carry a = σ/w_med while
network objectives are divided by σ/w_b), κ a mutable Param set in the ADMM wrapper, κ = 0 at the initialisation
solve; dispatch then internalises the marginal salvage loss of cycling. (b) Planning-only credit: F = I + Q − salvage
with dispatch unchanged (the critical review's §3.3 objection stands).
Fact that bears on the choice: `_get_remaining_calendar_life` gives a **2025 cohort zero remaining life at the
horizon end** (age 15 y = calendar life), so its salvage is identically zero on both instances (0 / .2 / .4 / .6 / .8
for 2025…2037). Under (a) dispatch is salvage-aware only for 2028+ cohorts; the reference unit (2025) is unaffected.
Cost: (a) one Param and one objective term, zero-solve gates G-CC-1…3, the A/B as a campaign item; every storage
cell is re-evaluated anyway. Recommendation: **(a)**, with the F2 fact stated in the manuscript.

## R2 — Salvage basis and parameters (convention; author supplies citations)

Today: `cost_basis = EXPECTED_INSTALLATION_ENERGY_COST` (purchase-year expected energy cost),
`energy_recovery_fraction = 1.0`, `recycling_floor_fraction = 0.0`, remaining-life fraction at terminal, health
normalised above `minimum_soh`, energy only (converter not credited). With the installation year now a decision
variable this basis decides the timing result (a 2037 cohort recovers 0.8 × fraction of its energy cost after one
block of operation).
Questions. (i) Cost basis: horizon-end expected energy cost (operationally the **2037** column — the cost reader
requires an exact year column; a 2040 value needs data or an extrapolation rule) vs purchase-year. (ii) Recovery
fraction: a cited second-life / resale value (the author will supply the source) vs 1.0. (iii) Recycling floor:
0 vs a scrap value. (iv) Health basis for a calendar-expired cohort above the SoH floor: zero (today) vs a
health-only residual. (v) Converter: no credit (today) vs pro-rata.
Recommendation: horizon-end (2037) cost; cited fraction well below 1; floor 0; keep calendar-expired = 0 with the
sentence that the 15-year calendar life is the asset's end of life by assumption; converter uncredited. Identical
on both sides (LB at maximum credit).

## R3 — Certificate wording on the general lattice

The search space becomes z ∈ ℤ^{2|N||Y|} (30). The Addendum A2 completion ("every admissible lattice point within one
∞-norm step") is combinatorial there (3³⁰). Proposed: the **coordinate unit neighbourhood** — ±1 in one coordinate,
plus the activation moves of each inactive cohort (P 0 → 0.25 with E ∈ {0.5, 1.0}); ≈ 60–70 points before
admissibility, 30–50 after the budget; statement "mesh-local optimality in the coordinate unit neighbourhood";
completion cap and N^max restated. Recommendation: adopt; the OrthoMADS positive basis carries the convergence
theory in either case.

## R4 — Wording if the bound is valid but non-discriminating

D0 (spec v2) classifies the relaxation gap absolutely (H1 ≤ 1e-4 of Q; H2 ≤ 1e-2; H3) **and** differentially
(P6: |Σgap(unit) − Σgap(x = 0)| against the 259 k€ storage value). The record (P5.5-D) says the dominant looseness
is the TN rank gap of the meshed 9-bus, plausibly plan-independent, not storage circulation. If H2 with P6 holding:
the certificate UB − LB is non-discriminating but the LB ranks plans. Requested: the admissible sentence(s) for
that case — e.g. "the plan is within g % of the global optimum of the AC-feasible lattice problem; g exceeds the
between-plan differences, so plan selection rests on the mesh-local certificate; the relaxed problem's optimal plan
coincides with the incumbent (evidence, not proof)". Recommendation: rule the wording now so the D0 outcome does not
reopen the question.

## R5 — closed by the author (Gurobi `QCPDual = 1` primary, Clarabel fallback per block, IPOPT cross-check).

## R6 — W184 (E3) and W185 (E2)

They would produce numbers under the superseded convention (end-of-block SoH, 5-year blocks, no salvage, SRP1).
Options: hold and re-scope the NRF arm and the penalty pilot into the new campaign (one convention in the paper) vs
run as ordered. Cost of holding: none now; the launcher work is reused. Recommendation: **hold**.

## F4 — a valid tightening of the LB side (wording of Decision 2)

SoH_T ≤ φ_cal^age is dispatch-independent, so the LB may credit salvage at normalised health
(φ^age − soh_min)/(1 − soh_min) (0.447 for a 2028 cohort at φ = 0.985, soh_min = 0.7) instead of 1.0, and bound
available energy by φ^age × rated — both affine in E, both still valid relaxations. Recommendation: adopt; replaces
"SoH ≡ 1" by "calendar-only SoH" in the LB definition.

## Also for information (no ruling needed)
D0 is route A: the W86 reference pair's `certified_models.pkl` (x = 0 cycle 132, unit cycle 112) were persisted on
the Studio and are hash-verified on the 24 GB machine; D0 is therefore zero-ADMM-solve. T-perf (six declared solves
here, six on the Studio) measures where solve time goes before any formulation lever is considered; a lever would
enter the convention-change spec as C6 under the Step 1b candidate-3′ gate.
