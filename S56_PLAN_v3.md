# S56 plan v3: timing, staging and operational uncertainty under consistent TSO–DSO coupling (2026-11-21)

**Predecessor:** `S56_PLAN_v2.md` (`1bbfed9b`), kept unchanged.

**What v3 incorporates:**
- the Expert's feedback on v2;
- the author's decisions 1–7;
- the Expert's rulings E1–E7, on the search policy, on R and on the first-order correction;
- the Advisor's cohort and search-policy reviews;
- step 0 and the zero-solve audits.

The ruling texts are recorded in `TASKS.md` (S56 section).

**Status:** implementation and zero-solve or minute-scale validation are under way. No search, pilot or full evaluation
has run. The archived results are untouched.

## 1. What the study is

A planning study of investment timing, staged additions and operational uncertainty for shared storage, under
physically consistent TSO–DSO coordination. Two planning arms, expected-profile (EV) and stochastic (SP), share:
- one grid and one investment domain;
- one terminal-value convention and one flexibility-price assumption (2.0× first, 1.0× as reference);
- one set of numerical comparison rules.

Both arms are evaluated under the same stochastic model, with operation re-optimised.

**Claims are scoped** to the selected rule outcomes. For a search's end state the claim is: "No improvement beyond the
applicable comparison criterion was identified among the evaluated neighboring plans". Unresolved comparisons stay
unresolved. No global-optimality claim is made.

## 2. Formulation (decided unless marked open)

| item | decision | authority |
|---|---|---|
| TN–DN coupling | per-scenario: the TN responds to the realized DN exchange; the DSO's expected exchange is the scenario-common day-ahead commitment; DN realizations are not forced onto it | author (a); Expert |
| Voltage coupling | V-1 (per-scenario V channel, pin unwired) or V-2 (pin + scenario-wise ΔV gate); **open**, decided from steps 3–4 | Expert |
| Row 18 | **zero premium primary**; reactive charge removed; no P-only premium sensitivity in the core campaign | Expert; author 4 |
| System cost | internal TN–DN settlement fully excluded; Addendum 38 (C) withdrawn in this mode; investment, operating cost and terminal value each counted once; consensus residuals reported separately; local-objective consistency to be verified | author 6 |
| Scenarios | 3 market × 3 operation; independent draws paired by index = a **small sample of the assumed independence model**; investment-cost trajectories (0.35/0.55/0.10) and price scenarios (1/3) independent (an assumption) | author 1; Expert |
| Dependence stress test | comonotone re-pairing by daily net demand (built, `0f4d39ea`); two fixed-plan evaluations after plan selection; not calibrated, not worst-case | author 3 |
| Storage cohorts | **arrangement A**: complete units, AC-coupled; per-cohort schedules, SoC, window, closure, converter P/Q, degradation; inter-cohort transfer **allowed**, losses and degradation accounted, transferred energy reported; within-cohort simultaneity restricted or demonstrably negligible; legacy H3 kept for archived reproduction; the formulation recorded in records and cache keys | Expert E1, E2 |
| Storage energy decomposition between the ESSO and the networks | **open**: the network aggregate rows are not automatically a valid outer approximation under circulation (the loss (1/η_dch − η_ch)·min(C, D)·Δt). Either aggregate rows with a valid gross-flow treatment, or energy physics in the ESSO with valid injection constraints in the networks. Single- and multi-cohort closure conventions aligned | Expert E2; Advisor designing |
| Re-optimisation without TV | retained for multi-cohort plans. Non-uniqueness is not invalidity. Any secondary allocation rule is documented and shown immaterial, with no artificial charge | Expert E3 |
| Oldest cohort first | a possible result, not imposed; per-cohort throughput reported. The 0.428 MW first-hour figure is conditional on initial energy, efficiencies and capacities | Expert E4 |
| Terminal value | in the optimisation, including the ESSO objective, under the existing energy-residual convention: installation-year expected energy cost × remaining calendar life × (SoH_T − 0.7)/0.3, discounted from 2040; an assumed residual value, not a resale price. The converter residual is excluded explicitly; its materiality is checked by zero-solve repricing (≈ 13 k€ for 2030 and ≈ 24 k€ for 2035 per 0.25 MVA) | author; step 0 |
| No-TV sensitivity | three distinct levels: (a) reprice at fixed dispatch, zero-solve (an upper bound per plan on the re-optimised no-TV objective, not a ranking); (b) re-optimise operation for selected plans; (c) re-search, only if (b) changes the ranking among evaluated plans. No additional full searches requested | author; Expert E3 |
| Flexibility price | 2.0× on DSO `cost_flex`, which reaches the coefficients of `flex_p_down` and `flex_q_down` (all scenario rows). Downward reactive flexibility is an activation cost, shown as its own line in the cost breakdown. Not scaled: wholesale prices, investment, penalties | author 2; audit J |
| ESSO slack penalty | 1e4 via the params key (in `32e21bb1`, default 1e3). Re-verified on the final formulation, with TV, the Q slack and positive controls | Expert E7 |
| Initialisation | a reactive-power slack is allowed. Final acceptance checks the P and Q ESSO slacks; physical limits are not altered to fit a request | Expert E5 |

## 3. Numerical resolution and acceptance

**Target.** R = **10,000 €** is the minimum separation for a determinate comparison of total discounted planning cost,
in both arms.

**Allowance.** R/4 = **2,500 €** is an initial calibration allowance for the numerical variation of a **paired
comparison as a whole**: both evaluations and all diagnostics together, not one allowance per channel. Its composition
rule (what enters, how each is converted to € and combined) is being designed by the Advisor and is frozen before the
pilot.

**Decision.**
- **Determinate:** |ΔF| ≥ R and the paired numerical variation ≤ R/4.
- **Otherwise unresolved.**

**Physical gates** (additional ceilings, not evidence that the economic target is met):
- per-scenario interface |ΔP_s| ≤ 12.5 kW (5 % of the 0.25 MVA unit);
- |ΔQ_s| reported in MVAr;
- ΔV per the voltage ruling;
- ESSO P and Q slacks;
- ESS-channel residuals (MW/MVAr);
- every cohort constraint (balance, window, converter, closure, within-cohort simultaneity);
- inter-cohort transfers monitored separately (not an acceptance failure by themselves).

**Calibration.** Against the centralized comparisons (steps 3, 5) and tighter-tolerance checks:
- the working tolerance is the loosest at which the comparison error stays within the allowance at every later cycle;
- its cost is then measured;
- **if the target cannot be attained, the limitation is reported before any change to R**;
- no rule relaxes a threshold for runtime.

**Search resolution σ** for the search's acceptance: from the pilot's settled pairs, by a rule fixed before the pilot,
consistent with R. It replaces the historical 18,449.66 € fit residual.

**The first-order objective correction** (priced residual) is a **diagnostic only**:
- kept in the validation records with verified units, signs and weights;
- not used to replace an objective, rank plans, or appear beside the main results.

## 4. Search policy (opportunistic polling)

**Shared by both arms and both price levels:**
- **Domain:** the staged domain, 37,041 admissible plans on 2025/2030/2035, or its five-block counterpart if adopted.
- **Neighbourhood:** Addendum 88 R3 plus removal of any cohort, diagonal resizes, and budget-redistribution pairs when an
  increase is budget-infeasible. It is enumerated in full and de-duplicated before evaluation.
- **Order:** deterministic, by last-accepted-move direction, then class rank, node, year and label. It uses no evaluated
  value and nothing from the other arm.
- **Classification:** each candidate is classed at the exit as improvement, unresolved, worse or uncertified.
- **Confirmation:** a settled confirmation at the tight tolerance precedes acceptance.
- **Acceptance:** the first confirmed improver becomes the incumbent.
- **Failure:** an unsuccessful poll is completed.
- **Cache:** write-once, keyed with the full configuration: coupling, cohort formulation, TV, multiplier, scenario
  instance.

**Budget.** N_max is equal per arm and set after the pilot.
- **Completed speculative evaluations count against it,** including those completed after an earlier candidate was
  accepted.
- **On exhaustion,** the search stops with its coverage reported.

**Starts.** x0, the reference unit and the recomputed historical plans; the same start rule in both arms.

**Order dependence.** First-improvement acceptance replaces the old tie convention with **dependence on scan order**.

**EV control searches:**
- a complete-poll search;
- an opportunistic search with the class order reversed.

**History.** The 5.6 k€ separation of the published 2× search's top two does not establish a uniquely preferred plan at
the selected resolution.

**Terminal report:**
- k of |N| evaluated;
- counts by class;
- the unresolved list with Δ and the criterion;
- the termination reason;
- the split of evaluations: decision-relevant, speculative, cache hits, confirmations;
- hours;
- the share of the domain evaluated.

## 5. Grid

**The three-block grid (2025/2030/2035) is used first** for the EV and SP pilots, as authorized. The five-block grid
(2025–2037, 3-year blocks, end 2040) is preferred if feasible. Its pilot and the final grid choice follow the measured
three-block results:
- **Feasibility criteria:**
  - the corrected SP evaluation, staged plan included, at a footprint ≤ 20.7 GiB;
  - affordable time per evaluation at the agreed search policy.
- **Same grid in both arms.**
- **End-of-block SoH kept;** the five-block 'mid' parameter file is refused.
- **Grid comparison:** with common-base inputs; a fixed 2025 unit against x0 on both grids, reported as incremental
  storage value. Search-date effects are grid-specific, because the grids are not nested.

## 6. Reuse and build (status)

| component | status |
|---|---|
| SRP1-grid 3 × 3 instance and EV twin; injection hook and assertion | built (`e681f664`) |
| Dependence stress-test instance (both grids) | built (`0f4d39ea`) |
| Step 0 zero-solve gates: probabilities, TV assertions, H3 derating table, converter and cost-basis repricing | done (`aeaab933`) |
| ESSO slack-penalty key | done (`32e21bb1`) |
| Per-scenario coupling option (F1, V-1/V-2, reactive-charge switch) | in progress (step 1) |
| ESSO cohort formulation, arrangement A, with the legacy key | in progress |
| Storage-energy decomposition (network side) | design in progress (Advisor) |
| Centralized comparator: no storage | in progress (step 3) |
| Centralized comparator: per-cohort storage | after the decomposition |
| Neighbourhood module and \|N\| over the domain; residual scale K | in progress (zero-solve) |
| Opportunistic search driver (O1–O15 as amended) and its determinism gate | to build |
| Harness: staged candidates, keys (coupling, cohort formulation, TV, instance), per-scenario and ESS residual capture, cost breakdown with downward reactive flexibility | partly in step 1; rest after it |
| EV and dependence hooks in the harness child (assertion before the multiplier hook) | to build |
| Diagnostic scripts that read the old ESSO penalty constant directly | fix before any run with the key set |

## 7. Validation sequence

Each step runs under a frozen spec with armed solve guards and recorded predictions.

| # | step | authorization |
|---|---|---|
| 0 | zero-solve gates | done |
| 1 | coupling option + identity gates | existing |
| 2 | ESSO slack-penalty key (zero-solve gates done) + solved re-verification with the final formulation (positive controls) | existing |
| 3 | centralized comparator, x0, α = 0, two blocks (merged = Σ ω single-scenario) | existing |
| 3b | ESSO cohort tests, isolated: identity, the 0.25/1 + 0.25/0.5 example, aged + new windows, circulation with every cohort returning to its initial energy, within-cohort simultaneity, non-uniqueness without TV, expiry | existing |
| 3c | storage-energy decomposition implemented + aggregation tests (incl. the circulation example) + centralized per-cohort comparator (E3 check) | existing (minute-scale) |
| 4 | ADMM on two blocks: pinned against F1 + V-1/V-2, α = 0; calibration of the working tolerance against step 3 | **authorized** (≈ 10–20 h) |
| 5 | storage case and TV in operation against the centralized comparator | existing |
| 6 | staged integrity: one full single-scenario staged evaluation, mixed duration, TV on | existing |
| 7 | concurrency gate with staging and TV (timing fields excluded in advance) | existing |
| 8 | **pilot:** EV pairs (both grids, native and common-base draws); SP on 2025/2030/2035 with x0, the reference unit and a staged plan, to settling. Measures t_SP, memory, σ and the calibration of the paired allowance | **authorized** (EV, three-block SP) |
| 8b | five-block SP pilot | after step 8 |

## 8. Cost projection (to be replaced by step 8's measurements)

| item | 2025/2030/2035 |
|---|---|
| EV search, opportunistic, 5-way speculative (0.9–1.5 h per evaluation) | ≈ 15–50 h, plus the two EV control searches |
| SP search, opportunistic, serial (t_SP 6.5–15 h, unmeasured) | ends at its start: 80–375 h; 3 moves: 160–525 h; 6 moves: 305–855 h |
| common evaluation (cross table, partly cached) | 0–45 h |
| dependence stress test (2 fixed-plan SP evaluations) | 13–30 h |

**These are projections, not limits of the model.** The SP memory footprint fits on the three-block grid.
**Before any search** the author names the search form and its hours from step 8's measurements. Each SP launch over 4 h
is listed with its run id and estimate.

## 9. Corrections to plan v2

1. **"Not affordable" (finding 1)** is now stated as projected serial hours that exceed the time authorized, with
   unmeasured inputs. It is **not** a memory or model limit; "minimum credible" is withdrawn.
2. **Row 18 (finding 3):** the D0 comparison is a TN-only re-solve at fixed DN dispatch. The reactive-charge basis was
   searched in the code and records only.
3. **X6 (keep H3 and flag):** rejected by the Expert. The SoC excursion is up to 14.5 % in the domain, not 4–8 %.
4. **A2:** "a valid 3-point sample" is replaced by "a small sample of the independence model".
5. **The no-TV levels:** fixed-dispatch repricing bounds the re-optimised objective per plan and does not rank. The SP
   re-search trigger is level (b).
6. **The converter re-ranking** ranks evaluated plans only. It says nothing about what a search under that objective
   would select.
7. **Search:** complete polls and the F-first tie rule are replaced by §4.
8. **Grid criterion:** "\|N\| ≥ 30" is replaced by the enumerated \|N\| (in progress). "Likely fails" is a projection.
9. **"D0 already suggests a small coupling effect"** is withdrawn: −2.6 k€ against an ≈ 18 k€ threshold is unresolved.
10. **"Mesh-local optimal" and "x_EV is mesh-local optimal under SP"** are replaced by the Expert's sentence (§1).
11. **"First-improvement removes selection dependence"** is replaced by "it substitutes scan-order dependence" (§4).

## 10. Manuscript

- **Archived 3 × 3 results:** a disclosure draft is ready (`S56_ARCHIVED_3X3_DISCLOSURE_DRAFT.md`) for the author to
  place. The diagnostics go in the supplement.
- **The corrected evaluations** will form the main narrative.
- **Replace or retain:** a recommendation on which tables, figures and sensitivities to replace or retain follows once
  step 8 and the first searches report.
