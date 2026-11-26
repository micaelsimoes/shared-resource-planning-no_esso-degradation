# S56 plan v2: timing, staging and operational uncertainty under consistent TSO–DSO coupling (2026-11-21)

This plan answers the author's three messages of 2026-11-21:
- the revised scope;
- 2.0× flexibility prices first;
- the five-block grid preferred if feasible.

It builds on these predecessors:
- design notes `S56_DESIGN_NOTE_v0.md` and `S56_DESIGN_NOTE_v1.md`;
- `S56_SCENARIO_COUPLING_PROPOSAL.md`;
- `P5_15_S56_STATUS_REPORT.md`.

The evidence behind it:
- audits A–D, E, F, I and J, and the TN-only re-solve D0 (v1, v2), all under `data/SRP1/Results/P515S56/`;
- the Advisor's reviews (decisions D1–D10 below).

**Nothing in this plan has been run beyond zero-solve audits and minute-scale diagnostics.** The archived results are
untouched.

## Three findings that change the plan

1. **A stochastic (SP) search with complete polls is not affordable on this machine, on either grid.**
   - **Projected time:** 390–1,800 h on 2025/2030/2035 (minimum credible ≈ 290–675 h), and more on five blocks.
   - **Assumptions:** SP evaluation 6.5–15 h, concurrency 1, 45–120 evaluations per search. The 5 h SP figure on the
     SRP1 grid has never been measured, and the 1.3–3× coupling factor is a guess.
   - **Consequence:** the plan has a **decision point after the pilot** on the form of the SP arm.
2. **Mixed-duration cohorts at one node lose aggregate power** (derived from code, to be confirmed on a built model).
   - **Why:** throughput is attributed to cohorts in proportion to installed energy (H3,
     `shared_energy_storage_data.py:806–809, 1812–1822`), and each cohort's own converter caps its share (l. 752–753).
     So |P_agg| ≤ min_c s_c / share_c.
   - **Example:** 0.25 MVA / 1 MWh plus 0.25 MVA / 0.5 MWh delivers at most 0.375 MW of 0.5 MVA.
   - **Effect:** a staged search would be biased against such plans.
3. **Row 18 is not a cost the deviation causes, once the TN re-dispatches.**
   - **Size:** at α = 0.5 it is 25.08 M€, about 140× the TN's measured cost of serving the deviations (178,866 €, D0) and
     27× the cost the pinned model omits (D1).
   - **Reactive power:** it charges reactive deviations at the same α·π̄ (`model_construction_helpers.py:1979–1981`),
     with no economic basis found. It also holds Q at the commitment, the restriction the author asked to avoid.

## Decisions needed

### Expert

| # | decision | recommendation |
|---|---|---|
| X1 | **Voltage coupling under option (a).** V-1: a per-scenario V channel with the pin unwired (reverses Addendum 38 (D)). V-2: keep the expectation channel and the pin, plus a scenario-wise ΔV gate | build both behind the option; decide from the centralized bracket Q_pin − Q_free and the convergence in validation steps 3–4; V-1 as the formulation of record if both pass |
| X2 | **Row 18 and the accounting.** Its role once the TN re-dispatches; the reactive charge; withdrawing Addendum 38 (C) ("deviation part economic") in the new mode, or deviation energy is counted twice | **drop the Q charge**. α = 0 as the primary corrected configuration: the centralized comparator becomes a plain merged stochastic OPF, and EV and SP share one cost functional. A P-only premium as a fixed-plan sensitivity. Q_sys = Σ_s ω_s (TN generation + DN flexibility + tabled slack terms); the settlement, pin and AL terms excluded |
| X3 | **Acceptance.** ω-weighted Boyd norm r² = r̄² + Σ ω (r^d_s)²; scenario-wise gates max \|ΔP_s\|, \|ΔQ_s\| ≤ f_max·S_min and ω-RMS ≤ f_rms·S_min with S_min = 0.25 MVA; ΔV gate; objective stability (10-cycle bar plus terminal-step ratio) | f_max = 0.2 (0.05 MW), f_rms = 0.04 (0.01 MW), ε_V = 1e-3 p.u. max, 3e-4 RMS. The **rule** (the tightest f_max ∈ {0.4, 0.2, 0.1} met within 1.5× the Boyd-exit cycles) is fixed before the pilot; the **values** after validation, before any search |
| X4 | **Search resolution σ.** It replaces the 18,449.66 € fit residual, which is not an error bound | rule fixed before the pilot: σ = max over the pilot's settled pairs and both arms of (\|V_exit − V_settled\| + settled band sum), floored at 2τ, common to both arms. Validated on held-out settled confirmations; a failure stops |
| X5 | **Economic significance of residuals.** Report the signed priced residual PR = Σ w Σ ω Σ_t (π_s + μ_s) ΔP_s and its absolute form as a first-order diagnostic, **never as a bound** | a comparison is indeterminate if \|PR(x) − PR(x′)\| > 0.25 × max(band sum, σ) |
| X6 | **H3 reading and mixed-duration derating** | keep H3 for this study and flag derated plans. The per-cohort SoC excursion is ≤ 4–8 % for age gaps of 5–10 years, read as an attribution rule |
| X7 | **R8 applicability.** ESSO slack penalty 1e4 via a params key (default 1e3 unchanged). It is needed because terminal value in the ESSO objective otherwise exceeds the slack bar (4.78× on the SRP1 grid, 5.84× on five blocks) | confirm that "the new instance" in Addendum 89 R8 covers the S56 instances |
| X8 | **Neighbourhood additions** to Addendum 88 R3: (d) removal of any cohort; (e) diagonal resizes (±1, ±1), (±1, ±2); (f) redistribution pairs when an increase is budget-infeasible | adopt. Coordinate steps are almost all duration-infeasible (36 of 36 at x0), and R3 alone leaves 4 moves at the 992 k€ plan |
| X9 | **The form of the SP arm** if the pilot confirms the cost (D10 options i–iii) | decided after the pilot, with measured t_SP |

### Author

| # | decision | recommendation |
|---|---|---|
| A1 | α (the row-18 premium) | 0 primary; 0.5 on P only as a sensitivity (follows X2) |
| A2 | **Scenario dependence.** Index-pairing of independently drawn per-network operation scenarios is a valid 3-point sample of the independent product measure. To state: TN and DN operation independent given season and year; market independent of operation; no weather coherence; price independent of RES | accept and state. Add a **comonotone-pairing sensitivity** (rank by daily net demand; marginals and EV unchanged) as a fixed-plan check |
| A3 | ESS-workbook cost-trajectory probabilities (0.35/0.55/0.10) against price-scenario probabilities (1/3 each) | state them as independent uncertainties |
| A4 | Doubling the `flex_q_down` price with 2.0×, as in the published 2× cells | keep, for consistency; its capacity is 1e-5 p.u., so the effect is negligible |
| A5 | **Grid inputs.** Draws are keyed by (year, day), so the grids share only 2025. Twins built from a common base draw, scaled by growth, isolate the grid effect | build common-base twins for the grid-comparison pilot only (a scenario-construction decision) |
| A6 | Grid for the searches (five-block preferred if feasible) | decided after the pilot, on the criteria below |
| A7 | **Machine time** | see "Blocked on the author" |

## Blocked on the author

- **Machine time beyond the existing authorization:**
  - validation step 4 (ADMM on two blocks, 5–6 runs of 1.7–3.3 h, ≈ 10–20 h);
  - step 8 (the pilot: EV 3–5 h; SP on the SRP1 grid 20–45 h; five-block SP 32–90 h, only if memory allows);
  - every SP search launch.
- **Push from PyCharm.**

## What is reused, modified or built

| component | status | evidence |
|---|---|---|
| SRP1-grid 3 × 3 instance and EV twin, injection hook and in-child assertion | **reuse** | E, 65/65 checks; SP `73d0a230`, EV `0040b146` |
| Five-block 3 × 3 instance (`w89`) and its EV variant | **reuse** for SP, **but not** `SRP1_5x3y_ESS_Params.json` (bundles 'mid'). A new 'end' + TV params file, with a guard that refuses 'mid' | A, I |
| Flexibility multiplier `flex_price_multiplier` | **reuse**. It scales DSO `cost_flex` in every scenario row, which reaches `flex_p_down` and `flex_q_down` (the only flexibility terms in Q). Not scaled: wholesale prices, π̄, the settlement, the ESSO, investment, penalties. The TSO flexibility charge is identically 0 | J, 29/29 |
| EV + multiplier ordering | **modify**: the EV assertion before the multiplier hook, or expecting m × mean on DSO blocks | J |
| Terminal value in the ESSO objective (`salvage_value.in_objective`, κ = a/σ) | **reuse**, independent of the SoH point and blocks. κ parity is an identity; it **needs** a solved check (step 5) | I; Advisor D5 |
| R8 ESSO slack penalty | **build**: a params key, default unchanged (production change, small) | I |
| Multi-cohort ESSO (H3) | **reuse, validate** (step 6). A later cohort leaves an earlier cohort's rows unchanged (zero-solve B1–B5) | I |
| General lattice `p516_lattice.py` | **reuse**; **extend** with moves (d)–(f) | I, J; C5 G-CC-4 30/30 |
| Staged search driver | **build** (≈ 400–800 lines): the old `run_mads` is fixed at 7 variables and 10 primes | I, J |
| Harness: multi-cohort candidate, keys, TV option, per-scenario residual capture | **modify** (G1–G3, G5 of audit I; ≈ 200–300 lines plus gates) | I |
| Per-scenario coupling F1 + V-1/V-2 + Q-charge switch, behind a default-off option | **build** (production, about 10 sites) | coupling proposal, change list |
| Centralized multi-scenario comparator with shared ESS | **build**: `combine_networks` empties the shared ESS and has no commitment terms | G (V0) |
| Concurrency gate | **reuse** the v1 harness with timing fields excluded in advance | F |

## Formulation as planned (subject to the Expert items)

- **Coupling.**
  - **Channels:** the existing expectation channel plus a per-scenario deviation channel on P, Q (and V under V-1).
  - **TSO side:** the TSO's per-scenario `interface_delta` freed; `pc` stays fixed.
  - **The commitment:** the DSO's expected exchange, scenario-common.
  - **Storage:** schedules stay scenario-common (hard non-anticipativity by aliasing). This is distinct from the network
    dispatch, which becomes scenario-dependent on both sides.
- **Terminal value:**
  - **The objective:** F = I + Q_gross − S, used once in the search driver.
  - **S:** installation-year expected energy cost × remaining calendar life × (SoH_T − 0.7)/0.3, discounted from 2040.
    The 2025 cohort's S = 0 by construction (asserted).
  - **Converter residual:** excluded, explicitly. It is material for later cohorts (≈ 13 k€ for 2030 and 24 k€ for 2035
    per 0.25 MVA), so every evaluated plan is also re-ranked with F − C_conv(x), zero-solve. The same repricing covers
    the alternative horizon-end cost basis.
- **No-terminal-value sensitivity, three levels:**
  - (a) reprice at unchanged dispatch, zero-solve, for all evaluated plans;
  - (b) re-optimise operation with κ = 0 for the selected plans;
  - (c) re-search: EV always; SP only if (a) changes the SP ranking.
- **Flexibility price:** 2.0× for the first searches; 1.0× as a reference. The three levels are named in every table:
  repricing an unchanged dispatch, re-optimising operation for a fixed plan, re-running the search.
- **Search:**
  - **Domain:** the staged domain (37,041 plans on 2025/2030/2035).
  - **Neighbourhood:** R3 + (d)–(f), on a unit mesh.
  - **Polls:** complete, in deterministic order.
  - **Moves:** on a determinate improvement at the exit, beyond max(band sum, σ), followed by settled confirmation.
  - **Ties:** (F, I, label).
  - **Barrier:** extreme, through the production feasibility check.
  - **Caps:** completion cap 60; N_max 120 per arm.
  - **Starts:** x0, the reference unit and the recomputed historical plans; search from the best, with the same rule in
    both arms.
  - **Reporting:** stopping reason, k of \|N\| at the final incumbent, evaluations as a share of the domain, hours per
    arm. **No claim of global optimality.**

## Grid: feasibility criteria (declared before the pilot)

The five-block grid is adopted for EV and SP if both of these hold:
1. **Memory:** the peak footprint of a corrected five-block SP evaluation, staged plan included, is ≤ 0.85 × 24.4 =
   20.7 GiB.
2. **Time:** one complete SP poll (\|N\| ≥ 30 on 30 integer variables) fits the authorized time.

**Projection:** criterion 1 is marginal (the archived pinned run is at 19–20.5 GiB). Criterion 2 likely fails for SP.

**Fallback:** five blocks for EV searches and for SP fixed-plan valuation; the searches on 2025/2030/2035.

**What is separated:**
- the **fixed-plan valuation sensitivity**: a 2025 unit against x0 on both grids, with common-base inputs, reported as
  incremental storage value;
- **search-date effects**: grid-specific, because the grids' installation dates are not nested.

End-of-block SoH is kept on both grids.

## Validation sequence

Each step runs under a frozen spec with armed solve guards and recorded predictions.

| # | step | key predictions | cost | authorization |
|---|---|---|---|---|
| 0 | **Zero-solve gates:** probability and scenario-count identity across networks; expired-cohort and 2025-TV = 0 assertions; H3 derating over the 37,041 plans (confirms finding 2 on built models); converter and cost-basis repricing table; TV coefficients on both grids; capture-path assertions for per-scenario ΔP/ΔQ/ΔV | derating < 1 only at mixed-duration nodes; 2025 TV = 0 | < 1 h | existing |
| 1 | **Implementation** of F1 + V-1/V-2 + Q-charge switch (default off); identity: SRP1 two-cycle bitwise with the option **on** (inert at 1 scenario); 3 × 3 two-cycle bitwise with it **off**; every preserved fixture loads | all hold | 1–1.5 h of runs | existing |
| 2 | R8 params key plus gate re-run | slack ≈ 0; ESSO iterations within ±20 % | < 0.5 h | existing |
| 3 | **Centralized comparator,** two blocks (fewest and most TN-curtailment hours): x0, α = 0, merged stochastic OPF; pin on/off; α = 0.5 on P | merged = Σ ω_s single-scenario optimum to 1e-6; (Q_pin − Q_free)/Q ≤ 1e-5 | 0.5–2 h | existing |
| 4 | **ADMM on the same blocks:** pinned, F1 + V-2, F1 + V-1, α = 0 and 0.5, two initialisations | \|C_ADMM − C_central\| ≤ max(3 bar, 2τ); cycle ratio 1.3–3.0; ≥ 80 % of the top-20 residual scenario-hours are curtailment hours; cycles to f_max = 0.2 ≤ 1.5× the exit cycles. **Stop if the ratio > 3 or the gate is not met** | ≈ 10–20 h | **needed** |
| 5 | **Storage case,** single scenario: merged network with one storage against the coordinated run. **TV in operation:** a 2035-cohort plan, κ on and off | cost within the bar; throughput change 0–10 %; F_on ≤ F_off-repriced + σ | 2–3 h | existing |
| 6 | **Staged integrity:** ESSO-only A/B (a 2025 cohort with and without a 2030 cohort); one full single-scenario staged evaluation | the earlier cohort's SoH chain unchanged; SoH replica to 1e-9; H3 residual ≤ 1e-8 | 1–1.5 h | existing |
| 7 | **Concurrency gate** with staging and TV (EV, 3 cycles, timing fields excluded in advance) | bitwise | 0.5 h | existing |
| 8 | **Pilot:** EV pairs on both grids, native and common-base draws; SP on the SRP1 grid: x0, unit and a staged plan, to settling; five-block SP x0, unit and staged if memory and time allow | SP SRP1-grid 6.5–15 h, 11–13 GiB; five-block 10–30 h, 20–22 GiB (memory gate may fail); \|V_SP − V_EV\| < σ (low confidence); \|V_(a)(u) − V_pinned(u)\| < σ | EV 3–5 h; SP 20–45 h (+ 32–90 h on five blocks) | **needed** |

**Decision point after step 8** (the author and the Expert, with measured t_SP and memory):
- **Grid:** A6.
- **Form of the SP arm:**
  - (i) a full SP search;
  - (ii) search on a reduced SP ensemble and evaluate on the full one (an instance decision);
  - (iii) the minimum SP deliverable: one complete SP poll at x_EV plus the 2 × 2 cross table. This is labelled "x_EV is or
    is not mesh-local optimal under SP" and is **not** a comparable-effort search.

## Projected cost of the first 2.0× searches (to be replaced by step 8's measurements)

| item | 2025/2030/2035 | five blocks |
|---|---|---|
| EV search (0.9–1.5 h per evaluation, 4.2× throughput, 60–120 evaluations) | 16–50 h | 40–120 h |
| SP search, complete polls (6.5–15 h per evaluation, concurrency 1) | 390–1,800 h (minimum ≈ 290–675 h) | 470–3,600 h |
| common evaluation (3 plans, partly cached) | 0–45 h | 0–90 h |

**Before the first searches the author must authorize:** steps 4 and 8, then the search form after step 8 (X9 / A6)
with its measured hours, and each SP launch.

## Starting now, within the existing authorization

1. Step 0, zero-solve.
2. Step 1 implementation, behind default-off options, with its identity gates.
3. Step 2, the R8 params key, implemented with its default unchanged; its use waits on X7.

Steps 3, 5, 6 and 7 follow as each is ready. Step 4 and the pilot wait for authorization.

## Manuscript (after results)

A recommendation on which tables, figures and sensitivities to replace or retain follows once step 8 and the first
searches report. One disclosure is owed now, whatever is ruled: in the archived 3 × 3 section, state that deviation
energy is settled financially, with D0's measured effect.

## Risks

- **ADMM on kinked TN costs.** μ_s must travel about −π_s in the 215 curtailment scenario-hours, so the cycle ratio may
  exceed 3.
- **A staged optimum may come from H3 derating or from the terminal-value basis, not from operation.** The derating and
  repricing tables discriminate.
- **A pilot null must not be over-read** for the larger staged domain (author's point 5). D0 already suggests a small
  coupling effect on storage value: −2.6 k€ against 18 k€.
