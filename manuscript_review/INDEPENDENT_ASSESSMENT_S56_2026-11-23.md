# Independent assessment of the S56 study (2026-11-23)

## Basis

**Read:** `TASKS.md` (S56 orders and rulings, l. 5–66), `S56_PLAN_v2.md`, the cohort and coupling proposals, `P5_15_S56_STATUS_REPORT.md`, the current and submitted `main.tex`, the reviewer comments, and the storage code in `shared_energy_storage_data.py`, `model_construction_helpers.py` and the ESSO coupling in `shared_resources_planning.py`.

**Checkout:** `2b91962a` on `feature/derivative-free-planning`, as in your brief. I could not reach the remote, so I cannot say whether the Mac Studio is ahead.

**State at this commit:**
- No cohort correction is in the code. The ESSO has no stored-energy variable (`shared_energy_storage_data.py:461–475`).
- `coupling_option_v1/` holds only its frozen spec. There are no step 3 or step 4 results.
- `S56_PLAN_v3.md`, cited in `TASKS.md`, is not in the checkout.

**Not inspected:** the JSON evidence files under `P515S56/` (I relied on their summaries in `TASKS.md` and the status report), the supplement, and the search driver. Nothing was run or modified.

**Evidence tags:** **[V]** verified in a file I read; **[M]** follows by reasoning from verified formulas; **[H]** hypothesis that needs a test.

## Verdict

The corrected model is the right direction, and three of your agreed choices should stand: scenario-wise exchanges, settlements excluded from system cost, and no deviation premium.

I would change four things before any expensive run:
1. Settle the storage-energy decomposition. As proposed it silently forbids the transfers that ruling E2 allows.
2. Apply the cohort rows to single-cohort plans too. Otherwise the search compares plans under two different operational models.
3. Treat installation timing as conditional on the terminal-value convention. The alternatives differ by two to three times R.
4. Replace the matched stochastic search with a stochastic evaluation of the plans the expected-profile search selects. The information structure cannot support more.

## 1. Is the research question coherent and focused?

Partly. The study now carries three questions:
- (a) Does shared storage pay under coordinated operation with degradation?
- (b) When and in what stages should it be installed, once terminal value is credited?
- (c) Does stochastic planning choose a different plan from expected-profile planning?

**(a) and (b) are what the reviewers asked for [V].** Reviewer 3, comment 1, asks for a salvage term and a reassessment of timing and sizing, and names additions in 2034 and 2037. The draft reply to Reviewer 2 promises a scenario-independent plan evaluated under scenario-dependent operation.

**(c) is a new question that no reviewer raised.** It is the most expensive part (80–855 h projected) and the least likely to give a resolvable answer (section 5). I would drop it as a search and keep the stochastic model as the evaluation model.

**The 2× flexibility price needs careful framing.** It is the level at which earlier runs gave a positive margin [V: `main.tex` l. 1510]. Presenting it as the primary case would read as choosing the assumption for the result. Keep 1× as the baseline, where the expected finding is no storage. Present 2× as the price condition under which storage enters the plan, so that timing and staging can be studied at all.

**Two scope statements belong in the paper:**
- Investment-cost uncertainty acts only through its mean, because the plan is scenario-independent and the objective risk-neutral [M].
- All dates are fixed at the start, so the study gives no value to waiting. "Staging is not preferred" cannot be read as "deferral has no option value" [M].

## 2. Blocking issues and stated limitations

**Blocking (resolve before pilots or searches):**

| # | issue | status |
|---|---|---|
| B1 | Storage-energy decomposition under circulation (section 3) | [V] code, [M] consequence |
| B2 | Cohort rows active only with two or more cohorts, so single-cohort and staged plans are evaluated under different models (section 3) | [V] proposal l. 50, 85–89 |
| B3 | Within-cohort simultaneous charging and discharging in the ESSO (section 3) | [M] |
| B4 | Scenario-wise coupling validated against a centralized solve | [V] not yet run |
| B5 | Numerical resolution: R = 10 k€ is below every comparison threshold reported so far, 12.6–18.0 k€ (`main.tex` l. 1393–1398, 1533) | [V] |

On B5: R is 1.5 × 10⁻⁵ of the 654 M€ operating cost. The useful evidence is that changing the penalty moved a *paired difference* by only 1.6 k€ (your 16 November analysis). So R may be attainable for paired single-scenario comparisons. It is unproven for the stochastic model [H].
- **Test:** repeat the pair (no storage, reference unit) from three initialisations and take the spread of the difference as the measured resolution. About 2 h in the expected-profile model; one replicate in the stochastic model costs one extra pair.
- **Arithmetic check:** a persistent 12.5 kW residual at 70 €/MWh is worth about 115 k€ over 15 years. The physical ceiling therefore does not imply R, as you noted.

**Limitations to state, not to fix:**
- **Degradation is constrained, not optimized [V, M].** The networks receive available energy as a parameter (`shared_resources_planning.py:6132–6133`). The ESSO objective is slack, a throughput regulariser, the terminal credit and the consensus terms (`shared_energy_storage_data.py:873–880`). No agent sees the value of capacity lost to fade, so wear is free until the SoH floor or the terminal credit bites. Evaluated storage value is then, if anything, understated. The paper should not describe the dispatch as degradation-optimal.
- **End-of-block SoH.** The one mid-block comparison moved the margin by 7.8 k€ (`main.tex` l. 1443–1445), close to R. Shorter blocks reduce this effect.
- **Open-loop staging, four representative days, the two modelled value streams, one test system.**

## 3. Cohort and network decomposition

**The network copies are not an outer approximation once transfers occur [V, M].**
- The network balance uses gross flows, ΔE = (η_ch C − D/η_dch)Δt, with relaxed complementarity C·D ≈ 0 and a daily closure on the aggregate (`model_construction_helpers.py:933–992`).
- The proposal leaves the networks unchanged and gives each cohort its own balance and closure in the ESSO (proposal l. 56–58, 72–74).
- Consensus is on net power only. Summing the cohort closures and subtracting the network closure gives (1/η_dch − η_ch) Σ_t min(C_t, D_t) Δt = 0.
- **Consequence:** at exact consensus no circulation is possible. A schedule with transfers can exist only inside the closure tolerance, the complementarity tolerance or the consensus residual, where the networks cost energy that the cohorts do not hold.

So the proposal as written already implements a no-transfer model, and it contradicts ruling E2.

**There are two consistent closures. Choose one explicitly.**

| | (i) transfers excluded | (ii) transfers admitted |
|---|---|---|
| Change | none in the networks; the circulation monitor becomes an acceptance gate | on multi-cohort node-years, deactivate the aggregate complementarity row in the network copies and keep the gross-flow balance |
| Why it is valid | without circulation, cohort sums satisfy every network row, so the outer-approximation claim holds [M] | with equal efficiencies, the aggregate gross-flow balance is the exact sum of the cohort balances; windows, closures, C + D ≤ ΣS and the circle remain valid sums [M] |
| Effect | conservative for staged, mixed-duration plans | the networks may over-use circulation, but the ESSO stays the physical truth and cost depends only on P and Q |
| Cost | none | two row families behind the formulation key; single-cohort identity kept |

(ii) is the smallest change that honours E2. It is equivalent to adding one loss variable m ≥ 0 to the aggregate balance.

**Decide by a cheap bound [H].** Run the price-taker schedule per cohort with and without transfers on the mixed-duration example. If the gain is below R/4, adopt (i) and state it as a limitation. Otherwise adopt (ii). The cost is seconds. My rough bound for the example in the proposal (0.072 MW of extra power for one hour a day) is up to about 25 k€ over the horizon, assuming a 100 €/MWh net spread captured every day for ten years. I would not assume it is negligible.

**B2: activate the cohort rows for every plan in the corrected mode [V, M].** Under the proposal a single-cohort ESSO has no energy state, and a staged one has windows tied to its own fading capacity. The two differ in feasible set and in how fade feeds back. A staged-versus-single comparison would then mix a modelling effect with an economic one. Keep bitwise identity only under the legacy key. This also resolves the closure-convention point, since every cohort then closes the same way.

**B3: within-cohort complementarity needs an explicit row [M].** The ESSO relies on a 10⁻⁵ throughput regulariser to keep a cohort from charging and discharging at once (`shared_energy_storage_data.py:476–539`). That argument holds while nothing rewards wasting energy. With cohort balances and closures, wasting energy can help the ESSO follow a network request it cannot otherwise close. Use the network's own relaxed row (ĉ·d̂ ≤ 10⁻⁴) per cohort.

**Other points:**
- The 10⁴ slack penalty should be re-verified after B1–B3, as you intend.
- Whether the efficiencies are equal across installation years should be asserted in a gate, since exactness of (ii) depends on it. I did not check it in the data.

## 4. Terminal value and installation timing

**The convention multiplies two depreciation measures [V].** The credit is cost × remaining-life fraction × (SoH − SoH_min)/(1 − SoH_min) (`main.tex` l. 2083–2088; `shared_energy_storage_data.py:917–931`). Under the baseline calibration both measures reach zero together at 15 years. Either one alone describes the remaining service, and the product understates it for mid-life units [M].

**Illustration (my calculation).** Take a 2035 unit at 2040 with SoH ≈ 0.875, the reference unit's value after its first block.

| convention | factor | credit for the 2035 reference unit |
|---|---|---|
| product (current) | 0.667 × 0.583 = 0.39 | 56.5 k€ [V: `main.tex` l. 1562] |
| minimum of the two | 0.583 | ≈ 85 k€ |
| remaining life only | 0.667 | ≈ 97 k€ |
| converter residual added | | + ≈ 24 k€ per 0.25 MVA [V: `TASKS.md` step 0] |

**These differences are two to four times R.** The existing timing table already shows the dependence: 2025, 2030 and 2035 give 64.4, 56.7 and 54.5 k€ after salvage, unresolved, and the ordering is set by the credit (`main.tex` l. 1560–1562).

**What this supports.** A timing statement that holds across conventions is defensible. "The optimal installation year is X" is not.

**Requirements:**
1. Reprice every evaluated plan under the four conventions above (zero-solve). You plan part of this already.
2. Re-optimize the selected plans under at least one alternative convention. Repricing at fixed dispatch does not show that the plan survives.
3. Add a consistency check (zero-solve): compare each late cohort's credit with the value its last-block operating margin would give over its remaining life. If the credit exceeds that value, the plan is carried by salvage and not by operation.

**Cycling older cohorts first.** The credit prices wear at 0, about 2 and about 4 €/MWh for the 2025, 2030 and 2035 cohorts [V: proposal l. 16]. A replacement-cost measure gives roughly 10 €/MWh (my calculation from k = 35,851 and 254 k€/MWh). Wear is therefore cheap for every cohort and free for the oldest.
- Oldest-first is then a rational outcome of the convention, bounded by converter, window and SoH floor, as your ruling says.
- Report per-cohort throughput, and describe the pattern as a consequence of the horizon-end convention, not as a physical recommendation.

**Energy-only credit and duration.** Excluding the converter favours long-duration late additions. Check whether the selected plan's duration changes under the converter-included repricing.

## 5. What the scenario structure can show

**A common storage schedule gives a lower bound on storage value [M].** Any scenario-adaptive schedule can reproduce the common one. So Q(x) with adaptive storage is at most Q(x) with a common schedule, while Q(0) is the same.
- "Storage does not pay" under a common schedule does not carry over to storage that responds to realized conditions.
- "Storage pays" does carry over.

**The stochastic and expected-profile plans are likely to coincide [M, H].** The schedule is common and prices enter linearly, so both models earn the mean-profile spread. Differences arise only at nonlinearities: curtailment, network limits, convex generation cost. The Advisor recorded the same expectation (status report l. 89–90), and D0 moved the reference unit's value by −2.6 k€ against an 18 k€ threshold [V].

**Three index-paired draws cannot separate uncertainty from sampling [M].** A difference between the two plans could come from these three draws. Standard practice needs replications or an out-of-sample evaluation (Kleywegt, Shapiro and Homem-de-Mello 2002; Mak, Morton and Wood 1999).

**Supportable claims:**
- The plan's cost and value under scenario-wise consistent operation with non-adaptive storage, on a small sample.
- Whether the expected-profile model misprices the plan (stochastic evaluation minus expected-profile evaluation).
- Sensitivity to cross-network dependence, from the planned stress test.

**Not supportable:** a value of the stochastic solution, an "optimal plan under uncertainty", or anything about tail conditions.

**One bracket I would add [H].** Evaluate no storage and the selected plan separately in each of the nine scenarios with the existing single-scenario model, and average. This is the fully adaptive (wait-and-see) value: 18 evaluations, about 5 h at five-way concurrency. Together with the common-schedule value it brackets the effect of the information structure.
- If the bracket is narrower than R, the common-schedule assumption is immaterial here.
- If it is wide, the paper must say that storage value depends on how far storage can respond to realizations.

Caveat: each scenario is treated as persisting over the horizon.

## 6. Smallest convincing package

| step | content | cost | status |
|---|---|---|---|
| 1 | Formulation: B1 decision test; B2 and B3 in the ESSO; ESSO-only tests i, ii, iii, iv of the proposal, plus a circulation case with closure in both models | seconds to minutes | in progress under the existing authorization |
| 2 | Coupling: centralized comparator and ADMM on two blocks, no storage | 10–20 h | authorized (steps 3–4) |
| 3 | Storage against centralized: one single-scenario block with one staged, mixed-duration plan | 2–3 h | existing authorization (steps 5–6) |
| 4 | Resolution: replicate pairs (section 2, B5) | ≈ 2 h expected-profile | needs your decision |
| 5 | Memory probe: build the corrected five-block stochastic model and run two cycles | minutes | needs your decision |
| 6 | Expected-profile searches at 1× and 2× on the chosen grid, with complete polls at the final incumbent and the reversed-order control | 16–120 h projected, parallel | pilot authorized; searches not |
| 7 | Stochastic evaluation of no storage, the selected plan(s) and the reference unit; then one complete stochastic poll around the selected plan at 2× if time allows | 3–4 evaluations, then ≈ 30 more | pilot authorized; poll not |
| 8 | Zero-solve repricing (section 4); wait-and-see bracket (section 5); dependence stress test (planned) | ≈ 5 h + 2 evaluations | bracket needs your decision |

**On step 5.** Peak memory is largely set at build and first factorization, so the five-block decision does not need a 10–30 h pilot [H].

**On the grid.** I would use five three-year blocks if the probe passes. That grid matches the submitted paper and Reviewer 3's dates, and reduces the end-of-block effect. The two grids' installation years are not nested, so choose before step 6 and do not mix them.

**Dropped:** the matched stochastic search and the stochastic 1× search.

## 7. Search and implementation

**Use the structure already measured [V, H].** Operating value was close to linear in capacity at node 7, and the 2× optimum exhausted the budget (your 16 November analysis; `main.tex` l. 1502, 1523).
- Start with a designed screen in the expected-profile model: one unit per node, year and duration, plus a short size ladder. About 20–30 evaluations at five-way concurrency.
- Fit a separable surrogate of marginal value per node and year, and solve the budgeted selection on it to propose candidates.
- Confirm each candidate by actual evaluation and a complete poll. Acceptance still rests on evaluated costs, so the comparison is not weakened. This is the search step of mesh adaptive direct search (Booker et al. 1999; Audet and Hare 2017).
- **Benefit:** fewer evaluations than blind polling over 37,041 plans, and a marginal-value table that is itself a publishable result.

**Report a set, not a point.** With first-improvement acceptance and a 5.6 k€ historical separation, report all evaluated plans within R of the incumbent as indistinguishable.

**Warm starts [H].** Starting a neighbour from the incumbent's consensus values and multipliers may cut cycles and correlate numerical error within a pair, which helps the resolution. It may also make results path-dependent. Test on two pairs against cold starts before adopting.

**Not worth doing now:** within-cycle parallelism and scenario decomposition of the stochastic evaluation. Both are production changes with uncertain gain.

## 8. Contributions, narrative and placement

**Contributions I would claim:**
1. A formulation linking shared-storage investment to decentralized AC TSO–DSO operation, with cohort-level storage states and dispatch-dependent cycling and calendar aging.
2. A cut-free evaluation and search procedure for the nonconvex operational problem, validated against a centralized solve.
3. Case-study findings on a 108-bus system.

**Findings the evidence could support, in order:**
- Coordination is worth far more than storage at this scale.
- Aging removes about a fifth of storage value.
- Storage enters the plan only at higher flexibility prices; the paper identifies where.
- Installation timing depends on the terminal-value convention.
- The results hold under scenario-wise consistent operation on a small sample.

**Placement:**

| where | content |
|---|---|
| Main text | problem and information structure in one paragraph; investment formulation; storage and cohort model; degradation; terminal value; solution procedure with one paragraph on numerical resolution; experimental design; four result subsections (coordination and baseline, degradation, price conditions and selected plan, timing with terminal value and uncertainty) |
| Appendix | ADMM updates, stopping test and acceptance criteria; scenario-wise coupling; terminal-value conventions; settings |
| Supplement | marginal-value table, poll records, repricing table, wait-and-see bracket, dependence stress test, archived 3 × 3 diagnostics with the disclosure |
| Response letter only | audit history, legacy allocation rule, bitwise gates, D0 details, tolerance calibration procedure |

**Length.** The current main text is about 4,500 words against 5,400 in the submission, with 2,400 words of appendix (my count, equations and tables excluded). Length is no longer the main problem. The remaining report-like feel comes from threshold and acceptance vocabulary in the results (`main.tex` l. 1296, 1307, 1415, 1511, 1525). Define the resolution once in the method, then report differences as value ± resolution.

**Keywords.** If the stochastic search is dropped, "two-stage stochastic programming" should go, and the abstract should say the plan is selected on expected profiles and evaluated under scenarios.

## Prioritized recommendation

**Retain:**
- scenario-wise exchange consistency, settlements excluded, no deviation premium;
- complete AC-coupled units (arrangement A);
- terminal value in both planning and operation;
- opportunistic polling with complete final polls;
- 1× as baseline, 2× as the stated price condition;
- the scoped wording on neighbourhood results.

**Change before expensive runs:**
1. Decide the decomposition (i) or (ii) by the price-taker bound.
2. Activate cohort rows for all plans in the corrected mode.
3. Add within-cohort complementarity in the ESSO.
4. Measure the resolution by replicate pairs; do not assume R.
5. Decide the grid by the memory probe.
6. Replace the stochastic search by stochastic evaluation plus one poll.

**Defer:**
- the value-of-stochastic-solution question;
- adaptive storage schedules beyond the bracket;
- adaptive (closed-loop) investment;
- degradation-optimal dispatch;
- within-cycle parallelism.

**Claims the evidence could support:**
- No improvement beyond the measured resolution was found among the evaluated neighbours of the selected plan.
- Storage recovers, or does not recover, its cost at each price level under coordinated operation with aging.
- The timing preference is, or is not, robust to the terminal-value convention.
- The expected-profile model prices the selected plan within, or outside, the resolution of its stochastic evaluation.

**Claims it could not support:** global or local optimality, an optimal plan under uncertainty, a value of the stochastic solution, or timing conclusions that depend on one salvage convention.

## References

- Audet, C., Hare, W. (2017). *Derivative-Free and Blackbox Optimization*. Springer.
- Booker, A. J., et al. (1999). A rigorous framework for optimization of expensive functions by surrogates. *Structural Optimization* 17.
- Kleywegt, A. J., Shapiro, A., Homem-de-Mello, T. (2002). The sample average approximation method for stochastic discrete optimization. *SIAM Journal on Optimization* 12(2).
- Mak, W.-K., Morton, D. P., Wood, R. K. (1999). Monte Carlo bounding techniques for determining solution quality in stochastic programs. *Operations Research Letters* 24.

I cited these from memory and did not look them up; please check the details before use.
