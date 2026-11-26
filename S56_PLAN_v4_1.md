# S56 plan v4.1: timing, staging and operational uncertainty under consistent TSO–DSO coupling (2026-11-23)

**Predecessor:** `S56_PLAN_v4.md` (`c65201d1`), kept unchanged. v4.1 applies the Expert's review of v4 and of
`P5_15_S56_EXPERT_HANDBACK_2.md` (`28143ef2`) and the author's decisions 5–6 (2026-11-23); the changes are listed in §11,
and the differences between the recorded rulings and the Expert's written assessment in §12.

**v4 incorporates:**
- the Expert's rulings of 2026-11-23 on `P5_15_S56_EXPERT_HANDBACK_REPORT.md` (`c5cf38a1`), items 1–7;
- the author's decision on the form of the SP arm (the same day);
- the status of validation steps 1–4.

The ruling texts are recorded in TASKS.md (S56 section, `0d6fbfb0`).

**Status.**
- **Done:**
  - the coupling option and its gates (step 1);
  - the centralized comparator without storage (step 3);
  - the step 4 build, its diagnostics and its smoke run.
- **Running or authorized now:**
  - step 4 main run;
  - the cohort follow-up (network-side D1 switch, complementarity row, efficiency gate);
  - the memory probes;
  - the single-cohort A/B;
  - the EV replicate pairs.
- **Not yet authorized:** EV searches, SP evaluations beyond the pilot, the SP poll and the wait-and-see bracket.
  Run ids and measured hours are listed before each is asked for.
- **Archived results:** untouched.

## 1. What the study is

A planning study of investment timing, staged additions and operational uncertainty for shared storage, under
physically consistent TSO–DSO coordination.

**Plans are selected by expected-profile (EV) searches:** one search per price level (1× and 2×) on the chosen grid, from
the best start, using opportunistic polling with a complete poll at the final incumbent. The two EV control searches run
at 2×; if the 1× search makes any move, they also run at 1× and their hours are listed first (§4).
- 1× is the baseline.
- 2× is the stated price condition under which storage enters the plan.

**Plans are evaluated under the stochastic model (SP), with operation re-optimized:**
- x0, the EV-selected plan(s) and the reference unit, at 1× and at 2×;
- **one complete SP poll** around the EV-selected plan at 2×, if the measured t_SP allows. The author names the hours
  before launch;
- a **wait-and-see bracket** at both price levels: x0 and the selected plan (the reference unit at 1× if no plan is
  selected there) evaluated separately in each of the nine scenarios with the single-scenario model,
  probability-averaged, and reported beside the common-schedule SP value;
- the **dependence stress test** (comonotone re-pairing), as planned.

**The matched SP search is withdrawn** (author, 2026-11-23).

**Claims are scoped:**
- For a search's end state the claim is: "No improvement beyond the applicable comparison criterion was identified
  among the evaluated neighboring plans."
- Unresolved comparisons stay unresolved.
- **No value of the stochastic solution is claimed, no "optimal plan under uncertainty", and no global optimality.**

## 2. Formulation

Changes from v3 are in bold.

| item | decision | authority |
|---|---|---|
| TN–DN coupling | per-scenario (F1). The TN responds to the realized DN exchange. The DSO's expected exchange is the scenario-common day-ahead commitment; DN realizations are not forced onto it | author (a); Expert |
| Voltage coupling | **V-1 is the formulation of record**: a per-scenario V channel, with the pin unwired. V-2 (soft pin) is kept in step 4 as a measured, non-deciding arm. Evidence: the hard-pin cost Δ_pin is 9.7× the per-block allowance on 2035\|Summer | **Expert X4-1** |
| Row 18 | zero premium primary; reactive charge removed | Expert; author 4 |
| System cost | the internal TN–DN settlement is excluded in full (implemented, `f6bd136e`); investment, operating cost and terminal value are each counted once; consensus residuals are reported separately | author 6 |
| Scenarios | 3 market × 3 operation, paired by index: a small sample of the assumed independence model | author 1; Expert |
| Dependence stress test | comonotone re-pairing (`0f4d39ea`); fixed-plan SP evaluations after plan selection; not calibrated, not worst-case | author 3 |
| Storage cohorts | arrangement A: complete units, AC-coupled. Transfers are allowed and reported. **Within-cohort simultaneity is restricted by an active per-cohort complementarity row plus the per-cohort row p^ch_c + p^dch_c ≤ s_c**, on multi-cohort node-years. Legacy H3 is kept behind its key | Expert E1, E2, **ruling 3** |
| Storage energy decomposition | **D1 (a):** on multi-cohort node-years only the network aggregate `sess_comp` is deactivated; exact per-cohort physics live in the ESSO; P/Q consensus is unchanged; one formulation key governs both sides. **Gate:** equal η_ch and η_dch across the cohorts at every node, because the exactness of the aggregate recursion rests on it | **Expert ruling 4** |
| Closure | **D2:** every energy state returns to 0.5·E_av. Several cohorts: exact per-cohort closure in the ESSO; the network closure is then implied with zero slack. **Gate:** network closure slack ≤ τ_cl | **Expert ruling 4** |
| Re-optimisation without TV | retained for multi-cohort plans. Non-uniqueness is not invalidity | Expert E3 |
| Oldest cohort first | a possible result, not imposed | Expert E4 |
| Terminal value | in the optimisation, ESSO included, under the energy-residual convention. The converter residual is excluded explicitly. **Sensitivity, §3a** | author; **(c)** |
| Flexibility price | 1× baseline and 2× on DSO `cost_flex`, which reaches `flex_p_down` and `flex_q_down`. Downward reactive flexibility is shown as its own cost line | author 2 |
| ESSO slack penalty | 1e4 via the params key. Re-verified on the final formulation with positive controls | Expert E7 |
| Initialisation | a reactive-power slack is allowed; final acceptance checks both the P and the Q ESSO slacks | Expert E5 |

## 3. Numerical resolution and acceptance

**R and R/4.**
- **R = 10,000 €** separates determinate comparisons of total discounted planning cost.
- **R/4 = 2,500 € applies to pairs.** Steps 3–4 calibrate a single evaluation without storage. The pair rule is first
  tested in step 5.

**Composition, D3 and D4 as accepted:**
- **Budget:** V_pair = S(x) + S(x′) + |PR(x) − PR(x′)| + G(x) + G(x′) ≤ 2,500 €. This is a sum.
- **Cancellation guard:** A(x) + A(x′) ≤ 5,000 €.
- **Determinate** iff |ΔF| ≥ R, V_pair ≤ R/4, both guards pass and every physical gate passes for both evaluations.
- **PR only withholds determinacy.** It never ranks or replaces an objective.
- **PR is recomputed from the entry-level capture** (`430dfba8`): every channel times its multiplier. Production's
  per-cycle `priced_residual_pr` holds only the P-deviation price, so it is not D3's PR.

**Solver depth (Expert X4-2):**
- No configuration change.
- Calibration and acceptance use tail-depth cycles only.
- **Measured floor at fixed central interface values:**
  - at production depth, +500 to +600 € per block (2.6× the per-block allowance), one-signed;
  - at tail depth, +41 to +46 € (0.2×).
- The production-depth floor is reported as a limitation.
- **Step 5 adds** a pair (x0, reference unit) measured at production depth and at tail depth. Candidates classed
  "worse" at the exit are never confirmed at tail depth, so this pair measures what that classification can rest on.

**Per-block allowance for single-evaluation calibration (steps 3–4):** e_b ≤ (R/4)·w_b/Σ₁₂w. This is 228.64 € for
2025|Winter and 187.56 € for 2035|Summer. It is applied as a per-block gate on observables, because Boyd norms are
global.

**Resolution evidence beside V_pair (d).** The pair (x0, reference unit) is evaluated from three initialisations in the
EV model. The spread of the paired difference is reported beside V_pair. One SP replicate is weighed at the pilot.

**Physical gates.** These are additional ceilings, not evidence that the economic target is met:
- per-scenario |ΔP_s| ≤ 12.5 kW, |ΔQ_s| in MVAr, ΔV_s;
- ESSO P and Q slacks;
- ESS-channel residuals;
- every cohort constraint;
- the network closure slack;
- the efficiency-equality gate.

Transfers are monitored, not gated.

**Stop conditions** rest on inequalities that must hold, stated with the local-solution caveat. A recorded prediction
whose failure is not a defect (for example, the sign of D_b) is not a stop.

**First-order correction:** diagnostic only.

### 3a. Terminal-value sensitivity (c)

1. **Zero-solve repricing of every evaluated plan under four conventions:**
   - the current product (remaining-life fraction × SoH headroom);
   - min(remaining-life fraction, SoH headroom);
   - remaining life only;
   - the current product plus the converter residual.
2. **Re-optimisation of the selected plans** under one alternative convention: **remaining life only plus the converter
   residual** (the largest credit), so that it and the current product bracket the others.
3. **Zero-solve check:** each late cohort's credit, against its last-block operating margin extended over its
   remaining life.
4. **The no-TV levels of v3 stand:**
   - (a) fixed-dispatch repricing;
   - (b) re-optimisation for selected plans;
   - (c) re-search, only if (b) changes a ranking.

## 4. Search (EV arm only)

**Domain and neighbourhood:**
- The staged domain on the chosen grid.
- The neighbourhood is R3, plus removal of any cohort, plus diagonal resizes as **same-sign steps only** (Expert ruling
  6), plus redistribution pairs.
- It is enumerated in full and de-duplicated. |N| on 2025/2030/2035: min 3, median 39, max 75.

**Polling:**
- Opportunistic polling as in v3 §4: deterministic order, settled confirmation before acceptance, first confirmed
  improver accepted, an unsuccessful poll completed.
- **Completion cap 80** (Expert ruling 5): every plan in the domain can be polled completely.

**The search runs in this order (g):**
1. **EV screen:** one unit per node, year and duration, plus a short size ladder.
2. **Marginal-value table:** reported from the screen.
3. **Candidates** proposed from the table, each confirmed by evaluation and a complete poll.
4. **One opportunistic search per price level, from the best start** (plan v2 rule). The start pool is x0, the reference
   unit, the recomputed historical plans and the confirmed screen candidates. The marginal-value table and the candidates
   are built separately at 1× and 2×: the same start *rule* does not mean the same starts.
5. **Complete poll at the final incumbent.**

**Controls:** the two EV control searches, a complete-poll search and one with the class order reversed, at **2×**. If the
1× search makes any move, they run at 1× too, with their hours listed first (author 5).

**Reporting.** Every evaluated plan within R of the final incumbent is reported as indistinguishable from it. The
terminal report is as in v3 §4.

**SP poll.** One complete SP poll around the EV-selected plan at 2×, only if t_SP allows and with the hours named by the
author.

## 5. Grid

**The grid is the author's decision, taken from the memory probes (e), before the EV searches. Grids are not mixed.**

**Probes:** build and two cycles of the corrected 12-block SP instance on 2025/2030/2035, and of the five-block instance
(2025–2037, end 2040), each with a staged plan. Each reports its peak footprint against 20.7 GiB.

**Already measured on the two-block step 4 instances:**
- peak RSS 2.1–3.4 GiB;
- 46–52 s/cycle on two blocks.

**Rules for both grids:**
- End-of-block SoH is kept.
- The five-block 'mid' file is refused.
- **Derived instances must keep the instance's first year.** Production grows prices and loads from it, so dropping
  2025 lowers the 2035 energy price by a factor of 1/1.28.

**Grid comparison** (common-base inputs; a fixed 2025 unit against x0) as in v3 §5.

## 6. Build status

| component | status |
|---|---|
| SRP1-grid 3 × 3 instance, EV twin, injection hook and assertion | built (`e681f664`); **EV plumbing in the harness child still to build** |
| Dependence stress-test instance | built (`0f4d39ea`) |
| Per-scenario coupling option (F1, V-1/V-2), Q convention, sidecar fixes | **done** (`f86ec267`, `f6bd136e`, `2c328c58`); G-a to G-f hold |
| Harness S56 per-cycle capture (primal copies, all duals, ρ, tail state, per-block Q) and grouping | **done** (`430dfba8`); pinned-arm capture in progress |
| ESSO cohort formulation, arrangement A | built (`8c9fd6ec`); **follow-up in progress:** complementarity and sum rows, D2 closure, efficiency gate, single-cohort activation option for the A/B |
| Network-side D1 switch in `configure_shared_ess_operational_state` | **in progress** (with the cohort follow-up) |
| Centralized comparator without storage | **done** (`5d870c81`, `1948bb3e`, `c7720654`) |
| Centralized comparator with storage: unit appended after the merge, **capacities fixed at the ADMM exit values, no degradation, no TV, operating cost only**, plus the DN 7 reference-generator limit (f) | to build (step 5) |
| Neighbourhood module and residual scale K | done (`b87819c4`); cap 80 and same-sign diagonals ruled |
| EV screen, marginal-value table, opportunistic driver, determinism gate | to build |
| TV repricing under four conventions; late-cohort credit check | to build (zero-solve) |
| Diagnostic scripts that read the old ESSO penalty constant directly | fix before any run with the key set |

## 7. Validation sequence

| # | step | status |
|---|---|---|
| 0 | zero-solve gates | done |
| 1 | coupling option and identity gates | **done** |
| 2 | ESSO slack-penalty key; solved re-verification on the final formulation | key done; solved check after the cohort follow-up |
| 3 | centralized comparator, x0, α = 0 | **done** (see the note below the table) |
| 3b/3c | ESSO cohort tests; decomposition and aggregation tests (circulation example), with the complementarity row and the efficiency gate | 3b done (`a1ee41b6`); **3c in progress** |
| 4 | ADMM on two blocks (instance A 2025\|Winter; instance B 2025 + 2035 Summer). Arms: **V-1 (calibration), V-2 (measured, non-deciding)**, pinned AA-off and AA-on. Calibration on tail-depth cycles. D_b ≥ 0 is a recorded prediction, not a stop | **authorized; main run next.** Diagnostics D-1 to D-3 and the smoke are done (`1948bb3e`, `6633f35f`) |
| 4b | **memory probes (e)** | **authorized**, after step 4 (machine) |
| 4c | **single-cohort A/B (b):** reference unit, EV model; the "on" arm is **exactly the multi-cohort construction at n = 1** (SoC, window, closure, complementarity, sum row, converter circle, the Q slack and the aggregate-circle handling); network side unchanged. \|dF\| reported against R/4 **and against the 4d replicate spread** (4d runs first or together), with throughput (EFC/day) and the SoH path per block. If above R/4, a proposal to activate the rows on every node-year in the corrected mode, legacy key excepted | **authorized** |
| 4d | **EV replicate pairs (d):** x0 and the reference unit from three initialisations | **authorized** |
| 5 | **stochastic two-block pair** (x0, reference unit; V-1, α = 0) against a centralized comparator: merged network, the reference unit appended at its bus, the DN 7 limit added, the unit's **available energy and rating fixed per year at the ADMM run's exit values**; no degradation chain and no terminal value in the comparator; **operating cost only**, R/4 tested on the pair difference. Plus the **production- vs tail-depth pair offset**, and the **single-scenario TV check with a 2035-cohort plan, κ on and off** (plan v2) | to build |
| 6 | staged integrity: one full single-scenario staged evaluation, mixed duration, TV on | existing |
| 7 | concurrency gate with staging and TV, timing fields excluded in advance | existing |
| 8 | pilot: EV pairs, and SP evaluations of x0, the reference unit and a staged plan, to settling. Measures t_SP, memory and the paired-allowance calibration | authorized (EV, three-block SP) |

**Step 3 note.** Central costs per representative day:
- 2025|Winter: 175,188.68 €;
- 2035|Summer: 153,017.21 €;
- 2025|Summer: 185,674.16 €, merged only (single 12 stalls from both starts).

The merged-minus-singles spread is +0.035 €/day on 26 pairs.

**Report-back stop point:**
- step 4 against its predictions;
- the memory probes;
- the step 5 pair;
- the A/B;
- the replicate spread;
- this plan.

What was measured is kept separate from what is projected.

## 8. Cost projection for this study form

**These are projections, except where marked measured.** They are serial machine hours unless stated.
- **t_EV:** 0.9–1.5 h per evaluation. EV runs up to 5-way concurrently (measured throughput ≈ 4.2× at 5-way,
  single-scenario).
- **t_SP:** 6.5–15 h per evaluation on the three-block grid, **unmeasured**. One SP evaluation at a time, subject to
  probe (e).

| item | basis | hours |
|---|---|---|
| Step 4 main run and continuation | design; smoke measured 46–52 s/cycle on two blocks | 5–12 (inside the authorized 10–20) |
| Memory probes (e), two instances | build + two cycles; the 12-block 3 × 3 took ≈ 25 min for two cycles (G-f r3, measured) | ≈ 1–2 |
| Single-cohort A/B (b) | 2 EV evaluations | 2–3 |
| EV replicate pairs (d) | 6 EV evaluations, up to 5-way | 2–4 elapsed |
| Step 5 pair, two blocks, with its centralized solves | as step 4, two candidates | 5–12 |
| EV screen | (nodes × years × durations + ladder) EV evaluations; count from the module before launch | ≈ 10–25 elapsed at 5-way (projection) |
| EV searches, 1× and 2× | opportunistic, 5-way speculative; 15–50 h each (v3) | 30–100 |
| EV control searches (complete-poll; reversed order) | complete polls, median \|N\| 39, 3–6 moves | 45–110 |
| SP evaluations: x0, EV-selected plan(s), reference unit × {1×, 2×} | 6–8 × t_SP | 40–120 |
| Wait-and-see bracket | 2 plans × 9 scenarios × {1×, 2×} single-scenario evaluations | 7–11 elapsed at 5-way |
| Dependence stress test | 2 fixed-plan SP evaluations | 13–30 |
| SP poll around the EV plan at 2× | \|N\| of that plan (13–75; median 39) × t_SP | ≈ 85–1,125; named by the author from the measured t_SP |

**Excluding the SP poll, the study projects at about 150–400 machine hours.** Most of the range comes from t_SP and the
number of EV moves. **Before each item that is not yet authorized,** the Planner lists run ids and measured hours.

## 9. Changes from v3

1. The matched SP search is withdrawn. The SP arm is evaluation, one optional poll, the wait-and-see bracket and the
   dependence stress test (§1).
2. V-1 is the formulation of record; V-2 is measured in step 4 only (§2).
3. The decomposition (D1) and closure (D2) are decided. The complementarity and sum rows are active. The
   efficiency-equality gate is added (§2).
4. D3 and D4 are accepted. PR is recomputed from the entry-level capture. Calibration is on tail-depth cycles. R/4
   applies to pairs (§3).
5. The D_b sign is a recorded prediction, not a stop. Stop conditions rest on must-hold inequalities (§3).
6. The TV sensitivity is extended to four conventions plus the late-cohort check (§3a).
7. The search adds the EV screen, the marginal-value table, cap 80 and same-sign diagonals. Every plan within R of the
   incumbent is reported indistinguishable (§4).
8. The grid is decided by the author from the memory probes, before the EV searches (§5).
9. New validation items: 4b memory, 4c single-cohort A/B, 4d replicates. Step 5 is redefined as a stochastic pair
   against a centralized solve with storage, with the DN 7 limit in the comparator (§7).
10. Projection redone (§8).

## 10. Manuscript

- **Wording** as in §1: no value-of-stochastic-solution claim and no "optimal plan under uncertainty".
- **The archived 3 × 3 disclosure draft** (`S56_ARCHIVED_3X3_DISCLOSURE_DRAFT.md`) stands. Step 4's D_b gives the
  two-block re-optimized counterpart of the 0.93 M€ figure. It is reported against its own bars and without sign
  expectations beyond the recorded prediction.
- **D_b** is a two-block figure; in any manuscript text it is not a counterpart of the 12-block 0.93 M€ without that
  qualification.
- **Limitations to state** (Expert assessment §2, §5; no change to the formulation): degradation is constrained, not
  optimized — no agent prices capacity lost to fade (the networks take available energy as a parameter reset between
  cycles; the ESSO objective is the feasibility penalty plus κ × the salvage credit), so the dispatch is not
  degradation-optimal; end-of-block SoH (one mid-block comparison moved a margin by 7.8 k€); open-loop staging, four
  representative days, two value streams, one test system; investment-cost uncertainty acts only through its mean; all
  dates fixed at the start, so no value of waiting is measured; a common storage schedule gives a lower bound on
  storage value ("pays" carries over to adaptive storage, "does not pay" does not).
- **Keywords and abstract:** with the SP search withdrawn, "two-stage stochastic programming" goes; the plan is selected
  on expected profiles and evaluated under scenarios.
- **Replace or retain:** a recommendation follows once the EV searches and the SP evaluations report.

## 11. Changes from v4

1. EV search: one search per price level from the best start; screen candidates join the start pool; marginal-value
   table and candidates built separately at 1× and 2× (Expert 1).
2. Step 5 comparator corrected: capacities fixed at the ADMM exit values, no degradation chain, no TV, operating cost
   only; R/4 on the pair difference. TV stays in the single-scenario check with a 2035-cohort plan, κ on and off
   (Expert 2). Planner code check agrees with the Expert's reading: network capacities are parameters reset from the
   ESSO estimate between cycles (`shared_resources_planning.py` l. 6811–6812 at `c2ea95f6`); the ESSO objective is the
   feasibility penalty plus κ × salvage credit (`shared_energy_storage_data.py` l. 930–938).
3. Single-cohort A/B: the on arm is exactly the multi-cohort construction at n = 1; EFC/day and SoH path reported;
   |dF| read against the 4d replicate spread (Expert 3).
4. D_b qualified as a two-block figure in manuscript text (Expert 4).
5. TV re-optimisation alternative: remaining life only plus converter residual (Expert addition to §3a).
6. EV controls at 2× (at 1× too if the 1× search moves); wait-and-see bracket at both levels, reference unit at 1× if no
   plan is selected there (author 5, 6).
7. §10: limitations and keyword changes from the Expert's assessment.

## 12. Recorded rulings against the Expert's written assessment

Source: `manuscript_review/INDEPENDENT_ASSESSMENT_S56_2026-11-23.md` (`ea74a215`), written against `2b91962a`. Where it
and TASKS.md / this plan differ, TASKS.md and this plan govern (Expert 0). Listed, not re-opened.

| # | assessment | recorded ruling / plan | difference |
|---|---|---|---|
| 1 | B1: choose (i) transfers excluded or (ii) admitted **by a price-taker bound test** | D1 (a) ≡ (ii), accepted directly | the bound test is not run; the price-taker initialisation refuses > 1 cohort (`shared_ess_price_taker.py`), so it would need new tooling |
| 2 | B2: **activate cohort rows for every plan** in the corrected mode | single-cohort A/B (4c) first; activation proposed only if \|dF\| > R/4 | conditional rather than unconditional |
| 3 | §6 step 3: storage against centralized with **one single-scenario block and a staged, mixed-duration plan** | step 5 uses the reference unit (single cohort), stochastic two-block pair | no centralized check of a staged multi-cohort plan is scheduled; step 6 checks staged integrity without a comparator |
| 4 | §7: candidates proposed by a **separable surrogate and a budgeted selection** on it | marginal-value table → candidates | the selection mechanism is not specified in the plan |
| 5 | §7: **warm starts** tested on two pairs against cold starts before adoption | not recorded | not adopted, not scheduled |
| 6 | §6: grid — five blocks if the probe passes | the author decides from the probes | none in substance |
| 7 | §2: degradation constrained, not optimized; scope statements (mean-only cost uncertainty, no value of waiting, common schedule a lower bound) | not in the rulings | now in §10 as limitations |
| 8 | §8: keyword "two-stage stochastic programming" removed | not in the rulings | now in §10 |

Items B3 (within-cohort row), the efficiency gate, the four TV conventions, the late-cohort check, replicate pairs,
the wait-and-see bracket, the memory probe, 1×/2× framing and the withdrawn SP search agree with the recorded rulings.
