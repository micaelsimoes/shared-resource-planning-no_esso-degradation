# Revision Context — Shared Resources Planning

Repository:
`/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation`

Preparation-copy note: this document may be edited in another checkout before
being transferred to the Mac Studio. All authorized execution must nevertheless
use the Mac Studio repository path above.

## Role

Act as a technical planner and mathematical-programming reviewer for the shared energy-storage planning repository.

Read this file first. When checking the mathematical formulation, also consult `simoes_2026_revisions.pdf` where relevant and inspect the current implementation before proposing changes. Prefer reviewer-driven implementation and validation plans before production edits.

This file is the repository-wide source of context. **As of 2026-10-07 the active
authority is `PLANNER_BRIEF_2026-09-13.md` with its Addenda 1–67**; it supersedes
`COWORK_HANDOFF.md` and the P5.12-R scope formerly governed by
`LOCAL_NLP_STABILITY_PLAN.md`, which is retained as a historical record.

---

# CURRENT SOURCE OF TRUTH — 2026-10-07 (tables frozen; methods text accepted; manuscript revision under way)

Authority: `PLANNER_BRIEF_2026-09-13.md` Addenda 52–68; `STEP6_ROUND1_CORRECTIONS.md`. Tables FROZEN: `frozen_step6_tables_v1_590088fe.json` (W160).
Methods text: `export/paragraphs_v5.md` (Addendum 67). Expert's revision plan: `STEP6_REVISION_MAP.md`. Latest handoff:
`P5_15_ADDENDUM68_ROUND1_REPORT.md` (W168–W171, zero-solve). Current order: `TASKS.md` (Addendum 68 section) —
**stopped for review** on its Decisions 1–9. Sections below this one are history for their topics; where they conflict
with this section, this section governs.

## Manuscript facts established from code and case files (W164–W167)

- **Two horizons.** SRP1: 2025/2030/2035, 5-year blocks, one scenario. 3 × 3: 2025/2028/2031/2034/2037, 3-year blocks,
  3 market × 3 operation scenarios (`SRP1__s53_3x3.json`). Map §B §3 describes only the first.
- **Investment cost.** `SRP1_ESS.xlsx` (`7ce1d1ab` = HEAD) has three trajectories weighted 0.35/0.55/0.10; I(x) is
  their expectation (2025: 253,877.68 €/MWh, 256,317.32 €/MVA). The energy row is 1.25× the submitted file's (÷4 h,
  not ÷5).
- **Storage in the multi-scenario instance.** One schedule per block, common to all market × operation scenarios
  (hard non-anticipativity by aliasing); deviations on the network side; row 18 is a DSO-side interface premium and
  does not touch the storage; daily SoC closure is soft (5 % + 10⁻⁵, penalised in Q).
- **§2.3 / Appendix A as printed differ from the code** in 38 of 68 audited items (`P5_15_W166_EQUATION_CODE_AUDIT.md`);
  the rewrite must follow that audit, not the printed equations.
- **x = 0 reference `ref:7aa017f0`** (anchor of every T1 B row) was certified by settling rule v1 (W101); its status
  under v6 is unevaluated (report Decision 4).
- **Every table certificate re-decided under v6** from records (W142 `w142_v6_from_records.json`; W168 for x = 0):
  same cycle in every case (x = 0 181, unit 172, Phase B, year ladder, b_*); `pb_y2025_n5` stays excluded. In
  evaluations continued from an earlier run the holds began at that run's stopping cycle, not k0 + 1.
- **Slacks at the certified point (W169):** closure and ESSO P-net slacks sit at IPOPT's relaxed lower bound
  (−1e-8 p.u. per variable) at every current certificate; the min(pch, pdch) ESSO check is recorded, not enforced
  (≤ 3.6e-5 S).
- **Search as run ≠ Algorithm 1 as printed** (W171b T1-1…T1-5): 7 variables (P, E per node + one common year);
  n + 1 poll (s47/s51) with completion at every unit poll; search evaluations at the production exit with σ_Q
  acceptance; F2 certificate is not a positive-spanning-set certificate.
- **0.933 (3 × 3 R prediction)** = price-only ratio of discount-weighted average spreads, 3 × 3 horizon over SRP1
  horizon; on totals 0.915 (W170).
- Number checker: `p515_s53_w164_manuscript_number_check.py` — re-run each round with new declarations into
  `w164_manuscript_check/overleaf_<commit>/`.

## Certification (frozen: criterion v6, stage spec `96c23404`, extension `84775dc4`, A64 spec `44a2dce8`)

**The certification rule.** A cell certifies at the first cycle where all of these hold:
- Boyd residuals pass, after which the holds apply: AA off, tight tail on (compl 1e-6), ρ frozen;
- ≥ 3 turning points since the first residual pass;
- swings are not growing (swing floor τ/10);
- the range over the last W cycles is ≤ τ = 4,539.07 €;
- the gap clause holds: |t_sum| ≤ τ/2;
- every cycle in the window is clean (reading γ / window (a)). Clean means Optimal, or Acceptable on a primary
  attempt within 10× the tail tolerances.

There is also a monotone branch (L = 60, |step|·60 ≤ τ).

**Differences between cells.**
- Determinate between certified cells iff the margin is ≥ max(3 × larger band, 2τ).
- Uncertified form: bar 3·max(|gap|, |slack|).

## Results (gross, settlement excluded; net beside, validated by form + salvage identity, W154b)

- **The campaign:** 46 SRP1 cells (42 + the 4 Addendum 64 cells): 36 certified, 10 uncertified (5 gap clause /
  dual dead zone, 2 lapse resets, 3 growth test). Every non-clean cycle after the first residual pass is a TSO
  recovery; 2035 Spring recurs.
- **Headline:** x = 0 optimal at SRP1. The unit's value − I is −64,417 (4.93×). The coordination benefit against the
  best no-reverse-flow arm is +90.9 M€ (13.9 %), which replaces the withdrawn 18.25 %.
- **Break-even (node 7, slope b + c/4):** 171.3–192.6 k€/MWh. The margin to energy cost is ≥ 61.3 k€/MWh
  (conservative: `d_4a82a64a` treated as an interval).
- **Flexibility ladder (1 MWh unit):**
  - m = 1.5: −14.5 k, within the uncertified bar;
  - m = 1.75: +14.2 k (1.32×);
  - m = 2: +46.3 k (3.77×).

  Break-even lies between 1.5 and 1.75.
- **Energy ladder at m = 2:** every step is determinate.
- **Ageing at soh_min 0.70:** no aged arm pays (−31.6 to −73.7 k). Without ageing the unit is at break-even (−4.1 k,
  within resolution).
- **soh_min 0.50:** +4.9 k against 0.70, within resolution. Value − I at 0.50 is −59.5 k, determinate.
- **Other claims:**
  - the F2 certificate holds (12/12 neighbours positive);
  - x = 0 wins on every C and G row;
  - every Phase B certificate is determinate;
  - value − I is negative and determinate at every discount rate.

## Step 6 package decisions — ruled (Addenda 65–67)

Net substitute, the ≥ 0.95 τ count of ten, `d_36686489`, and the unsourced items were ruled in Addendum 65; the methods
text in Addenda 66–67.

The post-revision rule question is recorded: "a non-clean cycle cannot be a turning point" (`j_a11d7966`,
`d_4a82a64a`).

**Post-revision cleanup:**
- the TSO recoveries (2035 Spring, 2030 Autumn and Winter, 2035 Summer);
- DSO5 2035 Winter and DSO7 2025 Winter Acceptable exits.

**Machine-local:** `BASH_MAX_TIMEOUT_MS` = 14,400,000 in `.claude/settings.local.json` lets attached runs exceed the
tool's 2 h limit.

---

# HISTORICAL (was current 2026-09-18) — 2026-09-18 (P5.15 Step 3 CLOSED; the ADMM oracle is fixed)

Supersedes every earlier "current" section for the ADMM configuration and the Step 3 results. Authority:
`PLANNER_BRIEF_2026-09-13.md` Addenda 1–23. Closing evidence: `P5_15_S39_ORACLE_REPORT.md`,
`P5_15_S40_STEP3_CLOSURE_REPORT.md`, `P5_15_S41_STEP3_CLOSED_REPORT.md`, and the handoffs `P5_15_ADDENDUM2[0-2]_EXPERT_REPORT.md`.

## The baseline ADMM configuration — THE oracle (Track B closed)

Arm D (`s39_D`), written into `data/SRP1/SRP1_params.json` at `fb3de341`; the case file alone reproduces D bitwise.
- **Coordination:** τ = 0 (no TSO proximal term); ρ_v 0.0077 and ρ_pf 0.198 initial, V and PF residual balancing
  live; ρ_ess 0.01 with the **two-phase ESS schedule** — exempt from balancing until the ESS Boyd dual ratio is < 1 on 5
  consecutive cycles, then standard balancing, one-way; freeze after 10 unchanged cycles plus an absolute freeze at cycle
  200; cold standalone initialization; σ fixed 9.363536e7, S_ref 2.5 MVA, D5 ESSO scaling.
- **Certification bar:** all three channels inside their Boyd tolerances (ε_abs 1e-5, ε_rel 1e-4), every local solve
  successful, for **10 consecutive cycles**, cap 300. Terminal ratios and rule ten are reported, not gated.
- **At C\*:** certified at cycle 139, gross_operational_cost **650,966,975.2943751**, bar 7,898.63 (max objective step
  over the last 10 cycles). ESS exemption lifted at cycle 31; ρ_ess 0.01 → 0.015 → 0.0225, frozen from 43; ρ_pf
  0.198 → 0.132 at cycle 2, frozen from 12; no clamp on any channel.
- **Reproducibility:** τ = 0 determinism established — D reproduced bitwise over all 139 cycles three times
  (`51a5fb48`, `57d523d0`, `2e6c5570` runs).

## The R2.5 package (Addenda 22–23)

What nonconvex consensus ADMM delivers here is block-stationarity at the certified tolerance, not global optimality
(Hong, Luo & Razaviyayn, SIAM J. Optim. 2016; Wang, Yin & Zeng, J. Sci. Comput. 2019). The package:
1. **Certified consensus:** 10 plain cycles inside the Boyd tolerances.
2. **Step 3.5 — interval-hull polish gap: PASSED.** Every coupling entry bounded to the interval of the agents' achieved
   values at cycle 139; unscaled base objective; primal warm start; no multiplier import. All 48 blocks solved without
   retries; Δ = −2,012.21 (TSO −33.82, DSO −1,978.39), **|Δ|/cost = 0.000309 %** against 0.1 %; max |Δ_i| 775 on DSO7
   2035 Spring; none flagged (`2e6c5570`).
3. **Configuration-reproducibility band:** four configurations certified under the same bar reach operating points
   within 0.011 % in system cost through a generation / internal-flexibility trade-off (determinate, not stopping slack).
4. **Acknowledged DSO SMOPF multimodality:** ~0.3 %.

## Rules adopted for every campaign

- One frozen oracle configuration for every candidate in a campaign; costs from different configurations never share a
  table; every reported cost carries its bar.
- Manuscript reproducibility statement (method section, next to the stopping rule), Addendum 23 wording: at C\* four
  configurations certified under the same bar reach operating points within 0.011 % in system cost, through a
  generation/internal-flexibility trade-off; the campaign uses one frozen configuration and compares candidates only
  within it; the reported cost is that configuration's. Not an uncertainty band on candidate differences.

## Findings recorded at closure

- **Node 7 (Addendum 22's reading withdrawn; wording per Addendum 27):** the DN's interface branch at node 7 is the only
  active interface constraint (23 of 288 periods; nodes 5 and 9 peak at 51 % / 67 %); the storage is idle in those
  periods; relief comes from DSO-side flexibility; the storage's dispatch follows the networks' price and flexibility
  signals under the ageing constraints (available capacity, SoH floor); no wear price is charged. "Congestion-relief
  value" and "reason for siting" are struck. The Step 5 Phase A screen must include node-7-empty candidates.
- **The exact-fix polish (17/48 infeasible, v11) was confounded by a harness defect.** `p56a_oracle._interface_expression`
  predates Addendum 12's reparametrization: it builds the interface flow as `pc + flex_up − flex_down` and omits
  `interface_delta_p/q`, which now carries all interface deviation, so `apply_common_values` equated two unequal
  constants (the TSO violations were frozen at 0.20–0.84 pu across warm/cold/adaptive restarts). The earlier readings
  ("ill-posed at active coupling constraints"; node 7's rating explaining six TSO failures) are **not established**.
  Only the v11 exact-fix run is affected (P5.5–P5.8 consumers predate Addendum 12; the hull harness bypasses the
  helper). The Addendum 23 manuscript sentence on the exact-fix outcome needs restating; the helper is unfixed pending
  authorization.
- **Step 3.6:** the TSO whole-model clone is replaced by a lightweight capture (bitwise gate passed, `8214be0d`); the
  remaining per-cycle clone cost (~3 s) is the DSO node-7 failure snapshot. The within-cycle persistent-worker path
  measured ~1.1× (screening's 97 % / 5.09× was mis-accounted: parent state sync counted as absorbable); one bounded task
  remains, then the path is paused. Step 5 parallelism is candidate-level.

## After Step 3 (Addendum 24, 2026-09-19)

- **`p56a_oracle` helper fixed** (`2051309c`): it returns the model's own `pc_adn`/`qc_adn`. The exact-fix re-run at
  D's point: 11/12 TSO blocks solve (v11: 0/12), so the stale helper caused v11. The prediction scored 5/12: the
  rating-midpoint check measured the DSO node-7 interface branch rating, not a TSO row. Addendum 24's conditional
  manuscript clause is **not supported**; the principle sentence stands (`63a4d7b5`).
- **Persistent-worker path paused** after its bounded task: DSO node-7 clone removed (bitwise; no whole-model clone left
  on the serial per-cycle path); bound-restore fix; parallel ≈ serial wall because per-worker round trips cost about
  what they parallelize. Step 5 parallelism is candidate-level.
- **Step 3.7 Anderson acceleration: not adopted.** It certified at 109 cycles against the ≤ 80 gate; cost, decomposition
  and hull-polish items passed. The code stays in, default off, flag-off bitwise-verified. **The campaign runs D**
  (`76095561`).
- **Resource:** peak RSS of one serial certified evaluation 2.55 GB, so about 10–12 concurrent candidate evaluations
  fit on 32 GB.
- **Still open:** the superseded stages' checklists that fail by design against the new case file
  (`WORKER_REPORT_S40_CASE_FILE.md`).

## Step 4 opened — configuration-selection run (Addendum 25, 2026-09-19)

Report: `P5_15_S44_SELECTION_REPORT.md`.
- **Campaign harness built** (`p515_s44_campaign_harness.py`). Its gate passed on the Planner's ruling
  (`P5_15_S44_GATE_RULING.md`): C\* through the harness, concurrent with two others, reproduces D exactly. The alias
  tie-break is fixed (sorted by name).
- **D certifies at all four selection candidates:** C\* 139, paper's plan 139, node-7-empty 136, 2×C\* 187. Cap 500,
  zero local-solve failures; per-node zero storage is evaluable.
- **AA `keep_memory` variant** (a rejection keeps the memory) certifies at C\* in 107 against 109 for Step 3.7's arm,
  and goes forward.
- **The adoption rule reads "AA adopted":** AA certifies faster than D at all four candidates (107/116/107/180), and
  (b)–(d) pass at each. **The case-file update is held for review.** The 2×C\* margin is 7 cycles.
- **Paper scale** (5 years × 4 days × 25 scenarios): 80 blocks, each about 25× SRP1, because scenarios sit inside each
  block. The model state is about 18 GiB; the pristine snapshot clones push the build past the 24 GiB watchdog. The
  cycle was not timed.
- **Addendum 26 confirmations:**
  - the committed `SRP1_ESS.xlsx` is not confirmed as the corrected file (a newer one exists on `paper_revisions`,
    `7ce1d1ab`);
  - C\* and the paper's plan are over the €1M budget;
  - `max_capacity` caps energy (5 MWh), not power;
  - x = 0 builds;
  - ESSO capacity duals are available without extra solves; their units and sign are unestablished.
- **Node 7 wording:** "the DN's interface branch at node 7 is the only active interface constraint".

## Step 4 Phase A under way (Addendum 27, 2026-09-19). Governs the configuration, cost file and harness.

Authority: Addendum 27 (with the author's decisions) and `STEP4_DFO_METHOD.md`. Frozen spec v15:
`data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json` (`0a188005`).

**The oracle is AA-on (`keep_memory`), from the case file.**
- `data/SRP1/SRP1_params.json` carries `admm.anderson_acceleration` = {enabled, memory 5, 1e-10, keep_memory} (`b5629311`). The loader reads it; case files without the key load unchanged.
- **Re-verified:** the case-file-alone run at C\* reproduces the AA C\* evaluation: 107 cycles, 650,982,939.9389359, every trajectory field.
  - The one differing field is `terminal_salvage_value`: ×1.25, ≤ 8.4e-35 EUR, from the corrected cost file.
  - Ruled PASS in `P5_15_S45_REVERIFY_RULING.md` (`b2c86a1a`).
- **The AA saving falls with storage size:** 23 / 17 / 21 / 4 % at C\* / paper plan / node-7-empty / 2×C\*.
- **Running D from now on** needs a declared campaign spec with an explicit `enabled: False` override. An undeclared spec now refuses to run.

**Cost file.** The corrected `SRP1_ESS.xlsx` is from `7ce1d1ab` (`2cada62b`, sha256 `e17bd588…e39cd6`).
- Energy costs are exactly ×1.25; power costs are unchanged.
- It enters I(x) and the salvage reporting expression, not Q(x).
- I(x) with it (`9e623dd3`, EUR 2025):

| candidate | I(x), EUR |
|---|---|
| paper plan | 1,237,798 |
| C\* | 3,696,250 |
| lattice plan 1.5 / 3.0 | 1,146,109 (now over the €1M budget) |
| 0.25 / 0.5 | 191,018 |
| 0.25 / 1.0 | 317,957 |

- **Budget frontier:** the largest budget-feasible E per duration is the same at every node.

| duration | 2025 | 2030 | 2035 |
|---|---|---|---|
| 2 h | 2 MWh | 3 MWh | 4 MWh |
| 4 h | 3 MWh | 4 MWh | 5 MWh |

**Master constraints and objective.**
- `max_capacity` = energy ≤ 5 MWh per node.
- B = €1M, applied in Phase B only.
- F(x) = I(x) + Q(x), with Q = `gross_operational_cost`. Salvage is reported and excluded.
- Salvage is ~0 for 2025 cohorts. It is material for 2030/2035 cohorts: up to 51.6k / 95.5k EUR per MWh of residual energy.

**Probability audit** (`6b343bb2`, `P5_15_S45_PROBABILITY_AUDIT_NOTE.md`): no defect at SRP1 or at paper scale.
- The workbook's investment-cost scenario probabilities feed I(x), the budget and salvage only.
- Every operational term uses the networks' own probabilities.
- The attribute name `shared_ess_data.prob_market_scenarios` is misleading; it is left as is, since preserved pickles carry it.

**Harness (Phase A).**
- The bar is now the max |Δ gross_operational_cost| over the last 10 cycles; it was the net-recourse step.
  - Every committed bar is unchanged (max |diff| 0.0).
  - The net-step value is kept, reported as `bar_net_recourse_step_reported`.
- A case-file-AA evaluation cannot pass as a D reference.
- Error records carry the configuration fields (`c1469fab`).
- Phase A concurrency is **7**: measured peak 2.40–2.55 GiB per evaluation; about 21 GiB non-reclaimable-free.
- **The harness evaluates investment year 2025 only**: the child refuses any other year, and the spec freeze has no year. The A1 year ladder and A2 staging need a harness/gates extension before they can run. Production supports any year and multi-cohort nodes.

**Paper scale.**
- The change plan is in hand (snapshot switch `'off'`; four single-scenario paths; timed cycle).
- The scale script currently refuses under the AA-on case file until it declares AA.
- Risks:
  - post-solve memory is unmeasured;
  - the σ calibration check may trip;
  - the row-18 scenario-deviation penalties activate at 25 scenarios;
  - the paper's investment years (2025/28/31/34/37) do not map onto SRP1's.

**Order** (Addendum 27): A0 → paper-scale task → A1 (ladders, year ladder, A2, A3) → **stop for review** → Phase B.

### Phase A COMPLETE — stopped for review (2026-09-21). Report: `P5_15_ADDENDUM27_PHASE_A_REPORT.md`

- **68/68 evaluations certified** (A0 8, A1a 30, A1b 20, A2 5, A3 5); no barrier points. Five bitwise determinism
  reproductions in total, four of them across campaigns.
- **x = 0 minimises F on SRP1 under the corrected costs, within the frozen AA-on configuration.**
  - F(x = 0) = 653,859,461. The smallest margin is +52,801 (node 5, 0.25 / 0.5).
  - It holds across size, duration, node, investment year, node combinations, and gross vs net-of-salvage.
- **Q is linear and separable within resolution.**
  - value = 14,295 + 227,727·E + 51,714·P EUR (node 7, 2025, n = 16, residual rms 10,285).
  - Node combinations are additive (the triple is 1.000 of the sum).
  - Mechanism: the storage injects at the DSO reference (interface) bus, so it cannot relieve the binding DN interface branch.
- **σ_Q ≈ 10–18k (1.6–2.8e-5 of Q)** replaces the provisional 1.1e-4.
- **Break-even energy cost:** 176,576 EUR/MWh at the margin and 197,728 for the first 4 h unit. Under the *original* costs the smallest unit is indeterminate.
- **Caveats (Advisor):**
  - the margin is below the 72k between-configuration spread at C\*;
  - the end-of-block SoH convention is conservative against storage (~9 %);
  - SRP1 is single-scenario, so option value is unmeasured.
- **Paper scale.**
  - The build fits with snapshot clones off (19.6 GiB).
  - One cycle was NOT timed: both attempts ran out of memory during initialization.
  - Projection: ≥ 1,272 s per cycle, 33–54 h per certified evaluation, ~40–50 GiB. It does not fit this 32 GiB machine.
- **Pending author decisions (report §7):**
  - Phase B as a cache-served formality;
  - an SoH-convention sensitivity arm;
  - a reduced-scenario option-value probe;
  - staging (needs a multi-cohort candidate form);
  - the paper-scale route.

### Addenda 28–29: ageing batch done — stopped for review (2026-09-21). Report: `P5_15_ADDENDUM28_AGEING_REPORT.md`

- **Phase A was accepted** (Addendum 28).
- **Settled by the author:**
  - Staging is not wanted.
  - The reduced-scenario variant is deferred.
  - Paper scale goes to a ≥ 64 GiB machine for three evaluations, after the code-path generalization and row 18.
  - SRP1 concurrency is 5 (Addendum 29).
- **Ageing batch at n7 0.25 / 1.0, 2025.** Five model variants, all certified.

  | variant | value − I (EUR) | verdict |
  |---|---|---|
  | C2 | −23,084 | indeterminate by bars, negative vs σ_Q |
  | C4 | −41,004 | negative |
  | C2 + calendar fade | −44,193 | negative |
  | C3 mid-block | −53,844 | negative |
  | no ageing | **+485** | break-even |

- **The ageing convention does not decide the sign.** Available energy rises as predicted, but value is sub-proportional to it: elasticity ≈ 0.6 at fixed power.
- **Zero-solve reports:**
  - **Structural finding.** The storage is at the upstream terminal (DN bus 1) of the single constrained interface branch 1–2. It is not idle in the binding slots; it simply cannot change the constrained flow.
  - **Discount rate.** Value-to-cost is 0.905 / 0.823 / 0.726 / 0.651 at 0 / 2 / 5 / 8 %.
  - **Captured spread.** 52.1 €/MWh per cycle, against a 98 €/MWh daily 4 h spread.
- **Case file.** `max_energy_to_power_factor` is 4; the oracle is unaffected.
- **Next, after the review:**
  1. The Phase B formal record.
  2. The memory task, run alone, with its decision rule recorded in spec v16. **Stop for review** after it.

### Addenda 30–31: baseline C2 + φ_cal 0.985 + soh_min 0.70; Phase B record done — stopped for review (2026-09-22). Report: `P5_15_ADDENDUM30_PHASE_B_REPORT.md`

- **Baseline in the case file** (`2466401d`). The ESS ageing parameters now enter the evaluation key (`65525006`). C3 results are the sensitivity set and are never mixed with the baseline.
- **Re-certification.**
  - C\* certified at 87 cycles; the smallest node-7 unit at 112 (value 259,428; value − I −58,529).
  - The 0.70 floor binds in 2035 at every point.
- **A1a under the baseline:** 30 of 30 certified.
  - x = 0 still minimises F; the closest point is n5 0.25 / 1.0 at +53,607.
  - Node-7 surface: value = 10,379 + 233,136·E + 52,699·P.
  - Break-even energy cost: 182.2k €/MWh at the margin, 195–200k for the first unit.
- **Phase B formal record:** terminated at x = 0, and the unit-poll certificate holds. All 14 feasible neighbours were evaluated and all are worse; the closest is +33,459.
  - Completion ruling A2: at every unit poll, every feasible lattice point within one step of the incumbent is added to the poll.
  - Reparametrization (option b) is referred to the author.
- **The storage prices at the bus-7 TSO marginal cost** (4 h spread 80.6, against the market's 98.0).
  - Value split: TSO generation 60 %, DSO flexibility 40 %, spread across all three DNs.
  - Captured spread: 60.4 €/MWh per cycle under the baseline.
- **Paper-scale prediction:** R = 0.937 on the market spread.
- **Signal size:** 3.97e-4 of the system cost, not "four orders of magnitude".
- **Open questions:**
  - citations for the baseline values, not yet supplied;
  - the reparametrization (option b);
  - the NOMAD comparison;
  - a terminal TSO capture;
  - negative penalty-component levels.
- **Next:** the memory task (Addendum 29), then **stop for review**.

### Addenda 32–34 — memory route, price mechanism, flexibility break-even. Stopped for review (2026-09-22). Report: `P5_15_ADDENDUM32_34_REPORT.md`

- **Paper-scale route decided by the rule: FAILS on cycle time.**
  - Footprint after initialization **24.78 GiB** (threshold 26) — the memory fix works; the bookkeeping growth (≈ 140 MiB per paper DSO block, Pyomo solution copies nothing reads) is released behind a switch, default off, bitwise-gated.
  - One ADMM cycle **42.3 min** (limit 25) ⇒ ≈ 78 h per evaluation. **Route: ≥ 64 GiB machine, or the SRP1-with-caveat fallback.** A larger machine does not fix a CPU-bound cycle.
  - No memory leak exists. σ calibration passed at paper scale (0.597). One DSO block unrecovered in cycle 1.
- **What sets the storage's price** (`dd0a86b3`, `d68c814d`): in the peak hours all TN conventional units are at zero output and the bus-7 price equals the DSOs' **daily flexibility energy-balance shadow price** μ (plus the hourly flexibility price where P-down is marginal), carried in by the ADMM interface dual. Verified to ≤ 0.078 €/MWh over 270 hours; reproduces the −12.3 €/MWh flatness term (residual 6e-7). No DN-local storage exists; congestion is zero everywhere.
- **Flexibility-price ladder** (`dc468ab4`): value − I = −58,529 (×1) / −31,075 (×1.5) / **+48,396 (×2, resolution 4,847: pays)** / +116,617 (×3). Break-even is between ×1.5 and ×2, where the flexibility price (≈ 97 €/MWh) reaches the market 4 h spread (98). The predicted 65–80 €/MWh ceiling is exceeded (86.5, 102.9): DSO flexibility use is structural, not optional. **Q(0) rises 653.9M → 876.0M across the ladder — different systems, not a sensitivity band.**
- **Zero-solve findings:** negative penalty levels are IPOPT bound-relaxation residue (absolute Q biased ≤ 2e-5, differences ≤ 181 €), no unbounded slack; reactive flexibility is structurally absent (bounds 2e-5 p.u.); the storage's value is 97 % active power; voltage support is unmonetized (the 1.1 pu bound never binds at bus 7); the DSOs buy no priced flexibility in the binding hours (relief there is free P-up), so Addendum 33's congestion/shifting dichotomy captures nothing as defined.
- **NOMAD** (`2c1272f4`): installed in a separate environment; the in-house Householder/rounding/bounds/(n+1) machinery reproduces it on 144 polls. STEP4 §5.2's "double/halve as in NOMAD 4" is wrong (NOMAD uses 1-2-5).
- **Open for the author:** paper-scale route; whether a data anchor exists for the flexibility price; the congestion/shifting definition; the untested marginal-MWh prediction; STEP4 corrections; the baseline citations (spec v18 records them as proposed, author to confirm).

### Addenda 38–39 — F2 demonstration, row 18, and the multi-scenario pilot. Stopped for review (2026-09-24)

Reports: `P5_15_ADDENDUM38_REPORT.md`, `P5_15_ADDENDUM39_PILOT_REPORT.md`. Specs v21 `13cb828c`, v22 `5d8df1e8`.

- **F2 (flexibility price ×2) is the paper's demonstration case.** Every marginal MWh pays; Phase B left the 2025 budget corner for a **2030 two-node plan** (0.25/0.5 at node 5 + 1.0/3.5 at node 7, I = 992,268), improving F by 102,117 at 5.5× resolution — the method finding what a ladder cannot. The corner and cache-hit predictions were refuted. The run stopped at the completion cap with mesh-local optimality **not** established; the certificate continuation (standard unit poll, snap-to-feasible, cap 60) is specified and pending.
- **Row 18 as signed is implemented, merged and gated** (`700cf13c`): linear DSO premium inside Q(x), TSO pinned, one scenario-free storage variable with the old copies retained unwired, settlement split (deviation energy economic), voltage pin solver-only, old quadratic unwired. Three gates pass; inert at one scenario, so every committed result stands.
- **The multi-scenario pilot certified** (`f239ee02`): 5 years × 4 days × 2×2 scenarios, α = 0.50, x = 0 at 74 cycles and the smallest unit at 72, in 4 h at concurrency 2 (~3.3 min/cycle).
  - **R = 0.937 confirmed at 0.942** — a prediction recorded before any multi-scenario run, from the mean price profile alone. The mean-profile argument transfers, which strengthens the SRP1-with-caveat fallback.
  - Value 244,321 (resolution 15,511); **value − I = −73,636**, so the unit still does not pay at 2×2.
  - **Interface dispersion at α = 0.50 is substantial**: 16.1 % of mean flow at worst, peak 34.5 MW. Addendum 39's "schedule-honouring regime" premise is **refuted**.
  - Settlement split reconciles per block to 1e-15; σ ratios 0.489 / 0.479 inside the band.
- **α: the standalone α\* = 0.1 does not transfer**, and both earlier explanations are withdrawn. It is an artefact of the **initialisation** economy, where interface import is unpriced and the DSO holds its schedule by curtailing RES at a flat 1 €/MWh. Under coordination the premium suppresses **market-price arbitrage** (91 % of the deviation at α = 0; the earned covariance falls 140-fold by α = 1). **α\*(coordinated) cannot be located from 2-cycle arms** — the coordination gap dominates the premium by more than an order of magnitude — so the R3.6 row as ordered would show no transition.
- **Pre-run audit caught two real bugs**, invisible at one scenario: the Excel dispersion sheet read row 18's unwired storage copies (spurious 98.75 MW), and the S31C flexibility volumes were inflated by the scenario count. Both fixed and gated.
- **Models** (2026-09-24): Worker and Planner pinned to `claude-opus-5-5` (Planner from its next restart), Advisor on Fable 5.1.
- **Open:** the R3.6 row (settled sweep, 2–4 h, or report α = 0.50 alone); row 18 at initialisation (fix or document); the paper-scale route (route (c) 3×3 now ≈ 16 h per evaluation); the F2 certificate continuation.

### Addenda 40–41 — the initialisation fix, and the curtailment audit (2026-09-24, CLOSED)

- **Row 18 at initialisation is now STRUCTURALLY inactive, and initialisation is bitwise identical across α.**
  - The problem: the ADMM initialisation economy prices the interface import at zero, so charging a deviation premium there is inconsistent — and it made the standalone α\* ≈ 0.1 an artefact.
  - The first fix (zeroing the mutable `row18_alpha`) was **rejected on review**. With the charge zeroed but the rows and the nonnegative unbounded pair retained, each index admits a zero-cost ray (d⁺+t, d⁻+t), and the perturbed KKT system has **no solution** for any μ > 0 — stationarity forces λ = z⁺ = z⁻ = 0 against d·z = μ, so the central path does not exist. IPOPT would most likely still **succeed**, at arbitrary d. The objection is therefore determinism and confounding, not failure: α > 0 arms would have initialised from a differently-conditioned NLP than the α = 0 arm, on the very α row about to be measured.
  - **Adopted:** deactivate the defining rows and fix the pair at 0 during initialisation; at activation compute e from the initialisation solution, set the minimal split d⁺ = max(e,0), d⁻ = max(−e,0), unfix, then activate. `row18_alpha` keeps the run's value; `row18_alpha_admm` dropped — one mechanism, and a per-index structural state is directly assertable.
  - **Licensing evidence (zero-solve `.nl` probe, 4 arms):** `sha256(A) == sha256(B)` exactly, both label settings, identical `.row`/`.col` — the α = 0 build and the fixed-and-deactivated α = 0.5 build write byte-identical `.nl` files. C (Param-only) = +4nT columns, +2nT rows, as predicted.
  - **A recorded prediction was REFUTED and stands as such.** The negative control D was predicted to lose columns to `linear_presolve` substitution; it did not. `NLWriter.__call__` sets `config.linear_presolve = False` on the API production uses (`SolverFactory` → `Block.write` → `WriterFactory('nl')`), so **linear presolve never runs in production**. The hazard (D is a hard non-anticipativity row `pg_adn − expected == 0`) is real but **visible** as +2nT rows, not silent. Planner's and Advisor's shared error: both read the CONFIG default, neither checked the call-site override. The per-index "fixed implies row inactive" assertion remains the operative control.
  - **SRP1 two-cycle bitwise gate PASSED** (r2, `d5a00bd9`; merge `3343c4da`). The first real IPOPT solves through the new initialisation/activation path: **153/153** declared solves with `GUARD.verify(153) == []`, counter **36/36** calls to each new function with **0 acting** (row 18 is unwired at one scenario, so both must no-op), **0 diffs** against the committed C\* reference, 0 trajectory mismatches, 0 genuine diffs against the W48 arm (1 provenance diff, the arm name), retired-quadratic tripwire 0. Gross operational cost 740,922,817.2425401, equal to the W39/W48 arm value in `e75a575e`. **The structural fix leaves SRP1 bitwise identical.**
  - The gate needed one script repair first: `COUNTER.install()` ran BEFORE the precondition stage, so `combined_code_presence()` read the counter's pass-through wrappers via `inspect.getsource` and six body-inspecting checks failed (21/21 live, 15/21 wrapped). Fixed by arming the counter on the run-lock acquisition, between the presence stage and the arm — deliberately NOT by checking against `COUNTER._originals`, which would test saved copies instead of the code that runs, and NOT by wrapping `W10.run_arm`, whose source the capture-path checklist itself reads.
  - **Known, recorded, deliberately not fixed:** `row18_gate_addendum.json` records those six checks as `false`, because the committed S51G gate writes it after W35 returns while the counter is still installed. The field is **not gating**; both gating evaluations returned 21/21. A fix was attempted and abandoned: the gate also writes to a production eval directory under `P56A/evals/` named after the ARM, which the output root cannot redirect, so a re-run would require an arm rename that changes the gate's provenance semantics — too much risk for six non-gating values. The caveat is recorded beside the artifact instead, and the untested script change was reverted so the gate stands at exactly the version r2 verified.
- **Curtailment audit (Addendum 41, `51bd342d`, zero solves) — decision rule selects branch 1: no re-baseline.**
  - **The 1 €/MWh tie-breaker is NOT in force on the certified path.** The curtailment penalty is zeroed for the TSO and the DSO in the ADMM subproblems, so every certified Q was computed at 0. The 1 €/MWh constant applies only to the initialisation build and the uncoordinated benchmark. Addendum 41's branch 1 therefore describes a mechanism that acts on no committed result, and branch 2 would **double-count**: with the explicit penalty at 0, curtailment is already implicitly priced, because lost injection forces a compensating import settled at the scenario price.
  - **Reachable (TSO-side) curtailment is below resolution everywhere** — C/bar ≤ 0.037. DN-side is material (C/bar 4.7–9.1) but unreachable from the HV busbar. First-order effect on storage **value**: 331.5 against a resolution of 15,511 (0.021).
  - **SRP1 has essentially no surplus curtailment.** All 364 curtailed generator-hours are **capability-bound**: the inverter gives up active power to supply reactive power at its S limit, with a voltage bound active in the same network-hour for 99.6 % of the energy.
  - **The DSO7 transformer binds on IMPORT** (`270278b9`), at all 23 at-rating hours. So the transformer co-occurrence is a **consequence** of the lost RES, not its cause: reactive support → active power given up → import rises → transformer saturates. An import-constrained DN wants more local RES, not less, and HV-side storage cannot push through a saturated transformer — unreachability is established by mechanism, not only by convention.
  - **The 100 MVA rating is correct** and matches the case file; the "200 MVA" premise was wrong (that is DSO5's). No units defect, no blast radius. W55's own tolerance check FAILs at 1e-9 against a 4.4e-8 solver residual — ruled benign, FAIL left standing with the committed analysis, per the salvage-field precedent.
  - **Row 18 induces physical curtailment.** On the pilot, 44 generator-hours (1,016 MWh) are genuine below-capability curtailment, all satisfying the row-18 condition d ≤ 0 and π_s < α·π̄, at hours where π_s is 1.9–15.1 €/MWh against a ~25 €/MWh premium. Addendum 39 attributed curtail-and-reimport to the initialisation economy **only**; that attribution is now **wrong**. This is a reportable property of the proposed imbalance premium: it can trade physical waste for financial imbalance.
  - **W53 caveat recorded, not re-run:** its transformer set came from the index of `m.r`, defined over all 32 branches, so all 32 were flagged as transformers. Counts are unaffected — the only active branch row is `branch_flow_limit[31]`, the real transformer.
- **For the author:** (a) how to rule DN-side curtailment that is material but unreachable, given the recorded rule names no branch for it; (b) whether active power given up for reactive support counts as curtailment for the Art. 13 argument — 100 % of SRP1's volume turns on it; (c) confirmation of the 3×3 route.
- **F2 certificate (Addendum 39 ruling 1; campaign `s53_f2_certificate_r1`, spec `803571c0`) — THE CERTIFICATE HOLDS.** `m = 2` MODEL VARIANT, never the baseline.
  - Termination `poll_failure_at_unit_mesh`. Final incumbent **unchanged**: `y2030__n5_p0.25_e0.5__n7_p1_e3.5`, F = 810,787,758.26, I = 992,268.11, Q = 809,795,490.15, bar = 2,248.59. So Addendum 38's open ruling 1 is answered — the 2030 two-node plan the search found on leaving the budget corner **is mesh-locally optimal at the unit mesh**.
  - 1 poll; 14 directions snapped to **10 distinct feasible** points, so completion was not triggered (10 ≥ n+1 = 8); **7 new evaluations of a 60 budget**, 3 cache hits, **0 barrier points**, all 7 certified. The earlier Phase B stop (61 feasible neighbours against a cap of 30) is resolved by the standard unit poll at a fraction of the projected cost.
  - **One unresolved indeterminate, and the certificate is scoped accordingly.** `y2030__n5_p0.25_e1__n7_p1_e3` is nominally BETTER by 6,571.29 against a resolution of 18,449.66, so it is not accepted under A4 — and not excluded. Its I = 992,268.108 equals the incumbent's 992,268.108: the two plans differ only by moving 0.5 MWh from node 7 to node 5 at identical cost. **The inter-node split of a fixed budget is below the method's resolution.** The certificate states "no determinate improvement", not "no improvement".
  - σ_Q = 18,449.66 is the binding half of the threshold at m = 2 (A6; C3-era, not re-measured at m = 2).
  - **Cost outlier recorded:** `y2030__n5_p0.25_e0.5__n7_p1.25_e3_m2` took wall 10,077 s against siblings at 4,580–5,381 s — roughly twice the cycles of any other cell in its batch, and it converged rather than stalled.
- **Owed, recorded, not dropped:** re-point the s51 single-block gate to the priced economy (it would otherwise pass without testing anything); add curtailment capture to the α-row spec, in place of a separate 4 h persisted pilot run.

### Addenda 44 — the α row accepted; polish re-measurement superseded by Addenda 45–48 (2026-09-25)

Report: `P5_15_ADDENDUM40_42_REPORT.md`. Specs v24 `3ac8c185`, v25 `407a4b33`; row spec `70965374`.

- **The α row is complete and accepted.** Five certified x = 0 cells plus the unit at α = 0.5. C(α) = Q − charge monotone increasing, P(α) monotone decreasing, **all four adjacent envelope inequalities within** in both the Q and V forms; every cell settled at ≤ 6.6 % of threshold. **The α row is reported as C(α) with the charge separate**; Q alone is not a system-cost curve.
- **Two mechanisms, no threshold.** **82 % of deviation volume is removed by α = 0.1** (P 324.07M → 57.80M); market share of Σωd² falls 96.5 % → 15.3 %. Then physical waste takes over: DSO curtailment **1,720 → 97,148 MWh** between α = 0.5 and 1.0, worth **7.72M €**, about a third of the 21.6M total cost of commitment. Σωd² is **not** monotone — it rises from α = 0.5 to 1.0 while E|d| falls.
- **Resolution convention changed (Addendum 44 ruling 6): the bar-sum is the operative test**, σ_Q is provenance-only (SRP1/C3-era, never re-derived for 2×2). value = 267,549 at **16.16×**; value − I = −50,408 at **−3.05×**.
- **"R confirmed" is WITHDRAWN.** R restates to 1.0313; the recorded ±0.01 band is falsified, but the ratio resolution is **0.152**, so R − 0.937 = 0.094 is 0.62× resolution and **indeterminate**. The mean-profile argument stands as an **analytical** basis, untested at this resolution.
- **The initialisation fix did NOT determinately change the storage's value.** The +23,228 shift is 0.72× the **four-cell** bar-sum (32,062) — the governing error when comparing two value estimates. σ_Q's 1.26× is not the right bar.
- **Scaling mismatch: real, effect indeterminate.** The unit cell's TSO solves used objective scaling 0.001 against 0.0047–0.0071 for x = 0 (μ 4.545e-8 vs 1.845e-6), identical on both solves across all 480 blocks — the cause is structural (388 more complementarity pairs). First estimated at 18,790 (1.14× resolution); **corrected to 15,079 (0.91×) — INDETERMINATE**, because the first parser read the **hull-polish** solve rather than the terminal ADMM solve.
- **Addendum 44 ruling 5 CANNOT be executed as worded.** A fixed-configuration polish on the certified cells with no new ADMM runs requires `certified_models.pkl`, which is **absent in all six cells** (`persist_certified_models: False`, per the W48 memory ruling). Only **cycle-7** FrozenSMOPF snapshots and the ESSO terminal models survive; the polish never re-solves the ESSO. Options, for the author: (a) re-run with persistence + a bitwise-reproduction gate — **never demonstrated on the 2×2 instance**; (b) polish rebuilt models from a non-certified start — a different experiment; (c) record the mismatch as a quantified sub-resolution caveat and pin the scaling **before** the 3×3 run. **Pinning `obj_scaling_factor` alone does not pin the effective scale** — IPOPT multiplies it by the gradient-based factor, so `nlp_scaling_method` must also be pinned.
- **Curtailment classification corrected.** A third class `at_availability` (c ≤ k·TOL_MW, **k = 2 derived** from the barrier geometry: 1/(r+1) + 1/r = ρ with μ ≤ T gives r = (1+√5)/2) separates entries sitting at availability within the barrier offset from genuine curtailment. The headline **97,148 MWh is classification-independent**; the below-capability subset moves −0.0036 %. **"Dual-established" is withdrawn**: |dual| × slack is constant by barrier complementarity, so dual magnitude reports slack size, not bindingness — the **primal** indicator is the sound basis.
- **Art. 13 (ruling 3):** the SRP1 volumes are **"active-power reduction under reactive support at the inverter capability limit"**, not curtailment and not compensated; the α = 1.0 volumes are self-inflicted under the premium. Author verifies the legal text.
- **DN-side curtailment (ruling 2):** unreachable, no effect on value, branch 1 stands, reported as a system feature.
- **3×3 (ruling 4):** confirmed with the **prefix draw** and R = 0.9331 recorded, after the polish re-measurement. **The 5×5 pair does not run on this machine**; the memory refactor is follow-up work — §7 established it is infeasible as-is anyway (the row set itself varies by day through four data-dependent `Constraint.Skip` sites; `objective_scale` is **not** a mutable Param, contrary to the brief).
- **Harness fixes done** (`6c97a166`, `930f1076`): per-pair heartbeat files (a shared file let pair 3's launch destroy pair 2's end state); boolean gate flags in **both** producers — the campaign harness and `p515_g_g1_g4_admm_gates.py` (21 sites, none feeding a hash), verified by AST equality with a mutation test proving the check is not vacuous.
- **Stringified-boolean audit (`9f00377a`): no committed claim is wrong**, scoped. `determinate_at_gt_error_bar` has **no reader** in any `.py` file on any branch; `objective_convergence` has three truthiness consumers, all reading inputs that hold real booleans. Determinacy claims derive from floats, not from the flag. 195 `"False"` strings existed and none was read.
- **Owed:** 42(1) linear-solver benchmark, 42(3) persistent-worker timing, the positive-capacity hazard extension, D1, and the zero-solve α = 2 surplus estimate (ruling 1). **Open anomaly:** hull polish raises gross on all six cells and reverses sign on one; post-certification and excluded from Q.

---
### Addenda 45–48 — barrier gap closed, tight tail adopted, new SRP1 reference; 3×3 pair on the Mac (2026-09-25)

- **The barrier-gap question is CLOSED: there is no scaling artefact.** Measured on the terminal ADMM solves, the gap difference between the x = 0 and unit cells is sub-resolution on every pair: SRP1 Phase A R = 0.0038, SRP1 C2 baseline R = 0.0023, 2×2 complete R = 0.9102. The 2×2 figure decomposes (decomposition frozen before the harness) as **incomplete convergence +15,143.67 (100.5 %)**, scaling 0.00, pair count −77.71. Mechanism: the floor is min(tol, compl_inf_tol·s)/11; below the tol cap n·μ/s reduces to n·compl_inf_tol/11, **independent of s**. 15 of 20 x0 TSO terminal solves at 2×2 stopped 5.7–8.6× above their floors; 337 of 352 terminal solves examined reach the floor.
  - **History of the finding, recorded as such:** first reported at 1.14× resolution as a possible systematic bias (§8a of the Addenda 40–42 report) — the parser had read the hull-polish solve; corrected to 0.91×; then decomposed to zero scaling contribution. It changed explanation three times, each time smaller. The value figures were never affected.
- **The scaling pin (Addendum 45 item 1) → FALLBACK_PURE_C.** `nlp_scaling_method = user-scaling` does pin the effective factor, but **C2 fails at every value** (n7u/x0 TSO median iteration ratio 2.13–2.25 against 2.25 at baseline) and C3 fails at every value. **The between-cell difference is structural** — the unit's TSO carries 388 more complementarity pairs — and cannot be removed by matched scaling. The fallback is overdetermined. Gradient-based scaling retained.
  - **IPOPT truncates its options-list line at 254 characters** (`Snprintf(buffer, 255, ...)`); long `output_file` paths merge the next option onto the same line, then lose the use counter (205+), then the path tail (212+). Any harness reading IPOPT option lists line by line is exposed when eval ids are long.
- **Tight tail ADOPTED (Addendum 46 ruling 7, closed by Addendum 48).** `compl_inf_tol = 1e-6` on all network solves, only in the certifying cycles, switched by the same all-channels-inside-tolerance predicate that turns AA off. The ruling named 1e-4, but that is IPOPT's default inherited by the DSOs; the TSO sets 5e-4 and all early-stopping solves were TSO — so the tail applies to all network solves. **Measured:** all three SRP1 references re-certify at exactly their reference cycles (C\* 87, unit 112, x0 132); per-cycle gross **bitwise identical before the first tail cycle** (79/104/124); 48/48 terminal solves at the μ floor; retry counts unchanged. ΔQ = −780.99 / −677.22 / −729.66 (−1.04 to −1.20 × 10⁻⁶ relative); only x0's shift is resolvable beyond stopping slack (1.95×).
  - **Expert's prediction miss, owned in Addendum 48:** "within 1e-6" exceeded by 4–20 % — the premise conflated μ reaching its floor with the complementarity residual reaching its tolerance. The consistent negative sign on 3/3 cells is the **barrier-path signature: a systematic offset, not noise, which cancels in value differences.** W83's −3,400 € had the right sign and was ~5× too large.
- **New SRP1 reference R = 259,375.33** (was 259,427.77; ΔR = −52.44, ≈ 2 × 10⁻⁴ of R). **Tail scope: the reference pair only.** Phase A/B, ageing, flexibility and α cells stay as certified — the common ≈ −1.1 × 10⁻⁶ offset cancels in every value difference. The manuscript states both tolerances and the measured offset in the certification paragraph.
- **Gate G6: final scope** = the final accepted attempt per block, in the tail window and the terminal round, written into the spec for every future cell. **The two earlier re-scopings (v30, v32) were post-hoc and are recorded as such.** Standing rule restated by Addendum 48: a gate's scope is part of the frozen spec, fixed before the run. Tail acceptance rests on the three measured facts above, not on G6.
- **Bar caveat:** `bar_tail = bar_ref` bitwise on all three cells because each bar window's largest step is pre-tail, so the "two-run bar" is **not an independent measure** and the tail neither improves nor re-measures the reproducibility band.
- **`identity_holds` = False on `s47_recert`** is a stale-formula artefact (the run predates the retry term). The count reconciles on both cells, nothing depends on the flag, and the reference recomputes bitwise equal to 259,427.76775527. No False flag travels without its explanation.
- **3×3 pair: on the Mac, now.** A 3×3 child peaks at ~22 GiB without persistence, ~27 GiB with it, on 32 GiB — so **concurrency 1, sequential, ≈ 25–31 h**, and nothing else runs on the Mac meanwhile. **Option (b) adopted**: `release_solution_bookkeeping` on, measured first at zero solves (prediction ≈ 3 GiB/child). **Persistence only if the measured runtime peak with persistence ≤ 0.85 × memory available after the reboot**; otherwise the hull polish is omitted at 3×3 and reported with the SRP1 figure (3.09 × 10⁻⁶ relative). The smoke becomes a **two-arm 3-cycle bitwise comparison**, (b) on vs off, measuring the 3×3 runtime peak directly. Prefix draw [1,2,3]×[1,2,3] verified bit-for-bit on all 9,040 scenario arrays; R = 0.9331 recorded, restated against 259,375.33. **Addendum 45 ruling 5 is void** and option (c), the memory refactor, is rejected before the pair.
- **A VM exists** (64 GB, 24 cores, x86, Ubuntu 24.04, storage-limited); Addendum 47/48 set its storage discipline. First VM job is the equivalence gate, then 42(1)/(3) at paper scale.
- **Pre-reboot state (W90, W91; spec v36 `14bbddc7`, predecessor v35 `8aa98dbf`).** Option (b) measured at zero solves by rebuilding the post-solve objects from synthetic `.sol` files validated byte-for-byte against a real IPOPT pair: **saving ≈ 3.98 GiB per 3×3 child** (range 2.92–3.98). `release_solution_bookkeeping` does **not** enter `evaluation_key`, so the frozen keys carry over (x0 `f6e9cd53`, unit `c82522f4`).
  - **The persistence margin rule is footprint-consistent**: persistence iff g × P_on + T_persist ≤ 0.85 × A_post, where P_on is the **footprint high-water mark** of the (b)-on smoke child (`ri_lifetime_max_phys_footprint`, captured exactly by a launcher-side monitor with no harness change; no fallback to RSS), T_persist = 8.37 GiB (footprint), A_post = the post-reboot preflight. v35 mixed an RSS peak with a footprint transient — and the choice of measure **decided** the verdict. `g` = 1.0344 remains RSS-based (no committed full run has a footprint record); a refusal that holds at g = 1 is independent of g's basis.
  - **G6 vacuity hole closed**: at least ⌈0.9 × B⌉ floor-testable accepted final attempts in T (72/80 at 3×3). The 10 % line is a judgment and the spec says so.
  - **The 3.09 × 10⁻⁶ SRP1 hull-polish figure is provenanced**: `P515S41/hull_polish/hull_polish_results.json` (sha `809e1c91`), stored as 0.000309 %. **Caveat:** it is from the 2026-09-18 Step 3.5 configuration, not the current tight-tail α = 0.5 setup.
  - **Known and deliberately not fixed before the smoke:** the per-arm smoke preflight thresholds (bon 18.50, boff 21.41 GiB) still mix an RSS model with a footprint saving. The smoke is what **measures** the true footprint peak; the pair's preflight must be derived from the smoke's measured footprint peaks.
- **Order:** (b) zero-solve measurement → author reboots → two-arm smoke gate → 3×3 pair → **stop for review**. The Advisor's uncoordinated-benchmark definition proceeds during the pair (no compute).
- **3×3 pair launched 2026-09-26 10:20:56 UTC** (`s53_w91_3x3_pair`, spec `231558f0`) after the two-arm smoke gate PASSED: S16 bitwise identical (b) on vs off; (b) saves **4.60 GiB** by footprint (both predictions missed high); persistence refused (27.54 > 20.75, also at g = 1). The (b)-off arm's extra wall time was in the terminal workbook write, not the ADMM cycles.
- **3×3 PAIR CERTIFIED (Addenda 48–50; report `P5_15_ADDENDUM48_50_3X3_REPORT.md`).** x = 0 cycle 72, Q 842,832,534.76; node-7 cycle 69, Q 842,595,839.51. **value = 236,695 (13.2×); value − I = −81,262, determinate 4.5× — the storage does not pay at 3×3.** **R = 0.9126** against the reference 259,375.33, ratio resolution 0.1405 — indistinguishable from the recorded 0.9331.
  - **The objective converges more slowly than the residuals.** Both cells were still descending at certification (x = 0 −4,006, node-7 −1,688 €/cycle), both following one structure: Anderson acceleration off → a swing → monotone descent. **The drift predates the tight tail on both cells** (from cycles 53 and 50; tail at 64 and 61), so Addendum 50's "tail transient inside the window" reading is not supported. Rule ten cannot see it (its threshold is ~20× the step). x = 0 descends 2–2.8× faster, so continued descent shrinks the value and makes value − I more negative — the sign is robust, the size of R is not.
  - **The μ-floor criterion measures IPOPT's objective scaling S, not depth**: unscaled complementarity = scaled / S, so S = 0.001 blocks fail complementarity and clamp to the floor while S = 0.009 blocks stop one update above it. 159 of 160 terminal solves are converged by IPOPT's own definition; one (node-7 DSO7|2034|Autumn) exited at Acceptable Level with complementarity 2.93× the tolerance.
  - **H_row18 and H_B refuted on both cells; H_C mixed.**
  - **The continuation cannot run as ordered**: the terminal state is not restorable, and persisted models would not have sufficed. Route A (deterministic replay gated bitwise, then continue) ≈ 22.8 h — awaiting the author.
- **Continuation (Addendum 51, route A stage 1; report `P5_15_ADDENDUM51_CONTINUATION_REPORT.md`).** The x = 0 replay reproduced the certified run **bitwise, 72 of 72 cycles** — every AA decision and the tail engagement included — the instance's reproducibility result. After certification the objective **settled through a damped oscillation** (min −12,014 at cycle 77, rebound, settled): frozen early stop at cycle 88 recorded **4,492 €**; post-hoc damped-cosine limit **−6,235 €** (range 5.1–6.3 k€, half-period 9.15 cycles, per-cycle contraction 0.892). **Stage 2 not run**; R ∈ **[0.888, 0.913]** with the drift as uncertainty — indistinguishable from the recorded 0.9331. **x = 0 optimal under the baseline at both instances.**
  - **The early-stop rule is unsound under oscillation:** a step bound does not bound the distance to the limit (a 500 € bound admits ~1,464 € amplitude). Input to the ordered settling-criterion Advisor review.
  - **The oscillation is carried by Spring and the TSO**, spread over ~30 blocks; the DSO7 blocks earlier analyses led with belong to a separate one-directional redistribution that nearly cancels.
  - **The stringified-boolean defect (W74–W76) recurred** in new code (G14) — a fix applied per writer does not prevent recurrence.
- **Addendum 49 — uncoordinated benchmark defined** (Advisor definition, adopted with changes). Three arms at x = 0 on SRP1 — **passive** (flexibility fixed at zero), **price-taker** (consensus terms removed, λ = ρ = 0), **coordinated** (the certified cell) — each the production subproblem with coupling removed. **The paper's claim is min(passive, price-taker) − coordinated**: the sign of passive − price-taker is not guaranteed where λ_t < π_t (RES-covered hours), where a price-taker activates flexibility the system does not need. **Coordination proper is the value of pricing DN flexibility at λ_t rather than at the wholesale price**; it lives where λ_t ≠ π_t. TSO arm: interface P/Q as fixed mutable Params, **the 9 × 10¹⁰ tracking penalty removed** (it can leave the TN's economic gradient below the dual-infeasibility tolerance). Gates: common-Q reproduction of the certified x = 0 cost from persisted models; 24-solve penalty-vs-fixed check (reported); three starts per arm, **band measured on the arms** (the C3-era 0.3 % is not used); consistency re-evaluation at the TN's actual interface voltage. **The 18.25 % coordination benefit is WITHDRAWN** — measured on the transfer-payment recourse Addendum 11 retired. Nothing runs before the pair finishes; code and tests are written meanwhile, not executed.

---

# SUPERSEDED "CURRENT" SECTION — 2026-09-14 (P5.15 Step 1 closed through gates G1–G5)

This section supersedes every earlier "current" section, including the 2026-09-10/11 block below,
which is retained unedited as a historical record. Authority: `PLANNER_BRIEF_2026-09-13.md`,
Addenda 1–6. Stage reports: `P5_15_G1_REPORT.md`, `P5_15_G2_REPORT.md`, `P5_15_G3F_REPORT.md`,
`P5_15_G4_REPORT.md`, `P5_15_G5_REPORT.md`, and the four expert handoffs `P5_15_EXPERT_HANDOFF*.md`.


## Amendment — 2026-09-15 (Addenda 9–11): the prior recourse was a transfer payment. Governs all statements below.

Authority: `PLANNER_BRIEF_2026-09-13.md` Addenda 9–11. Reports: `P5_15_S30_REPORT.md`,
`P5_15_S31_PENALTY_TABLE_DRAFT.md` (signed), `P5_15_S31_BASELINE_REPORT.md`.

- **Governing finding.** 99.8 % of the pre-signature recourse was the TSO's flexibility charge on the ADN
  interface loads (penalty-table row 3), priced against an anchor fixed at model creation. At Step 3.0's first cycle the
  recourse was 2,808,928,724 and the row-3 charge 2,803,570,588; at the capped post-signature campaign's terminal point
  the charge would be 2.65e9 against a recourse of 5.18e8. **Every candidate ranking, `C*`, the 0.37 % storage effect
  and the templated bias were measured on a transfer payment, not on system cost.** Every recourse value reported before
  this amendment (including Step 3.0's 817,520,272.93) is superseded as an economic quantity.
- **Step 3.0** (`P515S30`, bitwise identical to `P515G1B`) remains the **determinism reference only**.
- **Penalty table signed** (Addendum 10): categories E, D, R, A; D terms stay in the solver objective, are excluded from
  reported Q(x), and are reported as violations (measured at bound-relaxation floor: excluding them moves Q by 0.0025 %).
  Implemented rows 3, 5, 8, 9, 14 and defects D2, D3, D6 (`fc200780`). Row 18 (soft non-anticipativity priced at
  `α·π̄`) is a Step 5 item.
- **Removing row 3 alone is not a valid formulation** (Addendum 11). The capped campaign (`P515S31_run`) did not
  converge in 90 cycles: interface primal PF residual 44.5 → 12.4× tolerance, dual PF 161 → 8.9, recourse still climbing.
  The TSO was left an unpriced direction at each interface (terminal downward interface flexibility ≈ 108–113 pu·periods
  in every block). **Remedy authorized: row 3′, the cancelling transfer** — reinstate the charge `+T` in the TSO as category
  **A-transfer** and add the symmetric revenue `−T` in the DSO on its copy of the interface variables, same price, same
  anchor (the DSO's warm-start, uncoordinated interface exchange). **Q(x) = system cost**, the transfer excluded by
  construction and reported separately per DSO as "TSO flexibility procurement cost". Interface flexibility variables are
  not fixed at zero. The converged row-3′ campaign becomes Step 3's economic baseline.

## Amendment — 2026-09-16 (Addendum 15): the coordination was stiffer than the economics. Governs the storage results.

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 15. Reports: `P5_15_S32_BOYD_GATE_REPORT.md`,
`P5_15_S33_E2_GATE_REPORT.md`, `P5_15_STEP32_EXPERT_REPORT.md`.

- **Governing finding.** In every ADMM run to date the local objective is divided by a scale
  (`effective_scale = σ / block_weight` ≈ 2.0e5–2.5e5, σ = 9.363536e7 for C\*) while the augmented-Lagrangian terms —
  duals, ρ/2 quadratics and the TSO proximal term — are added **undivided**, and the ESSO objective is not divided at
  all (defect D5). At ρ = 1 the coordination therefore outweighs the local economics by 2–3 orders of magnitude.
  The shared-ESS channel's constant-speed drift in gate `s33e2` is this asymmetry made visible: a linear arbitrage
  gradient walked down at a rate set by ρ and the normalization, with the dual residual **ρ-invariant** (it measures
  the forcing, not the motion). It is **not** a stopping-rule defect and **not** a pricing defect.
- **Every prior "storage effect" figure was measured with the storage immobilised by coordination stiffness.** This is
  a second, independent reason — beyond the transfer-payment amendment above — why the 0.37 % storage effect, the `C*`
  ranking and the templated bias cannot be read as economic results. Gate `s33e2` measured storage at **0.79 % of
  rating** and EFC/day 0.067 against a 1.4612 threshold, drifting toward arbitrage at a rate that would need ~14,900
  cycles to reach rating.
- **Mechanism evidence** (zero-solve, guards armed): throughput rises rather than falls, so the ESSO throughput term is
  not the driver; failure-induced multiplier bias is excluded; the drift correlates with the within-day price deviation
  at cos = −0.516 at every node and window (charge ≈33/MWh below the day mean, discharge ≈29 above); shared-ESS usage
  is priced at zero (row 8) and the ESSO carries no degradation cost, so only round-trip loss bounds the arbitrage.
- **Decisions in force.**
  1. **No suboptimality bound in the paper** (a convex-case result on a nonconvex recourse). Optimality evidence for
     R2.5 is: the Boyd §3.3.1 residuals with **noise-floor-derived tolerances** (ε_abs = 1e-5 from the E4 measurement:
     solver noise V 1.0e-14, PF 1.3e-8, ESS 6.6e-8), the **3.5 polish gap**, and the **count of active bounds**.
  2. **No shared-ESS usage price; row 8 stands.** After D5, ε must be **re-verified at the new scaling** by the
     detector identity and an EFC-vs-ε ablation before it is trusted.
  3. **Convergence with active voltage bounds is acceptable**, and the count is reported as a result (gate `s33e2`:
     52 of 864 terminal interface voltages within 1e-6 pu of a bound, 339 within 0.005 pu, node maxima exactly 1.1 pu).
  4. The **~30-cycle cost oscillation** (range 185,571 over cycles 101–150, 2.8 × the objective tolerance) is attributed
     in one zero-solve pass; not blocking.
  5. **Step 3.4 proceeds now with 3.3(b) folded in**, as one production commit: ESSO objective on the networks' σ
     convention (D5); σ a fixed recorded per-network constant; ρ dimensionless relative to the scaled objective,
     starting low, with balancing frozen once ρ has been unchanged for 10 consecutive cycles; ESS residual normalized
     by a fixed `S_ref` = 2.5 MVA. Then zero-solve checks, a two-cycle preflight, and one gate at cap 150 with the
     prediction recorded in advance: storage reaches its arbitrage equilibrium within ~50 cycles, EFC/day rises from
     0.067 to O(1), and all three channels pass Boyd with 3 consecutive cycles. **Pass closes 3.2–3.4 together and
     opens 3.5; fail stops for review** with the ρ, γ, residual and EFC trajectories.
- **Status of Step 3.2 gates.** `s32` (fixed γ = 1): 150 cycles, cap, no channel ever passed its dual test.
  `s33e2` (γ = τρ, ρ frozen after cycle 30, 3 consecutive cycles): 150 cycles, cap; **V certified from cycle 48**
  (partly by bound saturation), PF not certified but decaying ×0.552 per 50 cycles, **ESS not certifiable at
  ε_abs 1e-5** under the old scaling. Both gates are superseded as convergence evidence by the 3.4 re-scaling, but
  their diagnostics stand.
- **Correction (2026-09-16, same day; `P5_15_EFC_ARCHAEOLOGY_NOTE.md` §6).** The stiffness reading is **kept as a
  rate claim** and **withdrawn as a destination claim**. Storage utilisation (EFC/day) was 0.97–1.34 in every run
  through `s31` and fell to 0.0285 at `s31c`, while σ, `effective_scale`, the undivided AL terms and the unscaled ESSO
  objective were **identical** across all of them — so stiffness cannot by itself explain that change. The verified
  mechanism at `s31c` is **price deletion**: row 3 charged `cost_flex·(flex_p_down + flex_q_down)` on the TSO's
  ADN-interface loads (`model_construction_helpers.py:1692–1712`), the DSO's import is its reference generator and is
  excluded from `generation_cost` (`:1663–1670`), and the shared ESS sits in the TSO node balance (`:1329`) — so
  TSO-side storage displaced the charged volume and was implicitly remunerated by it. Removing row 3 deleted that
  remuneration. The charge was one-sided and had no DSO receipt term, which is why it was removed and must not be
  restored as-is. **Both the 0.067 and the 1.1-era figures are transients**, so neither is a target: the 3.4 gate reads
  EFC/day against a derived price-taker benchmark `EFC*`, not against 1.1. Consequently the statement above that prior
  storage figures were measured with storage immobilised must be read as: **prior storage figures were measured either
  under the transfer-payment objective that implicitly paid for cycling, or on unconverged transients, and in all cases
  with coordination stiffness limiting the rate of movement.**
- **Manuscript list additions** (append to the list below): the storage-effect figures must be regenerated after the
  3.4 gate under the corrected scaling, stating explicitly that the previous values were measured with the storage
  immobilised; the stopping rule and its noise-floor-derived tolerances are reported as method; the active-bound count
  is reported as a result; and the σ constants and ρ policy are recorded with the ADMM configuration.

## Amendment — 2026-09-16 (Addenda 16–17): the first certified evaluation, and what it does and does not certify

Authority: `PLANNER_BRIEF_2026-09-13.md` Addenda 16 and 17. Reports: `P5_15_S35REF_REPORT.md`,
`P5_15_Z2_FLOOR_SLACK_NOTE.md`, `P5_15_S35PT_GATE3_REPORT.md`, `P5_15_ADDENDUM16_EXPERT_REPORT.md`.

- **Run 1 (`s35ref`) is the programme's first certified evaluation.** Boyd stop at cycle 477 (all three channels
  passing on 475–477); **system cost 651,039,166, settled** (rule ten 0.0039). The storage channel stopped at 0.988 of
  its threshold: **EFC/day ≥ 1.059, pinned from below only.** The harness `stopped_by` field is defective for stops
  needing more than one consecutive cycle; the trajectory-derived v2 evaluator is the accepted reading.
- **Success at C\* is Boyd certification.** Addendum 16's "equilibrium at the SoH threshold with the floor active" was
  a prediction, falsified by Z2 before run 1 was launched. **At C\* storage is network-limited, not
  degradation-limited:** terminal SoH 0.659, floor dual ≈ 0, and even the price-taker upper bound leaves SoH at 0.589.
  This is a result about the candidate.
- **Insight (iii)**, the degradation shadow price, is demonstrated in Step 5 wherever the SoH floor binds (the
  k = 10,000 arm, larger or longer-cycled plans). If it binds nowhere, the paper says so and the degradation effect is
  carried by available capacity and salvage. It is not manufactured.
- **Gate 3 (price-taker initialization) read.** The schedule was initialized correctly, but the pace is set by the
  storage duals building up from zero. Initialization must therefore be **primal and dual, and history-free**.
- **Claims to avoid (adopted verbatim from `P5_15_ADDENDUM16_EXPERT_REPORT.md` §6):** EFC, storage utilisation, SoH or
  storage value "at the optimum"; a symmetric EFC bound; "cost certified to 0.03 %"; "price-taker initialization removes
  the walk"; "a different fixed point" (nothing supports one); "over-relaxation will fix storage"; gate 3's EFC per
  cohort-year as equilibrium values.
- **Manuscript list additions** (append to the list below):
  - storage use is reported as the one-sided statement "EFC/day ≥ 1.059 at the certified point" unless the Addendum 17
    gate certifies EFC;
  - the network-limited, not degradation-limited, character of storage at C\* is reported as a result;
  - the claims to avoid above apply to the manuscript verbatim.

## Amendment — 2026-09-17 (Addenda 17–20): the storage lever worked; the PF channel now sets the pace

Authority: `PLANNER_BRIEF_2026-09-13.md` Addenda 17–20. Reports: `P5_15_ADDENDUM17_EXPERT_REPORT.md`,
`P5_15_S37_RHO_ESS_REPORT.md`, `P5_15_ADDENDUM19_EXPERT_REPORT.md` (supersedes the stage report on two ESS figures).

- **Dual initialization is closed as a lever (Addendum 19).** Per-agent storage duals are identified only up to the span
  of the duplicated active storage rows. Run 1 remains the certified evaluation.
- **The storage walk speed scales as 1/ρ_ess, and residual balancing is mis-specified on the ESS channel.** The ESS dual
  residual is gradient-dominated, so balancing drove ρ_ess up. **The ESS channel is exempt from balancing.** At fixed
  ρ_ess = 0.01 / 0.001 (cap 150), the storage channel first passes at cycle 95 / 87 (run 1: 475), and EFC/day ≥ 1.06 by
  cycle 33 / 3.
- **Neither ρ_ess arm certified; the PF dual residual set the verdict.** Its ratio is ρ_ess-invariant (within 1.3 % of
  run 1 at every matched cycle; run 1 first passes PF at 226). The accepted reading is HP1 + HP2: the τ = 1 TSO proximal
  term halves the TSO step, and ρ_pf is too large late while the cycle-60 backstop froze balancing. The √2 in the PF
  dual is correct accounting, not a cause. At low ρ_ess the storage channel is **not settled** (under-damped once the
  gradient is spent). A two-phase ESS schedule is recorded for later, not adopted.
- **Policy changes (Addendum 20):**
  - the cycle-60 backstop is removed for all channels; the freeze rule is 10 unchanged cycles (with a prior action)
    plus an absolute freeze at cycle 200;
  - certification runs use cap 300, which is a budget, not a criterion; tolerances are unchanged.
- **Authorized now (frozen spec v9, `P515S38/frozen_s38_pf_pace_spec_v9_7a2b4ab7.json`):**
  - a zero-solve replay of the balancing rule;
  - arm A: τ = 0 with ρ_pf held; fallback τ = 0.25 only on a pre-registered TSO-instability trigger;
  - arm B: τ = 1 with PF balancing live.
  - Both arms run at cap 300, cold, with ρ_ess = 0.01 fixed and exempt, and per-entry PF capture as standard.
  - The combined configuration is run only if both arms help (PF first pass ≤ 180); the adopted configuration then
    becomes the oracle and Step 3.5 follows.
- **Step 3.6** proceeds in parallel, with an interim screening target of S = 3 / X = 70 %. The two "cheap wins" are
  dropped, as they are already defaults. Timing measurements never share the machine with an arm.
- **Claims to avoid:**
  - EFC 1.14 / 1.16, or any cost from the uncertified ρ_ess arms, as equilibrium or lower-cost results;
  - "ρ_ess = 0.001 / 0.01 is adopted";
  - "the proximal term is the cause" before the arms report.

## Amendment — 2026-09-17 (Addendum 21): the PF pace is fixed; new certification bar; τ = 0 adopted

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 21. Reports: `P5_15_S38_PF_PACE_REPORT.md`,
`P5_15_ADDENDUM20_EXPERT_REPORT.md`.

- **HP1 + HP2 confirmed, with HP1 the larger lever.** Arm A (τ = 0) reached a Boyd stop at cycle 133 with PF first
  passing at 131; arm B (PF balancing live) stopped at 151 with PF at 146; run 1 stopped at 477 with PF at 226. The late
  PF residual concentrates in active power at the node 7 interface; the mechanism has not been examined.
- **τ = 0 globally is the oracle setting.** The proximal term leaves the method, consistent with the paper's Algorithm 2.
  Per-channel τ is a recorded fallback, used only on evidence of TSO instability.
- **Criterion (c), storage terminal ratio < 0.9, is withdrawn.** It tests which channel closes the stop, and run 1 fails
  it at 0.988. **The certification bar from now on:** all channels inside their Boyd tolerances for 10 consecutive
  cycles. Terminal ratios and rule ten are reported, not gated. The rule-nine cost bar uses the maximum objective step
  over the last 10 cycles. **Run 1 remains certified under Addendum 16.**
- **Authorized, in order (spec v10, `P515S39/frozen_s39_oracle_spec_v10_f1b2b999.json`):**
  1. arm C: τ = 0, PF balancing live, ρ_ess 0.01 exempt, cap 300, new bar;
  2. arm D: C plus the two-phase ESS schedule. If D certifies with storage terminal ratio < 0.5, it is the production
     oracle; otherwise C is.

  In parallel: the zero-solve node 7 look, and the Step 3.6 TSO-clone replacement. Its preflight and gate come after C
  and D, followed by the 10-cycle re-measurement and then persistent workers. Once the oracle is fixed: Step 3.5 (polish
  gap), then Step 3 closes.
- **Claims to avoid:**
  - A/B cost differences to run 1 as results;
  - "criterion (c) failed" as a finding about the method;
  - "the proximal term is removed" before an oracle arm certifies without it.

### Manuscript list — regeneration obligations (open)

1. **All Table 8 costs** regenerated under the system-cost definition, with a new **transfer-payment column** (TSO
   flexibility procurement cost per DSO).
2. **The Fig. 5 generation/flexibility split** regenerated under the system-cost definition.
3. The penalty-classification table (signed) in the appendix, with category D semantics and category R for ε.
4. The ε price effect (ablation C) and the ~1.39 % spurious throughput in previously published SoH trajectories.
5. The homogeneous-fleet cohort approximation (H3).
6. Row 18's imbalance-price definition `α·π̄` and its α sensitivity (Step 5).
7. Stopping-rule definitions, ρ policy, σ constants and the polish gap (Step 3.2–3.5) — now: the 10-consecutive-cycle
   Boyd bar; the oracle's penalty table (τ = 0, two-phase ESS, freeze 10/200); σ 9.363536e7; the **interval-hull polish**
   definition and its gap (0.000309 %).
8. The reproducibility statement (Addendum 23 wording) in the method section, next to the stopping rule.
9. The exact-fix sentence — **to be restated**: the v11 exact-fix outcome was confounded by the stale `p56a_oracle`
   interface helper (see the 2026-09-18 current section).
10. Node 7 restated to what holds; "congestion-relief value" and "reason for siting" struck.
11. References: Hong, Luo & Razaviyayn (2016); Wang, Yin & Zeng (2019); Walker & Ni (2011); Fu, Zhang & Boyd (2020);
    the AA paragraph and convergence figure only if Step 3.7 is adopted.

## Amendment — 2026-09-15: Step 1 closed (Addenda 7–8). Supersedes conflicting statements below.

Authority: `PLANNER_BRIEF_2026-09-13.md` Addenda 7–8. Report: `P5_15_STEP1_CLOSING_REPORT.md`.

- **Production baseline now:** `EPS_ESSO_THROUGHPUT = 1e-5` (was 1e-3); ESSO `tol = 1e-10`, `acceptable_tol = 1e-9`
  (was 1e-8 / 1e-7); explicit `recovery.enabled` (default true, all networks and the ESSO, `case33_1` included) with a
  tier-2 retry (cold + adaptive μ); the dead `limited-memory` entries removed. Statements below that give ε = 1e-3 or
  ESSO tol = 1e-8 describe the superseded baseline.
- **H-ε supported (ablation C, frozen criteria):** at ε = 1e-3 the throughput regularization priced storage cycling at
  the ADMM fixed point, cutting year-1 EFC/day 1.112 → 0.972. **Attribution of G1's change:** cycling → ε; recourse →
  mostly Candidate 4 (ablation A, ≈90 %); Candidate 1 → neither (ablation B).
- **New baseline G1 re-run:** 71 cycles, zero local failures, 22 network failures all recovered on tier 1, year-1
  EFC/day 1.103, SoH 0.840 → 0.731 → 0.627, recourse 817,520,272.93 (within the bar of the old G1), terminal
  `lg(mu)` −11.0, spurious throughput 0.0029 %. **G3-full re-run:** 80 cycles, zero local failures; tier 2 not needed.
- **Resolved hazards:** recovery eligibility no longer depends on configuration contents (G2 converges with node 5
  eligible, 71 cycles); the failure-parser "empty rows" were the snapshot inventory, now written separately.
- **Open for Step 3:** tier 2 inside a campaign; determinism of the new baseline; G2 at ε = 1e-5; the detector reading
  0.57–0.94 of its idle prediction at `lg(mu)` = −11.0; every result at ε = 1e-3 or earlier is superseded as a baseline.

## Withdrawn

- **The `C*` feasibility-boundary claim** (P5.14-L/N) is withdrawn. The pre-reformulation infeasibility
  at 1.00 MVA was on the capacity rows (`rated_s_capacity_unit`), not a feasibility boundary; after
  Candidate 2 (investments as parameters) G3-init passes at 1.00, 1.25 and 1.62 MVA.
- **The "programme closed" verdict** (`COWORK_HANDOFF.md`) is withdrawn. The numerical programme is open.
- **Every previously reported SoH trajectory and degradation number is not reusable.** The
  pre-reformulation model enforced complementarity only through a relaxed, penalized row
  (`pch_hat·pdch_hat ≤ slack + 1e-4`); measured at the P5.14-N C\* control it carried **1.386–1.391 %
  spurious throughput** (`max min(pch,pdch)/s_max = 0.007729`). The mechanism applies to every
  pre-reformulation run; the fraction is measured only at C\*. The published
  `1.0 → 0.8387 → 0.7284 → 0.6248` (node 5's trajectory, quoted without a node) is affected.

## Established

- **Warm-start mechanism (Step 0).** The ESSO `maxIterations` failures were a warm-start policy defect
  (imported bound multipliers with 1e-9 pushes throttling the dual step), not intrinsic conditioning.
  Production pushes resolve to 1e-5 (DSO) and 1e-6 (TSO).
- **Solver and recovery policy (Step 1a).** Warm-start pushes at IPOPT defaults unless configured;
  `max_iter = 500`; one cold retry on `internalSolverError`, `maxIterations` or `infeasible`.
- **Reformulated ESSO (Step 1, `b03c9b14`).** Investments are parameters; log-domain cumulative SoH
  `soh = prev·exp(−D)·φ_cal^n`; nine slack families and the complementarity/normalization rows deleted
  (callables retained for fixture unpickling); throughput regularization `ε·Σ(pch+pdch)`, `ε = 1e-3`.
- **Set 1 network changes (Step 1b).** Candidates 1 (power-factor rows unwired), 2 (capacity variables →
  parameters; sensitivity channel retired), 4 (flexibility band as slacked equality) and 5 (single-scenario
  deviation penalties off). Candidate 3′ dropped. `_add_benders_cut` disabled entirely.
- **Multimodality.** The DSO SMOPF has local optima ~0.3 % apart in objective at material capacity
  (Candidate 3′ evidence). `Q(x)` is defined up to the local optimum the deterministic path selects.
- **Remedy (h) and the leak mechanism.** ESSO `tol = 1e-8`, `acceptable_tol = 1e-7`. The residual
  simultaneous charge/discharge is an **interior-point barrier residual set by IPOPT's terminal barrier
  parameter** (`lg(mu)` = −8.6 in every ESSO solve of G1–G4): at idle periods `x_small = μ/(s_obj·ε)`
  (≈2.5e-5 absolute), with one large leg `x_small ≈ μ/(2·s_obj·ε)`. All cycle periods in G1–G4 are
  barrier-set by `zL·x/μ`. Spurious throughput ≈ 0.009 % at C\*. **The reported quantity is the measured
  per-solve detector**; the closed-form estimate holds on the ε fixture only and is not quoted at C\*.
  The earlier "μ-insensitive" reading came from treating the summary `Complementarity` line as μ.
- **Log handling (P5.15-F, `7ca40b93`).** One fresh ESSO log per solve, stamped node/cycle, in the logs
  directory; last-match parsing; absolute `results_dir`; failure snapshots never abort a campaign.
- **H3.** Pro-rata cohort-split rows (`N_active − 1`, parameter shares) — correct and **inert on every
  current instance** (single active cohort); not tested by any gate.

## Gate outcomes (Step 1)

| gate | outcome |
|---|---|
| G5 — A1/A3/A4 agreement, reformulated ESSO | **PASS** (re-specified): net power agrees to 4.2e-16; D/SoH within the summed leak estimates |
| G1 — C\* control, reconciliation against old control | **Converged** (72 cycles, recourse 817,618,798.07, rule ten 0.891); **reconciliation FAILS**: measured ΔSoH 4.1–9.6× the leak-predicted Δ, year-1 EFC/day 1.112 → 0.972, recourse 10.05× the rule-nine bar. The reformulated model reaches a different operating point; the gate cannot attribute the change among the model changes. |
| G2 — `k = 10,000` | **FAILS**: no convergence in 90 cycles; node 5 `case33_1` 2035 Autumn failed every cycle 66–90 and was never retried. Initialization slack-dominated (SoH floor binding). Pair difference indeterminate. |
| G3-init — 1.00 / 1.25 / 1.62 MVA | **PASS** |
| G3-full — 1.62 MVA / 3.24 MWh at node 7 | **Converged** (80 cycles, recourse 819,016,107.91); **zero-failures NOT met**: one unrecovered DSO failure (node 7 `case33_2` 2035 Autumn, cycle 38; pre-solve block preserved) and one ineligible failure |
| G4 — G1 repeated | **PASS**, bitwise identical |

TSO local failures across G1–G4: every one recovered on the single cold retry.

## Open hazards and defects

1. **Recovery eligibility is decided by dead configuration.** `_is_recoverable_network_failure` requires a
   non-empty `recovery_options`; `case33_1` has none (never eligible — the driver of G2's non-convergence),
   and `case33_2`/`case33_3` hold only `hessian_approximation: limited-memory`, which the retry discards.
   **Removing those "dead" entries would silently disable recovery there.**
2. **Two preserved comparators overwritten.** `data/SRP1/Results/FrozenSMOPF/matched_success_*_cycle7.pkl`
   (audited hashes in `P3_AUDIT_REPORT.md`) were overwritten during G1 and are not recoverable. Campaigns now
   redirect `results_dir` to their own root and hash-check the shared directory.
3. **Harness reporting defects.** The detector-vs-prediction ratio mixes a `min/s_max` ratio with an absolute
   prediction (visible at `s_max ≠ 1`); the print-based failure parser emits a few empty rows (G2: 2, G3-full: 3).
4. **Pickle hashes are not byte-stable** and are not a determinism test.

## Not established

- Which of the Step-1 changes moved the C\* operating point (G1).
- Whether G2 converges if node 5's failing block is eligible for recovery.
- Whether the unrecovered G3-full block fails intrinsically or path-dependently.
- H3 on any multi-cohort instance.

## Execution discipline in force

Campaigns run as one tool-tracked background process, stderr captured, exclusive lock, per-cycle heartbeat,
fresh output root and eval id per arm, never concurrent, never detached (`CLAUDE.md`). Large per-period
capture directories are hash-recorded in `evidence_manifest_sha256.json`, not committed.

---

# SUPERSEDED SOURCE OF TRUTH — 2026-09-10, amended 2026-09-11 (superseded 2026-09-14; retained unedited)

This section supersedes every older "current", "active" or "immediate"
instruction later in this file. Those sections remain historical evidence only.

## P5.12-R pre-execution update — 2026-09-11 (supersedes the repository-state bullets below)

Repository transition. The first P5.12-R R0 preflight (`data/SRP1/Results/P512R/`
`provenance.json`, `initial_repository.json`, `runtime_identity.json`,
`accepted_hashes.json`, recorded at HEAD `dd000167` with the two governing
documents modified) is historical evidence and must not be modified. HEAD is now
`ba202e2b0e937306f3c163de2173951c1d0c24f0`, ahead 3 / behind 0 of
`origin/feature/derivative-free-planning` (no fetch). The intervening commits
change no `.py` or parameter file and cannot alter production numerics:

- `d20220dd` commits the governing-document edits whose bytes the old R0 had
  already hashed as approved modifications, plus `CLAUDE.md` (agent
  instructions, read by no code);
- `8b83b139` adds `.claude/agents/{advisor,planner,worker}.md` and
  `.claude/settings.json` (agent configuration, imported by no code);
- `ba202e2b` changes `CLAUDE.md` only.

Harness readiness. A planner pre-execution review of the drafted
`p512_r_presolve_recapture.py` (never run) found it NOT READY: `model_state()`
crashed on non-indexed Pyomo Sets present in every block (`vmag_nodes`,
`apparent_power_limited_branches`), which would have ended the single run at the
cycle-20 capture; the target-solve count was asserted, not measured; the harness
hash, HEAD, branch, staged state and tracked-path set were unchecked; the
baseline path was hard-coded to the historical R0 files. A first Worker repair
fixed these and passed a no-solve rehearsal but was REJECTED because the
end-of-run integrity and no-target-solve evidence was recorded, not enforced.
A narrow amendment added a pure `final_verdict()` so `CAPTURED` requires
end-of-run integrity, zero target guard/solve/process-launch activity and an
unadvanced target IPOPT log; its no-solve rehearsal
(`data/SRP1/Results/P512R_REHEARSAL/20260911T160420Z/`, zero solver calls and
launches, all nine verdict negative controls as expected) was ACCEPTED.

Authorized frozen harness: `p512_r_presolve_recapture.py`, SHA-256
`f0f120c26ec2c50b774ff42051c233b87fba0341e3d959faafe70283301d86f0`. It must not be
edited before or during the run.

Fresh R0 required. Because the old R0 no longer matches the tree, a new baseline
is created by the Planner at `data/SRP1/Results/P512R/R0_v2_ba202e2b/` only after
these governing-document edits are complete. Its generator script and
definition are stored inside that directory; it does not claim equivalence to
the missing original R0 generator. P5.12-C report SHA-256
`36812879af01aa3cd549b62cdb424af33a8dd080866041398a930c942e9a24b5` and the four
historical R0 file hashes are added to its protected-artifact list.

Frozen-state rule. From baseline creation until the P5.12-R report: HEAD,
branch and upstream reference unchanged; nothing staged; all tracked paths and
bytes equal to the baseline (the uncommitted governing-document edits are
recorded in its `approved_diff`); no new tracked files; harness bytes equal to
the SHA above; accepted artifacts and historical R0 files unchanged; the same
interpreter and IPOPT identity. No git operation or tracked-file edit is
permitted. Any drift stops the stage; the baseline is not regenerated without
new authorization.

P5.12-R remains capture-only. It authorizes one cold-RESCALED replay to capture
the cycle-21 target input and stop before `solver.solve`. It does not authorize
solving the target, a retry, Arm A/B, KKT analysis or any repair.

## Cycle-21 mechanism status after P5.12-R — 2026-09-11

P5.12-R completed: `P5.12-R CAPTURED`, report
`P5_12_R_PRESOLVE_RECAPTURE_REPORT.md`. The following supersedes the
causal ranking in `P5_12_B_CYCLE21_FORENSIC_REPORT.md` sections 8-9. That
report is historical evidence and is not rewritten; its recorded numerical
failure stands.

Superseded: B's `HIGH` ranking of "multiplier / bound-proximity pathology" and
the proposed bound-multiplier A/B rest on `ipopt_zL_in` / `ipopt_zU_in` values
that IPOPT never received.

Established by P5.12-R and the subsequent independent review:

- production refreshes warm-start multipliers with `_in.update(_out)`, a merge,
  so entries absent from the previous `.sol` keep older values;
- all 54 such stale entries lie on `pg`/`qg` variables whose lower and upper
  bounds are exactly equal, and which Pyomo has not marked fixed;
- `fixed_variable_treatment` is nowhere configured, so IPOPT 3.14's default
  `make_parameter` applies: the 102 equal-bound variables are removed from the
  solved problem. The `.nl` declares 9268 variables, IPOPT reports 9166, and the
  Jacobian nonzero counts differ by the same 102;
- IPOPT's own iteration-0 multiplier norms confirm the exclusion. At objective
  scaling `1e-3`, `||curr_z_L||_inf` is `101.13796` at cycle 20 and `101.16099`
  at cycle 21, matching the refreshed multiplier on a non-fixed variable
  (`slack_shared_es_soc_final_up`, about `1.011e5` unscaled). The stale
  `7.995e6` and `1.021e7` values would appear as `7995` and `10.2`.

Therefore the following are CONTRADICTED as causal hypotheses: the magnitude of
the large stale values; staleness of multiplier entries as such; and the single
entry (`qg[2,0,0,21]`) whose staleness differs between cycles 20 and 21.

The merge behaviour of `_in.update(_out)` remains a code-hygiene concern,
because stale entries would reach the solver under a different
`fixed_variable_treatment`. It is not the current failure mechanism and no
production change is authorized.

The active mechanism is UNRESOLVED. Two hypothesis classes lead:

1. the cycle-21 problem data enters a difficult or degenerate region as the
   interior-point trajectory progresses;
2. the solve is path-sensitive to small differences in the starting state.

Supporting observation, not a conclusion: both cycles follow nearly the same
path for roughly 44 iterations from the default initial barrier parameter
`mu = 0.1`, through the same objective values and the same `mu` reductions, and
diverge only once `mu` has fallen to about `2e-6`. Cycle 20 made a similar
excursion (`inf_pr` about `0.123`) and recovered; cycle 21 jammed with
persistent bound-multiplier safeguard flags and step lengths near `5e-5`.

## Arm A result — 2026-09-11: ARM A EXACT REPRODUCTION

Evidence: `data/SRP1/Results/P512ArmA/` (manifest, equality gate, IPOPT log
`4f66a7ef…9d58`, consumed NL and `.sol`, `SolverResults`, replay script SHA-256
`4379f7cb…e643`). Established:

- the frozen P5.12-R cycle-21 state is a deterministic replay fixture;
- regenerating the prepared state from the frozen before-setup capture
  reproduces the frozen after-setup state exactly under the capture contract
  (ordered-state digest `3fd294d7…a85f`, all five suffixes included);
- the NL consumed by Arm A is byte-identical to the frozen prepared NL
  (`5934341b…39a7`); the symbol map matches (`c13732e8…d2ff`);
- the effective IPOPT configuration matches except the authorized `output_file`
  redirection, which protects the frozen P5.12-R log;
- the complete Arm A IPOPT trace reproduces the P5.12-B cycle-21 trace exactly:
  0 differing lines in 243497, after normalizing only the log-file boundary
  artifact (one leading blank line in a fresh file) and the elapsed-time field.
  Iteration-0 norms, the full 3000-row iteration table, termination and the
  final unscaled objective/residual/evaluation-count block are identical;
- exactly one target solve occurred (one option echo, one `EXIT`, counter 1);
- Arm A additionally preserves the final `.sol`, which the historical run did
  not.

This establishes reproducibility of the failure, not its cause. The active
mechanism remains UNRESOLVED between the two classes recorded above.

## P5.12-K no-solve forensic result — 2026-09-12

Report `data/SRP1/Results/P512K/P5_12_K_NO_SOLVE_KKT_FORENSIC_REPORT.md`,
verdict `H_TRANSFER REJECTED — ATTRIBUTION INCONCLUSIVE`. Zero solver calls
(both counters 0); 5 of 6 Jacobian builds used, about 0.32 s each. Established:

- **H_TRANSFER is rejected.** The stationarity reconstruction is calibrated
  against IPOPT to high accuracy: relative error `9.1e-09` at the converged
  cycle-20 state (target `1.1323057973496387e-03`), `2.5e-14` at the cycle-21
  start and `6.3e-15` at the cycle-20 start. The transferred cycle-20
  multipliers are consistent with the problem they came from.
- The large initial cycle-21 stationarity residual is created primarily by the
  ADMM consensus-parameter transition, concentrated at
  `expected_interface_pf_p[23]` (100% of the residual change at its argmax).
  Its magnitude does NOT explain the failure: the successful cycle-20 start has
  essentially the same residual (`1.8369213729230112e+04` versus
  `1.8366791482401353e+04`).
- **H_PUSH is rejected** as the explanation of that residual: the bound-push
  contribution is at most `0.0999`, not `1e4`.
- The shared-ESS SOC reset is **normal per-cycle production behaviour**, not a
  cycle-21 anomaly. `configure_shared_ess_operational_state`
  (`model_construction_helpers.py:1020`) re-initializes SOC to
  `e_capacity * ENERGY_STORAGE_RELATIVE_INIT_SOC` (0.50) on every DSO
  coordination update (`shared_resources_planning.py:4543`). The cycle-20 start
  is equally flat (all 24 periods `8.942982436855709e-05`). This reset, not the
  push, drives the iteration-0 feasibility level (raw argmax
  `sess_soc_def[0,0,0,6]`, `7.59e-05` raw versus `6.62e-05` pushed).
- **Start-state diagnostics do not distinguish** the successful cycle-20 solve
  from the failing cycle-21 solve: same residual magnitude, same argmax
  variable, same flat SOC, same starting feasibility.
- The first clear behavioural divergence is downstream, around iterations
  70-77, when bound-multiplier safeguard activity begins (2924 of 3001 rows
  flagged at cycle 21 from iteration 77; 0 of 116 at cycle 20).
- Terminal violations concentrate on the reference-bus active-power balances
  and the OLTC branch: model branch index 31 is `branch_id 1`, bus 1 -> bus 2,
  `is_transformer`, `vmag_reg`; node index 0 is the type-3 reference bus 1 and
  node index 1 is bus 2. Periods 15, 8 and 9 dominate. Complementarity never
  converged: `zL*(x-l)` mean `1.77e-03` against terminal `mu = 1.84e-06`, with
  all 7390 entries above `mu`. This is an observed failure geometry, NOT a
  demonstrated cause.

Housekeeping: "mapping hash" must distinguish the harness's internal content
digest (`digest()` over the atom-encoded record, e.g. `c13732e8...d2ff`) from
the on-disk file SHA-256 of `*_mapping.json` (e.g. `b39cd878...`). They are
different hash spaces; both were verified consistent.

## P5.12-P path-sensitivity result — 2026-09-12: PATH SENSITIVITY SUPPORTED

Evidence `data/SRP1/Results/P512P/` (two variant directories with equality
gates, logs, consumed NL and `.sol`, `SolverResults`). Exactly two solves ran.

Scope of the claim: path sensitivity is established **specifically with respect
to `warm_start_bound_push` on the frozen cycle-21 fixture**. The factor-of-10
perturbations are small, modelling-invariant numerical-path perturbations; they
must NOT be described as mathematically "infinitesimal".

Verified singular perturbation: both variants' 17-key option sets differ from
Arm A in exactly `output_file` (log redirect) and `warm_start_bound_push`;
IPOPT's own echo confirms the received value while `bound_push` stayed `1e-05`;
both prepared states hash to `3fd294d7...a85f`; the consumed NL, the gate
exports, Arm A's consumed NL and the frozen capture all hash to
`5934341b...39a7`. The NLP, start point, multipliers, parameters and bounds were
byte-identically untouched.

| | Arm A `1e-5` | Variant 1 `1e-6` | Variant 2 `1e-4` |
|---|---|---|---|
| result | maxIterations | Optimal | Optimal |
| iterations | 3000 | 89 | 151 |
| final dual infeasibility | `5.16e+01` | `1.66e-03` | `1.35e-07` |
| final constraint violation | `6.19e-02` | `1.96e-07` | `6.88e-12` |
| final complementarity | `6.20e-03` | `1.06e-05` | `9.09e-06` |
| safeguard rows | 2924 of 3001, from iter 77 to 3000 | 0 | 21, iters 82-102 only |

Iteration-0 unscaled dual infeasibility is identical in all three runs
(`1.8366791482401353e+04`), while complementarity forms a clean x10 ladder with
identical mantissa (`1.0116e-01` / `1.0116e+00` / `1.0116e+01`), confirming the
knob acted as intended and only as intended.

Additional established points:

- **Safeguard activation alone is not sufficient for failure.**
- Variant 2 enters the safeguard regime near the same iteration as Arm A
  (82 versus 77) but escapes after 21 contiguous iterations and converges.
- **The discriminator is therefore persistence versus escape after entering the
  difficult regime, not merely safeguard onset.**
- Neither variant develops Arm A's terminal reference-bus / OLTC violation
  geometry, and neither reproduces its complementarity stagnation.
- **No production parameter change is justified from this single frozen
  instance.** Two data points, one per side, on one captured failure would be
  tuning to that instance.

Not established: any mechanism; well-posedness of the NLP near this point; and
whether comparable knife-edge behaviour affects other blocks, cycles or
candidates. That last question bears on oracle reliability and remains open.

## P5.12-T trajectory forensic result — 2026-09-12

Verdict `ESCAPE SIGNATURE OBSERVED — MECHANISM UNRESOLVED`. Evidence
`data/SRP1/Results/P512T/` plus `P5_12_T_TRAJECTORY_FORENSIC_REPORT.md`. Zero
solves. Established, with the trajectory facts verified independently by the
Planner from the raw logs:

- **All three trajectories undergo a comparable feasibility excursion**, each
  from a healthy state: V1 peaks at iteration 60 (`inf_pr = 2.22e-01`), Arm A at
  70 (`1.49e-01`), V2 at 78 (`1.67e-01`). All three reach `lg(mu) = -5.7` at
  iterations 36 / 40 / 40.
- **Excursion onset and magnitude do not predict the outcome.** Arm A has the
  smallest peak and is the only run that never recovers. Recovery below
  `inf_pr = 1e-2` takes 2 iterations (V1), 32 (V2), and never for Arm A.
- **The distinguishing behaviour is recovery versus persistent stall.**
- **Arm A begins deteriorating before safeguard messages appear**: `inf_pr`
  rises from `2.6e-05` (iter 65) to `1.49e-01` (iter 70) and `alpha_pr` decays
  0.43 -> 0.063 -> 0.034 -> 0.004, roughly 7 iterations before the first `z`
  marker at 77. **Safeguard activity is therefore a symptom of the stalled
  regime, not the trigger, on current evidence.** On exit, V2's markers cease at
  103, about 5 iterations before its step unlock at 108-110.
- Variant 2 enters a similar safeguard regime (iterations 82-102) and later
  escapes: markers cease -> steps recover -> full steps and residual collapse at
  110 -> `mu` advances to `-8.0` at 113 -> multipliers release.
- **Multiplier growth is retired as a candidate mechanism.** At iteration 40 all
  three runs carry transiently large norms (`z_L ~ 5.8e4`, `y_c ~ 3.3e4`),
  including both that converge; by iteration 60 all three settle to identical
  values. Arm A is characterized by subsequent **stasis**: `z_L = 1.0131e+02`,
  `y_d = 1.1516e+02`, `y_c = 6.3039e+02` held to 4-5 significant figures for
  2940 iterations.
- **No solver-telemetry quantity observed so far provides a useful
  pre-excursion predictor** of success versus failure. All observed signals are
  terminal facets of one stalled regime, not predictive.
- The Arm A terminal state is a **stalled fixed-point-like regime**: barrier
  parameter pinned at `lg(mu) = -5.7`, accepted steps ~`5.4e-05`, `||d|| ~ 1.7`,
  persistent infeasibility (`6.19e-02`) and complementarity failure
  (`6.20e-03` against `mu = 1.84e-06`).

Housekeeping: stage scripts and reports are placed inconsistently — some
reports live in the repository root (`P5_12_R`, `P5_12_T`), others in their
stage directory (`P5_12_K`). Record only; do not reorganize historical
artifacts now.

## Solver-telemetry facts carried into P5.12-G — 2026-09-12

Independent Advisor review of the proposed geometric diagnostic, with every
load-bearing claim verified by the Planner directly from the preserved logs:

- **`delta_c` and `delta_x` are different quantities and must never be combined
  into one "regularization event" count.** `delta_c` is constraint-side
  regularization; `delta_x` is primal/Hessian-side.
- **Arm A has `delta_c = 0` across all 3000 factorizations.** This is direct
  counter-evidence to strong-form constraint-Jacobian rank failure of the kind
  that would require IPOPT constraint regularization. It does **not** prove
  LICQ, full numerical rank, or good conditioning.
- Only 27 Arm A events carry nonzero perturbation, and those are **`delta_x`**,
  all before iteration 70 — i.e. before the stall. The converged controls show
  substantially MORE `delta_x` activity (36 of 89 for V1, 70 of 151 for V2, that
  is 40-46% of iterations, versus 0.9% for Arm A). **`delta_x` frequency is
  therefore not evidence for the Arm A failure mechanism.**
- The stalled step is accepted on the **first** line-search trial
  (`alpha = 5.42e-05`, `ls 1`), with sufficient reduction and filter
  acceptability both succeeding and `ALPHA_MIN = 1.06e-13` eight orders lower.
  The step is truncated, not rejected.
- Safeguard corrections are numerically tiny (~`3.4e-08` to `3.87e-08`) and must
  be quantified independently; their textual prominence in the log is not
  evidence of numerical importance.
- **Correction to the earlier geometric proposal:** an *active bound* must not
  be described as having a "near-zero gradient". Constraint-row geometry
  (anomalously small Jacobian rows on active/near-active rows) and bound
  geometry (variables near a bound, with multipliers and step-limiter evidence)
  are separate objects. A variable may be called *blocking* only if preserved
  direction/telemetry evidence supports it; otherwise it is a **near-bound
  candidate**.
- Current stronger candidate: Arm A becomes trapped by local bound/active-set
  geometry and fraction-to-boundary truncation. This remains a mechanism
  CANDIDATE, not an established cause.

Housekeeping correction: IPOPT in the current configuration **does** supply
constraint scaling — all three logs report `c scaling provided` and
`d scaling provided` alongside the `1.000000e-03` objective scaling. This
supersedes the carried-forward P5.3-A2 statement that only the objective was
scaled. Do not rewrite the historical P5.3 reports.

Authorized next: ONE bounded no-solve Tier 0 + Tier 1 forensic (telemetry, then
activity and constraint-row/bound geometry). **Tier 2 spectral/SVD analysis is
NOT authorized** and may only be reconsidered against its declared triggers. No
solve is authorized. Production repair, Arm B, crossover, multiplier-warm-start
changes and investment search remain unauthorized.

## Repository and accepted branch state

The authorized worker checkout is the Mac Studio repository:

`/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation`

The active branch remains:

`feature/derivative-free-planning`

Historical (superseded by the 2026-09-11 update above): at the independent
P5.12-C planner audit, verified again before the 2026-09-10 document update,
the Mac Studio state was:

- HEAD `dd0001675e4e38cbf4ec0282df312469bcb435e5`;
- verified unified baseline `72468913499f5f9eca77ddbef6ed71fd61be7a97` is an ancestor;
- the sole later commit, `dd000167`, preserves P5.11/P5.12-A/P5.12-B reports,
  harnesses and evidence plus `.gitignore`, with no production change;
- upstream `origin/feature/derivative-free-planning`, ahead 0 / behind 0
  against the local reference, without fetching;
- tracked and staged files clean before this authorized governing-document
  update; pre-existing untracked and ignored artifacts remain untouched;
- `P5_12_C_BOUND_MULTIPLIER_AB_REPORT.md` is untracked; no P5.12-C checkpoint
  commit or arm harness was found.

This is not a fresh verification of the remote server. The P5.11 cycle-25
diagnostic pickle remains ignored and local; its recorded hash verifies, but
it is not a cycle-21 restart point. No repository synchronization is authorized.

Historical P5.12-A end state:

- HEAD `808590d23ad250627be9ff9d4ae51d059cbdbcdb`;
- tracked working tree clean;
- upstream `origin/feature/derivative-free-planning`;
- ahead 8, behind 0;
- no merge, pull, fetch, push, rebase, cherry-pick, reset or commit performed by
  the P5.12-A worker.

HEAD moved after P5.11 through a user merge that touched only
`REVISION_CONTEXT.md` and `LOCAL_NLP_STABILITY_PLAN.md`. That merge selected the
origin documents and again pruned the explicit P5.6-D, P5.7 and P5.8 record.
The accepted lineage remains recoverable from pre-merge commit `fbdb2f6a` and
local ancestor `bff465d7`. The recovered summary below restores those decisions.

The earlier ahead-of-origin and untracked-report observations are historical;
the checkpoint state above supersedes them. Do not push or rewrite history.

## Current canonical Mac Studio runtime

The current paper-instance runtime on the Mac Studio is:

`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`

P5.11 independently enumerated ten interpreters and found exactly one matching
the accepted provenance. `/opt/anaconda3` does not exist on the Mac Studio.

Accepted runtime identity:

- Python `3.11.11`, arm64;
- NumPy `2.4.2`;
- pandas `3.0.0`;
- SciPy `1.17.0`;
- Pyomo `6.9.5`;
- copulas `0.14.0`;
- IPOPT `3.14.18` / ASL `20241111` at `/usr/local/bin/ipopt`;
- HSL `ma97`;
- canonical realized-scenario checksum
  `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`.

The old `/opt/anaconda3/envs/opf_env_py311/bin/python` path later in this file is
historical machine-specific provenance and is not the Mac Studio command.

## Recovered accepted P5.6-C through P5.9 lineage

The following accepted history was pruned by the documentation-only merge and
is restored here from the accepted pre-merge lineage.

### P5.6-C — landscape and coverage gate

Accepted verdict:

`P5.6-C-C — derivative-free search is not ready`.

T0 versus T4 Spearman correlation was `-0.033`; 6 of 9 improvement signs
reversed. Fixed continuation rescued every tested direct failure but changed
already-VALID controls by about `2.9e6`, so it could not be used only on failure
without mixing incompatible oracle surfaces. The reported P5.6-B
`se|ALL|x19` crash was withdrawn as non-reproducible. End-to-end costs were
corrected to about 135 seconds for VALID and 186 seconds for failed evaluations.

### P5.6-D — uniformly refined nonlinear oracle

Accepted verdict:

`P5.6-D-C — uniform refinement does not stabilize the investment landscape`.

Uniform refinement reached 100% coverage from `K=4` and substantially improved
direction consistency, but did not stabilize ordering. The accepted
`tau_planning_refined` was `811438.05`, the best candidate changed at every
tested depth transition, and node 9 / 2025 reversed its preferred investment
sign between `K=8` and `K=12`. No accepted `K_STAR` exists.

The best known feasible nonlinear planning incumbent from that stage was
`se|node5|2025|-10%` with total `825109566.571083`.

### P5.7 — branch-selection diagnosis

Accepted verdict:

`P5.7-A — unique operational oracle can likely be recovered`.

Objective scaling was the primary diagnosed mechanism. Production formed each
augmented subproblem objective as `base / effective_scale + consensus terms`,
with `effective_scale` about `9.41e4 .. 1.16e5`. Re-solving all 48 subproblems
from their own converged point changed the recovered base objective by
`-383401.68` under CURRENT scaling and `-8892463.76` under RESCALED scaling,
explaining about 96.5% of the ADMM-to-polish gap.

Initialization sensitivity was small by comparison: four starts of the same
polish NLP spread by only `303.24` on an objective around `8.3e8`. Active-set
differences were sparse, and continuation improved reachability rather than
fixed-consensus local quality. The earlier conjecture that cold starts used a
candidate-dependent TSO anchor was later refuted by P5.8.

### P5.8 — ADMM objective-scaling validation

Accepted verdict:

`P5.8-B — objective scaling improves stability but additional ADMM issues remain`.

`RESCALED = effective_scale * CURRENT` was verified as an exact positive
multiple on all 48 blocks. RESCALED recovered much more of the base objective
and improved the maximum unscaled constraint violation by about 70 times, but
the initial RESCALED replay still drifted, often broke exact-consensus polish,
and approached the active-power interface tolerance. The inherited
`consecutive_converged_cycles` channel was also exposed.

The then-current objective stopping threshold was about 25 times the planning
signal, while the anchor was shown to remain effectively common across starts.
No production scaling or stopping-rule change was authorized by P5.8.

### P5.9 — partial stabilization and parameter diagnosis

P5.9 carried RESCALED evaluation forward and established the facts later used
by P5.10:

- `rho_v` was numerically inert across the tested 667-fold range;
- ESS consensus remained far inside tolerance;
- the adaptive `rho_pf` rule compressed requested values 300, 500 and 1000 to
  approximately 88.89, 98.77 and 131.69 because its dual residual scaled with
  rho;
- the CURRENT eight-depth comparison still had two ranking flips and
  cross-depth relative uncertainty `459695.59` against a mean signal
  `18085.03`, an uncertainty/signal ratio of `25.42`;
- RESCALED improved coverage and the landscape but did not yet establish a
  stable, explicitly configured polished oracle.

P5.9 did not authorize production adaptive-rho changes or investment search.

## Accepted P5.10 stabilized-oracle evidence

Accepted report:

`P5_10_STABILIZED_RESCALLED_ADMM_ORACLE_REPORT.md`

SHA-256:

`fe674ee534ba9b7cf72c188b756a1ad585133727b53aad453091531fd289311e`

P5.10 made no production changes. It established an explicit harness-level
oracle configuration and identified four carried-state channels:

1. rho stored in the template;
2. objective scaling stored in cloned models;
3. `consecutive_converged_cycles`;
4. `last_recourse`.

`last_recourse` was a latent candidate-history asymmetry. Neutralizing all
history channels left the base result bit-identical and must remain part of the
declared oracle contract.

The accepted stabilized diagnostic oracle uses:

- `RESCALED` objective construction;
- fixed `rho_v = 1.5`, `rho_ess = 1.0`;
- fixed `rho_pf`, with 300 the faster working value and 1000 retained as a
  separate certification cross-check;
- adaptive rho disabled;
- histories neutralized before every evaluation/generation;
- midpoint anchor held identical across candidates;
- exact-consensus polish as the authoritative endpoint.

At fixed `rho_pf` 300, 500 and 1000, base, node-5 and node-9 candidates all
solved and polished directly with zero failed blocks. `rho_pf = 300` was about
2.4 times cheaper than 1000 while producing the same qualitative three-point
landscape. The original P5.10 selection rule between 300 and 1000 was internally
contradictory, so neither value is a new production setting.

Exact-consensus polish is mandatory. The unpolished ADMM states were locally
feasible but did not represent one common physical point. At `rho_pf = 1000`,
about 1.6 kW of interface disagreement produced an approximately 850-unit polish
correction against an investment signal around 600. Polished and unpolished
oracles ranked candidates in opposite orders.

The current adaptive-rho rule self-cancels under the RESCALED residual
definition because the measured dual residual scales linearly with rho and
therefore triggers rho reduction. Requested `rho_pf` values 300, 500 and 1000
decayed to approximately 88.89, 98.77 and 131.69. No adaptive-rho redesign was
implemented or authorized.

Across nine stabilized terminating cycles, `stationarity_pf` was the sole
binding convergence criterion. Objective stationarity, consensus, voltage and
ESS criteria were materially inside tolerance. This is an open diagnostic
question, not authorization to retune convergence tolerances.

The eight-generation replay at RESCALED, fixed `rho_pf = 1000`, adaptive rho
off and neutralized history produced:

- all generations VALID;
- zero failed polish blocks;
- node 9 preferred at every depth;
- zero ranking flips;
- cross-depth relative uncertainty reduced from about `459695.59` to `22.09`;
- uncertainty/signal reduced from `25.42` to `0.67`;
- best-to-second mean gap only `32.87`, still thin relative to the numerical
  floor `tau_numerical = 10`;
- absolute base-chain drift about `-201099`, so only same-depth relative
  comparisons are scientifically valid.

P5.10 validated only three candidates. It did not establish a broad investment
landscape, a self-consistent cold RESCALED T0, a production adaptive-rho policy,
or readiness for an investment-search campaign.

## Accepted P5.11 consolidation evidence

Accepted report:

`P5_11_STABILIZED_ORACLE_CONSOLIDATION_REPORT.md`

SHA-256:

`a9f9e3422a533d993ccd890cbee53a4a8e1dcbbc3469e95af3a592ec04281b3b`

P5.11.0 passed:

- the Mac Studio checkout and history were identified correctly;
- the governing documents were not rolled back to P5.6-C, although they had not
  yet been extended with P5.10;
- all P5.10 harnesses and 21 evidence files were present, tracked and unmodified;
- the R0 provenance gate passed;
- CURRENT polished total reproduced bit-identically as
  `828021090.3608505`;
- RESCALED pre-polish recourse reproduced bit-identically as
  `825814074.4930633`.

P5.11.1 attempted to construct a diagnostic T0 cold under RESCALED, with fixed
rho `1.5 / 300 / 1.0`, adaptive rho disabled and carried histories neutralized.
The 48 augmented objectives were RESCALED at construction. The diagnostic state
identity was:

- template id `P511-SELFCONSISTENT-T0`;
- consensus/dual state SHA-256
  `14a7ea85ce283a595a35b1768c93c2a60bda872d4f5138c5168b3691a625cbf0`;
- artifact file SHA-256
  `3629784b277724e1f2c406f4a6dbd4599ea92b9d7ee126ac67bb2148909a67bd`;
- configuration hash `a080612dedf2727e`.

The cold RESCALED construction did **not** converge within the frozen production
cap of 25 ADMM cycles. `cycle_convergence` was false at every cycle. Recourse
moved from approximately `2.3520e9` at cycle 1 to `1.4612e9` at cycle 13 and
`1.4024e9` at cycle 25; `primal_pf` was `3.5893e-4` at cycle 25 and had worsened
relative to cycle 13.

Candidate evaluations warm-started from this non-converged state returned local
and polish successes, but they do not validate a self-consistent template. The
inherited-versus-diagnostic relative-delta discrepancies were:

- `rho_pf = 300`: `122.37` for node 5 and `27.59` for node 9;
- `rho_pf = 1000`: `68.61` for node 5 and `81.55` for node 9.

All exceeded the required `22.09` bound, so the P5.11.1 stopping rule fired.
P5.11.2, P5.11.3 and P5.11.4 were correctly not run.

This result does not invalidate P5.10. The inherited-template oracle reproduced
the accepted gates exactly and its P5.11 values matched P5.10. What failed was
the attempted hardening through a cold-built RESCALED T0.

## Accepted P5.12-A cold-RESCALED diagnostic, with qualification

Accepted report text SHA-256:

`a1424ec1b47944ad1aa5a9038306ab1fec4870bf0a3ed41993a29ba1f2c89e96`

Accepted stage verdict:

`P5.12-A FAIL — cold RESCALED diagnostic encountered a numerical failure`.

Repository/runtime provenance, both P5.10 reproduction gates and the P5.11
cycle-25 state hashes passed. No tracked production file or parameter changed.

The decisive authorized evidence is the first local NLP failure at cycle 21.
Retrospective instrumentation also established that the P5.11 cold build had
local failures at cycles 21-24 that its earlier harness did not record. This
strengthens the rejection of the P5.11 diagnostic T0.

P5.12-A did not enforce its stop rule correctly. Instead of one trajectory that
terminated at the first failure, it launched four concurrent cold runs capped at
25, 50, 75 and 100. The runs continued after cycle 21, and later states were
formed while production retained schedules across failed local solves.
Therefore:

- cycle 21 is accepted as the binding failure evidence;
- cycle 25-100 results are exploratory only;
- the reported repeated failures from cycle 83 onward do not establish a valid
  "permanent breakdown" conclusion;
- detached polish results after the first failure cannot certify a T0;
- concurrency is an avoidable experimental confound for future replay.

The inherited CURRENT-built / RESCALED-evaluated template remains the only
validated construction path. P5.10 continues to stand for its restricted
three-candidate, exact-consensus-polished surface. The cold RESCALED construction
is invalid and must not be used for candidate ranking.

## Accepted P5.12-B evidence, with independent-review qualifications

P5.12-B reproduced `DSO:case33_3 | 2025 | Spring | cycle 21`, terminating at
IPOPT's 3000-iteration limit. The matched cycle-20 solve was optimal in 115
iterations. Final unscaled constraint violation rose from `7.21495e-7` to
`0.06192134`, and dual infeasibility from `0.00113231` to `51.59806`.
The terminal failure was not restoration or a MA97 error.

The report, harness, aggregate JSON, provenance and filtered log hashes match
those in `P5_12_C_BOUND_MULTIPLIER_AB_REPORT.md`. The original raw log also
survives at
`data/SRP1/Results/P56A/evals/p512b_forensic/logs/optim_log_case33_3_2025_Spring.log`,
SHA-256 `57987d2e0be0fcb9849b9db5bd67b79795a7fa8c201a05692705d5e8faa1e0c5`.

The following qualifications supersede B's full-preservation and immediate-stop
claims without withdrawing its recorded numerical failure:

- B saved selected summaries, four leading primal values and short fingerprints,
  not a complete indexed model or replayable state. There is no complete
  cycle-20 restart checkpoint or exact cycle-21 snapshot manifest.
- `NetworkData.optimize` clones before `run_smopf`; `_create_smopf_solver` then
  refreshes bound-input suffixes from output suffixes. B's summaries therefore
  do not establish the actual ordered multipliers exported to IPOPT.
- The failure was DSO solve 25 of 36. Eleven more sequential DSO solves ran
  before the wrapper aborted, although no DSO-stage consensus update or cycle 22
  followed. This was not an immediate stop at the first observed failure.
- Equal counts do not establish identical structure or active-set membership;
  differences of L1 norms do not establish norms of pointwise state changes.

Large bound multipliers remain a structural suspect, not a demonstrated cause.
The accepted production `vmag_nodes` refactor remains a numerical-stability
improvement; the remaining local-NLP failures are a separate residual-conditioning
problem.

## Accepted P5.12-C preflight stop

The complete report is `P5_12_C_BOUND_MULTIPLIER_AB_REPORT.md`. Its stop was valid:
the required exact input was unavailable, and its authorization prohibited
trajectory replay. Neither arm ran; there is no causal or convergence result.
Its required closing phrase "A/B COMPLETE" denotes report delivery only.
No production or governing file was changed and nothing was committed or pushed
by that task. R0 was not rerun, so C makes no fresh runtime-provenance claim.

The independent audit verified all six hashes in C's artifact table. Searches
including ignored files found only historical P5.7, cycle-25 P5.11 and cycle-7
comparator pickles. Summaries and solver-log norms cannot recover the missing
indexed primals, parameters and multipliers losslessly. The later cycle-25 state
cannot substitute or be inverted into the target input. No external original
snapshot has been verified.

The later C authorization supersedes B's proposed experiment: the future frozen
local A/B retains constraint duals and suppresses only `ipopt_zL_in` and
`ipopt_zU_in` in Arm B. It does not compare two ADMM trajectories. Neither arm
is authorized by P5.12-R.

## CURRENT AUTHORIZED STAGE — P5.12-R deterministic pre-solve recapture

Following planner audit and user approval, authorize one capture stage only in
the existing Local checkout. Replay the frozen base cold-RESCALED configuration
through cycle 20, preserving the target's cycle-20 input and a complete
cycle-20 checkpoint. Enter cycle 21 in ordinary order only as far as
`DSO:case33_3 | 2025 | Spring` (24 preceding DSO solves under the recorded order).

Capture both the pre-setup model for comparison with B and the exact input after
solver setup refreshes bound multipliers, immediately before `solver.solve`.
Serialize and verify the state, then terminate WITHOUT solving the target.
This narrowly authorizes the recapture replay previously prohibited during C;
it does not authorize either A/B arm, a repair, a retry or a continuation.

Require complete indexed state, solver export ordering, effective settings,
full SHA-256 manifests and a no-solve reload-equivalence check. Historical
summary matching is not proof of exact historical full-state equality because
no original full-state hash exists. Any later Arm A must still reproduce the
3000-iteration failure under separate authorization.

`LOCAL_NLP_STABILITY_PLAN.md` is authoritative for the complete P5.12-R protocol,
evidence requirements and immediate stopping rules. Stop after its report for
planner review. This document update does not itself execute the worker stage.

## Locked prohibitions during P5.12-R

Do not change:

- production nonlinear formulation;
- IPOPT settings;
- production ADMM settings or iteration cap;
- adaptive-rho logic;
- convergence tolerances or stationarity definitions;
- objective scaling in production;
- TSO proximal regularization;
- ESSO degradation or active-energy degradation work;
- Benders logic or convex models;
- anchor or initialization policy;
- accepted P5.10/P5.11/P5.12-A/P5.12-B/P5.12-C evidence.

Do not run the full planning problem, derivative-free search, expanded candidate
population, stationarity sensitivity, another fixed-rho value, polish, either
A/B arm, the target cycle-21 solve, any later block or any post-failure update.

---

# HISTORICAL SOURCE OF TRUTH — 2026-09-08

This section supersedes older solver-policy, P5.4 planning and P5.5-A instructions recorded later in this file where they conflict with the current state.

## Current accepted nonlinear production checkpoint

The accepted active-energy ESS production lineage is:

- `a4a0bae8` — shared-network active-energy ESS productionization;
- `1e86d40e` — ordinary network ESS active-energy parity;
- `58f4911b` — ESSO active-energy conversion and throughput correction;
- `c3526ec8` — lifecycle/sensitivity audit state and post-B/C/D validation;
- `93974d83` — dimensionless charge/discharge complementarity across network ESS, ordinary ESS and ESSO, including ESSO aggregate complementarity;
- `06e921e5` — P5.4-D2-P sensitivity-clean shared-S productionization; redundant positive-capacity S-dependent numerical bounds removed with no feasible-set change.

Accepted nonlinear coordination evidence includes:

- `2917b9c9` — historical fixed-candidate live distributed ADMM diagnostic on the H1 production baseline; net-P/Q coordination only.

Diagnostic/planning evidence includes:

- `b0e53bc4` — P5.4-E2 complementarity-significance instrumentation;
- `65b261ba` — P5.4-D2 S/E sensitivity root-cause audit;
- `e1afa8e9` — original D3 cut-consistency audit, now retained as **NONCANONICAL historical evidence** because it was executed in `srp_env`;
- `1a68f409` — original D4 branch-recovery/hardened-cut audit, also **NONCANONICAL historical evidence**;
- `51a3f4e4`, `5bd1f0ce`, `cbc043d4` — P5.4-R canonical-environment revalidation, which supplies the authoritative D3/D4 paper-instance verdicts.

The old pre-P5.4 checkpoint `f77d829359ffd873367f556882546bc2dcc8ec99` remains historical only.

## Canonical paper environment and reproducibility identity — HARD REQUIREMENT

All future paper-instance validation, benchmark and planning-oracle work must use:

`/opt/anaconda3/envs/opf_env_py311/bin/python`

Canonical environment/provenance from P5.4-R:

- Python `3.11.11`;
- NumPy `2.4.2`;
- pandas `3.0.0`;
- SciPy `1.17.0`;
- Pyomo `6.9.5`;
- IPOPT `3.14.18` at `/usr/local/bin/ipopt`;
- HSL linear solver `ma97`;
- Gurobi / `gurobipy` `13.0.1`, academic licence, expiry `2027-04-10`.

Canonical SRP1 configuration:

- random seed `2026`;
- years `2025`, `2030`, `2035`;
- representative days `Spring`, `Summer`, `Autumn`, `Winter`;
- `24` instants;
- `1` market scenario;
- `1` operation scenario per network;
- TSO `case9`;
- DSOs `case33_1` at node 5, `case33_2` at node 7, `case33_3` at node 9;
- combined realized scenario checksum:

`5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`.

The fail-fast provenance gate introduced in P5.4-R is now a required infrastructure convention: paper-case harnesses must record interpreter/environment/package/solver provenance and abort if the checksum differs. A rejected environment must never silently produce evidence that is mixed with canonical results.

`srp_env` is noncanonical for paper validation because it generates a different stochastic realization (`4d948b9b...`). Seed `2026` alone is therefore **not** a sufficient reproducibility specification.

## Current live nonlinear network solver policy

The live source and current JSON files supersede older notes referring to earlier warm-start policies.

Current cold network configuration:

- IPOPT exact-Hessian primary path;
- MA97 for TSO and DSOs;
- `tol = 1e-5`;
- `acceptable_tol = 1e-4`;
- `acceptable_iter = 5` in the network parameter files;
- `case9`: `bound_push`, `bound_frac`, `slack_bound_frac`, `slack_bound_push` = `1e-6`;
- `case33_1`, `case33_2`, `case33_3`: the same four push/fraction values = `1e-5`.

Current warm-start handling in `network._create_smopf_solver`:

- bound and constraint multipliers are supplied when `from_warm_start=True`;
- TSO warm starts override `acceptable_iter = 0` and `acceptable_tol = tol`, preventing acceptable-level early termination from reintroducing the previously diagnosed voltage-barrier artifact;
- DSO warm starts retain configured acceptable settings unless explicitly overridden;
- warm-start push/fraction settings inherit configured network values unless explicitly provided.

Recovery remains limited to IPOPT `internalSolverError`; configured limited-memory recovery exists for `case33_2` and `case33_3`, while `case33_1` still has no explicit `recovery_options` block. Do not retune the nonlinear solver during P5.6-C.

---

# HISTORICAL PLANNING CHECKPOINT — P5.5-D accepted; P5.6-A/B reviewed; P5.6-C authorized

P5.4-R remains the canonical basis for the nonlinear production formulation and for retirement of the old nonlinear-recourse derivative cuts. P5.4-G was never run and remains permanently blocked.

The rigorous continuous-convex / MISOCP lower-bound route has been tested and retired as the planning architecture. P5.5-D remains accepted as:

`P5.5-D-C — practical rigorous lower-bound architecture is unavailable`.

The decisive P5.5-D evidence remains:

- the continuous convex hull of the H1 charge/discharge set is already the active-sum triangle `pch >= 0`, `pdch >= 0`, `pch + pdch <= S`;
- a lower-bound-safe fixed-mode screen reduces simultaneous circulation from ~`0.5 S` to ~`0.01 S` but changes the full objective by only ~`3.23e5` (`0.049%`) and closes only ~`0.42%` of the gap;
- the fixed-mode value still sits about `10.51%` below the nonlinear feasible reference, so the unrestricted MISOCP cannot provide the required planning-scale lower bound;
- the binary stage was correctly not run;
- the remaining relaxation defect is dominated by the meshed TSO AC relaxation, while DSO SOC, transformed OLTC and the ESSO available-energy relaxation are comparatively negligible;
- full-size conic dual certification is unavailable and is no longer an active planning requirement.

The active development branch remains:

`feature/derivative-free-planning`

created from the accepted P5.5-D HEAD with P5.5 history preserved.

P5.6-A built and validated the deterministic nonlinear candidate-evaluation oracle and remains accepted as:

`P5.6-A PARTIAL — nonlinear oracle works but feasibility, purity, branch stability or computational cost remains unresolved`.

P5.6-B then stabilized the **declared oracle policy** enough to specify the search mechanics, but exposed two deeper questions: whether different deterministic template generations induce the same investment ranking, and whether direct-start failures are false hidden infeasibility rather than physical infeasibility. P5.6-B is accepted as:

`P5.6-B PARTIAL — oracle policy, anchor robustness, start policy or search resolution remains unresolved`.

The next authorized stage is:

**P5.6-C — landscape robustness and oracle coverage gate**.

P5.6-C is the final gate before any derivative-free optimization campaign. It does **not** authorize the 200-evaluation search yet. It must determine whether the frozen T0 surface is representative enough for investment ranking, whether deterministic continuation can rescue false hidden-infeasible points, define the planning-level robustness threshold separately from numerical repeatability, lock deterministic parallel poll semantics, and identify a trustworthy practical MADS/pattern-search implementation path.

## Accepted P5.6-A nonlinear-oracle evidence

The complete original nonlinear coupled system is audited, including the original nonlinear ESSO, all 48 network SMOPFs and all coordinated TSO/DSO/ESSO quantities.

Canonical START-1 base evaluation:

- coordinated residual max `5.551e-17`;
- original nonlinear ESSO max constraint violation `2.035e-13`;
- largest network nonlinear residual `1.093e-05 p.u.` at `DSO5|2030|Winter`, treated as local IPOPT feasibility tolerance rather than a coordination mismatch;
- gross operational cost `829291677.522120`;
- physical salvage `3439.659877`;
- net operational recourse `829288237.862242`;
- investment cost `50000.000000`;
- total planning objective `829338237.862242`.

The old P5.5-D polished value near `729.36e6` is superseded. P5.6-A proved that `95.5%` of the apparent ~13% gain came from an invalid polish convention that moved achieved interface power into fixed TSO `pc` and thereby erased production flexibility cost. With the corrected polish, the genuine START-1 improvement over the ADMM-evaluated branch is ~`9.21e6` (`1.0982%`), mainly slack/flexibility cleanup.

The A12 call-history impurity is closed for the oracle by per-evaluation deep copies, isolated solver-log directories and reset diagnostics. Repeated evaluations of the same candidate under the same declared policy are bit-identical after different call histories.

The candidate API:

- checks first-stage feasibility before operational solves;
- runs nonlinear ADMM, original nonlinear ESSO, exact-consensus polish and full physical audit;
- returns explicit statuses instead of fabricated penalty objectives;
- caches only VALID completed evaluations under a key containing candidate, canonical checksum, configuration/oracle version, start/template id and anchor convention.

The canonical base candidate lies exactly on the minimum energy-to-power ratio `E = 2S`; negative E-only perturbations are therefore first-stage infeasible and must not be used as ordinary search directions.

## Accepted P5.6-B policy evidence

### Best certified incumbent

P5.6-B re-audited the START-2 base result against the same complete original-nonlinear ESSO/network/coupling certificate and promoted it to the current best rigorous feasible nonlinear planning incumbent for the canonical base investment:

- total planning objective `828021090.360850`;
- net operational recourse `827971090.360850`;
- gross operational cost `827974518.717105`;
- physical salvage `3428.356255`;
- investment cost `50000.000000`.

The START-1 value `829338237.862242` remains a valid but weaker feasible point.

### Template chain and frozen recurring start

The deterministic base-template chain does **not** converge to a fixed point over the authorized refinements:

- `T0`: `828021090.360850`;
- `T1`: `827415318.563944`;
- `T2`: `826824028.845478`;
- `T3`: `826405022.193437`;
- `T4`: `825961521.882321`.

Total drift from T0 to T4 is about `-2.059568e6`, with no reliable deceleration. Therefore no claim is made that a stabilized fixed-point template exists.

For reproducibility, the recurring direct search start is frozen **by rule** as:

`T0` = the archived cold solution of the canonical base candidate,

id:

`a81f7f5191dd42dbf50d1726149b8909`.

T0 is deterministic and pure, but knowingly not the best available template. The key unresolved question is whether T0, T2 and T4 induce essentially the same **relative investment landscape** despite different absolute objective levels.

### Anchor policy is locked

Under the 12-candidate P5.6-B population, midpoint and DSO anchoring have the same success rate on operationally evaluated candidates (`8/11 = 72.7%`). Under T0, DSO anchoring rescues none of the midpoint failures. Anchor deltas range from about `-13365.80` to `+11175.98`, candidate rankings change, and Spearman rank correlation is only `0.9048`.

The recurring search-anchor policy is therefore locked to:

`MIDPOINT ONLY`.

Do **not** mix anchors within the recurring search objective. Both anchors may still be evaluated during final incumbent certification and the best VALID fully certified solution retained.

### Recurring start policy is locked, but coverage is incomplete

For the strategic dual-start subset, cold is never better than T0 on objective where both succeed; it is worse by about `1.32e6` to `1.09e7`. Therefore the recurring direct search policy is:

`T0 ONLY`.

Cold is demoted to final-certification / periodic-audit use.

However, the two starts have different failure sets: at least one master-feasible candidate that is `POLISH_FAILURE` under T0 is VALID under cold, and the budget-boundary candidate crashes under both. Therefore a direct T0 failure is **not** evidence of physical infeasibility. This is the principal oracle-coverage issue for P5.6-C.

### Numerical repeatability and search resolution

Under the fully locked T0 + midpoint policy, repeated evaluations are bit-identical. P5.6-B proposed `tau_search = 10.0` against a measured numerical repeatability floor of exactly zero and a smallest adjacent observed objective gap of `33.27`.

P5.6-C must refine the terminology and separate:

- `tau_numerical = 10.0` — numerical comparison threshold on the declared deterministic oracle surface;
- `tau_planning` — investment-improvement robustness threshold derived from cross-template **relative-to-base** landscape variation.

Do not inflate `tau_numerical` to cover template/cold branch differences. Those are heuristic landscape uncertainty, not numerical noise.

### Search coordinates and current method recommendation

The recommended first-stage coordinates are:

`h = E - phi_min*S`, with `phi_min = 2`,

so the search uses `(S,h)` with `S >= 0`, `h >= 0`, `E = 2S + h`. This is a bijective change of variables on the minimum-duration constraint and avoids losing half the native S/E poll directions at the base point. Remaining max-duration, cumulative-capacity, cohort/lifetime and budget constraints are checked exactly; do not silently project infeasible requested points.

P5.6-B selected deterministic MADS/pattern-search as the preferred family, specifically OrthoMADS in principle, because the oracle is expensive, deterministic, nonsmooth, has hidden operational failures, supports caching and allows independent poll evaluations. This is a **design recommendation**, not yet an implementation authorization. P5.6-C must audit whether a trustworthy implementation already exists and must not casually hand-code OrthoMADS incorrectly.

### Search cost

Under T0 + midpoint:

- successful uncached evaluation ~`115 s`;
- blended observed evaluation cost ~`134 s`;
- final multi-start/both-anchor certification ~`740 s`;
- one-off T0 construction ~`518 s`;
- cache hit ~`0.040 s`.

Four upper-level workers are the current recommended maximum. Eight workers risk resource contention; IPOPT process crashes have already been observed under heavy concurrency.

## P5.6-B planner interpretation

P5.6-B settled the **declared** anchor convention, recurring direct start, coordinates, numerical repeatability and provisional search family. What remains unresolved is more precise than the old B verdict wording:

1. **template/branch landscape representativeness** — does T0 preserve investment rankings/improvement signs relative to later deterministic templates T2/T4?;
2. **oracle domain coverage / false hidden infeasibility** — can a deterministic history-independent continuation fallback rescue T0 failures, including remote/budget-near candidates?;
3. **planning-level robustness threshold** — derive `tau_planning` from cross-template relative-delta variation, distinct from `tau_numerical`;
4. **deterministic parallel semantics** — no asynchronous “first finished improvement wins”; future polls must be batch-deterministic;
5. **search implementation path** — audit NOMAD/PyNOMAD/repository support before implementation.

These are the active P5.6-C questions.

# P5.5 ARCHITECTURE STATUS

## P5.5-A — historical PARTIAL, structural inventory accepted

P5.5-A established the actual SRP1 controllable-resource set and the convexity map. Accepted facts remain:

- conventional generator P/Q are controllable;
- RES P/Q, curtailment and PF control are retained;
- active and reactive load flexibility are retained;
- load curtailment is disabled;
- ordinary ESS is absent from SRP1;
- shared ESS P/Q/SOC are present;
- each `case33_*` has **one controllable continuous OLTC** on branch 1, bus 1 -> bus 2, with `r in [0.83,1.17]`;
- there is no discrete tap logic, no phase shift, and no controllable capacitor bank/switched shunt in SRP1;
- `case33_1/2/3` are radial after preprocessing (33 buses, 32 branches, cyclomatic number 0);
- `case9` is meshed with one independent cycle (9 buses, 9 branches, cyclomatic number 1);
- the planning master is an LP with no integer/binary or bilinear master-side terms.

The intended convex family remains a lifted **W-space AC SOC/QC relaxation**. DC-OPF or LinDistFlow substitution is not authorized.

## Two-oracle bound principle remains useful; continuous-convex Benders is not yet viable

For planning candidate `x = (S,E)` define:

`R(x)` = globally solved convex-relaxed AC SMOPF recourse;

`Q_AC_feas(x)` = feasible full nonlinear AC SMOPF recourse, eventually AC-polished before being labelled a rigorous upper bound.

Required relation:

`R(x) <= Q_AC*(x) <= Q_AC_feas(x)`.

The lower-bound / feasible-upper-bound sandwich remains the right conceptual benchmark. The nonlinear model remains the physical truth model and upper-bound/incumbent validator. However, P5.5-C shows that the current **continuous** SOC relaxation cannot yet supply a useful certified planning lower bound or a usable cut oracle. A different lower-bound architecture is now under decision, not assumed.

## P5.5-B — reviewed as PARTIAL; major mathematical closures accepted

P5.5-B is accepted as:

`P5.5-B PARTIAL — one or more lower-bound/cut/interface issues remain unresolved`.

The `PARTIAL` status does **not** reflect a failed formulation. It reflects two items that can only be closed on the implemented real convex model: relaxation tightness/usefulness and formal cut certification.

### B1 accepted — only production-safe tightening

The dormant ±30-degree angle constraints are **not** part of the nonlinear production feasible set and must not be enabled in a certified lower-bound relaxation merely because rules exist in the code.

Only constraints that are:

- active in production; or
- mathematically implied by the production feasible set

may tighten the convex LB model.

Retain production `WijR >= 0`. Derive any further QC bounds only from active voltage bounds, active branch limits, the rank/SOC relation and other production-safe implications.

### B2 accepted — transformed continuous OLTC eliminates explicit tap bilinearity

For each DSO transformer use:

`U_i = r^2 * W_ii`

`C_ij = r * W_ij^R`

`D_ij = r * W_ij^I`.

All production transformer P/Q expressions become affine in `U_i`, `C_ij`, `D_ij`, `W_jj`, verified against the production equations to floating-point noise (~`5.9e-15`).

The physical transformer rank relation becomes:

`C_ij^2 + D_ij^2 = U_i * W_jj`

and is relaxed by the same rotated SOC used on ordinary branches:

`C_ij^2 + D_ij^2 <= U_i * W_jj`.

Continuous tap existence is represented exactly by:

`r_min^2 * W_ii <= U_i <= r_max^2 * W_ii`,

with `W_ii > 0` and recoverable `r = sqrt(U_i/W_ii)`.

For SRP1 there is no tap cost, tap-movement penalty, intertemporal tap coupling, discrete tap position or phase shift. Therefore `r` and `r_sqr` may be eliminated from the convex LB model. The earlier A-stage McCormick proposal is superseded.

### B3 accepted — P/Q/voltage semantics and available-capacity coupling now implemented in P5.5-C

The DSO `REF` generator is the mathematical representation of TSO import/export rather than a separately costed physical generator. The TSO ADN quantities are load-positive; DSO REF `pg/qg` are generation-positive; shared-ESS `pnet/qnet` are load-positive.

The first centralized convex prototype must retain separate TSO/DSO/ESSO copies and replace zero-residual ADMM consensus by exact affine equalities rather than immediately collapsing variables.

P5.5-C confirmed and implemented the planner correction that production passes ESSO **available** S/E capacities to the TSO/DSO operational models. The centralized model therefore imposes exact capacity coupling, after consistent unit conversion:

`S_network_TSO = S_network_DSO = S_available_ESSO`

`E_network_TSO = E_network_DSO = E_available_ESSO`.

Network SOC limits and anchors must use `E_available`, not rated investment E directly.

### B4 accepted — objective-level ESSO relaxation is lower-bound safe

The convex LB oracle may drop:

- ESS charge/discharge complementarity and H1 hats/links;
- the non-negative complementarity objective penalty;
- ESSO degradation and SoH equalities;
- minimum-SoH restriction;
- associated non-negative slack penalties.

The minimal convex ESSO energy relation is:

`0 <= E_available <= E_rated`.

This enlarges the feasible set and is jointly affine in available/rated E. The retained physical cost terms remain unchanged. The resulting LB may be weak, but its direction is safe.

### B5 planner correction — salvage must be a maximum-credit affine lower-bound term

Current net recourse is gross operating cost minus terminal salvage. Salvage is therefore a **credit**: a larger salvage value lowers the objective.

For a rigorous planning lower bound, the master must subtract an **upper bound on physically attainable salvage**. Under the intended non-negative salvage coefficients, the maximum credit occurs at terminal `E_available = E_rated` / `SoH = 1`, not at `soh_min`.

P5.5-C completed the sign trace and derived the exact affine term:

`V_salvage_max(x) = sum gamma[node,cohort] * E_investment[node,cohort]`

and use:

`-V_salvage_max(x)`

in the planning lower-bound objective. The explicit salvage formula has no direct S coefficient once this safe affine bound is used.

Salvage is excluded from operational `R(x)`.

### B6/B7 accepted as prototype evidence, not formal cut certification

Gurobi is the selected prototype solver and `gurobi_persistent` is the preferred interface.

A toy convex capacity problem verified:

- fixing-row dual matches the analytic value-function derivative;
- `QCPDual=1` returns conic duals;
- an `ObjVal`-anchored cut can be slightly invalid at the generating point;
- an `ObjBound`-anchored cut passed a 12-point sweep.

However, this does **not** yet prove that:

`[ObjBound(x_k) - sigma] + g_k^T (x-x_k)`

is globally valid on the real model. A scalar primal-dual gap protects the anchor, but does not by itself certify slope error away from the anchor.

The desired rigorous contract remains one affine dual function:

`L_k(x) = beta_k + g_k^T x`

with `beta_k` and `g_k` derived from one demonstrably dual-feasible conic solution, so weak duality proves:

`L_k(x) <= R(x)`

for every admissible x.

P5.5-C investigated this on the real convex model and found no usable full-model dual certificate; the replacement planning master therefore remains blocked.

## Gurobi is the selected convex-oracle solver

Canonical environment provides:

- `gurobipy 13.0.1`;
- academic licence valid through `2027-04-10`;
- `gurobi`, `gurobi_direct`, and `gurobi_persistent` available;
- `QCPDual=1` supported;
- linear and quadratic/conic duals available;
- `ObjVal`, `ObjBound`, and reported gap available.

Use `gurobi_persistent` first because planning-candidate updates will change capacity-fixing RHS values repeatedly and stable component-to-row handles are useful for dual extraction. `gurobi_direct` is the fallback.

For a scalar convex lower bound, use `ObjBound`. Do not treat `ObjVal` as a certified LB.

---

## P5.5-C — completed PARTIAL; continuous conic oracle implemented and diagnosed

Accepted implementation/evidence from the canonical `opf_env_py311` run:

- additive convex-oracle path only; nonlinear production SMOPF/ADMM/H1/D2-P/master untouched;
- parent model size `322596` variables, `244152` constraints and `35549` Gurobi second-order cones across 48 AC blocks;
- transformed OLTC implementation remains numerically exact and contributes negligible relaxation gap (`rho_tr` worst about `4.66e-06`);
- local nonlinear-to-convex mapped-point tests pass: objective accounting closes to machine precision and block-local violations match the nonlinear models' own residuals;
- the full-parent mapped state is **not yet an exact global-feasibility proof** because its source is an ADMM-converged state with nonzero interface consensus residuals (about `1.341 MW` active-power mismatch at worst and nonzero voltage mismatch);
- continuous full-model primal solve is reproducible with `ObjVal = 653192095.073059`, but Gurobi returns no dual-backed full-model certificate (`ObjBound = -inf` / QCP dual failure); this value is therefore **not a certified lower bound**;
- using that uncertified primal only as a tightness diagnostic gives a recourse gap of about `21.9%` to the best known nonlinear feasible base branch `836586463.43`, roughly `256-259 x` canonical `tol_cut`;
- dominant looseness is relaxed shared-ESS mode switching: `min(pch,pdch)/S` reaches about `0.49893`;
- second-order looseness is on the meshed TSO: AC rank gap worst about `2.4499e-02` and single-cycle angle inconsistency about `1.2344e-02 rad`;
- DSO SOC relaxation is essentially exact in comparison; ESSO `E_available/E_rated >= 0.99754`, so the relaxed degradation-energy interval is not a material gap source;
- the real-model sampled cut remains unusable: on the largest certifiable restriction the S/E dual signal is tiny/inconsistent with finite differences and apparent cut safety is dominated by solver `ObjBound-ObjVal` slack.

Planner interpretation:

1. Do **not** spend another stage trying to certify the current 22%-loose continuous oracle.
2. Do **not** add an arbitrary continuous circulation cap; because `(S,0)` and `(0,S)` are both H1-feasible operating modes, any convex set containing both contains `(S/2,S/2)`. The observed ~0.5S circulation may be intrinsic to the continuous convex hull of the mode disjunction.
3. A lower-bound-safe disjunctive outer approximation of H1 is worth one controlled diagnostic before abandoning rigorous lower bounds.
4. The planning master remains blocked.

# HISTORICAL ACTIVE STAGE — P5.6-C landscape robustness and oracle coverage gate

P5.6-C is the **final pre-search gate**. It must not launch the full derivative-free investment optimization campaign.

Accepted/frozen unless contradicted by C evidence:

- nonlinear production mathematics and solver/ADMM policy;
- `feature/derivative-free-planning`;
- current best certified base-investment incumbent `828021090.360850`;
- recurring direct template `T0`, id `a81f7f5191dd42dbf50d1726149b8909`;
- recurring anchor `MIDPOINT ONLY`;
- recurring direct start `T0 ONLY`;
- coordinates `(S,h)` with `h = E - 2S`;
- `tau_numerical = 10.0` as the provisional deterministic numerical threshold;
- deterministic MADS/pattern-search as the preferred search family in principle;
- four upper-level workers as the practical parallelism ceiling unless new measurements justify otherwise.

## C1 — correct the P5.6-B interpretation

Record that P5.6-B **did** lock the declared midpoint anchor and T0 direct start. The unresolved questions are landscape representativeness, domain coverage/false hidden infeasibility and deterministic search implementation semantics, not whether the anchor policy itself is still open.

## C2 — cross-template investment-landscape test

Use the deterministic P5.6-B benchmark population and evaluate the jointly reachable candidates under:

- `T0`;
- `T2`;
- `T4`;

with midpoint anchor only, a fresh candidate-specific baseline, original nonlinear ESSO, exact-consensus polish and full feasibility audit. Cache keys must include template id.

For every template and candidate compute:

`Delta_Tk(x) = Q_Tk(x) - Q_Tk(base)`.

The deltas, not absolute levels, are the central quantity.

Across jointly VALID candidates report:

- Spearman rank correlation;
- Kendall rank correlation;
- complete rankings;
- max rank displacement;
- number of pairwise ordering reversals;
- worst reversal magnitude;
- per-candidate `Delta_T0`, `Delta_T2`, `Delta_T4`, range and standard deviation;
- diagnostic affine fits `Q_T2 = a2 + b2*Q_T0` and `Q_T4 = a4 + b4*Q_T0`.

Do not infer robustness from correlation alone if a candidate that improves over base under one template becomes worse under another by a material amount.

Classify the landscape as stable only if improvement signs/rankings are sufficiently preserved and template drift is predominantly a common offset. If stable, keep T0 for the recurring search. If unstable, do not launch MADS; return for a fixed better template, fixed ensemble or continuation-oracle decision.

## C3 — deterministic continuation fallback

Direct T0 failure is not physical infeasibility. Test a fixed, history-independent continuation from the canonical base investment `x0` to a master-feasible target `x`:

`x(lambda) = (1-lambda)*x0 + lambda*x`

with fixed schedule:

`lambda = 0.25, 0.50, 0.75, 1.00`.

Because the first-stage feasible set is a polyhedron, every segment point remains first-stage feasible whenever both endpoints are feasible.

Starting from the frozen T0/base operational state, solve each continuation point in order and use each complete VALID state only to initialize the next. Every point must run the original nonlinear ESSO, production ADMM, midpoint-only exact-consensus polish and full feasibility audit. Do not switch anchors and do not use an adaptive lambda schedule in C.

Test continuation on:

- all T0/midpoint `POLISH_FAILURE` candidates from B;
- the budget-boundary `SOLVER_CRASH` candidate;
- at least two T0-direct VALID controls.

For controls compare direct-T0 and continuation objectives. This determines whether continuation is a neutral coverage rescue or a materially different branch-selection policy.

## C4 — lock oracle coverage policy

Compare:

- direct T0 only;
- direct T0, fixed continuation only on failure;
- fixed continuation for every candidate.

Prefer direct-T0 + continuation-on-failure only if continuation materially improves coverage **and** direct-vs-continuation objectives on already-valid controls are compatible enough that fallback does not mix incompatible objective surfaces.

If continuation materially changes valid-control objectives, either use continuation for every candidate or return PARTIAL. Never silently mix two incompatible surfaces.

Recompute success rate and ordinary/failed/blended evaluation costs under the chosen coverage policy.

## C5 — separate numerical and planning robustness thresholds

Use:

`tau_numerical = 10.0`

unless C evidence shows a larger deterministic numerical floor.

Define separately:

`tau_planning`

from cross-template **relative-to-base** delta variation. `tau_planning` governs whether an investment improvement is robust to known branch/template ambiguity. It is not a solver-repeatability tolerance.

Future search acceptance may use `tau_numerical` internally on the locked black-box surface, but promotion/scientific claims must respect `tau_planning`.

## C6 — deterministic parallel polling semantics

Correct the P5.6-B “parallel opportunistic first improvement” idea. Future search must use deterministic **batch polling**:

1. generate the ordered deterministic poll set;
2. apply exact first-stage feasibility checks;
3. remove duplicate/cache-hit points;
4. select the batch by deterministic poll order;
5. evaluate up to four candidates in parallel;
6. wait for the whole selected batch;
7. among all VALID results choose the best objective;
8. break ties within `tau_numerical` by deterministic poll index;
9. only then update incumbent and mesh.

No asynchronous “first process to finish wins” rule is allowed, because it would make the search path depend on OS/solver completion order.

## C7 — search implementation inventory

Without installing anything, audit:

- NOMAD / PyNOMAD availability in the canonical environment;
- existing repository pattern/MADS utilities;
- licences;
- hidden-constraint evaluator support;
- exact linear feasibility screening;
- deterministic direction/seed control;
- cache integration;
- deterministic batch/parallel evaluation.

Do not casually hand-code “OrthoMADS” without its actual mesh/direction requirements. If a trustworthy MADS implementation is unavailable or disproportionate to integrate, a deterministic generalized pattern search with a positive-spanning poll set is acceptable for the first campaign, with its limitations explicit.

## C8 — pre-search classification

End one:

`P5.6-C-A — derivative-free search is ready to launch`

only if template rankings are sufficiently stable, coverage policy is deterministic/acceptable, planning robustness is quantified, deterministic batch semantics are fixed and a practical search implementation path exists;

or

`P5.6-C-B — derivative-free search is ready only on a restricted validated domain`

if the local landscape is robust and nearby coverage is good but remote/budget-near candidates remain unreachable. Any restricted domain must be specified explicitly from tested successful directions;

or

`P5.6-C-C — derivative-free search is not ready`

if template generations materially reorder investment improvements, continuation cannot provide acceptable deterministic coverage, or no practical deterministic search implementation path exists.

Then:

`P5.6-C COMPLETE — ready for planner review before any optimization campaign`.

## Locked production decisions during P5.6-C

Do not change:

- nonlinear AC SMOPF equations;
- active-energy ESS formulation;
- D2-P sensitivity-clean shared-S formulation;
- H1 dimensionless complementarity and `ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`;
- net-P/Q-only nonlinear ADMM coordination;
- IPOPT tolerances/options;
- MA97/exact-Hessian policy;
- recovery/adaptive-rho/proximal policies;
- existing master/Benders equations;
- P5.5 convex/MISOCP diagnostic code.

Do not yet run or productionize:

- the 200-evaluation derivative-free optimization campaign;
- replacement outer planning loop;
- full/production MISOCP planning;
- distributed convex ADMM;
- QCP/Benders cut recovery;
- TSO SDP/QC strengthening.

# COMPLETED STAGE — P5.5-D lower-bound architecture decision gate

Accepted verdict:

`P5.5-D-C — practical rigorous lower-bound architecture is unavailable`.

The zero-binary fixed-mode screen was decisive: mode control reduced simultaneous circulation from ~`0.5 S` to ~`0.01 S` but closed only ~`0.42%` of the objective gap. Because `Q_MISOCP <= Q_fixed_mode` for minimization, the unrestricted MISOCP could not close the remaining ~`10.51%` gap, so no binary stage was run. The remaining relaxation defect is dominated by the meshed TSO AC model, but stronger TSO convexification is deferred as optional benchmark work rather than the active planning path.

## Old P5.4-G status

P5.4-G remains permanently blocked for the old nonlinear-recourse derivative cuts. Any later planner is a new architecture.

## Deferred items

Keep deferred during P5.6-C:

- physical complementarity tolerance `1e-5` / `1e-6` A/B;
- B1 exact `f_ref=0`;
- RES B2-R until defensible converter `Smax` data exists;
- nonlinear solver/ADMM retuning;
- distributed convex ADMM;
- production MISOCP planning;
- TSO SDP/QC strengthening except as future diagnostic benchmark;
- Benders/QCP dual-cut recovery;
- actual derivative-free optimization run;
- surrogate-assisted planning unless selected later by explicit planner review.

# HISTORICAL DECISION — P5.3-B complete; P5.4 authorized

P5.3-B is complete. The B-series results are now authoritative and supersede the older “current P5.3 execution order” later in this file. Those older sections remain historical evidence only.

The accepted production checkpoint is still:

`f77d829359ffd873367f556882546bc2dcc8ec99`

The successful B3 formulation is **not yet a production commit**. It is the approved production direction to be implemented and validated end to end in P5.4.

## P5.3-B1 — exact reference-angle gauge

Diagnostic change:

`f_ref = 0`

instead of the current narrow reference-angle band.

Result:

- gauge freedom was removed cleanly;
- the `f_ref` variable disappeared from the NLP;
- equality-rank deficiency was unchanged because `sess_snet_def` remained present;
- the three original positive-bootstrap failures were repaired, but three different failures appeared;
- the failure count therefore remained 3 and two harsher modes (`Error in step computation`, `Restoration Failed`) appeared.

Decision:

**B1 = CONTINUE TESTING — do not productionize yet.**

Retest `f_ref = 0` only after the active-power ESS formulation is stable in production. It is not part of the initial P5.4 production baseline.

## P5.3-B2-R — RES capability semantics

The current curtailable-RES formulation uses:

`0 <= pg <= P_available`

and:

`pg^2 + qg^2 <= S_available^2`,

with current synthetic:

`Q_available = 0`

so:

`S_available = P_available`.

Therefore stochastic irradiance/wind availability directly sets the P/Q capability-circle radius. In reduced SRP1:

- every live RES cold start lies exactly on `sg_capability`;
- `sg_capability` is the binding reactive restriction in 100% of live cold-start points;
- 17 realized low-output points in `(1e-5, 1e-4] p.u.` create very small active capability circles;
- static `qmin/qmax` are effectively unreachable over much of the operating range.

However, the repository contains **no explicit, defensible inverter/converter apparent-power rating**. `Pmax`, `Qmax`, PF limits, historical maxima, and arbitrary oversizing factors must not be reinterpreted as `S_converter` without documented physical semantics.

Decision:

**B2-R = DEFER — insufficient physical rating data for safe reformulation.**

Future data-model direction:

- add an explicit optional generator field such as `Smax` [MVA];
- store it as a dedicated apparent-power rating, e.g. `Generator.s_rated` in p.u.;
- require at minimum `Smax > 0` and `Smax >= Pmax` when provided;
- keep legacy cases on the current formulation when no rating is supplied;
- do not adopt arbitrary plausibility thresholds such as `Smax > 2*Pmax` without equipment evidence.

Important separate semantic point:

An explicit `Smax` alone would **not** enable reactive-only/STATCOM operation at `pg = 0`, because the current PF cone also forces `qg = 0` at `pg = 0`. Converter rating and reactive-only operating policy are separate future modelling decisions.

Stochastic-model findings retained for later work:

- `abs()` reflection of negative RES samples is quantitatively negligible but should eventually be replaced by physical lower clipping (`max(sample, 0)`) for correctness;
- the material support issue is upper overshoot;
- historical-max exceedance is not automatically a physical violation — future auditing should quantify exceedance of installed `Pmax` / capacity factor > 1;
- cross-generator spatial correlation is currently not preserved.

Do not modify the RES formulation or copula during P5.4.

## P5.3-B3 — active-power shared-ESS prototype — ACCEPTED PRODUCTION DIRECTION

The diagnostic network formulation replaced the apparent-power charge/discharge geometry with active-power battery dynamics.

Core accepted physical direction:

`pnet = pch - pdch`

`SOC_t = SOC_(t-1) + eta_ch * pch * dt - pdch * dt / eta_dch`

with the current representative-day time basis verified as `dt = 1 h`.

Converter capability:

`pnet^2 + qnet^2 <= S_rated^2`.

Active charging/discharging envelope derived from the old feasible set:

`pch + pdch <= S_rated`.

Complementarity moves from:

`sch * sdch`

to:

`pch * pdch`

with the existing `ESS_COMPLEMENTARITY_TOLERANCE` unchanged.

The diagnostic prototype removed/deactivated the shared-network internal rows/variables that depended on `sch/sdch`, including:

- `shared_es_sch`;
- `shared_es_sdch`;
- `sess_snet_def`;
- `sess_pch_link`;
- `sess_pdch_link`;
- old `sess_s_limit`;
- old apparent-power `sess_soc_def`;
- old `sess_comp`.

### Structural result

B3 removed the dominant structural defect completely:

- DSO zero-gradient equality rows: `24 -> 0`;
- TSO zero-gradient equality rows: `72 -> 0`;
- full equality Jacobian changed from exact rank deficiency to full row rank;
- DSO `sigma_min(full)` became approximately `5.925e-3`;
- TSO `sigma_min(full)` became approximately `3.287e-2`;
- the previous `sess_snet_def` curvature peak of about `18806` disappeared;
- largest remaining reported structural curvature was about `138`, a reduction of roughly 136x.

### Bootstrap robustness result

Exact positive-bootstrap A/B:

Production A:

- DSO success: `33/36`;
- persistent failures: `3`;
- total network IPOPT iterations: `33073`;
- mean iterations: `689`;
- median iterations: `468`;
- max: `3000`;
- runtime: about `274 s`.

Active-power B:

- DSO success: `36/36`;
- TSO success: `12/12`;
- ESSO unchanged success: `3/3`;
- primary failures: `0`;
- recoveries: `0`;
- persistent failures: `0`;
- total network IPOPT iterations: `1545`;
- mean iterations: `32.2`;
- median iterations: `27.5`;
- max iterations: `109`;
- runtime: about `37 s`.

The three original P5 failures were eliminated **without relocation**.

### Physics result

Required unit tests passed:

- pure Q with `pch = pdch = pnet = 0` produces exactly `Delta_SOC = 0`;
- pure charging changes SOC by `eta_ch * pch * dt`;
- pure discharging changes SOC by `-pdch * dt / eta_dch`;
- converter P/Q capability behaves correctly;
- zero-capacity gating remains safe.

This corrects the previous physical inconsistency where reactive apparent power affected stored battery energy.

### Remaining numerical issues after B3

These do **not** block the production direction, but must remain explicit.

Active-power complementarity:

`pch * pdch <= ESS_COMPLEMENTARITY_TOLERANCE * S_rated^2`

remains under-resolved at tiny bootstrap capacity. Its RHS is roughly `1.1e-12` to `1.0e-11`, far below the network IPOPT absolute tolerance, and a tiny accepted physical violation was measured.

Converter capability:

`pnet^2 + qnet^2 <= S_rated^2`

is also below the absolute solver feasibility scale for the smallest bootstrap devices.

These are now **inequality-resolution problems**, not equality-rank deficiencies.

Do not immediately respond with another arbitrary row multiplier. If later live ADMM/planning residual audits show material physical violations, prefer a true dimensionless ESS internal-variable formulation.

### Network/coordination compatibility

The B3 consumer trace established that load-bearing coordination components use `pnet/qnet`:

- nodal balance;
- ADMM consensus;
- expected shared-ESS P/Q schedules;
- scenario-deviation terms;
- Benders sensitivity extraction.

Two non-load-bearing consumers must be corrected during productionization:

1. the exported shared-ESS apparent-power/result field that currently derives from `sch - sdch`;
2. the ADMM residual diagnostic that reports charge/discharge using `sch/sdch`.

Do not silently change output semantics. If an existing `s_ess` field means apparent-power magnitude, use `sqrt(pnet^2 + qnet^2)` and document/rename as needed. If any consumer expects a signed quantity, preserve compatibility explicitly. Active charging/discharging quantities must be labelled in MW, not MVA.

## Current production decision

**B3 = PRODUCTIONIZE CANDIDATE / approved production direction.**

This does **not** mean the diagnostic wrapper is accepted production code. P5.4 must implement the active-energy semantics consistently across the production network model, ordinary ESS where relevant, ESSO, degradation/throughput, diagnostics, result exports, sensitivities, lifecycle handling, and live distributed execution.

The P5.2 narrow-band workaround is now abandoned. Do not productionize it. Do not continue tuning `sess_snet_def` or its `kappa` scale.

## Next authorized stage

**P5.4 — End-to-end active-energy ESS productionization.**

`LOCAL_NLP_STABILITY_PLAN.md` contains the detailed P5.4 execution protocol and takes precedence for implementation and validation.

---

# Current local-NLP checkpoint — P1 through P5.2-A3

## P1/P2 — voltage-magnitude structural conditioning

The original local failure family was traced to redundant explicit voltage-magnitude variables/equalities and later to TSO acceptable-level barrier behavior.

Accepted structural change:

- keep `e`, `f`, `vmag_sqr`, and `vmag_sqr = e^2 + f^2` on all physical nodes;
- create explicit `vmag` and `vmag_sqr = vmag^2` only where `vmag` is actually consumed:
  - DSO: reference/interface node;
  - TSO: active DSO-interface nodes.

This eliminated the decisive frozen `case33_2 / node 7 / 2025 Winter / cycle 10` exact-Hessian failure and subsequently passed live operational smoke tests.

The TSO warm-start policy was then tightened so acceptable-level restoration cannot stop at a materially larger barrier parameter. The resulting operational run converged with zero active voltage slack at convergence.

## P3/P4 — shared-ESS nonlinear equality normalization

The network-side shared-ESS magnitude row was identified as a major local KKT conditioning trigger:

`g_sess = (sch - sdch)^2 - pnet^2 - qnet^2 = 0`

At small dispatch all four derivatives can approach zero. Exact-Hessian MA97 failures in both DSO and TSO frozen states were removed by scaling the existing row in place.

Accepted production formulation:

`kappa_e * g_sess = 0`

with fixed numerical scale:

`kappa_e = 1 / S_rated[e]`

for active positive-capacity shared ESS. Zero/near-zero shared ESS uses the existing operational gating: variables are fixed to zero, operational rows are deactivated, and the finite placeholder scale never affects the active NLP.

Because shared-ESS capacity can change on a reused live model, the implementation keeps the scale synchronized with installed power capacity and transforms an imported row multiplier consistently when the scale changes.

This change reduced the P2.10 live primary local failures from 14 to 2 in the accepted P4.5 smoke and retained ADMM convergence.

## P4.6 — ordinary ESS sign convention and normalization

Ordinary network ESS now uses the canonical load-positive convention:

`es_pnet = es_pch - es_pdch`

so:

- `pnet > 0`: charging / active consumption;
- `pnet < 0`: discharging / active injection;
- `qnet > 0`: reactive absorption;
- `qnet < 0`: reactive injection.

Nodal balance and result processing use the same convention end to end.

Ordinary `ess_snet_def` is normalized analogously:

`kappa_es[e] * ((sch - sdch)^2 - pnet^2 - qnet^2) = 0`

with immutable build-time:

`kappa_es[e] = 1 / S_rated[e]`.

Unlike shared ESS, ordinary ESS has no zero-capacity gating. An explicitly instantiated ordinary ESS must therefore have rated apparent power greater than `1e-10 p.u.`; zero/near-zero explicit ratings are rejected at construction.

The OP1 validation case with two `0.005 p.u.` devices produced `kappa_es = 200`, remained a clean primary exact-Hessian solve, approximately halved IPOPT iterations, and preserved the physical equality and output sign convention.

## P5 — reduced planning baseline

The exact current P5 reduced planning baseline was run from production checkpoint `f77d8293...`.

Iteration 1, zero investment:

- operational ADMM converged in 9 cycles;
- no local primary network failures;
- no ESSO failures;
- zero active voltage slacks at the accepted solution;
- recourse stationarity passed.

Iteration 2, production positive-bootstrap candidate:

- initialization failed before ADMM;
- `case33_1 / node 5 / 2030 Winter` -> `maxIterations`;
- `case33_1 / node 5 / 2035 Winter` -> `maxIterations`;
- `case33_3 / node 9 / 2025 Summer` -> `maxIterations`;
- no recovery was attempted because `maxIterations` is outside the current recoverable class.

The previously problematic `case33_2 / node 7` and TSO shared-ESS-interface failure families did not reappear in P5.

## P5.1 / P5.1-B — small-capacity shared-ESS scaling diagnostics

The positive-bootstrap power ratings are very small:

- 2025: `0.010635 MVA = 1.0635e-4 p.u.` -> production `kappa ~= 9403`;
- 2030: `0.021270 MVA = 2.1270e-4 p.u.` -> production `kappa ~= 4701.5`;
- 2035: `0.031905 MVA = 3.1905e-4 p.u.` -> production `kappa ~= 3134.3`.

Capping `kappa` proved that row scaling directly controls convergence in several sensitive cold starts, but no scalar cap was robust across the full initialization population:

- cap 100 cleared the three original P5 failures but introduced a different `case33_3 / 2025 Autumn` failure;
- a tested scaling ladder showed strongly non-monotone behavior;
- `Kmax = 1000` was the only tested cap that solved the four targeted states simultaneously, but full initialization then failed at four different DSO states, including node 7.

Conclusion:

**do not productionize a scalar cap on `1/S_rated`.** Row scale is influential, but capping relocates path-dependent failures rather than robustly eliminating them.

## P5.2-A / A2 / A3 — narrow-band diagnostic

Hypothesis tested:

The hard equality

`g_sess = 0`

has exactly zero gradient at:

`sch = sdch = pnet = qnet = 0`.

A finite scalar multiplier cannot remove this exact zero-gradient equality degeneracy.

Diagnostic replacement:

`-epsilon_rel * S_rated^2 <= g_sess <= +epsilon_rel * S_rated^2`

while keeping the accepted production `kappa = 1/S_rated` unchanged in the scaled row.

### `epsilon_rel = 1e-5`

All eight known sensitive states ultimately solved, and the full 51-solve positive-bootstrap initialization had zero persistent failures. One targeted network state (`case33_2 / node 7 / 2030 Summer`) required the existing limited-memory recovery after a primary `internalSolverError`.

### epsilon sensitivity

Targeted sensitivity considered `1e-5`, `3e-5`, and `1e-4`:

- `1e-5`: outstanding node-7 case remained recovery-dependent;
- `3e-5`: target became a clean primary success but a previously successful node-5 control failed outright;
- `1e-4`: target and all three matched controls succeeded on the primary exact-Hessian path.

### Full initialization at `epsilon_rel = 1e-4`

Strong solver-side result:

- 51/51 local initialization solves successful;
- 36/36 DSO;
- 12/12 TSO;
- 3/3 ESSO;
- 48/48 network solves clean on the primary exact-Hessian path;
- zero network recovery attempts;
- zero persistent failures;
- initialization would enter ADMM.

Blocking physical/numerical finding:

The nominal band is below IPOPT's effective constraint-feasibility resolution for the tiny bootstrap devices.

Across 1728 active shared-ESS network rows:

- max `|g| / S_rated^2 = 2.6331e-4`;
- mean = `1.2871e-5`;
- 95th percentile = `5.9805e-5`;
- max nominal band utilization = `2.6331`;
- 126 rows (7.29%) exceeded 0.5 nominal utilization;
- 22 rows (1.27%) exceeded 0.9;
- 20 rows were at or beyond the nominal boundary within the audit criterion;
- worst cases were concentrated in TSO rows;
- maximum apparent-power mismatch remained small in absolute terms (`~48 VA`, max `DeltaS/S_rated ~= 2.25e-3`) but the declared band itself was not a reliable physical error budget.

Interpretation:

- converting the hard zero-gradient equality into an inequality is strongly beneficial structurally;
- the current tolerance-band construction is not yet a principled production physical constraint because the declared band is finer than the network solver's feasibility resolution;
- **do not productionize the P5.2 narrow band yet**;
- stop epsilon and scalar-kappa tuning and perform a broader structural conditioning audit.

---

# P5.3 — completed structural SMOPF review (historical execution record)

P5.3 is complete. The quantitative audit, corrected RES/Jacobian follow-up, and isolated B-series reformulation tests are retained below as historical execution evidence. The authoritative P5.3-B decisions and P5.4 next stage are stated near the top of this file.

`LOCAL_NLP_STABILITY_PLAN.md` contains the detailed execution protocol and takes precedence for the B experiments.

## P5.3-A / A2 — authoritative completed findings

The original P5.3-A row-wise audit correctly identified the shared-ESS nonlinear geometry as the dominant structural risk, but two global Jacobian conclusions were later corrected in P5.3-A2. The following statements are now authoritative.

### 1. `sess_snet_def` is the sole source of exact equality-row rank deficiency at the bootstrap cold start

Current production shared-ESS row:

`kappa * ((sch - sdch)^2 - pnet^2 - qnet^2) = 0`

with:

`kappa = 1 / S_rated`.

At the natural zero-dispatch cold start:

- every active `sess_snet_def` row has exactly zero first derivative;
- DSO models contain 24 exactly-zero equality rows;
- TSO models contain 72 exactly-zero equality rows;
- therefore the full equality Jacobian has `sigma_min = 0` and is exactly row-rank deficient;
- after removing only those exactly-zero rows, the tested reduced equality Jacobians have full row rank with no additional nullity.

Corrected reduced-spectrum conditioning on representative models:

- DSO reduced equality Jacobian condition number: approximately `8.98e4`;
- TSO reduced equality Jacobian condition number: approximately `1.42e3`.

The earlier claim that the TSO equality Jacobian was materially worse conditioned than the DSO Jacobian is **withdrawn**. The corrected result is the opposite on the nonzero subspace.

The accepted P4 normalization also gives the shared row curvature:

`2 * kappa = 2 / S_rated`,

reaching approximately `18806` at the smallest positive-bootstrap rating. Thus `sess_snet_def` combines:

- exact zero first derivative;
- an always-active equality;
- exact rank deficiency;
- curvature growing as `O(1/S_rated)`.

This remains the highest-priority structural defect.

### 2. `sess_comp` remains HIGH risk

The bilinear shared-ESS complementarity relaxation:

`sch * sdch <= ESS_COMPLEMENTARITY_TOLERANCE * S_rated^2`

has, at the positive-bootstrap scale:

- cold-start Jacobian norms around `2e-8` to `6e-8`;
- an RHS/margin scaling with `S_rated^2`, reaching roughly `1e-12` at the smallest bootstrap capacity;
- a physical inequality margin many orders below the network IPOPT feasibility tolerance.

Do not hide this with another arbitrary scalar normalization. The active-power ESS prototype must re-audit complementarity after moving it to `pch * pdch`.

### 3. Corrected column diagnostics

The previous report of approximately 48 near-zero DSO Jacobian columns (`pij/qij`) was a diagnostic artifact.

Root cause:

- `r_sqr` has no Pyomo initial value;
- reverse-mode numeric differentiation failed on rows referencing it;
- the original audit swallowed the exceptions and skipped 120 DSO equality rows;
- this made `pij/qij` appear disconnected even though their defining equations contain unit coefficients.

After supplying a nominal diagnostic value, the derivative failures disappear and the `pij/qij/pji/qji` own-variable coefficients are exactly 1 as expected.

The near-zero DSO-column conclusion and the earlier `f_ref`-column red flag are therefore **withdrawn**. The remaining production observation is only that `r_sqr` lacks an explicit cold-start initialization.

### 4. RES `sg_capability` remains HIGH risk

For curtailable RES:

`pg^2 + qg^2 <= sg_available^2`.

The current reduced SRP1 population has:

- 3732 active `sg_capability` rows;
- zero cold-start margin for these rows because the initial `pg` is placed at availability;
- gradient norms down to approximately `5.44e-5` for the lowest live availability values;
- curvature 2.

There are 17 realized RES availability values in `(1e-5, 1e-4] p.u.`. These are the main low-output nonlinear RES rows to inspect in B2-R.

### 5. Old exact RES B2 is cancelled for SRP1

All 144 curtailable SRP1 generator instances have:

`power_factor_control = True`.

Their stochastic reactive availability is identically zero, but `qg` remains a controlled variable inside the PF cone. Therefore:

- `gen_pf_profile` is never instantiated in SRP1;
- there is no cross-multiplied fixed-profile equality to clean up;
- replacing `pg^2 + qg^2 <= S_available^2` by `pg <= S_available` would change the feasible set because reactive power is not fixed to zero.

The old P5.3-B2 exact PF-profile cleanup is therefore a no-op for SRP1 and is superseded by **B2-R — RES capability semantics and conditioning**.

### 6. Current RES availability/converter semantics need review

Synthetic RES currently has `q_available = 0`, so:

`sg_available = sqrt(pg_available^2 + qg_available^2) = pg_available`.

The same stochastic active-power availability is therefore used as the radius of the P/Q capability circle. Reactive capability collapses as stochastic active availability falls.

This may conflate:

- stochastic primary-resource availability; and
- inverter/converter nameplate MVA capability.

B2-R may test a separated formulation only if the repository contains an explicit, defensible converter apparent-power rating. Do not invent one from a heuristic.

### 7. RES stochastic-support findings

The historical-data copula/KDE scenario process itself remains the baseline, but P5.3-A2 established:

- negative inverse-transformed RES samples before `abs()` are very rare (`0` to `0.17%` per 2400 values in the recorded calls);
- positive mass created solely by reflecting negatives through `abs()` is negligible (`<= 0.01%` of post-`abs` mass);
- the important support issue is **upper overshoot**: some season/type calls have up to approximately `33.5%` of synthetic values above the historical maximum;
- in the realized reduced SRP1 population, 30.6% of RES values are exact zero;
- there are no realized values in `(0, 1e-5]`;
- therefore the current `EQUALITY_TOLERANCE = 1e-5` availability switch is not being exercised marginally in this reduced run;
- 17 live values lie in `(1e-5, 1e-4]` and instantiate small active capability circles.

The `abs()` hypothesis is downgraded. Future stochastic-model work should prioritize physical upper support and spatial dependence.

### 8. Spatial RES correlation is not preserved

The copula is fitted per `(season, RES type)` with 24 hourly dimensions, so it preserves temporal dependence within a daily profile.

Physical generator identity is pooled out at fit time. Each same-type physical generator then samples independently from the common synthetic pool using a generator-specific seed.

Therefore the current workflow preserves temporal dependence but **does not preserve cross-generator spatial correlation**.

Do not redesign the copula during P5.3-B; keep this as a later scenario-model revision.

### 9. DSO interface-voltage semantics are intentional

At initial model construction, a DSO reference voltage is tightly initialized/bounded around the local generator setpoint. However, the production ADMM setup explicitly frees the interface magnitude while retaining the reference angle.

Therefore the earlier concern that the DSO interface magnitude remained effectively pinned throughout ADMM is **withdrawn**. The cold-start pinning is an initialization boundary condition; the ADMM magnitude is deliberately released.

This strengthens the exact reference-angle B1 test: the code already intends to retain the angle reference, so `f_ref = 0` is the cleaner gauge formulation to validate.

### 10. IPOPT scales the objective but not the constraint rows

Current network solves rely on IPOPT's default gradient-based NLP scaling. Production logs show the large raw objective gradient is scaled internally to approximately the configured maximum-gradient scale.

However, constraint scaling is not supplied. Thus the raw disparity among:

- zero-gradient rows;
- `~1e-8` complementarity rows;
- `~1e5` branch-related rows;

remains exposed to the KKT system. MA97 has also reported scaling activation due to excess delays.

Do not respond by retuning IPOPT during P5.3. Prefer formulation improvements.

## Historical P5.3 execution order (completed)

For practical debugging and validation, proceed in this order:

### B1 — exact reference-angle gauge

Test:

`f_ref = 0`

against the current narrow `+/- EQUALITY_TOLERANCE` reference-angle band.

This is the lowest-risk, mathematically exact cleanup and should be validated first.

### B2-R — RES capability semantics and conditioning

Run this **second**, before the larger ESS refactor, because it is easier and quicker to debug and validate.

First inspect every curtailable RES generator for an explicit, defensible converter/inverter apparent-power rating.

If such a rating exists, the diagnostic candidate is conceptually:

`0 <= pg <= P_available`

for stochastic resource availability, together with:

`pg^2 + qg^2 <= S_converter^2`

for converter capability, retaining the existing PF-control constraints.

This is a deliberate feasible-set change, not an exact algebraic rewrite.

If no defensible `S_converter` exists, stop B2-R before implementation and recommend the minimum data-model extension instead. Do not infer a rating from an arbitrary heuristic.

Keep the stochastic-support recommendations separate from the NLP formulation experiment.

### B3 — active-power ESS structural prototype

Run this **third**. It remains the highest-payoff structural reformulation, but it is deliberately postponed until after B2-R because B2-R is faster to isolate and validate.

Diagnostic target:

`pnet = pch - pdch`

`SOC_t = SOC_{t-1} + eta_ch * pch * Delta_t - pdch * Delta_t / eta_dch`

`pnet^2 + qnet^2 <= S_rated^2`

with complementarity on `pch * pdch`.

The prototype should determine whether `sch/sdch`, `ess_snet_def`, `sess_snet_def`, and the associated link equations can be removed safely from the network SMOPF.

This is a deliberate physical reformulation and is not authorized for production until the consumer trace, physics checks, rank/conditioning tests, and bootstrap solver comparison pass.

End-to-end ESSO throughput/degradation conversion remains a follow-on stage if B3 is favorable.

## Current decision discipline

- Do not test further shared-ESS epsilon values.
- Do not test further scalar `kappa` caps.
- Do not use solver-option tuning as a substitute for formulation work.
- B1, B2-R, and B3 are isolated experiments and each starts from the same accepted production baseline.
- A favorable B result is reported for planner review before any productionization or stacking.

# P5.3 invariants and prohibitions (historical)

During P5.3:

- do not tune IPOPT tolerances;
- do not increase `max_iter` as a solution;
- do not switch production MA97/MA57 policy;
- do not change ADMM rho rules or tolerances;
- do not change recourse-stationarity criteria;
- do not change common ADMM objective scaling;
- do not change TSO proximal regularization;
- do not change Benders/local-cut logic;
- do not add generic feasibility slacks;
- do not productionize the P5.2 narrow band;
- do not implement a scalar cap on shared-ESS `kappa`;
- do not change the stochastic samples during the exact RES algebra A/B;
- do not add calendar degradation;
- do not change terminal salvage;
- do not silently change complementarity tolerances.

P5.3 diagnostic A/B branches must be isolated. Reference-angle (B1), RES capability semantics (B2-R), and active-power ESS (B3) prototypes each start from the same accepted production baseline rather than stacking changes.

---

# Completed repository-wide work retained from earlier stages

## First-stage investment formulation

- ESS power and energy investments are scenario-independent variables indexed by ESS and investment year.
- Scenario-dependent investment-cost coefficients/probabilities remain active.
- The budget uses expected scenario-weighted expenditure.
- Results report one implementable physical investment plan.

## Solver separation

- LP master: Clp.
- Nonlinear operational subproblems: IPOPT.
- NLP and LP solver paths/configuration remain separated.

## Benders-type objective accounting

- Master estimate = investment cost + alpha.
- Gross recourse aggregates discounted/annualized TSO and DSO base SMOPF objectives.
- Net recourse subtracts terminal shared-ESS salvage.
- ESSO feasibility penalties and ADMM augmentation are excluded from economic recourse.
- The procedure is described as Benders-type with local sensitivity cuts; global lower-bound guarantees are not claimed.
- Operational non-convergence or material ESSO infeasibility stops the outer loop rather than generating a formal feasibility cut.

## Minimum SoH

- Minimum SoH remains enforced through the available-energy inequality and the rated-energy/cumulative-SoH identity.
- `soh_min = 0.50` remains intentionally hard-coded for the current baseline.
- Configurable minimum SoH remains deferred.

## ADMM convergence and adaptive penalties

- Convergence is evaluated after complete DSO-TSO-ESSO cycles.
- Interface voltage, P/Q flow, and shared-ESS consensus residuals are monitored separately.
- Economic recourse stationarity is required.
- Adaptive rho updates use tolerance-normalized residual balancing and hold groups already satisfying both primal and dual criteria.
- Failed local NLP cycles do not count as converged and do not update penalties from unreliable residuals.
- The method remains a nonconvex ADMM heuristic.

## Failure gating

- Failed initialization stops before ADMM.
- Failed ADMM-cycle blocks do not replace retained successful schedules or update their coupled duals.
- Success predicates use termination condition, not solver status alone.

## Terminal salvage

- Terminal salvage is an outer net-recourse credit, not part of the ESSO feasibility objective.
- SRP1 values battery energy capacity only, with remaining-calendar-life and normalized-health factors.
- Power-converter capacity is excluded from battery-health salvage.
- Calibration of the provisional salvage fractions remains required before final paper runs.

## Incumbent preservation and local validation

- The best feasible incumbent is preserved separately from later rejected candidates.
- Outer termination is explicitly classified.
- Local sensitivity/finite-difference validation infrastructure exists, but final reviewer-facing derivative validation remains deferred until the formulation is stable.

## Direct voltage-magnitude slack formulation

- Rectangular component slacks and `e_actual/f_actual` auxiliaries were removed.
- Voltage-limit relaxation uses lower/upper nonnegative squared-voltage slacks around the physical `e,f` voltage.
- Reference/interface/enforced-PV nodes retain hard voltage behavior as configured.

## Directional branch apparent-power limits

- Apparent-power limits are enforced at both terminals where required.
- Sending/reverse terminal reactive-flow definitions use consistent half-shunt accounting.
- Apparent-power auxiliary variables are indexed only where actually needed.
- Directional branch-loading/slack results are exported consistently.

---

# Active-energy SOC and degradation correction — IMPLEMENTED BASELINE

The earlier physical inconsistency in which apparent charge/discharge could alter stored battery energy has now been corrected in production.

Current network ESS baseline:

`pnet = pch - pdch`

`SOC_t = SOC_(t-1) + eta_ch * pch * Delta_t - pdch * Delta_t / eta_dch`.

Reactive power remains constrained by converter apparent-power capability but does not directly change battery stored energy.

Ordinary network ESS uses the same active-energy convention.

The ESSO model has no SOC state variable. Its existing degradation/throughput path has been corrected to use cell-side active-energy throughput:

`E_throughput = sum_d sum_t weight_d * Delta_t * (eta_ch * P_ch[d,t] + P_dch[d,t] / eta_dch)`

while preserving the pre-existing representative-day, cohort, year, equivalent-cycle and SoH semantics.

Local charge/discharge complementarity has since been resolved by accepted P5.4-H1 using dimensionless internal charge/discharge variables with `ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`. That nonlinear formulation remains the accepted production baseline; complementarity is deliberately relaxed only in the proposed future convex lower-bound model if required for convexity.

---

# Calendar degradation

Calendar degradation remains **deferred**. The active-energy SOC/cycling baseline has been validated, but calendar-degradation implementation is not part of P5.5-C and requires separate planner authorization.

Target conceptual extension:

`SoH_cumul[k,y] = SoH_cumul[k,y-1] * SoH_cycle[k,y]^(365 * Delta_y) * phi_cal[k]^Delta_y`

with `0 < phi_cal <= 1` and disabled compatibility case `phi_cal = 1`.

Keep `phi_cal` conceptually separate from the existing calendar-life/retirement parameter used for cohort retirement and salvage.

Do not begin calendar-degradation implementation during P5.5-C.

---

# Final-paper work still deferred

After the numerical and physical formulation stabilizes:

1. validate nonzero salvage on a controlled later investment cohort;
2. validate local sensitivities/finite differences at polished operational states;
3. revisit incumbent-centered trust-region/local-cut stabilization only if the evidence requires it;
4. run no-degradation / cycling-only / cycling+calendar experiment matrix;
5. run calendar-retention and discount-rate sensitivities;
6. calibrate provisional salvage parameters;
7. reconcile manuscript equations, algorithms, terminology, convergence claims, numerical tables, and response letter with the verified implementation.

Never describe the local-cut master estimate as a rigorous global lower bound or the procedure as globally convergent Benders decomposition.

---

# ACTIVE TRACK — ADMM recourse stabilization (from 2026-09-12)

Supersedes the previous "Immediate instruction" block (P5.12-R, complete). The outer
Benders-like layer is **deferred**: once stabilization holds, that heuristic will either
be rethought or replaced by a new outer layer, and the discussion is explicitly out of
scope until then.

## Premise correction — stabilization is not a convexity argument

**Do not carry the rationale that stabilizing ADMM makes the recourse convex, or that it
is what Benders validity waits on.** These are different objects:

- **ADMM stabilization** concerns whether the algorithm reliably reaches a fixed point
  for a given investment vector `x`. It is a property of the iteration.
- **Benders validity** concerns whether `Q(x)`, the recourse value as a function of `x`,
  is convex. It is a property of the underlying problem.

The recourse is an AC OPF and is nonconvex in the power-flow equations. It remains
nonconvex however well the iteration behaves. The repository already recorded this:
P5.8's verdict was "ADMM objective scaling validated, but it is **necessary and not
sufficient**" (`bff465d7`) and P5.9's was "rescaling fixes coverage and the landscape,
**coordination does not transfer**" (`fe6b5c63`).

## The correct rationale for stabilizing first

1. **Reproducibility.** P5.12-P showed `Q` depends on the solver path. Until `Q(x)` is
   deterministic, nothing downstream — cut validity, convexity, candidate rankings — can
   be measured at all.
2. **Cost.** At the observed descent ratio and ~51 local solves per cycle, one candidate
   evaluation approaches ~4,600 local solves. That gates every outer method equally,
   Benders and derivative-free alike.
3. **Testability.** Convexity cannot even be tested without a cheap, reproducible oracle.

## "Deferred" does not mean "legacy"

The Benders layer is **live production code**, not something only the paper describes:
`add_benders_cut` (`shared_resources_planning.py:180`), `upper_bound`/`lower_bound`
(`:291-293`), `gap_abs`/`gap_rel` (`:302-303`, `:430-433`) and `num_max_iters` (`:332`)
are all in the active path, and `planning_parameters.py:2,14` imports and instantiates
`BendersParameters` — despite the branch name `feature/derivative-free-planning`.

## Order of work

**Step 0 — activate C3 before stabilizing.** `(N, D, R) = (10000, 0.80, 0.50)`,
`k = 11542`, under a frozen impact gate. This changes the degradation term and therefore
the recourse; stabilizing against the current objective would mean stabilizing against
one about to change. Folded into the same gated stage as two determinism fixes that are
stabilization work rather than housekeeping: pin `fixed_variable_treatment` explicitly
(removing the silent dependence on the IPOPT 3.14 default), then replace the
`_in.update(_out)` warm-start merge with a clean assignment — provably inert once the
first is pinned, and it removes the latent path by which stale multipliers could reach
the solver.

**S1 — criterion validity. Zero solves. First.** `stationarity_pf` is the binding
criterion on 93 of 93 cycles in `p510_e_criteria.json`, unchanged across
`rho_pf` in {300, 500, 1000} and in the endpoint and replay cases, while the objective
criterion sits 3.47x from satisfied at cycle 20. Two criteria — one never satisfiable and
one merely far away — is a sign that part of "ADMM does not converge" may be an artefact
of a mis-scaled test rather than a property of the iteration. Analyse the 93 preserved
`binding_slack` values from existing artifacts: is the pf stationarity measure
approaching satisfaction, static or diverging; is it scale-dependent; is its
normalization commensurate with `objective_tolerance = max(1e3, 1e-3 * recourse)`; and is
the per-family `dual_<group>_mean_ratio` construction the right aggregation? Freeze a
spec first. **Change no tolerance, criterion or parameter — diagnose only.**

**S2 — convergence rate. The cost driver.** Verify first: the per-cycle objective
decrements from cycle 6 form a geometric sequence whose ratio climbs
`0.87 -> 0.94 -> 0.97 -> 0.982`, with the cycle-20 decrement still `4.95e6` against a
`1.42e6` tolerance — extrapolating to roughly 69 further cycles with about 19% of the
objective still to fall. Record plainly what follows: **the cycle-21 failure interrupted
a trajectory roughly a quarter of the way through, not one near convergence.** Then, before
proposing any lever, survey the prior art on the abandoned branches (read-only; no
checkout, no merge): `origin/admm_residual_balancing_tests`, `residual_balancing_mod`,
`origin/check_convergence_per_adn` (relevant to S1), `admm_initialization`,
`admm_prev_iter_vars`, `primal_value_update`, `admm_loop_corrections`,
`consensus_vars_sess_prev_iter`, `warm_start_tests`. Report what was tried, what the
commit messages claim, and whether any of it was ever evaluated. Already tried, not to be
repeated without new reason: `rho_pf` in {300, 500, 1000}, adaptive penalty off, objective
rescaling. Record remaining untried levers as candidates with rationale; run none.

**S3 — local solve reliability. Deprioritized.** One failure in 1,095 solves across the
P5.12-R trajectory. The rate is recorded explicitly because the class has absorbed
several weeks and the number reframes its urgency. The cycle-21 mechanism stays open and
unpursued; the breadth line stays closed at n = 3 with the fixture pool exhausted.

## Prepared but NOT run — the convexity probe

Three collinear investment points and a secant test on `Q`: a single violation of
`Q(0.5(x1+x2)) <= 0.5(Q(x1) + Q(x2))` would permanently retire the global-cut claim.
`benders_parameters` already carries `sensitivity_probe` and `finite_difference` groups.
The design is to be frozen and costed two ways — the cheap template-and-polish path in
the manner of P5.10-F, and a cold oracle — stating clearly that **the two test different
functions**. It runs after stabilization, as part of the outer-layer discussion.

## Out of scope until stabilization holds

Outer-layer redesign, cut validity, the 0.50% gap question, cohort SOC realizability,
penalty classification, the P5.10 recomputation. **Option A remains the interim
throughput reading**; B and C stay as formulation items. Penalty classification remains
blocking for the ranking-baseline re-derivation.

## Standing rules for this track

Frozen, hashed specifications before execution; gates carrying both a determinism control
and a negative control; no solves outside an authorized stage; the five artifact rules of
`CLAUDE.md`; `git log --full-history` for any file archaeology. **Report and stop at each
stage boundary** — S1 before S2, and S2's prior-art survey before any lever is proposed.

## P5.12-G / W / X / Y / Z results — 2026-09-12

All four stages were planned, executed and reported by the Planner alone: the Worker
has been unavailable since P5.12-G (external API spend limit). This removes the
independent-execution check the workflow relies on. The concrete cost is recorded
under "Cost of the missing independent check" below, as an instance rather than a
generality.

### P5.12-G — NO ARM-A-UNIQUE LOCAL-GEOMETRY SIGNATURE

Report `P5_12_G_GEOMETRY_FORENSIC_REPORT.md`; evidence `data/SRP1/Results/P512G/`.
Zero solves. Gates: 9166 optimization columns, 0 AD failures, 4 Jacobian builds.

Negatives: `delta_c` is zero in **all three** runs, so strong-form constraint-Jacobian
rank failure requiring IPOPT constraint regularization is unsupported (this does not
prove LICQ, full rank or good conditioning). `delta_x` fires 36 / 27 / 70 times
(V1 / Arm A / V2) and **never during the Arm A stall**; the converged controls
regularize far more, so its frequency cannot explain the failure. Row-norm ladders are
identical across all four states and the Arm-A-unique small-gradient set is empty.
One Arm-A-unique near-bound variable at 1e-4 (`flex_p_up[2,0,0,6]`). The blocking
variable is **not identifiable** from preserved artifacts because `d_x` was never
preserved. Safeguard corrections span 3.37e-08 to 9.7e-07, mean 4.02e-08 — cosmetic
by magnitude. Line-search rejection is excluded: every line-search event in all three
runs is single-trial.

SUPPORTED (carried, not only the negatives): **primal/dual activity disagreement is
materially stronger in Arm A** — 1912 rows dual-active but primal-inactive against
249 (V1) and 251 (V2), with 6182 dual-active rows against 4483 each. G's own confound
stands as G stated it: Arm A is infeasible at termination (6.19e-02) while both
controls are feasible (1.96e-07, 6.88e-12), and a violated row carries a large
multiplier with a large slack, which produces exactly this pattern. The signal is
therefore substantially, and possibly wholly, a restatement of infeasibility.

Tier-2 spectral/SVD analysis remains **unauthorized**: triggers 1 and 3 are not met and
trigger 2 is confounded as above.

### P5.12-W / X / Y — breadth probe, and P5.12-Z re-derivation

`BREADTH PROBE INCONCLUSIVE`. Reports `P5_12_W_BREADTH_PROBE_REPORT.md`,
`P5_12_X_COMPARATOR_BREADTH_REPORT.md`, `P5_12_Y_TSO_COMPARATOR_REPORT.md`.
All classification numbers were re-derived from primary artifacts by
`p512_z_classification.py` under a frozen specification; results in
`data/SRP1/Results/P512Z/classification.json`. Zero solves in Z; guards are
**enforcing** (they replace `OptSolver.solve` and `_execute_command` with functions
that raise), so a completed run with both counters at zero is genuine evidence.

**Y ladder authorization.** The user approved, **before execution**, applying the
+/-10x ladder around this fixture's own historical baseline: BASELINE `1e-6`,
LOW `1e-7`, HIGH `1e-5`. The perturbation FAMILY is common across fixtures; the
absolute values differ because case9's configured `bound_push` is `1e-6`. This
**closes** the `NOT RUN — INSTRUCTION CONFLICT, REFERRED TO USER` status recorded in
the P5.12-X report.

Both comparator reconstruction gates passed **byte-identically** against the historical
cycle-7 traces: DSO 0 differing lines in 7691, TSO 0 in 3118, normalizing only
`output_file`, `Total seconds in IPOPT` and the fresh-file leading blank line.
Preserved comparator fixtures are therefore faithfully replayable from a preserved
pre-solve model plus deterministically reconstructed context.

| fixture | validity | objective | interface (all 3 families) | branch | path |
|---|---|---|---|---|---|
| `TARGET_cycle20` (2025 Spring, w=460) | ROBUST | EQUIVALENT (2.21e-10 / 1.33e-08) | EQUIVALENT (worst 7.45e-11 / 8.00e-09) | EQUIVALENT | not material (0.9% / 3.5%) |
| DSO `case33_2` c7 (2025 Autumn, w=455) | ROBUST | **INDETERMINATE** (1.357e-05 / 3.650e-06) | EQUIVALENT (worst 1.60e-08) | EQUIVALENT | not material (12.1% / 37.4%) |
| TSO `case9` c7 (2025 Summer, w=455) | ROBUST | EQUIVALENT (1.72e-16 / 0.0) | EQUIVALENT (2.14e-16, `pf_p` and `pf_q` exactly 0.0) | EQUIVALENT | not material (2.7%) |

Breadth counts over previously-unknown fixtures, with `TARGET_cycle21` excluded as
KNOWN POSITIVE CONTROL: **3 tested; 0 outcome flips; 0 operational-output/branch
flips; 0 path-only; 2 robust on every axis; 1 robust except objective-equivalence
INDETERMINATE.**

Weighted objective differences (descriptive only, weight = `N_year * D_day *
annualization`): W 1.37e-04 and 8.23e-03 planning units; X **2.186** and 0.588;
Y 2.65e-08 and 0. X's 2.19 is ~10% of the accepted 22.09 cross-depth uncertainty and
~7% of the 32.87 best-to-second gap — non-negligible, not decisive, and it does not
alter any threshold.

**Two structural limits qualify how much weight the INCONCLUSIVE verdict carries.**
All three previously-unknown fixtures are `matched_success` captures, so selection is
conditioned on historical success and the sample cannot estimate the prevalence of
fragility. And none sits in the failure's difficulty stratum: baselines of 37, 91 and
115 iterations against the failure's 3000, the TSO fixture having only 2934 columns
and converging in 0.08 s with zero safeguard activity and arms agreeing to 1e-16, so
its diagnostic power against oracle fragility is low. Neither limit changes any frozen
threshold or classification.

**Fixture-pool exhaustion.** The preserved replayable base is spent at n = 3. Breadth
cannot be advanced from preserved artifacts; new preserved captures would be required.

**Defect phrasing, to be used wherever this enters the record.** At cycle 21 the
production-configured `1e-5` is the value that **fails**, while `1e-6` and `1e-4` both
converge. The defect is that the production setting is the failing one on this
instance — not that perturbation breaks a working solve.

### Corrected fact — constraint scaling is fixture-dependent

Superseding the earlier over-generalized note: Arm A and DSO `case33_2` report
`c scaling provided`; TSO `case9` reports **`No c scaling provided`**. Constraint
scaling therefore differs by fixture, and that is itself a modelling-invariant
conditioning difference between agent types bearing on cross-agent comparability.
Recorded alongside the fixture-dependent equal-bound removal count (W 102, X **100**,
Y 102, against NL columns 9268 / 9268 / 3036 and IPOPT-reported 9166 / 9168 / 2934,
each cross-checked against its own log). Historical P5.3 reports are not rewritten.

### Gap 3 — structurally moot, with a reusable corollary

`C_nl` and `C_opt` give identical maxima and identical threshold counts. This is
**structural, not a coincidence of these data**: the columns in `C_nl \ C_opt` are
exactly the equal-bound variables IPOPT removes under `make_parameter`, whose lower and
upper bounds are equal, so their values are identical in baseline and every arm by
construction, their scaled distance is identically zero, and they can never be the
argmax nor cross 1e-3. For a maximum or a threshold count the two sets must agree, for
any fixture. **Corollary:** the distinction would NOT be moot for any future mean-,
sum- or norm-based metric, where 100-102 guaranteed zeros would dilute the statistic.

### Interpretation note — the TSO interface zeros

`expected_interface_pf_p` and `_pf_q` differ by exactly 0.0 between arms in the Y
fixture. Verified from the NL b-section rather than from the pickle: both families are
**bound code 3 (free)** across all 72 entries each, while `expected_interface_vmag` is
**code 2 (lower bound only)**. So the two families that are exactly 0.0 are the free
ones, and the single family that moved — by 2.14e-16 — is the bounded one. The zeros
are therefore a genuine measurement at `.sol` precision and a property of the fixture's
solution, not a pinning artifact and not a measurement gap.

The same NL b-section independently confirms the equal-bound accounting: Y's histogram
is 102 code-4, 1770 code-0, 300 code-2, 864 code-3, so 3036 - 102 = 2934 matches
IPOPT's reported column count, and 1770 / 300 match the log's "lower and upper bounds"
and "only lower bounds" lines. Amendment 1's accounting and the structural-mootness
argument both stand on the NL alone, independent of any harness.

### Cost of the missing independent check

Two interface cells were recorded as EQUIVALENT without measurement: `TARGET_cycle20`
was never measured at all, and TSO `case9` was measured on `expected_interface_vmag`
alone while `_pf_p` and `_pf_q` were present and uncompared. The earlier rationale
that this was "consistent with a transmission block's interface role" is **false and
withdrawn** — all three fixtures carry all three families (24 / 24 / 72 entries).
P5.12-Z completed both measurements and both classifications were confirmed unchanged.
This is the concrete cost of losing the independent-execution check, and it is the
strongest available argument for restoring the Worker or an equivalent check before
any capture campaign.

### Housekeeping

- Classification evidence had **no preserved generator** until P5.12-Z; the W/X/Y
  numbers came from unpreserved ad-hoc steps. `p512_z_classification.py` plus its
  frozen specification now supply a reproducible derivation.
- `fresh_planning`-based stages are **not write-contained**: X and Y wrote 52 scenario
  diagram PDFs under `data/SRP1/Diagrams/`, outside the stage directories (untracked,
  no frozen or tracked artifact touched). Do not assert write containment for any
  future `fresh_planning` stage. P5.12-Z made no such call and its containment was
  confirmed after the run.
- **Duplicate fixture basenames carry different hashes**: the `matched_success_*`
  pickles exist in both `data/SRP1/Results/FrozenSMOPF/` and
  `data/SRP1/Results/P512R/production_snapshots/FrozenSMOPF/` (TSO `6b94d87c…` vs
  `2814948e…`; DSO `0bf972b9…` vs `d2342ba2…`, consistently +91 bytes). X and Y used
  and froze the `production_snapshots` copies. Only the path disambiguates.
- `p512_y_tso_replay.py` self-identifies as P5.12-X in its docstring, usage line and
  console tag. Harness bytes are left exactly as run.
- **Frozen-artifact naming rule (general convention).** The approved P5.12-Z
  specification was overwritten in place when the user's amendments were applied. v1,
  SHA-256 `5545fc65bfddf077844f2525796b0ef31a729f0845124d913257c434982eb261`, is
  **unrecoverable** — its bytes exist nowhere in the repository and the hash is
  recorded from the approving message. The executed specification is v2,
  `45edc424cd52edc9611a7bc220107f8fb30fefa4fadc5c3c486dc209cbdeb66b`, now stored as
  `frozen_formula_spec_v2_45edc424.json` with lineage in `frozen_spec_lineage.json`.
  Going forward: **frozen artifacts are named with version and content hash, are never
  replaced in place, and each version records its predecessor's hash.**
- **Evidence-staging rule (general convention).** Every committed report travels
  with its primary evidence base **and** that evidence's hash inventory. The first
  P5.12 preservation commit violated this in alternating directions: the G and T
  journals were committed without their manifests, while the K manifest was committed
  without its journal, leaving exactly one half of each pair for three stages. Like the
  frozen-artifact naming rule, this is a one-off correction promoted to a convention,
  because promotion is what prevents the same class of omission recurring.
- **Pattern, not three unrelated corrections.** Duplicate fixture basenames with
  different hashes; a frozen plan carrying the fixture-specific constant 9166 as if
  general; and a frozen approved artifact replaced in place — in each case an
  identifier fails to uniquely denote its content. Remedy: version-and-hash naming,
  operational rather than literal constants in frozen plans, path-qualified fixture
  identity.
- `COWORK_HANDOFF.md` was drafted by the user before P5.12-Y completed and is
  **superseded** by this section; it is not committed as current.
- **Provenance vocabulary.** Distinguish NUMERICAL FROZEN STATE (production source,
  numerical inputs, frozen fixtures, solve harness, solver/runtime configuration) —
  unchanged and verified throughout — from ORCHESTRATION-ONLY APPROVED DRIFT
  (`.claude/agents/planner.md` and `worker.md` model/effort edits). The repository was
  not literally frozen; the drift is recorded, not ignored, and does not touch the
  numerical experiment.
- Timestamps are standardised on Z-suffixed UTC. P5.12-R's trajectory ran 1176.3 s
  (19.6 min), which supersedes the ~16 min estimate used when sizing a future
  instrumented capture.

## Zero-salvage objective transition — incomparability register (2026-09-12)

The author's decision to exclude terminal salvage from the objective is
**definitional, not numerical**: it fixes what the oracle is supposed to compute.
It therefore has to precede any capture campaign rather than run alongside it —
captures taken under the current objective would be captures of an objective about to
change. This register enumerates, now, which existing results carry a salvage credit,
because the list is cheap to produce today and expensive to reconstruct afterwards.

**Magnitude note, added 2026-09-12 (from P5.14-G).** The definitional incomparability
stands and is unaffected by what follows. But the *numerical* weight of the salvage term
at the recourse level is now measured: from P5.9-B's `cost_families`,
`terminal_salvage_value` is **3,452.29** against `net_operational_recourse`
**828,248,310** — **4.17 ppm**. A future reader should therefore **not over-weight the
salvage boundary when comparing recourse values**: at this magnitude the salvage
convention cannot explain a recourse difference of even a few hundredths of a percent.
It must still be respected in full for (a) salvage-specific claims, (b) any ranking whose
best-to-second gap is narrow enough for 4 ppm to matter, and (c) the definitional
question of what the objective *is*, which is not a magnitude question at all.

**Mechanism.** `net_operational_recourse = gross_operational_cost -
terminal_salvage_value` (`shared_resources_planning.py:722-733`); the per-cycle ADMM
`recourse` diagnostic is that net value (`:395`); the block decomposition carries a
`('SALVAGE', None, None, None)` term with negative sign so the blocks sum to the net
recourse (`:736-745`).

### AFFECTED — carry a salvage credit; NOT comparable with any future zero-salvage result

- P5.6-A START-1: net recourse `829288237.862242`, total `829338237.862242`
  (salvage `3439.659877`).
- P5.6-B START-2, the current best certified incumbent: net recourse
  `827971090.360850`, total `828021090.360850` (salvage `3428.356255`).
- P5.6-B template chain totals T0..T4: `828021090.360850`, `827415318.563944`,
  `826824028.845478`, `826405022.193437`, `825961521.882321`, and the T0->T4 drift
  `-2.059568e6`.
- P5.6-D: best incumbent `se|node5|2025|-10%` total `825109566.571083`;
  `tau_planning_refined = 811438.05`.
- P5.4-R canonical: `Q0 = 838496830.813414`; D4 cold base `838496830.81`; best
  recovered branch `836586463.43`; `tol_cut = 7.164e5`.
- P5.10 / P5.11 reproduction gates: CURRENT polished total `828021090.3608505`;
  RESCALED pre-polish recourse `825814074.4930633`.
- **P5.10 landscape statistics**: cross-depth relative uncertainty `22.09`,
  uncertainty/signal `0.67`, best-to-second mean gap `32.87`, base-chain drift
  `-201099`. Also P5.9's CURRENT eight-depth uncertainty `459695.59` against signal
  `18085.03`.
- P5.11 / P5.12-A / P5.12-R per-cycle recourse markers `2352009862.13208` (cycle 1),
  `1461174062.836328` (13), `1424778916.2652295` (20), `1402384338.113119` (25), and
  every per-cycle recourse in the P512A and P512R trajectory records.
- P5.7 recovered-objective deltas `-383401.68` (CURRENT) and `-8892463.76` (RESCALED),
  computed on the same net convention.

### UNAFFECTED — salvage-free by construction

- `gross_operational_cost` figures (`829291677.522120`, `827974518.717105`).
- **All block-level IPOPT objectives and residuals** in P5.12-B / R / Arm A / P / W /
  X / Y / Z — `1348.4251823032162`, `354.15536755956111`, `337946.00264311919` and
  their arms — because salvage is an outer ESSO-derived credit that never enters a
  network block objective.
- All solver telemetry: iteration counts, safeguard counts, `delta_c` / `delta_x`,
  step sizes, residuals, complementarity, barrier progression.
- All P5.12-Z classification metrics: primal distances, interface distances, iteration
  ratios. The weighted figures (X LOW `2.186` planning units, etc.) are block-objective
  differences multiplied by the planning weight and are therefore salvage-free — but
  they are expressed in planning units and may only be compared against a
  same-convention planning signal.
- P5.5 convex-oracle results, which already use a different salvage convention
  (`-V_salvage_max`, salvage excluded from `R(x)`).

### Consequences to carry

1. The P5.10 acceptance statistics (`22.09`, `32.87`, `0.67`) are stated on the
   salvage-inclusive objective. **Any future ranking evidence gathered under zero
   salvage must re-derive its own uncertainty and signal baseline**; these numbers may
   not be reused as thresholds across the transition.
2. For the recorded early-cohort base candidate the credit is about `3.4e3` on `8.3e8`
   (~`4e-6` relative), so the numerical shift is small while the **definition** differs.
   The Expert's warning stands: early cohorts mask this because remaining-life salvage
   can be zero; a late cohort would not.
3. The transition must be applied consistently across operational evaluation, candidate
   ranking and any sensitivity or cut accounting, or results from the three will not be
   mutually comparable either.

### Convertibility audit — most of the affected list survives the transition

Audited against preserved artifacts rather than assumed. **Conversion rule:** the
zero-salvage value is `gross_operational_cost`, which is stored directly wherever the
cost families were preserved; equivalently `net + terminal_salvage_value`, and for a
total, `total + terminal_salvage_value`. This is arithmetic on existing artifacts —
**no re-solve**.

**CONVERTIBLE — gross and/or salvage preserved alongside the reported value:**

- P5.6-A START-1 — `P56A/p56a_a1_certificate.json` holds
  `gross_operational_cost = 829291677.5221196`, `physical_salvage = 3439.6598771430236`,
  `net = 829288237.8622425`, `total_objective = 829338237.8622425`.
- **P5.4-R `Q0`** — the same certificate's `admm` block holds
  `gross_operational_cost = 838500270.3685532`, `terminal_salvage_value =
  3439.5551395370985`, `net_operational_recourse = 838496830.8134136`.
- P5.6-B template chain T0..T4 — `P56B/p56b_b1_template.json` carries the keys.
- P5.6-D `tau_planning_refined` — `P56D/p56d_depths.json`, `p56d_k12.json`.
- P5.7 (3 files), P5.8 (4 files), P5.9 (14 files) carry the keys.
- **P5.10, including the acceptance statistics.** `p510_b_fixedrho_pf{300,500,1000}.json`
  carry `cost_families` per row, and — decisively — the eight-generation replay files
  `p510_f_replay_*.json` carry `cost_families` **per generation** — each file declares
  `generations: 8` and holds exactly 8 `gross_operational_cost` occurrences, for all
  three candidates (e.g. gross `827965125.950184`, salvage `3469.7529292681957`, net
  `827961656.1972548`). The **inputs** to `22.09` / `32.87` / `0.67` are therefore
  preserved at every generation. Their **definition** is not: see the separate entry
  below. They are not simply "recoverable".
- All per-cycle recourse markers — `P512A` trajectory files store
  `gross_operational_cost` beside `recourse`, and `P512R/run.json` stores gross,
  salvage and recourse together. Cycle 20 reconciles **to double precision, not to the
  last printed digit**: `1424782485.8041718 − 3569.5389424068017 = 1424778916.26522939`
  against the recorded `1424778916.2652295`, a relative residual of about `7.6e-17`.
  That is float round-off in the decimal transcription, not a data problem; a reader
  recomputing from the printed strings will see a last-digit difference and should
  expect it.

**NEEDS A ONE-OFF CROSS-CHECK — the stage's own artifacts lack the split:**

- P5.11's two reproduction gates (`828021090.3608505`, `825814074.4930633`): none of
  the six `P511/*.json` files carries the cost families. The same quantities do appear
  in `P56B` / `P510` artifacts that do carry them, so the split is very likely
  recoverable by matching values across stages — but that match must be verified, not
  assumed.
- P5.4-R `tol_cut = 7.164e5`: `P54R` itself carries no cost families; `P54R_D3` has
  four key-bearing files. Its derivation from the recourse scale must be re-traced.

**Consequence.** The earlier statement that the affected results are simply
"incomparable" was too strong. Most are **convertible by arithmetic on preserved
artifacts**; only the two items above need tracing. What remains true is that no
salvage-inclusive number may be compared against a zero-salvage number without
performing the conversion first.

### Dependency — a zero-salvage objective is not automatically an economic objective

`_get_operational_recourse_components` states in its own docstring that the quantity
"excludes scenario-deviation regularization and ADMM augmentation terms, but **may
include artificial penalty terms and is therefore not necessarily a pure economic
operating cost**" (`shared_resources_planning.py:723-724`).

So `gross_operational_cost` — precisely the quantity the zero-salvage convention
promotes to the objective — carries penalties of unstated status by its own admission.
A zero-salvage ranking baseline is therefore **not** an economic baseline until those
penalties are classified, which is the Expert's requirement to state which penalties
are economic costs and which must vanish for physical feasibility.

**Sequencing consequence, recorded:** penalty classification lands in the *same
quantity* as the salvage transition and cannot be deferred behind it. Both must be
settled before any ranking baseline is re-derived, or the baseline would have to be
derived twice.

**Concrete instance — the complementarity slack (added 2026-09-12, from P5.13-A).**
`slack_es_ch_comp_per_unit` (`shared_energy_storage_data.py:569`, penalized at `:641`
via `PENALTY_ESSO_SLACK`) would silently absorb the Option B throughput definition:
expectation-valued directional powers violate the 1e-4 complementarity row by ~2500x
in the product, the run does not fail, and the violation appears as cost inside
`gross_operational_cost`. This is a penalty **absorbing** a modelling inconsistency
rather than **signalling** one. It raises penalty classification from accounting
hygiene to failure detection: a penalty classified as operating cost cannot signal a
modelling inconsistency, it prices one. The classification must therefore state, per
penalty, whether a nonzero value is an economic cost or a *detector* that must be
zero — and for this slack the two readings differ in kind.


### P5.10 acceptance statistics — inputs preserved, definition UNPRESERVED

Searched rather than assumed. `22.09`, `32.87` and `0.67` appear in **no artifact**.
The apparent JSON matches are incidental substrings inside longer unrelated numbers
(`22.09141692720717` is a residual in the P512A trajectories; `22.091315306584715` is
in `P53B1`); a context search returns nothing. `p510_e_criteria.py` computes
`planning_signal` and `objective_tolerance` and nothing resembling these three, and
`p510_e_criteria.json` preserves `planning_signal = 33031.0` and `tau_numerical = 10.0`,
which are different quantities. The three figures exist only in Markdown: the P5.10 and
P5.11 reports, the P5.12-X and P5.12-Y reports, and these two governing documents.

So the position is narrower than "recoverable": **the inputs are preserved, the
definition is not.** Nothing on disk states what the cross-depth uncertainty ranges
over, how best-to-second is formed, or what the ratio divides. Recovering them means
reconstructing a formula whose only source is report prose.

This is the **fourth instance of the identifier/denotation pattern, and the most
consequential**, because these three function as acceptance thresholds for candidate
ranking — a heavier load than the classification metrics P5.12-Z re-derived.

**Protocol when the recomputation is authorized** (a Z-shaped stage, with one addition
that was unavailable in Z):

1. Freeze a formula specification reconstructed from the P5.10 report's prose, and have
   it approved, under the frozen-artifact naming rule.
2. **First reproduce the salvage-inclusive `22.09` / `32.87` / `0.67` under the old
   convention** from the preserved `cost_families`. That reproduction is what
   distinguishes a reconstructed definition from a guessed one.
3. Only once step 2 passes, compute the zero-salvage values.

If step 2 fails, the definition is wrong and the zero-salvage figures would have
inherited the error silently. The asymmetry that makes this work: in P5.12-Z the prose
numbers were the thing being checked and so could not also validate the method; here
the old-convention numbers are a genuine independent target, because what is unknown is
the formula, not the data.

**Hold** this recomputation until the throughput and penalty-classification items are
settled. Its inputs are not going anywhere, and re-deriving a ranking baseline before
the penalty status of `gross_operational_cost` is decided would mean doing it twice —
the same argument that made penalty classification blocking rather than sequential.

## P5.13-A — expected directional throughput versus expected net power (2026-09-12)

Definitional decision with code trace; no solver work. Full record in
`P5_13_A_THROUGHPUT_DEFINITION.md` (including corrections C1-C6 of 2026-09-12).
Frozen diagnostic specification, current version:
`data/SRP1/Results/P513A/frozen_throughput_diagnostic_spec_v2_8b2d8f77.json`.

**Established by code trace.** Network shared-ESS variables are per scenario
(`[e, s_m, s_o, p]`). Coordination averages the **signed net** quantity —
`dn_interface_expected_sess_p_def` (`model_construction_helpers.py:1798`) sums
`pi_m * pi_o * shared_es_pnet`, so opposite-sign dispatch cancels there. Consensus
parameters are indexed by period only. The ESSO then re-decomposes that expected net
schedule into its own directional `es_pch_per_unit` / `es_pdch_per_unit`, which carry
**no scenario index**, and degradation is driven by those
(`shared_energy_storage_data.py:~480`). So the degradation-driving quantity is the
directional throughput **of the expected schedule**, not the expectation of realized
directional throughput. The two differ whenever scenarios disagree in sign.

**The structural zero.** `SRP1.json` has `NumMarketScenarios = 1` and
`num_operation_scenarios = 1` for the TSO and all three DSOs, so this discrepancy is
**identically zero in this configuration by construction, not by measurement**.
Recorded explicitly: computing it here would yield zero and would be a false negative
that reads as validation. No quantification is attempted.

**One finding, not two.** The evaluation oracle hard-wires scenario `(0,0)` when reading
coordinated quantities (`p56a_oracle.py:275-287`, `:393-397`), while production's model
construction does sum over scenarios. Moving to the paper's 25 market/operational
combinations is therefore **not a data change the oracle would absorb** — it requires
changing that path. The Expert's point that the reduced-case certificate cannot carry
unchanged into the full stochastic study is a **direct consequence of this indexing**,
not a separate concern.

**Open definitional decision (author's to make).** Option A: degradation represents the
committed common schedule — the implementation is then correct as written and the
obligation is editorial, since the paper must not claim realized cycling. Option B:
degradation represents realized operation, driven by `E[pch]` and `E[pdch]` rather than
by a directional decomposition of the single `E[pnet]`. **Blocking:** whichever is
chosen changes the degradation term and hence the objective, so it must be settled
before any ranking baseline is re-derived — the same argument that made penalty
classification blocking.

**Option B is blocked by enforced complementarity — corrected 2026-09-12.** The first
record of this stage called the trade-off under B a weakening of "the ESSO-level
complementarity interpretation". That was too weak and made B look like the cheaper
option. Complementarity is *actively enforced*, and B's correct values violate it by
orders of magnitude. `ESS_COMPLEMENTARITY_TOLERANCE = SMALL_TOLERANCE = 1e-4`
(`definitions.py:80-81`) applies to **normalized** variables in `[0, 1]`, so equal
directional values are capped at `sqrt(1e-4)` = **1% of rating**. A two-scenario ±50/50
split needs `0.5` in each component: **50x over per component, 2500x over in the
product** — binding by three orders of magnitude, not marginally tight.

Three sites, two failure modes:

| # | Row | Location | Under Option B |
|---|---|---|---|
| 1 | `pch_hat * pdch_hat <= slack_es_ch_comp_per_unit + 1e-4` (per cohort) | `shared_energy_storage_data.py:569` | **slack-absorbed**; slack penalized at `:641` via `PENALTY_ESSO_SLACK`, landing in `gross_operational_cost` |
| 2 | `es_pch_hat_agg * es_pdch_hat_agg <= 1e-4` (aggregate) | `shared_energy_storage_data.py:618-620` | **no slack — infeasible** |
| 3 | `shared_es_pch_hat * shared_es_pdch_hat <= 1e-4` (per scenario) | `model_construction_helpers.py:836-838`, penalty `:1726` | rows hold (per-realization); the P5.4-H1.6 *rationale* for site 2 does not survive |

Site 3 requires care: the network rows are per scenario and are not violated by B.
What B breaks is the P5.4-H1.6 argument (`shared_energy_storage_data.py:599-605`) that
produced site 2 — that the ESSO aggregate feasible set must match the network's, "or
ADMM would be reconciling two different feasible sets". Under B the ESSO's directional
variables are expectations while the network's are per-realization, so the two sides
no longer describe the same object.

**Corrected cost of Option B.** Not "two expectations rather than one": the two ESSO
complementarity sites must be redefined or removed; the ADMM channel changes, because
`update_shared_energy_storage_model_to_admm` (`shared_resources_planning.py:3783-3827`)
reconciles exactly one expected net pair `(es_pnet, es_qnet)` against `(p_req, q_req)`
with one dual pair and one `rho`, so a directional pair adds a consensus quantity, a
dual, a residual and a stopping test per node and period; P5.4-H1.6 must be re-derived;
and the `(0,0)` oracle restriction still blocks any configuration that could exhibit
the discrepancy.

**Site 1 belongs to the penalty-classification item.** Adopt B's expectations without
the model change and nothing fails — the correct expected throughput appears as
`PENALTY_ESSO_SLACK` cost inside `gross_operational_cost`, the exact quantity whose
penalty status is already blocking. This is a concrete instance of a penalty
**absorbing** a modelling inconsistency rather than **signalling** one, and is the
strongest available argument that penalty status must be settled rather than
documented.

**Option C examined — NOT established.** Proposal: the networks pass `E[pch]`, `E[pdch]`
used *only* as inputs to the degradation law, leaving ESSO dispatch semantics and the
complementarity rows untouched. Three obstacles, the first structural:

- **O1 (index mismatch).** The law consumes `es_avg_ch_dch_per_unit[y_inv, y]`
  (`shared_energy_storage_data.py:479-490`), a per-**cohort**, per-year scalar already
  aggregated over days and periods, feeding a per-cohort SoH chain (`:509-532`) with
  per-cohort `es_e_rated_per_unit`. The network models have no cohort decomposition
  (`shared_es_pch[e, s_m, s_o, p]`). An exogenous expectation can constrain only the
  cohort sum, leaving the split the law consumes undetermined. This is the sharper form
  of the per-period concern: the consumed quantity is not per-period at all.
- **O2 (price channel).** In the operational subproblem the ESSO objective is
  `feasibility_penalty` plus the AL terms; degradation enters through constraints
  (throughput → degradation → SoH → `es_e_available_per_unit`) and reaches the economics
  via available capacity and salvage. Make throughput exogenous and degradation stops
  depending on any ESSO **dispatch** variable, severing the only channel — the ESSO's
  preference expressed through `dual_p_req` — by which cycling cost reaches the
  networks, whose objectives contain no degradation term (`penalty_ess_usage`
  `model_construction_helpers.py:1652`; complementarity penalty `:1726`). No agent then
  prices cycling against dispatch.
- **O3 (reconciled variant).** Reconciling throughput as a consensus quantity with its
  own dual forces the ESSO to reproduce an expected gross throughput while also meeting
  the `es_pnet` consensus and the complementarity rows — exactly the conflict that is
  infeasible at site 2 and slack-absorbed at site 1. The violation moves onto a residual
  that cannot close without slack.

Option C must not be cited as a cheaper alternative. O1 needs an explicit
cohort-allocation rule; O2/O3 show the hoped-for decoupling exists only in the form
that removes the degradation price. Secondary: any new consensus channel changes the
ADMM operator whose stability is the subject of the open cycle-21 investigation.

**Frozen specification, v2.**
`data/SRP1/Results/P513A/frozen_throughput_diagnostic_spec_v2_8b2d8f77.json`
(SHA-256 `8b2d8f77…a5ed`), lineage in the same directory's `frozen_spec_lineage.json`;
v1 `bf56149e…d311` preserved unmodified. v2 (a) preserves the derivation of the sign
expectation `A >= B` — for nonnegative directional powers `|E[X - Y]| <= E[X + Y]`, and
per-scenario complementarity makes `X + Y = |X - Y|` within a realization, so this is
`|E[z]| <= E[|z|]`; opposite-sign dispatch cancels in the expectation of the net and
adds in the expectation of the throughput — with its three assumptions and the
efficiency-independent equality condition; and (b) **corrects** v1's blanket rule that
any `A < B` is a diagnostic defect. With the site-1 slack active, both directional
values can be strictly positive and `B` can exceed `A`; `A < B` is then a *finding*
(slack absorption), and the diagnostic must report active slack alongside every `A`/`B`
pair. Neither version has been executed.

## P5.13-B — cycling calibration: `cl_nom`, `dod_nom`, `soh_min`, `t_cal` (2026-09-12)

Code trace, git history and arithmetic; no solver work, no production change. Full
record in `P5_13_B_CYCLING_CALIBRATION.md`.

**Primary finding — the (count, depth) pair was never maintained as a pair.** Recovered
from `git log --full-history` over `shared_energy_storage.py` (default history
simplification omits commits and must not be used for this question).
`cl_nom = 10000` dates from the initial commit (`4c6fef21`, 2024-04-08) and has **never
been changed**: an original modelling choice, not a leftover. If it was ever paired with
a depth, that depth was **0.95**, not 0.80 — `dod_nom` went `0.95 -> 0.90` the next day
(`ceb76b9a`) and `0.90 -> 0.80` twenty months later (`f952c969`, 2025-12-05), with the
count revisited at neither. `soh_min` did not exist when `cl_nom` was introduced; when it
appeared (`1a318339`) it was `0.10`, becoming `0.50` only on 2026-08-03 (`edcfbd95`).
So the defect is not that one of two readings of `cl_nom` is right — it is that a paired
`(count, depth)` specification was never maintained as a pair, because the law consumes
only the count.

**The apparent calibration is an accident.** Today's constants imply, at 10000 cycles of
depth 0.80, a retention `exp(-0.8) = 0.449`, close to today's `soh_min = 0.50`, which
makes the implementation look calibrated. It is not: the `0.80`/`0.50` pairing is the
residue of two independent later edits made twenty months apart, neither mentioning the
other quantity. Treated exactly like the two cancelling defects below — a coincidence
that must never be cited as justification.

**`t_cal` — a governance finding.** Seven value changes after the initial commit
(`20 -> 10 -> 20 -> 15 -> 10 -> 15 -> 20 -> 15`), two of them buried in commits
describing unrelated work: `39f83c47` "Planning. Test." (`15 -> 10`) and `7e3b3246`
"vmag_sqr redefined to vmag." (`10 -> 15`). `t_cal` gates cohort activation
(`shared_energy_storage_data.py:269`, `:432`, `:500`) and the salvage remaining-life
fraction (`:712-717`), so any result depending on it is interpretable only with its
commit pinned. Independent argument for the parameter move.

**The manuscript's 70% is DIVERGENT, not stale — revises the `EXPERT_REVIEW.md:51`
item.** `soh_min = 0.70` exists (`a3c76922`, 2026-07-29, "Default minimum SoH updated.")
but is **not on this branch**: it lives on `paper_revisions`, which is not an ancestor of
HEAD; the branches diverged at `285cbb42` (2026-07-28). `paper_revisions` is at **0.70
today**; `feature/derivative-free-planning`, which produced every P5.x result, went
`0.10 -> 0.50` on 2026-08-03 and is at **0.50 today**. The values are concurrent, not
sequential. The paper's 70% is supported by the code on its own branch and contradicted
by the code that generated the numbers; Table 7's 49.62% is consistent with the 0.50
floor binding. `soh_min` enters both the feasibility floor (`:529`) and the salvage
valuation (`:680`, `:748`, `:787`). The earlier phrasing — that the 70% claim had no
mechanism behind it — is **withdrawn**: it had one, on the other branch. The 60%/80%
sensitivity cases remain unsupported on either branch. Note also that the 0.50 value has
its own undocumented origin: `edcfbd95`'s message does not mention changing a default
that had stood at 0.10 for over two years.

**The law assumes a reference DoD of 1.0.** The throughput accumulator is cell-side
energy (`eff_ch*pch*dt + pdch*dt/eff_dch`, `:479-490`), so a full-depth cycle contributes
exactly `2E`; the normalization `2*cl_nom*E` (`:509`) is thus `cl_nom` cycles at
**DoD = 1.0** while the object declares depth `0.80` — the `1.25x` arm of the unmaintained
pair.

**`cl_nom` functions as a decay constant, not as cycles-to-EOL.** The SoH chain is
multiplicative per day (`:513`/`:515`, `:524`/`:526`, exponent `365*5 = 1825`), so
retention is `~exp(-EFC_total/cl_nom)`: at exactly `cl_nom` EFC the model retains
**36.8%**; SoH 0.50 at **6931 EFC**, SoH 0.80 at **2231 EFC**, rate-independent.
**Correction to earlier advice:** defaulting to the manufacturer-count reading because it
is conservative was wrong as general advice — depth and end-of-life retention point in
opposite directions. Depth gives `1.25x` towards shorter life; but manufacturer counts
are conventionally quoted to 70-80% SoH, and under that convention the implemented law
**understates** life by roughly `3.6x`. The conservative-default instruction is withdrawn.

**Also recorded.** (a) The comment at `:511` calls `es_soh_per_unit` the "Annual SoH"; it
is the **daily** retention factor — a factor-365 trap. (b) `es_degradation_per_unit ~ 1e-4`
against a bound of 1.0, multiplied by `20000*E_rated` in a bilinear equality: a
**candidate** row-scaling item for the NLP-stability work, with no causal claim.

**The decision, reframed as a triple.** State the calibration as `(N, D, R)` — cycles,
reference depth, end-of-life retention — and derive the law's constant
`k = N*D/(-ln R)`, using `2*k*E_rated` in place of `2*cl_nom*E_rated`. The law then
consumes all three, so the pair cannot silently break again, and `dod_nom` becomes
load-bearing rather than vestigial. The identity `EFC_to_R = -ln(R)*k = N*D` holds for
every triple, so each candidate reaches its own `R` at exactly `N*D` cycles and the
candidates differ only in curve steepness.

| # | `N` | `D` | `R` | `k` | `k`/10000 | 15-yr retention at 1 EFC/day |
|---|---|---|---|---|---|---|
| C1 as implemented | 10000 | 1.00 | 0.368 | 10000 | 1.000 | 0.5784 |
| C2 conventional datasheet | 10000 | 0.80 | 0.80 | 35851 | 3.585 | 0.8584 |
| C3 this branch's floor | 10000 | 0.80 | 0.50 | 11542 | 1.154 | 0.6223 |
| C4 manuscript branch's floor | 10000 | 0.80 | 0.70 | 22429 | 2.243 | 0.7834 |

Under C1 a cohort cycled once daily loses 42% of capacity over the horizon and
approaches the 0.50 floor; under C2 it loses 14%, the floor stops binding and the
salvage term changes materially. C3 is within 8% of present behaviour at 15 years; C2 and
C4 are not. C4 is on the table only because of the branch divergence, which means
choosing C2 or C3 also decides what the paper must say. **Anything other than
`k = 10000` changes the degradation term and hence the objective — blocking against the
ranking-baseline re-derivation.** The choice of `(N, D, R)` is the author's.

## P5.13-C — ESS ageing constants relocated into the parameters file (2026-09-12)

First production code change in this sequence, authorized as a behaviour-preserving
refactor. **Gate PASS.** No solve. Full record in `P5_13_C_PARAM_MOVE_REPORT.md`.

**What changed.** `t_cal`, `cl_nom`, `dod_nom`, `soh_min` are now read from
`data/SRP1/SharedESS/SRP1_ESS_Params.json` (`ageing` block) and applied in
`create_shared_energy_storages` via `self.params.ageing.apply_to(...)`;
`read_parameters_from_file` runs on the immediately preceding line
(`shared_resources_planning.py:6104-6105`), and each has exactly one call site. Parameter
defaults equal the former hard-coded values exactly, so an absent key changes nothing.
`shared_energy_storage.py` keeps the literals as annotated fallbacks (comment-only diff).

**Gate.** `data/SRP1/Results/P513C/frozen_param_move_gate_v1_bd8ab535.json`, frozen and
hashed before any edit; harness `p513_c_param_move_gate.py` (`99cae59e…b122`), which
builds the ESSO models through the production path. **Correction (P5.13-D):** that path
**solves** — `create_shared_energy_storage_model` calls `shared_ess_data.optimize(...)`
(`shared_resources_planning.py:3156`), three IPOPT solves per capture. The original
"runs no solver" claim was asserted, not traced, and is withdrawn. The evidence is
unaffected and in fact broader than described, since the compared state includes
post-solve variable values. Invariants:
ordered semantic state (I1), `.nl` bytes (I2), the four constants on every ESS object in
every year (I3), objective/penalty/salvage expression strings (I4), diff confinement (I5).

**Three runs, not one.** (a) *Determinism control* `pre` vs `pre2` on the unmodified
tree — identical, including `.nl` bytes; without it the gate would prove nothing.
(b) *Neutrality* `pre` vs `post_final` — **PASS on I1-I4**. (c) *Negative control*
`probe`, with `cycle_life_nominal` 10000 -> 12345 and `minimum_soh` 0.50 -> 0.55 —
**FAIL as required**, differing in I1, I2, I3 and the salvage expression, proving the
parameters file is genuinely load-bearing rather than inert. Probe reverted and
neutrality re-verified.

**Type preservation is the load-bearing detail.** `_read_optional_number` keeps a JSON
integer an integer: `cl_nom` enters a constraint expression, so coercing `10000` to
`10000.0` would change the rendered model and the `.nl` bytes. That is the easiest way
for a "neutral" refactor to stop being neutral.

**The calibration triple is DECLARED, NOT CONSUMED.** The schema carries
`ageing.calibration` (`status`, `cycles_n`, `reference_dod_d`, `eol_retention_r`, the
last `null`). Nothing in model construction reads it; `characteristic_constant()`
(`k = N*D/(-ln R)`) is called nowhere; any status other than `DECLARED_NOT_CONSUMED`
raises. Consuming it changes the objective and is the author's open decision, so it
cannot ride along inside a refactor advertised as neutral.

**Disclosed deviation (I5).** The frozen gate permitted three files; a fourth,
`shared_energy_storage.py`, was touched with a comment only (no `+`/`-` line contains an
assignment). Recorded as a deviation rather than absorbed by widening the frozen
specification after the fact; reverting it costs nothing.

**Limits.** The gate proves the model handed to the solver is byte-identical, so every
existing result remains valid without re-running any of them; identical `.nl` and
identical options imply an identical solve. But all three ESSO models produced the same
I1 and I2 hashes, so the gate exercises **one structure replicated three times**, not
three independent ones. The 15 ordered-state files (20 MB) are hash-recorded in the
manifests rather than committed.

## P5.13-D — Step 0: C3 activation and two determinism fixes — GATE FAILED AS FROZEN (2026-09-12)

Full record in `P5_13_D_C3_ACTIVATION_REPORT.md`. Frozen gate
`data/SRP1/Results/P513D/frozen_c3_impact_gate_v1_8ee17c92.json`. **The production
change is implemented but NOT COMMITTED**, pending the author's decision.

**The error, first.** The frozen gate asserted "no solver is invoked" and forbade solves.
That premise was false and was inherited from P5.13-C without tracing the call:
`create_shared_energy_storage_model` -> `shared_ess_data.optimize(...)`
(`shared_resources_planning.py:3156`) performs **three IPOPT solves per capture**. Two
consequences, both recorded: P5.13-C's committed "no solve" claim is corrected in place,
and my own gate was breached by my own harness. The breach is of my specification, not
of the authorization, and it is the direct cause of the mis-specified invariant below.

**Results.** E2 (the changed rows carry `2k = 23083.120654223414`, 6 of 30 rows, none
still carrying `20000`), E3 (`k = 11541.560327111707`), E5 (reversion byte-identical),
E6 (consistency guard), E7 (warm-start assignment drops stale keys) and E8 (option pin
on both paths, configuration still overriding) all **PASS**. **E1 FAILS as frozen**: the
degradation component changed as intended, and 19 `vars.*` components also changed —
but only in their **values**, which are post-solve values, and which must move when the
constant moves. E1 demanded byte-identical variables, impossible for a capture that
solves.

**The substantive result stands:** C3's structural impact is confined to the six
degradation rows, where the constant changes from `2*cl_nom` to `2k`, a factor
`1.1541560327111706`.

**E5 proves more than reversibility.** The reversion capture ran with the *new* code —
`cl_eff`, the pinned `fixed_variable_treatment`, the warm-start assignment — and only
the JSON status reverted, yet it reproduced the old-code baseline byte-identically,
including post-solve values. So the two determinism fixes are **empirically neutral**,
not merely neutral by argument, and C3 is reversible through one JSON field.

**The three changes.** (a) The law consumes `cl_eff` = `k` when the calibration is
ACTIVE and `cl_nom` otherwise, with a guard raising unless `cycles_n == cl_nom` and
`reference_dod_d == dod_nom` — which is what stops the count and depth drifting apart
again (P5.13-B, F0); `dod_nom` is now load-bearing. (b) `fixed_variable_treatment`
pinned to `make_parameter`, the value read from this machine's binary via
`ipopt --print-options` rather than from memory, applied before configuration so a case
file can still override. (c) Both warm-start merge sites (`network.py:516-517`,
`shared_energy_storage_data.py:889-890`) replaced by
`helper_functions.replace_warm_start_suffix`, which clears before updating.

**Resolved: reverted, re-registered, re-run — gate v2 PASSES.** The working tree was
reverted to the committed state; gate v2
(`data/SRP1/Results/P513D/frozen_c3_impact_gate_v2_afc863a1.json`, predecessor
`8ee17c92…`) was frozen; the change was re-applied from a preserved patch
(`9656b344…`) and the captures re-run. The reason pre-registration was not negotiable
here: this is the first production code change in the sequence, made by an agent that is
both its author and its only verifier, so the gate is not one check among several — it
is the only independent check that exists. Disclosed on the face of v2: the v1 captures
had been observed, so E1c's component list is not blind; its quantitative expectations
are derived from the algebra of the law instead.

**Solve profile — specified and armed, never denied.** `SolveProfileGuard`
(`p513_solve_profile_guard.py`) permits solves reached through
`shared_energy_storage_data._run_solver_attempt`, counts them and raises on any other
call site. All four captures: **3 solves, 3 process launches, 0 blocked**, matching the
declared count exactly in both directions. Launches equalling solves also shows no
recovery retry fired.

**Results.** E1a (`.nl` `961f81fd…` -> `119b3868…`, reversion back to `961f81fd…`),
E1b (sole differing component `constraints.energy_storage_capacity_degradation`, 6 of 30
rows), E1c (no unexpected movers; worst move `es_degradation_per_unit` at 13.356%),
E2, E3, E4 (`v2_post` vs `v2_post2` identical), E5, E6, E7, E8 — **all PASS**. The
sharpest datum: the law gives `delta = throughput/(2kE)`, so `es_degradation_per_unit`
must scale by `cl_nom/k`; the pre-registered prediction `0.8664339757` and the observed
`0.8664416996` agree to five significant figures, the residual being the ESSO
re-optimizing throughput.

**E5 at its proper weight.** `v2_revert` ran with the *new* code — `cl_eff`, the pinned
option, the warm-start assignment — with only the JSON status reverted, and reproduced
the baseline byte-identically including post-solve values, on a path that solves; its
`.nl` hash also equals the P5.13-C baseline captured under the old code. The two
determinism fixes are therefore **empirically neutral, not neutral by argument**.

**What is now structurally closed.** `effective_cycle_constant()` raises unless
`cycles_n == cl_nom` and `reference_dod_d == dod_nom`, so the count and the depth cannot
drift apart silently again — the actual defect behind the calibration episode
(P5.13-B, F0). `dod_nom` is load-bearing.

## P5.13-F — P5.12-T's zero-solve claim re-audited under armed guards (2026-09-12)

Same defect class as the P5.13-C/D error, found in an earlier stage and recorded as such
rather than as a separate incident. The guards raise as well as count
(`_blocked_solve` increments, then raises `SolverInvocationBlocked`), so a stage that
installs them **cannot** make a false zero-solve claim. Three stages installed them —
P5.12-G, P5.12-K, P5.12-Z. Three did not — P5.12-T, P5.13-C, P5.13-D.

P5.12-T's committed "Zero solves were performed" was therefore an argument, not
evidence — the same distinction on which the P5.12-R first-repair rejection turned.
`p513_f_t_reaudit.py` re-runs T's harness **unchanged** with every solve path blocked,
redirecting output to `data/SRP1/Results/P512T_REAUDIT` so the original artifacts are
untouched. Result: **0 solves, 0 process launches, 0 blocked**, T's own 41 parser checks
still passing. The enforced zero replaces the asserted one; verdict in
`data/SRP1/Results/P513D/t_reaudit_verdict.json`.

**Promoted to a convention.** `CLAUDE.md` now carries a sixth evidence rule: enforce
solve claims with armed guards, never assert them, with the permitted count declared in
advance and checked exactly — too few fails as loudly as too many, since it means the
path under test did not run.

## S1 — criterion validity for `stationarity_pf` (2026-09-12)

Diagnose only; zero solves **enforced** (`SolveProfileGuard` blocking mode: 0 solves,
0 launches); nothing changed. Frozen spec
`data/SRP1/Results/P514S1/frozen_s1_criterion_spec_v1_6c1d0a81.json`; full record in
`P5_14_S1_CRITERION_VALIDITY_REPORT.md`.

**Rediscovery, recorded (added from S2).** S1's *mechanism framing* — the dual residual
being linear in rho and the coupling with an absolute tolerance — was already documented
in `p59_b_adaptive.py`'s docstring, with a two-arm A/B attached and a top-level JSON key
`dual_residual_is_linear_in_rho`. S1 rediscovered it. S1's **results** are not in that
docstring and stand: the strong form falsified (terminal slacks 1.1035 / 1.0644 / 1.0005,
so the criterion is satisfiable), the `p ~ 0.42-0.68` partial-compensation
quantification, the terminal-circularity insight, the absolute-versus-relative tolerance
mismatch in kind, and the 8x max-versus-mean asymmetry.

**Headline: the primary hypothesis is partially confirmed and its strong form is
contradicted.** The rho/tolerance coupling is real by construction, but `stationarity_pf`
is **not unsatisfiable** — it is satisfied at termination in all nine preserved runs.
What higher `rho_pf` costs is cycles, not satisfiability.

**Harness defect, caught by the predeclared cross-check.** The first run re-derived 9
slacks against 93: the top-level `rows` of each `p510_b` artifact are three terminal
summaries, while the per-cycle data lives in `cycle_detail` (3 repeats x 6, 9, 16 =
18+27+48 = 93). Corrected, the cross-check matches **93 of 93 to 1e-12**. Pre-correction
numbers are void.

**P1/P2 as predeclared.** `g = dual_pf_mean/rho_pf` is not invariant: `g(1000)/g(300)` =
0.338 full-trajectory, 0.498 at matched cycles — the increments shrink as rho rises.
Slack ratio `slack(1000)/slack(300)`: **0.877** full (predeclared class
"compensating — mechanism ABSENT") versus **0.600** at matched cycles 1-6 (class
"partial"); exponent `p` in `slack ~ rho^(-p)` = 0.106 full, **0.422** matched, **0.675**
at cycle 1 alone. The spec predeclared that a disagreement between the two would itself
be a finding; it is, and §3 of the report gives its cause.

**Selection effect — terminal values are circular.** The run stops when residual
convergence is met, so terminal slack is pinned just above 1 by the stopping rule in
every setting (1.1035 / 1.0644 / 1.0005 for rho 300/500/1000). Cross-rho comparison must
be made at matched cycle indices; the full-trajectory median is contaminated by differing
lengths and by this pinning. Honest reading: **partial rho domination, `p ~ 0.42-0.68`.**

**The criterion improves monotonically.** Log-slack trend per cycle is positive in every
setting (+0.0293, +0.0196, +0.0106). Median slack by cycle at rho 300:
0.262, 0.486, 0.623, 0.780, 0.965, 1.104. "Binding on 93 of 93" therefore means *the
active constraint that sets the stopping time*, not a test that can never be met.

**The cost is cycles: 6 -> 9 -> 16 per repeat for rho_pf 300 -> 500 -> 1000.** Raising
rho by 3.33x nearly triples the cycles required, at ~51 local solves per cycle. This is
an S2 quantity. (Configuration `RESCALED_v1.5 ... ad0 nh1`, history-neutralised; not
interchangeable with the production trajectory of P5.12-R.)

**Q2 — mismatch in kind.** The stationarity tolerance in force is **0.01 absolute**
(re-derived as `dual_pf_mean / dual_pf_mean_ratio`), applied to a residual already scaled
by rho and normalized by `interface_rating`, while the objective tolerance is relative
(`max(1e3, 1e-3 * recourse)`). Boyd et al. §3.3.1 use tolerances with both absolute and
relative parts; `rho*(z^k - z^(k-1))` itself is textbook.

**Q3 — aggregation asymmetry.** Consensus uses max AND mean per family; stationarity uses
the mean alone. Median `max/mean` is 8.37 / 8.02 / 8.12 across settings, so a max-based
stationarity criterion would bind about **8x harder**. Recorded, not recommended.

**Scope limit (predeclared).** All 93 cycles share one binding criterion, so the dataset
holds no counterexample and supports no claim about what distinguishes binding from
non-binding criteria — the selection-on-success shape of the P5.12-W/X/Y probe. P1/P2 are
within-criterion and unaffected.

### Prior-art survey (read-only) — two results that change the picture

**Most of the named branches are merged, not abandoned.** Six are ancestors of HEAD with
zero commits ahead: `origin/admm_residual_balancing_tests`,
`origin/check_convergence_per_adn`, `admm_initialization`, `admm_prev_iter_vars`,
`primal_value_update`, `admm_loop_corrections`. `consensus_vars_sess_prev_iter` does not
exist. Only `residual_balancing_mod` (42 commits, 2024-06) and `warm_start_tests`
(5 commits, 2024-05) carry unmerged work.

**`residual_balancing_mod` is misnamed.** It *removes* the blocks labelled
"Augmented Lagrangian -- Interface power flow (residual balancing)" and replaces the
interface normalization with an average-interface-power form — the ancestor of today's
`/interface_rating`. All commit messages are "Update".

**Boyd-style residual balancing is already in production and is disabled.**
`shared_resources_planning.py:5472-5502` raises rho when
`primal_ratio > increase_balance_ratio * dual_ratio` and lowers it when
`dual_ratio > decrease_balance_ratio * primal_ratio`, applying the factor to `rho_v`,
`rho_pf`, `rho_ess` across TSO and all DSO models; `admm_parameters.py:18` defaults
`adaptive_penalty = False` and every P5.10 config is labelled `ad0`. It balances
**ratios** — residual over tolerance — which is exactly the quantity the coupling
distorts. **No evaluation artifact exists for any branch**, so what these experiments
showed is not recoverable.

**No lever is proposed. S1 stops here; S2 is not started.**

## S2 — cost model and lever designs (design stage, 2026-09-12)

Zero solves **enforced** (0 solves, 0 launches); nothing changed; no lever run. Full
record in `P5_14_S2_COST_AND_LEVER_DESIGN.md`; artifacts under
`data/SRP1/Results/P514S2/`.

**Correction that comes first: residual balancing HAS been evaluated.** `p59_b_adaptive.py`
and `data/SRP1/Results/P59/p59_b_adaptive.json` are a two-arm A/B of that lever with a
control, and its docstring already states the mechanism S1 rediscovered — dual linear in
rho, the update rule therefore pushing rho back down, "measured below, not asserted".
S1's mechanism analysis is a rediscovery. My S1 wording ("no evaluation artifact for any
branch") was true of branches and invited the wrong generalisation to stages.

**What P5.9-B measured** (same template, same start `rho_pf` = 1000, first generation):
adaptive **off** 16 cycles / 783.6 s at fixed rho; adaptive **on** 6 cycles / 295.4 s with
`rho_pf` driven to **131.687** and the objective better by 109,131 of 8.28e8. A **2.67x
cycle reduction**, and the coupling is **self-correcting, not oscillatory**. The off arm's
16 cycles independently reproduces P5.10-B at `rho_pf` 1000.

**Proximal regularization is the lever with genuinely no evaluation** — searched; only
provenance dumps and a docstring noting it was unchanged.

**Cost model.** 51 local solves per cycle (36 DSO + 12 TSO + 3 ESSO), verified from the
P5.12-R ledger (1095 = 780 + 252 + 63).

| Path | cycles | ADMM solves | `Q(x)` depends on |
|---|---|---|---|
| cold, uncapped | **>= 21** | **>= 1071** | `x` alone |
| warm from template, gen 1, rho_pf 300 | 6 | 306 | `x` and the template |
| warm from template, gen 1, rho_pf 1000 | 16 | 816 | `x` and the template |
| warm, gen 1, adaptive from 1000 | 6 | 306 | `x` and the template |
| warm **continuation**, gens 2+ | 1 | 51 | `x` and the whole campaign history |

Polish is not separately counted in the ledgers; inferred as one pass over 48 network
blocks per generation and excluded from the counts — an inference, not a measurement.

**Withdrawn figures.** The ~69-further-cycles / ~4,600-solves estimate is withdrawn: the
cold run carried `cap: 21` in its own configuration and was stopped by configuration, not
by stalling. The 15x cold-to-warm ratio is withdrawn with it. The supported ratio is
**>= 3.5x** (>= 1071 against 306), with the true cold cost unknown and bounded below only —
replacing one unsupported number with another would repeat the original error.

**Headline.** Cold at rho_pf 300, capped at 21, still descending; warm-from-template at
rho_pf 300, converged in 6. Same rho, same tolerances. **Initialization dominates rho as a
cost lever, and the P5.10 sweep varied the weaker of the two.**

**The tension for the outer layer.** The cheap path is cheap *because* it inherits a
template, and inheritance is what makes `Q(x)` depend on history rather than on `x` alone
— the same concern the Expert raised about P5.10's inherited-template comparison. A cheap
oracle whose value depends partly on where it started, or an independent oracle at several
times the cost: any outer method needs the second to mean anything, any campaign needs the
first to be affordable. Generations 2+ at 1 cycle / 51 solves are 20x cheaper than a cold
evaluation's lower bound and depend on the entire preceding campaign. A formulation
question, not a tuning one.

**Frozen designs, costed, NOT run.** `AB1 — adaptive rho on the COLD path`
(`frozen_ab1_adaptive_cold_v1_5d996737.json`): the warm path is already answered, so the
untested case is the cold one; predeclared **attractor hypothesis** — the update rule's
fixed point is a property of the problem and the dead band, not the starting value, so
from `rho_pf` 300 the adaptive arm should settle in [80, 200], falsified by settling
outside it or by oscillation; cost **>= 2142 solves for both arms**, a lower bound.
`AB2 — DSO proximal regularization` (`frozen_ab2_dso_proximal_v1_76e0051d.json`): the
never-measured asymmetry (`tso.enabled true`, `dso.enabled false`, both gamma 1.0), warm
path, prediction that DSO block movement falls with **no directional prediction for cycle
count**; cost ~612 solves for both arms. **AB2 runs first if both are authorized**, and
its control must reproduce P5.10-B's 6 cycles before any treatment number is read.

**Prior art, two unmerged branches.** `residual_balancing_mod` (42 commits) is misnamed —
it removes the blocks labelled residual balancing and introduces average-interface-power
normalization, the ancestor of `/interface_rating`; all messages are "Update".
`warm_start_tests` (5 commits) is 40 lines setting `from_warm_start = True`; messages
"Blegh", "Debug", "Correction", "Small correction", "Test." Neither has an evaluation
artifact, so their claims are unrecoverable — correctly scoped to branches this time.

**No lever run. S2 stops here.**

## AB1 — adaptive rho on the cold path (2026-09-12)

Both arms complete under the pre-registered design
`data/SRP1/Results/P514S2/frozen_ab1_adaptive_cold_v2_6e23cac0.json` (committed
`eaaa196a` before either arm ran). Guards armed: **0 blocked solves in either arm**. Full
record in `P5_14_AB1_REPORT.md`.

| | treatment (adaptive) | control (fixed rho) |
|---|---|---|
| cycles to convergence | **32** | **none — capped at 50** |
| local solves / wall clock | 1683 / 1293 s | 2601 / 2119 s |
| final `rho_pf` | 3.4683 | 300 |
| final recourse | 826,829,641 | 1,310,469,508 (descending) |

**The cold-path cost without adaptation is STILL unknown.** The control hit the cap, which
was predeclared as inconclusive; no cost may be extrapolated from it. The bound moves from
21 cycles (P5.12-R) to "more than 50".

**Attractor prediction FALSIFIED.** Predicted `rho_pf` in [80, 200], most likely 133.33;
observed **3.4683** after eleven consecutive decreases, held for 20 cycles, having passed
through 133.33 at cycle 2. No oscillation. **The falsification is scoped to the
prediction, not to the rule:** what fails is naive initialization-independence. Cold and
warm are different iterate regimes, and a rule that settles low where iterates move freely
and high near a solution is behaving as state-dependent penalty balancing should;
discarding the mechanism on this evidence would be the wrong lesson. The same rule and dead band settle at **131.687 warm** and
**3.4683 cold** — a factor of 38. The adaptive rule's operating point is a property of the
trajectory it is placed on, reinforcing the S2 finding that initialization dominates.

**The result inverts — the arms were not held to the same standard.** Dividing rho out of
`dual = rho*|dz|/base` recovers the physical iterate step: treatment at its declared
convergence `6.079e-04`; control at cycle 32 `3.326e-04`; control at cycle 50
`2.907e-04`. **The control's iterates were moving ~2x LESS than the treatment's, yet the
control fails the test and the treatment passes**, because the standards differ by
`300/3.4683 = 86.50`. On a common standard neither converged: to pass at rho 300 the step
must be under `3.33e-05`, and the treatment sits at **18.2x** that while the control sits
at **8.7x**. Measured identically, the control is CLOSER to convergence than the treatment
ever got. "Adaptation converged the cold path in 32 cycles" is therefore substantially a
criterion artifact — the S1 coupling at its extreme.

**What is real: the descent.** At matched cycle 32 the treatment's recourse is **40%
lower** (826.8 M vs 1372.7 M). The control's decrements over its last five cycles are
3.273, 3.206, 3.183, 3.152, 3.127 M, a ratio of 0.890 per ten cycles against a 484 M
remaining gap; **no extrapolation is offered**, per the frozen design. Mechanism: at rho
300 the consensus penalty dominates each local objective, holding iterates near consensus
(`primal_pf = 2.15e-04`, 21x tighter) but descending slowly; at rho 3.47 agents optimize
economically, consensus is looser but inside tolerance, and the objective falls faster.

**Two findings larger than the A/B question.**
1. *The stationarity criterion is incommensurable across rho settings* — not merely
   scale-sensitive. Any convergence comparison between an adaptive and a fixed-rho run
   compares two different tests, so "cycles to convergence" is not a valid cross-rho cost
   measure unless restated on a rho-free quantity.
2. *Consensus alone is never a valid convergence proxy.* The control sits inside
   `primal_pf <= 0.01` — 21x tighter than the treatment's converged point — while its
   objective still falls 3.1 M per cycle at cycle 50. **Corrected reading:** the 40%
   recourse spread is NOT evidence of a wide objective band at tolerance, because the
   control is mid-descent; two points on one descent trajectory differing by 40% show only
   that one has not finished. The finding *vindicates* the composite residual-AND-objective
   test, which correctly refused to declare the control converged where consensus alone
   would have.

**Caveats.** Both arms run under C3, so no objective here is comparable with pre-C3
figures. One local solve failed (treatment, cycle 11, during the rho collapse; production
held the penalty update and the run recovered) — 1 in 4284 solves across the stage. Solve
identities: 1683 vs `51x32` and 2601 vs `51x50`, each differing by exactly one
initialization block of 51; reported, not absorbed.

**Verdict.** `adaptation_helps` **not established** — the cycle comparison it rests on is
invalid. `adaptation_harms` not established. The cost question is **INCONCLUSIVE**. AB1
licenses no cost figure for an independent oracle; it licenses a sharper question: on a
rho-free standard, what does the cold path cost, and is the consensus tolerance tight
enough for `Q(x)` to mean anything?

## P5.14-G — candidate recovery and the rho-free comparison (2026-09-12)

Zero solves **enforced** (0/0/0/0). Artifact
`data/SRP1/Results/P514AB1/p514g_candidate_and_rhofree.json`.

**Candidate recovered, and the eighth-rule defect closed retroactively.** AB1 recorded
its settings exhaustively and never named its problem instance. The candidate is
nonetheless recoverable because it is deterministic: AB1 builds it with
`srp._build_positive_bootstrap_candidate`, and P5.9-B reaches **the same function**
through `BC.population -> p56a_candidates.base_vector` (`:39-43`). Re-derived here and
**cross-checked byte-for-byte against the candidate P5.12-R recorded verbatim: identical**.
So AB1, P5.9-B and P5.12-R all evaluate the same instance.

**But the proposed cold-versus-warm inversion still cannot be tested — for a different
reason.** AB1 ran under **C3** (`k = 11541.56`); P5.9-B ran **pre-C3** (`k = 10000`), so
the 0.12% gap is **confounded with the formulation change**.

**WITHDRAWN 2026-09-12 (P5.14-X22): the confound is numerically NIL on this instance.**
The warm fixed cell run under C3 reproduces P5.10-B's pre-C3 run to all sixteen digits —
recourse `827885239.5417057`, `dual_pf_mean_ratio 0.906183793111952`, `primal_pf
9.051837140987118e-05`, 6 cycles. I invoked the C3 boundary to block the cold-versus-warm
comparison and asserted the confound's existence without measuring its size, which is the
same error pattern as the salvage channel below. The definitional incomparability stands;
the numerical effect on this candidate is nil, so the comparison IS available. Likely
mechanism, recorded as a hypothesis: at `e = 0.0213` p.u. the degradation term's coupling
into available energy is far below what the network dispatch can resolve. The analysis
that follows is retained because its channel reasoning is correct and would apply on an
instance with material storage.

*Channel corrected.* The confound does NOT run mainly through salvage. From P5.9-B's own
`cost_families`, `terminal_salvage_value` is **3,452.29** against a
`net_operational_recourse` of **828,248,310** — **4.17 ppm**. The gap in question is
827,845,392 - 826,829,641 = **1,015,751**, so the entire salvage term is **0.34% of the
gap** and cannot account for it even if C3 doubled it. The dominant channel is the other
branch: SoH -> available capacity -> usable storage energy in every period -> dispatch ->
`gross_operational_cost`. **A C3 warm-path run therefore measures a dispatch-level
difference, not a salvage-level one.**
Reading that gap as two oracles agreeing on `Q` would repeat the salvage-incomparability
error across a boundary we created ourselves. The comparison requires a warm-path run
**under C3**, which is a solve and is on hold.

**The rho-free comparison, free from stored data.** Step `= dual_pf_mean / rho_pf` is the
mean interface increment with rho divided out; the single fixed physical standard is
`tol/rho_ref = 0.01/300 = 3.333e-05`.

| arm | cycles | first step | final step | multiple of the physical threshold | reached it |
|---|---|---|---|---|---|
| treatment | 32 | 0.220027 | 6.079e-04 | **18.24x** | no |
| control | 50 | 0.220027 | 2.907e-04 | **8.72x** | no |

Both arms start at an identical step, which independently confirms a shared instance and
initialization. **Neither reached the physical threshold**, and the control's step is
**2.09x smaller** at each arm's best.

**That does not mean the control is closer to a solution — and the earlier framing
conceded too much to it.** At matched cycle 32, same candidate and same formulation, the
control's recourse is 1,372.7 M against the treatment's 826.8 M: a 40% worse objective
with smaller iterate steps. **Small `|dz|` under a stiff penalty is not near-convergence;
it is being stuck.** A penalty of 300 pins the blocks near consensus, suppressing motion
while preventing the coupled system from descending, so the control's advantage on the
motion standard measures **immobility, not proximity to a solution**.

**The sharper conclusion: both motion-based standards are gameable by rho, in opposite
directions.** `rho*|dz|` inflates with rho, so raising rho pushes the test away from
satisfaction; bare `|dz|` deflates with rho, so raising rho pushes the test toward
satisfaction. Neither is a rho-independent measure of anything. **Of the three members of
the composite test, the objective criterion is the only rho-independent one — and it is
precisely the member that refused the control.**

**Remedy direction, recorded as the standing finding of S1 and S2 together:** not a
better-scaled stationarity measure, but an **optimality-based criterion rather than a
motion-based one** — the objective's own convergence, or a KKT-style residual on the
original coupled problem rather than on the consensus iteration.

## P5.14-X22 — the 2x2 under one formulation (2026-09-12)

Four cells, one candidate, one formulation (C3), guards armed, **0 blocked solves**.
Frozen spec `data/SRP1/Results/P514X22/frozen_2x2_spec_v1_55b10d16.json`. Full record in
`P5_14_X22_2x2_REPORT.md`.

| cell | cycles | solves | final `rho_pf` | terminal `gross_operational_cost` |
|---|---|---|---|---|
| cold_fixed | 50 **(capped)** | 2601 | 300 | 1,310,473,449 |
| cold_adaptive | 32 | 1683 | 3.4683 | 826,833,558 |
| warm_fixed | 6 | 357 | 300 | 827,888,692 |
| warm_adaptive | **4** | **255** | 88.8889 | 827,738,664 |

Cycle 1 is identical within each initialization, confirming one shared instance and two
shared starting points.

**C3 is numerically inert here — my confound caution is withdrawn.** `warm_fixed` under C3
reproduces P5.10-B's pre-C3 run to sixteen digits. Definitional incomparability stands;
the numerical effect on this candidate is nil. By-products: byte-level determinism of the
warm path across the P5.13-C/D production edits, and a retroactive demonstration that
those edits were neutral on a *solving* trajectory rather than only on a model build.

**Adaptation's advantage is COLD-SPECIFIC** (predeclared outcome). Cold pair at matched
cycle 32: **−39.8%**. Warm pair at terminal: **−0.018%**. Adaptation mostly compensates for
a bad cold start rather than improving the method. Warm it is still a **29% cost saving**
(4 cycles / 255 solves against 6 / 357) — a cost gain, not an objective gain.

**~~The two oracles AGREE on `Q` to ~0.1%~~ — RETRACTED 2026-09-12 by Track C1.** The
agreement was already indeterminate under the ninth rule (offset 0.99x its own error bar),
and C1 showed it wrong in substance: one decade tighter, the offset **grew 9.39x** to
8,507,695 — 1.03% of the objective, at 65.6x its error bar. The two initializations do
**not** share a fixed point. What survives is the *direction*: the templated oracle is
systematically high. Original text retained below, struck, because the retraction is the
finding.
cold_adaptive vs warm_fixed **−0.1275%**; vs warm_adaptive **−0.1094%**; warm pair
−0.0181%; our warm_adaptive vs P5.9-B's pre-C3 warm adaptive **+0.0133%**. A cold
independent start and a warm templated start agree on `Q` to about one part in a thousand:
**evidence that `Q(x)` is well defined rather than an artifact of where the iteration
began** — the strongest result of the stabilization effort so far. Qualification: the sign
is **systematic**, the templated oracle sitting **~0.11% high**, which matters for any
ranking whose best-to-second gap is narrower than that. `cold_fixed` is capped and
contributes only a bound.

**The cost of independence, measured:** cold_adaptive / warm_adaptive = **6.60x**;
/ warm_fixed = **4.71x**. This replaces the withdrawn 15x and the earlier ">= 3.5x" bound.
The outer-layer tension is now quantified: **0.11% of objective bias against a factor of
6.6 in cost.**

**The attractor prediction holds in the regime where it was derived.** `warm_adaptive`
settled at **88.889**, inside the predicted [80, 200]; the cold cell settled at 3.4683,
outside it. The rule's operating point is a property of the iterate regime. Even warm it is
start-dependent along the multiplicative grid (P5.9-B 131.687 from 1000; this cell 88.889
from 300), so the band captures a range, not a unique fixed point.

**Solve identity:** every cell is `51 x cycles + 51` — one initialization block —
1683/2601/357/255 for 32/50/6/4 cycles. The harness's "UNEXPLAINED" label was wrong; its
explanation list allowed only a 48-block polish or zero, and no polish ran in the warm
cells.

## P5.14-Y2 — the resolution limit on `Q`, established before the experiment (2026-09-12)

Frozen design `data/SRP1/Results/P514Y2/frozen_y2_offset_design_v1_d6f7dd9e.json`.
**NOT authorized to run; held.** Zero solves in producing it.

**The rho-endpoints are exactly quantized on the decrease grid**, which makes the traces
self-checking: `300/1.5^11 = 3.4683059832`, `300/1.5^3 = 88.8888888889`,
`1000/1.5^5 = 131.6872427984`, all reproducing the observed values exactly. The rule only
ever decreased, never increased, never landed off-grid. **Attractor reframed:** warm from
300 lands at 88.9 and warm from 1000 at 131.7 — *adjacent grid points*. Two starts a
factor of 3.3 apart converging to neighbours is **convergence to a region**, not start
dependence; the genuine dependence is on **regime** (warm ~10^2, cold ~3.5), which is
state-dependent balancing behaving as it should. Corresponding caution: the predicted band
[80, 200] spans ~2.4x, about two grid intervals, so the warm cell landing inside it is
**weak confirmation**, not a clean hit.

**The offset against the signal, on the correct footing.** `PLANNING_SIGNAL = 33031.0`
(`p510_e_criteria.py:53`) is the *pre-stabilized* best-to-second gap; the stabilized
oracle's gap is **32.87** with cross-depth uncertainty **22.09** (ratio 0.67 — the
Expert's "not strong separation"). These are the same quantity on two oracles and must not
be divided by one another. The measured cold-versus-warm offset is **1,055,598** absolute
(vs `warm_fixed`), i.e. ~32,000x the stabilized signal. That is not fatal, because rankings
are differences *within* one path and a constant offset cancels exactly — but it sets the
bar: the offset must reproduce across candidates to within **~33 absolute units out of
1.06e6, i.e. 0.0031%**.

**Pre-registered resolution analysis — the bar cannot be met, and this is established
before running anything.** Arithmetic is not the limit: the warm path reproduced a pre-C3
run to 16 digits, so the pipeline is bitwise deterministic. The limit is **where each run
stops**. `Q` is read at the point where the per-cycle objective change falls under
tolerance, so it is known only to about the size of the last step taken:

| cell | terminal per-cycle objective change |
|---|---|
| cold_adaptive | 706,604 |
| warm_fixed | 59,459 |
| warm_adaptive | 212,489 |

The error bar on the cold/warm_fixed offset is their sum, **766,062** — so the measured
offset of 1,055,598 is only **1.4x its own error bar**, and the ranking bar of 33 units is
**~23,000x finer than that**. The experiment as posed is **INDETERMINATE by construction**.

**The finding this yields, which is larger than either outcome.** The objective criterion —
the only rho-independent member of the composite test — is itself far too loose to resolve
the ranking it supports. `objective_tolerance = 827,945` against a stabilized signal of
**32.87**: a factor of **25,188**, and still **25x** the *pre-stabilized* 33,031. `Q` is
defined only to within ~±8.3e5 by its own stopping rule while the quantity to be resolved
is 32.87. The empirical cross-depth uncertainty of 22.09 is four orders of magnitude
smaller than that bound, which means **the templated ranking works — where it works — by
the determinism of identical code paths cancelling the slack, not because the criterion
resolves anything.** That is fragile: any perturbation of a trajectory (a different
candidate, a solver retry, a code edit) can move a stopping point by up to a tolerance
width.

**Cost of a resolution-adequate oracle — indicative only.** Continuing the cold cell at its
observed late decrement ratio 0.8216 until the per-cycle change falls under 33 units
implies ~50.7 more cycles, ~2,588 more solves per evaluation. **This is an extrapolation of
a decrement ratio — the exact operation that has gone wrong three times in this project —
and the ratio is not stable (0.87 -> 0.98 in the R trajectory, 0.8216 here). Order of
magnitude only; it must never be quoted as a cost.** The resulting total (~4,271) lands
near the withdrawn 4,600 by a different route; that is a coincidence and does not vindicate
the withdrawn figure, whose derivation was wrong.

**Recommended redesign, frozen with the above.** Measure the two initializations at a
**common, criterion-free reference** — an identical fixed cycle budget — reporting the
offset together with each cell's per-cycle change at that budget as an explicit error bar;
then ask only whether the offset reproduces *within the stated error bar*, which is the
strongest claim the current stopping rule supports. What would actually settle it is
tightening the objective tolerance until the terminal per-cycle change falls below the
signal — a parameter change, not authorized, with the indicative cost above.

## Track A — ADMM stabilization track CLOSED (2026-09-12)

Documentation only; no solves. Full record in `P5_14_TRACK_A_STABILIZATION_CLOSURE.md`.
S1, S2, AB1 and the 2x2 stand as recorded, with the rediscovery, the corrections and the
withdrawn claims intact.

**The resolution finding is the track's principal result.** `objective_tolerance = 827,945`
against a stabilized best-to-second gap of **32.87** — a factor of **25,188**, and still
25x the pre-stabilized `PLANNING_SIGNAL = 33,031`. The observed cross-depth uncertainty of
**22.09** sits four orders of magnitude below that bound, and the **path-identity
mechanism** explains why: identical code paths visiting identical iterates stop at identical
points, so the stopping slack cancels exactly in a difference. P5.10's ranking stability is
a consequence of **determinism, not resolution**. It generalises to reruns of the same
candidates on the same code and data; it does **not** generalise to any comparison where the
paths differ — a different candidate, a solver retry, a code edit, a different
initialization — each of which can move a stopping point by up to a tolerance width, i.e.
~25,000 times the signal.

**Consequence: `Q`'s resolution, not its convexity, is the immediate disqualifier for
cut-based methods.** A cut or a finite-difference slope built on differences of `Q` inherits
an uncertainty four orders of magnitude larger than the effects being ranked. Convexity
remains unestablished and is now a second-order question.

**The codebase already encodes the gate it fails.** `benders_parameters.py:17` sets
`minimum_signal_to_noise_ratio = 10.0` and the case file sets
`benders.finite_difference.enabled = false`. The current ratio is
`32.87 / 827,945 = 3.97e-05`, failing that gate by **2.5e5** — more than five orders of
magnitude. Someone built the right check and switched it off: independent corroboration
from the opposite direction.

**Ninth rule promoted** to `CLAUDE.md`: report a difference with its resolution; a
difference smaller than the error implied by where each computation stopped is
indeterminate, not a result.

**Closure reason.** `Q` well defined across structurally different initializations (~0.1%,
systematic); ADMM converges (32 cycles cold adaptive, 4-6 warm); local solves fail once in
1,095 (once in 4,284 across AB1/X22); a warm evaluation costs 255 solves; **no remaining
lever with an identified decision-relevant payoff**. The cycle-21 mechanism stays open and
unpursued; the breadth line stays closed at n = 3.

## Track B — configuration reconciliation: OVERRIDE, not drift (2026-09-12)

Zero solves, enforced. Full record in `P5_14_TRACK_B_CONFIG_RECONCILIATION.md`; evidence
`data/SRP1/Results/P514B/b_config_reconciliation.json`. **Nothing changed; the decision is
the author's.**

**Per-stage governing configuration** (tolerances *derived* from each stage's own numbers,
not read from the case file): every preserved stage ran at `rho_v = 1.5`, `rho_pf` 300 or
1000, stationarity and consensus tolerances 0.01. `adaptive_penalty` was False everywhere
except P5.9-B's treatment arm and the 2x2's adaptive cells. `num_max_iters` is not
serialized by the older stages — a gap against the eighth rule.

**Mechanism: programmatic override.** `data/SRP1/SRP1_params.json` has carried
`rho.{v,pf,ess} = 1.0` and `adaptive_penalty = true` **continuously since 2025-12-15
(`784346d7`)**, which pre-dates every preserved stage. The file was never edited to diverge;
the harnesses override rho and the adaptive flag in memory via
`p59_rho.apply_rho_to_params` / `set_adaptive_penalty`. What *did* move in the file are the
**tolerances** (stationarity `0.05 -> 0.001 -> 0.01`; objective rel
`0.0005 -> 0.005 -> 0.001 -> 0.01 -> 0.001`) and `num_max_iters 50 -> 25` on 2026-09-04 —
and tolerances are **not** overridden, so each stage ran with whatever the file held.

**A plain production run today** would use `rho = 1.0` everywhere, `adaptive_penalty = True`,
`num_max_iters = 25`, stationarity 0.01, objective (1000, 1e-3), TSO proximal on, DSO
proximal off, C3 active — **matching no preserved stage**.

**Decision put to the author with both readings.** (1) *Deliberate baseline*: P5.9-B and the
2x2 both show adaptation works and finds its own level, so `adaptive = True` with
`rho.pf = 1.0` is rational; if so, record it as the new intended baseline and state the
corollary that **every preserved result was produced under a now-superseded configuration**.
One measurable caveat: the rule has only ever been started *above* its settling range, so
only the decrease branch has ever fired; starting at 1.0 would require the increase branch,
which no preserved run exercises. (2) *Drift*: the tolerance edits are labelled "Debug." and
moved five times in six weeks, so the file should be restored to the evidence-base values.

## Tracks C, D, E — status (2026-09-12)

**C0 frozen** (`data/SRP1/Results/P514C/frozen_c0_tolerance_sweep_v1_43da5cb6.json`): one
decade, `admm.tol.objective.rel` 1e-3 -> 1e-4, overridden **in the harness**, the case file
untouched. Configuration inherited from the 2x2's adaptive cells so the tolerance is the
only varied setting. Predeclared classes: proportional shrink / plateau / indeterminate.
Recorded in the spec: **at 1e-3 the cold/warm offset (905,573) was already smaller than its
own error bar (919,092)**, so the baseline comparison is itself indeterminate — C1 tests
whether one decade changes that. C1 running; **C2 stops for the author's decision** on a
second decade, whose cost is an order of magnitude uncertain (12 cycles at r = 0.82, 114 at
r = 0.98) and must be measured.

**D proposed, nothing run** (`P5_14_TRACK_D_CANDIDATE_RESIZE_PROPOSAL.md`): the conditioning
problem is in the experiment, not the solver. P5.10's 0.0000040% perturbation gives
`signal/resolution = 0.00004x`; the paper's claimed effects give **20.1x** (2.01% storage)
and **182.5x** (18.25% coordination + storage), both clearing the built-in SNR gate of 10.
The caveat to be measured rather than assumed: at real capacities neither the 255-solve warm
cost nor the 1-in-1,095 reliability transfers, because both were measured where the storage
is nearly inert. Penalty classification becomes live again in this track.

**E delivered** (`P5_14_TRACK_E_MANUSCRIPT_RECONCILIATION.md`): claim / status / correction
for the Expert's five items plus five established since — the 0.50% gap is
`benders.tol_rel = 0.005`, a configured stopping tolerance reported as an achieved gap; the
70% SoH is a live `paper_revisions` divergence; the 60/80% sensitivities were not runnable
while the constants were hard-coded; ranking stability rests on path-identity cancellation;
and branch governance must be stated. Penalty classification is recorded as a blocking
dependency the paper should make visible.

## Track C1 — one tolerance decade: the oracles DIVERGE (2026-09-12)

Frozen spec `43da5cb6` before either cell ran; case file never edited; guards clean;
identities exact (`3519 = 51x68+51`, `306 = 51x5+51`). Full record in
`P5_14_TRACK_C1_TOLERANCE_DECADE.md`.

| | cold 1e-3 | cold 1e-4 | warm 1e-3 | warm 1e-4 |
|---|---|---|---|---|
| cycles | 32 | **68** | 4 | **5** |
| recourse | 826,829,641 | **819,145,341** | 827,735,215 | **827,653,037** |
| terminal step | 706,604 | 47,545 | 212,489 | 82,178 |
| solves | 1683 | **3519** | 255 | 306 |

**The offset GREW.** At 1e-3: 905,573 against an error bar of 919,092 — ratio 0.99,
**indeterminate**. At 1e-4: **8,507,695** against 129,723 — ratio **65.6, determinate**,
and 1.0279% of the objective. Growth of **9.39x** where proportional shrink would have
given ~0.1x. **This is outside the three predeclared classes** (shrink / plateau /
indeterminate) and is recorded as such rather than mapped onto "plateau", though its
implication is the plateau implication amplified tenfold.

**The two initializations do NOT share a fixed point.** Tightening let the cold path
descend a further 7,684,300 over 36 extra cycles while the warm path moved 82,178 in one.
Tightening pulls them apart rather than together. **The 2x2 headline is retracted**: the
0.1% agreement was a coincidence of two early stopping points.

**Mechanism, and a correction to Track A.** The stopping rule tests the **rate** of
objective change, not proximity to an optimum, so a path whose increments decay quickly
stops early regardless of its objective. Warm stopped at **99.3%** of its threshold, cold
at 58%. Track A recorded the objective criterion as "the only rho-independent member" —
true, but it is **not an optimality test**: **all three members of the composite test are
motion-based**, and no reweighting of them yields an optimality criterion. The remedy
direction — a KKT residual on the original coupled problem — is reinforced and broadened.

**Cost, measured.** Independence is now **11.5x** (3519/306), up from 6.60x at 1e-3: the
cost of independence grows as the tolerance tightens, because the cold path is the one
still descending. **The ratio-based decade estimate was falsified by measurement**: it
predicted ~11.7 extra cold cycles, the measured cost was **36** — a 3.1x underestimate.
The decrement-ratio extrapolation is now **retired as a method**, not merely cautioned
against. The realised ratio drifted 0.8216 -> 0.8451, so a second decade costs **at least**
36 cycles (>= 1,836 solves, >= 35 minutes) and plausibly double; no single figure is
offered.

**C2 stops here for the author's decision on a second decade.**

## C2 decision and its consequence — the templated oracle is disqualified (2026-09-12)

**C2: no second decade.** The sweep answered its question. The two initializations do not
share a reachable fixed point — the cold path has never stopped descending at any tolerance
tried, and the warm path is parked in its template's basin. A third point would confirm what
the mechanism already predicts, at >= 36 cycles and plausibly double: a confirmation, not a
decision.

**Decrement-ratio extrapolation is RETIRED as a method**, not merely cautioned against. It
was tested directly and failed: predicted ~11.7 extra cold cycles for the decade, measured
**36** — a 3.1x underestimate. It must not be used to produce a cost, a horizon or a
remaining-gap estimate.

**The consequence that decides the project.** The templated oracle is **1.03% high**, and it
is high because it stops early in a worse basin; in a minimization that systematically
**overstates cost**. The paper's claimed incremental storage benefit is **2.01%**, so the
bias is roughly **half the effect the paper exists to report**.

**The hopeful reading is closed off by the mechanism, not merely unmeasured.** A constant
bias would cancel in an incremental comparison, and we could never measure constancy across
candidates. But the bias *is* the amount by which a warm path fails to leave its template's
basin, and in a campaign **each candidate inherits its own template** — so there is no reason
to expect constancy, and the mechanism actively predicts candidate-dependence: a candidate
whose template sits in a better basin shows a smaller bias. **The templated oracle cannot
support the paper's headline comparison at any price.**

**The cold path is the oracle** — the only one not reporting an artifact of its own
initialization. Independence costs 11.5x and rises with the tolerance, but at a 1% bias
against a 2% effect the cheap path is unusable regardless of price.

**And for the comparison the paper actually makes, today's resolution suffices.** Error bars
per the ninth rule (a pair-difference carries both cells' terminal steps):

| tolerance | solves/evaluation | error bar on a pair | 2.01% storage | 18.25% combined | four plans |
|---|---|---|---|---|---|
| `rel` 1e-3 | 1,683 | 1,413,207 | **11.8x** | 106.9x | **6,732** |
| `rel` 1e-4 | 3,519 | 95,090 | **175.0x** | 1589.1x | **14,076** |

Both clear the built-in gate of 10, but 1e-3 is **marginal** for the headline effect at
11.8x. **Track D is now the priority, executed on the cold path**, and no further tolerance
work is needed for it. The author decides the candidate set; the Planner recommends 1e-4.

**Tenth rule adopted** in `CLAUDE.md`: report the terminal-step-to-threshold ratio for every
cell of every evaluation. A settled run stops well inside its threshold; one terminating at
~99% of it is being *stopped*, not converging. C1's contrast — warm **99.3%**, cold
**58.0%** — diagnosed the mechanism, and reporting it earlier would have flagged the 2x2's
warm cells before their agreement was read as a result.

**Objective convention recorded** in `CLAUDE.md`: `gross_operational_cost` and
`net_operational_recourse` differ by exactly the terminal salvage credit — 3,448.87 on the
X22 warm-adaptive cell, which is why that table reads 827,738,663 (gross) where C1's reads
827,735,215 (net). State the convention on every table.

## Track D1 — three-cell campaign SUSPENDED (2026-09-12)

Frozen spec `2c8c7bef` before any cell ran; overrides `budget = 5.0e6` (derived so
`max_capacity` binds before the budget at every admissible ratio, and verified inert on the
cold path) and `rel = 1e-4`. Full record in `P5_14_TRACK_D1_CAMPAIGN_REPORT.md`.
**Neither comparison is available.**

| cell | outcome | solves |
|---|---|---|
| 1 — uncoordinated, no storage | **BLOCKED — production defect** | 0 |
| 2 — coordinated, no storage | succeeded | 3,621 |
| 3 — coordinated, 1.00 MVA / 4.00 MWh | **FAILED at initialization** | 51 |

**Cell 1: the uncoordinated mode is broken in the current tree.**
`_add_dso_scenario_deviation_penalty` unconditionally references `expected_shared_ess_p`
(`shared_resources_planning.py:2826`); the ADMM path builds it first (`:3013-3018`), the
uncoordinated path does not (`:5850-5859`). Introduced by `99a59fec` (2026-08-18). The
`no_coordination` workbook predates it, so the mode worked once and has been broken ~3
weeks. **Not fixed** — whether an uncoordinated DSO should carry a shared-ESS deviation
penalty is a formulation question, not a typo.

**Cell 2 succeeded:** 70 cycles, 3,621 solves (`51x70+51`), recourse **820,746,762.46**
(= gross, no salvage), rule-ten ratio **0.8414**, zero local-solve failures, final
`rho_pf` 5.2025, **no non-vanishing slacks** among 17 families at 1e-6. *Spec premise
corrected*: cell 2 does build all 17 slack families at zero capacity — trivially zero
rather than absent.

**Cell 3 failed at initialization.** The candidate passed first-stage feasibility; the ESSO
subproblem at **node 7** converged to a locally infeasible point (165 iterations,
constraint violation **6.84e-05**, dual infeasibility 1.0e3), and ADMM never started.
**Nodes 5 and 9 solved optimally at identical capacity**, so it is node-specific.
**Whether this is genuine infeasibility or numerical failure is NOT established** — the
violation is of the same order as `ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`, which hints at
numerical difficulty, but two of three solving settles nothing. The distinction decides
whether it is a finding about the plan or about the tool.

**The transfer caveat was measured and DID NOT transfer — the campaign's principal
result.** Local-solve reliability at negligible capacity was 1 in 1,095 (1 in 4,284 across
AB1/X22); at 1.00 MVA / 4.00 MWh it was **1 failure in 3 ESSO solves**, aborting the run.
Evaluation cost at real capacity is **not measurable** because the run never started.

**The ESSO recovery path is inapplicable by design.**
`_is_recoverable_shared_ess_failure` (`shared_energy_storage_data.py:934-940`) fires only
on `internalSolverError`; node 7 terminated as `infeasible`, so no retry occurred —
confirmed by the solve count `51 = 36+12+3`. The recovery covers solver crashes, and the
failure mode that occurs at real capacity is the one it does not cover.

**A structural confound that would affect cell 3 even if it ran.**
`max(|rating|, shared_ess_normalization_floor_mva = 0.10)` means cell 2 (zero) and the C1
cold cell (0.0106) both normalize the shared-ESS consensus terms by **0.10**, while cell 3
would use **1.00** — a factor-of-10 difference in the ESS consensus weighting, on top of
the capacity difference. Any cell3-minus-cell2 difference would confound the two.

**An unexplained anomaly in the one available pair.** Cell 2 (no storage) against the C1
cold cell (bootstrap 0.0106 MVA), identical in every other respect: difference
**1,601,421** against an error bar of 116,608 — **13.7x, determinate**. But the bootstrap
plan is ~10.6 kW / 21 kWh per node costing ~7,046 per node, so 1.6M of operating saving
**cannot be storage value**. Either zero capacity differs structurally beyond the
normalization floor, or the two trajectories settled in different basins (the
path-dependence C1 established). **Recorded as an anomaly, not a measurement** — and a
warning that "no storage" may not be a clean control.

**To complete the campaign:** a decision on the uncoordinated path's penalty; a diagnosis
of node 7; a resolution of the normalization-floor confound; an explanation of the anomaly.
None authorized.

## GOVERNING FACT — the oracle cannot evaluate a plan large enough to matter (2026-09-13)

Recorded as the governing fact of the numerical programme, not as one blocker among four.
**Every stabilization finding — `Q` well defined, ADMM converging, one local failure in
1,095 — was established in a regime where the storage is economically negligible. At the
first capacity where it is not, the evaluation aborts before ADMM starts.**

One sub-result separates cleanly: **the cost model transferred and the reliability model
did not.** Cell 2 took 3,621 solves against 3,519 predicted, **2.9% over**; the 1-in-1,095
local-solve failure rate became **1-in-3**.

## P5.14-L — capacity ladder: C* = 0.96875 MVA / 3.875 MWh per node (2026-09-13)

Frozen spec `c1ea6393` with both go/no-go branches and the prime suspect predeclared. Seven
rungs, initialization stage only, ADMM never started, 358 solves, ~5 minutes. Full record in
`P5_14_L_CAPACITY_LADDER_REPORT.md`.

| `s` MVA/node | 0.25 | 0.50 | 0.75 | 0.875 | 0.9375 | **0.96875** | 1.00 |
|---|---|---|---|---|---|---|---|
| outcome | PASS | PASS | PASS | PASS | PASS | **PASS** | **FAIL (node 7)** |

**A clean, monotone threshold**, bracketed to within **3.2%**. By the predeclared rule that
makes it **a finding about the plan**: `case33_2` cannot host 4.00 MWh at ratio 4 but can
host 3.875 MWh. **Largest evaluable capacity: `C* = 0.96875 MVA / 3.875 MWh` per node**
(2.91 MVA / 11.6 MWh system-wide).

**The predeclared prime suspect is FALSIFIED.** The unslacked aggregate complementarity row
(`shared_energy_storage_data.py:618-620`) is satisfied to **1e-08**. The violation sits on
two pure definitional identities — `es_s_rated_per_unit == es_s_investment` (`:442`) and its
sum (`:454`) — at **6.845e-05**, matching IPOPT's reported figure exactly. Those are linear
equalities in otherwise free variables and cannot be infeasible alone, so that is where
restoration left residual, not a physical conflict. The sharp threshold and the restoration
artifact are **compatible**: a genuine feasibility boundary, with the residual reported on
definitional rows. What is ruled out is the specific hypothesis that the unslacked
aggregate row is the binding obstruction.

**Limit: the ladder tested INITIALIZATION ONLY.** `C*` is the largest capacity whose ESSO
subproblems solve at initialization, **not** the largest that completes a full ADMM
evaluation — and P5.12 established that a trajectory can initialize cleanly and fail later.
Scope: ratio 4, single cohort 2025, `rel` 1e-4, `rho_pf` 300 adaptive, C3.

**Go/no-go status.** The bounded error bar at `rel` 1e-4 is ~**164,000**, i.e. **0.02%** of
the objective, against a claimed 2.01% effect of ~16.6e6 — **~100x the bar**. So resolution
is *not* the binding constraint at `C*`; an effect would have to be under 1% of the claim to
be unresolvable. Deciding GO/NO-GO needs **one cold evaluation at `C*` paired against cell
2** (~3,600 solves, ~35 min). Not run.

## Refinements and design constraints recorded alongside the ladder (2026-09-13)

**Rule nine refined** in `CLAUDE.md`: the bar bounds *stopping slack*, not *path
divergence*. Two runs still descending toward different limits can differ by far more than
the sum of their terminal steps — C1 showed exactly that when tightening grew the offset
9.4x. "Determinate at N x the error bar" licenses only "not explained by stopping slack",
never "a real difference in the limit". The bar is **local**, valid only when both runs have
settled, and **rule ten is what tells you whether they have**. Applied to the D1 anomaly:
cell 2's rule-ten ratio is **0.8414** against the C1 cold cell's **0.5800**, so a third and
cheapest explanation joins the two already recorded — **both runs are simply at different
points on their descents**, which alone can account for 1.6M without invoking structure or
basins.

**Cell 1 is less open than it looked.** `_add_dso_scenario_deviation_penalty` assembles
**three** components (`shared_resources_planning.py:2829-2833`): `scenario_deviation_voltage`,
`scenario_deviation_interface_power` and `scenario_deviation_shared_ess`. An uncoordinated
DSO has no shared ESS by construction, so the shared-ESS component should simply be absent
while the other two remain — a guarded component, not a formulation question. It fits the
gate pattern trivially, since behaviour preservation is free when the mode currently raises.
**Not applied; awaiting authorization.** Recorded separately: *a production mode nobody runs
is a mode nobody notices breaking* — three weeks of silent breakage.

**The normalization confound is a design constraint on the comparison, not an observation.**
`max(|rating|, 0.10)` gives cell 2 (zero) and the bootstrap cell the same 0.10 normalization
while any real plan uses its own rating — a tenfold difference on the ESS consensus channel
at 1.00 MVA, layered on the capacity difference. **Any future cell-3-versus-cell-2
comparison must either hold the normalization fixed or quantify its contribution
separately.**

## P5.14-M — C* evaluation BLOCKED by a third production defect (2026-09-13)

Frozen spec `5a6210aa` before the run. GO was authorized and **cannot execute**. Full
record in `P5_14_M_CSTAR_BLOCKED.md`.

Initialization succeeded at `C* = 0.96875 MVA / 3.875 MWh` as the ladder predicted, ADMM
started, and cycle 1 raised `KeyError: 'sch'` in
`_print_worst_primal_residual_diagnostics` (`shared_resources_planning.py:5186`) — **a
diagnostic print statement, not the solver.**

**Producer/consumer mismatch.** The `charge_discharge` dict is built with `pch`/`pdch`
(`:5014`, `:5023`, `:5035`); the printer reads `sch`/`sdch` (`:5186-5187`), the retired
apparent-power names. The printer dates from `1777457d` (2026-09-02); the producer was
converted by **`58f4911b`, "P5.4-C: ESSO active-energy conversion" (2026-09-06)** — which
converted the producer and not the consumer. Broken for a week.

**Why it never fired before:** the diagnostic is guarded by
`primal['ess'] > tol['consensus']['ess']` = 0.1 (`:5152`). At negligible capacity the ESS
primal residual never reaches 0.1; at material capacity it is exceeded on cycle 1. The
defect was not *caused* by material capacity — it was finally *reached* by it.

**The pattern is now the substantive finding.** Three blockers, all in paths that only
material capacity or an unused mode reaches: the uncoordinated mode (`99a59fec`,
2026-08-18), node 7's genuine feasibility boundary (not a defect), and this printer
(`58f4911b`, 2026-09-06). **The tool has been exercised only in a regime where its
storage-specific code paths are inert** — so the code that handles a materially loaded
shared ESS has never been executed, and its defects have accumulated unobserved. That is
the governing fact restated with a mechanism.

**Severity differs sharply:** the printer is two dictionary keys and changes no number;
cell 1 is one guarded penalty component with a formulation reading behind it; node 7 is not
a defect but a real capacity limit — the only one of the three that is information about the
system rather than about the code.

**Status:** `C*` not obtained, GO unexecuted, no artifact written (`d1_cell3.json` still
holds the 1.00 MVA run). The three-point capacity series remains at two points. **No
production change made** — and fixing a third defect in a path no test exercises is itself a
decision rather than a patch.

## P5.14-M — C* evaluation COMPLETED; the anomaly resolves as path divergence (2026-09-13)

The success branch. 67 cycles, converged inside the cap, **zero local-solve failures**, no
non-vanishing slacks, identity exact (`3468 = 51x67+51`). First storage-benefit number this
tool has produced at material capacity. Full record in `P5_14_M_CSTAR_RESULT.md`.

**Three-point series** (all cold, `rel` 1e-4, C3, `rho_pf` 300 adaptive):

| point | `s` MVA/node | recourse | rule ten | solves |
|---|---|---|---|---|
| cell 2 | 0 | 820,746,762.5 | 0.8414 | 3,621 |
| C1 cold | 0.0106 | 819,145,341.2 | 0.5800 | 3,519 |
| **C\*** | **0.96875** | **816,121,464.2** | **0.9329** | 3,468 |

Monotone decreasing. Pairs: `0 -> 0.0106` **1,601,421** (13.7x est bar); `0.0106 -> C*`
**3,023,877** (24.4x); `0 -> C*` **4,625,298** (31.9x est, 28.2x bounded, **0.5635%** of
base) against the paper's claimed 2.01%.

**The D1 anomaly is RESOLVED as path divergence.** Per MVA added: step 1 delivers
**50,193,933/MVA**, step 2 **1,052,023/MVA** — the first step is **47.7x** more valuable per
MVA, which is economically incoherent for storage value (10.6 kW/node cannot deliver half
what 958 kW/node delivers). **The 1.6M first step is not storage value.** Consequence: the
`0 -> C*` figure is an **upper bound** on the storage effect, not a measurement, since it
inherits the contaminated step. The cleaner quantity is the second step alone, **3,023,877
(0.37%)**, between two cells that both contain a shared ESS.

**Limits as predeclared.** The `0 -> C*` pair is the weakest: **both** endpoints stopped
near their bounds (rule ten 0.8414 and 0.9329), and under the refined ninth rule the bar
bounds stopping slack rather than path divergence and is valid only where both runs have
settled. So 31.9x licenses "not explained by stopping slack" and **not** "a real difference
in the limit". The normalization confound is live in that same pair (0.10 against 0.96875,
**9.7x** on the ESS consensus channel).

**Two corrections to earlier framing.** (a) **Reliability DID transfer** — zero failures in
3,468 solves at C\*. The 1-in-3 rate was the *feasibility boundary* at 1.00 MVA, not general
degradation at material capacity; my earlier statement is right about 1.00 and wrong as a
general claim. (b) **Cost transferred slightly downward** — 3,468 solves against 3,519 at
bootstrap, 1.5% fewer.

**What C\* still does not exercise.** `gross == recourse` exactly, so **salvage is zero**:
investing in 2025 only with `t_cal = 15` over a 15-year horizon leaves the single cohort
fully depreciated at the terminal. The terminal-value component of the Track E inertness
table is therefore **still inert — by plan shape, not magnitude**. Updated status at C\*:
consensus channel **live**, ESSO constraint set **live**, terminal value **inert**, ageing
model **UNKNOWN**.

**The follow-up this makes necessary (not authorized):** a C3-style perturbation at C\*
capacity — change the degradation constant and see whether the recourse moves, as it did
not at bootstrap capacity. Until then the 0.37% effect **cannot be attributed to a model in
which storage degrades.**

## P5.14-M corrections and the EFC gap (2026-09-13)

**The per-MVA ratio is WITHDRAWN.** The 47.7x mixed conventions: step 1 used C1's per-node
*cumulative* capacity (0.0319047) while step 2 used a system-wide denominator with C1 at its
*annual* investment. Computed consistently — the system-wide factor of three cancels, so
only C1's convention matters — the ratio is **15.6x** (per-node cumulative) or **23.6x**
(horizon-average). **The ratio is not load-bearing**: C1's capacity is a rising trajectory
while C\*'s is constant, so per-MVA normalization is ambiguous and the choice swings the
answer by 50%. **The absolute statement carries the conclusion and needs no convention:
1.6M of operating saving from 32 kW per node cumulative is not credible on its face.**

**C\* is the limiting endpoint of EVERY comparison**, not only the weakest. Its rule-ten
ratio is **0.9329**. All three pairs contain C\*, so the **0.37% second step (endpoints
0.5800 and 0.9329) is limited by C\*'s near-bound stop** exactly as the cell-2 pair is, and
must be reported with that qualification.

**The zero salvage was a deliberate trade, recorded as such.** The 2025-only single-cohort
shape was chosen to sidestep cohort-SOC realizability; its cost is a structurally zero
terminal value. **The design traded one inert component for another.** A later cohort would
carry remaining life and nonzero salvage, at the price of reintroducing multi-cohort
allocation.

**The realized EFC/day at C\* is NOT recoverable — a reporting miss.** The campaign spec
required it with its margin to the 1.4612 threshold; the harness captured per-cycle ADMM
diagnostics and slack maxima but not `es_avg_ch_dch_per_unit`, `es_degradation_per_unit` or
`es_soh_per_unit_cumul`, and the ESSO models were not serialized. **It cannot be obtained
with zero solves.** Options: (a) instrumented re-run of the C\* baseline, ~3,468 solves and
~35 min, giving the exact converged EFC and doubling as a determinism check on
816,121,464 — **recommended**; (b) initialization-only probe, 51 solves, indicative only
since initialization cycling is not converged cycling; (c) instrument the perturbation arm,
which gives the perturbed arm's EFC rather than the baseline's.

**Why it gates the perturbation.** Negligible cycling would make an inert ageing model
*physically correct* rather than a defect, and the perturbation predictable rather than
informative; substantial cycling near 1.4612/day would make inertness a genuine defect and
the perturbation exactly the test.

**The perturbation, predeclared if warranted:** set `ageing.calibration.status` back to
`DECLARED_NOT_CONSUMED`, restoring `k = 10000` — the same 15.4% change already characterised
at bootstrap capacity, hence like-for-like. Against the **164,149** bounded bar: moving more
means the ageing model is live and the 0.37% is attributable to a degrading-storage model;
moving less or not at all means the model is inert even where storage does real work, which
**combined with salvage at exactly zero would mean this run values effectively ideal,
non-degrading storage with no terminal value, and the paper's degradation modelling
contributes nothing to any number it reports.**

## P5.14-N — the ageing model at material capacity: LIVE, and destabilizing (2026-09-13)

Frozen spec `f1bbddf4` before either arm; rule eleven asserted before execution in both.
Full record in `P5_14_N_AGEING_PERTURBATION.md`.

**Control arm: the determinism gate PASSES exactly** — recourse `816,121,464.1554238`,
**delta = 0.0**, with cycles 67, solves 3,468, rule ten 0.9329 and zero local-solve failures
all matching. That confirms the campaign's only material-capacity number **and** proves the
added capture non-perturbing, so no separate neutrality run was needed. ESSO models
serialized (5.3 MB).

**The EFC answer: cycling is substantial.** 1.1124 EFC/day peak, ~0.99 average, uniform
across the three nodes — **above the 0.76 interpretability guide and below the 1.4612
floor-binding threshold** (margin 0.349, 76.1% of threshold). Cumulative SoH falls
`1.0 -> 0.8387 -> 0.7284 -> 0.6248`, a **37.5% capacity loss** over the horizon.

**Perturbation arm (`cl_eff` 11,541.56 -> 10,000, the same 15.4% change that moved nothing
at bootstrap capacity): the mechanism responded exactly and the solve destabilized.**
Degradation/day rose `9.64e-05 -> 1.112e-04`, a ratio of **1.1535** against the expected
1.1542; terminal SoH fell `0.6248 -> 0.5867`. But the run **hit the 90-cycle cap with 74 of
90 cycles carrying local-solve failures**, against **zero** in the control, and produced no
recourse.

**Verdict.** The predeclared branches both required a value to compare against the 164,149
bar, and a capped cell yields none — **the objective comparison is INCONCLUSIVE**. But the
question behind them is answered, the other way from the one we were braced for:

> **The ageing model is not inert at material capacity. It is live and consequential enough
> that a 15.4% change in its constant takes the problem from converging in 67 cycles with
> zero local-solve failures to not converging in 90 cycles with 74 failure-cycles.**

Physically coherent: smaller `k` means faster degradation, so `es_e_available = es_e_rated *
soh_cumul` shrinks faster and tightens the feasible set through the horizon. The SoH floor
is **not** the cause — terminal SoH 0.5867 is still above 0.50, margin 0.0867 against the
control's 0.1248.

**What this licenses.** It **retires** the "degradation modelling contributes nothing"
branch *at material capacity* — at bootstrap capacity it contributed nothing measurable, at
C\* it dominates the solve's behaviour. It does **not** attribute the 0.37% storage effect
to degradation value; that needed the inconclusive comparison. And it **raises a new
robustness concern**: the C\* baseline sits close to a regime where a 15.4% parameter change
causes local-solve failure in 82% of cycles — the converged baseline is real, but its
neighbourhood is not benign.

**Two harness defects recorded.** (a) The determinism gate was applied to an arm it does not
apply to, so the perturbed artifact carries a **spurious** `FAIL — NON-DETERMINISM` verdict;
the gate compares against the control's reference and the perturbation is supposed to
differ. Corrected in `n1_k10000_gate_correction.json` rather than by editing the result. The
real determinism result is the control's PASS. (b) Neither report stores per-cycle rows, so
the cycle at which failures began is unrecoverable — rule eleven covered the ESSO quantities
the spec named, and the spec did not name the trajectory. The rule worked as written; the
gap was in the requirement.

## NUMERICAL PROGRAMME CLOSED — Track D and the numerical work (2026-09-13)

### The failure mode: numerical fragility, not infeasibility

| source | `Optimal` | `Acceptable` | **`Max Iterations`** | `Locally infeasible` |
|---|---|---|---|---|
| network solves (4,368) | 4,348 | 19 | **1** | 0 |
| ESSO node 5 / 7 / 9 (last 120 each) | 120 / 47 / 112 | — | **0 / 73 / 8** | 0 |

Failures are **`Maximum Number of Iterations Exceeded`** in the **ESSO** subproblems,
overwhelmingly at **node 7**; `max_iter` is unset so each burned IPOPT's default 3,000
iterations. **No solve reported local infeasibility.** Decisive by contrast: **the same node
7, at 1.00 MVA in the ladder, returned `Converged to a point of local infeasibility`** — the
solver says "infeasible" when it finds infeasibility, and it did not say so here.
Qualification: `maxIterations` proves non-convergence within budget, not that the set is
untightened; what it establishes is that the perturbed problem **did not look infeasible to
IPOPT**. So the paper's sentence is the fragility one: a 6.1% cut in terminal available
energy (2.421 -> 2.274 MWh) leaves the solver unable to navigate 82% of cycles, not because
the plan becomes infeasible but because the subproblem becomes numerically intractable.

### A positive result, on the component we most suspected

`degradation/day` moved `9.64e-05 -> 1.112e-04`, ratio **1.1535** against the law's
`11541.56/10000 = 1.15416` — a **0.06% match**. **The first direct validation that the
degradation implementation does what the law says**, on the component we had four
independent reasons to think inert. **The ageing model is live and consequential at material
capacity; that branch is closed favourably.**

### P5.12 reframed retrospectively

P5.12 spent weeks on **one local NLP failure in 1,095 solves** at bootstrap capacity as a
rare anomaly. It was an **early symptom of a fragility that becomes dominant at realistic
scale** — one parameter-step from the material-capacity baseline it is the normal behaviour,
74 of 90 cycles. This makes that effort look **better**: it was chasing something real, at
the only magnitude where it was then visible.

### The campaign's question, answered negatively

| component | status |
|---|---|
| resolution | fine — 28x headroom at `rel` 1e-4 |
| reliability at a point | fine — zero failures in 3,468 solves at C\* |
| **reliability across the design space** | **not established; the one data point is catastrophic** |

A campaign varies the **candidate**, a far larger perturbation than 15.4% of a fixed model
constant. If a constant-step of that size produces 82% cycle failure, **a campaign cannot be
expected to hold together.** That is what Track D was for.

**C\*'s own numbers are implicated.** The 816,121,464 baseline sits one small parameter-step
from 82% failure and stopped at 93% of its threshold. **Wherever the 0.37% storage effect is
reported, this must be reported with it.**

### Closure

**Track D and the numerical programme are CLOSED** — not because nothing further could be
learned, but because the question they were asked has been answered specifically and
documented: **the tool cannot currently support a planning campaign at material capacity,
and we now know why in terms precise enough to write down.** Further stages would
characterise a fragility already established as disqualifying.

**Track E is the remaining work**, with a considerably sharper set of claims to reconcile
than when it started.

### Two process fixes promoted to `CLAUDE.md` stage templates

- **Scope a gate per arm** — a gate comparing every arm against a control reference is
  ill-defined for an arm designed to differ, and produced a spurious non-determinism verdict
  here.
- **Record per-cycle state by default** — any stage whose run can fail records the
  trajectory as standard. This was the **third** capture gap; rule eleven asserts what a spec
  requires, and a default is what stops a fourth arising from a spec that did not think to
  ask.
