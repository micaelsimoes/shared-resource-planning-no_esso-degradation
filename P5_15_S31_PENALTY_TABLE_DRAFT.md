# P5.15 Step 3.1 — Penalty classification and TSO/DSO objective consistency (DRAFT for author decision)

**Status: DRAFT — not signed.** Drafted by the Planner, 2026-09-15, with zero solves, from an independent code inventory by
the Advisor whose load-bearing citations the Planner re-read (§6). Authority: `PLANNER_BRIEF_2026-09-13.md`
Addendum 9 item 3.1. **The expert reviews; the author decides each row.** 3.2 onward waits on the signed table.

Code at `e19a580d` + harness commits (production unchanged since `92e3fafe`). Abbreviations: `defs` = `definitions.py`,
`mch` = `model_construction_helpers.py`, `net` = `network.py`, `srp` = `shared_resources_planning.py`,
`sesd` = `shared_energy_storage_data.py`, `np` = `network_parameters.py`.

## 0. What the recourse is today (verified)

- Every network uses `obj_type = COST`, so the objective is (mch:1557–1563):
  `gen_cost + flex_cost + load_curt_cost + gen_curt_penalty + ess_usage_cost + slack_penalties + ess_complementarity_penalties`.
- **Recourse** `gross_operational_cost` = Σ over TSO and DSOs of each network's base objective at its current
  weights (srp:740–746); `net` = `gross` − terminal salvage. Scenario-deviation and ADMM (AL, proximal) terms are
  excluded. The code's own comment (srp:742): *"may include artificial penalty terms and is therefore not necessarily a
  pure economic operating cost."* **Step 3.1 is what removes that caveat.**
- `_prepare_*_objectives_for_admm` zeroes some weights before ADMM (srp:3563–3572 TSO, 3712–3724 DSO); a zeroed weight
  contributes 0 to recourse.
- TSO/DSO base objectives are divided by `effective_scale` inside ADMM; the ESSO base objective is not (noted for 3.4).
- The ESSO objective (`feasibility_penalty`) is **not** part of recourse; only the salvage credit is.

## 1. The classification table

**E** = economic cost (part of Q(x)); **D** = feasibility detector (must be ~0 at a valid solution; reported, not
priced); **R** = numerical regularizer (a third category, proposed — see row 20); **A** = ADMM term (not a penalty).
"Author" marks rows where the classification is a modelling choice only the author can make.

| # | Term | Where | Weight (value) | In recourse now? | Current | **Recommendation** | Decision needed |
|---|---|---|---|---|---|---|---|
| 1 | Generation energy cost | TSO; DSO excluding its reference generator (mch:1566–1573) | market price | yes | E | **E** | — (DSO reference exclusion keeps imports priced once) |
| 2 | Flexibility activation cost, network-internal loads (`flex_p_down + flex_q_down`) | TSO, DSO (mch:1588–1598) | `cost_flex` (market data) | yes | E | **E** | **Author**: only the downward direction is priced, and Q at an active-power price — confirm intended |
| 3 | **Flexibility cost on the TSO's ADN interface load** | TSO only: ADN load created with `fl_reg=True` (srp:10606, 10614); `pc`/`qc` fixed to the consensus value, `flex_*` freed to the interface rating (srp:2963–2980); priced by the same function as row 2 | `cost_flex` | **yes** | neither — prices movement of the interface away from a fixed anchor | **Remove** (see defect D1) | **Author** — this row touches the definition of Q(x) |
| 4 | Load curtailment cost | TSO, DSO (mch:1613–1623) | `COST_CONSUMPTION_CURTAILMENT` = 300 (defs:46) | yes, but inactive (`l_curt` false) | E | **E** (value of lost load) | — |
| 5 | RES curtailment penalty | DSO active; **TSO weight zeroed** (srp:3567; DSO zeroing commented out, srp:3719); term mch:1638–1646 | `PENALTY_GENERATION_CURTAILMENT` = 1 (defs:59) | DSO yes; TSO 0 | ambiguous | **Identical on both sides**; default **0** unless the paper models a real curtailment cost | **Author**: E (compensation) or zero on both |
| 6 | Flexibility-usage penalty | congestion-management objective only (mch:1686–1696) | 1e-2 (defs:61), zeroed | inactive under COST | — | omit from appendix | — |
| 7 | Load-curtailment penalty | congestion-management objective only (mch:1661–1671) | 1e2 (defs:60) | inactive under COST | — | omit | — |
| 8 | ESS usage cost, network side — shared ESS (mch:1713–1718) and local ESS if `es_reg` (mch:1719–1722) | TSO, DSO | `PENALTY_ESS_USAGE` = 0.1 (defs:62); **zeroed on both sides** in ADMM (srp:3566, 3718) | weight 0 in ADMM | would be E | **Shared ESS: keep 0** (cycling cost belongs to the ESSO) | **Author**: the single zeroing also removes local-ESS usage cost — confirm intended |
| 9 | Bilinear ESS complementarity penalty `pch·pdch` (local and shared) | TSO, DSO (mch:1786–1790, 1794–1798) | `PENALTY_ESS_COMPLEMENTARITY` = 1e2 (defs:56) | yes | D | **D** — the hard constraint already exists (mch:730–732, 876–878); penalty redundant and indefinite | — (recommend removing the penalty term) |
| 10 | Local ESS day-balance slack | TSO, DSO (mch:1791–1792; bounded to 5 % of E) | `PENALTY_ESS_BALANCE` = 1e3 (defs:55) | yes | D | **D** | — (hard equality feasible by idling) |
| 11 | Shared-ESS day-balance slack | TSO, DSO (mch:1799–1800; variables **unbounded**, net:379–381) | `PENALTY_SHARED_ESS_BALANCE` = 1e3 (defs:58) | yes | D | **D** | Verify the ESSO imposes the same cyclic condition before hardening (not inspected) |
| 12 | Squared-voltage slacks | TSO, DSO (mch:1745–1746) | `PENALTY_VOLTAGE_SQUARED` = 5e4 (defs:49) | yes | D | **D, kept as a penalty**, reported ~0 at the reported point (hard limits risk infeasible intermediate ADMM iterates — inferred) | **Author**: D, or E if violations are meant to be priced |
| 13 | Flexibility day-balance slack, **P** (Candidate 4) | TSO, DSO (mch:1762–1771; bounds 0–0.01 pu) | `PENALTY_FLEXIBILITY` = 1e3 (defs:52) | yes | D | **D** (hard `p_up == p_down` is feasible; the old ±band caused active-set switching) | — |
| 14 | Flexibility day-balance slack, **Q**; TSO ADN-load flexibility slacks | TSO, DSO (mch:1772; Q constraint unwired, net:433; ADN loads skipped, mch:562–564) | 1e3 (defs:52) | yes | **orphaned** — penalized slack with no constraint | **Remove** (defect D2) | — |
| 15 | Node-balance slacks | TSO, DSO (mch:1747–1750) | `PENALTY_NODE_BALANCE` = 1e6 (defs:51) | inactive | D | **D** | — |
| 16 | Branch-flow slacks | TSO, DSO (mch:1752–1760) | `PENALTY_CURRENT` = 1e3 (defs:50) | inactive | D | **D** | — |
| 17 | Transformer/tap, generator, interface/consensus slacks; constants `PENALTY_GENERATION`, `PENALTY_ESS`, `PENALTY_SHARED_ESS`, `PENALTY_SETTLEMENT`, `EXPECTED_VALUE_PENALTY`, `COST_GENERATION_CURTAILMENT` | — | defined in defs but unreferenced in the ADMM path | no | — | omit (not in the objectives) | — |
| 18 | Scenario-deviation penalties (voltage, interface P/Q, shared ESS) | TSO (srp:2821–2855), DSO (srp:2858–2886); skipped at one scenario (Candidate 5) | `PENALTY_SCENARIO_DEVIATION` = 9e4 (defs:75); shared ESS 1e4 (defs:76) | no | regularization; inactive in SRP1 | depends on the model | **Author**: if interfaces are here-and-now, a softened non-anticipativity constraint (→ hard equality); if scenario-dependent interfaces are allowed, remove |
| 19 | ESSO aggregate slacks `slack_es_pnet_up/down` | ESSO (sesd:750–758) | `PENALTY_ESSO_SLACK` = 1e3 (defs:63) | no | D | **D** — measured ~0 from cycle 1 (G2 re-run); binding only at initialization (end-of-life rule, recorded as intended) | — |
| 20 | `EPS_ESSO_THROUGHPUT · Σ(pch+pdch)` | ESSO (sesd:767–773) | 1e-5 (defs:74) | no | regularizer | **R — proposed third category**: cannot be zero at a valid solution; report as a regularizer with its measured bias (price effect at 1e-3; 0.0029 % spurious throughput at the baseline) | **Author**: accept category R in the appendix |
| 21 | Terminal salvage value | recourse only (subtracted, srp:750; sesd:775–779) | salvage parameters (SRP1_ESS_Params.json) | subtracted | E | **E** (credit) | — |
| 22 | AL, proximal and previous-iterate terms | TSO, DSO, ESSO (srp:3649–3705, 3800–3828, 3862–3875) | `rho`, `gamma` (SRP1_params.json) | no | A | **A** — label as ADMM terms, not penalties | — |

## 2. TSO/DSO inconsistencies to resolve

| # | Item | TSO | DSO | Recommendation |
|---|---|---|---|---|
| A1 | RES curtailment | weight 0 (srp:3567) | weight 1 retained (srp:3719 commented) | **Make identical** (row 5); author sets the value |
| A2 | ESS usage | 0 (srp:3566) | 0 (srp:3718) | consistent; **keep**, with the stated reason (row 8) |
| A3 | Scenario deviation | sums over all ADNs and shared ESS | own interface only | consistent in weights; inactive in SRP1; **keep** (row 18 decision governs) |
| A4 | Interface flexibility cost | charged (row 3) | no counterpart | **Remove, or the author justifies** |
| A5 | Proximal regularization | enabled | not enabled | out of 3.1 scope (does not enter recourse); recorded |
| A6 | Previous-iterate AL | no linear dual term | has one | inactive; recorded |
| A7 | Reference-generator exclusion from generation cost | not applicable | excluded (mch:1570) | **Keep** (prevents double pricing of imports) |
| A8 | Flexibility day-balance on interface loads | skipped (mch:562–564) | — | keep the skip; remove its orphan slacks (row 14) |

## 3. Defects found (independent of the classification decisions)

| # | Defect | Effect | Proposed handling |
|---|---|---|---|
| **D1** | TSO charges `cost_flex` on the ADN interface load against an anchor fixed at model creation (row 3) | Q(x) depends on the initialization anchor | **Raise before 3.2**; resolved by the row-3 decision |
| **D2** | Penalized slacks with no constraint: flexibility Q day-balance; TSO ADN-load flexibility slacks | pure barrier noise inside recourse | remove (row 14) |
| **D3** | `get_feasibility_violation` = `feasibility_penalty / PENALTY_ESSO_SLACK` (sesd:109–110), but `feasibility_penalty` now includes the ε throughput term | the reported ESSO "violation" is contaminated by row 20 | report slacks directly |
| D4 | ESS day-balance slacks sit inside the "complementarity" expression and are logged as `ess_comp` | misleading diagnostics | split in logging |
| D5 | TSO/DSO base objectives divided by `effective_scale`; ESSO base not | ESSO penalties inflated ~σ relative to network costs inside ADMM | **defer to 3.4** |
| D6 | Shared-ESS day-balance slack unbounded (net:379–381); local-ESS slack bounded | asymmetry | bound, or harden per row 11 |
| D7 | Previous-iterate AL asymmetry (A6) | inactive | record |
| D8 | `convex_oracle.py` duplicates `_prepare_*` logic and zeroes RES curtailment | historical module, marked not built | record only |

## 4. What the committed artifacts can and cannot say about magnitudes

The logs record per-block **deltas** of objective components, not **levels**, and network models are not serialized.
So for rows 9–14 the terminal level at the new baseline is **unknown from artifacts**. Indications only:
voltage-slack deltas fall to ≤ 1e-6 late in the run; the flexibility P slack sat at its 0.01 pu bound at cycle 1 and
then fell (inferred); the orphan Q slacks (row 14) are nonzero at O(0.1–1). Whether reclassification **changes Q(x)**
or is a presentation change therefore cannot be decided from existing evidence.

## 5. Recommendation for the next gate (for the author)

Before or with 3.2, capture per-block **levels** of each objective component at the terminal cycle — zero extra solves,
using the existing component functions — with three splits: day-balance separated from complementarity; TSO ADN-load
flexibility cost separated from internal flexibility cost; orphan slacks separated. Reading rule: if every row the
author signs as **D** is at or below the stopping-slack bar at terminal, the reclassification does not move Q(x);
otherwise the signed table changes the recourse, and that change is reported.

## 6. Verification performed by the Planner

Re-read directly: objective composition (mch:1557–1563); flexibility cost (mch:1588–1598); RES curtailment term
(mch:1638–1646); flexibility slack penalty including the orphan Q slack (mch:1762–1772); Q balance constraint unwired
(net:430–434); `_prepare_*` zeroing (srp:3563–3572, 3712–3724); recourse definition (srp:740–750); TSO ADN-load fixing
and flexibility freeing (srp:2955–2980) and `fl_reg=True` on ADN loads (srp:10600–10616); ESSO `feasibility_violation`
(sesd:103–110) and objective (sesd:765–785); shared vs local ESS slack bounds (net:350–382). All matched the inventory.
Rows not listed were taken from the Advisor's inventory without an independent re-read.
