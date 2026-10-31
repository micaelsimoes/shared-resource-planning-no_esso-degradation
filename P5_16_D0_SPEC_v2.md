# P5.16 D0 — relaxation gap of the current formulation at the two SRP1 references (spec v2)

Planner, 2026-10-30. Supersedes `P5_16_D0_SPEC_v1.md` (kept). Advisor verdict on v1: build with changes — seven
blocking, all integrated; every factual claim below was re-verified by the Planner against the record or the code.
Status: **ready for the Worker once the author answers §2's Studio check**; nothing runs before that answer.

## 1. Question (unchanged) and the decision rule (changed)

Is the SOC relaxation of the **current** production recourse tight enough for a valid lower bound to be informative?
Two readings are now separated, because the record (P5.5-D2/D4) shows the dominant looseness is the **TN rank gap of
the meshed 9-bus**, which is plausibly plan-independent, not storage circulation (negligible in absolute terms):
- **Absolute:** Σ_groups gap / Q_cert per cell → H1 ≤ 1e-4, H2 ≤ 1e-2, H3 otherwise.
- **Differential (decision relevance):** |Σgap(unit) − Σgap(x = 0)| against the storage value 259,375 € and the
  bar-sum. If the gap is plan-independent within the bars, the LB ranks plans even when the absolute gap is H2 — a
  fact for the expert's R4 wording (the certificate UB − LB stays non-discriminating; the LB-optimal plan coinciding
  with the UB incumbent is evidence, not proof).

## 2. The certified points — two routes

**Cells (named):** the W86 tight-tail reference pair, same frozen configuration, both with
`persist_certified_models: true`:
- x = 0: `data/SRP1/Results/P515S53/tight_tail_w86/campaign_s53_w86_tail_recert/evals/5cfe69a615ae3708_x0/`,
  certified cycle 132; `certified_models.pkl` sha256 `d816bb4b1ade4b44ad4a77872443fe2668986663c524fc4239815d4657974e21`;
  `esso_models_s39_D.pkl` `15f384090089e123a295d8a3f5d0d6b2dc4e953effed55b28eae456a38529d8d`.
- unit 7: 0.25 MVA / 1 MWh (2025): `…/evals/ca8927e75d628bd1_n7_4h_e1/`, certified cycle 112;
  `certified_models.pkl` `fe06ca485003b4fc9c0fd9788279bd287dd813d659179b311885e8ab03efda04`;
  `esso_models_s39_D.pkl` `95f7b38a6ec21833f7b5521ec8332d8022ef6801b4095d1410225438b2cedfc9`.
- Gross costs are read from each cell's `evaluation_record.json` (the Worker states them in the report).
- The settled x = 0 (W101/W106, cycle 181) differs from this x = 0 by +14,970.62 € (2.3e-5 of Q): reported beside
  the results as UB uncertainty, not used as the mapped point.

The pickles are hash-recorded in the child manifests but **absent from this checkout**: the cells ran from the Studio
checkout `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation`. The v1 premise "no certified
cell has persisted network models" was true of this machine only.

- **Route A (preferred, zero ADMM solves):** the author checks the Studio for the four files above; if present with
  matching sha256, they are copied to this machine under the same relative paths (read-only), hash-verified on load,
  and the live blocks are the certified models. The Addendum 87 "not zero-solve" correction is then reversed.
- **Route B (only if absent):** in-process reproduction of each cell under its frozen configuration, gated bitwise
  against its own record with a comparator **parameterised to that record** (the `p515_s53_srp1_bitwise_gate.py` →
  W48/W39/W35 → W10 lineage) — not `p515_s40_polish_gap._reproduction_check`, which hard-codes the s39_D cell
  (cycle 139). ≈ 1.5–2 h each on the 24 GB machine, attended, alone.
- Either route: **capture before mutation** — the certified Var vector of every block, the live blocks and ESSO
  pickled under `P516D0/` with hashes, `state['dual_vars']` and the consensus vectors.

## 3. Construction — deactivate-and-add on the live production blocks

Production already carries `vmag_sqr`, `voltage_product_real`, `voltage_product_imag` and writes every balance, flow
and thermal row in them; rank-1 enters only through the definitional rows `vmag_sqr_def` (Wii = e² + f²),
`vmag_def` (interface nodes), `voltage_product_real_rule`, `voltage_product_imag_rule`, and `r_sqr_def` (taps). Per
cloned block:
1. Deactivate those five families; add the rotated cone `WijR² + WijI² ≤ Wii·Wjj` per active branch and period
   (written as the bilinear inequality; Wii ≥ v_min² > 0). Dormant angle rows stay dormant (production has them
   commented out). `voltage_product_real ≥ 0` may stay.
2. Transformer branches only (inventory first — whether SRP1's case9/case33 flag `is_transformer` is recorded):
   the B2 (U, C, D) envelope for the `r·cross` and `r_sqr·vmag_sqr` bilinears.
3. **Network copies' capacity Params:** production sets `shared_es_s/e_rated_fixed` from the ESSO's *available*
   capacity each cycle. Variant **R** (the LB object of Phase 2): reset to **rated** on TSO and DSO copies.
   Variant **A**: leave the certified available values (tighter; still a valid LB of production). Both solved; the
   difference attributes the degradation-relaxation share of the gap.
4. **ESSO:** SoH ≡ 1 (exp chain, floor, bilinear available rows deleted; rated through the window); H3 rows inert at
   both cells (≤ 1 cohort per year) and dropped in general; **`slack_es_pnet_up/down` fixed at 0** (they are
   penalised only in the ESSO objective, which the gross objective excludes — free slacks would decouple `es_pnet`
   from the cohort powers and collapse Q_LB); the ε-throughput term is **not** in the LB objective (gross excludes
   the ESSO), so the pch/pdch split is non-unique — accepted for a bound.
5. Objectives: deactivate every ADMM / rescaled / ESSO-penalty objective; LB objective = Σ_b w_b ·
   [`objective_function_rule`(b) − contracted settlement_b − voltage pin_b] with w_b = `_get_admm_block_weight`, built
   from the production expression components (`_get_operational_recourse_components` lineage) — never re-typed.
6. Coupling as exact equalities on the models' own quantities with s_base / v_base conversions: voltage on
   `vmag_sqr` (Wii) between TSO ADN node and DSO reference; interface P/Q on the TSO's `pc_adn/qc_adn` (or
   `pg_adn/qg_adn`) against the DSO's interface Expressions — **not** `p56a_oracle._interface_expression` (the S41
   defect); storage P/Q between TSO, DSO and ESSO copies (`shared_es_pnet/qnet` ↔ `es_pnet/qnet`). No AL or
   proximal terms. The DSO reference's ±1e-5 `f` box has no W counterpart and is dropped (rotation invariance) —
   listed.
7. Coordination reshaping is inherited from the live blocks (TSO: ADN voltage slacks fixed 0, `pc/qc` fixed at the
   DSO cycle-0 anchor, flex legs fixed 0, `interface_delta_p/q` freed ±rating; `penalty_shared_ess_usage = 0`,
   `penalty_gen_curtailment = 0`, `interface_settlement_weight = 1`; DSO ref: `e` freed, ref slacks ub 0) — recorded,
   not re-applied.

## 4. Gates (fail-fast, in order)

- **G-D0-0 Builder dry run (zero solves, before any load/reproduction):** build the relaxation on the cycle-7
  FrozenSMOPF TSO snapshot and the s39_D ESSO pickle; row inventory and mapped-point machinery exercised.
- **G-D0-1 Point acquisition:** route A hash match on all four files, or route B bitwise reproduction. Stop on failure.
- **G-D0-2 Row inventory:** every active production row of every live block classified **retained / replaced-by /
  dropped-with-argument**, with counts asserted per family. (A missing row only *lowers* Q_LB and is invisible to any
  value gate — this inventory is the only detector.)
- **G-D0-3 Identities at the mapped point (per block, per row):** retained rows: residual equals the production row's
  residual, |Δ| ≤ 1e-12; cones: ≤ 1e-12 relative at the rank-1 point; transformer envelope ≤ 1e-9; coupling rows:
  residual equals the inter-copy mismatch read from the live blocks, reported per channel in physical units beside
  eps_pri (Boyd's r is an L2 norm over all blocks — a per-row tolerance from it would be vacuous). Objective identity
  per block (TSO + 3 DSO) and per group, threshold 1e-8 relative, achieved value recorded (expect ~1e-15).
- **G-D0-4 Containment, sign corrected:** Q_LB(group) ≤ Q_cert(group) + solver gap. A violation means an **extra or
  tighter** row or a wrong objective (never a missing row). Also **P7** below.
- **G-D0-5 Solver and duals:** Gurobi `QCPDual 1`, `NonConvex 0`, `BarQCPConvTol ≤ 1e-9`, `BarConvTol ≤ 1e-10`,
  `LogFile` captured; any log containing "failed to compute QCP dual" marks the solve dual-less (recorded, Clarabel
  trigger); IPOPT `tol 1e-9` on the same relaxation (smooth form); agreement ≤ 1e-7 relative or ≤ the sum of the two
  solvers' own gaps; each solver's primal-dual gap reported as **Q_LB's resolution** (must be ≤ 0.1 × 1e-4 × Q for
  H1 to be decidable). Dual bound as the **Lagrangian margin**: with y ≥ 0 on inequalities and all variables boxed,
  g = L(x*, y) − Σ_i |∇_x L(x*, y)_i| · w_i ≤ p*; require g ≤ Q_LB ≤ g + 1e-6 |Q_LB|. Informative for D0, gating in
  Phase 2.
- **G-D0-6 Guard:** `SolveProfileGuard` permitted site = **one leaf function** that only calls the relaxation solver
  and from which no production entry point is reachable; declared count = 2 × (number of relaxation solves),
  verified exactly.

## 5. Scope and measurements

**All 12 (year, day) groups per cell** (the point acquisition dominates the cost; a group is ≈ 1/12 of the P5.5-C
monolith that solved). Per group and summed: (a) gap (Q_cert − Q_LB)/Q_cert, variants R and A; (b) at the unit:
the P5.5-D4 **fixed-mode re-solve cost** (each copy's charge/discharge mode fixed to the certified sign per period)
in € and % of group gross — replaces the pch/pdch ratio, undefined at x = 0; (c) TN rank residual per branch-period
and the cycle-angle residual, plus a **two-way freeze attribution** (TN frozen at certified W vs DSO + ESSO frozen)
bracketing the interaction — labelled as such, not as a "rank-gap cost"; (d) DSO rank residuals; (e) OLTC envelope
slack if transformers exist; (f) dual consistency at the unit: −(QCPi_E · 1 MWh + QCPi_S · 0.25 MVA) per group vs
ΔQ_LB(group) = Q_LB(x0) − Q_LB(unit), and ΔQ_LB vs W183e's per-day contribution c; (g) wall time, size, peak RSS.

## 6. Predictions (recorded before the run; each falsifiable)

P1 DSO blocks rank-exact ≤ 1e-5 (P5.5-C). P2 fixed-mode re-solve cost at the unit ≤ 0.05 % of group gross (P5.5-D4:
0.049 %). P3 Σgap/Q point estimate **1e-3** (H2) for variant R; H1 would falsify the mechanism reading, H3 the TN-gap
reading. P4 Gurobi returns duals on all 48 relaxation solves. P5 dual consistency: sign negative and within a factor
3 of ΔQ_LB per group. **P6 (decisive): |Σgap(unit) − Σgap(x0)| ≤ the bar-sum of the two cells** (gap plan-
independent); point estimate ≤ 5 k€ against the 259,375 € value. **P7: Q_LB(unit) ≤ Q_LB(x0) per group** to solver
tolerance (containment: the unit's feasible set includes x = 0's). P8 OLTC envelope slack ≤ 1e-5 if transformers
exist. P9 TN rank residual ≈ 2.45e-2 relative (P5.5-C8), cycle-angle residual ≈ 1.2e-2 rad.

**Expert predictions (Addendum 88, recorded before any conic solve; variants per §11):** at x = 0 the variants R, A
and C coincide identically. Unit cell: per group gap_A ≤ gap_C ≤ gap_R; summed over the 12 groups
gap_R − gap_C ∈ [10, 50] k€ and gap_C − gap_A ∈ [5, 40] k€; **P6 holds on A** (network gap plan-independent) and
**fails on R** by about the degradation share (≥ 15 k€).

## 7. Captures (every quantity below has a capture path asserted before the run)

Pickles of all live blocks + ESSO before mutation; `dual_vars` and consensus vectors; each relaxation exported to
`.mps`/`.lp` with a name map (enables Clarabel later without re-acquisition); mapped-point vector and per-row residual
table; Gurobi and IPOPT logs and multiplier vectors; storage mode pattern per copy; rank residual per branch-period;
per-block gross components; the `e_available` values the live blocks carried; peak RSS; the cell key and route in
every artefact. Root `data/SRP1/Results/P516D0/` (new); harness `p516_d0_*.py`; report `P5_16_D0_REPORT.md`.

## 8. Not permitted
Production code changes; any ADMM run other than route B's two reproductions; any solve outside the leaf; the Studio
as a compute node (file copy only); writes under existing roots.

## 9. Interpretation
Absolute class from Σgap/Q (variant R). Decision relevance from P6/P7. Reported together; the expert's R4 wording is
chosen on both. G-D0-3 failure stops the realignment (the relaxation is not of production's problem).

## 10. Planner rulings after T0.1a (2026-10-31; `P5_16_D0_T01A_REPORT.md`, evidence `P516D0/dry_run/`)

Dry run: zero solves (guard 0/0), inventory complete on every block, cones 6.8e-16, objective identity 0.0; two
predictions failed (P-a, P-b) for reasons that are properties of the instance, not of the builder.

1. **Retained-row identity gate restated to floating-point resolution.** The 1e-12 absolute bar cannot be met on
   case33_2: `branch_flow_limit` carries |y|² ≈ 1.08e5, so one ulp of W (2.2e-16, from mapping e² + f²) moves the row
   by 4.8e-11 — exactly the first-order bound. Gate becomes: (i) **same-point identity, production row vs relaxed row
   at identical W: Δ = 0 exactly** (primary; held on 5,261 + 2,808 rows); (ii) at the mapped point
   |Δ| ≤ max(1e-12, 8·eps·Σ|terms|) per row (held: 0 rows above first-order bound + 4× resolution). The 1e-10 stop
   line is unchanged.
2. **Inventory "replaced" class** = the five definitional families **plus the envelope-rewrite rows**: on the OLTC
   branch (one per case33_k; none on case9) the rows containing r·WijR, r·WijI, r_sqr·Wii (`node_balance_p/q` at the
   two transformer nodes, `pij/qij/pji/qji_def`) are rebuilt in (U, C, D) from the production row's own degree-2
   representation — matched to 1.2e-13. Nothing else is replaced; nothing is dropped on a network block except
   ruling 3.
3. **`sess_comp` is dropped in the relaxation** (`pch_hat · pdch_hat ≤ 1e-4`, the `BILINEAR_RELAXATION` branch of
   `model_construction_helpers.sess_comp_rule`; 24 DSO + 72 TSO rows): a nonconvex row whose removal enlarges the
   feasible set (lower-bound safe; P5.5-D2: the convex hull of the complementarity set inside the box is the
   triangle production already imposes). **Any other nonconvex row found on the certified blocks is listed in the
   inventory and is a stop**, not a silent drop. Reported to the expert as information in Addendum 88.
4. **FrozenSMOPF loadability** (21/24 snapshots fail to unpickle: `s_bounds` deleted in P5.4-B `1e86d40e`, an
   unwire-never-delete violation predating the rule): not needed for D0 (the W86 certified pickles post-date it);
   recorded as a housekeeping repair (retained unwired stub + x = 0 bitwise gate) for a later task.
5. Side writes: production's data loader regenerates untracked diagrams in `data/SRP1/Diagrams/`; the harness now
   redirects them. No tracked or cited file touched.

**T0.1b (next, zero solves):** the same builder on the route-A certified blocks of both cells, all 12 groups, both
capacity modes; capture before mutation (pickled live blocks, Var vectors, inter-copy mismatches); inventory with
the nonconvex-row listing; coupling rows (§3.6) built and their residuals reported as the inter-copy mismatch per
channel; gates per rulings 1–3; `.lp` exports with name maps for every group. **T0.2** (the 96 declared conic solves
+ IPOPT cross-checks) follows only after T0.1b's gates hold.

## 11. Addendum 88 amendments (expert, 2026-10-31) — binding before T0.2

- **D-1. Variant A is not a lower bound of production.** Fixing available energy at the certified values excludes
  dispatches that age the unit less; A is a bound only conditional on the certified SoH trajectory — an
  **attribution diagnostic, never quoted as a bound** (any artefact that does is a stop).
- **D-2. Variant C added:** calendar-only available energy per F4 — on SRP1's end-of-block convention
  rated × φ_cal^{age at block end} on the network copies (and the ESSO's available rows). Same gates, same exports.
  C is the Phase-2 object.
- **D-3. Readings:** the mechanism question (is the network gap plan-independent?) is read on **A**; the decision
  relevance for R4 is read on **C**; R is reported, its differential contains the plan-dependent degradation
  relaxation by construction, so a P6 failure on R alone says nothing about the rank gap. **Absolute class on C.**
- **D-4.** The §10 rulings are confirmed (identity at fp resolution with same-point Δ = 0 primary; envelope-rewrite
  class; `sess_comp` dropped; FrozenSMOPF repair later).
- **D-5. Transfer.** D0 measures the superseded convention (SRP1, end-of-block); its class transfers to the five-block
  instance on the reading that the network gap is per block and plan-independent; **the first Phase-2 LB evaluation
  on the new instance re-measures it at x = 0 and the unit — a gate, not an assumption.**
- **R4 wording is ruled** (Addendum 88 R4): G in k€ and g = G/F(x̂) always together and beside the stakes; decomposition
  of G where measured; "no plan on the lattice can lower the total cost by more than G below that of x̂" in every case;
  forbidden words listed there.
- **T0.1b** proceeds as ordered and adds variant C to its builds and exports (follow-up task if it has returned
  without it).

## 12. Planner rulings after T0.1b (2026-10-31; `P5_16_D0_T01B_REPORT.md`, evidence `P516D0/t01b/`, manifest `4e5bbd45…`)

T0.1b: zero solves; Q1–Q5 all held; same-point Δ = 0 on 810,912 row checks; coupling residuals equal the recorded
inter-copy mismatches (worst 0.139 × eps_pri); objective identity 5e-15 per block, 12-group sums equal the recorded
gross to 2e-16 / 6e-16. The stop rule fired on the retained-row Δ at the rank-1 mapped point (max 2.08e-9) because the
**certified points satisfy the definitional rows only to IPOPT tolerance** (`vmag_sqr_def` 4.1e-11, `r_sqr_def`
1.8e-11), not to one ulp as the T0.1a fixtures did; |y|² ≈ 1e5 amplifies that. 0 rows exceed their first-order W
bound + 4·eps·Σ|terms|. Rulings:

1. **Gate restated (option (a)).** The mapped point is the exact rank-1 point (cones ≤ 7.3e-16). Ruling 1(ii) and the
   stop line become: |Δ_row| ≤ first-order W bound (|∂row/∂W| · |δW|, δW = the certified definitional residual) +
   4·eps·Σ|terms|, per row; the same-point Δ = 0 remains the primary construction test. The certified definitional
   residuals are reported per block as the **mapping offset**. Re-evaluated from `identities.json`; no re-run needed
   for retained rows.
2. **Envelope-rewrite rows:** the same first-order statement (the scratch check that found all 960 affected rows within
   6.110e-9 vs a bound of 6.113e-9 is repeated **inside the evidence root** in T0.1c); at the certified-W point they
   match to ≤ 1.3e-13.
3. **Variant C defined (SRP1, end-of-block convention):** for a cohort installed in block y_inv, in block y ≥ y_inv,
   age_end(y) = Σ_{k = y_inv..y} num_years_k; available energy E_C = E_rated · φ_cal^{age_end(y)} with φ_cal =
   `calendar_retention_per_year` (0.985) from the ESS params; available power S unchanged (rated). ESSO: the SoH ≡ 1
   available rows become `es_e_available_per_unit == es_e_rated_per_unit · φ^{age_end}`; network copies'
   `shared_es_e_rated_fixed` set to E_C. At x = 0 all three variants coincide identically (prediction).
4. **Capacity-derived bounds follow the variant.** Variant R (and C) reset every bound derived from available capacity
   — in particular the `slack_shared_es_soc_final_up/down` bounds set by `configure_shared_ess_operational_state`
   (e·fraction + 1e-5) — to the variant's capacity, by calling the production function with that capacity or by the
   same arithmetic; the builder inventories every Var bound that depends on capacity and lists it. "R = production at
   rated capacity" then holds in every bound.
5. **Containment comparator for T0.2 (G-D0-4).** The certified gross contains −13,012.57 € (x = 0) / −13,032.47 €
   (unit) of detector-penalty terms from slacks at ≈ −1e-8 inside IPOPT's bound relaxation (Addenda 32–34: a known
   residue, ≤ 2e-5 of Q absolutely, ≤ 181 € in differences; here 19.90 € between the cells). A bound-respecting conic
   solve cannot reach them. **G-D0-4 compares Q_LB against Q_cert⁺ = gross with every slack-penalty term re-evaluated at
   the bound-projected slacks (max(slack, 0))**, i.e. the AC-feasible-within-tolerance value; both gross and Q_cert⁺
   are reported with the correction stated. The absolute class and P6 are read against Q_cert⁺; the correction cancels
   in P6 to 19.90 €.
6. **Builder committed with the evidence** (`p516_d0_build_relaxation.py`, the +90-line extension: `drop_sess_comp`,
   `nonconvex_active_rows`, the BILINEAR_RELAXATION form check).

**T0.1c (next, zero solves):** variants R, A, C for both cells and all 12 groups with rulings 1–4 applied; the
envelope first-order check preserved; Q_cert⁺ computed per group and cell; exports for all three variants (72 LP
files); the capacity-derived-bound inventory. Predictions: all retained and envelope rows within the restated gate in
every block-mode; the three variants coincide at x = 0 bitwise in the exported LPs; Q_cert⁺ − gross = 13,012.57 /
13,032.47 €. **T0.2** then runs the declared conic solves on R, A, C.

## 13. Planner rulings after T0.1c (2026-10-31; `P5_16_D0_T01C_REPORT.md`, evidence `P516D0/t01c/`, manifest `d2955128…`)

T0.1c: zero solves; C1, C3, C4, C5 held; 1,209,360 retained-row checks and 41,472 envelope checks inside the restated
gate (max ratio 0.9999999982 — on single-W-term rows the first-order bound is the row's own Δ, so the gate carries
information on multi-term rows); Q_cert⁺ − gross = 13,012.57 € (x = 0) / 13,032.47 € (unit), equal to the recorded
detector penalty to 1.8e-12 €; E_C/E_R = 0.985^{5,10,15} to 1e-15; two value-dependent bounds per network storage copy,
reset for R and C. **C2 ruled semantically held:** R and A LPs identical in body at x = 0; C differs only in the 27
calendar available-energy rows (name and a φ^{age} coefficient on `es_e_rated_per_unit`, a Var pinned to 0 by the
retained rated-capacity rows) — same feasible set, same objective. Noted, unverified: the network copies carry an
available energy 6e-6 relative below the ESSO's certified value (hypothesis: the pre-ESSO-solve iterate of the
certified cycle) — reported in T0.2 beside variant A, not acted on.

**T0.2 declared solves:** Gurobi on every exported group LP — 3 variants × 2 cells × 12 groups = 72 — plus, on
variant C at the unit cell, the P5.5-D4 fixed-mode re-solve (12) and the two-way freeze attribution (24): 108 Gurobi
solves; IPOPT smooth-form cross-check on variant C, both cells, 12 groups each: 24 solves. **132 declared**, leaf
guard, counted exactly. Group separability of the relaxed ESSO (no row couples different (year, day) once the SoH chain
is gone) is asserted by inventory before solving, so Σ_groups Q_LB is the joint relaxed optimum.
