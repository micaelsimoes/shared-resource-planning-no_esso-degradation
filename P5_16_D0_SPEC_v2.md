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
