# P5.16 D0 — one-block relaxation gap on the current formulation (spec v1, for Advisor review)

Planner, 2026-10-30. Authority: author's direction of 2026-10-30 (`REVISION_REALIGNMENT_ANALYSIS_2026-10-30.md` v2;
`PLANNER_BRIEF_ADDENDUM87_DRAFT_2026-10-30.md`, D0 order). Status: **draft** — the Advisor reviews this before any
Worker task is issued; the Worker builds from the reviewed version (v2) only.

## 1. Question

Is the SOC (W-space) relaxation of the **current** production recourse — gross objective convention, P5.15 ESSO with
SoH ≡ 1, ε-regularised complementarity, exact coupling — tight enough, on SRP1, for a valid lower bound to be
informative about the storage decision? Outcome classes (analysis §6.3): H1 gap ≤ 1e-4 of the block's Q; H2 ≤ 1e-2;
H3 otherwise or circulation > 0.05. P5.5-C's 21.9 % does not transfer (pre-Addendum-11 objective, apparent-power
ESSO, old complementarity).

## 2. What runs (declared; nothing else)

**Not zero-solve.** No certified SRP1 cell has persisted network models (only ESSO terminal models survive — the
Addendum 44 finding holds for SRP1 too; `find` over `data/SRP1/Results` returns no `certified_models.pkl`). The
certified block points are therefore obtained the way `p515_s41_hull_polish.py` obtained them: **reproduce the cell
in-process to its certificate, gated bitwise against the committed report, then work on the live Pyomo models.**

- R-x0: in-process reproduction of the x = 0 reference under the current frozen configuration
  (baseline spec `data/SRP1/Results/P515S47/frozen_s47_baseline_spec_v17_ff0056b8.json`), via
  `p515_g_g1_g4_admm_gates.run_admm_arm(...)` exactly as `p515_s40_polish_gap.py` / `p515_s41_hull_polish.py` call
  it, with `p515_s40_polish_gap._reproduction_check` **by import** against the committed certificate of that cell
  (the Worker names the record path, its sha256, the certified cycle and the gross cost in the spec v2 **before**
  running; the Planner verifies them against `REVISION_CONTEXT.md` "Results" and the s47 recert records).
- R-unit: the same for 7: 0.25 MVA / 1 MWh (2025).
- Each reproduction ≈ 1.5–2 h on the 24 GB machine (SRP1 ≈ 2.2 GiB; 4 P-cores), attended, both streams captured,
  alone on the machine; the Studio is not used. Reproduction failure (any non-bitwise field outside the documented
  field-list drift) **stops D0**: the relaxation would not be of the certified point.
- Conic solves: per selected block, one Gurobi solve (`QCPDual = 1`, `NonConvex = 0`, barrier, no crossover
  dependence) and one IPOPT solve of the same relaxation in smooth form. No other solver call; the production
  network/ESSO solve paths are not re-entered after the reproduction — enforced by `SolveProfileGuard` armed for the
  conic phase with permitted production count **0**.

## 3. Block selection

For each cell, one (year, day) coupling group: the 2025 day with the largest storage contribution in the unit cell's
records (the declared day of W183e, 2025 Spring, if the records confirm it; otherwise the day the records rank
first — stated). The group is the TSO block + the three DSO blocks + the ESSO restricted to that cohort-year, i.e. the
smallest unit on which exact coupling closes. A second group (2035, same day) is added for the unit cell only if time
allows, to see the degraded-capacity end of the horizon; it is optional and labelled so.

## 4. The relaxation (built from the live, certified production blocks)

Design reference: `convex_oracle.py::build_ac_block` (P5.5-B2 W-space families; **historical, does not build** —
reused as design, not imported). Rules that do not touch (e, f, r) are called from production by reference; only the
voltage families are rewritten:

1. W variables: Wii, WijR, WijI per branch and period; Wii = e² + f² and the pair products at the certified point
   define the **mapped point**. Production's voltage bounds verbatim (PV setpoint pinning, slack-relaxed limits).
2. Rank coupling relaxed to the rotated SOC WijR² + WijI² ≤ Wii·Wjj; transformer taps by the B2 uniform treatment
   (continuous r ∈ [r_min, r_max] envelope). Dormant ±30° angle rows **not** enabled (B1).
3. Nodal balances, branch flows and thermal limits in W.
4. Converter circle ‖(p, q)‖ ≤ S for storage and flexible resources as in production (jointly convex in S).
5. ESSO: SoH ≡ 1 — `exp` chain, floor row and the bilinear `available_e_capacity_unit` rows deleted; available =
   rated through the cohort window (linear in x); H3 pro-rata rows (`es_pnet_cohort_share_h3`) **dropped**;
   ε-throughput term kept (linear). Salvage not in a block (terminal) — out of scope here.
6. Coupling: exact equalities between the TSO and DSO interface copies (V, P, Q) and between the three copies of the
   storage P/Q, replacing the consensus terms; **no** AL/proximal terms; the ADMM objective deactivated and the
   unscaled base objective activated (`p515_s40_polish_gap._switch_to_base_objective` by import, unchanged).
7. Objective: `gross_operational_cost` of the group — contracted interface settlement and voltage pin excluded,
   scenario-deviation costs and slack penalties included — computed by the **production** component functions
   (`_get_operational_recourse_components` / block components) on the relaxed blocks, so the convention is shared by
   construction, not re-implemented.
8. Coordination reshaping as production applies it in the distributed solve (TSO: ADN loads' bounds and flex
   ratings, node voltage slacks fixed; DSO: interface magnitude freed, angle reference retained) — the relaxation
   bounds the **coordinated** feasible set.

## 5. Gates (fail-fast, in order)

- G-D0-1 **Reproduction** bitwise (R-x0, R-unit) — stop on failure.
- G-D0-2 **Mapped-point identity.** The certified point mapped into the relaxation: (a) every relaxed row satisfied
  within max(1e-6, the recorded consensus residual of its channel) — the coupling rows within the certified
  interface/storage residuals, every other row within IPOPT's constraint tolerance; (b) the group's gross objective
  recomputed on the mapped point equals the certified per-block sum to ≤ 1e-8 relative. **Failure stops the
  realignment** — the relaxation is not of production's problem.
- G-D0-3 **Convexity/containment sanity.** Q_LB(block) ≤ Q_cert(block) by ≥ −1e-9 relative (the solver's primal
  value may exceed the true infimum only within tolerance; a violation beyond it means a missing row or a wrong
  objective). Gurobi and IPOPT primal values agree to ≤ 1e-6 relative.
- G-D0-4 **Dual quality (Gurobi).** `Pi`/`QCPi` returned for every row; our own dual-feasibility residual
  ‖c − Aᵀy − z‖∞ and the dual objective bᵀy computed **by the harness** (not read from `ObjBound`); dual objective ≤
  primal value; gap between them ≤ 1e-6 relative. If duals are not returned, record it (this is the Clarabel
  trigger for Phase 2), do not stop.

## 6. Measurements (per block, both cells)

(a) relative gap (Q_cert − Q_LB)/Q_cert; (b) min(p_ch, p_dch)/S per storage copy and dissipated energy (1 − η_ch η_dch)
× circulated energy, in MWh and €; (c) TN rank residual per branch-period (Wii·Wjj − WijR² − WijI²)/(Wii·Wjj) and its
cost attribution: the gap re-solved with the TN rows fixed to the certified W (TN exact) vs free; (d) DSO rank
residuals (expected exact); (e) OLTC envelope slack; (f) at the unit: Gurobi `QCPi` on the storage capacity rows vs
the Phase A slopes (227.7 k€/MWh, 51.7 k€/MVA) scaled to the block's weight — sign and magnitude; (g) wall time,
variables, cones, peak RSS per conic solve.

## 7. Predictions (recorded here, before the run)

P1 DSO blocks rank-exact to ≤ 1e-5 (P5.5-C precedent). P2 circulation ≤ 0.05 at both cells (dissipation unrewarded
under the current objective). P3 overall block gap ≤ 1e-2 (H1 or H2); the Planner's point estimate is H2 (TN rank
gap on the meshed 9-bus). P4 Gurobi returns duals on blocks of this size (≈ 1/12 of the P5.5-C monolith). P5 the
capacity dual at the unit has the Phase A sign (negative: more capacity lowers Q) and magnitude within a factor 3 of
the scaled slope.

## 8. Outputs and evidence rules

Harness `p516_d0_block_gap.py` (+ `p516_d0_build_relaxation.py`); root `data/SRP1/Results/P516D0/` (new; no cited
artefact touched); per-block JSON with the instance identifier (cell key, year, day), both solver values, the
measurements, the gate verdicts, and a `manifest_sha256.json`; stdout+stderr logs; the reproduction reports.
Report `P5_16_D0_REPORT.md` in the standard order (Decisions needed / Blocked / Changed / Found with each prediction
against its outcome / Not confirmed). Capture-path assertion before the run for every quantity in §6.

## 9. Not permitted

Production code changes; any ADMM run other than R-x0 and R-unit; any solve after the conic phase through the
production paths; the Studio; writing under any existing results root.

## 10. Questions for the Advisor

1. Is the mapped-point gate (§5 G-D0-2) correctly scoped — in particular the tolerance on the coupling rows, and
   whether the objective identity should be checked per block or per group?
2. Is dropping the H3 rows and the ε term's treatment lower-bound-safe exactly as stated?
3. Is (c)'s cost attribution of the TN rank gap (fix TN W to certified vs free) a valid decomposition?
4. Anything in production's coordination reshaping (§4.8) the relaxation must replicate that this list omits?
5. Is the smooth-form IPOPT cross-check worth its cost, or should IPOPT be dropped for D0?
