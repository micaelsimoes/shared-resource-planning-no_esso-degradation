# Addendum 87 (DRAFT by the Planner, 2026-10-30) — realignment direction; convention change; D0 order; W184/W185 held

**Status: draft for the author's adoption.** The brief is the author's and expert's record; this text becomes
Addendum 87 only when the author appends it (amended as needed) to `PLANNER_BRIEF_2026-09-13.md`. Until then nothing
here authorizes a run. Basis: `REVISION_REALIGNMENT_ANALYSIS_2026-10-30.md` (v2) and the Advisor review it records.

## Author's decisions (taken 2026-10-30)

1. **Direction.** The revision is realigned with the submitted paper's intent under valid mathematics: a convex
   relaxation of the operational recourse supplies a **lower bound** and the Benders master; the certified ADMM
   evaluation of the full AC model supplies the **upper bound** and every reported physical quantity; the MADS
   frame supplies the mesh-local certificate around the master's plan. The relaxation never produces a reported
   operating point. The decision to proceed past D0 (below) is conditional on D0's outcome (H1/H2 proceed; H3 falls
   back to the current manuscript plus paper-scale upper bounds).
2. **Salvage is part of the planning objective F.** Identical on both sides: realised terminal credit on the UB side,
   maximum credit (SoH ≡ 1, affine in x) on the LB side.
3. **Degradation convention for the UB instance:** mid-block SoH point (`available_energy_soh_point = 'mid'`) and
   3-year blocks (2025/2028/2031/2034/2037, calendar life = five blocks). Applied together with decision 2 in one
   re-evaluation campaign. All storage-plan cells in the cache are superseded by it; x = 0 cells are not.
4. **Lattice unchanged:** 0.25 MVA / 0.5 MWh, 2 h ≤ E/P ≤ 4 h per cohort, 5 MWh cumulative rated energy per node and
   year, €1 M expected discounted budget over all years.
5. **Full (node, year) decision space** on both sides: per-node investment years and staged additions. The UB search
   runs on the general lattice z ∈ ℤ^{2|N||Y|} (30 on the paper's year set).
6. **Machines.** No VM request. Studio: the re-evaluation campaign and 2–3 full 5 × 4 × 25 upper bounds (x = 0 and the
   plan; serial, or 3–4 persistent workers if a two-cycle memory smoke passes on 32 GB). The 24 GB machine: D0 and the
   LB side. A VM is requested only if D0 → H1 and further paper-scale UBs are needed, with measured hours.

## Rulings requested from the expert

**Author's agreement recorded 2026-10-30 (this session):** R1 as recommended (option (a), terminal salvage term in the
ESSO objective); R2 as recommended (horizon-end expected energy cost, energy-only, cited recovery fraction); R5 closed
(Gurobi `QCPDual = 1` primary, Clarabel open-source fallback per block, IPOPT cross-check; no licence request, no
environment change before D0). The expert's confirmation of R1 is still requested, since it supersedes a governing
decision. Author authorized D0 and the convention-change zero-solve tasks to proceed.

**Advisor review of the convention-change spec (2026-10-30; `P5_16_CONVENTION_CHANGE_SPEC_v2.md` §0) — facts the
expert must see before confirming R1, and two author items:**
- F2: a 2025 cohort's salvage is identically zero (remaining calendar life 0 at the horizon end), so under R1 (a)
  dispatch is salvage-aware only for 2028+ cohorts; for a 2025-only plan the critical review's §3.3 objection stands.
  **Author/expert:** should a calendar-expired cohort with SoH above the floor carry a health-based residual value?
- F4 (expert wording): the LB may credit salvage and bound available energy at the calendar-only SoH φ_cal^age
  (dispatch-independent, hence valid) instead of SoH ≡ 1 — about 2× tighter on the salvage credit.
- F9 (**author**): "horizon-end cost" is operationally the 2037 cost column; a 2040 value needs data or a rule.
- **Author:** `energy_recovery_fraction` is 1.0 and `recycling_floor_fraction` 0.0 today — cited values needed.

- **R1 — salvage mechanism.** Option (a): terminal salvage term in the ESSO objective, so dispatch internalises the
  marginal salvage loss of cycling (this supersedes the governing decision "no explicit degradation cost"); option (b):
  planning-only credit, dispatch unchanged (the critical review's §3.3 objection then stands). Planner recommends (a).
- **R2 — salvage basis.** Purchase-year expected energy cost (as implemented in
  `shared_energy_storage_data.py::_build_terminal_salvage_value_expression`) vs horizon-end expected cost;
  energy-only vs converter pro-rata; `energy_recovery_fraction` and `recycling_floor_fraction` values with citations.
  Planner recommends horizon-end cost, energy-only, cited fraction. Fixed before any freeze.
- **R3 — certificate wording on the general lattice.** Coordinate unit neighbourhood (±1 in one coordinate, plus the
  activation moves of inactive cohorts) replaces the ∞-norm box completion of Addendum A2; completion cap and N^max
  restated accordingly.
- **R4 — non-discriminating bound.** Whether a valid gap larger than the between-plan differences (H2) is acceptable
  wording, stated with the sentence that plan selection rests on the mesh-local certificate.
- **R5 — conic solver.** Admission of MOSEK (academic) or Clarabel to the canonical environment for the LB side;
  Gurobi's QCP duals are not accepted as certificates (P5.5-C).
- **R6 — W184 (E3) and W185 (E2).** Planner recommends holding both and re-scoping the robustness checks (NRF arm,
  penalty pilot) into the new campaign, so the paper carries numbers from one convention. W183i (zero-solve) may
  complete as ordered; its launcher work is reused.

## D0 order — authorized by the author 2026-10-30; spec `P5_16_D0_SPEC_v1.md` (Advisor review first)

**Correction to the earlier draft:** D0 is **not** zero-solve. No SRP1 certified cell has persisted network models,
so the certified block points are obtained by reproducing the two cells in-process to their certificates, gated
bitwise against the committed reports (`p515_s41_hull_polish.py` precedent), then building the relaxation on the
live models; ≈ 1.5–2 h each on the 24 GB machine, attended. Conic solves follow with the production solve paths
guarded at zero.

- **T0.1 Build.** For the two certified SRP1 cache points x = 0 and 7: 0.25 MVA / 1 MWh (2025), one (year, day) block
  each: W-space SOC relaxation of TSO + 3 DSOs + ESSO with SoH ≡ 1 and the H3 pro-rata rows dropped, exact coupling,
  the `gross_operational_cost` convention (contracted settlement and voltage pin excluded, deviation costs and slack
  penalties included), built from production constructors wherever the rule does not touch (e, f, r); only the
  voltage families are rewritten. **Gate:** the certified ADMM point mapped into the relaxation violates no row by
  more than the recorded consensus residuals and reproduces the block's gross objective to ≈ 1e-8. Failure stops
  the realignment: the relaxation is not of production's problem.
- **T0.2 Measure.** Per block: relative gap (Q_cert − Q_LB)/Q_cert; min(p_ch, p_dch)/S and dissipated energy; TN rank
  residual in cost terms; conic dual infeasibility; capacity duals at the unit against the Phase A slopes
  (227.7 k€/MWh, 51.7 k€/MVA) scaled to one block.
- **Predictions (recorded before the run):** DSO blocks exact to ≤ 1e-5; circulation ≤ 0.05; overall gap ≤ 1e-2
  (H1 or H2). Interpretation: ≤ 1e-4 → H1; 1e-4–1e-2 → H2 (R4 applies); > 1e-2 or circulation > 0.05 → H3.
- Harness `p516_d0_*.py` under `data/SRP1/Results/P516D0/`; report `P5_16_D0_REPORT.md`; Advisor reviews the spec
  before the Worker builds. Output paths new; no cited artefact touched.

## Convention-change tasks (zero solves except the named gate) — authorized on adoption, after R1–R3

- ESSO salvage term per R1/R2; `'mid'` as the baseline SoH point; the single-scenario five-block case file
  (2025–2037); cumulative-capacity admissibility and general-lattice canonicalization per decision 5 and R3.
- **Gate:** x = 0 bitwise identical on the ESSO and every network block (salvage and storage rows inert);
  every preserved fixture unpickles; the unit cell builds under the new convention.
- Then spec v2 freeze (configuration hash, horizon, SoH point, salvage basis, lattice, certificate rule) — the
  re-evaluation campaign is authorized by a later addendum, one gate per task.

## Stops

As `CLAUDE.md`. In addition: D0's T0.1 gate failure; any D0 reading in H3; any salvage basis change after freeze.
