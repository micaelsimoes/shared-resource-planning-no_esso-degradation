# Revision realignment — analysis, strategies and implementation plan (v2)

Planner, 2026-10-30. Inputs: `manuscript_submitted/main.tex`, `manuscript_review/main_edd7a63_round8.tex`
(the current revision, Overleaf `a87e195`), `manuscript_review/Reviewers Comments.docx`,
`manuscript_review/manuscript_critical_review_2026-10-25.md`, `STEP4_DFO_METHOD.md`,
`PLANNER_BRIEF_2026-09-13.md` (Addenda 1–86), `REVISION_CONTEXT.md`, `TASKS.md`, the production code
(`shared_resources_planning.py`, `shared_energy_storage_data.py`, `convex_oracle.py`) and one Advisor review
(this session). Nothing was run; no file other than this one was changed. This document is a proposal for the
author's and expert's decision — it authorizes nothing.

---

## Decision record (author, 2026-10-30, this session) — v2 supersedes the "Decisions needed" list below

Taken by the author:
- **D1 → (c):** the D0 diagnostic decides between the current Step-7 route and the realignment; the realignment is
  the intended direction.
- **D3:** salvage enters the planning objective F (hence the ESSO objective on the UB side and the maximum-credit
  affine form on the LB side). Formulation change — expert ruling requested on the mechanism; basis under D9.
- **D4:** no VM request now. The Studio (M2 Max, 32 GB) carries the UB re-evaluation campaign and 2–3 full
  5 × 4 × 25 upper bounds; this 24 GB machine carries D0 and the LB side. A VM is requested only if D0 → H1 and
  more paper-scale UBs are needed, with measured hours.
- **D6:** degradation convention for the UB instance: mid-block SoH point and 3-year blocks (2025/28/31/34/37),
  applied with D3 in **one** re-evaluation campaign. Every storage-plan cell in the cache is superseded; x = 0 is not.
- **D7:** lattice kept (0.25 MVA / 0.5 MWh, 2 h ≤ E/P ≤ 4 h, 5 MWh cumulative per node, €1 M expected budget).
- **D8:** the full (node, year) space is searched on both sides: LB master over (P_{n,y}, E_{n,y}) with per-node
  years and staging; UB/MADS on the 30-variable general lattice with the certificate redefined to the coordinate
  unit neighbourhood (expert wording requested).

Requested from the expert (draft Addendum 87, `PLANNER_BRIEF_ADDENDUM87_DRAFT_2026-10-30.md`):
- **D2:** acceptable wording if the bound is valid but non-discriminating (H2).
- **D3 mechanism:** terminal salvage term in the ESSO objective (salvage-aware dispatch) versus a planning-only
  credit; the governing decision "no explicit degradation cost" is affected.
- **D5:** conic solver for the LB side (Gurobi QCP duals failed in P5.5-C; MOSEK academic or Clarabel) and its
  admission to the canonical environment.
- **D9:** salvage basis — purchase-year vs horizon-end expected unit cost; energy-only vs converter; recovery
  fraction value and citation. Fixed before any freeze; identical on both sides.
- **D10:** W184 (E3) and W185 (E2) paused and re-scoped under the new convention (recommended) or run as ordered.
- **D8 wording:** the mesh-local certificate in the coordinate unit neighbourhood; completion cap and N^max.
- **D0 authorization:** the two-task zero-ADMM-solve diagnostic of §7.1, its gates and recorded predictions.

## Next steps (order)

1. Author adopts Addendum 87 (or returns it amended) and forwards the expert items; author decides D5's licence.
2. Planner → Advisor: review of the D0 spec and of the convention-change spec (D3 + D6 + D8 closed-form checks)
   before any Worker task.
3. Worker: **T0.1/T0.2 (D0)** on this machine — zero ADMM solves; decision point per §7.1.
   In parallel, Worker: the convention change under zero-solve gates — ESSO salvage term (D3), `'mid'` baseline,
   the 5-year-horizon single-scenario case file, cumulative-capacity admissibility and the general-lattice
   canonicalization (D8); gate: x = 0 bitwise identical on the ESSO + all network blocks (salvage is zero there).
4. After D0 and the gates: spec v2 frozen (configuration hash, horizon, SoH point, salvage basis, lattice,
   certificate rule); Studio re-evaluation campaign (references, ladders, benchmark arms) under one configuration.
5. If D0 → H1/H2: Phase 2 (LB oracle, dual extraction with margin, cache validation, lattice MILP master) on
   this machine; UB at the master's plan + coordinate unit poll on the Studio; then 2–3 full 5 × 4 × 25 UBs.
6. Manuscript Phase 4 with the expert.

---

## Decisions needed

1. **Direction.** (a) Continue the current Step-7 campaign (E1–E6, Addenda 82–86) to submission — the
   honest-reframing paper now in `main_edd7a63_round8.tex`; cost ≈ 50–90 h compute + round 9, weeks.
   (b) Realign: add a **valid lower-bound side** (convex relaxation of the recourse, Benders on it) under the
   existing certified ADMM upper bound, restore staged timing and all 25 scenarios on the lower-bound side,
   evaluate a few paper-scale plans on the VM; cost ≈ 15–20 Worker tasks plus VM time, several weeks.
   (c) Decide between (a) and (b) **after a two-task, zero-ADMM-solve diagnostic (D0, §7.1)** that measures
   whether the relaxation is tight enough to matter. **Recommendation: (c).** D0 is cheap, settles the one
   question (b) stands or falls on, and does not disturb the current order (W183i is zero-solve; W184/W185
   are 1 + 8 evaluations and remain useful under either direction).
2. **Acceptable wording if the bound is valid but non-discriminating** (gap larger than the between-plan
   differences, §6.3 case H2). The paper could then state a true optimality gap in the original form (as the
   submitted 0.50 % did, invalidly) but must say plan selection rests on the mesh-local certificate.
   The expert should rule in advance whether that wording is worth the lower-bound effort.
3. **Salvage convention in the planning objective.** Addendum 58 ruling 3: the investment-year result is
   decided by whether salvage is in F. Any lower bound must use the same convention as the upper bound, so
   this must be fixed before the master is built (currently: excluded, reported net). Author decision.
4. **Paper-scale upper-bound evaluations on the VM** — which plans (x = 0, the 0.25 MVA/1 MWh unit, the
   2× flexibility-price plan, the submitted paper's plan rounded to the lattice), in what order, under the
   persistent-worker evaluator. Author (machine allocation).
5. **Conic solver.** The lower bound needs reliable SOCP duals. Gurobi is installed but returned no dual
   certificate for this model class in P5.5-C (`ObjBound = −inf`); MOSEK (academic licence) or Clarabel
   (pip) are the alternatives. Author action: licence or package in the canonical environment.

## Blocked on the author

- VM access and the per-core speed measurement (one 3×3 cycle timed on the VM) — needed for every
  paper-scale estimate in §7.
- Decisions 3 and 5 above before Phase 2 (§7.2) can start.

---

## 1. The submitted paper and what the reviewers asked

**Claimed (submitted):** a "bi-level" planning model — master: siting, sizing and *staged* investment over five
representative years (2025–2037) under three investment-cost trajectories, with per-trajectory investments
probability-averaged into an "expected" plan; subproblem: ADMM-coordinated TSO/DSO full AC-OPF
(IEEE 9 + 3 × IEEE 33, 4 seasonal days, 5 × 5 scenarios per block) with a shared-ESS agent tracking SoH.
Benders optimality cuts from "sensitivities", feasibility cuts from slack activation; converged in four
iterations to a 0.50 % gap. Headline: coordination with shared ESS cuts operating cost by 18.25 % and RES
curtailment by 92 % and removes voltage violations; all budget invested in 2025 at node 7; SoH reaches 49.6 %;
discretization (1/3/5-year) changes siting and sizing.

**Reviewers (R2, R3 — the substantive ones):** the recourse is nonconvex, so (i) the cuts are not valid lower
bounds, the "sensitivities" are of undefined origin and local duals are not subgradients; (ii) the slack-based
feasibility cuts are not Farkas certificates; (iii) ADMM convergence, residuals, tolerances, penalty rule and
failure handling are unspecified; (iv) averaging mutually exclusive scenario investments is not an
implementable plan; (v) apparent-power throughput as cell degradation is wrong and calendar ageing is
missing; (vi) SOC dynamics, cost annualization and salvage are not shown; (vii) no degradation-neutral
benchmark. R3 offers the exit explicitly: *provide convexity / strong-duality / cut-validity arguments, or
characterize the scheme as a heuristic with local guarantees.* R1 is editorial.

## 2. What the revision did, and what it lost

| Aspect | Submitted | Current (`edd7a63` / `a87e195`) | Status |
|---|---|---|---|
| Outer method | Benders-type cuts, bounds, 0.50 % gap | Granular MADS (OrthoMADS) on a 0.25 MVA / 0.5 MWh lattice; extreme barrier; mesh-local certificate; **no bound, no gap** | Correct replacement, but the bounds narrative is gone |
| Recourse objective | Included a TSO interface flexibility charge | System cost (`gross_operational_cost`), contracted settlement excluded | **Addendum 11: 99.8 % of the old recourse was a transfer payment.** The 18.25 % cannot return in any form |
| Investment plan | Per-scenario investments averaged | One scenario-independent plan | Necessary fix |
| Scenarios in planning | 25 per block in the search | Search on SRP1 (3 years × 4 days × **1** scenario); 3 × 3 only for fixed plans | Lost: multi-scenario planning |
| Staging | Master variables per year (found: all in 2025) | Formulation keeps it; search restricted to one common year | Lost: staging search |
| Instance | 5 years × 4 days × 25 scenarios (80 blocks) | SRP1; paper-scale **never evaluated** (42 min/cycle ⇒ ≈ 78 h per evaluation on the Mac) | Lost: scale |
| Degradation | Apparent-power throughput, no calendar | Active energy, calendar ageing, log-domain SoH, floor, datasheet calibration | Fixed |
| ADMM | Pseudocode | Boyd residual test, 10-cycle certification bar, recovery tiers, bitwise reproducibility | Fixed, but heavy on diagnostics |
| Cost inputs | Energy cost ÷ 5 (error) | Corrected (× 1.25) | Must be disclosed |
| Finding | Storage pays; all budget in 2025 | Coordination alone saves 12.2 % vs a price-taker benchmark; **no evaluated plan recovers its cost** at baseline; unit margin −64.4 k€ aged, ≈ 0 unaged; pays at 1.75–2× flexibility price | A changed conclusion |
| Prose | Planning and physical outcomes | Acceptance criteria, certificates, evaluation ledger (appendix prose ×5) | The editorial drift the author feels |

## 3. Diagnosis — why it drifted

These are the causes, in order of weight; each is a measured fact in the record, not an opinion.

1. **The recourse the original results were measured on was a transfer payment** (Addendum 11). Once the
   objective became system cost, the storage effect shrank to ≈ 3.5 × 10⁻⁴ of Q per MWh and the coordination
   headline fell from 18.25 % (invalid) to 12.2 % (valid, different benchmark). No method restores the old
   numbers; the paper must own this.
2. **Q(x) has a resolution problem, not primarily a convexity problem** (REVISION_CONTEXT ~l.4380):
   σ_Q ≈ 10–18 k€ against lattice steps of 64–127 k€ and a storage signal of ≈ 228 k€/MWh. Finite-difference
   or local-dual cuts were measured as signal/noise ≈ 4 × 10⁻⁵, and the capacity duals disagreed with finite
   differences twice (P5.5-C; Phase A). That is why cuts were abandoned, and it is correct.
3. **Paper scale was never affordable on the Mac** (18–25 GiB, 42 min/cycle). The whole search programme
   moved to SRP1 by necessity, and the manuscript inherited the narrowing.
4. **Q is affine in x on SRP1** (227.7 k€/MWh, 51.7 k€/MVA, additive across nodes). The optimum is a corner
   of the admissible set — so the MADS machinery, correct as it is, finds nothing a ladder would not, and the
   text spends its weight on certificates rather than on the planning problem.
5. **Defensive writing.** Having withdrawn invalid claims, the revision replaced them with acceptance
   machinery (certification, bars, ledgers) rather than with a new positive mathematical object. The
   critical review of 25 Oct says the same.

## 4. What can and cannot come back

**Cannot (and should not):** the 18.25 % / 92 % headline (transfer payment); per-scenario investment tables
(non-implementable); local-dual Benders cuts as valid bounds; a storage-pays-at-baseline conclusion (Addendum 31:
at paper scale the *level* of the mean price profile, not its dispersion, sets the value; the measured ratio
R ≈ 0.94 says paper scale is slightly *worse* than SRP1 for storage).

**Can, in a valid form:**
- **Bounds and a true optimality gap** — from a convex relaxation of the recourse (§6).
- **All 25 scenarios and staged years in the planning problem** — on the lower-bound side, where they are
  cheap (§6.2), with the upper bound evaluated at the proposed plan.
- **The paper-scale instance** — a handful of fixed-plan upper-bound evaluations on the VM with within-cycle
  block parallelism (Addendum 18 design).
- **Operational-benefit results** (voltage, curtailment, flexibility) — from the certified x = 0 / plan /
  benchmark cells already persisted; these were dropped from the text, not from the record.
- **A certified negative result** — if the relaxation is tight: F(0) − LB bounds the net benefit of *any*
  lattice plan, which turns "no evaluated plan pays" into a theorem-grade statement.

## 5. Strategies considered

| | Strategy | Assessment |
|---|---|---|
| S1 | Keep MADS; restore breadth by compute (general lattice, multi-scenario search, paper scale on the VM) | A paper-scale *search* is unreachable under any strategy (19–36 evaluations per poll at ≈ 8–15 h each even on the VM). A multi-scenario search mostly re-finds the mean-profile result (Addendum 31). Staging search is decided by the salvage convention (Addendum 58). **Keep only its paper-scale fixed-plan evaluations.** |
| S2 | Surrogate-assisted DFO; "Benders-type" local cuts as search heuristic (STEP4 §5.6) | Local-dual cuts rest on a signal measured twice as unusable. A regression surrogate is harmless but decorative on an affine Q. **Worth doing only with the convex LB as the surrogate** (then it is S3's by-product). |
| S3 | Valid bounds via a convex relaxation of the recourse; Benders on the relaxation; certified ADMM as the upper bound | Mathematically valid under stated conditions (§6.1); restores bounds/gap, multi-scenario and staged planning on the LB side; cheap at paper scale. **Risk: the gap may exceed the storage stakes** (§6.3). Decided by D0. |
| S4 | Convexify the evaluation too | Reports relaxed operating points (simultaneous charge/discharge, rank-gap voltages) — the class of flaw the reviewers caught, moved elsewhere. **Rejected.** |
| S5 | Honest reframing only (current manuscript) | Defensible, limited; **the fallback at every stage below**, which is its value. |

**Recommended: S3 + S1-restricted + S2-as-LB-surrogate, built so that S5 remains the fallback at each stage.**
The Advisor's independent review reached the same ranking and supplied the validity conditions and the
P5.5-C caveats below.

## 6. The lower-bound side — what it is and what it licenses

### 6.1 Construction and validity

Let x ∈ X be a lattice plan (sizes, durations, budget, staged years), I(x) its closed-form investment cost,
Q_AC(x) the global optimum of the coupled nonconvex recourse, and Q_cert(x) the objective of the certified
ADMM point (a feasible point up to the stated consensus tolerances). Define Q_LB(x) as the optimal value of
the convex program obtained from the *production* coordinated recourse by:

1. rank relaxation of the W-space voltage products to rotated second-order cones (the `convex_oracle.py`
   design, P5.5-B2), transformed continuous OLTC (measured exact to ≈ 5 × 10⁻⁶);
2. **SoH ≡ 1** — delete the `exp` chain, the floor row and the bilinear `available_e = rated × soh` rows, so
   available capacity equals rated capacity, linear in x through the cohort window. Valid *because* the
   governing decision puts no degradation cost in the objective and salvage is outside F: degradation only
   tightens constraints, so removing it enlarges the feasible set;
3. drop the H3 pro-rata cohort rows (coefficients are ratios of investments — non-affine in x; with
   SoH ≡ 1 cohorts are interchangeable, so nothing is lost);
4. drop or keep the ε-throughput regularisation (nonnegative, linear);
5. exact coupling equalities in place of consensus;
6. the **identical** objective convention: `gross_operational_cost`, contracted settlement and voltage pin
   excluded, scenario-deviation costs and slack penalties included.

Then x enters only through affine rows and the jointly convex cone ‖(p, q)‖ ≤ S, so Q_LB is convex in x and
Q_LB(x) ≤ Q_AC(x) ≤ Q_cert(x) for every x ∈ X. Slacked rows give Slater; SOCP strong duality holds; a
**dual-feasible** conic solution gives an affine function L_k(x) ≤ Q_LB(x) for all x by weak duality
(anchor the cut at the dual objective with a dual-infeasibility margin — never at the primal value; P5.5-C's
`ObjBound = −inf` is the failure mode). The master min_{x∈X} I(x) + α, α ≥ L_k(x), is a small lattice MILP
with staged years and multi-cut over scenarios; it converges finitely and its value is a valid

  LB ≤ min_{x∈X} [I(x) + Q_AC(x)] ≤ UB := I(x̂) + Q_cert(x̂).

### 6.2 Cost of the LB side

With SoH ≡ 1 the inter-year coupling disappears: the LB separates into one SOCP per (year, day) block carrying
its 25 scenarios through the first-stage (day-ahead) variables — the Addendum 42 structure. A paper-scale block
is ≈ 2× the P5.5-C SRP1 parent (322 k variables, 35 k cones): minutes in a conic solver; 20 blocks on 24 cores
≈ 10–30 min per LB evaluation of all 500 scenario-OPFs. Multi-scenario planning and staged years therefore
cost almost nothing on this side. This is what S1 cannot deliver and the strongest argument for S3.

### 6.3 What the paper could claim — three cases, settled by D0

- **H1 — relaxation tight to ≈ 10⁻⁴ of Q.** The gap discriminates between plans; F(0) − LB bounds the value
  of *any* lattice plan; the negative baseline result becomes certified; the 2× flexibility-price case gets a
  certified positive gap. Strongest answer to R2/R3.
- **H2 — tight to 10⁻³–10⁻².** A valid LB and gap in the original form, but the gap (≥ 650 k€ at 0.1 %)
  exceeds the between-plan differences (the whole lattice moves F by ≲ 0.5 % of Q). Plan selection still
  rests on the MADS certificate and the measured affinity; the text must say so. Decision 2.
- **H3 — storage circulation or TN rank gap material (> 10⁻²).** Convex cuts are useless for sizing; the only
  rigorous object is a one-shot MISOCP bound at the final candidates (valid but not convex in x). Fall back to
  S5 + paper-scale fixed-plan evaluations.

Why H1 is plausible now and was not in P5.5-C: that 21.9 % gap was measured against the pre-Addendum-11
recourse (transfer payment, apparent-power ESSO, old complementarity), with circulation min(p_ch, p_dch)/S ≈ 0.5
as the dominant looseness. Under the current objective curtailable RES has zero marginal value, so dissipation
is unrewarded and circulation should be inert; the DSO blocks were already essentially exact. This is a
prediction to record, not a fact.

### 6.4 ADMM convergence — the minimal defensible statement (R2.5, R2.6)

The production scheme is a three-agent-type Gauss–Seidel consensus ADMM with nonconvex *constraints* in every
block; Hong–Luo–Razaviyayn (2016) and Wang–Yin–Zeng (2019) do not cover it as stated, and citing them as a
guarantee would be wrong. State: (i) no global guarantee is claimed; (ii) termination is the Boyd (2011 §3.3.1)
absolute + relative residual test on every channel held for ten consecutive cycles, with terminal-step-to-
threshold ratios reported; (iii) at a fixed point with each block at a KKT point of its augmented subproblem,
the block points and scaled duals satisfy the KKT conditions of the coupled problem (first-order conditions
are additive, proximal terms vanish) — so a certified point is an approximate first-order stationary,
AC-feasible-within-tolerance point: exactly the *local* characterisation R3 offered; (iv) the empirical record
(cycles to certification, 1 local failure in 1,095 solves, the degenerate-dual dead zone as a limitation).
The LB side needs no ADMM; a remark that the convex LB is distributable under standard two-block guarantees
suffices.

## 7. Implementation plan

Phases are ordered so that S5 (the current manuscript) is the fallback at every point. Each Worker task gets
one gate, a recorded prediction, and no production-path change outside its scope. Nothing below is authorized
by this document.

### 7.1 Phase 0 — D0: one-block relaxation gap on the current formulation (2 Worker tasks, zero ADMM solves)

- **T0.1 Build.** From the two certified SRP1 points in the cache (x = 0 and the 0.25 MVA/1 MWh unit), one
  (year, day) block each: W-space relaxation of TSO + 3 DSOs + ESSO(SoH ≡ 1, H3 rows dropped), exact coupling,
  gross objective, in a conic solver returning duals; built from production constructors where they do not
  touch (e, f, r), rewriting only the voltage families (reuse `convex_oracle.py` as a design, not as code —
  it does not build). Gate: the certified ADMM point mapped into the relaxation violates no row by more than
  the consensus residuals and reproduces `gross_operational_cost` to ≈ 10⁻⁸. **Failure of this gate stops
  everything** — it means the relaxation is not of production's problem.
- **T0.2 Measure.** Per block: relative gap (Q_cert − Q_LB)/Q_cert; min(p_ch, p_dch)/S and dissipated energy;
  TN rank residual in cost terms; conic dual infeasibility; the capacity duals at the unit against the Phase A
  slopes scaled to one block (227.7 k€/MWh, 51.7 k€/MVA). Predictions recorded before the run: DSO blocks exact
  to ≤ 10⁻⁵; circulation ≤ 0.05; overall gap in H1 or H2.
- **Decision point** (author + expert): H1 → Phase 2 with the certified-negative claim; H2 → Phase 2 only
  with Decision 2 wording agreed; H3 → Phase 1 + Phase 4 only.

### 7.2 Phase 1 — compute track, independent of D0 (3–5 tasks; starts when the VM is available)

- **T1.1** VM provisioning (author) and one timed 3×3 cycle for per-core speed.
- **T1.2–1.3** Addendum 18 persistent-worker evaluator (within-cycle block parallelism, 24 workers) with its
  bitwise gate against the sequential builder (the earlier attempt projected 5× and delivered 1.1× — the
  Advisor reviews the design before implementation).
- **T1.4–1.5** Paper-scale upper-bound evaluations of the named plans (Decision 4), one attached run per task,
  `persist_certified_models` on, trajectory recorded, the capture checklist asserted before the run. Expected
  ≈ 8–15 h each; 3–6 evaluations. Fallback: the 3×3 numbers in hand.

### 7.3 Phase 2 — the lower-bound oracle (4–6 tasks; after D0 passes and Decisions 3, 5)

- **T2.1** ESSO SoH ≡ 1 block, cohort window linear in x, scenario structure (first-stage variables common
  across the 25 scenarios of a block), exact coupling — the full-instance LB model builder.
- **T2.2** Dual extraction with margin; cut constructor L_k(x) from dual-feasible points.
- **T2.3** Validation gates on the existing cache (≈ 100+ certified points, free): Q_LB(x) ≤ Q_cert(x) on
  every point; mapped-point identity on each; cut validity L_k(x) ≤ Q_LB(x) on held-out x (validate on points
  the cut did not use).
- **T2.4** Lattice MILP master: sizes, durations, budget, staged years; multi-cut over blocks/scenarios;
  generalized-Benders loop with the LB evaluations in parallel per block.
- **T2.5** SRP1 run (LB, UB at the master's plan, gap), then paper-scale LB (all 25 scenarios, 5 years) and the
  UB from Phase 1 at the master's plan. Prediction recorded: the master settles in ≤ 6 iterations (Q affine).

### 7.4 Phase 3 — LB as the MADS search step (1 task)

Replace STEP4 §5.6's local-dual cuts with the convex LB as the surrogate in the search step; stated as a
heuristic inside the convergent frame. Keeps the manuscript's Benders lineage in its honest form.

### 7.5 Phase 4 — manuscript (2–3 tasks, with the expert)

- Method §2: the two-sided structure (relaxed LB oracle with Benders master; certified AC upper bound;
  mesh-local search), the validity statement of §6.1 as a proposition with its conditions, the ADMM paragraph
  of §6.4. Delete the acceptance machinery from the main text; one compact appendix.
- Results reorganised around questions (plan and gap; break-even in cost and flexibility price; degradation;
  coordination value with the benchmark's reverse-flow restriction stated; multi-scenario and staged results
  as LB-side optima with the UB at the plan; paper-scale evaluations; numerical scope). Restore the operational
  figures (voltage, curtailment, flexibility) from persisted cells.
- Response letter: own the three corrections (transfer-payment recourse, cost file, invalid cuts) plainly.

### 7.6 Effort

≈ 15–20 Worker tasks (D0 2; compute 3–5; LB 4–6; surrogate 1; manuscript 2–3) plus VM time; the first decision
point is reached after 2 tasks. At this project's task cadence that is several weeks, not days.

## 8. Claims each outcome licenses

| Outcome | Licensed claim | Not licensed |
|---|---|---|
| D0 gate fails | Nothing new; S5 | Any bound |
| H3 | S5 + paper-scale UB evaluations; optional MISOCP bound at final candidates | Convex cuts, gap narrative |
| H2 | Valid LB/UB/gap; multi-scenario and staged optima of the relaxed problem; "plan within g % of the AC-feasible optimum on the lattice" with g stated and the sentence that g exceeds between-plan differences | Plan discrimination from the gap |
| H1 | All of H2 plus: certified bound on the value of any lattice plan; certified negative result at baseline; certified positive at 2× flexibility price | Anything about the continuous (non-lattice) problem; apportioning the gap |

In every case: the UB point is AC-feasible within the stated consensus tolerances (or polished to exact
feasibility first); the gap aggregates relaxation looseness, the degradation relaxation and UB suboptimality.

## 9. Risks and their early detectors

| Risk | Detector |
|---|---|
| Relaxation not of production's problem (silent invalidity) | T0.1 mapped-point identity gate; T2.3 Q_LB ≤ Q_cert on the whole cache |
| Gap ≫ stakes | T0.2 — before any further investment |
| Circulation rewarded somewhere (high-RES hours at paper scale) | T0.2 circulation ratio; repeat on a high-RES paper-scale block in T2.5 |
| Conic duals unreliable (P5.5-C) | T0.2 dual-infeasibility; cuts anchored at the dual objective with margin |
| Persistent-worker speed-up not realised | T1.2 gate measures the cycle, with the Advisor's design review first |
| Convention mismatch LB vs UB (salvage, settlement) | Decision 3 fixed before T2.1; one shared objective constructor |
| Author-level drift | Every phase stops at its named review point; S5 remains the fallback |

## Not confirmed

- Conic solver availability in `opf_env_py311` (Gurobi binary present; MOSEK/Clarabel not found in the system
  interpreter; the canonical environment was not located from `CLAUDE.local.md`, which is absent).
- VM per-core speed relative to the Mac; the status of the Addendum 18 persistent-worker gate (42(3)).
- Any relaxation-gap measurement on the *current* formulation — the whole point of D0.
- That `convex_oracle.py` pieces are reusable beyond design; it has not built since P5.15-1b.
