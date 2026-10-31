# Planner Brief — ESSO/SMOPF conditioning programme (authorized 2026-09-13)

Author: Micael Simões. Prepared with the external expert (Cowork session). This brief
**supersedes** the "NUMERICAL PROGRAMME CLOSED" verdict in `REVISION_CONTEXT.md` and the
"Current next proposed action" in `COWORK_HANDOFF.md`. Background and reasoning are in
`EXPERT_REVIEW_2_ACTION_PLAN.md` (read it first; §1 contains the diagnosis this brief acts on).

## Governing decisions (author, final)

- The ESSO agent **stays** in the ADMM loop. Option B (forward recursion) is rejected.
- No explicit cycling/degradation cost is added anywhere. Degradation is tracked and propagated;
  the only intertemporal trade-off enforced in operation is the SoH floor. The manuscript will say so.
- The investment-fixing slacks in the ESSO subproblem are **retired**: investments become fixed
  parameters. The "feasibility cut from slack activation" path in the master is retired with them.
- ESSO charge/discharge complementarity rows are **replaced** by an ε-linear regularization of
  throughput in the ESSO objective. Network complementarity rows are unchanged.
- The SoH chain moves to the log domain (continuous-compounding form); calendar ageing enters
  as `phi_cal` with default `1.0` (neutral until the author sets it).
- Open author decisions that do **not** block Steps 0–2: salvage reinstatement, calibration triple
  (C2 vs C3), outer-method choice. Do not change them.

## Authorization

**Steps 0, 1 and 2 are authorized now**, in that order for Worker execution (Step 2 may run in
parallel as an Advisor read-only task). **Stop and report to the author at the end of each step.**
Steps 3–6 are roadmap only — not authorized.

Process for Steps 0–2: ordinary Worker tasks with a short report each (`P5_15_<step>_REPORT.md`
plus evidence directory `data/SRP1/Results/P515<step>/`). The frozen-specification ceremony is
**not** required for these diagnostics; the `CLAUDE.md` evidence rules still apply to any number
that will be reused (record the instance, the convention, the error bar, the rule-ten ratio).
Solver options are hypotheses, not frozen state: the Worker may vary them freely on preserved
instances. Production adoption of a change still needs the gate stated in each step.

Check first that the Worker is available (it was unavailable during P5.12-G onward). If it is not,
the Planner executes and says so in the report, as before.

---

## Step 0 — ESSO warm-start policy A/B on the preserved failing instance

**Hypothesis (H-WS).** The `maxIterations` failures of the ESSO subproblem are caused by the
warm-start policy in `shared_energy_storage_data.py:_create_solver` — imported bound multipliers
with `warm_start_mult_bound_push = 1e-9` (and 1e-9 for the other four pushes) — which throttles
the dual step by the fraction-to-boundary rule. Evidence: in `optim_log_node_7.txt` (P5.14-N
perturbation arm) every failing solve starts primal-feasible, `lg(mu)` freezes at −3.8 after
iteration 1, `alpha_du` ≈ 2e-8–5e-7 for 3,000 iterations, dual infeasibility grows 7e-3 → 1.8e3.
The last node-7 solve of that arm is one of the failures, so the serialized model is a genuine
failing instance.

**Instance.** `data/SRP1/Results/P514N/esso_models_k10000.pkl`, node 7 (also nodes 5 and 9 as
controls — they converged). The pickled model still holds `p_req`, `q_req`, `dual_p_req`,
`dual_q_req`, `rho`, and the `ipopt_zL_out/zU_out` suffixes of the last solve. Do not rebuild the
model; load it and re-solve through the production path (`_run_solver_attempt`) with
`option_overrides`. Verify before solving that the loaded model's `admm_objective` is active and
the request parameters are non-trivial; record their hash.

**Arms** (each an independent solve from the loaded state; only options differ):

| arm | settings |
|---|---|
| A0 | current production policy (reproduction gate: must hit `maxIterations`) |
| A1 | warm start, `warm_start_mult_bound_push = 1e-3`, `warm_start_bound_push = 1e-3`, `warm_start_slack_bound_push = 1e-3`, `_frac` options at IPOPT defaults |
| A2 | warm start primal only: do not export `ipopt_zL_in/zU_in` (clear the suffixes), pushes as A1 |
| A3 | A1 + `mu_strategy = adaptive` |
| A4 | cold start (`warm_start_init_point = no`, no suffix export) |
| A5 | A0 + `max_iter = 500` (cost-of-failure check only) |

**Evidence per arm.** Exit status, iterations, wall time, objective (`admm_objective`), max
constraint violation, and the full iteration table saved. Success criterion for a candidate
policy: converged in < 200 iterations with objective equal to A4's to 1e-6 relative and
constraint violation < 1e-6. If A0 does not reproduce `maxIterations`, stop and report (the
instance is not what we think it is).

**Then repeat** A0–A4 on the preserved cycle-21 DSO fixture using the P5.12-P harness
(`p512_p_path_sensitivity.py`) with the network policy (`network.py:521-534`, 1e-6 pushes),
overriding to IPOPT defaults / primal-only / adaptive / cold. Same evidence.

**Permitted changes:** none to production in this step. Harness only (`p515_0_esso_warmstart_ab.py`).

**Report must state:** which arms converged, whether H-WS is supported or falsified separately
for the ESSO and the DSO fixture, and a recommended production policy for each solver family
(one option block), plus the recommendation to (a) set `max_iter` for the ESSO (500) and networks,
and (b) extend `_is_recoverable_shared_ess_failure` to fire on `maxIterations` and
`infeasible` with a cold retry. Production adoption of the policy happens in Step 1's gate run,
not here.

---

## Step 1 — ESSO subproblem reformulation

**Objective.** Remove the avoidable nonlinearity and mis-scaling from `_build_subproblem`
(`shared_energy_storage_data.py:346-675`) so that the ESSO is an LP plus one `exp` per
cohort-year and the converter circle.

**Changes (all in `shared_energy_storage_data.py` unless noted):**

1. **Investments as parameters.** `es_s_investment`, `es_e_investment` become mutable `Param`
   (or the existing `_fixed` params are used directly); delete the four investment slacks and
   their penalty terms. `es_s_rated_per_unit`, `es_e_rated_per_unit`, `es_s_rated`, `es_e_rated`
   then become parameters (or expressions) — the Worker chooses whichever keeps
   `_configure_esso_cohort_state`, `get_updated_capacities`, `_get_terminal_salvage_value_*`,
   `_map_available_capacity_sensitivities_to_investments` and `update_model_with_candidate_solution`
   working. Consequences to handle: `pch ≤ s_max`, the normalization rows and the floor become
   linear; the cohort-deactivation logic can stay as is.
2. **Log-domain SoH chain.** Per cohort-year define `D[y_inv,y]` (annual fractional loss,
   `NonNegativeReals`, initialize 0) with the linear row
   `D[y_inv,y] * (2 * cl_eff * E_rated_per_unit[y_inv,y]) == 365 * num_years * es_avg_ch_dch_per_unit[y_inv,y]`
   (E_rated is now a constant), and
   `es_soh_per_unit_cumul[y_inv,y] == prev * exp(-D[y_inv,y]) * phi_cal ** num_years`
   with `phi_cal` read from `ageing.calendar_retention_per_year` (default 1.0, neutral).
   Delete `es_soh_per_unit`, `es_degradation_per_unit`, `es_degradation_per_unit_cumul` and
   their slack families (keep the two `soh_cumul` slacks only if the Worker finds a caller that
   needs them; otherwise delete). Keep `num_years` semantics exactly as today (from the
   investment-year block — note the Expert's remark that this is only correct for equal-width
   blocks; record, do not change). Floor: `soh_cumul >= soh_min` (linear).
3. **Complementarity by regularization.** Delete `energy_storage_complementarity`, the
   `*_hat_*` variables, the normalization rows, `slack_es_ch_comp_per_unit`, and the aggregate
   complementarity row. Add to `feasibility_penalty` the term
   `EPS_ESSO_THROUGHPUT * Σ (pch + pdch)` over active cohort-periods, with
   `EPS_ESSO_THROUGHPUT` in `definitions.py`, initial value `1e-3` (units: same as the AL
   terms, which are normalized by `2*rating`; the Worker must check the resulting gradient scale
   against the AL gradient at the P5.14-N control state and report it). Keep `pch, pdch ≥ 0`,
   `pch ≤ s_max`, `pdch ≤ s_max`, and the converter circle. Document in the code that
   complementarity now follows from LP structure, not from a constraint, and add a
   post-solve check that `min(pch, pdch) ≤ 1e-6 * s_max` for every active cohort-period
   (report the maximum violation; it is a detector, not a cost).
4. **Solver policy.** Adopt the Step 0 recommendation for the ESSO; set `max_iter`; extend the
   recovery path as recommended.
5. **Master side.** Retire the slack-based "recovery/feasibility cut" branch in
   `shared_resources_planning.py` (`add_benders_cut` and callers) — deactivate, do not delete,
   with a comment pointing at this brief. Verify nothing else reads the deleted ESSO variables
   (`grep` for every deleted name across the repository, including `p5*` harnesses that remain
   in use; historical harnesses may break and that is acceptable, list them).

**Not permitted:** changes to network models, ADMM driver logic, tolerances, rho, or the
`(N, D, R)` calibration; adding any degradation cost; changing throughput definition (Option A
stays).

**Gate (must pass before commit):**

- G1 — Re-run the P5.14-N control arm (C\*, `k = 11,541.56`, cold, `rel = 1e-4`, `rho_pf` 300
  adaptive) with the new ESSO. Required: per-cohort SoH trajectory within 1e-6 absolute of the
  control (`1.0 → 0.8387 → 0.7284 → 0.6248`); EFC/day within 1e-4; recourse within the rule-nine
  error bar of `816,121,464.16` (bar ≈ 164,000) — **report the difference and the bar; do not
  demand equality**, the `exp` form and the removed slacks make bit-identity impossible. Record
  cycles, solves, rule-ten ratio, ESSO iteration counts (mean/max) and zero local failures.
- G2 — Re-run the P5.14-N perturbation arm (`k = 10,000`). Required: converges within the cap,
  zero `maxIterations` exits at any node. Report recourse and the pair difference against G1
  with its bar (this also answers the P5.14-N objective comparison left inconclusive).
- G3 — Capacity ladder (P5.14-L harness) extended to 1.00, 1.25 and 1.62 MVA / 3.24 MWh at
  node 7 (the paper's plan), initialization stage: all must PASS. Then one full cold evaluation
  at 1.62 MVA / 3.24 MWh (node 7 only, other nodes zero) — converged, zero failures.
- G4 — Determinism: G1 run twice, identical.

**Report must state:** per-change diff summary, the iteration-count distribution before/after,
G1–G4 results with error bars and rule-ten ratios, the complementarity detector maximum, and any
harness broken by the deletions.

---

## Step 2 — SMOPF constraint-family conditioning audit (Advisor, read-only; may run in parallel)

**Objective.** A formulation-level audit of one DSO model (`case33_2`, node 7, the fixture that
failed at cycle 21) and the TSO model (`case9`), replacing solver-telemetry forensics with a
constraint-family inventory. No solves.

For every constraint family in `model_construction_helpers.py` / `network.py`: nonlinearity type
(linear / bilinear / quadratic / trig / product-zero-gradient), variables involved with their
bounds and typical magnitude at the preserved converged solution (P5.12 fixtures), whether the
row has zero gradient at that solution, its penalty coefficient (from `definitions.py`) against
the objective scale, and whether it is one of the P5.3 HIGH-risk families (`sess_comp`, RES
`sg_capability`). Also inventory the warm-start policy and every IPOPT option in force.

**Deliverable.** `P5_15_2_SMOPF_CONDITIONING_AUDIT.md` with a ranked shortlist of at most five
reformulation candidates, each with: the row(s), the proposed replacement, what it changes
mathematically, the expected conditioning effect, and the cheapest test. **No implementation.**
The Planner reviews and brings the shortlist to the author.

---

## Roadmap — NOT authorized (for orientation only)

- Step 3 — ADMM stopping rule: Boyd-style absolute+relative primal/dual residual test on the
  original consensus problem; objective-change test auxiliary; rule-ten ratio reported per
  evaluation; one recorded configuration baseline (Track B decision).
- Step 4 — Outer layer: derivative-free master search over `(S, h = E − φ_min S)` with the cold
  oracle; no cuts, no gap claim. Pending the author's outer-method decision.
- Step 5 — Reviewer campaign at full configuration (5 years, 4 days, 25 scenarios via the
  scenario-summing production path): uncoordinated (after fixing the
  `_add_dso_scenario_deviation_penalty` guard), coordinated without storage, coordinated with the
  optimized plan; R3.6 matrix; salvage; discount rate (fixed-plan first); `soh_min` cases.
- Step 6 — Manuscript reconciliation (Track E list + `EXPERT_REVIEW_2_ACTION_PLAN.md` §2).

---

# Addendum 1 — after Steps 0 and 2 (2026-09-13, author + external expert)

## Corrections to this brief

- **Fixture identification.** The cycle-21 failing fixture is **`case33_3`, node 9, 2025 Spring**
  (`data/SRP1/Results/P512R/cycle21_pre_setup/snapshot.pkl`). `case33_2` node 7, 2025 Autumn,
  cycle 7 is the *converged* comparator. Step 0 used the correct fixture; the text of Step 0 and
  Step 2 above is wrong and is corrected by this note.
- **ESSO instances.** Nodes 7 **and 9** of `esso_models_k10000.pkl` are failing instances; only
  node 5 is a control.

## Step 0 verdict and the production policy it licenses

H-WS is **supported** on both solver families (two ESSO instances, one DSO fixture; A0
reproduces `maxIterations` on all three; A1/A2/A4 converge to the same point). Adopted policy,
to be implemented in Step 1a and validated by Step 1's gate runs:

- **Primary:** warm start with imported bound multipliers; all five `warm_start_*` pushes/fracs
  at IPOPT defaults (1e-3 / 1e-3 / 1e-3 / 1e-3 / 1e-3), for the ESSO **and** both network
  families. `warm_start_mult_bound_push` is its own option — the derivation from `bound_push`
  at `network.py:534` is removed.
- **`max_iter = 500`** for the ESSO and the networks.
- **Recovery** fires on `maxIterations`, `infeasible` and `internalSolverError`, and is **one
  change**: cold start (`warm_start_init_point = no`, suffixes cleared, including `model.dual`)
  with the primary options and the exact Hessian. The limited-memory Hessian is dropped from
  `recovery_options` so a recovery success identifies its cause.
- **`mu_strategy = adaptive` is not used**, as primary or recovery. A3 found a *different* local
  optimum on the ESSO (lower objective, −0.01279 vs −0.01198). This is evidence of multimodality
  in the un-reformulated ESSO and becomes gate G5 below.
- The TSO's warm-start override of `acceptable_iter = 0` / `acceptable_tol = tol` is removed
  (undocumented asymmetry with the DSO); both families use the case-file values.

## Step 1 — amendments

**Step 1a (first commit, own gate):** fix the `option_overrides` clobbering in
`shared_energy_storage_data.py:_create_solver` (apply the warm-start block *before* the override
merge, or make it `.get(key, default)` like `network.py`), remove the `bound_push` derivation in
`network.py:534`, implement the policy above in both `_create_solver` functions and both
`_is_recoverable_*` functions. Gate: the Step 0 `clobber_probe` shows overrides respected; A0
re-run with no override still reproduces `maxIterations` on node 7 (i.e. the reproduction path is
intact); A1 re-run through `option_overrides` alone converges in ~30 iterations.

**Pre-Step-1 zero-solve check:** read a preserved P5.12-R `.nl` and state whether constraint
multipliers (`dual` suffix) are exported. If yes, clear `model.dual` on cold solves (part of 1a)
and re-run Step 0's A4 once on node 7; if the objective is unchanged to 1e-6, A4 stands.

**Gate G5 (new):** on the reformulated ESSO, re-solve the node-7 and node-9 instances under A1,
A3 and A4. All three must agree to 1e-6 relative. Disagreement means the reformulation left a
nonconvexity and Step 1 does not close until it is identified.

**Gate reporting:** count `maxIterations` / recovery events across G1–G4 for both families; the
policy is adopted if the count is zero at C\* and at 1.62 MVA.

## Step 2 — candidate decisions

| candidate | decision |
|---|---|
| 1 — shared-ESS power-factor rows tying `\|q\|` to `pch + pdch` | **Accepted, form (a)**: delete both rows; the converter circle carries the reactive limit. Removes the LICQ-degenerate idle vertex and the ESSO/network reactive-set asymmetry. Manuscript sentence on power-factor limits to be removed (Track E). Implement in **Step 1b** (network side), same gate runs. |
| 2 — network capacity `Var`s → `Param`s | **Accepted — unblocked** (author confirmed the derivative-free outer method, 2026-09-13; the Benders sensitivity channel is retired). Implement in Step 1b. |
| 3 — RES converter capability separated from availability | **WITHDRAWN as a feasible-set change** (author, 2026-09-13): RES apparent capability stays equal to the available active power; no reactive support when idle (`Pmax`/`Qmax` are MATPOWER-style box limits, not a nameplate). To be stated in Section 3 of the manuscript. **Replaced by Candidate 3′ — conditioning only, feasible set unchanged:** (a) normalize `sg_capability` per period, `(pg/pg_avail)² + (qg/pg_avail)² ≤ 1` for `pg_avail > ε`; (b) for periods with `pg_avail ≤ ε` build no row and fix `pg = qg = 0` by bounds. Gate: identical dispatch to 1e-8 on a converged fixture; report cold-start iteration count before/after. Step 1b, own commit. |
| 4 — flexibility day-balance band → slacked equality | **Accepted** (`slacks.flexibility.day_balance = true`, penalty as configured). Step 1b. |
| 5 — guard scenario-deviation penalties off at one scenario | **Accepted** (provable no-op). Step 1b. |

Step 1b is a **separate commit** from the ESSO reformulation (Step 1 items 1–5), each with its
own diff summary, so that a gate regression can be attributed. Order: 1a → ESSO reformulation
→ 1b → gates G1–G5.

## Step 3 agenda (still NOT authorized; recorded so it is not lost)

The audit's two dominant findings belong here and outrank the stopping rule:
(1) the DSO objective is divided by `effective_scale ≈ 1.05e5` while the AL terms are undivided
— P5.7 measured this as 96.5 % of the ADMM-to-polish gap; the objective and the AL must sit on
one scale; (2) the shared-ESS AL normalization `2·max(S, 0.10)` gives `1/S²` curvature
(952 → 2.5e5 over the campaign range) — normalize by a fixed per-node reference rating instead.
Then the Boyd-style residual tests and one recorded configuration baseline (Track B).

---

# Addendum 2 — after Steps 1a / 1 / 1b Set 1 (2026-09-13, author + external expert)

- **Candidate 3′ is DROPPED.** Same feasible set, different local optimum (0.3 % objective,
  1.95e-5 dispatch), cold-start iterations 39 → 65. The note at `sg_avail_rule` stands. The
  finding to carry is the multimodality itself: **the DSO SMOPF has local optima ~0.3 % apart
  in objective at material capacity** (consistent with A3's 1e-5 on the cycle-21 fixture and
  Track C1's 1.03 % templated-vs-cold bias). Record in `REVISION_CONTEXT.md` as a standing
  fact: `Q(x)` is defined up to the local optimum the deterministic path selects.
- **`_add_benders_cut` disabled entirely: confirmed** (guarded deactivation, comment → brief).
- **New `CLAUDE.md` rule:** deactivate and unwire; never delete a callable that a preserved
  fixture may resolve at unpickling. Candidate 1's "delete both rows" wording is corrected to
  "unwire both rows".
- **Erratum accepted:** the DSO production warm-start pushes resolve to **1e-5** (case-file
  `bound_push`), the TSO's to 1e-6.
- **ε check (before the gates):** two arms, `EPS_ESSO_THROUGHPUT` 1e-3 vs 1e-7, on the
  reformulated node-7 instance. Report (i) `pnet` displacement (predict < 1e-6 p.u.),
  (ii) complementarity detector `max min(pch, pdch)/s_max` in both arms (predict ≤ 1e-6 at
  1e-3; drift at 1e-7 argues *for* 1e-3), (iii) ESSO iterations. Gate G1–G5 on **1e-3**. The
  displacement framing (ε / AL curvature) is the accepted comparator; gradient ratio is not.
- The preference for `slack_es_pnet_down` over `pdch` near the SoH floor is the floor acting
  through its only channel; it lands in `gross_operational_cost` and is an instance of the
  penalty-classification item (Track E §D), not a non-uniqueness of the split.
- `convex_oracle.py`: mark historical (P5.5-C), leave unrepaired. Dead `limited-memory`
  entries in four case files: author removes. Harnesses: repair only
  `p514_n_instrumented_cstar.py` and `p514_l_capacity_ladder.py`; list the other seven as
  broken-historical.
- **Dispatch order:** harness repair → ε check → G1–G5 → `REVISION_CONTEXT.md` rewrite → stop.

---

# Addendum 3 — after the gate hold (`P5_15_EXPERT_HANDOFF.md`) (2026-09-13)

The barrier identity `min(pch, pdch) = μ_final / (2·s_obj·ε)` is accepted as measured. The
conclusions drawn from it are amended:

- The leak is proportional to `μ_final`, which the ESSO's `tol` controls. §6 of the handoff
  omitted this lever. At `tol = 1e-8` the leak is ~5e-6 (≈0.01 % of throughput on the fixture,
  ≈0.004 % at C\*), below the recourse resolution and two orders below the local-optimum spread.
- The identity gives a **closed-form bound** on spurious throughput,
  `2·N_periods·μ_final/(2·s_obj·ε)` — the first such bound any formulation here has had. It is
  reported on every ESSO solve and stated in the manuscript.
- Remedy (d)/(e2) (single signed power with `√(p²+δ²)`) is **NOT authorized**: it contributes
  spurious throughput `≈ δ·(η_ch + 1/η_dch)/2` at every idle period and curvature `1/δ` there;
  it changes the owner of the artifact, not its class. Kept as fallback if the detector at
  `tol = 1e-8` exceeds 1e-4 relative throughput at C\*.

**Authorized:**

1. **Remedy (h):** ESSO `tol = 1e-8`, `acceptable_tol = 1e-7` (case file, author applies or
   Worker via `option_overrides` now that overrides are honoured). Detector
   `max min(pch, pdch)/s_max` and the analytic bound logged per ESSO solve.
2. **H3 rule (formulation change, authorized):** per active cohort,
   `pnet_cohort[y_inv, y, d, p] = (E_rated[y_inv, y] / Σ E_rated[·, y]) · pnet_agg[y, d, p]` —
   linear, parameters only (homogeneous-fleet approximation; to be stated in the manuscript,
   answering Expert §2.4 / R3.3 cohort realizability).
3. **G1 re-specified as a reconciliation gate, per node** (5, 7, 9 separately): the measured
   Δ(SoH_y) between the old control (`esso_models_control.pkl`) and the reformulated run must
   equal the Δ predicted from the two leak fractions (old 1.386–1.391 %, new = the run's own
   detector) through `D ∝ throughput`, `SoH = Π exp(−D)`, to within 10 % of that Δ. Same for
   EFC/day. Recourse within the rule-nine bar. The old trajectory is a known-biased comparator,
   not a target.
4. Amend the comment at `shared_energy_storage_data.py:400-412`: the LP statement stands; the
   inference does not; state the barrier identity and the bound.
5. Nine broken-historical harnesses, none repaired: accepted (count corrected from seven).
6. Then G1–G5 (G1 as above) → `REVISION_CONTEXT.md` rewrite, which must also record that every
   previously reported SoH trajectory carried ≈1.4 % spurious throughput from the relaxed
   complementarity row, so no published degradation number is reusable.

---

# Addendum 4 — after `P5_15_EXPERT_HANDOFF_2.md` (2026-09-13)

- Remedy (h) **confirmed**: μ_final ↓101.5×, leak ↓102.1× — the identity is predictive. Spurious
  throughput 0.0091 % at `tol = 1e-8`. ESSO closed as a component pending G1–G4.
- **"Estimate", not "bound"**: the identity holds exactly only at the terminal barrier optimum;
  measured/estimate scatters ±0.35 %. Manuscript wording: *estimate*; gates use the measured
  detector. The Planner's override of Addendum 3's wording is accepted.
- **G5 re-specified** (the 1e-6 absolute criterion was mis-set, same class as the detector
  threshold): (i) net power `pch − pdch` across A1/A3/A4 to 1e-10 — **passed** (1e-16);
  (ii) `es_D_per_unit` and `es_soh_per_unit_cumul` across arms within the sum of the two arms'
  analytic leak estimates propagated through `D ∝ throughput` (in practice ≤ 1e-4 absolute on
  SoH). On the reported numbers (7.6e-5 on D, 2.2e-5 on SoH) **G5 passes**; record it as passed
  with the re-specified criterion and the decomposition table as the evidence that no
  nonconvexity remains.
- **G1 stands as specified.** The reconciliation Δ is dominated by the old control's 1.39 %
  leak, not by the new run's 0.009 %; Δ(SoH) ≈ 1.4e-3 against a 10 % criterion is well resolved.
- The ESSO detector and the per-solve estimate are logged in every G run, per cycle.
- Commit this round explicitly by name. Then G1–G4 → `REVISION_CONTEXT.md` rewrite → stop.

---

# Addendum 5 — after `P5_15_EXPERT_HANDOFF_3.md` (2026-09-14)

- **G3-init passes** at 1.00 / 1.25 / 1.62 MVA, all nodes, zero failures. Credit recorded to
  Candidate 2.
- **ESSO log-handling fix authorized as one change**, four parts: (1) ESSO logs go to the same
  logs directory `network.py` uses; (2) **one log file per ESSO solve**, stamped cycle/node —
  the per-cycle detector trajectory is then a directory listing, not a parse; (3) the parser
  reads the *last* `Complementarity`/`Objective` lines; (4) the network failure handler resolves
  `results_dir` to an absolute path at construction and saves the failure snapshot without
  aborting. Tests: two-cycle probe with distinct `mu_final` per cycle; a deliberately triggered
  network failure that snapshots and continues; single-solve detector values unchanged from the
  committed results.
- **No harness-only workaround** for gate reporting. These logs are the paper's reporting path.
- **Zero-solve check before G1:** from the ladder logs compute `mu_final/(2·s_obj·ε)` at the
  initialization call site. If it reproduces ~2.4e-5, the 5× gap is an `s_obj` difference at
  initialization (consistent with the identity; record it). If not, `tol = 1e-8` is not in force
  on that path — fix the plumbing in the same commit.
- **The lost TSO failure at C\*** is the first network failure under the new policy; on the G1
  re-run it must be captured, counted and classified (exit status, iterations, recovery outcome).
- **`CLAUDE.md` rules adopted:** campaign harnesses capture stderr, refuse to run concurrently,
  and are never detached (`screen`/`nohup`/background); one gate per Worker task with the exact
  command.
- Order: stage this round by name (lock file stays untracked) → fix + tests → zero-solve check →
  G1 → G2 → G3-full → G4, sequentially → `REVISION_CONTEXT.md` rewrite → stop.

---

# Addendum 6 — after `P5_15_EXPERT_HANDOFF_4.md` (2026-09-14)

- Log fix and its three tests: **accepted.** The 51-solve deviation for the ladder re-run is
  accepted as declared.
- **Campaign launch:** one background run started through the tool itself, stderr captured,
  lock guard on, exit notification; no `screen`/`nohup`/`&`. The harness writes a heartbeat
  file (cycle, timestamp) every cycle.
- **Commit the fix before G1.**
- **At C\* quote the measured detector only.** Manuscript: the estimate is the fixture-validated
  mechanism; the reported quantity is the measured per-solve detector.
- **Finding 1 — the C\* leak is not barrier-set** (μ-insensitive). Add to G1's per-cycle
  logging, per period and cohort: `pch`, `pdch`, `pnet`, and the IPOPT bound multipliers
  `zL`/`zU` of both legs. Test per period: `zL_small·x_small ≈ μ` ⇒ barrier-set; otherwise the
  O(1) multiplier present (upper bound of the large leg, SoH-floor price through `D`, warm-start
  push not decayed) names the mechanism. Characterization only; not a blocker (≈0.05 %
  spurious throughput at 2.6e-5).
- The Addendum 5 `s_obj` explanation of the 5× gap is **withdrawn** (`s_obj = 0.1` on both
  paths; `tol = 1e-8` in force). The gap is covered by the same capture.
- Order: commit → G1 (with capture) → G2 → G3-full → G4 → `REVISION_CONTEXT.md` rewrite → stop.

---

# Addendum 7 — after `P5_15_EXPERT_HANDOFF_5.md` (2026-09-14)

**Standing result:** the ESSO is closed as a component — detector flat across 72–90 cycles,
terminal μ identical in every solve, leak barrier-set (63,072/63,072), 5× gap predicted to 3
figures, G4 bitwise. Addendum 6's "not barrier-set" premise is withdrawn (the summary
`Complementarity` line is an optimality error, not μ). Remaining fragility is network-side and
configurational.

1. **Recovery policy (authorized):** explicit `recovery.enabled` (default true) for every network
   and the ESSO, independent of `recovery_options`; `case33_1` on the same policy; remove the
   `limited-memory` entries only after this lands. **Tier-2 retry** after a failed cold retry:
   cold + `mu_strategy = adaptive`, logged and counted separately. Test the tier on the saved
   cycle-38 block first (item 3). Then **re-run G2**, capturing ESSO slack values as well as
   duals.
2. **G1 attribution — ablation before re-baselining, two runs max.** Ablation A: G1
   configuration with Candidate 4 reverted (flexibility band restored). If old EFC/day and
   recourse are recovered within the bar, Candidate 4 is the cause; record that the old numbers
   carried a spurious flexible-energy source. Otherwise ablation B: G1 with Candidate 1 re-wired.
   In every case the new formulation is the baseline afterwards; the ablation serves the
   manuscript's account of the change.
3. **Cycle-38 block (authorized, single solves):** current policy, cold, cold + adaptive μ,
   `tol` one decade looser. Classify intrinsic vs path-dependent.
4. **G2 initialization:** intended behaviour (the paper's end-of-life rule acting at cycle 1 via
   the slack); recorded, no formulation change. Convergence question deferred to the G2 re-run.
5. **Comparators:** accept the loss and record it; do not regenerate.
6. Order: recovery policy + cycle-38 tests → ablation A (→ B) → G2 re-run → report → stop.

---

# Addendum 8 — after `P5_15_EXPERT_HANDOFF_6.md` (2026-09-14)

- **G2 converges with node 5 eligible** (71 cycles, zero failures): G2's earlier non-convergence
  was the recovery-eligibility defect. Recourse indeterminate vs G1 (within the bar). Recorded.
- **Ablation A/B read:** Candidate 4 accounts for ~90 % of the recourse change; neither
  Candidate 4 nor Candidate 1 explains the year-1 EFC/day drop 1.112 → 0.972.
- **Hypothesis H-ε (expert):** the ε regularization acts as a throughput *price* at the ADMM
  fixed point — the consensus dual transmits ε to the networks, whose economic layer sits at
  1e-3–1 after the `effective_scale` division (audit §2.7) — i.e. a cycling cost by the back
  door. Addendum 3's displacement argument held at fixed duals only.
- **Ablation C (authorized, one run):** G1 configuration with `EPS_ESSO_THROUGHPUT = 1e-5`.
  Frozen criteria: year-1 EFC/day within 2 % of 1.112 **and** detector ≈ 100× G1's (identity
  prediction ≈ 2.5e-3) ⇒ H-ε supported. Then production: ε = 1e-5 with ESSO `tol = 1e-10`,
  `acceptable_tol = 1e-9` (leak ≈ 1e-5, ~0.02 % throughput), G1 re-run as the new baseline,
  price effect reported in the manuscript. If C does not move EFC: record as unattributed to
  Step 1b/ε; proceed with ε = 1e-3.
- `limited-memory` entries: **remove now.** Tier 2: exercised by the G3-full re-run after the ε
  decision; no dedicated run. Cycle-38 block: classified path-dependent, recovered by tier 2 —
  closed. Parser fix: yes.
- Order: parser fix + entries removal → ablation C → production ε/tol per outcome → G1 re-run
  (new baseline) → G3-full re-run → report → **Step 1 closes; Step 3 begins.**

---

# Addendum 9 — Step 1 CLOSED; Step 3 AUTHORIZED (2026-09-15)

Step 1 closes on `P5_15_STEP1_CLOSING_REPORT.md` (commit `e19a580d`). New production baseline:
ε = 1e-5, ESSO `tol` 1e-10 / `acceptable_tol` 1e-9, explicit recovery policy with tier 2,
`limited-memory` entries removed. Detector ratio 0.57–0.94 at the new tolerance: expected
(terminal μ at `mu_min`, legs near `bound_relax_factor` scale); recorded, no action. Tier 2
reported as not yet exercised in a campaign: correct; it is counted when it fires.

## Step 3 — ADMM objective consistency, penalty policy and stopping rule

**Purpose.** Make the ADMM fixed point the recourse the paper defines, and make the stopping rule
able to certify it. Every campaign is one background run per the Addendum 5/6 process; cold
start; C\* candidate unless stated; ~40 min each.

### 3.0 — Baseline determinism (one run)
G1 at the new baseline, repeated: bitwise on per-cycle recourse, residuals, SoH, detector. This
is the reference every Step 3 change is measured against.

### 3.1 — Penalty classification and objective consistency (author decisions; zero solves)
Produce the table the Expert asked for on day one: for every penalty term in TSO, DSO and ESSO
objectives (`definitions.py` constants and `_prepare_*_objectives_for_admm`), state whether it is
an **economic cost** (part of the recourse) or a **feasibility detector** (must be zero at a
valid solution and is reported, not priced). Then make the TSO and DSO objective composition
consistent: same treatment of RES curtailment, ESS usage and scenario deviation on both sides,
or a stated reason for a difference. **Author decides each row.** The Planner drafts the table
with the current status and a recommendation per row; the expert reviews before the author
signs. This row-by-row table goes into the manuscript's appendix (R2.5/R3.7).

### 3.2 — Stopping rule (production change, one gate run)
Replace the composite motion test with Boyd et al. §3.3.1: per channel, primal residual
`r = ‖x − z‖`, dual residual `s = ρ‖z^k − z^{k−1}‖`, tolerances `ε_pri = √p·ε_abs + ε_rel·max(‖x‖, ‖z‖)`,
`ε_dual = √n·ε_abs + ε_rel·‖ρu‖`, with `ε_abs`, `ε_rel` in the case file. The objective-change
test becomes a **diagnostic** (reported, not a criterion). Rule-ten ratio reported per channel.
Gate: the 3.0 run under the new rule converges; report cycles and the terminal residuals.

### 3.3 — ρ policy and ESS normalization (production change, one to two gate runs)
(a) Residual balancing **on** as the baseline (the P5.9-B/AB1 mechanism), initial ρ from the
case file, dead band as configured; report the ρ trajectory per channel. (b) Normalize the
shared-ESS consensus residual by a **fixed per-node reference rating** (the maximum installable
`S_ref = E_max/φ_min`, 2.5 MVA here) instead of `2·max(S, 0.10)`, so the AL curvature is
candidate-independent. Gate: converges under 3.2's rule; ESS-channel residual and ρ trajectory
reported; recourse compared with 3.0 with the bar.

### 3.4 — Objective scale (production change, one gate run)
Keep the `f/σ` scaling IPOPT needs, but make σ a **fixed, recorded per-network constant** in the
case file rather than a per-run computed `effective_scale`, so `Q(x)` is evaluated under
identical scaling for every candidate. Report the constant and the run's agreement with 3.3.

### 3.5 — Polish-gap gate (48 network solves, no ADMM)
At the converged 3.4 point, re-solve every network block with the **unscaled base objective** at
fixed consensus (P5.7's test). Gate: the recourse change is < 0.1 % of the recourse. This is the
measurement that the ADMM point is base-objective-optimal, and it is the number the paper
reports as the decomposition's optimality evidence (R2.5). If it fails, the ρ/σ ratio is the
lever, and the Planner stops for review before touching it.

### Also in Step 3
G2 at the new baseline (needed for the R3.6 matrix later; one run, after 3.4). H3 stays inert
(single cohort) — a two-cohort probe is deferred to Step 5.

### Step 3 closes when
3.0–3.5 pass, `REVISION_CONTEXT.md` head records the ADMM configuration as **the** baseline
(Track B decision closed), and the manuscript list carries: penalty table, stopping-rule
definitions, ρ policy, σ constants, polish gap. Then Step 4 (outer method).

---

# Addendum 10 — Step 3.0 passed; 3.1 expert review of the penalty table (2026-09-15)

3.0: bitwise deterministic at the new baseline — the Step 3 reference. Report accepted.

**Semantics of category D (add to the table header):** a detector term stays in the solver
objective (feasibility of intermediate iterates) but is **excluded from the reported Q(x)**; its
terminal value is reported as a violation, and a nonzero value flags the solution rather than
pricing it. Applies to every benchmark case, including uncoordinated (voltage violations
reported separately from cost).

**Expert recommendations on the author rows** (author decides):

| row | recommendation | reason |
|---|---|---|
| 3 / D1 | **Remove** | transfer payment double-counted (DSO has no receipt term); anchor-dependent, so Q(x) inherits initialization; the consensus dual already carries the DSO's marginal cost |
| 2 | **E**, stated in the paper | [27] convention (downward only; Q at P price) is defensible but must be declared |
| 5 | **Identical both sides; 0** (or small identical value as **R**) | curtailed RES has zero marginal cost and is replaced by priced generation via the interface; an explicit penalty double-prices; curtailment stays a reported metric |
| 8 | **Shared ESS 0; split the flag** so local-ESS usage is not silently zeroed | inert in SRP1; hygiene for future cases |
| 9 | **Remove** the network complementarity penalty | hard relaxed row exists; the penalty is indefinite and redundant |
| 12 | **D** | per the semantics above |
| 18 | **Author's decision (2026-09-15): soft non-anticipativity retained, made defensible.** Interface P, Q: deviation of each scenario from the day-ahead schedule priced at `c = α·π̄`, with `π̄` the probability-weighted market price of the representative day and `α` an author-set **imbalance premium** (default 0.25; sensitivity over α ∈ {0.1, 0.25, 0.5, 1.0}; α = 1 is "deviation at the full energy price", the author's conservative case); **linear** in `\|dev\|` (two nonnegative variables); category **E**, in Q(x). **Borne by the DSO** (author, 2026-09-15): the deviation terms live in the DSO block, on its per-scenario interface P **and** Q against the committed schedule; the same `α·π̄` is used for Q (stated as an assumption). The TSO operates on the committed schedule and carries no dispersion term. Interface V: **no deviation term** (physical state, not a commitment). **Shared-ESS dispatch scenario-independent** in the network models (hard NA for the committed schedule the ESSO degrades — closes the P5.13-A discrepancy by construction). Results report realized interface deviations per scenario (max/expected, MW and % of rating) and per-scenario AC feasibility. Implement in Step 5. | replaces an arbitrary 9e4 (effective 85.7 after scaling — hard NA in disguise) with a priced, interpretable term; consistent with Option A |
| 20 | **Accept category R** | honest label; appendix sentence with measured bias and nil price effect at 1e-5 |
| D2, D3, D6 | **Fix with the table** | orphan slacks removed; ESSO slacks reported directly; shared-ESS day-balance slack bounded like local |

**Sequence after signature:** implement signed rows + D2/D3/D6 (one commit) → one baseline
campaign with **per-block component levels** captured at the terminal cycle (three splits per
draft §5) → reading rule per draft §5; report whether Q(x) moved and by how much, with the bar
→ 3.2.

---

# Addendum 11 — after `P5_15_S31_BASELINE_REPORT.md` (2026-09-15)

**Governing finding:** 99.8 % of the pre-signature recourse (817.5 M) was the TSO's interface
flexibility charge (row 3). Every candidate ranking, `C*`, the 0.37 % storage effect and the
templated bias were measured on a transfer payment, not on system cost. Record in
`REVISION_CONTEXT.md` and in the manuscript list: all Table 8 costs and the Fig. 5
generation/flexibility split are to be regenerated under the system-cost definition.

**Diagnosis accepted:** removing the charge left the TSO an unpriced direction at each interface;
the consensus dual is the only price and ADMM discovers it slowly (90-cycle climb, tripled
failures, DSO flex cost 209 M).

**Remedy — row 3′, the cancelling transfer (authorized):**
- TSO block: reinstate the interface flexibility charge `+T = c_flex · |p_int − p_anchor|`
  (and Q as today), category **A-transfer**.
- DSO block: add the symmetric revenue `−T` on the DSO's copy of the interface variables, same
  price, **same anchor**.
- Anchor: the DSO's warm-start (uncoordinated) interface exchange, passed to the TSO as a
  parameter at initialization; stated in the paper as the flexibility baseline. Verify whether
  the current anchor already is this quantity.
- `Q(x)` = system cost; the transfer is excluded by construction (cancels at consensus) and is
  **reported separately** per DSO as "TSO flexibility procurement cost". Interface flexibility
  variables are NOT fixed at zero.
- Gate: converged campaign at C\* (cycles, failures, rule-ten); `T_TSO + T_DSO` at the terminal
  point reported (should vanish to the consensus residual); system-cost recourse becomes the
  economic baseline for Step 3. 3.0 remains the determinism reference only.

**Discriminator (zero-solve, run first):** terminal interface flexibility variables and interface
residuals from the capped campaign; expected to show large unpriced interface movement.

**3.2 note:** a recourse-relative objective tolerance is ill-defined when the recourse can start
near zero — the objective-change test is diagnostic-only, as already specified.

**Order:** discriminator → row 3′ implementation (one commit, zero-solve checks, fixtures
unpickle) → converged baseline campaign with component levels → report → 3.2.

---

# Addendum 12 — row 3′ made well-posed (after `P5_15_S31B_ROW3PRIME_REVIEW.md`, 2026-09-15)

The Planner's objections are accepted: `−c·|Δ|` is concave and a cancelling transfer only
shifts the dual. Two changes, one per problem:

1. **Transfer = interface energy settlement at the market price (signed, linear).** DSO block:
   `+π_t · p_int` (imports paid at the scenario's hourly wholesale price); TSO block:
   `−π_t · p_int`. Cancels exactly at consensus; no anchor needed in the optimization (constant).
   Price is `π_t`, **not** `c_flex` (internal activation is already priced at `c_flex`, row 2).
   Category A-transfer. Interpreted as a dual warm start near the marginal value of interface
   energy. `δ = p_int − a` (a = DSO uncoordinated exchange) is kept for **reporting** flexibility
   volumes only.
2. **Null space removed by reparametrization.** TSO interface: `p_int = pc + δ_P`,
   `δ_P ∈ [−rating, +rating]`, one signed variable; same for `δ_Q`. Flexibility is not fixed
   at zero. Confirmed consistent with Addendum 11.
3. **No settlement on Q.** Reactive exchange unremunerated (stated in the paper); the DSO's
   `flex_q_down` cost (row 2) carries it; the TSO sees it through the dual.

Reporting: `Q(x)` = system cost with settlements excluded by construction; per DSO report the
interface settlement `Σ_t π_t·p_int,t` and the flexibility volume `δ_P`, `δ_Q`.

Gate unchanged from Addendum 11: converged campaign at C\*; terminal `T_TSO + T_DSO` (should
vanish to the consensus residual); system-cost recourse; cycles, failures, rule-ten; per-block
component levels. Then 3.2.

---

# Addendum 13 — after `P5_15_S31C_ECONOMIC_BASELINE_REPORT.md` (2026-09-15)

- **Row 3′ closed.** Cancellation identity holds (`T_TSO + ΣT_DSO` = price-weighted consensus
  residual to 2e-7); interfaces unsaturated; δ_P shifts energy between periods with small nets.
  Settlement and signed interface variable are kept jointly; **no attribution ablation.**
- **s31c is Step 3's reference trajectory, not a converged baseline** (cap 90, rule-ten 1.96,
  still descending; system cost 661.40 M). No higher-cap re-run now. Step 3 comparisons are made
  at matched cycles and at the terminal point.
- **Diagnosis recorded:** consensus met from cycle 18, economic crawl ~1e-4/cycle thereafter —
  the ρ/objective-curvature signature (S1/C1). 3.2 and 3.3 are coupled and run together.
- **Zero-solve extraction from s31c (first):** ρ trajectory per channel; per-channel primal and
  dual residual ratios per cycle; whether residual balancing fired during the crawl.
- **Gate 3.2+3.3(a) (one run, cap 150):** Boyd residual tests (`ε_abs`, `ε_rel` in the case
  file; objective-change test diagnostic-only, reported with its rule-ten ratio); residual
  balancing on from the case-file initial ρ. Report: cycles, terminal residuals per channel, ρ
  trajectory, system cost vs s31c at matched cycles and terminal, failures.
- Then 3.3(b) fixed ESS reference rating → 3.4 fixed σ → 3.5 polish gap, each one run, each
  compared with the previous.

---

# Addendum 14 — after `P5_15_S32_BOYD_GATE_REPORT.md` (2026-09-15)

- **Stopping rule vindicated:** the Boyd rule refused a run the old rule would have certified
  at cycle 48 with 5.04 M still to fall. Tolerances are **not** loosened.
- **Diagnosis accepted:** fixed proximal weight γ = 1 while balancing drives ρ to 0.2 and below
  turns the TSO update into a proximal-gradient step (constant drift), and a feedback loop
  (small Δz → small ρ-part dual residual → ρ lowered further) drives the ρ collapse.
- **Proximal term = stabiliser, not method.** Tie it to the penalty: **γ_c = τ·ρ_c per channel,
  τ = 1** (Deng & Yin 2016); one sentence + citation in the paper. γ = 0 is the fallback only if
  E2 fails to converge.
- **Consecutive converged cycles: 3.**
- **Balancing freeze:** adapt ρ only during the first 30 cycles, then hold (ADMM requires an
  eventually constant ρ; the ρ_v dead-band oscillation is the symptom of not holding).
- **Authorized:** E1 (zero solves: interface voltages vs bounds; ESSO objective scale vs TSO/DSO
  σ — this is defect D5's evidence), E4 (a few solves: local-solver noise floor → sets `ε_abs`),
  then **E2** (one run, cap 150, γ = τρ, freeze at 30, 3 consecutive cycles). E3 only if E2 fails.
- 3.3(b) and 3.4 wait for 3.2 to pass; **3.4 explicitly includes D5** (ESSO objective on the same
  σ convention as the networks).
- Report: cycles, terminal residuals per channel (primal / dual, with the proximal share of the
  dual), ρ and γ trajectories, system cost vs s31c and s32 at matched cycles, failures, noise
  floor from E4.

---

# Addendum 15 — after `P5_15_STEP32_EXPERT_REPORT.md` (2026-09-16)

**Governing reading of §2.5:** the ESS channel's constant-speed drift toward price arbitrage is
the audit's scaling finding made visible — objective divided by σ ≈ 1e5, AL undivided, so the
consensus penalty resists the economics by 2–3 orders of magnitude and a linear gradient is
walked down at `g·a²/ρ` per cycle. Not a stopping-rule defect, not a pricing defect. **Every
prior "storage effect" was measured with the storage immobilised by coordination stiffness.**
The arbitrage is the storage's value and must be allowed to occur.

Decisions:
1. **No suboptimality bound in the paper** (convex-case result on a nonconvex recourse).
   Optimality evidence for R2.5 = Boyd residuals with noise-floor-derived tolerances (§2.2 is
   kept as a reported method), the 3.5 polish gap, and the count of active bounds.
2. **No shared-ESS usage price.** Row 8 stands. After D5, **re-verify ε** with the detector
   identity and an EFC-vs-ε ablation before trusting it at the new scaling.
3. **Convergence with active voltage bounds is acceptable**; report the count as a result.
4. **The ~30-cycle mode:** one zero-solve attribution (period per channel), ≤ 1 hour; not
   blocking.
5. **3.4 now, with 3.3(b) folded in (one commit):** (a) ESSO objective on the same σ convention
   as the networks (D5); (b) σ a fixed, recorded per-network constant; (c) ρ dimensionless
   relative to the scaled objective, starting low; balancing **frozen when ρ has not changed for
   10 consecutive cycles**, not at a fixed cycle; (d) ESS residual normalized by fixed
   `S_ref = 2.5 MVA`. Zero-solve checks; preflight two cycles.
   **Gate (one run, cap 150), prediction recorded in advance:** storage reaches its arbitrage
   equilibrium within ~50 cycles; EFC/day rises from 0.067 to O(1); all three channels pass
   Boyd with 3 consecutive cycles. Pass ⇒ 3.2 + 3.3 + 3.4 close together → 3.5 polish gap.
   Fail ⇒ stop for review with the ρ, γ, residual and EFC trajectories.

---

# Addendum 16 — after `P5_15_STEP34_EXPERT_REPORT.md` (2026-09-16)

- Addendum 15's mechanism confirmed; its ~50-cycle timescale withdrawn (ADMM on a linear-until-
  bound direction is gradient-like, step ~1/ρ; lowering ρ costs local failures). **The walk must
  be shortened by initialization, not by ρ.**
- **Corrections in §2 accepted**; the row-3 price-deletion mechanism is recorded as verified.
- **D5 form accepted** (argmin-identical scaling of the ESSO coordination terms, detector-gated).
- **Success target:** equilibrium **at the SoH threshold with the degradation constraint active**;
  the floor multiplier (shadow price of degradation) is reported as a result — insight (iii).
- **ρ_ess starts at 0.1125** (balancing's value), balancing on, freeze after 10 unchanged cycles.
- **Storage usage price: closed** (author decided twice); no ε revisit.

**Authorized, in order:**
1. **Reference equilibrium run:** s34 configuration, cap 500, ρ_ess start 0.1125. Report the
   cycle at which all three channels pass (3 consecutive), terminal EFC/day vs threshold, the
   floor multiplier, system cost, failures. This is the endpoint every later oracle is judged
   against.
2. **Price-taker initialization (implement in parallel):** initialize the shared-ESS dispatch in
   every network block and the ESS consensus copies at the `EFC*` price-taker LP solution
   (candidate- and price-dependent, history-free; SoH floor and efficiency included in the LP if
   cheap, else the LP as is); initialize the ESS dual consistently (zero, or the LP's price
   differential in the AL units — state which). Zero-solve checks; two-cycle preflight.
3. **Gate (cap 150):** the initialized run must reproduce run 1's terminal system cost within the
   rule-nine bar, EFC/day within 2 %, and pass Boyd (3 consecutive) on all channels. Pass ⇒
   3.2–3.4 close, this is the production oracle, 3.5 polish gap follows. Fail ⇒ stop with
   trajectories.
- Over-relaxation (Boyd §3.4.3, α ≈ 1.5) is recorded as the next lever if 3 fails; not now.

---

# Addendum 17 — after `P5_15_ADDENDUM16_EXPERT_REPORT.md` (2026-09-16)

- **Run 1 is the programme's first certified evaluation** (Boyd stop at 477; system cost
  651,039,166 settled, rule ten 0.0039). Storage channel stopped at 0.988 — EFC/day ≥ 1.059,
  pinned from below only. Harness `stopped_by` defect: v2 evaluator accepted.
- **Success at C\* = Boyd certification.** Addendum 16's "at the SoH threshold, floor active" was
  a prediction; Z2 falsified it. At C\* storage is network-limited, not degradation-limited
  (terminal SoH 0.659, floor dual ~0). Recorded as a result about the candidate.
- **Insight (iii)** demonstrated in Step 5 wherever the floor binds (k = 10,000 arm, larger or
  longer-cycled plans); if nowhere, the paper states it and the degradation effect is carried by
  available capacity and salvage. Not manufactured.
- **Gate 3 read:** schedule initialized correctly; the pace is set by dual build-up from zero.
  Initialization must be primal **and dual**, and **history-free**.

**Authorized, in order:**
1. Zero-solve: (a) dual-direction comparison gate 3 vs run 1; (b) price-taker LP with run 1's
   terminal nodal prices; (c) the same LP with **cycle-0 LMPs** from the initialization OPF
   solves (node-balance duals at the storage buses, per scenario/period). Report EFC from (b)
   and (c) against run 1's 1.059 and against each other.
2. **Dual initialization from cycle-0 LMPs:** per agent, the marginal value of its storage copy
   at the LP schedule, mapped into the AL's normalized units, projected onto the consensus
   zero-sum invariant. **Mapping test before any run:** applied to run 1's terminal prices it
   must reproduce run 1's terminal storage duals (per agent) to a stated tolerance. Run 1's
   prices are validation only; production uses cycle-0 LMPs (deterministic per candidate).
3. **ρ starts per channel = balancing's C\* values** (0.0116 / 0.088 / 0.1125) as case-file
   defaults; balancing on; freeze after 10 unchanged cycles.
4. **Gate (cap 150):** Boyd on all three channels (3 consecutive); cost within the rule-nine bar
   of run 1; storage channel terminal ratio < 0.9 (settled, not stopped). Pass ⇒ 3.2–3.4 close,
   production oracle = LMP-initialized ADMM, 3.5 follows, and EFC is a certified quantity.
   Fail ⇒ stop with trajectories; reporting then follows §6's one-sided statement.
- Over-relaxation parked. **Claims to avoid** (§6) adopted verbatim for `REVISION_CONTEXT.md`
  and the manuscript list.

---

# Addendum 18 — Step 3.6: within-cycle block parallelism (author, 2026-09-16)

**Rationale.** Within a cycle the only dependency is across agent types (DSO → TSO → ESSO). All
blocks of one type (36 DSO, 12 TSO, 3 ESSO) are independent given the consensus state, so they
are solved concurrently. Target: cycle wall time from ~39 s to ~8–10 s on 8 workers.

**Design (Planner audits the existing `*_parallel` path first):**
- Persistent worker processes, each owning a fixed subset of blocks; models built once; only
  consensus parameters, duals and interface results exchanged per cycle.
- Each worker single-threaded (`OMP_NUM_THREADS=1`; MA57, or MA97 single-thread): bitwise
  reproducibility and no core oversubscription. Multithreaded factorization is NOT used.
- Private Pyomo temp directory and unique log paths per worker (the likely cause of the P5.6-B
  concurrency crashes).
- Results assembled in a fixed block order; ADMM updates unchanged.
- **Zero-solve first:** from existing per-solve logs, sum "Total seconds in IPOPT" per cycle
  against the ~39 s wall time, per block type. If Pyomo NL write/read overhead dominates, record
  an in-memory NLP interface (PyNumero/cyipopt) as a later, separate item with a
  numerical-equivalence gate.

**Gate:** two-cycle preflight serial vs parallel — **bitwise identical** trajectory, residuals,
detector; then one matched full run identical to its serial reference. Cycle time reported.

**Sequencing:** implementation and zero-solve checks may proceed while the Addendum 17 gate
occupies the machine; the preflight and gate run only after it, never mixed with a numerical
gate. Step 3.6 precedes Step 4.

---

# Addendum 19 — after `P5_15_ADDENDUM17_EXPERT_REPORT.md` (2026-09-17)

- **F1–F3 accepted.** Per-agent storage duals are identified only up to the span of duplicated
  active storage rows (standard redundancy in consensus ADMM); the in-span split is a harmless
  random walk; the identifiable part is small at the optimum. **Dual initialization is not the
  lever; Addendum 17's route is closed.** Run 1 remains the certified evaluation. No replay.
- **The bottleneck is the primal walk speed:** per-cycle block move `≈ ∇f_i/(ρ_ess·a²)` (~2e-4
  normalized units per cycle at ρ_ess = 0.1125), which set run 1's 477 cycles and gate 3's
  drift alike. The dual increment is ρ-invariant; the primal step is `∝ 1/ρ_ess`.
- **Residual balancing is mis-specified on the ESS channel:** its dual residual is dominated by
  the unrelieved economic gradient and does not shrink with ρ, so balancing drives ρ_ess *up*
  (0.05 → 0.1125). **The ESS channel is exempted from balancing.**

**Authorized — the ρ_ess experiment (two runs, cap 150, cold, no LP initialization):** run 1's
configuration with ESS balancing off and `ρ_ess ∈ {0.01, 0.001}`. Predictions recorded in advance:
storage step per cycle scales ~1/ρ_ess; EFC/day reaches 1.06 within ~50 cycles at 0.01; at least
one arm certifies (Boyd, 3 consecutive, storage ratio < 0.9, cost within the rule-nine bar of run
1). Monitor local-solve failures and ESS step-direction sign changes (oscillation). If both
certify, adopt the larger ρ_ess. Report cycles, EFC trajectory, step sizes, failures.

- **Option 1 (settlement active at initialization): parked** — gives LP(π) duals, low expected
  value; revisit only if both arms fail.
- **Step 3.6 proceeds, decoupled:** first per-phase timing per block (model update, NL write,
  IPOPT, `.sol` read, bookkeeping); persistent workers must absorb per-block overhead or the
  path is not built. Cheap wins to test: Pyomo NL writer v2; no symbolic labels in the `.sol`
  round trip. Preflight and gate after the ρ_ess arms.
- Step 4 warm-continuation design (§7) accepted as designed; not run.

---

# Addendum 20 — after `P5_15_ADDENDUM19_EXPERT_REPORT.md` (2026-09-17)

- **ρ_ess did its job** (storage first pass 87–95 vs 475; EFC 1.06 by cycle 33). Neither arm's
  storage channel is *settled* (under-damped at low ρ once the gradient is spent). **Recorded
  for later, not now:** a two-phase ESS schedule — ρ_ess low while the storage walks; ESS
  balancing re-enabled once its dual ratio has been < 1 for 5 consecutive cycles.
- **The PF channel sets the verdict.** Its dual ratio decays at ~0.988/cycle, invariant to
  ρ_ess. Reading: HP1 + HP2 — the TSO proximal term (τ = 1) halves the TSO step every cycle,
  and ρ_pf is too large for the late phase while the cycle-60 backstop froze balancing before
  the imbalance appeared. HP3's √2 is correct accounting of a real term, not a cause.
- **The cycle-60 backstop is removed for all channels.** Freeze rule = Addendum 15's
  (10 unchanged cycles) plus an absolute freeze at cycle 200.
- **Cap is a budget, not a criterion:** certification runs use cap 300; tolerances untouched.
- **Step 3.6 proceeds now in parallel**; interim target S = 3 / X = 70 % accepted as interim,
  the measurement sets the real one; the two "cheap wins" are dropped (already defaults).

**Authorized:**
(a) zero-solve replay of the balancing rule on the saved trajectories (would ρ_pf have been
    lowered, and when);
(b) two arms, cap 300, cold, ρ_ess = 0.01 fixed and ESS-exempt, **per-entry PF capture on**
    (standard instrumentation), no backstop:
    **A** — τ = 0 (proximal off); fallback τ = 0.25 only if the TSO shows instability
    (failures or oscillation);
    **B** — PF balancing active, freeze-after-10, absolute freeze 200.
    Prediction: PF first pass well before 226 in at least one arm; A the stronger candidate.
    Report PF first-pass cycle, ρ/γ trajectories, per-entry PF decomposition, cost vs run 1's
    bar, failures, certification verdict.
(c) if both help: one combined confirmation run; the adopted configuration becomes the oracle
    and 3.5 follows.

---

# Addendum 21 — after `P5_15_ADDENDUM20_EXPERT_REPORT.md` (2026-09-17)

- **HP1 + HP2 confirmed**; HP1 (proximal term) the larger lever. PF no longer sets the pace.
- **Criterion (c) (storage terminal ratio < 0.9) is WITHDRAWN** — it tests which channel closes
  the stop (closing channel ≈ 0.95 by construction); run 1 fails it at 0.988.
- **Certification bar, from now on:** all channels inside their Boyd tolerances for **10
  consecutive cycles**; per-channel terminal ratios and the rule-ten objective ratio reported,
  not gated. Cost bar (rule nine) uses the **maximum objective step over the last 10 cycles**.
  Run 1 remains certified under Addendum 16's rule.
- **τ = 0 globally** is the oracle setting; the proximal term is removed from the method
  (consistent with the paper's Algorithm 2). Per-channel τ is a recorded fallback, implemented
  only on evidence of TSO instability on some candidate.
- **Authorized, in order:**
  1. **Arm C** (as Addendum 20 (c)): τ = 0, PF balancing live (freeze-after-10, absolute 200),
     ρ_ess = 0.01 fixed and exempt, cap 300, per-entry PF capture, new bar. Predictions recorded
     first: PF first pass ≤ 131; certification by ~145.
  2. **Arm D** (separate): C plus the two-phase ESS schedule — ρ_ess = 0.01 while the storage
     walks; ESS balancing re-enabled once its dual ratio has been < 1 for 5 consecutive cycles.
     Prediction: certifies with the storage channel well inside tolerance (terminal ratio
     < 0.5). If so, **D is the production oracle**; else C.
  3. **Zero-solve, in parallel:** node 7 interface (P, 2030) at the stop — interface flow vs
     limits, voltage, flexibility bounds, storage dispatch at node 7.
  4. **Step 3.6:** replace the per-cycle TSO whole-model clone with a lightweight capture
     (mutable parameter values + warm-start suffixes; block rebuilt deterministically on demand);
     re-measure over 10 cycles; then build persistent workers. Preflight/gate after arms C and D.
- After the oracle is fixed: **3.5 polish gap**, then Step 3 closes.

---

# Addendum 22 — after `P5_15_ADDENDUM21_EXPERT_REPORT.md` (2026-09-17)

- **Both arms certify under the 10-cycle bar.** PF closed (HP1 + HP2 exhausted).
- **Oracle = D** (τ = 0, PF balancing live, two-phase ESS schedule, ρ_ess 0.01 → lift on 5
  sub-unity cycles). The "storage terminal < 0.5" condition is **withdrawn** (a prediction with
  criterion (c)'s defect). **What the oracle delivers:** a certified cost with its stated bar (max
  objective step over the last 10 cycles) at the fewest cycles; the bar is reported and used in
  every comparison; close Step 4 decisions get longer runs via the cold-audit cadence. Storage at
  ~0.8 of tolerance is converged to tolerance; no further lever.
- **Rules adopted:** one frozen oracle configuration for every candidate in a campaign; costs
  from different configurations never share a table. Zero-solve component decomposition of the
  58–72k differences (C, D vs run 1) authorized.
- **Node 7 (author's question):** the interface rating is **not** changed for convergence. The
  binding interface with the storage at rating in the same periods is the storage's congestion-
  relief value and the paper's reason for siting at node 7. Report it as a result (utilization
  and dispatch table in the binding periods). Run the period-by-period cross-check of PF-residual
  entries vs at-rating periods (zero solves) to record the mechanism.
- **Step 3.6:** integrate the clone replacement (Worker's pristine-block-per-year/day design is
  correct; record the interface-load re-fix bug it avoided), bitwise preflight, 10-cycle
  re-measurement, then persistent workers; bitwise gate.
- **Closure of Step 3:** write D's configuration into `data/SRP1/SRP1_params.json` (closes
  Track B), run **3.5 polish gap** on D's certified point (gate < 0.1 %), `REVISION_CONTEXT.md`
  head, Step 3 closed.
- **Step 3.7 — Anderson acceleration (authorized after 3.5 and after the 3.6 bitwise gate):**
  type-II AA, memory 5, on the joint (consensus, dual) iterate, active only once ρ is frozen;
  safeguarded (accept the extrapolated step only if the combined residual norm falls vs the plain
  step, else plain step); regularized least squares; harness flag default off. Gate: two-cycle
  identity with the flag off; with the flag on, certification ≤ 80 cycles at C\* and certified
  cost within D's bar. Prediction recorded before the run. Citation: Zhang, O'Donoghue & Boyd
  (2020); Fu, Zhang & Boyd (2020).

# Addendum 23 — after `P5_15_ADDENDUM22_EXPERT_REPORT.md` (2026-09-18)

- **Status accepted.** Items (1)–(3) done; τ = 0 determinism established (D reproduced bitwise
  over 139 cycles, twice). Step 3 stays open on 3.5 only. The Planner's three findings are
  accepted and change three earlier decisions (below).
- **Step 3.5 restated — interval-hull polish.** Diagnosis accepted: the certified point satisfies
  consensus only to the Boyd tolerance, so fixing a coupling entry at the midpoint places it half
  a residual outside any block whose own achieved value sits on that block's feasible boundary
  (node 7 at rating explains 6 TSO failures; the other 11 are the same mechanism on constraints
  not yet identified, or a harness defect). Exact fixing is ill-posed at active constraints, and
  the certified point is not exactly feasible for it. **New test:** every coupling entry is
  constrained, entrywise, to the closed interval between the agents' achieved values at the
  certified cycle (two agents for V and PF; three for ESS; a degenerate interval is a fixed
  value). Each block is then feasible by construction (its own achieved point lies in the hull);
  base objective as in P5.7; primal warm start from the certified block solution, IPOPT-default
  bound push, no multiplier import; production recovery policy; all 48 blocks must solve.
  Gap Δ = Σ_i [f_i(polished) − f_i(certified)] is ≤ 0 by construction (each block's start is
  feasible); **gate |Δ|/cost < 0.1 %**; any block with Δ_i > 1e-6·cost is flagged as a solver
  artefact, not distortion. Report: total Δ, TSO/DSO split, max |Δ_i|, and the number of hull
  bounds active at the polished points (residual coupling pressure). The polished sides may
  differ by up to the certified residual — the test measures augmented-Lagrangian distortion at
  the certified tolerance, not feasibility restoration. Rejected: projection (needs the active
  constraint identified per entry; six TSO failures unexplained), one-sided fixing (mirrors the
  problem onto the other block), Boyd-tolerance box (well-posed but looser than the hull, which
  is the tightest entrywise box feasible for every agent). **Before the run (zero solves):** read
  the restoration-phase logs of the six autumn/winter TSO and five DSO-2025 failures; confirm the
  polish harness's starting point, and that it does not re-fix interface loads from the mutated
  consensus (the §6.1 constructor bug). A harness defect is fixed before the hull run. The
  exact-fix outcome (17/48 infeasible) is recorded as a methodological finding — one sentence in
  the manuscript.
- **What 3.5 is.** Not optimality evidence for a nonconvex problem: block-stationarity at the
  certified tolerance. The R2.5 package is: certified consensus (10 plain cycles inside the Boyd
  tolerances); hull polish gap (bound on AL distortion); the configuration-reproducibility band
  (four configurations ≤ 0.011 %); acknowledged DSO SMOPF multimodality (~0.3 %). Cite Hong,
  Luo & Razaviyayn (SIAM J. Optim. 2016) and Wang, Yin & Zeng (J. Sci. Comput. 2019) for what
  nonconvex consensus ADMM delivers.
- **Node 7 — Addendum 22's reading withdrawn.** Report only what holds: the node 7 interface is
  the only active coupling constraint (23/288 periods; nodes 5 and 9 peak at 51 %/67 %); the
  storage is idle in those periods; relief comes from DSO-side flexibility; the storage's dispatch
  is set by price arbitrage and wear cost. "Congestion-relief value" and "reason for siting" are
  struck from the manuscript items. The rating constraint's multiplier at the certified TSO
  solves is reported if already captured (zero solves); the polish-derived multiplier is not
  used (degenerate with the hull bound where both coincide). No separate counterfactual run: the
  Step 5 Phase A screen must include node-7-empty candidates, so the siting question is answered
  by the campaign under the frozen oracle.
- **Cost differences (§5) accepted as determinate.** Manuscript, method section, next to the
  stopping rule: at C\* four configurations certified under the same bar reach operating points
  within 0.011 % in system cost, through a generation/internal-flexibility trade-off; the campaign
  uses one frozen configuration and compares candidates only within it; the reported cost is that
  configuration's. Not presented as an uncertainty band on candidate differences.
- **Step 3.6 persistent workers — one bounded task, then paused.** Per-phase timing over 5
  cycles on the parallel path (no algorithmic change); fix the implicit→explicit bound restore
  and its warnings; remove the DSO node-7 failure-snapshot clone with the same capture technique
  (bitwise gate). Then the path is paused whatever the timing shows: Step 5's parallelism is at
  the candidate level (Phase A screen and OrthoMADS polls are embarrassingly parallel, ≥ 2n
  evaluations per poll, no state sync), so within-cycle parallelism is not on the critical path.
  Record peak RSS of one serial certified run (the 3.7 run) to size candidate-level parallelism
  on 32 GB. The screening's 97 %/5.09× figure is recorded as mis-accounted (parent sync counted
  as absorbable).
- **Step 3.7 proceeds now** (after the hull polish; the persistent-worker gate is no longer a
  prerequisite). Amendments to Addendum 22: (i) AA active from cycle 1, memory cleared at every
  ρ change on any channel (balancing step, ESS exemption lift, freeze); "once frozen" is the
  special case — in D, V/PF ρ change at cycles 1–2 only, ESS at 32–33, all frozen from 44, so the
  old rule would have left AA cycles 44–139 only. (ii) Iterate = stacked (z, u) over all channels
  in the residual-test normalization (V pu, PF pu, ESS/S_ref); g = F(w) − w; type-II, memory 5,
  Tikhonov 1e-10. (iii) Safeguard on the combined scaled residual, no extra solves: an AA step is
  taken only while the residual at the current cycle is below the residual before the last
  accepted AA step; otherwise the plain iterate is used and the memory cleared (the Fu–Zhang–Boyd
  envelope form is acceptable; the choice is recorded). (iv) AA off once all channels are inside
  tolerance: the 10 certifying cycles are plain ADMM, so the certificate is independent of AA;
  AA resumes if a channel leaves tolerance. (v) Gate: two-cycle bitwise identity with the flag
  off; flag on: certification ≤ 80 cycles at C\*; certified cost within **1.5e-4 relative of D**
  (the observed configuration spread, replacing "within D's bar" — §5 shows different trajectories
  legitimately land 40–72k apart); decomposition vs D reconciling to the two known components;
  hull polish gap < 0.1 % at the AA point. (vi) Adoption: pass → AA-on becomes the frozen
  campaign configuration, case file updated before Step 5, D superseded and never mixed; fail
  on cycles → AA stays off and the campaign runs D. Prediction recorded before the run;
  accepted/rejected steps and resets recorded per cycle.
- **Order:** 3.5 log read → hull polish (Step 3 closes on pass; `REVISION_CONTEXT.md` head) →
  persistent-worker task → 3.7. One background run at a time. Stop for review after Step 3
  closes and after 3.7.
- **Manuscript items added:** hull-polish definition and the exact-fix sentence; reproducibility
  statement; node 7 restated; AA paragraph and convergence figure if adopted; references Walker &
  Ni (2011), Fu, Zhang & Boyd (2020), Hong et al. (2016), Wang et al. (2019).

# Addendum 24 — after `P5_15_S41_STEP3_CLOSED_REPORT.md` (2026-09-18)

- **Step 3 closed; accepted.** Hull polish at D's certified point: 48/48 solved, no retries;
  Δ = −2,012 (3.1e-6 of cost, TSO −34 / DSO −1,978), Δ ≤ 0 as predicted, max |Δ_i| 775, six
  blocks positive by ≤ 0.39 (solver noise, none near the flag). Third bitwise reproduction of D.
  The Planner's gate correction is right: the gate quantity is Σ per-block change (sign-guaranteed).
  **Report both numbers:** the block-objective change (−2,012, 3e-6) as the construction guarantee
  and the system-cost change (+11,967, 1.8e-5, settlement excluded) as the headline for R2.5 —
  their difference is the settlement transfer, which cancels only at exact consensus. Record for
  the manuscript: the settlement's non-cancelling remainder at the certified tolerance is 36,679
  (5.6e-5 of cost) and is excluded from the reported cost by construction (Q = gross operational
  cost). Hull-bound counts are to be reported excluding degenerate intervals (zero-solve, if cheap).
- **Stale helper (`p56a_oracle._interface_expression`) — finding accepted.** The v11 exact-fix
  run is void: the helper built the TSO interface flow as `pc + flex_up − flex_down`, omitting
  `interface_delta_p/q` (Addendum 12), so `apply_common_values` equated two unequal constants
  (violations 0.20–0.84 pu frozen from iteration 0). Addendum 23's causal reading of the 17
  failures and the node-7 rating attribution are **withdrawn as explanations of that run**; the
  hull test stands on principle (well-posed at active constraints by construction), not on that
  run. **Authorized:** fix the helper so it takes the interface flow from the model's own
  expression (`pc_adn` or equivalent) rather than re-deriving it; if it must be hand-built,
  include `interface_delta_p/q` and add a zero-solve regression check that the helper equals the
  model expression at the certified snapshots. **Re-run** the exact-fix polish at D's point as
  the fix's test (48 solves), prediction recorded first: the six spring/summer TSO blocks where
  node 7's midpoint exceeds 100 MVA report local infeasibility with violation ≤ 5e-4 MVA; the six
  autumn/winter TSO blocks solve; the five DSO-2025 blocks are recorded without prediction. If all
  12 TSO blocks solve, the rating-midpoint check (utilization 1.000005) is re-examined for the
  quantity it measures. Manuscript sentence (replaces Addendum 23's): "The coupling variables
  are confined to the interval between the agents' certified values rather than fixed, because
  the certified point satisfies consensus only to tolerance and exact fixing is not guaranteed
  feasible where a coupling constraint is active" — plus, if the prediction holds, "at the
  baseline point exact fixing is infeasible in the blocks where the node 7 interface is at its
  rating." No manuscript use of the v11 run.
- **Step 3.7 code choices accepted:** a cycle with a local-solve failure skips AA and clears the
  memory; the safeguard mark is unchanged on rejection; the 2.7e-16 dual round-trip is
  acceptable with the flag on — with the flag off the AA code must not touch the iterate (the
  bitwise gate verifies it). Peak RSS recorded.
- **Order:** helper fix + exact-fix re-run (short; closes Step 3's last loose end) →
  persistent-worker bounded task (Addendum 23) → Step 3.7 integration, flag-off two-cycle bitwise
  gate, flag-on run under the Addendum 23 gate. One background run at a time. Stop for review
  after 3.7.

# Addendum 25 — after `P5_15_S43_STEP37_REPORT.md` (2026-09-19)

- **Step 3.7 result accepted.** AA certifies at 109 (−22 %), cost +7,456 from D (inside the
  band), reconciles, hull polish 0.000347 %. Gate (a) failed on a choice of mine: Addendum 23
  (iii) cleared the memory on every rejection, and 27 of 109 cycles were spent with an empty
  memory. That was over-conservative: every visited (w, g) pair is a valid sample of the map
  whichever step produced it, so a rejection need not discard the history — only a map change
  (ρ step, exemption lift, freeze) or a failure cycle should. **One bounded variant arm at C\*,
  authorized:** identical to the 3.7 run except that a rejected step keeps the memory (plain
  iterate taken, mark unchanged, pair retained). Single change; prediction recorded before the
  run (expert: 80–95 cycles, gate (a) uncertain); gates (b)–(d) unchanged. The better AA arm at
  C\* (109 or the variant) goes forward to the selection run below. Gate (c) tightening
  (residual ≤ 1.0, other components identically 0) is right.
- **Adoption is decided by robustness, not by the C\* number.** The ≤ 80 bar was a build/no-build
  threshold; AA is built and flag-off bitwise-verified, so the campaign question is whether AA-on
  saves cycles at every candidate while passing (b)–(d) at each. **Configuration-selection run
  (opens Step 4):** three candidates besides C\* — the paper's plan (node 7 only, 1.62 MVA /
  3.24 MWh), node-7-empty (C\* at nodes 5 and 9), 2×C\* at all nodes — each evaluated cold under D
  and under the chosen AA arm (6 certified runs). This also answers the question Step 4 must
  answer before any search: **does D certify across the design space?** (P5.14-N's 82 % failure
  regime one parameter step from the old baseline is the reason.) Rule: AA-on is adopted as the
  frozen campaign configuration only if it certifies in fewer cycles than D at all four
  candidates with (b)–(d) passing at each (band 1.5e-4 relative to D's cost at the same
  candidate); otherwise D. Case file updated before Phase A if adopted. A candidate at which D
  itself fails to certify within cap 500 stops the run for review.
- **Campaign harness — first Step 4 task, tested by the selection run.** N concurrent
  evaluations, each its own process (`OMP_NUM_THREADS=1`), results directory, heartbeat, exit
  code and per-cycle record; one campaign-level lock replaces the one-run lock for campaign use
  only; frozen spec per campaign. Gate: a C\* evaluation through the harness, concurrent with
  two others, reproduces D bitwise. Concurrency for the selection run: 3 (7.7 GB).
- **Alias tie-break non-determinism:** fix (sort by name, not over a set) in the same bounded task
  as the harness — a non-deterministic report field will eventually fail a bitwise gate. The
  parallel-arm `esso_models_pickle.bytes` difference stays documented; that path is paused.
- **Scale (Step 4/5 design input; measure, do not assume).** Peak RSS 2.55 GB at SRP1 scales
  with the block count (48 network blocks), so the paper instance (5 years × 4 days × 25
  scenarios ≈ 42× SRP1) is ≈ 100 GB and ≈ 42× the cycle time (≈ 25 min/cycle, ≈ 2 days per
  certified evaluation) if the architecture is unchanged — not runnable on 32 GB. **Zero-solve
  measurement authorized:** instantiate the paper-scale instance (build only), report peak RSS
  and block count; if it fits, time one cycle. The campaign instance (scenario count) and the
  paper-scale verification route (memory-flat build–replay–solve–discard evaluation, scenario
  reduction, or a larger machine) are the author's decision, taken on that measurement.
- **Exact-fix re-run:** prediction falsified (5/12) — recorded. The 100 MVA figure is the **DSO
  node-7 interface branch rating**, not a TSO row; the node 7 result and manuscript wording say
  so ("the DN's interface branch at node 7 is the only active interface constraint"). The
  conditional clause is struck; the principle sentence stands; the hull test remains the
  adopted 3.5 test.
- **Order:** alias fix + campaign harness (bitwise gate) → AA variant arm at C\* → selection run
  (6 evaluations, 3 concurrent) → paper-scale build measurement. Stop for review after the
  selection run with the scale measurement attached.

# Addendum 26 — Step 4 design decisions (author + expert, 2026-09-19)

- **Design variables (author):** per-node capacities are **granular** at 0.25 MVA (power) and
  0.5 MWh (energy) — the order of one PCS string block and one battery rack; commercial 20-ft
  units (3.5–7 MWh, 2–4 h) are multiples. Duration bounded **2 h ≤ E/P ≤ 4 h** (case file
  `max_energy_to_power_factor` 10 → 4). **Investment year is a decision variable**: the general
  lattice is $(P_{n,y}, E_{n,y})$ over nodes and representative years (the model's existing
  per-year investment structure and cohorts); the single-cohort form $(P_n, E_n, y_n)$ is the
  Phase A ladder form and the Phase B fallback.
- **Method definition:** `STEP4_DFO_METHOD.md` (v1) — master problem, oracle contract, MADS with
  granular variables (OrthoMADS poll; Phase A designed search; optional Benders-type local-model
  search), implementation (NOMAD 4 via PyNomad recommended, in-house OrthoMADS fallback), budget
  table, claims. The Planner writes the Step 4 spec from it after the selection run and scale
  measurement (Addendum 25) return.
- **Oracle rules for the campaign:** cold start for every candidate — warm continuation
  (Addendum 17 §7) is rejected for the campaign because it makes Q(x) history-dependent; cache
  keyed by canonical x; extreme barrier for non-certified evaluations, recorded with cause; the
  oracle's resolution σ_Q (provisional 1.1e-4 of Q) re-measured on the Phase A unit-step ladders
  and reported with the final claim.
- **Reference candidates on the lattice:** C\* → 1.0 MVA / 4.0 MWh at all nodes; the paper's
  plan → 1.5 MVA / 3.0 MWh at node 7. The selection run (Addendum 25) keeps the original
  off-lattice C\* and plan, since it tests configuration robustness, not the campaign.
- **Author's decisions (2026-09-19, later):** `budget` (in €) and `max_capacity` are **kept** as
  master constraints; the investment cost file had a small error and was corrected (the Planner
  confirms the committed `SRP1_ESS.xlsx` is the corrected one; its hash goes into the campaign
  spec); **salvage excluded** from I(x) — at most a Step 5 sensitivity on the final plan.
  Implication recorded: at the file's unit costs the budget binds at about one Megapack-class unit
  per node in 2025 (≈ 1.0 MVA / 3.5 MWh at 4 h; 1.5 MVA / 3.0 MWh at 2 h; more in later years),
  so the master problem is an allocation problem (node, duration, year, split) and the paper's
  claim is "optimal under the budget". **Phase A adjustment:** the A1 single-node ladders run
  **without** the budget (the value-of-storage curves are the sensitivity that answers "why this
  budget"), capped only by `max_capacity`; the budget applies from A2 and throughout Phase B.
- **To confirm before the Step 4 spec (Planner, zero solves):** I(x) and budget slack at the
  paper's plan and at C\* with the committed cost file; whether x = 0 (no storage) is evaluable;
  whether the ESSO capacity multipliers are available without extra solves (for the optional
  search step).
- **Open author decisions:** P^max/E^max per node beyond `max_capacity`; campaign instance after
  the scale measurement; general lattice vs single-cohort in Phase B (budget table §7).

# Addendum 27 — after `P5_15_ADDENDUM25_EXPERT_REPORT.md` (2026-09-19)

- **D certifies at all four candidates** (139/139/136/187, zero local-solve failures). The
  robustness question for C\*-sized and larger designs is answered.
- **AA-on (`keep_memory`) adopted as the frozen campaign configuration.** Rule met at every
  candidate (107/116/107/180; (b)–(d) pass; every cost difference reconciles to the two known
  components). The 4 % margin at 2×C\* is not a concern — the rule is "fewer cycles with the
  checks passing", and 2×C\* lies outside `max_capacity` anyway; record that the saving falls
  with storage size. Before Phase A: write AA-on into `data/SRP1/SRP1_params.json`, then verify
  that the case-file-alone run reproduces the AA C\* evaluation bitwise (107 cycles,
  650,982,939.94). Both cycle predictions (expert 80–95, Planner 78–100) missed: memory loss was
  not the main limit on AA — recorded. Harness gate ruling accepted (tie order from the sort fix).
- **Master constraints were never on the oracle's path** (`budget`, `max_capacity` live only in
  the retired Benders master): the DFO enforces them in closed form. `max_capacity` caps **energy**
  at 5 MWh per node — production semantics kept; `STEP4_DFO_METHOD.md` corrected (P ≤ 2.5 MVA
  follows from the duration bound). Ladders become E ∈ {1, …, 5} MWh at 2 h and 4 h.
- **Budget finding for the author:** with the committed cost file the paper's own plan costs
  €1,073k — over the €1M budget; with `7ce1d1ab` (`paper_revisions`, energy × 1.25) every
  reference candidate is over it. Two author decisions are open and block only Phase B: (i) which
  cost file is the corrected one (bring it onto this branch; hash into the spec); (ii) budget
  framing — recommended: primary Phase B without the budget (the size where marginal value meets
  marginal cost, bounded by `max_capacity`) plus a budget-constrained Phase B sharing the cache as
  the reported sensitivity. Q(x) does not depend on the cost file, so A0/A1 proceed now.
- **Salvage:** a reporting expression only — the ESSO subproblem objective is the ε-throughput
  regularizer plus the ADMM terms; no salvage credit and no wear price enter it. "Salvage excluded"
  is therefore a reporting choice with no case-file change. Correction to the node 7 wording: the
  storage's dispatch follows the networks' price and flexibility signals **under the ageing
  constraints (available capacity, SoH floor); no wear price is charged** — replace "wear cost"
  wherever Addendum 23 used it. Q6: yes, "the DN's interface branch at node 7 is the only active
  interface constraint".
- **Phase A opens with A0** (Planner's §5.3, as `STEP4_DFO_METHOD.md` §4.1 now records): x = 0,
  the smallest lattice unit at each node (0.25 MVA with 0.5 and 1.0 MWh), the lattice plan
  1.5 / 3.0 at node 7 — 8 evaluations, one batch, under AA-on; a non-certified point stops for
  review. Then A1 (90 evaluations, 10 slots, hull polish on incumbents only).
- **Paper scale — bounded task authorized:** (a) the pristine snapshot clones switchable, off for
  campaign and verification runs; (b) the four single-scenario code paths generalized, **preceded
  by a zero-solve audit of every consumer of `shared_ess_data.prob_market_scenarios` on the
  oracle path** — from the code, the ESS workbook's three investment-cost scenarios feed the
  ESSO master's investment cost, the salvage expected unit cost and the results writer, while the
  networks use their own `network.prob_market_scenarios`; the audit confirms no operational term
  is weighted by the workbook's probabilities in SRP1 or at paper scale, and names the defect if
  one is; (c) re-measure the build and **time one cycle**, run alone (18 GiB). Provisional campaign
  design: search on SRP1; paper-scale evaluation of the final incumbent serially with snapshots
  off; confirmed or revised on the timed cycle.
- **ESSO multipliers:** deferred; if §5.6 is ever used, sign and units are validated first by a
  finite-difference check at C\* (two certified runs).
- **Order:** case file → AA-on + bitwise re-verification → probability audit (zero solves) → A0 →
  paper-scale bounded task incl. the timed cycle (machine alone) → A1 ladders. Phase B waits for
  the two author decisions. Stop for review after A1 with the timed cycle attached.

## Addendum 27 — author's decisions (2026-09-19, later)

- **Cost file:** `data/SRP1/SharedESS/SRP1_ESS.xlsx` at `7ce1d1ab` ("Costs updated",
  `paper_revisions`) is the corrected file. Bring that file alone onto this branch (checkout of
  the path from that commit, committed by name), record its sha256 in the campaign spec, and
  recompute I(x) and the budget position of the reference candidates with it (zero solves). It
  does not touch the oracle: Q(x) evaluations already ordered stay valid.
- **Budget:** the campaign runs **under the €1M budget** as its primary and only Phase B — the
  submitted paper's framing, with the plan re-optimized because the cost data were corrected. The
  "what the budget forgoes" question is answered by the A1 ladders, which run without the budget by
  construction; an unbudgeted Phase B is **not** run unless time permits after Step 5 (the cache
  carries over). Under a binding budget the search is over allocation (node, duration, year, split)
  in a small region, and poll directions that leave the budget cost nothing — Phase B is cheaper
  than the unbudgeted estimate.
- **Phase A reduced for cost** (`STEP4_DFO_METHOD.md` §4 updated): A0 as recorded (8); A1 = the
  full ladder E ∈ {1,…,5} MWh at 2 h and 4 h at **2025 for all three nodes** (30) and the year
  ladder (2030, 2035) **at the best node and duration only** (20); A2 combinations and staging
  (≈ 6); A3 resolution probe folded into A1's best ladder as unit-step points (≈ 4). ≈ 70
  evaluations, ≈ 7 batches of 10, ≈ 10 h. Phase B single-cohort first (≤ 10 directions per poll,
  one batch per poll), staging checked by one probe at the incumbent; the general lattice only if
  the probe shows staging helps.
- Both author decisions are taken, so nothing blocks the order above; Phase B follows A1 without a
  separate stop, under a frozen spec that names the cost-file hash, the budget, the lattice, the
  configuration and the poll design. Stop for review after A1 (with the timed cycle) and again
  when Phase B terminates or exhausts its budget.

# Addendum 28 — after `P5_15_ADDENDUM27_PHASE_A_REPORT.md` (2026-09-21)

- **Phase A accepted: 68/68 certified; x = 0 minimises F on SRP1** under the corrected costs, the
  C3 ageing calibration and the frozen AA-on configuration. The result is accepted as a finding,
  not as the paper's conclusion — see below. Process deviations accepted: snapshots on for SRP1
  campaigns; concurrency 7; the year ladder at both durations; the A3 anchor.
- **What the objective is.** Value is linear in energy (227.7k ± 3.7k €/MWh against a corrected
  cost of 253.9k), nearly independent of power (51.7k/MVA against 256.3k) and additive across
  nodes (1.000×): at these capacities the storage is a price-taker arbitrageur and the master
  objective is affine within σ_Q (10–18k, 4–7× finer than the provisional figure). Two
  consequences: (i) the optimum under any budget is a corner — x = 0 when value/MWh < cost/MWh,
  the largest budget-feasible 4 h energy at the best node and year when it exceeds it — so Phase B
  is short by nature; (ii) **the paper-scale question reduces to the value per MWh at paper
  scale**, i.e. x = 0, the smallest unit and one larger design: three evaluations, not a campaign.
- **Structural finding (manuscript).** The storage injects at the DN's reference bus; the DN
  interface branch rating is the sum of the ratings of the branches adjacent to that bus
  (`network.py:get_interface_branch_rating`), so the binding element at node 7 is downstream of
  the injection and the storage cannot relieve it; the TN side does not bind. Value at the
  interface node is therefore arbitrage only. **Planner: confirm the branch endpoints (from-bus,
  to-bus) of the binding branch and the storage's bus in one zero-solve line** before this goes in
  the manuscript. It is a consequence of the interface-siting design, to be stated as such, with a
  future-work note on DN-internal siting.
- **The sign is decided by the ageing convention, and that is the paper's finding.** C3 reads the
  10,000-cycle count as cycles to 50 % retention; the datasheet convention C2 reads it as cycles
  to 80 % (`P5_13_B_CYCLING_CALIBRATION.md`: k 35,851 vs 11,542; 15-year retention at 1 EFC/day
  0.858 vs 0.622); C4 (0.70, the manuscript branch's floor) sits between. Since value is
  proportional to available energy, the expert's estimate of the value multiplier relative to C3
  is ≈ 1.23 (C2), ≈ 1.16 (C4), ≈ 1.09 (mid-block health, Advisor caveat 2), ≈ 1.36 (no ageing),
  against ×1.21 needed at the smallest unit and ×1.32 at 5 MWh. Calendar ageing is currently off
  (φ_cal = 1.0); an LFP calendar fade of ≈ 1.5 %/year (φ_cal ≈ 0.985, ≈ 0.80 after 15 years)
  would take back ≈ 10 %. **The calibration's EOL retention is a physical statement about the
  datasheet count, not a policy; the SoH floor is the policy** — C3 conflated them. **Authorized
  (one batch, ≈ 5 evaluations at the smallest node-7 4 h unit; Q(0) is unchanged by the ageing
  model):** C2; C4; C2 with φ_cal = 0.985; C3 with mid-block health; no ageing. Predictions as
  above, recorded before the run; each variant labelled as a model variant, never mixed with the
  baseline in a table. The **author decides the baseline calibration and the floor together**
  (C2/C4 and soh_min 0.50 vs the manuscript's 0.70) with a datasheet citation; if the baseline
  changes, one re-certification check at C\* under AA-on precedes any further campaign.
- **Phase B under the current baseline:** run as the formal record only (the cached neighbours
  make it a few evaluations). If the author's calibration decision makes storage pay, Phase B is
  re-run under the new baseline — that is the demonstration of the method on a non-trivial case.
- **Manuscript:** report the break-even energy cost (176.6k €/MWh at the margin, 197.7k for the
  first unit) with the ageing-sensitivity band as the transferable result; the value-of-storage
  figure with the budget marked; the linearity and additivity; the structural finding; the plan
  under the budget as the corner it is. The results section changes fundamentally versus the
  submission (corrected cost data; the numerical overhaul) and the response letter says so.
- **Paper scale.** The provisional design cannot run on 32 GB (≈ 36 GiB during initialization,
  ≈ 41 h per evaluation). Route: **(a) a ≥ 64 GiB machine for three evaluations** exploiting
  linearity, after (b) the four single-scenario code paths are generalized (they fail loudly above
  one scenario until then — agreed) and row 18 with the author's α is in place; the author says
  what machine is available. Fallback (d): SRP1 results with the caveat, plus the reduced-scenario
  SRP1 variant (5 market × 1 operation; 12–16 GiB, 1 slot) at x = 0 and the smallest unit as the
  uncertainty result — **deferred** until the ageing batch and the machine question are settled;
  expectation recorded: with a non-anticipative storage schedule and value linear in prices, the
  expected value over market scenarios ≈ the value at the mean prices, so uncertainty is not
  expected to close a 20 % gap. Memory work (c): not now. Staging: not wanted.
- **Method document:** §1.4 step costs updated to the corrected file (energy step 126,939); the
  case file's `max_energy_to_power_factor` set to 4 for consistency (master-only parameter; no
  oracle effect).
- **Discount rate (author's question): cannot decide the sign.** Cost is paid in 2025
  undiscounted; value arrives with factors 1 / 0.906 / 0.820 at 2 %, so a zero rate raises the
  PV-weighted available energy from ≈ 2.02 to ≈ 2.21 (+9 %, ≈ +10 % under C2), against the +21 %
  needed; 5 % and 8 % cut value by ≈ 12 % and ≈ 21 %. **Zero-solve item:** from the Phase A
  per-block records, re-weight the per-representative-year components of Q(x) − Q(0) at the
  smallest node-7 4 h unit for 0/2/5/8 % and report value-to-cost at each (this is also the R3.6
  discount-rate row); report the average daily price spread in the SRP1 market data against the
  captured spread implied by the value per MWh (≈ 55–60 €/MWh per full cycle). The rate stays at
  2 % and is not tuned toward a result.
- **Order:** branch-endpoint confirmation and φ_cal/SoH trajectory report (zero solves) → ageing
  batch → Phase B formal record → (b) code-path generalization only once the machine is known.
  Stop for review after the ageing batch.

# Addendum 29 — memory (author's observation of 1.5 GB swap at concurrency 7, 2026-09-21)

- **SRP1 concurrency: 5.** Rule: the largest concurrency with no steady-state swap growth (7 × 2.3–
  2.55 GiB plus parent and OS reached swap; the Planner measured no per-cycle slowdown, but the
  remaining SRP1 work is small, so the margin is free). Effective for the ageing batch and the
  Phase B record.
- **Paper scale is one process, so concurrency does not help it.** Its 36 GiB during initialization
  is 18 GiB of models plus +0.22 GiB on each block's first solve. Route (c) reopened as a **bounded
  profiling task** (after the ageing batch; machine alone): resident memory before and after the
  first and second solve of the same block, with `load_solutions` on/off and with NL symbol maps
  and solution objects cleared after value extraction; classify the growth as one-time
  bookkeeping, inherent IPOPT/NL workspace, or a per-solve leak (a second-solve increment is a
  leak and is fixed regardless of machine). Fix what is bookkeeping; gate: bitwise identity on two
  SRP1 cycles. Then re-measure build + initialization at paper scale and time one cycle.
- **Decision rule (recorded before the run):** resident after initialization ≤ 26 GiB, flat across
  cycles, cycle ≤ 25 min → the three paper-scale evaluations (x = 0, smallest unit, one larger
  design; ≈ 40 h each, ≈ 5 days serial, machine dedicated) run on this machine after the
  code-path generalization and row 18 with the author's α; otherwise a ≥ 64 GiB machine, or the
  SRP1-with-caveat fallback with the reduced-scenario variant.
- **Order:** as Addendum 28, then the memory task. Stop for review after the ageing batch and
  again after the memory task.

# Addendum 30 — after `P5_15_ADDENDUM28_AGEING_REPORT.md` (2026-09-21)

- **Result accepted: the ageing convention does not decide the sign.** Five variants at the
  smallest node-7 4 h unit, all certified: C2 −23k (indeterminate by bars, negative vs σ_Q), C4
  −41k, C2 + fade −44k, mid-block −54k, no ageing +485 (break-even). Larger units stay clearly
  negative under every variant. **Addendum 28's reading is withdrawn.**
- **Expert's prediction failure, recorded.** The available-energy ratios were right (1.219 vs
  1.23; 1.372 vs 1.36); "value ∝ available energy" was wrong. At fixed power the extra energy is
  cycled into lower-spread hours (elasticity 0.60–0.62; value per available MWh 359k → 318k), and
  the mid-block convention is offset endogenously by faster wear (EFC 1.06 vs 1.01). A static
  estimate cannot see the dispatch response; the mechanism table (`p515_s46_ageing_mechanism.py`)
  is a manuscript item.
- **Baseline (expert's recommendation; author decides):** C2 (10,000 cycles at 0.80 DoD to 0.80
  retention — datasheet convention) + calendar fade φ_cal = 0.985 (0.80 at the 15-year calendar
  life) + soh_min 0.70 (the manuscript's floor), with citations recorded in the spec. Under this
  set the SoH reaches ≈ 0.68 at the horizon, so the floor is expected to bind in the last block —
  the degradation-aware constraint live, to be shown; it is an unrun variant (the batch used 0.50
  throughout). Cost of switching ≈ 40 evaluations (re-certification at C\* and the smallest unit;
  A1a re-run for the figure; Phase B record), ≈ 9 h at concurrency 5. Alternative: keep C3/0.50
  with the sensitivity band as the defence, at zero cost. The C3 results become the sensitivity
  set either way and never share a table with the baseline.
- **Phase B:** formal record under the chosen baseline, after the ladders.
- **Manuscript items:** the break-even energy cost with the ageing band as measured; the
  elasticity mechanism; the discount curve (0.905 / 0.823 / 0.726 / 0.651 of cost at 0/2/5/8 %);
  the captured spread (52.1 €/MWh per full cycle against 98.0 available; round-trip efficiency
  0.93 and degradation-limited cycling explain the ratio); the structural finding in Z1's wording
  — the storage sits at the upstream terminal (DN bus 1) of the single constrained branch 1–2, its
  injection enters the interface balance, not the branch — with the correction that it is
  **active** in the binding slots (≥ 99 % of rating in 2 of 23) but cannot change the constrained
  flow; future work: DN-internal siting. The headline: at the interface node on this instance,
  shared storage has arbitrage value only, falls 8–20 % short of break-even at corrected 2025
  costs depending on the ageing convention, and a non-degrading battery only breaks even.
- **Order:** baseline decision → re-certification checks → A1a re-run → Phase B record → memory
  task (Addendum 29) alone. Stop for review after the Phase B record and after the memory task.

# Addendum 31 — market uncertainty (author's question, 2026-09-21)

- **SRP1 carries no uncertainty** (one market, one operation scenario); the paper instance has
  5 × 5 inside each block. **In this model uncertainty cannot improve the case:** the storage
  schedule is scenario-independent (non-anticipative; the DSO pays per-scenario interface
  deviations, row 18), and for a common schedule the arbitrage revenue is linear in each
  scenario's prices, so the expected value over market scenarios equals the value under the
  probability-weighted **mean price profile**, whose daily spread is at most the average of the
  scenarios' spreads (misaligned peaks flatten it). Option value requires scenario-dependent
  recourse dispatch — a model change (per-scenario ESS consensus channel), future work, not this
  revision. Operation scenarios do not change this: a common schedule shifts the contracted
  interface and every scenario equally, leaving deviations unchanged.
- **What can change the case at paper scale is the level, not the dispersion.** Zero-solve check
  before the memory task: the daily 4 h spread (Z4's definition) of each of the five market
  scenarios and of their weighted mean profile; which scenario or average SRP1's single scenario
  corresponds to; the ratio of the mean-profile spread to SRP1's 98 €/MWh is the first-order
  estimate of the paper-scale value per MWh relative to SRP1's, recorded as a prediction for the
  paper-scale evaluations.
- **Manuscript item:** one sentence stating that the non-anticipative schedule makes the
  stochastic value the mean-price value, with recourse dispatch as future work.
- **Author's sanity check (cycle economics), recorded.** €1M over 15 × 365 days is €183/day;
  €1M buys ≈ 3.1 MWh at 4 h, so ≈ €59/day per rated MWh, i.e. ≈ €60–65 per MWh per daily cycle
  after discounting and health loss. The smallest unit cycles 1.0–1.15 EFC/day, the floor is
  slack, and each cycle earns €52.1/MWh: 52 × 1.01 × 5,475 ≈ €288k undiscounted against a cost of
  €318k — short by ≈ 10 % before discounting and ageing, ≈ 20 % after. The math is consistent; the
  gap is the captured margin. **Zero-solve item:** why €52 against the €98 market spread —
  efficiency explains ≈ €7; at x = 0 report the daily 4 h spread of the TSO's nodal marginal cost at
  bus 7 (dual of the bus-7 balance in the certified TSO blocks) and how the market price scenario
  enters the TSO objective; attribute Q(0) − Q(x) at the smallest unit between TSO generation
  cost and each DSO component. If the storage sees the TSO marginal-cost spread rather than the
  market spread, the paper-scale prediction above is restated on that quantity. Manuscript
  discussion: at ≈ €320k/MWh installed, 15 years and one cycle a day, break-even needs ≈ €60 per
  MWh-cycle; pure arbitrage rarely clears it, and the interface siting forgoes the stacked network
  value that usually does.

# Addendum 32 — after `P5_15_ADDENDUM30_PHASE_B_REPORT.md` (2026-09-22)

- **Accepted.** Baseline set and read back; C\* certifies in 87 cycles under it; the 0.70 floor
  binds in 2035 everywhere (the degradation-aware constraint is live); all 30 ladder points and the
  14-point unit poll certified; x = 0 minimises F with a certificate that holds; break-even
  182.2k €/MWh at the margin, 195–200k for the first unit. Sixth bitwise reproduction. The C3 set
  is the sensitivity set and appears in no baseline table. The "four orders of magnitude" wording
  is withdrawn for §5's sentence (signal ≈ 4 × 10⁻⁴ of system cost, resolvable because the bar
  and the polish gap hold the cost to ≈ 2 × 10⁻⁵).
- **Q1 — citations (expert's proposal; author confirms).** Datasheets found: EVE MB31 314 Ah LFP
  — 8,000 full cycles to 70 % SOH at 0.5P, 25 °C, i.e. (8000, 1.0, 0.70) → k = 22,430, which is
  C4's k exactly and gives the 0.70 floor its citation; Hithium 314 Ah — 13,000+ cycles to
  65 % SOH → k ≈ 30,200. The baseline k = 35,851 (10,000 cycles at 0.80 DoD to 0.80) sits at the
  **favourable** end of the datasheet range, which strengthens a negative result: storage does not
  pay even under an optimistic cycle life; C4 is the datasheet-exact sensitivity. **No re-run.**
  Calendar fade φ = 0.985 (0.80 at the 15-year calendar life): a model assumption, cited to LFP
  calendar-ageing studies (candidates for the author to verify: Naumann et al., J. Energy Storage
  2018; Keil et al., J. Electrochem. Soc. 2016). Spec v17's "to be supplied" fields are filled by
  the author.
- **Q2 — Phase B at x = 0.** The completion ruling is sound and stays: a finite set of mesh
  points is a legal MADS **search step**, so polling every feasible unit neighbour at unit mesh is
  within the framework and gives the certificate its meaning; no reparametrization for this
  revision (recorded as an implementation refinement — snap-to-feasible poll points — for the
  method description, since the affine objective made Phase B trivial here).
- **Q3 — NOMAD.** Authorized only in a **separate** environment, never the canonical one, for the
  stub check of directions and mesh (STEP4 §6 (ii)); time-capped at one bounded Worker task. If it
  does not build on Apple silicon within that, record "not done" and the manuscript describes the
  in-house OrthoMADS with the algorithm citations, not NOMAD.
- **Q4 — one terminal TSO capture: yes.** One solve-bearing capture at x = 0 (a 2030 day, where
  the marginal-cost/market ratio is 0.76): which generator or constraint sets the bus-7 marginal
  cost in the top-4 and bottom-4 hours against the market price — the physical explanation of the
  −12.3 €/MWh flatness term for the manuscript.
- **Q5 — memory task proceeds next**, machine alone, under spec v16's decision rule.
- **Q6 — negative penalty levels: zero-solve check, yes.** A penalty on a slack cannot be negative
  unless the slack is unbounded below or the report signs it as a credit; read the slack bounds and
  values in the persisted models and classify (reporting convention vs unbounded slack). An
  unbounded slack would bias absolute Q, not differences, and is fixed before any absolute cost is
  quoted.
- **Recorded for the paper-scale evaluations:** prediction R = 0.937 on the market spread (mean
  profile 91.7 vs SRP1 98.0; SRP1's scenario = paper scenario 1 in 2025), with the caveat that the
  storage sees the bus-7 marginal cost (spread 80.6, 0.823 of market) — the restated ratio needs
  paper-scale solves. Attribution at the smallest unit: 60 % TSO generation, 40 % DSO flexibility
  across all three DNs.
- **Order:** citations (author) → Q6 check and the TSO capture (zero/one solve, alongside) →
  memory task alone → stop for review. NOMAD stub task when the machine is otherwise idle.

# Addendum 33 — flexibility price and siting (author's question, 2026-09-22)

- **Facts from the case data:** `cost_flex` is an hourly profile ≈ 25–55 €/MWh (growth factors
  applied), charged on the DSOs' downward P **and** Q flexibility (`flexibility_cost` in
  `model_construction_helpers.py`). It is cheap relative to the 98 €/MWh energy spread, so DSO
  flexibility competes with storage for the same arbitrage, flattens the interface marginal cost
  (part of the −12.3 €/MWh term) and is why 40 % of the storage's value appears as displaced
  flexibility in all three DNs.
- **Raising the price would raise the value, with a ceiling:** less flexibility at x = 0 widens
  the marginal-cost spread and each shifted MWh displaces a dearer alternative, but once
  flexibility is priced above the spread the DSOs stop using it and the storage's value converges
  to pure arbitrage on the TSO marginal cost — at most ≈ 65–80 €/MWh per cycle against 60.4 now
  (+10–30 %): near break-even for the smallest unit (needs +23 %), not for the marginal MWh
  (+39 %). **No flexibility-price change is authorized without a data anchor** (real DSO
  congestion-management prices) supplied by the author; a price raised until storage pays reads
  as tuning.
- **The physical lever is siting.** The ESS is modelled at DN bus 1 (HV side of the interface
  transformer); a shared BESS at an interface substation is normally connected at the MV busbar
  (bus 2), downstream of the transformer, where it relieves the binding branch, sees the DN-side
  locational price and displaces congestion-relief flexibility. **Model variant, Advisor review
  before implementation:** ESS at DN bus 2; the TSO's explicit ESS term at bus 7 removed (the
  injection enters the interface flow), TSO coupled through the interface only; ESS consensus
  between DSO and ESSO. Gate: x = 0 unchanged bitwise. Two evaluations under the baseline: the
  smallest node-7 4 h unit and 3 MWh / 0.75 MVA. **Prediction:** value at the smallest unit rises
  by ≈ €50–100k (0.25 MW over the binding periods at ≈ 45 €/MWh plus the locational price
  effect); the sign of value − I is open. If storage pays at the MV busbar and not at the HV side,
  that contrast is the paper's finding on where the value of shared storage comes from.
- **Zero-solve first:** at x = 0 and at the smallest unit, split each DSO's flexibility cost into
  node-7 binding-period slots (congestion relief) and the rest (economic shifting); report the
  applied `cost_flex` profile; note the equal pricing of P and Q flexibility.
- **Order:** zero-solve split alongside the running order → Advisor review of the variant →
  variant build and gate → two evaluations, after the memory task (machine availability).

# Addendum 34 — siting settled; voltage channel; flexibility-price break-even (author, 2026-09-22)

- **Siting variant withdrawn.** The interface transformer is DSO-operated, so a shared ESS on
  its MV side would breach unbundling; the asset stays at the HV busbar (DN bus 1). Recorded as
  the manuscript's explanation of why the storage has no active-power network value on the DN
  side. Addendum 33's zero-solve flexibility split stays.
- **Reactive channel (zero solves):** the storage's Q costs no degradation and, injected at the HV
  busbar, sets the DN's slack voltage; DSOs pay `cost_flex` for reactive flexibility too. At the
  smallest node-7 unit under the baseline: split the storage's value and each DSO's flexibility
  saving into P and Q components; the storage's Q dispatch as a fraction of rating; interface
  voltage-bound activity (entries at 1.1 pu) at x = 0 and with the unit. If Q is idle, voltage
  support is unmonetized because nothing binds — network value exists only where a constraint
  binds, and this system is unstressed.
- **Stress is not applied by hand** (it would read as manufacturing the result). The legitimate
  stressed instance is paper scale: later years with load/RES growth and high-RES operation
  scenarios (reverse flows, DN voltages up) — one more reason the memory task matters.
- **Flexibility price as a break-even axis** (author's choice of lever), after the memory task:
  uniform multipliers 1.5 / 2 / 3 on the `cost_flex` profile via harness override (no workbook
  edit); each level evaluated at x = 0 (Q(0) changes with the price) and at the smallest unit —
  six evaluations. Reported as "the flexibility price at which the smallest unit pays, if any",
  with the ceiling stated (once flexibility exceeds the spread the DSOs stop using it; value
  converges to pure arbitrage on the TSO marginal cost, ≈ 65–80 €/MWh per cycle vs 60.4).
  **Predictions:** smallest unit indeterminate between 2× and 3×; the marginal MWh pays at no
  level. The growth-rate variant is not preferred: it back-loads the increase into the years of
  lowest PV weight and health.
- **Flexibility pricing convention (author's question):** P and Q downward flexibility are
  **already** priced by the same curve (`flexibility_cost`: `c_flex[p]·baseMVA·(flex_p_down +
  flex_q_down)`; upward flexibility free). Reactive priced at the active curve overprices Q by
  about an order of magnitude against real reactive services, which suppresses DSO Q-flexibility
  and inflates the storage's Q-derived value. Not a lever to raise value; a convention to state
  in the manuscript. **Conditional sensitivity:** if the P/Q split shows a material Q share, one
  evaluation pair (x = 0 and the smallest unit) with Q flexibility priced at 10–20 % of P — the
  robustness check a reviewer would ask for; expected to lower, not raise, the storage's value.
- **Order:** P/Q split alongside → memory task → flexibility ladder.

# Addendum 35 — after `P5_15_ADDENDUM32_34_REPORT.md` (2026-09-22)

- **Accepted in full.** Memory fix (bookkeeping, switch off by default, bitwise gate), no leak;
  paper-scale initialization completed and σ calibration passed; NOMAD verification done in a
  separate environment; negative penalty levels are IPOPT bound-relaxation residue (bias ≤ 2e-5 on
  Q, ≤ 181 € on values) — a third category, no defect; reactive flexibility is structurally
  absent (bounds pinned) and the storage's value is 97 % active — the manuscript must not claim
  reactive flexibility, and voltage support earns nothing because no limit binds at bus 7.
- **Expert's predictions refuted, recorded:** break-even lies between ×1.5 and ×2 (not ×2–×3);
  the value ceiling (65–80 €/MWh per cycle) was exceeded (86.5 at ×2, 102.9 at ×3) because DSO
  flexibility use is **structural**, not optional — the DSOs keep buying at higher prices and each
  displaced MWh is dearer. The mechanism proof supersedes the ceiling argument.
- **The price mechanism is the paper's central mechanism.** In the expensive hours the DSOs shed
  imports, the TN is covered by its zero-cost RES with every conventional unit at zero, and the
  bus-7 price is the DSOs' flexibility shadow price (daily balance dual μ plus the hourly price
  where P-down is marginal) — demand-side flexibility sets the price on the vertical step of the
  supply curve. The shared storage therefore competes with, and is valued at, the distributed
  demand flexibility it displaces; it pays when that flexibility costs about the market's daily
  spread (break-even at m ≈ 2: mean 96.7 vs spread 98.0 €/MWh). Manuscript: the flatness
  decomposition (−6.68 capped peaks, −5.25 hour reselection, −0.39 losses), the hour-by-hour
  identity, and the sentence above.
- **Flexibility framing (Q2):** the ladder is reported as a **break-even axis** — "the flexibility
  price at which the smallest unit pays" — with the different-systems caveat (Q(0) +34 % at ×3);
  the baseline price is not changed. If the author finds a data anchor for real DSO
  congestion-management prices, the paper adds one sentence locating the break-even within the
  observed range; without it, nothing is claimed about realism.
- **Q3 — the congestion/shifting dichotomy is dropped.** The node-7 rating binds in cheap hours
  as an *import limit on shifting* (unpriced P-up into those hours), not as peak congestion;
  report the mechanism as measured and describe the flexibility cost as the priced down-leg.
- **Q4 — marginal-MWh test authorized:** node 7, 4 h, 2 MWh at ×2 and ×3 (2 evaluations).
  Prediction: at ×2 the marginal MWh pays (value/MWh ≈ 300k > 254k) and the optimum under the
  budget is the budget corner. **If confirmed, scenario F2** (m = 2, labelled a flexibility-price
  scenario, never the baseline) becomes the **demonstration case for the planning method**: the
  node-7 4 h ladder under F2 (5 points), then Phase B under the budget from the best point (≈ 20–30
  evaluations, ≈ 6–8 h at concurrency 5) — the paper's non-trivial optimal plan, obtained by the
  full machinery, with the baseline plan (x = 0) beside it.
- **Q5 — STEP4 §5.2 corrected** (in-house double/halve with Halton directions; NOMAD 4.6 uses a
  1-2-5 ladder and random directions; direction/rounding/bound machinery verified equal on 144
  polls). The signal-size sentence is §5's. **Q6 — harness task authorized** (per-event solve
  identity in the scale script), and the run needs an explicit **unrecovered-failure policy**
  (continue with the last iterate, count, no unrecovered failure in the 10 certifying cycles)
  before any paper-scale evaluation.
- **Q1 — paper-scale route (author decides; expert's recommendation):** the rule failed on time
  (42.3 min/cycle, ≈ 78 h per evaluation; CPU-bound, so a bigger machine helps only by freeing the
  Mac). Options: **(d) full instance, serial, two evaluations only** — x = 0 and the smallest unit,
  ≈ 6.5 days with the machine dedicated, the third (larger design) dropped because linearity is
  established; the mean-profile argument (R = 0.937) is the analytical backup; **(c)** a reduced
  paper instance (3 market × 3 operation scenarios, cycle ≈ 15 min, ≈ 28 h per evaluation, ≈ 2.5
  days for two) with the reduction stated; **(a)** a ≥ 64 GiB Linux server if available — same
  78 h, but the Mac stays free for the F2 demonstration and Step 5; **(e)** a within-cycle parallel
  evaluator returning arrays, not models (blocks partitioned across 8 workers; ≈ 3–4× on the
  cycle) — bounded engineering with a hard gate, only if none of the above is acceptable.
  Recommendation: (a) if a server exists, else (d); (c) if a week of the Mac is not acceptable.
- **Step 5 list (for the next stop):** the coordination-benefit result — uncoordinated dispatch
  vs ADMM-coordinated at x = 0 (the paper's other main claim; a few SRP1 evaluations after the
  `_add_dso_scenario_deviation_penalty` guard fix); R3.6 rows: discount (done, zero-solve),
  soh_min (0.50 vs 0.70 under C2 + fade, one evaluation), salvage (zero-solve add-back), α
  (paper scale only); certification statistics for the method section.
- **Order:** harness task + unrecovered-failure policy → marginal-MWh test (2) → F2 demonstration
  if confirmed → paper-scale evaluations per the author's route choice (machine dedicated) →
  Step 5. Stop for review after the marginal test and F2.

# Addendum 36 — multi-scenario pilot (author, 2026-09-22)

- **Pilot before any paper-scale evaluation:** the paper's structure — 5 representative years,
  4 days, row 18 active with **α = 0.50** (author's choice; conservative side of imbalance
  pricing; the R3.6 sensitivity {0.25, 0.5, 1.0} still runs on the multi-scenario instance; the
  paper-scale run uses the same α) — at **2 market × 2 operation scenarios**; baseline ageing,
  AA-on, campaign cost file and budget. Blocks ≈ 4× SRP1's; expected cycle of a few minutes,
  evaluation ≈ 8–10 h.
- **Prerequisite:** generalize the four single-scenario code paths (hull-polish helpers,
  settlement reporting, `p56a_oracle.load_baseline`, `run_admm_arm` solve count); **gate: bitwise
  two-cycle reproduction on SRP1** after the change.
- **Runs:** x = 0 and the smallest node-7 4 h unit, concurrency 2; certification under the
  10-cycle bar; hull polish on both; two-cycle bitwise reproduction of one; full results output
  for the manuscript (per-scenario costs and interface profiles, interface dispersion under soft
  non-anticipativity, ESS dispatch and SoH trajectory, settlement, penalty-table components,
  Excel writers).
- **Report:** cycle time and memory (second point on the block-size scaling law, to price the
  3 × 3 route); value per MWh vs SRP1 and vs the R = 0.937 prediction; anything the
  generalization changed.
- **Order:** marginal-MWh test now (SRP1, while the Worker generalizes) → generalization + SRP1
  gate → pilot → F2 demonstration if confirmed → paper-scale route decision on the pilot's
  timing → paper-scale run → Step 5. Stop for review after the pilot.

# Addendum 37 — row 18 implementation rulings (expert; author confirms, 2026-09-22)

- **Implement row 18 as signed** (Addendum 10/11): a linear charge on each DSO's per-scenario
  interface deviation, priced at α·π̄, inside Q(x) as an economic (E) term — a proxy for the
  balancing cost the deviation causes, not a transfer (no TSO receipt); no TSO term; no V term;
  the shared-ESS schedule hard non-anticipative. The existing quadratic pin (9e4 on V and
  interface P/Q, 1e4 on storage, both sides, outside Q) is **unwired, not deleted**; at one
  scenario it was vacuous, so every SRP1 measurement is unaffected — the SRP1 bitwise gate
  proves it. The marginal-MWh result (pays at ×2; F2 confirmed) and the generalization gate are
  accepted.
- **(i) Hourly premium.** $c_t = \alpha\,\bar\pi_t$ with $\bar\pi_t$ the probability-weighted mean
  over market scenarios of the hour's price (per representative year and day). A daily constant
  would misprice deviations in cheap versus dear hours and invite deviating where energy is cheap.
- **(ii) Deviation against the coupled expectation.** $d_{s,t} = p^{\mathrm{int}}_{s,t} - \bar p_t$
  with $\bar p_t = \sum_s \omega_s\, p^{\mathrm{int}}_{s,t}$ a variable of the DSO block (its own
  day-ahead expectation); likewise for Q. Never against the incoming consensus value — that
  would price movement relative to an ADMM iterate, the defect removed in row 3. Charge
  $\sum_{s,t} \omega_s\, c_t\,(d^+_{s,t} + d^-_{s,t})$ with $d = d^+ - d^-$, $d^\pm \ge 0$ (LP
  form; no |·| in the NLP).
- **(iii) Drop the TSO term; keep per-scenario consensus.** The TSO serves each scenario's actual
  interface flow through the per-scenario consensus channels and decides nothing about the
  deviation, so it carries no deviation cost. The quadratic "regularisation" it loses was
  identically zero in every convergence measurement (one scenario), so nothing measured depended
  on it; if the multi-scenario ADMM needs help, that is a ρ matter for the PF channel, not a
  penalty.
- **(iv) Storage hard non-anticipativity = one variable, not copies.** In each network block the
  shared-ESS P and Q at the interface node are a single scenario-free variable per period,
  referenced in every scenario's balance rows — no per-scenario copies tied by equalities (the
  duplicated-row dual non-identifiability found in Step 3 must not be reintroduced). The ESSO has
  no scenario index; the ESS consensus channel is unchanged.
- **Gates:** SRP1 two-cycle bitwise identity (row 18 vacuous at one scenario); at 2 × 2, the
  charge and the dispersion reported; a two-cycle limit check with α → large reproducing the
  pinned behaviour (dispersion → 0). **Dispersion metric:** per DSO, RMS and max of $d_{s,t}$
  over scenarios and hours, in MW and as a share of the mean interface flow, plus the total
  charge; the R3.6 row is α ∈ {0.25, 0.5, 1.0} with α = 0 as the free-deviation reference.
- The Advisor's review is reconciled against these rulings; any disagreement comes back with
  the reason before implementation. The 2 × 2 smoke test under the current quadratic form is
  accepted as a code-path check only; its dispersion is not a result.

# Addendum 38 — after `P5_15_ADDENDUM35_37_REPORT.md` (2026-09-22)

- **Accepted:** the marginal MWh pays at ×2 (second MWh 360,723 vs 317,957; value slightly
  concave; the F2 optimum is expected at the budget corner, 3 MWh, which the F2 ladder tests);
  F2 confirmed; the four code paths generalized and gated (SRP1 bitwise; 2 × 2 smoke 8/8); the
  unrecovered-failure policy verified as production behaviour; the per-event solve identity fixed.
  Third scaling point: 3.3 s per DSO block at 2 × 2 → ≈ 3 min/cycle, 7–10 h per pilot evaluation,
  ≈ 3.9 GiB.
- **Addendum 37 (iii)'s premise was wrong:** there are no per-scenario consensus channels; both
  sides couple only the commitments, and each side's per-scenario deviation is its own variable
  within ±rating. Corrected as follows.
- **(A) TSO pinned.** The TSO's per-scenario interface deviations are fixed at zero: it operates
  on the committed schedule in every scenario; the DSO's deviations are settled against the
  market at the scenario price plus the premium — **day-ahead commitment with imbalance
  settlement**, the signed intent of soft non-anticipativity. Per-scenario consensus channels
  (the TSO re-dispatching to each scenario's actual flow) are the exact alternative, rejected for
  this revision (×25 channels, convergence at paper scale unknown) and recorded as future work.
- **(B) Storage reconciliation confirmed:** one scenario-free variable per period referenced in
  every scenario's balance rows; per-scenario variables retained unwired (fixtures loadable);
  non-anticipativity on charge and discharge separately (throughput drives degradation).
- **(C) Deviation energy enters Q(x).** With the TSO pinned the settlement no longer cancels; its
  residual is the price–deviation covariance, i.e. the energy the DSO deviated by valued at the
  scenario price (consistent with market-priced TN generation). Split the settlement: the
  contracted part is a transfer and cancels; the deviation part is economic and enters Q(x) with
  the premium. Addendum 13's cancellation gate becomes "residual = covariance term by
  construction"; both reported.
- **(D) Voltage pin kept** as a solver-only term excluded from Q(x) — hard non-anticipativity on
  the interface voltage, what a TSO holding a substation setpoint does; the per-scenario
  mismatch is reported; low-stakes here (no voltage limit binds at node 7).
- **Also:** the single-block A/B at α ∈ {0, 0.5, large} before the pilot (dispersion is a
  threshold outcome: report α·π̄ against each DSO's internal flexibility price); a premium floor
  only if any hour's mean price is non-positive; the manuscript describes the commitment-plus-
  imbalance structure, the DSO-side settlement and the pinned TSO. Gates as in Addendum 37.
- **Sequencing:** F2 runs now on SRP1 while row 18 is built; then the pilot at α = 0.50.

# Addendum 39 — after `P5_15_ADDENDUM38_REPORT.md` (2026-09-23)

- **Accepted.** F2: every marginal MWh pays (values not monotone — the 4th and 5th MWh worth
  more than the 3rd; report as measured); Phase B left the 2025 budget corner for a **2030
  two-node plan** (0.25 / 0.5 at node 5 + 1.0 / 3.5 at node 7; I = 992,268; value − I +228,304,
  +102,117 over the corner at 5.5× resolution) — the paper's demonstration that the method finds
  what a ladder cannot, with the timing reversal explained (discounting and the cost decline buy
  more capacity in 2030; under the ×2 price that outweighs investing earlier, whereas under the
  baseline price the year ladder ran the other way). Both corner predictions refuted, recorded.
  Row 18 built to the rulings and gated (SRP1 bitwise; α limit checks; settlement split 1e-12;
  no floor needed); inert at one scenario, so every committed result stands.
- **Ruling 1 — the F2 certificate is worth having, and it is cheaper than 61 evaluations.** The
  completion rule (every feasible point within one unit step, in the ∞-norm box) was a remedy for
  x = 0, where every poll direction was infeasible; at an interior incumbent the MADS certificate
  is **poll failure over a positive spanning set at unit mesh**, not the full box. Continue from
  the incumbent with the standard unit poll: the 2n OrthoMADS directions rounded to the lattice,
  infeasible ones snapped to the nearest feasible lattice point, the completion set added only if
  fewer than n + 1 feasible poll points remain — ≤ 18 evaluations per poll, ≈ 4–5 h at
  concurrency 5. If a poll succeeds, iterate; cap the continuation at 60 evaluations and report
  either the certificate or the budget exhaustion honestly. The cap of 30 on the completion set
  stays.
- **Ruling 2 — locate the threshold, then bracket it.** At α = 0.50 dispersion is ≈ 0 for a priced
  reason (unpriced upward flexibility makes holding the schedule cheaper than `cost_flex`
  suggests — a convention to state in the manuscript). Single-block test first at α ∈ {0, 0.01,
  0.02, 0.05, 0.1, 0.25} to find α\* where dispersion vanishes; the R3.6 row is then {0, α\*/2,
  α\*, 2α\*, 0.5}, so the sensitivity shows the transition. The pilot's baseline stays α = 0.50:
  the regime in which DSOs honour their day-ahead commitment, and the paper's statement is "at
  premiums above α\* the DSOs hold their schedules; dispersion appears only below it".
- **Sequencing:** pilot first (its timing decides the paper-scale route); the F2 unit poll may
  run concurrently at concurrency 3 if the campaign harness and memory allow (2 × 3.9 + 3 × 2.2
  GiB), otherwise after the pilot. Stop for review after the pilot, with the F2 poll attached if
  done.
- **Models (author's question):** Claude Opus 5.5 (`claude-opus-5-5`) is current. If the `opus`
  alias already resolves to it, nothing changes; if the Planner reports Opus 5, pin
  `model: claude-opus-5-5` for the Worker now (next delegation) and for the Planner at its next
  restart; the Advisor stays on Fable 5.1. Model changes are safe at task boundaries; the
  bitwise gates protect the numerics; record the switch by name and date.

# Addendum 40 — after `P5_15_ADDENDUM39_PILOT_REPORT.md` (2026-09-24)

- **Accepted.** Both pilot evaluations certified (74 / 72 cycles, 4 h for the pair); **R = 0.937
  confirmed at 0.942** — the mean-profile argument transfers, which licenses SRP1 results with a
  stated scenario caveat wherever a larger instance is unaffordable; storage does not pay at
  2 × 2 (−73,636, determinate); two scenario-indexing bugs in the manuscript outputs caught by
  the pre-run audit and gated. **Addendum 39's premise is withdrawn:** α = 0.50 is a dispersing
  regime (16 % of mean flow at node 5, peak 34.5 MW); the single-block α\* = 0.1 was an artefact
  of the initialisation economy (unpriced import, RES curtailed at a flat 1 €/MWh) and the
  "unpriced upward flexibility" explanation is refuted — neither reaches the manuscript.
- **The mechanism under coordination is the right one and is kept as the baseline's story:** at
  α = 0 the deviation is 91 % market-price arbitrage (the DSO shifts import between price
  scenarios and earns the covariance); the premium prices that away (market part ÷300,
  covariance ÷140 at α = 0.5) while the physical flexibility volumes are unchanged; the residual
  dispersion at α = 0.5 is the operation scenarios differing physically — the recourse a DSO
  should exercise and pay for. §5's manuscript paragraph is **confirmed as proposed**.
  **Methodological footnote recorded:** α thresholds cannot be read from unsettled arms (the
  coordination gap prices deviation at ≈ 225 €/MWh at cycle 2 against a premium of 5–10); only
  certified sweeps count.
- **Ruling 1 — R3.6 α row on a settled sweep:** α ∈ {0, 0.1, 0.25, 0.5, 1.0} at x = 0 on the
  2 × 2 instance, certified (≈ 5–6 h at concurrency 2), reporting per α: dispersion (RMS share of
  mean flow, peak), the arbitrage/operation split of Σωd², the covariance earned, the row-18
  charge, and Q(0) — the "cost of commitment" curve. Run **after** ruling 2.
- **Ruling 2 — fix row 18 at initialisation:** inactive at the initialisation solve, activated
  with the settlement weight (the Advisor's minimal fix); inert at one scenario, SRP1 bitwise gate
  required. It is a configuration change for multi-scenario runs, so the pilot's α > 0
  artifacts are superseded: the α row re-runs α = 0.5 at x = 0, and the smallest unit at
  α = 0.5 is re-run once (≈ 2 h) so the pilot pair stays consistent; the R comparison is
  re-stated on the re-run.
- **Ruling 3 — paper-scale route (expert's recommendation; author decides):** make **3 × 3 the
  paper's multi-scenario instance** (≈ 9 min/cycle, ≈ 16 h per evaluation, ≈ 8 GiB, concurrency
  2 → the headline pair x = 0 and the smallest unit in about a day), with a recorded selection
  rule — the three market scenarios whose renormalized mean profile has the 4 h spread closest to
  the five-scenario mean (91.7 €/MWh) and three operation scenarios spanning low / mid / high
  RES — and the R prediction for that set computed zero-solve and recorded **before** the run.
  The 5 × 5 pair (≈ 78 h each) is optional at the end, as a confirmation row, if the machine can
  be spared for a week; the R result makes it a confirmation, not a requirement. The α row stays
  at 2 × 2. The manuscript states the reduction and cites the confirmed R prediction as the
  reason it is safe.
- **Ruling 4 — F2 certificate runs next** (SRP1; standard unit poll with snap-to-feasible; cap
  60), before the init fix, since it needs none of the multi-scenario code.
- **Next design item for Step 5:** the **uncoordinated benchmark** must be defined before it is
  run — what each agent assumes about the interface, how the resulting operating point is made
  consistent and costed, and that its economy (α = 0, unpriced import) differs from the
  coordinated one — Advisor review of the definition first.
- **Order:** F2 certificate → init fix + SRP1 gate → α row (2 × 2, x = 0) + the unit at α = 0.5
  → 3 × 3 selection rule and R prediction (zero solves) → author's route confirmation → 3 × 3 pair
  → uncoordinated benchmark definition → Step 5. Stop for review after the α row with the 3 × 3
  prediction attached.

# Addendum 41 — renewable curtailment cost (author's question, 2026-09-24)

- **Principle.** In a system-cost objective curtailment is priced endogenously at its
  replacement cost (extra conventional generation in the TN, extra priced imports in the DN); an
  explicit penalty beyond a tie-breaker double-counts unless it has an institutional basis. The
  basis exists: non-market-based redispatching, including RES curtailment for grid reasons, must
  be financially compensated by the ordering system operator under Regulation (EU) 2019/943,
  Art. 13 (paragraph to be verified by the author) — from the operators' perspective curtailment
  costs ≈ the market price of the curtailed energy. Data-anchored, not tuning.
- **Effect on storage is empirical:** a curtailment cost acts only where curtailment occurs at
  the optimum. On SRP1 the evidence says it does not (TN demand on the RES plateau, curtailment
  ruled out as the cause of price flatness, DN interface binding on import); it could appear in
  high-RES operation scenarios, where an HV-busbar storage earns the cost by absorbing TN-side
  surplus (not DN-side surplus behind the transformer).
- **Zero-solve read first** (before the α row and the 3 × 3 selection, so nothing is evaluated
  twice): curtailed RES energy per network, hour and scenario at x = 0 and at the smallest unit,
  on SRP1 and on the 2 × 2 pilot, with cause where identifiable and the applied penalty.
- **Decision rule, recorded:** below resolution everywhere → the 1 €/MWh tie-breaker stays,
  stated in the manuscript with the measured volumes; material in any scenario the storage's bus
  can reach → the baseline moves to curtailment priced at the hourly market price in both
  networks, as a data parameter, with an SRP1 bitwise gate at the old value and a re-baseline of
  x = 0 and the smallest unit on SRP1 before anything else runs; the 3 × 3 selection is frozen
  only after this decision.

# Addendum 42 — computational efficiency (discussion with the author, 2026-09-24)

- **Structure.** Scenarios sit inside each block, coupled only through the first-stage variables
  (commitment p̄, scenario-free storage schedule, pinned TSO interface, voltage pin); given those,
  each scenario is an independent ≈ 11k-variable OPF. The block is a two-stage program.
- **Three measurements, no algorithm change, when the machine is idle** (between the F2 poll and
  the α row; each ≤ 30 min): (1) **linear-solver benchmark** — one SRP1 block, one 2 × 2 block, one
  paper-scale block, each solved with ma27 / ma57 / ma97 at `OMP_NUM_THREADS=1`; wall time and
  iterations. The network blocks are pinned to MA97 (`p54r_provenance.py`), whose multithreaded
  advantage is unused under single-threaded concurrent evaluations. (2) **NL variable count vs
  model variables** at x = 0 (SRP1 and 2 × 2): retired variables (unwired storage copies, pinned
  TSO deviations, old quadratic slacks) must be *fixed* so the NL writer drops them; free unused
  variables are shipped to IPOPT. (3) **Existing persistent-worker path timed on the 2 × 2
  instance, two cycles**: the round trip that cost as much as a 0.4 s solve on SRP1 is a small
  fraction of a 3–19 s solve; report the speed-up and whether parent and workers both hold the
  models (memory).
- **Decision before the 3 × 3 pair** (each is a new configuration → re-certification at C\*,
  x = 0 and the smallest unit on SRP1, ≈ 3 h): switch the linear solver if the benchmark shows a
  clear gain; start ρ at D's terminal values (0.017 / 0.132 / 0.0225) if the Planner's read of the
  balancing history supports it; inexact early solves (IPOPT tolerance schedule tied to the
  residual level) only if the first two are adopted and time allows. None of these is applied
  mid-campaign; all are frozen before the multi-scenario results that reach the paper.
- **Future work (manuscript):** flatten the scenarios into the outer ADMM as separate blocks with
  non-anticipativity consensus on the first-stage variables (identical to today's form at one
  scenario, hence gated bitwise on SRP1), on memory-flat workers with mutable parameters and
  arrays across the process boundary — the fully parallel, ≈ 2 GB evaluator; a branch-flow SOCP
  relaxation for the radial DNs as a further option. **Not for this revision.** Skipping
  re-solves of nearly unchanged blocks is rejected: it would hollow out the certification tail.
- **Memory (author's follow-up):** resident memory is Pyomo, not IPOPT — per paper-scale DSO
  block ≈ 230 MiB of model, ≈ 137 MiB of suffix data (duals, bound multipliers) plus 45–58 MiB of
  duplicated warm-start copies; IPOPT's factorization is transient. Two refactors with exact
  gates, independent of the algorithm and of the parallelism question: (a) **one pristine model
  per network (or per network-year) with mutable parameters**, replaying each block's data and
  consensus state before its solve and keeping primal/dual vectors as arrays — model memory ÷4 to
  ÷20, replay ≪ solve; gate: byte-identical NL files, hence bitwise trajectories; (b) warm-start
  suffixes stored as numpy arrays rather than Suffix dictionaries — the ≈ 7 GiB of dual data to
  well under 1. Not needed for the 3 × 3 pair (fits today); decided after 42(2)/42(3) report, on
  whether the 5 × 5 confirmation pair is wanted on this machine. They are the foundation of the
  flattened-scenario architecture, so the effort carries forward.

# Addendum 43 — Addendum 41 outcome (2026-09-24)

- **Branch 1; no re-baseline; 3 × 3 unblocked.** Reachable (TN-side) curtailment is below
  resolution everywhere (C/bar ≤ 0.037); a compensation-based price would move the storage's
  value by 331 € against a resolution of 15,511.
- **Rule premise corrected:** the 1 €/MWh tie-breaker is **not in force** in the coordinated
  subproblems (zeroed for TSO and DSO; it acts only in the initialisation build and the
  uncoordinated benchmark). **Manuscript wording:** curtailment carries no explicit penalty in
  the coordinated problem; its cost is the replacement energy — settled at the scenario price on
  the DN side, dispatched at generation cost on the TN side; at the certified points reachable
  curtailment is below resolution. Branch 2 would have double-counted; it is withdrawn.
- **Node 7 refinement (manuscript):** on SRP1 all 364 curtailed generator-hours are
  inverter-capability-bound — RES inverters give up active power for reactive support with a
  voltage bound active in the same network-hour (99.6 %); there is essentially no surplus
  curtailment, and the DSO7 transformer binds on import as a **consequence** of the lost
  injection. The interface binding is voltage-driven inside the DN. The storage's reactive
  capability cannot reach it: the TN supplies reactive power at no cost, so the interface
  voltage is already what the coordination wants, and the overvoltage is local to the feeder
  ends. **Zero-solve confirmation when convenient:** DSO7's curtailed energy at the smallest unit
  vs x = 0 (unchanged → chain confirmed).
- **Addendum 42 scheduling accepted:** items (1) and (3) in the idle window after α-row pair 1,
  machine alone; item (2) now, build-only. The Planner's note that item (2) targets the same
  free-variable defect row 18 had (zero-cost free variables → a barrier ray without a central
  path) is the right reading.

# Addendum 44 — after `P5_15_ADDENDUM40_42_REPORT.md` (2026-09-25)

- **Accepted, with §4's corrections.** Expert's own correction: "R confirmed at 0.942"
  (Addendum 40) is **withdrawn** — the ratio's resolution at 2 × 2 is 0.152, so neither 0.942
  nor the restated 1.031 tests 0.937; the mean-profile argument stands as the analytical basis,
  untested at this resolution. The init-fix change in value (+23k) is indeterminate. The reactive
  leg's charge share is zero to tolerance. The curtailment indicator is primal.
- **α row accepted as the R3.6 row (five cells).** Manuscript: **C(α) = Q − charge is the
  cost-of-commitment curve** (21.6 M€, 2.66 %, from α = 0 to 1), with the charge reported
  separately as the transfer/proxy; Q alone overstates commitment cost 2×. Two mechanisms:
  arbitrage suppression essentially complete by α = 0.1; deliberate surplus curtailment above
  0.5 where π_s < α·π̄ (97 GWh, 7.7 M€ at α = 1.0) — a premium above the energy's value induces
  waste, which is why α ∈ [0.25, 0.5] is the operating range. Σωd² non-monotone at the top: stated.
  **No α = 2.0 cell**: a zero-solve estimate of curtailed surplus at α = 2 from the scenario
  prices and surplus volumes replaces it. Report the charge difference between the x = 0 and unit
  cells at α = 0.5 (value convention: Q per Addendum 38 C; if the charge difference is above
  resolution, C-based value is reported beside it).
- **§8a scaling hazard — priority, and the fix is a fixed-configuration polish, not a
  re-evaluation.** Each certified Q carries an interior-point barrier gap of order pairs × μ /
  scale; gradient-based scaling gave the storage cell a 7× smaller scale (388 more
  complementarity pairs), so the two gaps differ by ≈ 19k (> the two-cell resolution). Remedy:
  re-solve the existing certified cells under **one pinned `obj_scaling_factor` and `tol`
  1e-8** for every polish solve (zero new ADMM runs); report per cell the polished Q, the
  **block-objective Δ** (the Δ ≤ 0 guarantee is on the block objective, which includes settlement
  and charge — polished gross exceeding certified gross is expected, as at SRP1), and value and
  R under the polished convention; extend the gap estimate to the DSO blocks (zero solves).
  **Prediction:** value at α = 0.5 converges to ≈ 250–260k, the init-fix change disappears,
  polished R ≈ 1.0 ± 0.1. If it resolves the pair, **the polished Q becomes the reported cost
  convention** in every table (certified Q = the ADMM certificate). Pinning scaling inside the
  ADMM itself is a configuration change and is not done now.
- **DN-side curtailment:** unreachable from the HV busbar → no effect on the storage's value;
  branch 1 stands; reported as a system feature. **Art. 13 framing:** active power surrendered at
  the inverter S-limit to provide reactive support under a voltage constraint is a
  capability-curve consequence of a grid-code obligation, not a redispatch instruction, and is
  not compensated — manuscript term "active-power reduction under reactive support", not
  curtailment; the α = 1.0 volumes are self-inflicted under the premium and not compensable. The
  author verifies the legal text.
- **3 × 3 confirmed** with production's prefix draw [1,2,3] × [1,2,3] and R = 0.9331 recorded
  (reproducibility over a 0.6 % refinement), run after the polish re-measurement; 3 × 3 prediction
  band [0.93, 1.09] recorded with the resolution caveat. **5 × 5 does not run on this machine**;
  the memory refactor is **infeasible as-is** (four RES-dependent `Constraint.Skip` sites, three
  baked-in float families) and is recorded as follow-up work; the mean-profile argument plus
  3 × 3 is the paper's multi-scenario evidence.
- **σ_Q:** SRP1/C3-era provenance stated in the manuscript; the four-cell bar-sum is the operative
  test at 2 × 2. **Harness:** per-pair heartbeat files; boolean gate flags; the
  `row18_gate_addendum.json` caveat stands.
- **Order:** harness fixes → fixed-configuration polish re-measurement (stop for review, short)
  → 3 × 3 pair → 42(1)/(3) idle measurements → uncoordinated benchmark definition (Advisor) →
  Step 5.

# Addendum 45 — ruling 5 resolved: option (c) with a reference re-run (2026-09-25)

- **§8a corrected:** the log parser read the polish, not the terminal ADMM solve; the corrected
  barrier-gap difference is 15,079 = 0.91× the two-cell resolution — **indeterminate**. The
  scaling mismatch itself is real and structural (unit cell 0.001 vs 0.0047–0.0071; 388 more
  complementarity pairs); only its estimated effect on value is sub-resolution. Certified terminal
  models were not persisted for the α row (W48 memory ruling), so Addendum 44's polish
  re-measurement cannot run as worded; options (a) and (b) rejected (disproportionate; a
  different experiment).
- **Ruling: (c) + reference re-run.** The pin is `nlp_scaling_method = user-scaling` with one
  `obj_scaling_factor` for every network solve in every cell (a user factor alone is multiplied by
  the gradient-based one and pins nothing); the value chosen by a two-cycle SRP1 test between
  0.001 and 0.005 (both cells converge, similar iterations); production tolerances unchanged —
  matching, not tightening. **Fallback** if user-scaling degrades conditioning (constraint
  scaling reverts to 1): pure (c), gradient-based retained, caveat recorded, no re-run.
- **If the pin holds, it is a new frozen configuration for everything from here on:** re-certify
  C\*, re-evaluate x = 0 and the smallest unit on SRP1 (the new reference, ≈ 3 h); the value shift
  vs the old reference is the empirical size of the artefact. Reason: R compares 3 × 3 to SRP1,
  and a scaling-convention difference between them would be of the order of the effect R
  measures. The α row is not re-run; its storage cell carries the sub-resolution caveat.
- **Zero solves:** the corrected gap estimate on the SRP1 x = 0 / smallest-unit pair (terminal
  ADMM solves) against Phase A's bars — whether the ladders carry a material gap difference.
- **3 × 3 pair** under the pinned configuration, persisting certified models if 16 GiB allows,
  with the R prediction restated against the new SRP1 reference before the run.
- **Hull polish:** no action; the polish inherits the pin; a block-objective sign reversal that
  persists under matched scaling becomes a finding. **Value convention:** Q stands — the charge
  difference between the x = 0 and unit cells at α = 0.5 is 3,077 (0.015 %), the empirical
  counterpart of the storage cancelling out of the interface deviation.
- **Order:** pin test → SRP1 reference re-run → SRP1 gap estimate → 3 × 3 pair → 42(1)/(3) idle →
  uncoordinated benchmark definition. Stop for review after the 3 × 3 pair.

# Addendum 46 — barrier-gap question closed; convergence-depth tail rule (2026-09-25)

- **Closed: there is no scaling artefact.** Correctly identified terminal solves give ΔG =
  −77.71 € (structural pair count) on SRP1 Phase A, on the C2 reference and, once incomplete
  convergence is removed, at 2 × 2; the 2 × 2 figure of 15,066 (0.91× resolution) was fifteen
  x = 0 TSO solves stopping at μ 5.7–8.6× their floor after 10–21 iterations. Method validated
  (the identification rule reproduces both the corrected and the mis-read figures). **Addendum
  44's polished-convention proposal and Addendum 45's pin are void**; gradient-based scaling
  stays; item 2 of Addendum 45 (reference re-run under a pin) is void. Kept: pinning cannot
  equalise cells — the difference is structural (388 more pairs; iteration ratio 2.1–2.25 at
  every factor) — so any future "pin the configuration" proposal is answered. Expert's own
  note: two addenda were spent on a parser-based estimate acted on before the parser was
  verified; the standing rule is that a log-derived quantity is validated by reproducing a known
  figure before it drives a decision.
- **Convergence depth (item 7): minimal (c).** `compl_inf_tol` 1e-4 → 1e-6 **only in the
  certifying cycles**, switched by the all-channels-inside-tolerance condition that turns AA off;
  all cycles before the tail bitwise unchanged; per-solve floor status reported for every cell
  from now on. Rationale: the depth gap scales with complementarity pairs, so at 3 × 3 it is
  plausibly at the resolution (a likely 35 h re-run avoided for ≈ 3 h now); the certified Q sits
  at the μ floor on every block, values are depth-matched at every scale, the certificate is
  stronger. **Prediction:** SRP1 references (C\*, x = 0, smallest unit) unchanged or within 1e-6
  relative (SRP1 already at the floor 192/192); at 2 × 2 the fifteen early stops would reach the
  floor. **Fallback** if the tight tail fails to certify or moves the reference materially: the
  unchanged configuration with the depth caveat recorded (the Planner's §8 reading).
- **3 × 3 pair confirmed** under production + tight tail, R restated against the SRP1 reference
  re-run under it (or 259,427.77 under the fallback), certified models persisted if 16 GiB
  allows. Before reusing that reference: zero-solve look at `campaign_s47_recert`'s
  `identity_holds`. §9 items proceed.
- **Order:** tight-tail SRP1 re-certification → s47 identity look → 3 × 3 pair → 42(1)/(3) idle →
  uncoordinated benchmark definition. Stop for review after the 3 × 3 pair.

# Addendum 47 — design notes (author + expert, 2026-09-25; picked up after the 3 × 3 pair)

- **Linear solver — decision rule for 42(1):** switch only if the **paper-scale block** gains
  ≥ 1.5× at the same iteration count with no restoration entries; a switch is a new configuration
  and costs a ≈ 3 h SRP1 re-certification before the 3 × 3 pair. Expected: MA57 > MA97
  single-threaded at every size; MA27 possibly best on SRP1 blocks.
- **Memory — common-topology replay (author's proposal, adopted as the design):** one model per
  network over the **union** of assets across years; per (year, day) replay data as mutable
  Params and switch absent assets off. Blockers from the static audit map directly: the four
  RES-dependent `Constraint.Skip` sites become constraints always constructed and **deactivated**
  by the replay under the former skip condition; the three baked-in float families (RES
  availability, prices, `effective_scale`) become mutable Params; absent assets fixed at zero
  with rows deactivated. **Gate: byte-identical NL files** against the per-block build (the NL
  writer excludes deactivated rows and fixed variables and orders by declaration, so superset
  index sets can reproduce the file exactly) → bitwise trajectories, no re-certification. A
  switch Param leaving zero-coefficient rows would break byte identity and force re-runs — not
  the design. Costs: replay ≪ solve; warm-start vectors as arrays per block; a per-block
  provenance hash of the replayed state; snapshot/clone machinery adapted. 80 models → 4.
  Sequenced after the 3 × 3 pair, Advisor review of the design first; prerequisite for the 5 × 5
  pair and the flattened-scenario architecture.
- **INESC TEC Linux VM (≥ 64 GB; cores and architecture to be confirmed):** the Mac stays the
  single home of the decision record; the VM is a compute target. The Worker launches remote
  runs over SSH from its own shell; the repo syncs through git (the VM pulls the commit a run
  starts from); artifacts rsync back, are hash-recorded and committed from the Mac after
  verification; lock/heartbeat/exit-code conventions per machine; SSH and rsync patterns
  allow-listed for the VM host. Environment: Python 3.11 and the pinned package set; IPOPT with
  HSL (conda-forge IPOPT + compiled `libcoinhsl` via `hsllib`, METIS for MA97); provenance
  extended with hostname, CPU, BLAS, IPOPT/HSL builds. **Cross-machine bitwise reproduction is
  not expected;** equivalence gate: a certified SRP1 run reaching the same certification cycle
  with |ΔQ|/Q ≤ 1e-8 — pass → results may share tables with the machine stated; fail → the VM is a
  separate labelled configuration. With 64 GB two serial 5 × 5 evaluations fit concurrently; with
  enough cores and a positive 42(3) result the persistent-worker path brings the paper-scale cycle
  to minutes.

# Addendum 48 — tight tail adopted; 3 × 3 pair on the Mac under memory rules; VM storage discipline (2026-09-25)

- **Tight tail adopted; ruling 7 closed.** Accepted facts: all three SRP1 references re-certify at
  exactly their reference cycles (C\* 87, unit 112, x = 0 132); per-cycle gross bitwise identical
  before the first tail cycle (79/104/124); 48/48 terminal solves at the μ floor with
  `compl_inf_tol` 1e-6 in force; retry counts unchanged. ΔQ = −1.04 to −1.20 × 10⁻⁶ relative on all
  three cells. **Expert's prediction miss, owned:** "unchanged or within 1e-6" was exceeded by
  4–20%; its premise conflated μ reaching its floor with the complementarity residual reaching its
  tolerance — a 1e-4 termination leaves complementarity slack that the 1e-6 tail closes. The
  consistent negative sign on 3/3 cells is the barrier-path signature (the barrier holds the
  iterate interior; the objective falls as complementarity tightens): a **systematic offset, not
  noise**, which cancels in differences — ΔR = −52.44 € (2 × 10⁻⁴ of R). The fallback was correctly
  not triggered: certification held at identical cycles and the reference moved immaterially.
- **New SRP1 reference R = 259,375.33** adopted; the 3 × 3 prediction R = 0.9331 stands, restated
  against it. **Scope of the tail: the reference pair only.** Phase A/B, ageing, flexibility and α
  cells stay as certified under the previous tail — a common ≈ −1.1 × 10⁻⁶ offset cancels in every
  value difference to ≈ 2 × 10⁻⁴ — no re-runs; the manuscript states both tolerances and the
  measured offset in the certification paragraph.
- **Gate G6:** the final scope (final accepted attempt per block, tail window and terminal round)
  is written into the spec for every future cell; the two post-hoc re-scopings are recorded as
  such. Acceptance of the tail rests on the three measured facts above, not on G6. Standing rule
  unchanged: a gate's scope is part of the frozen spec, fixed before the run. **Bar caveat
  recorded:** bar_tail = bar_ref bitwise because each window's largest step is pre-tail, so the
  tail neither improves nor independently re-measures the reproducibility band; the 0.011% band
  stands as measured. x = 0's shift at 1.95× its own bar is consistent with a systematic effect.
  **`identity_holds` = False** is accepted as a stale-formula artefact; recompute it under the
  current formula or drop the flag from the summary — no False flag travels without its
  explanation.
- **3 × 3 pair: on the Mac, now — not held for the VM.** The VM's date is in IT's hands, the pair
  is the paper's critical result, 25–31 h of Mac time is affordable, and the Step 5 SRP1 rows
  (≈ 1 h each) follow it. Decisions: **(1)** sequential pair at concurrency 1 — accepted; nothing
  else runs on the Mac during it (author's one-run rule); the Advisor's uncoordinated-benchmark
  definition (no compute) proceeds meanwhile. **(2) Option (b) adopted:**
  `release_solution_bookkeeping` on, measured first at zero solves at 3 × 3 (prediction: ≈ 3 GiB per
  child). **Persistence only if the measured runtime peak with persistence ≤ 0.85 × memory
  available after the reboot** (≥ 15% headroom for a 15 h run); prediction: it will not fit
  (≈ 24 GiB against ≈ 24), so the rule, not the prediction, decides. Without persistence the hull
  polish is omitted at 3 × 3 and reported as such with the SRP1 figure (3.09 × 10⁻⁶ relative, below
  the band); Addendum 45 ruling 5 is void, so the only remaining consumers of terminal models are
  the polish and future post-hoc looks — those, if ever needed, come from the VM. Option (c)
  rejected before the pair (Addendum 47 sequencing stands). **(3)** Reboot before launch —
  author's action; the preflight re-measures and records available memory in provenance.
- **Spec v34 → v35** before launch: concurrency 1; (b) on; persistence per the margin rule; the
  smoke gate becomes a **two-arm 3-cycle bitwise comparison** ((b) on vs off; per-cycle gross and
  residuals identical; ≈ 1.2 h) that also measures the 3 × 3 runtime peak directly instead of
  scaling from 2 × 2; predictions restated before launch (R = 0.9331 against 259,375.33; memory
  peaks; 10.1–12.5 min/cycle; per-solve floor status). Frozen eval keys, prefix draw
  [1,2,3] × [1,2,3] and the SRP1 (b) bitwise gate carry over.
- **VM — Addendum 47 supplement (confirmed: 64 GB, 24 cores, x86, Ubuntu 24.04 CLI-only,
  storage-limited).** 40 GB workable, 60 GB comfortable, under four rules the Worker builds in from
  the start: (i) venv or cleaned conda; shallow clone without `Results`
  (`--depth 1 --filter=blob:none`); IPOPT/HSL build trees deleted; snapd removed; journald capped;
  (ii) swap 8 GB; (iii) NL/sol scratch on an 8 GB tmpfs through `TMPDIR`; (iv) certified-model dumps
  written only for flagged cells (references, reported plans), rsynced to the Mac on certification
  and pruned — at most two cells in flight (≈ 10 GB at 3 × 3, ≈ 20 GB at 5 × 5). Also: matplotlib
  `Agg`; runs launched inside tmux or `systemd-run`, never as a foreground SSH command; packages
  `build-essential gfortran cmake meson ninja-build pkg-config libblas-dev liblapack-dev
  libmetis-dev git rsync tmux`. First VM job: the equivalence gate; then 42(1)/(3) at paper scale,
  where the 24 cores matter (block-solve concurrency per child is the lever; memory per concurrent
  IPOPT/MA97 process to be measured). The VM queue after that is set at the post-pair review.
- **Order:** (b) zero-solve measurement → reboot → two-arm smoke gate → 3 × 3 pair (sequential) →
  **stop for review**. Then 42(1)/(3) idle measurements → Step 5 rows.

# Addendum 49 — uncoordinated benchmark: three-arm definition, fixed interface, common Q (2026-09-26)

- **Ruling 1 — three arms, adopted, with the claim anchored differently.** Arms at x = 0 on SRP1:
  **passive** (production DSO block with all flexibility fixed at zero, substation voltage at the
  setpoint, curtailment only for feasibility); **price-taker** (production DSO block with the
  consensus terms removed — λ = 0, ρ = 0 — so the DSO sees exactly the price its local objective
  carries in production and nothing else, at the setpoint voltage); **coordinated** = the certified
  x = 0 cell. Each arm is the production subproblem with coupling removed, not a re-implementation.
  Decomposition reported as the Advisor defines it, **but the paper's claim is "coordination beats
  the best uncoordinated arrangement": benefit = min(passive, price-taker) − coordinated.** The
  sign of passive − price-taker is not guaranteed: in the hours where the TN's interface marginal
  value λ_t is below the price the DSO sees (RES-covered hours — Addendum 32's mechanism, conventional
  at zero), a price-taker activates downward flexibility the system does not need, and its system
  cost can exceed the passive arm's. If that happens it is a finding (a misaligned tariff is worse
  than passivity), not an error. **Mechanism to state in the paper:** coordination proper is the
  value of pricing DN flexibility at λ_t rather than at the wholesale price; it lives where
  λ_t ≠ π_t — congested or voltage-constrained interfaces, and RES-covered hours. **Zero-solve look
  first:** from the certified x = 0 cell, tabulate λ_t (interface-P consensus dual) against π_t per
  node and hour; the hours with λ_t ≠ π_t and c_flex < π_t bound coordination proper from above.
  Recorded as the prediction before any arm runs; the Advisor's "inside the band" expectation
  stands beside it as the competing prediction.
- **Ruling 2 — hard-fixed interface loads, adopted; the penalty goes.** A 9 × 10¹⁰ tracking penalty
  dominates IPOPT's gradient-based objective scaling and can leave the TN's economic gradient below
  the dual-infeasibility tolerance — an "optimal" flag over an unresolved dispatch, biasing the
  benchmark upward. The TSO arm is the production TSO block with interface P and Q fixed to the DSO
  arm's schedule as (mutable) Params, consensus terms removed, voltages within their normal
  bounds — the exact limit of the penalty, well scaled. The 24-solve check (penalty vs fixed at the
  same targets: TN cost, interface residual, iteration count, dual infeasibility at termination)
  runs and is reported; it decides whether the hazard was live, not whether the ruling holds.
- **Ruling 3 — new production function beside the old, adopted.** The existing path stays unwired
  and untouched (SRP1 bitwise gate); the new one is built from the production model constructors
  with a check that each arm's model has the coordinated block's variable and constraint counts
  minus the removed consensus rows plus the fixed rows. **Common-Q gate before any arm is
  reported:** the new evaluation function applied to the certified x = 0 cell's persisted terminal
  models must reproduce its certified gross_operational_cost (bitwise, or the difference explained
  to the last digit) — the log-derived-quantity rule applied to the evaluation function. The
  tie-breaker Param is set to its production value in every arm; the arms differ only in decision
  objectives and coupling.
- **Consistency convention (SRP1).** TN voltages bounded, not fixed at the setpoint. After the two
  solves, re-evaluate each DN at the TN's actual interface voltage with the DSO's decisions fixed:
  report max |ΔV| at nodes 5/7/9 per hour and any DN voltage or thermal violation. If a DN limit is
  violated, one sequential pass (DSO re-solved at the actual voltage → TSO re-solved) defines the
  arm's cost, and the pass's effect is reported. The band is widened by what is measured, never
  softened.
- **Resolution, measured on the arms themselves.** Each arm solved from three starts (production
  cold; warm from the certified coordinated solution; perturbed); the spread is the arm's
  multimodality band, reported next to the coordinated cell's reproducibility band (0.011%) and
  the DSO band re-measured in Step 5's certification statistics. The C3-era 0.3% is not used as a
  number. A difference is claimed only above the larger band.
- **Manuscript:** the 18.25% coordination benefit is **withdrawn now**, not after the measurement —
  it was measured on the transfer-payment recourse Addendum 11 retired. The replacement is the
  three-arm table with bands; if coordination proper is inside the band on SRP1, the paper says so
  and places the coordination claim where λ_t ≠ π_t (the multi-scenario instance under row 18, and
  any stressed variant the author chooses later — author's call, with the SRP1 numbers in hand).
- **Timing.** Nothing solves before the pair finishes; the Worker may write the new function and
  its tests during the pair and runs nothing — not even tests that build a model (the pair holds
  ≈ 22 of 24 GiB). Order after the pair review: λ_t vs π_t look → common-Q gate → 24-solve check →
  three arms × three starts → consistency re-evaluation → report. The 3 × 3 convention: second
  Advisor note after the pair, as proposed.
- **Clarification (2026-09-26) — the tie-breaker has two roles, and "its production value" meant
  the evaluation.** *Evaluation Q:* tie-breaker **0** in every arm — the value the certified
  coordinated Q was computed with (the ADMM subproblems zero it), so the common-Q gate on the
  x = 0 cell fixes it by construction; if that cell curtailed nothing the gate cannot discriminate
  0 from 1 and the rule stands by principle. *Decision objectives:* **0** for the price-taker (the
  production subproblem with coupling removed; curtailment is already costed through the interface
  price) and for the coordinated arm (production). **1 €/MWh (the build default) for the passive
  arm only,** because with flexibility fixed at zero and no price the passive DSO's objective would
  be empty — a pure feasibility problem whose curtailment, and therefore whose interface schedule
  and TN cost, IPOPT would pick arbitrarily. As the sole objective term the tie-breaker is the
  minimum-curtailment selection rule and its value is immaterial to the arg-min; verify by
  re-solving the passive arm at 0.1 and 10 €/MWh and reporting the interface schedule difference
  (expected: within solver tolerance; any residual is the non-unique distribution of curtailment
  among units, which the three-start band already covers). Report curtailed energy per arm.
- **λ_t recovery.** Two independent sources, both to be read and compared: the consensus dual for
  the interface-P channel, which enters the DSO and TSO subproblems as a (mutable) Param and is
  therefore readable from the persisted terminal models of the W86 x = 0 cell; and the TN nodal
  dual at the interface bus (IPOPT constraint dual of the TN power balance, in the models'
  suffixes). At certification they agree up to the ADMM scaling — on the interface-P channel
  **σ and the interface rating only** (S_ref and D5 act on the ESS channel alone; Planner's
  correction from source, 2026-09-26) — which must be undone to €/MWh before comparison with
  π_t; the check that the units are right is reproducing
  Addendum 32's known figure — the bus-7 marginal cost equal to the DSO flexibility shadow price
  — before the table drives any prediction (the log-derived-quantity rule).

# Addendum 50 — 3 × 3 x = 0 certified; G6 failure benign; objective not settled at certification (2026-09-26)

- **Stop was correct; the running node-7 cell continues.** A failed prediction with no named
  fallback is a stop; stopping the cell would destroy the data that decides the finding. The
  certification rule is not changed mid-pair.
- **Ruling 1 — G6: option (a).** Recorded as failed (1/80 at the μ floor against ≥ 72/80), with
  its mechanism, and **not re-scoped** for this pair. **Expert's premise, owned:** G6 encoded "at
  the μ floor" as the depth criterion because SRP1 sat there 192/192 under production tolerance;
  but the floor is IPOPT's `mu_min` — a lower **clamp** derived from `compl_inf_tol`, not a target.
  IPOPT terminates when the four error metrics meet their tolerances, and a solve that does so one
  monotone μ-update above the clamp (μ/floor ≈ 3, inside the (1, 5] band a 0.2 decrease factor
  implies) is converged by the only definition that matters. Under the tail the clamp fell
  100–500× while the solves went 33× deeper; "above the floor" is the floor moving. **Verify from
  logs, zero solves:** μ/floor ∈ (1, 5] for all 80 (any > 5 breaks the one-update hypothesis for
  that solve); the four terminal error metrics against the tail tolerances for the final accepted
  attempt. **Prediction for node-7:** G6 as frozen fails likewise (≤ 3/80 at floor, median μ/floor
  ≈ 3). **G6 for future specs**, frozen before the next run: depth = `Optimal Solution Found` under
  the tail tolerances for the final accepted attempt in the tail window and terminal round;
  μ_terminal/μ_floor reported per solve, not gated.
- **G6 does not explain the drift.** The μ-level effect of a 1e-4 → 1e-6 tolerance change was
  ≈ 700 € in total at SRP1; the residual 1–5× floor variation is far below that. A descent of
  4,000 €/cycle and accelerating is **ADMM-level movement**: after the tolerance switch the fixed
  point of the inexact-solve map shifted and the iteration is moving toward the new one — or
  moving between basins. Two findings, one cause of exposure: by construction the tail fires when
  the channels enter tolerance and the 10-cycle window starts at once, so the tail's transient
  sits **inside** the certification window; at SRP1 the transient was small enough not to matter.
- **Ruling 2 — the certified Q stands as "certified on residuals", its ± bar claim does not.**
  Steps of −1,009 … −4,006 € over cycles 66–72, each larger than the last, mean the bar (largest
  step in the window, 13,954 €) does not bound the remaining descent — the CLAUDE.md refinement
  applies verbatim: the bar bounds stopping slack, not a path still descending. Rule ten
  (terminal-step-to-threshold) is computed now for x = 0 and for node-7 at certification.
- **Diagnostics from records, zero solves, both cells once node-7 certifies** (x = 0 may start
  now — records only, no model loading while the cell runs): (i) per-block decomposition of ΔQ per
  cycle 63–72 — uniform across blocks (barrier creep, H_B) vs concentrated in a few DSO blocks
  (basin transition, H_C); (ii) per-cycle movement of the consensus variables and duals per
  channel; (iii) primal/dual residual margins inside the window as fractions of tolerance;
  (iv) **the row-18 terms per cycle — Σω|d| and the premium (H_row18):** the d⁺/d⁻ split is an
  L1 kink, degenerate for an interior method, and the tighter tolerance resolves the DSO's
  scenario deviations, moving p̄ and hence the TSO — a 3 × 3-specific mechanism absent at SRP1
  (d ≡ 0) that would explain why the 3 × 3 tail effect is 20× SRP1's; (v) local-solve iteration
  counts and μ per cycle. **Predictions:** H_row18 — the premium and deviation terms account for
  most of ΔQ and the DSO blocks with the largest ΔQ have the largest deviation changes;
  H_B — uniform, decreasing ΔQ across blocks; H_C — a few DSO blocks with large ΔQ and longer
  solves.
- **Ruling 3 — post-certification continuation, after the pair and the diagnostics.** Each cell
  continued from its terminal state for up to 30 cycles (≈ 3.3 h each, one at a time), after a
  **bitwise identity gate**: the terminal cycle re-run from the recorded state reproduces its Q
  bitwise; if the state is not restorable bitwise, the continuation is a separately labelled run.
  Stop when |ΔQ| < 500 €/cycle for 3 consecutive cycles or at 30 cycles. Report the total
  post-certification descent per cell and its effect on the value. **Expert's prediction:** a
  hump — steps peak within ≈ 5 cycles and decay; total remaining descent 20–60 k€ per cell;
  inter-cell difference < 10 k€, inside R's resolution. If the steps are still increasing after
  10 cycles or the total exceeds 100 k€, H_C is likely and no 3 × 3 result is used before the
  certification design gains a settling criterion. This is a measurement of the stopping slack
  the bar failed to bound, not a re-certification; the certified Q's are unchanged.
- **Reporting.** The value and R are reported three ways: at certification; after continuation;
  and with the drift as an explicit uncertainty. The manuscript's certification paragraph for the
  multi-scenario instance says "certified on residuals at cycle N; the objective settled within
  X € over M further cycles." **Future specs** (frozen before any further 3 × 3 cell, Advisor
  review first, informed by the continuation data): a settling criterion in the certification
  window — non-increasing objective steps over the last five cycles, or a restart of the 10-cycle
  count at tail engagement — is proposed, not yet adopted.

# Addendum 51 — 3 × 3 pair certified; continuation by replay (route A, staged); G6 restated (2026-09-27)

- **Results accepted.** At 3 × 3 the smallest unit does not pay: value − I = −81,262 €, determinate
  at 4.5×; **x = 0 optimal under the baseline at both instances** — and the drift can only widen
  this (x = 0 falls 2–2.8× faster than the storage cell, so a continued fall shrinks the value).
  R = 0.9126 at certification against the recorded 0.9331: indistinguishable at the resolution.
  159/160 final solves converged by IPOPT's definition; the one "Acceptable Level" exit (compl
  2.93× tolerance) is reported with its cell, block and retry history, not acted on.
- **Expert's explanation withdrawn.** Addendum 50 attributed the descent to the tail's transient
  inside the certification window; on both cells the fall began before the tail engaged, so the
  attribution fails and H_row18 as a cause of the *drift* falls with it. What stands: the objective
  converges more slowly than the Boyd residuals on this instance, and the bar does not bound the
  remaining descent. Note that tail-on and AA-off coincide by design (cycle 63 on x = 0), so an
  AA-off transient — plain ADMM correcting extrapolated duals — remains a candidate for the
  accelerating steps in the window; the continuation discriminates it.
- **Ruling — route A, staged; B rejected (memory rule, production changes); C rejected** because the
  paper's certification claim for the multi-scenario instance needs the settling behaviour
  quantified, not only stated, and the settling criterion for every further multi-scenario run
  (VM included) needs this data. **Stage 1: x = 0** — re-run under the identical spec to
  certification, **checked bitwise against the 72 recorded cycles** (this replay is also the
  instance's reproducibility measurement for the manuscript; a divergence at cycle k is reported
  with its magnitude and the run is then labelled separately, still informative), then continue
  **30 cycles** with the certification rule disabled and the certifying regime unchanged (tail on,
  AA off, ρ frozen) — a separately labelled run under spec v37. Stop early at |ΔQ| < 500 €/cycle for
  3 cycles. **Analysis:** fit the post-certification steps to a geometric sequence; report the
  ratio and the extrapolated remaining descent D = step/(1 − ratio) with the fit's validity
  (increasing steps → no extrapolation). **Stage 2 decision rule, recorded now:** the storage
  cell's continuation runs if D_x0 (measured plus extrapolated) exceeds the value's resolution
  (bar-sum ≈ 18 k€); otherwise R is reported as the range [(V − D_x0)/R_ref, V/R_ref] with the
  storage cell's descent bounded above by D_x0. ≈ 11 h for stage 1; ≈ 12 h more if stage 2 runs.
- **Expert's predictions, recorded:** replay bitwise through cycle 72; steps peak within a few
  cycles of certification and then decay geometrically with ratio 0.80–0.95; D_x0 = 20–60 k€;
  stage 2 triggered; R_settled ≥ 0.80. **Competing outcomes and what each means:** (H1) hump then
  geometric decay → AA-off transient plus a slow mode; the settling criterion is a step bound at
  the end of the window; (H2) near-constant steps through cycle 102 (ratio ≈ 1) → the residual
  tolerances are too loose for this instance; the certified Q's are reported as upper bounds and
  the criterion must be objective-based (or ε_rel tightened) before any further 3 × 3 result is
  used; (H3) growing steps or a jump → basin transition; Advisor review before anything else.
- **Diagnostics owed.** Addendum 50's records-only items (i)–(v) — per-block ΔQ decomposition,
  consensus and dual movement per channel, residual margins, row-18 terms per cycle, iteration
  counts and μ — are reported for both cells before stage 1 launches, or the omission is stated.
  Add the ρ history and the AA state around cycles 55–72 on each cell.
- **G6 for future specs — adopted as the Planner states it:** `Optimal Solution Found` plus the four
  error metrics within the tail tolerances for the final accepted attempt; μ/floor reported per
  solve, never gated — the floor test measured scaling, not depth. Frozen into v37.
- **Manuscript.** Multi-scenario instance: x = 0 optimal (determinate at 4.5×, drift-robust);
  R reported at certification and settled (or as a range); certification paragraph carries the
  replay reproducibility result and the settling statement "certified on residuals at cycle N; the
  objective descended a further X € over M cycles (ratio r)". The SRP1 certification statement is
  unaffected (its objective was settled at certification: the tail moved it 1e-6 in total).
- **One-run rule.** Stands unless the author rules otherwise: the SRP1 benchmark arms (≈ 1 GB,
  minutes) could run beside the continuation only with measured headroom ≥ 3 GiB above the
  continuation's peak — the author's call, since the rule is his.
- **Order:** diagnostics report (records only) → spec v37 frozen with predictions → stage 1 → stage 2
  by the rule → pair report (value, R three ways, reproducibility, settling) → **stop for review**.
  Then the SRP1 benchmark (Addendum 49 order) and Step 5 rows.

# Addendum 52 — 3 × 3 pair closed; settling criterion to the Advisor; gate-result hygiene (2026-09-27)

- **Pair closed; results accepted** (`P5_15_ADDENDUM51_CONTINUATION_REPORT.md`, `58ff8d88`).
  Replay **bitwise 72/72** — the multi-scenario instance's reproducibility statement. After
  certification the objective **oscillated and settled**: 4,492 € further fall under the frozen
  rule, settled value 6,235 € below the certified one by the post-hoc fit (range 5.1–6.3 k€,
  ≈ 7 × 10⁻⁶ relative). Stage 2 not triggered (threshold 17,965 €). **R ∈ [0.888, 0.913]**,
  indistinguishable from the recorded 0.9331 — the mean-profile prediction holds. **x = 0 optimal
  under the baseline at both instances.**
- **Expert's predictions against outcomes:** replay bitwise — held; transient then settling — held
  in kind, but the shape was a damped oscillation, not a hump with geometric decay; **D_x0 = 20–60 k€
  — missed by an order of magnitude** (measured 4.5–6.3 k€); stage 2 triggered — missed;
  R_settled ≥ 0.80 — held trivially. The accelerating steps of cycles 66–72 were one half-swing of
  an oscillation, not the onset of a slide, and I read them as the latter. The corollary I stated in
  Addendum 50 — that the bar could not bound the remaining descent — was true as a matter of
  construction and false as a matter of fact here: 6.2 k€ settled inside the 14.0 k€ bar. The
  standing statement is the narrow one: the bar bounds stopping slack once the objective has
  settled; on this cell it did, and the continuation is what shows it.
- **Settling criterion → Advisor review (Addendum 50's item), now with data and a constraint.** A
  step bound is phase-dependent on an oscillating objective (the 500 € rule stopped at a turning
  point and admits a ≈ 1,464 € swing). The criterion must bound the **envelope**: the range of Q
  over a window at least one oscillation period long (period read from cycles 72–88), or the
  amplitude of a damped-oscillation fit validated on points it did not use. Candidate: certify when
  the 10-cycle range of Q ≤ a fraction of the value resolution the campaign targets, in addition to
  the Boyd residuals. The review also takes the mechanism question: the oscillation lives mainly in
  Spring and the TSO — a marginal-resource switch at the interface (λ_t near the DSO flexibility
  shadow price in RES-covered Spring hours) is the candidate to test against the per-block records,
  and the criterion must be robust to it. No compute; runs in parallel with the benchmark; adopted
  into the spec before any further multi-scenario cell, Mac or VM.
- **Gate-result hygiene — adopted, before the SRP1 benchmark runs.** One shared writer for gate
  results; string-typed flags refused at write; a repository-wide test that loads every gate-result
  JSON under the results tree and asserts boolean typing, so the next recurrence fails a test, not a
  run. Rule into CLAUDE.md: a defect fixed twice is fixed at the writer, not at the caller. Second
  rule from the same report, also into CLAUDE.md: **a fit is validated only on points it did not
  use.** Both Planner self-corrections accepted as stated.
- **Manuscript.** Multi-scenario instance: x = 0 optimal (determinate at 4.5×; drift-robust);
  value ratio 0.89–0.91 (certified to settled) against 0.933 predicted from the mean-profile
  spread; reproducibility bitwise over 72 cycles; certification paragraph: "certified on residuals
  at cycle 72; a 16-cycle continuation showed a damped oscillation settling 6.2 k€ (7 × 10⁻⁶) below
  the certified value, within the 14.0 k€ certification bar." SRP1 statements unchanged.
- **Order:** gate-writer + type test (Worker, code + test, minutes) → SRP1 benchmark per Addendum 49
  (λ_t vs π_t look → common-Q gate → 24-solve check → three arms × three starts → consistency
  re-evaluation → report) → Step 5 rows. Advisor settling review in parallel. Stop for review after
  the benchmark report.

# Addendum 53 — settling criterion adopted (δR = 0.07); SRP1 references not settled: continuation first (2026-09-27)

- **Ruling 1 — certification rule for multi-scenario cells: the Advisor's refined criterion,
  adopted**, on top of the Boyd residuals: after the residuals pass, at least two sign changes of the
  objective step (the period is measured, not assumed); successive swings not growing; the range of
  Q over one full period ≤ τ; no fit in the decision. **Add a monotone branch:** if no sign change
  occurs within 2× the longest period seen on the instance, certify when the steps are decreasing
  and the range over that window ≤ τ. **δR = 0.07 → τ = δR·V_SRP1/4 = 4,539 € per cell.** Rationale:
  the manuscript claims R to ± 0.035, which is the band already measured at 3 × 3 and makes the
  statement "0.89–0.91 against 0.933 predicted, at the resolution" honest; resolving the 0.03 gap to
  the prediction would need δR ≈ 0.035 (τ ≈ 2.3 k€) — a VM-scale refinement, recorded, not
  ordered. Cost accepted: ≈ 3–4 h more per 3 × 3 cell. Addendum 52's 10-cycle candidate is
  withdrawn (it certifies node-7 mid-descent).
- **Ruling 2 — the SRP1 references first, before anything else that solves.** The finding that
  matters most in this note: on all three SRP1 references the residuals first pass at the cycle
  AA switches off, and Q then falls 38–53 k€ below every earlier trough; x = 0 was certified at the
  bottom of that fall, the unit cell on the rebound. **The SRP1 value (259,375 €) therefore carries
  an unquantified settling band, and the SRP1 sign margin — value − I ≈ −59 k€ — is of the same
  order.** "x = 0 optimal at SRP1" is not safe until this is measured; at 3 × 3 it is (−81 k€ against
  a measured 6 k€ slack). **Order:** continuation of **all three references** (x = 0, unit, C\*) as at
  3 × 3 — bitwise replay to certification, then continue under the certifying regime until the new
  criterion certifies or 100 cycles, ≈ 3.5 h in total — under spec v38 with the criterion as the
  stop rule. **Expert's predictions:** settled slacks per cell within [5, 25] k€; the value's
  settled change |ΔV| ≤ 20 k€, so the sign margin survives; the three slacks alike to within
  ≈ 5 k€ (the certification rule fires at the same phase of the same transient, and the clean
  affinity across dozens of SRP1 cells argues the slack is mostly common-mode). **If |ΔV| > 40 k€
  the SRP1 sign conclusion is reopened.** Then **triage:** every SRP1 difference the manuscript
  reports (Phase A ladders, break-even, ageing ×1.126, flexibility ladder, α row) whose margin is
  below 3× the largest settled slack among the three references is flagged "pending" and listed
  with its margin; a re-settling campaign (continuations of the flagged cells, ≈ 1.2 h each on the
  Mac) is sized, and if it exceeds ten cells it is the VM's first job after the equivalence gate.
- **Ruling 3 — the "ratio resolution 0.1405" is retired.** It measured the largest step at the
  start of the window, i.e. stopping slack, not settling; every resolution statement from here
  derives from τ and measured settling.
- **Ruling 4 — benchmark.** The uncoordinated arms (minutes) run as soon as W100 is reviewed; the
  coordinated arm is reported only after the x = 0 continuation, as a settled value, not a 31 k€
  band. Nothing else changes in Addendum 49.
- **Records.** The per-period interface consensus duals (λ_t per node) join the default per-cycle
  records from now on — a write-only change; the mechanism question could not be tested for want
  of them.
- **Corrections accepted; ownership.** Addendum 51's "SRP1 settled at certification" is withdrawn —
  the expert inferred settling from the tail's small total effect, which says nothing about the
  AA-off transient; Addendum 52's "SRP1 statements unchanged" falls with it, pending the
  continuation. The Planner's own correction stands: the objective converges at 0.892/cycle, faster
  than the slowest residual (0.975); the residual criterion stops the iteration while Q is still
  ≈ 20 k€ from its limit because the window opens exactly when AA switches off and the plain-ADMM
  transient begins.
- **Design note for the Advisor, no compute:** the certification window coincides with the AA-off
  transient by construction. On the continuation data, evaluate whether switching AA off earlier
  (residuals within 10× tolerance) would let the transient decay before the window opens and
  shorten certified runs under the new criterion. Not adopted now.
- **Manuscript.** 3 × 3 conclusions unaffected. **All SRP1 numbers on hold** until the
  three-reference continuation reports — one working day at most. The certification paragraph
  gains, for both instances, the settled-versus-certified figure and the criterion statement.
- **Order:** W100 review → spec v38 (criterion; predictions) → SRP1 three-reference continuation
  → **stop for review** (triage and any re-settling campaign are sized, not launched) → benchmark
  arms → Step 5 rows.

# Addendum 54 — SRP1 value survives settled; C\* creep diagnostic; re-settling campaign under one configuration (2026-09-28)

- **Results accepted** (`P5_15_ADDENDUM53_SRP1_CONTINUATION_REPORT.md`, `ebe34021`). Replays
  bitwise 3/3. x = 0 settles +14,971 € (± 4.2 k€) above its old certificate, the unit +20,806 €
  (± 4.4 k€); **V_SRP1 = 253,540 €** (was 259,375); **value − I = −64,417 €, determinate at 7.5× — x = 0
  optimal at SRP1 under the baseline.** New certification cycles 181 and 172 (were 132 and 112).
  **3 × 3 R restated against the settled reference: [0.909, 0.934] against 0.9331 predicted** — the
  mean-profile prediction is confirmed within resolution; the 0.03 gap was the unsettled reference.
  Predictions: P1 held (x = 0), indeterminate (unit); P2 held; **P3 missed narrowly on the two
  scoreable cells** (slacks 5.8 k€ apart, against "within 5 k€") — the common-mode argument was
  right in substance and optimistic in degree. The Advisor's period projection (20 cycles) missed:
  the SRP1 period is 29–30; the period is instance-specific and the criterion measures it.
- **Criterion wording fixed.** "Two sign changes" and "range over one full period" were
  inconsistent: two turning points bound a half-period; a full period needs three. **The
  three-turning-point reading is the criterion.** The Planner confirms in the report whether it
  was in v38 before the run; if it was applied after, the report states both readings' certification
  cycles and the frozen one stands as the certificate, with the stricter one reported.
- **Ruling 1 — C\*: option (a), 100-cycle diagnostic extension (≈ 2.7 h), with a hypothesis to test.**
  A steady −265 €/cycle with no decay and a slowly rising pf_primal is not a settling iteration; it is
  a creep along a nearly flat direction. **H_ess-flat:** the ESSO carries no economic term (ε-throughput
  regularizer only), so with substantial storage the ESS schedule is determined only through consensus
  and the augmented Lagrangian is nearly flat along re-timings of that schedule; ADMM discovers the
  residual arbitrage slowly. x = 0 (no ESS) and the unit (small ESS) settle; C\* (the corner plan)
  creeps. **Record per cycle:** per-block ΔQ (TSO, DSO, ESSO), Σ|Δp_ess| per node, the channel
  whose primal residual is rising. **Predictions:** the creep sits in TSO generation cost with ESS
  schedules still moving (Σ|Δp_ess| not decaying) and the rising residual is the ESS channel; the
  rate stays within a factor 2 of −265 €/cycle over 100 cycles. **Outcomes:** confirmed → the
  certification rule gains a **creep branch** ("residual-certified; objective drifting at r €/cycle;
  settled value unbounded"), and the manuscript's storage cells carry a drift band, with a
  formulation note that an economic tie-breaker in the ESSO would remove the flat direction
  (author's decision; not for this revision unless a decision hinges on a creeping cell);
  refuted (creep in DSO blocks, or decays, or the residual leaves tolerance) → Advisor review before
  the campaign is sized.
- **Ruling 2 — re-settling campaign: re-run, not pin.** The flagged cells are **re-run under the
  current production configuration** (tight tail + new criterion), so every manuscript SRP1 number
  shares one configuration; gate per cell: the trajectory is bitwise identical to the original run
  up to the first tail cycle (the tail re-certification showed this holds). All differences are
  restated against the **settled** references. **Phase-mismatched cells first:** the six Phase B
  neighbours, the 2030 cell and the F2 certificate (≈ 8 cells, ≈ 14 h on the Mac) — those stopped
  mid-descent against x = 0 at its trough, so settling can shrink their margins. The threshold
  (3× the largest settled slack; 62.4 k€ provisional) is finalised after C\*. The remaining cells
  (≈ 25–34) go to the VM if it is online within the week, else the Mac continues — **author's
  machine-time call.**
- **Ruling 3 — benchmark on the settled models:** the common-Q gate and the λ_t vs π_t look use the
  cycle-181 x = 0 models; the reference Q for the gate is the settled value. The uncoordinated arms
  (≈ 1 h with the look and the gate) run **between the C\* diagnostic and the campaign**, so Step 5's
  benchmark is in hand early.
- **Housekeeping:** the truncated message on `279c732b` stays; history is not rewritten.
- **Manuscript.** SRP1: settled value 253,540 €; x = 0 optimal (−64.4 k€, 7.5×). Multi-scenario:
  R ∈ [0.909, 0.934] against 0.933 predicted from the mean-profile spread — confirmed within
  resolution. Certification paragraph: the criterion (Boyd residuals + three turning points, swings
  not growing, one-period range ≤ τ = 4,539 €), the cycles under it, and the settled-minus-certified
  figures for both instances. Every other SRP1 number remains on hold pending its re-run.
- **Order:** C\* diagnostic → benchmark look + common-Q gate + arms → phase-mismatched 8 cells →
  **stop for review** (campaign threshold final; VM decision; benchmark report).

# Addendum 55 — benchmark gate C4 scope; negative curtailment in the shared helper (2026-09-28)

- **Ruling — option (a).** The failing C4 row (node 7, 2030 Summer, hour 6: DSO-side and TSO-side λ
  0.058 apart against 0.05) is a **consensus-agreement** quantity, not a units quantity: units errors
  are systematic factors and every true units item reproduces to ≈ 1e-13. The S48 point is the old
  cycle-132 certificate, which Addenda 50–54 established was residual-certified but unsettled, so an
  incomplete dual agreement there is expected, and the same item passes at 0.026 on the settled
  cycle-181 models. Moving the item to the settled models is the scope Addendum 54 already gave the
  look; it is not a tolerance widened after the fact. **Tolerance stays 0.05 €/MWh.** Benchmark spec
  v2, zero solves. Record the 0.058 → 0.026 tightening as a settling datum, and keep the row's
  address: node 7, Summer, hour 6 is a candidate λ ≠ π hour for the mechanism table.
- **The −19.24 € TSO curtailment is investigated before any arm is reported** — bounded to records,
  ≤ 30 min. A negative curtailment is sign-impossible; at 1 €/MWh it is ≈ 19 MWh block-weighted,
  three or more orders above anything IPOPT's bound relaxation could produce, so it is either an
  availability-profile mismatch in the helper (wrong scenario, hour or scaling field) or a TN RES
  unit whose model bound is not its availability. Identify the block, unit and hours and the
  mechanism. If the helper is wrong, fix it and **re-run the common-Q gate**; if the model bound is
  the cause, it is in production's certified Q as well — report it, do not change the model. A
  sign-impossible value in a helper every arm shares is exactly what the log-derived-quantity rule
  exists for.
- **Launcher self-collision (C\* refusal):** the W108 fix, the v41 re-freeze identical but for code
  pins, per-check refusal logging and the dry-run to solver start are approved as stated. Pattern
  to note in CLAUDE.md alongside the gate-writer rule: **a pre-run check that scans committed
  artefacts excludes the run's own.** The look's usability rule, the frozen W93 settings and the
  curtailment figures (589.43 at cycle 181 vs 589.18 at 132; 660.65 MWh raw) are accepted as
  confirmed.
- **Order unchanged:** C\* diagnostic (v41) → curtailment look → benchmark spec v2 → λ look + common-Q
  gate + arms → phase-mismatched 8 cells → stop for review.

# Addendum 56 — the RES bound slack: kept for this revision; curtailment reporting convention (2026-09-28)

- **Finding accepted** (W109, `f6e3533f`): RES output is bounded by availability **+ 1e-5 pu**
  (`model_construction_helpers.py:188`); with curtailment unweighted, output sits at that edge, most
  visibly at dawn and dusk. Every entry is inside the declared band; the −19.24 € (TSO) and −42.67 €
  (DSO) totals are 1,332 such entries times probabilities and block weights of 373–460. The helper is
  production's own term and matches production bit for bit on 48/48 blocks; W53 had measured the
  same thing on the old models. **Expert's premise in Addendum 55 withdrawn:** "three orders above
  bound relaxation" compared an aggregate against a per-entry tolerance. Compare like with like.
- **Ruling — the slack stays for this revision.** Removing it is a model change: every certified
  number would need re-certification, including the 3 × 3 pair and its continuation (≈ 35 h) and the
  settled SRP1 references, and R would otherwise compare two different models. What it buys is
  ≈ 1.2 × 10⁻⁵ of absolute Q (the −7.9 k€ first-order free-energy estimate) that is **common-mode
  across cells** — the slack is exploited fully wherever energy has positive marginal value, with or
  without storage — so every reported difference is unaffected to second order. **Scheduled for the
  post-revision formulation cleanup**, together with the ESSO economic tie-breaker if H_ess-flat is
  confirmed. Technical note for that cleanup: the slack is unnecessary with IPOPT — lb = ub is handled
  by `fixed_variable_treatment = make_parameter`, and the replay design's mutable bounds write lb = ub
  to the NL file cleanly; before removal, clamp any negative availability in the data at zero, since
  guarding against that is the one legitimate reason such a slack is ever added. The author owns the
  line; the decision not to touch it now is the expert's.
- **Reporting convention.** Repository and benchmark tables: **net, production's definition, is the
  frozen primary**, with the positive and negative parts and raw MWh beside it on the same weighting
  (as W111 already builds them). **Manuscript:** curtailed energy is the **positive part**, in MWh,
  and the model description carries one sentence: RES output is bounded by availability with a
  1e-5 pu numerical slack; the resulting over-production totals ≈ 62 MWh-equivalent over the horizon
  (≈ 10⁻⁵ of renewable energy) and is reported separately. The −7.9 k€ common-mode effect on Q goes
  in the reproducibility note, not the results.
- **Order unchanged.** C\* extension running; spec v2 code edits only until it finishes; then the λ
  look, common-Q gate and arms; then the eight cells; stop for review.

# Addendum 57 — benchmark arms redefined under a no-reverse-flow rule; consensus-gap clause; campaign rulings (2026-09-28)

Rulings on `P5_15_ADDENDUM54_56_CONSOLIDATED_NOTE.md` (`ce96d492`).

- **Decision 1 — the arms were mis-defined by the expert, not mis-built.** Verified against
  `data/SRP1/case9/case9_2025.json` and `case9_2035.json`: the TN's only loads are the three interface
  loads at buses 5, 7, 9 (`fl_reg` 1); generators are three CONV at Pmin 0 (250/300/270 MW) plus wind
  and PV (132 MW in 2025, 245 MW in 2035), all curtailable to zero. The TN is a pure transit network:
  if the DNs net-export in aggregate beyond ≈ 4 MW of losses, no TN operating point exists, whatever
  the generation. Addendum 49's arms let an uncoordinated DN export into a network that cannot
  absorb it, which is a definition error; the coordinated solution resolves the same surplus by
  curtailing DN RES (hence its 166–209 MW imports at the hours where the passive DN exports).
  **Ruling: (a) and (b), both, in benchmark spec v3.** (a) The sweep: continue each arm past failing
  blocks, report-only, and record per block and hour whether the TN can accept the DN schedule and by
  how much not — the manuscript's qualitative statement, "without any interface rule the TN cannot
  accept the DNs' exchange in n of 12 blocks (h hours)". (b) The recourse, in its static form:
  **every uncoordinated arm carries a no-reverse-flow constraint at each interface** (p_int ≥ 0,
  import only) — the connection-agreement rule that is today's practice and the only interface rule a
  DSO can apply without coordination. The DN then curtails under its own arm economy (passive:
  minimum curtailment as the sole term; price-taker: its own objective under the bound); the TSO arm
  is unchanged (fixed interface P/Q, voltages bounded). The benchmark measures **dynamic coordination
  against a static interface limit**, and the claim is against the best NRF arm. Report beside it,
  zero-solve from the Q181 models, the number of interface-hours with reverse flow in the coordinated
  solution — if any, part of the measured benefit is the value of allowing reverse flow, and the
  paper says so. (c), a TN sink, is an instance change and is **not** recommended for this revision.
  The λ-look results stand as recorded (λ ≠ π on 848/864 rows; proper in 807/864 hours; common-Q gate
  bitwise).
- **Decision 2 — the consensus gap: (b), a gap clause, plus (a) for old certificates.** The
  priced interface gap t_sum is a first-order inconsistency in Q itself (energy appearing or vanishing
  at the interface at price π), and the settled references show it closes with settling (23.8 k€ →
  143 €; 34.1 k€ → 663 €). **Certification gains the clause |t_sum| ≤ τ/2** (2,270 €) on top of the
  settling rule; the settled references satisfy it, old certificates do not, and C\* (gap growing
  6.9 → 15.6 k€ with pf rising) is correctly never certified. t_sum recorded per cycle; every restated
  difference reported in gross and Q_cc; verdicts on gross for certified cells (consistent to τ/2 by
  construction). **Old certificates not re-run:** report gross, Q_cc and t_sum; a difference is
  determinate only if its margin exceeds 3 × max(20.8 k€, |t_sum| of the cell) in both gross and Q_cc
  terms, else pending — the Planner recomputes the triage list under this rule from records (zero
  solves) and reports how many cells move. (c), a creep branch on gross, rejected as the note says.
  **Monotone branch amended** for the campaign spec: steps decreasing over the window **and**
  |last step| × L ≤ τ, with L = 60 (2× the longest period measured on the instance) — a linear
  remaining-descent bound, no fit; C\* at 111 €/cycle fails it, as it should. The references'
  certificates are untouched (oscillation branch). P_MAX and L are instance-measured quantities
  recorded in each spec. **C\* manuscript statement adopted as proposed.** H_ess-flat scored as
  reported: the flat direction is real, the ESS channel was the wrong residual to name (expert's
  P_c).
- **Decision 3 — campaign:** (a) the 2030 and 2035 year-ladder cells are **first C2 evaluations**,
  no bitwise gate, both years so the ladder is one configuration (+1 cell); the Planner confirms
  each plan is legal on the current lattice (E/P ≤ 4) and substitutes the nearest legal plan, stated,
  if not. (b) **Same holds as the references** after the first residual pass (AA off, tail on,
  ρ frozen), for parity. (c) **F2 challenger first**, uncertified reporting form frozen in advance;
  note that with τ = 4,539 € the F2 pair's 6,338 € margin is below 2τ, so the settled verdict on
  that neighbour will be "within resolution" — acceptable: the F2 result is the demonstration that
  storage pays at ×2 and a two-node plan emerges, not that plan's optimality against a neighbour at
  6 k€. Design otherwise as the note states. **Machine time: the expert recommends the Mac, now
  (≈ 14–17 h); the VM is not online — author's call.**
- **Order:** benchmark spec v3 (sweep + NRF arms + reverse-flow count) → arms (minutes) → campaign
  spec (criterion with gap clause and amended monotone branch; predictions) → F2 challenger → the
  rest of the nine (ten) cells → **stop for review** with the benchmark report, the settled
  differences in gross and Q_cc, and the recomputed triage.

# Addendum 58 — coordination benefit measured; F2 dead zone; year ladder is convention-dependent (2026-09-29)

Rulings on `P5_15_ADDENDUM57_BENCHMARK_AND_RESETTLE_REPORT.md` (`5612b8f1`).

- **Benchmark accepted; the claim as the manuscript states it.** Coordination beats the best static
  no-reverse-flow arrangement by **+90.9 M€ (13.9 %)**, determinate, like-for-like reviewed: +70.2 M€ of
  TSO conventional energy (0.81 TWh of TN renewables the uncoordinated DSOs leave curtailed by
  mis-timing their demand against the hours when TN renewables are free) and +20.7 M€ of DN flexibility;
  the 4 reverse-flow interface-hours in the coordinated solution (of 864) mean the benefit is not the
  value of allowing reverse flow, and the paper says so. Without any interface rule the TN cannot accept
  the DSOs' exchange in 1/12 blocks (passive) and 8/12 (price-taker). **Interpretation sentence,
  mandatory:** the benefit is the value of dispatching DN flexibility against the TN's marginal value
  λ_t rather than the wholesale price π_t; coordination — or a locational real-time signal computed by
  the TSO, which is coordination by another name — delivers it, a static rule with wholesale exposure
  does not. This pre-empts the "a tariff would do that" objection with the mechanism rather than a
  denial. **Predictions:** the expert's (Addendum 49: proper coordination resolvable and positive,
  living where λ_t ≠ π_t) held; the Advisor's "inside the band" missed; nobody predicted the size.
  **Both arms are tabulated**, not only the best; and **the two blocks where coordination does not
  hold are reported with their magnitude against the DSO multimodality band**: within the band →
  stated as such; beyond it → the coordinated point in that block is a worse local optimum of the
  joint problem than the static arm's, the certified Q(x = 0) is an upper bound there, and the paper
  says so as a limitation (a warm-started re-solve from the arm's point is a post-review check, not
  now). The 18.25 % figure is replaced, not restored: different recourse, different definition.
- **Ruling 1 — F2 pair: report as is.** The gap clause refused a settled objective because of a
  **dual dead zone** in one TSO hour: storage discharge drives the TN's conventional output to its
  lower bound while DN flexibility is marginal, so the interface dual is set-valued (λ ∈ [0, c_flex])
  and consensus converges at the pace of a degenerate dual — a known ADMM property, not a defect, and
  ≈ 500 cycles is not worth buying. Report both F2 cells as "objective settled; interface-consensus
  gap unresolved (degenerate dual, documented)", with margins in gross and Q_cc; the F2 conclusion
  (storage pays at ×2; a two-node plan emerges) stands if the +102 k€ margin exceeds 3 × max(gap,
  slack) in both terms, with the caveat stated. Manuscript limitations paragraph: when the TN's
  marginal cost is degenerate, the interface dual is set-valued and certification reports the case.
  The post-revision tie-breaker (ESSO economic term, Addendum 56's cleanup) removes the dead zone.
- **Ruling 2 — pb_y2025_n5 accepted, flagged.** An "Acceptable" IPOPT exit meets the acceptable
  tolerances, so its objective slack is bounded and far below the 51 k€ margin. **Future specs:** a
  certifying cycle requires `Optimal Solution Found` on every block; a non-Optimal accepted solve
  makes the cycle non-certifying (retry, or the count restarts).
- **Ruling 3 — the year ladder is convention-dependent, and the paper says so.** Gross (the frozen
  convention, no salvage) has 2035 worse than 2030 by +42.5 k€, determinate; net of salvage the
  difference is −2.3 k€, within resolution. The mechanism is the convention: without salvage a later
  investment pays full cost for life left unused at the horizon. **Primary stays gross** (author's
  earlier decision, Addendum 25 era); the year-ladder row carries the net-of-salvage figure beside it
  and one sentence: the investment-year comparison in this instance is decided by the salvage
  convention, not by operation. Whether the ladder stays in the baseline tables is the author's
  call; x = 0 optimal does not depend on it.
- **Machine time (author).** 37 cells, ≈ 50–60 h on the Mac; the VM is not available. Expert's
  recommendation: run them, in **manuscript-priority order** so the paper can be drafted on settled
  numbers as they land — Phase A ladders (the affine slope) → flexibility ladder (break-even ×2) →
  ageing → the remainder — then the Step 5 rows (≈ 10 cells). Step 6 drafting proceeds in parallel
  on the settled results already in hand (benchmark, 3 × 3, references, Phase B, year ladder).
- **Order:** author's machine-time ruling → campaign in priority order, reports at each claim's
  completion (no stop between claims unless a prediction fails) → Step 5 rows → **stop for review**
  before Step 6 tables are frozen.
- **Supplement (2026-09-29) — ageing arms at minimum SoH 0.70, option (b), overriding the Planner's
  (a).** The baseline is C2 with the 0.70 floor, and the floor already binds in 2035 under the
  baseline itself; a harsher calibration binding it earlier is the same physics, not a confound —
  the unit reaches end-of-life sooner, and that is what the row is meant to show. Reproducing the
  arms at 0.50 would make the ageing row the one row in the table that is not a one-parameter
  perturbation of the baseline, against Ruling 2's purpose (one configuration), and would let the
  harsher arms cycle a battery the baseline declares dead. Salvage is reporting-only and does not
  touch gross. The row reports, per arm, the year the floor binds (if any) beside the value; the
  ageing statements measured at 0.50 (Addendum 30 era) are restated under the new row, and the
  soh_min row of Step 5 (floor varied at C2) is the row that isolates the floor. **Prediction:**
  the arms where the floor binds show a smaller value change per unit of cycle life than the
  0.50-based elasticity implied, since end-of-life caps the value before cycle life does.
  **pb_y2025_n5: re-run under the new rule at the end of the priority queue** (≈ 1.5 h), so every
  Phase B certificate has the same strength and the flag leaves the table.

# Addendum 59 — the non-Optimal rule: reading (γ) (2026-09-30)

- **Ruling: (γ).** The rule exists so that the certificate's *evidence* is clean: no cycle whose data
  enter the certification test at the certifying cycle may carry a non-Optimal accepted solve on any
  block. Precisely, in spec v4: the **certifying window** is the span of cycles the test reads at the
  certifying cycle — the three turning points and the period between them for the range test, and
  the preceding swing for "swings not growing" — and every cycle in it must be all-Optimal; the
  gap clause and residuals are evaluated at the certifying cycle as before. **No reset** of the
  count: an Acceptable exit outside the window is history the clean window re-establishes over;
  (α) lets one marginal block veto a cell forever, which is a property of the solver's tolerance
  at that block, not of the ADMM's convergence. (δ) is rejected because a recovery jump inside the
  window is a solver artefact the range test would then be measuring. The retry tier is not touched
  (solve path unchanged). pb_y2025_n5's original certificate stays excluded under (γ) — the
  Acceptable cycle sat inside its window — and its queued re-run stands.
- **Report the recurring block.** Three Acceptable exits on the same DSO block on cell 1 say that
  block is marginal at the tight tail in some cycles; record which block and hours, per cell, for
  the reproducibility note. If the same block recurs across cells, that is a finding for the
  post-revision cleanup (a scaling or bound issue at that block), not a reason to change the rule
  now.
- **v4 re-freeze as the Planner proposes:** reading (γ); gate G8 persistence follows production's
  certificate, not the settling label; W135's router patch in the same freeze so the ageing and pb
  cells need no second one. **Cell 1 re-run** under v4 rather than re-read from v3 records —
  1.5 h buys a certificate whose frozen spec names its rule, and the trajectory to cycle 173 is
  predicted bitwise identical (record that prediction). Cell 1's settled figures (band 1,978 €, gap
  closed, s = +18.7 k€) are accepted; the ≈ 58 k€ provisional B margin landing in the Planner's
  formula-based range settles that discriminating prediction against the Advisor's.
- **Order:** v4 freeze → cell 1 re-run → resume in priority order (≈ 57 h v3 cells, 9 h ageing and
  pb) → Step 5 rows → stop for review before Step 6 tables are frozen.
- **Supplement (2026-09-30) — the certifying window is reading (a).** Addendum 59 mis-described the
  test: it reads the last W cycles at k\*, not a three-turning-point span plus a preceding swing, so
  the enumeration reached back past the evidence and re-created the α veto it had just ruled out
  (cell 1 vetoed to its cap by Acceptable cycles 120–141; pb_y2025_n9 losing a certificate it earned).
  **The principle stands and decides:** the certifying window is **exactly the set of cycles the
  certification test reads at k\*, as implemented** — on the current implementation the last W
  cycles, with the turning-point count, the swing comparison, the range and the gap clause all read
  inside it. Spec v4 states W and asserts that no sub-test reads outside it; if a future
  implementation reads further back, the window is the union of what is read, by this principle, not
  a new ruling. Under (a): cell 1 certifies at 173 (the endorsed prediction), pb_y2025_n9 keeps 174,
  pb_y2025_n5 stays excluded, the other ten keep k\*. **DSO7 2025 Winter** is the Acceptable block on
  both affected cells: named in the reproducibility note; post-revision cleanup item (Addendum 59).
- **Order unchanged:** v4 freeze (criterion v4 with (a), G8 fix, W135 patch, v4 branch) → cell 1
  re-run → priority order (≈ 66 h) → Step 5 rows → stop for review before Step 6 tables are frozen.

# Addendum 60 — the veto bounded: an Acceptable exit within 10× the tail tolerances is clean (2026-09-30)

- **Ruling: option 2.** DSO7 2025 Winter exits "Acceptable" on primary attempts at complementarity
  2.51× the tight-tail tolerance after ≈ 43 iterations, on four cells, every 10–15 cycles on cell 3 —
  so under v4 no window is ever clean and the block vetoes the certificate to the cap, which is the
  α outcome by another route. The veto exists to keep **solver artefacts** — recovery jumps like
  pb_y2025_n5's TSO exit at 2,234× — out of the certifying window. A primary attempt that stops with
  **all four IPOPT metrics within 10× the tight-tail tolerances** is not an artefact: it is a solve
  deeper than anything the production tail (1e-4) certified before Addendum 46, and treating it as a
  veto would make the tail's own history inconsistent. **v5:** such an exit counts as clean;
  recoveries and larger violations still veto; the 10× bound is one order of magnitude, not a value
  tuned to 2.51. Re-freeze (≈ 1–2 h), re-run cell 3 (≈ 1.4 h). **Cells certified under v4 keep their
  certificates** — v5 is strictly more permissive on the veto and identical otherwise, so a v4
  certificate is a v5 certificate; the report states the spec each cell certified under and, where
  v5 would have certified earlier, the v5 cycle from records, without re-runs. Option 3 stays in the
  post-revision cleanup: a block whose complementarity floors at 2.5× the tolerance has a scaling or
  bound-multiplier issue worth finding, not now. **Prediction:** cell 3 certifies under v5; the nine
  break-even-fit cells meet the same block and certify.
- **Reproducibility note:** one DSO block (DSO7, 2025 Winter) converges to ≈ 2.5× the complementarity
  tolerance in some cycles and is accepted under the 10× rule; the rule and the bound are stated.
- **Claim point #3 accepted:** n5 58.0 k€ (6.7×), n9 61.1 k€ (7.3×), n9_e3 195.7 k€ (2.0× the
  uncertified bar); Planner's formula-based predictions 2 of 3, Advisor's 0 of 3; cell-1 bitwise
  prediction held.
- **Order:** v5 freeze → cell 3 re-run → break-even-fit cells (≈ 11 h) → remainder in priority order
  → Step 5 rows → stop for review before Step 6 tables are frozen.

# Addendum 61 — swing noise floor; what τ bounds; window reads enumerated (2026-09-30)

Rulings on `P5_15_ADDENDUM60_SWING_TEST_NOTE.md` (`ef0c3390`).

- **1. Swing noise floor τ/10 (≈ 454 €), adopted.** A three-cycle blip left swings of 226 € and 125 €
  in d_c52e1670's history, and the all-pairs "not growing" test then fails against every normal
  swing forever — a test comparing noise to signal. Swings below τ/10 are not swings of the
  oscillation being measured and are excluded from the growth comparison; replayed on every
  committed record this changes no certificate and certifies d_c52e1670 at 150. The last-pair test
  is rejected as weaker (it forgets an earlier growth trend). **Consistency refinement, to replay
  before adoption:** the same floor applied to the turning-point count (a sign change producing a
  swing below τ/10 is not a turning point). If that replay changes no certificate, v6 carries both;
  if it changes any, v6 carries the swing floor only and the inconsistency is recorded for the
  cleanup. The break-even prediction failed on its first cell for a rule reason, not a solver one —
  recorded as failed under v5.
- **2. What τ bounds — stated, with a determinacy floor.** Two continued cells moved 0.74–0.88 τ
  after the rule certified them: inside the δR = 0.07 budget (each cell ≤ τ), at its edge. The
  manuscript says so with the measured figures rather than claiming τ as an assumption: "cells
  continued past certification moved at most 0.9 τ." **Determinacy rule for differences between
  certified cells: margin ≥ max(3 × the larger bar, 2τ)** — the 2τ floor (≈ 9.1 k€) is the honest
  consequence of cells moving up to τ each; it changes no verdict to date (B margins 43–61 k€; F2
  already within resolution). No margin added to δR: the 3 × 3 R band came from settled
  continuations, not from τ.
- **3. Window reads enumerated per certificate** — accepted as the record for the Addendum 59
  supplement: the window is the union of what the test reads, and v4/v5 list those reads per
  certificate rather than asserting a bound the implementation cannot.
- **Accepted as reported:** cell 3 certified at 148, bitwise as predicted; certificates to date;
  reproducibility note items (DSO7 2025 Winter ≈ 2.5×, clean under 10×; TSO 2035 Spring recoveries,
  non-clean).
- **Order:** v6 freeze (swing floor; turning-point floor if invariant on replay; determinacy floor
  in the report stage) → eight remaining break-even-fit cells (≈ 11 h) → remainder in priority
  order → Step 5 rows → stop for review before Step 6 tables are frozen.

# Addendum 62 — the 5 MWh cell stays uncertified; stop conditions narrowed; break-even fit with bands (2026-10-01)

Rulings on `P5_15_ADDENDUM61_D_CELL_NOTE.md` (`c9b9d189`). All three Planner recommendations adopted.

- **1. v6 stands; d_36686489 is recorded uncertified.** The refusal is the right verdict: the cell
  was still sliding at −85 €/cycle at its cap after five TSO recovery exits, and the principled patch
  (excluding swings that span a non-clean cycle) certifies it only by leaving nothing to compare,
  after which it moves −1.01 τ — breaking the manuscript's "≤ 0.9 τ" statement. A rule that certifies
  by running out of evidence is not a rule. **Mechanism to record:** this is the largest storage cell,
  and repeated TSO recoveries after the residual pass are what Addendum 58's dead zone looks like at
  baseline price — storage discharge driving the TN's conventional output to its lower bound while DN
  flexibility is marginal, so the interface dual is set-valued. The limitations paragraph gains "the
  degenerate case appears with large storage" and names this cell. d_3632b0ae's certificate at
  0.997 τ stays flagged.
- **2. Stop conditions narrowed — into CLAUDE.md.** A certification-status prediction ("certifies")
  has a fallback by construction: the uncertified reporting form, which already decides large
  margins. Its failure is recorded and the campaign carries on. The campaign stops only on a harness
  fault, a gate failure other than G6/G27, or a failed **margin or sign** prediction — which includes
  the case where the uncertified form leaves a manuscript claim indeterminate (margin below
  max(3 × larger bar, 2τ)). Three of the last four stops were "certifies" predictions failing with no
  consequence for any verdict; that is a day of Mac time.
- **3. Break-even fit with bands.** The affine fit includes the uncertified point as an **interval**
  (its uncertified bar plus the 2τ floor), the slope is reported with the min/max over that interval
  and the break-even as a range; the certified-only fit is shown beside it. The conclusion must hold
  under both fits — it does, at ≈ 72 k€/MWh margin to cost against any plausible shift. Prediction
  on record: both fits agree on the slope within 5 %.
- **Order:** remaining five D cells (≈ 7 h) → claim point #12 → remainder in priority order → Step 5
  rows → stop for review before Step 6 tables are frozen. No further stops on certification status.

# Addendum 63 — flexibility ladder at m = 1.5: (a) + (c); stops that idle the machine (2026-10-02)

Rulings on `P5_15_ADDENDUM62_H_CLAIM_STOP_NOTE.md` (`53cd9e57`).

- **Decision 1 — (a) and (c), as the Planner recommends; (b) rejected.** At m = 1.5 the unit's
  value − I is −14.45 k€ against an 18.75 k€ bar: the sign is not resolved and the report says so in
  the frozen uncertified form, with the cause stated — a settled cost refused only by a consensus
  gap holding at ≈ −2.5 k€ against the τ/2 limit of 2.27 k€, the F2 pattern at a quarter of the size.
  (b) would scrape through at 1.01× on Q_cc, which is a rule changed to rescue a claim; running
  longer buys nothing because the gap is not moving. **(c) restates the ladder as a crossing:** the
  unit pays at m = 2 (+46.3 k€, determinate at 3.77×, prediction held); at m = 1.5 the sign is
  unresolved; under monotonicity of value − I in m the break-even multiplier lies **at or below
  ≈ 1.75** (linear interpolation from the lower edge of the m = 1.5 bar), nominally 1.62, with the
  lower end of the bracket open — the m = 1.5 bar admits a positive value, so "at most 1.75" is the
  only bound the data supports. **Monotonicity is justified, not assumed:** by the envelope theorem
  on both optima, d(value)/dm equals the base-price-weighted flexibility that storage displaces,
  which is non-negative; the manuscript carries that sentence. **Optional, author's machine time:**
  one cell at m = 1.75 (≈ 1.5 h, after the I and J cells) brackets the crossing from the other side
  and would turn "at most 1.75" into an interval; recommended if the ladder is a headline row.
  Prediction scored: sign held, size missed low, determinacy missed. Interface prices of
  h_f9eae48f are read zero-solve for the dead-zone table.
- **Decision 2 — the I and J cells restart now**, with the author's go-ahead; they do not depend on
  this claim. **Rule into CLAUDE.md:** a stop raised for a reporting ruling does not idle the machine
  when the queued cells are independent of the ruling — the stop applies to decisions that change
  what runs next, and the report is written while the queue proceeds.
- **Order:** I and J cells → claim point after J (≈ 5–6 h) → remainder → optional m = 1.75 → Step 5
  rows → stop for review before Step 6 tables are frozen.

# Addendum 64 — campaign complete; rulings on the consolidated handback; Step 6 package (2026-10-05)

Rulings on `P5_15_ADDENDUM63_CONSOLIDATED_HANDBACK.md` (`0a56ba88`). Campaign: 42 cells, 32 settled;
60 claims, 50 determinate; no recorded sign prediction failed.

- **Author's decisions, expert's recommendation: run both.** (1) The m = 1.75 cell (≈ 1.5 h): the
  flexibility ladder is a headline row — "storage pays at ×2; break-even between 1.5 and 1.75" is a
  sentence a referee can use, "at or below 1.75" is not. (2) The soh_min 0.50 row (plumbing + ≈ 1.5 h):
  it is the one Step 5 row that needs a solve, it isolates the floor the ageing row shows binding in
  2035, and the design is bounded — variant key in new files, Advisor design check, bitwise gate
  against the settled C2_calfade unit through 172, frozen spec. Together ≈ half a day; the Mac is idle.
- **3. Slope: b + c/4**, the 4 h marginal slope the break-even price is built from. The 5 % prediction
  is scored at the midpoints and held on both readings (0.59 %, 0.31 %); the worst corners (7.1 %,
  3.7 %) are the intervals' sensitivity and are reported as such, not as the prediction's test.
- **4. Certificates on a non-clean turning point: keep, flag, uncertified form beside** — the veto's
  window was defined as the cycles the test reads, and these turning points sit outside it, so the
  certificates are valid under the frozen rule; no verdict depends on them (J 4 MWh at 11.1× when
  treated as unsettled; break-even margin 61.3 k€/MWh instead of 63.5). **The manuscript uses the
  conservative figures** (61.3 k€/MWh; J at its uncertified bar) so that nothing rests on the flag.
  Post-revision rule question, recorded: "a non-clean cycle cannot be a turning point" — the mirror of
  d_c52e1670, where a blip blocked a certificate.
- **5. The determinacy floor changed one verdict — expert's Addendum 61 prediction failed on it.**
  E:C2_calfade vs C3 (+9,329 €, 0.71× threshold) is reported within resolution; S1 restated "within
  resolution of C3 at a 0.70 floor". Recorded as the prediction's failure, not re-litigated.
- **6. Ageing restated at the 0.70 floor:** ε_AE 1.04–1.85 (was ≈ 0.6 at 0.50); the expert's
  floor-binding prediction — point estimate held on C2_calfade, within resolution (0.20×); on C2 and
  C4 the floor never binds and ε_k rises determinately — scored **"consistent in direction, not
  resolved at this precision"**, S5 restated. The late-life-tail reading of why the floor raises ε_AE
  stays a hypothesis in the limitations, not a result.
- **7. Dead-zone table:** (a) l_0ee93aca entered by its signature with the cause stated (two TSO
  recoveries reset the rule before the gap clause); (b) h_f9eae48f's TSO-lever column report-only,
  labelled post hoc.
- **Before the tables freeze:** the net-of-salvage scorer, validated on 8 of 60 rows, reproduces three
  more recorded rows from records, zero-solve — chosen from different claim families — before any
  net figure enters a table. The TSO recoveries' cause (2035 Spring and three other blocks) goes to
  the post-revision cleanup with the logs preserved, as the handback says.
- **Results the manuscript carries** (gross primary; net beside where ruled): the handback's table
  stands, with two sentences the expert wants verbatim: *without ageing the unit is at break-even
  (−4.1 k€, within resolution); with the baseline calibration and the 0.70 floor it loses 31.6–73.7 k€
  across the aged arms, determinately* — the degradation-aware framing in one line — and the
  coordination interpretation sentence of Addendum 58.
- **Step 6 package, after the two cells:** the Planner assembles (i) the results tables, gross primary
  and net beside, with bars, multiples and certification status per cell; (ii) the certification
  paragraph — Boyd residuals, the settling criterion with three turning points, swing floor τ/10,
  range ≤ τ = 4,539 €, gap clause τ/2, the 10× clean-exit rule and the non-Optimal window rule, the
  measured post-certification movement (≤ 0.9 τ), bitwise replays, and the uncertified form; (iii) the
  limitations paragraph — the dual dead zone with large storage and high flexibility price, the TSO
  recoveries, the RES bound slack with its ≈ 62 MWh-equivalent, the monotone-branch caveat; (iv) the
  reproducibility note — bitwise 72/72 at 3 × 3, 3/3 and every gated cell at SRP1, the Acceptable
  blocks, the machine and solver provenance; (v) the prediction scorecard as supplementary material;
  (vi) a response-to-reviewers map from each reviewer item (R3.6 rows included) to the table or
  paragraph that answers it, with the 18.25 % figure's replacement stated explicitly. **Stop for
  review** with the package; the author and the expert review the tables before the manuscript text
  is written.
- **Housekeeping.** The brief is edited only by the expert and synced to the repository; the Planner
  commits it but never edits it. The "file changed on disk" was the Addendum 63 sync landing after
  the Planner's commit; if it was the handback instead, the author diffs it before forwarding.

# Addendum 65 — Step 6 package reviewed; tables freeze after five closing reads (2026-10-06)

Review of `P5_15_STEP6_PACKAGE.md` (`54c7f899`) and tables `w157_step6_tables_a64` (`b913ea94`). The expert
read the package in full. **Verdict: accepted, with the rulings and closing reads below; the tables freeze
once the reads are in.** The two Addendum 64 cells landed as predicted: the unit pays at m = 1.75
(+14.2 k€, 1.32×) so the break-even sits between 1.5 and 1.75; the 0.50 floor adds +4.9 k€, within
resolution, and the unit still loses 59.5 k€ determinately — the floor is not what makes storage
uneconomic at SRP1.

- **1. Net-of-salvage validation: substitute accepted.** Net differs from gross by the salvage term
  alone; the form reproduces all 60 gross results bitwise, the identity Q − salvage − Q_net = 0 holds on
  all 50 cell records, and the 20 independent G rows match to 5 × 10⁻⁸ €. The label "validated by form +
  salvage identity" stays on every non-G net figure. Net remains secondary.
- **2. The ≥ 0.95 τ count: ten**, the Planner's scope — every certificate the tables use, superseded
  certificates excluded, bitwise twins counted once. The certification paragraph says: "ten of the
  certificates the tables use stop within 5 % of τ; the rule bounds each cell's stopping error at
  about τ, not well inside it." `i_5a6a88b4` (monotone, 0.939) is flagged in T2 beside them.
- **3. `d_36686489`: corrected attribution accepted, and the expert's Addendum 62 mechanism is
  withdrawn for this cell.** The records show a small positive gap (+1.1 k€) and a growth-test failure
  after five TSO recoveries — not the dead zone. The limitations paragraph states the recoveries.
- **4. Unsourced items — three become sourced by cheap reads, one by a sentence:**
  - *Threading.* Read the launch environment the v6 `launch_commands` imply and the current shell for
    `OMP_NUM_THREADS`; record "unset (OpenMP default)" or the value. Note in the reproducibility
    paragraph that HSL_MA97 is designed for bit-compatible results independent of the thread count
    (Hogg & Scott, HSL_MA97 report, 2013) — the Planner confirms the statement against the HSL
    documentation in the author's `coinhsl-2023.11.17` archive before it is written — which is why the
    bitwise replays hold regardless of the setting.
  - *Linear solver banner.* The campaign's IPOPT logs are preserved on disk though not committed: read
    one v6 banner for "running with linear solver ma97" (and one ESSO log for ma57) and record file, hash
    and line, zero-solve. The P5.12-R banner is then corroboration, not the source.
  - *Python, Pyomo, macOS at campaign time.* Read them now from the same machine and environment and
    verify the environment is unchanged since the campaign (`conda-meta/history` timestamps, no entries
    after the last v6 launch); record "read 2026-10-06; environment unchanged since the campaign".
  - *Re-optimised plan per ageing arm.* Not run, and a sentence suffices: under every aged arm the unit —
    the smallest lattice point — loses determinately, and Phase A measured the value as affine in size
    with slope below cost, so x = 0 is the optimal plan under each aged arm provided that affinity
    holds under the arm (stated as the assumption); under no ageing the unit is at break-even within
    resolution and the plan is indeterminate at this precision. R3.6 is answered with the fixed-plan
    rows plus this sentence.
  - *Remaining drift of the monotone certificates at L = 60.* Stays unsourced and is stated as such;
    seven of eight have range/τ ≤ 0.171 and are not at issue.
- **Closing reads before the freeze (zero-solve):** (a) confirm from T2's "certifying spec" column that
  every cell the manuscript tables use was certified under the current configuration or is a settled
  reference — if any old-configuration certificate remains in a manuscript figure, name it and state
  its configuration in the reproducibility note; (b) T6 for the manuscript shows **both** uncoordinated
  arms, not only the best (Addendum 58); (c) the banner, threading and version reads above.
- **Sentences accepted** as proposed for the m = 1.75 and soh_min rows (with "≈ 1.63 by linear
  interpolation" in the first), and the two verbatim sentences confirmed in place. The two non-H1
  benchmark blocks being positive beyond the band means Addendum 58's upper-bound limitation is not
  triggered — no sentence needed, as the package says.
- **Scorecard:** accepted as supplementary material in full, the expert's misses included (#2–4, #8,
  #29, #30 in part, #45 unresolved). It is the paper's honest record of how the numbers were reached.
- **Division of labour for Step 6 writing.** The Planner exports T1–T10 manuscript-ready (CSV and LaTeX;
  k€ to one decimal, €/MWh to the nearest 10, multiples to two decimals) with the four draft paragraphs
  and the reviewer map; the author writes the manuscript and the response letter; the expert reviews
  the methods, certification and limitations text once drafted. **Main-text set (expert's suggestion,
  author decides):** headline x = 0 at SRP1; break-even fit; flexibility ladder 1.5/1.75/2; R3.6 ageing
  rows; year ladder gross and net; discount; benchmark with both arms and the mechanism sentence; the
  3 × 3 instance (x = 0 optimal, R ∈ [0.909, 0.934]). **Supplementary:** Phase B, C/G, L (F2),
  certification statistics, dead-zone table, scorecard.
- **Order:** closing reads → tables frozen (hash recorded) → manuscript-ready export → the author
  drafts → expert review of methods text. The Mac has no queued run; the post-revision cleanup list
  (RES slack, ESSO tie-breaker, DSO7/DSO5 Acceptable blocks, TSO recoveries, non-clean turning points,
  monotone drift at L = 60) is carried in TASKS.md as a separate section.

# Addendum 66 — tables frozen; expert's review of the methods paragraphs (2026-10-07)

Frozen tables `frozen_step6_tables_v1_590088fe.json` (sha256 `590088fe…`), export W160, paragraph check
W161 (`2a1d7f92`: 163/164 figures match the tables, the 164th checked against W153). **The drafting set is
accepted.** The certification-statistics relabel (65 from the first residual pass; 63.5 from the restart
after a lapse) is correct and stays.

- **Two wording rulings.** (1) *Reproducibility:* the recorded statement — one thread everywhere, set and
  logged per evaluation, HSL built without OpenMP so MA97 ran serially — replaces the MA97
  bit-compatibility sentence; the expert's claim was not confirmable from the archive and its citation
  was imprecise, so it is withdrawn and the recorded fact is used. (2) *Ageing:* "≈ 0.6" in Addendum 65
  is corrected to the records' **0.41–0.62** on the three comparable arms at 0.50.
- **Methods review of `paragraphs_v2.md` — edits the Planner applies as v3 so the author integrates
  clean text.** General: strip every internal reference from manuscript prose — addendum and decision
  numbers, W-numbers, cell identifiers (`d_36686489` → "the 5 MWh evaluation"), "re-settled", C\* →
  "the reference corner plan"; identifiers stay in the supplementary tables with a key.
  *Certification paragraph:* (a) "every local NLP must solve successfully" → "every local NLP returns a
  clean solution (defined below)"; (b) **replace** "Because the objective converges more slowly than the
  residuals" — the Planner's own correction in Addendum 53 stands — with "Passing the residual test does
  not imply that the objective has settled: on the reference evaluations the objective moved by a
  further 15–21 k€ after the residuals first passed"; (c) "penalty frozen" → "the ADMM penalty
  parameters ρ frozen"; (d) define P̂ (the period measured between the first and third turning points)
  and P_max (the longest period measured on the instance, 30 cycles); (e) τ: "τ = δR·V/4, with δR = 0.07
  and V the settled SRP1 value of the reference unit, so that a ratio of two values — four evaluations,
  each in error by at most τ — is resolved to δR"; (f) "Ten of the N certificates the tables use" with N
  filled; (g) **define the settling slack** of an uncertified evaluation exactly as the scorer computes
  it (the objective's movement from its old certificate, or from the first residual pass where no old
  certificate exists — the Planner states which); (h) "42 re-settled SRP1 cells" → "42 SRP1 evaluations
  run under this rule"; "four cells added under Addendum 64" → "four further evaluations".
  *Limitations:* (i) "sits at a kink" → "is at zero activation, where its cost is non-smooth"; define λ
  as the interface price (the consensus dual); (j) drop "(Decision 3)"; (k) "frozen near 0.69" → "the
  power-flow primal residual, as a fraction of its tolerance, stuck near 0.69"; (l) give the recovery
  count: "21 cycles in 10 evaluations, all in transmission blocks"; (m) the hypothesis paragraph keeps its
  bracket label in the draft and loses it in the manuscript, where the sentence "This is read as …
  has not been measured" already marks it. *Reproducibility:* accepted as v2; add the version lines
  (IPOPT 3.14.18 with MA97/MA57 from the campaign banners, Python 3.11.11, Pyomo 6.9.5, macOS and the
  M2 Max 32 GB machine) in one sentence, with "read on 2026-10-06 from the unchanged environment" in
  the supplementary note rather than the main text.
- **Reviewer map and scorecard:** accepted; the map's "editorial" rows are the author's to write.
- **From here:** the author drafts the manuscript and response letter from the export and v3
  paragraphs; the expert reviews the drafted methods, results and limitations sections and the
  response letter's technical answers when the author sends them; the Planner stands by for figure
  checks against the frozen tables (every number in the manuscript traced to a frozen-table cell or a
  named record, as W161 did for the paragraphs). No run is queued; the cleanup list is in TASKS.md.

# Addendum 67 — methods paragraphs final (v5); three departures from the expert's edits accepted (2026-10-08)

`paragraphs_v5.md` (`097421f8`); figure check W163 on v4 (`ff66c75f`), 116/116; v5 adds three wording
changes so the text says what the code does, and one recorded figure (tail offset 1.0–1.2 × 10⁻⁶).
**Accepted as the manuscript's methods text.** The three departures are the Planner correcting the
expert against code and records, and each is right:

- **τ uses V = 259,375.33**, the reference value in force when the criterion was adopted and frozen into
  every spec (0.07 × 259,375.33 / 4 = 4,539.07 €), not the later settled value — the expert's edit named
  the wrong V. The text says τ was fixed from the reference then in force and kept frozen; the 2.3 %
  difference to the settled value is immaterial and is not re-derived.
- **"Successful", not "clean", for the residual test:** the test accepts any exit the modelling layer
  accepts as optimal, IPOPT's acceptable level included; cleanliness is the certification rule's
  separate decision. Two concepts, two words.
- **The 15–21 k€ movement is anchored at the earlier residual-based certificates**, ten cycles after
  the first residual pass, not at the pass itself — the sentence reads "after their residual-based
  certification".
- **"One machine"** has no host field in the evaluation records and stands on the author's attestation,
  supported by the identical interpreter path and memory size recorded in every run; the
  reproducibility note says so in those words.
- **From here:** the author drafts; the expert reviews the drafted methods, results, limitations and the
  response letter's technical answers; the Planner traces every manuscript number to a frozen-table cell
  or named record. The programme's decision record closes at this addendum until the draft arrives.

## Update obligations

At the end of Step 1 the Planner rewrites the "CURRENT SOURCE OF TRUTH" head of
`REVISION_CONTEXT.md` to reflect: the withdrawal of the `C*` feasibility-boundary claim, the
warm-start mechanism, the reformulated ESSO, and the reopened numerical programme. Historical
sections are not rewritten. `COWORK_HANDOFF.md` is marked superseded by this brief.

# Addendum 68 — rulings on `P5_15_STEP6_SUPPORT_REPORT.md` (W164–W167); §2 corrected to the code (2026-10-07)

Support order complete: number checker W164 (86/86 letter numbers match; one wording mismatch), figures 1–5 W165
(deterministic PDFs, captions drafted), equation-vs-code audit W166 (68 rows on the **submitted** §2.3/Appendix A at
Overleaf `6191c6c`: 22 match, 38 differ, 8 retired), nomenclature and year tables W167. **Accepted.** The audit was run
on the old text by construction (the order pinned `6191c6c`); the revised §2 pasted since (Parts A–D of
`section2_expert_draft.tex`) is corrected here to what the audit found, in `STEP6_ROUND1_CORRECTIONS.md` §A, and is
re-audited as W171 once pasted.

**Decisions (the report's 1–7).**

1. **R3.3 wording — (a).** The schedule is one per block, common to all market × operation scenarios, in all three
   agents; non-anticipativity is hard (aliasing), the priced term is the DSO-side interface premium (row 18) and never
   touches the storage; the closure is soft (slack ≤ 0.05 E^Av + 1e-5, penalised 1e3 €/MWh inside Q). The letter's
   R3.3 is rewritten from the Worker's plain statement (`STEP6_ROUND1_CORRECTIONS.md` B.1); the CONFIRM comment goes.
   The same facts enter §2.3.3 and §2.3.6 (corrections A.3, A.6).
2. **Two horizons — (a).** SRP1: 2025/2030/2035, five-year blocks. The 3 × 3: 2025/2028/2031/2034/2037, three-year
   blocks. §3 states both, the T11 caption states the 3 × 3 horizon, §4.5 says the ratio R compares the instances as
   run (scenario set and block length both differ) and does not isolate the scenario effect; the letter's R1.2 adds
   "run on a finer five-block horizon" (B.6). No re-run. W170 records how the 0.933 prediction was computed, so that
   §4.5 can say what it was a prediction of.
3. **§3.1 — as recommended.** Three trajectories at 0.35/0.55/0.10; I(x) is their expectation (linear in the costs);
   the 2025 expectations are 256,317.32 €/MVA and 253,877.68 €/MWh. The table is replaced by W167's 2025/2030/2035
   fragment (energy row ×1.25 against the submitted file, which divided the 4 h cost by 5). §3.1 says I(x) is the
   expectation; the letter's "Further changes" names the correction (÷4 h, not ÷5).
4. **Figure 4(a), the x = 0 reference — (b).** The reference `ref:7aa017f0` (k* 181) was certified under settling rule
   v1; the paper describes v6. W168 replays the v6 decision on its committed per-cycle record (zero-solve). Rulings in
   advance: the frozen tables are **not** re-frozen on the outcome — if v6 certifies at k*' ≠ 181 the certified value
   differs by less than the band (0.93 τ) and the caption and §4.7 state both cycles; if the record ends before v6 can
   decide, the Planner stops and reports, and a bounded continuation (≤ 60 cycles) is a separate ruling; if v6
   refuses on the record, T1's B rows carry a footnote and §4.7 says under which rule the anchor was certified. The
   caption keeps both facts in every case. **Prediction:** certifies under v6 at a cycle in [174, 195], window range
   ≤ τ, value within 0.93 τ of the tabulated one.
5. **Letter wording — all seven fixed** (B.2–B.9): "the other comparison that depends on the convention"; the Phase B
   neighbour named (two-node plan at the doubled flexibility price); T8 has no terminal-AE column → PV-weighted
   available energy, and the floor never binds within the horizon; R3.6's calibration sentence rewritten with all five
   aged arms in order; "48 blocks" → 48 network blocks plus three storage agents; R1.5 → Figure 1 of the revised
   manuscript; R2.8 stays a status note until the references are in. "Revision 1" / "first reply" are consistent.
   "Single machine" stands on the author's attestation (Addendum 67).
6. **main.tex l. 1058 (60 %/80 % floors "are evaluated") — removed** in the §3.4 rewrite, with the 0.5 %/2 %
   calendar sentence (map §B).
7. **Figure 1 panel (b) — kept.** Both fits use all ten points; the 2 h panel shows the reader the same fit on the
   other duration.

**§2 corrections from the audit (`STEP6_ROUND1_CORRECTIONS.md` §A; the author pastes, W171 verifies).** The audit's
H-impact findings that the pasted §2 had not yet absorbed: scenarios are joint within one block model, not instantiated
per scenario (A.1); available energy uses the end-of-block SoH, and the agent publishes the degraded capacities to the
networks every cycle — the path by which degradation reaches operation (A.2); the network storage model has the
active-sum limit p^ch + p^dch ≤ S, the capability circle, the relaxed normalised complementarity (1e-4), the 0.10–0.90
window, the 0.5 E^Av initial state and the soft closure with its penalty inside Q (A.3); the agent's net power carries a
penalised slack pair (1e3) and the pro-rata allocation acts on the net power, vacuous for every single-cohort plan in
the tables (A.4); the agent's objective is the slack penalty + ε-throughput (1e-5) + AL terms, excluded from Q, with the
circle on (P^Net, Q^Net) (A.5); the deviation part of the settlement is inside Q and the contracted part is not, the
voltage pin is solver-only (A.6, A.7); the 3 × 3 cells were stopped by the production exit (ten consecutive passing
cycles) and continued, not by the settling holds — §2.2.7 and §3.5 say which instance used which rule, and T11 is a band
(A.7). Notation: n_w for the window (W is taken), 𝒱 for the reference value (V is voltage); the cohort window written
half-open (A.8). Parameters for §3.5, as found in force: η 0.97/0.96; SoC 0.10–0.90, initial 0.5; closure slack
0.05 E^Av + 1e-5 at 1e3 €/MWh; complementarity 1e-4; ε 1e-5; c^σ 1e3; S_ref 2.5 MVA (normalisation 2 S_ref = 5 MVA);
ρ initial v 0.0077 / pf 0.198 / ess 0.01; balancing ratio 5 (pf decrease 3), ×/÷1.5, clamp [1e-4, 1e4], per-channel
freeze after 10 unchanged cycles, backstop 200, ESS exempt until its dual ratio < 1 on 5 cycles; Boyd 1e-5/1e-4; ten
consecutive cycles; caps 3 × 3 500, SRP1 gated N_old + 100 / ungated min(k0 + 109, 300); σ = 93,635,360;
κ_ESSO = σ / median w_b (227,210.997 SRP1; 386,258.694 3 × 3); w_b = Y_y D_d 1.02^{−(y−y0)}; AA type-II memory 5,
Tikhonov 1e-10, ratchet safeguard, `keep_memory`, cleared on ρ change and on a solve failure, off when every channel
passes; tail compl_inf_tol 1e-6 (TSO production 5e-4, DSO 1e-4); ESSO IPOPT tol 1e-10 / acceptable 1e-9, MA57; networks
tol 1e-5 / acceptable 1e-4, MA97, max_iter 500, recovery acceptable_tol 1e-4 / acceptable_iter 1; proximal γ = 0; row 18
α = 0.5; voltage pin 9e4 solver-only; baseMVA 100.

**Appendix A.** The audit's A4–A35 rows describe a different algorithm from the printed one (three-agent global-variable
consensus with a z-update and per-agent duals; normalisation 1/(2 S_ref); κ_ESSO; one ρ per channel under residual
balancing; convergence checked once per cycle after the ESSO; targets = z; interface channels on the expected interface
normalised by the DN interface rating, V unnormalised; the TSO proximal term at γ = 0; the publication of available
capacities). The expert drafts Appendix A from the audit's "Implemented but not in the text" list, in the same form as
§2; it is audited after it is pasted (round 3).

**Blocked on the author, resolved by order:** the reviewers' comment document and `manuscript_submitted/` are committed
to the repository from the Air checkout (`manuscript_review/`, `manuscript_submitted/`; `manuscript/` added to
`.gitignore`), so W164 can check the quotations and `latexdiff` has its base. The 15 year-indexed figures stay
2025-only; the l. 1642 sentence is corrected to say that later years are drawn by the same generator with the year in
the seed (author).

**Next order:** `STEP6_ROUND1_CORRECTIONS.md` §C (W168–W171). Predictions recorded there. Nothing else runs.

# Addendum 69 — rulings on `P5_15_ADDENDUM68_ROUND1_REPORT.md` (W168–W171); errata to Addendum 68 (2026-10-08)

Order complete; accepted. Predictions: W168 held (v6 certifies the x = 0 reference at 181, range/τ 0.927, ΔQ = 0);
W169 held on the reading ruled below; W171 **failed** (two H rows in §2.3, five in §2.1 and three in §2.2 that the
prediction did not cover). The failures are the expert's, recorded here with their cause: §2.2.7 was drafted from
`paragraphs_v2` plus the Addendum 66 edits instead of from the accepted v5 (Addendum 67), so two definitions the Planner
had corrected regressed; "never binds" in Decision 5 came from the expert's misreading of the Figure 3 caption; Algorithm
1 was written from `STEP4_DFO_METHOD.md` rather than from the three search campaigns' code. Corrections:
`STEP6_ROUND2_CORRECTIONS.md`.

**Errata to Addendum 68.** Decision 5: the end-of-life floor binds in 2035 under the baseline (C2_calfade), C3_unit
and C3_midblock, and not within the horizon under C2 and C4 (T8 `floor_year_070`; figure 3); the letter's R3.6 and
the §4.3 text say so (round 2, D.1). The §3.5 parameter list: the recovery options (acceptable_tol 1e-4, acceptable_iter
1) are set for the TSO only; the DSOs run without them.

**Decisions (the report's 1–9).**

1. **Floor** — as the Planner states; erratum above; letter D.1.
2. **l. 639** — the Planner's sentence adopted (round 2, B.3): every certificate was decided, or re-decided from its
   committed record, under the rule and certifies at the cycle reported; in evaluations continued from an earlier run
   the holds began at that run's stopping cycle. `ref:bd504ecf` needs no new replay (W142 re-decided it at 172).
3. **P̂ and the settling slack** — the v5 definitions restored verbatim (round 2, B.2–B.3): P̂ spans the three most
   recent turning points; the slack is the movement between the earlier, residual-based certificate and the cap. The
   M-impact clauses of T2-4–T2-6 (τ/100 sign floor, three excluded cycles, lapse reset, the monotone branch's
   no-sign-change window and half-window test, gap clause and clean veto on the monotone window, the gap-corrected
   condition and the largest-over-the-pair rule for uncertified bars, "at least" for the threshold) are printed too: a
   referee re-deriving the rule must get the rule.
4. **Algorithm 1 — rewritten to the search as run, and the claims reduced to the recorded certificates** (round 2,
   A.1–A.4). Decision vector: one (S, E) per node and one investment year common to all nodes, n = 7; per-node timing
   and staging not searched (stated; the master admits them). Two variants named: A (n + 1 OrthoMADS directions,
   infeasible rejected, full admissible unit box at Δ = 1, cap 30, stop for review beyond) for s47 and s51; B (2n
   directions snapped to the unit frame, Δ = 1 throughout, completion to n + 1) for s53. Acceptance as run: production
   exit (ten consecutive passing cycles, cap 500) and the margin max(bar_x + bar_inc, σ_Q), σ_Q = 18,449.66 €; every
   evaluation in a reported comparison re-settled afterwards on the final code under v6, comparisons at
   max{3b, 2τ}. Certificates as recorded: at x = 0 all fourteen admissible unit neighbours evaluated (the full box; no
   positive spanning set exists at the boundary); at the F2 plan the final poll's neighbours, which do not positively
   span, and the plan is reported as better than each evaluated neighbour (seven determinately, five within
   resolution), **not** as a mesh-local optimum. The MADS convergence theory is cited as the framework's property, not
   as a result of these runs. The recourse equation of §2.1 is written as the within-block two-stage problem with the
   scenario-free u⁰ (storage schedule, TSO interface exchange, DSO committed import). The three search campaigns ran on
   earlier code states, gated inert at SRP1 by the bitwise gates the brief records; §4.7 says so.
5. **§2.3** — T3-1 and T3-2 fixed (round 2, C.1, C.6): consensus on the net active and reactive power only, each model
   with its own split; the commitment expectation and deviations run over market × operation pairs and vanish only with
   a single scenario. The L rows T3-3 (consumption sign in both copies), T3-5 (pre-day state), T3-6 (penalty in each
   network copy), T3-9, T3-11 fixed with them.
6. **W169 scoring** — "at the lower bound to solver tolerance": held at every current certificate (the superseded
   `pb_y2025_n5` certificate is outside the tables under Addenda 58/59). The two printed sentences adopted (C.3, C.5);
   the min(P^Ch, P^Dch) detector is stated as recorded, not enforced, with its maximum (3.6 × 10⁻⁵ of the rating).
   Coverage for `d_c52e1670` accepted as scoped.
7. **The 0.933 prediction** — cited as recorded: the price-spread prediction for the draw that was run, a ratio of
   discount-weighted average 4 h spreads over the two instances' horizons (90.53/97.02); 0.915 on the totals basis
   given beside it; both inside the measured [0.909, 0.934]; neither isolates the scenario effect (Decision 2 of
   Addendum 68). Not replaced after the run.
8. **Letter** — R3.5's clause on the 0.5 %/2 % sentence dropped (the reviewers never saw that text); the second half
   of B.6 applied; the cost-file correction named in "Further changes"; the stray `"` removed; R1.5 and "Figure 15"
   wait for the PDF the reviewers read, which the author adds to `manuscript_submitted/`.
9. **Housekeeping** — k(C4) = 22,429; §3.5 and §4.7 remain forward references until written; TN generation priced at
   the scenario market price stated in §3; Q's component list completed (market-priced TN generation, DN flexibility,
   load curtailment at its price, the closure and feasibility slack penalties at their solver bound, RES curtailment
   uncosted and reported as energy — round 2, B.5); the per-element σ captures stay hash-recorded.

**Also from W171b, adopted:** the 3 × 3 sentence of §2.2.7 says that only the reference evaluation was continued and
that the band is on the ratio (B.3); the search's cache is keyed on candidate and configuration (A.2).

**Next order:** `STEP6_ROUND2_CORRECTIONS.md` §E (W172 Appendix A audit and the DN interface ratings; W173 the two §2.1
CONFIRM points and the neighbour counts; W174 re-check after the round-2 paste). Predictions: W173 — the x = 0 count
is 14 and the F2 poll-set count is 12, with 17 the number of evaluated box points; W172 — at most three H rows in
Appendix A, none in the z-update, the dual updates or the residual-balancing rule.

# Addendum 70 — rulings on `P5_15_ADDENDUM69_ROUND2_REPORT.md` (W172–W173); §3.5–3.6 drafted (2026-10-08)

W172 (Appendix A audit, 141 statements: 120 consistent, 1 H, 3 M, 17 L; the DN interface ratings read) and W173 (the
§2.1 CONFIRM points and the neighbour counts) accepted. Addendum 69 and the round-2 file were not in the repository
when the order ran (the author's commit on the Air had not been made); the predictions are scored here from the
report. **W172 prediction** ("at most three H rows; none in the z-update, the dual updates or the residual-balancing
rule"): held on the count (one H row, in A.6) and on the equations; the *sentence* calling the z-update "the minimiser
of the sum of the three agents' storage terms" was imprecise (M), which the prediction did not cover. **W173
prediction** ("x = 0 count 14; F2 poll set 12; 17 evaluated box points"): 14 held, 17 held, **12 failed** — the final
poll had ten points, and the twelve `l_` cells of Addendum 63 include one that is not a box neighbour. Corrections:
`STEP6_ROUND3_CORRECTIONS.md`.

**Decisions (the report's 1–7).**

1. **A.6 (H)** — the Planner's sentence adopted: at one scenario the commitment charge and the voltage regularisation
   are not constructed; the interface settlement stays in every local objective at full weight and is excluded from
   the reported cost as a transfer (A.1 of round 3).
2. **The three M rows** — fixed: the expectation over market × operation pairs (A.2a); the z-update sentence
   (unit weights; κ^E scales the agent's local problem only, A.2b); the end-of-cycle order AA step → balancing →
   memory clear → exit (A.2c). **The 17 L rows** — all taken (A.3 d–s): Gauss–Seidel on the interface channels only;
   the sum over blocks equals Q after the stated exclusions; κ^E's meaning; ℒ defined and κ^E shown in §2.3.5's
   objective; per-agent copies on failure; the continuation case in the ρ, AA and tail holds; the √3 in s_E and which
   duals enter ‖λ‖; the cap in "stopped by the rule"; the AA iterate on the interface channels; the tail as a declared
   option restoring each operator's own tolerance; the retry triggers and two tiers; the initialisation order
   (conversion first, storage duals at zero, one dual-ascent step on the interface); the gap recorded by the harness;
   the TSO's fixed initialisation exchange and the bounded adjustment.
3. **Recovery settings** — per agent, in §3.6: TSO acceptable_tol 1e-4 / acceptable_iter 1; DSOs a cold restart at
   their primary settings; storage agent 1e-9 / 1; tier 2 adds the adaptive barrier strategy; triggers maxIterations,
   infeasible, solver error. Addendum 68's list corrected accordingly (erratum).
4. **Interface ratings** — R^I = 2.0 / 1.0 / 1.5 p.u. (200 / 100 / 150 MVA) at nodes 5 / 7 / 9, into §3.6; the IEEE-33
   branch table's branch 1 corrected per ADN (author; round 3, C.3).
5. **F2 sentence** — the Planner's counts adopted verbatim (round 3, B.d): seventeen of 61 admissible unit neighbours
   evaluated (ten in the final poll, seven earlier), none determinately better; of the thirteen re-evaluated, better
   than twelve (seven determinately, five within resolution) and within resolution of the thirteenth. x = 0: fourteen.
6. **Algorithm 1** — Δ₀ = 4 stated and listed as an input; the loop as run (at most 60 polls; a poll refused only when
   its new points would exceed N^max − N); variant B's completion adds the whole admissible unit box under the cap of
   30; and the sentence that in variant A every rounded poll direction was inadmissible at every poll, so every
   evaluated point came from the unit-neighbour completion (round 3, B.a–d).
7. **§3.5 and §3.6 written** (round 3, C.4–C.5): "Shared Energy Storage Parameters" (η, SoC window and initial state,
   closure slack and price, complementarity tolerance, calendar life, ε^E and c^σ, the lattice and budget; the ageing
   calibration table with the seven arms as run, k(C3) = 11,542, k(C4) = 22,429, φ 0.985, floor 0.70 / 0.50, the
   SoH evaluation point) and "Evaluation and Certification Settings" (solver versions and options per agent, the
   retry policy, σ, S^ref, R^I, κ^E, ρ₀, balancing, Boyd tolerances, AA, production exit and caps, α and the voltage
   weight, τ and P_max, the search's Δ₀, budgets, completion cap and σ_Q, and the two static arrangements of the
   benchmark). §2's "Section 3.4"/"3.5" references become \ref{sec:case_ess_params} / \ref{sec:case_settings}. Also:
   §3.1 states that I(x) is the probability-weighted sum, equal to the cost at the expected trajectory, and names the
   ÷4 correction; §3.3 states that TN generation is priced at the scenario's wholesale price; the two red paragraphs at
   the end of §3.4 go.

**Next order:** `STEP6_ROUND3_CORRECTIONS.md` §D (W174 after the author confirms rounds 2 and 3 are in Overleaf; W175
the CONFIRM points of §3.5–3.6). Predictions: W174 — no H row remains in §2 or Appendix A, and every §3.5–3.6 value
matches its source; W175 — the benchmark paragraph needs one wording change at most (the passive arm's tie-breaker).

# Addendum 71 — rulings on `P5_15_ADDENDUM70_ROUND3_REPORT.md` (W174a/b, W175); §2, §3.1, §3.5–3.6 and Appendix A closed for the revision (2026-10-08)

Accepted. W174a: no wrong number anywhere in the manuscript, the letter, the cover letter or the highlights (380 match,
12 approximate by printed rounding; letter 97 match, 62 quotation numbers verified). W174b: all ten W171b H rows and
W172's A6-1 resolved; every round-2 and round-3 block present. W175: the mid-block SoH point, the no-ageing arm and the
benchmark definitions confirmed.

**Predictions.** W174 (values match their sources): **held**. W174 (no H row left in §2 or Appendix A): **failed** — the
Planner's rating of l. 647 is adopted: after "this threshold", "accepts a poll point … only by a determinate margin"
reads as the §2.2.7 rule and restates the claim W171b T1-4 had found false; the sentence was a round-1 leftover that
round 2 did not update. W175 (one wording change at most): **failed on count** — the tie-breaker and common-Q sentence
was predicted; the consistency pass (each DN re-solved once at the TN's interface voltage, then the TSO again), under
which every reported arm value stands and which moved the passive starts by −44.0/−69.8/−69.8 k€ and the price-taker
by +234 €, was not; the 13.9 % claim is unaffected.

**Decisions (the report's 1–8), all as recommended; corrections in `STEP6_ROUND4_CORRECTIONS.md`.** (1) The two
passages lost in the round-3 paste restored verbatim — the AA-off clause (the expert's step-11 instruction quoted the
clause without its lead-in) and Algorithm 2's `k ← k + 1` (the A.2(c) block did not say the line after it stays).
(2) l. 647 rewritten: the search accepted by its own resolution rule; every proposed cell re-evaluated under the §2.2.7
rule before being reported. (3) l. 449: two runs stopped on a failed unit poll, one for review at the completion cap;
"at every evaluated poll". (4) "One frozen configuration" scoped to the reported certified evaluations in §3.6 and in
Appendix A; the searches ran earlier code states without the tail, their incumbents re-evaluated under the final one.
(5) The benchmark paragraph takes the Worker's two sentences, plus "at the plan without shared storage, at the
single-scenario instance". (6) l. 756 and l. 866 → `sec:case_ess_params` (the round-3 mapping rule was too blunt).
(7) **Chemistry — author's decision:** "lithium iron phosphate" stays only with sources for the 10,000-cycle
statement, the 8,000-cycle datasheet (EVE MB31) and the calendar retention; otherwise "lithium-ion" with the NREL ATB
category, in §3.5, the abstract and the letter's R1.2(iv). The expert's recommendation is the first, since the EVE
datasheet is LFP and the retention and counts are LFP-typical, but the text must not name a chemistry the inputs do not
record. (8) The nineteen L rows: the four "damaged in the paste" ones (NEW-5, NEW-6, NEW-9, NEW-10 — three of them in
the expert's own round-3 text) and the two "elsewhere" ones fixed now; the hold-origin, window-bound and indexing rows
listed for the whole-paper read. The C4 row of the ageing table gets a footnote: entered as 10,000 cycles at 0.80 DoD,
the same product Nδ, which is all the calibration uses. Ten answered CONFIRM comments deleted.

**Status.** With round 4 applied, §2, §3.1, §3.5–3.6 and Appendix A are closed for this revision; what remains for them
is the final-pass list (notation and index points) and the nomenclature. Next writing: the author's §3 data
figures and §3.3–3.4 text; then §4 from the frozen tables and the W165 figures (map round 2), where the expert drafts
§4.7 and the limitations paragraph from `paragraphs_v5` and the author drafts 4.1–4.6 with the export's rounding; then
§1, §5, front matter and the appendices B–E (round 3). The Planner stands by for the number check at each round's
close; no run is queued. Still open on the author: the PDF the reviewers read (figure numbering in R1.5 and "Figure
15"); the datasheet citations.

# Addendum 72 — round 4 verified in Overleaf (`c71ca2e`); one leftover; W176 order (2026-10-08)

**Verification (expert, by diff of the collapsed-whitespace blocks against `main.tex` at Overleaf HEAD `c71ca2e`,
2,156 lines).** Thirteen of the fourteen round-4 edits are in verbatim, including the two restorations (the AA-off
clause at A.4 and Algorithm 2's `k ← k + 1`, now at l. 1655); every old wording is gone; no `[CONFIRM` and no
`[AUTHOR]` comment remains. **Leftover:** item 4 was applied at its first site only — l. 748 (η, SoC, ε^Cl, c^Cl, ε^C)
now points to `sec:case_ess_params`, but l. 858 ("The values of $c^{\sigma}$ and $\varepsilon^{\text{E}}$ are given in
Section~\ref{sec:case_settings}") still points to §3.6; it becomes `\ref{sec:case_ess_params}`. The α reference
(l. 899) and the Appendix A recovery-settings reference (l. 1605) correctly stay on `sec:case_settings`.

**Chemistry (Addendum 71 item 7): the author took the second option.** §3.5 l. 1062 reads "utility-scale lithium-ion
battery (the NREL ATB utility-scale battery category~\cite{nrel_ess_costs})", the abstract (l. 143) says "utility-scale
lithium-ion battery storage", and the letter's R1.2(iv) now reads "the reference technology is utility-scale
lithium-ion battery storage; the cycling calibrations are datasheet readings (Section~3.5)". No chemistry is named
anywhere that the inputs do not record. The ageing table's rows remain "datasheet" statements without a named
datasheet; whether to cite the sheets behind the 10,000- and 8,000-cycle counts is the author's call at the final
pass, and the "Section~4.3" literal in the §3.5 ageing paragraph is on the final-pass list with the others.

**W176 order (zero solves; after the l. 858 fix is in Overleaf, at that HEAD).** (a) Re-check the round-4 lines only
against the records: the fourteen edits of `STEP6_ROUND4_CORRECTIONS.md` and the l. 858 reference, each rated as in
W174b; (b) re-run the number check (W174a's script) at that HEAD; (c) list the literal "Section~4.x" references
remaining in §2–§3 and confirm no `[CONFIRM` comment remains. Report as `P5_15_ADDENDUM72_ROUND4_REPORT.md`; no
production, spec, case or frozen file changes. Then stand by for the number check at each writing round's close.

**Predictions.** W176(a): no M or H row among the round-4 lines; any L row is already on the final-pass list of
`STEP6_ROUND4_CORRECTIONS.md`. W176(b): zero MISMATCH in `main.tex` and the letter, with the same twelve approximates as
W174a (printed roundings) and the submitted-version count reduced only where §3.5–3.6 text replaced submitted text.

# Addendum 73 — §4 frame delivered (`section4_expert_draft.tex`); W177 order (2026-10-08)

**What the file is.** The expert's frame for Section 4, replacing `main.tex` l. 1126–1445 (from `\section{Results}` to
the line before the CONCLUSIONS banner). It carries: the section preamble with the salvage credit (eqs.
`salvage_credit`, `salvage_remaining_life`, from Part D.2, still `[CONFIRM — W177]`); nine subsections in the
revision-map order (4.1 baseline, 4.2 break-even, 4.3 degradation, 4.4 coordination, 4.5 multi-scenario, 4.6
investment year and discount, 4.7 certification statistics and computational performance, 4.8 limitations, 4.9 key
insights); nine tables composed from the frozen T1–T11 rows with every number copied from the W160 `.tex` export at
its printed rounding and the source rows named in a comment under each table; the five W165 figures with captions
adapted from `captions.md`; the expert's text in full for the preamble, 4.7, 4.8 and a proposed 4.9; and, for 4.1–4.6,
an `[AUTHOR — write]` slot per subsection listing the facts, the accepted sentences to use verbatim (paragraphs_v5
sentences 1–4) and the claims not to make. The main/supplementary split is the export README's: T1 (all claims), T2,
T5, T6 (all starts), T9, T10, the full statistics and the scorecard go to Appendix E in round 3; the author decides.
The file compiles (article wrapper; no error, no overfull box).

**Expert's own choices to be aware of.** (1) "The reference unit" for the 0.25 MVA / 1 MWh plan at node 7, not "the
smallest unit": the lattice admits 0.25 MVA / 0.5 MWh and that plan is evaluated (T1 C row, 49.7 k€); the letter's
"smallest unit" is listed for its final pass. (2) The S column of the value-ladder table is E over the duration
(the cell names carry duration and E); marked for W177. (3) The energy ladder at m = 2 is labelled node 7, 4 h from
the candidate keys shared with the B cells; marked for W177. (4) The F2 plan is described as 2030, node 5
0.25 MVA / 0.5 MWh and node 7 1 MVA / 3 MWh, read from the L-row names; marked for W177. (5) The benchmark table prints
the best start's own curtailment (807.6 / 0.6 price-taker; 781.7 / 201.2 passive). (6) The 3 × 3 "16 cycles" of the
post-hoc continuation is from the brief's record, marked for W177. (7) "41 new recourse evaluations (14, 20, 7)" is
W173's realized use, summed; marked. (8) "0 sign changes in 60 claims" (salvage) is the map's W153 line; marked.

**W177 order (zero solves), in two parts.** (a) Now, on `section4_expert_draft.tex` at the repository root: check
every `[CONFIRM — W177]` point against code or records (the salvage equation (a)–(d) against
`_get_terminal_salvage_value_*`; the S/E/duration of the B and C cells; the I/J candidate keys; the F2 plan; the 16
cycles; the 41; the W153 sign-change line; the 2025 cohort's zero salvage) and check every number of the nine composed
tables and the five captions against the frozen JSON; report as `P5_15_ADDENDUM73_W177A_REPORT.md`. (b) When the
author's 4.1–4.6 text is in Overleaf: the number check and an audit of §4 as pasted (every statement rated as in
W174b), at that HEAD. **Predictions.** W177(a): zero MISMATCH in the composed tables and captions; the salvage
equation needs one correction among its four points; the four other structural CONFIRMs (S/E, I/J keys, F2 plan,
16 cycles) hold. W177(b): scored when ordered.

**Author's steps.** Overleaf: upload the five PDFs from `export/figures/`; delete l. 1126–1445; paste the file at
l. 1126; replace the literal references — l. 415 "Section~4" → `Section~\ref{sec:results}`, l. 445 "Section~4.7" →
`Section~\ref{sec:case_settings}` (σ_Q is stated there), l. 631 "Section~4.5" →
`Subsection~\ref{subsec:res_multiscenario}`, l. 1066 "Section~4.3" → `Subsection~\ref{subsec:res_degradation}`,
l. 1116 "(Section~4.4)" → `(Subsection~\ref{subsec:res_coordination})`; compile (expect only the `[AUTHOR — write]`
slots empty); then write 4.1–4.6 into the slots and edit 4.9. No old §4 label is referenced outside §4 except
`sec:results`, which is kept.

# Addendum 74 — rulings on W176a (`P5_15_W176A_ROUND4_LINES.md`) and W176b (number check at `8b76423`) (2026-10-08)

Accepted. W176a: every round-4 edit present (14 of 14, plus l. 858), every old wording gone, no H or M row; the
round-4 lines resolve Addendum 71's H, W174b's NEW-1 and NEW-3 and the two "elsewhere" rows. W176b: zero MISMATCH in
`main.tex` and the letter (388 match, 12 approximate), the twelve approximates unchanged, the 559 submitted-version
figures untouched (all in §4–§5 and Appendix E, to be rewritten); 9 new declarations, all matched except one named
setting.

**Predictions.** W176(a) "no M or H row": **held**. W176(a) "any L row already on the final-pass list": **failed as
scored** — two new L rows, both in the expert's own round-4 wording: R4-1 (l. 445, "each is described by its recorded
certificate": s51 stopped for review and has no certificate) and R4-2 (l. 639, "the coarser resolution rule": σ_Q is
coarser than the certified-pair threshold but finer than the uncertified-form bars of every F2 comparison). Both are
corrected in `STEP6_ROUND5_CORRECTIONS.md`, not deferred. W176(b): zero MISMATCH **held**, same twelve approximates
**held**, "submitted-version count reduced only where §3.5–3.6 replaced submitted text" **held vacuously** (no
reduction occurred; §3.5–3.6 held no submitted-version figure).

**Decisions.** (1) R4-1 and R4-2 rewritten (round 5, items 1–2). (2) "ATB" dropped from §3.5: the bib entry is NREL's
"Utility-Scale Battery Storage, 2024" and no input names the ATB; "the NREL utility-scale battery category" is what
the record supports (item 3). (3) The consistency-pass sentence states what the runs did: every DN block was re-solved
once at the TN interface voltage (36 of 36 in each of the six runs; the code's trigger is global), so "where a DN
limit is violated" goes (item 4; W176a unexpected finding 2). (4) Literal "Section 4" references: eight, mapped in
round 5 onto the §4 labels of Addendum 73; this supersedes the five of Addendum 73. (5) The note under l. 639 ("every
cell it proposed was re-evaluated") is read as the incumbents, as l. 445 says; no change. (6) The letter's R2.8
in-text placeholder and the `\rchanges{}` bracket placeholders are the author's, at the letter's final pass, with the
§4 labels now available. (7) The IPOPT acceptable-exit return code and the ATB page remain unverified; neither is
claimed by the text.

**Order.** Round 5 is applied with the §4 paste; W177(a) runs now on `section4_expert_draft.tex` (Addendum 73);
W177(b) at the HEAD where §4 and round 5 are in. No new prediction beyond Addendum 73's.

# Addendum 75 — §4 frame verified in Overleaf (`e6e58e8`); figure numbering settled from the submitted source (2026-10-08)

**§4 paste.** `main.tex` at Overleaf `e6e58e8` (2,310 lines) carries `section4_expert_draft.tex` verbatim at
l. 1127–1602 (comments stripped, the blocks are identical; the seven `[AUTHOR — write]` slots and nine
`[CONFIRM — W177]` comments intact; the old §4 gone; the seams to §3.6 and §5 clean); the five figure PDFs are in the
clone; the l. 635 reference now points to `subsec:res_certification`. The 26 remaining `tab:cs3_`/`fig:cs3_` labels
are §3's and Appendix E's own (round 3).

**Figure numbering (closes W171a G3/G4 as far as the source allows).** `manuscript_submitted/main.tex` has 18 figure
environments: 1 framework, 2 test-system diagram, 3 convergence, 4 voltage, 5 operational costs, 6 RES generation
(two sub-panels), 7 curtailment, then B.8 (market prices), C.9 (TN RES) and D.10–D.18 (ADN profiles; elsarticle's
`\appendix` prefixes the section letter and does not reset the counter). The reviewers' comments (docx): Reviewer 1
"(5) Figure 2 could be improved in size for clarity"; Reviewer 3 "The explanation of Figure 15 in Section 4.4 is not a
direct conclusion. Please explain the differences between each subgraph…". No figure 15 exists in §4.4 of the source
(its figures are 4–7), so the reviewer's number comes from the PDF they were served, which the author can confirm in
the submission system; the section is unambiguous. **Rulings for the letter (author to confirm):** R1.5 answers the
test-system diagram, Figure 2 in the source, which the revision has already redrawn as two panels (TN and DN) at a
larger scale — the current reply, about the framework figure, answers the wrong figure unless the served PDF numbered
the graphical abstract as Figure 1; proposed: "The test-system diagram (Figure~2) was redrawn as two panels, the
transmission network and the distribution-network template, each at a larger scale; the framework figure
(Figure~1) was also redrawn, with fewer elements, to reflect the revised architecture." R3.2's "Figure~15" sentence
avoids the number: "On the figure of Section~4.4 the reviewer refers to as Figure~15: the results figures of that
section belonged to the superseded results and were removed; each results figure of the revision is discussed by what
differs between its panels rather than by restating its values." Letter line numbers at its final pass.

**Next.** The author writes 4.1–4.6 into the slots and edits 4.9; W177(a) on the frame is pending; W177(b) when §4 is
complete. The expert composes Appendix E (T1, T2, T5, T6 all starts, T9, T10, the full certification statistics and
the scorecard) from the export for round 3 meanwhile.

# Addendum 76 — W177(b) order: the revised manuscript at Overleaf `6ff8f3c`; Appendix E inputs (2026-10-14)

**What changed since Addendum 75.** The author's whole-document read (2026-10-09 to 10-13) moved and rewrote text in
§2, §3.6, §4, §5 and Appendix A; every change, with the expert's corrections and the Overleaf commits, is listed in
`manuscript_review/W177B_CHANGE_LOG.md` (committed with this addendum). In short: §2.2.7 keeps the recourse value,
a concise certification paragraph, resolution and determinacy, and the full rule v6 moved to new Appendix A subsection
"Certification Rule" (`app:admm_updated_certification`); §3.6 is "Planning and Certification Settings", and the solver
and ADMM settings moved to Appendix A "Solver and Coordination Settings" (`app:admm_updated_settings`); §2.3.5 (agent
objective) moved to Appendix A.2 as f_E; the author wrote §4.1–4.6 (the `[AUTHOR — write]` slots are gone; their fact
lists are in Overleaf git at `cfee0ba`); conclusions, highlights and keywords were rewritten; Fig. `fig:res_ageing` was
redrawn by the expert from Table `tab:res_ageing` (`fig3_ageing_arms_v2.pdf`; the W165 PDF carried code labels).

**W177(b) order (zero solves).** Pull the Overleaf clone; work at `6ff8f3c` or later and state the HEAD.
- **Part 0 — W177(a).** If `P5_15_ADDENDUM73_W177A_REPORT.md` exists, attach it and say which of its findings the
  pasted text still carries. If it does not, run its checks now: the `[CONFIRM — W177]` comments are in `main.tex`
  (salvage equation (a)–(d); F2 plan; the 16 cycles; the 41 evaluations; W153 line; G rows).
- **Part 1 — numbers.** Every number in the abstract, the highlights, §4.1–4.7 prose, table notes and captions, and
  the Conclusions against the frozen JSON (`590088fe`) and the W160 export; MATCH / MISMATCH / DERIVED (with the
  arithmetic) per number. Derived numbers to check explicitly: "24–33 %" (§4.2), "7–9 %" (§4.5 and conclusions),
  "90.9 M€", "654 M€" if present, the O3 differences 9.3 / 42.1 / 28.5 k€ (§4.3).
- **Part 2 — statements.** Every claim of §4.1–4.6 and the Conclusions rated as in W174b. Name in the report:
  (a) §4.1 "all fourteen admissible unit neighbours … none reduced the total discounted cost beyond the search
  resolution"; (b) §4.2 the F2 plan: 17 of 61 neighbours evaluated, 13 re-evaluated, "determinately better than seven
  and within resolution of the other six"; (c) §4.6 salvage: the number of comparisons (60), sign changes (0) and
  verdict changes — the slot said one (the F2 neighbour), the letter (R3.1) says two (F2 neighbour and 2035 − 2030):
  give the count and the list, and whether the 2035 − 2030 pair is among the 60; (d) Conclusions "every other
  evaluated plan, from 0.5 to 5 MWh and at one or two nodes, also loses determinately"; (e) §4.5 "the planning search
  again returned x = 0" at the multi-scenario instance.
- **Part 3 — structure.** (a) Appendix A "Certification Rule": every clause of rule v6 present (holds and k₀ reset,
  C1–C5, the noise/counting sentence, M1–M3, uncertified form), and the §2.2.7 summary consistent with it;
  (b) Appendix A "Solver and Coordination Settings" + §3.6: every setting of the frozen configuration present once,
  values against the campaign specs; (c) Appendix A.2: f_E (`eq:esso_objective`: slack penalty + ε^E throughput) and
  the agent's constraint list against the code; (d) nomenclature: symbols defined but unused, used but undefined;
  (e) `fig3_ageing_arms_v2.pdf`: values, thresholds and binding years against T8, and the row mapping (C2 = cycling
  only, C2_calfade = baseline, C3_unit = retention 50 %, C3_midblock = retention 50 % mid-block, C4 = datasheet
  8,000 cycles); (f) undefined references and duplicate labels (the expert's static check found none at `6ff8f3c`).
- **Part 4 — Appendix E inputs.** Appendix E still holds the submitted results table, while §2.2.7 promises the
  bands, gaps and slacks of every evaluation used in a reported comparison. Commit and push the W160 export (the
  `.tex`/CSV of T1, T2, T5, T6 with all starts, T9, T10, the full certification statistics and the scorecard, and the
  export README) under `manuscript_review/export_W160/`, unchanged, with the frozen-JSON hash in the README. The expert
  composes Appendix E from it (Addendum 73); §4.7's clause breakdown and reproducibility paragraph move there with it.

**Report** as `P5_15_W177B_REPORT.md`; commit and push it with the export. **Predictions.** Part 1: zero MISMATCH in
numbers carried from the tables; the derived numbers hold. Part 2: (c) is two verdict changes; nothing contradicted.
Part 3: no undefined reference; at least one nomenclature symbol defined but no longer used after the moves.

**Amendment to Addendum 76 (2026-10-14, same day).** Two items join W177(b). (1) §3 opening: it still described the
submitted instance (five years 2025–2037, 5 × 5 scenarios); the expert's fix states both instances as run
(`manuscript_review/section3_intro_conclusions_fixes.tex`, M1, with a `[CONFIRM — W177]` on which scenario the
single-scenario instance carries and on the 3 × 3 prefix draw [1,2,3] × [1,2,3]) — check it at the HEAD you audit.
(2) The letter: `manuscript_review/response_to_reviewers_v2.tex` (final-pass draft: section, table, figure and
equation numbers from a compile of `6ff8f3c`; "reference unit" throughout; R3.6's "optimal plan" claim aligned with
§4.8; R1.5 and R3.2 per Addendum 75) — run Part 1 and Part 2 on it as on the manuscript, and check every numbered
pointer (Section/Table/Figure/Eq./Appendix) against the compiled manuscript at the HEAD you audit.

# Addendum 77 — rulings on W177(b) (`4bef2b2c`, Overleaf `e10c104`); the manuscript at `9a697cc`; W178 order (2026-10-14)

**Received.** Parts 0–4 at `e10c104` and the export (`829e92e5`). The number check is clean where it matters: 0 MISMATCH
in 388 table tokens, 16/16 derived numbers; no undefined reference at `e10c104`. The failures are wording, and the
report is right on each. **Expert's error, recorded:** "the search returned x = 0" at the 3 × 3 instance came from the
expert's own slot facts (Addendum 73, after Addendum 64's "x = 0 optimal" for the 3 × 3); no search was run there.

**Rulings (the report's decisions 1–6).** The manuscript and letter edits are in
`manuscript_review/w177b_fixes.tex` (A1–A20, B1–B4) and the letter `manuscript_review/response_to_reviewers_v4.tex` (v3 with the equation numbers of Overleaf `820ee1b` + the four follow-up fixes below).
1. **Salvage count — one verdict among the 60; accepted as recommended.** §4.6 and R3.1 now say: no sign and one verdict
   among the 60 (the F2 neighbour, 0.95× → 1.22×); the 2035 − 2030 pair is outside the 60 and changes verdict (and
   sign) net of salvage (A13; letter L1).
2. **3 × 3 "planning search" — accepted.** §4.5: "The planning search was not repeated at this instance; the reference
   unit was evaluated against the plan without storage" (A11); R1.2 says the reference unit again loses (L2).
3. **Letter text absent from the paper — the paper is completed, not the letter trimmed, except one.** Abstract names
   lithium-ion (A1); the introduction's uncertainty-scope sentence is restored (A3; the author had dropped it again);
   §2.2.7 opens with "no convergence guarantee" (A4); §4.8 states that the 5 × 5 set could not be evaluated within the
   memory available (A15). R3.7/3.8 is trimmed instead: the paper has no annual figures (L4). The two Appendix E
   pointers stand; the expert composes Appendix E from the export now.
4. **R1.4 "exact" — accepted.** Letter and §4.6: the 0/5/8 % rows re-weight the 2 % evaluations; exact at x = 0,
   first-order at the unit plan (A12, L3).
5. **Table numbers — accepted:** `\addtocounter{table}{-1}` after each nomenclature `longtable` (A2). Verified by a
   compile in the expert's container (demo graphics): Tables 1–15 restored. **Equations did move** after `e10c104`
   (the author added eight numbered equations in §2.2–2.3); the letter v4 carries the numbers of the author's version
   (`820ee1b`, cohort-rating equation split in two) with the follow-up fixes: Q(x) (13); cohort ratings (14)–(15);
   SOC block (20)–(28), SOC limits (22); agent and degradation (29)–(35); salvage (38)–(39).
6. **Carried rows.** Salvage equation: the coded form is printed (energy cost only; health factor normalised above
   SoH^Min), with a where-clause saying the converter is not credited (A8). A-1: three cases named in A.3, A.5 and §3.6
   (fresh; the three reference continuations; re-runs on an earlier record, replayed to k₀ and capped at N_old + 100)
   (A6, A17, A19). A-3: the gap is priced at the expected wholesale price, block-weighted, on active power (A4, A19).
   Conclusions: the three M rows rewritten (A16). §4 preamble: status wording (A7). J 4 MWh row at its uncertified bar
   per Addendum 64 ruling 4 (A9). 0.50-floor row: value 258.5 k€ (= 318.0 − 59.5); AE and EFC/day from W155 (W178 c).
   Nomenclature: π_t row → π̄_t; s dropped from the t_sum row; C^Op_s added (A2). Stale pointers to A.5/A.8 (A11, A14,
   A18, A20). Cache-seeding clause rewritten from the report's (2b) reading (A6). Hard-coded "Subsection 3.1" (A5).

**New since `e10c104` (expert's static and compiled check of `9a697cc`).** The author rewrote §2.1–2.3. Found:
§2.3.2 printed `eq:available_power`/`eq:available_capacity` twice (duplicate labels) and dropped `eq:cohort_rated`,
leaving S^{Rated,Unit}/E^{Rated,Unit} undefined and Appendix A.2's `\eqref{eq:cohort_rated}` undefined (B1, a clean
§2.3.2); §2.3.4 cited `eq:cohort_available` for the energy product (B2); §2.1 omitted the completion-cap stop (B3);
the leader–follower sentence is commented again (B4, optional). With all fixes applied the expert's compile has 0
errors, 0 undefined references, 0 duplicate labels.

**W178 order (zero solves), when the author has applied the fixes; state the Overleaf HEAD.**
- (a) **§2.1–§2.3 as rewritten after `e10c104`, against the code:** the active-cohort set and rated sums (eqs. 7–9),
  the cohort and available-capacity equations (14–19), the network storage block (20–28: net power, SoC recursion,
  limits, initial state, closure and its slack bound, the P^Ch + P^Dch ≤ S^Av limit, the capability circle,
  complementarity), and the agent's constraint list in A.2; every statement rated as in W174b.
- (b) **A1–A20 and B1–B4 applied as worded;** Part 1 and Part 2 re-scored at the new HEAD on the rows they address.
- (c) **The 0.50-floor row:** AE and EFC/day (PV-weighted, as the other rows) from the W155 record.
- (d) **The letter v4** (`response_to_reviewers_v4.tex`, once in Overleaf): Parts 1–2 on its changed replies (R1.2,
  R1.4, R2.2, R2.5, R2.10–12, R3.1, R3.7–8) and its numbered pointers against the manuscript (static is fine; the
  expert's compile numbers are in ruling 5).
**Follow-up at Overleaf `820ee1b` (author's application of the fixes, 2026-10-14 evening).** Applied with
stylistic line breaks; B4 declined (author's choice). Four slips fixed by the expert on a copy
(`manuscript_review/main_820ee1b_fixed.tex`, 4 hunks): the §2.3.2 heading had been dropped (sections 2.3.2–2.3.5
renumbered, against the letter); the cohort-rating equation was split in two with the label on the second (the A.2
range would have skipped S); the nomenclature lost the daily C^Op_{y,d,m,o} row when C^Op_s was added; §2.3.4's
"nonlinear rows" sentence was cut mid-way. Audit W178 at the HEAD that carries these.
**Report** `P5_15_W178_REPORT.md`. **Predictions:** (a) the network storage block and the cohort equations hold
against code; at most one M row, in the wording of the closure slack or of the power limit; (b) all applied; (d)
0 pointer MISMATCH.

# Addendum 78 — rulings on W178 (`5a7d663e`, Overleaf `9fd19e7`); follow-up applied at `e926fb7`; Appendix E composed; W179 order (2026-10-15)

**Received.** `P5_15_W178_REPORT.md` with W178A (§2), W178B (fixes) and W178C (letter). Every prediction held:
(a) the network storage block and the cohort equations match the code, 0 M, 2 L; (b) A1–A20 and B1–B3 applied, B4
declined by the author; (c) 0.50-floor row AE 0.778, EFC/day 1.08 from W155; (d) 151 of 156 letter pointers MATCH,
5 point to the submitted version on purpose, 0 MISMATCH.

**Rulings.**
1. **Wording rows → expert follow-up G1–G12, applied.** `manuscript_review/main_9fd19e7_fixed.tex` (+ `.diff`, 14
   hunks) and letter v5. G1–G2 "Appendix Appendix" / "Section Appendix" prefixes; G3 0.50-floor AE/EFC; G4 day weights
   92/91/91/91 in §3 (W178C Blocked 1(b)); G5 conclusions stopping status incl. the production exit; G6 holds "from the
   cycle after" the stopping cycle (A.3, A.5); G7 4.2-1; G8 4.3-1; G9 S3-12; G10 §4.8 local-solution limitation (LL-8);
   G11 two resolved CONFIRM comments removed; G12 nomenclature Ŝ/Ê. Letter v5: R1.4 fuel via the wholesale price
   (§3.3) + A.1; R2.10 converter limits (26)–(28); R2.3 §2.2.6; LM-3 "inference"; LL-8 §4.8. The author applied both
   at Overleaf `e926fb7`; the expert verified `main.tex` byte-identical to `main_9fd19e7_fixed.tex` and
   `response_to_reviewers_draft.tex` byte-identical to v5.
2. **L4 (R3.7/3.8).** Option 1 taken: day counts added to §3 (G4), Appendix E recomposed (below). With the submitted
   per-block table gone, "no annual figures" holds for every cost; EFC/day (Table 11, §4.3) is a utilisation rate, not
   an annual figure — L, accepted.
3. **LM-9 and 4.6-2.** Closed by the Appendix E below, subject to W179.

**Appendix E composed (expert, 2026-10-15).** `manuscript_review/appendixE_evaluation_record.tex`, generated by
`manuscript_review/gen_appE.py` from `export_W160/*.csv` (no table entry typed by hand); full manuscript
`main_e926fb7_appE.tex` (+ `main_e926fb7_appE.diff`, 5 hunks); letter `response_to_reviewers_v6.tex`.
- **Appendix E "Evaluation Record"** (label `app:results` kept): E.1 comparisons, Table E.20 — 58 rows = T1's 55
  distinct comparisons (60 claims less the four G `:net` duplicates and the CHECK headline row) + the 3 distinct T10
  claims (the g070 REPORT_BESIDE row equals row 58, stated in the note); E.2 evaluations, Table E.21 — the 61 rows of
  T2, numbered 1–61 and referenced by number from E.20, E.22, E.24; E.3 the neighbours of x = 0 re-evaluated under the
  rule, Table E.22 (T5 + the two C rows); E.4 the static arrangements by start, Table E.23 (T6); E.5 the gap-refused
  evaluations, Table E.24 (T9); E.6 certification statistics and the reproducibility paragraph (moved from §4.7).
- **Main text:** §4.6 salvage sentence — "the 55 comparisons of Table E.20 reported both gross and net", row 34, and
  the year pair "changes both the sign and the verdict" (+42.5 → −2.3); the G-rows sentence added (85.1–223.6 k€, rows
  17–20, all determinate gross and net) and its CONFIRM removed; §4.7 keeps the counts by cause, the τ statement, the
  0.9τ and 15–21 k€ movements, Fig. 9 and the search/wall-time paragraph, and moves the branch split, the medians and
  the reproducibility paragraph to E.6; §4.8 keeps the −8.8 to −9.4 k€ fingerprint and moves the node split and the
  residual ratio to E.5.
- **Letter v6:** R2.2 pointer (cycles of every evaluation in Appendix E); R3.1 "Among the 55 comparisons … reported
  both gross and net"; reproducibility note → Appendices E.6 and A.8; new "Further changes" bullet on Appendix E.
- **Expert checks:** compile 0 errors, 0 undefined references; every label outside Appendix E has the same number as at
  `e926fb7` (aux compared). Re-derived from the CSVs (`verify.py`): 55 T1 comparisons, 0 sign changes, 1 verdict change
  (row 34, 0.95× → 1.22×); the 42 = T2 cells less the three B cells and the four references (IDs listed in E.6);
  32 = 24 osc. + 8 mon.; 10 = 5 gap + 2 lapse + 3 growth; median k* 174, k* − k₀ median 65 (21–89), range/τ median
  0.86; the ten ≥ 0.95τ = IDs 2, 3, 5, 10, 11, 12, 14, 17, 20, 23; 46 = 49 certified − superseded − 2 twins; 21
  non-clean cycles in 10 evaluations; T9 fingerprint −8.8 to −9.4, 55/14/31 %, 0.68–0.70.

**Interpretations made by the expert (each a `[CONFIRM — W179]` or listed here):**
(i) the 60 → 55 + 3 mapping above; (ii) the plan behind every cell in Table E.21 (decoded from the claim ids and T2
candidate keys; in particular `yl_y2030`/`yl_y2035` = the reference unit, the six `pb_*` = 0.25/0.5 at one node, F2
returned plan `ref:5ca4f86c` = 2030, n5 0.25/0.5 + n7 1/3.5, challenger `ref:e28de4ac` = n5 0.25/1 + n7 1/3);
(iii) the "Rule" column: "run" for v6, ext v3, A64 v1 and the uncertified cells with no spec; "record" for the v1
references, v2 (W118, incl. the two uncertified F2 references), v4, v5 and `d_c52e1670`; (iv) E.3: the other six box
neighbours of x = 0 (two-node 2025 {5,7}, {7,9}; 2030 {5,9}, {7,9}; three-node 2025, 2030) were evaluated by the search
only; (v) T6: the unit of "NRF violations (n)", the reading of the consistency pass, and "warm" = started from the
certified coordinated evaluation; (vi) E.6: "119 exits at IPOPT's acceptable level … all in distribution blocks" for
W153's "Acceptable-clean events after N: 119, all DSO"; (vii) the continued cells and their movements (14: 0.74τ;
3: 0.78–0.88τ); (viii) the "three single-scenario reference continuations" (which cells) — their replay cycles are
blank in T2 and print "—"; (ix) k₀ = 123 for x = 0 (Fig. 9) is not in T2 — E.21 prints "—" with a note citing Fig. 9.

**W179 order (zero solves), at the Overleaf HEAD that carries `main_e926fb7_appE.tex` and letter v6; state the HEAD.**
- (a) **Appendix E against the frozen JSON (`590088fe`):** every entry of Tables E.20–E.24 MATCH/MISMATCH; every
  number and count in the E prose (intro, E.1, E.3, E.5, E.6) rated; items (i)–(ix) confirmed or corrected.
- (b) **Main-text and letter changes of this addendum:** §4.6 (salvage sentence, G-rows sentence), §4.7, §4.8; letter
  R2.2, R3.1 and the two "Further changes" bullets; pointer check of the letter v6 at the HEAD.
- (c) **G1–G12 and letter v5 as applied at `e926fb7`:** re-score the W178 rows they address.
- (d) **Closure slack (§2.3.3, "Both daily closure slacks were zero, within solver tolerance, at every certified
  evaluation reported in this study"):** the W169 inventory covered 45 of 51 certified points; complete it for the
  six missing and re-rate the claim.
- (e) **Two statements the new appendix makes checkable:** §4.8 Scope "every evaluated unit neighbour of the plan
  without storage determinately worse" (8 under the rule, per (iv)); §2.2.7 Determinacy "the bands, gaps and slacks
  of every evaluation used in a reported comparison are tabulated in Appendix E" (the multi-scenario evaluations are
  in Table 13 and the static arrangements in E.23 — say whether the wording needs a qualifier).
**Report** `P5_15_W179_REPORT.md`. **Predictions:** (a) 0 MISMATCH in table entries; at most two of (i)–(ix)
corrected, most likely (v) and (vi); (b) 0 pointer MISMATCH; (c) all W178 rows addressed close; (d) the six missing
points also have zero closure slack within tolerance, so the claim holds at 51 of 51; (e) the §4.8 Scope wording needs
"under the certification rule" or "at the search resolution" for the six search-only neighbours.

# Addendum 79 — rulings on W179 (`43c72e06`, Overleaf `48099ab`); round 6 corrections; W180 order (2026-10-15)

**Received.** `P5_15_W179_REPORT.md` with W179a (Appendix E vs `590088fe`), W179b (text and letter; G1–G12 and
letter v5) and W179c (closure slack; two scope statements). Zero solves, guards armed and verified at 0, pickle
blocked; mutation-tested checks (a), negative control (b). Overleaf `48099ab` verified: `main.tex` =
`main_e926fb7_appE.tex`, letter = v6 (md5 by the expert, sha256 by the Planner).

**Predictions of Addendum 78 against outcomes.** (a) 0 MISMATCH in table entries — **held** (1,623 entries: 1,443
frozen MATCH, 180 non-frozen agree with their named sources); "at most two of (i)–(ix) corrected, likely (v), (vi)" —
**failed**: (v), (vii), (viii) corrected, (vi) confirmed. (b) 0 pointer MISMATCH — **held** (165: 160 MATCH, 5
SUBMITTED). (c) all W178 rows close — **failed**: 18 of 20, C-1 and R14-2 narrowed but still L. (d) 51 of 51 —
**failed on the count**: the six missing points hold (five x = 0 cells exactly 0 by construction; `d_c52e1670` at
k* = 150 at the relaxed bound, from the committed per-cycle record), but E.21 row 16 (`pb_y2025_n5`, the superseded
W118 v2 certificate, now reported) carries a positive closure sum of +1.76 × 10⁻⁶ MWh, so 50 of 51. (e) §4.8 Scope
qualifier — **held** (8 under the rule, 3.40–8.09×; 6 search-only, 3.47–7.17× the search resolution). Three
failures, all wording or count; no run depends on them (CLAUDE.md stop rule satisfied by this ruling).

**Rulings (the report's decisions 1–6).**
1. **Row 16: kept, relabelled (option a).** Status "superseded^s", Rule "---", its cycle, range and band shown; the
   `^s` legend says it was certified under an earlier version of the rule and is not decided under the rule of A.5
   within its recorded cycles. §4.7 scoped to "every certificate used in a reported comparison", with one sentence
   recording the superseded certificate; §2.3.3 scoped likewise and quoting row 16's figure. The E.22 note (51.3 k€,
   3.79×) stays as a recorded fact. The 46-certificate count is unchanged (49 T2-certified cells − 16 − two twins).
2. **§2.3.3 closure sentence** rewritten in W179c's form (a): at the lower bound to within the solver's bound
   relaxation (10⁻⁶ MWh per slack) at every certificate used in a comparison and at both multi-scenario evaluations;
   row 16's sum 1.8 × 10⁻⁶ MWh, 5 × 10⁻⁶ of E^Av; "without storage the closure row is not built" replaces "within
   solver tolerance" at x = 0. W169 erratum (closure lives in `ess_complementarity_penalties`, carried per cycle by
   `creep_diagnostic_per_cycle.jsonl`) recorded; nothing W169 scored changes.
3. **Appendix E wording**, all taken: E.4 restated in the three phases of `p515_s53_w116_benchmark_nrf.py` (count
   in the re-evaluation with the NRF rows off, in network-hours of 864; consistency = Q(pass) − Q(first phase);
   E.23's Q after the pass, curtailment before it); E.6 "three evaluations continued, 12, 14 and 3, 0.004τ / 0.74τ /
   0.78–0.88τ"; E.6 replays "through the stopping cycles of the residual-based runs they continue, 132, 112 and 87";
   intro "ratios to two decimals, except the residual ratio of Table E.24, to three"; E.3 "its own resolution (σ_Q,
   or the sum of the two bars where larger)" plus a sentence on the six search-only neighbours (3.47–7.17×, earlier
   code state, production exit); E.5 and §4.8 note that the split and residual were not recorded for 40 and 41.
4. **Qualifiers**, as recommended: §4.8 Scope "worse, determinately for the eight re-evaluated under the
   certification rule (Table E.22) and beyond the search resolution for the other six"; §2.2.7 Determinacy "every
   single-scenario evaluation used in a reported comparison … ; the multi-scenario evaluations … Table 13" and
   "every incumbent it proposed"; §4.7 closing sentence likewise (search evaluations by count, 3 × 3 pair in Table
   13). Accepted L residues: discount-rate thresholds (in Table 15, not E.21) and the benchmark's 71.9 k€ band (E.23
   note).
5. **C-1 and R14-2** reworded to what the text holds: conclusions "with the remaining movement of the plan without
   storage estimated from a continued run … the search's own evaluations at the search resolution"; letter R1.4
   "Fuel costs are not represented separately: … priced at the wholesale energy price of the scenario (Section 3.3),
   so fuel-cost uncertainty enters only through the market-price scenarios." Letter R2.2 "every evaluation behind a
   reported comparison", R2.5 "every evaluation repeated from an earlier record (Appendix E.6)", Appendix E bullet
   "every single-scenario recourse evaluation those comparisons use".
6. **Small items**: A.3 l. 1899 prefix dropped; §2.2.6 l. 697 "held fixed in the operational subproblem, where they
   enter as the parameters Ŝ^Inv, Ê^Inv of Subsection 2.3.2" (new label `subsubsec:capacity`); the four
   `[CONFIRM — W179]` comments removed (0 CONFIRM comments remain; the `% [AUTHOR]` citation note stays).

**Round 6 delivered (expert, 2026-10-15).** `STEP6_ROUND6_CORRECTIONS.md` (repo root);
`manuscript_review/main_48099ab_round6.tex` (+ `.diff`, 16 hunks), `response_to_reviewers_v7.tex`
(+ `letter_v6_v7.diff`, 5 hunks), `gen_appE.py` and `verify_appE.py` patched (33 PASS), `appendixE_evaluation_record.tex`
regenerated. Compile: 0 errors, 0 undefined, 0 multiply-defined; every label number identical to `48099ab`, one label
added; the generator reproduced the committed appendix bitwise before the patch and the new block equals its output.

**W180 order (zero solves), at the Overleaf HEAD that carries `main_48099ab_round6.tex` and letter v7; state the HEAD
and verify `main.tex` byte-identical to that file.**
- (a) **Appendix E regenerated:** run `gen_appE.py` and `verify_appE.py` from `export_W160/` at the HEAD; the block
  in `main.tex` must equal the generator's output; re-run the W179a entry check (only row 16's Status and Rule may
  differ from `48099ab`); rate the E.3, E.4, E.5, E.6 and intro sentences of round 6 against the sources named in
  `STEP6_ROUND6_CORRECTIONS.md` §A.10 (W173 item2.x0.rows; `p515_s53_w116_benchmark_nrf.py` phases; W163 N25, N49).
- (b) **Row 16:** confirm that no other E.21 row prints a certificate issued under an earlier rule version without a
  v6 decision on its record (every "record" row has a v6-from-records certificate or is a reference continuation).
- (c) **The ten main-text edits** (§A.1–A.9) and **letter v7** (§B): verbatim at the HEAD; letter pointer check
  (165 pointers; "Appendix E.6" new); re-score C-1, R14-2, C47-1, C48-1, R22-b, FC-2c, the G12 follow-on and the
  A.3 prefix; 0 `[CONFIRM` and 0 `<word>~\ref{app:` at the HEAD.
- (d) **§2.3.3 as reworded:** re-rate against W179c (50 current certificates + 2 T11 points at the bound; row 16's
  figure; x = 0 row not built).
- **Report** `P5_15_W180_REPORT.md`; commit and push; nothing else runs.
**Predictions (expert):** (a) the regenerated block equals `main.tex` bitwise and 0 MISMATCH in 1,623 entries with
exactly two entries changed (row 16 Status, Rule); all five prose sentences MATCH; (b) 0 further such rows; (c) 0
pointer MISMATCH and all eight re-scored rows close, 0 CONFIRM, 0 prefixes; (d) the sentence holds at 52 of 52
points. After W180: `latexdiff` against `manuscript_submitted/`, the cover letter, submission.

# Addendum 80 — rulings on W180 (`a1ab9e99`, Overleaf `e5322f8`); round 7; W181 order (2026-10-16)

**Received.** `P5_15_W180_REPORT.md` with W180a–d; zero solves (guard 0, pickle 0). Overleaf `e5322f8`: `main.tex`
byte-identical to `main_48099ab_round6.tex`, letter to v7; the clone diff equals the expert's two diffs line for line.

**Predictions of Addendum 79 against outcomes.** (a) regenerated block bitwise, 33 PASS, 0 MISMATCH in 1,623 entries,
exactly two cells changed, five prose sentences MATCH — **held**; (b) row 16 the only superseded certificate —
**held**; (c) 0 pointer MISMATCH, 0 CONFIRM, 0 prefixes — **held**; "all eight re-scored rows close" — **failed**
(7 of 8; R22-b still L: the 3 × 3 pair fits neither of R2.2's classes); (d) 52 of 52 — **failed on the count**: the
sentence holds at 50 of 50 (48 current E.21 certificates + the 2 multi-scenario points; W179c's 51 already included
the T11 pair, so the expert double-counted it). Both failures are wording or count; no number or run depends on
them. W180d also tightened W179c's distance-to-bound figure (≤ 3.76 × 10⁻¹⁰ MWh over the 44; not in the manuscript).

**Rulings.**
1. **R2.2:** option (b) — "every single-scenario evaluation behind a reported comparison" (letter v8).
2. **§2.3.3:** "Without storage the closure row is deactivated and its slacks are fixed at zero" (rests on the code
   only; the NL-writer wording is not used).
3. **Ageing citations (author).** The `% [AUTHOR]` note was never in `main.tex` (it lived in the round-3 paste and
   was dropped), and §3.5 cites only `nrel_ess_costs`; the two datasheet cycle-life statements and the 0.985/yr
   calendar retention have **no source in the manuscript**. The comment is restored under Table 6 as a reminder.
   Blocked on the author: add the citations (round 4 item 6 names the EVE MB31 datasheet for the 8,000-cycle row;
   a source for the 10,000-cycle row; an LFP calendar-ageing study) before submission, and tell the expert which, so
   the letter's R3.4/R3.5 replies can name them.
4. **Wording residues:** Table 13 "(cycle 72 for the plan without storage, 69 for the unit)"; E.4 second phase
   "re-evaluation … with its decisions held at the first phase and its voltage, thermal and no-reverse-flow rows
   switched off"; E.21 note "evaluations 40 and 41 are re-decided from the records of earlier runs". Accepted as L:
   0.004τ to three decimals; the six search-only neighbours' bands; the W153 cause rule for rows 39 and 41.
5. **Check scope:** W181 is the short check the Planner proposed, not a re-audit.

**Round 7 delivered (expert, 2026-10-16).** `STEP6_ROUND7_CORRECTIONS.md` (repo root); `manuscript_review/
main_e5322f8_round7.tex` (+ `.diff`, 6 hunks: §2.3.3, the `[AUTHOR]` comment, Table 13, the generator's header
comment, the E.21 note, E.4), `response_to_reviewers_v8.tex` (+ `letter_v7_v8.diff`, 2 hunks), `gen_appE.py`
patched, `appendixE_evaluation_record.tex` regenerated, `verify_appE.py` unchanged (33 PASS). Compile 0 errors, 0
undefined, 0 multiply-defined; every label number identical to `e5322f8`.

**W181 order (zero solves), at the Overleaf HEAD that carries `main_e5322f8_round7.tex` and letter v8; state the
HEAD.** (a) `main.tex` byte-identical to `main_e5322f8_round7.tex`, letter to v8; the clone diff `e5322f8 → HEAD`
equals `main_e5322f8_round7.diff` + `letter_v7_v8.diff`. (b) Appendix E regenerated bitwise from the patched
`gen_appE.py`; `verify_appE.py` 33 PASS; the W180a entry check re-run (0 MISMATCH; 0 cells changed — only the E.4
paragraph and the E.21 note differ). (c) Re-rate the five touched sentences (§2.3.3 against the three code lines;
Table 13 against `campaign_results.json`; E.4 against the W116 harness phase B; the E.21 note against W180b/W163
N49; R2.2) and close R22-b. (d) Report whether the three ageing citations are present under §3.5 at the HEAD
(expected absent unless the author has added them; if present, name the keys). Report `P5_15_W181_REPORT.md`;
commit and push; nothing else runs.
**Predictions (expert):** (a) identical and the diffs equal; (b) bitwise, 33 PASS, 0 MISMATCH, 0 cells changed; (c)
all five MATCH and R22-b closes; (d) absent, or present with three keys. **After W181:** the `latexdiff` against
`manuscript_submitted/` (expert, in the container, demo graphics), the cover letter, and submission once the
citations are in.

# Addendum 81 — W181 held; ageing citations; marked copy (latexdiff); letter v9; W182 order (2026-10-16)

**Received.** `P5_15_W181_REPORT.md` (`6931c781`, Overleaf `cd57efc`): every prediction of Addendum 80 held — identity,
clone diff = expert diffs, Appendix E bitwise with 0 cells changed, 33 PASS, five sentences MATCH, R22-b closed,
ageing citations absent. Rounds 6 and 7 are closed.

**Ageing citations (author, Overleaf `fa7dcb6` → `8f051a2`).** The expert traced the 10,000-cycles-at-0.80-DoD-to-0.80
statement: it is the case file's own input (`cycles_n = 10000`, `reference_dod_d = 0.80`; S35 Z2, P5.13-B) and no
manufacturer document states it; Addendum 30 Q1 only placed it "at the favourable end of the datasheet range". Ruling:
C2 is stated as an assumption bracketed by a datasheet, not as a datasheet reading. §3.5 now reads "cycle-life
statements of the kind given in manufacturers' datasheets"; "The baseline assumes 10,000 cycles at 80 % depth of
discharge to 80 % retention (k = 35,851), at the favourable end of current LFP cell datasheets, and adds a calendar
retention of 0.985 per year …, a modelling assumption in the range reported for LFP cells [43]"; "the datasheet
statement of 8,000 full cycles to 70 % retention [44]"; Table 6 caption "cycle count, depth of discharge and end
retention of the cycle-life statement". Sources verified by the expert: [43] Naumann, Schimpe, Keil, Hesse, Jossen,
J. Energy Storage 17 (2018) 153–169, doi 10.1016/j.est.2018.01.019; [44] EVE Power Co., Ltd., Product Specification
PBRI-MB31-D06-01 Version A (20 Nov 2023), Table 3: "25℃ 8000 cycles, 70% SOH" at 0.5P, 2.5–3.65 V. Keil et al. 2016
not used (record not verified). Expert compile at `8f051a2`: 0 errors, 0 undefined; every label number identical to
round 7; references [1]–[42] unchanged, [43]–[44] new; the EVE entry's title and month were brace-protected after a
first render lowercased "V/Ah/Version A".

**Marked copy.** `manuscript_review/main_latexdiff.tex`: latexdiff 1.4.0 (standalone, run with Perl on the Air; the
script is parked untracked at `manuscript_review/latexdiff-so`), `manuscript_submitted/main.tex` against `main.tex` at
`8f051a2` with its 381 `\textcolor{blue}{…}` revision wrappers stripped first (a marked copy shows additions in blue
already; latexdiff wrapped the wrapper's argument, equations included, in `\DIFadd`, which `ulem` cannot typeset —
the stripped source compiles with every label number identical to `8f051a2`). Options: `--math-markup=whole`,
`--exclude-textcmd="section,subsection,subsubsection,caption"`, PICTUREENV extended to
`table|longtable|tabular|algorithm|figure|landscape` (every results table is new; inline marking inside tables broke
them: 111 errors in the first run). Post-processed by script: one `\emergencystretch=2em` assignment latexdiff had
wrapped; 18 `\ref`/`\eqref` to labels that exist only in the submitted version (deleted figures and tables) print the
label name in brackets instead of "??"; inside display-math environments the 61 `\DIFadd`/`\DIFdel` wrappers are
replaced by colour-only `\DIFaddm`/`\DIFdelm` (no underline or strike-through, which `ulem` cannot apply across
`aligned` blocks). Expert compile with a stand-in for `ulem` (absent in the container): 0 errors, 0 undefined, 98
pages; a static check finds no `ulem`-marked argument containing an environment or paragraph break. The final PDF
with underline and strike-through is compiled on Overleaf, which has `ulem`.

**Letter v9** (`response_to_reviewers_v9.tex`, + `letter_v8_v9.diff`, 4 hunks): opening paragraph "cycle-life
calibration bracketed by a manufacturer datasheet"; R1.2(iv) "one of them read from a manufacturer datasheet [44]";
R3.4 response extended (which calibration reads what; favourable baseline strengthens the negative result); R3.5 cites
[43]. Compiles alone. The letter has no bibliography: [43] and [44] are the manuscript's numbers at `8f051a2`.

**W182 order (zero solves), at the Overleaf HEAD carrying letter v9; state the HEAD.** (a) `main.tex` byte-identical to
`main.tex` at `8f051a2` (the only manuscript change since `cd57efc` is §3.5 / Table 6 caption / the two `\cite` / the
two bib entries); rate the §3.5 paragraph and the Table 6 caption against the frozen calibration constants (k =
35,851; 22,429; φ = 0.985; floor 0.70) and against the two sources as described above (the Planner reads the bib
entries, not the web). (b) Letter v9: pointer check; [43] and [44] against the `.bbl`/`.aux` ordering derived from
the source (elsarticle-num, citation order). (c) `main_latexdiff.tex`: confirm it was generated from `main.tex` at
`8f051a2` (the %DIF header names the files; compare the ADD-side text outside DIF markup with `main.tex`); count the
`\DIFadd`/`\DIFdel` blocks; confirm no table, figure or algorithm environment carries inline DIF markup; list any
`\ref` that would be undefined. (d) `STEP6_*`, brief and change log present at HEAD. Report `P5_15_W182_REPORT.md`;
commit and push. **Predictions:** (a) MATCH on all four constants and both sources; (b) 0 pointer MISMATCH, [43]/[44]
confirmed; (c) generated from `8f051a2`, 0 inline markup in environments, 0 undefined refs; (d) present.
**After W182:** the cover letter (expert draft from `manuscript_submitted/cover_letter.tex`), then submission: revised
`main.tex` (blue), `main_latexdiff.tex` compiled on Overleaf as the marked copy, letter v9, highlights.

# Addendum 82 (v2, replaces v1 of the same date, never acted on) — simulation brief E1–E6; decisions D1–D5; manuscript at `edd7a63`; W182 order (2026-10-25)

**Received.** The author's brief `simulation_study_brief_for_expert_2026-10-25.md`; decisions D1–D5 with six protocol
clarifications; `manuscript_review/STEP7_PLAN_REVIEW_2026-10-25.md`; `manuscript_review/manuscript_critical_review_2026-10-25.md`.
The expert's plan is `STEP7_SIMULATION_PLAN_2026-10-25.md` v2 (repo root). **No run is authorized by this addendum**;
the campaign spec is frozen after W182 and runs are authorized one gate per task by a later addendum.

**Manuscript state.** Overleaf `edd7a63`, eight commits after `8f051a2`: 1,021 changed lines in `main.tex`, four new
equations and `app:salvage_credit`, `tab:res_capacity_summary` added, `tab:res_flex_ladder` and two figures removed,
§3.3/3.4 merged, retitled subsections, headline 12.2 % against the price-taker benchmark. Letter v9 unchanged. The
marked copy compiled by the author is of `8f051a2`. Addendum 81's W182 is withdrawn.

**Rulings.**
1. **E1 (D2, D3, clarification 1):** both arms on the final code head under the frozen configuration, variant B
   throughout, common seed set {x = 0; 7: 0.25/1 2025; 7: 0.5/2 2025; the historical 2× incumbent}, seed costs
   evaluated under each ageing model, N^max = 30 new search evaluations per arm, cross-evaluations certified under
   rule v6, primary setting 2×. The historical 2× incumbent (row 40, gap clause) is a reference row and a seed, not
   the comparator. Baseline price: a fourteen-point unit-neighbourhood screen under the no-ageing model, labelled as
   such (clarification 2); the aged 1× search (s47) is not re-run.
2. **E2 (clarification 3):** stage 1 = the eight-evaluation penalty pilot (ρ₀ × 0.5, × 2); each pair's verdict by its
   own rule-v6 quantities; the baseline threshold descriptive only; range of margins reported separately from the
   per-pair verdict; stage 2 (perturbed coordinated start, four evaluations) conditional on Z6; no centralized model.
3. **E3 (clarifications 4, 5):** one certified C_NRF evaluation with the NRF row on both interface copies, a declared
   option off by default; the reporting rule for the static-vs-ADMM comparison is the plan's §3 (two uncertainties
   side by side; "resolved under the declared procedure" only beyond three times each; "determinate" reserved for
   rule-v6 pairs); the mechanism sentence of §4.4 re-derived from the new decomposition; the four reverse-flow hours
   are context, not a bound.
4. **E5 deferred, claim narrowed (clarification 6); E4 deferred, wording per D4.** Exact sentences in plan §6 and §8.
5. **E6:** zero-solve export from rows 1–2; day rule declared; inputs committed or hash-recorded first (Z5).
6. **Predictions are hypotheses** (plan §1, §8): H1a–d, H2, H3a–b recorded; no outcome extends the campaign.
7. **Counts and estimates reconciled:** E1 ≤ 72 evaluations at 2× (seeds 7–8, search ≤ 60, cross-evaluations 4) +
   14 for the screen; E3 1; E2 8 (+4); concurrency check 5. Serial ≈ 84–89 h; ≈ 50 h with four slots after the check.

**W182 order (zero solves), at Overleaf `edd7a63` or later; state the HEAD and the code head of the working tree.**
- **A. Re-audit of the author's edits since `8f051a2`.** (A1) the four new equations (`eq:acceptance_tolerance`,
  `eq:interface_deviation`, `eq:interface_deviation_nonnegative`, `eq:soh_initial`) and `app:salvage_credit` against
  the code in the W166 form; the presence of an equation is not evidence that the implementation changed — rate what
  the code does. (A2) every changed or moved number in the 49 hunks against the frozen JSON and the named records
  (W163 checker, new declarations version). (A3) a label map old → new for every table/figure/equation at `edd7a63`,
  and the letter v9 pointer check against it (removed `tab:res_flex_ladder`, the two removed figures, shifted
  numbers). (A4) every `\ref` or prose mention that still points at removed tables/figures. (A5) captions and
  sentences whose `\textcolor{blue}` the author dropped — list only. (A6) Appendix E block bitwise equal to
  `gen_appE.py` output; `verify_appE.py` 33 PASS. (A7) sentences in the abstract, highlights (both copies),
  contributions, §4.5, §5 and the letter that claim or imply a multi-scenario search, a value of stochastic selection,
  an installation-timing preference, or that "certification" bounds error or resolves a ratio (audit §3.1, §3.4,
  §7); list with line numbers. (A8) out-of-budget ladder points (4 h / 4–5 MWh, 2025) — every place they appear and
  whether each is labelled as a probe outside the admissible set (audit 3.6); the 13.9 % figure wherever it is still
  called a saving (letter, `highlights.tex`, cover letter); the abstract's technology and cost-uncertainty sentences
  (audit §6).
- **B. Evidence inventory.** (Z1) the no-ageing arm's search-path implementation identical to T8's (k → ∞, φ = 1,
  floor inactive, SoH ≡ 1); whether Q(0) is exactly independent of the ageing model (storage rows deactivated at
  x = 0); the zero-capacity handling of the normalised complementarity, cohort allocation and degradation rows at
  S^Av = 0, with code lines (audit 3.7). (Z2) row 40 (`ref:5ca4f86c`): remaining cap and reachability of a rule-v6
  certificate. (Z3) one table over E.21: code head of the run, final-head yes/no, configuration hash. (Z4) wall time
  per cycle and peak RSS of the 3 × 3 cells and of a five-block single-scenario build, from the records (for the E4
  file only). (Z5) E6 inputs: which quantities are in committed records, which only in `ess_schedule_per_cycle.jsonl`
  (uncommitted) or `certified_models.pkl`; commit or hash-record the exact files for rows 1 and 2. (Z6) whether the
  benchmark harness's perturbed start applies to the coordinated procedure without new code; where ρ₀ enters, so that
  × 0.5 / × 2 is a configuration parameter; where the NRF row would be declared as an option on both interface copies
  and what the harness records for interface P on each side. (Z7) from row 1's records: the four reverse-flow
  interface-hours with magnitude, block weight and price — without solves; state "not recoverable" if the records
  lack them.
- **C. Pre-campaign readings (brief §2).** (C1) cost boundary term by term with code lines and inside/outside Q,
  including deviation settlement and commitment charges (audit 3.8). (C2) horizon-end date and its mapping to terminal
  SoH and the final block (audit 3.3). (C3) budget admissibility of every plan named in the manuscript (expected
  2025 cost from Table 1; the two-node 2030 plan ≈ 992.3 k€ per the audit — confirm from the file). (C4) the
  concurrency path: bitwise gate present on the final code; whether a 4-slot batch is runnable without code change.
- **Report** `P5_15_W182_REPORT.md` with part reports; commit and push; nothing else runs. **Predictions (expert):**
  A1 at most one M row (the interface-deviation sign convention); A3 ≥ 10 letter pointers MISMATCH; A6 bitwise, 33
  PASS; A7 ≥ 6 sentences listed; Z1 Q(0) exactly independent; Z2 not reachable within the cap; Z6 ρ₀ is a
  configuration parameter and the NRF row needs a new declared option (code change, both copies); Z7 recoverable from
  `certified_models.pkl` only.

# Addendum 83 — rulings on W182 (`dfd09ca5`, Overleaf `edd7a63`, code head `a17f6ba7`, production `575bfec4`); round 8 scope; campaign spec v1; W183 order (2026-10-26)

**Received.** `P5_15_W182_REPORT.md` with parts a–g; seven zero-solve parts (guard 0, no build, pickle blocked). The
HEAD is 67 commits after `8f051a2` (Addendum 82 said eight: the count was of the eight Overleaf commits the expert had
seen listed; corrected). Letter v9 committed (`2f67f63a`) and byte-identical to the clone's.

**Predictions against outcomes.** A1 ≤ 1 M — held (0 M; the interface-deviation sign matches the code). A2 — 701
MATCH, 31 DERIVED, 0 MISMATCH. A3 ≥ 10 pointer MISMATCH — held (65 of 173). A6 bitwise — **failed** (Appendix E
hand-edited in six hunks; 0 of 1,637 entries changed); 33 PASS — held. A7 ≥ 6 — held (16, plus 11 borderline). Z1 —
held by code reading only. Z2 — held (row 40 used its whole cap of 281; the gap clause alone refuses). Z6 — held on
both parts. Z7 — **failed favourably** (the four reverse-flow hours are in the committed
`interface_settlement_detail_s31c.json`: node 7, 2025, Autumn h19 −3.157 MW, Spring h22 −2.402, Spring h2 −1.602,
Summer h22 −0.067; weights 455/460/460/455; prices 161.80/134.53/117.73/131.72 €/MWh). Neither failure changes a
number or a run; the A6 fallback is ruling 1.

**Rulings.**

1. **Appendix E: option (a).** The expert ports the author's six hunks into `gen_appE.py` (round 8) and the bitwise
   gate stays. Rule from now on: Appendix E prose is edited through the generator — the author sends the sentence,
   the expert patches the generator, the block is regenerated; a hand edit of the block fails the next W-check.
   Hunk 2's `\ref{tab:res_flex_ladder}` → `Fig.~\ref{fig:res_flex_ladder}` is adopted in the generator.
2. **Objective convention.** Everywhere the convention is stated (Appendix E intro, §2.2.6/C^Op paragraph, Table 13,
   TASKS.md, the brief's convention line, the letter): "Q: gross operational cost, **contracted interface settlement
   excluded**"; one sentence at Table 13 / §4.5: at the multi-scenario instance the deviation settlement and the
   commitment premium (+25.1 M€, about 3.0 % of Q at x = 0) are inside Q, the contracted settlement is not. No number
   changes. The expert's convention line in this brief is corrected accordingly from here on.
3. **E1–E3 code changes (formulation points ruled; Worker tasks in W183).**
   - (i) **No-ageing search launcher:** a new launcher (harness code; production untouched) that accepts
     `model_variant` and `flex_price_multiplier` together, both in the evaluation key, concurrency declared as a
     parameter with a memory preflight. `eol_retention_r` = **0.50**, the T8 declaration, so that the committed
     no-ageing cell of the reference unit (row 54) carries the same key and the key space stays consistent.
   - (ii) **ρ₀ scaling:** a declared, keyed evaluation option `rho0_scale` applied in the configuration hook through
     `p59_rho.apply_rho_to_params` (all three channels and the ESSO); the S39 refusal is relaxed only when the option
     is declared; the scale enters the evaluation key. Harness code only.
   - (iii) **E2 stage 2 deferred.** Option (c), the `price_taker` initialisation, perturbs only the storage channel and
     is not a perturbed coordinated start; it would be mislabelled as one. No production hook is built for this
     revision. The paper states that E2 is a sensitivity to initial penalty levels and does not establish independence
     from the starting point (plan §4, as already drafted).
   - (iv) **NRF option, formulation.** The row is on the **interface exchange carried by the consensus channel**:
     the TSO copy `pc_adn ≥ 0` and the DSO copy `pg_adn ≥ 0`, in the DN-import direction, on both DSO builders, as in
     W116's arm (`pg_adn`), so that the coordinated arm and the static arms constrain the same quantity. At x = 0 this
     equals the reference-generator row. For x > 0 the exchange net of the storage at the reference bus is the
     physical transformer flow, and that is what "no reverse flow at the interface" means; the manuscript states this
     definition once with the benchmark (§3.6/§4.4). Implemented as a declared per-network option (precedent
     `shared_ess_model` in the params files), default off, plus a harness evaluation option and key entry; the harness
     records the binding hours from `p_int_tso_expected_mw` / `p_int_dso_expected_mw` at terminal. A production change
     under the "deactivate and unwire, never delete" and fixture-loading rules; with the option off, the NL files of
     the baseline build must be byte-identical before and after the change (gate in W183).
   - (v) **Concurrency.** Decision deferred to the E1 step and made cheap: the E1 seed evaluations run serially first
     (needed anyway), then the same four are re-run as one 4-slot batch and compared on the gated per-cycle fields
     (the W101/W142 replay comparison); bitwise identity authorizes 4 slots for the E1 search batches, otherwise E1
     runs serially. The stale `p515_s44_gate.py` is left as is (deactivated by its literals; not edited). The 3 × 3
     peak RSS (15.2 GiB) is irrelevant to E1 (SRP1 cells, 2.2 GiB).
   - (vi) **Pinning.** Campaign spec v1 pins the production head **and** the sha256 of every harness and launcher
     file it runs (`p515_s44_campaign_harness.py`, `p515_g_g1_g4_admm_gates.py`, the new launcher, `p56a_oracle.py`,
     the settling modules), plus the case, params and cost files.
   - (vii) **Gate G0 (first E1 gate, zero solves, build permitted):** at x = 0, the ESSO and network NL files with
     `ageing_enabled` on and off are byte-identical; and the combination `model_variant` + `flex_price_multiplier`
     applies both (NL of the unit at 2× under N differs from the unit at 2× under A only in the deactivated/altered
     ageing rows; key contains both). Prediction: identical at x = 0.
4. **E6.** The author is asked to commit the two JSONL files unmodified (`git add -f`, 31.5 MB, precedent the 13.7 MB
   `interface_duals_per_cycle.jsonl`); the expert authorizes a zero-solve read of row 2's `certified_models.pkl`
   (`d0a3d512…`) under an armed guard, pickle-block lifted for that file only, output a committed JSON of the hourly
   SoC and schedule for the declared day. Export after the author's commit.
5. **Round 8 (expert, next deliverable; independent of the campaign results).** Manuscript: the Appendix E generator
   port (ruling 1); the convention wording (ruling 2); the 4 main.tex claim sentences (M-ABS-1/2, M-HL-1, M-CON-1)
   reworded per D4 and audit §3.1; the out-of-budget evaluations marked once — a superscript in Tables 4/6 and
   Fig. 5/E.20–E.21 rows and one sentence in §3.6 ("the 2025 ladder rungs above 1 M€ — 1/4, 1.25/5 at 4 h; 1.5/3,
   2/4, 2.5/5 at 2 h — are valuation probes outside the admissible set") rather than 34 edits (author to confirm
   this method); the `\textcolor{blue}` restored on the 23 lines of W182c A5 (rewritten captions and notes are revised
   text; the reviewers' convention applies); the wording list: P^I direction and its storage exclusion stated at the
   interface definition; y^End = 2040 (1 January) and the terminal-SoH index stated once (§2.2 or `app:salvage_credit`);
   "load curtailment" removed from the Q list (never built); l. 1290 growth rates moved/labelled; l. 2919 τ pointer →
   `eq:acceptance_tolerance`; l. 1573 "first binds in 2035" qualified single-scenario (2037 at 3 × 3); the §3.3
   generator-pricing sentence restored (it was lost in the merge and the letter's R1.4 relies on it). Letter v10:
   pointers renumbered from W182c's label map (65 rows; the l. 97 "Section 3.3" pointer resolved by the restored
   sentence); [43]/[44] → [42]/[43]; l. 76/77/80 made true of the abstract; 13.9 % → "the price-taker arrangement
   costs 13.9 % more; the saving is 12.2 % of its cost" at l. 53 and l. 80; the nine letter sentences of A7 reworded
   ("numerically accepted under the declared rule" where "certified" was wrong; resolution statements without
   "bounds"/"resolved to δ_R"; w.r. never rendered as break-even). `highlights.tex` replaced by the main.tex
   highlights with 12.2 % and without "every evaluation is certified". Cover letter: rewritten (bi-level/Benders text
   out; what changed and why; the narrowed requests). Everything compiled here; numbering checked against W182c's
   map. The results-dependent text (§4.3 E1 table, §4.4 E3 sentences, §4.7 E2 paragraph, letter R2.5/R2.6/R3.6, the
   marked copy) is round 9, after the campaign.
6. **Counts and limits (replace plan §9).** E3 1; E2 stage 1 8 (stage 2 deferred); E1 at 2×: seeds 7 (x = 0 once, by
   G0) + 1 concurrency batch (4, discarded as evidence if bitwise) + search ≤ 30 + ≤ 30 + cross-evaluations 4 = ≤ 75;
   screen 14; E6 0. Serial worst case ≈ 2.5 + 10 + 57 + 10.5 ≈ 80 h; ≈ 48 h if the batch gate passes. Author's D1
   is the authorization for runs over 4 h within these limits; every extension needs a ruling.

**Campaign spec v1 (expert, frozen before W184).** Roots `P515S55/e3_cnrf`, `P515S55/e2_rho`, `P515S55/e1_2x`,
`P515S55/e1_screen_1x`; eval keys with `model_variant`, `flex_price_multiplier`, `rho0_scale`, `nrf_option`; the
capture checklist asserted before each run (rule 11): per cell Q, status, k\*/end, band or gap/slack, stopping ratio,
candidate hash, code and harness hashes, closure/σ/circle/SOC/interface diagnostics, binding NRF hours (E3), E^Av
and terminal SoH per node (E1); predictions H1a–d, H2, H3a–b as in the plan; stopping limits as ruling 6; no archived
root written.

**W183 order (Worker tasks authorized by this addendum; zero solves except the builds named; one gate per task).**
- (W183a) `rho0_scale` option (ruling 3 ii) with a unit test that the key changes and the S39 refusal stays for an
  undeclared ρ. (W183b) The E1 search launcher (3 i) with concurrency as a parameter and the memory preflight; a dry
  run that builds the key set of the seed set under both models without solving. (W183c) The NRF option (3 iv):
  production change on both copies and both DSO builders; fixture-loading check of every preserved pickle; **gate:**
  option off → NL files byte-identical to the pre-change build for x = 0 and the reference unit; option on at x = 0 →
  exactly 12 × 24 × 3 new rows per copy, none active elsewhere. (W183d) Gate G0 (3 vii). (W183e) The E6 export
  (ruling 4) once the JSONL commit is in. (W183f) Spec v1 freeze: hashes, keys, checklist assertion, and the serial
  two-cycle bitwise gate (`p515_s53_srp1_bitwise_gate.py`) re-run on the changed code at both DSO builders. Report
  `P5_15_W183_REPORT.md`; nothing runs beyond the builds and the two-cycle gate. **Predictions:** G0 identical at
  x = 0; the NRF-off NL identity holds; the two-cycle gate passes; the E1 key dry run yields 7 distinct seed keys.
- Then, each by its own addendum line after W183 reports: **W184** E3 (one certified evaluation) → **W185** E2
  stage 1 (eight) → **W186** E1 seeds, the batch gate, arm N, arm A, cross-evaluations → **W187** the 1× screen.
  E3's report rewrites §4.4 from its records before E2 starts; a changed verdict in E2 stops the chain before E1.

# Addendum 84 — round 8 delivered (claims, consistency, submission files, Appendix E generator port); W183h check (2026-10-26)

**Round 8 (expert, 2026-10-26; `STEP6_ROUND8_CORRECTIONS.md`), base Overleaf `edd7a63`.** `manuscript_review/
main_edd7a63_round8.tex` (+ `.diff`, 31 hunks), `response_to_reviewers_v10.tex` (+ `letter_v9_v10.diff`, 12 hunks),
`highlights_v3.tex`, `cover_letter_v2.tex` (+ diffs), `gen_appE.py` with the author's six `edd7a63` hunks ported and
the round-8 edits (convention; $^{b}$ on the seven over-budget evaluations in E.20/E.21; y^End = 2040 and the
terminal-SoH index in E.1; the τ pointer to `eq:acceptance_tolerance`; blue on the lattice caption),
`appendixE_evaluation_record.tex` regenerated, `verify_appE.py` 33 PASS. Main text: 21 edits (abstract, highlight 1,
contributions 1 and 4, $C^{\text{Op}}$ list, §3.3 growth rates and the restored generator-pricing sentence, §3.5
convention and the over-budget sentence, §4.2 fit scope and Fig. 5 caption, Table 6 marks, blue restored on Tables
3/4/6/7 captions and notes, §4.3 floor year, §4.5 cost boundary at 3 × 3, Fig. 8 caption, §4.4 convention, A.1 P^I
direction). Letter v10: 65 pointers renumbered from W182c's map and verified against the round-8 `.aux` (every label
number identical to `edd7a63`; 76 pages), [42]/[43], 12.2 % saving / 13.9 % cost increase, the abstract descriptions,
the nine A7 sentences, R3.6 scope narrowed (the affine-ladder argument withdrawn), v9's R3.4 error corrected (the
datasheet calibration loses less, −45.2 vs −64.4 k€). Compile: 0 errors, 0 undefined; the three submission files
compile alone. Convention line of this brief from here on: Q = `gross_operational_cost`, **contracted interface
settlement excluded**; net = Q − terminal salvage credit.

**Rulings.** (1) The audit's §7 inconsistencies are closed by round 8 except the prediction-record bullet (letter
l. 286), which is the author's call. (2) The A7 borderline sentences M-CON-3, M-45-1/2 and M-5-1 stand (they describe
the formulation or quote the paper's own qualified wording). (3) The results-dependent text waits for W184–W187
(round 9).

**W183h (zero-solve; runs alongside W183a–g, no harness use), at the Overleaf HEAD carrying round 8; state the HEAD.**
(a) `main.tex` byte-identical to `main_edd7a63_round8.tex`; letter to v10; `highlights.tex` to v3; `cover_letter.tex`
to v2. (b) Appendix E block bitwise equal to the patched `gen_appE.py` output; `verify_appE.py` 33 PASS; entry check
(W180a method): 0 MISMATCH, 0 entries changed, the $^{b}$ marks on exactly rows 5, 6, 9, 10, 11, 38, 39 of E.21 and
4, 5, 8, 9, 10, 26, 27, 30, 31 of E.20. (c) Letter v10 pointer check against the HEAD (W182c method): expect 0
MISMATCH, 0 AMBIGUOUS, 5 SUBMITTED. (d) Re-rate the 16 A7 sentences and the A8 places at the HEAD (expect 0 of the 16
remaining; the over-budget points labelled at every table/figure occurrence, by the §3.5 sentence or the $^{b}$
legend). (e) Numbers in the 21 main-text edits against the frozen JSON and the named records (growth rates
1.00/2.50 % from `case9_operational_data.xlsx`; 1.15–1.91 M€ from W182d's table; "about 3 %" from W182g; 2037 from
W182g). (f) 0 `[CONFIRM`, 0 `<word>~\ref{app:`; blue present on the 23 A5 lines. Report `P5_15_W183H_ROUND8.md`
(part of W183's report or separate); commit and push. **Predictions:** (a)–(c) as stated; (d) 0 remaining; (e)
0 MISMATCH; (f) hold.

# Addendum 85 — rulings on W183 (`ca56713c`; production `6f43479e`, harness `c34ebdc5`, E1 launcher `320d503c`); the two-cycle gate read as field-list drift; E1 poll rule B+; spec v1 declarations; W183g order; W184/W185 authorized conditionally (2026-10-26)

**Received.** `P5_15_W183_REPORT.md` with parts 0, a–f and e (reports `P5_15_W183_0_NL_REFERENCE.md`,
`…W183A_OPTIONS.md`, `…W183B_E1_LAUNCHER.md`, `…W183C_NRF.md`, `…W183D_G0.md`, `…W183E_E6_EXPORT.md`,
`…W183F_BITWISE_GATE.md`). Solves 153, all in the two-cycle gate, declared and verified by the armed guard; every other
part built under `SolveProfileGuard(permitted=())` at 0 or built nothing; pickle loads 115 (W183c gate 1) + 3 (W183e).
The chain is stopped at W184 on two findings: the two-cycle gate returned `GATE_PASS=False` as frozen, and the E1
launcher's dry run showed that variant B refuses the first poll for every storage seed. Neither needs a solve to rule.
Overleaf is at `a87e195` with round 8 (the expert verified the four files byte-identical to the round-8 deliverables:
`main.tex` 39f6fd86…, letter v10 2aeb33b2…, highlights v3 4cedd45c…, cover letter v2 88eee8ce…; repo `bd3a9f35`);
W183h has not run (W183 did not read Overleaf) and is re-ordered below unchanged.

**Predictions against outcomes (Addendum 83).** G0 identical at x = 0 — **held** (W183-0 before the change, W183d on
the final head: ESSO NL identical per node; network NL and component digests identical on 48 sequential + 36 parallel
blocks; the unit at 2× differs N vs A only in the six degradation rows). The NRF-off NL identity — **held** (696/696
blocks; option on: 864 rows per copy, all active, nothing else differs). The two-cycle gate passes — **failed as
frozen** on one item of twelve files, read in ruling 1 as field-list drift (not a code effect). The E1 dry run yields
7 distinct seed keys — **held** (7 + the G0 alias x = 0 under N). Unpredicted finding, recorded: at k = 0 the snapped
2n poll of each storage seed has 6–7 distinct feasible points, so the frozen completion (31, 34, 61 box points) is
refused by the cap of 30 — a property of the lattice, computed without a solve (W183b's table, `w183b_e1_dryrun.json`
`first_poll_if_each_seed_were_the_incumbent`, k = 0…9 per seed).

**Rulings.**

1. **Two-cycle bitwise gate: field-list drift; (b′) now, (c) as the standing rule; the W48 reference stays.** The
   44 "genuine" diffs are the 22 `PER_CYCLE_RESPONSE_FIELDS` keys that W64 (`70950e8c`) added to the per-cycle record
   after the W48 reference arm was written: absent there, `null` in the new arm on both rows; the 29 shared trajectory
   keys are bitwise equal; the new arm's `per_cycle_record.jsonl` is byte-identical (`f8506905…`) to the post-W64
   W83 arm's; the other 11 compared files have 0 genuine diffs, Q = 740,922,817.2425401 equals W48's, 153/153 solves,
   57/57 presence. Nothing in the W183 code changes is visible in the ADMM trajectory.
   - (b′) **Zero-solve:** run the gate's own comparator (`W10.classify`, the gate's file set and tolerances) on the
     committed W183f arm (`P515S55/w183f_srp1_bitwise_gate/arm/`) against the committed W83 arm
     (`P515S53/tight_tail_w83/srp1_bitwise_gate/arm/`) as the whole-arm reference, files hash-checked against the
     W83 manifest first. Output to a new write-once directory `P515S55/w183f_srp1_bitwise_gate_vs_w83/`; the gate's
     own outputs and verdict are not edited. **Prediction:** 0 genuine diffs on all 12 files; provenance diffs ≤ 3,
     all under `rule_eleven_checklist` (`w10_arm`, `w183a_nrf_option`, `w84_convergence_depth_tail`).
   - (c) **Standing comparator rule (from this addendum on):** a key absent in the reference record and `null` or
     absent in the new arm is `field_list_drift`, listed by key, never `genuine`, provided every key shared by the
     two records is bitwise equal; a key present in both with different values stays `genuine`. Implemented in the
     wrapper layer (`p515_s55_w183f_bitwise_gate.py` v2 or a successor): the committed gate
     `p515_s53_srp1_bitwise_gate.py` and its frozen verdict `GATE_PASS` are not edited; the wrapper records both
     `GATE_PASS` (as frozen) and `GATE_PASS_RULED` (this rule) with the drift keys. The W48 arm remains the
     whole-arm reference (older than the tail and than W64; its 29 keys are the trajectory).
   - The identical W83 failure (`eb8cf81f`, never ruled) is settled retroactively by the same reading; recorded here,
     no re-run.
   - **Reading of W183f:** passed under rule (c) once (b′) returns 0 genuine diffs; the CLAUDE.md stop on W184 is
     lifted on that result (W183g (a) reports it before anything solves).
   - **Scope ("at both DSO builders").** Accepted as the Planner read it: the solve-bearing gate covers the sequential
     builder, the only one the configuration runs (`ParallelExecution: false`, pinned in spec v1); the parallel
     builder is covered by W183c's zero-solve NL + component identity (36 parallel blocks × 8 configurations within
     the 696/696). Recorded as a limit: no solve-bearing gate covers the parallel builder, which is not on the
     campaign path; a run that turned `ParallelExecution` on would need its own gate.

2. **E1 poll rule — B+: variant B with a direction-index advance where the frozen rule would refuse.** Facts: the
   completion of a one-node incumbent at the unit (31 feasible box points), of 7: 0.5/2 (34) and of the historical
   incumbent (61) exceeds the cap of 30 and the arm budget N^max = 30; at k = 0 every storage seed has 6–7 distinct
   feasible points after snapping (7 for the historical incumbent, 6 for the two single-node seeds), so the frozen
   rule stops every storage incumbent at poll 0; the unit reaches 8 distinct points at k = 2 only, 7: 0.5/2 at k = 2
   (9) and k = 3 (8), the historical incumbent at every k except 0 and 8; x = 0 has 3–5 distinct points at every k
   and a completion of 14, which runs. Rejected: (a) a seeds-only E1 (it answers R3.6's planning question not at
   all, and the cross-evaluations would compare seeds, not searched plans); (b) a cap of 61 (one poll of 61 × 0.75 h
   ≈ 46 h per arm exhausts the budget for a one-step full-box certificate); (c) a per-seed hand-picked start (as the
   Planner says).
   - **Ruled, declared here before any E1 solve, identical in both arms and at every poll:** when fewer than n + 1 = 8
     distinct feasible poll points remain after rounding and snapping, (i) if the completion of the incumbent has
     ≤ 30 feasible points it is used, as frozen; (ii) if it has more than 30, the Halton index advances by one and the
     2n directions are re-drawn, rounded and snapped, at most 10 consecutive indices, and the first index with at
     least n + 1 distinct feasible points is polled — the skipped indices and their snap tables are written into the
     poll record; (iii) if no index in the window yields n + 1 points, `STOP_FOR_REVIEW_completion_cap` as frozen.
     Everything else is the frozen variant B of Addendum 39 ruling 1 / spec v23: 2n = 14 OrthoMADS directions, the
     rounding, `SNAP_RULE`, Δ = 1 at every poll, acceptance of a poll point only above max{bar_x + bar_inc, σ_Q =
     18,449.66 €}, the A7 barrier rule, ≤ 30 new evaluations per arm, ≤ 10 launched polls, the index continuing from
     the last polled index + 1. Start index k₀ = 0 (t = 17) in both arms. The rule reads the snap table only — it
     carries no objective value, so it is a design declaration, not an outcome choice; the geometry table that
     motivates it is computable before any solve and is committed (W183b). The paper states the rule in one sentence
     where E1 is reported (round 9) with the reason (the completion of a one- or two-node incumbent exceeds the cap
     and the budget). The certificate scope stays as frozen: a terminal poll failure is "poll failure over the
     recorded poll set at unit mesh" with the spanning test recorded; it is a full-box certificate only when the
     completion was used; the E1 report states the scope per arm.
   - **Implementation:** the E1 launcher owns its poll loop (`run_continuation_e1`, a copy of `C.run_continuation`
     with the advance step (ii) inserted before the cap refusal); `C.build_poll` (pure) is imported unchanged; the
     committed `p515_s53_f2_certificate.py` is not edited; the poll record gains `halton_indices_skipped` and
     `advance_rule`. **Gate (zero-solve dry run, W183g (b)):** with the advance disabled, the launcher's loop
     reproduces the imported loop's poll-0 decision for each of the 8 seed-as-incumbent cases (refusal for the six
     storage cases, the 14-point completion for x = 0) with bitwise-equal poll records; with the advance enabled it
     polls the unit at k = 2 (8 points), 7: 0.5/2 at k = 2 (9), the historical incumbent at k = 1 (8), x = 0 at k = 0
     (completion 14) — these follow from W183b's committed table and are checks, not hypotheses; W183b's k = 0…9
     table is reproduced; the A7 rule and every other decision path are unchanged.

3. **Barrier stop rule: A7 confirmed** (stop for review after 2 new barrier evaluations in one poll or 3 overall); plan
   §9's "stop for review on a barrier point" is read as A7; every barrier point is reported with its cause.

4. **Spec v1 declarations (W183b items 3–6; report decision 3).** (i) Halton start k₀ = 0 per arm, under ruling 2.
   (ii) Arm A declares no `model_variant` (the baseline by absence; keys equal to E.21 rows 35 and 40), arm N declares
   the variant and its label; every entry carries `flex_price_multiplier` = 2.0 and its label — confirmed. (iii) No
   post-certification on seed and search entries (production exit, cap 500, 10 cycles) — confirmed. (iv) The batch
   gate re-runs **unit_A_2x, unit_N_2x, n7_05_2_A_2x, n7_05_2_N_2x** (the two node-7 plans under both models — the
   plan's "four node-7 seeds") as one 4-slot batch under `P515S55/e1_2x_batch4` (root-scoped cache, never seeding an
   arm), compared with the serial records on the gated per-cycle fields (`PER_CYCLE_TRAJECTORY_FIELDS`, W101/W142
   replay method), peak RSS recorded; bitwise identity authorizes 4 slots for the search batches, otherwise serial.
   (v) The G0 pointer for `--freeze`/`--run` is `P515S55/w183_nl_post_w183c_off/w183_nl_digest.json.gz` `ca195bd3…`
   with `w183d_g0/g0.json` `8cef45d6…` — confirmed. (vi) Cross-evaluations: x_A and x_N under A and under N (4; 2 if
   x_A = x_N), certified protocol = the ext-v6 `settling_resettle` v6 declaration as in spec `10f4aa7a` on the final
   code, phase `cross` under `P515S55/e1_2x`; a returned plan that is a seed is still re-evaluated under the certified
   protocol. (vii) The 1× screen: the 14 admissible unit neighbours of x = 0 under N at 1× **plus x = 0 at 1× on the
   final code** (production exit; evaluated once, the alias for N by G0, as in E1) = **15 evaluations** under
   `P515S55/e1_screen_1x`; compared by max{bar sum, σ_Q}; E.21 row 1 (the committed certified value of x = 0 at 1×) and row 54
   (the reference unit under N) reported beside as fixed-plan data, never as the comparator. (viii) E3: x = 0, 1×,
   `nrf_option` True (key and label), certified protocol as (vi), root `P515S55/e3_cnrf`, one evaluation. (ix) E2
   stage 1: {x = 0, reference unit} × {1×, 2×} × {`rho0_scale` 0.5, 2.0} = 8, certified protocol, root
   `P515S55/e2_rho`. (x) NRF binding tolerance 1e-3 MW — confirmed; E3 reports both measures (binding hours by the
   expected-value rule; per block the minimum row body and the count of rows within tolerance), states that at one
   scenario they coincide, and gives the per-node minima. (xi) W183g builds the thin fixed-plan launcher (precedent
   ext v6: `freeze_campaign_spec` + `evaluate`, pins, write-once roots, the capture checklist asserted before each run)
   for (vi)–(ix).

5. **Harness points (decision 4): both kept.** The `previous_iter.ess` guard under a declared `rho0_scale` is correct
   (the case file has it off). The `w183a_nrf_option` checklist entry in undeclared records is accepted; it is a
   provenance field, which ruling 1 (c) already treats as such in comparisons.

6. **E6 (decision 5).** Panel (b) uses the cost measure (€): per-block change against x = 0 of the TN conventional cost
   and of the DN flexibility cost, from `component_levels_terminal.json`; the caption says cost, not volume; no further
   pickle read. The supporting node-7 schedule for all 12 blocks stays. The three guarded loads are accepted as
   recorded; from now on a guarded-read authorization means as many guarded reads as the task needs, each logged with
   its run, so trial reads need no separate authorization. The declared day 2025 Spring (40,181.22 €, 15.8 % of
   253,539.62 €, determinate by 118×) is adopted; the figure is drawn by the expert in round 9 from `e6_export.json`
   (zero-solve); the 2035 Spring oscillation (block range 4.25–4.34 k€ over the certification windows) is noted for
   §4.7/E.7, no action.

7. **Runtime coverage of the "not confirmed" items.** NRF rows through the ADMM cycles: E3's terminal capture runs on
   the run's own terminal models, which are the last cycle's models, so 864 active rows per copy at terminal is the
   runtime evidence (W184 reports it). Scaled ρ through the cycles: the per-cycle record carries `rho_{v,pf,ess}_after`
   and `rho_{v,pf,ess}_action`; W185 checks on every E2 cell that the cycle-0 values are S39 × s exactly when the
   action is a hold, and S39 × s times the recorded adaptation otherwise. The parallel builder in a worker process is
   not on the campaign path (ruling 1). The launchers' `--freeze`/`--run` paths are first exercised by W184's single
   attended run and by W186's attended seeds phase.

8. **Order change.** W185 (E2 stage 1) follows W184 (E3) without waiting for the §4.4 text: the rewrite of §4.4 from
   E3's records is round-9 text work and E2's design does not depend on E3's outcome; the plan's "E3's report rewrites
   §4.4 before E2 starts" is replaced by "§4.4 is rewritten in round 9 from E3's records". The E2 stop rule (H2 fails
   → stop before E1) stands; W186/W187 wait for Addendum 86.

9. **Counts and time (replace Addendum 83 ruling 6):** E3 1; E2 8; E1 ≤ 75 (seeds 7, batch 4, searches ≤ 30 + ≤ 30,
   cross-evaluations ≤ 4); screen 15; E6 0. Serial worst case ≈ 2.5 + 10 + 57 + 11.25 ≈ 81 h; ≈ 49 h with the batch
   gate passed. D1 covers runs over 4 h within these limits; every extension needs a ruling.

**W183g order (Worker tasks authorized by this addendum; zero solves; builds only where named; one gate per task).**
- (a) Ruling 1: the (b′) comparator run against the W83 arm; wrapper v2 with `GATE_PASS_RULED` and the drift keys;
  nothing re-run. **Prediction:** 0 genuine, ≤ 3 provenance.
- (b) Ruling 2: E1 launcher v2 with the owned loop and the B+ advance; the dry-run gate of ruling 2 (advance off ≡
  imported loop at poll 0 for 8 cases, bitwise; advance on → k = 2/2/1/0, poll sizes 8/9/8/14, skipped indices
  recorded; W183b's table reproduced); the four batch seeds and the roots of ruling 4 declared in the spec.
- (c) Ruling 4 (xi): the fixed-plan launcher with phases E3 (1 entry), E2 (8), cross (≤ 4, declared after W186's
  searches), screen (15); zero-solve dry runs: keys recompute, E3's key carries `nrf_option`, E2's keys carry
  `rho0_scale` 0.5/2.0 and the price, the screen's 15 keys under N at 1× with the x = 0 alias, pins, write-once
  roots, the committed-spec scan (a committed holder of a key is reported and never reused: no cost is carried
  across code states), memory preflight at concurrency 1. **Prediction:** key counts 1/8/15; 0 committed holders of
  the E3 and E2 keys.
- (d) Spec v1 freeze: production head `6f43479e` and every module hash, harness `c34ebdc5…`, gates `6f4631e1…`,
  both launchers, oracle, settling v1–v6, the 81 inputs, the W2 and Phase A tables; keys of every phase; the B+ rule
  text; the counts of ruling 9; the capture checklist; `ParallelExecution: false` pinned; committed.
- (e) W183h as ordered in Addendum 84, at Overleaf `a87e195`; the HEAD stated.
- Report `P5_15_W183G_REPORT.md` (W183h separate or inside); commit and push. Nothing solves in W183g.

**W184 (E3) — authorized conditionally, no further addendum needed.** Launches, attended, as a separate commit with
its launch log, once W183g (a)–(d) hold (0 genuine in (a); the dry-run gates in (b) and (c); the freeze in (d)) and
W183g is reported. One certified evaluation of C_NRF (ruling 4 (viii)); stop for review if the certified status is
not reached within its cap (report the uncertified form with gap, slack and bar) or if `nrf_capture_error` is set.
Report `P5_15_W184_REPORT.md`: Q(C_NRF) with status, k\*/end, band or gap/slack, stopping ratio; the two conditional
differences with the plan §3 reporting rule and denominators (Q(P_NRF) − Q(C_NRF) relative to Q(P_NRF); Q(C_NRF) −
Q(C_OPEN) relative to Q(C_NRF)), recomputed from the records; the decomposition into TN conventional energy and DN
flexibility and the curtailment columns, as Table 12/E.23; the terminal NRF capture (rows and active rows per copy,
binding hours by both measures, per-node minima); the closure/σ/circle/SOC/interface diagnostics. **Hypotheses (plan
§3, plus one):** H3a Q(C_NRF) − Q(C_OPEN) positive and within resolution (< 2τ = 9.1 k€); H3b Q(P_NRF) − Q(C_NRF) =
90.9 ± 0.1 M€, 12.2 % at one decimal; H3c the four reverse-flow hours of C_OPEN (node 7, 2025: Autumn h19, Spring h22,
Spring h2, Summer h22) are among the binding hours of C_NRF, and 864 rows per copy are active at terminal.

**W185 (E2 stage 1) — authorized conditionally.** Follows W184's report without a further addendum if W184 ran to a
declared status and the terminal capture read 864 active rows per copy; eight certified evaluations serially (ruling
4 (ix)); the cycle-0 ρ check of ruling 7 on every cell; the margin and verdict rule of plan §4; report
`P5_15_W185_REPORT.md`. **Hypothesis:** H2 as in the plan (paired margins within the baseline pair's threshold of
their baseline values at each price; no pair's verdict changes); the ρ read-back holds on 8/8 cells. **Stop:** H2
fails at either price → stop before E1 and report. W186 (E1) and W187 (screen) wait for Addendum 86 after W185.

# Addendum 86 — rulings on W183g/W183h (`3a299aa0`); the fixed-plan launcher as a Step-7 hooks module with a gated router patch; the ρ read-back restated; E2 runs all eight; E1 per-poll launches; round-9 wording queued; W183i order; W184/W185 re-authorized conditionally (2026-10-26)

**Received.** `P5_15_W183G_REPORT.md` (+ `…W183G_A_GATE_RULED.md`, `…W183G_B_E1_V2.md`, `…W183H_ROUND8.md`); zero
solves everywhere (guard 0, pickle 0, builds 0); production `6f43479e` and harness `c34ebdc5…` unchanged; E1 launcher v2
`b0841b09…` (`7cf88b15`). Overleaf `a87e195` confirmed byte-identical to round 8 by the Planner (cmp) as by the expert.
W183g (c)/(d), W184 and W185 held on decision 1. The Studio's push reached origin: `3a299aa0` is on the Air.

**Predictions against outcomes (Addendum 85 / 84).** (a) 0 genuine vs the W83 arm — **held**; `GATE_PASS_RULED` True
with three negative controls; provenance ≤ 3 — **failed on the count** (4: the fourth is the W83 reference arm's own
`w83_convergence_depth_tail` read-back, configuration of the reference, not trajectory; the W184 condition "0 genuine"
is met). (b) advance off ≡ imported loop (8 cases, byte-equal), advance on → k = 2/2/1/0 with sizes 8/9/8/14, A7 and
every other path unchanged, the v1 assertions — **all held** (175/175). W183h: identity, Appendix E (33 PASS, 1,623
entries 0 MISMATCH, the `$^{b}$` rows exact), letter v10 pointers (167 MATCH / 0 / 0 / 5 SUBMITTED), the 16 A7
sentences (0 remaining), the 21 edits' numbers (0 MISMATCH; "about 3 %" = 2.887 %), 0 `[CONFIRM`, blue 23/23 —
**held**; over-budget labelled at every occurrence — **failed 23/25** (E.24 rows 39 and 6, no mark, no legend); 0
`<word>~\ref{app:` — **failed, 7** (introduced at `edd7a63`; W182c's pattern accepted "Section"/"Appendix" before
`app:`). The two-cycle gate is therefore **passed under the ruled reading**; the stop of W183f is lifted. The W183f
wrapper, the committed gate and their outputs are untouched.

**Rulings.**

1. **Fixed-plan launcher: option (a), the W155 pattern, with the router patch gated.** The harness's `settling_resettle`
   validators accept only cells of their own fixed tables (ext v6: the seven W135 cells), and the declaration carries
   the cell identity in the key; the `10f4aa7a` declaration validates and is refused with the cell renamed. Reusing a
   committed cell name (b) is rejected, as the Planner recommends: a false identity in every key and record. Running
   E3/E2 at the production exit (c) is rejected: H2 is a rule-v6 verdict and E3's status is part of its result.
   Ordered: a new hooks module `p515_s55_step7_resettle_hooks.py` (own schema; cells E3 1, E2 8, cross ≤ 4 added after
   W186; the v6 rule, state and capture checklists by import; the ext-v6 declaration shape with
   `nrf_option_expected` / `rho0_scale_expected` / `flex_price_multiplier_expected` and entry checks) and the
   one-branch addition in `resettle_hooks_module`. This changes the harness sha256 from `c34ebdc5…`, which is
   authorized here; spec v1 pins the patched hash, and the E1 launcher's pins are taken at its freeze on the patched
   harness. **Gates (W183i (a)), before any solve:** (i) every committed eval key recomputes (the W183a T1 method,
   7,213 entries of the 252 v2 specs); (ii) the router routes every committed schema to the same module as before;
   (iii) neutrality: the new branch is taken by no committed spec, and a planted spec of the new schema is routed to
   the new module; (iv) **the two-cycle bitwise gate re-run on the patched harness** (153 solves, about 2 min; the
   harness is in the gate's clean-file list) into a new write-once root, read under the ruled comparator against the
   W48 reference and the W83 arm — ordered, not optional: it closes the harness question at the cost of two minutes.
   Prediction for (iv): 0 genuine, `GATE_PASS_RULED` True, provenance ≤ 4. The 1× screen stays a thin production-exit
   phase of the same launcher (no resettle declaration). The cross-evaluations use `P515S55/e1_2x/cross` as their own
   phase root (decision 3, choice (d)).

2. **Ruling 7 restated.** The recorded `rho_*_after` is a mean over models; the Params equal S39 × s exactly (W183a T3)
   and the mean carries the summation's ulp. The check on every E2 cell is **bitwise equality with the replicated
   arithmetic** `sum([x] * n_g) / n_g` in the canonical interpreter, per channel g ∈ {v, pf, ess}: x = S39_g × s for a
   factor-1 action (held, fixed, exempt, held after failure); x = clamp(S39_g × s × f) for increased/decreased (f = 1.5
   or 1/1.5, clamp [1e-4, 1e4]); n_g the recorded model count of the channel. The restatement reproduces the committed
   W183f row (s = 1) on all three channels (pf 0.1980000000000002, n = 48; ess 0.010000000000000004, n = 51; v
   0.011549999999999998, n = 48). A cell that does not reproduce bitwise is a harness-fault stop. The check runs on
   cycle 0 and is reported for every cycle (the per-cycle actions are recorded, so the recursion is replicable).

3. **W185 and campaign declarations.** (i) **E2 completes all eight cells.** H2 is evaluated after each completed pair
   in the order (1×, 0.5), (1×, 2.0), (2×, 0.5), (2×, 2.0) and reported per pair; a failed pair does **not** stop E2 —
   the paper's fallback ("the margin as a range over configurations") needs all eight, the cost of finishing is at
   most 7.5 h, and the chain already stops before E1 (W186 waits for Addendum 87 in any case). The Planner's
   recommendation to stop at the first failing pair is not adopted, for that reason; a failed H2 is recorded as a
   failed prediction either way. (ii) `persist_certified_models` **on** for E3 and for all eight E2 cells (precedent
   `10f4aa7a`; the pickles stay untracked under the roots; the manifests hash them). (iii) **E1 per-poll launches
   (W183i (c)):** the E1 launcher gains a resumable mode — one launch per poll (seeds phase likewise in batches of its
   own size), the resume state rebuilt only from the phase root's own evaluation records (manifest-hashed, keys
   recomputed under the arm's model, the same root-scoping as the cache loader), the run precondition relaxed only for
   a resume into a phase root whose records all verify, and a refusal when the rebuilt state and the launcher's
   `k_next`/budget/incumbent disagree with the last poll record. Gate (zero-solve, stub records): a run split at every
   poll boundary reproduces the in-process loop's records byte-equal (the W183g (b) method), including after a B+
   advance and after an A7 stop; a planted foreign record in the root is refused. (iv) **The five B+ interpretation
   choices of W183g (b) are confirmed:** (a) the window includes the first refused index; (b) on exhaustion the
   terminating record is the first examined index's, byte-equal to the frozen loop's, with the nine further indices in
   `halton_indices_skipped`; (c) `advance_rule.polled_index` null on a cap refusal; (d) `--freeze`/`--run` accept only
   `P515S55/e1_2x` and `P515S55/e1_2x_batch4`, cross under `e1_2x/cross`; (e) the renamed dry-run output. (v) The
   window stays at 10: the scan's exhausting cases (the unit at k = 3…12, 7: 0.5/2 at k = 4…13) are reachable only if
   such a point becomes the incumbent at those indices, which the seed-incumbent rule and the unit-step polls make
   unlikely but not impossible; rule (iii) applies and a `STOP_FOR_REVIEW_completion_cap` is a reportable E1 outcome,
   ruled on if it occurs, never patched mid-run.

4. **Round-9 wording queue (expert; no run depends on it).** (i) The seven `Appendix~`/`Section~\ref{app:…}` prefixes
   at `main.tex` l. 530, 1631, 1639, 1742, 1763, 1780, 2002 → the bare `\ref` (the label prints its own word on
   elsarticle); the W182c/W183h pattern is tightened to `\b(Appendix|Section)~\\ref\{app:` for the next check. (ii)
   E.24 rows 39 and 6 (7: 1.25/5, 2025) get the over-budget mark and the caption its legend, through `gen_appE.py`.
   (iii) The E.23 start mark becomes `$^{s}$` (generator), so `$^{b}$` means over budget only. (iv) "Within numerical
   resolution of break-even" at l. 1520 and l. 1806 and in cover letter v2 l. 19 → the letter-v10 wording ("within the
   numerical resolution of zero, so its sign is not resolved"). (v) The cover letter's bracketed paragraph is filled
   from W184–W187. (vi) The letter's prediction-record bullet (l. 286): the author's call stands; the expert's
   recommendation is to drop it and keep the record internal (audit §4). All six go into the round-9 deliverables.

**W183i order (Worker tasks authorized by this addendum; zero solves except gate (a)(iv)'s 153).**
- (a) Ruling 1: hooks module + router patch; gates (i)–(iv); report before anything else runs.
- (b) The fixed-plan launcher on the patched harness: phases E3 (1), E2 (8), screen (15), cross (≤ 4, declared after
  W186); dry runs as Addendum 85 W183g (c) (keys recompute; E3's key carries `nrf_option`; E2's keys carry
  `rho0_scale` 0.5/2.0 and the price; the screen's 15 keys under N at 1× with the x = 0 alias; pins; write-once roots;
  the committed-spec scan; preflight). Prediction: key counts 1/8/15; 0 committed holders of the E3 and E2 keys.
- (c) Ruling 3 (iii): E1 launcher v3 with per-poll launches; its gate.
- (d) Spec v1 freeze on the patched harness: as Addendum 85 W183g (d), plus the ρ check of ruling 2 and the
  declarations of ruling 3 in the spec text; the E1 launcher's pins refreshed at freeze.
- Report `P5_15_W183I_REPORT.md`; commit and push.

**W184 (E3) and W185 (E2) — re-authorized conditionally, no further addendum.** W184 launches, attended, once W183i
(a), (b) and (d) hold ((c) may follow in parallel; it is needed before W186 only), under its Addendum 85 line (one
certified evaluation of C_NRF, `persist_certified_models` on, hypotheses H3a–H3c, the terminal NRF capture, stop for
review on an unreached status or `nrf_capture_error`). W185 follows W184's report under its Addendum 85 line with
rulings 2 and 3 (i)–(ii) of this addendum (eight cells serially, all completed; H2 per pair; the ρ read-back on every
cell; prediction: H2 holds at both prices and the read-back holds 8/8). W186 (E1) and W187 (screen) wait for
Addendum 87 after W185.


# Addendum 87 — realignment direction; convention change; D0 and T-perf orders; W184/W185 held (drafted by the Planner 2026-10-30; adopted by the author 2026-10-31)

**Adoption (author, 2026-10-31).** Decisions 1–6 and the agreements recorded under "Author's agreement" are the
author's. The items under "Rulings requested from the expert" (R1 confirmation, R3, R4, R6, F4 wording, and the
salvage-parameter convention) are forwarded in `EXPERT_REQUEST_2026-10-31.md` and remain open until Addendum 88.
D0 (`P5_16_D0_SPEC_v2.md`, route A — the four W86 pickles verified on the 24 GB machine), the convention-change
zero-solve tasks (`P5_16_CONVENTION_CHANGE_SPEC_v2.md`) and T-perf (`P5_16_TPERF_SPEC_v1.md`) are authorized.

**Status: adopted** (see Adoption below). Basis: `REVISION_REALIGNMENT_ANALYSIS_2026-10-30.md` (v2) and the Advisor review it records.

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

**Correction of the correction (Advisor review of spec v1, verified):** the W86 tight-tail reference pair (x = 0
cycle 132, unit cycle 112) ran with `persist_certified_models: true` from the Studio checkout; their
`certified_models.pkl` and ESSO pickles are hash-recorded (spec v2 §2) but absent from the 24 GB machine's checkout.
**Author action:** check the Studio for the four files; if present with matching sha256, D0 is zero-ADMM-solve
(hash-verified loads, route A). Only if absent: in-process reproduction with a comparator parameterised to the W86
records (route B, ≈ 1.5–2 h each). Spec v2 `P5_16_D0_SPEC_v2.md` also: deactivate-and-add construction on
production's own W-space rows; ESSO P-net slacks fixed at 0; rated-vs-available capacity variants; row inventory and
1e-12 identity gates; all 12 groups per cell; the differential prediction P6 (gap plan-independent) beside the
absolute class — relevant to R4.

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

## T-perf (author-authorized 2026-10-31)
Diagnostic `P5_16_TPERF_SPEC_v1.md`: six declared timed solves on the certified unit-cell blocks here and six on
paper-scale blocks on the Studio — NL-write / IPOPT / load split, iterations, scaling factors, RSS — producing a
ranked lever table. A formulation lever adopted from it (per-unit base / scaling, W-auxiliary structure) enters
the convention-change spec as C6 with the Step 1b candidate-3′ gate, so it rides the single re-evaluation campaign.

## Stops

As `CLAUDE.md`. In addition: D0's T0.1 gate failure; any D0 reading in H3; any salvage basis change after freeze.
