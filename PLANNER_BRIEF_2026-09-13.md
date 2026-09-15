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
| 18 | **Author's decision (2026-09-15): soft non-anticipativity retained, made defensible.** Interface P, Q: deviation of each scenario from the day-ahead schedule priced at `c = α·π̄`, with `π̄` the probability-weighted market price of the representative day and `α` an author-set **imbalance premium** (default 0.25; sensitivity over α ∈ {0.1, 0.25, 0.5, 1.0}; α = 1 is "deviation at the full energy price", the author's conservative case); **linear** in `\|dev\|` (two nonnegative variables); category **E**, in Q(x). Interface V: **no deviation term** (physical state, not a commitment). **Shared-ESS dispatch scenario-independent** in the network models (hard NA for the committed schedule the ESSO degrades — closes the P5.13-A discrepancy by construction). Results report realized interface deviations per scenario (max/expected, MW and % of rating) and per-scenario AC feasibility. Implement in Step 5. | replaces an arbitrary 9e4 (effective 85.7 after scaling — hard NA in disguise) with a priced, interpretable term; consistent with Option A |
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

## Update obligations

At the end of Step 1 the Planner rewrites the "CURRENT SOURCE OF TRUTH" head of
`REVISION_CONTEXT.md` to reflect: the withdrawal of the `C*` feasibility-boundary claim, the
warm-start mechanism, the reformulated ESSO, and the reopened numerical programme. Historical
sections are not rewritten. `COWORK_HANDOFF.md` is marked superseded by this brief.
