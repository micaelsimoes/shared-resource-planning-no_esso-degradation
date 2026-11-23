# P5.15 S56 step 3: centralized comparator at x = 0, α = 0, two blocks (2026-11-23)

**Authority:** `S56_PLAN_v3.md` §7 row 3 (existing authorization); TASKS.md S56 section; the Planner's task of 2026-11-23.

**Spec:** `data/SRP1/Results/P515S56/step3_central_v1/frozen_step3_central_spec_v1_4017dc70.json` (v1, no predecessor).

**Harness:** `p515_s56_central_ms_x0.py` (sha256 `e2205575…`). The harness, the spec and the trial were committed in `9d37e0df` before the launch.

**Instance:** `s56_srp1grid_3x3`, case sha256 `71d34849…`, scenario checksum **`73d0a230…`** (read and checked).

**Run:**
- attached and alone, with stdout and stderr captured; stderr is 0 bytes;
- total wall time 311 s;
- guards exact: 22 solves and 22 launches from `_solve_one`, none blocked; 0 pickle calls.

**Objective convention (every table):** gross. That is `model.objective` with interface-settlement weight 0, no row 18
and no voltage pin:
- generation + flexibility + load-curtailment cost + slack terms;
- settlement excluded (the merged model has no interface);
- no salvage (x = 0);
- € per representative day, probability-weighted over the block's 9 scenarios (ω_s = 1/9 here).

**C_central(block) = Σ_s ω_s C_s**, where C_s is the objective of the single-scenario model. The block weight is stated
beside it, not applied: years[y] × days[d] / 1.02^(y − 2025), as in `NetworkData.get_primal_value`.

## Decisions needed

1. **The 2025|Winter comparator is incomplete.**
   - **What happened:** single 02 (s_m = 0, s_o = 2) hit the iteration limit (500 iterations, 13.0 s). It stalled at an
     objective of 171,075.595 € with an unscaled constraint violation of 2.6e-5 and dual infeasibility of 6.3e5. From
     about iteration 250 the line search was backtracking 15 times per step.
   - **Consequence:** Σ_s ω_s C_s cannot be formed for this block, so P3, P4 and P5 cannot be evaluated there.
   - **What is still available:** both merged starts reached the same point, 175,188.714 € (they differ by 2.2e-9 €).
     The merged solution's own scenario-02 term is 171,052.349 €.
   - **Options** (each needs a spec v2, because v1 permits no other solve):
     - **(a) Recommended.** One guarded solve of single 02, warm-started from the scenario-02 slice of the merged
       solution. It takes seconds and tests whether that point is the single model's optimum.
     - **(b)** One cold re-solve of single 02 with the production recovery tiers on (cold start, then
       `mu_strategy adaptive`).
     - **(c)** Accept the merged value as C_central for 2025|Winter. It would carry the uniform +0.035 € per scenario
       stopping offset measured on the other 17 pairs (see Found).
2. **Which step 4 arms calibrate against this comparator** (Advisor design review, which is active).
   - **F1 + V-1** (`per_scenario_voltage_mode = 'per_scenario'`) at α = 0 is the same problem, up to the relaxations
     listed in Found, none of which binds here.
   - **F1 + V-2** (`'pin'`) and the **pinned (legacy) arm** are **not** the same problem: the voltage pin couples the
     scenarios. For those arms this comparator is a relaxation, that is a lower bound, not an equality target.
3. **The block choice did not produce a TN-curtailment block.**
   - The zero-solve surplus proxy picked 2035|Summer as the block with the most curtailment (84 scenario-hours of
     system surplus).
   - The centralized solution of that block has **0 MWh of TN curtailment in every scenario**, and at most 0.33 MWh
     per day of ADN curtailment.
   - The proxy ignores flexibility and losses, and the surplus is absorbed without curtailment.
   - If step 4 needs a block where TN curtailment actually occurs (the coupling proposal's V1 intent), the block has to
     be chosen differently, under a new spec.

## Blocked on the author

None.

## Changed

- **`9d37e0df`** (before the launch): the audited harness, the frozen spec v1 `4017dc70`, the trial spec draft
  (`trial_spec_draft_v0.json`, block choice null), the trial results with their manifest, and the trial logs.
- **This commit:**
  - `run_v1/central_ms_x0_results.json` and `run_v1/manifest_sha256.json`;
  - the 22 IPOPT logs;
  - `run_v1_stdout.log` and `run_v1_stderr.log`;
  - `run_v1_manifest_sha256.json` (the hash inventory of every committed file);
  - this report.
- **Not committed:** the reader's diagram PDFs under `*/instance_read/`. They are hash-recorded in the two run
  manifests.
- **No production file was edited.** TASKS.md is left to the Planner.

**Audit fixes to the interrupted draft** (no redesign):
1. **The test for a loaded solution.**
   - **Fix:** `solution_loaded` now uses production's `solver_result_succeeded(result)`.
   - **Why:** the draft used `result is not None`. A failed solve returns a result without loading it, so the draft
     would have read failures as loaded (the S56 G harness defect).
   - **Effect here:** single 02 is correctly recorded as not loaded.
2. **The predictions.**
   - **Fix:** they were rewritten to the Planner's P1–P5, and the guards are now verified before the predictions are
     evaluated, because P1 reads them.
   - **Why:** the draft carried an unrelated P1–P4, including a memory limit.
3. **New phase-A checks:**
   - `solver_params.verbose` is off, because production's `_run_smopf` calls `exit()` after a failed solve when it is
     on;
   - the block-choice premise holds: every TN conventional generator has P_min = 0.
4. **New capture only:**
   - per-solve IPOPT-child peak RSS, sampled every 0.1 s, beside the running maximum of `ru_maxrss`;
   - the block weight;
   - relaxation margins;
   - the instance checksum in the results and the manifest.
5. **The mapped-point evaluation before the warm solve is capture-only**, so a None-valued Var cannot abort the
   declared solve set.

The changes were checked as follows:
- zero-solve exercises of the capture, warm-map and evaluate paths (scratch, guard armed, 0 solves);
- the official trial;
- a zero-solve trial against the frozen spec (block choice recomputed equals the spec).

## Found

### The zero-solve gates (trial and launch, both blocks)

15 of 15 checks pass.
- **Size:** the merged model has 343,656 Vars and 216,675 (2025|Winter) and 217,863 (2035|Summer) active rows.
- **No coupling:** every row carries exactly one scenario label, equal to its own index.
- **Slice identity:** each of the 9 single models equals the merged model restricted to its scenario, bitwise:
  - 24,075 to 24,207 rows per scenario;
  - every Var bound and fixing;
  - every per-scenario objective expression.
- **Objectives at a random point:** the merged objective equals Σω of the single objectives to 2.8e-16 relative.
- **Block choice:**
  - fewest curtailment: 2025|Winter (0 surplus hours);
  - most curtailment: 2035|Summer (84 scenario-hours, 983 MWh weighted).

### Predictions (recorded in the spec before launch) against outcomes

| prediction | outcome |
|---|---|
| P1: phase A passes; 22 solves and 22 launches, guards exact | **held**: 0 failed checks; 22 / 22 / 0 blocked; pickle 0/0 |
| P2: all 22 solves Optimal | **failed**: 21/22. 2025\|Winter single 02 reached "Maximum Number of Iterations Exceeded" (500); recovery off as specified, not re-run |
| P3: warm-started merged = Σω C_s within 1e-6 (per block) | 2035\|Summer **held**: +0.0346 €, 2.26e-7. 2025\|Winter **not evaluable** (single 02 not loaded), so P3 is recorded as not held |
| P4: default-start merged = Σω C_s within 1e-5 (per block) | 2035\|Summer **held**: +0.0346 €, 2.26e-7. 2025\|Winter **not evaluable**. Not a stop |
| P5: central cost per block in the plan convention, block weight stated, checksum in every artifact | 2035\|Summer **complete**. 2025\|Winter **incomplete** (C_central = Σω C_s undefined), so P5 is recorded as not held. The checksum is in the spec, the results (declared and read) and both manifests |

### Central cost per block (gross, € per representative day)

| block | weight | C_central = Σω C_s | merged, default start | merged, warm start | merged − Σω |
|---|---|---|---|---|---|
| 2025\|Winter | 455.0 | **undefined** (8 of 9 singles; 02 failed) | 175,188.713709 | 175,188.713709 | — |
| 2035\|Summer | 373.258 | **153,017.214164** | 153,017.248745 | 153,017.248745 | +0.034580 (2.26e-7) |

- **Parts for 2035|Summer:**
  - generation + flexibility: Σω = 153,019.902 (merged 153,019.924);
  - slack terms: Σω = −2.688 (merged −2.675).
- **Weighted for 2035|Summer:** 153,017.214 × 373.258 = 57,114,972 €.

### Resolution of the merged-versus-singles comparison

On all 17 converged pairs (8 in 2025|Winter, 9 in 2035|Summer), the merged per-scenario objective lies above the
single's by **+0.03450 to +0.03505 €**, a nearly uniform amount.
- **Composition:** +0.022 € comes from generation + flexibility and +0.0125 € from the slack terms.
- **Where the solves stopped:**
  - the merged solves stop at μ = 2.5e-9, above their floor, with objective scaling 0.009;
  - the singles stop at their floor, 9.1e-11, with scaling 0.001.
- **The warm start:**
  - it began at the singles' point: objective = Σω C_s exactly, maximum violation 1.3e-8;
  - after 254 iterations it ended 0.0346 € higher, at the same point as the default start (they differ by 6.4e-10 €).
- **Reading (a hypothesis, not tested):** the offset is the stopping slack of the merged interior-point solve, not a
  difference between the problems. This is consistent with the bitwise slice identity.
- **Comparison bar this implies:**
  - about 0.035 € per representative day (≈ 13 € on the weighted planning scale for 2035|Summer);
  - far below R = 10,000 €.

### Per solve: status, wall time and memory

| block | solve | IPOPT exit | iterations | wall (s) | IPOPT peak RSS (GiB) | objective (€/day) |
|---|---|---|---|---|---|---|
| 2025\|Winter | merged, default start | Optimal | 94 | 27.6 | 1.067 | 175,188.714 |
| | singles 00, 01, 10, 11, 12, 20, 21, 22 | Optimal | 62–86 | 1.9–2.7 | 0.116–0.131 | see results |
| | single 02 | Maximum Number of Iterations Exceeded | 500 | 13.0 | 0.146 | not loaded |
| | merged, warm start | Optimal | 91 | 26.4 | 0.993 | 175,188.714 |
| 2035\|Summer | merged, default start | Optimal | 95 | 29.0 | 1.071 | 153,017.249 |
| | singles (9) | Optimal | 79–190 | 2.4–4.9 | 0.116–0.146 | see results |
| | merged, warm start | Optimal | 254 | 67.9 | 1.284 | 153,017.249 |

Notes on the table:
- **IPOPT peak RSS** is sampled every 0.1 s. The exact running maximum (`ru_maxrss` over the children) is 1.284 GiB.
- **The Python process** peaked at 1.98 GiB.
- **Largest constraint violation after a solve:**
  - ≤ 1.3e-8 for the singles and the 2025 merged solves;
  - 4.2e-7 for both 2035 merged solves.
- **Per-solve values for every solve:** `run_v1/central_ms_x0_results.json` → `solves.<block>.<solve>.solve`.

### Relaxations of the merge against the coordinated models

None of them binds on either block. Scenario-hours outside each limit, merged default and warm:

| limit | scenario-hours outside | closest approach |
|---|---|---|
| dropped DN reference-generator P limits | 0 | DN 7 in 2035\|Summer: 0.56 MW below its 100 MW limit |
| dropped DN reference-generator Q limits | 0 | |
| \|P\| above the interface rating | 0 | DN 7 in 2035\|Summer: 0.56 MW below its 100 MVA rating |
| DN reference-node V limits | 0 | V reaches the 1.1 pu upper limit (margin ~1e-8) |

- **The voltage limits are not a relaxation.** The DN reference-node limits and the TN bus limits are identical
  (0.9–1.1 pu at all three DNs), so the 1.1 pu limit binds in both models alike.

### Audit: step 4 at α = 0 with per-scenario coupling, against this merged model

Recorded in the spec as `audit_equivalence_with_step4`. Code read at HEAD `e8474703`.

- **The commitment P̄/Q̄** (`expected_interface_pf_p/q`) is a Var **defined** on each side by a row
  P̄ = Σ_s ω_s P_s. It adds no restriction.
- **At α = 0 the commitment is unpriced.** Row 18 is not constructed: `add_scenario_commitment_terms` requires
  `premium_alpha > 0`. The commitment's only other use is the expectation consensus channel, which per-scenario consensus
  implies.
- **The TN realized exchange is not tied to the commitment.** The TSO's per-scenario `interface_delta` copies are freed
  within ±R (`_wire_per_scenario_interface_coupling`).
- **Under V-1 the pin is not constructed on either side.**
- **The settlement.** The ADMM path sets the settlement weight to 1 on both sides. At per-scenario consensus the TSO term
  (−π_s P^T_s) and the DSO term (+π_s P^D_s) cancel scenario by scenario with equal ω_s. The step-1 Q convention excludes
  the settlement in full.
- **Conclusion:** F1 + V-1 targets this problem. V-2 and the pinned arm do not (Decision 2).
- **Known relaxations, reported and not enforced:**
  - the dropped DN reference-generator limits;
  - the DN reference node merged into the TN bus;
  - the ADMM TSO fixes the voltage slacks at the ADN buses to 0, which the merged model does not;
  - no ±R box on `interface_delta` around the TSO's fixed pc.

## Not confirmed

- **2025|Winter C_central.** Not established: single 02 did not converge, and no other solve was permitted.
- **The stopping-slack reading of the +0.035 € offset.** It is a hypothesis. No tighter-tolerance merged solve was run.
- **The ±R box on the TSO `interface_delta` around pc.** It cannot be checked against this run: pc is the DSO's
  initial consensus value, known only once step 4 runs. Only |P| ≤ R was checked.
- **Global optimality.** Not established for any solve. Two merged starts reaching the same point is evidence of a
  common basin, not a proof.
- **The equivalence audit** is a code reading of `shared_resources_planning.py` (`_wire_per_scenario_interface_coupling`,
  `create_transmission_network_model`, `create_distribution_networks_models_sequential`, `_prepare_*_objectives_for_admm`,
  `_get_operational_recourse_components`) and of `model_construction_helpers.py` (`add_scenario_commitment_terms`,
  `objective_function_rule`, the interface rules). It was not run: no ADMM model was built with the option on in this
  step.
