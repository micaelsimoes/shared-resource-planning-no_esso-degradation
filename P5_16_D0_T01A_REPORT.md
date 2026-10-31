# P5.16 D0 / T0.1a — relaxation builder dry run (G-D0-0), zero solves

Worker, 2026-10-31. Spec: `P5_16_D0_SPEC_v2.md` (sha256 `c93ef819…0f3`), sections 3, 4 (G-D0-0 / G-D0-2 / G-D0-3), 7.
Evidence: `data/SRP1/Results/P516D0/dry_run/` (untracked data root), hash inventory `manifest_sha256.json`
(sha256 `04033ff7429c8e515f899a872a5d807381fad340276cf19299a2dadab69f6e67`, 64 files, child exit code 2 = a gate failed,
no stop rule fired).

## Blocked on Planner

1. **Retained-row gate failed on the DSO block (4.79e-11 against the 1e-12 bar), below the 1e-10 stop line.** It is
   not a construction error, and I did not change the tolerance. All of the excess is on `branch_flow_limit`, branch 8
   of case33_2, where the row coefficient |y|² = 1.08e5. Mapping W from (e, f) moves W by one ulp (2.2e-16) from the
   snapshot's own W, so the row residual moves by 1.08e5 × 2.2e-16 × 2 = 4.79e-11. That shift equals the first-order
   bound (4.789141e-11) and is within the row's own floating-point evaluation resolution
   (eps·Σ|terms| = 5.11e-11). Three checks support this:
   (a) evaluated at the *same* point, the production row and the relaxed row give exactly the same residual
   (max |Δ| = 0.0 over 5,261 DSO rows and 2,808 TSO rows);
   (b) if W is kept at the snapshot's own values (diagnostic M0), the retained Δ is exactly 0.0;
   (c) no row has Δ above its first-order bound plus 4 × its evaluation resolution (0 rows).
   **Decision:** keep the absolute 1e-12 gate (case33_2's current-limited rows cannot meet it for any W mapping that
   is exact rank-1), or restate it relative to row scale (for example Δ ≤ k·eps·Σ|terms|). The same question will
   arise on the certified blocks at T0.1.
2. **The task's inventory gate conflicts with the B2 envelope on the DSO.** "Replaced class is exactly the
   definitional families" (and P-a) cannot hold on a block with a transformer. The OLTC (branch 31, bus 1→2) puts
   r·WijR, r·WijI and r_sqr·Wii into 192 more rows: `node_balance_p/q` at the transformer's two nodes (48 + 48) and
   `pij/qij/pji/qji_def` (24 each). These have to be rewritten in (U, C, D); otherwise the block stays nonconvex. I
   classified them as `replaced / envelope_rewrite`, so every row is still in exactly one class and nothing is dropped.
   Each rewrite is generated from the production row's own degree-2 standard representation, not re-typed, and is
   exact to 1.16e-13. **Decision:** accept that the replaced class is the 5 definitional families plus the
   envelope-rewrite rows.
3. **Retained nonconvex row: `sess_comp`.** This is `shared_es_pch_hat · shared_es_pdch_hat ≤ 1e-4`: 24 rows on the
   DSO block and 72 on the TSO block. It is a bilinear, nonconvex inequality, and current production builds it too:
   SRP1's `case9_params.json` and `case33_2_params.json` both set `shared_ess_model: BILINEAR_RELAXATION`
   (`network.py` l.512, `sess_comp_rule` in `model_construction_helpers.py` l.933). Spec v2 section 3 does not mention
   it. As the task requires, I retained it, so the relaxed network blocks are **not convex**, and Gurobi with
   `NonConvex 0` (spec G-D0-5) will refuse them. Dropping it is safe for the lower bound (it removes an inequality),
   but the task does not allow dropping rows on a network block. **Decision for the expert or Planner:** drop
   `sess_comp` from the relaxation (recommended, with that one-line argument), or relax it some other way.
4. **Most FrozenSMOPF pickles do not load: 21 of 24 fail.** The failures include the preferred
   `frozen_DSO_node7_case33_2_2025_Summer_cycle7.pkl`, both `failure_TSO_*` pickles,
   `reference_case33_2_2025_Winter_cycle10.pkl` and all five `P3Preserved/*`. Every one fails with
   `AttributeError: Can't get attribute 's_bounds' on model_construction_helpers`; `s_bounds` was deleted in P5.4-B
   (`1e86d40e`, before the unwire-never-delete rule). I used the two that load:
   `matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl` (the DSO block, in place of Summer) and
   `matched_success_TSO_case9_2025_Summer_cycle7.pkl`. Restoring loadability would need a retained, unwired `s_bounds`
   callable in production, which is outside this task. **Decision:** whether to order that.
5. **A write outside the permitted directory during exploration, before the harness existed.** One exploratory call to
   `SharedResourcesPlanning.read_planning_problem()` regenerated the 52 untracked scenario-diagram PDFs in
   `data/SRP1/Diagrams/` (production's known data-loading side effect, `REVISION_CONTEXT.md` l.3330). No tracked or
   cited artefact was touched. The evidence run redirects these writes to `dry_run/side_writes/Diagrams/` and checks
   containment: 0 files modified outside the output directory, and tracked status unchanged.

## Changed

- New, untracked, staged by filename only (not committed): `p516_d0_build_relaxation.py` (sha256 `a538a008…b763` at
  run time), `p516_d0_dry_run.py` (`4376ab63…9bce` at run time), and this report. No production file was edited.
- The run was made at repo HEAD `bca88de3` (the Planner's T-perf commit landed during the task). Interpreter
  `/opt/anaconda3/envs/opf_env_py311/bin/python`, Python 3.11.11, Pyomo 6.9.5, numpy 2.4.2. The SRP1 scenario checksum
  is `5a02b77c…0358`.
- Builder API, as the task asks:
  - `inventory_rows(block, kind)`
  - `relax_network_block(block, capacity_mode, rated_capacity)`
  - `relax_esso_block(esso_model)`
  - `lb_objective(blocks)` and `attach_lb_objective`
  - `mapped_point(block, relaxed, w_source='ef')`
  - `export_lp(relaxed, path)`
  - Additions: `production_rows_at_point` (same-point check), `objective_form` and per-row form labels (convexity
    audit), and `model_fingerprint` (non-mutation proof).
  - The docstrings cite every family with its line numbers. The builder checks each definitional family's rule
    function by name (`vmag_sqr_def`, `vmag_def`, `voltage_product_real_rule`, `voltage_product_imag_rule`,
    `r_sqr_rule`) and raises on drift.
- Harness: the parent creates the output directory atomically, which refuses an existing run or a concurrent copy. It
  runs a child process with stdout and stderr merged into `run.log` and writes the manifest. The blocking
  `SolveProfileGuard` is installed before any production import. Predictions are written to `predictions.json` before
  any computation. A capture-path checklist (15 quantities) is asserted before the run and again before writing. All
  JSON goes through `gate_result_io.dump`. `P516_D0_REHEARSAL_DIR=/tmp/...` is a rehearsal-only override that refuses
  anything outside /tmp. Two rehearsals into /tmp and the evidence run produced byte-identical LP files.

## Found

Inputs: DSO `3b0aba78…8564`, TSO `1bc0aa68…9304`, ESSO `esso_models_s39_D.pkl` `cfac19ea…3732` (nodes 5, 7, 9). Mapped
points are these cycle-7 / terminal snapshots, not the certified points.

| Gate (task) | Block | Achieved | Bar | Verdict |
|---|---|---|---|---|
| Inventory: each active row in exactly one class; inventory = builder's deactivated/active sets | DSO, TSO (both modes), ESSO ×3 | consistent (DSO 2,568 replaced / 5,261 retained; TSO 720 / 2,808; ESSO 15 dropped / 1,185 retained) | exact | PASS |
| Retained-row \|Δ\| | DSO (R and A) | **4.789e-11** (`branch_flow_limit[8,0,0,12]`) | 1e-12 | **FAIL** (below 1e-10 stop; see Blocked 1) |
| Retained-row \|Δ\| | TSO (R and A) | 3.872e-13 (`branch_flow_limit[3,0,0,9]`) | 1e-12 | PASS |
| Retained-row \|Δ\| | ESSO 5/7/9 | 1.36e-20 | 1e-12 | PASS |
| Cone relative residual, W | DSO / TSO | 6.78e-16 / 5.40e-16 | 1e-12 | PASS |
| Cone relative residual, transformer (U, C, D) | DSO | 6.10e-16 | 1e-12 | PASS |
| Envelope rewrite \|Δ\| (spec G-D0-3) | DSO, 192 rows | 1.16e-13 | 1e-9 | PASS (also below 1e-12) |
| Envelope bounds r_lb²·Wii ≤ U ≤ r_ub²·Wii at the mapped point | DSO | min relative slack 0.272 | ≥ 0 | holds |
| Objective identity, relative | DSO, TSO, summed (R and A) | 0.0 | 1e-8 | PASS |
| Guard (blocking, declared 0) | process | 0 solves, 0 execs, 0 blocked; gurobipy, highspy, clarabel, cplex, xpress, mosek not imported | exact 0 | PASS |
| Input not mutated (fingerprint) | all 5 inputs | unchanged | exact | PASS |
| Containment | process | 0 files outside the output directory | 0 | PASS |

Objective convention (all objective values): gross = `objective_function_rule(b)` − contracted settlement_b − voltage
pin_b, unweighted per block, with w_b = `_get_admm_block_weight` = 455.0 for both blocks (equal to the fixture's own
`admm_block_weight`). On these fixtures the settlement and pin terms are both 0, because neither fixture has them.
Instance: the fixtures named above. DSO gross 710.533919031818, TSO gross 203,515.47633546146, the same on the relaxed
and original blocks. **This identity is close to vacuous: no objective term involves W.**

Predictions (recorded in `predictions.json` before the run):

- **P-a, exactly five definitional families replaced: FAILED.**
  - DSO: all 5 families are replaced, plus 192 envelope-rewrite rows in 6 more families.
  - TSO: 4 families are replaced; `r_sqr_def` has 0 rows because case9 has no transformer.
  - Row counts: DSO 792 / 24 / 768 / 768 / 24 (`voltage_mag_sqr_def`, `voltage_mag_def`, `voltage_product_real_def`,
    `voltage_product_imag_def`, `r_sqr_def`); TSO 216 / 72 / 216 / 216 / 0.
- **P-b, retained identity ≤ 1e-13: FAILED.** Achieved 4.79e-11 on the DSO and 3.87e-13 on the TSO, both from
  `branch_flow_limit`. Every other family is ≤ 1.14e-13.
- **P-c, cone ≤ 1e-15: HELD.** Achieved 6.78e-16.
- **P-d, objective identity ≤ 1e-12: HELD.** Achieved exactly 0.0.
- **P-e, transformers present: HELD, via the DN interface only.**
  - case9 has 0 transformers: no `transformers` key in `case9_2025.json`, 0 in the fixture.
  - Each case33_k has 1 OLTC (bus 1→2, `vmag_reg` 1, r ∈ [0.83, 1.17]) in the current case files. In the loaded
    DSO fixture this is branch 31, which carries 24 branch-periods.

Capacity variants:

- Variant A keeps the values the fixture carried: e_avail = 1.8114e-4 pu on the DSO, and
  1.6299e-4 / 1.8114e-4 / 1.8073e-4 pu on the TSO.
- Variant R sets the rated values from `network.shared_energy_storages[e].{s,e}`: e = 2.1270e-4 pu
  (1.9143e-4 for TSO ESS 0).
- In R the only rows that read the capacity Params are `shared_energy_storage_s/e_sensitivities`: 2 on the DSO and 6 on
  the TSO. They are excluded from the R gate and listed, with Δ ≤ 3.20e-5, which equals rated − available.

ESSO (all three nodes are identical in structure; the only investment is cohort 2025, 0.96875 MVA / 3.875 MWh):

- **Dropped rows**, each with its argument in `ESSO_DROP_ARGUMENTS`:
  - 9 `available_e_capacity_unit` rows, replaced by 9 rows `E_avail == E_rated`
  - 3 SoH exp-chain rows
  - 3 `soh_min` floor rows
  - 0 active H3 rows (all 864 deactivated)
- **Retained:** the 3 linear D rows.
- **Slacks:** 288 + 288 `slack_es_pnet_*` are fixed at 0. Their snapshot values were at most 1.0e-8 in magnitude and
  sum to −2.88e-6 per family (slightly negative).
- **Objectives:** both deactivated, replaced by a zero objective for the LP writer.
- No non-polynomial row remains active.

Exports: 7 LP files with symbolic labels and a name map:

- DSO R and A: 7,709 vars / 6,293 cons each
- TSO R and A: 2,604 / 3,024 each
- ESSO ×3: 1,195 / 1,194 each. The three ESSO files are byte-identical: an LP file carries no values, and the
  investments are equal.

## Not confirmed

- **Live certified blocks are not tested.** These fixtures predate P5.15-1b and Step 3.1-C:
  - they carry the retired `sess_phi_limit_*` rows, the `shared_energy_storage_s/e_sensitivities` rows, and a bilinear
    `sess_pch_hat_link` (with S as a Var);
  - they lack `interface_settlement_weight`, so `network.get_primal_value` and
    `srp._get_local_interface_settlement` raise AttributeError on them.
  - The objective was therefore evaluated through `_PreSettlementFixtureView`, which supplies 0 for the two missing
    attributes. It reproduces each fixture's own stored `objective` exactly (difference 0.0).
  - The fixtures' objective contains the bilinear `shared_es_pch · shared_es_pdch` complementarity penalty (minimum
    eigenvalue −2.275e6), which current production removed in P5.15 Step 3 row 9. On the live blocks the family list,
    the capacity-row exclusion set and the objective form will differ. The classifier is generic, but G-D0-2 and
    G-D0-3 must be re-run on the route A blocks.
- **Coupling rows (spec section 3 item 6) are not built or tested:** single blocks from different runs cannot be
  coupled meaningfully. `voltage_mag_def` is deactivated, so `vmag` and `expected_interface_vmag_def` no longer bind W;
  the coupling has to go on Wii, as the spec says.
- **No relaxation was solved**, so feasibility of the exported LPs, solver acceptance and the gap are unknown. Gurobi
  would in any case reject the retained `sess_comp` rows and the fixtures' nonconvex objective (Blocked 3).
- **Where I looked** for the `s_bounds` history: `git log -S "def s_bounds"` on `model_construction_helpers.py` and
  `P5_4_ACTIVE_ENERGY_ESS_PRODUCTIONIZATION_REPORT.md` l.322. I did not search other branches.
