# P5.15 S56: step 3 completion (v2) and the step 4 pre-run diagnostics D-1, D-2, D-3 (2026-11-23)

**Authority:** `S56_PLAN_v3.md` §7 rows 3–4; TASKS.md S56 section (Planner rulings recorded at `fc5dc766`);
`S56_STEP4_DESIGN.md` decision 8; the Planner's task of 2026-11-23 (rulings 1–3; one frozen diagnostics spec).

**Spec:** `data/SRP1/Results/P515S56/step4_diag_v1/frozen_step4_diag_spec_v2_ff466289.json` (version 2; predecessor
step 3 spec v1 `4017dc70`). **Launcher:** `p515_s56_step4_diag.py`. Both committed in `59cda418` with a zero-solve
trial (`trial_v1`) before launch.

**Run:** `step4_diag_v1/run_v1`, attached, alone, both streams captured; 586 s wall; **50 of 50 declared solves, every
guard exact**; no harness fault.

**Objective convention (every table):** C values are **gross** — the step 3 merged / single-scenario
`model.objective` (settlement weight 0, no row 18, no pin, no salvage), € per representative day, ω-weighted.
D-1 values are production `_get_operational_recourse_block_components` — **settlement excluded in full**, pin excluded,
salvage separate — in € of total discounted cost (w_b applied). Allowances are (R/4)·w_b/Σ₁₂w, R = 10,000 €.

## Decisions needed

1. **2025|Summer has no C_central.** In S3-2, singles 12 and 22 hit the iteration limit (500), with the same stall
   signature as run_v1's 2025|Winter single 02 (μ 1.84e-6, unscaled dual infeasibility 2.0e6 / 2.6e6, constraint
   violation 2.4e-4 / 4.4e-4). Recovery was off, as declared; nothing was re-run. Both merged starts converged to the
   same point (185,674.155753 €/day; they differ by 1.0e-8 €).
   - **(a) Recommended.** A spec v3 with two warm single solves (12, 22) from the S3-2 merged-default slice, which is
     saved (`run_v1/primal/S3_2__logs__2025_Summer__merged_default.json.gz`). This remedy worked for 2025|Winter
     single 02 (P2v2 and P3v2 held). 2 solves, seconds each.
   - **(b)** Use the merged per-scenario terms for 12 and 22 (bias ≈ +0.035 €/day per scenario, as measured).
   - **(c)** Leave 2025|Summer without a comparator. It is a passenger block of instance B.

   The D-3 box check for 2025|Summer is also missing. It can be done zero-solve at step 4's own initialisation
   round, or in the spec v3 of (a), which then needs instance B's init round again (11 solves).
2. **D-3 box reading.** On the two evaluable blocks, **no value lies beyond IPOPT's bound relaxation** (1e-8·max(1,
   |bound|)). Read strictly, though, 48 values (2025|Winter) and 38 values (2035|Summer) sit above the 1.1 pu voltage
   limit, by at most 4.6e-10 pu.
   - That tolerance was **fixed after a draft zero-solve trial**, on run_v1 data, had shown the strict excess. It is
     recorded in the spec (`d3.bound_tol_note`).
   - Because 2025|Summer cannot be evaluated, the evaluator records D-3 boxes as **not held**.
   - The Planner should rule on the tolerance reading.
3. **For the Expert, from the design's own conditions:**
   - **X4-2 is triggered.** D-1's production-depth offset exceeds the allowance on both blocks (2.61× and 2.66×). At
     tail depth it falls to 0.20× and 0.22×.
   - **X4-1:** the recommended rule ("eliminate V-2 if the hard-pin cost exceeds the allowance") is met on
     2035|Summer. There Δ_pin weighted = 1,823.6 € = 9.72× the allowance. On 2025|Winter it is 0.81×.

## Blocked on the author

None.

## Changed

| commit | content |
|---|---|
| `a07d2e71` | Instance B (ruling 2): `p515_s56_step4_instances.py --instance-b`, `step4_instances_v2/`, gates G1–G5, inventory `step4_instances_v2_manifest_sha256.json` |
| `59cda418` | `p515_s56_step4_diag.py`, spec v2 `ff466289`, zero-solve `trial_v1` and its inventory — before launch |
| this commit | `run_v1` evidence (results, children, IPOPT logs, the S3-R and S3-1 primal files), stdout/stderr, `run_v1_manifest_sha256.json`, this report |

- **No production file was edited.**
- **Instance B:** `step4_instances_v2/SRP1__s56_step4_2025_2035_Summer.json`, case sha256 `b1cd5440…`, scenario
  checksum `ade7456c…`.
- **The 2035-only instance is retired.** Its files are kept and it is not used.
- **The tooling change** to `p515_s56_step4_instances.py` is a new `--instance-b` mode. G1 now restricts the derived
  snapshot to the block, which is a no-op for single-block instances. The σ proxy reports "not available" for a block
  that has no run_v1 solve.

**Configuration:**
- 12-block instance: `s56_srp1grid_3x3`, checksum `73d0a230…`.
- Step 3 settings reused verbatim from `4017dc70`:
  - tol 1e-5, compl_inf_tol 1e-6, MA97;
  - recovery off;
  - objective alignment.
- ADMM children:
  - option on, 'per_scenario', row 18 Q charge off, α = 0, x0;
  - σ_fixed 93,635,360;
  - p58 rescale;
  - thread caps 1;
  - the case-file AA and tail declaration of the G-f configuration.
- D-1:
  - recovery off;
  - production depth: TSO 5e-4; DSO key not set, which is IPOPT's 1e-4;
  - tail depth 1e-6.

**Not committed (hash-recorded in `run_v1/manifest_sha256.json`):** 29 of the 32 primal files, about 30 MB; the
instance-read diagrams; the children's derived-read and arm directories.

## Found

### Predictions against outcomes

| prediction (verbatim in the spec) | outcome |
|---|---|
| **S3-R gate:** the reproduced merged objective is bitwise equal to run_v1, and every per-scenario interface P, Q, V equals it | **held**: 175,188.7137093924 (`0x1.562a5b5ad451ap+17`) and 153,017.2487446982 (`0x1.2adc9fd6ddc3fp+17`), both ==; 81/81 interface lists equal per block (max diff 0); 94 / 95 iterations, as run_v1 |
| **P2v2:** single 02 Optimal | **held**: Optimal, 69 iterations, μ at floor 9.09e-11 |
| **P3v2:** objective = 171,052.349 − offset ∈ [0.030, 0.040] € | **held**: 171,052.31402652664, offset **0.034964 €** |
| step 3 P3 for 2025\|Winter (warm merged vs Σω C_s ≤ 1e-6) | **held**: rel 2.0e-7 (+0.034960 €) |
| step 3 P4 for 2025\|Winter (default merged ≤ 1e-5) | **held**: rel 2.0e-7 (+0.034960 €) |
| **S3-2:** all Optimal; P3 / P4 as in run_v1 | **failed**: 9/11 Optimal; singles 12 and 22 maxIter; P3 / P4 not evaluable (decision 1). Merged default 185,674.1557530493, warm 185,674.1557530391 |
| **D-3 σ:** the assert passes on both instances (B: settles G3) | **held**: A σ_computed 5.518e7 (ratio 0.589; largest agent DSO9); B 5.282e7 (ratio 0.564; largest TSO 2035\|Summer); window [3.12e7, 2.81e8]. **G3 for B: inside** |
| **D-3 boxes:** every value inside its box | **not held as frozen** (2025\|Summer not evaluable). 0 values beyond tolerance on 2025\|Winter and 2035\|Summer; strict count 48 and 38 (decision 2) |
| **D1a:** production-depth \|offset\| ≥ allowance (H2) | **held**: +597.67 € (2.61× 228.64), +499.03 € (2.66× 187.56) |
| **D1b:** tail-depth \|offset\| ≤ allowance / 3 | **held**: +45.67 € (≤ 76.21), +40.58 € (≤ 62.52) |
| **D-2:** Δ_pin ≥ −0.04 €/day | **held**: +0.4056 and +4.8856 €/day |

### Per-block table

| block | w_b | allowance (€) | C_central (€/day, gross) | D-1 offset, production (€) | D-1 offset, tail (€) | Δ_pin €/day (weighted €; × allowance) | D-3 minimum margin |
|---|---|---|---|---|---|---|---|
| 2025\|Winter | 455.0 | 228.64 | **175,188.678750** | +597.67 (1.314 €/day; 2.61×) | +45.67 (0.100 €/day; 0.20×) | +0.4056 (+184.53 €; 0.81×) | P: 47.33 MW (TSO δ box), 37.02 MW (DSO ref-gen); Q: 98.66 / 89.93 MVAr; V: −4.6e-10 pu (strict), inside tolerance |
| 2035\|Summer | 373.2585 | 187.56 | **153,017.214164** | +499.03 (1.337 €/day; 2.66×) | +40.58 (0.109 €/day; 0.22×) | +4.8856 (+1,823.59 €; 9.72×) | P: 27.34 MW (TSO δ box), **0.563 MW** (DSO7 ref-gen limit 100.001 MW); Q: 97.95 / 85.53 MVAr; V: −1.9e-10 pu (strict), inside tolerance |
| 2025\|Summer | 455.0 | 228.64 | **undefined** (singles 12, 22 maxIter) | not run (not in the order) | not run | not run | not evaluable (no central point) |

- **C_central** for 2025|Winter = Σω of run_v1's 8 singles plus S3-1's single 02. For 2035|Summer it is run_v1's
  value. Weighted: 79,710,848.8 € and 57,114,972.2 €.
- **Δ_pin** = C_hardpin − C_central. Against the reproduced merged objective instead, it is +0.3706 and +4.8510 €/day.
  It therefore includes the merged solve's ≈ +0.035 €/day stopping offset.
- **D-3 margins** are min(value − lower, upper − value) per family, over 648 values each (3 DNs × 9 scenarios ×
  24 hours). The smallest relative margin (margin / half-width) is −4.6e-9 for 2025|Winter and −1.9e-9 for
  2035|Summer, both on V. The smallest P relative margin is 0.0056 (DSO7 ref-gen, 2035|Summer).

### D-1 detail (production `_get_operational_recourse_block_components`, settlement excluded in full)

**Method:**
- Every per-scenario interface copy was fixed at the central point: 648 each of TSO δ_p, δ_q, V and DSO pg_ref, qg_ref,
  V_ref.
- The interface reproduces the central values to 1.4e-14 MW.
- There is no DN reference-node load or shunt, and the shared-ESS term is 0.
- The models are in production's cycle-1 Param state (zero-solve, stubbed `optimize`). Every objective was evaluable
  before the solve.
- All 16 solves were Optimal and loaded. There was no infeasible local problem, so nothing points to H4.

| block | depth | agent | iterations | μ_final | floor status | vars at lb / ub | ineq rows at ub | free vars |
|---|---|---|---|---|---|---|---|---|
| 2025\|Winter | production | DSO5 / DSO7 / DSO9 / TSO | 77 / 73 / 93 / 40 | 4.28e-8 / 1.12e-8 / 1.62e-8 / 3.35e-8 | at / at / at / at | 33,906/5,635 · 33,524/3,289 · 33,913/4,654 · 3,366/1,164 | 663 · 660 · 708 · 1,136 | 87,480 ×3 · 18,360 |
| 2025\|Winter | tail | DSO5 / DSO7 / DSO9 / TSO | 76 / 67 / 87 / 21 | 2.51e-9 / 9.39e-11 / 1.67e-10 / 2.24e-10 | above / at / at / at | 35,138/8,063 · 35,124/7,449 · 35,099/8,264 · 3,444/1,164 | 869 · 866 · 817 · 1,136 | same |
| 2035\|Summer | production | DSO5 / DSO7 / DSO9 / TSO | 146 / 97 / 118 / 74 | 1.66e-8 / 9.16e-9 / 1.08e-8 / 2.46e-8 | at ×4 | 32,034/6,285 · 31,660/4,190 · 30,435/6,829 · 3,387/933 | 721 · 815 · 809 · 1,176 | same |
| 2035\|Summer | tail | DSO5 / DSO7 / DSO9 / TSO | 125 / 90 / 86 / 32 | 2.04e-10 / 6.72e-11 / 1.37e-10 / 1.79e-10 | at ×4 | 34,150/8,110 · 34,359/6,953 · 31,157/11,193 · 3,524/936 | 854 · 956 · 948 · 1,177 (+58 at lb) | same |

- Active bound = |value − bound| ≤ 1e-6 in model units. Inequality rows at lb are 0 everywhere except 58 on the TSO
  2035|Summer at tail.
- **Production − tail, per agent (€):**
  - 2025|Winter: TSO 67.28, DSO5 157.98, DSO7 163.18, DSO9 163.56; sum 552.0.
  - 2035|Summer: TSO 62.18, DSO5 131.18, DSO7 131.50, DSO9 133.58; sum 458.4.
- **Offset split per day** (local parts − C_central parts):

  | block | depth | generation + flexibility (€/day) | slack terms (€/day) |
  |---|---|---|---|
  | 2025\|Winter | production | +0.751 | +0.563 |
  | 2025\|Winter | tail | +0.018 | +0.082 |
  | 2035\|Summer | production | +0.741 | +0.596 |
  | 2035\|Summer | tail | +0.007 | +0.102 |

  At tail depth the remainder therefore sits mostly in the slack terms. This is an observation; it is not attributed.

### D-2 detail

- **Construction:** 576 added rows, vmag_sqr[TN_dn, s, t] = vmag_sqr[TN_dn, (0,0), t].
- **Warm start:** from the S3-R primal file (278,424 Vars set).
- **Solves:** both Optimal, at 76 and 100 iterations, with μ 2.5e-9 (above the floor, as the merged solves).
- **After the solve:**
  - pin residual 0;
  - interface V spread across scenarios ≤ 1.9e-12 pu;
  - max violation 1.0e-8;
  - 0 TN curtailment scenario-hours.
- **Parts** (generation + flexibility / slack): 175,191.759 / −2.675 and 153,024.775 / −2.675 €/day.

### D-3: ADMM-only bounds and where I looked

**Searched:**
- Code reading, listed in the spec `d3.code_read`:
  - `create_transmission_network_model`;
  - `interface_pf_p/q_transmission_def`;
  - `_wire_per_scenario_interface_coupling`;
  - `create_distribution_networks_models_sequential`;
  - `update_*_to_admm`;
  - `combine_networks`;
  - `vmag_bounds` / `vmag_sqr_bounds` / the voltage rows.
- A programmatic diff of every ADMM-prepared block (A: 4 blocks; B: 8 blocks) against a plain build with the same
  candidate. It compares (lb, ub, fixed, fixed value) per Var, Var and Constraint components present on one side only,
  and constraint active states.

**Found, identical on every block:**
- **TSO, at the ADN bus / load (648 entries each):**
  - voltage slacks fixed at 0;
  - pc / qc bounds removed (plain ±1e-5) and fixed at the DSO initial expectation;
  - the four flexibility legs fixed at 0;
  - every `interface_delta_p/q` copy freed within ±rating (plain: fixed at 0).
- **DSO, at the reference node (216 entries):** `e` bounds **widened** from [0.9999, 1.0001] to [0, 1.1501]. This
  loosens rather than tightens.
- **New on both sides:** the expectation Vars, with `expected_interface_vmag` ≥ 0, their defining rows, and the
  inactive shared-ESS expectation rows. No constraint changed active state, and none is absent.
- **Merge-level** (step 3 `relaxation_limits`): the DN reference-generator P/Q limits (±100.001 MW / MVAr in the
  models) and the DN reference-node voltage limits exist only in the coordinated models.

**Boxes checked:**
- TSO δ box: pc ± rating;
- TSO V: vmag ∩ vmag_sqr bounds ∩ [v_min, v_max], with the slacks fixed at 0;
- DSO reference-generator pg / qg bounds;
- DSO reference-node V.

The values checked are each block's central single-scenario interface values.

### Instance B gates (zero solves, `a07d2e71`)

| gate | 2025\|Summer | 2035\|Summer |
|---|---|---|
| G1 data bitwise vs the 12-block block | **holds**: 644/644 arrays + probabilities; object walks 0 diffs; network cost arrays equal | **holds**: same |
| G2 weight ratio | 1.0 (455.0) | 1.0 (373.2585) |
| G3 σ in window | proxy not available zero-solve → **settled at D-3: inside** (B σ_computed 5.282e7) | proxies straddled → **settled at D-3: inside** |
| G4 harness and ESSO accept | **holds** (investment year 2025, harness default) | **holds** |
| G5 NL identity (init + cycle 1, plain/labelled/.row/.col) | **8/8** at scale 1.0 | **8/8** at scale 1.0 |

### Solve counts, wall time, RSS

| process | declared | observed (solves / launches) | sites | wall | peak RSS |
|---|---|---|---|---|---|
| parent | 16 (S3-R 2, S3-1 1, S3-2 11, D-2 2) | 16 / 16, and 16 / 16 on the step 3 harness guard | `_solve_declared` 16 | 585 s (586 s `/usr/bin/time`) | python 2.13 GB; IPOPT ≤ 1.07 GiB sampled |
| child A | 15 (init 7, D-1 8) | 15 / 15 | `_init_round` 7, `_d1_solve` 8 | 82 s (init 15.5 s) | python 1.70 GB; IPOPT 0.36 GB |
| child B | 19 (init 11, D-1 8) | 19 / 19 | `_init_round` 11, `_d1_solve` 8 | 119 s (init 32.2 s) | python 2.53 GB; IPOPT 0.36 GB |
| total | **50** | **50 / 50**, nothing blocked; pickle 0/0 in every process | | | |

- **No recovery attempt** in the initialisation round: exec counts equal solve counts.
- **Initialisation exits:** every network solve was Optimal. The TSO 2025|Winter solve stopped above its floor
  (μ 1.84e-6). Every ESSO solve was ok/optimal.
- **stderr:**
  - the children's stderr is 0 bytes;
  - `run_v1_stderr.log` holds only `/usr/bin/time -l` output (my wrapper);
  - the harness wrote nothing to stderr.
- **Imported repository modules** were committed clean in all three processes.

## Unexpected findings

- **Cold single-scenario solves are fragile.** 3 of 27 have now hit maxIter (run_v1 2025|Winter 02; 2025|Summer 12
  and 22). All three stall at the same μ (1.8449e-6) with dual infeasibility ~1e6. The merged solves and the warm
  single were not affected.
- **D-3 diff, two plain-build facts:**
  - the TSO's plain ADN load has pc/qc bounds of ±1e-5;
  - the DSO reference node's `e` is bounded to [0.9999, 1.0001] in a plain build and widened by the ADMM path.
- **The DSO7 reference-generator P limit is within 0.56 MW** of the 2035|Summer central exchange (DN7 s01 t13, 99.44
  MW). This is consistent with step 3's closest approach.

## Not confirmed

- **2025|Summer C_central, its step 3 P3/P4, and its D-3 boxes.** Searched: `run_v1/step4_diag_results.json` →
  `solves.S3_2|2025|Summer`. Two singles were not loaded.
- **What makes up the remaining tail-depth offset** (0.10–0.11 €/day, mostly in the slack terms). Not separated into
  H1–H4. Not checked: whether the central singles' slack terms and the local ones measure the same quantities. The
  TSO fixes its ADN-bus voltage slacks at 0, which the merged model does not.
- **The production-depth floor reading.**
  - Every production-depth local solve stopped at its μ floor, and tail depth removed 552 / 458 € of the offset.
  - This shows the stopping-depth effect. It does not show that the tail-depth remainder is a floor.
- **The D-1 Param state.** It is production's cycle-1 pre-solve state, except that the TSO's requests are the DSOs'
  *initial* expectations (no cycle-1 DSO solve precedes). With the interface fixed, these Params enter only constant
  terms and the IPOPT path. Their effect was not isolated.
- **Global optimality of any solve**, including the hard-pin merged solve, which was run from one start only.
- **That the D-3 inventory is complete beyond Var bounds and fixings.** Param values and objective terms are not diffed
  (they are not bounds). Merge-level differences come from code reading and step 3's `relaxation_limits`.
- **That the stopping-slack reading of the +0.035 €/day merged − singles offset** is right. It now covers run_v1's
  17 pairs (+0.03450 to +0.03505 €), single 02 (+0.034964 €) and the 7 converged 2025|Summer pairs (+0.03485 to
  +0.03538 €). It remains a hypothesis. No tighter-tolerance merged solve was run.
