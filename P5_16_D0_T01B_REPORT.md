# P5.16 D0 / T0.1b — relaxation on the route-A certified blocks (both cells, 12 groups, R and A), zero solves

Worker, 2026-10-31. Spec: `P5_16_D0_SPEC_v2.md` (sha256 `af7d3ac1…`, read at run time; sections 2, 3, 4, 7, 10 and
the §11 amendments that appeared during the task). Evidence: `data/SRP1/Results/P516D0/t01b/` (untracked data root,
952 MB, 209 files), hash inventory `manifest_sha256.json` sha256
`4e5bbd45234598094593a8daca807a576590eee52f492f5281b63cf621876562`. Child exit code **3: a stop rule fired**. Git HEAD
at the run: `95148ca7`.

## Blocked on Planner

1. **Stop rule fired: retained-row Δ > 1e-10 at the rank-1 mapped point.** It fired in 88 of 192 block-modes (368
   rows). The maximum is **2.08e-9** on `branch_flow_limit[(8,0,0,3)]`, case33_3 (DSO9), 2030 Winter, in both cells.
   Ruling 1 (ii), |Δ| ≤ max(1e-12, 8·eps·Σ|terms|), fails on **3,086 rows in 144 block-modes**.
   - **Cause.** The certified points are rank-1 only to IPOPT tolerance, not to 1 ulp as the T0.1a fixtures were.
     The worst definitional-row residuals at the certified points are 4.1e-11 (TSO `voltage_mag_sqr_def`), 3.3e-11
     (TSO `voltage_product_real_def`), 2.2e-11 (DSO `voltage_mag_sqr_def`) and 1.8e-11 (DSO `r_sqr_def`). Mapping W
     to e² + f² moves W by those amounts, and the case33 `branch_flow_limit` coefficients (|y|² ≈ 1e5) amplify the
     shift.
   - **Evidence that the construction is right.** 0 rows exceed the first-order W bound + 4·eps·Σ|terms|. Same-point
     Δ is exactly 0 on all 810,912 row checks.
   - **No single mapped point passes every §10 gate on these blocks.** If W is kept at the certified values (the
     "snapshot" diagnostic), retained Δ is exactly 0, but the cone residual is up to **1.84e-11** (W) and
     **3.67e-11** (transformer) relative. That is above the 1e-12 cone gate in 20 and 14 block-modes respectively.
   - **Decision.**
     - (a) Restate ruling 1 (ii) and the stop line as |Δ| ≤ first-order W bound + 4·eps·Σ|terms|. This holds on every
       retained row. The mapped point stays exactly rank-1, so cones are ≤ 7.3e-16.
     - (b) Gate the retained rows at the snapshot-W point, and restate the cone gate relative to the certified
       definitional residual.
     - (c) Something else.
   - **Recommendation: (a).** Same-point Δ = 0 stays the primary gate, and the bound tracks each point's measured
     rank-1 residual.
   - **Cost of (a).** No re-run is needed for the retained rows: `identities.json` stores, for every row with
     Δ > 1e-13, its Δ, `first_order_w_bound` and `eval_resolution`, plus per-block counts. The envelope rows (item 2)
     would need one small addition.
2. **The envelope-rewrite gate (≤ 1e-9) failed in 10 block-modes** (5 groups × 2 modes). The maximum is **6.16e-9**
   on `pij_def[(31,0,0,21)]`, the OLTC branch, DSO9 2025 Autumn. The mechanism is the same as item 1: those groups
   carry the largest `r_sqr_def` / `vmag_sqr_def` residuals at the certified point (1.6–2.2e-11). At the snapshot-W
   point the same rows match to ≤ 1.27e-13. A scratch check outside the evidence root
   (`/tmp/t01b_explore/env_bound.py`, reading the evidence captures) puts every one of the 960 rewritten rows in
   those 5 block-modes within its first-order bound (maximum Δ 6.110e-9 against a bound of 6.113e-9). That check is
   **not preserved evidence**; ordering (a) would add it to the harness.
3. **Variant C was not built.** Spec §11 D-2 (Addendum 88, committed in `95148ca7` while this task was running) adds
   calendar-only available energy to T0.1b "(follow-up task if it has returned without it)". My order named rated
   and available only. C also needs a definition that the order does not give: the φ_cal exponent at block end on
   SRP1, and the change to the ESSO's available rows. **A follow-up task is needed.**
4. **The builder change is not staged.** The task allows `git add` of the harness and this report only.
   `p516_d0_build_relaxation.py` is modified and unstaged (sha256 `f08491c1…` at run time). It needs to be staged and
   committed with this evidence.
5. **Ruling needed before T0.2 (containment G-D0-4).** The certified gross includes **−13,012.57 €** (x0) and
   **−13,032.47 €** (unit) of detector penalties (`detector_penalty_total` in each `evaluation_record.json`). They
   come from slack variables sitting at about −1e-8, inside IPOPT's bound relaxation. In the captured vectors, the
   certified point violates variable bounds by up to 1.0e-8 (slacks, `pg`, `flex_p_up`).
   - A bound-respecting conic solve cannot reach those negative contributions. Q_LB ≤ Q_cert + gap can therefore fail
     by up to about 13 k€ (2.0e-5 of Q) with no row defect.
   - The difference between the two cells is 19.90 € (the shared-ESS day-balance slack). So it barely touches P6, but
     it is the same order as P6's 5 k€ point estimate and below the H1 resolution.
   - **Decision:** what T0.2 compares Q_LB against: gross, `economic_recourse_all_D_excluded`, or gross corrected by
     the detector terms.
6. **Variant R keeps one capacity-dependent Var bound at the available value.** The bounds of
   `slack_shared_es_soc_final_up/down` on the network copies are set by `configure_shared_ess_operational_state`
   (MCH l.1117) as e × `ESS_DAY_BALANCE_SLACK_FRACTION` + 1e-5. The builder resets only the two capacity Params
   (spec §3 item 3).
   - Example: the unit cell in 2035 has ub 3.6e-4 pu (available 0.007), where rated (0.01) would give 5.1e-4.
   - R is still a relaxation of production, which has the same bound. It is not "production at rated capacity" in
     this one bound.
   - **Decision:** whether R (and C) also reset it.

## Changed

- `p516_d0_build_relaxation.py`: extended, not rewritten. Additions:
  - `DROPPED_NETWORK_FAMILIES` (ruling 3, with its LB-safety argument)
  - `CONVEX_FORMS`
  - `_check_sess_comp_rows`, which verifies the BILINEAR_RELAXATION form `1.0·pch_hat[k]·pdch_hat[k] ≤ 1e-4` and
    raises on any other form
  - `nonconvex_active_rows`
  - an opt-in `drop_sess_comp=False` on `inventory_rows` and `relax_network_block`

  The defaults are unchanged. A rehearsal of `p516_d0_dry_run.py` into `/tmp` reproduces all 7 T0.1a LP files and
  `name_map.json` byte-identically, against the committed manifest `04033ff7`. All 42 gate rows, the predictions and
  the build records are identical too.
- `p516_d0_t01b.py`: new harness (sha256 `9d122f75…`). The parent creates the root atomically (refuses if it exists),
  merges stderr into `run.log`, takes git provenance before and after, and writes the manifest. In the child:
  - the blocking `SolveProfileGuard` (permitted = ()) is armed before any production import;
  - a `sys.addaudithook` hook raises on any process launch and records every write-open and filesystem mutation;
  - `predictions.json` is written before any input is loaded;
  - a 20-item capture-path checklist is asserted before the run and again before writing;
  - all JSON goes through `gate_result_io.dump`;
  - the four input sha256 are checked against spec §2 before anything is loaded (refuse = exit 5).
- Output files: `predictions.json`, `inventory.json`, `identities.json`, `coupling.json`, `objective_identity.json`,
  `lp/index.json` plus 48 `.lp` files and 48 name maps, `captures/` (24 group pickles, 2 ESSO pickles, 26
  Var/mutable-Param vector files), `side_writes/Diagrams/` (production's diagram PDFs, redirected), `run.log`,
  `provenance_parent*.json`, `manifest_sha256.json`.
- Staged by filename: `p516_d0_t01b.py`, `P5_16_D0_T01B_REPORT.md`. Nothing committed.
- Configuration: interpreter `/opt/anaconda3/envs/opf_env_py311/bin/python`, Pyomo 6.9.5. The case files match each
  cell's record: `SRP1_params.json` `dbfdb2a0…`, `SRP1_ESS_Params.json` `39106f93…`.
- Wall time 570 s for the child.

## Found

**Instances.** Both cells are W86 tight-tail, configuration s39_D, objective convention `gross_operational_cost`
(settlement-excluded, salvage not subtracted).

| cell | eval dir | candidate_key | investment (MVA, MWh) @ node 5 / 7 / 9 | certified cycle | recorded gross (€) | rule-ten terminal step / threshold | eps_pri v / pf / ess |
|---|---|---|---|---|---|---|---|
| x0 | `5cfe69a615ae3708_x0` | `8435c718…` | 0 / 0 / 0 | 132 | 653,858,731.5686293 | 0.0024 | 3.426e-3 / 1.375e-3 / 7.20e-4 |
| unit_n7 | `ca8927e75d628bd1_n7_4h_e1` | `db77e154…` | 0 / (0.25, 1.0) / 0 | 112 | 653,599,356.2420883 | 0.0557 | 3.426e-3 / 1.375e-3 / 7.90e-4 |

**Inputs and provenance.**
- All four input sha256 match spec §2, and are unchanged after loading and at the end of the run.
- Production `_get_operational_recourse_components`, called on the pickled models, reproduces the recorded gross
  exactly (relative difference 0.0) for both cells.
- Boyd r_v and r_pf recomputed from the blocks' `expected_*` Vars (SRP l.7196–7240 formulas) equal the recorded
  values: x0 to 0.0 relative, unit to ≤ 7.0e-16.
- So the pickles are the certified iterate.

**Predictions** (recorded in `predictions.json` before any load):

| | prediction | outcome |
|---|---|---|
| Q1 | no nonconvex row remains after dropping `sess_comp` | **HELD.** 0 non-convex rows in 246 audits (192 relaxed blocks, 48 assembled groups, 6 ESSO). `sess_comp` dropped: 24 rows per group on TSO and DSO7 at the unit; 0 at x0 (rows inactive at zero capacity). Form verified on every dropped row. |
| Q2 | same-point Δ = 0 on every retained row, all 24 groups | **HELD.** Δ = 0.0 exactly on 810,912 row checks (24 groups × 2 modes × 4 blocks), and on the 6 ESSO. |
| Q3 | coupling residuals = inter-copy mismatches, and ≤ recorded eps_pri per channel | **HELD.** See the coupling table. |
| Q4 | objective identity ≤ 1e-12 per block; 12-group sum = recorded gross to ≤ 1e-8 | **HELD.** Worst block 5.15e-15. Sum vs recorded: 1.8e-16 (x0), 5.5e-16 (unit), both modes. |
| Q5 | unit e_available < rated in 2030 and 2035; = rated at x = 0 | **HELD.** Node-7 copies (TSO e1, DSO7 e0) carry 0.0077984 (2030) and 0.0070000 (2035) pu against rated 0.01 pu. At x0 every copy carries 0 = rated. 2025 is also below rated (0.0088039; end-of-block SoH 0.8804), which the prediction did not address. |

**Gates.** 2,216 gate rows. Two kinds failed: ruling 1 (ii) at the rank-1 mapped point (144 rows) and envelope
≤ 1e-9 (10 rows); see Blocked items 1–2. Everything below passed:
- inventory == builder and the per-family counts, on every block and mode;
- each active row classified exactly once;
- 26 capture round-trip fingerprints;
- cones at the rank-1 point: ≤ 7.33e-16 (W) and ≤ 7.16e-16 (transformer U, C, D);
- objective identities;
- coupling equalities, identical in R and A;
- live blocks not mutated;
- inputs unchanged;
- guard 0 / 0, with 0 process launches, 0 solver libraries imported, and 0 write-opens in the repository outside
  the root.

Containment detail:
- The filesystem events outside the root are `os.makedirs` attempts by production's loader on the existing
  `data/SRP1/Results` and `data/SRP1/Results/Logs` (no file written), and on `~/.matplotlib`.
- Tracked git status was unchanged across the run.
- In the first rehearsal another agent staged `PLANNER_BRIEF_ADDENDUM88_2026-10-31.md`, which the parent's status
  diff caught. My process wrote nothing there (audit hook).

**Inventory.** Rows summed over the 12 groups:

| cell | block | active | replaced: cone | replaced: decoupled | replaced: r_sqr envelope | replaced: envelope rewrite | dropped: `sess_comp` | retained |
|---|---|---|---|---|---|---|---|---|
| unit_n7 | TSO | 33,960 | 7,776 | 864 | 0 | 0 | 288 | 25,032 |
| unit_n7 | DSO7 | 93,492 | 27,936 | 288 | 288 | 2,304 | 288 | 62,388 |
| unit_n7 | DSO5 / DSO9 | 89,916 each | 27,936 | 288 | 288 | 2,304 | 0 | 59,100 |
| x0 | TSO | 30,780 | 7,776 | 864 | 0 | 0 | 0 | 22,140 |
| x0 | DSO7 | 90,312 | 27,936 | 288 | 288 | 2,304 | 0 | 59,496 |
| x0 | DSO5 / DSO9 | 89,916 each | 27,936 | 288 | 288 | 2,304 | 0 | 59,100 |

ESSO: 9 `available_e` rows dropped per node. Unit node 7 also drops 3 SoH exp-chain rows and 3 floor rows (its 3 D
rows are retained). No new class was needed. Retained ESSO Δ is 0; the rows touching the fixed slacks move ≤ 1.5e-15,
because the up and down slacks are equal at the certified point.

**Coupling (spec §3.6) at the certified point.**

How the rows are built:
- 504 rows per group-mode, exact equalities, defined in `coupling.json` `definitions`.
- The TSO side is `pc_adn` / `qc_adn`: network.py l.440–441, `is_transmission` branch; MCH l.1241 / 1265,
  pc[adn_load] + `interface_delta_p/q`; the flex legs are fixed at 0 in the ADMM path (SRP l.4266–4300).
- The TSO block has no `pg_adn`. l.444–445 is the DSO else-branch, and the certified TSO block's Expression list
  confirms it.
- Constants: vb = 345 kV on both sides (the DSO reference bus 1 has base_kv 345); s_base = 100 MVA on both sides;
  interface rating 200 MVA; S_ref = 2.5 MVA, so a = 0.2.
- "Normalized" means Boyd units: v as magnitude-equivalent ΔW / (√W_t + √W_d) on the TSO base; pf as ΔMW / rating;
  ess as a·Δ.

| cell | channel | max \|residual\| | max \|residual − mismatch\| | max normalized | L2 normalized | eps_pri | max / eps_pri |
|---|---|---|---|---|---|---|---|
| x0 | v (Wii) | 4.36 kV² | 2.8e-11 | 1.95e-5 | 3.04e-5 (= recorded r_v) | 3.43e-3 | 0.0057 |
| x0 | pf P / Q | 2.15e-2 MW / 3.09e-4 Mvar | 1.3e-14 | 1.07e-4 | 3.26e-4 (= r_pf) | 1.37e-3 | 0.078 |
| x0 | ess (all four rows) | ≤ 7.9e-25 MW | ≤ 9e-41 | 1.6e-25 | — | 7.20e-4 | 2e-22 |
| unit_n7 | v (Wii) | 1.73 kV² | 2.8e-11 | 7.27e-6 | 2.42e-5 (= r_v) | 3.43e-3 | 0.0021 |
| unit_n7 | pf P / Q | 2.48e-2 MW / 3.03e-4 Mvar | 1.3e-14 | 1.24e-4 | 4.76e-4 (= r_pf) | 1.37e-3 | 0.090 |
| unit_n7 | ess P TSO–ESSO / DSO–ESSO | 5.14e-4 / 5.49e-4 MW | 3.6e-17 | 1.10e-4 | 1.7e-4 / 2.1e-4 | 7.90e-4 | 0.139 |
| unit_n7 | ess Q TSO–ESSO / DSO–ESSO | 2.15e-5 / 1.16e-5 Mvar | 2.9e-18 | 4.3e-6 | — | 7.90e-4 | 0.0054 |

The implied TSO–DSO storage mismatch at the unit is 5.49e-4 MW (P) and 9.90e-6 Mvar (Q). The ESS L2 values are
pairwise, so they are not Boyd's r_ess, which uses x_i − z; z is not in the pickles.

**Exports.**
- 48 LP files (24 R + 24 A), each group being TSO + 3 DSO + 3 ESSO + the coupling rows.
- Size: 26,292–27,310 variables, 22,371–23,435 rows, 6.4–6.7 MB each, with a name map.
- The objective is linear: Σ_b w_b·gross_b, with the ESSO contributing 0.
- Every A entry in `lp/index.json` carries the D-1 label "attribution diagnostic, NOT a lower bound of production".
- Two rehearsals produced byte-identical LP files, name maps and vector files for the groups they shared. The
  re-pickled captures are not byte-stable across processes; their content equality is gated by fingerprint instead.

## Not confirmed

- **No relaxation was solved.** Feasibility of the LPs, solver acceptance (Gurobi's recognition of the rotated cones
  written as bilinear inequalities, `NonConvex 0`), and the gap are all unknown.
- **Group decomposition.** Each group LP contains the whole-horizon ESSO models, with coupling only on its own
  (year, day). After relaxation, the ESSO's cross-group rows are the D rows and the cohort rows. My argument that
  the sum of group optima is still a relaxation of the joint problem rests on two observations: D enters no retained
  row, and the cohort capacities are Params. No solve tested it.
- **Boyd r_ess** was not recomputed: the end-of-cycle z is not in the pickles. `state['dual_vars']` and the
  consensus dicts are not in the pickles either. The duals and consensus targets in force at each block's own solve
  are captured as mutable Params in `captures/*/vectors_*.json`.
- **The envelope-row first-order bound** (Blocked item 2) and the bound-violation statistic (Blocked item 5) come
  from scratch reads of the evidence captures, not from the harness. Their formulas are in
  `/tmp/t01b_explore/env_bound.py` and in item 5's description. Neither is preserved in the root.
- **Variant C** was not built (Blocked item 3).
- Where I looked for the `pg_adn` / `pc_adn` question: network.py l.420–446, MCH l.1235–1310, SRP l.4230–4330 and
  l.9396–9480, and the component list of a certified TSO block.
