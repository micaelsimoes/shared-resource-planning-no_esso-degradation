# P5.16 D0 / T0.1c — variants R, A, C on the certified blocks with the §12 rulings applied (zero solves)

Worker, 2026-10-31. Spec: `P5_16_D0_SPEC_v2.md` §2, §3, §11 (D-1…D-5), §12 (sha256 `600fdd6d…` at run time).
Evidence: `data/SRP1/Results/P516D0/t01c/` (untracked data root, 779 MB, 205 files). Hash inventory
`manifest_sha256.json` sha256 **`d29551285683ede902b71188ba6973f33e15d53018f8a77da6e5fd833514afb5`**.
Child exit code **0**: 3,637 gate rows, all pass, 0 stops. Guard 0 / 0, 0 process launches, 0 solver libraries
imported, 0 write-opens in the repository outside the root. Git HEAD at the run: `7324aa9c`. Child wall time 907 s.

## Blocked on Planner

1. **Prediction C2 failed as worded (byte identity at x = 0).** The task lists no stop for it, so the run carried on.
   - **R ≡ A at x = 0.** In all 12 groups the R and A LP files are identical except line 1. Line 1 is Pyomo's
     source-name comment and names the variant (the T0.1b convention). So raw bytes differ and the bodies are
     identical.
   - **C differs from R and A in 45 body lines per x = 0 group, all in the ESSO's 27 available-energy rows:**
     - 27 row-name lines: `d0_available_e_eq_calendar(k)` against `d0_available_e_eq_rated(k)`.
     - 18 coefficient lines, in the in-window pairs (y ≥ y_inv): −φ^age_end against −1 on `es_e_rated_per_unit`.
   - **Why:** §12.3 makes C's row `es_e_available == φ^age_end · es_e_rated_per_unit`. At x = 0,
     `es_e_rated_per_unit` is a free Var. The retained `rated_e_capacity_unit` rows pin it to
     `es_e_investment_fixed = 0`, so the feasible sets and the LB objective coincide, but the files cannot.
   - **Options:**
     - (a) Accept C2 as "semantically identical, body-identical across R and A; C differs only in the ESSO
       available rows (names and in-window coefficients on a Var pinned to 0)". **Recommended.** No re-run.
     - (b) Write C's rows on the investment Param (`E_avail == φ^age · es_e_investment_fixed` in-window). This still
       differs from R's `E_avail == E_rated` form, so it does not buy byte identity. Not recommended.
     - (c) Restate C2 for later tasks as body identity modulo the ESSO available rows.
2. **Committing the evidence.**
   - As instructed, I staged only the harness, the builder and this report.
   - CLAUDE.md requires a report to be committed with its evidence base and hash inventory. The committable small
     artefacts are:
     - `manifest_sha256.json` (32 KB)
     - `predictions.json`
     - `provenance_parent*.json`
     - `run.log` (78 KB)
     - `objective.json` (269 KB)
     - `capacity.json` (506 KB)
     - `identities.json` (1.8 MB)
   - `inventory.json` (72 MB) and the LP / name-map files (≈ 700 MB) are hash-recorded in the manifest.
   - Which of these to add is the Planner's call (T0.1b committed its manifest plus its small artefacts).

## Changed

- **`p516_d0_build_relaxation.py`** (+248 / −12 lines, sha256 `8dc070c8…`). Opt-in extension; every default is
  unchanged.
  - `relax_network_block(..., capacity_mode='calendar', calendar_capacity=..., derived_bounds=...)`.
  - `relax_esso_block(..., calendar_factors=...)` builds the component `d0_available_e_eq_calendar`.
    `_set_esso_mapped_point` and `mapped_point` follow the factor, or the component name, when present.
  - `envelope_first_order_table`: the §12.2 check. Its formula is the T0.1b scratch formula, now in the builder.
  - `CAPACITY_BOUND_CODE_SITES`, the static inventory.
  - `capacity_bound_inventory` and `production_capacity_bounds`. Both call the production
    `configure_shared_ess_operational_state` on a scratch clone, never on the input or on the relaxed clone.
- **Reproducibility of earlier tasks (rehearsals into `/tmp`).**
  - `p516_d0_dry_run.py` reproduces all 7 T0.1a LP files and `name_map.json` byte-identically against the
    committed manifest `04033ff7`. Its predictions are identical and its exit code is the same (2).
  - One dry-run gate differs: "containment: writes outside OUT_DIR" = 3. That check is mtime-based over the
    repository. It caught another agent's concurrent edits to `p516_tperf_a2.py`, `p516_tperf_profile.py` and
    `.idea/workspace.xml`.
  - `p516_d0_t01b.py` with 1 group reproduces all 34 shared files byte-identically (4 LP, 4 name maps, 26 vector
    captures). Its exit code is the same (3).
- **`p516_d0_t01c.py`** (new, sha256 `66057e2d…`).
  - Imports `CELLS`, `RECORD_FILES`, `sha256_file`, `_git` and the event lists from `p516_d0_t01b.py`.
  - Parent and child follow T0.1b: atomic root, so it refuses if the root exists; stdout and stderr merged into
    `run.log`; git provenance before and after; manifest.
  - In the child, the blocking guard and the audit hook are installed first. `predictions.json` is written before
    any load. A 21-item capture-path checklist is asserted before the run and again before writing.
  - Refuses with exit 5 on any input hash mismatch. Checked: the 4 pickles against §2; the ESS params and case
    params against each cell's `evaluation_record`; the T0.1b manifest against `4e5bbd45…`; and the 3 T0.1b files
    read, against that manifest.
  - Rehearsal-only env vars are `P516_D0_T01C_REHEARSAL_DIR` / `_GROUPS`.
- **Launch.** `/opt/anaconda3/envs/opf_env_py311/bin/python p516_d0_t01c.py`, a single process with both streams
  captured by the harness parent.
  - The run (about 15 min) exceeded the tool's 10-min foreground limit, so it ran under the tool's session-owned
    background mode and I polled it. It was not `screen`, `nohup` or shell `&`.
  - The exit code was read from the console log.
- **Staged by filename:** `p516_d0_build_relaxation.py`, `p516_d0_t01c.py`, `P5_16_D0_T01C_REPORT.md`. Nothing
  committed.
- **Tracked status during the run.** It changed, but not because of this run: other agents committed T-perf / C5
  work (HEAD went `0b92e24a` → `7324aa9c` → `d9620092` during the task). My child's audit hook records 0
  repository writes.

## Found

**Instances.** W86 tight-tail cells, configuration s39_D. Gross convention: settlement-excluded, salvage not
subtracted.

| cell | candidate_key | certified cycle | recorded gross (€) | production gross recomputed on the pickles |
|---|---|---|---|---|
| x0 | `8435c718…` | 132 | 653,858,731.5686293 | rel. diff ≤ 1e-8 (gated); 12-group LB-objective sum rel 1.8e-16 (R = A = C) |
| unit_n7 (0.25 MVA / 1 MWh @ 7, 2025) | `db77e154…` | 112 | 653,599,356.2420883 | 12-group sum rel 5.5e-16 (R = A = C) |

**Predictions** (in `predictions.json`, written before any input was loaded):

| | prediction | outcome |
|---|---|---|
| C1 | every retained and envelope row within the restated gate, all block-modes | **HELD.** See C1 detail below. |
| C2 | R, A, C LP files of each x = 0 group byte-identical | **FAILED as worded.** R and A bodies are identical (12 / 12); raw bytes differ in line 1, the variant name. C differs in 45 lines per group, all in the ESSO available rows (Blocked item 1). |
| C3 | Q_cert⁺ − gross = 13,012.57 (x0), 13,032.47 € (unit), to the cent | **HELD.** See C3 detail below. |
| C4 | at the unit, E_A ≤ E_C ≤ E_R in every (cohort, block); E_C/E_R = 0.985^5, ^10, ^15 for the 2025 cohort | **HELD.** See the capacity table below. |
| C5 | the capacity-derived inventory is exactly the two SoC-final slack bounds per network copy, plus the ESSO's | **HELD for value-dependent bounds.** See C5 detail below. |

C1 detail:
- 288 block-modes (2 cells × 12 groups × 3 variants × 4 blocks).
- Retained rows over the gate: 0 of 1,209,360 gated row checks (7,008 capacity-Param rows excluded in R and C at the unit).
- Envelope-rewrite rows over the gate: 0 of 41,472.

C3 detail:
- x0: correction **13,012.566792632126 €**. The second computation path gives 13,012.56679263243.
- unit: correction **13,032.465347571979 €** (path 2: 13,032.465347572272).
- Both equal −`detector_penalty_total` of the record to 1.8e-12 €.
- Every detector slack contribution is ≤ 0; the positive parts are exactly 0.

C5 detail:
- 2 value-dependent bounds per shared-ESS network copy: the ub of `slack_shared_es_soc_final_up/down`. That is
  144 per cell over 72 copies.
- The ESSO carries **0** capacity-dependent bounds.
- The inventory also lists **zero-gated** bounds: `shared_es_pch/pdch/pnet/qnet`, 96 per copy. These depend on
  capacity only through "capacity == 0" and are identical across R, A and C at both cells.

**C1 / §12.1–12.2, restated gates** (per variant, summed over 12 groups × 4 blocks; R, A and C identical except
where noted):

| cell | retained row checks (gated) | over restated gate | max \|Δ\| | max Δ / threshold | old ruling 10.1(ii) failures | Δ > 1e-10 | cones W / transformer (max rel) | same-point Δ |
|---|---|---|---|---|---|---|---|---|
| x0 | 199,836 | 0 | 2.08e-9 | 0.9999999982 | 928 | 111 | 7.3e-16 / 6.7e-16 | 0 exactly |
| unit_n7 (R, C) | 202,116 (+3,504 capacity-Param rows excluded) | 0 | 2.06e-9 | 0.9999999978 | 615 | 73 | 7.1e-16 / 7.2e-16 | 0 exactly |
| unit_n7 (A) | 205,620 | 0 | 2.06e-9 | 0.9999999978 | 615 | 73 | same | 0 exactly |

- **Envelope-rewrite rows (§12.2).**
  - 6,912 rows per cell-variant (192 per DSO block-mode), 0 over bound + 4·eps·Σ|terms|.
  - Maximum Δ:

    | cell | max Δ | first-order bound | Δ / threshold | row |
    |---|---|---|---|---|
    | x0 | 6.110e-9 | 6.113e-9 | 0.99995 | `pij_def[(31,0,0,21)]`, DSO9, 2025 Autumn |
    | unit | 6.161e-9 | 6.164e-9 | 0.99995 | same row |

  - This reproduces the T0.1b scratch figure (6.110e-9 against 6.113e-9) inside the evidence root.
  - Under the superseded ≤ 1e-9 gate, 8 (x0) and 12 (unit) rows would have failed per variant.
- **Mapping offset** (the maximum certified definitional residual δW, per cell and block kind):

  | cell | block | `vmag_sqr_def` | `voltage_product_real_def` | `r_sqr_def` |
  |---|---|---|---|---|
  | x0 | TSO | 4.15e-11 | 3.27e-11 | — |
  | x0 | DSO | 2.21e-11 | 1.71e-12 | 1.83e-11 |
  | unit | TSO | 9.52e-12 | 9.33e-12 | — |
  | unit | DSO | 2.22e-11 | 2.30e-12 | 1.85e-11 |

  Per block, it is in `identities.json` → `groups.<cell>.<group>.mapping_offset`.
- **W-quadratic audit.** 0 of 65,064 retained non-linear rows carry a W variable in a quadratic term. So the
  first-order W bound (linear coefficients only) is first-order complete on the gated rows.
- **ESSO** (both forms, SoH ≡ 1 and calendar):
  - Same-point Δ is 0.
  - Retained Δ is at most 1.48e-15 (unit ESSO7), within the first-order offset of the fixed `slack_es_pnet` (288
    rows per node).
  - The added available-e rows show residual 0 at the mapped point.
  - 0 nonconvex rows.
- **Other gates, all passed:**
  - coupling residuals equal the T0.1b recorded residuals bit for bit (36,288 rows, 0 unequal);
  - objective identity per block and group ≤ 1e-8 (achieved ≤ 2e-16 per group);
  - 0 nonconvex rows in any assembled group;
  - inventory equals the builder on every block-mode;
  - the live-block fingerprints equal T0.1b's `fp_before` (51 / 51 per cell), so the captures are reused by
    reference, with 26 / 26 T0.1b capture files hash-checked;
  - the live blocks are not mutated and the inputs are unchanged at the end.

**Variant C and capacities (§12.3).**
- Inputs:
  - φ_cal = 0.985, from `SRP1_ESS_Params.json` (`39106f93…`, equal to each cell's recorded ESS-params hash) and
    equal on every planning ESS object.
  - Years map `{2025: 5, 2030: 5, 2035: 5}` from `SRP1.json` (`61a794a7…`), equal to the planning object's.
  - `available_energy_soh_point = end`, ageing enabled.
  - age_end(y_inv, y) = Σ_{k = y_inv..y} num_years_k, and 0 for y < y_inv.
- Unit cell, node-7 network copies (TSO e1 and DSO7 identical), pu on 100 MVA, s = 0.0025 in all three variants.
  Energy capacity **E_ (MWh-equivalent pu)** and the `slack_shared_es_soc_final` ub (lb 0):

| block | E_R | E_A carried | E_C | ub R | ub A (certified) | ub C |
|---|---|---|---|---|---|---|
| 2025 | 0.01 | 0.0088038878 | 0.0092721650 | 5.1000e-4 | 4.5019e-4 | 4.7361e-4 |
| 2030 | 0.01 | 0.0077984018 | 0.0085973044 | 5.1000e-4 | 3.9992e-4 | 4.3987e-4 |
| 2035 | 0.01 | 0.0069999999 | 0.0079715625 | 5.1000e-4 | 3.6000e-4 | 4.0858e-4 |

- **C4, ESSO cohort 2025 at node 7:**

  | block | E_A (certified) | E_C | E_R | E_C / E_R | 0.985^n |
  |---|---|---|---|---|---|
  | 2025 | 0.88039 | 0.92722 | 1 | 0.92722 | 0.985^5 |
  | 2030 | 0.77984 | 0.85973 | 1 | 0.85973 | 0.985^10 |
  | 2035 | 0.70000 | 0.79716 | 1 | 0.79716 | 0.985^15 |

  The ratios equal 0.985^5k to ≤ 1e-15.
- x = 0: all three capacity vectors are 0 on every copy and every derived bound is [0, 0], so nothing differs.

**Capacity-derived bounds (§12.4).**
- **Method: the production function.** For every block (96 per cell) a scratch clone was configured with
  `configure_shared_ess_operational_state` at three probes, (0, 0) and two different positive capacities, and every
  Var bound was diffed.
- **Static code sites inventoried** (`CAPACITY_BOUND_CODE_SITES`, in `capacity.json`):
  - MCH l.1153–1169: the slack ub, value-dependent.
  - MCH l.1186–1196: zero-gated bounds.
  - MCH l.1137–1150 and l.1088: zero-gated fixings.
  - network.py l.393–407: capacity-independent.
  - ESSO, shared_energy_storage_data.py l.463–468, 548–550, 567–568, 689, 1585: no capacity-value bounds.
  - SRP l.4270–4306, 4657–4669, 5408–5414: interface / reference, not capacity.
- **ESSO probe.** The production `_configure_esso_cohort_state` at (0, 0), (0.25, 1), (0.5, 2) per cohort: 0 bound
  changes. It changes fixings and row activity only.
- **Values used for R and C.** They are read from the same production function called with the variant capacity on
  a scratch clone, then copied onto the relaxed clone. They equal the arithmetic e·0.05 + 1e-5 exactly (0
  mismatches).
- The function was not called on the relaxed clone, because it also resets the `shared_es_soc` values (24 per copy)
  and so would move the mapped point.
- **What the call changes.** At the variant capacity it changes no bound beyond the two slack families, no fixed
  flag and no row activity, in 96 / 96 blocks.
- **Fidelity.** Called at the carried capacity, it reproduces the certified bounds in 96 / 96 blocks.
- **Bounds changed.** 48 bounds changed against the certified values in R and 48 in C (unit cell: 12 groups ×
  {TSO e1, DSO7} × up/down). None changed at x0.
- **Against T0.1b.** The unit R LPs differ from T0.1b's R in exactly those 4 bound lines per group. The A LPs are
  body-identical to T0.1b's (24 / 24), as are the x0 R LPs (12 / 12).
- **D-1 labelling.** Every A entry in `lp/index.json` (24 / 24) carries "attribution diagnostic, NOT a lower bound
  of production".

**Q_cert⁺ (§12.5).** Gross convention, € (per group and per block in `objective.json` → `q_cert_plus`):

| cell | gross (Σ blocks) | correction | Q_cert⁺ | voltage slack | flex P day-balance | shared-ESS day-balance |
|---|---|---|---|---|---|---|
| x0 | 653,858,731.57 | 13,012.57 | 653,871,744.14 | 12,057.44 | 955.13 | 0 |
| unit_n7 | 653,599,356.24 | 13,032.47 | 653,612,388.71 | 12,057.44 | 955.13 | 19.90 |

- **How the slacks were identified.** The slack families are the variables of the six production detector
  functions: `voltage_slack_penalty`, `node_balance_…`, `branch_flow_…`, `flexibility_p_day_balance_…`,
  `local_ess_day_balance_…` and `shared_ess_day_balance_slack_penalty`. These are the `_get_local_detector_components`
  lineage, which mirrors `_get_local_slack_penalty_components`.
- **Checks on the identification:**
  - their certified value equals production `_get_local_detector_components` per block to ≤ 1e-12;
  - every one has lb = 0;
  - none appears in a quadratic or nonlinear gross term;
  - the gross carries each with the detector coefficient (0 mismatches).
- **Two independent paths.** Path 1 projects the detector expressions; path 2 projects the gross expression's own
  linear terms. They agree to ≤ 3e-10 € per cell.
- **Where the correction comes from.** 58,176 negative voltage slacks, 2,304 flex slacks and (unit only) 48
  shared-ESS slacks. 0 slacks exceed their ub.
- **Between the cells,** the correction differs by 19.90 € (the shared-ESS day-balance term).
- Per group, the correction falls from 1,203 / 1,205 € (2025 Spring) to 976 / 978 € (2035).

**Unexpected findings.**
- **The restated retained-row gate is met with essentially no margin on single-W-term rows, by construction.** For a
  row with one W term (e.g. `voltage_magnitude_lower_cons`, `node_balance_p` at a TSO node), Δ = |c·δW|, which is
  the first-order bound itself. The maximum ratio 0.9999999982 is that case. On those rows the gate can only fail by
  more than 4·eps·Σ|terms| of rounding, so it is informative mainly on multi-term rows.
- **The network copies' carried available energy differs slightly from the ESSO's certified value.**
  - In 2025: 0.0088038878 pu carried against 0.880394190 / 100 = 0.0088039419 pu in the ESSO.
  - The difference is 5.4e-8 pu (relative 6e-6). 2030 shows a similar small offset.
  - My unverified hypothesis: the copies carry the ESSO iterate from the capacity update before the certified
    cycle's ESSO solve.
  - C4 holds on both sets of values.

## Not confirmed

- **No relaxation was solved.** LP feasibility, solver acceptance and the gaps are unknown, as at T0.1b.
- **The A-capacity lag hypothesis** (Unexpected findings). I did not trace the order of `update_*_coordination_model`
  against the ESSO solve in the certified cycle.
- **The ESSO capacity-bound probe** exercised `_configure_esso_cohort_state` directly. `_update_model_with_candidate_solution`
  was not run end to end, and it also sets the investment Params and the H3 share rows (Params and rows, not
  bounds).
- **Scope of the static bound-site search:** `setub` / `setlb` / `bounds=` / `.fix(` in
  `model_construction_helpers.py`, `network.py`, `shared_resources_planning.py` and `shared_energy_storage_data.py`.
  Separately, `setub` / `setlb` was searched in every non-`p5*` top-level `.py`: `debug.py` and
  `uncoordinated_benchmark.py` also contain them, and neither is on a shared-ESS capacity path. The empirical probe
  covers only what `configure_shared_ess_operational_state` and `_configure_esso_cohort_state` do.
- **Boyd r_ess** is not recomputed, as at T0.1b: z is not in the pickles.
