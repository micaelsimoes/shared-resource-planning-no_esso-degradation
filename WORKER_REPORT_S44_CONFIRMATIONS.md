# Worker Report — P5.15 Addendum 26 zero-solve confirmations (spec v14)

**Worker, 2026-09-19.** Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 26; `STEP4_DFO_METHOD.md` §1.2–1.3 and §9;
frozen spec `data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json` (sha256 `e4500e27…22c9b`), key
`addendum26_confirmations_zero_solve`.

**Zero solves, enforced.** `SolveProfileGuard(permitted=())` was installed before any production import and stayed
installed for the whole run. `verify(0)` passed exactly: `permitted_solve 0, permitted_exec 0, blocked_solve 0,
blocked_exec 0`. The run used the script as committed at `96fa8cc8` (script sha256 `95aba947…7c52c2`, recorded in every
output file). It ran attached, with stdout and stderr captured to `run_console.log`, and exited with code 0.

## Task received

Four zero-solve confirmations: (1) the provenance of the cost file; (2) I(x) and the budget slack at the paper's plan,
at C\* and at the two Addendum 26 lattice reference points, with units, and whether `budget` / `max_capacity` are active
in the model; (3) whether the oracle can evaluate x = 0 at every node; (4) whether the ESSO capacity multipliers are
available without extra solves. Resumed from a previous Worker whose session ended; see "Unexpected findings" 6.

## Files inspected

`shared_energy_storage_data.py` (master `_build_master_problem` :291-391, ESSO `_build_subproblem`, `_optimize`,
cohort gating, reader :1641-1705), `shared_resources_planning.py` (`_run_operational_planning`, `create_*_model`,
S_ref helpers :4406-4431, first-stage check :1851-1886, σ resolution :3365-3490, ESSO ADMM update :4744-4815),
`network.py`, `network_data.py` (`_get_sensitivities`, candidate unit conversion :2903-2924),
`model_construction_helpers.py` (network shared-ESS zero gating), `p56a_oracle.py`, `p515_g_g1_g4_admm_gates.py`
(`_construct_arm_planning`, `run_admm_arm`, the ESSO capture), `p514_n_instrumented_cstar.py`, `p58_rescale.py`,
`p513_solve_profile_guard.py`, `data/SRP1/SRP1.json`, `SRP1_params.json`, `SharedESS/SRP1_ESS_Params.json`, both git
blobs of `SRP1_ESS.xlsx`, and the S42 exact-fix artifacts and manifest, G3F_r2 and P514D artifacts,
`REVISION_CONTEXT.md` (Track D1).

## Files modified / created

- `p515_s44_addendum26_confirmations.py` (new; commits `acb74b5b`, `c3960b68`, `96fa8cc8`).
- Outputs in `data/SRP1/Results/P515S44/confirmations/`: `item1_cost_file_provenance.json`,
  `item2_investment_cost_and_budget_slack.json`, `item3_zero_investment_evaluability.json`,
  `item4_esso_capacity_multipliers.json`, `production_stdout_capture.log`, `run_console.log`, `run_exit_code.txt`,
  and `manifest_sha256.json`, which hashes the other 7 files.
- A side effect of calling the oracle's own construction: the empty directory
  `data/SRP1/Results/P56A/evals/p515s44_confirm_x0_20260919T081435Z/logs` (created by `O.fresh_planning`; untracked)
  and the empty `confirmations/item3_arm_construct/results/`.
- No production file, case file, `p515_g_g1_g4_admm_gates.py`, or harness-Worker file was touched.

## Changes made

The script only. Every file:line and every verdict in the outputs is computed at run time (source lines are located by
string search, and the verdicts are derived from git and model queries). None is written in advance as prose.

## Commands / experiments run

```
python p515_s44_addendum26_confirmations.py > data/SRP1/Results/P515S44/confirmations/run_console.log 2>&1   # exit 0
python p515_s44_addendum26_confirmations.py --manifest
```

Before this run I did two dry runs with all outputs redirected to the session scratchpad (`OUT_DIR` and
`p56a_oracle.WORK_DIR` patched). The first exposed a trace artifact, now fixed in `c3960b68` (see item 3, step 8b);
nothing from the dry runs is in the repository.

---

## Results

### 1. Cost file provenance — **the committed file is NOT the latest cost update in the repository**

- **Committed file.** `data/SRP1/SharedESS/SRP1_ESS.xlsx`: sha256
  **`1458147446e9b70190465f42cef761af9cfd8b209e91035d858e2e63941d1414`**, 25,590 bytes, git blob `5ef61f5d`. The
  working copy is byte-identical to HEAD and `git status` is clean. The xlsx's own core properties give
  `modified 2025-06-12 15:58:18`.
- **History reachable from HEAD** (rename-tracked). Only renames and copies since 2025-06-12:
  `072b1310` 2026-02-03 (R100, "Spaces removed"), `784346d7` 2025-12-15 (R100 from `OP3/…/CS7_ESS.xlsx`), `a4a75c98`
  2025-12-11 (C100). The last content changes in that lineage are `ed5895ee` / `d90733fe`, 2025-06-12, "Test.
  Planning", on `CS7_ESS.xlsx`.
- **All refs (local and remote).** This path has exactly **2 distinct blobs** across refs:
  `5ef61f5d` (HEAD and 18 other refs) and **`8a7e3744`** (sha256 `e17bd5887e1d0738005ae17c3144593527081c9a0776e19cfaa50aafefe39cd6`,
  xlsx `modified 2026-07-29 12:47:08`). The second blob is carried only by `paper_revisions` and
  `origin/paper_revisions`. It was introduced by **`7ce1d1ab` 2026-07-29 18:49 "Costs updated."**, which is **not an
  ancestor of HEAD**. Its parent blob is exactly HEAD's blob. The merge base of HEAD and `paper_revisions` is `285cbb42`
  (2026-07-28), which carries `5ef61f5d`: this branch forked one day before the cost update.
- **What that commit changed** (cell-by-cell comparison of the two blobs):
  - `Investment Cost, Energy`: every one of the 105 numeric cells is multiplied by exactly **1.25** (ratio
    1.25 ± 2e-16).
  - Unit label of both NREL sheets: `[$/KWh]` → `[$/kW]` (1 cell each).
  - `Investment Cost, Power`, `Scenarios`, conversion rate and cost breakdown: identical.
- **Unit-cost content** as production reads it (committed file, € per MVA / € per MWh; case years):

  | | c^S S1 / S2 / S3 | c^E S1 / S2 / S3 |
  |---|---|---|
  | 2025 | 214,171.33 / 267,528.72 / 342,165.62 | 169,706.26 / 211,985.89 / 271,127.09 |
  | 2030 | 168,720.15 / 224,250.29 / 278,097.04 | 133,691.41 / 177,692.69 / 220,360.07 |
  | 2035 | 153,929.27 / 206,958.26 / 268,576.72 | 121,971.33 / 163,990.73 / 212,816.31 |

  Scenario weights **0.35 / 0.55 / 0.10** are stored in the xlsx, sheet `Scenarios`, row 1, columns C–E. The discount
  factor **0.02** is not in the xlsx: it is `data/SRP1/SRP1.json` `"DiscountFactor"` (line 15), assigned at
  `shared_resources_planning.py:8008`.

  Expected unit costs (weighted and discounted):

  | year | committed c^S €/MVA | committed c^E €/MWh | `8a7e3744` c^E €/MWh |
  |---|---|---|---|
  | 2025 | 256,317.32 | 203,102.14 | 253,877.68 |
  | 2030 | 190,384.09 | 150,857.60 | 188,572.00 |
  | 2035 | 159,606.93 | 126,470.23 | 158,087.78 |

  The "≈ €256k/MVA, ≈ €203k/MWh" quoted in STEP4 §1.2, and the budget-binding points in Addendum 26, match the
  **committed** blob. They do not match `8a7e3744`.
- **Documents identifying "the corrected file".** Searched: `git grep` of all tracked `*.md`, `*.py`, `*.json`,
  `*.txt` at HEAD, and a full read of `PLANNER_BRIEF_2026-09-13.md`, `STEP4_DFO_METHOD.md`, `EXPERT_REVIEW.md`,
  `EXPERT_REVIEW_2_ACTION_PLAN.md` and `CLAUDE_LEGACY_BACKUP.md` in the working tree. The tokens searched are listed in
  `item1…json`. **None names a commit, blob or hash for the corrected file.** The only related hit:
  `P512R/.../initial_repository.json` records sha256 `14581474…` for this path, so P5.12-R and later used the
  committed blob. Not searched: anything outside this repository, reflog-only or dangling objects, and stashes (the
  stash list is empty).
- **Verdict (scoped).** The committed file on this branch is the pre-`7ce1d1ab` version. The repository holds a later
  "Costs updated." version, with energy costs × 1.25, on `paper_revisions`. Whether `7ce1d1ab` is the correction the
  author refers to cannot be established from the repository. **Provenance of the committed file as "the corrected
  one" is not supported.**

### 2. I(x) and budget slack (€, x in MVA / MWh, all 2025; B = `budget` = 1,000,000 €)

I(x) is `model.investment_cost` of the production Benders master. The master is built by
`shared_ess_data.build_master_problem()`, the candidate is loaded with the production
`load_candidate_solution_into_master_model`, and I(x) is read with `pe.value` (no solve). The slack B − I(x) is read
from the master's own budget row (`ub − body`). Cross-check: `p56a_oracle.investment_cost` agrees within ≤ 9.3e-10 € at
every point. Salvage is not in I(x): the objective is `investment_cost + alpha`
(`shared_energy_storage_data.py:382`).

| point | I(x), committed file | B − I(x) | I/B | first-stage check | I(x), blob `8a7e3744` | B − I(x) |
|---|---|---|---|---|---|---|
| **paper's plan**, n7 1.62 / 3.24 | **1,073,285.01** | **−73,285.01** | 1.073 | infeasible (budget) | 1,237,797.74 | −237,797.74 |
| **C\***, n5/7/9 0.96875 / 3.875 | **3,105,984.62** (1,035,328.21 per node) | **−2,105,984.62** | 3.106 | infeasible (budget) | 3,696,250.22 | −2,696,250.22 |
| **lattice C\***, all 1.0 / 4.0 | **3,206,177.68** (1,068,725.89 per node) | **−2,206,177.68** | 3.206 | infeasible (budget) | 3,815,484.10 | −2,815,484.10 |
| **lattice plan**, n7 1.5 / 3.0 | **993,782.41** | **+6,217.59** | 0.994 | feasible | 1,146,109.02 | −146,109.02 |
| (context) node7_empty | 2,070,656.42 | −1,070,656.42 | 2.071 | infeasible (budget) | 2,464,166.82 | −1,464,166.82 |
| (context) two_c_star, 1.9375 / 7.75 | 6,211,969.25 | −5,211,969.25 | 6.212 | infeasible: budget **and** max_capacity (7.75 > 5.0, all 9 rows) | 7,392,500.45 | −6,392,500.45 |
| x = 0 | 0 | +1,000,000 | 0 | feasible | 0 | +1,000,000 |

**Is `budget` / `max_capacity` active?**

- **Benders master: yes, as active rows.**
  - `energy_storage_maximum_capacity` (`shared_energy_storage_data.py:338`): 9 active rows.
  - `energy_storage_power_to_energy_factor` (:344): 18 active rows.
  - `energy_storage_investment`, the budget (:360): 1 active row.
  - The master is built only at `shared_resources_planning.py:349`, inside `_run_planning_problem`.
- **Where production reads them.** An AST scan of 26 production modules finds `budget` / `max_capacity` accessed only
  by `_build_master_problem`, `_check_candidate_first_stage_feasibility` (:1851-1886),
  `_build_positive_bootstrap_candidate`, `_validate_local_sensitivities_with_finite_differences` and the parameter
  readers.
- **Not on the oracle's path.** Every call site of those readers is inside `_run_planning_problem` or
  `_complete_missing_sensitivities_with_probe`. **None is in `_run_operational_planning`, the oracle's path.** In the
  oracle harness the budget is overridden to **5.0e6** (`p515_g_g1_g4_admm_gates.py:1116`,
  `N.BUDGET`, `p514_n_instrumented_cstar.py:43`), and that override is inert because nothing on the path reads it.
- **`max_capacity` bounds energy, not power.** Its row variables are `['es_e_rated']`: E_rated (cohort-accumulated,
  every year) ≤ 5.0 **MWh**. There is no power cap. The case file currently has E/P between 2 and 10.

### 3. x = 0 at every node — construction, S_ref, consensus, ESSO, evaluator and captures all work at zero solves

Traced with the oracle's own construction (`_construct_arm_planning`, D arm `apply_rho=False`,
`investment_map = {5, 7, 9: (0, 0)}`), followed by production's initialization sequence from
`_run_operational_planning`. Each agent's `.optimize` was replaced, on that planning instance only, by an interceptor
that records the call and returns "no result". The declared intercept count was DSO 3 / TSO 1 / ESSO init 1 / ESSO
coordination 1; the observed count matched it **exactly**.

| step | outcome |
|---|---|
| 1. construction | OK; `_is_zero_investment_candidate` True; total capacity 0 everywhere |
| 2. first-stage check | feasible |
| 3. `create_admm_variables` | OK; ESS consensus entries for nodes 5, 7, 9 |
| 4–5. DSO (3 nodes) and TSO (12 blocks) build | OK; `shared_es_s_rated_fixed = 0`; every `sess_*` row deactivated (converter, sum-limit, SoC def/limits/final); `expected_shared_ess_p` fixed at 0 |
| 6. ESSO build + candidate + TSO request | OK; all 3 cohorts inactive at every node; 0 free pch/pdch; `energy_storage_limits`, degradation, throughput and H3 rows all inactive; `rated_s/e_capacity_unit` 6+6 rows active with RHS 0; 576 `operation_agg` rows active |
| 8a. ADMM objective preparation | OK |
| 8b. `_compute_common_admm_objective_scale` | **raises** on unsolved blocks. This is a trace artifact: it reads the initialization's *solved* objective values. The trace continued with the case-file fixed σ = 93,635,360 (declared substitution). |
| 8c. ESSO AL scale (227,210.997), `update_*_to_admm` (TSO/DSO under `p58_rescale.patched_admm_objectives`), ESSO AL objective, consensus initialization | OK; price-taker branch not taken (`standalone`) |
| 9. S_ref | fixed **2.5 MVA** at every node and year, from `_admm_shared_ess_reference_mva`; TSO 0.025 pu on 100 MVA; installed S seen = 0. Had the reference been None, x = 0 would normalize by the 0.10 MVA floor (the Track D1 confound): **not the case under D** |
| 10. ESSO structure | 288 converter-circle rows per node active; `es_pnet` / `es_qnet` free; RHS `es_s_rated[y]` pinned to 0 through active equalities |
| 11–12. capacities published (all 0); ESSO coordination update | OK; `p_req` / `dual_p_req` set (0) |
| 13. legacy and Boyd residuals | OK; ESS channel r = s = 0, eps_pri = eps_dual = 7.2e-4, 5,184 entries, passes trivially |
| 14. captures | OK, nothing raises: EFC/day max **None** (guarded `not rated`); degradation fraction 0; complementarity 0 over 0 periods; available capacity and SoH 0; salvage 0; slack violation 0; harness `_capture_esso_solve` writes 0 records per node; zL pre-check correctly not marked done |

**What x = 0 leaves open (solve-level, not decidable at zero solves):**

1. **The ESSO converter circle at zero rating.** `es_pnet² + es_qnet² ≤ es_s_rated[y]²`
   (`shared_energy_storage_data.py:731`) with the RHS pinned to 0 gives the feasible set {(0, 0)} for 288 (y, d, p)
   per node. The row's gradient vanishes at that point, so MFCQ fails.
   - Historical evidence that IPOPT handles it: G3-full (`P515G3F_r2`, and the same map in `P515G3F_B`) ran with nodes
     5 and 9 at zero. It produced **81/81 successful ESSO rounds at each zero node**, with diagnostics present for every
     round and 0 "Shared ESS … did not converge" lines.
   - That evidence is from 2026-09-14 code and configuration, **not D**.
2. **The fixed-σ calibration assertion.** σ_computed must be within ×3 of 93,635,360, and σ_computed needs the x = 0
   initialization solves.
3. **Report fields that are undefined at x = 0.** Per-candidate EFC/day and terminal SoH are None or "not applicable".
   The §2.5 report format must allow for that.

**Scoped evidence scan.** I scanned all 75 `data/SRP1/Results/**/g_*.json`. **No all-nodes-zero run exists under the
current harness.** The only all-zero evaluation found is Track D1 cell 2 (`P514D/d1_cell2.json`, 2026-09-12, pre-P5.15
code): 70 cycles, recourse 820,746,762.46, 0 local failures. `p515_s44_campaign_harness*.py` did not exist at run time.

### 4. ESSO capacity multipliers — **available at zero extra solves (raw); their meaning as ∂Q/∂x is not established**

- **Rows linking ESSO operation to installed capacity:**
  - `rated_s_capacity_unit` :551 and `rated_e_capacity_unit` :552: `es_*_rated_per_unit[y_inv, y] ==
    es_*_investment_fixed[y_inv]`, one per cohort-year.
  - The degradation D-row :654, which has E_inv as a coefficient.
  - Downstream rows: `rated_s_capacity` :563, `energy_storage_limits` :692, the converter circle :731 and
    `available_e_capacity_unit` :584.
- **Duals are imported on every ESSO solve.** `model.dual = pe.Suffix(IMPORT_EXPORT)` is declared at :805 (with
  `ipopt_zL/zU_out` at :801) and loaded by `model.solutions.load_from(result)` at :1296.
- **Committed evidence (no solve), `esso_models_s39_D.pkl`** (tracked, sha256 `6a840eb0…`, matches the S42 manifest;
  C\* under D, terminal cycle 139):
  - Every **active** row of the ESSO capacity-related families at every node has a dual: 1,200 dual entries per node.
  - Node 7, `rated_s_capacity_unit` [1..6]: −76.76, −50.26, −20.48 (cohort 2025, y = 2025 / 2030 / 2035); −50.28,
    −20.50 (cohort 2030); −20.50 (cohort 2035).
  - `rated_e_capacity_unit`: ≈ −7.0e-11 for the invested 2025 cohort, and −0.01818 for the empty cohorts.
  - Nodes 5 and 9 are similar (−80.04 / −80.01 for cohort 2025 at y = 2025).
- **Units and sign.** These are in the ESSO's `admm_objective` units per MVA or MWh. That objective is the feasibility
  penalty plus `al_scale_esso` (227,211) × the AL terms, with ρ_esso 0.0225 at termination. It is **not € of Q**, and
  the sign convention is not established.
- **The energy multiplier is structurally ≈ 0 in the ESSO.** The ESSO has no SoC state, so the operational value of E
  lives in the network SoC rows.
- **Network side, `certified_models.pkl`** (untracked, hash-recorded, sha256 `4e3d8bfc…a797`, matches the manifest):
  duals are present on 100% of active `sess_converter_capability`, `sess_active_sum_limit` and `sess_soc_*` rows in all
  12 TSO and 36 DSO blocks.
- **Per-cycle capture already records some ESSO duals.** `esso_capture/s39_D/node*_cycle139.jsonl` (untracked,
  hash-recorded) records duals for 4 ESSO families every cycle (`_ESSO_DUAL_FAMILIES`,
  `p515_g_g1_g4_admm_gates.py:252`), but **not** `rated_*_capacity_unit`.
- **The production Benders sensitivity channel is retired.** `network_data.py:2851`: `_get_sensitivities` returns
  "unavailable".
- **What a zero-extra-solve capture would take.** A new read site at the terminal or certified cycle, for example a
  `post_run_hook` that reads:
  - `model.dual` on `rated_s/e_capacity_unit` and the D-row, per node and cohort, from the ESSO models; and
  - if the network share is wanted, the `sess_*` row duals × ∂row/∂(`shared_es_*_rated_fixed`) in the 48 blocks.

  Each would then be converted from its own scaled objective (σ, block weight, p58 effective scale, AL scale) to €.
  This needs no solve. Whether the sum is a usable ∂Q/∂x at a nonconvex certified consensus point is **a question for
  the Planner and Advisor**. I have not validated it; validating it with a finite-difference check would need solves.

## Validation

- Guard `verify(0)` passed; the interceptor count matched its declaration exactly.
- Master I(x) and the independent transcription agree to ≤ 9.3e-10 €.
- The master's budget-row slack equals B − I(x) at every point.
- The S42 artifact hashes match their manifest.
- The script sha256 is recorded in all outputs and equals the committed file.
- Distinctions:
  - Code executes: yes.
  - x = 0 construction and captures: demonstrated at zero solves.
  - x = 0 solve behaviour under D: **not demonstrated**.
  - Multiplier availability: demonstrated.
  - Multiplier validity as a gradient: **not demonstrated**.

## Unexpected findings

1. **Cost file (item 1).** The newer `paper_revisions` version raises energy costs by 25%. Under it, even the lattice
   plan (1.5 / 3.0) exceeds the budget, by 146,109 €. The Addendum 26 binding-point arithmetic was done with the
   committed (older) file. The same `paper_revisions` commits also change `SRP1.json` (years 2025–2037 in steps of 3; a
   single 365-day day type), which is context only.
2. **Budget.** The paper's plan exceeds B = 1e6 by 73,285 € (7.3%) and C\* by a factor of 3.1, both on the committed
   file.
3. **`max_capacity` is an energy cap (≤ 5.0 MWh), not a power cap.** STEP4 §1.2 states P ≤ 5 MVA and E ≤ 4 P^max.
   Under production's definition two_c_star (7.75 MWh) violates it at all 9 node-years. The selection run does not
   check it, because the oracle path never calls the first-stage check.
4. **Budget override.** A DFO master that calls `_check_candidate_first_stage_feasibility` on a planning object built by
   `_construct_arm_planning` would read **budget 5e6** (the `N.BUDGET` override), not 1e6.
5. **σ calibration assertion.** It is candidate-dependent: it compares against the solved initialization. At x = 0 and
   at large plans it is unverified.
6. **The previous Worker's draft.** It left an **uncommitted, never-run** draft at this filename (sha256
   `82490759…0ebf`; a copy is kept outside the repository in the session scratchpad). I rewrote it rather than
   committing it, because it:
   - asserted results as prose before any run (for example "every checked row carries a non-None dual" and
     "3 of the 4 candidates infeasible");
   - declared other branches "not searched", when a read-only `git log --all` search finds `7ce1d1ab`;
   - read a pre-reformulation artifact (`P515S32_run/esso_models_baseline.pkl`) for item 4.

## Remaining issues

- Which cost file is authoritative needs the author. If it is `8a7e3744`, bringing it onto this branch is a case-file
  change I am not authorized to make.
- x = 0 under D is unverified at solve level: the degenerate ESSO circle and the σ assertion remain open.
- Converting the multipliers to € and validating them as ∂Q/∂x has not been done.

## Questions for Planner

1. Is `7ce1d1ab` ("Costs updated.", energy × 1.25, on `paper_revisions`) the correction the author means? If so, every
   Addendum 26 budget statement changes: for example, the lattice plan becomes infeasible.
2. Should `max_capacity` be read as production defines it (E ≤ 5 MWh), or as STEP4 §1.2 states it (P ≤ 5 MVA)?
3. For §5.6: should a terminal-cycle multiplier read site (ESSO-only, or ESSO plus network) be specified, with a
   separate solve-based validation?

## Commits

- Script: `acb74b5b`, `c3960b68`, `96fa8cc8`.
- Outputs, manifest and this report: the commit that adds this file.
