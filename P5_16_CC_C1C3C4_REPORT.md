# P5.16 convention change — C1 + C3 + C4, gates G-CC-1, G-CC-2, G-CC-3, G-CC-5 (spec v4) — Worker report

Spec: `P5_16_CONVENTION_CHANGE_SPEC_v4.md`; authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 88 (R1.1–R1.5) and its
Amendment (C2 closed). **Zero solves**: the guard (`SolveProfileGuard(permitted=())`) was verified at 0 in every run.
Nothing is committed. Every file named below is staged by filename.

## Blocked on Planner

1. **STOP: G-CC-3 slack safety FAILED** (a named stop: Addendum 88 R1.4 (ii), the task's stop list).
   - The salvage marginal per unit of `es_pnet`, in ESSO objective units, peaks at **5.84** for a 2037 cohort
     (discharge, Spring day, SoH_T at its calendar-only upper bound; **5.75** at the replayed nominal point).
   - The bar is 1e-3 × `PENALTY_ESSO_SLACK` = **1.0**, so the peak is **5.84×** the bar. It is still 0.58 % of the penalty.
   - Per cohort (worst entry): 2028 **1.55**, 2031 **2.93**, 2034 **4.40**, 2037 **5.84**. Every 2028+ cohort fails.
   - Definition used: one MW of a cohort's pch/pdch at **one representative period**, which is the unit that `es_pnet`
     and the slack pair are measured in. Its salvage marginal is κ · c · SoH_T, where c already carries the period's
     weight n · nd · dt / eff (3 y × 92 d × 1 h / 0.96 for the worst entry). That is why one MW-period is worth
     ≈ 1,394 € of salvage for a 2037 cohort.
   - The **per-MWh reading** (κ · |∂S/∂TP|) gives 0.005 (2028) to 0.020 (2037) objective units per MWh. That would
     pass the bar by 50×. The expert's "slack safety holds" matches this reading, not the per-variable one.
   - The ruling is not mine. Options:
     - (a) Raise `PENALTY_ESSO_SLACK` before freeze, as R1.4 (ii) says. It needs ≥ 5.85e3 for the 1e-3 bar; 1e4
       would give a margin of 1.7. This is a production-configuration change.
     - (b) The expert rules that the bar is per MWh of throughput.
     - (c) The expert re-sets the bar.
   - Not acted on. Nothing was changed.
2. **G-CC-1 fixture check: 112 of 139 unpickle**, against a stop item ("a fixture fails to unpickle"). I judged this
   **not caused by the edit** and continued with G-CC-2/5/3, which are zero-solve and independent of it. Please
   confirm or overrule.
   - 22 FrozenSMOPF pickles fail on `model_construction_helpers.s_bounds`. That symbol was deleted in P5.4-B
     `1e86d40e` and is already recorded in `P5_16_D0_T01A_REPORT.md` and Addendum 88 D-4 ("FrozenSMOPF loadability
     repair: later, as housekeeping"). `model_construction_helpers.py` is untouched.
   - 5 named anchors are **absent on this machine**: `P512R/cycle21_pre_setup/snapshot.pkl`,
     `P512R/cycle21_prepared/snapshot.pkl`, the two `P512R/production_snapshots/...cycle7.pkl`, and
     `FrozenSMOPF/failure_TSO_case9_2025_Spring_cycle1.pkl`.
   - The same 134 present paths were unpickled with the **HEAD** production modules: 112 ok at HEAD and 112 ok after
     the edit, with **identical outcomes and no newly failing fixture**. The HEAD modules were exported read-only with
     `git archive HEAD` to /tmp, and a module `__file__` assertion confirmed they were the ones loaded.
   - The literal check `fixtures_all_unpickle` stays **FAIL** in the record and is not relaxed.
   - The list has 139 entries, not W183c's 115: W183c's own `declared_fixture_list()` was run on this machine, and it
     globs `FrozenSMOPF/**`.
3. **Wording of the € parity identity (G-CC-3 (i))**, for the Planner to settle.
   - Read literally ("a 1 € change of S moves the ESSO objective exactly as a 1 € change of a block's PV cost moves
     that block's ADMM objective"), it does **not** hold. ESSO: κ = 4.125e-3 per €. Block: 1/σ = 1.068e-8 per €.
     The two differ by exactly a = σ/w_med = 386,259, by design (F1: κ = a/σ).
   - The a-normalised identity (κ/a = 1/σ = d block ADMM objective / d PV €) **holds to 8.7e-13** (ESSO), 3.9e-12
     (DSO7) and 8.7e-13 (TSO).
   - I report the second as the gate.
4. **Sign wording in G-CC-3**: "∂objective/∂throughput = κ·∂S/∂throughput, negative".
   - Measured: ∂S/∂TP < 0, as predicted. ∂objective/∂TP = κ·∂(−S)/∂TP > 0.
   - The identity is checked as ∂obj/∂TP = κ·∂(salvage_credit_obj)/∂TP = −κ·∂S/∂TP.

## Changed

Production diff: 3 files, +183 / −6. All changes are switch-controlled; the off path constructs nothing new.

- `shared_energy_storage_parameters.py`
  - `SalvageValueParameters.in_objective`, default **False**. It is read strictly as a JSON boolean, and
    in_objective without enabled raises.
  - `SharedEnergyStorageParameters.available_energy_soh_point`, default **None** (not declared). It is read from a
    top-level key and validated as 'end' or 'mid'.
- `shared_energy_storage_data.py`
  - (C3) `SharedEnergyStorageData.read_parameters_from_file` assigns `available_energy_soh_point` only when the file
    declares it.
  - (C1) `_build_subproblem`: with the switch on it builds
    - Param `salvage_objective_kappa` (mutable, initialised **0**);
    - Expression `salvage_credit_obj`;
    - objective = `feasibility_penalty + κ · salvage_credit_obj`;
    - a plain dict `_salvage_throughput_coefficients` (not a component).

    The else branch is the original statement verbatim. The switch is read with `getattr(..., False)` because
    pickled parameter objects lack the attribute.
  - New `_build_salvage_credit_objective_expression`:
    − Σ disc_T · f · c^E(y_inv) · rlf · [floor·E + (1−floor)·E·(SoH[y_inv,T] − soh_min)/(1 − soh_min)].
    - E is the Param `es_e_investment_fixed`, so the term is linear in the one Var `es_soh_per_unit_cumul[y_inv, T]`.
    - It is guarded by rlf > 0 and T inside the cohort window (same tcal_norm as the rated and degradation rows).
    - Under 'mid' it uses the end-of-block terminal SoH.
    - It never touches the bilinear `e_available` row.
    - f and floor are read from the salvage parameters (today 1.0 and 0.0), so the value equals −`salvage_value` at
      feasible points (measured: rel 0.0 at the 2028 unit).
  - `_get_esso_complementarity_diagnostics` (diagnostic only): **uses the effective marginal**, as Addendum 88
    R1.4 (iii) requires, not a caveat.
    - When the term is present, the bound is Σ 2·μ/(s_obj·(a_ch + a_dch)) over the same active cohort-periods, with
      a = ε + κ·c·SoH_T. The ε-only bound is kept as `spurious_throughput_bound_eps_only`.
    - Without the term, the record is byte-for-byte the old one (old code path, same keys).
- `shared_resources_planning.py` (κ plumbing only)
  - `update_shared_energy_storage_model_to_admm(..., objective_scale=None)`.
  - When the ESSO carries κ, it adds Param `admm_common_objective_scale` and sets
    κ = `admm_esso_al_scale / admm_common_objective_scale` (a = 1 if the AL scale is not applied). It raises if σ is
    missing.
  - The single production caller (`_run_operational_planning`) now passes `objective_scale=objective_scale`.

New files:
- `data/SRP1/SRP1_5x3y_1x1.json` (`f916e0c6…`).
  - Produced by `p515_s44_scale_measurement.derive_case('paper', {num_market_scenarios: 1,
    num_operation_scenarios: 1})`, with only `SharedEnergyStorage.params_file` re-pointed.
  - Years {2025, 2028, 2031, 2034, 2037: 3}; NumMarketScenarios 1; operation scenarios 1;
    `PlanningParameters` = `SRP1_params.json` (σ_fixed 93,635,360, esso_al_scale sigma_over_median_block_weight).
- `data/SRP1/SharedESS/SRP1_5x3y_ESS_Params.json` (`08ade261…`).
  - A copy of `SRP1_ESS_Params.json` plus `"available_energy_soh_point": "mid"` and `"salvage_value": {"in_objective":
    true}`. All other values are today's: f = 1.0, floor 0.0, basis EXPECTED_INSTALLATION_ENERGY_COST, ageing baseline,
    solver.
- `p516_cc_gates.py` (`933ba245…`, the harness for all four runs).
  - It reuses W183's `build_config`, `Capture`, `block_state`, `compare_blocks`, W183c's `gate1_fixtures` (child
    process), the committed `p516_d0_dry_run.py` (executed into /tmp, not edited), and production functions throughout.
  - Its `/tmp` rehearsal root was used for smoke runs; those outputs are deleted.

Untouched (git status clean, checked inside every run): `SRP1.json`, `SRP1_ESS_Params.json`, `SRP1_ESS.xlsx`,
`model_construction_helpers.py`, network models, `P516D0/`, `p516_d0_*.py`.

## Found (prediction → outcome)

### G-CC-1 — switch-off identity, SRP1 (`P516CC/g_cc1/post_off`, manifest `f8080629…`)

- **Reference**: built from **HEAD `f9336940`** *before* the edit, by the committed, unmodified
  `p515_s53_w183_nl_digest.py` (`--expect-production-commit HEAD`, configs x0_A_1x and unit_A_1x, ALL_OK).
  - Stored in `g_cc1/ref_head` (digest JSON `94f4ed06…`, manifest `bf4436b1…`).
- **Bitwise**: HELD.
  - Scope: both configurations × both builders (sequential 51 blocks, parallel 36).
  - Every block's NL is identical in all four files (plain, labelled, .row, .col), and every component digest is
    identical.
  - No κ / `salvage_credit_obj` was constructed.
- **Secondary**: also bitwise identical to the committed Studio reference `w183_nl_post_w183c_off` (production
  `6f43479e`).
- **Fixtures**: see Blocked on Planner item 2. Literal FAIL (112/139); no new failure against HEAD modules.
- **T0.1a dry run**: HELD.
  - The committed manifest is `04033ff7…`.
  - All 7 LP files are byte-identical to it.
  - The child exit code is 2 both then and now (the recorded P-a/P-b failures).
  - A pre-edit rehearsal had also reproduced them.

### G-CC-2 — switches on in memory, SRP1 files unchanged (`P516CC/g_cc2/x0_unit_on`, manifest `cc71a153…`)

There are two arms: the **full convention** (C1 + C3 'mid') and **C1 only**. Both are compared with the G-CC-1 run.

| prediction | outcome |
|---|---|
| network blocks bitwise identical | **HELD** (both arms, both configs, both builders) |
| x = 0 ESSO value-inert (S ≡ 0) | **HELD**: the standard repn of `salvage_credit_obj` has no free variable, constant 0, at every node, so the objective is identical as a function. Solution identity is inferred from that; nothing was solved |
| x = 0 ESSO NL bitwise (expert, recorded as prediction) | **HELD for C1**: the C1-only arm's ESSO plain NL is bitwise identical at x = 0, **and at the unit** (κ = 0 is folded by the writer). With C3 on it differs at x = 0 as well: 'mid' rewrites `available_e_capacity_unit` (J coefficients) at x = 0 too. That is a C3 consequence; the spec's prediction concerned C1 |
| unit-cell diff restricted to C1 / C2 expression / C3 | **HELD** |

How the diff was enumerated:
- The comparison is name-mapped and column-order-invariant (W183's own row analysis assumes identical columns;
  'mid' reorders them).
- **Components**: `Param:salvage_objective_kappa`, `Expression:salvage_credit_obj`, `Objective:objective` (C1);
  `Expression:salvage_value`, `Expression:salvage_credit` (the reporting expression's existing 'mid' branch,
  i.e. the C2 salvage-expression class); `Constraint:available_e_capacity_unit` (C3).
- **NL entries differing**: only `C|available_e_capacity_unit` and `J|available_e_capacity_unit`. No objective entry
  differs, because κ = 0 at the would-be solve.
- **Header** (unit node 7): nonlinear-variable counts `12 0 0 → 11 0 0` and Jacobian nonzeros `3251 → 3253`.

### G-CC-5 — five-block instance (`P516CC/g_cc5/g0`, manifest `fe6765a6…`)

All checks HELD:
- **Load**: the case loads with no write into `data/SRP1/Diagrams`.
- **Blocks**: 80 network + 3 ESSO blocks were built at x = 0 and at the 2028 unit (node 7, 0.25 MVA / 1.0 MWh) through
  `run_operational_planning` with every optimize stubbed.
- **Memory**: peak RSS **1.60 GiB**.
- **tcal_norm** = 5, with 15/3 = 5.0 exact.
- **Weights**: all 80 equal 3 · days · 1.02^{−(y−2025)} to ≤ 1e-15.
- **Terminal discount**: 0.7430147299885189 = 1.02^{−15}.
- **rlf**: [0.0, 0.2, 0.4, 0.6, 0.8].
- **Switches**: 'mid' and in_objective were read from the file; the salvage defaults are frozen.
- **κ**: 0 at the initialisation stub.
- **Market scenarios**: `NumMarketScenarios` = 1 (case file); `prob_market_scenarios` = [0.35, 0.55, 0.10] (ESS
  workbook 'Scenarios' sheet). They are independent, as the spec notes.
- **c^E(y)**, probability-weighted, €/MWh: 2025 253,878; 2028 226,471; 2031 205,100; 2034 195,805; 2037 186,511.
- **G0**: the x = 0 NL and block digests of the 83 blocks are recorded in `g_cc_results.json` → `g0_x0_digests`
  (sha256 of that map, sorted JSON: `c1c7cd45…`). The ESSO plain NL is `13d18b4d…` at all three nodes.
- **σ band**: prediction recorded (σ_computed/σ_fixed inside [1/3, 3], point ≈ 0.6). **Not testable here**: it needs
  the initialisation solve.

### G-CC-3 — salvage calculus, five-block 2028 unit (`P516CC/g_cc3/unit2028`, manifest `cfb5ae45…`)

Holding:
- **κ = 0 at the initialisation**: HELD.
  - κ read inside the initialisation ESSO stub is 0.0 at all three nodes.
  - In the source, `create_shared_energy_storage_model` precedes `update_shared_energy_storage_model_to_admm`.
  - The Param is initialised 0.00.
- **κ**: 4.125137066901099e-3.
  - It equals `admm_esso_al_scale / admm_common_objective_scale` (386,258.69 / 93,635,360) exactly.
  - It equals the hand replica a/σ with w_med = 242.416 (median of the 80 hand-computed weights) to 0.0 relative.
  - It equals 1/w_med exactly.
- **∂objective/∂throughput = κ·∂(credit)/∂throughput**: HELD.
  - FD vs κ·FD agree to ≤ 2.3e-9, and FD vs AD to ≤ 8.7e-9 (all ≤ 1e-6).
  - AD vs closed form agree to ≤ 2.5e-16.
  - The ADMM objective carries the term (FD of `admm_objective` = FD of the base objective, rel 0.0).
  - Method: AD through the rows with pyomo `differentiate`; FD through the throughput aggregate with the SoH chain
    replayed from the model's own equality rows.

∂S/∂TP per MWh of cell throughput (horizon), all negative except 2025:

| cohort | ∂S/∂TP €/MWh | expert prediction (factor 2) | Advisor order of magnitude |
|---|---|---|---|
| 2025 | **0 exactly** (outside the term; FD change 0.0) | 0 — HELD | — |
| 2028 | **−1.227** | ≈ 1 — HELD | ≈ 2 |
| 2031 | −2.362 | — | — |
| 2034 | −3.594 | — | — |
| 2037 | **−4.850** | ≈ 5 — HELD | ≈ 8 |

The value is the same for every block of a cohort's window (spread ≤ 1.1e-8). It is measured at the replayed nominal
point: SoH_T 0.784 (2028) to 0.941 (2037).

- **€ parity**: HELD in the a-normalised form (Blocked on Planner item 3).
  - DSO block case33_2|2028|Summer via `flex_p_down[0,0,0,0]`; TSO block via `pg[0,0,0,0]`. Each variable is outside
    the AL terms.
  - Unset ADMM request and dual Params were set to 0 for evaluation only (counts recorded).
- **Slack safety**: **FAILED**, 5.84 against a bar of 1.0. This is a STOP (Blocked on Planner item 1).
- **Complementarity diagnostic**: uses ε + κ·c·SoH_T.
  - On the built model, the effective pair marginal ranges from 2e-5 (2025 cohort) to 11.11 and matches the hand
    computation exactly.
  - No IPOPT log exists, so both bounds are None (by design, no default substituted).

## Not confirmed

- **σ-band assertion** on the new instance: it needs the initialisation solve; not run.
- **Absent anchors**: the CLAUDE.md-named cycle-21 anchor and four other named anchors do not exist on this machine,
  so their unpickling after the edit is unverified. Where I looked: the W183c declared list and `os.path.isfile`.
- **Solution identity at x = 0** under the new convention is inferred from the identity of the objective as a
  function and from value-inertness; no solve was made.
- **Bitwise identity on the Studio** was not run there. The G-CC-1 secondary comparison does match a Studio-built
  reference.
- **Other harness callers** of `update_shared_energy_storage_model_to_admm` (10 `p5*.py` scripts) do not pass
  `objective_scale`. They keep working with the switch off, but would raise with it on.
  - `p515_s34_zero_solve_checks.py` compares the function's source with a pre-module and will now see a different
    source. Not run.
- **Campaign hook**: whether `p515_s44_campaign_harness`'s ESS-params pin and model-variant path accept the new
  params file was not checked (out of scope; it reads the `ageing` block only).

## Evidence (staged by filename; NL/items bulk hash-recorded in the manifests, not staged)

- `P516CC/g_cc1/ref_head/{w183_nl_digest.json, manifest_sha256.json, launch.log}`
- `P516CC/g_cc1/post_off/{g_cc_results.json, manifest_sha256.json, predictions.json, fixtures_unpickle.json,
  fixtures_unpickle_head_modules.json, fixtures_head_paths.json, t01a_dry_run_launch.log, t01a_rehearsal_run.log,
  t01a_rehearsal_manifest_sha256.json}`
- `P516CC/g_cc2/x0_unit_on/{g_cc_results.json, manifest_sha256.json, predictions.json}`
- `P516CC/g_cc5/g0/{g_cc_results.json, manifest_sha256.json, predictions.json, planning_read_stdout.log}`
- `P516CC/g_cc3/unit2028/{g_cc_results.json, manifest_sha256.json, predictions.json, planning_read_stdout.log}`
- `P516CC/g_cc{1_post_off,2_x0_unit_on,5_g0,3_unit2028}_launch.log` (stdout + stderr, attached runs)
