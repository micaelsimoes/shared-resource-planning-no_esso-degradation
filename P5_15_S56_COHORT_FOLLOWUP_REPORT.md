# P5.15 S56 — ESSO cohort follow-up: report (spec v1 `287c0da6`)

Worktree `.claude/worktrees/s56-cohort-followup`, branch `s56-cohort-followup` (from `c65201d1`). Not merged.
Test run 2026-11-23 15:52:53–17:16:31 UTC (5,018 s), base commit `437efb48`, interpreter
`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`, IPOPT `/usr/local/bin/ipopt` (3.14.18, ma57) through
`NLP_SOLVER_PATH`. No campaign or ADMM run. 13 ESSO solves in total (2 before-tree, 11 after-tree), all node 7, each
under an armed `SolveProfileGuard` that was verified exactly at its declared count.

## Blocked on Planner

1. **T8 failed as frozen. The cause is the row's tolerance, not IPOPT's termination.** Measured L = 2.475e-4 MWh/day
   against a predicted ≤ 1e-6 (0.8508 before the change). The complementarity row holds at its bound:
   pch·pdch/S_c² = 1.00006e-4 against a tolerance of 1e-4. L is unchanged when μ_final changes 3.1-fold. The row is
   `(pch/S_c)(pdch/S_c) ≤ 1e-4`, the normalised `sess_comp` form frozen in spec v1. At the H_B operating point
   (pch ≈ 0.122 MW in charge hours 1–4, S_c = 0.25) it permits pdch ≤ 5.1e-5 MW per hour, so L ≥ 2.0e-4 MWh/day
   whenever the dump incentive is present. Under that row, the ≤ 1e-6 prediction could not have held. Options, for
   the Planner or Expert (a formulation question, so not a Worker decision):
   (a) accept the leak as the relaxation's designed slack, the same tolerance the network `sess_comp` uses, and
       re-state the criterion as "row at or inside its tolerance";
   (b) a tighter cohort-row tolerance. This changes the formulation and needs a new spec and an NL-identity re-check
       of the multi-cohort builds only;
   (c) an exact complementarity. This changes the formulation.
   Cost of (a): none. Cost of (b): one build + 2 ms solves.
   I recommend (a) unless the manuscript needs within-cohort simultaneity below 1e-6.
2. **Two items in the task's scope are not built**, because spec v1 forbids harness edits ("not_permitted: harness
   edits"):
   - the reading of the Q slack, the per-cohort closure and the detectors into the exit certificates (E6);
   - `cohort_formulation` in the harness keys and records.

   The gate function `model_construction_helpers.shared_ess_network_closure_slack_max` exists but is not wired. These
   items need a spec v2 that names the harnesses.
3. **T9: 14 changes the prediction did not list** (5 of 5 predicted changes occurred). Ten have an exact explanation
   (below). The other four are small and not diagnosed. They are recorded as failed against the frozen T9 "unchanged
   elsewhere" clause, and nothing is re-scoped.
4. **Evidence size.** `cohort_followup_v1/run/` is committed in full: 3,400 files, 71 MB, of which 63 MB are the
   3,360 gzipped NL files that T1's bitwise claim rests on. The spec lists them as outputs, and
   `P515S55/w183_nl_reference_pre` is precedent. If the repository should not carry 63 MB, the alternative is to
   hash-record them only: their sha256 values are already in `cohort_followup_tests.json` and in the manifest.
5. **Merge.** Merging into `feature/derivative-free-planning` needs the author's OK.

## Changed

- **Production commit** (this commit, on `s56-cohort-followup`). The five files are bitwise the ones the run used:
  their sha256 equals the run's `trees.after.overlay_sha256`, and they equal `tree_after/` byte for byte.
  - `model_construction_helpers.py` `b5888548…`
  - `network_data.py` `ac3e7122…`
  - `shared_energy_storage_data.py` `0c3f032e…`
  - `shared_energy_storage_parameters.py` `8aba50cd…`
  - `shared_resources_planning.py` `c6e874d2…`
- **Spec** v1 `287c0da6` (predecessor none), committed before the run in `437efb48`. There is no spec v2: no new solve
  was needed (see Found, T8).
- **Evidence**: `data/SRP1/Results/P515S56/cohort_followup_v1/run/`, written by the run, plus `manifest_sha256.json`
  (3,408 entries: run files, launch logs, script, diagnostics module, five production files). Written after the run
  by the script's own `--manifest` mode. Also `run_launch_stdout.log` and `run_launch_stderr.log` (0 bytes).
- **T8 diagnosis** (zero-solve, new): `p515_s56_cohort_followup_t8_diag.py` writes
  `cohort_followup_v1/t8_diagnosis/t8_diagnosis.json` and `manifest_sha256.json`, with captured stdout/stderr. The
  formulas are in the script docstring. Guard `permitted=()`, verified at 0 solves and 0 launches.
  - Disclosure: the first execution counted the T9 closure-slack variables from the detector block-days and got 4
    instead of 32 for one-block tests. That output was moved aside unmodified to the scratchpad
    (`…/scratchpad/t8_diagnosis_first_run_flawed_t9_count/`). The script was fixed to count from the test plan,
    cross-checked against T3's measured 16 closure rows, and re-run to the vacated path. All non-T9 content is
    identical between the two executions.
  - The diagnosis was done after viewing the data. It is exploratory and has no recorded predictions.

### Parameter key and defaults

- One key, already present since `8c9fd6ec`: the optional top-level `cohort_formulation` of the shared-ESS parameters
  JSON (`SRP1_ESS_Params.json`). It is read strictly.
- Admissible values:
  - `legacy_h3`: the default, used when the key is absent. It is absent from every committed input; the pre-launch
    check confirmed that.
  - `independent_units`
  - **`independent_units_all_node_years`**: new. The single-cohort A/B option, off by default.
- There is no other new key. The network flag `shared_ess_cohort_gross_flow` is a runtime attribute set per candidate,
  not a parameter. It is a plain Python attribute and never reaches an NL file.

### Which rows are built on which node-years

| key | ESSO, node-year with ≥ 2 active cohorts | ESSO, 1 active cohort | network `sess_comp` of the unit (TSO + DSO blocks) |
|---|---|---|---|
| `legacy_h3` | today's H3 rows; no new row | today | active (today) |
| `independent_units` | per cohort-period: SoC, window (2), converter circle, **sum `pch+pdch ≤ s_c`**, **complementarity `(pch/S_c)(pdch/S_c) ≤ 1e-4`**; per cohort-day: **exact closure `e_T = 0.5·E_av`** (closure-slack Vars no longer constructed); per block-period: aggregate Q row + Q-slack pair; aggregate circle deactivated | today (new rows built deactivated, Vars fixed) | **deactivated** on ≥ 2-cohort node-years only; every other storage row unchanged |
| `independent_units_all_node_years` | as `independent_units` | **as the multi-cohort construction** (threshold 1) | as `independent_units` (the switch stays at ≥ 2 cohorts) |

- Efficiency-equality gate: under either corrected key, the ESSO build raises `ValueError` unless every cohort at the
  node has the same η_ch and the same η_dch.
- Unwired, not deleted: the 8c9fd6ec closure-slack Vars and their objective term. The `hasattr`-guarded configure
  branch is kept for models built before this change, and every retired symbol is retained (T2).

## Found

Each T is stated against its frozen prediction. Source: `run/cohort_followup_tests.json`. The verdict reads T1, T2, T3,
T4_T5, T6, T7 and T10 True, and children_ok True; every child exited with rc 0.

- **T1 — held. Default path bitwise unchanged.** Every compared file was identical:
  - ESSO legacy before vs after: 96/96.
  - ESSO legacy determinism (before vs before_rerun): 96/96.
  - Explicit `legacy_h3` key vs absent key: 96/96.
  - `independent_units` on single-cohort and x0 builds, before vs after: 84/84; vs legacy: 84/84.
  - Network legacy before vs after: 480/480. Determinism: 480/480. Key vs default: 480/480.
  - Network `independent_units` on single-cohort builds: 432/432 and 432/432.

  Informational, as predicted: under `independent_units` on p2025_2030, the ESSO differs in the 4 node-7 cells
  (8/12 identical), and the network differs in exactly the 16 TSO + DSO7 blocks of 2030 and 2035, by exactly the
  `sess_comp` rows.
- **T2 — held. The fixtures still load.**
  - 118 declared fixtures load under both trees (118 present, 118 ok each); no outcome differs.
  - `hasattr` on fresh after-tree builds: both corrected keys carry no `slack_es_soc_final_per_unit_*`. Legacy builds
    carry none of the seven new row families. Network blocks carry no retired component.
  - Retained branch: a model built by the before-tree code under `independent_units`, with the closure-slack Vars
    present (pickle `850efb6d…`), was updated by the after-tree candidate update without raising.
  - Retained symbols: `sess_comp_rule`, `sess_soc_final_rule`, `sess_phi_limits_*` and `sess_*_sensitivities` are all
    present, as are the four `_esso_cohort_*` helpers.
- **T3 — held.**
  - SRP1, plan 2025 + 2030. ESSO complementarity rows by block 2025/2030/2035: 0/192/192; sum rows the same.
    Network `sess_comp` at node 7 under `independent_units`: 24/0/0. Under legacy: 24/24/24. No other unit or row
    family changed, and reconfiguration is idempotent.
  - Five-block 3×3, plan 2025 + 2031. ESSO complementarity and sum rows 0/0/192/192/192. `sess_comp` 24/24/0/0/0.
- **T4 / T5 — held.**
  - T4, against network rows with `sess_comp` off: violation 6.9e-18 (≤ 1e-12). Per-cohort sample feasibility
    4.4e-16 (≤ 1e-12). Transfer fraction 1.0 (≥ 0.9; 400/400 samples).
  - T5, `sess_comp` on: closure residual − λX = 5.8e-16 (≤ 1e-12).
- **T6 — held; all 17 predictions.**
  - P hour 1 = 0. P hour 2 = 0.0214025 MW (predicted 0.0214). X = 0.28968 MWh.
  - Loss 0.0207604 MWh.
  - Legacy closure slack needed: 0.0207604 MWh, TSO and DSO7.
  - Network (a) slack 0. ESSO cohort-row violation 2.8e-17.
- **T7 — held.** Raising η_ch, then η_dch, of the 2030 cohort by 0.01 makes the build raise under `independent_units`
  and under `independent_units_all_node_years`. It does not raise under `legacy_h3`, nor after the value is restored.
- **T8 — FAILED as frozen.** Within-cohort L max after = **2.475e-4 MWh/day** (2025 cohort, 2030|Spring) against
  **≤ 1e-6**. T8b (two cohorts) = 2.475e-4 against ≤ 1e-6: also failed. The before-tree baseline reproduces
  cohort_esso v2 at 0.8508163061725996 (held; tolerance 1e-6). T8b before = 0.8508163 (informational). L fell by a
  factor of 3,438. **Diagnosis: genuine simultaneous charge and discharge at the complementarity row's bound, not an
  interior-point barrier residual.** Evidence, all from saved outputs (`t8_diagnosis.json`):
  - *Row form and tolerance.*
    - The row is `(pch_c/S_c)(pdch_c/S_c) ≤ ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`, with S_c = 0.25 MVA. That is
      pch·pdch ≤ 6.25e-6 MW².
    - IPOPT runs with `tol` 1e-10 and `acceptable_tol` 1e-9 (remedy-(h) overrides). Its default `bound_relax_factor`
      appears as "Variable bound violation 1.0e-8".
  - *The row is at its bound.*
    - Row body max = 1.0000843e-4, which exceeds the tolerance by 8.4e-9 (inside IPOPT's 1e-8 bound relaxation).
    - At the production diagnostic's argmax (2030|Spring, period 3), pch = 0.122420 MW and pdch = 5.1057e-5 MW, so
      pch·pdch/S_c² = **1.000058e-4**.
    - On all 8 multi-cohort block-days, the charge-hour reconstruction from the stored-energy trajectory gives
      1.000058e-4 to 1.000084e-4. That reconstruction assumes the max-min hour is a max-charge hour, and is checked
      directly only at the argmax.
    - A barrier residual of an inactive row would leave the body below the bound. Here it sits at the bound to
      58–84 parts per million on every block-day.
  - *Independent of μ.*
    - μ_final: 3.278e-11 for T8, 1.051e-11 for T8b, a ratio of 3.12.
    - The leak is the same in both: max hourly min 5.10567138e-5 vs 5.10567139e-5 (ratio 0.9999999996); L ratio
      0.99999982.
    - A barrier residual scales with μ_final, as in production's barrier identity
      min = μ/(s_obj·(a_ch+a_dch)). A leak set by the tolerance does not.
  - *IPOPT final state, T8.* Optimal in 56 iterations. lg μ = −11.0. Complementarity 3.28e-11 scaled (3.28e-10
    unscaled), against `tol` 1e-10. Constraint violation 2.2e-13. Bound violation 1.0e-8. IPOPT's "complementarity"
    is the primal–dual barrier measure, not the model's complementarity row.
  - *Barrier floor in the same solve.* Cohorts with no incentive have L = −2.393e-7 MWh/day, which is
    −9.97e-9 MW × 24 h: variables at the relaxed bound −1e-8. The 2025 cohort's leak is 1,034 times that floor in
    magnitude.
  - *Against the legacy "barrier identity" leak of 1.03e-4 MWh (v2 t_iv_peak):*
    - That leak has pch = pdch = 9.128e-6 MW at its argmax (symmetric) and is spread over the day: 23.6
      hour-equivalents (L / max-hourly-min).
    - Its row body is 1.2e-5, which is 12% of the tolerance (not binding). After the change it is still there, at
      1.014e-4 MWh.
    - The T8 leak is asymmetric (pch/pdch = 2,398) and concentrated: 4.85 hour-equivalents on 2030 days, 4.69 on 2035
      days. That matches the 4 charge hours 1–4 of the request profile, and the same 4-hour structure as the
      0.85 MWh/day dump before the change (4.03 / 3.75 hour-equivalents).
    - The two leaks have similar daily magnitudes (2.4× apart) but different mechanisms.
- **T9 — predicted changes 5/5 occurred. "Unchanged elsewhere" failed for 14 quantities.** The v2 predictions
  re-applied hold for 19 of 20; the one that fails is `t_iv_peak_within_cohort`, which v2 also failed ("mis-scaled").
  All 11 after-tree solves were optimal with exactly one solve each.
  - Predicted and changed (t_v_tv_on):
    - `admm_objective` −147.5658 → −147.2438.
    - `feasibility_penalty` −0.009778 → −0.009448.
    - Throughput of the 2025 cohort in the 2035 block: 2.5167 → 1.0683 MWh/day.
    - Within-cohort L: 0.8508 → 2.475e-4.
    - Inter-cohort gross transfer max: 0.8508 → 0.1497 MWh/day.
  - Not predicted, **explained exactly (10)**: `feasibility_penalty` of t_ii_a_indep, t_ii_b, t_iii_indep,
    t_iv_control, t_iv_peak, t_v_tv_off_oldest and t_v_tv_off_newest, and `admm_objective` of the two TV-off tests.
    Each change is +n × 1e3 × 1.0e-8:
    - n = 32, 32, 16, 32, 32, 40 and 40 closure-slack Vars in v2;
    - Δ = 3.19997e-4, 3.19997e-4, 1.59999e-4, 3.19997e-4, 3.19997e-4, 3.99996e-4 and 3.99996e-4.

    The implied per-Var value is −9.99991e-9 in every case. That equals the P slack's relaxed lower bound in the same
    solves (−9.999909e-9). In v2, each unwired closure-slack Var sat at its bound-relaxed −1e-8 and was priced at
    `PENALTY_SHARED_ESS_BALANCE` = 1e3. Removing them (D2) moves the penalty by 1e-5 per Var. The frozen reason
    ("v2 closure slacks were ~0, so values move by less than the rule") missed this: the rule's 1e-6 is smaller than
    32 × 1e-5.
  - Not predicted, **not diagnosed (4)**. All are small:
    - t_ii_b pre-charged cohort stored-energy rise: 0.128304 → 0.128065 MWh (−2.39e-4). Delivered power is unchanged
      at 0.5 MW.
    - t_iii_indep minimum window margin: 0.0072179 → 0.0071837 MWh (−3.4e-5).
    - t_iv_peak: within-cohort L −1.9e-6 MWh; inter-cohort transfer +1.66e-6 MWh. Inside the spec's declared
      "leak-level" uncertainty for t_iv_peak.
    - t_v_tv_on `tracking_max_abs_mw`: 0.02683 → 0.02834 MW. This is the T8 case, where the binding row changes the
      solution. It was not on the predicted list.
- **T10 — held; 30 of 30 inventory checks.**
  - Reference unit, `independent_units_all_node_years`: per active cohort-block 96 SoC, 192 window, 4 closure,
    96 converter, 96 sum and 96 complementarity rows. Per block, 96 Q rows and a free Q-slack pair. The aggregate
    circle has 0 of 96 rows active. The network `sess_comp` stays at 24 (no flag).
  - Under `independent_units` the reference unit equals legacy (no new rows active), which is the default-off check.

## Unexpected findings

- **The H_B dump still has a partial route.** After the change, on multi-cohort days with TV on:
  - inter-cohort gross transfer is 0.116–0.150 MWh/day, with conversion losses of 0.0083–0.0107 MWh/day;
  - before the change these were 0.30–0.85 and 0.021–0.061.

  E2 permits inter-cohort transfer and no row restricts it. Whether this transfer is the dump incentive moved to the
  inter-cohort route, or ordinary dispatch, is not confirmed.
- **Some network build paths skip the gross-flow sync.** `uncoordinated_benchmark.py` (`build_dso_arm_models`,
  `build_tso_arm_model`) calls `NetworkData.update_model_with_candidate_solution` without
  `_sync_shared_ess_cohort_gross_flow`. Under a corrected key those models silently keep `sess_comp` (today's model),
  unless a stale attribute from an earlier candidate exists; in that case the staleness guard raises if the capacities
  differ. No test exercised this. Every call site in `shared_resources_planning.py` lies after the sync inside
  `_run_operational_planning`, or inside the no-coordination benchmark, which syncs too.
- **The production diagnostic does not flag this leak.** Its aggregate spurious-throughput check gives a bound of
  9.44e-3 MWh and a measurement of 3.87e-3 MWh for T8, so the leak passes as barrier-level. A per-pair check
  (row body against tolerance) is what shows it.

## Not confirmed

- **Per-hour min(pch, pdch) profile.** It was not preserved. The run recorded only the per-cohort block-day sum, the
  hourly max and the solver's argmax record; the models and solutions were not serialised. Looked in: `child_solves_*.json`
  (detectors, solve logs) and the scratch `work_solves_*` directories (IPOPT logs only). The 4.85 hour-equivalents are
  consistent with a binding row in hours 1–4 plus about 4e-5 MWh elsewhere, but the split is not observed.
- **Row multipliers.** Not recorded. "Binding" is inferred from the body at its bound and from μ-independence, not
  from a positive dual. Two ms solves under a spec v2 would settle this, together with the per-hour profile: the T8
  repeat with dual import, and the same with the row tolerance at 1e-6.
- **The incentive behind the dump** (salvage κ term, H_B) is taken from cohort_esso v2. It was not re-derived here.
- **The four undiagnosed T9 changes.** A plausible reading is a shift of the barrier's interior point on a degenerate
  optimal face once the slacks were removed and the rows added. It is not tested.
- **Scratchpad state.** Nothing newer than the run's end (17:16) exists in the worktree or the cohort scratch
  directory. The previous Worker's last note ("a rehearsal … before freezing") matches the rehearsal disclosed in spec
  v1, which ran before the 15:52 freeze (`reh2_*`, 15:46–15:47). No unfrozen spec or script was found or adopted.
  `scratchpad/tables.md` and `scratchpad/dry/` (17:05–17:07) belong to the step-4 post-diagnostics, not to this
  task. They were not used.
