# P5.12-K -- NO-SOLVE KKT / warm-start forensic report

This forensic diagnoses the consistency and initial residual structure of the frozen replay fixture. It does not by itself explain why cycle 21 fails while cycle 20 succeeds.

## 0. Zero-solver evidence

- `OptSolver.solve` calls: 0
- `SystemCallSolver._execute_command` calls: 0
- zero confirmed: True

## 1. Step 0 -- frozen-state equality

- all frozen-input file hashes match: True
- semantic digest cycle21_pre_setup: match=True
- semantic digest cycle21_prepared: match=True
- semantic digest cycle20_prepared: match=True
- checkpoint state digest match: True
- checkpoint block digest match: True

### Mapping-hash finding

The task-given "mapping" hash equals manifest.json's internal export.mapping_sha256 field, which is the harness's digest() (compact-JSON, atom-encoded) content hash of the in-memory {mapping, not_exported_variables} record -- NOT the sha256 of the on-disk *_mapping.json file bytes (that is sha(), a different hash space, computed over the pretty-printed dump()). original_mapping.json, reload_mapping.json and armA_prepared_mapping.json are byte-identical to each other (all sha256 b39cd878...), and export.mapping_sha256 is identical across the cycle20_checkpoint block, cycle21_prepared and Arm A, confirming the SAME symbol<->variable-name correspondence and the same not_exported_variables identity set across all three states. This is not a data-integrity failure.

### S20 vs S21 variable-value equality

- n common vars: 10756, value diffs: 25, bound diffs: 0, fixed diffs: 0
- **STOP-CONDITION TRIGGERED: var values differ.** argmax={'name': 'shared_es_soc[0,0,0,13]', 'S20': 2.749961901839369e-05, 'S21': 8.945159532174192e-05, 'delta': 6.195197630334823e-05}
- top differing variables (all `shared_es_soc`/`shared_es_e_rated`):
  - shared_es_soc[0,0,0,13]: S20=2.749961901839369e-05 S21=8.945159532174192e-05 delta=6.195197630334823e-05
  - shared_es_soc[0,0,0,19]: S20=0.00015012668389283762 S21=8.945159532174192e-05 delta=-6.06750885710957e-05
  - shared_es_soc[0,0,0,18]: S20=0.00014981743244070055 S21=8.945159532174192e-05 delta=-6.0365837118958634e-05
  - shared_es_soc[0,0,0,12]: S20=3.0062754874218894e-05 S21=8.945159532174192e-05 delta=5.9388840447523026e-05
  - shared_es_soc[0,0,0,17]: S20=0.00014837338218727804 S21=8.945159532174192e-05 delta=-5.892178686553612e-05
  - shared_es_soc[0,0,0,11]: S20=3.109534670302164e-05 S21=8.945159532174192e-05 delta=5.835624861872028e-05
  - shared_es_soc[0,0,0,16]: S20=0.00014578751909235952 S21=8.945159532174192e-05 delta=-5.63359237706176e-05
  - shared_es_soc[0,0,0,10]: S20=3.5079233175475396e-05 S21=8.945159532174192e-05 delta=5.437236214626652e-05
  - shared_es_soc[0,0,0,9]: S20=3.559059010972211e-05 S21=8.945159532174192e-05 delta=5.3861005212019806e-05
  - shared_es_soc[0,0,0,8]: S20=3.770537905347023e-05 S21=8.945159532174192e-05 delta=5.1746216268271685e-05

### S20 vs S21 suffix equality (dual, ipopt_zL_out, ipopt_zU_out)

- dual: n_common_keys=7826, n_diffs=0, only_S20=0, only_S21=0
- ipopt_zL_out: n_common_keys=7460, n_diffs=0, only_S20=0, only_S21=0
- ipopt_zU_out: n_common_keys=7298, n_diffs=0, only_S20=0, only_S21=0

### Param group differences

- **vmag_req**: 1 differing Param(s), e.g. vmag_req n_diffs=24 inf_norm=0.00027015556839482535
- **p_pf_req**: 1 differing Param(s), e.g. p_pf_req n_diffs=24 inf_norm=0.0014631987511639633
- **q_pf_req**: 1 differing Param(s), e.g. q_pf_req n_diffs=24 inf_norm=0.0006978160350294485
- **p_ess_req**: 1 differing Param(s), e.g. p_ess_req n_diffs=24 inf_norm=1.124670930871124e-07
- **q_ess_req**: 1 differing Param(s), e.g. q_ess_req n_diffs=24 inf_norm=7.392700709657249e-10
- **dual_vmag_req**: 1 differing Param(s), e.g. dual_vmag_req n_diffs=24 inf_norm=1.3991924308119554e-05
- **dual_pf_p_req**: 1 differing Param(s), e.g. dual_pf_p_req n_diffs=24 inf_norm=6.311408611736624e-06
- **dual_pf_q_req**: 1 differing Param(s), e.g. dual_pf_q_req n_diffs=24 inf_norm=1.2946280573800095e-05
- **dual_ess_p_req**: 1 differing Param(s), e.g. dual_ess_p_req n_diffs=24 inf_norm=9.866713309794888e-07
- **dual_ess_q_req**: 1 differing Param(s), e.g. dual_ess_q_req n_diffs=24 inf_norm=1.2120552631319354e-09
- **shared_es_e_rated_fixed**: 1 differing Param(s), e.g. shared_es_e_rated_fixed n_diffs=1 inf_norm=4.3541906369657735e-08

**Step-0 gate:** {'var_values_identical': False, 'suffixes_identical': True, 'param_diffs_confined_to_named_groups': True, 'FROZEN_STATE_EQUALITY_HOLDS': False}

## 2. Step 1 -- feasibility / bound-push calibration

### S21

- raw equality violation max: 7.585267333401663e-05 (argmax sess_soc_def[0,0,0,6])
- pushed equality violation max: 6.616143671444699e-05 (argmax sess_soc_def[0,0,0,6])
- gate target: 6.617113671444699e-05; gate pass (>=3 sig figs): True
- reference (cycle-20 converged) context: 7.214950263900732e-07

### S20start

- raw equality violation max: 7.59047884816037e-05 (argmax sess_soc_def[0,0,0,6])
- pushed equality violation max: 6.621355005825931e-05 (argmax sess_soc_def[0,0,0,6])
- gate target: None; gate pass (>=3 sig figs): False
- reference (cycle-20 converged) context: None

## 3. Step 2 -- objective-gradient calibration

- ||grad f||_inf = 100000.0 at slack_shared_es_soc_final_up[0,0,0]
- implied IPOPT scaling min(1,100/||grad||_inf) = 0.001
- gate target = 0.001; gate pass = True
- objective value reconstruction relative mismatch: 3.1416103941895254e-16
- objective gradient reconstruction relative mismatch (inf): 3.637978807091713e-17
- gradient reconstruction pass (<1e-9): True

## 4. Step 3 -- stationarity reconstruction

Calibrated convention: `r = grad_f(x) - J(x)^T*dual_raw(x) - zL_raw(x) - zU_raw(x)`, using Pyomo's own `dual`/`ipopt_zL_*`/`ipopt_zU_*` suffix values exactly as stored (no re-signing beyond this single fixed convention).

### S20
- ||r||_inf = 0.0011323058076182222, ||r||_2 = 0.0012750395407129798, argmax = e[0,0,0,7]
- gate target = 0.0011323057973496387, relative error = 9.068737026318777e-09, calibrated = True

### S21_raw
- ||r||_inf = 18366.791482401815, ||r||_2 = 62555.60835630291, argmax = expected_interface_pf_p[23]

### S21_pushed
- ||r||_inf = 18366.791482401815, ||r||_2 = 62555.60835893277, argmax = expected_interface_pf_p[23]
- gate target = 18366.791482401353, relative error = 2.515536308795948e-14, calibrated = True

### S20start_raw
- ||r||_inf = 18369.21372923023, ||r||_2 = 62572.416728976095, argmax = expected_interface_pf_p[23]

### S20start_pushed
- ||r||_inf = 18369.21372923023, ||r||_2 = 62572.416731597776, argmax = expected_interface_pf_p[23]
- gate target = 18369.213729230112, relative error = 6.3375233988207285e-15, calibrated = True

**Calibration status: CALIBRATED**

## 5. Step 4 -- attribution on residual vectors

- ||dR||_inf = 18366.791482399472, argmax = expected_interface_pf_p[23], value at argmax = 18366.791482399472
- reconstruction error (inf): 0.0 (relative: 0.0)
- ADMM_parameter_change: ||.||_inf=18366.791482399472, value at dR-argmax=18366.791482399472, fraction=1.0, sign_matches=True
- bound_and_mult_push: ||.||_inf=0.09991965587185184, value at dR-argmax=0.0, fraction=0.0, sign_matches=False
- other_remainder: ||.||_inf=0.0, value at dR-argmax=0.0, fraction=0.0, sign_matches=False
- dominance status: DOMINANT:ADMM_parameter_change
- note: Step 0 found that S20 and S21 do NOT share an identical primal point (shared_es_soc / shared_es_e_rated differ). "other_remainder" therefore absorbs both genuine nonlinear interaction effects AND that uncontrolled primal-point shift; it is not a clean third category.

## 6. Step 5 -- Arm A terminal-iterate forensic (independent)

- iteration rows parsed: Arm A=3001, cycle20=116
- 'z' marker count: Arm A=2924 (first at iteration 77), cycle20=0
- 'Some value in z_U becomes too large' message count: Arm A=2924, cycle20=0
- final barrier mu (Arm A) = 1.8449144625279508e-06 (target 1.8449144625279508e-06, match=True)
- final constraint violation (unscaled, Arm A) = 0.061921343776988665 (target 0.061921343776988665, match=True)
- top offending rows (first 10 of 30):
  - node_balance_p[0,0,0,15]: residual=0.06192134377698866, side=equality, equality=True
  - pij_def[31,0,0,15]: residual=0.061921343776977555, side=equality, equality=True
  - pji_def[31,0,0,15]: residual=0.061804368331065174, side=equality, equality=True
  - node_balance_p[1,0,0,15]: residual=0.06180436833097858, side=equality, equality=True
  - pij_def[31,0,0,8]: residual=0.060468594542801735, side=equality, equality=True
  - node_balance_p[0,0,0,8]: residual=0.0604685945427913, side=equality, equality=True
  - pji_def[31,0,0,8]: residual=0.06032165528749389, side=equality, equality=True
  - node_balance_p[1,0,0,8]: residual=0.06032165528710767, side=equality, equality=True
  - node_balance_p[0,0,0,9]: residual=0.05844650580332822, side=equality, equality=True
  - pij_def[31,0,0,9]: residual=0.058446505803326554, side=equality, equality=True
- zL*(x-l) distribution: {'n': 7390, 'max': 0.0024149109321272653, 'mean': 0.0017736019021494585, 'median': 0.001830969306555249, 'n_exceeding_terminal_mu': 7390}
- zU*(u-x) distribution: {'n': 7266, 'max': 0.00221057042466828, 'mean': 0.0017991253785547764, 'median': 0.001844448610919535, 'n_exceeding_terminal_mu': 7264}
- inequality-row caveat: Inequality-row residuals above are the PHYSICAL body-vs-bound violation at the final .sol primal point; they are NOT the internal IPOPT slack-equality residual, since the .sol file carries no slack values for inequality rows.

## Verdict

Step 0 found S20/S21 are NOT at an identical primal point: `shared_es_soc`/`shared_es_e_rated` differ (see Step 0 var-comparison above). The premise "S20 vs S21 is a Param-only difference at an identical primal point" is falsified by evidence. The Step 4 ΔR decomposition (reported above for completeness, informational only) is therefore contaminated by this uncontrolled primal-point shift and cannot support a clean ADMM-vs-PUSH dominance claim, even though its raw numbers nominally point at the ADMM/parameter-change component.

H_TRANSFER REJECTED — ATTRIBUTION INCONCLUSIVE
