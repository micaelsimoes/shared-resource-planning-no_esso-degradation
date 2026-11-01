# P5.15 W159 — closing reads before the Step 6 tables freeze (Addendum 65)

Generated 2026-10-06T08:07:43.234582+00:00 by `p515_s53_w159_closing_reads.py` (sha256 `1eacbc238dca`), git HEAD `af09a806244b`; zero solves (guard verified at 0), pickle blocked (0 calls). Every value below is read from records; the JSON `w159_closing_reads.json` carries the full evidence.

## (a) Certifying spec and as-run configuration of every evaluation T1–T10 use

Current configuration (operational definition: `data/SRP1/Results/P515S53/w142_resettle_v6/frozen_s53_resettle_spec_v6_96c23404.json inputs_in_force_now (stage spec v6 96c23404)`): case file `dbfdb2a0`, ESS params `39106f93` (E/P bounds [2.0, 4.0] h), AA keep_memory, tight tail compl_inf_tol 1e-06, BASELINE (C2 + phi_cal 0.985 + soh_min 0.70) (k 35,851.36).

| evaluation | stage (certifying spec) | rule | status / k* | started (UTC) | classification | variant / differences | tables |
|---|---|---|---|---|---|---|---|
| e_soh050 (`3f01b9ea`) | A64 v1 (rule v6) | settling_criterion_v6 | certified / 184 | 2026-10-05T12:23:30 | current + declared design variant | SoH floor row [0.5] (current 0.7) -- declared (the soh_min row) | T10 |
| g070_neutrality (`5007232b`) | A64 v1 (rule v6) | settling_criterion_v6 | certified / 172 | 2026-10-05T09:13:10 | current + declared design variant | model variant {'eol_retention_r': 0.8, 'calendar_retention_per_year': 0.985, 'available_energy_soh_point': 'end', 'ageing_enabled': True} (identical in effect to the baseline) | T10 |
| h_unit_m175 (`dc3ab932`) | A64 v1 (rule v6) | settling_criterion_v6 | certified / 121 | 2026-10-05T11:26:51 | current + declared design variant | flexibility price multiplier 1.75 (declared) | T10 |
| h_x0_m175 (`b9f9be35`) | A64 v1 (rule v6) | settling_criterion_v6 | certified / 139 | 2026-10-05T10:30:03 | current + declared design variant | flexibility price multiplier 1.75 (declared) | T10 |
| ref:7aa017f0, x0 settled (W101 d110bd1a, certified at 181) (`d110bd1a`) | W101 continuation (settling_criterion v1 rule) | settling_criterion | certified / 181 | 2026-09-27T20:22:38 | current | — | T1, T2, T3, T4, T5, T6, T7, T8, T10 |
| ref:bd504ecf, unit n7_4h_e1 settled (W101 3f084f2f, certified at 172) (`3f084f2f`) | W101 continuation (settling_criterion v1 rule) | settling_criterion | certified / 172 | 2026-09-27T21:45:35 | current | — | T1, T2, T3, T7, T10 |
| W118 yl_y2030 (`6e4f11a9`) | W118 r2 (rule v2) | settling_criterion_v2 | certified / 169 | 2026-09-29T15:45:22 | current | — | T4 |
| W118 yl_y2035 (`3a2387f4`) | W118 r2 (rule v2) | settling_criterion_v2 | certified / 167 | 2026-09-29T19:39:50 | current | — | T4 |
| pb_y2025_n5 (`ca29c5e8`) | W118 r2 (rule v2) | settling_criterion_v2 | certified / 167 | 2026-09-29T08:22:28 | current | — | T5 |
| pb_y2025_n7 (`b9b8e4be`) | W118 r2 (rule v2) | settling_criterion_v2 | certified / 172 | 2026-09-29T14:07:40 | current | — | T5 |
| pb_y2025_n9 (`b7bce5a8`) | W118 r2 (rule v2) | settling_criterion_v2 | certified / 174 | 2026-09-29T12:07:08 | current | — | T5 |
| pb_y2030_n5 (`3846675a`) | W118 r2 (rule v2) | settling_criterion_v2 | certified / 182 | 2026-09-29T09:58:25 | current | — | T5 |
| pb_y2030_n7 (`828f03fa`) | W118 r2 (rule v2) | settling_criterion_v2 | certified / 195 | 2026-09-29T06:37:12 | current | — | T5 |
| pb_y2030_n9 (`344c936f`) | W118 r2 (rule v2) | settling_criterion_v2 | certified / 192 | 2026-09-29T04:42:08 | current | — | T5 |
| ref:5ca4f86c (`24c5ccb6`) | W118 r2 (rule v2) | settling_criterion_v2 | certified / 281 | 2026-09-28T20:52:01 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2, T9 |
| ref:e28de4ac (`1fe91e86`) | W118 r2 (rule v2) | settling_criterion_v2 | certified / 261 | 2026-09-28T18:46:58 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2, T9 |
| e_c2 (`ba969ad2`) | ext v3 (rule v6) | settling_criterion_v6 | certified / 172 | 2026-10-04T00:32:07 | current + declared design variant | model variant {'eol_retention_r': 0.8, 'calendar_retention_per_year': 1.0, 'available_energy_soh_point': 'end', 'ageing_enabled': True}: calibration k 35,851.36 (10,000 cycles at DoD 0.8 to retention r = 0.80), phi_cal [1.0], SoH point ['end'] | T1, T2, T8 |
| e_c2_calfade (`bbd82994`) | ext v3 (rule v6) | settling_criterion_v6 | certified / 172 | 2026-10-04T03:06:00 | current + declared design variant | model variant {'eol_retention_r': 0.8, 'calendar_retention_per_year': 0.985, 'available_energy_soh_point': 'end', 'ageing_enabled': True} (identical in effect to the baseline) | T1, T2, T8 |
| e_c3_midblock (`8b1eace9`) | ext v3 (rule v6) | settling_criterion_v6 | certified / 174 | 2026-10-04T04:21:59 | current + declared design variant | model variant {'eol_retention_r': 0.5, 'calendar_retention_per_year': 1.0, 'available_energy_soh_point': 'mid', 'ageing_enabled': True}: calibration k 11,541.56 (10,000 cycles at DoD 0.8 to retention r = 0.50), phi_cal [1.0], SoH point ['mid'] | T1, T2, T8 |
| e_c3_unit (`409740fb`) | ext v3 (rule v6) | settling_criterion_v6 | certified / 160 | 2026-10-03T23:20:13 | current + declared design variant | model variant {'eol_retention_r': 0.5, 'calendar_retention_per_year': 1.0, 'available_energy_soh_point': 'end', 'ageing_enabled': True}: calibration k 11,541.56 (10,000 cycles at DoD 0.8 to retention r = 0.50), phi_cal [1.0], SoH point ['end'] | T1, T2, T8 |
| e_c4 (`c59d7006`) | ext v3 (rule v6) | settling_criterion_v6 | certified / 173 | 2026-10-04T01:48:59 | current + declared design variant | model variant {'eol_retention_r': 0.7, 'calendar_retention_per_year': 1.0, 'available_energy_soh_point': 'end', 'ageing_enabled': True}: calibration k 22,429.39 (10,000 cycles at DoD 0.8 to retention r = 0.70), phi_cal [1.0], SoH point ['end'] | T1, T2, T8 |
| e_no_ageing (`1b10e9e1`) | ext v3 (rule v6) | settling_criterion_v6 | certified / 177 | 2026-10-04T05:39:30 | current + declared design variant | model variant {'eol_retention_r': 0.5, 'calendar_retention_per_year': 1.0, 'available_energy_soh_point': 'end', 'ageing_enabled': False}: calibration no ageing (ageing disabled), phi_cal [1.0], SoH point ['end'] | T1, T2, T8 |
| pb_y2025_n5_v6 (`be95e576`) | ext v3 (rule v6) | settling_criterion_v6 | certified / 199 | 2026-10-04T07:00:26 | current | — | T1, T2, T5 |
| b_0dd237f0 (`38d09af2`) | v4 | settling_criterion_v4 | certified / 173 | 2026-09-30T09:10:13 | current | — | T1, T2 |
| b_2a0ba8b2 (`33447912`) | v4 | settling_criterion_v4 | certified / 173 | 2026-09-30T07:46:52 | current | — | T1, T2 |
| b_4649234b (`4de68ced`) | v5 | settling_criterion_v5 | certified / 148 | 2026-09-30T14:36:09 | current | — | T1, T2 |
| d_c52e1670 (`af42a163`) | v5 | settling_criterion_v5 | not_certified / 198 | 2026-09-30T15:46:22 | current | — | T1, T2, T3 |
| c_156ce2d1 (`3a4c10c8`) | v6 | settling_criterion_v6 | certified / 176 | 2026-10-02T14:54:09 | current | — | T1, T2 |
| c_6597a79d (`3a0d03e8`) | v6 | settling_criterion_v6 | certified / 193 | 2026-10-02T16:25:24 | current | — | T1, T2 |
| d_3632b0ae (`1fe57699`) | v6 | settling_criterion_v6 | certified / 153 | 2026-10-01T07:01:54 | current | — | T1, T2, T3 |
| d_36686489 (`4de61708`) | v6 | settling_criterion_v6 | not_certified / 191 | 2026-10-01T08:13:10 | current | — | T1, T2, T3, T9 |
| d_4a82a64a (`14b00a04`) | v6 | settling_criterion_v6 | certified / 142 | 2026-09-30T21:25:50 | current | — | T1, T2, T3 |
| d_9246ed01 (`579a1370`) | v6 | settling_criterion_v6 | certified / 138 | 2026-10-01T21:44:22 | current | — | T1, T2, T3 |
| d_a12d95a2 (`9bc66d9e`) | v6 | settling_criterion_v6 | not_certified / 203 | 2026-10-01T12:33:48 | current | — | T1, T2, T3 |
| d_c7fee8be (`aa0b99f5`) | v6 | settling_criterion_v6 | certified / 145 | 2026-10-01T16:48:53 | current | — | T1, T2, T3 |
| d_d3709599 (`64408bae`) | v6 | settling_criterion_v6 | certified / 175 | 2026-10-01T10:26:05 | current | — | T1, T2, T3 |
| d_f759dd48 (`5f1b1aa6`) | v6 | settling_criterion_v6 | not_certified / 198 | 2026-10-01T15:08:19 | current | — | T1, T2, T3 |
| g_37b5c499 (`bf2d8348`) | v6 | settling_criterion_v6 | certified / 185 | 2026-10-02T17:56:45 | current | — | T1, T2 |
| g_47dce43c (`46ff23a1`) | v6 | settling_criterion_v6 | certified / 174 | 2026-10-02T19:17:29 | current | — | T1, T2 |
| g_48749148 (`22331aa7`) | v6 | settling_criterion_v6 | certified / 170 | 2026-10-02T20:32:39 | current | — | T1, T2 |
| g_9abf31d4 (`5560d0f0`) | v6 | settling_criterion_v6 | certified / 172 | 2026-10-02T21:47:46 | current | — | T1, T2 |
| h_50dea31c (`354b1bda`) | v6 | settling_criterion_v6 | certified / 146 | 2026-10-02T01:26:06 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2 |
| h_74eda68d (`dca08628`) | v6 | settling_criterion_v6 | certified / 150 | 2026-10-02T02:26:28 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2 |
| h_aa8a76d7 (`22dd5ef8`) | v6 | settling_criterion_v6 | certified / 153 | 2026-10-01T22:54:36 | current + declared design variant | flexibility price multiplier 1.5 (declared) | T1, T2, T9 |
| h_f9eae48f (`aed2d618`) | v6 | settling_criterion_v6 | not_certified / 194 | 2026-10-01T23:58:58 | current + declared design variant | flexibility price multiplier 1.5 (declared) | T1, T2, T9 |
| i_5a6a88b4 (`e046a8cf`) | v6 | settling_criterion_v6 | certified / 198 | 2026-10-02T07:10:07 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2 |
| j_5f3cccb4 (`cc7f0059`) | v6 | settling_criterion_v6 | not_certified / 270 | 2026-10-02T12:22:19 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2, T9 |
| j_a11d7966 (`87efbc3d`) | v6 | settling_criterion_v6 | certified / 158 | 2026-10-02T10:44:31 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2 |
| j_f3aa335e (`af066d6b`) | v6 | settling_criterion_v6 | not_certified / 260 | 2026-10-02T08:45:31 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2 |
| l_0ee93aca (`86d27f42`) | v6 | settling_criterion_v6 | not_certified / 284 | 2026-10-03T02:08:45 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2, T9 |
| l_195156fa (`1fd23b8d`) | v6 | settling_criterion_v6 | certified / 252 | 2026-10-03T21:19:23 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2 |
| l_2ab0ce2d (`75266862`) | v6 | settling_criterion_v6 | certified / 400 | 2026-10-03T06:08:20 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2 |
| l_45aa25a6 (`296e09ff`) | v6 | settling_criterion_v6 | not_certified / 291 | 2026-10-03T14:21:27 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2, T9 |
| l_76c78064 (`571ed31e`) | v6 | settling_criterion_v6 | certified / 229 | 2026-10-03T12:43:45 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2 |
| l_7b199ef9 (`331e76c7`) | v6 | settling_criterion_v6 | certified / 188 | 2026-10-02T23:07:08 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2 |
| l_7c455554 (`9906389c`) | v6 | settling_criterion_v6 | not_certified / 282 | 2026-10-03T16:34:27 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2, T9 |
| l_7db09f6c (`7f5a857b`) | v6 | settling_criterion_v6 | certified / 251 | 2026-10-03T10:49:14 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2 |
| l_8e4c220e (`5775a50c`) | v6 | settling_criterion_v6 | certified / 233 | 2026-10-03T09:09:43 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2 |
| l_b2251bc5 (`abf4ac0a`) | v6 | settling_criterion_v6 | not_certified / 320 | 2026-10-03T18:43:28 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2, T9 |
| l_df1a5525 (`6c8f6353`) | v6 | settling_criterion_v6 | certified / 236 | 2026-10-03T04:18:38 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2 |
| l_e1da0984 (`2587ab49`) | v6 | settling_criterion_v6 | certified / 235 | 2026-10-03T00:28:33 | current + declared design variant | flexibility price multiplier 2.0 (declared) | T1, T2 |

### Old-configuration figures still in a manuscript table

- **T8 column 'eps_AE (0.50, superseded)' -- the Addendum 28 ageing-batch mechanism table data/SRP1/Results/P515S46/ageing_mechanism/ageing_mechanism.json (b5eca2a2), formula elasticity = ln(value / value_C3) / ln(AE / AE_C3)**
  - `data/SRP1/Results/P515S45/campaign_s45_a1a/evals/7eb1ce62c2509f54_n7_4h_e1/evaluation_record.json` (ageing-batch point listed in ageing_mechanism.json inputs): ESS params at git_head `fbb1296f`: calibration {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8, 'eol_retention_r': 0.5}, minimum_soh 0.5, E/P [2.0, 10.0] h, calendar retention absent from the file
    read back from the record: variant None; calibration not read back; phi_cal None; SoH floor None; SoH point None; tail in force None; certified at 125; started 2026-09-20T13:08:51.319945+00:00; classification OLD CONFIGURATION
  - `data/SRP1/Results/P515S46/campaign_s46_ageing/evals/06f092d164f13819_n7_4h_e1_no_ageing/evaluation_record.json` (ageing-batch point listed in ageing_mechanism.json inputs): ESS params at git_head `e127daa7`: calibration {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8, 'eol_retention_r': 0.5}, minimum_soh 0.5, E/P [2.0, 4.0] h, calendar retention absent from the file
    read back from the record: variant {'eol_retention_r': 0.5, 'calendar_retention_per_year': 1.0, 'available_energy_soh_point': 'end', 'ageing_enabled': False}; calibration no ageing (ageing disabled); phi_cal [1.0]; SoH floor [0.5]; SoH point ['end']; tail in force None; certified at 118; started 2026-09-21T13:11:22.014558+00:00; classification OLD CONFIGURATION
  - `data/SRP1/Results/P515S46/campaign_s46_ageing/evals/65a5da775d1ff5b2_n7_4h_e1_c4/evaluation_record.json` (ageing-batch point listed in ageing_mechanism.json inputs): ESS params at git_head `e127daa7`: calibration {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8, 'eol_retention_r': 0.5}, minimum_soh 0.5, E/P [2.0, 4.0] h, calendar retention absent from the file
    read back from the record: variant {'eol_retention_r': 0.7, 'calendar_retention_per_year': 1.0, 'available_energy_soh_point': 'end', 'ageing_enabled': True}; calibration k 22,429.39 (10,000 cycles at DoD 0.8 to retention r = 0.70); phi_cal [1.0]; SoH floor [0.5]; SoH point ['end']; tail in force None; certified at 119; started 2026-09-21T13:11:22.003506+00:00; classification OLD CONFIGURATION
  - `data/SRP1/Results/P515S46/campaign_s46_ageing/evals/98e2857016a16d1c_n7_4h_e1_c2_calfade/evaluation_record.json` (ageing-batch point listed in ageing_mechanism.json inputs): ESS params at git_head `e127daa7`: calibration {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8, 'eol_retention_r': 0.5}, minimum_soh 0.5, E/P [2.0, 4.0] h, calendar retention absent from the file
    read back from the record: variant {'eol_retention_r': 0.8, 'calendar_retention_per_year': 0.985, 'available_energy_soh_point': 'end', 'ageing_enabled': True}; calibration k 35,851.36 (10,000 cycles at DoD 0.8 to retention r = 0.80); phi_cal [0.985]; SoH floor [0.5]; SoH point ['end']; tail in force None; certified at 121; started 2026-09-21T13:11:22.008009+00:00; classification OLD CONFIGURATION
  - `data/SRP1/Results/P515S46/campaign_s46_ageing/evals/c6b53015fcf65e24_n7_4h_e1_c2/evaluation_record.json` (ageing-batch point listed in ageing_mechanism.json inputs): ESS params at git_head `e127daa7`: calibration {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8, 'eol_retention_r': 0.5}, minimum_soh 0.5, E/P [2.0, 4.0] h, calendar retention absent from the file
    read back from the record: variant {'eol_retention_r': 0.8, 'calendar_retention_per_year': 1.0, 'available_energy_soh_point': 'end', 'ageing_enabled': True}; calibration k 35,851.36 (10,000 cycles at DoD 0.8 to retention r = 0.80); phi_cal [1.0]; SoH floor [0.5]; SoH point ['end']; tail in force None; certified at 119; started 2026-09-21T13:11:22.000548+00:00; classification OLD CONFIGURATION
  - `data/SRP1/Results/P515S46/campaign_s46_ageing/evals/ed4a1acc7059784d_n7_4h_e1_c3_midblock/evaluation_record.json` (ageing-batch point listed in ageing_mechanism.json inputs): ESS params at git_head `e127daa7`: calibration {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8, 'eol_retention_r': 0.5}, minimum_soh 0.5, E/P [2.0, 4.0] h, calendar retention absent from the file
    read back from the record: variant {'eol_retention_r': 0.5, 'calendar_retention_per_year': 1.0, 'available_energy_soh_point': 'mid', 'ageing_enabled': True}; calibration k 11,541.56 (10,000 cycles at DoD 0.8 to retention r = 0.50); phi_cal [1.0]; SoH floor [0.5]; SoH point ['mid']; tail in force None; certified at 123; started 2026-09-21T13:11:22.011352+00:00; classification OLD CONFIGURATION
  - `data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7aa017f09989b56d_x0/evaluation_record.json` (Q(0) of the 0.50-era values (S46 campaign_results baseline_inputs.Q0_eval_dir; not itself listed in ageing_mechanism inputs, reached through the pinned campaign_results)): ESS params at git_head `8bd0102c`: calibration {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8, 'eol_retention_r': 0.5}, minimum_soh 0.5, E/P [2.0, 10.0] h, calendar retention absent from the file
    read back from the record: variant None; calibration not read back; phi_cal None; SoH floor None; SoH point None; tail in force None; certified at 132; started 2026-09-19T18:50:42.002646+00:00; classification OLD CONFIGURATION
  - T8 values: {'C3_unit': None, 'C2': 0.6005243309458448, 'C4': 0.4132607018916576, 'C2_calfade': 0.6757370557796795, 'C3_midblock': 0.13000447119227676, 'no_ageing': 0.6198441005274611}
  - rows: ['T8: C2 -- column eps_AE (0.50, superseded)', 'T8: C4 -- column eps_AE (0.50, superseded)', 'T8: C2_calfade -- column eps_AE (0.50, superseded)', 'T8: C3_midblock -- column eps_AE (0.50, superseded)', 'T8: no_ageing -- column eps_AE (0.50, superseded)']

### Superseded certificate displayed

- T5 row pb_y2025_n5 (W118 r2 certificate, rule v2): superseded by the v6 re-run pb_y2025_n5_v6 (claim C:y2025__n5_p0.25_e0.5): the W118 certificate sat on a non-Optimal cycle (Addendum 59)

### T6 uncoordinated arms (benchmark runs, spec v5 bca69f97; not settling certificates)

- `nrf_arm_passive_cold_r2`: instance `8435c718`, arm network compl_inf_tol 1e-06, linear solvers ['ma97'], case / ESS params sha recorded: False
- `nrf_arm_passive_perturbed_r2`: instance `8435c718`, arm network compl_inf_tol 1e-06, linear solvers ['ma97'], case / ESS params sha recorded: False
- `nrf_arm_passive_warm_from_certified_r2`: instance `8435c718`, arm network compl_inf_tol 1e-06, linear solvers ['ma97'], case / ESS params sha recorded: False
- `nrf_arm_price_taker_cold_r2`: instance `8435c718`, arm network compl_inf_tol 1e-06, linear solvers ['ma97'], case / ESS params sha recorded: False
- `nrf_arm_price_taker_perturbed_r2`: instance `8435c718`, arm network compl_inf_tol 1e-06, linear solvers ['ma97'], case / ESS params sha recorded: False
- `nrf_arm_price_taker_warm_from_certified_r2`: instance `8435c718`, arm network compl_inf_tol 1e-06, linear solvers ['ma97'], case / ESS params sha recorded: False

## (b) T6 and the two uncoordinated arms

T6 carries Q and band of both arms: **True**. Fields:

| field | in T6 JSON | in T6 Markdown | source |
|---|---|---|---|
| passive: Q (best of 3 starts) | True | True | report_v3.json per_arm_nrf.passive.q_best |
| passive: best start | True | True | report_v3.json per_arm_nrf.passive.best_start |
| passive: multimodality band | True | True | report_v3.json per_arm_nrf.passive.multimodality_band_eur |
| passive: Q by start (cold / warm_from_certified / perturbed) | False | False | report_v3.json per_arm_nrf.passive.q_by_start |
| price_taker: Q (best of 3 starts) | True | True | report_v3.json per_arm_nrf.price_taker.q_best |
| price_taker: best start | True | True | report_v3.json per_arm_nrf.price_taker.best_start |
| price_taker: multimodality band | True | True | report_v3.json per_arm_nrf.price_taker.multimodality_band_eur |
| price_taker: Q by start (cold / warm_from_certified / perturbed) | False | False | report_v3.json per_arm_nrf.price_taker.q_by_start |
| claim (best arm, price-taker): benefit, relative, multiple, determinate | True | True | report_v3.json claim.benefit_eur / benefit_relative / determinate |
| passive arm against coordinated: Q_passive - Q181 | True | False | report_v3.json claim.decomposition.passive_NRF_minus_coordinated_eur |
| passive minus price-taker: Q_passive - Q_price_taker | True | False | report_v3.json claim.decomposition.passive_NRF_minus_price_taker_NRF_eur |
| passive arm: relative benefit and multiple of its own band | False | False | NOT RECORDED in report_v3.json (only the best-arm claim carries benefit_relative and determinate); derivable from per_arm_nrf.passive.q_best, coordinated.q and the bands by the claim definition |
| coordinated band (reproducibility 0.011 %) | True | True | report_v3.json claim.bands_eur.coordinated_reproducibility_0.011pct |
| reverse-flow interface-hours in the coordinated solution (and MWh) | True | True | report_v3.json coordinated_reverse_flow_count.totals.material |
| NRF arms: reverse flow excluded by the arm definition (pg_adn >= 0 rows) | False | False | report_v3.json no_reverse_flow_definition (Addendum 57 Decision 1(b)); per-arm JSON no_reverse_flow |
| NRF arms: consistency re-evaluation NRF violations (n, max excess pu) per start | False | False | report_v3.json consistency_nrf.<arm run>.nrf_violations |
| sweep_passive_cold: blocks / hours the TN cannot accept | True | True | report_v3.json sweep.sweep_passive_cold |
| sweep_passive_cold: which blocks fail | False | False | report_v3.json sweep.sweep_passive_cold.failing_blocks |
| sweep_price_taker_cold: blocks / hours the TN cannot accept | True | True | report_v3.json sweep.sweep_price_taker_cold |
| sweep_price_taker_cold: which blocks fail | False | False | report_v3.json sweep.sweep_price_taker_cold.failing_blocks |
| per-block decomposition of each arm (12 TSO + 36 DSO blocks) | False | False | per-arm run records nrf_arm_<arm>_<start>_r2.json phase_C_sequential_pass.evaluation.block_components (the arm cost source is phase_C_sequential_pass); not in report_v3.json |
| curtailment by agent per arm and start | False | False | report_v3.json curtailment_table.arms_and_variants_phase_A |

## (c) Provenance reads

### 1. Threading

- Launch-command environment assignments: {'v6': {'n_commands': 34, 'env_assignments_before_interpreter': ['NLP_SOLVER_PATH']}, 'ext v3': {'n_commands': 7, 'env_assignments_before_interpreter': ['NLP_SOLVER_PATH']}, 'A64 v1': {'n_commands': 4, 'env_assignments_before_interpreter': ['NLP_SOLVER_PATH']}}
- Shell profiles naming a thread variable: none (checked: ['/Users/micaelsimoes/.zshrc', '/Users/micaelsimoes/.zprofile', '/Users/micaelsimoes/.zshenv', '/Users/micaelsimoes/.bash_profile', '/Users/micaelsimoes/.bashrc', '/Users/micaelsimoes/.profile', '/etc/zprofile', '/etc/zshrc', '/etc/profile', '/etc/bashrc'])
- Claude shell snapshots: [('snapshot-zsh-1791272785439-2f08pz.sh', '2026-10-06T07:46:25', False)]
- .env names a thread variable: {'OMP_NUM_THREADS': False, 'MKL_NUM_THREADS': False, 'OPENBLAS_NUM_THREADS': False, 'VECLIB_MAXIMUM_THREADS': False, 'NUMEXPR_NUM_THREADS': False, 'OMP_DYNAMIC': False, 'OMP_THREAD_LIMIT': False}
- launchctl getenv: {'OMP_NUM_THREADS': None, 'MKL_NUM_THREADS': None, 'OPENBLAS_NUM_THREADS': None, 'VECLIB_MAXIMUM_THREADS': None, 'NUMEXPR_NUM_THREADS': None, 'OMP_DYNAMIC': None, 'OMP_THREAD_LIMIT': None}
- Tracked Python naming thread variables: production (non-p5) ['admm_persistent_workers.py']; harness lines ['320: THREAD_CAP_ENV = {', "321:     'OMP_NUM_THREADS': '1',"] … ['1789: child_env.update(THREAD_CAP_ENV)'] ["2233: raise SystemExit(f'CHILD REFUSES: thread caps not in force in the child environment: {bad}')"]
- This process (the current shell's export): {'OMP_NUM_THREADS': None, 'MKL_NUM_THREADS': None, 'OPENBLAS_NUM_THREADS': None, 'VECLIB_MAXIMUM_THREADS': None, 'NUMEXPR_NUM_THREADS': None, 'OMP_DYNAMIC': None, 'OMP_THREAD_LIMIT': None}
- Every table evaluation's child saw OMP_NUM_THREADS = '1': **True**

### 2. Linear-solver banners

| cell | log | sha256 | matches launch manifest | first banner line | banner | all banners |
|---|---|---|---|---:|---|---|
| d_4a82a64a | `data/SRP1/Results/P56A/evals/p515s44_s53_w142_resettle_v6_d_4a82a64a_14b00a04ffbcfd33_run/logs/optim_log_case9_2025_Summer.log` | `f451d4af4369551c` | True | 26 | This is Ipopt version 3.14.18, running with linear solver ma97. | 143 × ['ma97'] |
| d_4a82a64a | `data/SRP1/Results/P56A/evals/p515s44_s53_w142_resettle_v6_d_4a82a64a_14b00a04ffbcfd33_run/logs/optim_log_case33_2_2025_Summer.log` | `7cd079e562b09f83` | True | 25 | This is Ipopt version 3.14.18, running with linear solver ma97. | 143 × ['ma97'] |
| d_4a82a64a | `data/SRP1/Results/P56A/evals/p515s44_s53_w142_resettle_v6_d_4a82a64a_14b00a04ffbcfd33_run/logs/optim_log_esso_node5_cycle001.txt` | `830ff5a9b11b5af6` | True | 8 | This is Ipopt version 3.14.18, running with linear solver ma57. | 1 × ['ma57'] |
| e_soh050 | `data/SRP1/Results/P56A/evals/p515s44_s53_w155_a64_r2_e_soh050_3f01b9eaaab13c82_run/logs/optim_log_case9_2025_Summer.log` | `31ff057541fd63f5` | True | 26 | This is Ipopt version 3.14.18, running with linear solver ma97. | 185 × ['ma97'] |
| e_soh050 | `data/SRP1/Results/P56A/evals/p515s44_s53_w155_a64_r2_e_soh050_3f01b9eaaab13c82_run/logs/optim_log_esso_node7_cycle001.txt` | `2e0d0829490a337f` | True | 8 | This is Ipopt version 3.14.18, running with linear solver ma57. | 1 × ['ma57'] |

All 495 logs the d_4a82a64a launch manifest lists: 495 hash-match, 0 missing; banners network {'ma97': 6887}, ESSO {'ma57': 429}; versions ['3.14.18']; files without a banner 0.

### 3. Versions now and environment changes

- Read 2026-10-06T08:07:40.567563+00:00: Python `3.11.11 | packaged by conda-forge | (main, Dec  5 2024, 08:47:03) [Clang 18.1.8 ]`; Pyomo 6.9.5; sw_vers `ProductName: macOS ProductVersion: 27.0 BuildVersion: 26A428`
- Window: first v6 cell start 2026-09-30T21:25:50.591628+00:00; last launch end 2026-10-05T13:42:31.230468+00:00
- conda-meta/history: 1 entr(y/ies); last 2026-09-09 14:21:48 (local); entries at or after the first v6 start: []
- Environment files: newest mtime 2026-09-09T13:22:23.053606+00:00; modified at or after the first v6 start: 0
- softwareupdate --history rows at or after the first v6 start: []; all rows: ['macOS Sonoma 14.2.1                                14.2.1     09/01/2024, 09:41:38', 'Command Line Tools for Xcode                       15.3       20/04/2024, 10:58:02', 'macOS Sonoma 14.4.1                                14.4.1     20/04/2024, 11:00:52', 'macOS Sonoma 14.5                                  14.5       29/05/2024, 18:28:49', 'macOS Sonoma 14.6.1                                14.6.1     30/08/2024, 15:08:13', 'Command Line Tools for Xcode                       16.0       17/10/2024, 18:06:03', 'Safari                                             18.0.1     17/10/2024, 18:06:03', 'macOS Sonoma 14.7                                  14.7       17/10/2024, 18:07:16', 'macOS Sequoia 15.0.1                               15.0.1     18/10/2024, 09:24:55', 'Command Line Tools for Xcode                       16.1       04/12/2024, 07:29:59', 'macOS Sequoia 15.1.1                               15.1.1     04/12/2024, 07:31:16', 'Command Line Tools for Xcode                       16.2       23/01/2025, 15:16:02', 'macOS Sequoia 15.2                                 15.2       23/01/2025, 15:17:24', 'macOS Sequoia 15.3.1                               15.3.1     21/02/2025, 19:58:28', 'Command Line Tools for Xcode                       16.3       01/05/2025, 16:49:30', 'macOS Sequoia 15.4.1                               15.4.1     01/05/2025, 16:50:51', 'macOS Sequoia 15.5                                 15.5       27/05/2025, 01:51:26', 'Command Line Tools for Xcode                       16.4       23/06/2025, 09:02:09', 'macOS Sequoia 15.6                                 15.6       08/08/2025, 01:47:26', 'macOS Sequoia 15.6.1                               15.6.1     22/08/2025, 02:33:57', 'Command Line Tools for Xcode 26.2                  26.2       18/12/2025, 18:07:05', 'Safari                                             26.2       18/12/2025, 18:07:05', 'macOS Sequoia 15.7.3                               15.7.3     18/12/2025, 18:08:11', 'Safari                                             26.5       19/05/2026, 16:13:57', 'macOS Sequoia 15.7.7                               15.7.7     19/05/2026, 16:15:02', 'macOS Tahoe 26.5                                   26.5       19/05/2026, 17:27:32', 'Command Line Tools for Xcode 26.5                  26.5       30/05/2026, 15:57:50', 'macOS Tahoe 26.5.1                                 26.5.1     02/06/2026, 18:08:51', 'macOS Tahoe 26.5.2                                 26.5.2     16/07/2026, 09:30:43', 'Command Line Tools for Xcode 26.6                  26.6       08/09/2026, 19:24:48', 'macOS Tahoe 26.6.2                                 26.6.2     08/09/2026, 19:26:59', 'Command Line Tools for Xcode 26.5                  26.5       08/09/2026, 19:46:58', 'Command Line Tools for Xcode 26.6                  26.6       08/09/2026, 19:46:58', 'Command Line Tools for Xcode 26.5                  26.5       09/09/2026, 12:11:33', 'Command Line Tools for Xcode 26.6                  26.6       09/09/2026, 12:11:33', 'Command Line Tools for Xcode 26.5                  26.5       09/09/2026, 13:39:45', 'Command Line Tools for Xcode 26.6                  26.6       09/09/2026, 13:39:45', 'Command Line Tools for Xcode 27.0                  27.0       10/09/2026, 19:03:50', 'macOS 27                                           27         18/09/2026, 18:14:27']
- /usr/local/bin/ipopt: mtime 2025-07-09T13:59:41+00:00, sha256 `b316abbe38cd851d` matches the v6 pin: True; `--version`: `Ipopt 3.14.18 (aarch64-apple-darwin24.5.0), ASL(20241111)`
- Changes at or after the first v6 start found by these reads: none

### 4. HSL_MA97 bit-compatibility statement

- Source archive: `/Users/micaelsimoes/Downloads/coinhsl-2023.11.17.tar.gz` (72 members); doc-like members ['coinhsl-2023.11.17/ChangeLog', 'coinhsl-2023.11.17/LICENCE', 'coinhsl-2023.11.17/README', 'coinhsl-2023.11.17/meson_options.txt']; PDF members []
- Binary archive: `/Users/micaelsimoes/Downloads/CoinHSL.v2023.11.17.aarch64-apple-darwin-libgfortran5.tar.gz`; doc-like members ['CoinHSL.v2023.11.17.aarch64-apple-darwin-libgfortran5/share/licenses/CoinHSL/LICENCE', 'CoinHSL.v2023.11.17.aarch64-apple-darwin-libgfortran5/share/licenses/CompilerSupportLibraries/GPL-3.0+', 'CoinHSL.v2023.11.17.aarch64-apple-darwin-libgfortran5/share/licenses/METIS/LICENSE.txt', 'CoinHSL.v2023.11.17.aarch64-apple-darwin-libgfortran5/share/licenses/OpenBLAS32/LICENSE']; PDF members []
- Pattern hits in the source archive: [('coinhsl-2023.11.17/hsl_ma97/hsl_ma97d.f90', 3086, '   ! Reset number of threads for this level (if it has been changed)'), ('coinhsl-2023.11.17/hsl_ma97/hsl_ma97d.f90', 3937, "! In the serial case could just use dsyrk (but don't for bit-compatibility)."), ('coinhsl-2023.11.17/hsl_ma97/hsl_ma97s.f90', 3086, '   ! Reset number of threads for this level (if it has been changed)'), ('coinhsl-2023.11.17/hsl_ma97/hsl_ma97s.f90', 3937, "! In the serial case could just use ssyrk (but don't for bit-compatibility).")]
- Pattern hits in the binary archive: []
- Installed libcoinhsl `230fffdd6e4d9f1e`: ma97 symbols 220; OpenMP symbols []; links libgomp/libomp False; equals the ipopt3.14.18hsl5.5.0 package member: True; runtime message line refs into hsl_ma97d.f90 [6777, 8238, 8718] -> archive source lines {6777: 'deallocate(fkeep%alloc)', 8238: 'deallocate(stack_ptr%mem)', 8718: 'end module hsl_ma97_double'}
- ~/ThirdParty-HSL build `8a4f39e48fbce79a` (mtime 2024-01-09) equals installed: False

## Checks

- a_every_table_eval_key_resolved_to_one_committed_run_record: True
- a_every_run_record_committed_clean: True
- a_0_50_era_records_match_the_ageing_mechanism_pins: True
- a_stage_by_directory_agrees_with_T2_certifying_spec: True
- a_campaign_spec_sha256_matches_every_record: True
- c2_sample_logs_exist_and_match_their_launch_manifest: True
- c2_every_d_4a82a64a_manifest_log_present_and_hash_matching: True
- inputs_committed_clean: True
- inputs_with_manifest_match: True
- w157_tables_json_matches_its_manifest: True
- zero_solve_guard_verified_0: True
- pickle_load_and_loads_0: True

