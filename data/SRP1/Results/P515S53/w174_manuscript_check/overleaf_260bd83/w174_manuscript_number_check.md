# W174a -- number check of the manuscript .tex files at Overleaf 260bd83 (declarations v3) and the section 3.5-3.6 value table

Overleaf clone `manuscript/6a67305f25e8348fb71380c3` at commit `260bd834e0a27befa545592a8c477fd329553785` (declared `260bd83`); files: `cover_letter.tex` sha256 `363ad821`, `highlights.tex` sha256 `d568f329`, `main.tex` sha256 `cb237b2d129e689a94dd50abec9a9cca739567ef97a486ec012a901f0e04b6df`, `response_to_reviewers_draft.tex` sha256 `fc523153`, `section2_expert_draft.tex` sha256 `7e347dc8`.
Frozen tables `frozen_step6_tables_v1_590088fe.json` (sha256 `590088fe`). Script `p515_s53_w174_manuscript_number_check.py` (imports `p515_s53_w171_manuscript_number_check.py` @ a379b481 and through it W164-W160, none edited). Submitted source `manuscript_submitted/main.tex` (sha256 `ca07d7db`); reviewers' document `manuscript_review/Reviewers Comments.docx` (sha256 `c7afdc8b`). ZERO SOLVES (guards verified 0), pickle blocked. Nothing in the clone is edited.

Statuses: match (declared / rule against a named record / auto-unique / auto-ambiguous), MISMATCH, approximate (section 3.5-3.6 rows only: the printed figure is the source rounded at the printed precision), no table counterpart, submitted-version figure, reviewer quotation, verified, unchecked (with the reason), no source found (section 3.5-3.6 rows only). Outside sections 3.5-3.6 a figure equal to its source at the written precision is "match" (W164 rule). Excluded LaTeX structure is counted separately.

## Counts per file -- manuscript

| file | tokens | excluded | in scope (body) | comments | match declared | match rule | match auto-unique | match auto-ambiguous | MISMATCH | approximate | no table counterpart | submitted-version | reviewer quotation, verified | unchecked | no source found | unassigned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cover_letter.tex | 5 | 3 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| highlights.tex | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| main.tex | 2516 | 220 | 2130 | 166 | 297 | 83 | 0 | 0 | 0 | 12 | 0 | 559 | 0 | 1345 | 0 | 0 |
| response_to_reviewers_draft.tex | 318 | 10 | 306 | 2 | 97 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 62 | 148 | 0 | 0 |

## Counts -- draft, not compiled (`section2_expert_draft.tex`; apart from the manuscript counts)

| file | tokens | excluded | in scope (body) | comments | match declared | match rule | match auto-unique | match auto-ambiguous | MISMATCH | approximate | no table counterpart | submitted-version | reviewer quotation, verified | unchecked | no source found | unassigned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| section2_expert_draft.tex | 231 | 0 | 92 | 139 | 60 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 167 | 0 | 0 |

Every token assigned: **True**; stale declarations: []; tokens claimed twice: [].

## Sections 3.5-3.6 (main.tex l. 1066-1150): every value and named setting, one by one

Rows: 161 -- no source found 1, match 148, approximate 12 (values with a numeric token: {'match': 129, 'approximate': 12}; named settings without a token: {'no source found': 1, 'match': 19}).

**Recorded prediction (expert, Addendum 70): every section 3.5-3.6 value matches its source.** Outcome: HELD on the values: 141 value rows -- match 129, approximate 12 (approximate = the source rounded at the printed precision); named settings: no source found 1, match 19; not matching: values [], settings ['V01.1']

Not rows: the named settings of the benchmark paragraph (l. 1138-1143: no-reverse-flow rule, passive and price-taker arm definitions, TSO dispatch at fixed interface exchanges) are the `% [CONFIRM -- W175]` analysis and are not rows here; only its printed numbers ("two", "three") are.

| id | line | printed | quantity | value in force | source | status | precision / note |
|---|---:|---|---|---|---|---|---|
| V01.1 | 1070 | utility-scale lithium iron phosphate battery | chemistry of the shared ESS | no model parameter; the recorded datasheets of the cycle-life calibrations are LFP | data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json citations.cycle_life[*].source (sha256 8bda2a0a, blob 5d113e54): ['EVE MB31 314 Ah LFP datasheet', 'Hithium 314 Ah LFP datasheet']; data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json citations.status (sha256 8bda2a0a, blob 5d113e54): 'PROPOSED BY THE EXPERT (Addendum 32); AUTHOR TO CONFIRM AND VERIFY' | **no source found** | no configuration field states a chemistry (searched git-tracked *.py and every data JSON for "LFP" / "iron phosphate" (git grep): one hit file, data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json citations; no model or case-file field carries a chemistry); the only record is the citations block of spec v18, status "PROPOSED BY THE EXPERT; AUTHOR TO CONFIRM AND VERIFY", whose two datasheets |
| V02.1 | 1070 | $\eta^{Ch} = 0.97$ | eta^Ch (shared ESS, network models and agent) | 0.97 | shared_energy_storage.py:14 (sha256 d45c72d0); model_construction_helpers.py:969 (sha256 a20a224f); shared_energy_storage_data.py:655 (sha256 9acd095f) | **match** | class default; assignments to .eff_ch/.eff_dch in production modules: [('network.py', 1186), ('network.py', 1187)] (local-ESS loader only: True) |
| V02.2 | 1070 | $\eta^{Dch} = 0.96$ | eta^Dch | 0.96 | shared_energy_storage.py:15 (sha256 d45c72d0) | **match** |  |
| V03.1 | 1070 | 10 % | SoC^Min (percent of E^Av) | 0.1 | definitions.py:39 (sha256 7e5719af); model_construction_helpers.py:894 (sha256 a20a224f) | **match** | shared_es_e_rated_fixed = e_available / s_base (shared_resources_planning, every cycle): the fraction applies to the degraded available energy |
| V03.2 | 1070 | 90 % | SoC^Max (percent of E^Av) | 0.9 | definitions.py:38 (sha256 7e5719af); model_construction_helpers.py:902 (sha256 a20a224f) | **match** |  |
| V03.3 | 1070 | $SoC^{Min} = 0.10$ | SoC^Min (fraction of E^Av) | 0.1 | definitions.py:39 (sha256 7e5719af); model_construction_helpers.py:894 (sha256 a20a224f) | **match** | shared_es_e_rated_fixed = e_available / s_base (shared_resources_planning, every cycle): the fraction applies to the degraded available energy |
| V03.4 | 1070 | $SoC^{Max} = 0.90$ | SoC^Max (fraction of E^Av) | 0.9 | definitions.py:38 (sha256 7e5719af); model_construction_helpers.py:902 (sha256 a20a224f) | **match** |  |
| V04.1 | 1070 | 50 % | SoC^0 (percent; initial = closure target) | 0.5 | definitions.py:40 (sha256 7e5719af); model_construction_helpers.py:973 (sha256 a20a224f); model_construction_helpers.py:987 (sha256 a20a224f) | **match** |  |
| V04.2 | 1070 | $SoC^{0} = 0.50$ | SoC^0 (fraction) | 0.5 | definitions.py:40 (sha256 7e5719af); model_construction_helpers.py:973 (sha256 a20a224f); model_construction_helpers.py:987 (sha256 a20a224f) | **match** |  |
| V05.1 | 1070 | $\varepsilon^{Cl} = 0.05$ | closure slack bound eps^Cl (fraction of E^Av) | 0.05 | model_construction_helpers.py:410 (sha256 a20a224f); model_construction_helpers.py:1166 (sha256 a20a224f); data/SRP1/case9/case9_params.json:22,26 (sha256 f3eff050); data/SRP1/case33_1/case33_1_params.json:22,26 (sha256 8e9e5a53); data/SRP1/case33_2/case33_2_params.json:22,26 (sha256 31b5fedf); data/SRP1/case33_3/case33_3_params.json:22,26 (sha256 a19bd5b9) | **match** | slacks.shared_ess.day_balance per case file: {'case9': True, 'case33_1': True, 'case33_2': True, 'case33_3': True} |
| V05.2 | 1070 | $10^{-5}$ p.u. | closure slack numerical allowance (per unit at baseMVA 100) | 1e-05 | definitions.py:90 (sha256 7e5719af) | **match** | in model units: per unit at baseMVA 100 (the slack and e_capacity are e_available / s_base), i.e. 1e-3 MWh -- STEP6_ROUND1_CORRECTIONS A.3(e) writes "10^-5 p.u." |
| V05.3 | 1070 | $c^{Cl} = 10^{3}$ EUR/MWh | closure slack price c^Cl (EUR/MWh) | 1000.0 | definitions.py:58 (sha256 7e5719af); model_construction_helpers.py:2341 (sha256 a20a224f) | **match** | objective term base * 1e3 * slack_pu with base = network.baseMVA (100): 1e3 per MWh of slack |
| V06.1 | 1070 | $\varepsilon^{C} = 10^{-4}$ | network complementarity tolerance eps^C (normalised) | 0.0001 | definitions.py:91 (sha256 7e5719af); definitions.py:92 (sha256 7e5719af); model_construction_helpers.py:951 (sha256 a20a224f) | **match** | shared_ess_model per case file: {'case9': 'BILINEAR_RELAXATION', 'case33_1': 'BILINEAR_RELAXATION', 'case33_2': 'BILINEAR_RELAXATION', 'case33_3': 'BILINEAR_RELAXATION'}; hat = P / S_rated_fixed (hat-link rows) |
| V07.1 | 1070 | $T^{Cal} = 15$ years | calendar lifetime T^Cal (years) | 15 | data/SRP1/SharedESS/SRP1_ESS_Params.json:17 (sha256 39106f93) ageing.calendar_life_years; data/SRP1/Results/P515S53/w142_resettle_v6/frozen_s53_resettle_spec_v6_96c23404.json (sha256 96c23404) inputs_in_force_now.configuration_now.ess_ageing_baseline.calendar_life_years = 15 | **match** | consumed by the salvage credit (calendar_life_basis REMAINING_FRACTION_AT_TERMINAL) and by the calendar retention statement below |
| V08.1 | 1070 | $\varepsilon^{E} = 10^{-5}$ | eps^E (ESSO throughput regularisation) | 1e-05 | definitions.py:74 (sha256 7e5719af); shared_energy_storage_data.py:844 (sha256 9acd095f) | **match** |  |
| V08.2 | 1070 | $c^{\sigma} = 10^{3}$ | c^sigma (ESSO P-net slack penalty) | 1000.0 | definitions.py:63 (sha256 7e5719af); shared_energy_storage_data.py:829 (sha256 9acd095f); data/SRP1/SharedESS/SRP1_ESS_Params.json:6 (sha256 39106f93) | **match** | ESS params slacks = True |
| V09.1 | 1070 | $\Delta^S = 0.25$ MVA | lattice unit Delta^S (MVA) | 0.25 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9; data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) min/max_energy_to_power_factor; p515_s47_phase_b_record.py:186 (sha256 97f573e5) | **match** |  |
| V09.2 | 1070 | $\Delta^E = 0.5$ MWh | lattice unit Delta^E (MWh) | 0.5 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9; data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) min/max_energy_to_power_factor; p515_s47_phase_b_record.py:187 (sha256 97f573e5) | **match** |  |
| V09.3 | 1070 | 2 h | minimum duration E/P (h) | 2.0 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9; data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) min/max_energy_to_power_factor | **match** |  |
| V09.4 | 1070 | 4 h | maximum duration E/P (h) | 4.0 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9; data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) min/max_energy_to_power_factor | **match** |  |
| V09.5 | 1070 | at most 5 MWh per node | maximum energy capacity per node (MWh) | 5.0 | data/SRP1/SharedESS/SRP1_ESS_Params.json:3 (sha256 39106f93) max_capacity; p515_s47_phase_b_record.py:188 (sha256 97f573e5) | **match** |  |
| V09.6 | 1070 | a budget of 1 M EUR | investment budget (M EUR) | 1000000.0 | data/SRP1/SharedESS/SRP1_ESS_Params.json:2 (sha256 39106f93) budget (EUR); p515_s47_phase_b_record.py:190 (sha256 97f573e5) | **match** |  |
| V10.1 | 1074 | 10,000 cycles | baseline datasheet cycles N | 10000 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calibration.cycles_n | **match** |  |
| V10.2 | 1074 | 80 % depth of discharge | baseline depth of discharge (percent) | 0.8 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calibration.reference_dod_d | **match** |  |
| V10.3 | 1074 | cycles to 80 % retention | baseline retention at the cycle count (percent) | 0.8 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calibration.eol_retention_r; data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C2_calfade.eol_retention_r | **match** |  |
| V10.4 | 1074 | $k = 35,851$ | baseline k = N D / (-ln R) | 35851.3609417964 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.k_closed_form_by_arm.C2_calfade; T8 tables.ageing.rows[arm=C2_calfade].k; closed form from data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calibration = 35851.3609417964 | **approximate** | printed at 0 decimal place(s): 35,851 = 35851.3609417964 rounded |
| V10.5 | 1074 | $\phi^{Cal} = 0.985$ per year | calendar retention phi^Cal per year | 0.985 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calendar_retention_per_year; data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C2_calfade.calendar_retention_per_year | **match** |  |
| V10.6 | 1074 | 80 % retention over the 15-year calendar life | retention after the calendar life without cycling (percent) = phi^T | 0.7971562536487744 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calendar_retention_per_year ^ ageing.calendar_life_years = 0.985^15; data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json citations.calendar_fade.statement (sha256 8bda2a0a, blob 5d113e54): "phi_cal = 0.985 per year (0.80 at the 15-year calendar life)" | **approximate** | printed at 0 decimal place(s): 80 = 79.71562536487744 rounded |
| V10.7 | 1074 | 15-year calendar life | calendar life (years) in the retention statement | 15 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calendar_life_years | **match** |  |
| V10.8 | 1074 | $SoH^{Min} = 0.70$ | end-of-life floor SoH^Min (baseline) | 0.7 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.minimum_soh; data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.floor_row_lower_expected | **match** |  |
| V11.1 | 1074 | the same count without calendar fade | cycling-only sensitivity: the baseline count without calendar fade | {'C2': {'ageing_enabled': True, 'available_energy_soh_point': 'end', 'calendar_retention_per_year': 1.0, 'eol_retention_r': 0.8}} | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C2 | **match** |  |
| V11.2 | 1074 | the same count read as cycles to 50 % retention | C3 retention at the cycle count (percent) | {'C3_unit': 0.5, 'C3_midblock': 0.5} | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C3_unit.eol_retention_r; data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C3_midblock.eol_retention_r | **match** |  |
| V11.3 | 1074 | evaluated at the end of the block or at its midpoint | C3 SoH evaluation point | ('end', 'mid') | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C3_unit / C3_midblock.available_energy_soh_point | **match** | the mid-block formula itself is W175's CONFIRM point |
| V12.1 | 1074 | a second datasheet of 8,000 full cycles | C4 datasheet cycles (full cycles) | 8000 | data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json citations.cycle_life[EVE MB31] (sha256 8bda2a0a, blob 5d113e54): '8,000 full cycles to 70 % SOH at 0.5P, 25 C', "(cycles 8000, DoD 1.0, EOL 0.70) -> k = 22,430 = C4's k; also cites the 0.70 floor" | **match** | the run encodes this datasheet as the file's (N, D) = (10000, 0.8) with N x D = 8000 = 8,000 x 1.0, the only product the model consumes (k = N D / -ln R). The datasheet statement is as RECORDED in spec v18 (status "PROPOSED BY THE EXPERT; AUTHOR TO CONFIRM AND VERIFY"; the Planner has not accessed the datasheet); the [AUTHOR] comment at l. 1099-1100 still asks for the citation |
| V12.2 | 1074 | to 70 % retention | C4 retention at the cycle count (percent) | 0.7 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C4.eol_retention_r | **match** |  |
| V13.1 | 1074 | no ageing | no-ageing sensitivity | {'no_ageing': {'ageing_enabled': False, 'available_energy_soh_point': 'end', 'calendar_retention_per_year': 1.0, 'eol_retention_r': 0.5}} | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.no_ageing.ageing_enabled | **match** | implementation of the arm is W175's CONFIRM point |
| V13.2 | 1074 | the baseline with a 0.50 floor | floor of the 0.50-floor sensitivity | 0.5 | data/SRP1/Results/P515S53/w155_a64_cells/frozen_s53_a64_cells_spec_v1_44a2dce8.json cells.e_soh050.declaration.minimum_soh (sha256 44a2dce8, blob 6e83fd7a) (arm C2_calfade) | **match** |  |
| V21.1 | 1080 | the state of health that sets a block's available energy (end of block, or its midpoint) | SoH point definition (caption) | ['end', 'mid'] | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms[*].available_energy_soh_point | **match** | the formula is W175's CONFIRM point |
| V14.1 | 1087 | 10,000 | N^DS (cycles) | 10000 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calibration.cycles_n (the arms vary eol_retention_r only) | **match** |  |
| V14.2 | 1087 | 0.80 | delta^DS (DoD) | 0.8 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calibration.reference_dod_d | **match** |  |
| V14.3 | 1087 | 0.80 | R^DS (retention) | 0.8 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C2_calfade.eol_retention_r | **match** |  |
| V14.4 | 1087 | 35,851 | k_e | 35851.3609417964 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.k_closed_form_by_arm.C2_calfade; closed form N D / -ln R = 35851.3609417964; T8 tables.ageing.rows[arm=C2_calfade].k = 35851.3609417964 | **approximate** | printed at 0 decimal place(s): 35,851 = 35851.3609417964 rounded |
| V14.5 | 1087 | 0.985 | phi^Cal | 0.985 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C2_calfade.calendar_retention_per_year | **match** |  |
| V14.6 | 1087 | end | SoH point | end | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C2_calfade.available_energy_soh_point | **match** |  |
| V14.7 | 1087 | 0.70 | SoH^Min | 0.7 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.floor_row_lower_expected | **match** |  |
| V15.1 | 1088 | 10,000 | N^DS (cycles) | 10000 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calibration.cycles_n (the arms vary eol_retention_r only) | **match** |  |
| V15.2 | 1088 | 0.80 | delta^DS (DoD) | 0.8 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calibration.reference_dod_d | **match** |  |
| V15.3 | 1088 | 0.80 | R^DS (retention) | 0.8 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C2.eol_retention_r | **match** |  |
| V15.4 | 1088 | 35,851 | k_e | 35851.3609417964 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.k_closed_form_by_arm.C2; closed form N D / -ln R = 35851.3609417964; T8 tables.ageing.rows[arm=C2].k = 35851.3609417964 | **approximate** | printed at 0 decimal place(s): 35,851 = 35851.3609417964 rounded |
| V15.5 | 1088 | 1.000 | phi^Cal | 1.0 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C2.calendar_retention_per_year | **match** |  |
| V15.6 | 1088 | end | SoH point | end | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C2.available_energy_soh_point | **match** |  |
| V15.7 | 1088 | 0.70 | SoH^Min | 0.7 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.floor_row_lower_expected | **match** |  |
| V16.1 | 1089 | row label: 50 | row label: retention (percent) | 0.5 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C3_unit.eol_retention_r | **match** |  |
| V16.2 | 1089 | 10,000 | N^DS (cycles) | 10000 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calibration.cycles_n (the arms vary eol_retention_r only) | **match** |  |
| V16.3 | 1089 | 0.80 | delta^DS (DoD) | 0.8 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calibration.reference_dod_d | **match** |  |
| V16.4 | 1089 | 0.50 | R^DS (retention) | 0.5 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C3_unit.eol_retention_r | **match** |  |
| V16.5 | 1089 | 11,542 | k_e | 11541.560327111707 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.k_closed_form_by_arm.C3_unit; closed form N D / -ln R = 11541.560327111707; T8 tables.ageing.rows[arm=C3_unit].k = 11541.560327111707 | **approximate** | printed at 0 decimal place(s): 11,542 = 11541.560327111707 rounded |
| V16.6 | 1089 | 1.000 | phi^Cal | 1.0 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C3_unit.calendar_retention_per_year | **match** |  |
| V16.7 | 1089 | end | SoH point | end | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C3_unit.available_energy_soh_point | **match** |  |
| V16.8 | 1089 | 0.70 | SoH^Min | 0.7 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.floor_row_lower_expected | **match** |  |
| V17.1 | 1090 | row label: 50 | row label: retention (percent) | 0.5 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C3_midblock.eol_retention_r | **match** |  |
| V17.2 | 1090 | 10,000 | N^DS (cycles) | 10000 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calibration.cycles_n (the arms vary eol_retention_r only) | **match** |  |
| V17.3 | 1090 | 0.80 | delta^DS (DoD) | 0.8 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calibration.reference_dod_d | **match** |  |
| V17.4 | 1090 | 0.50 | R^DS (retention) | 0.5 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C3_midblock.eol_retention_r | **match** |  |
| V17.5 | 1090 | 11,542 | k_e | 11541.560327111707 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.k_closed_form_by_arm.C3_midblock; closed form N D / -ln R = 11541.560327111707; T8 tables.ageing.rows[arm=C3_midblock].k = 11541.560327111707 | **approximate** | printed at 0 decimal place(s): 11,542 = 11541.560327111707 rounded |
| V17.6 | 1090 | 1.000 | phi^Cal | 1.0 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C3_midblock.calendar_retention_per_year | **match** |  |
| V17.7 | 1090 | mid | SoH point | mid | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C3_midblock.available_energy_soh_point | **match** |  |
| V17.8 | 1090 | 0.70 | SoH^Min | 0.7 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.floor_row_lower_expected | **match** |  |
| V18.1 | 1091 | row label: 8,000 | row label: datasheet cycles | 8000.0 | data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json citations.cycle_life[EVE MB31].statement (sha256 8bda2a0a, blob 5d113e54) | **match** |  |
| V18.2 | 1091 | 8,000 | N^DS (datasheet cycles) | 8000.0 | data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json citations.cycle_life[EVE MB31].calibration (sha256 8bda2a0a, blob 5d113e54): "(cycles 8000, DoD 1.0, EOL 0.70) -> k = 22,430 = C4's k; also cites the 0.70 floor" | **match** | as run: the file's (N, D) = (10000, 0.8); N x D equal (8000 = 8000 x 1.0), the only product k uses; datasheet statement as recorded in spec v18 (proposed by the expert, author to confirm) |
| V18.3 | 1091 | 1.00 | delta^DS (datasheet DoD) | 1.0 | data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json citations.cycle_life[EVE MB31].calibration (sha256 8bda2a0a, blob 5d113e54): "(cycles 8000, DoD 1.0, EOL 0.70) -> k = 22,430 = C4's k; also cites the 0.70 floor" | **match** | as run: the file's (N, D) = (10000, 0.8); N x D equal (8000 = 8000 x 1.0), the only product k uses; datasheet statement as recorded in spec v18 (proposed by the expert, author to confirm) |
| V18.4 | 1091 | 0.70 | R^DS (retention) | 0.7 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C4.eol_retention_r | **match** |  |
| V18.5 | 1091 | 22,429 | k_e | 22429.38601645703 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.k_closed_form_by_arm.C4; closed form N D / -ln R = 22429.38601645703; T8 tables.ageing.rows[arm=C4].k = 22429.38601645703 | **approximate** | printed at 0 decimal place(s): 22,429 = 22429.38601645703 rounded |
| V18.6 | 1091 | 1.000 | phi^Cal | 1.0 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C4.calendar_retention_per_year | **match** |  |
| V18.7 | 1091 | end | SoH point | end | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C4.available_energy_soh_point | **match** |  |
| V18.8 | 1091 | 0.70 | SoH^Min | 0.7 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.floor_row_lower_expected | **match** |  |
| V20.1 | 1092 | $\infty$ | no ageing: k_e | {'ageing_enabled': False, 'k_field': 11541.560327111707} | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.no_ageing.ageing_enabled; p515_s44_campaign_harness.py:629 (sha256 9c6bda19) | **match** | ageing disabled (the harness docstring: SoH == 1 everywhere), the reading of k = infinity; the k field of the arm (11,541.56) is not consumed when ageing is off -- the implementation is W175's CONFIRM point |
| V20.2 | 1092 | 1.000 | no ageing: phi^Cal | 1.0 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.no_ageing.calendar_retention_per_year | **match** |  |
| V20.3 | 1092 | --- / --- / --- | no ageing: (N, delta, R), SoH point, SoH^Min | {'eol_retention_r_field': 0.5, 'soh_point_field': 'end'} | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.no_ageing | **match** | not applicable with ageing off (SoH == 1, so no floor binds); the fields the arm carries are inert |
| V19.1 | 1093 | row label: 0.50 | row label: floor | 0.5 | data/SRP1/Results/P515S53/w155_a64_cells/frozen_s53_a64_cells_spec_v1_44a2dce8.json cells.e_soh050.declaration.minimum_soh (sha256 44a2dce8, blob 6e83fd7a) | **match** |  |
| V19.2 | 1093 | 10,000 | N^DS (cycles) | 10000 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calibration.cycles_n (the arms vary eol_retention_r only) | **match** |  |
| V19.3 | 1093 | 0.80 | delta^DS (DoD) | 0.8 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calibration.reference_dod_d | **match** |  |
| V19.4 | 1093 | 0.80 | R^DS (retention) | 0.8 | data/SRP1/Results/P515S53/w155_a64_cells/frozen_s53_a64_cells_spec_v1_44a2dce8.json cells.e_soh050.declaration (sha256 44a2dce8, blob 6e83fd7a).eol_retention_r | **match** |  |
| V19.5 | 1093 | 35,851 | k_e | 35851.3609417964 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.k_closed_form_by_arm.C2_calfade; closed form N D / -ln R = 35851.3609417964 | **approximate** | printed at 0 decimal place(s): 35,851 = 35851.3609417964 rounded |
| V19.6 | 1093 | 0.985 | phi^Cal | 0.985 | data/SRP1/Results/P515S53/w155_a64_cells/frozen_s53_a64_cells_spec_v1_44a2dce8.json cells.e_soh050.declaration (sha256 44a2dce8, blob 6e83fd7a).calendar_retention_per_year | **match** |  |
| V19.7 | 1093 | end | SoH point | end | data/SRP1/Results/P515S53/w155_a64_cells/frozen_s53_a64_cells_spec_v1_44a2dce8.json cells.e_soh050.declaration (sha256 44a2dce8, blob 6e83fd7a).available_energy_soh_point | **match** |  |
| V19.8 | 1093 | 0.50 | SoH^Min | 0.5 | data/SRP1/Results/P515S53/w155_a64_cells/frozen_s53_a64_cells_spec_v1_44a2dce8.json cells.e_soh050.declaration.minimum_soh (sha256 44a2dce8, blob 6e83fd7a) | **match** |  |
| V22.1 | 1109 | IPOPT 3.14.18 | IPOPT version | 3.14.18 | data/SRP1/Results/P515S53/w159_closing_reads/w159_closing_reads.json c.c3.ipopt.version_output.stdout (sha256 a49ebbdc, blob 635f5762): 'Ipopt 3.14.18 (aarch64-apple-darwin24.5.0), ASL(20241111)'; data/SRP1/Results/P515S53/w159_closing_reads/w159_closing_reads.json c.c2.tally_d_4a82a64a_all_manifest_logs.versions (sha256 a49ebbdc, blob 635f5762): ['3.14.18'] (495 logs) | **match** | scope of the record: the W159 closing reads (2026-10-06) of the v6 campaign's logs and binary; the environment is recorded unchanged since the first v6 start, not for the earlier search campaigns |
| V22.2 | 1109 | Pyomo 6.9.5 | Pyomo version | 6.9.5 | data/SRP1/Results/P515S53/w159_closing_reads/w159_closing_reads.json c.c3.pyomo_version (sha256 a49ebbdc, blob 635f5762) | **match** |  |
| V22.3 | 1109 | Python 3.11.11 | Python version | 3.11.11 | data/SRP1/Results/P515S53/w159_closing_reads/w159_closing_reads.json c.c3.sys_version (sha256 a49ebbdc, blob 635f5762): '3.11.11 / packaged by conda-forge / (main, Dec  5 2024, 08:47:03) [Clang 18.1.8 ]' | **match** |  |
| V23.1 | 1110 | MA97 | network linear solver | {'case9': 'ma97', 'case33_1': 'ma97', 'case33_2': 'ma97', 'case33_3': 'ma97'} | data/SRP1/case9/case9_params.json:40 (sha256 f3eff050); data/SRP1/case33_1/case33_1_params.json:40 (sha256 8e9e5a53); data/SRP1/case33_2/case33_2_params.json:40 (sha256 31b5fedf); data/SRP1/case33_3/case33_3_params.json:40 (sha256 a19bd5b9); data/SRP1/Results/P515S53/w159_closing_reads/w159_closing_reads.json c.c2.tally_d_4a82a64a_all_manifest_logs.network (sha256 a49ebbdc, blob 635f5762): {'ma97': 6887} | **match** |  |
| V23.2 | 1110 | MA57 | storage-agent linear solver | ma57 | data/SRP1/SharedESS/SRP1_ESS_Params.json:37 (sha256 39106f93); data/SRP1/Results/P515S53/w159_closing_reads/w159_closing_reads.json c.c2.tally_d_4a82a64a_all_manifest_logs.esso (sha256 a49ebbdc, blob 635f5762): {'ma57': 429} | **match** |  |
| V23.3 | 1110 | single-threaded throughout | threading | OMP/MKL/OpenBLAS/vecLib/NumExpr threads = 1 | data/SRP1/Results/P515S53/w159_closing_reads/w159_closing_reads.json c.c1.all_table_evaluations_ran_with_OMP_NUM_THREADS_1 (sha256 a49ebbdc, blob 635f5762) = True; p515_s44_campaign_harness.py:320 (sha256 9c6bda19); p515_s44_campaign_harness.py:2233 (sha256 9c6bda19) | **match** | record scope: every evaluation the frozen tables use (W159); "throughout" beyond the tables (the search campaigns) runs through the same harness and its child refusal |
| V24.1 | 1111 | $10^{-5}$ | network convergence tolerance (tol) | {'case9': 1e-05, 'case33_1': 1e-05, 'case33_2': 1e-05, 'case33_3': 1e-05} | data/SRP1/case9/case9_params.json:37 (sha256 f3eff050); data/SRP1/case33_1/case33_1_params.json:37 (sha256 8e9e5a53); data/SRP1/case33_2/case33_2_params.json:37 (sha256 31b5fedf); data/SRP1/case33_3/case33_3_params.json:37 (sha256 a19bd5b9) | **match** |  |
| V24.2 | 1111 | $10^{-4}$ | network acceptable tolerance | {'case9': 0.0001, 'case33_1': 0.0001, 'case33_2': 0.0001, 'case33_3': 0.0001} | data/SRP1/case9/case9_params.json:38,50 (sha256 f3eff050); data/SRP1/case33_1/case33_1_params.json:38 (sha256 8e9e5a53); data/SRP1/case33_2/case33_2_params.json:38 (sha256 31b5fedf); data/SRP1/case33_3/case33_3_params.json:38 (sha256 a19bd5b9) | **match** |  |
| V24.3 | 1112 | $5 \times 10^{-4}$ | TSO complementarity tolerance (compl_inf_tol) | 0.0005 | data/SRP1/case9/case9_params.json:41 (sha256 f3eff050) | **match** |  |
| V24.4 | 1112 | $10^{-4}$ (DSOs) | DSO complementarity tolerance | {'case_files': {'case33_1': None, 'case33_2': None, 'case33_3': None}, 'ipopt_default': 0.0001} | network.py:632 (sha256 18acaa84); data/SRP1/case33_1/case33_1_params.json: no compl_inf_tol key; data/SRP1/case33_2/case33_2_params.json: no compl_inf_tol key; data/SRP1/case33_3/case33_3_params.json: no compl_inf_tol key | **match** | IPOPT default (no DSO case file sets the option) |
| V24.5 | 1112 | at most 500 iterations | network iteration limit (max_iter) | 500 | network.py:559 (sha256 18acaa84); case files: max_iter {'case9': None, 'case33_1': None, 'case33_2': None, 'case33_3': None} (no override) | **match** |  |
| V24.6 | 1113 | $10^{-6}$ | tight-tail complementarity tolerance | {'v6': {'compl_inf_tol': 1e-06, 'enabled': True}, '3x3': {'compl_inf_tol': 1e-06, 'enabled': True}} | data/SRP1/Results/P515S53/w142_resettle_v6/frozen_s53_resettle_spec_v6_96c23404.json (sha256 96c23404) inputs_in_force_now.configuration_now.convergence_depth_tail; data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha256 231558f0) configuration.convergence_depth_tail | **match** |  |
| V25.1 | 1113 | $10^{-10}$ | storage-agent tolerance | 1e-10 | shared_energy_storage_data.py:1082 (sha256 9acd095f); shared_energy_storage_data.py:108 (sha256 9acd095f); shared_energy_storage_data.py:1104 (sha256 9acd095f); data/SRP1/SharedESS/SRP1_ESS_Params.json:34 (sha256 39106f93) (file value, overridden) | **match** | applied after the file options to every ESSO solve |
| V25.2 | 1114 | $10^{-9}$ | storage-agent acceptable tolerance | 1e-09 | shared_energy_storage_data.py:1082 (sha256 9acd095f); shared_energy_storage_data.py:108 (sha256 9acd095f) | **match** |  |
| V26.1 | 1114 | ends at the iteration limit, infeasible or in a solver error | retry triggers | ['maxIterations', 'infeasible', 'internalSolverError'] | network.py:785 (sha256 18acaa84); network.py:786 (sha256 18acaa84); network.py:787 (sha256 18acaa84); shared_energy_storage_data.py:1218 (sha256 9acd095f) | **match** | network.py _is_recoverable_network_failure and shared_energy_storage_data _is_recoverable_shared_ess_failure test exactly these three termination conditions |
| V26.2 | 1114 | is retried ... and then once more | retry enabled for every agent (tiers 1 and 2) | {'case_file_recovery_blocks': {'case9': None, 'case33_1': None, 'case33_2': None, 'case33_3': None}, 'ess_recovery_block': None, 'defaults': (True, True)} | solver_parameters.py:23 (sha256 2801ef16); solver_parameters.py:24 (sha256 2801ef16); solver_parameters.py:53 (sha256 2801ef16) | **match** | no case file or ESS file carries a solver.recovery block |
| V26.3 | 1114 | retried as a cold restart with the agent's recovery settings | tier 1 = cold restart | warm_start_init_point = 'no' | network.py:976 (sha256 18acaa84); shared_energy_storage_data.py:1281 (sha256 9acd095f) | **match** |  |
| V26.6 | 1114 | DSOs: their primary settings | DSO recovery settings | {'case33_1': None, 'case33_2': {}, 'case33_3': {}} | data/SRP1/case33_1/case33_1_params.json solver.recovery_options = None; data/SRP1/case33_2/case33_2_params.json solver.recovery_options = {}; data/SRP1/case33_3/case33_3_params.json solver.recovery_options = {}; network.py:973 (sha256 18acaa84) | **match** | empty or absent recovery_options: the cold restart keeps the primary options (acceptable_tol 0.0001, acceptable_iter 5) |
| V26.9 | 1114 | and then once more with the adaptive barrier strategy | tier 2 | mu_strategy = 'adaptive', one attempt | network.py:999 (sha256 18acaa84); shared_energy_storage_data.py:1335 (sha256 9acd095f); network.py:994 (sha256 18acaa84) | **match** | tier 2 runs only after a failed tier-1 retry whose failure is itself recoverable; same recovery options plus mu_strategy adaptive |
| V26.4 | 1115 | TSO: acceptable tolerance $10^{-4}$ | TSO recovery acceptable_tol | 0.0001 | data/SRP1/case9/case9_params.json:49 (sha256 f3eff050) solver.recovery_options.acceptable_tol; network.py:972 (sha256 18acaa84) | **match** |  |
| V26.5 | 1115 | after one acceptable iteration | TSO recovery acceptable_iter | 1 | data/SRP1/case9/case9_params.json:49 (sha256 f3eff050) solver.recovery_options.acceptable_iter | **match** |  |
| V26.7 | 1116 | storage agent: acceptable tolerance $10^{-9}$ | storage-agent recovery acceptable_tol | 1e-09 | data/SRP1/SharedESS/SRP1_ESS_Params.json:39 (sha256 39106f93) solver.recovery_options = {'acceptable_tol': 0.0001, 'acceptable_iter': 1}; shared_energy_storage_data.py:1285 (sha256 9acd095f); shared_energy_storage_data.py:1082 (sha256 9acd095f) | **match** | the file's 1e-4 is replaced by ESSO_TOL_OVERRIDES (applied to the retry too) |
| V26.8 | 1116 | after one iteration | storage-agent recovery acceptable_iter | 1 | data/SRP1/SharedESS/SRP1_ESS_Params.json:39 (sha256 39106f93) solver.recovery_options.acceptable_iter | **match** |  |
| V27.1 | 1119 | $\sigma = 93,635,360$ | sigma (common objective scale) | 93635360.0 | data/SRP1/SRP1_params.json:49 (sha256 dbfdb2a0) | **match** | the 3x3 spec runs the same case file (data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha256 231558f0) configuration.case_file = data/SRP1/SRP1_params.json) |
| V28.1 | 1120 | $S^{ref} = 2.5$ MVA | S^ref (MVA) | 2.5 | data/SRP1/SRP1_params.json:50 (sha256 dbfdb2a0); shared_resources_planning.py:5028 (sha256 0610d745); shared_resources_planning.py: 12 lines divide by (2 * shared_ess_rating) | **match** | floor 0.1 MVA inactive while S_ref is set (_shared_ess_admm_normalization_mva returns reference_mva) |
| V29.1 | 1120 | 2.0 | R^I (case33_1, TN node 5) p.u. | 2.0 | network.py:83 (sha256 18acaa84); network.py:90 (sha256 18acaa84); shared_resources_planning.py:4222,5148,5417 (sha256 0610d745); case files data/SRP1/case33_1/case33_1_2025.json (sha256 5cde824b, blob 5922cb45), data/SRP1/case33_1/case33_1_2030.json (sha256 01144534, blob ae77ca16), data/SRP1/case33_1/case33_1_2035.json (sha256 674b2d00, blob 58211b8d), data/SRP1/case33_2/case33_2_2025.json (sha256 8d6bd989, blob c24bfddc), data/SRP1/case33_2/case33_2_2030.json (sha256 e4065145, blob d1924a42), data/SRP1/case33_2/c | **match** | fixed in every year file 2025, 2028, 2030, 2031, 2034, 2035, 2037: [(200.0, 100.0)]; incident branches [(('transformers', 1),)] |
| V29.2 | 1120 | 1.0 | R^I (case33_2, TN node 7) p.u. | 1.0 | network.py:83 (sha256 18acaa84); network.py:90 (sha256 18acaa84); shared_resources_planning.py:4222,5148,5417 (sha256 0610d745); case files data/SRP1/case33_1/case33_1_2025.json (sha256 5cde824b, blob 5922cb45), data/SRP1/case33_1/case33_1_2030.json (sha256 01144534, blob ae77ca16), data/SRP1/case33_1/case33_1_2035.json (sha256 674b2d00, blob 58211b8d), data/SRP1/case33_2/case33_2_2025.json (sha256 8d6bd989, blob c24bfddc), data/SRP1/case33_2/case33_2_2030.json (sha256 e4065145, blob d1924a42), data/SRP1/case33_2/c | **match** | fixed in every year file 2025, 2028, 2030, 2031, 2034, 2035, 2037: [(100.0, 100.0)]; incident branches [(('transformers', 1),)] |
| V29.3 | 1120 | 1.5 | R^I (case33_3, TN node 9) p.u. | 1.5 | network.py:83 (sha256 18acaa84); network.py:90 (sha256 18acaa84); shared_resources_planning.py:4222,5148,5417 (sha256 0610d745); case files data/SRP1/case33_1/case33_1_2025.json (sha256 5cde824b, blob 5922cb45), data/SRP1/case33_1/case33_1_2030.json (sha256 01144534, blob ae77ca16), data/SRP1/case33_1/case33_1_2035.json (sha256 674b2d00, blob 58211b8d), data/SRP1/case33_2/case33_2_2025.json (sha256 8d6bd989, blob c24bfddc), data/SRP1/case33_2/case33_2_2030.json (sha256 e4065145, blob d1924a42), data/SRP1/case33_2/c | **match** | fixed in every year file 2025, 2028, 2030, 2031, 2034, 2035, 2037: [(150.0, 100.0)]; incident branches [(('transformers', 1),)] |
| V29.4 | 1120 | 200 | R^I (case33_1) MVA | 200.0 | case files data/SRP1/case33_1/case33_1_2025.json (sha256 5cde824b, blob 5922cb45), data/SRP1/case33_1/case33_1_2030.json (sha256 01144534, blob ae77ca16), data/SRP1/case33_1/case33_1_2035.json (sha256 674b2d00, blob 58211b8d), data/SRP1/case33_2/case33_2_2025.json (sha256 8d6bd989, blob c24bfddc), data/SRP1/case33_2/case33_2_2030.json (sha256 e4065145, blob d1924a42), data/SRP1/case33_2/case33_2_2035.json (sha256 6888045f, blob bb4420b1), data/SRP1/case33_3/case33_3_2025.json (sha256 c9f430ce, blob ee92dfbc), data/ | **match** |  |
| V29.5 | 1120 | 100 | R^I (case33_2) MVA | 100.0 | case files data/SRP1/case33_1/case33_1_2025.json (sha256 5cde824b, blob 5922cb45), data/SRP1/case33_1/case33_1_2030.json (sha256 01144534, blob ae77ca16), data/SRP1/case33_1/case33_1_2035.json (sha256 674b2d00, blob 58211b8d), data/SRP1/case33_2/case33_2_2025.json (sha256 8d6bd989, blob c24bfddc), data/SRP1/case33_2/case33_2_2030.json (sha256 e4065145, blob d1924a42), data/SRP1/case33_2/case33_2_2035.json (sha256 6888045f, blob bb4420b1), data/SRP1/case33_3/case33_3_2025.json (sha256 c9f430ce, blob ee92dfbc), data/ | **match** |  |
| V29.10 | 1121 | 9 | TN node of case33_3 | 9 | data/SRP1/SRP1.json (sha256 61a794a7) DistributionNetworks[*].connection_node_id | **match** |  |
| V29.6 | 1121 | 150 | R^I (case33_3) MVA | 150.0 | case files data/SRP1/case33_1/case33_1_2025.json (sha256 5cde824b, blob 5922cb45), data/SRP1/case33_1/case33_1_2030.json (sha256 01144534, blob ae77ca16), data/SRP1/case33_1/case33_1_2035.json (sha256 674b2d00, blob 58211b8d), data/SRP1/case33_2/case33_2_2025.json (sha256 8d6bd989, blob c24bfddc), data/SRP1/case33_2/case33_2_2030.json (sha256 e4065145, blob d1924a42), data/SRP1/case33_2/case33_2_2035.json (sha256 6888045f, blob bb4420b1), data/SRP1/case33_3/case33_3_2025.json (sha256 c9f430ce, blob ee92dfbc), data/ | **match** |  |
| V29.7 | 1121 | 100 MVA base | base (MVA) of the per-unit ratings | [100.0] | case files data/SRP1/case33_1/case33_1_2025.json (sha256 5cde824b, blob 5922cb45), data/SRP1/case33_1/case33_1_2030.json (sha256 01144534, blob ae77ca16), data/SRP1/case33_1/case33_1_2035.json (sha256 674b2d00, blob 58211b8d), data/SRP1/case33_2/case33_2_2025.json (sha256 8d6bd989, blob c24bfddc), data/SRP1/case33_2/case33_2_2030.json (sha256 e4065145, blob d1924a42), data/SRP1/case33_2/case33_2_2035.json (sha256 6888045f, blob bb4420b1), data/SRP1/case33_3/case33_3_2025.json (sha256 c9f430ce, blob ee92dfbc), data/ | **match** |  |
| V29.8 | 1121 | 5 | TN node of case33_1 | 5 | data/SRP1/SRP1.json (sha256 61a794a7) DistributionNetworks[*].connection_node_id | **match** |  |
| V29.9 | 1121 | 7 | TN node of case33_2 | 7 | data/SRP1/SRP1.json (sha256 61a794a7) DistributionNetworks[*].connection_node_id | **match** |  |
| V30.1 | 1121 | $\kappa^{E} = \sigma/\operatorname{median}_b w_b$ | kappa^E formula | sigma_over_median_block_weight | data/SRP1/SRP1_params.json:51 (sha256 dbfdb2a0); shared_resources_planning.py:4062 (sha256 0610d745) | **match** |  |
| V30.2 | 1122 | 227,211 | kappa^E single-scenario instance | 227210.996652 | data/SRP1/SRP1_params.json:51 (sha256 dbfdb2a0); shared_resources_planning.py:4062 (sha256 0610d745); shared_resources_planning.py:4037 (sha256 0610d745); data/SRP1/SRP1.json (sha256 61a794a7); data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json (sha256 2a64e3c5) | **approximate** | printed at 0 decimal place(s): 227,211 = 227210.996652 rounded recomputed: sigma / median w_b, median 412.10751847261173 over 48 blocks |
| V30.3 | 1122 | 386,259 | kappa^E multi-scenario instance | 386258.694309 | data/SRP1/SRP1_params.json:51 (sha256 dbfdb2a0); shared_resources_planning.py:4062 (sha256 0610d745); shared_resources_planning.py:4037 (sha256 0610d745); data/SRP1/SRP1.json (sha256 61a794a7); data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json (sha256 2a64e3c5) | **approximate** | printed at 0 decimal place(s): 386,259 = 386258.694309 rounded recomputed: sigma / median w_b, median 242.4161873368304 over 80 blocks |
| V31.1 | 1123 | 0.0077 | rho^V_0 | {'case9': 0.0077, 'case33_1': 0.0077, 'case33_2': 0.0077, 'case33_3': 0.0077} | data/SRP1/SRP1_params.json:67 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:73 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:83 (sha256 dbfdb2a0) | **match** | one value for every agent |
| V31.2 | 1123 | 0.198 | rho^PF_0 | {'case9': 0.198, 'case33_1': 0.198, 'case33_2': 0.198, 'case33_3': 0.198} | data/SRP1/SRP1_params.json:67 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:73 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:83 (sha256 dbfdb2a0) | **match** | one value for every agent |
| V31.3 | 1123 | 0.01 | rho^E_0 | {'case9': 0.01, 'case33_1': 0.01, 'case33_2': 0.01, 'case33_3': 0.01, 'esso': 0.01} | data/SRP1/SRP1_params.json:67 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:73 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:83 (sha256 dbfdb2a0) | **match** | one value for every agent |
| V32.1 | 1123 | 5 | residual-balance ratio mu | 5.0 | data/SRP1/SRP1_params.json:53 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json (sha256 dbfdb2a0) admm.adaptive_penalty = True | **match** |  |
| V32.2 | 1124 | 3 | ratio for the decrease of rho^PF | 3.0 | data/SRP1/SRP1_params.json:54 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json (sha256 dbfdb2a0) admm.adaptive_penalty = True | **match** |  |
| V32.3 | 1124 | 1.5 | increase / decrease factor | 1.5 | data/SRP1/SRP1_params.json:55 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:56 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json (sha256 dbfdb2a0) admm.adaptive_penalty = True | **match** |  |
| V32.4 | 1124 | 10^{-4} | rho lower bound | 0.0001 | data/SRP1/SRP1_params.json:57 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json (sha256 dbfdb2a0) admm.adaptive_penalty = True | **match** |  |
| V32.5 | 1124 | 10^{4} | rho upper bound | 10000.0 | data/SRP1/SRP1_params.json:58 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json (sha256 dbfdb2a0) admm.adaptive_penalty = True | **match** |  |
| V32.6 | 1124 | ten | per-channel freeze after unchanged cycles | 10 | data/SRP1/SRP1_params.json:59 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json (sha256 dbfdb2a0) admm.adaptive_penalty = True | **match** |  |
| V32.7 | 1125 | 200 | freeze backstop cycle | 200 | data/SRP1/SRP1_params.json:60 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json (sha256 dbfdb2a0) admm.adaptive_penalty = True | **match** |  |
| V32.8 | 1125 | one | storage-channel exemption: dual ratio below | 1.0 | data/SRP1/SRP1_params.json:62 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json (sha256 dbfdb2a0) admm.adaptive_penalty = True | **match** |  |
| V32.9 | 1126 | five | storage-channel exemption: consecutive cycles | 5 | data/SRP1/SRP1_params.json:62 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json (sha256 dbfdb2a0) admm.adaptive_penalty = True | **match** |  |
| V33.1 | 1126 | 10^{-5} | Boyd eps_abs | 1e-05 | data/SRP1/SRP1_params.json:42 (sha256 dbfdb2a0); admm_parameters.py:363 (sha256 40d0e972) | **match** |  |
| V33.2 | 1127 | 10^{-4} | Boyd eps_rel | 0.0001 | data/SRP1/SRP1_params.json:42 (sha256 dbfdb2a0); admm_parameters.py:363 (sha256 40d0e972) | **match** |  |
| V34.1 | 1127 | Anderson acceleration | Anderson acceleration enabled | True | data/SRP1/SRP1_params.json:123 (sha256 dbfdb2a0); admm_anderson_acceleration.py:3 (sha256 24cf3bea) | **match** |  |
| V34.2 | 1127 | 5 | AA memory | 5 | data/SRP1/SRP1_params.json:123 (sha256 dbfdb2a0) | **match** |  |
| V34.3 | 1127 | 10^{-10} | AA regularisation (Tikhonov) | 1e-10 | data/SRP1/SRP1_params.json:124 (sha256 dbfdb2a0) | **match** |  |
| V35.1 | 1127 | ten consecutive passing cycles | production exit: consecutive passing cycles | {'SRP1_params': 10, '3x3_spec': 10} | data/SRP1/SRP1_params.json:45 (sha256 dbfdb2a0); data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha256 231558f0) required_consecutive_cycles; shared_resources_planning.py:3390 (sha256 0610d745) | **match** |  |
| V35.2 | 1128 | a cap of 500 cycles | cycle cap at the multi-scenario instance | 500 | data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha256 231558f0) cap | **match** |  |
| V36.1 | 1129 | $\alpha = 0.5$ | commitment premium alpha (3x3) | ['{"alpha": 0.5, "floor": null}'] | data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha256 231558f0) candidates[*].interface_deviation_premium | **match** |  |
| V36.2 | 1129 | $9 \times 10^{4}$ | voltage regularisation weight (3x3, solver-side) | 90000.0 | definitions.py:75 (sha256 7e5719af); model_construction_helpers.py:1940 (sha256 a20a224f); shared_resources_planning.py:849 (sha256 0610d745) | **match** | excluded from the reported Q |
| V37.1 | 1131 | 0.07 | delta_R | 0.07 | settling_criterion.py:55 (sha256 fbe5550b) | **match** |  |
| V37.2 | 1132 | 259,375.33 | reference value V (EUR) | 259375.33 | settling_criterion.py:56 (sha256 fbe5550b) | **match** |  |
| V37.3 | 1132 | 4,539.07 | tau = delta_R V / 4 (EUR) | 4539.0682750000005 | settling_criterion.py:57 (sha256 fbe5550b) | **approximate** | printed at 2 decimal place(s): 4,539.07 = 4539.0682750000005 rounded |
| V37.4 | 1132 | 30 | P_max (cycles) | 30 | data/SRP1/Results/P515S53/w142_resettle_v6/frozen_s53_resettle_spec_v6_96c23404.json (sha256 96c23404) stop_rule.p_max = {'L': 60, 'source': 'W102 x0 P_hat 29, W103 unit P_hat 30 (as fc791891)', 'value': 30} | **match** | L_MONO = 2 P_max = 60 |
| V38.1 | 1133 | capped 100 cycles after that run's stopping cycle | cap of a continued evaluation (cycles after the earlier run's stop) | N_old + 100 | data/SRP1/Results/P515S53/w142_resettle_v6/frozen_s53_resettle_spec_v6_96c23404.json (sha256 96c23404) cells[*].cap_rule / gated / N_old; data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) cells[*].cap_rule; data/SRP1/Results/P515S53/w155_a64_cells/frozen_s53_a64_cells_spec_v1_44a2dce8.json cells[*].cap_rule (sha256 44a2dce8, blob 6e83fd7a) | **match** | continued (replay-gated on the earlier run, cap = N_old + 100): 31 cells; fresh (dynamic cap): 14 cells; violations [] |
| V38.2 | 1134 | $k_0 + 109$ | cap of a fresh evaluation: cycles after k0 | 109 | data/SRP1/Results/P515S53/w142_resettle_v6/frozen_s53_resettle_spec_v6_96c23404.json (sha256 96c23404) cells[*].cap_rule / gated / N_old; data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) cells[*].cap_rule; data/SRP1/Results/P515S53/w155_a64_cells/frozen_s53_a64_cells_spec_v1_44a2dce8.json cells[*].cap_rule (sha256 44a2dce8, blob 6e83fd7a); settling_criterion_v2.py:76 (sha256 3db13b0e) | **match** |  |
| V38.3 | 1134 | 300 | cap of a fresh evaluation: ceiling | 300 | data/SRP1/Results/P515S53/w142_resettle_v6/frozen_s53_resettle_spec_v6_96c23404.json (sha256 96c23404) cells[*].cap_rule / gated / N_old; data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) cells[*].cap_rule; data/SRP1/Results/P515S53/w155_a64_cells/frozen_s53_a64_cells_spec_v1_44a2dce8.json cells[*].cap_rule (sha256 44a2dce8, blob 6e83fd7a); settling_criterion_v2.py:77 (sha256 3db13b0e) | **match** |  |
| V39.1 | 1134 | $\Delta_0 = 4$ | Delta_0 (variant A) | {'S47': 4, 'S51': 4} | p515_s47_phase_b_record.py:194 (sha256 97f573e5); data/SRP1/Results/P515S47/campaign_s47_phase_b/campaign_spec_s47_phase_b_8cfa264e.json extra.poll_design.delta_0 (sha256 8cfa264e, blob 77baa5d0); data/SRP1/Results/P515S51/campaign_s51_f2_phase_b/campaign_spec_s51_f2_phase_b_5ce295e1.json extra.poll_design.delta_0 (sha256 5ce295e1, blob e254e90c) | **match** |  |
| V39.2 | 1135 | budgets of 20 new evaluations | budget of new evaluations, variant A | {'S47': 20, 'S51': 20} | p515_s47_phase_b_record.py:200 (sha256 97f573e5); data/SRP1/Results/P515S47/campaign_s47_phase_b/campaign_spec_s47_phase_b_8cfa264e.json extra.poll_design.max_new_evaluations (sha256 8cfa264e, blob 77baa5d0); data/SRP1/Results/P515S51/campaign_s51_f2_phase_b/campaign_spec_s51_f2_phase_b_5ce295e1.json extra.poll_design.max_new_evaluations (sha256 5ce295e1, blob e254e90c) | **match** |  |
| V39.3 | 1135 | the two variant-A searches | number of variant-A searches | ['S47', 'S51'] | data/SRP1/Results/P515S47/campaign_s47_phase_b/campaign_spec_s47_phase_b_8cfa264e.json extra.poll_design.design (sha256 8cfa264e, blob 77baa5d0) = orthomads_n_plus_1_neg; data/SRP1/Results/P515S51/campaign_s51_f2_phase_b/campaign_spec_s51_f2_phase_b_5ce295e1.json extra.poll_design.design (sha256 5ce295e1, blob e254e90c) = orthomads_n_plus_1_neg; data/SRP1/Results/P515S53/campaign_s53_f2_certificate_r1/campaign_spec_s53_f2_certificate_r1_803571c0.json extra.poll_design.design (sha256 803571c0, blob 6a35a084) = ortho | **match** |  |
| V39.4 | 1135 | 60 for the variant-B certificate | budget of new evaluations, variant B | 60 | p515_s53_f2_certificate.py:156 (sha256 59f75ce4); data/SRP1/Results/P515S53/campaign_s53_f2_certificate_r1/campaign_spec_s53_f2_certificate_r1_803571c0.json extra.poll_design.max_new_evaluations (sha256 803571c0, blob 6a35a084) | **match** |  |
| V39.5 | 1136 | a completion cap of 30 points | completion cap (points) | {'S47': 30, 'S51': 30, 'S53': 30} | p515_s47_phase_b_record.py:205 (sha256 97f573e5); p515_s53_f2_certificate.py:155 (sha256 59f75ce4); data/SRP1/Results/P515S47/campaign_s47_phase_b/campaign_spec_s47_phase_b_8cfa264e.json extra.poll_design.completion_cap (sha256 8cfa264e, blob 77baa5d0); data/SRP1/Results/P515S51/campaign_s51_f2_phase_b/campaign_spec_s51_f2_phase_b_5ce295e1.json extra.poll_design.completion_cap (sha256 5ce295e1, blob e254e90c); data/SRP1/Results/P515S53/campaign_s53_f2_certificate_r1/campaign_spec_s53_f2_certificate_r1_803571c0.json | **match** |  |
| V39.6 | 1136 | $\sigma_Q = 18,449.66$ EUR | sigma_Q (EUR) | [18449.663947025518] | data/SRP1/Results/P515S47/campaign_s47_phase_b/campaign_results.json sigma_Q_eur (every occurrence) (sha256 906f9da3, blob 292f7f22); data/SRP1/Results/P515S51/campaign_s51_f2_phase_b/campaign_results.json sigma_Q_eur (every occurrence) (sha256 2cdd5d60, blob f4d8428d); data/SRP1/Results/P515S53/campaign_s53_f2_certificate_r1/campaign_results.json sigma_Q_eur (every occurrence) (sha256 61b9c501, blob b55b1f99); data/SRP1/Results/P515S47/campaign_s47_phase_b/campaign_results.json A6 (sha256 906f9da3, blob 292f7f22): | **approximate** | printed at 2 decimal place(s): 18,449.66 = 18449.663947025518 rounded one value in all three search campaigns |
| V40.1 | 1138 | two static arrangements | number of static arrangements | ['passive', 'price_taker'] | T6 tables.benchmark.w160_additions.arms_in_full (keys) | **match** | the arm definitions are W175's CONFIRM point |
| V41.1 | 1143 | the best of three solver starts | solver starts per arrangement | {'passive': 3, 'price_taker': 3} | T6 tables.benchmark.w160_additions.arms_in_full.{passive,price_taker}.Q_by_start (count of starts) | **match** | cold, perturbed, warm_from_certified; each arm = the minimum over its starts |

### Files cited by the value table (sha256 and git blob at HEAD; code citations carry file:line and sha256)

| path | sha256 | git blob |
|---|---|---|
| STEP4_DFO_METHOD.md | d05bae1e66f56b27 | b548d1712653 |
| admm_anderson_acceleration.py | 24cf3bea9b07725a | 8ab3945488f9 |
| admm_parameters.py | 40d0e972dcadfb4d | 98e778178595 |
| data/SRP1/Results/P515S47/campaign_s47_phase_b/campaign_results.json | 906f9da353ebe215 | 292f7f22e9c1 |
| data/SRP1/Results/P515S47/campaign_s47_phase_b/campaign_spec_s47_phase_b_8cfa264e.json | 8cfa264e7ab0d92a | 77baa5d018a6 |
| data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json | 8bda2a0a94d6f6f2 | 5d113e54dcf0 |
| data/SRP1/Results/P515S51/campaign_s51_f2_phase_b/campaign_results.json | 2cdd5d6048919980 | f4d8428d959d |
| data/SRP1/Results/P515S51/campaign_s51_f2_phase_b/campaign_spec_s51_f2_phase_b_5ce295e1.json | 5ce295e17c5165ea | e254e90c34a4 |
| data/SRP1/Results/P515S53/campaign_s53_f2_certificate_r1/campaign_results.json | 61b9c501a2dc3a95 | b55b1f99e375 |
| data/SRP1/Results/P515S53/campaign_s53_f2_certificate_r1/campaign_spec_s53_f2_certificate_r1_803571c0.json | 803571c0efcf828e | 6a35a084315c |
| data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json | 84775dc41428ba6a | 455bc13f07bb |
| data/SRP1/Results/P515S53/w142_resettle_v6/frozen_s53_resettle_spec_v6_96c23404.json | 96c234045f15fa4f | 8dcc55880706 |
| data/SRP1/Results/P515S53/w155_a64_cells/frozen_s53_a64_cells_spec_v1_44a2dce8.json | 44a2dce8c2bb12f4 | 6e83fd7a9188 |
| data/SRP1/Results/P515S53/w159_closing_reads/w159_closing_reads.json | a49ebbdc7219c805 | 635f5762f40e |
| data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json | 2a64e3c5bd06c30c | dd8dd781cb48 |
| data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json | 231558f070579732 | 63d7f5e40eba |
| data/SRP1/SRP1.json | 61a794a7ce7a3fb9 | 9a6d7373c833 |
| data/SRP1/SRP1_params.json | dbfdb2a07d12bfed | a80056d0e3fd |
| data/SRP1/SharedESS/SRP1_ESS_Params.json | 39106f934bf3edbf | 45034b489d46 |
| data/SRP1/case33_1/case33_1_2025.json | 5cde824b69378604 | 5922cb450866 |
| data/SRP1/case33_1/case33_1_2030.json | 0114453432afd3aa | ae77ca162a58 |
| data/SRP1/case33_1/case33_1_2035.json | 674b2d00809d93af | 58211b8d1a83 |
| data/SRP1/case33_1/case33_1_params.json | 8e9e5a536fd1ef2b | 6976e6952131 |
| data/SRP1/case33_2/case33_2_2025.json | 8d6bd989f05925b7 | c24bfddc6af3 |
| data/SRP1/case33_2/case33_2_2030.json | e4065145c214b1b5 | d1924a42c4d1 |
| data/SRP1/case33_2/case33_2_2035.json | 6888045f33a07f33 | bb4420b14303 |
| data/SRP1/case33_2/case33_2_params.json | 31b5fedf87b96724 | d53faa5e1846 |
| data/SRP1/case33_3/case33_3_2025.json | c9f430ce32b0b959 | ee92dfbcb025 |
| data/SRP1/case33_3/case33_3_2030.json | de8aa07d29d4be87 | eaac8137bbe9 |
| data/SRP1/case33_3/case33_3_2035.json | f73ae82b1b240a8d | fb3a28144432 |
| data/SRP1/case33_3/case33_3_params.json | a19bd5b9d18a26de | 66a2a6b258fe |
| data/SRP1/case9/case9_params.json | f3eff0502c606a90 | 975332dd52ce |
| definitions.py | 7e5719af129df111 | 2f649080e8e7 |
| model_construction_helpers.py | a20a224f11008de8 | 5b0de5fca140 |
| network.py | 18acaa84af7a0438 | ffa61728b2bc |
| p515_s44_campaign_harness.py | 9c6bda1969eca917 | 5e38a38413bc |
| p515_s45_a1_campaign.py | 812c5abf8b874dfe | 2f6cb3e7b310 |
| p515_s47_phase_b_record.py | 97f573e52fb57483 | c96a4c9300f5 |
| p515_s53_f2_certificate.py | 59f75ce46b42f410 | c769cf75338c |
| settling_criterion.py | fbe5550b34a4062f | b890245c2f5f |
| settling_criterion_v2.py | 3db13b0eb7a093f4 | d1c906274a34 |
| shared_energy_storage.py | d45c72d0db9a3743 | 7199052ed194 |
| shared_energy_storage_data.py | 9acd095fb5e50e9b | ff4162c2348a |
| shared_resources_planning.py | 0610d745a41c86c7 | 050bc411715e |
| solver_parameters.py | 2801ef16853d7e63 | 2171f834c2b4 |

### Tokens in l. 1066-1150 that are not values (assigned by rule)

| line | written | status | reason |
|---:|---|---|---|
| 1070 | 0 | unchecked | formula constant in sections 3.5-3.6 (new, round 3 C.4-C.5) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b  |
| 1074 | 4.3 | unchecked | section / table / figure / equation / algorithm number written in prose; main.tex at this commit: section 4.3 exists ("Computational Performance") |
| 1099 | two | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1100 | MB31 | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1100 | 8,000 | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1101 | W175 | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1101 | 1 | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1101 | 2 | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1101 | 2 | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1101 | 625 | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1101 | 626 | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1102 | 1 | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1108 | one | unchecked | number word in sections 3.5-3.6 (new, round 3 C.4-C.5) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1122 | single | unchecked | compound adjective (single-scenario) |
| 1133 | one | unchecked | number word in sections 3.5-3.6 (new, round 3 C.4-C.5) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1138 | 4.4 | unchecked | section / table / figure / equation / algorithm number written in prose; main.tex at this commit: section 4.4 exists ("Operational Planning Results") |
| 1144 | W175 | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1144 | 49 | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1144 | 57 | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1144 | 58 | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1145 | 1 | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1145 | 0 | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1145 | 0 | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1146 | three | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1147 | 0 | unchecked | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec3_5_6; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |

## MISMATCH (all files)

| file | line | col | written | counterpart at written precision | source | note |
|---|---:|---:|---|---|---|---|
| section2_expert_draft.tex (draft) | 74 | 23 | 2n | n+1 | p515_s47_phase_b_record.py:199 (sha256 97f573e5); p515_s47_phase_b_record.py:509 (sha256 97f573e5); p515_s51_f2_phase_b.py:803 (sha256 87ab3b2d) | Algorithm 1 writes "the 2n OrthoMADS directions"; both planning searches as run (Phase B and the F2 Phase B) poll n + 1 directions (the n Householder columns and minus their sum; STEP4 section 3 allows "2n ... or the n+1 minimal positive basis") |
| section2_expert_draft.tex (draft) | 79 | 34 | 1 | unit poll | p515_s47_phase_b_record.py:607 (sha256 97f573e5); p515_s47_phase_b_record.py:204 (sha256 97f573e5); p515_s47_phase_b_record.py:205 (sha256 97f573e5) | Algorithm 1 adds the completion set "If /P/ < n + 1"; the code adds it at every poll of unit size (UNIT_POLL_COMPLETION and delta == DELTA_MIN), refusing (stop for review) above 30 points |
| section2_expert_draft.tex (draft) | 233 | 37 | third | third-last to last | settling_criterion_v2.py:166 (sha256 3db13b0e) (inherited by settling_criterion_v6); W163 N11 written 'the three most recent turning points: from the third-last to the last' | "measured between the first and third turning points": the code measures P_hat between the third-last and the last turning point (the three most recent); the two agree only while exactly three turning points have occurred since k0 |
| section2_expert_draft.tex (draft) | 496 | 31 | 22,430 | 22429 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.k_closed_form_by_arm.C4 (= T8 C4 k 22429.386) | 8,000 x 1.0 / -ln 0.70 = 22,429.39 |

## Approximate / no table counterpart / no source found

- `main.tex` line 1074: `35,851` (approximate) -- printed at 0 decimal place(s): 35,851 = 35851.3609417964 rounded
- `main.tex` line 1074: `80` (approximate) -- printed at 0 decimal place(s): 80 = 79.71562536487744 rounded
- `main.tex` line 1087: `35,851` (approximate) -- printed at 0 decimal place(s): 35,851 = 35851.3609417964 rounded
- `main.tex` line 1088: `35,851` (approximate) -- printed at 0 decimal place(s): 35,851 = 35851.3609417964 rounded
- `main.tex` line 1089: `11,542` (approximate) -- printed at 0 decimal place(s): 11,542 = 11541.560327111707 rounded
- `main.tex` line 1090: `11,542` (approximate) -- printed at 0 decimal place(s): 11,542 = 11541.560327111707 rounded
- `main.tex` line 1091: `22,429` (approximate) -- printed at 0 decimal place(s): 22,429 = 22429.38601645703 rounded
- `main.tex` line 1093: `35,851` (approximate) -- printed at 0 decimal place(s): 35,851 = 35851.3609417964 rounded
- `main.tex` line 1122: `227,211` (approximate) -- printed at 0 decimal place(s): 227,211 = 227210.996652 rounded recomputed: sigma / median w_b, median 412.10751847261173 over 48 blocks
- `main.tex` line 1122: `386,259` (approximate) -- printed at 0 decimal place(s): 386,259 = 386258.694309 rounded recomputed: sigma / median w_b, median 242.4161873368304 over 80 blocks
- `main.tex` line 1132: `4,539.07` (approximate) -- printed at 2 decimal place(s): 4,539.07 = 4539.0682750000005 rounded
- `main.tex` line 1136: `18,449.66` (approximate) -- printed at 2 decimal place(s): 18,449.66 = 18449.663947025518 rounded one value in all three search campaigns
- `response_to_reviewers_draft.tex` line 277: `single` (no table counterpart) -- no committed record states the machine count; stands on the author's attestation (Addendum 67; Addendum 68 Decision 5 "Single machine stands on the author's attestation")

## Version-2 declarations re-located (version 2 -> version 3)

| v2 id | file | fragment found (times) | outcome | v3 declaration | reason / what replaced it |
|---|---|---:|---|---|---|
| L03v2 | response_to_reviewers_draft.tex | 1 | carried | L03v2 | fragment found once; evaluated unchanged |
| L05v2 | response_to_reviewers_draft.tex | 1 | carried | L05v2 | fragment found once; evaluated unchanged |
| L08v2 | response_to_reviewers_draft.tex | 1 | carried | L08v2 | fragment found once; evaluated unchanged |
| L22v2 | response_to_reviewers_draft.tex | 1 | carried | L22v2 | fragment found once; evaluated unchanged |
| L37v2 | response_to_reviewers_draft.tex | 1 | carried | L37v2 | fragment found once; evaluated unchanged |
| L40v2 | response_to_reviewers_draft.tex | 1 | carried | L40v2 | fragment found once; evaluated unchanged |
| L43v2 | response_to_reviewers_draft.tex | 0 | removed | - | R3.5: the clause "The sentence of the submitted version announcing 0.5 % and 2 % cases ... was removed" is deleted (Addendum 69 D8: the reviewers never saw that text); no number left |
| L52v2 | response_to_reviewers_draft.tex | 1 | carried | L52v2 | fragment found once; evaluated unchanged |
| L21v2 | response_to_reviewers_draft.tex | 1 | carried | L21v2 | fragment found once; evaluated unchanged |
| L32v2 | response_to_reviewers_draft.tex | 1 | carried | L32v2 | fragment found once; evaluated unchanged |
| L45v2 | response_to_reviewers_draft.tex | 1 | carried | L45v2 | fragment found once; evaluated unchanged |
| L54 | response_to_reviewers_draft.tex | 1 | carried | L54 | fragment found once; evaluated unchanged |
| L55 | response_to_reviewers_draft.tex | 1 | carried | L55 | fragment found once; evaluated unchanged |
| L56 | response_to_reviewers_draft.tex | 1 | carried | L56 | fragment found once; evaluated unchanged |
| L57 | response_to_reviewers_draft.tex | 1 | carried | L57 | fragment found once; evaluated unchanged |
| LC1 | response_to_reviewers_draft.tex | 1 | carried | LC1 | fragment found once; evaluated unchanged |
| LC2 | response_to_reviewers_draft.tex | 1 | carried | LC2 | fragment found once; evaluated unchanged |
| L01 | response_to_reviewers_draft.tex | 1 | carried | L01 | fragment found once; evaluated unchanged |
| L02 | response_to_reviewers_draft.tex | 1 | carried | L02 | fragment found once; evaluated unchanged |
| L04 | response_to_reviewers_draft.tex | 1 | carried | L04 | fragment found once; evaluated unchanged |
| L06 | response_to_reviewers_draft.tex | 1 | carried | L06 | fragment found once; evaluated unchanged |
| L07 | response_to_reviewers_draft.tex | 1 | carried | L07 | fragment found once; evaluated unchanged |
| L09 | response_to_reviewers_draft.tex | 1 | carried | L09 | fragment found once; evaluated unchanged |
| L10 | response_to_reviewers_draft.tex | 1 | carried | L10 | fragment found once; evaluated unchanged |
| L11 | response_to_reviewers_draft.tex | 1 | carried | L11 | fragment found once; evaluated unchanged |
| L12 | response_to_reviewers_draft.tex | 1 | carried | L12 | fragment found once; evaluated unchanged |
| L13 | response_to_reviewers_draft.tex | 1 | carried | L13 | fragment found once; evaluated unchanged |
| L14 | response_to_reviewers_draft.tex | 0 | removed | L304, L301 | R1.2 (round 2 B.6 second part): "the horizon (three representative years standing for five-year blocks)" -> "... five-year blocks at the single-scenario instance; five three-year blocks at the multi-scenario instance" |
| L15 | response_to_reviewers_draft.tex | 1 | carried | L15 | fragment found once; evaluated unchanged |
| L16 | response_to_reviewers_draft.tex | 1 | carried | L16 | fragment found once; evaluated unchanged |
| L17 | response_to_reviewers_draft.tex | 1 | carried | L17 | fragment found once; evaluated unchanged |
| L18 | response_to_reviewers_draft.tex | 1 | carried | L18 | fragment found once; evaluated unchanged |
| L19 | response_to_reviewers_draft.tex | 1 | carried | L19 | fragment found once; evaluated unchanged |
| L20 | response_to_reviewers_draft.tex | 1 | carried | L20 | fragment found once; evaluated unchanged |
| L23 | response_to_reviewers_draft.tex | 1 | carried | L23 | fragment found once; evaluated unchanged |
| L24 | response_to_reviewers_draft.tex | 1 | carried | L24 | fragment found once; evaluated unchanged |
| L25 | response_to_reviewers_draft.tex | 1 | carried | L25 | fragment found once; evaluated unchanged |
| L26 | response_to_reviewers_draft.tex | 1 | carried | L26 | fragment found once; evaluated unchanged |
| L27 | response_to_reviewers_draft.tex | 1 | carried | L27 | fragment found once; evaluated unchanged |
| L28 | response_to_reviewers_draft.tex | 1 | carried | L28 | fragment found once; evaluated unchanged |
| L29 | response_to_reviewers_draft.tex | 1 | carried | L29 | fragment found once; evaluated unchanged |
| L30 | response_to_reviewers_draft.tex | 1 | carried | L30 | fragment found once; evaluated unchanged |
| L31 | response_to_reviewers_draft.tex | 1 | carried | L31 | fragment found once; evaluated unchanged |
| L34 | response_to_reviewers_draft.tex | 1 | carried | L34 | fragment found once; evaluated unchanged |
| L35 | response_to_reviewers_draft.tex | 1 | carried | L35 | fragment found once; evaluated unchanged |
| L36 | response_to_reviewers_draft.tex | 1 | carried | L36 | fragment found once; evaluated unchanged |
| L38 | response_to_reviewers_draft.tex | 1 | carried | L38 | fragment found once; evaluated unchanged |
| L39 | response_to_reviewers_draft.tex | 1 | carried | L39 | fragment found once; evaluated unchanged |
| L41 | response_to_reviewers_draft.tex | 1 | carried | L41 | fragment found once; evaluated unchanged |
| L42 | response_to_reviewers_draft.tex | 1 | carried | L42 | fragment found once; evaluated unchanged |
| L44 | response_to_reviewers_draft.tex | 1 | carried | L44 | fragment found once; evaluated unchanged |
| L46 | response_to_reviewers_draft.tex | 1 | carried | L46 | fragment found once; evaluated unchanged |
| L47 | response_to_reviewers_draft.tex | 1 | carried | L47 | fragment found once; evaluated unchanged |
| L48 | response_to_reviewers_draft.tex | 1 | carried | L48 | fragment found once; evaluated unchanged |
| L49 | response_to_reviewers_draft.tex | 1 | carried | L49 | fragment found once; evaluated unchanged |
| L50 | response_to_reviewers_draft.tex | 1 | carried | L50 | fragment found once; evaluated unchanged |
| L51 | response_to_reviewers_draft.tex | 1 | carried | L51 | fragment found once; evaluated unchanged |
| L53 | response_to_reviewers_draft.tex | 1 | carried | L53 | fragment found once; evaluated unchanged |
| CL1 | cover_letter.tex | 1 | carried | CL1 | fragment found once; evaluated unchanged |
| CL2 | cover_letter.tex | 1 | carried | CL2 | fragment found once; evaluated unchanged |
| M01 | main.tex | 1 | carried | M01 | fragment found once; evaluated unchanged |
| M02 | main.tex | 1 | carried | M02 | fragment found once; evaluated unchanged |
| M03 | main.tex | 1 | carried | M03 | fragment found once; evaluated unchanged |
| M04 | main.tex | 1 | carried | M04 | fragment found once; evaluated unchanged |
| M05 | main.tex | 1 | carried | M05 | fragment found once; evaluated unchanged |
| M06 | main.tex | 1 | carried | M06 | fragment found once; evaluated unchanged |
| M07 | main.tex | 1 | carried | M07 | fragment found once; evaluated unchanged |
| M17 | main.tex | 1 | carried | M17 | fragment found once; evaluated unchanged |
| M08 | main.tex | 1 | carried | M08 | fragment found once; evaluated unchanged |
| M09 | main.tex | 1 | carried | M09 | fragment found once; evaluated unchanged |
| M10 | main.tex | 1 | carried | M10 | fragment found once; evaluated unchanged |
| M11 | main.tex | 1 | carried | M11 | fragment found once; evaluated unchanged |
| M12 | main.tex | 0 | removed | V10, V14-V18 | the red section 3.4 paragraph (70 % floor) is deleted (round 3 C.3); the floor SoH^Min = 0.70 is printed in section 3.5 (text and ageing table) |
| M13 | main.tex | 0 | removed | - | deleted with the red paragraph (Addendum 68 D6; round 3 C.3): the 60 % / 80 % floors were never evaluated; no number left |
| M14 | main.tex | 0 | removed | V10 | the red calendar paragraph (1 %/yr, 14 %, 15-year, 20 years) is deleted (round 3 C.3); the calendar retention 0.985 / 80 % over the 15-year calendar life is printed in section 3.5 |
| M15 | main.tex | 0 | removed | - | deleted with M14 (round 3 C.3): the 0.5 % / 2.0 % calendar cases were never run; no number left |
| M16 | main.tex | 1 | carried | M16 | fragment found once; evaluated unchanged |
| S201 | main.tex | 1 | carried | S201 | fragment found once; evaluated unchanged |
| S202 | main.tex | 1 | carried | S202 | fragment found once; evaluated unchanged |
| S203 | main.tex | 1 | carried | S203 | fragment found once; evaluated unchanged |
| S204 | main.tex | 1 | carried | S204 | fragment found once; evaluated unchanged |
| S205 | main.tex | 0 | removed | S318 | "(measured between the first and third turning points)" -> "the number of cycles spanned by the three most recent turning points" (round 2 B.2; Addendum 69 D3) |
| S206 | main.tex | 1 | carried | S206 | fragment found once; evaluated unchanged |
| S207 | main.tex | 1 | carried | S207 | fragment found once; evaluated unchanged |
| S208 | main.tex | 1 | carried | S208 | fragment found once; evaluated unchanged |
| S209 | main.tex | 1 | carried | S209 | fragment found once; evaluated unchanged |
| S210 | main.tex | 0 | removed | S326 | "over a window of $2P_{\max}$ cycles with $P_{\max}$" -> "... cycles starting after $k_0$ with $P_{\max}$" (round 2 B.2) |
| S211 | main.tex | 1 | carried | S211 | fragment found once; evaluated unchanged |
| S212 | main.tex | 1 | carried | S212 | fragment found once; evaluated unchanged |
| S213 | main.tex | 1 | carried | S213 | fragment found once; evaluated unchanged |
| S214 | main.tex | 1 | carried | S214 | fragment found once; evaluated unchanged |
| S215 | main.tex | 1 | carried | S215 | fragment found once; evaluated unchanged |
| S216 | main.tex | 1 | carried | S216 | fragment found once; evaluated unchanged |
| S217 | main.tex | 0 | removed | S323 | "called determinate when it exceeds" -> "... when it is at least" (round 2; Addendum 69 D3) |
| S218 | main.tex | 0 | removed | S324 | "three times the larger of that evaluation's consensus gap and settling slack" -> "three times the largest of their consensus gaps and settling slacks" (round 2; Addendum 69 D3) |
| S219 | main.tex | 0 | removed | S304, S307, S313, S314 | Algorithm 1 rewritten to the search as run (round 2 A.1-A.4; Addendum 69 D4): variant A n + 1 directions, variant B 2n |
| S220 | main.tex | 0 | removed | S305 | the completion line rewritten: "every admissible lattice point within one unit step of z^inc in the infinity-norm" (variant A, at Delta = 1) |
| S221 | main.tex | 1 | carried | S221 | fragment found once; evaluated unchanged |
| S222 | main.tex | 1 | carried | S222 | fragment found once; evaluated unchanged |
| S223 | main.tex | 1 | carried | S223 | fragment found once; evaluated unchanged |
| S224 | main.tex | 1 | carried | S224 | fragment found once; evaluated unchanged |
| S225 | main.tex | 0 | removed | S305, S307 | "\If{$/\mathcal{P}/ < n + 1$}" -> variant A "\If{$\Delta = 1$}" (unit-box completion) and variant B "if fewer than $n+1$ distinct admissible points result" (round 2 A.2, round 3 B.c) |
| S226 | main.tex | 0 | removed | S310 | "$\Delta^p \gets 2\Delta^p$" -> "$\Delta \gets 2\Delta$ (variant A) or $1$ (variant B)" |
| S227 | main.tex | 0 | removed | S311 | "\lIf{$\Delta^p = 1$}" -> "\lIf{$\Delta = 1$}" |
| S228 | main.tex | 0 | removed | S312 | "$\Delta^p \gets \Delta^p / 2$" -> "$\Delta \gets \Delta/2$" |
| S229 | main.tex | 1 | carried | S229 | fragment found once; evaluated unchanged |
| S230 | main.tex | 0 | removed | S322 | "evaluated under the production exit --- ten consecutive cycles passing the residual test" -> "... under the production exit (ten consecutive passing cycles with every local solve successful)" (round 2 B.3) |
| d201 | section2_expert_draft.tex | 1 | carried | d201 | fragment found once; evaluated unchanged |
| d202 | section2_expert_draft.tex | 1 | carried | d202 | fragment found once; evaluated unchanged |
| d203 | section2_expert_draft.tex | 1 | carried | d203 | fragment found once; evaluated unchanged |
| d204 | section2_expert_draft.tex | 1 | carried | d204 | fragment found once; evaluated unchanged |
| d205 | section2_expert_draft.tex | 1 | carried | d205 | fragment found once; evaluated unchanged |
| d206 | section2_expert_draft.tex | 1 | carried | d206 | fragment found once; evaluated unchanged |
| d207 | section2_expert_draft.tex | 1 | carried | d207 | fragment found once; evaluated unchanged |
| d208 | section2_expert_draft.tex | 1 | carried | d208 | fragment found once; evaluated unchanged |
| d209 | section2_expert_draft.tex | 1 | carried | d209 | fragment found once; evaluated unchanged |
| d210 | section2_expert_draft.tex | 1 | carried | d210 | fragment found once; evaluated unchanged |
| d211 | section2_expert_draft.tex | 1 | carried | d211 | fragment found once; evaluated unchanged |
| d212 | section2_expert_draft.tex | 1 | carried | d212 | fragment found once; evaluated unchanged |
| d213 | section2_expert_draft.tex | 1 | carried | d213 | fragment found once; evaluated unchanged |
| d214 | section2_expert_draft.tex | 1 | carried | d214 | fragment found once; evaluated unchanged |
| d215 | section2_expert_draft.tex | 1 | carried | d215 | fragment found once; evaluated unchanged |
| d216 | section2_expert_draft.tex | 1 | carried | d216 | fragment found once; evaluated unchanged |
| d217 | section2_expert_draft.tex | 1 | carried | d217 | fragment found once; evaluated unchanged |
| d218 | section2_expert_draft.tex | 1 | carried | d218 | fragment found once; evaluated unchanged |
| d219 | section2_expert_draft.tex | 1 | carried | d219 | fragment found once; evaluated unchanged |
| d220 | section2_expert_draft.tex | 1 | carried | d220 | fragment found once; evaluated unchanged |
| d221 | section2_expert_draft.tex | 1 | carried | d221 | fragment found once; evaluated unchanged |
| d222 | section2_expert_draft.tex | 1 | carried | d222 | fragment found once; evaluated unchanged |
| d223 | section2_expert_draft.tex | 1 | carried | d223 | fragment found once; evaluated unchanged |
| d224 | section2_expert_draft.tex | 1 | carried | d224 | fragment found once; evaluated unchanged |
| d225 | section2_expert_draft.tex | 1 | carried | d225 | fragment found once; evaluated unchanged |
| d226 | section2_expert_draft.tex | 1 | carried | d226 | fragment found once; evaluated unchanged |
| d227 | section2_expert_draft.tex | 1 | carried | d227 | fragment found once; evaluated unchanged |
| d228 | section2_expert_draft.tex | 1 | carried | d228 | fragment found once; evaluated unchanged |
| dC01 | section2_expert_draft.tex | 1 | carried | dC01 | fragment found once; evaluated unchanged |
| dC02 | section2_expert_draft.tex | 1 | carried | dC02 | fragment found once; evaluated unchanged |
| dC03 | section2_expert_draft.tex | 1 | carried | dC03 | fragment found once; evaluated unchanged |
| dC04 | section2_expert_draft.tex | 1 | carried | dC04 | fragment found once; evaluated unchanged |
| dC05 | section2_expert_draft.tex | 1 | carried | dC05 | fragment found once; evaluated unchanged |
| dC06 | section2_expert_draft.tex | 1 | carried | dC06 | fragment found once; evaluated unchanged |
| dC07 | section2_expert_draft.tex | 1 | carried | dC07 | fragment found once; evaluated unchanged |
| dC08 | section2_expert_draft.tex | 1 | carried | dC08 | fragment found once; evaluated unchanged |
| dC09 | section2_expert_draft.tex | 1 | carried | dC09 | fragment found once; evaluated unchanged |
| dC10 | section2_expert_draft.tex | 1 | carried | dC10 | fragment found once; evaluated unchanged |

## Reviewer quotations against the reviewers' document

Document: `manuscript_review/Reviewers Comments.docx` sha256 `c7afdc8b`, word/document.xml sha256 `c054751b`, 32 paragraphs (W171a normalisation).

| quote | segment | letter lines | chars | result | match ratio | docx paragraph | start | differences (letter -> document) |
|---:|---:|---|---:|---|---:|---:|---|---|
| 0 | 0 | 63 | 51 | verbatim | 1.000 | 0 | Regarding the Abstract, the following are relevant |  |
| 0 | 1 | 65 | 246 | verbatim | 1.000 | 0 | The Authors could tell the reader a little more ab |  |
| 0 | 2 | 66 | 116 | verbatim | 1.000 | 3 | The size of the test systems on which the framewor |  |
| 0 | 3 | 67 | 476 | differs | 0.994 | 5 | The lines below could also be improved: "Relative  | delete l.67: ...d also be improved: ['"' -> '']Relative to uncoordi...; delete l.67: ...g voltage violations['."' -> ''] Does 18.25% refer t... |
| 0 | 4 | 68 | 84 | verbatim | 1.000 | 7 | What type of battery ESS is the reference? This co |  |
| 0 | 5 | 69 | 294 | verbatim | 1.000 | 9 | The battery ESS is shared by both the Transmission |  |
| 0 | 6 | 70 | 97 | verbatim | 1.000 | 11 | Improving the Abstract taking into consideration t |  |
| 1 | 0 | 87 | 197 | verbatim | 1.000 | 13 | Several uncertainties will accompany the evolution |  |
| 1 | 1 | 89 | 88 | verbatim | 1.000 | 13 | It could be good for this type of Paper to elabora |  |
| 1 | 2 | 90 | 97 | verbatim | 1.000 | 15 | Additionally, could some sensitivity analysis be u |  |
| 2 | 0 | 98 | 47 | verbatim | 1.000 | 16 | Figure 2 could be improved in size for clarity. |  |
| 3 | 0 | 107 | 797 | verbatim | 1.000 | 16 | The claimed "bi-level" formulation is not clearly  |  |
| 3 | 1 | 108 | 638 | verbatim | 1.000 | 22 | The scenario treatment is conceptually problematic |  |
| 4 | 0 | 121 | 613 | verbatim | 1.000 | 22 | The interaction between ADMM, Benders decompositio |  |
| 5 | 0 | 127 | 648 | verbatim | 1.000 | 22 | The Benders cuts are not mathematically justified. |  |
| 5 | 1 | 128 | 508 | verbatim | 1.000 | 22 | The source of the sensitivity coefficients is uncl |  |
| 5 | 2 | 129 | 534 | verbatim | 1.000 | 22 | The feasibility cuts are not valid as written. The |  |
| 6 | 0 | 143 | 603 | verbatim | 1.000 | 22 | The ADMM convergence claim is weak for the stated  |  |
| 6 | 1 | 144 | 474 | verbatim | 1.000 | 22 | The adaptive ADMM penalty update is not justified. |  |
| 7 | 0 | 171 | 227 | verbatim | 1.000 | 22 | The literature review must include foundational re |  |
| 8 | 0 | 181 | 684 | verbatim | 1.000 | 22 | The degradation model is too simplified for the st |  |
| 9 | 0 | 191 | 612 | verbatim | 1.000 | 22 | The ESS active/reactive/apparent power relationshi |  |
| 9 | 1 | 192 | 576 | verbatim | 1.000 | 22 | The shared ESS formulation does not visibly contai |  |
| 10 | 0 | 210 | 1099 | verbatim | 1.000 | 23 | Regarding the master-problem objective in Section  |  |
| 11 | 0 | 218 | 1254 | verbatim | 1.000 | 24 | Regarding the Benders optimality cut in Section 2. |  |
| 12 | 0 | 231 | 924 | differs | 0.999 | 24 | Regarding Eqs. (21)-(31) in Section 2.3.2 and Appe | replace l.231: ... generic function h(['.' -> '⋅']), which is insuffic... |
| 13 | 0 | 239 | 1137 | verbatim | 1.000 | 26 | Regarding the degradation formulation in Section 2 |  |
| 14 | 0 | 246 | 940 | verbatim | 1.000 | 27 | Regarding the degradation model in Section 2.3.2,  |  |
| 15 | 0 | 253 | 801 | verbatim | 1.000 | 28 | Regarding insight (iii) in Section 4.6 and the rel |  |
| 16 | 0 | 261 | 820 | verbatim | 1.000 | 29 | Fig. 3 reports the Benders objective in , whereas  |  |
| 16 | 1 | 262 | 865 | verbatim | 1.000 | 30 | Regarding the representative-day setting in Sectio |  |

Quotation tokens: {'reviewer quotation, verified': 62}

## section 2 (revised, rounds 1-3) (main.tex l. 315-909): every number with its source

| line | written | status | check | counterpart (at written precision) | source / reason |
|---:|---|---|---|---|---|
| 319 | two | unchecked | not a figure |  | compound adjective (two-stage) |
| 319 | single | unchecked | not a figure |  | article sense ("a single ...") |
| 358 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 358 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 359 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 373 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 377 | 4 | unchecked | cross-reference |  | section / table / figure / equation / algorithm number written in prose; main.tex at this commit: section 4 exists ("Results") |
| 379 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 379 | two | unchecked | not a figure |  | compound adjective (two-stage) |
| 386 | two | unchecked | not a figure |  | compound adjective (two-stage) |
| 391 | two | unchecked | method statement |  | number word in section 2 (revised, rounds 1-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 413 | one | unchecked | method statement |  | number word in section 2 (revised, rounds 1-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 413 | one | unchecked | method statement |  | number word in section 2 (revised, rounds 1-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 413 | 2 | match | S301 | 2 | p515_s47_phase_b_record.py:191 (sha256 97f573e5); p515_s47_phase_b_record.py:185 (sha256 97f573e5) |
| 413 | +1 | match | S301 | +1 | p515_s47_phase_b_record.py:191 (sha256 97f573e5); p515_s47_phase_b_record.py:185 (sha256 97f573e5) |
| 414 | single | unchecked | not a figure |  | compound adjective (single-node) |
| 415 | 4 | unchecked | cross-reference |  | section / table / figure / equation / algorithm number written in prose; main.tex at this commit: section 4 exists ("Results") |
| 417 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 417 | 4 | match | S302 | 4 | p515_s47_phase_b_record.py:194 (sha256 97f573e5); data/SRP1/Results/P515S47/campaign_s47_phase_b/campaign_spec_s47_phase_b_8cfa264e.json extra.poll_design.delta_0 (sha256 8cfa264e, blob 77baa5d0); data/SRP1/Results/P515S51/campaign_s51_f2_phase_b/campaign_spec |
| 417 | 1 | match | S302 | 1 | p515_s53_f2_certificate.py:154 (sha256 59f75ce4); p515_s47_phase_b_record.py:195 (sha256 97f573e5); data/SRP1/Results/P515S53/campaign_s53_f2_certificate_r1/campaign_spec_s53_f2_certificate_r1_803571c0.json extra.poll_design.delta (sha256 803571c0, blob 6a35a0 |
| 417 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 419 | 60 | match | S303 | 60 | p515_s47_phase_b_record.py:201 (sha256 97f573e5); p515_s53_f2_certificate.py:157 (sha256 59f75ce4); data/SRP1/Results/P515S47/campaign_s47_phase_b/campaign_spec_s47_phase_b_8cfa264e.json extra.poll_design.max_polls (sha256 8cfa264e, blob 77baa5d0); data/SRP1/R |
| 422 | 1 | match | S304 | n + 1 | p515_s47_phase_b_record.py:199 (sha256 97f573e5); p515_s47_phase_b_record.py:509 (sha256 97f573e5); data/SRP1/Results/P515S47/campaign_s47_phase_b/campaign_spec_s47_phase_b_8cfa264e.json extra.poll_design (sha256 8cfa264e, blob 77baa5d0); data/SRP1/Results/P51 |
| 424 | 1 | match | S305 | 1 | p515_s47_phase_b_record.py:607 (sha256 97f573e5); p515_s47_phase_b_record.py:204 (sha256 97f573e5); data/SRP1/Results/P515S47/campaign_s47_phase_b/campaign_spec_s47_phase_b_8cfa264e.json extra.poll_design.unit_poll_completion (sha256 8cfa264e, blob 77baa5d0) |
| 425 | one | match | S305 | one | p515_s47_phase_b_record.py:46 (sha256 97f573e5) (COMPLETION_RULE) |
| 425 | 30 | match | S305 | 30 | p515_s47_phase_b_record.py:205 (sha256 97f573e5); p515_s47_phase_b_record.py:650 (sha256 97f573e5) |
| 428 | 1 | match | S306 | 1 | p515_s53_f2_certificate.py:154 (sha256 59f75ce4); data/SRP1/Results/P515S53/campaign_s53_f2_certificate_r1/campaign_spec_s53_f2_certificate_r1_803571c0.json extra.poll_design.delta (sha256 803571c0, blob 6a35a084) |
| 429 | 2n | match | S307 | 2n | p515_s53_f2_certificate.py:151 (sha256 59f75ce4); p515_s53_f2_certificate.py:153 (sha256 59f75ce4); data/SRP1/Results/P515S53/campaign_s53_f2_certificate_r1/campaign_spec_s53_f2_certificate_r1_803571c0.json extra.poll_design (sha256 803571c0, blob 6a35a084) |
| 429 | 1 | match | S307 | n + 1 | p515_s53_f2_certificate.py:152 (sha256 59f75ce4); data/SRP1/Results/P515S53/campaign_s53_f2_certificate_r1/campaign_spec_s53_f2_certificate_r1_803571c0.json extra.poll_design.min_feasible_poll_points (sha256 803571c0, blob 6a35a084) |
| 429 | 30 | match | S307 | 30 | p515_s53_f2_certificate.py:155 (sha256 59f75ce4); data/SRP1/Results/P515S53/campaign_s53_f2_certificate_r1/campaign_spec_s53_f2_certificate_r1_803571c0.json extra.poll_design.completion_cap (sha256 803571c0, blob 6a35a084) |
| 432 | ten | match | S308 | 10 | data/SRP1/Results/P515S47/campaign_s47_phase_b/campaign_spec_s47_phase_b_8cfa264e.json required_consecutive_cycles (sha256 8cfa264e, blob 77baa5d0); data/SRP1/Results/P515S51/campaign_s51_f2_phase_b/campaign_spec_s51_f2_phase_b_5ce295e1.json required_consecuti |
| 432 | 500 | match | S308 | 500 | data/SRP1/Results/P515S47/campaign_s47_phase_b/campaign_spec_s47_phase_b_8cfa264e.json cap (sha256 8cfa264e, blob 77baa5d0); data/SRP1/Results/P515S51/campaign_s51_f2_phase_b/campaign_spec_s51_f2_phase_b_5ce295e1.json cap (sha256 5ce295e1, blob e254e90c); data |
| 432 | ten | match | S308 | 10 | p515_s47_phase_b_record.py:245 (sha256 97f573e5) |
| 432 | two | match | S309 | 2 | p515_s47_phase_b_record.py:202 (sha256 97f573e5); p515_s47_phase_b_record.py:203 (sha256 97f573e5); data/SRP1/Results/P515S53/campaign_s53_f2_certificate_r1/campaign_spec_s53_f2_certificate_r1_803571c0.json extra.poll_design.barrier_stop (sha256 803571c0, blob |
| 432 | one | unchecked | S309 |  | "in one poll" (the scope of the per-poll count) |
| 432 | three | match | S309 | 3 | p515_s47_phase_b_record.py:202 (sha256 97f573e5); p515_s47_phase_b_record.py:203 (sha256 97f573e5); data/SRP1/Results/P515S53/campaign_s53_f2_certificate_r1/campaign_spec_s53_f2_certificate_r1_803571c0.json extra.poll_design.barrier_stop (sha256 803571c0, blob |
| 435 | 2 | match | S310 | 2 | p515_s47_phase_b_record.py:720 (sha256 97f573e5); data/SRP1/Results/P515S47/campaign_s47_phase_b/campaign_spec_s47_phase_b_8cfa264e.json extra.poll_design.success (sha256 8cfa264e, blob 77baa5d0) |
| 435 | 1 | match | S310 | 1 | data/SRP1/Results/P515S53/campaign_s53_f2_certificate_r1/campaign_spec_s53_f2_certificate_r1_803571c0.json extra.poll_design.on_success (sha256 803571c0, blob 6a35a084) |
| 437 | 1 | match | S311 | 1 | p515_s47_phase_b_record.py:195 (sha256 97f573e5); STEP4_DFO_METHOD.md:123 (sha256 d05bae1e) |
| 438 | 2 | match | S312 | 2 | p515_s47_phase_b_record.py:730 (sha256 97f573e5); STEP4_DFO_METHOD.md:190 (sha256 d05bae1e) |
| 444 | W173 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 444 | s47 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 444 | s51 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 445 | 1 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 445 | 30 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 445 | 20 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 445 | s47 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 445 | s51 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 445 | 60 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 445 | s53 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 445 | 1 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 446 | six | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 446 | s53 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 449 | one | unchecked | method statement |  | number word in section 2 (revised, rounds 1-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 449 | one | unchecked | method statement |  | number word in section 2 (revised, rounds 1-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 449 | one | unchecked | method statement |  | number word in section 2 (revised, rounds 1-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 449 | Two | match | S313 | 2 | data/SRP1/Results/P515S47/campaign_s47_phase_b/campaign_spec_s47_phase_b_8cfa264e.json extra.poll_design.design (sha256 8cfa264e, blob 77baa5d0); data/SRP1/Results/P515S51/campaign_s51_f2_phase_b/campaign_spec_s51_f2_phase_b_5ce295e1.json extra.poll_design.des |
| 449 | 1 | match | S313 | n + 1 | data/SRP1/Results/P515S47/campaign_s47_phase_b/campaign_spec_s47_phase_b_8cfa264e.json extra.poll_design.design (sha256 8cfa264e, blob 77baa5d0); data/SRP1/Results/P515S51/campaign_s51_f2_phase_b/campaign_spec_s51_f2_phase_b_5ce295e1.json extra.poll_design.des |
| 449 | 2n | match | S314 | 2n | data/SRP1/Results/P515S53/campaign_s53_f2_certificate_r1/campaign_spec_s53_f2_certificate_r1_803571c0.json extra.poll_design.design (sha256 803571c0, blob 6a35a084) |
| 449 | two | unchecked | method statement |  | number word in section 2 (revised, rounds 1-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 449 | 4.7 | unchecked | cross-reference |  | section / table / figure / equation / algorithm number written in prose; main.tex at this commit: section 4.7 DOES NOT EXIST |
| 449 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 449 | fourteen | match | S315 | 14 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item2.x0.b_box (sha256 0aa337fc, blob 05932598) {'n_feasible': 14, 'n_evaluated': 14, 'n_not_evaluated': 0, 'box_equals_final_poll_set': True} |
| 449 | seventeen | match | S316 | 17 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item2.F2.b_box (sha256 0aa337fc, blob 05932598).n_evaluated_in_search |
| 449 | 61 | match | S316 | 61 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item2.F2.b_box (sha256 0aa337fc, blob 05932598).n_feasible |
| 449 | ten | match | S316 | 10 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item2.F2.b_box (sha256 0aa337fc, blob 05932598).n_in_final_poll_set |
| 449 | seven | match | S316 | 7 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item2.F2.b_box (sha256 0aa337fc, blob 05932598).n_cached_outside_poll_set |
| 449 | thirteen | match | S317 | 13 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item2.F2.c_L_claims_box_neighbours (sha256 0aa337fc, blob 05932598).n |
| 449 | twelve | match | S317 | 12 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item2.F2.c_L_claims_box_neighbours (sha256 0aa337fc, blob 05932598).n_positive_plan_better |
| 449 | seven | match | S317 | 7 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item2.F2.c_L_claims_box_neighbours (sha256 0aa337fc, blob 05932598).n_positive_determinate |
| 449 | five | match | S317 | 5 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item2.F2.c_L_claims_box_neighbours (sha256 0aa337fc, blob 05932598).n_positive_within_bar |
| 452 | W173 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 452 | fourteen | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 452 | 0 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 452 | twelve | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 452 | F2 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 452 | T1 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 453 | W171b | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 453 | T1 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 453 | 5 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 453 | 61 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 453 | F2 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 453 | 44 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 453 | 17 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 454 | 12 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 454 | 17 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 459 | single | unchecked | not a figure |  | article sense ("a single ...") |
| 473 | 1 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 497 | +1 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 504 | +1 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 543 | 2 | unchecked | M07 |  | cited typical duration range (\cite{nrel_ess_costs}); the model bounds are E/P in [2.0, 4.0] h (SRP1_ESS_Params) |
| 543 | 10 | unchecked | M07 |  | cited typical duration range (\cite{nrel_ess_costs}); the model bounds are E/P in [2.0, 4.0] h (SRP1_ESS_Params) |
| 545 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 545 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 545 | 0.25 | match | S201 | 0.25 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 545 | 0.5 | match | S201 | 0.5 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 545 | 2 | match | S201 | 2.0 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 545 | 4 | match | S201 | 4.0 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 545 | 3 | unchecked | cross-reference |  | section / table / figure / equation / algorithm number written in prose; main.tex at this commit: section 3 exists ("Case Study") |
| 556 | 1 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 608 | 1 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 623 | 10^{-5} | match | S202 | 1e-05 | W163 N2 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): data/SRP1/SRP1_params.json admm.tol.boyd.eps_abs; W163 status match, written '10⁻⁵', value at written precision '1e-05' |
| 623 | 10^{-4} | match | S202 | 0.0001 | W163 N3 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): data/SRP1/SRP1_params.json admm.tol.boyd.eps_rel; W163 status match, written '10⁻⁴', value at written precision '0.0001' |
| 623 | 15 | match | S203 | 15 | W163 N4 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W101 three-reference summary (62bdeafe) reports.x0.s_signed (k€), measured from the old residual-rule certificate N; W163 status match,  |
| 623 | 21 | match | S203 | 21 | W163 N5 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W101 three-reference summary (62bdeafe) reports.n7_4h_e1.s_signed (k€), measured from the old residual-rule certificate N; W163 status m |
| 624 | one | unchecked | method statement |  | number word in section 2 (revised, rounds 1-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 630 | three | match | S204 | 3 | W163 N7 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v2.SettlingRuleV2.evaluate: at_least_3_turning_points = len(T) >= 3 (called by settling_criterion_v6.SettlingRuleV6.e |
| 630 | three | match | S318 | three most recent | settling_criterion_v2.py:166 (sha256 3db13b0e) (inherited by v6) |
| 630 | 100 | match | S319 | 100 | settling_criterion.py:58 (sha256 fbe5550b) |
| 630 | three | match | S319 | 3 | settling_criterion.py:59 (sha256 fbe5550b) |
| 631 | 10 | match | S206 | 10 | W163 N8 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v6.SWING_FLOOR = TAU / 10 (= GROWTH_TEST_FLOOR = TURNING_POINT_FLOOR); v6 spec stop_rule.swing_floor.F; W163 status m |
| 632 | 20 | match | S207 | 20 | W163 N9 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.W_MIN; v6 spec stop_rule.W.oscillatory; W163 status match, written '20', value at written precision '20' |
| 632 | 1.1 | match | S207 | 1.1 | W163 N10 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.W_FACTOR; W163 status match, written '1.1', value at written precision '1.1' |
| 633 | 2 | match | S208 | 2 | W163 N12 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v2.GAP_BOUND = TAU / 2 (v6 GAP_BOUND); W163 status match, written '2', value at written precision '2' |
| 634 | four | match | S209 | 4 | W163 N13 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v5.METRICS (v6) and v6 spec stop_rule.clean_rule.metric_table; W163 status match, written 'four', value at written p |
| 634 | ten | match | S209 | 10 | W163 N14 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v5.CLEAN_FACTOR (v6); v6 spec stop_rule.clean_rule.factor; PRIMARY_ATTEMPT; W163 status match, written '10', value a |
| 639 | 2P_ | match | S326 | 60 | W163 N15 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): v6 spec stop_rule.p_max.L and stop_rule.W.monotone ("L = L_MONO = 2 * P_MAX = 60"); W163 status match, written '2 P_max = 60', value at |
| 639 | 100 | match | S320 | 100 | settling_criterion.py:58 (sha256 fbe5550b); settling_criterion.py:32 (sha256 fbe5550b) |
| 639 | 2P_ | match | S211 | 60 | W163 N17 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): v6 spec stop_rule.constants.MONOTONE_LAST_STEP_CLAUSE "abs(dQ_k) * L_MONO <= TAU" with L_MONO 60; settling_criterion_v2 last_step_times |
| 639 | 4 | match | S321 | 4 | main.tex at the declared commit: the enumerate list of the certification rule (5 items) |
| 639 | 5 | match | S321 | 5 | main.tex at the declared commit: the enumerate list of the certification rule (5 items) |
| 639 | single | unchecked | not a figure |  | compound adjective (single-scenario) |
| 639 | ten | match | S322 | 10 | data/SRP1/SRP1_params.json:45 (sha256 dbfdb2a0) = 10; data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha256 231558f0) required_consecutive_cycles = 10; shared_resources_planning.py:3390 (sha256 0610d74 |
| 639 | 4.5 | unchecked | cross-reference |  | section / table / figure / equation / algorithm number written in prose; main.tex at this commit: section 4.5 exists ("Impact of Planning Horizon Discretization") |
| 643 | 4 | match | S212 | 4 | W163 N18 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.TAU = DELTA_R * R_REF / 4.0 (constants() formula); W163 status match, written '4', value at written precision '4' |
| 643 | 0.07 | match | S212 | 0.07 | W163 N19 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.DELTA_R; v6 spec stop_rule.constants.DELTA_R; W163 status match, written '0.07', value at written precision '0.07' |
| 643 | 259,375.33 | match | S213 | 259375.33 / 259375.33 | W163 N20 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.R_REF; W101 summary expert_P2.V_old (the SRP1 value Q(x0, N) − Q(unit, N) at the certificates in force when v39 froz |
| 643 | 4,539.07 | match | S213 | 4539.07 | W163 C1 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): constants.TAU; W163 status match, written '4,539.07', value at written precision '4539.07' |
| 643 | two | match | S214 | 2 | W163 N22 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): T11: R = V / V_SRP1_settled (a ratio of two values); W163 status match, written 'two', value at written precision '2' |
| 643 | four | match | S214 | 4 | W163 N21 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): TAU divisor 4 (N18); T11 rows: V = Q(0) − Q(unit) (two evaluations per value) and R = V / V_SRP1 (two values): 2 × 2 = 4 evaluations; W |
| 643 | ten | match | S215 | 10 | W163 C4 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): T2 at_or_above_0.95_tau_counted (count of true); W163 status match, written '10', value at written precision '10' |
| 643 | 5 | match | S215 | 5 | W163 N24 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): frozen constants.FLAG_RANGE_OVER_TAU / registry threshold 0.95: 1 − 0.95; W163 status match, written '5', value at written precision '5 |
| 643 | 0.9 | match | S216 | max 0.882 τ | W163 N25 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): records: Q(last) − Q(k*) of the runs that continued past a settling-rule certificate of the same evaluation (cell 1 b_2a0ba8b2 v3 run p |
| 643 | 4.7 | unchecked | cross-reference |  | section / table / figure / equation / algorithm number written in prose; main.tex at this commit: section 4.7 DOES NOT EXIST |
| 647 | two | match | S323 | 2 bars | W163 N73 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v6.determinate_certified(margin, bar_r, bar_o): the difference of two certified cells, one bar each; W163 status mat |
| 647 | 3 | match | S323 | 3 | W163 N28 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v6.DETERMINACY_BAR_FACTOR; determinacy_threshold behaviour; W163 status match, written '3', value at written precisi |
| 647 | 2 | match | S323 | 2 | W163 N29 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v6.DETERMINACY_TAU_MULTIPLE; determinacy_threshold(100, 200) = 2 TAU; W163 status match, written '2', value at writt |
| 647 | two | match | S323 | 2 bars | W163 N73 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v6.determinate_certified(margin, bar_r, bar_o): the difference of two certified cells, one bar each; W163 status mat |
| 647 | three | match | S324 | 3 × max(1000, 2000) = 6000 | W163 N26 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): p515_s53_w132_resettle_v3_campaign.resolve (the scorer): bar = 3.0 × max(gap, slack) over the uncertified cell(s); behaviour on a synth |
| 653 | one | unchecked | method statement |  | number word in section 2 (revised, rounds 1-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 664 | two | unchecked | method statement |  | number word in section 2 (revised, rounds 1-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 679 | 4 | unchecked | cross-reference |  | section / table / figure / equation / algorithm number written in prose; main.tex at this commit: section 4 exists ("Results") |
| 693 | +1 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 695 | +1 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 700 | R2.11 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 700 | R2.12 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 700 | R3.1 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 700 | R3.3 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (sec2; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 711 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 711 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 711 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 716 | 1 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 731 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 731 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 732 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 733 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 741 | 2 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 741 | 2 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 741 | 2 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 755 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 756 | single | unchecked | not a figure |  | article sense ("a single ...") |
| 756 | one | unchecked | method statement |  | number word in section 2 (revised, rounds 1-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 756 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 767 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 772 | single | match | S229 | single | T1 tables.claims[*].instance -> candidate_canonical.investment_year (instance dict, or tables.cells / cells_appended_w160 for a list instance) |
| 789 | 365 | match | S221 | 365 | shared_energy_storage_data.py:665 (sha256 9acd095f) |
| 804 | 365 | match | S222 | 365 | shared_energy_storage_data.py:703 (sha256 9acd095f) |
| 805 | 2 | match | S223 | 2 | shared_energy_storage_data.py:702 (sha256 9acd095f) |
| 811 | 2 | match | S224 | 2 | shared_energy_storage_data.py:702 (sha256 9acd095f) |
| 811 | one | unchecked | method statement |  | number word in section 2 (revised, rounds 1-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 815 | 1 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 817 | -1 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 817 | 1 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 864 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 864 | 2 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 864 | 2 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 864 | 2 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 866 | one | unchecked | method statement |  | number word in section 2 (revised, rounds 1-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 866 | 4 | match | S325 | 4e-05 | data/SRP1/Results/P515S53/w169_slack_inventory/w169_slack_inventory.json detector."max ratio at P over certified points" (sha256 f5831567, blob c19671c0) = 3.6321744275336843e-05 (e_c2) |
| 866 | 10^{-5} | match | S325 | 4e-05 | data/SRP1/Results/P515S53/w169_slack_inventory/w169_slack_inventory.json detector."max ratio at P over certified points" (sha256 f5831567, blob c19671c0) = 3.6321744275336843e-05 (e_c2) |
| 884 | 0 | unchecked | notation |  | formula constant in section 2 (revised, rounds 1-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 906 | one | unchecked | method statement |  | number word in section 2 (revised, rounds 1-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 906 | one | unchecked | method statement |  | number word in section 2 (revised, rounds 1-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 906 | single | unchecked | not a figure |  | article sense ("a single ...") |
| 906 | single | unchecked | not a figure |  | compound adjective (single-scenario) |

## section 3.1 Investment Costs (revised table and paragraph, round 3 C.1) (main.tex l. 947-975): every number with its source

| line | written | status | check | counterpart (at written precision) | source / reason |
|---:|---|---|---|---|---|
| 949 | three | unchecked | method statement |  | number word in section 3.1 Investment Costs (revised table and paragraph, round 3 C.1) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 956 | 2025 | match | C301 | 2025 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years ['2025', '2030', '2035'] |
| 956 | 2030 | match | C301 | 2030 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years ['2025', '2030', '2035'] |
| 956 | 2035 | match | C301 | 2035 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years ['2025', '2030', '2035'] |
| 958 | 1 | match | C302 | 1 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.num_scenarios (sha256 dff6a2b1, blob ac426244) |
| 958 | 35.00 | match | C302 | 35.00 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.scenario_probabilities[0] (sha256 dff6a2b1, blob ac426244) |
| 958 | 214.17 | match | C302 | 214.17 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.cost_power_eur_per_MVA.1 (sha256 dff6a2b1, blob ac426244).2025 (EUR; printed in k EUR) |
| 958 | 168.72 | match | C302 | 168.72 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.cost_power_eur_per_MVA.1 (sha256 dff6a2b1, blob ac426244).2030 (EUR; printed in k EUR) |
| 958 | 153.93 | match | C302 | 153.93 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.cost_power_eur_per_MVA.1 (sha256 dff6a2b1, blob ac426244).2035 (EUR; printed in k EUR) |
| 959 | 2 | match | C303 | 2 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.num_scenarios (sha256 dff6a2b1, blob ac426244) |
| 959 | 55.00 | match | C303 | 55.00 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.scenario_probabilities[1] (sha256 dff6a2b1, blob ac426244) |
| 959 | 267.53 | match | C303 | 267.53 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.cost_power_eur_per_MVA.2 (sha256 dff6a2b1, blob ac426244).2025 (EUR; printed in k EUR) |
| 959 | 224.25 | match | C303 | 224.25 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.cost_power_eur_per_MVA.2 (sha256 dff6a2b1, blob ac426244).2030 (EUR; printed in k EUR) |
| 959 | 206.96 | match | C303 | 206.96 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.cost_power_eur_per_MVA.2 (sha256 dff6a2b1, blob ac426244).2035 (EUR; printed in k EUR) |
| 960 | 3 | match | C304 | 3 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.num_scenarios (sha256 dff6a2b1, blob ac426244) |
| 960 | 10.00 | match | C304 | 10.00 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.scenario_probabilities[2] (sha256 dff6a2b1, blob ac426244) |
| 960 | 342.17 | match | C304 | 342.17 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.cost_power_eur_per_MVA.3 (sha256 dff6a2b1, blob ac426244).2025 (EUR; printed in k EUR) |
| 960 | 278.10 | match | C304 | 278.10 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.cost_power_eur_per_MVA.3 (sha256 dff6a2b1, blob ac426244).2030 (EUR; printed in k EUR) |
| 960 | 268.58 | match | C304 | 268.58 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.cost_power_eur_per_MVA.3 (sha256 dff6a2b1, blob ac426244).2035 (EUR; printed in k EUR) |
| 962 | 1 | match | C305 | 1 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.num_scenarios (sha256 dff6a2b1, blob ac426244) |
| 962 | 35.00 | match | C305 | 35.00 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.scenario_probabilities[0] (sha256 dff6a2b1, blob ac426244) |
| 962 | 212.13 | match | C305 | 212.13 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.cost_energy_eur_per_MWh.1 (sha256 dff6a2b1, blob ac426244).2025 (EUR; printed in k EUR) |
| 962 | 167.11 | match | C305 | 167.11 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.cost_energy_eur_per_MWh.1 (sha256 dff6a2b1, blob ac426244).2030 (EUR; printed in k EUR) |
| 962 | 152.46 | match | C305 | 152.46 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.cost_energy_eur_per_MWh.1 (sha256 dff6a2b1, blob ac426244).2035 (EUR; printed in k EUR) |
| 963 | 2 | match | C306 | 2 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.num_scenarios (sha256 dff6a2b1, blob ac426244) |
| 963 | 55.00 | match | C306 | 55.00 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.scenario_probabilities[1] (sha256 dff6a2b1, blob ac426244) |
| 963 | 264.98 | match | C306 | 264.98 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.cost_energy_eur_per_MWh.2 (sha256 dff6a2b1, blob ac426244).2025 (EUR; printed in k EUR) |
| 963 | 222.12 | match | C306 | 222.12 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.cost_energy_eur_per_MWh.2 (sha256 dff6a2b1, blob ac426244).2030 (EUR; printed in k EUR) |
| 963 | 204.99 | match | C306 | 204.99 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.cost_energy_eur_per_MWh.2 (sha256 dff6a2b1, blob ac426244).2035 (EUR; printed in k EUR) |
| 964 | 3 | match | C307 | 3 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.num_scenarios (sha256 dff6a2b1, blob ac426244) |
| 964 | 10.00 | match | C307 | 10.00 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.scenario_probabilities[2] (sha256 dff6a2b1, blob ac426244) |
| 964 | 338.91 | match | C307 | 338.91 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.cost_energy_eur_per_MWh.3 (sha256 dff6a2b1, blob ac426244).2025 (EUR; printed in k EUR) |
| 964 | 275.45 | match | C307 | 275.45 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.cost_energy_eur_per_MWh.3 (sha256 dff6a2b1, blob ac426244).2030 (EUR; printed in k EUR) |
| 964 | 266.02 | match | C307 | 266.02 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.cost_energy_eur_per_MWh.3 (sha256 dff6a2b1, blob ac426244).2035 (EUR; printed in k EUR) |
| 973 | three | match | C310 | 3 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.num_scenarios (sha256 dff6a2b1, blob ac426244) |
| 973 | 256.32 | match | C308 | 256.32 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.expected_power_eur_per_MVA.2025 (sha256 dff6a2b1, blob ac426244) |
| 973 | 253.88 | match | C308 | 253.88 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json as_read_by_production.expected_energy_eur_per_MWh_production_function.2025 (sha256 dff6a2b1, blob ac426244) |
| 973 | 2025 | match | C308 | 2025 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years |
| 973 | 4 | match | C309 | 4 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json formula_recompute_7ce1d1ab.formula_templates.energy (sha256 dff6a2b1, blob ac426244): "=HLOOKUP({c}$1,'Investment Cost NREL, EUR'!$B$1:$AF$4,{r},FALSE)/4*1000*('Cost breakdown N"... |
| 973 | 5 | match | C309 | 5 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json predecessor_072b1310_selected[sheet Investment Cost, Energy].first_formula_per_row (sha256 dff6a2b1, blob ac426244) (".../5*1000*...") |
| 973 | 4 | match | C309 | 4 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json formula_recompute_7ce1d1ab.formula_templates.energy (sha256 dff6a2b1, blob ac426244): "=HLOOKUP({c}$1,'Investment Cost NREL, EUR'!$B$1:$AF$4,{r},FALSE)/4*1000*('Cost breakdown N"... |

## section 3.3 sentence on TN generation (round 3 C.2) (main.tex l. 983-985): every number with its source

| line | written | status | check | counterpart (at written precision) | source / reason |
|---:|---|---|---|---|---|

## Appendix A (rewritten, rounds 2-3) (main.tex l. 1493-1719): every number with its source

| line | written | status | check | counterpart (at written precision) | source / reason |
|---:|---|---|---|---|---|
| 1497 | three | unchecked | not a figure |  | compound adjective (three-agent,) |
| 1497 | three | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1504 | one | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1504 | one | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1504 | one | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1504 | one | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1504 | one | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1504 | 1 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1508 | Three | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1508 | one | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1508 | one | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1508 | three | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1508 | one | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1512 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1517 | W172 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1517 | 0.02 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1517 | 1.02 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1517 | y0 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1518 | 93,635,360 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1518 | 3 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1519 | 3.5 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1535 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1535 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1539 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1539 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1542 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1543 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1543 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1543 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1554 | one | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1563 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1565 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1565 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1565 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1575 | one | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1579 | W172 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1579 | 5144 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1579 | 5161 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1579 | 5419 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1579 | 5433 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1580 | 5177 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1580 | 5188 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1580 | 5436 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1580 | 5441 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1580 | 5505 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1580 | 5523 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1581 | 1 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1581 | 2 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1581 | 3976 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1581 | 4011 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1581 | 5136 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1581 | 5142 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1582 | 5401 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1582 | 5407 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1582 | 0 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1582 | 0 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1588 | three | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1588 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1588 | 1 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1588 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1594 | 1 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1595 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1595 | 1 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1596 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1597 | 1 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1598 | 1 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1598 | 1 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1606 | three | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1611 | 1 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1611 | 1 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1611 | 1 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1613 | 1 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1614 | 1 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1614 | 1 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1621 | three | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1621 | three | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1625 | one | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1625 | 1.5 | match | A301 | 1.5 | data/SRP1/SRP1_params.json:55 (sha256 dbfdb2a0) |
| 1625 | 1.5 | match | A302 | 1.5 | data/SRP1/SRP1_params.json:56 (sha256 dbfdb2a0) |
| 1625 | 5 | match | A303 | 5.0 | data/SRP1/SRP1_params.json:53 (sha256 dbfdb2a0) |
| 1625 | 5 | match | A303 | 5.0 | data/SRP1/SRP1_params.json:53 (sha256 dbfdb2a0) |
| 1625 | 3 | match | A303 | 3.0 | data/SRP1/SRP1_params.json:54 (sha256 dbfdb2a0) |
| 1625 | 10^{-4} | match | A303 | 0.0001 | data/SRP1/SRP1_params.json:57 (sha256 dbfdb2a0) |
| 1625 | 10^{4} | match | A303 | 10000.0 | data/SRP1/SRP1_params.json:58 (sha256 dbfdb2a0) |
| 1625 | one | match | A304 | 1.0 | data/SRP1/SRP1_params.json:62 (sha256 dbfdb2a0) |
| 1625 | five | match | A304 | 5 | data/SRP1/SRP1_params.json:62 (sha256 dbfdb2a0) |
| 1625 | two | unchecked | not a figure |  | compound adjective (two-phase) |
| 1625 | ten | match | A304 | 10 | data/SRP1/SRP1_params.json:59 (sha256 dbfdb2a0) |
| 1625 | 200 | match | A304 | 200 | data/SRP1/SRP1_params.json:60 (sha256 dbfdb2a0) |
| 1625 | one | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1628 | W172 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1628 | 5 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1628 | 3 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1629 | 1.5 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1629 | 1.5 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1629 | 10 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1629 | 200 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1629 | 5 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1629 | 0.0077 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1629 | 0.198 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1629 | 0.01 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1630 | 3.5 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1636 | three | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1636 | 3 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1636 | one | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1636 | three | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1636 | 10^{-5} | match | A305 | 1e-05 | data/SRP1/SRP1_params.json:42 (sha256 dbfdb2a0) |
| 1636 | 10^{-4} | match | A305 | 0.0001 | data/SRP1/SRP1_params.json:42 (sha256 dbfdb2a0) |
| 1636 | ten | match | A306 | 10 | data/SRP1/SRP1_params.json:45 (sha256 dbfdb2a0) = 10; data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha256 231558f0) required_consecutive_cycles = 10; shared_resources_planning.py:3390 (sha256 0610d74 |
| 1636 | single | unchecked | not a figure |  | compound adjective (single-scenario) |
| 1640 | three | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1640 | 5 | match | A307 | 5 | data/SRP1/SRP1_params.json:123 (sha256 dbfdb2a0) |
| 1640 | 10^{-10} | match | A307 | 1e-10 | data/SRP1/SRP1_params.json:124 (sha256 dbfdb2a0) |
| 1640 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1640 | 2 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1644 | 10^{-6} | match | A308 | 1e-06 | data/SRP1/Results/P515S53/w142_resettle_v6/frozen_s53_resettle_spec_v6_96c23404.json (sha256 96c23404) inputs_in_force_now.configuration_now.convergence_depth_tail |
| 1644 | two | match | A309 | 2 | network.py:980 (sha256 18acaa84); network.py:1002 (sha256 18acaa84); shared_energy_storage_data.py:1335 (sha256 9acd095f) |
| 1647 | W172 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1647 | 7044 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1647 | 7304 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1648 | 5 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1648 | 1e | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1648 | 10 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1649 | 1e | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1649 | 6 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1650 | 1e | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1650 | 4 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1650 | 1 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1650 | 500 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1652 | 49 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1652 | 4 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1652 | 1715 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1652 | 1735 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1652 | 2011 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1652 | 10.1137/10078356X. | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1658 | one | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1678 | 0 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1679 | three | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1679 | one | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1680 | 1 | unchecked | notation |  | formula constant in Appendix A (rewritten, rounds 2-3) (an index offset, exponent, bound or coefficient of the printed formula; the equations and algorithms are audited against the code by W171b / W172 / W174b) |
| 1695 | three | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1699 | ten | match | A310 | 10 | data/SRP1/SRP1_params.json:45 (sha256 dbfdb2a0) = 10; data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha256 231558f0) required_consecutive_cycles = 10; shared_resources_planning.py:3390 (sha256 0610d74 |
| 1699 | single | unchecked | not a figure |  | compound adjective (single-scenario) |
| 1706 | W172 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1706 | 3066 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1706 | 3210 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1706 | 4197 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1706 | 4559 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1707 | 4880 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1707 | 4894 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1707 | 2951 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1707 | 2960 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1707 | 3216 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1707 | 3390 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1708 | 3697 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1708 | 6075 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1708 | 6082 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1708 | 6409 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1708 | 6411 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1714 | one | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1714 | one | unchecked | method statement |  | number word in Appendix A (rewritten, rounds 2-3) describing structure (no value to check; the statements are audited against the code by W171b / W172 / W174b) |
| 1714 | single | unchecked | not a figure |  | article sense ("a single ...") |
| 1717 | W172 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1717 | 18 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1717 | 5221 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1717 | 5343 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1717 | 1 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1718 | 1693 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1718 | 1724 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1718 | 1251 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1718 | 1304 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1718 | 9e4 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1718 | 1925 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |
| 1718 | 1943 | unchecked | comment (not typeset) |  | [CONFIRM] / [AUTHOR] instruction comment in a revised passage (app_a; the CONFIRM analysis is W172 / W173 / W175 scope); comments are not manuscript text |

## main.tex outside the revised passages: tokens on lines changed since W171a (42794d4)

| line | written | status | check | source / reason |
|---:|---|---|---|---|
| 1816 | 33 | match |  | case files data/SRP1/case9/case9_<year>.json and data/SRP1/case33_<n>/case33_<n>_<year>.json at 2025/2030/2035: len(nodes) {'case33_1_2025': 33, 'case33_1_2030': 33, 'case33_1_2035': 33, 'case33_2_2025': 33, 'case33_2_2030': 33, 'case33_2_2035': 33, 'case33_3_ |
| 1816 | 1 | match | C311 | network.py:83 (sha256 18acaa84); network.py:90 (sha256 18acaa84); case files data/SRP1/case33_1/case33_1_2025.json (sha256 5cde824b, blob 5922cb45), data/SRP1/case33_1/case33_1_2030.json (sha256 01144534, blob ae77ca16), data/SRP1/case33_1/case33_1_2035.json ( |
| 1816 | 200 | match | C311 | case files data/SRP1/case33_1/case33_1_2025.json (sha256 5cde824b, blob 5922cb45), data/SRP1/case33_1/case33_1_2030.json (sha256 01144534, blob ae77ca16), data/SRP1/case33_1/case33_1_2035.json (sha256 674b2d00, blob 58211b8d), data/SRP1/case33_2/case33_2_2025. |
| 1816 | 100 | match | C311 | case files data/SRP1/case33_1/case33_1_2025.json (sha256 5cde824b, blob 5922cb45), data/SRP1/case33_1/case33_1_2030.json (sha256 01144534, blob ae77ca16), data/SRP1/case33_1/case33_1_2035.json (sha256 674b2d00, blob 58211b8d), data/SRP1/case33_2/case33_2_2025. |
| 1816 | 150 | match | C311 | case files data/SRP1/case33_1/case33_1_2025.json (sha256 5cde824b, blob 5922cb45), data/SRP1/case33_1/case33_1_2030.json (sha256 01144534, blob ae77ca16), data/SRP1/case33_1/case33_1_2035.json (sha256 674b2d00, blob 58211b8d), data/SRP1/case33_2/case33_2_2025. |
| 1816 | 5 | match | C311 | data/SRP1/SRP1.json (sha256 61a794a7) DistributionNetworks[*].connection_node_id |
| 1816 | 7 | match | C311 | data/SRP1/SRP1.json (sha256 61a794a7) DistributionNetworks[*].connection_node_id |
| 1816 | 9 | match | C311 | data/SRP1/SRP1.json (sha256 61a794a7) DistributionNetworks[*].connection_node_id |
| 1816 | 5 | match | C311 | main.tex branch table row "1 & 1 & 2 & ..." column S^Rated = 200.00; case files data/SRP1/case33_1/case33_1_2025.json (sha256 5cde824b, blob 5922cb45), data/SRP1/case33_1/case33_1_2030.json (sha256 01144534, blob ae77ca16), data/SRP1/case33_1/case33_1_2035.jso |

## Response letter: every number with its source

| line | written | scope | status | check | counterpart (at written precision) | source / reason |
|---:|---|---|---|---|---|---|
| 1 | 2026-10-06 | comment | match | LC1 | 2026-10-06 | STEP6_REVISION_MAP.md line 3 (header: "Expert's plan for the author, 2026-10-06") |
| 5 | 590088fe | comment | match | LC2 | 590088fe | sha256 of data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json |
| 24 | 1 | body | unchecked | enumerator |  | revision number (title block) |
| 33 | three | body | match | L01 | 3 | the letter itself: \section*{Reviewer n} blocks |
| 35 | two | body | match | L02 | 2 | the letter itself: \paragraph blocks before Reviewer 1 |
| 38 | 2 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 38 | 3 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 40 | 0.50 | body | match | L03v2 | 0.50 | manuscript_submitted/main.tex (sha256 ca07d7db) line 866: 'to a relative optimality gap of 0.50\\%' |
| 42 | 2.2 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2 |
| 44 | 2.2.7 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 45 | 3.4 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 3.4 |
| 48 | 64.4 | body | match | L04 | 64.4 | T1 tables.claims[claim_id=CHECK:headline_V_minus_I_settled].d_gross (k EUR, absolute value) |
| 49 | 4.2 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.2 |
| 50 | 4.3 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.3 |
| 50 | 18.25 | body | match | L05v2 | 18.25 | manuscript_submitted/main.tex (sha256 ca07d7db) line 140: 'Relative to uncoordinated operation, coordinated operation with shared ESSs reduces operating costs by up to 18.25\\%' |
| 52 | 90.9 | body | match | L06 | 90.9 | T6 tables.benchmark.benefit (M EUR) |
| 52 | 13.9 | body | match | L06 | 13.9 | T6 tables.benchmark.benefit_relative (%) |
| 53 | 4.4 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.4 |
| 57 | Two | body | unchecked | L07 |  | count of the requests the paragraph names (Reviewer 1 attribution; Reviewer 3 re-optimisation); the third sentence (the 5 x 5 instance) is not a reviewer request |
| 57 | 1 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 57 | 3 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 57 | five | body | match | L08v2 | five | manuscript_submitted/main.tex (sha256 ca07d7db) line 664: 'is represented by five years (2025, 2028, 2031, 2034, and 2037)' |
| 57 | twenty-five | body | match | L08v2 | 25 | manuscript_submitted/main.tex (sha256 ca07d7db) line 673: 'resulting in 25 operating scenarios per representative day' |
| 57 | 3 | body | match | L09 | 3x3 / x0 | W163 N48 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W98 r2 replay reference data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/evals/f6e9cd53fdbb8ee8_x0/per_cycle_record.jsonl (t |
| 57 | 3 | body | match | L09 | 3x3 / x0 | W163 N48 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W98 r2 replay reference data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/evals/f6e9cd53fdbb8ee8_x0/per_cycle_record.jsonl (t |
| 57 | 4.5 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.5 |
| 60 | 1 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 62 | R1.2 | body | unchecked | enumerator |  | reviewer item label |
| 67 | 18.25 | body/rcomment | reviewer quotation, verified |  | 18.25 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) |
| 67 | 92.16 | body/rcomment | reviewer quotation, verified |  | 92.16 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) |
| 67 | 18.25 | body/rcomment | reviewer quotation, verified |  | 18.25 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) |
| 67 | 92.16 | body/rcomment | reviewer quotation, verified |  | 92.16 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) |
| 67 | 8th | body/rcomment | reviewer quotation, verified |  | 8th | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) |
| 67 | 10th | body/rcomment | reviewer quotation, verified |  | 10th | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) |
| 67 | 13th | body/rcomment | reviewer quotation, verified |  | 13th | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) |
| 67 | 14th | body/rcomment | reviewer quotation, verified |  | 14th | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) |
| 69 | one | body/rcomment | reviewer quotation, verified |  | one | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 9; letter quotation 0 segment 5 (verbatim) |
| 74 | R1.4 | body/rresponse | unchecked | enumerator |  | reviewer item label |
| 74 | 3 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 76 | 3.4 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 3.4 |
| 76 | single | body/rresponse | unchecked | not a figure |  | compound adjective (single-scenario) |
| 76 | 3 | body/rresponse | match | L10 | 3x3 / x0 | W163 N48 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W98 r2 replay reference data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/evals/f6e9cd53fdbb8ee8_x0/per_cycle_record.jsonl (t |
| 76 | 3 | body/rresponse | match | L10 | 3x3 / x0 | W163 N48 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W98 r2 replay reference data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/evals/f6e9cd53fdbb8ee8_x0/per_cycle_record.jsonl (t |
| 77 | five | body/rresponse | match | L54 | 5 | data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json (sha256 2a64e3c5) Years {'2025': 3, '2028': 3, '2031': 3, '2034': 3, '2037': 3} (count); data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha |
| 77 | 0.909 | body/rresponse | match | L11 | 0.909 | T11 tables.three_by_three.R_range_derived[0] (lower end, post-hoc settled descent) |
| 77 | 0.934 | body/rresponse | match | L11 | 0.934 | T11 tables.three_by_three.R_range_derived[1] (upper end, at certification) |
| 77 | single | body/rresponse | unchecked | not a figure |  | compound adjective (single-scenario) |
| 77 | 0.933 | body/rresponse | match | L12 | 0.933 | T11 tables.three_by_three.rows[quantity=R predicted from the mean-profile spread].value (0.9331) |
| 77 | 4.5 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.5 |
| 77 | 9 | body/rresponse | match | L13 | 9 | case files data/SRP1/case9/case9_<year>.json and data/SRP1/case33_<n>/case33_<n>_<year>.json at 2025/2030/2035: len(nodes) {'case9_2025': 9, 'case9_2030': 9, 'case9_2035': 9} |
| 77 | three | body/rresponse | match | L13 | 3 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) DistributionNetworks (count) |
| 77 | 33 | body/rresponse | match | L13 | 33 | case files data/SRP1/case9/case9_<year>.json and data/SRP1/case33_<n>/case33_<n>_<year>.json at 2025/2030/2035: len(nodes) {'case33_1_2025': 33, 'case33_1_2030': 33, 'case33_1_2035': 33, 'case33_2_2025': 33, 'case33_2_2030': 33, 'case33_2_2035': 33, 'case33_3_ |
| 77 | 108 | body/rresponse | match | L13 | 108 | case files data/SRP1/case9/case9_<year>.json and data/SRP1/case33_<n>/case33_<n>_<year>.json at 2025/2030/2035: len(nodes) 9 + 3 x 33 per year {'2025': 108, '2030': 108, '2035': 108} |
| 78 | three | body/rresponse | match | L304 | 3 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years ['2025', '2030', '2035'] (count) |
| 78 | five | body/rresponse | match | L301 | 5 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years {'2025': 5, '2030': 5, '2035': 5} (block lengths) |
| 78 | single | body/rresponse | unchecked | not a figure |  | compound adjective (single-scenario) |
| 78 | five | body/rresponse | match | L301 | 5 | data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json (sha256 2a64e3c5) Years {'2025': 3, '2028': 3, '2031': 3, '2034': 3, '2037': 3} (count) |
| 78 | three | body/rresponse | match | L301 | 3 | data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json (sha256 2a64e3c5) Years (block lengths) |
| 78 | 2.2.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 78 | T1 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 78 | T6 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 79 | 4.3 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.3 |
| 82 | 3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 82 | 4.5 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.5 |
| 83 | 4.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.3 |
| 85 | R1.4 | body | unchecked | enumerator |  | reviewer item label |
| 94 | 0 | body/rresponse | match | L15 | 0 | T7 tables.discount[0].rate (%) |
| 94 | 2 | body/rresponse | match | L15 | 2 | T7 tables.discount[1].rate (%) |
| 94 | 5 | body/rresponse | match | L15 | 5 | T7 tables.discount[2].rate (%) |
| 94 | 8 | body/rresponse | match | L15 | 8 | T7 tables.discount[3].rate (%) |
| 94 | -40.8 | body/rresponse | match | L16 | -40.8 | T7 tables.discount[0].value_minus_I (k EUR) at rate 0.0 |
| 94 | -64.4 | body/rresponse | match | L16 | -64.4 | T7 tables.discount[1].value_minus_I (k EUR) at rate 0.02 |
| 94 | -92.7 | body/rresponse | match | L16 | -92.7 | T7 tables.discount[2].value_minus_I (k EUR) at rate 0.05 |
| 94 | -114.6 | body/rresponse | match | L16 | -114.6 | T7 tables.discount[3].value_minus_I (k EUR) at rate 0.08 |
| 94 | T7 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 94 | 2 | body/rresponse | match | L17 | 2 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) DiscountFactor (%) |
| 95 | 4.6 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.6 |
| 95 | T7 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 97 | R1.5 | body | unchecked | enumerator |  | reviewer item label |
| 97 | 2 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 98 | 2 | body/rcomment | reviewer quotation, verified |  | 2 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 16; letter quotation 2 segment 0 (verbatim) |
| 99 | 1 | body/rresponse | match | L55 | 1 | main.tex at the declared Overleaf commit: the first figure environment (line 383, ['framework_v2.pdf'], label ['fig:two-stage_tool_framework']) |
| 100 | 2 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 103 | 2 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 105 | R2.1 | body | unchecked | enumerator |  | reviewer item label |
| 105 | R2.9 | body | unchecked | enumerator |  | reviewer item label |
| 107 | two | body/rcomment | reviewer quotation, verified |  | two | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 16; letter quotation 3 segment 0 (verbatim) |
| 107 | two | body/rcomment | reviewer quotation, verified |  | two | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 16; letter quotation 3 segment 0 (verbatim) |
| 107 | two | body/rcomment | reviewer quotation, verified |  | two | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 16; letter quotation 3 segment 0 (verbatim) |
| 107 | single | body/rcomment | reviewer quotation, verified |  | single | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 16; letter quotation 3 segment 0 (verbatim) |
| 108 | one | body/rcomment | reviewer quotation, verified |  | one | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 22; letter quotation 3 segment 1 (verbatim) |
| 110 | two | body/rresponse | unchecked | not a figure |  | compound adjective (two-stage) |
| 111 | one | body/rresponse | unchecked | L18 |  | article sense ("one ... plan" = a single plan) |
| 113 | single | body/rresponse | unchecked | not a figure |  | article sense ("a single ...") |
| 113 | 2.1 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.1 |
| 115 | single | body/rresponse | unchecked | not a figure |  | article sense ("a single ...") |
| 118 | 2.1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.1 |
| 118 | 1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 118 | 2.1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.1 |
| 118 | 2.2 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2 |
| 118 | 5 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 120 | R2.2 | body | unchecked | enumerator |  | reviewer item label |
| 122 | 1 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 122 | 0.25 | body/rresponse | match | L19 | 0.25 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 122 | 0.5 | body/rresponse | match | L19 | 0.5 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 122 | 2 | body/rresponse | match | L19 | 2.0 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 122 | 4 | body/rresponse | match | L19 | 4.0 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 122 | One | body/rresponse | unchecked | L20 |  | one evaluation per candidate (the evaluation cache keyed on the candidate); an algorithm statement, audited by the equation/algorithm audit (map section B 2.1), not a table figure |
| 122 | one | body/rresponse | unchecked | L21v2 |  | one local problem per network block per cycle (algorithm statement) |
| 122 | three | body/rresponse | match | L21v2 | 3 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a): Years ['2025', '2030', '2035'], Days ['Autumn', 'Spring', 'Summer', 'Winter'], TransmissionNetwork + 3 DistributionNetworks |
| 122 | four | body/rresponse | match | L21v2 | 4 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a): Years ['2025', '2030', '2035'], Days ['Autumn', 'Spring', 'Summer', 'Winter'], TransmissionNetwork + 3 DistributionNetworks |
| 122 | four | body/rresponse | match | L21v2 | 4 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a): Years ['2025', '2030', '2035'], Days ['Autumn', 'Spring', 'Summer', 'Winter'], TransmissionNetwork + 3 DistributionNetworks |
| 122 | single | body/rresponse | unchecked | L21v2 |  | compound adjective (single-scenario) |
| 122 | 48 | body/rresponse | match | L21v2 | 48 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a): Years ['2025', '2030', '2035'], Days ['Autumn', 'Spring', 'Summer', 'Winter'], TransmissionNetwork + 3 DistributionNetworks (3 x 4 x 4) |
| 122 | one | body/rresponse | unchecked | L21v2 |  | one ESSO problem per storage agent per cycle |
| 122 | three | body/rresponse | match | L21v2 | 3 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) DistributionNetworks (count) = the interface nodes; shared_energy_storage_data.py:96 (sha256 9acd095f) (one ESSO solve per active distribution-network node per cycle; each ESSO model spans every year  |
| 122 | one | body/rresponse | unchecked | L21v2 |  | one agent per interface node (structure) |
| 122 | 2.2.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 122 | 4.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.7 |
| 123 | 2.2 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2 |
| 123 | 1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 123 | 4.7 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.7 |
| 125 | R2.3 | body | unchecked | enumerator |  | reviewer item label |
| 125 | R2.4 | body | unchecked | enumerator |  | reviewer item label |
| 125 | R2.7 | body | unchecked | enumerator |  | reviewer item label |
| 133 | 2.2 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2 |
| 136 | 2.2.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 138 | 2.2 | body/rchanges | unchecked | L22v2 |  | revised-manuscript section number (map section B 2.2) |
| 138 | 2.2.7 | body/rchanges | match | L22v2 | 2.2.7 | manuscript_submitted/main.tex section structure (subsubsection "Benders' Cuts" at line 440; numbered by its position) |
| 138 | 2.2.7 | body/rchanges | match | L23 | 2.2.7 | STEP6_REVISION_MAP.md line 96: "replace by \"2.2.7 Recourse evaluation and certification\"" |
| 139 | 4.7 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.7 |
| 141 | R2.5 | body | unchecked | enumerator |  | reviewer item label |
| 141 | R2.6 | body | unchecked | enumerator |  | reviewer item label |
| 148 | three | body/rresponse | match | L24 | 3 | W163 N7 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v2.SettlingRuleV2.evaluate: at_least_3_turning_points = len(T) >= 3 (called by settling_criterion_v6.SettlingRuleV6.e |
| 150 | one | body/rresponse | match | L25 | >= 1 period | W163 N9 (W_MIN 20) and N10 (W_FACTOR 1.1): W = max(20, ceil(1.1 P_hat)) >= 1.1 P_hat > P_hat, i.e. at least one measured period |
| 151 | 2.2.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 152 | 42 | body/rresponse | match | L26 | 42 | W163 C5 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): T2 cells with certification_stats_included; W163 status match, written '42', value at written precision '42'; paragraphs_v5.md line 65 ( |
| 152 | 32 | body/rresponse | match | L26 | 32 | W163 C6 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): T2; W163 status match, written '32', value at written precision '32'; paragraphs_v5.md line 65 (097421f8) |
| 153 | 10 | body/rresponse | match | L26 | 10 | W163 N30 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): T2 cells with certification_stats_included, status uncertified (W153 totals.n_uncertified); W163 status match, written '10', value at w |
| 154 | 0.9 | body/rresponse | match | L27 | max 0.882 τ | W163 N25 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): records: Q(last) − Q(k*) of the runs that continued past a settling-rule certificate of the same evaluation (cell 1 b_2a0ba8b2 v3 run p |
| 155 | 4.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.7 |
| 155 | T2 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 158 | two | body/rresponse | unchecked | not a figure |  | compound adjective (two-phase) |
| 163 | 2.2.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 166 | three | body/rresponse | match | L28 | 3 | T6 tables.benchmark.w160_additions.arms_in_full.{passive,price_taker}.Q_by_start (count of starts) |
| 166 | T6 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 168 | 2.2.7 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 168 | 4.7 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.7 |
| 168 | T2 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 170 | R2.8 | body | unchecked | enumerator |  | reviewer item label |
| 171 | 10.1016/j.est.2024.114911 | body/rcomment | reviewer quotation, verified |  | 10.1016/j.est.2024.114911 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 22; letter quotation 7 segment 0 (verbatim) |
| 171 | 10.1109/ISGTEUROPE62998.2024.10863557 | body/rcomment | reviewer quotation, verified |  | 10.1109/ISGTEUROPE62998.2024.10863557 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 22; letter quotation 7 segment 0 (verbatim) |
| 171 | 10.1016/j.apenergy.2022.120569 | body/rcomment | reviewer quotation, verified |  | 10.1016/j.apenergy.2022.120569 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 22; letter quotation 7 segment 0 (verbatim) |
| 172 | three | body/rresponse | match | L29 | 3 | the letter itself: DOIs in the R2.8 comment |
| 172 | 1 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 174 | one | body/rresponse | unchecked | L30 |  | [AUTHOR: ...] instruction, not manuscript text |
| 175 | 10.1016/j.est.2024.114911 | body/rresponse | unchecked | reference identifier |  | DOI (reference-list identifier) |
| 175 | 1 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 176 | 10.1109/ISGTEUROPE62998.2024.10863557 | body/rresponse | unchecked | reference identifier |  | DOI (reference-list identifier) |
| 176 | 10.1016/j.apenergy.2022.120569 | body/rresponse | unchecked | reference identifier |  | DOI (reference-list identifier) |
| 177 | 1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 179 | R2.10 | body | unchecked | enumerator |  | reviewer item label |
| 186 | 21 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 186 | 28 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 187 | 2.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 189 | R2.11 | body | unchecked | enumerator |  | reviewer item label |
| 189 | R2.12 | body | unchecked | enumerator |  | reviewer item label |
| 202 | two | body/rresponse | unchecked | L31 |  | pronoun ("the two" = the network models and the storage agent) |
| 202 | 2.3 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 203 | 2.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 206 | 3 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 208 | R3.1 | body | unchecked | enumerator |  | reviewer item label |
| 210 | 2.2.1 | body/rcomment | reviewer quotation, verified |  | 2.2.1 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) |
| 210 | 1 | body/rcomment | reviewer quotation, verified |  | 1 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) |
| 210 | 2034 | body/rcomment | reviewer quotation, verified |  | 2034 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) |
| 210 | 2037 | body/rcomment | reviewer quotation, verified |  | 2037 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) |
| 210 | 15 | body/rcomment | reviewer quotation, verified |  | 15 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) |
| 210 | 2037 | body/rcomment | reviewer quotation, verified |  | 2037 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) |
| 210 | 2.4 | body/rcomment | reviewer quotation, verified |  | 2.4 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) |
| 210 | 2 | body/rcomment | reviewer quotation, verified |  | 2 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) |
| 210 | 4.1 | body/rcomment | reviewer quotation, verified |  | 4.1 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) |
| 212 | T1 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 212 | T4 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 213 | 60 | body/rresponse | match | L32v2 | 60 | T1 tables.claims (count) |
| 213 | two | body/rresponse | match | L32v2 | 2 | T1 tables.claims (gross_verdict != net verdict) + T4 tables.year_ladder (gross_v6.verdict != net_v6.verdict) |
| 213 | two | body/rresponse | match | L32v2 | 2 | T1 claim L:y2030__n5_p0.25_e0.5__n7_p1_e3: other_cell l_0ee93aca candidate_canonical.nodes (non-zero nodes); data/SRP1/Results/P515S53/w142_resettle_v6/frozen_s53_resettle_spec_v6_96c23404.json (sha256 96c23404) cells.l_0ee93aca.m_flex_price_multiplier |
| 213 | 2035 | body/rresponse | match | L34 | 2035 | T4 tables.year_ladder.per_year; data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years |
| 213 | 2030 | body/rresponse | match | L34 | 2030 | T4 tables.year_ladder.per_year; data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years |
| 213 | +42.5 | body/rresponse | match | L34 | 42.5 | T4 tables.year_ladder.D_gross (k EUR, 2035 - 2030) |
| 213 | -2.3 | body/rresponse | match | L35 | -2.3 | T4 tables.year_ladder.D_net (k EUR) |
| 213 | 4.6 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.6 |
| 213 | 2 | body/rresponse | unchecked | L36 |  | Table [2] of the revised manuscript (placeholder; the reviewer cites "Table 2 in Section 4.1" of the submitted version) |
| 214 | 2.2.1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 214 | 2.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 214 | 4.6 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.6 |
| 214 | T1 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 214 | T4 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 216 | R3.2 | body | unchecked | enumerator |  | reviewer item label |
| 216 | 0.50 | body | match | L37v2 | 0.50 | manuscript_submitted/main.tex (sha256 ca07d7db) line 866: 'to a relative optimality gap of 0.50\\%' |
| 218 | 2.2.7 | body/rcomment | reviewer quotation, verified |  | 2.2.7 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 11 | body/rcomment | reviewer quotation, verified |  | 11 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 4.3 | body/rcomment | reviewer quotation, verified |  | 4.3 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 26 | body/rcomment | reviewer quotation, verified |  | 26 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 27 | body/rcomment | reviewer quotation, verified |  | 27 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 29 | body/rcomment | reviewer quotation, verified |  | 29 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 11 | body/rcomment | reviewer quotation, verified |  | 11 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 0.50 | body/rcomment | reviewer quotation, verified |  | 0.50 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 3 | body/rcomment | reviewer quotation, verified |  | 3 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 15 | body/rcomment | reviewer quotation, verified |  | 15 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 4.4 | body/rcomment | reviewer quotation, verified |  | 4.4 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 15 | body/rcomment | reviewer quotation, verified |  | 15 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 222 | 2.2.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 223 | 4 | body/rresponse | match | L38 | 4 | W163 N18 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.TAU = DELTA_R * R_REF / 4.0 (constants() formula); W163 status match, written '4', value at written precision '4'; p |
| 223 | 0.07 | body/rresponse | match | L38 | 0.07 | W163 N19 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.DELTA_R; v6 spec stop_rule.constants.DELTA_R; W163 status match, written '0.07', value at written precision '0.07';  |
| 223 | single | body/rresponse | unchecked | L38 |  | compound adjective (single-scenario) |
| 224 | two | body/rresponse | match | L39 | 2 | W163 N22 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): T11: R = V / V_SRP1_settled (a ratio of two values); W163 status match, written 'two', value at written precision '2'; paragraphs_v5.md |
| 224 | 15 | body/rresponse | unchecked | L40v2 |  | figure number of the submitted version. In manuscript_submitted/main.tex the figure environments in order give Figure 15 = line 1637, D Distribution Networks > D.2 ADN Connected to TN Node~7, ['case33_2_flexibility_scenarios_2025_Spring.pdf']; Section 4.4 hold |
| 227 | 2.2.7 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 227 | 4.7 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.7 |
| 227 | 4 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 229 | R3.3 | body | unchecked | enumerator |  | reviewer item label |
| 231 | 21 | body/rcomment | reviewer quotation, verified |  | 21 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 12 segment 0 (differs) |
| 231 | 31 | body/rcomment | reviewer quotation, verified |  | 31 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 12 segment 0 (differs) |
| 231 | 2.3.2 | body/rcomment | reviewer quotation, verified |  | 2.3.2 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 12 segment 0 (differs) |
| 233 | 2.3.3 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 233 | R2.11 | body/rresponse | unchecked | enumerator |  | reviewer item label |
| 233 | R2.12 | body/rresponse | unchecked | enumerator |  | reviewer item label |
| 233 | one | body/rresponse | unchecked | L56 |  | a network block = one (representative year, representative day) pair (48 = 3 x 4 x 4, L21v2); structure, no value |
| 233 | one | body/rresponse | unchecked | L56 |  | as the first "one" |
| 233 | single | body/rresponse | unchecked | L56 |  | one scenario-free schedule (model_construction_helpers.sess_na_scenario / sess_row_is_duplicate); a model statement audited by W171b |
| 233 | one | body/rresponse | unchecked | L56 |  | as "a single schedule" |
| 233 | one | body/rresponse | unchecked | L57 |  | demonstrative ("this one schedule") |
| 233 | 2.3.6 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 235 | 2.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 237 | R3.4 | body | unchecked | enumerator |  | reviewer item label |
| 239 | 2.3.2 | body/rcomment | reviewer quotation, verified |  | 2.3.2 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 26; letter quotation 13 segment 0 (verbatim) |
| 239 | 22 | body/rcomment | reviewer quotation, verified |  | 22 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 26; letter quotation 13 segment 0 (verbatim) |
| 239 | 23 | body/rcomment | reviewer quotation, verified |  | 23 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 26; letter quotation 13 segment 0 (verbatim) |
| 241 | R2.10 | body/rresponse | unchecked | enumerator |  | reviewer item label |
| 242 | 2.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 244 | R3.5 | body | unchecked | enumerator |  | reviewer item label |
| 246 | 2.3.2 | body/rcomment | reviewer quotation, verified |  | 2.3.2 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) |
| 246 | 22 | body/rcomment | reviewer quotation, verified |  | 22 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) |
| 246 | 25 | body/rcomment | reviewer quotation, verified |  | 25 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) |
| 246 | 15 | body/rcomment | reviewer quotation, verified |  | 15 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) |
| 246 | 3 | body/rcomment | reviewer quotation, verified |  | 3 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) |
| 246 | two | body/rcomment | reviewer quotation, verified |  | two | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) |
| 246 | 15 | body/rcomment | reviewer quotation, verified |  | 15 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) |
| 248 | 0.985 | body/rresponse | match | L41 | 0.985 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) ageing.calendar_retention_per_year; data/SRP1/Results/P515S53/frozen_s53_spec_v41_fcea4b38.json (sha256 fcea4b38, last commit 477dba3e) configuration.identical_to_w104.ess_ageing_ |
| 248 | C2 | body/rresponse | unchecked | enumerator |  | ageing calibration name (T8 arm labels C2 / C2_calfade) |
| 248 | C2 | body/rresponse | unchecked | enumerator |  | ageing calibration name (T8 arm labels C2 / C2_calfade) |
| 248 | 31.6 | body/rresponse | match | L42 | 31.6 | T8 tables.ageing.rows[arm=C2].value_minus_I (k EUR, absolute value) |
| 248 | 64.4 | body/rresponse | match | L42 | 64.4 | T8 tables.ageing.rows[arm=C2_calfade].value_minus_I (k EUR, absolute value) |
| 248 | T8 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 249 | 3.4 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 3.4 |
| 249 | 4.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.3 |
| 249 | T8 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 251 | R3.6 | body | unchecked | enumerator |  | reviewer item label |
| 253 | 4.6 | body/rcomment | reviewer quotation, verified |  | 4.6 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 28; letter quotation 15 segment 0 (verbatim) |
| 253 | 5 | body/rcomment | reviewer quotation, verified |  | 5 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 28; letter quotation 15 segment 0 (verbatim) |
| 255 | 7 | body/rresponse | match | L44 | 7 | T1 tables.claims[claim_id=E:n7_4h_e1_C2_calfade:value_minus_I].instance (the non-zero node of the unit) |
| 255 | -4.1 | body/rresponse | match | L45v2 | -4.1 | T8 tables.ageing.rows[arm=no_ageing].value_minus_I (k EUR) |
| 255 | -31.6 | body/rresponse | match | L45v2 | -31.6 | T8 tables.ageing.rows[arm=C2].value_minus_I (k EUR) |
| 255 | -64.4 | body/rresponse | match | L45v2 | -64.4 | T8 tables.ageing.rows[arm=C2_calfade].value_minus_I (k EUR) |
| 255 | -45.2 | body/rresponse | match | L45v2 | -45.2 | T8 tables.ageing.rows[arm=C4].value_minus_I (k EUR) |
| 255 | -65.9 | body/rresponse | match | L45v2 | -65.9 | T8 tables.ageing.rows[arm=C3_midblock].value_minus_I (k EUR) |
| 255 | -73.7 | body/rresponse | match | L45v2 | -73.7 | T8 tables.ageing.rows[arm=C3_unit].value_minus_I (k EUR) |
| 255 | T8 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 256 | 0.70 | body/rresponse | match | L46 | 0.70 | W163 N38 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): v6 spec inputs_in_force_now ess_ageing_baseline.minimum_soh (the baseline label "soh_min 0.70"); W163 status match, written '0.70', val |
| 256 | 0.50 | body/rresponse | match | L46 | 0.50/0.50 | W163 N41 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): A64 e_soh050 r2 campaign spec (28c9478d) candidates[0].settling_resettle.minimum_soh and extra.minimum_soh (the override; configuration |
| 256 | +4.9 | body/rresponse | match | L46 | 4.9 | T10 tables.a64.rows[claim_id=E:soh050:delta_value_vs_070].d_gross (k EUR) |
| 256 | T10 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 256 | T8 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 256 | 2035 | body/rresponse | match | L302 | 2035 | T8 tables.ageing.rows[*].floor_year_070 |
| 256 | two | body/rresponse | match | L302 | 2 | T8 tables.ageing.rows[*].floor_year_070 (aged arms with None) |
| 257 | 4.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.3 |
| 257 | T8 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 257 | T10 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 259 | R3.7 | body | unchecked | enumerator |  | reviewer item label |
| 259 | R3.8 | body | unchecked | enumerator |  | reviewer item label |
| 261 | 3 | body/rcomment | reviewer quotation, verified |  | 3 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 29; letter quotation 16 segment 0 (verbatim) |
| 261 | 8 | body/rcomment | reviewer quotation, verified |  | 8 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 29; letter quotation 16 segment 0 (verbatim) |
| 261 | 0.50 | body/rcomment | reviewer quotation, verified |  | 0.50 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 29; letter quotation 16 segment 0 (verbatim) |
| 262 | 3 | body/rcomment | reviewer quotation, verified |  | 3 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 30; letter quotation 16 segment 1 (verbatim) |
| 262 | 8 | body/rcomment | reviewer quotation, verified |  | 8 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 30; letter quotation 16 segment 1 (verbatim) |
| 262 | 4.4.1 | body/rcomment | reviewer quotation, verified |  | 4.4.1 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 30; letter quotation 16 segment 1 (verbatim) |
| 262 | four | body/rcomment | reviewer quotation, verified |  | four | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 30; letter quotation 16 segment 1 (verbatim) |
| 262 | 8760 | body/rcomment | reviewer quotation, verified |  | 8760 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 30; letter quotation 16 segment 1 (verbatim) |
| 264 | one | body/rresponse | match | L47 | 1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.md T7 caption ("one discount factor per representative year applied to the five years of its block") |
| 265 | five | body/rresponse | match | L47 | 5 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years {'2025': 5, '2030': 5, '2035': 5} |
| 266 | 2025 | body/rresponse | match | L48 | 2025 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.md T7 caption ("I paid in 2025"); T7 tables.discount[*].I identical at every rate |
| 268 | 2.2.1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 268 | 3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 268 | 4 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 274 | five | body | match | L49 | 5 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years {'2025': 5, '2030': 5, '2035': 5} |
| 275 | 1 | body | match | L50 | 1 | T6 tables.benchmark.sweep.sweep_passive_cold.n_blocks |
| 275 | 12 | body | match | L50 | 12 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a): years x days |
| 275 | 8 | body | match | L50 | 8 | T6 tables.benchmark.sweep.sweep_price_taker_cold.n_blocks |
| 275 | 12 | body | match | L50 | 12 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a): years x days |
| 275 | 4.4 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.4 |
| 276 | 10^{-5} | body | match | L51 | 1e-05 | W163 N33 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): definitions.py EQUALITY_TOLERANCE = 1e-5; model_construction_helpers.py curtaillable pg upper bound pg_avail + EQUALITY_TOLERANCE; W109 |
| 277 | single | body | no table counterpart | L52v2 |  | no committed record states the machine count; stands on the author's attestation (Addendum 67; Addendum 68 Decision 5 "Single machine stands on the author's attestation") |
| 277 | single | body | match | L52v2 | thread caps 1 on every table evaluation | W163 N45 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W159 c1: harness THREAD_CAP_ENV lines (OMP / MKL / OPENBLAS / VECLIB / NUMEXPR = 1); all_table_evaluations_ran_with_OMP_NUM_THREADS_1;  |
| 278 | 49 | body | match | L53 | 49 | data/SRP1/Results/P515S53/w160_step6_frozen/export/paragraphs_v2.md (sha256 73b42c49, last commit 2a1d7f92) section (v) prediction scorecard: rows numbered 1..n |
| 279 | 4 | body | match | L303 | 4 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json formula_recompute_7ce1d1ab.formula_templates.energy (sha256 dff6a2b1, blob ac426244) |
| 279 | 5 | body | match | L303 | 5 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json predecessor_072b1310_selected (energy formula ".../5*1000*...") (sha256 dff6a2b1, blob ac426244) |
| 279 | 4 | body | match | L303 | 4 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json formula_recompute_7ce1d1ab.formula_templates.energy (sha256 dff6a2b1, blob ac426244) |
| 279 | 1.25 | body | match | L303 | 1.25 | data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json predecessor_072b1310_selected (energy formula ".../5*1000*...") (sha256 dff6a2b1, blob ac426244); data/SRP1/Results/P515S53/w167_nomenclature_years/question4_cost_file.json formula_reco |

## Findings -- letter

- **G1** (line 256, W171a finding re-checked): R3.6 now reads "(in 2035 under the baseline, the mid-block and the unit-retention calibrations; not within the horizon under the other two)"; T8 floor_year_070 = {'C3_unit': 2035, 'C2': None, 'C4': None, 'C2_calfade': 2035, 'C3_midblock': 2035, 'no_ageing': None}: consistent (L302 match). RESOLVED.
- **G2** (line -, W171a finding re-checked): The R3.5 clause on the 0.5 % / 2 % sentence is gone from the letter (Addendum 69 D8); the sentence itself is gone from main.tex (present at this commit: False; never in the submitted source: True). L43v2 removed with its sentence. RESOLVED.
- **G3** (line [100], W171a finding, unchanged): R1.5 \rchanges still says "Figure~2 and the graphical abstract" on lines [100]; Addendum 69 D8: waits for the PDF the reviewers read. OPEN.
- **G4** (line 224, W171a finding, unchanged): "Figure 15 of the submitted version" not derivable from the source (W171a G4); waits for the PDF (Addendum 69 D8). OPEN.
- **G5** (line [], W171a finding re-checked): stray double quote after "(Section~4.5)": lines []. RESOLVED.
- **G6** (line 78, W171a finding re-checked): B.6 qualifier: R1.2 now reads "three representative years standing for five-year blocks at the single-scenario instance; five three-year blocks at the multi-scenario instance" (True); counts checked (L301). RESOLVED.
- **G7** (line 279, W171a finding re-checked): "Further changes" names the cost-file correction (÷5 instead of ÷4, ×1.25): True; numbers checked (L303). RESOLVED.
- **G8** (line [176], W171a finding, unchanged): "The three references were added" beside the bracketed status ("not yet added" on lines [176]); B.8 keeps the note until the references are in. OPEN (by design).
- **K1** (line 213, claim checked): "two verdicts": T1 ['L:y2030__n5_p0.25_e0.5__n7_p1_e3'] and T4 2035 - 2030 (unchanged since W171a; L32v2).
- **K5** (line rcomment blocks, quotation check): 29 of 31 quotation segments verbatim, 2 differ, 0 not found.

## Findings -- main.tex

- **M1** (line 756, 866, cross-reference to the wrong subsection): Section 2 says the values of \eta, SoC^{\text{Min}}, \varepsilon^{\text{Cl}} (l. 756); c^{\sigma}, \varepsilon^{\text{E}} (l. 866) are "given in Section~\ref{sec:case_settings}" = ('3.6', 'Evaluation and Certification Settings'); they are printed in ('3.5', 'Shared Energy Storage Parameters') (l. {'\\eta': [1070], 'SoC^{\\text{Min}}': [1070], '\\varepsilon^{\\text{Cl}}': [1070], 'c^{\\sigma}': [1070], '\\varepsilon^{\\text{E}}': [1070], '\\alpha = 0.5': [1129]}). Round 3 C mapped every former "Section~3.5" to sec:case_settings, but section 3.5 as pasted holds these values. No number token (\ref is excluded).
- **M2** (line 449, 639, 643, 1074, 1138, literal forward references): literal section numbers in sections 2, 3.5-3.6 and Appendix A: [(449, 'Section~4.7'), (639, 'Section~4.5'), (643, 'Section~4.7'), (1074, 'Section~4.3'), (1138, 'Section~4.4')]; main.tex sections 3.x and 4.x at this commit: {'3.1': 'Investment Costs', '3.2': 'Market Data', '3.3': 'Transmission Network', '3.4': 'Active Distribution Networks', '3.5': 'Shared Energy Storage Parameters', '3.6': 'Evaluation and Certification Settings', '4.1': 'Shared ESS Investment Plan', '4.2': 'Degradation and Available Energy Capacity', '4.3': 'Computational Performance', '4.4': 'Operational Planning Results', '4.5': 'Impact of Planning Horizon Discretization', '4.6': 'Key Insights'} (section 4 is still the submitted text; "Section~4.7" etc. stay literal until section 4 is rewritten, round 3 C)
- **M3** (line -, W171a findings re-checked): W171a M1 (Section 3.5 absent): sections 3.5 and 3.6 exist (labels {'sec:case_studies': 914, 'sec:case_ess_params': 1067, 'sec:case_settings': 1106}). W171a M2 (calibrations cited to the unrevised 3.4): now \ref{sec:case_ess_params}. W171a M3 ("first and third turning points"): now "the three most recent turning points" (S318 match). W171a M4 ("2n OrthoMADS", "If |P| < n + 1"): Algorithm 1 now states variants A (n + 1, unit-box completion at Delta = 1) and B (2n, completion below n + 1) (S304-S307, S313-S314 match). RESOLVED.

## highlights.tex: every number with its source

no in-scope numeric token

## cover_letter.tex: every number with its source

- line 16 `four`: match -- the cover letter itself: \item entries of the list that follows
- line 24 `15`: match -- data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years (sum of block lengths)

## main.tex line rules re-pinned to cb237b2d

Revised passages (declared or ruled; never the auto index): sec2 l. 315-909; sec3_1 l. 947-975; sec3_3 l. 983-985; sec3_5_6 l. 1066-1151; app_a l. 1493-1719. Network tables l. 980-1065; Appendices B-D l. 1720-2080.

| W171a region (l.) | key | kind | 260bd83 (l.) |
|---|---|---|---|
| 1122-1134 | r47 | result | 1215-1227 |
| 1135-1272 | r_delete | result | 1228-1365 |
| 1273-1361 | r_delete | result | 1366-1454 |
| 1362-1381 | keyins | result | 1455-1474 |
| 1059-1121 | results | result | 1152-1214 |
| 1964-2077 | appE | result | 2081-2194 |
| 940-957 | cost | dropped (8 lines changed) | the submitted cost table (W171a l. 940-957) is replaced by the revised section 3.1 table (W167 fragment, round 3 C.1): its numbers are declared (C301-C308), not submitted-version |

## main.tex: submitted-version figures by section

| section | n | map lines cited |
|---|---:|---|
| Front matter | 3 | 117, 23 |
| 3 Case Study | 26 | 114, 117, 120 |
| 4 Results | 245 | 137, 159, 165, 169 |
| 5 Conclusions | 2 | 23 |
| E Results | 283 | 193 |

## W171a parameter checks re-evaluated on this run (the Addendum 68 list, source of several value-table rows)

| id | quantity | found | status | sources |
|---|---|---|---|---|
| P01 | eta_ch (shared ESS, network models and agent) | 0.97 | match | shared_energy_storage.py:14 (sha256 d45c72d0); model_construction_helpers.py:969 (sha256 a20a224f); shared_energy_storage_data.py:655 (sha256 9acd095f) |
| P02 | eta_dch | 0.96 | match | shared_energy_storage.py:15 (sha256 d45c72d0) |
| P03 | SoC^Min (fraction of E^Av) | 0.1 | match | definitions.py:39 (sha256 7e5719af); model_construction_helpers.py:894 (sha256 a20a224f) |
| P04 | SoC^Max | 0.9 | match | definitions.py:38 (sha256 7e5719af); model_construction_helpers.py:902 (sha256 a20a224f) |
| P05 | SoC^0 (initial = closure target) | 0.5 | match | definitions.py:40 (sha256 7e5719af); model_construction_helpers.py:973 (sha256 a20a224f); model_construction_helpers.py:987 (sha256 a20a224f) |
| P06 | closure slack bound eps^Cl (fraction of E^Av) | 0.05 | match | model_construction_helpers.py:410 (sha256 a20a224f); model_construction_helpers.py:1166 (sha256 a20a224f); data/SRP1/case9/case9_params.json:22,26 (sha256 f3eff050); data/SRP1/case33_1/case33_1_params.json:22,26 (sha256 8e9e5a53); data/SRP1/case33_2/case33_2_params.json:22,26 (sha256 31b5fedf); data/SRP1/case33_3/case33_3_params.json:22,26 (sha256 a19bd5b9) |
| P07 | closure slack numerical term | 1e-05 | match | definitions.py:90 (sha256 7e5719af) |
| P08 | closure slack penalty c^Cl (EUR/MWh) | 1000.0 | match | definitions.py:58 (sha256 7e5719af); model_construction_helpers.py:2341 (sha256 a20a224f) |
| P09 | network complementarity eps^C (normalised) | 0.0001 | match | definitions.py:91 (sha256 7e5719af); definitions.py:92 (sha256 7e5719af); model_construction_helpers.py:951 (sha256 a20a224f) |
| P10 | eps^E (ESSO throughput regularisation) | 1e-05 | match | definitions.py:74 (sha256 7e5719af); shared_energy_storage_data.py:844 (sha256 9acd095f) |
| P11 | c^sigma (ESSO P-net slack penalty) | 1000.0 | match | definitions.py:63 (sha256 7e5719af); shared_energy_storage_data.py:829 (sha256 9acd095f); data/SRP1/SharedESS/SRP1_ESS_Params.json:6 (sha256 39106f93) |
| P12 | S_ref (MVA); normalisation 2 S_ref | 2.5 | match | data/SRP1/SRP1_params.json:50 (sha256 dbfdb2a0); shared_resources_planning.py:5028 (sha256 0610d745); shared_resources_planning.py: 12 lines divide by (2 * shared_ess_rating) |
| P13 | rho initial v / pf / ess | {'v': {'case9': 0.0077, 'case33_1': 0.0077, 'case33_2': 0.0077, 'case33_3': 0.0077}, 'pf': {'case9': 0.198, 'case33_1': 0.198, 'case33_2': 0.198, 'case33_3': 0.198}, 'ess': {'case9': 0.01, 'case33_1': 0.01, 'case33_2': 0.01, 'case33_3': 0.01, 'esso': | match | data/SRP1/SRP1_params.json:67 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:73 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:83 (sha256 dbfdb2a0) |
| P14 | residual balancing: ratio 5 (pf decrease 3), x/÷1.5, clamp [1e-4, 1e4] | {'residual_balance_ratio': 5.0, 'residual_balance_ratio_pf_decrease': 3.0, 'increase_factor': 1.5, 'decrease_factor': 1.5, 'min': 0.0001, 'max': 10000.0} | match | data/SRP1/SRP1_params.json:53 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:54 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:55 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:56 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:57 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:58 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:48 (sha256 dbfdb2a0) |
| P15 | per-channel freeze after 10 unchanged cycles; backstop 200; ESS exempt until dual ratio < 1 on 5 cycles | {'freeze_after_unchanged_cycles': 10, 'freeze_backstop_cycle': 200, 'ess_exempt': {'dual_ratio_below': 1.0, 'consecutive_cycles': 5}} | match | data/SRP1/SRP1_params.json:59 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:60 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:62 (sha256 dbfdb2a0) |
| P16 | Boyd eps_abs / eps_rel | {'eps_abs': 1e-05, 'eps_rel': 0.0001} | match | data/SRP1/SRP1_params.json:42 (sha256 dbfdb2a0); admm_parameters.py:363 (sha256 40d0e972) |
| P17 | production exit: consecutive passing cycles | {'SRP1': 10, '3x3_spec': 10} | match | data/SRP1/SRP1_params.json:45 (sha256 dbfdb2a0); shared_resources_planning.py:3390 (sha256 0610d745); data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha256 231558f0) required_consecutive_cycles |
| P18 | cycle caps: 3x3 500; SRP1 gated N_old + 100, ungated min(k0 + 109, 300) | {'3x3_cap': 500, 'v6_gated_cells': 30, 'v6_ungated_cells': 4, 'CAP_AFTER_K0': 109} | match | data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha256 231558f0) cap; data/SRP1/Results/P515S53/w142_resettle_v6/frozen_s53_resettle_spec_v6_96c23404.json (sha256 96c23404) cells[*].cap_rule (gated: formula, cap = N_old + 100, per-cell ceiling; ungated: after_first_k0, ceiling); settling_criterion_v2.py:76 (sha256 3db13b0e) |
| P19 | sigma (common objective scale) | 93635360.0 | match | data/SRP1/SRP1_params.json:49 (sha256 dbfdb2a0) |
| P20 | kappa_ESSO = sigma / median w_b | {'SRP1': 227210.996652, 'SRP1_median_w_b': 412.10751847261173, 'SRP1_n_blocks': 48, '3x3': 386258.694309, '3x3_median_w_b': 242.4161873368304, '3x3_n_blocks': 80} | match | data/SRP1/SRP1_params.json:51 (sha256 dbfdb2a0); shared_resources_planning.py:4062 (sha256 0610d745); shared_resources_planning.py:4037 (sha256 0610d745); data/SRP1/SRP1.json (sha256 61a794a7); data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json (sha256 2a64e3c5) |
| P21 | w_b = Y_y D_d 1.02^-(y - y0) | {'SRP1_DiscountFactor': 0.02, '3x3_DiscountFactor': 0.02} | match | shared_resources_planning.py:3872 (sha256 0610d745); shared_resources_planning.py:3873 (sha256 0610d745); data/SRP1/SRP1.json (sha256 61a794a7); data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json (sha256 2a64e3c5) |
| P22 | AA: type-II, memory 5, Tikhonov 1e-10, ratchet safeguard, keep_memory, cleared on rho change and on a solve failure, off when every channel passes | {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory', 'type_II': True, 'ratchet_safeguard': True, 'cleared_on_rho_change': True, 'cleared_on_solve_failure': True, 'off_when_every_channel_passes': True} | match | data/SRP1/SRP1_params.json:123 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:124 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:125 (sha256 dbfdb2a0); admm_anderson_acceleration.py:3 (sha256 24cf3bea); admm_anderson_acceleration.py:398 (sha256 24cf3bea); admm_anderson_acceleration.py:442 (sha256 24cf3bea); admm_anderson_acceleration.py:477 (sha256 24cf3bea) |
| P23 | tail compl_inf_tol 1e-6 (production: TSO 5e-4, DSO 1e-4) | {'v6_tail': {'compl_inf_tol': 1e-06, 'enabled': True}, '3x3_tail': {'compl_inf_tol': 1e-06, 'enabled': True}, 'tso_case_file': 0.0005, 'dso_case_files': {'case33_1': None, 'case33_2': None, 'case33_3': None}, 'ipopt_default_in_network_py': 0.0001} | match | data/SRP1/Results/P515S53/w142_resettle_v6/frozen_s53_resettle_spec_v6_96c23404.json (sha256 96c23404) inputs_in_force_now.configuration_now.convergence_depth_tail; data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha256 231558f0) configuration.convergence_depth_tail; data/SRP1/case9/case9_params.json:41 (sha256 f3eff050); network.py:632 (sh |
| P24 | ESSO IPOPT tol / acceptable_tol / linear solver | {'override': '1e-10 / 1e-9', 'file_tol': 1e-06, 'file_acceptable_tol': 1e-05, 'linear_solver': 'ma57'} | match | shared_energy_storage_data.py:1082 (sha256 9acd095f); shared_energy_storage_data.py:108 (sha256 9acd095f); shared_energy_storage_data.py:1104 (sha256 9acd095f); data/SRP1/SharedESS/SRP1_ESS_Params.json:37 (sha256 39106f93) |
| P25 | networks IPOPT tol / acceptable / MA97 / max_iter / recovery acceptable_tol & acceptable_iter | {'options': {'case9': {'tol': 1e-05, 'acceptable_tol': 0.0001, 'linear_solver': 'ma97'}, 'case33_1': {'tol': 1e-05, 'acceptable_tol': 0.0001, 'linear_solver': 'ma97'}, 'case33_2': {'tol': 1e-05, 'acceptable_tol': 0.0001, 'linear_solver': 'ma97'}, 'ca | partial | data/SRP1/case9/case9_params.json:37 (sha256 f3eff050); data/SRP1/case33_1/case33_1_params.json:37 (sha256 8e9e5a53); data/SRP1/case33_2/case33_2_params.json:37 (sha256 31b5fedf); data/SRP1/case33_3/case33_3_params.json:37 (sha256 a19bd5b9); network.py:559 (sha256 18acaa84); network.py:972 (sha256 18acaa84) |
| P26 | proximal gamma (TSO; DSO off) | {'tso': {'enabled': True, 'gamma_policy': 'tied_to_rho', 'tau': 0.0}, 'dso_enabled': False} | match | data/SRP1/SRP1_params.json:95 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:96 (sha256 dbfdb2a0); shared_resources_planning.py:5081 (sha256 0610d745) |
| P27 | row 18 alpha (3x3 only) | ['{"alpha": 0.5, "floor": null}'] | match | data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha256 231558f0) candidates[*].interface_deviation_premium |
| P28 | interface-voltage pin weight (3x3 only; solver-only) | 90000.0 | match | definitions.py:75 (sha256 7e5719af); model_construction_helpers.py:1940 (sha256 a20a224f); shared_resources_planning.py:849 (sha256 0610d745) |
| P29 | baseMVA (TN and DNs) | [100.0] | match | case files data/SRP1/case9/case9_<y>.json and data/SRP1/case33_<n>/case33_<n>_<y>.json at 2025/2030/2035: baseMVA |
| P30 | lattice units and duration bounds | {'p_step': 0.25, 'e_step': 0.5, 'step4_p': True, 'step4_e': True, 'step4_duration': True, 'ep_min': 2.0, 'ep_max': 4.0} | match | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) min/max_energy_to_power_factor |
| P31 | storage agents per cycle (one per interface node, all years and days) | 3 | match | shared_energy_storage_data.py:96 (sha256 9acd095f); data/SRP1/SRP1.json (sha256 61a794a7) DistributionNetworks |

## Unchecked: by category (all files)

| file | category | n |
|---|---|---:|
| main.tex | comment (not typeset) | 166 |
| main.tex | address | 6 |
| main.tex | not a figure | 29 |
| main.tex | enumerator | 1 |
| main.tex | descriptive count | 4 |
| main.tex | notation | 75 |
| main.tex | cross-reference | 9 |
| main.tex | method statement | 48 |
| main.tex | literature value | 2 |
| main.tex | network/data parameter | 1005 |
| response_to_reviewers_draft.tex | enumerator | 48 |
| response_to_reviewers_draft.tex | cross-reference | 75 |
| response_to_reviewers_draft.tex | structural | 1 |
| response_to_reviewers_draft.tex | not a figure | 12 |
| response_to_reviewers_draft.tex | method statement | 8 |
| response_to_reviewers_draft.tex | author placeholder | 1 |
| response_to_reviewers_draft.tex | reference identifier | 3 |
| section2_expert_draft.tex | comment (draft instruction) | 111 |
| section2_expert_draft.tex | identifier | 2 |
| section2_expert_draft.tex | method statement | 10 |
| section2_expert_draft.tex | cross-reference | 15 |
| section2_expert_draft.tex | notation | 23 |
| section2_expert_draft.tex | not a figure | 4 |
| section2_expert_draft.tex | enumerator | 2 |

## Excluded LaTeX structure

| file | category | n | tokens |
|---|---|---:|---|
| cover_letter.tex | package/class option or argument | 3 | 12pt, utf8, T1 |
| highlights.tex | package/class option or argument | 2 | 12pt, 1in |
| main.tex | package/class option or argument | 8 | 12pt, 1p, 3p, 5p |
| main.tex | figure option/filename | 44 | 1.00, 0.50, 0.65, 0.90, 0.95, 0.475 |
| main.tex | table structure | 145 | 2, 5, 4, 8, 3, 1, 7, 9 |
| main.tex | label/ref/cite key | 2 | two |
| main.tex | length/layout | 17 | 0.5em, 4pt, 1.00, 5pt, 0.85, 1.40 |
| main.tex | environment option/argument | 4 | 0.90 |
| response_to_reviewers_draft.tex | package/class option or argument | 4 | 11pt, 1in, utf8, T1 |
| response_to_reviewers_draft.tex | macro definition | 3 | 1 |
| response_to_reviewers_draft.tex | macro parameter | 3 | 1 |

## Integrity checks

- clone_head_is_declared_commit: True
- file_set_equals_declared: True
- every_sha_equals_declared: True
- every_file_equals_its_blob_at_commit: True
- no_tex_modified_or_untracked_in_clone: True
- w171a_pin_readable_in_clone: True
- frozen_json_unchanged: True
- v2_declaration_set_reconstructed_equals_w171a_applied: True
- every_declaration_found_and_evaluated: True
- no_token_claimed_twice: True
- no_duplicate_declaration_id: True
- every_token_assigned: True
- every_map_citation_found_once: True
- statuses_in_vocabulary: True
- main_line_rules_pinned_to_this_main: True
- revised_passages_start_at_their_headings: True
- w171a_regions_kept_or_dropped_with_reason: True
- w171a_regions_equal_pinned_expectation: True
- every_v2_declaration_carried_superseded_or_replaced: True
- parameter_table_evaluated: True
- every_value_declaration_applied: True
- every_value_row_status_in_vocabulary: True
- every_quotation_segment_located: True
- docx_sha_equals_declared: True
