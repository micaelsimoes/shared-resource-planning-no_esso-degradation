# P5.15 Addendum 32 Q6 (W26) -- negative penalty levels

P5.15 Addendum 32 Q6 W26 -- negative penalty levels: reporting convention or unbounded slack (zero solves)

Objective convention: Q(x) = gross_operational_cost, settlement-EXCLUDED (the oracle cost convention); value = Q(0) - Q(x); EUR; "weighted" = unweighted * admm_block_weight (num_years * num_days / 1.02^(y - 2025)); terminal salvage is reported with each record.

Git HEAD e904955bf037a86c6ba371306c0e5ec399202cb7. Zero solves: armed SolveProfileGuard(permitted=()), counts {'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}, verify(0) failures [].

## Classification

**NEITHER (a) NOR (b) AS POSED: every slack feeding a negative level is correctly bounded (lb = 0, NonNegativeReals) and the report carries no sign convention; the negative levels are the solver's bound-relaxation residue (IPOPT bound_relax_factor = 1e-8, honor_original_bounds = no), reported and priced unclipped**

Code: every D-row level is PENALTY * (slack_up + slack_down) with a positive coefficient, summed without any reference subtraction or credit sign (P2), so (a) is excluded. Models: every slack entry has lower bound exactly 0 (domain NonNegativeReals; raw declared bounds also 0), and no terminal value lies below IPOPT's relaxed bound lb - 1e-8 (P3 checks, the five P515S44 model sets); for the P515S45-S47 records, whose models are not persisted, (b) is excluded by inference: the declaring code is unchanged since the model sets were built (ageing_independence) and every negative reported level lies at or above the relaxed-bound floor (P4 floor test). The negative values are within [-1e-8, 0): IPOPT relaxes every bound by bound_relax_factor = 1e-8 (not overridden anywhere in the searched scope) and, with honor_original_bounds = no, returns the final point unprojected; Pyomo loads it as-is; a slack whose penalty drives it to its lower bound sits at the relaxed bound. The report (and the objective re-evaluation behind Q) sums those values unclipped. Size: the projected-onto-bounds level differs from the reported level by exactly the reported level in every model set (no positive slack value); e.g. c_star_aa_keep_memory: bias -12801.4727 EUR = -1.966e-05 of gross. Absolute Q is therefore biased LOW by the negative slack values (|detector_penalty_total| <= 1.98e-05 of Q over all certified records); value differences move by d_det = det(x) - det(0), which is not identically 0 but at most 180.99 EUR (max |d_det|/bar = 0.0586); baseline n7_4h_e1: d_det = -19.35 EUR against value 259427.77 and bar 25104.65. Positive detector-family block levels: 8 in 4 files; 8 of them sit in a block whose terminal-cycle solve was a recovery retry (network_failures log); implied mean slack value per free entry = level / free_coef_sum: shared_ess_day_balance_slack 2.907e-09, shared_ess_day_balance_slack 2.910e-09, shared_ess_day_balance_slack 2.920e-09, shared_ess_day_balance_slack 2.923e-09, voltage_slack 2.690e-08 (p.u.; per-entry values are not available -- these models are not persisted). RES curtailment definitional (negative in TSO blocks): not a slack; pg's declared upper bound is pg_avail + EQUALITY_TOLERANCE (1e-5 p.u.), so pg_avail - pg >= -1e-5 - relaxation by construction (P3: no pg above its relaxed upper bound); its objective weight penalty_gen_curtailment is 0 in every block, so it does not enter Q.

## P1 Inventory

Scope: every component_levels_terminal.json tracked in git (git ls-files at HEAD) whose path contains /P515S44/, /P515S45/, /P515S46/ or /P515S47/; certification status from the sibling evaluation_record.json (absent = harness checks / gates / zero-check runs, listed but flagged). Files: 142; status counts {'certified': 129, 'no_evaluation_record': 10, 'not_certified': 3}. Negative block levels (rows): 23634 (full list in the JSON, `P1_inventory.negative_rows`).

| agent | component | block levels | negative | negative (certified) | positive | min unweighted | max unweighted | min weighted | max weighted |
|---|---|---|---|---|---|---|---|---|---|
| DSO | branch_flow_slack | 5112 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO | detector_penalty_total | 5112 | 5004 | 4644 | 0 | -0.833922 | 0 | -376.941 | 0 |
| DSO | ess_complementarity_bilinear_definitional | 5112 | 0 | 0 | 2124 | 0 | 0.00622335 | 0 | 2.56469 |
| DSO | flexibility_cost_internal | 5112 | 0 | 0 | 5004 | 0 | 37250.6 | 0 | 1.40569e+07 |
| DSO | flexibility_p_day_balance_slack | 5112 | 5004 | 4644 | 0 | -0.0639969 | 0 | -29.1724 | 0 |
| DSO | local_ess_day_balance_slack | 5112 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO | node_balance_slack | 5112 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO | res_curtailment_definitional_at_weight_1 | 5112 | 0 | 0 | 5004 | 0 | 40.1267 | 0 | 18458.3 |
| DSO | shared_ess_day_balance_slack | 5112 | 2124 | 1764 | 0 | -0.0019999 | 0 | -0.911636 | 0 |
| DSO | voltage_slack | 5112 | 5004 | 4644 | 0 | -0.767925 | 0 | -346.857 | 0 |
| TSO | branch_flow_slack | 1704 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| TSO | detector_penalty_total | 1704 | 1664 | 1544 | 4 | -0.114817 | 0.324533 | -52.8151 | 149.282 |
| TSO | ess_complementarity_bilinear_definitional | 1704 | 0 | 0 | 1508 | 0 | 0.00264716 | 0 | 1.2177 |
| TSO | flexibility_p_day_balance_slack | 1704 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| TSO | generation_cost | 1704 | 0 | 0 | 1668 | 0 | 185366 | 0 | 8.43417e+07 |
| TSO | local_ess_day_balance_slack | 1704 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| TSO | node_balance_slack | 1704 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| TSO | res_curtailment_definitional_at_weight_1 | 1704 | 1666 | 1546 | 2 | -0.00574403 | 0.00898656 | -2.61353 | 4.13382 |
| TSO | shared_ess_day_balance_slack | 1704 | 1504 | 1384 | 4 | -0.00572577 | 0.00175375 | -2.63331 | 0.803111 |
| TSO | voltage_slack | 1704 | 1664 | 1544 | 4 | -0.109091 | 0.32278 | -50.1818 | 148.479 |

Signed zeros (-0.0, not negative): {'TSO|res_curtailment_penalty': 1246}.

Positive detector-family block levels (8; detector-family block levels > 0, joined with network_failures_*.jsonl of the same evaluation (same agent/node/year/day); terminal_cycle_solve_was_a_recovery = a row exists at cycle == cycles_run (the block's terminal solve was a recovery retry)):

| file | status | block | component | unweighted | weighted | implied mean value per free entry (P4) | terminal-cycle solve was a recovery | network-failure rows, same block (cycle:class) |
|---|---|---|---|---|---|---|---|---|
| P515S45/campaign_s45_a1a/evals/7c66ddb648ce86a1_n5_2h_e5 | certified | TSO|2025|Spring | voltage_slack | 0.322779 | 148.479 | 2.689829121379665e-08 | True | 28:recovered_tier1, 105:recovered_tier1 |
| P515S45/campaign_s45_a1a/evals/7c66ddb648ce86a1_n5_2h_e5 | certified | TSO|2025|Spring | shared_ess_day_balance_slack | 0.00174589 | 0.803111 | 2.909820907327218e-09 | True | 28:recovered_tier1, 105:recovered_tier1 |
| P515S47/campaign_s47_a1a_baseline/evals/366864896179e01d_n7_4h_e5 | certified | TSO|2030|Autumn | voltage_slack | 0.32278 | 133.02 | 2.6898296101445413e-08 | True | 40:recovered_tier1, 42:recovered_tier1, 88:recovered_tier1, 91:recovered_tier1 |
| P515S47/campaign_s47_a1a_baseline/evals/366864896179e01d_n7_4h_e5 | certified | TSO|2030|Autumn | shared_ess_day_balance_slack | 0.00175375 | 0.722732 | 2.9229104829669817e-09 | True | 40:recovered_tier1, 42:recovered_tier1, 88:recovered_tier1, 91:recovered_tier1 |
| P515S47/campaign_s47_a1a_baseline/evals/58de187f7806c66e_n9_2h_e4 | certified | TSO|2035|Spring | voltage_slack | 0.32278 | 121.804 | 2.689829283027455e-08 | True | 97:recovered_tier1 |
| P515S47/campaign_s47_a1a_baseline/evals/58de187f7806c66e_n9_2h_e4 | certified | TSO|2035|Spring | shared_ess_day_balance_slack | 0.00174429 | 0.658227 | 2.90715582315648e-09 | True | 97:recovered_tier1 |
| P515S47/campaign_s47_a1a_baseline/evals/ecfcd325ddf18d43_n5_2h_e1 | certified | TSO|2035|Summer | voltage_slack | 0.32278 | 120.48 | 2.689829321670831e-08 | True | 113:recovered_tier1 |
| P515S47/campaign_s47_a1a_baseline/evals/ecfcd325ddf18d43_n5_2h_e1 | certified | TSO|2035|Summer | shared_ess_day_balance_slack | 0.00175173 | 0.653846 | 2.9195424080167677e-09 | True | 113:recovered_tier1 |

## P2 Code trace

- level_builder_harness_s31_block_components: `p515_g_g1_g4_admm_gates.py:1664`
- level_builder_harness_calls_production_detector: `p515_g_g1_g4_admm_gates.py:1687`
- level_builder_harness_weighted_is_weight_times_unweighted: `['p515_g_g1_g4_admm_gates.py:1750', 'p515_g_g1_g4_admm_gates.py:1763']`
- level_builder_harness_res_curtailment_definitional: `p515_g_g1_g4_admm_gates.py:1729`
- level_builder_harness_writer: `p515_g_g1_g4_admm_gates.py:1736`
- production_detector_components: `shared_resources_planning.py:749`
- production_detector_voltage_accumulation: `shared_resources_planning.py:766`
- production_detector_total_is_plain_sum: `shared_resources_planning.py:773`
- production_recourse_gross_minus_settlement_only: `shared_resources_planning.py:943`
- production_recourse_D_excluded_reporting_field: `shared_resources_planning.py:968`
- penalty_voltage: `model_construction_helpers.py:1906`
- penalty_flex_p_day_balance: `model_construction_helpers.py:1956`
- penalty_local_ess_day_balance: `model_construction_helpers.py:1988`
- penalty_shared_ess_day_balance: `model_construction_helpers.py:1998`
- objective_function_rule: `model_construction_helpers.py:1647`
- objective_adds_slack_penalties: `model_construction_helpers.py:1653`
- objective_adds_ess_day_balance_penalties: `model_construction_helpers.py:1654`
- get_primal_value_evaluates_objective_function_rule: `network.py:65`
- decl_slack_v_sqr_down: `network.py:282`
- decl_slack_v_sqr_up: `network.py:283`
- decl_slack_flex_p_balance_up: `network.py:321`
- decl_slack_es_soc_final_up: `network.py:376`
- decl_slack_shared_es_soc_final_up: `network.py:402`
- bounds_voltage_down: `model_construction_helpers.py:92`
- bounds_voltage_up: `model_construction_helpers.py:100`
- bounds_shared_ess_slack_loop: `model_construction_helpers.py:1088`
- bounds_shared_ess_slack_setub: `model_construction_helpers.py:1092`
- pg_bounds_curtaillable_upper_tolerance: `model_construction_helpers.py:189`
- res_curtailment_definitional_term: `model_construction_helpers.py:1802`
- prior_art_voltage_slack_diagnostics_clips_at_0: `model_construction_helpers.py:109`
- network_solver_options_merge: `network.py:556`

Every D-row penalty helper adds PENALTY * (slack_up + slack_down) with a POSITIVE constant coefficient (no subtraction, no reference level, no credit sign); the harness level is the production helper value times the probability (unweighted) and times admm_block_weight (weighted); detector_penalty_total is a plain sum. The "detector" semantics are a classification label (D rows = feasibility detectors reported separately), not a sign convention: economic_recourse_all_D_excluded = net - detector_penalty_total is an ADDITIONAL reporting field. A negative level therefore requires negative slack VALUES.

IPOPT bound-relaxation option search (every git-tracked *.py file (production and p5* harnesses) and every git-tracked data/SRP1 *.json outside Results/ (case, params, ESS params files); 404 files; needles ['bound_relax_factor', 'honor_original_bounds', 'constr_viol_tol']): 0 hits in production modules or case files; all hits: 0 (listed in the JSON). IPOPT documentation query: {'bound_relax_factor': ['bound_relax_factor                     0 <= (      1e-08) <  +inf      ', '   Factor for initial relaxation of the bounds.', '     Before start of the optimization, the bounds given by the user are'], 'honor_original_bounds': ['honor_original_bounds         ("no")', '   Indicates whether final points should be projected into original bounds.', '     Ipopt might relax the bounds during the optimization (see, e.g., option'], 'constr_viol_tol': ['constr_viol_tol                        0 <  (     0.0001) <  +inf      ', '   Desired threshold for the constraint and variable bound violation.', '     Absolute tolerance on the constraint and variable bound violation.'], 'version': 'Ipopt 3.14.18 (aarch64-apple-darwin24.5.0), ASL(20241111)'}

## P3 Model read-back (P515S44 certified model sets)

Ageing independence: earliest model-set launch 2026-09-19T12:38:03.980583+00:00; commits touching ['network.py', 'model_construction_helpers.py', 'definitions.py', 'helper_functions.py', 'shared_resources_planning.py', 'p515_g_g1_g4_admm_gates.py', 'network_data.py', 'network_parameters.py'] since then: 85c147fa1e4bd72646143aa6ab32a316a58153e7 2026-09-20T12:31:43 (40 changed lines, slack/penalty/bound token hits: 0); b127259308fcc18054fb912000b4f2ff1077e569 2026-09-19T22:51:34 (261 changed lines, slack/penalty/bound token hits: 0). ESSO module defines TSO/DSO slack vars: {'shared_energy_storage_data.py': []}.

| model set | candidate | sha256 = manifest | gross recomputed / reported - 1 | max abs(recompute - reported) | detector weighted (reported) | projected | bias | bias / gross |
|---|---|---|---|---|---|---|---|---|
| 837fc982565dbba3_c_star_aa_keep_memory | c_star_aa_keep_memory `578636daa6d6` | True | 3.66e-16 | 0.00e+00 | -12801.4727 | 0.0000 | -12801.4727 | -1.966e-05 |
| 4e53fa5560bbfa10_two_c_star_d | two_c_star_d `4e53fa5560bb` | True | 1.84e-16 | 0.00e+00 | -12801.4732 | 0.0000 | -12801.4732 | -1.975e-05 |
| 0796b6dceeb0f95f_node7_empty_aa_keep_memory | node7_empty_aa_keep_memory `e30704e6e4dd` | True | 1.83e-16 | 0.00e+00 | -12782.1213 | 0.0000 | -12782.1213 | -1.961e-05 |
| 8e48f3ec8993a283_paper_plan_aa_keep_memory | paper_plan_aa_keep_memory `d1a02e67107d` | True | 1.83e-16 | 0.00e+00 | -12762.7698 | 0.0000 | -12762.7698 | -1.954e-05 |
| a6ca6c94e033e460_two_c_star_aa_keep_memory | two_c_star_aa_keep_memory `4e53fa5560bb` | True | 1.84e-16 | 0.00e+00 | -12801.4732 | 0.0000 | -12801.4732 | -1.975e-05 |

Slack families, model set data/SRP1/Results/P515S44/campaign_s44_aa_variant/evals/837fc982565dbba3_c_star_aa_keep_memory (all blocks; values in p.u. of the variable):

| agent / family | n | fixed | lb=ub | free | domains | lb min..max | ub min..max (None) | value min | value max | negative | below lb | max below lb | below relaxed lb | above relaxed ub | penalty coef min..max | objective coef min..max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TSO|voltage_slack | 5184 | 1728 | 576 | 2880 | ['NonNegativeReals'] | 0.0..0.0 | 0.0..0.11250000000000004 (0) | -9.090909098225947e-09 | 0.0 | 2880 | 2880 | 9.091e-09 | 0 | 0 | 50000.0..50000.0 | 50000.0..50000.0 |
| TSO|flexibility_p_day_balance_slack | 72 | 72 | 0 | 0 | ['NonNegativeReals'] | 0.0..0.0 | 0.01..0.01 (0) | 0.0 | 0.0 | 0 | 0 | 0.000e+00 | 0 | 0 | None..None | None..None |
| TSO|shared_ess_day_balance_slack | 72 | 0 | 0 | 72 | ['NonNegativeReals'] | 0.0..0.0 | 0.0012432327938688766..0.0016315308772929765 (0) | -9.601473195813088e-09 | -9.471112172217523e-09 | 72 | 72 | 9.601e-09 | 0 | 0 | 100000.0..100000.0 | 100000.0..100000.0 |
| TSO|orphan_flex_q_day_balance (not in objective) | 72 | 72 | 0 | 0 | ['NonNegativeReals'] | 0.0..0.0 | 0.01..0.01 (0) | 0.0 | 0.0 | 0 | 0 | 0.000e+00 | 0 | 0 | None..None | None..None |
| DSO|voltage_slack | 57024 | 0 | 1728 | 55296 | ['NonNegativeReals'] | 0.0..0.0 | 0.0..0.11250000000000004 (0) | -9.818182169338604e-09 | 0.0 | 55296 | 55296 | 9.818e-09 | 0 | 0 | 50000.0..50000.0 | 50000.0..50000.0 |
| DSO|flexibility_p_day_balance_slack | 2304 | 0 | 0 | 2304 | ['NonNegativeReals'] | 0.0..0.0 | 0.01..0.01 (0) | -9.919024624108153e-09 | -9.896379144784008e-09 | 2304 | 2304 | 9.919e-09 | 0 | 0 | 100000.0..100000.0 | 100000.0..100000.0 |
| DSO|shared_ess_day_balance_slack | 72 | 0 | 0 | 72 | ['NonNegativeReals'] | 0.0..0.0 | 0.0012432327938688766..0.0016315308772929765 (0) | -9.911017421165964e-09 | -9.907079145181587e-09 | 72 | 72 | 9.911e-09 | 0 | 0 | 100000.0..100000.0 | 100000.0..100000.0 |
| DSO|orphan_flex_q_day_balance (not in objective) | 2304 | 2304 | 0 | 0 | ['NonNegativeReals'] | 0.0..0.0 | 0.01..0.01 (0) | 0.0 | 0.0 | 0 | 0 | 0.000e+00 | 0 | 0 | None..None | None..None |

All five model sets (roll-up totals):

| model set | slack entries | negative | below original lb | below relaxed lb | above relaxed ub | min value | lb min | domains |
|---|---|---|---|---|---|---|---|---|
| 837fc982565dbba3_c_star_aa_keep_memory | 67104 | 60624 | 60624 | 0 | 0 | -9.919024624108153e-09 | 0.0 | ['NonNegativeReals'] |
| 4e53fa5560bbfa10_two_c_star_d | 67104 | 60624 | 60624 | 0 | 0 | -9.919032585897034e-09 | 0.0 | ['NonNegativeReals'] |
| 0796b6dceeb0f95f_node7_empty_aa_keep_memory | 67104 | 60576 | 60576 | 0 | 0 | -9.919021742228913e-09 | 0.0 | ['NonNegativeReals'] |
| 8e48f3ec8993a283_paper_plan_aa_keep_memory | 67104 | 60528 | 60528 | 0 | 0 | -9.919016932181339e-09 | 0.0 | ['NonNegativeReals'] |
| a6ca6c94e033e460_two_c_star_aa_keep_memory | 67104 | 60624 | 60624 | 0 | 0 | -9.919032560228403e-09 | 0.0 | ['NonNegativeReals'] |

RES curtailment definitional (not a slack; objective weight read from the model):

- 837fc982565dbba3_c_star_aa_keep_memory: {'TSO': {'penalty_gen_curtailment_param_values': [0.0], 'min_avail_minus_pg': -1.0006793519947354e-05, 'n_pg_above_avail': 1305, 'n_entries': 1728, 'max_ub_minus_avail': 1.0000000000010001e-05, 'min_ub_minus_avail': 9.999999999996123e-06, 'n_pg_above_ub': 19, 'n_pg_above_relaxed_ub': 0}, 'DSO': {'penalty_gen_curtailment_param_values': [0.0], 'min_avail_minus_pg': -1.0009418725649157e-05, 'n_pg_above_avail': 486, 'n_entries': 3456, 'max_ub_minus_avail': 1.0000000000010001e-05, 'min_ub_minus_avail': 9.999999999996123e-06, 'n_pg_above_ub': 53, 'n_pg_above_relaxed_ub': 0}}
- 4e53fa5560bbfa10_two_c_star_d: {'TSO': {'penalty_gen_curtailment_param_values': [0.0], 'min_avail_minus_pg': -1.0006793519933232e-05, 'n_pg_above_avail': 1305, 'n_entries': 1728, 'max_ub_minus_avail': 1.0000000000010001e-05, 'min_ub_minus_avail': 9.999999999996123e-06, 'n_pg_above_ub': 19, 'n_pg_above_relaxed_ub': 0}, 'DSO': {'penalty_gen_curtailment_param_values': [0.0], 'min_avail_minus_pg': -1.0009418684703042e-05, 'n_pg_above_avail': 489, 'n_entries': 3456, 'max_ub_minus_avail': 1.0000000000010001e-05, 'min_ub_minus_avail': 9.999999999996123e-06, 'n_pg_above_ub': 53, 'n_pg_above_relaxed_ub': 0}}
- 0796b6dceeb0f95f_node7_empty_aa_keep_memory: {'TSO': {'penalty_gen_curtailment_param_values': [0.0], 'min_avail_minus_pg': -1.0006794251920972e-05, 'n_pg_above_avail': 1305, 'n_entries': 1728, 'max_ub_minus_avail': 1.0000000000010001e-05, 'min_ub_minus_avail': 9.999999999996123e-06, 'n_pg_above_ub': 19, 'n_pg_above_relaxed_ub': 0}, 'DSO': {'penalty_gen_curtailment_param_values': [0.0], 'min_avail_minus_pg': -1.0009418730029361e-05, 'n_pg_above_avail': 485, 'n_entries': 3456, 'max_ub_minus_avail': 1.0000000000010001e-05, 'min_ub_minus_avail': 9.999999999996123e-06, 'n_pg_above_ub': 53, 'n_pg_above_relaxed_ub': 0}}
- 8e48f3ec8993a283_paper_plan_aa_keep_memory: {'TSO': {'penalty_gen_curtailment_param_values': [0.0], 'min_avail_minus_pg': -1.0006793631098379e-05, 'n_pg_above_avail': 1305, 'n_entries': 1728, 'max_ub_minus_avail': 1.0000000000010001e-05, 'min_ub_minus_avail': 9.999999999996123e-06, 'n_pg_above_ub': 19, 'n_pg_above_relaxed_ub': 0}, 'DSO': {'penalty_gen_curtailment_param_values': [0.0], 'min_avail_minus_pg': -1.0009418764036717e-05, 'n_pg_above_avail': 484, 'n_entries': 3456, 'max_ub_minus_avail': 1.0000000000010001e-05, 'min_ub_minus_avail': 9.999999999996123e-06, 'n_pg_above_ub': 53, 'n_pg_above_relaxed_ub': 0}}
- a6ca6c94e033e460_two_c_star_aa_keep_memory: {'TSO': {'penalty_gen_curtailment_param_values': [0.0], 'min_avail_minus_pg': -1.0006793519933476e-05, 'n_pg_above_avail': 1305, 'n_entries': 1728, 'max_ub_minus_avail': 1.0000000000010001e-05, 'min_ub_minus_avail': 9.999999999996123e-06, 'n_pg_above_ub': 19, 'n_pg_above_relaxed_ub': 0}, 'DSO': {'penalty_gen_curtailment_param_values': [0.0], 'min_avail_minus_pg': -1.000941868527748e-05, 'n_pg_above_avail': 489, 'n_entries': 3456, 'max_ub_minus_avail': 1.0000000000010001e-05, 'min_ub_minus_avail': 9.999999999996123e-06, 'n_pg_above_ub': 53, 'n_pg_above_relaxed_ub': 0}}

## P4 Floor test and effect on Q

floor(agent, node, component) = min over the five model sets and all blocks of sum_{free slack entries} coef * (lb - min(1e-8 * max(1, |lb|), 1e-4)); free = not fixed and lb < ub; every reported negative detector level must satisfy level >= floor - 1e-12 * max(1, |floor|). Tested 15300 negative detector block levels; below floor: 0. Floors (unweighted): {'TSO|None|voltage_slack': -0.12000000000000009, 'TSO|None|node_balance_slack': 0.0, 'TSO|None|branch_flow_slack': 0.0, 'TSO|None|flexibility_p_day_balance_slack': 0.0, 'TSO|None|local_ess_day_balance_slack': 0.0, 'TSO|None|shared_ess_day_balance_slack': -0.006, 'TSO|None|orphan_flex_q_day_balance (not in objective)': 0.0, 'DSO|5|voltage_slack': -0.7679999999999708, 'DSO|5|node_balance_slack': 0.0, 'DSO|5|branch_flow_slack': 0.0, 'DSO|5|flexibility_p_day_balance_slack': -0.06400000000000004, 'DSO|5|local_ess_day_balance_slack': 0.0, 'DSO|5|shared_ess_day_balance_slack': -0.002, 'DSO|5|orphan_flex_q_day_balance (not in objective)': 0.0, 'DSO|7|voltage_slack': -0.7679999999999708, 'DSO|7|node_balance_slack': 0.0, 'DSO|7|branch_flow_slack': 0.0, 'DSO|7|flexibility_p_day_balance_slack': -0.06400000000000004, 'DSO|7|local_ess_day_balance_slack': 0.0, 'DSO|7|shared_ess_day_balance_slack': -0.002, 'DSO|7|orphan_flex_q_day_balance (not in objective)': 0.0, 'DSO|9|voltage_slack': -0.7679999999999708, 'DSO|9|node_balance_slack': 0.0, 'DSO|9|branch_flow_slack': 0.0, 'DSO|9|flexibility_p_day_balance_slack': -0.06400000000000004, 'DSO|9|local_ess_day_balance_slack': 0.0, 'DSO|9|shared_ess_day_balance_slack': -0.002, 'DSO|9|orphan_flex_q_day_balance (not in objective)': 0.0}. Max level/floor ratio: {'TSO|None|voltage_slack': 0.9090910098401167, 'TSO|None|shared_ess_day_balance_slack': 0.9542955886818798, 'DSO|5|voltage_slack': 0.9818180745461587, 'DSO|5|flexibility_p_day_balance_slack': 0.9909055973322661, 'DSO|5|shared_ess_day_balance_slack': 0.9909090940826608, 'DSO|7|voltage_slack': 0.9999026540371538, 'DSO|7|flexibility_p_day_balance_slack': 0.9999508772375642, 'DSO|7|shared_ess_day_balance_slack': 0.9999513273247057, 'DSO|9|voltage_slack': 0.9818181118865307, 'DSO|9|flexibility_p_day_balance_slack': 0.9909087924467874, 'DSO|9|shared_ess_day_balance_slack': 0.9909090931397252}. ratio = level / floor in (0, 1]: 1 means every free slack of that family sits at its relaxed bound; structure (which entries are free) is read from the P515S44 model sets; a record whose structure had MORE free entries could legitimately go below this floor

x = 0 record data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7aa017f09989b56d_x0 (`8435c71859dd`): detector total -12743.4186, Q(0) = 653859461.2279. Max |d_det| over certified x != 0 records: 180.9895; max |d_det| / bar(x): 0.0586; max |det| / Q: 1.975e-05.

Baseline (s47_recert:n7_4h_e1, `db77e1549af8`): detector -12762.7696, d_det vs x0 -19.3510, value 259427.7678, bar 25104.6546, |d_det|/bar 0.0008, |d_det|/|value| 7.459e-05.

Formulas: {'d_detector_vs_x0': 'det(x) - det(0), det = recourse_components.detector_penalty_total (weighted, EUR)', 'value': 'Q(0) - Q(x), Q = evaluation_record.certified_cost', 'effect_on_value': 'Q_reported(x) = Q_projected(x) + bias(x), bias = level_reported - level_projected (P3); value_reported = value_projected + bias(0) - bias(x). Where no slack value is positive (every P515S44 model set, P3) bias = det, so value_reported - value_projected = det(0) - det(x) = -d_detector_vs_x0. Where a block carries positive slack values (P1 positive_detector_rows) the report does not separate the positive and negative parts, so bias(x) is not determined from the report for that record'}

| record | label | key | Q(x) | detector | d_det vs x0 | value | bar | abs(d_det)/bar |
|---|---|---|---|---|---|---|---|---|
| P515S44/campaign_s44_aa_variant/4e53fa5560bbfa10 | two_c_star_d | `4e53fa5560bb` | 648138276.77 | -12801.47 | -58.05 | 5721184.46 | 1542.71 | 0.0376 |
| P515S44/campaign_s44_aa_variant/837fc982565dbba3 | c_star_aa_keep_memory | `578636daa6d6` | 650982939.94 | -12801.47 | -58.05 | 2876521.29 | 5895.48 | 0.0098 |
| P515S44/campaign_s44_gate/578636daa6d6360d | c_star | `578636daa6d6` | 650966975.29 | -12801.47 | -58.05 | 2892485.93 | 7898.63 | 0.0073 |
| P515S44/campaign_s44_gate/d1a02e67107d0bac | paper_plan | `d1a02e67107d` | 653029766.99 | -12762.77 | -19.35 | 829694.24 | 661.58 | 0.0293 |
| P515S44/campaign_s44_gate/e30704e6e4dd3765 | node7_empty | `e30704e6e4dd` | 651900014.16 | -12782.12 | -38.70 | 1959447.07 | 9104.68 | 0.0043 |
| P515S44/campaign_s44_selection_aa/0796b6dceeb0f95f | node7_empty_aa_keep_memory | `e30704e6e4dd` | 651918540.60 | -12782.12 | -38.70 | 1940920.63 | 6228.2 | 0.0062 |
| P515S44/campaign_s44_selection_aa/8e48f3ec8993a283 | paper_plan_aa_keep_memory | `d1a02e67107d` | 653029455.39 | -12762.77 | -19.35 | 830005.84 | 6861.74 | 0.0028 |
| P515S44/campaign_s44_selection_aa/a6ca6c94e033e460 | two_c_star_aa_keep_memory | `4e53fa5560bb` | 648151169.48 | -12801.47 | -58.05 | 5708291.75 | 1347.94 | 0.0431 |
| P515S44/harness_checks_r3/aa_ref | c_star | `578636daa6d6` | 650966975.29 | -12801.47 | -58.05 | 2892485.93 | 7898.63 | 0.0073 |
| P515S45/campaign_s45_a0_c7/26a3878f77cfba4f | n7_p0.25_e0.5 | `56b9730bfb7b` | 653736823.97 | -12762.77 | -19.35 | 122637.26 | 8289.58 | 0.0023 |
| P515S45/campaign_s45_a0_c7/3be858940664d29a | n9_p0.25_e1.0 | `a361ac932145` | 653602441.60 | -12762.77 | -19.35 | 257019.63 | 11766.43 | 0.0016 |
| P515S45/campaign_s45_a0_c7/7aa017f09989b56d | x0 | `8435c71859dd` | 653859461.23 | -12743.42 | 0.00 | 0.00 | 9629.98 | 0.0 |
| P515S45/campaign_s45_a0_c7/7eb1ce62c2509f54 | n7_p0.25_e1.0 | `db77e1549af8` | 653597653.72 | -12762.77 | -19.35 | 261807.50 | 11340.32 | 0.0017 |
| P515S45/campaign_s45_a0_c7/b58dfe3fb3e562fd | n5_p0.25_e1.0 | `1821665996ec` | 653614718.03 | -12762.77 | -19.35 | 244743.19 | 3815.98 | 0.0051 |
| P515S45/campaign_s45_a0_c7/d0963078cb50c989 | lattice_plan_n7_p1.5_e3.0 | `2cd1de8249cf` | 653099300.59 | -12762.77 | -19.35 | 760160.64 | 2690.2 | 0.0072 |
| P515S45/campaign_s45_a0_c7/dd97e3d0dc598843 | n5_p0.25_e0.5 | `33031c083b68` | 653721243.98 | -12762.77 | -19.35 | 138217.24 | 5749.46 | 0.0034 |
| P515S45/campaign_s45_a0_c7/fba341303944b6f2 | n9_p0.25_e0.5 | `c50a0e021a23` | 653755659.40 | -12762.77 | -19.35 | 103801.82 | 19052.45 | 0.001 |
| P515S45/campaign_s45_a1a/04ad31d9a00099a2 | n7_4h_e4 | `11725e45371a` | 652873847.31 | -12762.77 | -19.35 | 985613.92 | 11330.33 | 0.0017 |
| P515S45/campaign_s45_a1a/0dffd41717e21056 | n9_4h_e3 | `ace7403b9b27` | 653127641.06 | -12762.77 | -19.35 | 731820.17 | 6311.74 | 0.0031 |
| P515S45/campaign_s45_a1a/17abdd0fa401951f | n7_2h_e5 | `354b38814b9c` | 652589462.83 | -12762.77 | -19.35 | 1269998.39 | 4721.01 | 0.0041 |
| P515S45/campaign_s45_a1a/311ccaa39ed5ded3 | n7_2h_e2 | `423e5a7f623e` | 653325523.06 | -12762.77 | -19.35 | 533938.17 | 6800.67 | 0.0028 |
| P515S45/campaign_s45_a1a/3be858940664d29a | n9_4h_e1 | `a361ac932145` | 653602441.60 | -12762.77 | -19.35 | 257019.63 | 11766.43 | 0.0016 |
| P515S45/campaign_s45_a1a/47f0ff739d3ddbea | n5_4h_e5 | `f0bfdf1c8adb` | 652608641.73 | -12762.77 | -19.35 | 1250819.50 | 6906.92 | 0.0028 |
| P515S45/campaign_s45_a1a/5f530b6bf49fe137 | n5_2h_e4 | `f20fb813dfef` | 652796655.57 | -12762.77 | -19.35 | 1062805.66 | 9446.57 | 0.002 |
| P515S45/campaign_s45_a1a/6ad3641937f6a637 | n9_2h_e4 | `30d0f10d376b` | 652804572.33 | -12762.77 | -19.35 | 1054888.90 | 1096.03 | 0.0177 |
| P515S45/campaign_s45_a1a/71c0860a9f5f53a5 | n9_4h_e5 | `b2e9d339a0f7` | 652639190.60 | -12762.77 | -19.35 | 1220270.63 | 4920.51 | 0.0039 |
| P515S45/campaign_s45_a1a/7bf551efdebeeec1 | n7_4h_e2 | `cc4eb5df3858` | 653372839.70 | -12762.77 | -19.35 | 486621.53 | 6212.03 | 0.0031 |
| P515S45/campaign_s45_a1a/7c66ddb648ce86a1 | n5_2h_e5 | `302735fb1984` | 652564046.86 | -12562.43 | 180.99 | 1295414.37 | 3089.48 | 0.0586 |
| P515S45/campaign_s45_a1a/7c7afed64d16ff89 | n7_2h_e1 | `3993aebc641d` | 653598722.55 | -12762.77 | -19.35 | 260738.68 | 5953.06 | 0.0033 |
| P515S45/campaign_s45_a1a/7eb1ce62c2509f54 | n7_4h_e1 | `db77e1549af8` | 653597653.72 | -12762.77 | -19.35 | 261807.50 | 11340.32 | 0.0017 |
| P515S45/campaign_s45_a1a/80cf840cf4f80d0b | n7_4h_e5 | `cab98853bb31` | 652650749.06 | -12762.77 | -19.35 | 1208712.16 | 4082.62 | 0.0047 |
| P515S45/campaign_s45_a1a/80f2490bcb48991e | n5_2h_e1 | `2a1302016f89` | 653587843.31 | -12762.77 | -19.35 | 271617.92 | 4611.62 | 0.0042 |
| P515S45/campaign_s45_a1a/87282a2536866085 | n9_4h_e2 | `6ed97c64f357` | 653402723.90 | -12762.77 | -19.35 | 456737.33 | 11364.7 | 0.0017 |
| P515S45/campaign_s45_a1a/8b07a41303810662 | n9_2h_e1 | `75271798795b` | 653590205.50 | -12762.77 | -19.35 | 269255.73 | 5759.77 | 0.0034 |
| P515S45/campaign_s45_a1a/9b77039f5ee42cec | n5_4h_e2 | `3eed1a1ec1a7` | 653337271.46 | -12762.77 | -19.35 | 522189.76 | 10862.16 | 0.0018 |
| P515S45/campaign_s45_a1a/a719e99b6208078e | n7_2h_e4 | `7048a00e1304` | 652821679.08 | -12762.77 | -19.35 | 1037782.15 | 9999.97 | 0.0019 |
| P515S45/campaign_s45_a1a/a90fc9344ad0eb7c | n5_4h_e4 | `f92d165248fe` | 652853134.61 | -12762.77 | -19.35 | 1006326.62 | 9987.49 | 0.0019 |
| P515S45/campaign_s45_a1a/aa43b799f64cbd9d | n9_2h_e3 | `dd970259278d` | 653082632.87 | -12762.77 | -19.35 | 776828.36 | 9296.3 | 0.0021 |
| P515S45/campaign_s45_a1a/b58dfe3fb3e562fd | n5_4h_e1 | `1821665996ec` | 653614718.03 | -12762.77 | -19.35 | 244743.19 | 3815.98 | 0.0051 |
| P515S45/campaign_s45_a1a/baeb374612983211 | n9_4h_e4 | `466fb2a6c3b1` | 652852486.38 | -12762.77 | -19.35 | 1006974.85 | 9972.24 | 0.0019 |
| P515S45/campaign_s45_a1a/c7fc0dc56ab4951f | n5_2h_e3 | `bb375de471aa` | 653076367.44 | -12762.77 | -19.35 | 783093.79 | 2251.5 | 0.0086 |
| P515S45/campaign_s45_a1a/c83860658163d970 | n7_4h_e3 | `fb21d822e145` | 653123596.80 | -12762.77 | -19.35 | 735864.43 | 8423.92 | 0.0023 |
| P515S45/campaign_s45_a1a/d0963078cb50c989 | n7_2h_e3 | `2cd1de8249cf` | 653099300.59 | -12762.77 | -19.35 | 760160.64 | 2690.2 | 0.0072 |
| P515S45/campaign_s45_a1a/e09ea9b6571a4848 | n5_2h_e2 | `5097b0c6b14e` | 653310380.48 | -12762.77 | -19.35 | 549080.74 | 7961.73 | 0.0024 |
| P515S45/campaign_s45_a1a/e3acbedf92088337 | n5_4h_e3 | `32acd70c1b79` | 653109230.43 | -12762.77 | -19.35 | 750230.79 | 14465.9 | 0.0013 |
| P515S45/campaign_s45_a1a/ec4ae6c80301fd19 | n9_2h_e2 | `46628fb108a5` | 653307646.98 | -12762.77 | -19.35 | 551814.25 | 10139.92 | 0.0019 |
| P515S45/campaign_s45_a1a/fef8869bfce3b416 | n9_2h_e5 | `a2a2bd2b7e9d` | 652566422.19 | -12762.77 | -19.35 | 1293039.04 | 2763.9 | 0.007 |
| P515S45/campaign_s45_a1b/0bb38720f1a4426a | n7_2h_e3_y2030 | `0226e5e77299` | 653355838.71 | -12755.67 | -12.25 | 503622.52 | 15352.63 | 0.0008 |
| P515S45/campaign_s45_a1b/12d566208836c1ee | n7_4h_e3_y2035 | `1435efacb664` | 653610370.36 | -12749.24 | -5.82 | 249090.87 | 5366.46 | 0.0011 |
| P515S45/campaign_s45_a1b/25c32e351ead53e1 | n7_4h_e4_y2030 | `0025f3453f36` | 653221518.52 | -12755.67 | -12.25 | 637942.71 | 35469.38 | 0.0003 |
| P515S45/campaign_s45_a1b/37b5c4999f65749f | n7_2h_e1_y2030 | `e7c94efae91d` | 653669414.30 | -12755.67 | -12.25 | 190046.93 | 8669.74 | 0.0014 |
| P515S45/campaign_s45_a1b/4294aae365384186 | n7_2h_e5_y2035 | `947de89b21eb` | 653413139.47 | -12749.24 | -5.82 | 446321.76 | 830.54 | 0.007 |
| P515S45/campaign_s45_a1b/47dce43c9f4ab0aa | n7_2h_e1_y2035 | `803d1a6cf8f2` | 653790388.23 | -12749.24 | -5.82 | 69073.00 | 14682.46 | 0.0004 |
| P515S45/campaign_s45_a1b/48749148d0bebd0d | n7_4h_e2_y2030 | `76baf11ecc05` | 653527588.92 | -12755.67 | -12.25 | 331872.31 | 35239.5 | 0.0003 |
| P515S45/campaign_s45_a1b/4fd41116d3aea30b | n7_2h_e4_y2030 | `6d0daf8d44ff` | 653173091.65 | -12755.67 | -12.25 | 686369.57 | 15674.05 | 0.0008 |
| P515S45/campaign_s45_a1b/52e05bc14e85c4cd | n7_2h_e4_y2035 | `930e7a1528dd` | 653515395.07 | -12749.24 | -5.82 | 344066.16 | 6622.53 | 0.0009 |
| P515S45/campaign_s45_a1b/549476cd276a0350 | n7_4h_e1_y2030 | `c408836ee588` | 653689107.46 | -12755.67 | -12.25 | 170353.77 | 10181.32 | 0.0012 |
| P515S45/campaign_s45_a1b/69e0758622e1740c | n7_4h_e4_y2035 | `ab6edede03e3` | 653510195.81 | -12749.24 | -5.82 | 349265.42 | 14377.0 | 0.0004 |
| P515S45/campaign_s45_a1b/7075078094340ac5 | n7_2h_e3_y2035 | `f90159726c7d` | 653582342.13 | -12749.24 | -5.82 | 277119.10 | 6423.54 | 0.0009 |
| P515S45/campaign_s45_a1b/8704c8d2b3ff6135 | n7_2h_e2_y2035 | `389e6e079d67` | 653701282.93 | -12749.24 | -5.82 | 158178.29 | 8271.44 | 0.0007 |
| P515S45/campaign_s45_a1b/9abf31d494b664b8 | n7_4h_e2_y2035 | `ca1b7bac4ba5` | 653677237.89 | -12749.24 | -5.82 | 182223.34 | 22999.2 | 0.0003 |
| P515S45/campaign_s45_a1b/b5e6809061d352e3 | n7_4h_e3_y2030 | `3b77a5f6804b` | 653381538.31 | -12755.67 | -12.25 | 477922.92 | 7732.14 | 0.0016 |
| P515S45/campaign_s45_a1b/c8e931ae7693719e | n7_4h_e5_y2035 | `ed8c36fd40c6` | 653425375.57 | -12749.24 | -5.82 | 434085.66 | 34988.68 | 0.0002 |
| P515S45/campaign_s45_a1b/cbb9e21a7708feb9 | n7_2h_e2_y2030 | `44cfc7e5b063` | 653513161.04 | -12755.67 | -12.25 | 346300.19 | 14690.28 | 0.0008 |
| P515S45/campaign_s45_a1b/d14d13ec3c38c601 | n7_2h_e5_y2030 | `b1d7831dfc0d` | 653004397.43 | -12755.67 | -12.25 | 855063.80 | 10830.74 | 0.0011 |
| P515S45/campaign_s45_a1b/dab6a8a2df221ba9 | n7_4h_e1_y2035 | `7fca7670b9b7` | 653776244.70 | -12749.24 | -5.82 | 83216.53 | 4662.93 | 0.0012 |
| P515S45/campaign_s45_a1b/dfe2e8e971318a20 | n7_4h_e5_y2030 | `b1429b9d9a4a` | 653064055.98 | -12755.67 | -12.25 | 795405.25 | 11104.01 | 0.0011 |
| P515S45/campaign_s45_a2/3be0745fa72df1ac | presence_n7_n9 | `85d6d95a8ef8` | 653374248.72 | -12782.12 | -38.70 | 485212.50 | 20579.14 | 0.0019 |
| P515S45/campaign_s45_a2/98678e4ebd60579b | presence_n5_n9 | `f9689a4b1d27` | 653352696.86 | -12782.12 | -38.70 | 506764.36 | 5807.16 | 0.0067 |
| P515S45/campaign_s45_a2/b571ca1b3ec3bc86 | presence_n5_n7_n9 | `6fbf40202169` | 653095878.35 | -12801.47 | -58.05 | 763582.88 | 19572.27 | 0.003 |
| P515S45/campaign_s45_a2/c5ae12a1ce18b5b7 | presence_n5_n7 | `00b96fd5cdf5` | 653363180.63 | -12782.12 | -38.70 | 496280.59 | 9594.03 | 0.004 |
| P515S45/campaign_s45_a2/eae950d246a4c396 | lattice_c_star_p1.0_e4.0 | `a002e3ab10e2` | 650893053.38 | -12801.47 | -58.05 | 2966407.85 | 7607.65 | 0.0076 |
| P515S45/campaign_s45_a3/37275e28d2255a58 | res_n7_p0.75_e1.5_y2025 | `4f135d7a623c` | 653464681.80 | -12762.77 | -19.35 | 394779.42 | 5069.49 | 0.0038 |
| P515S45/campaign_s45_a3/7c99998d824189c7 | res_n7_p1_e2.5_y2025 | `3e15cb08487b` | 653213468.47 | -12768.74 | -25.32 | 645992.75 | 7500.91 | 0.0034 |
| P515S45/campaign_s45_a3/acd6c3d25096d2fd | res_n7_p0.5_e1.5_y2025 | `062178ab2f3a` | 653475798.16 | -12762.77 | -19.35 | 383663.06 | 7956.48 | 0.0024 |
| P515S45/campaign_s45_a3/c606fa793da2a3ff | res_n7_p0.75_e2.5_y2025 | `222e4f71cc13` | 653232459.33 | -12762.77 | -19.35 | 627001.89 | 8559.76 | 0.0023 |
| P515S45/campaign_s45_a3/f4affa64172cea44 | res_n7_p1.25_e2.5_y2025 | `fdfe88cd4342` | 653194764.65 | -12762.77 | -19.35 | 664696.58 | 7529.55 | 0.0026 |
| P515S45/reverify_aa_c_star/3e741dac72c9e1bc | c_star_aa_case_file | `578636daa6d6` | 650982939.94 | -12801.47 | -58.05 | 2876521.29 | 5895.48 | 0.0098 |
| P515S46/campaign_s46_ageing/06f092d164f13819 | n7_4h_e1_no_ageing | `db77e1549af8` | 653541019.35 | -12762.77 | -19.35 | 318441.88 | 9180.61 | 0.0021 |
| P515S46/campaign_s46_ageing/65a5da775d1ff5b2 | n7_4h_e1_C4 | `db77e1549af8` | 653582507.77 | -12762.77 | -19.35 | 276953.46 | 13980.19 | 0.0014 |
| P515S46/campaign_s46_ageing/98e2857016a16d1c | n7_4h_e1_C2_calfade | `db77e1549af8` | 653585696.83 | -12762.77 | -19.35 | 273764.40 | 11516.99 | 0.0017 |
| P515S46/campaign_s46_ageing/c6b53015fcf65e24 | n7_4h_e1_C2 | `db77e1549af8` | 653564588.26 | -12762.77 | -19.35 | 294872.97 | 14269.07 | 0.0014 |
| P515S46/campaign_s46_ageing/ed4a1acc7059784d | n7_4h_e1_C3_midblock | `db77e1549af8` | 653595347.98 | -12762.77 | -19.35 | 264113.25 | 14903.22 | 0.0013 |
| P515S47/campaign_s47_a1a_baseline/0dd237f0d286db79 | n9_4h_e1 | `a361ac932145` | 653597304.27 | -12762.77 | -19.35 | 262156.96 | 37547.85 | 0.0005 |
| P515S47/campaign_s47_a1a_baseline/1c1ef751d9334f40 | n5_2h_e2 | `5097b0c6b14e` | 653340344.29 | -12762.77 | -19.35 | 519116.94 | 4117.96 | 0.0047 |
| P515S47/campaign_s47_a1a_baseline/209b9d3362f045f4 | n9_2h_e5 | `a2a2bd2b7e9d` | 652549647.82 | -12762.77 | -19.35 | 1309813.41 | 22342.83 | 0.0009 |
| P515S47/campaign_s47_a1a_baseline/2a0ba8b2f3d3b99e | n5_4h_e1 | `1821665996ec` | 653595110.77 | -12762.77 | -19.35 | 264350.45 | 5120.02 | 0.0038 |
| P515S47/campaign_s47_a1a_baseline/2e73c7ae12b80205 | n9_2h_e2 | `46628fb108a5` | 653335542.24 | -12762.77 | -19.35 | 523918.98 | 8544.33 | 0.0023 |
| P515S47/campaign_s47_a1a_baseline/3632b0ae8de45fc6 | n7_4h_e4 | `11725e45371a` | 652857762.61 | -12762.77 | -19.35 | 1001698.62 | 43606.46 | 0.0004 |
| P515S47/campaign_s47_a1a_baseline/366864896179e01d | n7_4h_e5 | `cab98853bb31` | 652623187.72 | -12583.28 | 160.13 | 1236273.51 | 10512.6 | 0.0152 |
| P515S47/campaign_s47_a1a_baseline/3e57c18a1a134bef | n5_4h_e5 | `f0bfdf1c8adb` | 652598924.90 | -12762.77 | -19.35 | 1260536.33 | 37166.74 | 0.0005 |
| P515S47/campaign_s47_a1a_baseline/3ee9cd30873c7b3b | n5_2h_e3 | `bb375de471aa` | 653061517.12 | -12762.77 | -19.35 | 797944.11 | 43466.32 | 0.0004 |
| P515S47/campaign_s47_a1a_baseline/4649234b0f6624b9 | n9_4h_e3 | `ace7403b9b27` | 653083282.72 | -12762.77 | -19.35 | 776178.50 | 36068.03 | 0.0005 |
| P515S47/campaign_s47_a1a_baseline/4a82a64a21aa67cc | n7_4h_e3 | `fb21d822e145` | 653104540.55 | -12762.77 | -19.35 | 754920.68 | 28589.71 | 0.0007 |
| P515S47/campaign_s47_a1a_baseline/58de187f7806c66e | n9_2h_e4 | `30d0f10d376b` | 652828155.00 | -12598.42 | 145.00 | 1031306.23 | 24543.42 | 0.0059 |
| P515S47/campaign_s47_a1a_baseline/6a5f1810c7764ec6 | n5_2h_e5 | `302735fb1984` | 652570215.22 | -12762.77 | -19.35 | 1289246.01 | 4402.43 | 0.0044 |
| P515S47/campaign_s47_a1a_baseline/6e1cdc5ca7ff1c12 | n5_4h_e2 | `3eed1a1ec1a7` | 653352582.39 | -12762.77 | -19.35 | 506878.84 | 15084.38 | 0.0013 |
| P515S47/campaign_s47_a1a_baseline/7ec6d3a2e040b940 | n9_2h_e3 | `dd970259278d` | 653045261.03 | -12762.77 | -19.35 | 814200.20 | 34301.03 | 0.0006 |
| P515S47/campaign_s47_a1a_baseline/8160e09e0d7e7ad8 | n5_4h_e4 | `f92d165248fe` | 652840750.35 | -12762.77 | -19.35 | 1018710.88 | 10496.78 | 0.0018 |
| P515S47/campaign_s47_a1a_baseline/9246ed01f072b31f | n7_2h_e5 | `354b38814b9c` | 652554364.83 | -12762.77 | -19.35 | 1305096.39 | 15947.66 | 0.0012 |
| P515S47/campaign_s47_a1a_baseline/9e495a608aa822cd | n5_4h_e3 | `32acd70c1b79` | 653109946.09 | -12762.77 | -19.35 | 749515.14 | 5414.41 | 0.0036 |
| P515S47/campaign_s47_a1a_baseline/a12d95a2a952693c | n7_2h_e2 | `423e5a7f623e` | 653333260.69 | -12762.77 | -19.35 | 526200.53 | 26773.7 | 0.0007 |
| P515S47/campaign_s47_a1a_baseline/a8593c4fd712a218 | n9_4h_e2 | `6ed97c64f357` | 653372064.85 | -12762.77 | -19.35 | 487396.38 | 25527.13 | 0.0008 |
| P515S47/campaign_s47_a1a_baseline/bd504ecf5a288d44 | n7_4h_e1 | `db77e1549af8` | 653600033.46 | -12762.77 | -19.35 | 259427.77 | 25104.65 | 0.0008 |
| P515S47/campaign_s47_a1a_baseline/be0a39962f866f68 | n9_4h_e4 | `466fb2a6c3b1` | 652842882.77 | -12762.77 | -19.35 | 1016578.46 | 38797.42 | 0.0005 |
| P515S47/campaign_s47_a1a_baseline/c52e167077ed3bbd | n7_4h_e2 | `cc4eb5df3858` | 653364259.33 | -12762.77 | -19.35 | 495201.90 | 8969.5 | 0.0022 |
| P515S47/campaign_s47_a1a_baseline/c7fee8beeb210466 | n7_2h_e4 | `7048a00e1304` | 652802918.76 | -12762.77 | -19.35 | 1056542.47 | 17100.92 | 0.0011 |
| P515S47/campaign_s47_a1a_baseline/d3709599c7c42807 | n7_2h_e1 | `3993aebc641d` | 653588078.13 | -12762.77 | -19.35 | 271383.09 | 16015.79 | 0.0012 |
| P515S47/campaign_s47_a1a_baseline/d877282f87fc103c | n9_4h_e5 | `b2e9d339a0f7` | 652605117.12 | -12762.77 | -19.35 | 1254344.11 | 29300.74 | 0.0007 |
| P515S47/campaign_s47_a1a_baseline/e0fd79e013daa0d8 | n5_2h_e4 | `f20fb813dfef` | 652832585.93 | -12762.77 | -19.35 | 1026875.30 | 5926.56 | 0.0033 |
| P515S47/campaign_s47_a1a_baseline/eac75f1c2edc983a | n9_2h_e1 | `75271798795b` | 653590232.81 | -12762.77 | -19.35 | 269228.41 | 39039.99 | 0.0005 |
| P515S47/campaign_s47_a1a_baseline/ecfcd325ddf18d43 | n5_2h_e1 | `2a1302016f89` | 653588053.04 | -12600.20 | 143.21 | 271408.19 | 21760.49 | 0.0066 |
| P515S47/campaign_s47_a1a_baseline/f759dd4825d3f3af | n7_2h_e3 | `2cd1de8249cf` | 653075483.46 | -12762.77 | -19.35 | 783977.77 | 8849.71 | 0.0022 |
| P515S47/campaign_s47_phase_b/10c73abd8511010d | y2025__n7_p0.25_e0.5 | `56b9730bfb7b` | 653725854.46 | -12762.77 | -19.35 | 133606.77 | 12515.05 | 0.0015 |
| P515S47/campaign_s47_phase_b/156ce2d1d53d36f4 | y2025__n5_p0.25_e0.5__n9_p0.25_e0.5 | `ba670a65868a` | 653574496.71 | -12782.12 | -38.70 | 284964.52 | 12498.86 | 0.0031 |
| P515S47/campaign_s47_phase_b/1bff3ed2fb98302a | y2025__n9_p0.25_e0.5 | `c50a0e021a23` | 653722932.48 | -12762.77 | -19.35 | 136528.75 | 37575.28 | 0.0005 |
| P515S47/campaign_s47_phase_b/226df5da6e1eb902 | y2030__n5_p0.25_e0.5__n9_p0.25_e0.5 | `013befee221f` | 653687714.17 | -12767.92 | -24.50 | 171747.06 | 5673.47 | 0.0043 |
| P515S47/campaign_s47_phase_b/25fb61a4b5a4a7c0 | y2025__n7_p0.25_e0.5__n9_p0.25_e0.5 | `a0258d800b0d` | 653591087.65 | -12782.12 | -38.70 | 268373.58 | 3781.25 | 0.0102 |
| P515S47/campaign_s47_phase_b/4a8527254a85b9cc | y2025__n5_p0.25_e0.5 | `33031c083b68` | 653711463.53 | -12762.77 | -19.35 | 147997.70 | 30558.74 | 0.0006 |
| P515S47/campaign_s47_phase_b/4f5bccb02f7ba155 | y2030__n7_p0.25_e0.5__n9_p0.25_e0.5 | `be2c55eba95b` | 653678526.78 | -12767.92 | -24.50 | 180934.45 | 6311.61 | 0.0039 |
| P515S47/campaign_s47_phase_b/5204ec67e0ba7bf5 | y2025__n5_p0.25_e0.5__n7_p0.25_e0.5 | `1a639ffc70d8` | 653579253.85 | -12782.12 | -38.70 | 280207.37 | 7163.9 | 0.0054 |
| P515S47/campaign_s47_phase_b/56bb0b466d2a63c2 | y2025__n5_p0.25_e0.5__n7_p0.25_e0.5__n9_p0.25_e0.5 | `cc03ce8e9b04` | 653438213.31 | -12801.47 | -58.05 | 421247.92 | 34158.44 | 0.0017 |
| P515S47/campaign_s47_phase_b/6597a79dbe8ed61f | y2030__n5_p0.25_e0.5__n7_p0.25_e0.5 | `a5bb68bf2bd1` | 653706335.69 | -12767.92 | -24.50 | 153125.54 | 34456.68 | 0.0007 |
| P515S47/campaign_s47_phase_b/9e9c61f81f584b05 | y2030__n5_p0.25_e0.5__n7_p0.25_e0.5__n9_p0.25_e0.5 | `f0445c7218ca` | 653566019.14 | -12780.17 | -36.76 | 293442.08 | 6187.69 | 0.0059 |
| P515S47/campaign_s47_phase_b/a30a9faffdd74f9e | y2030__n9_p0.25_e0.5 | `da2f6bc13b10` | 653751038.53 | -12755.67 | -12.25 | 108422.70 | 8182.43 | 0.0015 |
| P515S47/campaign_s47_phase_b/d0c1f1605a44d5a8 | y2030__n5_p0.25_e0.5 | `e0e614845a23` | 653761255.12 | -12755.67 | -12.25 | 98206.11 | 7370.26 | 0.0017 |
| P515S47/campaign_s47_phase_b/d7030f598c2aeb4d | y2030__n7_p0.25_e0.5 | `53c1de5c5a53` | 653759847.48 | -12755.67 | -12.25 | 99613.74 | 5614.61 | 0.0022 |
| P515S47/campaign_s47_recert/070f833e1e318f85 | c_star | `578636daa6d6` | 650912327.20 | -12801.47 | -58.05 | 2947134.03 | 25146.49 | 0.0023 |
| P515S47/campaign_s47_recert/bd504ecf5a288d44 | n7_4h_e1 | `db77e1549af8` | 653600033.46 | -12762.77 | -19.35 | 259427.77 | 25104.65 | 0.0008 |

Checks: 82; failed: []; errors: [].

