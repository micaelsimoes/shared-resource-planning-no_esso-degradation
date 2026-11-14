# P5.15 Addendum 41 W53 -- renewable-curtailment audit (zero solves, no model construction)

Zero solves (armed SolveProfileGuard(permitted=()), counts {'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}, verify(0) failures []); model-construction blockers ['Network.build_model', 'NetworkData.build_model', 'SharedEnergyStorageData.build_subproblem', 'SharedEnergyStorageData.build_master_problem'], blocked calls []. all_checks_pass = True; failing: [].

**Conventions.** Volumes in MWh (1 h periods). `c = (pg_avail - pg) * baseMVA`, the production definitional curtailment (active power). Per-day values are per representative day and scenario; horizon values are weighted by `admm_block_weight` and omega_s (the Q weighting). Priced amounts `C` are EUR at the scenario's hourly market price on the positive part of c, in the same units as the certified Q = `gross_operational_cost` (settlement-excluded); they are what pricing curtailment at the market price would add at the CURRENT dispatch (an upper bound on the change in Q). `C on net c` is the penalty term itself evaluated at the current point (the term is linear in c, negatives included); `C above tol only` restricts to generator-hours with c > TOL_MW. TOL_MW = EQUALITY_TOLERANCE x baseMVA = 0.001 MW per generator-hour.

## 1. Instances and the resolution recoverable from committed records

| instance | campaign / eval key | candidate | cycles | Q (gross, settlement-excl.) | bar | rule-ten ratio | source | resolution recoverable | duals |
|---|---|---|---|---|---|---|---|---|---|
| srp1_x0 (SRP1, x = 0) | s48_x0_capture / d2c96b1480402a3b | 8435c718 | 132 | 653,859,461.23 | 9,629.98 | 0.0034 | persisted terminal models | generator x hour (1 scenario) | yes |
| srp1_unit (SRP1, node 7 0.25 MVA / 1.0 MWh (2025)) | s47_recert / bd504ecf5a288d44 | db77e154 | 112 | 653,600,033.46 | 25,104.65 | 0.0567 | component_levels_terminal.json | block (network x year x day), day sum | no |
| pilot_x0 (2x2 pilot, x = 0) | s52_pilot_nopersist / 7d53b6f21b686a44 | 8435c718 | 74 | 841,964,496.83 | 8,403.29 | 0.0056 | terminal operational workbook (production writer, pre post-certification) | generator x hour x scenario | no |
| pilot_unit (2x2 pilot, node 7 0.25 MVA / 1.0 MWh (2025)) | s52_pilot_nopersist / 711fce9aa74d6878 | db77e154 | 72 | 841,720,175.54 | 7,107.55 | 0.0112 | terminal operational workbook (production writer, pre post-certification) | generator x hour x scenario | no |

- **srp1_x0**: Full per-generator, per-hour resolution with every multiplier. Missing: nothing for this audit. Cost to obtain: -
- **srp1_unit**: No persisted models and no terminal workbook exist for this eval (s47 ran without post-certification persistence); the only committed curtailment quantity is the per-block net day sum `res_curtailment_definitional_at_weight_1`. Missing: the per-hour profile, the positive part (net only), the capability/interior split. Cost to obtain: one re-run of s47_recert point n7_4h_e1 (eval key bd504ecf) with persist_certified_models, as s48_x0_capture did for x = 0: ~0.87 h wall (recorded run 3146 s, 112 cycles), ~2.6 GB RSS, ~164 MB pickle, plus a bitwise reproduction gate against s47_recert.
- **pilot_x0**: Contrary to the pre-task caveat, per-hour and per-scenario curtailment IS recoverable: the hash-recorded workbook holds pg_avail and pg per curtaillable generator, hour and scenario; it reconciles with the certified component levels (check below). Missing: every multiplier (no models), so the cause is shown by primal indicators only. Cost to obtain: one re-run of the pilot pair with persist_certified_models (recorded walls 3.76 h and 4.06 h at concurrency 2, ~4 h for the pair) and the +5 GiB certified-model pickle transient per child that the no-persistence ruling avoided; the frozen campaign_s52_pilot (with persistence) is the named run.
- **pilot_unit**: Contrary to the pre-task caveat, per-hour and per-scenario curtailment IS recoverable: the hash-recorded workbook holds pg_avail and pg per curtaillable generator, hour and scenario; it reconciles with the certified component levels (check below). Missing: every multiplier (no models), so the cause is shown by primal indicators only. Cost to obtain: one re-run of the pilot pair with persist_certified_models (recorded walls 3.76 h and 4.06 h at concurrency 2, ~4 h for the pair) and the +5 GiB certified-model pickle transient per child that the no-persistence ruling avoided; the frozen campaign_s52_pilot (with persistence) is the named run.

## 2. Curtailment penalty in force, per path

Read-back from the certified artifacts: certified models (srp1_x0, 48 blocks): penalty_gen_curtailment = [0.0], interface_settlement_weight = [1.0]; component levels res_curtailment_penalty = srp1_x0 [0.0]; srp1_unit [0.0]; pilot_x0 [0.0]; pilot_unit [0.0]; pilot per-scenario penalty = pilot_x0 [0.0]; pilot_unit [0.0]. PENALTY_GENERATION_CURTAILMENT imported now = 1.0.

| path | weight on c | settlement weight | where (file:line at the run commits / HEAD) |
|---|---|---|---|
| ADMM TSO subproblem (every cycle; the certified Q) | 0 EUR/MWh | 1 | shared_resources_planning.py:4970 (set) after the build at model_construction_helpers.py:1660, model_construction_helpers.py:1664; call shared_resources_planning.py:2913 |
| ADMM DSO subproblem (every cycle; the certified Q) | 0 EUR/MWh | 1 | shared_resources_planning.py:5151 (set) after the build at model_construction_helpers.py:1660, model_construction_helpers.py:1664; call shared_resources_planning.py:2912 |
| initialisation solve (TSO and DSO, before _prepare_*_for_admm) | 1.0 EUR/MWh (the constant) | 0 | definitions.py:59; model_construction_helpers.py:1660, model_construction_helpers.py:1664; term model_construction_helpers.py:2118; solves shared_resources_planning.py:4240, shared_resources_planning.py:4328, shared_resources_planning.py:4441 |
| standalone / uncoordinated benchmark (_run_operational_planning_without_coordination) | 1.0 EUR/MWh (never reset there; see the reference list) | 0 | definitions.py:59; model_construction_helpers.py:1660, model_construction_helpers.py:1664; solves shared_resources_planning.py:4600, shared_resources_planning.py:8496, shared_resources_planning.py:4635, shared_resources_planning.py:8431 (enclosing functions in the per-commit table) |

Per-commit line numbers (pattern hits with the enclosing function):

| pattern role | s48_x0_capture run | s47_recert run | s52 pilot run | HEAD |
|---|---|---|---|---|
| the constant (initialisation / standalone weight) | definitions.py:59 (module level) | definitions.py:59 (module level) | definitions.py:59 (module level) | definitions.py:59 (module level) |
| bound tolerance (pg upper bound = pg_avail + tol) | definitions.py:90 (module level) | definitions.py:90 (module level) | definitions.py:90 (module level) | definitions.py:90 (module level) |
| Param created at the constant on every build (first hit: OBJ_MIN_COST, which every SRP1 network uses) | model_construction_helpers.py:1570 (setup_cost_parameters), model_construction_helpers.py:1574 (setup_cost_parameters) | model_construction_helpers.py:1570 (setup_cost_parameters), model_construction_helpers.py:1574 (setup_cost_parameters) | model_construction_helpers.py:1660 (setup_cost_parameters), model_construction_helpers.py:1664 (setup_cost_parameters) | model_construction_helpers.py:1660 (setup_cost_parameters), model_construction_helpers.py:1664 (setup_cost_parameters) |
| the objective term: penalty * B * (pg_avail - pg) | model_construction_helpers.py:1780 (gen_curtailment_penalty) | model_construction_helpers.py:1780 (gen_curtailment_penalty) | model_construction_helpers.py:2118 (gen_curtailment_penalty) | model_construction_helpers.py:2118 (gen_curtailment_penalty) |
| settlement weight 0 on every build (the initialisation import is unpriced) | model_construction_helpers.py:1611 (build_objective) | model_construction_helpers.py:1611 (build_objective) | model_construction_helpers.py:1701 (build_objective) | model_construction_helpers.py:1701 (build_objective) |
| pg_adn = pg[ref] - shared_es_pnet: the DSO exchange is net of the storage at the reference bus | model_construction_helpers.py:1216 (interface_pf_p_distribution_def) | model_construction_helpers.py:1216 (interface_pf_p_distribution_def) | model_construction_helpers.py:1297 (interface_pf_p_distribution_def) | model_construction_helpers.py:1297 (interface_pf_p_distribution_def) |
| the shared ESS enters the node balance of its own bus | absent | absent | model_construction_helpers.py:1419 (compute_node_load) | model_construction_helpers.py:1419 (compute_node_load) |
| TSO initialisation / benchmark solve at the build weights | shared_resources_planning.py:3808 (create_transmission_network_model) | shared_resources_planning.py:3808 (create_transmission_network_model) | shared_resources_planning.py:4240 (create_transmission_network_model) | shared_resources_planning.py:4240 (create_transmission_network_model) |
| DSO initialisation / benchmark solve (sequential) at the build weights | shared_resources_planning.py:3881 (create_distribution_networks_models_sequential) | shared_resources_planning.py:3881 (create_distribution_networks_models_sequential) | shared_resources_planning.py:4328 (create_distribution_networks_models_sequential) | shared_resources_planning.py:4328 (create_distribution_networks_models_sequential) |
| DSO initialisation solve (parallel worker) at the build weights | shared_resources_planning.py:3988 (create_distribution_network_model) | shared_resources_planning.py:3988 (create_distribution_network_model) | shared_resources_planning.py:4441 (create_distribution_network_model) | shared_resources_planning.py:4441 (create_distribution_network_model) |
| ADMM path: DSO weights reset after the initialisation solve | shared_resources_planning.py:2525 (_run_operational_planning) | shared_resources_planning.py:2525 (_run_operational_planning) | shared_resources_planning.py:2912 (_run_operational_planning) | shared_resources_planning.py:2912 (_run_operational_planning) |
| ADMM path: TSO weights reset after the initialisation solve | shared_resources_planning.py:2526 (_run_operational_planning) | shared_resources_planning.py:2526 (_run_operational_planning) | shared_resources_planning.py:2913 (_run_operational_planning) | shared_resources_planning.py:2913 (_run_operational_planning) |
| ADMM TSO subproblem: curtailment weight set to 0 | shared_resources_planning.py:4505 (_prepare_transmission_objectives_for_admm) | shared_resources_planning.py:4505 (_prepare_transmission_objectives_for_admm) | shared_resources_planning.py:4970 (_prepare_transmission_objectives_for_admm) | shared_resources_planning.py:4970 (_prepare_transmission_objectives_for_admm) |
| ADMM DSO subproblem: curtailment weight set to 0 | shared_resources_planning.py:4686 (_prepare_distribution_objectives_for_admm) | shared_resources_planning.py:4686 (_prepare_distribution_objectives_for_admm) | shared_resources_planning.py:5151 (_prepare_distribution_objectives_for_admm) | shared_resources_planning.py:5151 (_prepare_distribution_objectives_for_admm) |
| ADMM path: settlement weight set to 1 | shared_resources_planning.py:4513 (_prepare_transmission_objectives_for_admm), shared_resources_planning.py:4694 (_prepare_distribution_objectives_for_admm) | shared_resources_planning.py:4513 (_prepare_transmission_objectives_for_admm), shared_resources_planning.py:4694 (_prepare_distribution_objectives_for_admm) | shared_resources_planning.py:4978 (_prepare_transmission_objectives_for_admm), shared_resources_planning.py:5159 (_prepare_distribution_objectives_for_admm) | shared_resources_planning.py:4978 (_prepare_transmission_objectives_for_admm), shared_resources_planning.py:5159 (_prepare_distribution_objectives_for_admm) |
| TSO solve in the hierarchical / uncoordinated benchmarks (enclosing function shown) | shared_resources_planning.py:4135 (_run_operational_planning_hierarchical), shared_resources_planning.py:8025 (_run_operational_planning_without_coordination) | shared_resources_planning.py:4135 (_run_operational_planning_hierarchical), shared_resources_planning.py:8025 (_run_operational_planning_without_coordination) | shared_resources_planning.py:4600 (_run_operational_planning_hierarchical), shared_resources_planning.py:8496 (_run_operational_planning_without_coordination) | shared_resources_planning.py:4600 (_run_operational_planning_hierarchical), shared_resources_planning.py:8496 (_run_operational_planning_without_coordination) |
| DSO solve in the hierarchical / uncoordinated benchmarks (enclosing function shown) | shared_resources_planning.py:4170 (_run_operational_planning_hierarchical), shared_resources_planning.py:7960 (_run_operational_planning_without_coordination) | shared_resources_planning.py:4170 (_run_operational_planning_hierarchical), shared_resources_planning.py:7960 (_run_operational_planning_without_coordination) | shared_resources_planning.py:4635 (_run_operational_planning_hierarchical), shared_resources_planning.py:8431 (_run_operational_planning_without_coordination) | shared_resources_planning.py:4635 (_run_operational_planning_hierarchical), shared_resources_planning.py:8431 (_run_operational_planning_without_coordination) |

## 3. Volumes per network (horizon, Q weighting)

**srp1_x0** (SRP1, x = 0); bar 9,629.98.

| network | net c (MWh) | positive part (MWh) | above tol (MWh) | available RES (MWh) | net share % | network-hours above tol | C at hourly price (EUR) | C on net c | C above tol only |
|---|---|---|---|---|---|---|---|---|---|
| DSO5 | 166.69 | 184.39 | 147.35 | 3,345,194 | 0.00498 | 119 | 17,651.95 | 15,383.66 | 14,329.37 |
| DSO7 | 255.65 | 269.42 | 232.80 | 4,429,706 | 0.00577 | 43 | 13,007.12 | 11,236.70 | 9,109.53 |
| DSO9 | 186.43 | 197.61 | 119.51 | 4,030,889 | 0.00463 | 92 | 14,968.54 | 13,535.10 | 7,930.69 |
| TSO | -18.81 | 0.06 | 0.00 | 6,815,629 | -0.00028 | 0 | 0.24 | -2,347.86 | 0.00 |
| **all** | 589.96 | 651.48 | 499.66 | 18,621,418 | | | **45,627.84** | 37,807.60 | 31,369.59 |

Per network and scenario (omega-weighted contribution to Q): DSO5|0_0 net 166.69 MWh, C 17,651.95; DSO7|0_0 net 255.65 MWh, C 13,007.12; DSO9|0_0 net 186.43 MWh, C 14,968.54; TSO|0_0 net -18.81 MWh, C 0.24

**pilot_x0** (2x2 pilot, x = 0); bar 8,403.29.

| network | net c (MWh) | positive part (MWh) | above tol (MWh) | available RES (MWh) | net share % | network-hours above tol | C at hourly price (EUR) | C on net c | C above tol only |
|---|---|---|---|---|---|---|---|---|---|
| DSO5 | 574.46 | 591.08 | 544.18 | 3,130,098 | 0.01835 | 747 | 26,461.95 | 24,222.21 | 21,844.28 |
| DSO7 | 459.63 | 473.07 | 430.96 | 4,271,498 | 0.01076 | 429 | 17,342.50 | 15,515.34 | 12,611.91 |
| DSO9 | 644.01 | 655.73 | 585.06 | 3,656,757 | 0.01761 | 699 | 20,862.53 | 19,282.61 | 13,605.62 |
| TSO | -11.00 | 5.00 | 0.85 | 6,359,614 | -0.00017 | 2 | 310.12 | -1,832.96 | 1.60 |
| **all** | 1,667.10 | 1,724.87 | 1,561.04 | 17,417,966 | | | **64,977.10** | 57,187.19 | 48,063.40 |

Per network and scenario (omega-weighted contribution to Q): DSO5|0_0 net 41.54 MWh, C 4,435.01; DSO5|0_1 net 430.66 MWh, C 9,846.11; DSO5|1_0 net 40.20 MWh, C 4,654.61; DSO5|1_1 net 62.06 MWh, C 7,526.23; DSO7|0_0 net 102.52 MWh, C 3,877.15; DSO7|0_1 net 237.96 MWh, C 5,205.77; DSO7|1_0 net 68.49 MWh, C 4,581.10; DSO7|1_1 net 50.65 MWh, C 3,678.48; DSO9|0_0 net 464.49 MWh, C 5,892.62; DSO9|0_1 net 74.75 MWh, C 4,807.94; DSO9|1_0 net 54.81 MWh, C 5,279.99; DSO9|1_1 net 49.96 MWh, C 4,881.99; TSO|0_0 net -2.39 MWh, C 78.04; TSO|0_1 net -2.38 MWh, C 79.30; TSO|1_0 net -3.12 MWh, C 74.65; TSO|1_1 net -3.11 MWh, C 78.13

**pilot_unit** (2x2 pilot, node 7 0.25 MVA / 1.0 MWh (2025)); bar 7,107.55.

| network | net c (MWh) | positive part (MWh) | above tol (MWh) | available RES (MWh) | net share % | network-hours above tol | C at hourly price (EUR) | C on net c | C above tol only |
|---|---|---|---|---|---|---|---|---|---|
| DSO5 | 574.58 | 591.19 | 544.34 | 3,130,098 | 0.01836 | 749 | 26,486.81 | 24,247.12 | 21,875.73 |
| DSO7 | 454.85 | 468.29 | 426.06 | 4,271,498 | 0.01065 | 430 | 17,279.97 | 15,452.84 | 12,539.19 |
| DSO9 | 644.00 | 655.71 | 585.03 | 3,656,757 | 0.01761 | 700 | 20,873.61 | 19,293.69 | 13,618.43 |
| TSO | -17.10 | 0.35 | 0.00 | 6,359,614 | -0.00027 | 0 | 5.20 | -2,308.93 | 0.00 |
| **all** | 1,656.32 | 1,715.55 | 1,555.43 | 17,417,966 | | | **64,645.59** | 56,684.72 | 48,033.36 |

Per network and scenario (omega-weighted contribution to Q): DSO5|0_0 net 41.57 MWh, C 4,437.94; DSO5|0_1 net 430.67 MWh, C 9,853.39; DSO5|1_0 net 40.23 MWh, C 4,658.63; DSO5|1_1 net 62.11 MWh, C 7,536.86; DSO7|0_0 net 101.17 MWh, C 3,873.56; DSO7|0_1 net 236.69 MWh, C 5,199.36; DSO7|1_0 net 67.13 MWh, C 4,548.29; DSO7|1_1 net 49.86 MWh, C 3,658.77; DSO9|0_0 net 464.46 MWh, C 5,895.39; DSO9|0_1 net 74.77 MWh, C 4,810.60; DSO9|1_0 net 54.80 MWh, C 5,282.55; DSO9|1_1 net 49.97 MWh, C 4,885.07; TSO|0_0 net -4.21 MWh, C 1.70; TSO|0_1 net -4.22 MWh, C 1.52; TSO|1_0 net -4.33 MWh, C 1.08; TSO|1_1 net -4.34 MWh, C 0.91

**srp1_unit** (block resolution only); bar 25,104.65.

| network | net c (MWh) | unit - x0 (MWh) | C upper bound (EUR) | abs dC vs x0 upper bound (EUR) |
|---|---|---|---|---|
| DSO5 | 166.68 | -0.0104 | 28,942.77 | 3.25 |
| DSO7 | 255.59 | -0.0535 | 44,503.82 | 17.25 |
| DSO9 | 186.32 | -0.1171 | 32,778.71 | 22.15 |
| TSO | -18.80 | 0.0083 | 0.00 | 3.11 |

## 4. Per network, year, day and scenario (MWh per representative day; rows with a positive part >= 0.01)

**srp1_x0**

| network | year | day | scen | omega | net | pos. part | above tol | Sg_curt (MVAh) | hours > tol | priced (EUR/day) |
|---|---|---|---|---|---|---|---|---|---|---|
| DSO5 | 2025 | Spring | 0_0 | 1.0 | 0.0142 | 0.0165 | 0.0093 | 0.0000 | 7 | 1.49 |
| DSO5 | 2025 | Summer | 0_0 | 1.0 | 0.0157 | 0.0206 | 0.0111 | 0.0000 | 8 | 1.49 |
| DSO5 | 2025 | Autumn | 0_0 | 1.0 | 0.0188 | 0.0232 | 0.0198 | 0.0000 | 8 | 2.19 |
| DSO5 | 2025 | Winter | 0_0 | 1.0 | 0.0335 | 0.0390 | 0.0337 | 0.0000 | 9 | 3.65 |
| DSO5 | 2030 | Spring | 0_0 | 1.0 | 0.0229 | 0.0243 | 0.0163 | 0.0000 | 10 | 2.31 |
| DSO5 | 2030 | Summer | 0_0 | 1.0 | 0.0334 | 0.0374 | 0.0267 | 0.0000 | 13 | 3.24 |
| DSO5 | 2030 | Autumn | 0_0 | 1.0 | 0.0267 | 0.0309 | 0.0258 | 0.0000 | 8 | 3.21 |
| DSO5 | 2030 | Winter | 0_0 | 1.0 | 0.0425 | 0.0473 | 0.0403 | 0.0000 | 9 | 4.89 |
| DSO5 | 2035 | Spring | 0_0 | 1.0 | 0.0364 | 0.0373 | 0.0292 | 0.0000 | 14 | 4.76 |
| DSO5 | 2035 | Summer | 0_0 | 1.0 | 0.0759 | 0.0798 | 0.0701 | 0.0000 | 14 | 5.81 |
| DSO5 | 2035 | Autumn | 0_0 | 1.0 | 0.0381 | 0.0399 | 0.0336 | 0.0000 | 10 | 3.24 |
| DSO5 | 2035 | Winter | 0_0 | 1.0 | 0.0564 | 0.0605 | 0.0509 | 0.0000 | 9 | 7.57 |
| DSO7 | 2025 | Spring | 0_0 | 1.0 | 0.1055 | 0.1076 | 0.1036 | 0.0000 | 4 | 3.08 |
| DSO7 | 2025 | Summer | 0_0 | 1.0 | 0.0740 | 0.0776 | 0.0719 | 0.0000 | 3 | 3.29 |
| DSO7 | 2025 | Winter | 0_0 | 1.0 | 0.0066 | 0.0110 | 0.0021 | 0.0000 | 2 | 1.05 |
| DSO7 | 2030 | Spring | 0_0 | 1.0 | 0.1074 | 0.1091 | 0.1017 | 0.0000 | 4 | 2.61 |
| DSO7 | 2030 | Summer | 0_0 | 1.0 | 0.0568 | 0.0598 | 0.0525 | 0.0000 | 3 | 3.41 |
| DSO7 | 2030 | Winter | 0_0 | 1.0 | 0.0128 | 0.0166 | 0.0099 | 0.0000 | 7 | 1.76 |
| DSO7 | 2035 | Spring | 0_0 | 1.0 | 0.0529 | 0.0538 | 0.0461 | 0.0000 | 3 | 2.89 |
| DSO7 | 2035 | Summer | 0_0 | 1.0 | 0.1642 | 0.1673 | 0.1597 | 0.0000 | 6 | 8.71 |
| DSO7 | 2035 | Autumn | 0_0 | 1.0 | 0.0127 | 0.0146 | 0.0059 | 0.0000 | 4 | 1.11 |
| DSO7 | 2035 | Winter | 0_0 | 1.0 | 0.0154 | 0.0185 | 0.0111 | 0.0000 | 7 | 2.34 |
| DSO9 | 2025 | Spring | 0_0 | 1.0 | 0.0349 | 0.0359 | 0.0218 | 0.0000 | 9 | 1.25 |
| DSO9 | 2025 | Summer | 0_0 | 1.0 | 0.0531 | 0.0557 | 0.0424 | 0.0000 | 11 | 3.05 |
| DSO9 | 2025 | Autumn | 0_0 | 1.0 | 0.0149 | 0.0165 | 0.0023 | 0.0000 | 2 | 1.57 |
| DSO9 | 2025 | Winter | 0_0 | 1.0 | 0.0086 | 0.0126 | 0.0000 | 0.0000 | 0 | 1.25 |
| DSO9 | 2030 | Spring | 0_0 | 1.0 | 0.0373 | 0.0382 | 0.0173 | 0.0000 | 9 | 1.85 |
| DSO9 | 2030 | Summer | 0_0 | 1.0 | 0.0627 | 0.0659 | 0.0468 | 0.0000 | 12 | 4.84 |
| DSO9 | 2030 | Autumn | 0_0 | 1.0 | 0.0288 | 0.0302 | 0.0153 | 0.0000 | 7 | 3.55 |
| DSO9 | 2030 | Winter | 0_0 | 1.0 | 0.0191 | 0.0233 | 0.0072 | 0.0000 | 5 | 2.56 |
| DSO9 | 2035 | Spring | 0_0 | 1.0 | 0.0576 | 0.0582 | 0.0357 | 0.0000 | 11 | 4.51 |
| DSO9 | 2035 | Summer | 0_0 | 1.0 | 0.0842 | 0.0870 | 0.0739 | 0.0000 | 11 | 6.27 |
| DSO9 | 2035 | Autumn | 0_0 | 1.0 | 0.0295 | 0.0305 | 0.0163 | 0.0000 | 7 | 2.10 |
| DSO9 | 2035 | Winter | 0_0 | 1.0 | 0.0273 | 0.0310 | 0.0168 | 0.0000 | 8 | 4.31 |

**pilot_x0**

| network | year | day | scen | omega | net | pos. part | above tol | Sg_curt (MVAh) | hours > tol | priced (EUR/day) |
|---|---|---|---|---|---|---|---|---|---|---|
| TSO | 2037 | Spring | 0_0 | 0.25 | 0.0113 | 0.0124 | 0.0078 | 0.0124 | 1 | 0.18 |
| TSO | 2037 | Spring | 0_1 | 0.25 | 0.0120 | 0.0128 | 0.0078 | 0.0128 | 1 | 0.25 |
| DSO5 | 2025 | Spring | 0_0 | 0.25 | 0.0323 | 0.0346 | 0.0244 | 0.0000 | 10 | 2.84 |
| DSO5 | 2025 | Spring | 0_1 | 0.25 | 2.4398 | 2.4422 | 2.4267 | 2.4106 | 6 | 9.59 |
| DSO5 | 2025 | Spring | 1_0 | 0.25 | 0.0326 | 0.0348 | 0.0248 | 0.0000 | 10 | 3.91 |
| DSO5 | 2025 | Spring | 1_1 | 0.25 | 0.0351 | 0.0375 | 0.0239 | 0.0000 | 6 | 2.87 |
| DSO5 | 2025 | Summer | 0_0 | 0.25 | 0.0265 | 0.0315 | 0.0163 | 0.0000 | 8 | 2.29 |
| DSO5 | 2025 | Summer | 0_1 | 0.25 | 0.0341 | 0.0387 | 0.0219 | 0.0000 | 12 | 3.53 |
| DSO5 | 2025 | Summer | 1_0 | 0.25 | 0.0265 | 0.0315 | 0.0163 | 0.0000 | 8 | 2.48 |
| DSO5 | 2025 | Summer | 1_1 | 0.25 | 0.0344 | 0.0390 | 0.0220 | 0.0000 | 12 | 3.63 |
| DSO5 | 2025 | Autumn | 0_0 | 0.25 | 0.0317 | 0.0360 | 0.0316 | 0.0000 | 9 | 3.37 |
| DSO5 | 2025 | Autumn | 0_1 | 0.25 | 0.0154 | 0.0187 | 0.0084 | 0.0000 | 4 | 1.88 |
| DSO5 | 2025 | Autumn | 1_0 | 0.25 | 0.0237 | 0.0280 | 0.0244 | 0.0000 | 9 | 2.61 |
| DSO5 | 2025 | Autumn | 1_1 | 0.25 | 0.0135 | 0.0168 | 0.0068 | 0.0000 | 3 | 1.70 |
| DSO5 | 2025 | Winter | 0_0 | 0.25 | 0.0216 | 0.0271 | 0.0233 | 0.0000 | 8 | 2.53 |
| DSO5 | 2025 | Winter | 0_1 | 0.25 | 0.1102 | 0.1155 | 0.1054 | 0.0000 | 17 | 11.05 |
| DSO5 | 2025 | Winter | 1_0 | 0.25 | 0.0198 | 0.0254 | 0.0215 | 0.0000 | 8 | 2.42 |
| DSO5 | 2025 | Winter | 1_1 | 0.25 | 0.1088 | 0.1142 | 0.1037 | 0.0000 | 17 | 11.10 |
| DSO5 | 2028 | Spring | 0_0 | 0.25 | 0.0119 | 0.0142 | 0.0056 | 0.0000 | 2 | 0.88 |
| DSO5 | 2028 | Spring | 0_1 | 0.25 | 0.0742 | 0.0765 | 0.0632 | 0.0000 | 11 | 9.09 |
| DSO5 | 2028 | Spring | 1_0 | 0.25 | 0.0116 | 0.0140 | 0.0055 | 0.0000 | 2 | 1.16 |
| DSO5 | 2028 | Spring | 1_1 | 0.25 | 0.0741 | 0.0763 | 0.0632 | 0.0000 | 11 | 11.42 |
| DSO5 | 2028 | Summer | 0_0 | 0.25 | 0.0141 | 0.0193 | 0.0073 | 0.0000 | 5 | 1.71 |
| DSO5 | 2028 | Summer | 0_1 | 0.25 | 0.0236 | 0.0291 | 0.0178 | 0.0000 | 9 | 2.77 |
| DSO5 | 2028 | Summer | 1_0 | 0.25 | 0.0138 | 0.0190 | 0.0062 | 0.0000 | 4 | 1.50 |
| DSO5 | 2028 | Summer | 1_1 | 0.25 | 0.0273 | 0.0328 | 0.0223 | 0.0000 | 9 | 2.69 |
| DSO5 | 2028 | Autumn | 0_0 | 0.25 | 0.0191 | 0.0225 | 0.0128 | 0.0000 | 6 | 2.39 |
| DSO5 | 2028 | Autumn | 0_1 | 0.25 | 0.0158 | 0.0187 | 0.0132 | 0.0000 | 6 | 2.03 |
| DSO5 | 2028 | Autumn | 1_0 | 0.25 | 0.0263 | 0.0298 | 0.0217 | 0.0000 | 5 | 2.79 |
| DSO5 | 2028 | Autumn | 1_1 | 0.25 | 0.0313 | 0.0342 | 0.0286 | 0.0000 | 7 | 3.19 |
| DSO5 | 2028 | Winter | 0_0 | 0.25 | 0.0226 | 0.0282 | 0.0238 | 0.0000 | 8 | 3.10 |
| DSO5 | 2028 | Winter | 0_1 | 0.25 | 0.0628 | 0.0686 | 0.0612 | 0.0000 | 13 | 7.49 |
| DSO5 | 2028 | Winter | 1_0 | 0.25 | 0.0228 | 0.0284 | 0.0241 | 0.0000 | 8 | 3.04 |
| DSO5 | 2028 | Winter | 1_1 | 0.25 | 0.0617 | 0.0674 | 0.0600 | 0.0000 | 13 | 7.18 |
| DSO5 | 2031 | Spring | 0_0 | 0.25 | 0.0657 | 0.0669 | 0.0549 | 0.0000 | 13 | 6.54 |
| DSO5 | 2031 | Spring | 0_1 | 0.25 | 0.0152 | 0.0167 | 0.0051 | 0.0000 | 3 | 1.80 |
| DSO5 | 2031 | Spring | 1_0 | 0.25 | 0.0644 | 0.0657 | 0.0540 | 0.0000 | 13 | 7.23 |
| DSO5 | 2031 | Spring | 1_1 | 0.25 | 0.0152 | 0.0167 | 0.0051 | 0.0000 | 3 | 1.99 |
| DSO5 | 2031 | Summer | 0_0 | 0.25 | 0.0288 | 0.0329 | 0.0218 | 0.0000 | 11 | 2.99 |
| DSO5 | 2031 | Summer | 0_1 | 0.25 | 0.0381 | 0.0424 | 0.0293 | 0.0000 | 13 | 3.81 |
| DSO5 | 2031 | Summer | 1_0 | 0.25 | 0.0256 | 0.0297 | 0.0195 | 0.0000 | 12 | 2.74 |
| DSO5 | 2031 | Summer | 1_1 | 0.25 | 0.0391 | 0.0434 | 0.0298 | 0.0000 | 13 | 3.86 |
| DSO5 | 2031 | Autumn | 0_0 | 0.25 | 0.0217 | 0.0235 | 0.0173 | 0.0000 | 6 | 2.16 |
| DSO5 | 2031 | Autumn | 0_1 | 0.25 | 0.0499 | 0.0520 | 0.0439 | 0.0000 | 9 | 4.98 |
| DSO5 | 2031 | Autumn | 1_0 | 0.25 | 0.0165 | 0.0183 | 0.0113 | 0.0000 | 5 | 2.00 |
| DSO5 | 2031 | Autumn | 1_1 | 0.25 | 0.0353 | 0.0373 | 0.0310 | 0.0000 | 8 | 4.11 |
| DSO5 | 2031 | Winter | 0_0 | 0.25 | 0.0449 | 0.0497 | 0.0424 | 0.0000 | 9 | 5.54 |
| DSO5 | 2031 | Winter | 0_1 | 0.25 | 0.0602 | 0.0649 | 0.0581 | 0.0000 | 10 | 7.19 |
| DSO5 | 2031 | Winter | 1_0 | 0.25 | 0.0448 | 0.0495 | 0.0423 | 0.0000 | 9 | 5.55 |
| DSO5 | 2031 | Winter | 1_1 | 0.25 | 0.0606 | 0.0653 | 0.0585 | 0.0000 | 10 | 7.25 |
| DSO5 | 2034 | Spring | 0_0 | 0.25 | 0.0318 | 0.0333 | 0.0231 | 0.0119 | 7 | 2.37 |
| DSO5 | 2034 | Spring | 0_1 | 0.25 | 0.0606 | 0.0615 | 0.0509 | 0.0000 | 13 | 6.18 |
| DSO5 | 2034 | Spring | 1_0 | 0.25 | 0.0196 | 0.0211 | 0.0101 | 0.0000 | 6 | 2.88 |
| DSO5 | 2034 | Spring | 1_1 | 0.25 | 0.0568 | 0.0578 | 0.0477 | 0.0000 | 13 | 7.62 |
| DSO5 | 2034 | Summer | 0_0 | 0.25 | 0.0403 | 0.0437 | 0.0331 | 0.0000 | 13 | 3.77 |
| DSO5 | 2034 | Summer | 0_1 | 0.25 | 0.0392 | 0.0429 | 0.0268 | 0.0000 | 9 | 3.34 |
| DSO5 | 2034 | Summer | 1_0 | 0.25 | 0.0464 | 0.0498 | 0.0405 | 0.0000 | 13 | 5.15 |
| DSO5 | 2034 | Summer | 1_1 | 0.25 | 0.0391 | 0.0429 | 0.0282 | 0.0000 | 10 | 4.44 |
| DSO5 | 2034 | Autumn | 0_0 | 0.25 | 0.0507 | 0.0525 | 0.0472 | 0.0000 | 10 | 6.93 |
| DSO5 | 2034 | Autumn | 0_1 | 0.25 | 0.0271 | 0.0299 | 0.0225 | 0.0000 | 6 | 3.75 |
| DSO5 | 2034 | Autumn | 1_0 | 0.25 | 0.0525 | 0.0544 | 0.0491 | 0.0000 | 10 | 5.62 |
| DSO5 | 2034 | Autumn | 1_1 | 0.25 | 0.0283 | 0.0312 | 0.0239 | 0.0000 | 6 | 3.00 |
| DSO5 | 2034 | Winter | 0_0 | 0.25 | 0.0438 | 0.0475 | 0.0405 | 0.0000 | 9 | 5.56 |
| DSO5 | 2034 | Winter | 0_1 | 0.25 | 0.0690 | 0.0739 | 0.0655 | 0.0000 | 10 | 8.63 |
| DSO5 | 2034 | Winter | 1_0 | 0.25 | 0.0443 | 0.0480 | 0.0410 | 0.0000 | 9 | 5.90 |
| DSO5 | 2034 | Winter | 1_1 | 0.25 | 0.0625 | 0.0675 | 0.0593 | 0.0000 | 10 | 8.38 |
| DSO5 | 2037 | Spring | 0_0 | 0.25 | 0.0318 | 0.0326 | 0.0208 | 0.0000 | 8 | 2.28 |
| DSO5 | 2037 | Spring | 0_1 | 0.25 | 3.8260 | 3.8266 | 3.8149 | 3.6965 | 15 | 57.80 |
| DSO5 | 2037 | Spring | 1_0 | 0.25 | 0.0258 | 0.0267 | 0.0159 | 0.0000 | 9 | 2.82 |
| DSO5 | 2037 | Spring | 1_1 | 0.25 | 0.1119 | 0.1126 | 0.1002 | 0.0000 | 14 | 17.38 |
| DSO5 | 2037 | Summer | 0_0 | 0.25 | 0.0709 | 0.0742 | 0.0638 | 0.0000 | 9 | 5.68 |
| DSO5 | 2037 | Summer | 0_1 | 0.25 | 0.1103 | 0.1133 | 0.1025 | 0.0000 | 17 | 10.59 |
| DSO5 | 2037 | Summer | 1_0 | 0.25 | 0.0708 | 0.0740 | 0.0637 | 0.0000 | 9 | 6.55 |
| DSO5 | 2037 | Summer | 1_1 | 0.25 | 0.1089 | 0.1119 | 0.1015 | 0.0000 | 17 | 12.16 |
| DSO5 | 2037 | Autumn | 0_0 | 0.25 | 0.0344 | 0.0370 | 0.0284 | 0.0000 | 7 | 4.36 |
| DSO5 | 2037 | Autumn | 0_1 | 0.25 | 0.0342 | 0.0363 | 0.0301 | 0.0000 | 11 | 4.26 |
| DSO5 | 2037 | Autumn | 1_0 | 0.25 | 0.0343 | 0.0369 | 0.0284 | 0.0000 | 7 | 4.48 |
| DSO5 | 2037 | Autumn | 1_1 | 0.25 | 0.0292 | 0.0313 | 0.0243 | 0.0000 | 10 | 3.86 |
| DSO5 | 2037 | Winter | 0_0 | 0.25 | 0.0514 | 0.0558 | 0.0466 | 0.0000 | 9 | 7.10 |
| DSO5 | 2037 | Winter | 0_1 | 0.25 | 0.0547 | 0.0591 | 0.0498 | 0.0000 | 14 | 7.51 |
| DSO5 | 2037 | Winter | 1_0 | 0.25 | 0.0519 | 0.0563 | 0.0463 | 0.0000 | 9 | 7.33 |
| DSO5 | 2037 | Winter | 1_1 | 0.25 | 0.0569 | 0.0614 | 0.0529 | 0.0000 | 15 | 7.98 |
| DSO7 | 2025 | Spring | 0_0 | 0.25 | 0.5237 | 0.5257 | 0.5147 | 0.4379 | 6 | 4.38 |
| DSO7 | 2025 | Spring | 0_1 | 0.25 | 0.1274 | 0.1293 | 0.1250 | 0.0000 | 4 | 3.39 |
| DSO7 | 2025 | Spring | 1_0 | 0.25 | 0.0842 | 0.0862 | 0.0753 | 0.0000 | 5 | 4.86 |
| DSO7 | 2025 | Spring | 1_1 | 0.25 | 0.0960 | 0.0979 | 0.0943 | 0.0000 | 3 | 4.31 |
| DSO7 | 2025 | Summer | 0_0 | 0.25 | 0.0703 | 0.0739 | 0.0621 | 0.0000 | 8 | 3.94 |
| DSO7 | 2025 | Summer | 0_1 | 0.25 | 0.0558 | 0.0593 | 0.0523 | 0.0000 | 5 | 3.07 |
| DSO7 | 2025 | Summer | 1_0 | 0.25 | 0.0703 | 0.0739 | 0.0621 | 0.0000 | 8 | 4.37 |
| DSO7 | 2025 | Summer | 1_1 | 0.25 | 0.0432 | 0.0467 | 0.0378 | 0.0000 | 6 | 2.66 |
| DSO7 | 2025 | Autumn | 0_1 | 0.25 | 0.0117 | 0.0143 | 0.0064 | 0.0000 | 3 | 1.38 |
| DSO7 | 2025 | Autumn | 1_1 | 0.25 | 0.0093 | 0.0120 | 0.0039 | 0.0000 | 3 | 1.15 |
| DSO7 | 2025 | Winter | 0_1 | 0.25 | 0.0160 | 0.0204 | 0.0137 | 0.0000 | 6 | 1.86 |
| DSO7 | 2025 | Winter | 1_1 | 0.25 | 0.0160 | 0.0204 | 0.0133 | 0.0000 | 5 | 1.89 |
| DSO7 | 2028 | Spring | 0_0 | 0.25 | 0.1318 | 0.1338 | 0.1285 | 0.0000 | 5 | 3.74 |
| DSO7 | 2028 | Spring | 0_1 | 0.25 | 0.1508 | 0.1528 | 0.1490 | 0.0000 | 5 | 4.18 |
| DSO7 | 2028 | Spring | 1_0 | 0.25 | 0.1313 | 0.1333 | 0.1281 | 0.0000 | 5 | 6.23 |
| DSO7 | 2028 | Spring | 1_1 | 0.25 | 0.1419 | 0.1439 | 0.1408 | 0.0000 | 5 | 6.62 |
| DSO7 | 2028 | Summer | 0_0 | 0.25 | 0.0499 | 0.0530 | 0.0431 | 0.0000 | 6 | 3.61 |
| DSO7 | 2028 | Summer | 0_1 | 0.25 | 0.0441 | 0.0475 | 0.0312 | 0.0000 | 9 | 4.27 |
| DSO7 | 2028 | Summer | 1_0 | 0.25 | 0.0471 | 0.0502 | 0.0397 | 0.0000 | 6 | 2.85 |
| DSO7 | 2028 | Summer | 1_1 | 0.25 | 0.0550 | 0.0584 | 0.0431 | 0.0000 | 10 | 4.51 |
| DSO7 | 2028 | Autumn | 0_0 | 0.25 | 0.0081 | 0.0114 | 0.0023 | 0.0000 | 2 | 1.23 |
| DSO7 | 2028 | Autumn | 1_0 | 0.25 | 0.0101 | 0.0134 | 0.0051 | 0.0000 | 4 | 1.34 |
| DSO7 | 2028 | Winter | 0_0 | 0.25 | 0.0154 | 0.0198 | 0.0148 | 0.0000 | 7 | 2.11 |
| DSO7 | 2028 | Winter | 0_1 | 0.25 | 0.0089 | 0.0133 | 0.0035 | 0.0000 | 3 | 1.48 |
| DSO7 | 2028 | Winter | 1_0 | 0.25 | 0.0151 | 0.0194 | 0.0144 | 0.0000 | 7 | 2.02 |
| DSO7 | 2028 | Winter | 1_1 | 0.25 | 0.0088 | 0.0132 | 0.0024 | 0.0000 | 2 | 1.46 |
| DSO7 | 2031 | Spring | 0_0 | 0.25 | 0.1707 | 0.1721 | 0.1641 | 0.0000 | 5 | 5.14 |
| DSO7 | 2031 | Spring | 0_1 | 0.25 | 0.0768 | 0.0782 | 0.0683 | 0.0000 | 3 | 2.53 |
| DSO7 | 2031 | Spring | 1_0 | 0.25 | 0.1463 | 0.1478 | 0.1389 | 0.0000 | 4 | 5.64 |
| DSO7 | 2031 | Spring | 1_1 | 0.25 | 0.0767 | 0.0781 | 0.0683 | 0.0000 | 3 | 3.18 |
| DSO7 | 2031 | Summer | 0_0 | 0.25 | 0.0374 | 0.0415 | 0.0282 | 0.0000 | 9 | 3.64 |
| DSO7 | 2031 | Summer | 0_1 | 0.25 | 0.0183 | 0.0217 | 0.0152 | 0.0000 | 3 | 1.55 |
| DSO7 | 2031 | Summer | 1_0 | 0.25 | 0.0329 | 0.0370 | 0.0250 | 0.0000 | 8 | 3.30 |
| DSO7 | 2031 | Summer | 1_1 | 0.25 | 0.0196 | 0.0230 | 0.0151 | 0.0000 | 3 | 1.59 |
| DSO7 | 2031 | Autumn | 0_0 | 0.25 | 0.0236 | 0.0255 | 0.0162 | 0.0000 | 5 | 2.41 |
| DSO7 | 2031 | Autumn | 0_1 | 0.25 | 0.0218 | 0.0238 | 0.0125 | 0.0000 | 5 | 2.05 |
| DSO7 | 2031 | Autumn | 1_0 | 0.25 | 0.0228 | 0.0248 | 0.0163 | 0.0000 | 5 | 2.77 |
| DSO7 | 2031 | Winter | 0_0 | 0.25 | 0.0132 | 0.0170 | 0.0085 | 0.0000 | 6 | 1.91 |
| DSO7 | 2031 | Winter | 0_1 | 0.25 | 0.0104 | 0.0143 | 0.0054 | 0.0000 | 4 | 1.62 |
| DSO7 | 2031 | Winter | 1_0 | 0.25 | 0.0133 | 0.0171 | 0.0086 | 0.0000 | 6 | 1.92 |
| DSO7 | 2031 | Winter | 1_1 | 0.25 | 0.0104 | 0.0143 | 0.0055 | 0.0000 | 4 | 1.63 |
| DSO7 | 2034 | Spring | 0_0 | 0.25 | 0.1680 | 0.1694 | 0.1577 | 0.0000 | 6 | 4.58 |
| DSO7 | 2034 | Spring | 0_1 | 0.25 | 0.0911 | 0.0924 | 0.0783 | 0.0000 | 12 | 5.61 |
| DSO7 | 2034 | Spring | 1_0 | 0.25 | 0.1670 | 0.1684 | 0.1567 | 0.0000 | 6 | 7.61 |
| DSO7 | 2034 | Spring | 1_1 | 0.25 | 0.0735 | 0.0748 | 0.0647 | 0.0000 | 12 | 6.13 |
| DSO7 | 2034 | Summer | 0_0 | 0.25 | 0.1295 | 0.1324 | 0.1195 | 0.0000 | 7 | 7.38 |
| DSO7 | 2034 | Summer | 0_1 | 0.25 | 0.0587 | 0.0622 | 0.0563 | 0.0000 | 4 | 3.24 |
| DSO7 | 2034 | Summer | 1_0 | 0.25 | 0.1043 | 0.1072 | 0.0948 | 0.0000 | 5 | 8.86 |
| DSO7 | 2034 | Summer | 1_1 | 0.25 | 0.0565 | 0.0600 | 0.0545 | 0.0000 | 4 | 4.74 |
| DSO7 | 2034 | Autumn | 0_0 | 0.25 | 0.0177 | 0.0197 | 0.0139 | 0.0000 | 6 | 2.42 |
| DSO7 | 2034 | Autumn | 0_1 | 0.25 | 0.0073 | 0.0101 | 0.0049 | 0.0000 | 4 | 1.35 |
| DSO7 | 2034 | Autumn | 1_0 | 0.25 | 0.0254 | 0.0275 | 0.0199 | 0.0000 | 6 | 2.58 |
| DSO7 | 2034 | Autumn | 1_1 | 0.25 | 0.0073 | 0.0101 | 0.0035 | 0.0000 | 3 | 1.10 |
| DSO7 | 2034 | Winter | 0_0 | 0.25 | 0.0139 | 0.0177 | 0.0092 | 0.0000 | 6 | 2.14 |
| DSO7 | 2034 | Winter | 0_1 | 0.25 | 0.0222 | 0.0259 | 0.0175 | 0.0000 | 8 | 3.04 |
| DSO7 | 2034 | Winter | 1_0 | 0.25 | 0.0141 | 0.0179 | 0.0101 | 0.0000 | 7 | 2.25 |
| DSO7 | 2034 | Winter | 1_1 | 0.25 | 0.0199 | 0.0236 | 0.0152 | 0.0000 | 8 | 2.97 |
| DSO7 | 2037 | Spring | 0_0 | 0.25 | 0.1427 | 0.1435 | 0.1349 | 0.0000 | 7 | 3.04 |
| DSO7 | 2037 | Spring | 0_1 | 0.25 | 3.4280 | 3.4287 | 3.4123 | 3.3099 | 12 | 39.40 |
| DSO7 | 2037 | Spring | 1_0 | 0.25 | 0.1384 | 0.1392 | 0.1289 | 0.0000 | 7 | 6.63 |
| DSO7 | 2037 | Spring | 1_1 | 0.25 | 0.0835 | 0.0842 | 0.0701 | 0.0000 | 9 | 5.83 |
| DSO7 | 2037 | Summer | 0_0 | 0.25 | 0.0811 | 0.0832 | 0.0739 | 0.0000 | 11 | 6.68 |
| DSO7 | 2037 | Summer | 0_1 | 0.25 | 0.0767 | 0.0789 | 0.0722 | 0.0000 | 8 | 5.07 |
| DSO7 | 2037 | Summer | 1_0 | 0.25 | 0.0742 | 0.0764 | 0.0662 | 0.0000 | 11 | 7.25 |
| DSO7 | 2037 | Summer | 1_1 | 0.25 | 0.0764 | 0.0786 | 0.0718 | 0.0000 | 8 | 5.89 |
| DSO7 | 2037 | Autumn | 0_0 | 0.25 | 0.0107 | 0.0120 | 0.0031 | 0.0000 | 3 | 1.41 |
| DSO7 | 2037 | Autumn | 0_1 | 0.25 | 0.0181 | 0.0199 | 0.0117 | 0.0000 | 6 | 2.46 |
| DSO7 | 2037 | Autumn | 1_0 | 0.25 | 0.0092 | 0.0105 | 0.0031 | 0.0000 | 3 | 1.30 |
| DSO7 | 2037 | Autumn | 1_1 | 0.25 | 0.0180 | 0.0197 | 0.0117 | 0.0000 | 6 | 2.50 |
| DSO7 | 2037 | Winter | 0_0 | 0.25 | 0.0207 | 0.0236 | 0.0149 | 0.0000 | 8 | 3.04 |
| DSO7 | 2037 | Winter | 0_1 | 0.25 | 0.0085 | 0.0115 | 0.0039 | 0.0000 | 3 | 1.52 |
| DSO7 | 2037 | Winter | 1_0 | 0.25 | 0.0207 | 0.0236 | 0.0149 | 0.0000 | 8 | 3.12 |
| DSO7 | 2037 | Winter | 1_1 | 0.25 | 0.0085 | 0.0115 | 0.0039 | 0.0000 | 3 | 1.54 |
| DSO9 | 2025 | Spring | 0_0 | 0.25 | 2.2173 | 2.2180 | 2.1998 | 2.1562 | 13 | 10.03 |
| DSO9 | 2025 | Spring | 0_1 | 0.25 | 0.0287 | 0.0296 | 0.0160 | 0.0000 | 8 | 1.13 |
| DSO9 | 2025 | Spring | 1_0 | 0.25 | 0.0626 | 0.0633 | 0.0453 | 0.0000 | 11 | 4.20 |
| DSO9 | 2025 | Spring | 1_1 | 0.25 | 0.0282 | 0.0291 | 0.0149 | 0.0000 | 7 | 1.63 |
| DSO9 | 2025 | Summer | 0_0 | 0.25 | 0.0650 | 0.0675 | 0.0511 | 0.0000 | 12 | 4.16 |
| DSO9 | 2025 | Summer | 0_1 | 0.25 | 0.0521 | 0.0555 | 0.0412 | 0.0000 | 11 | 3.32 |
| DSO9 | 2025 | Summer | 1_0 | 0.25 | 0.0626 | 0.0652 | 0.0481 | 0.0000 | 12 | 4.47 |
| DSO9 | 2025 | Summer | 1_1 | 0.25 | 0.0504 | 0.0538 | 0.0391 | 0.0000 | 11 | 3.59 |
| DSO9 | 2025 | Autumn | 0_0 | 0.25 | 0.0148 | 0.0166 | 0.0087 | 0.0000 | 4 | 1.70 |
| DSO9 | 2025 | Autumn | 0_1 | 0.25 | 0.0211 | 0.0232 | 0.0056 | 0.0000 | 4 | 2.31 |
| DSO9 | 2025 | Autumn | 1_0 | 0.25 | 0.0151 | 0.0169 | 0.0086 | 0.0000 | 4 | 1.70 |
| DSO9 | 2025 | Autumn | 1_1 | 0.25 | 0.0231 | 0.0251 | 0.0072 | 0.0000 | 5 | 2.48 |
| DSO9 | 2025 | Winter | 0_1 | 0.25 | 0.0264 | 0.0303 | 0.0126 | 0.0000 | 6 | 3.11 |
| DSO9 | 2025 | Winter | 1_1 | 0.25 | 0.0264 | 0.0303 | 0.0127 | 0.0000 | 6 | 3.22 |
| DSO9 | 2028 | Spring | 0_0 | 0.25 | 0.0917 | 0.0924 | 0.0803 | 0.0001 | 15 | 5.13 |
| DSO9 | 2028 | Spring | 0_1 | 0.25 | 0.0302 | 0.0310 | 0.0180 | 0.0000 | 9 | 1.68 |
| DSO9 | 2028 | Spring | 1_0 | 0.25 | 0.0909 | 0.0916 | 0.0797 | 0.0000 | 15 | 6.89 |
| DSO9 | 2028 | Spring | 1_1 | 0.25 | 0.0301 | 0.0310 | 0.0180 | 0.0000 | 9 | 2.24 |
| DSO9 | 2028 | Summer | 0_0 | 0.25 | 0.0434 | 0.0467 | 0.0349 | 0.0000 | 11 | 3.62 |
| DSO9 | 2028 | Summer | 0_1 | 0.25 | 0.1004 | 0.1030 | 0.0898 | 0.0000 | 12 | 8.16 |
| DSO9 | 2028 | Summer | 1_0 | 0.25 | 0.0464 | 0.0497 | 0.0371 | 0.0000 | 10 | 3.28 |
| DSO9 | 2028 | Summer | 1_1 | 0.25 | 0.0976 | 0.1002 | 0.0866 | 0.0000 | 12 | 6.92 |
| DSO9 | 2028 | Autumn | 0_1 | 0.25 | 0.0301 | 0.0324 | 0.0167 | 0.0000 | 10 | 4.36 |
| DSO9 | 2028 | Autumn | 1_1 | 0.25 | 0.0298 | 0.0321 | 0.0156 | 0.0000 | 9 | 3.35 |
| DSO9 | 2028 | Winter | 0_0 | 0.25 | 0.0119 | 0.0158 | 0.0013 | 0.0000 | 1 | 1.95 |
| DSO9 | 2028 | Winter | 0_1 | 0.25 | 0.0155 | 0.0198 | 0.0072 | 0.0000 | 4 | 2.44 |
| DSO9 | 2028 | Winter | 1_0 | 0.25 | 0.0082 | 0.0121 | 0.0000 | 0.0000 | 0 | 1.44 |
| DSO9 | 2028 | Winter | 1_1 | 0.25 | 0.0162 | 0.0205 | 0.0083 | 0.0000 | 4 | 2.37 |
| DSO9 | 2031 | Spring | 0_0 | 0.25 | 0.0529 | 0.0538 | 0.0367 | 0.0000 | 12 | 4.12 |
| DSO9 | 2031 | Spring | 0_1 | 0.25 | 0.0473 | 0.0481 | 0.0275 | 0.0000 | 14 | 3.32 |
| DSO9 | 2031 | Spring | 1_0 | 0.25 | 0.0541 | 0.0550 | 0.0403 | 0.0000 | 14 | 4.70 |
| DSO9 | 2031 | Spring | 1_1 | 0.25 | 0.0473 | 0.0481 | 0.0275 | 0.0000 | 14 | 3.74 |
| DSO9 | 2031 | Summer | 0_0 | 0.25 | 0.0763 | 0.0799 | 0.0628 | 0.0000 | 11 | 6.57 |
| DSO9 | 2031 | Summer | 0_1 | 0.25 | 0.0594 | 0.0627 | 0.0453 | 0.0000 | 12 | 5.35 |
| DSO9 | 2031 | Summer | 1_0 | 0.25 | 0.0726 | 0.0761 | 0.0586 | 0.0000 | 11 | 6.21 |
| DSO9 | 2031 | Summer | 1_1 | 0.25 | 0.0590 | 0.0623 | 0.0459 | 0.0000 | 12 | 5.22 |
| DSO9 | 2031 | Autumn | 0_0 | 0.25 | 0.0296 | 0.0314 | 0.0197 | 0.0000 | 9 | 2.70 |
| DSO9 | 2031 | Autumn | 0_1 | 0.25 | 0.0262 | 0.0273 | 0.0120 | 0.0000 | 7 | 2.63 |
| DSO9 | 2031 | Autumn | 1_0 | 0.25 | 0.0341 | 0.0359 | 0.0235 | 0.0000 | 9 | 4.33 |
| DSO9 | 2031 | Autumn | 1_1 | 0.25 | 0.0302 | 0.0314 | 0.0162 | 0.0000 | 7 | 4.06 |
| DSO9 | 2031 | Winter | 0_0 | 0.25 | 0.0226 | 0.0268 | 0.0110 | 0.0000 | 7 | 3.29 |
| DSO9 | 2031 | Winter | 0_1 | 0.25 | 0.0188 | 0.0227 | 0.0079 | 0.0000 | 5 | 2.84 |
| DSO9 | 2031 | Winter | 1_0 | 0.25 | 0.0221 | 0.0263 | 0.0106 | 0.0000 | 7 | 3.26 |
| DSO9 | 2031 | Winter | 1_1 | 0.25 | 0.0186 | 0.0226 | 0.0079 | 0.0000 | 5 | 2.85 |
| DSO9 | 2034 | Spring | 0_0 | 0.25 | 0.0431 | 0.0440 | 0.0230 | 0.0000 | 9 | 2.94 |
| DSO9 | 2034 | Spring | 0_1 | 0.25 | 0.4784 | 0.4794 | 0.4664 | 0.4284 | 10 | 6.08 |
| DSO9 | 2034 | Spring | 1_0 | 0.25 | 0.0417 | 0.0426 | 0.0230 | 0.0000 | 9 | 3.78 |
| DSO9 | 2034 | Spring | 1_1 | 0.25 | 0.0515 | 0.0524 | 0.0386 | 0.0000 | 10 | 4.29 |
| DSO9 | 2034 | Summer | 0_0 | 0.25 | 0.0758 | 0.0791 | 0.0583 | 0.0000 | 15 | 6.27 |
| DSO9 | 2034 | Summer | 0_1 | 0.25 | 0.0748 | 0.0785 | 0.0651 | 0.0000 | 12 | 5.60 |
| DSO9 | 2034 | Summer | 1_0 | 0.25 | 0.0735 | 0.0768 | 0.0556 | 0.0000 | 15 | 7.94 |
| DSO9 | 2034 | Summer | 1_1 | 0.25 | 0.0739 | 0.0776 | 0.0632 | 0.0000 | 12 | 7.71 |
| DSO9 | 2034 | Autumn | 0_0 | 0.25 | 0.0237 | 0.0253 | 0.0134 | 0.0000 | 8 | 4.07 |
| DSO9 | 2034 | Autumn | 0_1 | 0.25 | 0.0471 | 0.0486 | 0.0380 | 0.0000 | 12 | 7.72 |
| DSO9 | 2034 | Autumn | 1_0 | 0.25 | 0.0228 | 0.0245 | 0.0115 | 0.0000 | 8 | 2.70 |
| DSO9 | 2034 | Autumn | 1_1 | 0.25 | 0.0466 | 0.0481 | 0.0367 | 0.0000 | 12 | 5.04 |
| DSO9 | 2034 | Winter | 0_0 | 0.25 | 0.0399 | 0.0438 | 0.0284 | 0.0000 | 11 | 5.44 |
| DSO9 | 2034 | Winter | 1_0 | 0.25 | 0.0398 | 0.0438 | 0.0293 | 0.0000 | 11 | 6.20 |
| DSO9 | 2037 | Spring | 0_0 | 0.25 | 4.9001 | 4.9007 | 4.8886 | 4.7981 | 13 | 19.46 |
| DSO9 | 2037 | Spring | 0_1 | 0.25 | 0.0410 | 0.0416 | 0.0242 | 0.0000 | 11 | 2.33 |
| DSO9 | 2037 | Spring | 1_0 | 0.25 | 0.1040 | 0.1046 | 0.0901 | 0.0000 | 12 | 8.45 |
| DSO9 | 2037 | Spring | 1_1 | 0.25 | 0.0414 | 0.0420 | 0.0240 | 0.0000 | 10 | 3.75 |
| DSO9 | 2037 | Summer | 0_0 | 0.25 | 0.0640 | 0.0661 | 0.0564 | 0.0000 | 10 | 4.96 |
| DSO9 | 2037 | Summer | 0_1 | 0.25 | 0.1020 | 0.1049 | 0.0910 | 0.0000 | 14 | 9.10 |
| DSO9 | 2037 | Summer | 1_0 | 0.25 | 0.0685 | 0.0706 | 0.0607 | 0.0000 | 11 | 5.91 |
| DSO9 | 2037 | Summer | 1_1 | 0.25 | 0.1020 | 0.1049 | 0.0900 | 0.0000 | 13 | 10.29 |
| DSO9 | 2037 | Autumn | 0_0 | 0.25 | 0.0565 | 0.0580 | 0.0433 | 0.0000 | 11 | 6.95 |
| DSO9 | 2037 | Autumn | 0_1 | 0.25 | 0.0254 | 0.0269 | 0.0107 | 0.0000 | 5 | 3.22 |
| DSO9 | 2037 | Autumn | 1_0 | 0.25 | 0.0567 | 0.0582 | 0.0436 | 0.0000 | 11 | 7.61 |
| DSO9 | 2037 | Autumn | 1_1 | 0.25 | 0.0254 | 0.0269 | 0.0108 | 0.0000 | 5 | 3.54 |
| DSO9 | 2037 | Winter | 0_0 | 0.25 | 0.0238 | 0.0277 | 0.0087 | 0.0000 | 5 | 3.81 |
| DSO9 | 2037 | Winter | 0_1 | 0.25 | 0.0322 | 0.0363 | 0.0148 | 0.0000 | 8 | 5.07 |
| DSO9 | 2037 | Winter | 1_0 | 0.25 | 0.0245 | 0.0284 | 0.0099 | 0.0000 | 6 | 3.91 |
| DSO9 | 2037 | Winter | 1_1 | 0.25 | 0.0295 | 0.0336 | 0.0132 | 0.0000 | 9 | 4.71 |

**pilot_unit**

| network | year | day | scen | omega | net | pos. part | above tol | Sg_curt (MVAh) | hours > tol | priced (EUR/day) |
|---|---|---|---|---|---|---|---|---|---|---|
| DSO5 | 2025 | Spring | 0_0 | 0.25 | 0.0324 | 0.0346 | 0.0244 | 0.0000 | 10 | 2.85 |
| DSO5 | 2025 | Spring | 0_1 | 0.25 | 2.4394 | 2.4418 | 2.4263 | 2.4103 | 6 | 9.59 |
| DSO5 | 2025 | Spring | 1_0 | 0.25 | 0.0326 | 0.0349 | 0.0249 | 0.0000 | 10 | 3.92 |
| DSO5 | 2025 | Spring | 1_1 | 0.25 | 0.0350 | 0.0375 | 0.0239 | 0.0000 | 6 | 2.87 |
| DSO5 | 2025 | Summer | 0_0 | 0.25 | 0.0265 | 0.0315 | 0.0163 | 0.0000 | 8 | 2.29 |
| DSO5 | 2025 | Summer | 0_1 | 0.25 | 0.0341 | 0.0387 | 0.0219 | 0.0000 | 12 | 3.52 |
| DSO5 | 2025 | Summer | 1_0 | 0.25 | 0.0265 | 0.0315 | 0.0163 | 0.0000 | 8 | 2.48 |
| DSO5 | 2025 | Summer | 1_1 | 0.25 | 0.0344 | 0.0390 | 0.0220 | 0.0000 | 12 | 3.62 |
| DSO5 | 2025 | Autumn | 0_0 | 0.25 | 0.0317 | 0.0360 | 0.0316 | 0.0000 | 9 | 3.37 |
| DSO5 | 2025 | Autumn | 0_1 | 0.25 | 0.0154 | 0.0188 | 0.0084 | 0.0000 | 4 | 1.88 |
| DSO5 | 2025 | Autumn | 1_0 | 0.25 | 0.0237 | 0.0280 | 0.0244 | 0.0000 | 9 | 2.61 |
| DSO5 | 2025 | Autumn | 1_1 | 0.25 | 0.0135 | 0.0168 | 0.0068 | 0.0000 | 3 | 1.70 |
| DSO5 | 2025 | Winter | 0_0 | 0.25 | 0.0216 | 0.0271 | 0.0233 | 0.0000 | 8 | 2.53 |
| DSO5 | 2025 | Winter | 0_1 | 0.25 | 0.1102 | 0.1156 | 0.1054 | 0.0000 | 17 | 11.06 |
| DSO5 | 2025 | Winter | 1_0 | 0.25 | 0.0198 | 0.0253 | 0.0215 | 0.0000 | 8 | 2.42 |
| DSO5 | 2025 | Winter | 1_1 | 0.25 | 0.1088 | 0.1142 | 0.1037 | 0.0000 | 17 | 11.10 |
| DSO5 | 2028 | Spring | 0_0 | 0.25 | 0.0119 | 0.0143 | 0.0067 | 0.0000 | 3 | 0.88 |
| DSO5 | 2028 | Spring | 0_1 | 0.25 | 0.0742 | 0.0765 | 0.0632 | 0.0000 | 11 | 9.08 |
| DSO5 | 2028 | Spring | 1_0 | 0.25 | 0.0116 | 0.0140 | 0.0056 | 0.0000 | 2 | 1.16 |
| DSO5 | 2028 | Spring | 1_1 | 0.25 | 0.0740 | 0.0763 | 0.0632 | 0.0000 | 11 | 11.41 |
| DSO5 | 2028 | Summer | 0_0 | 0.25 | 0.0141 | 0.0193 | 0.0073 | 0.0000 | 5 | 1.71 |
| DSO5 | 2028 | Summer | 0_1 | 0.25 | 0.0236 | 0.0291 | 0.0178 | 0.0000 | 9 | 2.77 |
| DSO5 | 2028 | Summer | 1_0 | 0.25 | 0.0138 | 0.0190 | 0.0062 | 0.0000 | 4 | 1.50 |
| DSO5 | 2028 | Summer | 1_1 | 0.25 | 0.0273 | 0.0328 | 0.0223 | 0.0000 | 9 | 2.69 |
| DSO5 | 2028 | Autumn | 0_0 | 0.25 | 0.0191 | 0.0225 | 0.0128 | 0.0000 | 6 | 2.39 |
| DSO5 | 2028 | Autumn | 0_1 | 0.25 | 0.0158 | 0.0187 | 0.0132 | 0.0000 | 6 | 2.03 |
| DSO5 | 2028 | Autumn | 1_0 | 0.25 | 0.0263 | 0.0298 | 0.0217 | 0.0000 | 5 | 2.79 |
| DSO5 | 2028 | Autumn | 1_1 | 0.25 | 0.0313 | 0.0342 | 0.0286 | 0.0000 | 7 | 3.19 |
| DSO5 | 2028 | Winter | 0_0 | 0.25 | 0.0226 | 0.0282 | 0.0238 | 0.0000 | 8 | 3.10 |
| DSO5 | 2028 | Winter | 0_1 | 0.25 | 0.0628 | 0.0685 | 0.0611 | 0.0000 | 13 | 7.49 |
| DSO5 | 2028 | Winter | 1_0 | 0.25 | 0.0228 | 0.0284 | 0.0241 | 0.0000 | 8 | 3.04 |
| DSO5 | 2028 | Winter | 1_1 | 0.25 | 0.0617 | 0.0674 | 0.0600 | 0.0000 | 13 | 7.18 |
| DSO5 | 2031 | Spring | 0_0 | 0.25 | 0.0657 | 0.0670 | 0.0550 | 0.0000 | 13 | 6.56 |
| DSO5 | 2031 | Spring | 0_1 | 0.25 | 0.0152 | 0.0167 | 0.0051 | 0.0000 | 3 | 1.80 |
| DSO5 | 2031 | Spring | 1_0 | 0.25 | 0.0645 | 0.0657 | 0.0541 | 0.0000 | 13 | 7.24 |
| DSO5 | 2031 | Spring | 1_1 | 0.25 | 0.0152 | 0.0167 | 0.0050 | 0.0000 | 3 | 1.99 |
| DSO5 | 2031 | Summer | 0_0 | 0.25 | 0.0288 | 0.0329 | 0.0218 | 0.0000 | 11 | 2.99 |
| DSO5 | 2031 | Summer | 0_1 | 0.25 | 0.0381 | 0.0424 | 0.0293 | 0.0000 | 13 | 3.81 |
| DSO5 | 2031 | Summer | 1_0 | 0.25 | 0.0256 | 0.0297 | 0.0195 | 0.0000 | 12 | 2.74 |
| DSO5 | 2031 | Summer | 1_1 | 0.25 | 0.0391 | 0.0434 | 0.0298 | 0.0000 | 13 | 3.85 |
| DSO5 | 2031 | Autumn | 0_0 | 0.25 | 0.0217 | 0.0235 | 0.0173 | 0.0000 | 6 | 2.16 |
| DSO5 | 2031 | Autumn | 0_1 | 0.25 | 0.0500 | 0.0520 | 0.0440 | 0.0000 | 9 | 4.98 |
| DSO5 | 2031 | Autumn | 1_0 | 0.25 | 0.0165 | 0.0183 | 0.0113 | 0.0000 | 5 | 2.00 |
| DSO5 | 2031 | Autumn | 1_1 | 0.25 | 0.0353 | 0.0374 | 0.0310 | 0.0000 | 8 | 4.12 |
| DSO5 | 2031 | Winter | 0_0 | 0.25 | 0.0449 | 0.0497 | 0.0424 | 0.0000 | 9 | 5.55 |
| DSO5 | 2031 | Winter | 0_1 | 0.25 | 0.0602 | 0.0649 | 0.0581 | 0.0000 | 10 | 7.19 |
| DSO5 | 2031 | Winter | 1_0 | 0.25 | 0.0448 | 0.0495 | 0.0423 | 0.0000 | 9 | 5.55 |
| DSO5 | 2031 | Winter | 1_1 | 0.25 | 0.0606 | 0.0653 | 0.0585 | 0.0000 | 10 | 7.25 |
| DSO5 | 2034 | Spring | 0_0 | 0.25 | 0.0319 | 0.0333 | 0.0231 | 0.0119 | 7 | 2.37 |
| DSO5 | 2034 | Spring | 0_1 | 0.25 | 0.0608 | 0.0617 | 0.0511 | 0.0000 | 13 | 6.21 |
| DSO5 | 2034 | Spring | 1_0 | 0.25 | 0.0197 | 0.0212 | 0.0101 | 0.0000 | 6 | 2.89 |
| DSO5 | 2034 | Spring | 1_1 | 0.25 | 0.0570 | 0.0580 | 0.0479 | 0.0000 | 13 | 7.66 |
| DSO5 | 2034 | Summer | 0_0 | 0.25 | 0.0403 | 0.0437 | 0.0331 | 0.0000 | 13 | 3.77 |
| DSO5 | 2034 | Summer | 0_1 | 0.25 | 0.0392 | 0.0430 | 0.0268 | 0.0000 | 9 | 3.34 |
| DSO5 | 2034 | Summer | 1_0 | 0.25 | 0.0464 | 0.0498 | 0.0405 | 0.0000 | 13 | 5.15 |
| DSO5 | 2034 | Summer | 1_1 | 0.25 | 0.0391 | 0.0429 | 0.0282 | 0.0000 | 10 | 4.44 |
| DSO5 | 2034 | Autumn | 0_0 | 0.25 | 0.0507 | 0.0525 | 0.0472 | 0.0000 | 10 | 6.93 |
| DSO5 | 2034 | Autumn | 0_1 | 0.25 | 0.0271 | 0.0299 | 0.0225 | 0.0000 | 6 | 3.75 |
| DSO5 | 2034 | Autumn | 1_0 | 0.25 | 0.0525 | 0.0544 | 0.0491 | 0.0000 | 10 | 5.63 |
| DSO5 | 2034 | Autumn | 1_1 | 0.25 | 0.0284 | 0.0312 | 0.0239 | 0.0000 | 6 | 3.00 |
| DSO5 | 2034 | Winter | 0_0 | 0.25 | 0.0438 | 0.0475 | 0.0405 | 0.0000 | 9 | 5.56 |
| DSO5 | 2034 | Winter | 0_1 | 0.25 | 0.0690 | 0.0740 | 0.0655 | 0.0000 | 10 | 8.63 |
| DSO5 | 2034 | Winter | 1_0 | 0.25 | 0.0443 | 0.0480 | 0.0410 | 0.0000 | 9 | 5.90 |
| DSO5 | 2034 | Winter | 1_1 | 0.25 | 0.0625 | 0.0675 | 0.0593 | 0.0000 | 10 | 8.38 |
| DSO5 | 2037 | Spring | 0_0 | 0.25 | 0.0319 | 0.0327 | 0.0209 | 0.0000 | 8 | 2.29 |
| DSO5 | 2037 | Spring | 0_1 | 0.25 | 3.8265 | 3.8271 | 3.8153 | 3.6963 | 15 | 57.91 |
| DSO5 | 2037 | Spring | 1_0 | 0.25 | 0.0260 | 0.0269 | 0.0160 | 0.0000 | 9 | 2.85 |
| DSO5 | 2037 | Spring | 1_1 | 0.25 | 0.1127 | 0.1134 | 0.1009 | 0.0000 | 15 | 17.55 |
| DSO5 | 2037 | Summer | 0_0 | 0.25 | 0.0709 | 0.0742 | 0.0638 | 0.0000 | 9 | 5.68 |
| DSO5 | 2037 | Summer | 0_1 | 0.25 | 0.1103 | 0.1133 | 0.1025 | 0.0000 | 17 | 10.59 |
| DSO5 | 2037 | Summer | 1_0 | 0.25 | 0.0708 | 0.0740 | 0.0637 | 0.0000 | 9 | 6.55 |
| DSO5 | 2037 | Summer | 1_1 | 0.25 | 0.1089 | 0.1119 | 0.1015 | 0.0000 | 17 | 12.16 |
| DSO5 | 2037 | Autumn | 0_0 | 0.25 | 0.0345 | 0.0370 | 0.0284 | 0.0000 | 7 | 4.36 |
| DSO5 | 2037 | Autumn | 0_1 | 0.25 | 0.0342 | 0.0363 | 0.0302 | 0.0000 | 11 | 4.26 |
| DSO5 | 2037 | Autumn | 1_0 | 0.25 | 0.0343 | 0.0369 | 0.0284 | 0.0000 | 7 | 4.48 |
| DSO5 | 2037 | Autumn | 1_1 | 0.25 | 0.0292 | 0.0313 | 0.0243 | 0.0000 | 10 | 3.86 |
| DSO5 | 2037 | Winter | 0_0 | 0.25 | 0.0514 | 0.0558 | 0.0466 | 0.0000 | 9 | 7.10 |
| DSO5 | 2037 | Winter | 0_1 | 0.25 | 0.0546 | 0.0591 | 0.0498 | 0.0000 | 14 | 7.50 |
| DSO5 | 2037 | Winter | 1_0 | 0.25 | 0.0519 | 0.0563 | 0.0464 | 0.0000 | 9 | 7.33 |
| DSO5 | 2037 | Winter | 1_1 | 0.25 | 0.0569 | 0.0613 | 0.0529 | 0.0000 | 15 | 7.98 |
| DSO7 | 2025 | Spring | 0_0 | 0.25 | 0.5234 | 0.5254 | 0.5144 | 0.4375 | 6 | 4.38 |
| DSO7 | 2025 | Spring | 0_1 | 0.25 | 0.1274 | 0.1293 | 0.1250 | 0.0000 | 4 | 3.39 |
| DSO7 | 2025 | Spring | 1_0 | 0.25 | 0.0842 | 0.0862 | 0.0753 | 0.0000 | 5 | 4.86 |
| DSO7 | 2025 | Spring | 1_1 | 0.25 | 0.0960 | 0.0979 | 0.0943 | 0.0000 | 3 | 4.31 |
| DSO7 | 2025 | Summer | 0_0 | 0.25 | 0.0703 | 0.0739 | 0.0621 | 0.0000 | 8 | 3.94 |
| DSO7 | 2025 | Summer | 0_1 | 0.25 | 0.0558 | 0.0593 | 0.0523 | 0.0000 | 5 | 3.07 |
| DSO7 | 2025 | Summer | 1_0 | 0.25 | 0.0703 | 0.0739 | 0.0621 | 0.0000 | 8 | 4.37 |
| DSO7 | 2025 | Summer | 1_1 | 0.25 | 0.0432 | 0.0467 | 0.0378 | 0.0000 | 6 | 2.66 |
| DSO7 | 2025 | Autumn | 0_1 | 0.25 | 0.0117 | 0.0143 | 0.0064 | 0.0000 | 3 | 1.38 |
| DSO7 | 2025 | Autumn | 1_1 | 0.25 | 0.0093 | 0.0120 | 0.0039 | 0.0000 | 3 | 1.15 |
| DSO7 | 2025 | Winter | 0_1 | 0.25 | 0.0152 | 0.0196 | 0.0129 | 0.0000 | 6 | 1.79 |
| DSO7 | 2025 | Winter | 1_1 | 0.25 | 0.0153 | 0.0197 | 0.0132 | 0.0000 | 6 | 1.83 |
| DSO7 | 2028 | Spring | 0_0 | 0.25 | 0.1317 | 0.1338 | 0.1285 | 0.0000 | 5 | 3.73 |
| DSO7 | 2028 | Spring | 0_1 | 0.25 | 0.1507 | 0.1527 | 0.1489 | 0.0000 | 5 | 4.17 |
| DSO7 | 2028 | Spring | 1_0 | 0.25 | 0.1312 | 0.1332 | 0.1280 | 0.0000 | 5 | 6.22 |
| DSO7 | 2028 | Spring | 1_1 | 0.25 | 0.1418 | 0.1439 | 0.1407 | 0.0000 | 5 | 6.61 |
| DSO7 | 2028 | Summer | 0_0 | 0.25 | 0.0499 | 0.0530 | 0.0431 | 0.0000 | 6 | 3.61 |
| DSO7 | 2028 | Summer | 0_1 | 0.25 | 0.0443 | 0.0477 | 0.0313 | 0.0000 | 9 | 4.29 |
| DSO7 | 2028 | Summer | 1_0 | 0.25 | 0.0471 | 0.0502 | 0.0397 | 0.0000 | 6 | 2.85 |
| DSO7 | 2028 | Summer | 1_1 | 0.25 | 0.0550 | 0.0584 | 0.0431 | 0.0000 | 10 | 4.52 |
| DSO7 | 2028 | Autumn | 0_0 | 0.25 | 0.0081 | 0.0114 | 0.0023 | 0.0000 | 2 | 1.23 |
| DSO7 | 2028 | Autumn | 1_0 | 0.25 | 0.0101 | 0.0134 | 0.0051 | 0.0000 | 4 | 1.34 |
| DSO7 | 2028 | Winter | 0_0 | 0.25 | 0.0154 | 0.0198 | 0.0148 | 0.0000 | 7 | 2.11 |
| DSO7 | 2028 | Winter | 0_1 | 0.25 | 0.0089 | 0.0132 | 0.0035 | 0.0000 | 3 | 1.48 |
| DSO7 | 2028 | Winter | 1_0 | 0.25 | 0.0151 | 0.0194 | 0.0143 | 0.0000 | 7 | 2.02 |
| DSO7 | 2028 | Winter | 1_1 | 0.25 | 0.0088 | 0.0132 | 0.0024 | 0.0000 | 2 | 1.46 |
| DSO7 | 2031 | Spring | 0_0 | 0.25 | 0.1708 | 0.1723 | 0.1642 | 0.0000 | 5 | 5.14 |
| DSO7 | 2031 | Spring | 0_1 | 0.25 | 0.0770 | 0.0783 | 0.0683 | 0.0000 | 3 | 2.54 |
| DSO7 | 2031 | Spring | 1_0 | 0.25 | 0.1464 | 0.1479 | 0.1391 | 0.0000 | 4 | 5.64 |
| DSO7 | 2031 | Spring | 1_1 | 0.25 | 0.0769 | 0.0782 | 0.0683 | 0.0000 | 3 | 3.19 |
| DSO7 | 2031 | Summer | 0_0 | 0.25 | 0.0375 | 0.0416 | 0.0283 | 0.0000 | 9 | 3.65 |
| DSO7 | 2031 | Summer | 0_1 | 0.25 | 0.0182 | 0.0217 | 0.0151 | 0.0000 | 3 | 1.55 |
| DSO7 | 2031 | Summer | 1_0 | 0.25 | 0.0329 | 0.0370 | 0.0250 | 0.0000 | 8 | 3.30 |
| DSO7 | 2031 | Summer | 1_1 | 0.25 | 0.0195 | 0.0230 | 0.0150 | 0.0000 | 3 | 1.59 |
| DSO7 | 2031 | Autumn | 0_0 | 0.25 | 0.0236 | 0.0255 | 0.0162 | 0.0000 | 5 | 2.41 |
| DSO7 | 2031 | Autumn | 0_1 | 0.25 | 0.0218 | 0.0238 | 0.0125 | 0.0000 | 5 | 2.05 |
| DSO7 | 2031 | Autumn | 1_0 | 0.25 | 0.0228 | 0.0248 | 0.0162 | 0.0000 | 5 | 2.77 |
| DSO7 | 2031 | Winter | 0_0 | 0.25 | 0.0132 | 0.0170 | 0.0085 | 0.0000 | 6 | 1.91 |
| DSO7 | 2031 | Winter | 0_1 | 0.25 | 0.0104 | 0.0143 | 0.0054 | 0.0000 | 4 | 1.62 |
| DSO7 | 2031 | Winter | 1_0 | 0.25 | 0.0133 | 0.0171 | 0.0086 | 0.0000 | 6 | 1.92 |
| DSO7 | 2031 | Winter | 1_1 | 0.25 | 0.0104 | 0.0143 | 0.0055 | 0.0000 | 4 | 1.63 |
| DSO7 | 2034 | Spring | 0_0 | 0.25 | 0.1555 | 0.1569 | 0.1452 | 0.0000 | 6 | 4.50 |
| DSO7 | 2034 | Spring | 0_1 | 0.25 | 0.0842 | 0.0855 | 0.0715 | 0.0000 | 12 | 5.57 |
| DSO7 | 2034 | Spring | 1_0 | 0.25 | 0.1546 | 0.1560 | 0.1442 | 0.0000 | 6 | 7.36 |
| DSO7 | 2034 | Spring | 1_1 | 0.25 | 0.0668 | 0.0681 | 0.0580 | 0.0000 | 12 | 5.99 |
| DSO7 | 2034 | Summer | 0_0 | 0.25 | 0.1295 | 0.1325 | 0.1195 | 0.0000 | 7 | 7.39 |
| DSO7 | 2034 | Summer | 0_1 | 0.25 | 0.0587 | 0.0621 | 0.0563 | 0.0000 | 4 | 3.24 |
| DSO7 | 2034 | Summer | 1_0 | 0.25 | 0.1044 | 0.1073 | 0.0949 | 0.0000 | 5 | 8.87 |
| DSO7 | 2034 | Summer | 1_1 | 0.25 | 0.0565 | 0.0600 | 0.0544 | 0.0000 | 4 | 4.74 |
| DSO7 | 2034 | Autumn | 0_0 | 0.25 | 0.0177 | 0.0197 | 0.0139 | 0.0000 | 6 | 2.42 |
| DSO7 | 2034 | Autumn | 0_1 | 0.25 | 0.0073 | 0.0101 | 0.0049 | 0.0000 | 4 | 1.35 |
| DSO7 | 2034 | Autumn | 1_0 | 0.25 | 0.0254 | 0.0274 | 0.0198 | 0.0000 | 6 | 2.58 |
| DSO7 | 2034 | Autumn | 1_1 | 0.25 | 0.0073 | 0.0101 | 0.0035 | 0.0000 | 3 | 1.10 |
| DSO7 | 2034 | Winter | 0_0 | 0.25 | 0.0139 | 0.0177 | 0.0092 | 0.0000 | 6 | 2.14 |
| DSO7 | 2034 | Winter | 0_1 | 0.25 | 0.0223 | 0.0260 | 0.0175 | 0.0000 | 8 | 3.05 |
| DSO7 | 2034 | Winter | 1_0 | 0.25 | 0.0141 | 0.0179 | 0.0101 | 0.0000 | 7 | 2.25 |
| DSO7 | 2034 | Winter | 1_1 | 0.25 | 0.0200 | 0.0237 | 0.0152 | 0.0000 | 8 | 2.98 |
| DSO7 | 2037 | Spring | 0_0 | 0.25 | 0.1310 | 0.1319 | 0.1233 | 0.0000 | 7 | 3.03 |
| DSO7 | 2037 | Spring | 0_1 | 0.25 | 3.4121 | 3.4128 | 3.3966 | 3.3049 | 13 | 39.35 |
| DSO7 | 2037 | Spring | 1_0 | 0.25 | 0.1263 | 0.1272 | 0.1170 | 0.0000 | 7 | 6.28 |
| DSO7 | 2037 | Spring | 1_1 | 0.25 | 0.0763 | 0.0771 | 0.0613 | 0.0000 | 9 | 5.63 |
| DSO7 | 2037 | Summer | 0_0 | 0.25 | 0.0814 | 0.0835 | 0.0742 | 0.0000 | 11 | 6.70 |
| DSO7 | 2037 | Summer | 0_1 | 0.25 | 0.0772 | 0.0794 | 0.0726 | 0.0000 | 8 | 5.11 |
| DSO7 | 2037 | Summer | 1_0 | 0.25 | 0.0745 | 0.0766 | 0.0665 | 0.0000 | 11 | 7.27 |
| DSO7 | 2037 | Summer | 1_1 | 0.25 | 0.0769 | 0.0791 | 0.0723 | 0.0000 | 8 | 5.93 |
| DSO7 | 2037 | Autumn | 0_0 | 0.25 | 0.0107 | 0.0120 | 0.0031 | 0.0000 | 3 | 1.41 |
| DSO7 | 2037 | Autumn | 0_1 | 0.25 | 0.0181 | 0.0199 | 0.0118 | 0.0000 | 6 | 2.46 |
| DSO7 | 2037 | Autumn | 1_0 | 0.25 | 0.0092 | 0.0105 | 0.0031 | 0.0000 | 3 | 1.30 |
| DSO7 | 2037 | Autumn | 1_1 | 0.25 | 0.0180 | 0.0197 | 0.0118 | 0.0000 | 6 | 2.50 |
| DSO7 | 2037 | Winter | 0_0 | 0.25 | 0.0206 | 0.0236 | 0.0149 | 0.0000 | 8 | 3.04 |
| DSO7 | 2037 | Winter | 0_1 | 0.25 | 0.0086 | 0.0116 | 0.0039 | 0.0000 | 3 | 1.53 |
| DSO7 | 2037 | Winter | 1_0 | 0.25 | 0.0206 | 0.0236 | 0.0149 | 0.0000 | 8 | 3.12 |
| DSO7 | 2037 | Winter | 1_1 | 0.25 | 0.0086 | 0.0116 | 0.0039 | 0.0000 | 3 | 1.55 |
| DSO9 | 2025 | Spring | 0_0 | 0.25 | 2.2170 | 2.2177 | 2.2005 | 2.1559 | 14 | 10.03 |
| DSO9 | 2025 | Spring | 0_1 | 0.25 | 0.0287 | 0.0297 | 0.0161 | 0.0000 | 8 | 1.14 |
| DSO9 | 2025 | Spring | 1_0 | 0.25 | 0.0626 | 0.0633 | 0.0453 | 0.0000 | 11 | 4.21 |
| DSO9 | 2025 | Spring | 1_1 | 0.25 | 0.0282 | 0.0292 | 0.0149 | 0.0000 | 7 | 1.64 |
| DSO9 | 2025 | Summer | 0_0 | 0.25 | 0.0651 | 0.0676 | 0.0511 | 0.0000 | 12 | 4.17 |
| DSO9 | 2025 | Summer | 0_1 | 0.25 | 0.0521 | 0.0555 | 0.0412 | 0.0000 | 11 | 3.32 |
| DSO9 | 2025 | Summer | 1_0 | 0.25 | 0.0627 | 0.0652 | 0.0482 | 0.0000 | 12 | 4.47 |
| DSO9 | 2025 | Summer | 1_1 | 0.25 | 0.0505 | 0.0539 | 0.0392 | 0.0000 | 11 | 3.60 |
| DSO9 | 2025 | Autumn | 0_0 | 0.25 | 0.0148 | 0.0166 | 0.0087 | 0.0000 | 4 | 1.70 |
| DSO9 | 2025 | Autumn | 0_1 | 0.25 | 0.0211 | 0.0232 | 0.0056 | 0.0000 | 4 | 2.31 |
| DSO9 | 2025 | Autumn | 1_0 | 0.25 | 0.0151 | 0.0169 | 0.0086 | 0.0000 | 4 | 1.70 |
| DSO9 | 2025 | Autumn | 1_1 | 0.25 | 0.0231 | 0.0251 | 0.0072 | 0.0000 | 5 | 2.48 |
| DSO9 | 2025 | Winter | 0_1 | 0.25 | 0.0264 | 0.0303 | 0.0126 | 0.0000 | 6 | 3.11 |
| DSO9 | 2025 | Winter | 1_1 | 0.25 | 0.0264 | 0.0303 | 0.0128 | 0.0000 | 6 | 3.22 |
| DSO9 | 2028 | Spring | 0_0 | 0.25 | 0.0915 | 0.0923 | 0.0802 | 0.0001 | 15 | 5.12 |
| DSO9 | 2028 | Spring | 0_1 | 0.25 | 0.0302 | 0.0310 | 0.0180 | 0.0000 | 9 | 1.68 |
| DSO9 | 2028 | Spring | 1_0 | 0.25 | 0.0907 | 0.0915 | 0.0796 | 0.0000 | 15 | 6.89 |
| DSO9 | 2028 | Spring | 1_1 | 0.25 | 0.0301 | 0.0309 | 0.0180 | 0.0000 | 9 | 2.24 |
| DSO9 | 2028 | Summer | 0_0 | 0.25 | 0.0434 | 0.0467 | 0.0349 | 0.0000 | 11 | 3.62 |
| DSO9 | 2028 | Summer | 0_1 | 0.25 | 0.1004 | 0.1030 | 0.0899 | 0.0000 | 12 | 8.17 |
| DSO9 | 2028 | Summer | 1_0 | 0.25 | 0.0464 | 0.0497 | 0.0371 | 0.0000 | 10 | 3.28 |
| DSO9 | 2028 | Summer | 1_1 | 0.25 | 0.0976 | 0.1002 | 0.0866 | 0.0000 | 12 | 6.92 |
| DSO9 | 2028 | Autumn | 0_1 | 0.25 | 0.0301 | 0.0324 | 0.0167 | 0.0000 | 10 | 4.36 |
| DSO9 | 2028 | Autumn | 1_1 | 0.25 | 0.0298 | 0.0321 | 0.0156 | 0.0000 | 9 | 3.35 |
| DSO9 | 2028 | Winter | 0_0 | 0.25 | 0.0119 | 0.0158 | 0.0013 | 0.0000 | 1 | 1.95 |
| DSO9 | 2028 | Winter | 0_1 | 0.25 | 0.0156 | 0.0198 | 0.0073 | 0.0000 | 4 | 2.44 |
| DSO9 | 2028 | Winter | 1_0 | 0.25 | 0.0082 | 0.0121 | 0.0000 | 0.0000 | 0 | 1.44 |
| DSO9 | 2028 | Winter | 1_1 | 0.25 | 0.0163 | 0.0205 | 0.0083 | 0.0000 | 4 | 2.37 |
| DSO9 | 2031 | Spring | 0_0 | 0.25 | 0.0529 | 0.0538 | 0.0367 | 0.0000 | 12 | 4.12 |
| DSO9 | 2031 | Spring | 0_1 | 0.25 | 0.0474 | 0.0481 | 0.0275 | 0.0000 | 14 | 3.32 |
| DSO9 | 2031 | Spring | 1_0 | 0.25 | 0.0541 | 0.0550 | 0.0403 | 0.0000 | 14 | 4.70 |
| DSO9 | 2031 | Spring | 1_1 | 0.25 | 0.0473 | 0.0481 | 0.0275 | 0.0000 | 14 | 3.74 |
| DSO9 | 2031 | Summer | 0_0 | 0.25 | 0.0763 | 0.0799 | 0.0628 | 0.0000 | 11 | 6.57 |
| DSO9 | 2031 | Summer | 0_1 | 0.25 | 0.0594 | 0.0627 | 0.0453 | 0.0000 | 12 | 5.35 |
| DSO9 | 2031 | Summer | 1_0 | 0.25 | 0.0725 | 0.0760 | 0.0585 | 0.0000 | 11 | 6.20 |
| DSO9 | 2031 | Summer | 1_1 | 0.25 | 0.0590 | 0.0623 | 0.0460 | 0.0000 | 12 | 5.22 |
| DSO9 | 2031 | Autumn | 0_0 | 0.25 | 0.0297 | 0.0314 | 0.0198 | 0.0000 | 9 | 2.71 |
| DSO9 | 2031 | Autumn | 0_1 | 0.25 | 0.0262 | 0.0273 | 0.0120 | 0.0000 | 7 | 2.63 |
| DSO9 | 2031 | Autumn | 1_0 | 0.25 | 0.0341 | 0.0359 | 0.0235 | 0.0000 | 9 | 4.34 |
| DSO9 | 2031 | Autumn | 1_1 | 0.25 | 0.0303 | 0.0314 | 0.0162 | 0.0000 | 7 | 4.06 |
| DSO9 | 2031 | Winter | 0_0 | 0.25 | 0.0226 | 0.0268 | 0.0110 | 0.0000 | 7 | 3.29 |
| DSO9 | 2031 | Winter | 0_1 | 0.25 | 0.0188 | 0.0227 | 0.0079 | 0.0000 | 5 | 2.85 |
| DSO9 | 2031 | Winter | 1_0 | 0.25 | 0.0221 | 0.0263 | 0.0106 | 0.0000 | 7 | 3.26 |
| DSO9 | 2031 | Winter | 1_1 | 0.25 | 0.0187 | 0.0226 | 0.0079 | 0.0000 | 5 | 2.85 |
| DSO9 | 2034 | Spring | 0_0 | 0.25 | 0.0430 | 0.0439 | 0.0230 | 0.0000 | 9 | 2.94 |
| DSO9 | 2034 | Spring | 0_1 | 0.25 | 0.4784 | 0.4793 | 0.4663 | 0.4283 | 10 | 6.08 |
| DSO9 | 2034 | Spring | 1_0 | 0.25 | 0.0415 | 0.0424 | 0.0229 | 0.0000 | 9 | 3.78 |
| DSO9 | 2034 | Spring | 1_1 | 0.25 | 0.0513 | 0.0523 | 0.0375 | 0.0000 | 10 | 4.29 |
| DSO9 | 2034 | Summer | 0_0 | 0.25 | 0.0759 | 0.0791 | 0.0583 | 0.0000 | 15 | 6.27 |
| DSO9 | 2034 | Summer | 0_1 | 0.25 | 0.0748 | 0.0785 | 0.0652 | 0.0000 | 12 | 5.60 |
| DSO9 | 2034 | Summer | 1_0 | 0.25 | 0.0736 | 0.0768 | 0.0556 | 0.0000 | 15 | 7.94 |
| DSO9 | 2034 | Summer | 1_1 | 0.25 | 0.0739 | 0.0776 | 0.0632 | 0.0000 | 12 | 7.72 |
| DSO9 | 2034 | Autumn | 0_0 | 0.25 | 0.0237 | 0.0253 | 0.0134 | 0.0000 | 8 | 4.08 |
| DSO9 | 2034 | Autumn | 0_1 | 0.25 | 0.0471 | 0.0487 | 0.0380 | 0.0000 | 12 | 7.72 |
| DSO9 | 2034 | Autumn | 1_0 | 0.25 | 0.0228 | 0.0245 | 0.0115 | 0.0000 | 8 | 2.70 |
| DSO9 | 2034 | Autumn | 1_1 | 0.25 | 0.0466 | 0.0482 | 0.0367 | 0.0000 | 12 | 5.05 |
| DSO9 | 2034 | Winter | 0_0 | 0.25 | 0.0399 | 0.0438 | 0.0284 | 0.0000 | 11 | 5.45 |
| DSO9 | 2034 | Winter | 1_0 | 0.25 | 0.0399 | 0.0438 | 0.0293 | 0.0000 | 11 | 6.20 |
| DSO9 | 2037 | Spring | 0_0 | 0.25 | 4.8999 | 4.9005 | 4.8882 | 4.7978 | 13 | 19.50 |
| DSO9 | 2037 | Spring | 0_1 | 0.25 | 0.0409 | 0.0416 | 0.0242 | 0.0000 | 11 | 2.35 |
| DSO9 | 2037 | Spring | 1_0 | 0.25 | 0.1039 | 0.1046 | 0.0900 | 0.0000 | 12 | 8.49 |
| DSO9 | 2037 | Spring | 1_1 | 0.25 | 0.0413 | 0.0420 | 0.0239 | 0.0000 | 10 | 3.78 |
| DSO9 | 2037 | Summer | 0_0 | 0.25 | 0.0640 | 0.0661 | 0.0564 | 0.0000 | 10 | 4.96 |
| DSO9 | 2037 | Summer | 0_1 | 0.25 | 0.1020 | 0.1049 | 0.0911 | 0.0000 | 14 | 9.11 |
| DSO9 | 2037 | Summer | 1_0 | 0.25 | 0.0685 | 0.0706 | 0.0607 | 0.0000 | 11 | 5.91 |
| DSO9 | 2037 | Summer | 1_1 | 0.25 | 0.1021 | 0.1050 | 0.0901 | 0.0000 | 13 | 10.29 |
| DSO9 | 2037 | Autumn | 0_0 | 0.25 | 0.0566 | 0.0580 | 0.0434 | 0.0000 | 11 | 6.95 |
| DSO9 | 2037 | Autumn | 0_1 | 0.25 | 0.0254 | 0.0269 | 0.0107 | 0.0000 | 5 | 3.22 |
| DSO9 | 2037 | Autumn | 1_0 | 0.25 | 0.0568 | 0.0582 | 0.0436 | 0.0000 | 11 | 7.61 |
| DSO9 | 2037 | Autumn | 1_1 | 0.25 | 0.0255 | 0.0270 | 0.0108 | 0.0000 | 5 | 3.54 |
| DSO9 | 2037 | Winter | 0_0 | 0.25 | 0.0238 | 0.0277 | 0.0087 | 0.0000 | 5 | 3.81 |
| DSO9 | 2037 | Winter | 0_1 | 0.25 | 0.0322 | 0.0363 | 0.0148 | 0.0000 | 8 | 5.07 |
| DSO9 | 2037 | Winter | 1_0 | 0.25 | 0.0245 | 0.0284 | 0.0099 | 0.0000 | 6 | 3.91 |
| DSO9 | 2037 | Winter | 1_1 | 0.25 | 0.0295 | 0.0336 | 0.0132 | 0.0000 | 9 | 4.72 |

**srp1_unit** (block resolution: day net only)

| block | net (MWh/day) | x0 net (MWh/day) | unit - x0 | penalty term in Q |
|---|---|---|---|---|
| DSO|5|2025|Spring | 0.0142 | 0.0142 | 0.00000 | 0.0 |
| DSO|5|2025|Summer | 0.0157 | 0.0157 | -0.00000 | 0.0 |
| DSO|5|2025|Autumn | 0.0188 | 0.0188 | 0.00000 | 0.0 |
| DSO|5|2025|Winter | 0.0335 | 0.0335 | 0.00000 | 0.0 |
| DSO|5|2030|Spring | 0.0229 | 0.0229 | -0.00003 | 0.0 |
| DSO|5|2030|Summer | 0.0334 | 0.0334 | -0.00000 | 0.0 |
| DSO|5|2030|Autumn | 0.0267 | 0.0267 | -0.00000 | 0.0 |
| DSO|5|2030|Winter | 0.0425 | 0.0425 | -0.00000 | 0.0 |
| DSO|5|2035|Spring | 0.0364 | 0.0364 | -0.00000 | 0.0 |
| DSO|5|2035|Summer | 0.0759 | 0.0759 | -0.00000 | 0.0 |
| DSO|5|2035|Autumn | 0.0381 | 0.0381 | 0.00000 | 0.0 |
| DSO|5|2035|Winter | 0.0564 | 0.0564 | 0.00000 | 0.0 |
| DSO|7|2025|Spring | 0.1055 | 0.1055 | -0.00002 | 0.0 |
| DSO|7|2025|Summer | 0.0740 | 0.0740 | 0.00000 | 0.0 |
| DSO|7|2030|Spring | 0.1073 | 0.1074 | -0.00006 | 0.0 |
| DSO|7|2030|Summer | 0.0568 | 0.0568 | -0.00004 | 0.0 |
| DSO|7|2030|Winter | 0.0128 | 0.0128 | -0.00000 | 0.0 |
| DSO|7|2035|Spring | 0.0530 | 0.0529 | 0.00005 | 0.0 |
| DSO|7|2035|Summer | 0.1641 | 0.1642 | -0.00004 | 0.0 |
| DSO|7|2035|Autumn | 0.0127 | 0.0127 | 0.00000 | 0.0 |
| DSO|7|2035|Winter | 0.0154 | 0.0154 | -0.00001 | 0.0 |
| DSO|9|2025|Spring | 0.0348 | 0.0349 | -0.00016 | 0.0 |
| DSO|9|2025|Summer | 0.0531 | 0.0531 | 0.00000 | 0.0 |
| DSO|9|2025|Autumn | 0.0149 | 0.0149 | 0.00000 | 0.0 |
| DSO|9|2030|Spring | 0.0372 | 0.0373 | -0.00011 | 0.0 |
| DSO|9|2030|Summer | 0.0627 | 0.0627 | -0.00000 | 0.0 |
| DSO|9|2030|Autumn | 0.0288 | 0.0288 | 0.00000 | 0.0 |
| DSO|9|2030|Winter | 0.0191 | 0.0191 | 0.00000 | 0.0 |
| DSO|9|2035|Spring | 0.0576 | 0.0576 | -0.00001 | 0.0 |
| DSO|9|2035|Summer | 0.0842 | 0.0842 | 0.00000 | 0.0 |
| DSO|9|2035|Autumn | 0.0295 | 0.0295 | 0.00001 | 0.0 |
| DSO|9|2035|Winter | 0.0273 | 0.0273 | -0.00000 | 0.0 |

## 5. Hourly profiles (MW, sum over the network's curtaillable generators, above tol) -- the network-day-scenarios with at least 0.1 MWh above tol

**srp1_x0**: 3 rows

| network | year | day | scen | h0 | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | h9 | h10 | h11 | h12 | h13 | h14 | h15 | h16 | h17 | h18 | h19 | h20 | h21 | h22 | h23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DSO7 | 2025 | Spring | 0_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.013 | 0.014 | 0.032 | 0.045 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2030 | Spring | 0_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.001 | 0.009 | 0.046 | 0.045 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2035 | Summer | 0_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.002 | 0.003 | 0.028 | 0.065 | 0.061 | 0.002 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**pilot_x0**: 24 rows

| network | year | day | scen | h0 | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | h9 | h10 | h11 | h12 | h13 | h14 | h15 | h16 | h17 | h18 | h19 | h20 | h21 | h22 | h23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DSO5 | 2025 | Spring | 0_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.004 | 0.004 | 0.001 | 2.414 | 0 | 0 | 0 | 0 | 0 | 0.001 | 0 | 0.002 | 0 |
| DSO5 | 2025 | Winter | 0_1 | 0.008 | 0.009 | 0.011 | 0.009 | 0.014 | 0.013 | 0.003 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.003 | 0.004 | 0.003 | 0.002 | 0.003 | 0.003 | 0.003 | 0.005 | 0.007 | 0.007 |
| DSO5 | 2025 | Winter | 1_1 | 0.008 | 0.009 | 0.009 | 0.015 | 0.012 | 0.008 | 0.003 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.004 | 0.004 | 0.003 | 0.002 | 0.003 | 0.003 | 0.003 | 0.005 | 0.007 | 0.007 |
| DSO5 | 2037 | Spring | 0_1 | 0.004 | 0.017 | 0.009 | 0.015 | 0.015 | 0.005 | 0 | 0 | 0.003 | 0.008 | 0.010 | 0.004 | 0.010 | 0.004 | 0.821 | 0 | 2.890 | 0 | 0 | 0 | 0 | 0.001 | 0 | 0 |
| DSO5 | 2037 | Spring | 1_1 | 0.004 | 0.017 | 0.008 | 0.014 | 0.014 | 0.004 | 0 | 0 | 0.003 | 0.008 | 0.010 | 0.002 | 0.010 | 0.004 | 0.001 | 0 | 0 | 0 | 0 | 0 | 0 | 0.001 | 0 | 0 |
| DSO5 | 2037 | Summer | 0_1 | 0.002 | 0.004 | 0.004 | 0.005 | 0.006 | 0.005 | 0 | 0 | 0.008 | 0.010 | 0.008 | 0.012 | 0.014 | 0.011 | 0.004 | 0.002 | 0.002 | 0 | 0 | 0 | 0 | 0 | 0.002 | 0.005 |
| DSO5 | 2037 | Summer | 1_1 | 0.002 | 0.003 | 0.004 | 0.005 | 0.006 | 0.005 | 0 | 0 | 0.008 | 0.010 | 0.008 | 0.011 | 0.014 | 0.011 | 0.004 | 0.002 | 0.002 | 0 | 0 | 0 | 0 | 0 | 0.002 | 0.005 |
| DSO7 | 2025 | Spring | 0_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.008 | 0.010 | 0.024 | 0.032 | 0.001 | 0.439 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2025 | Spring | 0_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.012 | 0.028 | 0.021 | 0.064 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2028 | Spring | 0_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.006 | 0.018 | 0.024 | 0.039 | 0.042 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2028 | Spring | 0_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.011 | 0.015 | 0.036 | 0.044 | 0.043 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2028 | Spring | 1_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.006 | 0.017 | 0.024 | 0.039 | 0.042 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2028 | Spring | 1_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.009 | 0.013 | 0.034 | 0.042 | 0.041 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2031 | Spring | 0_0 | 0 | 0 | 0 | 0 | 0.001 | 0 | 0 | 0 | 0 | 0 | 0.019 | 0.021 | 0.072 | 0.050 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2031 | Spring | 1_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.014 | 0.016 | 0.064 | 0.045 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2034 | Spring | 0_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.004 | 0 | 0.007 | 0.009 | 0.051 | 0.051 | 0.036 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2034 | Spring | 1_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.004 | 0 | 0.007 | 0.009 | 0.051 | 0.051 | 0.036 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2034 | Summer | 0_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.001 | 0.005 | 0.017 | 0.033 | 0.039 | 0.023 | 0 | 0.001 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2037 | Spring | 0_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.001 | 0.001 | 0.002 | 0.011 | 0.042 | 0.052 | 0.026 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2037 | Spring | 0_1 | 0 | 0.001 | 0 | 0 | 0.003 | 0.001 | 0 | 0 | 0 | 0 | 0.006 | 0.002 | 0.026 | 0.034 | 1.186 | 0 | 2.147 | 0.003 | 0 | 0 | 0 | 0 | 0.001 | 0.002 |
| DSO7 | 2037 | Spring | 1_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.001 | 0.001 | 0.002 | 0.007 | 0.041 | 0.051 | 0.026 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO9 | 2025 | Spring | 0_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.002 | 0.003 | 0.003 | 0.007 | 0.007 | 0.006 | 2.160 | 0.003 | 0.001 | 0.001 | 0 | 0.001 | 0.004 | 0.002 | 0 | 0 |
| DSO9 | 2034 | Spring | 0_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.004 | 0.009 | 0.002 | 0.006 | 0 | 0.430 | 0.003 | 0 | 0.006 | 0 | 0 | 0.003 | 0 | 0.002 | 0.002 |
| DSO9 | 2037 | Spring | 0_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.002 | 0.008 | 0.012 | 0.006 | 0.006 | 0.022 | 0.010 | 4.407 | 0.007 | 0.402 | 0.003 | 0 | 0.001 | 0.003 | 0 | 0 | 0 |

**pilot_unit**: 24 rows

| network | year | day | scen | h0 | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | h9 | h10 | h11 | h12 | h13 | h14 | h15 | h16 | h17 | h18 | h19 | h20 | h21 | h22 | h23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DSO5 | 2025 | Spring | 0_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.004 | 0.004 | 0.001 | 2.413 | 0 | 0 | 0 | 0 | 0 | 0.001 | 0 | 0.002 | 0 |
| DSO5 | 2025 | Winter | 0_1 | 0.008 | 0.009 | 0.011 | 0.009 | 0.014 | 0.013 | 0.003 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.003 | 0.004 | 0.003 | 0.002 | 0.003 | 0.003 | 0.003 | 0.005 | 0.007 | 0.007 |
| DSO5 | 2025 | Winter | 1_1 | 0.008 | 0.009 | 0.009 | 0.015 | 0.012 | 0.008 | 0.003 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.004 | 0.004 | 0.003 | 0.002 | 0.003 | 0.003 | 0.003 | 0.005 | 0.007 | 0.007 |
| DSO5 | 2037 | Spring | 0_1 | 0.004 | 0.017 | 0.009 | 0.015 | 0.015 | 0.005 | 0 | 0 | 0.004 | 0.008 | 0.010 | 0.004 | 0.010 | 0.004 | 0.820 | 0 | 2.890 | 0 | 0 | 0 | 0 | 0.001 | 0 | 0 |
| DSO5 | 2037 | Spring | 1_1 | 0.004 | 0.017 | 0.009 | 0.014 | 0.014 | 0.004 | 0 | 0.001 | 0.004 | 0.008 | 0.009 | 0.002 | 0.010 | 0.004 | 0.001 | 0 | 0 | 0 | 0 | 0 | 0 | 0.001 | 0 | 0 |
| DSO5 | 2037 | Summer | 0_1 | 0.002 | 0.004 | 0.004 | 0.005 | 0.006 | 0.005 | 0 | 0 | 0.008 | 0.010 | 0.008 | 0.012 | 0.014 | 0.011 | 0.004 | 0.002 | 0.002 | 0 | 0 | 0 | 0 | 0 | 0.002 | 0.005 |
| DSO5 | 2037 | Summer | 1_1 | 0.002 | 0.003 | 0.004 | 0.005 | 0.006 | 0.005 | 0 | 0 | 0.008 | 0.010 | 0.008 | 0.011 | 0.014 | 0.011 | 0.004 | 0.002 | 0.002 | 0 | 0 | 0 | 0 | 0 | 0.002 | 0.005 |
| DSO7 | 2025 | Spring | 0_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.008 | 0.010 | 0.024 | 0.032 | 0.001 | 0.439 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2025 | Spring | 0_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.012 | 0.028 | 0.021 | 0.064 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2028 | Spring | 0_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.006 | 0.018 | 0.024 | 0.039 | 0.042 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2028 | Spring | 0_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.011 | 0.015 | 0.036 | 0.044 | 0.043 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2028 | Spring | 1_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.006 | 0.017 | 0.024 | 0.039 | 0.042 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2028 | Spring | 1_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.009 | 0.013 | 0.034 | 0.042 | 0.041 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2031 | Spring | 0_0 | 0 | 0 | 0 | 0 | 0.001 | 0 | 0 | 0 | 0 | 0 | 0.019 | 0.021 | 0.072 | 0.050 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2031 | Spring | 1_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.014 | 0.016 | 0.064 | 0.045 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2034 | Spring | 0_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.004 | 0 | 0.007 | 0.009 | 0.051 | 0.051 | 0.023 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2034 | Spring | 1_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.004 | 0 | 0.007 | 0.009 | 0.051 | 0.051 | 0.023 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2034 | Summer | 0_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.001 | 0.005 | 0.017 | 0.033 | 0.039 | 0.023 | 0 | 0.001 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2037 | Spring | 0_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.001 | 0.001 | 0.002 | 0.011 | 0.042 | 0.052 | 0.014 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO7 | 2037 | Spring | 0_1 | 0 | 0.001 | 0 | 0 | 0.003 | 0.001 | 0 | 0 | 0 | 0 | 0.006 | 0.002 | 0.026 | 0.034 | 1.170 | 0.001 | 2.147 | 0.002 | 0 | 0 | 0 | 0 | 0.001 | 0.002 |
| DSO7 | 2037 | Spring | 1_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.001 | 0.001 | 0.002 | 0.007 | 0.041 | 0.051 | 0.014 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSO9 | 2025 | Spring | 0_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.001 | 0.002 | 0.003 | 0.003 | 0.007 | 0.007 | 0.006 | 2.160 | 0.003 | 0.001 | 0.001 | 0 | 0.001 | 0.004 | 0.002 | 0 | 0 |
| DSO9 | 2034 | Spring | 0_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.004 | 0.009 | 0.002 | 0.006 | 0 | 0.430 | 0.003 | 0 | 0.006 | 0 | 0 | 0.003 | 0 | 0.002 | 0.002 |
| DSO9 | 2037 | Spring | 0_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.002 | 0.008 | 0.012 | 0.006 | 0.006 | 0.022 | 0.010 | 4.406 | 0.007 | 0.402 | 0.003 | 0 | 0.001 | 0.003 | 0 | 0 | 0 |

## 6. Causes

Topology read from the certified srp1_x0 models (bus indices are 0-based model indices; DN index 0 = case33 bus 1, the 345 kV busbar; the DN transformer is branch 1-2): {"TSO": {"shared_ess_in_node_balance_of_bus_index": [4, 6, 8], "curtailable_gen_bus_index": {"3": 3, "4": 5, "5": 7, "6": 3, "7": 5, "8": 7}}, "DSO5": {"reference_gen": 0, "reference_bus_index": 0, "shared_ess_in_node_balance_of_bus_index": [0], "pg_adn_expr": "pg[0,0,0,0] - shared_es_pnet[0,0,0,0]", "curtailable_gen_bus_index": {"1": 5, "2": 22, "3": 24, "4": 17}}, "DSO7": {"reference_gen": 0, "reference_bus_index": 0, "shared_ess_in_node_balance_of_bus_index": [0], "pg_adn_expr": "pg[0,0,0,0] - shared_es_pnet[0,0,0,0]", "curtailable_gen_bus_index": {"1": 5, "2": 22, "3": 24, "4": 17}}, "DSO9": {"reference_gen": 0, "reference_bus_index": 0, "shared_ess_in_node_balance_of_bus_index": [0], "pg_adn_expr": "pg[0,0,0,0] - shared_es_pnet[0,0,0,0]", "curtailable_gen_bus_index": {"1": 5, "2": 24, "3": 22, "4": 17}}}

**srp1_x0**: 364 generator-hours above tol: 364 capability_bound, 0 interior. capability_bound = the apparent-power row sg_capability (pg^2 + qg^2 <= sg_avail^2) is active with a non-zero dual: pg is below pg_avail because the inverter supplies Q at its S limit (Sg_curt ~ 0) -- a binding constraint, not surplus. Every entry carries the active voltage / branch rows of its network-hour with their duals (JSON); the table splits by whether the DN transformer limit (a branch with a ratio Var) and a voltage bound are active then.

| network | class | transformer_limit_active | voltage_bound_active | generator_hours | sum_c_mw_over_entries_unweighted | sg_capability_dual_raw_min | sg_capability_dual_raw_max |
|---|---|---|---|---|---|---|---|
| DSO5 | capability_bound | False | True | 121 | 0.3666 | -67,751.3523 | -6,455.1586 |
| DSO7 | capability_bound | False | True | 22 | 0.0325 | -27,102.7809 | -7,639.4612 |
| DSO7 | capability_bound | True | True | 74 | 0.5321 | -109,723.6309 | -10,889.5193 |
| DSO9 | capability_bound | False | False | 4 | 0.0054 | -1,022.8023 | -853.1918 |
| DSO9 | capability_bound | False | True | 143 | 0.2903 | -100,692.0884 | -1,987.9696 |

Horizon split (horizon_split_by_class; above-tol entries; Q weighting; C at the hourly price):

| group | generator-hours | H above tol (MWh) | C above tol (EUR) |
|---|---|---|---|
| capability_bound | 364 | 499.66 | 31,369.59 |

Horizon split (horizon_split_by_class_transformer_voltage; above-tol entries; Q weighting; C at the hourly price):

| group | generator-hours | H above tol (MWh) | C above tol (EUR) |
|---|---|---|---|
| ('DSO5', 'capability_bound', 'transformer_limit_active=False', 'voltage_bound_active=True') | 121 | 147.35 | 14,329.37 |
| ('DSO7', 'capability_bound', 'transformer_limit_active=False', 'voltage_bound_active=True') | 22 | 12.69 | 1,230.16 |
| ('DSO7', 'capability_bound', 'transformer_limit_active=True', 'voltage_bound_active=True') | 74 | 220.11 | 7,879.37 |
| ('DSO9', 'capability_bound', 'transformer_limit_active=False', 'voltage_bound_active=False') | 4 | 2.36 | 8.18 |
| ('DSO9', 'capability_bound', 'transformer_limit_active=False', 'voltage_bound_active=True') | 143 | 117.15 | 7,922.51 |

**pilot_x0**: 2932 generator-hours above tol (2920 DN-side, 12 TSO-side). NO DUALS (models not persisted): the columns are primal indicators only. Shares below are of the above-tol energy summed over entries (unweighted). Share below the apparent capability: 0.6576; share in DN entries meeting the row-18 condition (d <= +tol and pi_s < alpha*pibar): 0.6589; share with an LV voltage at Vmax: 0.3424; max transformer 1-2 loading (workbook Flow_ij, 1.0 = at rating) at a curtailed hour: 1.000005001392292; share with the transformer at rating: 0.1126; share below the apparent capability AND meeting the row-18 condition: 0.6570; TSO entries with all conventional units at Pmin: 0 of 12.

| network | class_primal | row18_condition | lv_at_vmax | transformer_at_rating | generator_hours | sum_c_mw_over_entries_unweighted |
|---|---|---|---|---|---|---|
| DSO5 | P_for_Q_at_capability | False | True | False | 832 | 2.9245 |
| DSO5 | below_apparent_capability | True | False | False | 16 | 6.1376 |
| DSO7 | P_for_Q_at_capability | False | True | False | 377 | 0.6391 |
| DSO7 | P_for_Q_at_capability | False | True | True | 496 | 2.9187 |
| DSO7 | P_for_Q_at_capability | True | True | True | 8 | 0.0485 |
| DSO7 | below_apparent_capability | True | False | False | 12 | 3.7722 |
| DSO9 | P_for_Q_at_capability | False | True | False | 1163 | 2.4894 |
| DSO9 | below_apparent_capability | True | False | False | 16 | 7.3995 |
| TSO | below_apparent_capability | n/a | n/a | n/a | 12 | 0.0156 |

Horizon split (horizon_split_by_class; above-tol entries; Q weighting; C at the hourly price):

| group | generator-hours | H above tol (MWh) | C above tol (EUR) |
|---|---|---|---|
| ('DN', 'P_for_Q_at_capability') | 2876 | 543.83 | 41,630.34 |
| ('DN', 'below_apparent_capability') | 44 | 1,016.37 | 6,431.46 |
| TSO | 12 | 0.85 | 1.60 |

Horizon split (horizon_split_by_class_row18_trafo; above-tol entries; Q weighting; C at the hourly price):

| group | generator-hours | H above tol (MWh) | C above tol (EUR) |
|---|---|---|---|
| ('DSO5', 'P_for_Q_at_capability', 'row18_condition=False', 'transformer_at_rating=False') | 832 | 174.99 | 18,832.21 |
| ('DSO5', 'below_apparent_capability', 'row18_condition=True', 'transformer_at_rating=False') | 16 | 369.19 | 3,012.06 |
| ('DSO7', 'P_for_Q_at_capability', 'row18_condition=False', 'transformer_at_rating=False') | 377 | 38.26 | 3,811.07 |
| ('DSO7', 'P_for_Q_at_capability', 'row18_condition=False', 'transformer_at_rating=True') | 496 | 178.26 | 6,793.90 |
| ('DSO7', 'P_for_Q_at_capability', 'row18_condition=True', 'transformer_at_rating=True') | 8 | 2.80 | 18.31 |
| ('DSO7', 'below_apparent_capability', 'row18_condition=True', 'transformer_at_rating=False') | 12 | 211.64 | 1,988.63 |
| ('DSO9', 'P_for_Q_at_capability', 'row18_condition=False', 'transformer_at_rating=False') | 1163 | 149.52 | 12,174.85 |
| ('DSO9', 'below_apparent_capability', 'row18_condition=True', 'transformer_at_rating=False') | 16 | 435.53 | 1,430.77 |
| ('TSO', 'below_apparent_capability', 'row18_condition=n/a', 'transformer_at_rating=n/a') | 12 | 0.85 | 1.60 |

Hours with curtailment below the apparent capability (network, year, day, scenario, hour): ('DSO5', 2025, 'Spring', '0_1', 14); ('DSO5', 2034, 'Spring', '0_0', 14); ('DSO5', 2037, 'Spring', '0_1', 14); ('DSO5', 2037, 'Spring', '0_1', 16); ('DSO7', 2025, 'Spring', '0_0', 14); ('DSO7', 2037, 'Spring', '0_1', 14); ('DSO7', 2037, 'Spring', '0_1', 16); ('DSO9', 2025, 'Spring', '0_0', 14); ('DSO9', 2034, 'Spring', '0_1', 14); ('DSO9', 2037, 'Spring', '0_0', 14); ('DSO9', 2037, 'Spring', '0_0', 16)

**pilot_unit**: 2919 generator-hours above tol (2919 DN-side, 0 TSO-side). NO DUALS (models not persisted): the columns are primal indicators only. Shares below are of the above-tol energy summed over entries (unweighted). Share below the apparent capability: 0.6589; share in DN entries meeting the row-18 condition (d <= +tol and pi_s < alpha*pibar): 0.6600; share with an LV voltage at Vmax: 0.3411; max transformer 1-2 loading (workbook Flow_ij, 1.0 = at rating) at a curtailed hour: 1.000005001392106; share with the transformer at rating: 0.1107; share below the apparent capability AND meeting the row-18 condition: 0.6589; TSO entries with all conventional units at Pmin: 0 of 0.

| network | class_primal | row18_condition | lv_at_vmax | transformer_at_rating | generator_hours | sum_c_mw_over_entries_unweighted |
|---|---|---|---|---|---|---|
| DSO5 | P_for_Q_at_capability | False | True | False | 833 | 2.9278 |
| DSO5 | below_apparent_capability | True | False | False | 16 | 6.1370 |
| DSO7 | P_for_Q_at_capability | False | True | False | 375 | 0.6295 |
| DSO7 | P_for_Q_at_capability | False | True | True | 496 | 2.8762 |
| DSO7 | P_for_Q_at_capability | True | True | True | 8 | 0.0292 |
| DSO7 | below_apparent_capability | True | False | False | 12 | 3.7565 |
| DSO9 | P_for_Q_at_capability | False | True | False | 1163 | 2.4899 |
| DSO9 | below_apparent_capability | True | False | False | 16 | 7.3983 |

Horizon split (horizon_split_by_class; above-tol entries; Q weighting; C at the hourly price):

| group | generator-hours | H above tol (MWh) | C above tol (EUR) |
|---|---|---|---|
| ('DN', 'P_for_Q_at_capability') | 2875 | 540.02 | 41,603.83 |
| ('DN', 'below_apparent_capability') | 44 | 1,015.41 | 6,429.53 |

Horizon split (horizon_split_by_class_row18_trafo; above-tol entries; Q weighting; C at the hourly price):

| group | generator-hours | H above tol (MWh) | C above tol (EUR) |
|---|---|---|---|
| ('DSO5', 'P_for_Q_at_capability', 'row18_condition=False', 'transformer_at_rating=False') | 833 | 175.18 | 18,863.76 |
| ('DSO5', 'below_apparent_capability', 'row18_condition=True', 'transformer_at_rating=False') | 16 | 369.16 | 3,011.97 |
| ('DSO7', 'P_for_Q_at_capability', 'row18_condition=False', 'transformer_at_rating=False') | 375 | 37.70 | 3,788.91 |
| ('DSO7', 'P_for_Q_at_capability', 'row18_condition=False', 'transformer_at_rating=True') | 496 | 175.88 | 6,752.27 |
| ('DSO7', 'P_for_Q_at_capability', 'row18_condition=True', 'transformer_at_rating=True') | 8 | 1.69 | 11.03 |
| ('DSO7', 'below_apparent_capability', 'row18_condition=True', 'transformer_at_rating=False') | 12 | 210.78 | 1,986.98 |
| ('DSO9', 'P_for_Q_at_capability', 'row18_condition=False', 'transformer_at_rating=False') | 1163 | 149.57 | 12,187.85 |
| ('DSO9', 'below_apparent_capability', 'row18_condition=True', 'transformer_at_rating=False') | 16 | 435.47 | 1,430.58 |

Hours with curtailment below the apparent capability (network, year, day, scenario, hour): ('DSO5', 2025, 'Spring', '0_1', 14); ('DSO5', 2034, 'Spring', '0_0', 14); ('DSO5', 2037, 'Spring', '0_1', 14); ('DSO5', 2037, 'Spring', '0_1', 16); ('DSO7', 2025, 'Spring', '0_0', 14); ('DSO7', 2037, 'Spring', '0_1', 14); ('DSO7', 2037, 'Spring', '0_1', 16); ('DSO9', 2025, 'Spring', '0_0', 14); ('DSO9', 2034, 'Spring', '0_1', 14); ('DSO9', 2037, 'Spring', '0_0', 14); ('DSO9', 2037, 'Spring', '0_0', 16)

## 7. Resolution and the recorded decision rule

Resolution used: the bar (record.bar, the stopping-slack error of Q). Justification: pricing curtailment at the hourly market price changes Q by at most C(x) at a global optimum (the current point stays feasible; the price is non-negative), so C(x) <= bar(x) means the re-baseline cannot move Q by more than its own stopping error; at the value level, |d value| <= max(C(0), C(unit)) rigorously and ~ C(0) - C(unit) to first order, both compared with bar(0) + bar(unit), the committed value resolution. sigma_Q is not used (it is a dispersion calibration, not an error of Q). A volume threshold is not used as the criterion (volumes are reported for the manuscript) because the decision concerns Q and the storage value. Reading: C <= bar establishes "below resolution"; C > bar only means the bound does not exclude a Q change above the bar (the re-optimised change can be far smaller than C), i.e. "not shown to be below resolution".

| instance | resolution | bar | C all networks | C reachable (TSO) | C/bar | C reachable/bar | material (all) | material (reachable) |
|---|---|---|---|---|---|---|---|---|
| srp1_x0 | per hour | 9,629.98 | 45,627.84 | 0.24 | 4.7381 | 0.000025 | True | False |
| pilot_x0 | per hour | 8,403.29 | 64,977.10 | 310.12 | 7.7323 | 0.036905 | True | False |
| pilot_unit | per hour | 7,107.55 | 64,645.59 | 5.20 | 9.0953 | 0.000732 | True | False |
| srp1_unit | block (upper bound: day net volume x the day max price) | 25,104.65 | 106,225.31 | 0.00 | 4.2313 | 0.000000 | True | False |

| pair | value resolution | C(0) | C(unit) | first-order dC | dC/resolution | rigorous bound max C | bound/resolution |
|---|---|---|---|---|---|---|---|
| srp1 | 34,734.63 | 45,627.84 | 106,225.31 | 45.76 | 0.00132 | 106,225.31 | 3.0582 |
| pilot | 15,510.84 | 64,977.10 | 64,645.59 | 331.51 | 0.02137 | 64,977.10 | 4.1891 |

**Branch selected (mechanical application of the recorded rule):** NEITHER BRANCH AS WORDED: reachable (TSO-side) curtailment is below resolution everywhere, but DN-side (unreachable) curtailment exceeds the Q bar in at least one instance -- Planner ruling

## 8. What the re-baseline branch would require (not implemented)

- Data parameter: a curtailment-price mode read from the case file (default = today: the constant 1.0 at build, 0 in the ADMM subproblems), and a mode pricing c at the scenario's hourly market price network.cost_energy_p[s_m][p] in BOTH networks, i.e. the weight inside the term at model_construction_helpers.py:2118 becomes per (s_m, p), and the ADMM resets at shared_resources_planning.py:4970 / shared_resources_planning.py:5151 keep it instead of zeroing it.
- SRP1 bitwise gate at the old value: the two-cycle SRP1 gate pattern (p515_s52_srp1_bitwise_gate.py): 153 solves declared, recorded wall 105 s.
- Re-baseline on SRP1 before anything else runs: x = 0 (recorded 3193 s, 132 cycles) and the smallest unit (recorded 3146 s, 112 cycles): ~1.8 h serial, ~0.9 h at concurrency 2; plus the pilot pair if the pilot is to be restated (recorded 3.76 h + 4.06 h at concurrency 2).
- The 3 x 3 selection is frozen only after the decision (Addendum 41).

## 9. Checks

| check | value |
|---|---|
| inputs_verified_against_committed_manifests | True |
| capture_path_checklist_all_true | True |
| srp1_x0_reconciles_with_component_levels | True |
| srp1_x0_prices_identical_across_dsos | True |
| srp1_x0_prices_match_w25 | True |
| srp1_unit_duplicate_record_equal | True |
| pilot_x0_reconciles_with_component_levels | True |
| pilot_unit_reconciles_with_component_levels | True |
| pilot_x0_sg_curt_matches_multiscenario | True |
| pilot_unit_sg_curt_matches_multiscenario | True |
| pilot_prices_non_negative | True |
| pilot_pibar_matches_production_row18_premium | True |
| srp1_prices_non_negative | True |
| penalty_zero_in_every_certified_block | True |
| no_model_construction_call | True |
| solve_profile_guard_verified_0 | True |
