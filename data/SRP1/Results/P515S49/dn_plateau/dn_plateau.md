# P5.15 Addendum 32 Q4 follow-up W31 -- DN row setting the bus-7 price plateau; throughput-weighted flatness split (zero solves)

Instance: x = 0, candidate key `8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57`, eval key `d2c96b1480402a3b61aca4abc188e41c6009eb582d8e6ccdd380e51651f996c7`; models `data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/certified_models.pkl` sha256 `03b62593a23f748c819f18dce52c88c6a9802b8af3c52033ae09a3b88d10afce` (verified; not committed). Task B: BASELINE record `data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1` (candidate key `db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a`, terminal cycle 112), ESSO capture sha256 `5c566b024c3aea56891e21b21a6db824d59800304c28b6c12dadc690b1ee1d68`. W28 JSON dd0a86b3 sha256 `0f7c8002e8d64d97f48036ca7d5bb706addad1776c24c473fc8530a10cd5e4a7`; W25 JSON b7aca555 sha256 `5acbd2b32d924b47f23650142076180c420bd0ba201b0a63d4d89f4b671b7a31`.

Solve profile: armed SolveProfileGuard(permitted=()); counts {'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}; verify(0) failures []. all_checks_pass = True; failing: [].

Objective convention: prices / duals in EUR/MWh (DN: raw dual / B_dn, the active objective being the physical per-representative-day cost on these variables; TSO: dual / baseMVA, W28); Task B in W25 T4 units: EUR per MWh of full cell cycle, Q(x) = gross_operational_cost settlement-EXCLUDED, block weights w_r2 = num_years * num_days / 1.02^(y - 2025) or w_undisc = num_years * num_days.

## Method (formulas)

-- the DN price chain (formulas; B_dn = DN baseMVA = 100, dt = 1 h; every dual is converted by /B_dn):
  The active DSO objective is p58_rescaled_admm_objective. Its gradient on the physical variables is the
  PHYSICAL per-representative-day cost (verified per block: df/d pg[ref] / B_dn = pi, the market price of the
  interface settlement, and df/d flex_p_down / B_dn = c_flex[p]); so a raw dual / B_dn is EUR/MWh with no
  sigma / block-weight factor (unlike the TSO blocks, where W25 needs sigma / (w_block ...) for the ADMM terms only).
  KKT convention (W28): r_v = df/dv - sum_c lambda_c dc/dv - zL_v - zU_v, lambda = model.dual, zL >= 0, zU <= 0.
  (1) interface (DN reference bus 1, node 0; pg[0] = import from the TN, free):
        y0_p := dual(node_balance_p[0,p]) / B_dn = pi_p + lambda(expected_interface_pf_p_def[p]) / B_dn
      (stationarity of pg[0]); the second term is the ADMM consensus price correction (stationarity of
      expected_interface_pf_p: lambda_E = gradient of the ADMM terms).
  (2) TSO side: LMP_n,p (TSO bus n of the DN) = dual(node_balance_p[bus n]) / B_tso (W28). At ADMM consensus
      y0_p = LMP_n,p; the gap is reported per hour (it is the consensus residual, not a modelling term).
  (3) DN network: y_i,p := dual(node_balance_p[i,p]) / B_dn at load bus i; the DN loss / voltage factor is
      y_i,p / y0_p (reported, not decomposed further).
  (4) Load flexibility of load l at bus i (pc_node = pc + flex_p_up - flex_p_down; daily row
      flex_energy_balance_p[l]: sum_p flex_p_up = sum_p flex_p_down (+ slacks)):
        mu_l := dual(flex_energy_balance_p[l]) / B_dn                (a DAILY constant per load)
        stationarity of flex_p_up[l,p]:   y_i,p = mu_l + z_up          (z = (zL + zU)/B of the variable)
        stationarity of flex_p_down[l,p]: y_i,p = mu_l + c_flex_l,p - z_down
      (signs NOT assumed: both identities are recomputed from the model with numeric derivatives and checked to
      KKT_TOL_EUR; a flex variable is MARGINAL when |z| <= Z_MARG_EUR, then y_i,p = its row price within Z_MARG).
  Plateau test (per c_interface hour of the TSO, classified by W28): the marginal rows' prices -- mu_l + c_flex_l,p
  (P-down) or mu_l (P-up), built from the flex_energy_balance_p dual and the objective coefficient, NOT from y_i --
  are referred to the interface with the DN factor (row price / (y_i/y0)) and compared with y0 and LMP_n,p.
  KKT bracket (every flex row, marginal or not): rows with y_i >= row price (P-up with z_up > 0, P-down with
  z_down < 0) give lower bounds, the others upper bounds; referred to the interface; y0 must lie inside.
  Alternatives tested in the same hours: (ii) DN-local storage (existence read from the DN case files and from the
  model index sets), (iii) DN renewable curtailment (DN generator whose bus price is within Z_MARG of its own
  objective gradient, 0 here -> it would set the price);
  otherwise 'bracketed by flex rows'.
  cost_flex relation (tested, not asserted): mu_l compared with y_i in the hours where flex_p_up[l] is marginal (the
  hours INTO which shifted energy is consumed at the margin) and with the day's lowest y_i.

TASK B -- throughput-weighted flatness split (W25 T4 baseline, r2 and undiscounted weightings):
  W25: flatness_tw = -(1/T) sum_b w_b T_b [S_pi,b - S_LMP0,b], T_b = sum_p (eff_ch pch_b,p + pdch_b,p / eff_dch) dt
  / 2 (baseline ESSO terminal capture, cycle 112, eff 0.97 / 0.96), T = sum_b w_b T_b, S = top-4 minus bottom-4
  (own hours), LMP0 = bus-7 LMP at x = 0. W28 per block: S_pi,b - S_LMP,b = H_b - sum_c D_c,b (hour selection and
  the components of LMP7 - pi: reference_generator_limit (the c_interface / generator-limit term), losses,
  congestion, voltage_bounds, voltage_setpoint, angle, admm_interface_voltage, other, residual). Hence
      flatness_tw = -(1/T) sum_b w_b T_b H_b + sum_c (1/T) sum_b w_b T_b D_c,b.
  Both W25's (b) and W28 use LMP7 at x = 0 (W25 from record a0_c7 x0 via its identity; W28 from the s48 x0_capture
  duals, a bitwise-identical-trajectory reproduction): the split is exact up to the difference of the two LMP series
  (<= 1.7e-6 EUR/MWh per hour, W28) -- the residual is reported.

## Checks

| check | value |
|---|---|
| pickle_sha256_matches | True |
| pickle_eval_is_x0 | True |
| w28_json_committed_unmodified | True |
| w25_json_committed_unmodified | True |
| baseline_record_candidate_key | True |
| baseline_record_cycles_112 | True |
| baseline_esso_capture_present | True |
| capture_path_checklist_all_true | True |
| dn_objective_gradient_pg0_equals_market_price | True |
| tso_lmp_matches_w28 | True |
| kkt_resid_below_tol_all_dn_blocks | True |
| interface_identity_y0_eq_pi_plus_lambdaE | True |
| flex_identities_y_eq_row_price_plus_multiplier_all_slots | True |
| y0_inside_flex_kkt_bracket_all_hours | True |
| consensus_gap_below_tol_in_c_interface_hours | True |
| flex_balance_slacks_negligible | True |
| no_dn_local_storage_in_models | True |
| shared_storage_zero_rated_at_x0 | True |
| task_b_r2_T_matches_w25 | True |
| task_b_r2_w25_series_repro | True |
| task_b_r2_split_residual_below_tol | True |
| task_b_r2_split_sums_to_direct | True |
| task_b_undiscounted_T_matches_w25 | True |
| task_b_undiscounted_w25_series_repro | True |
| task_b_undiscounted_split_residual_below_tol | True |
| task_b_undiscounted_split_sums_to_direct | True |
| task_b_same_top_bottom_hours_all_blocks | True |
| solve_profile_guard_verified_0 | True |

## Task A -- DN-local storage (ii)

DN case files (energy_storages key read by network.py): case33_1_2025: 0; case33_1_2030: 0; case33_1_2035: 0; case33_2_2025: 0; case33_2_2030: 0; case33_2_2035: 0; case33_3_2025: 0; case33_3_2030: 0; case33_3_2035: 0.
Model index sets (2030 Spring): dso5: es_soc size 0, shared_es E [0.0] MWh, S [0.0] MVA, max |sess_soc_def dual| 0.00e+00 EUR/MWh; dso7: es_soc size 0, shared_es E [0.0] MWh, S [0.0] MVA, max |sess_soc_def dual| 0.00e+00 EUR/MWh; dso9: es_soc size 0, shared_es E [0.0] MWh, S [0.0] MVA, max |sess_soc_def dual| 0.00e+00 EUR/MWh.

## Task A -- mechanism per hour, all blocks (counts of hours)

| DN | year | day | c_interface hours | P-down marginal (mu+c) | P-up marginal (mu) | DN RES marginal (0) | bracketed only | max bracket width, bracketed-only hours | max |consensus gap| c_int | max flex identity dev incl. z | max KKT resid (EUR/MWh) | slack flex bal (pu) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 2025 | Spring | 7 | 6 | 0 | 0 | 1 |  | 0.0005 | 5.68e-14 | 2.84e-10 | 9.9e-09 |
| 5 | 2025 | Summer | 3 | 2 | 0 | 0 | 1 |  | 0.0003 | 5.68e-14 | 7.57e-10 | 9.9e-09 |
| 5 | 2025 | Autumn | 0 | 0 | 0 | 0 | 0 |  | 0.0000 | 5.68e-14 | 4.64e-11 | 9.9e-09 |
| 5 | 2025 | Winter | 3 | 3 | 0 | 0 | 0 |  | 0.0002 | 5.68e-14 | 7.23e-14 | 9.9e-09 |
| 5 | 2030 | Spring | 14 | 13 | 0 | 0 | 1 |  | 0.0018 | 4.26e-14 | 4.33e-09 | 9.9e-09 |
| 5 | 2030 | Summer | 12 | 11 | 1 | 0 | 0 |  | 0.0013 | 5.68e-14 | 9.33e-11 | 9.9e-09 |
| 5 | 2030 | Autumn | 9 | 1 | 8 | 0 | 0 |  | 0.0000 | 5.68e-14 | 7.49e-10 | 9.9e-09 |
| 5 | 2030 | Winter | 3 | 3 | 0 | 0 | 0 |  | 0.0019 | 5.68e-14 | 2.62e-13 | 9.9e-09 |
| 5 | 2035 | Spring | 14 | 12 | 0 | 0 | 2 | 33.210 | 0.0024 | 5.68e-14 | 3.05e-12 | 9.9e-09 |
| 5 | 2035 | Summer | 11 | 8 | 1 | 0 | 2 |  | 0.0015 | 5.68e-14 | 2.53e-13 | 9.9e-09 |
| 5 | 2035 | Autumn | 7 | 3 | 4 | 0 | 0 |  | 0.0005 | 5.68e-14 | 2.40e-12 | 9.9e-09 |
| 5 | 2035 | Winter | 7 | 2 | 5 | 0 | 0 |  | 0.0000 | 5.68e-14 | 5.00e-14 | 9.9e-09 |
| 7 | 2025 | Spring | 7 | 7 | 0 | 0 | 0 |  | 0.0048 | 5.68e-14 | 1.41e-09 | 9.9e-09 |
| 7 | 2025 | Summer | 3 | 3 | 0 | 0 | 0 |  | 0.0016 | 4.26e-14 | 1.37e-11 | 9.9e-09 |
| 7 | 2025 | Autumn | 0 | 0 | 0 | 0 | 0 |  | 0.0000 | 5.68e-14 | 7.83e-13 | 9.9e-09 |
| 7 | 2025 | Winter | 3 | 3 | 0 | 0 | 0 |  | 0.0023 | 5.68e-14 | 1.33e-13 | 9.9e-09 |
| 7 | 2030 | Spring | 14 | 14 | 0 | 0 | 0 |  | 0.0152 | 2.84e-14 | 3.43e-09 | 9.9e-09 |
| 7 | 2030 | Summer | 12 | 11 | 0 | 0 | 1 | 44.293 | 0.0102 | 5.68e-14 | 1.29e-11 | 9.9e-09 |
| 7 | 2030 | Autumn | 9 | 1 | 8 | 0 | 0 |  | 0.0000 | 7.11e-14 | 4.18e-13 | 9.9e-09 |
| 7 | 2030 | Winter | 3 | 3 | 0 | 0 | 0 |  | 0.0150 | 5.68e-14 | 6.58e-14 | 9.9e-09 |
| 7 | 2035 | Spring | 14 | 13 | 0 | 0 | 1 | 33.646 | 0.0155 | 5.68e-14 | 1.16e-11 | 9.9e-09 |
| 7 | 2035 | Summer | 11 | 10 | 0 | 0 | 1 | 55.543 | 0.0089 | 8.53e-14 | 6.57e-09 | 9.9e-09 |
| 7 | 2035 | Autumn | 7 | 3 | 3 | 0 | 1 | 54.768 | 0.0045 | 5.68e-14 | 5.18e-14 | 9.9e-09 |
| 7 | 2035 | Winter | 7 | 2 | 4 | 0 | 1 | 23.513 | 0.0004 | 5.68e-14 | 7.28e-14 | 9.9e-09 |
| 9 | 2025 | Spring | 7 | 6 | 0 | 0 | 1 |  | 0.0012 | 5.68e-14 | 4.12e-10 | 9.9e-09 |
| 9 | 2025 | Summer | 3 | 2 | 0 | 0 | 1 |  | 0.0003 | 4.26e-14 | 1.39e-10 | 9.9e-09 |
| 9 | 2025 | Autumn | 0 | 0 | 0 | 0 | 0 |  | 0.0000 | 5.68e-14 | 9.60e-13 | 9.9e-09 |
| 9 | 2025 | Winter | 3 | 3 | 0 | 0 | 0 |  | 0.0006 | 5.68e-14 | 3.32e-07 | 9.9e-09 |
| 9 | 2030 | Spring | 14 | 13 | 0 | 0 | 1 |  | 0.0038 | 4.26e-14 | 2.67e-09 | 9.9e-09 |
| 9 | 2030 | Summer | 12 | 11 | 1 | 0 | 0 |  | 0.0023 | 5.68e-14 | 4.00e-12 | 9.9e-09 |
| 9 | 2030 | Autumn | 9 | 1 | 7 | 0 | 1 | 50.806 | 0.0000 | 5.68e-14 | 4.25e-12 | 9.9e-09 |
| 9 | 2030 | Winter | 3 | 3 | 0 | 0 | 0 |  | 0.0033 | 5.68e-14 | 4.24e-12 | 9.9e-09 |
| 9 | 2035 | Spring | 14 | 12 | 1 | 0 | 1 |  | 0.0032 | 5.68e-14 | 1.45e-11 | 9.9e-09 |
| 9 | 2035 | Summer | 11 | 9 | 1 | 0 | 1 |  | 0.0026 | 5.68e-14 | 1.81e-10 | 9.9e-09 |
| 9 | 2035 | Autumn | 7 | 3 | 4 | 0 | 0 |  | 0.0011 | 4.26e-14 | 1.43e-11 | 9.9e-09 |
| 9 | 2035 | Winter | 7 | 2 | 5 | 0 | 0 |  | 0.0001 | 7.11e-14 | 1.06e-12 | 9.9e-09 |

### DN 5 (case33_1) 2030 Spring: hour by hour

| h | TSO setter | pi | LMP TSO bus | y0 DN ref | gap | lambda_E/B | mechanism | #P-down marginal | #P-down at ub (primal) | #P-up marginal | #P-up at ub (primal) | mu+c of marginal P-down (min..max) | mu of marginal P-up (min..max) | marginal rows referred to interface (mean) | LMP - referred | bracket [lo, hi] referred | nearest row distance | c_flex mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | c_interface | 132.01 | 100.755 | 100.755 | +0.0005 | -31.25 | flex_p_down_marginal | 28 | 6 | 0 | 0 | 100.85..103.38 | .. | 100.739 | +0.0161 | [100.755, ] | 0.0000 | 54.14 |
| 2 | c_interface | 133.20 | 98.537 | 98.537 | +0.0001 | -34.66 | flex_p_down_marginal | 31 | 2 | 0 | 0 | 98.60..101.18 | .. | 98.520 | +0.0172 | [98.537, ] | 0.0000 | 51.94 |
| 3 | c_interface | 135.99 | 99.303 | 99.304 | -0.0003 | -36.69 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 99.23..101.97 | .. | 99.296 | +0.0069 | [99.304, 99.304] | 0.0000 | 52.73 |
| 4 | c_interface | 134.90 | 97.989 | 97.990 | -0.0005 | -36.91 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 97.93..100.67 | .. | 97.987 | +0.0026 | [97.990, 97.990] | 0.0000 | 51.43 |
| 5 | c_interface | 141.12 | 99.637 | 99.638 | -0.0005 | -41.48 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 99.60..102.34 | .. | 99.637 | +0.0002 | [99.638, 99.638] | 0.0000 | 53.10 |
| 6 | c_interface | 144.86 | 99.129 | 99.130 | -0.0006 | -45.73 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 99.10..101.83 | .. | 99.129 | +0.0000 | [99.130, 99.130] | 0.0000 | 52.59 |
| 7 | c_interface | 152.60 | 101.541 | 101.541 | -0.0007 | -51.06 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 101.52..104.26 | .. | 101.542 | -0.0010 | [101.541, 101.541] | 0.0000 | 55.02 |
| 8 | c_interface | 131.21 | 98.869 | 98.869 | -0.0005 | -32.34 | flex_p_down_marginal | 29 | 0 | 0 | 0 | 98.88..101.61 | .. | 98.872 | -0.0036 | [98.869, 98.869] | 0.0000 | 52.37 |
| 9 | a_conv_interior | 88.06 | 88.345 | 88.346 | -0.0015 | 0.28 | flex_p_down_marginal | 19 | 0 | 0 | 0 | 89.06..90.73 | .. | 88.356 | -0.0108 | [88.346, 88.346] | 0.0000 | 42.05 |
| 10 | a_conv_interior | 43.13 | 44.006 | 44.006 | -0.0006 | 0.87 | flex_p_up_marginal | 0 | 0 | 4 | 28 | .. | 48.32..48.50 | 44.020 | -0.0146 | [44.006, 44.063] | 0.0000 | 33.04 |
| 11 | a_conv_interior | 38.03 | 38.734 | 38.734 | -0.0000 | 0.71 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 43.434] | 5.2486 | 30.10 |
| 12 | a_conv_interior | 34.89 | 35.532 | 35.532 | -0.0000 | 0.64 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 43.454] | 8.8410 | 29.58 |
| 13 | a_conv_interior | 17.75 | 18.082 | 18.082 | -0.0000 | 0.33 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 43.384] | 27.9363 | 28.77 |
| 14 | a_conv_interior | 14.37 | 14.618 | 14.618 | -0.0000 | 0.25 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 43.660] | 31.5366 | 26.59 |
| 15 | a_conv_interior | 3.69 | 3.741 | 3.741 | -0.0000 | 0.05 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 44.198] | 42.7070 | 25.17 |
| 16 | a_conv_interior | 7.90 | 8.004 | 8.004 | -0.0000 | 0.10 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 44.641] | 38.3903 | 25.03 |
| 17 | a_conv_interior | 15.39 | 15.606 | 15.606 | -0.0000 | 0.22 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 44.546] | 30.5538 | 27.06 |
| 18 | a_conv_interior | 46.97 | 47.216 | 47.216 | -0.0000 | 0.24 | flex_p_up_marginal | 0 | 0 | 1 | 0 | .. | 49.21..49.21 | 47.149 | +0.0669 | [47.149, 76.256] | 0.0698 | 31.37 |
| 19 | c_interface | 90.75 | 86.233 | 86.234 | -0.0012 | -4.52 | flex_p_down_marginal | 29 | 0 | 0 | 0 | 86.41..89.14 | .. | 86.245 | -0.0123 | [86.234, 86.234] | 0.0000 | 39.90 |
| 20 | c_interface | 128.94 | 101.034 | 101.033 | +0.0005 | -27.91 | flex_p_down_marginal | 23 | 9 | 0 | 0 | 101.10..103.09 | .. | 101.030 | +0.0040 | [101.033, 101.033] | 0.0000 | 54.60 |
| 21 | c_interface | 161.27 | 106.973 | 106.971 | +0.0018 | -54.30 | flex_p_down_marginal | 22 | 10 | 0 | 0 | 107.01..109.00 | .. | 106.965 | +0.0077 | [106.971, 106.971] | 0.0000 | 60.50 |
| 22 | c_interface | 152.21 | 101.485 | 101.483 | +0.0017 | -50.73 | flex_p_down_marginal | 27 | 6 | 0 | 0 | 101.50..104.08 | .. | 101.475 | +0.0094 | [101.483, 101.483] | 0.0000 | 54.99 |
| 23 | c_interface | 141.99 | 100.873 | 100.873 | -0.0002 | -41.12 | bracketed by flex rows | 0 | 32 | 0 | 0 | .. | .. |  |  | [100.333, ] | 0.5418 | 53.87 |
| 24 | c_interface | 178.86 | 99.099 | 99.098 | +0.0017 | -79.76 | flex_p_down_marginal | 19 | 14 | 0 | 0 | 99.49..101.06 | .. | 99.066 | +0.0329 | [99.098, 99.098] | 0.0000 | 52.48 |

mu_l (daily shadow value of shifted energy, EUR/MWh): min 46.51, max 49.24, mean 47.92. cost_flex relation: flex_p_up marginal in 5 load-hours (hours [10, 18]); there y_i - mu_l max abs 6.98e-02 EUR/MWh; mu_l - day-min y_i in [42.71, 45.20]; flex_p_down marginal in hours [1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 20, 21, 22, 24].

### DN 5 (case33_1) 2030 Summer: hour by hour

| h | TSO setter | pi | LMP TSO bus | y0 DN ref | gap | lambda_E/B | mechanism | #P-down marginal | #P-down at ub (primal) | #P-up marginal | #P-up at ub (primal) | mu+c of marginal P-down (min..max) | mu of marginal P-up (min..max) | marginal rows referred to interface (mean) | LMP - referred | bracket [lo, hi] referred | nearest row distance | c_flex mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | c_interface | 157.23 | 129.337 | 129.337 | -0.0000 | -27.89 | flex_p_down_marginal | 14 | 18 | 0 | 0 | 131.11..133.04 | .. | 129.313 | +0.0236 | [129.337, ] | 0.0000 | 53.91 |
| 2 | c_interface | 146.06 | 128.652 | 128.652 | -0.0001 | -17.41 | flex_p_down_marginal | 14 | 18 | 0 | 0 | 130.44..132.37 | .. | 128.639 | +0.0132 | [128.652, ] | 0.0000 | 53.24 |
| 3 | c_interface | 140.13 | 127.930 | 127.930 | -0.0001 | -12.20 | flex_p_down_marginal | 20 | 12 | 0 | 0 | 127.88..131.73 | .. | 127.921 | +0.0092 | [127.930, 127.930] | 0.0000 | 52.60 |
| 4 | c_interface | 136.48 | 125.855 | 125.855 | -0.0001 | -10.62 | flex_p_down_marginal | 22 | 10 | 0 | 0 | 125.86..129.70 | .. | 125.846 | +0.0090 | [125.855, 125.855] | 0.0000 | 50.57 |
| 5 | c_interface | 137.07 | 125.859 | 125.859 | -0.0001 | -11.21 | flex_p_down_marginal | 22 | 10 | 0 | 0 | 125.92..129.77 | .. | 125.856 | +0.0033 | [125.859, 125.859] | 0.0000 | 50.64 |
| 6 | a_conv_interior | 124.61 | 125.341 | 125.341 | -0.0003 | 0.73 | flex_p_down_marginal | 28 | 2 | 0 | 0 | 126.19..130.03 | .. | 125.343 | -0.0020 | [125.341, 125.341] | 0.0000 | 50.90 |
| 7 | a_conv_interior | 126.80 | 127.840 | 127.842 | -0.0015 | 1.04 | flex_p_down_marginal | 22 | 0 | 0 | 0 | 129.87..133.72 | .. | 127.843 | -0.0023 | [127.842, 127.842] | 0.0000 | 54.59 |
| 8 | c_interface | 137.40 | 131.722 | 131.722 | -0.0005 | -5.68 | flex_p_down_marginal | 22 | 4 | 0 | 0 | 131.97..136.02 | .. | 131.722 | -0.0000 | [131.722, 131.722] | 0.0000 | 56.94 |
| 9 | a_conv_interior | 100.57 | 101.239 | 101.239 | -0.0000 | 0.67 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [76.485, 125.947] | 24.3648 | 54.56 |
| 10 | c_interface | 81.90 | 77.438 | 77.438 | -0.0000 | -4.46 | flex_p_up_marginal | 0 | 0 | 2 | 0 | .. | 75.28..79.12 | 77.436 | +0.0025 | [77.438, 120.586] | 0.0000 | 48.46 |
| 11 | a_conv_interior | 70.69 | 71.684 | 71.685 | -0.0008 | 0.99 | flex_p_up_marginal | 0 | 0 | 4 | 28 | .. | 77.36..77.61 | 71.634 | +0.0505 | [71.623, 71.685] | 0.0001 | 43.95 |
| 12 | a_conv_interior | 63.31 | 64.257 | 64.257 | -0.0000 | 0.94 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 69.958] | 6.3247 | 40.03 |
| 13 | a_conv_interior | 54.66 | 55.485 | 55.485 | -0.0000 | 0.83 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 69.847] | 15.9589 | 41.52 |
| 14 | a_conv_interior | 47.25 | 47.927 | 47.927 | -0.0000 | 0.67 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 70.158] | 24.5922 | 39.02 |
| 15 | a_conv_interior | 44.44 | 44.984 | 44.984 | -0.0000 | 0.54 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 70.918] | 28.3811 | 38.19 |
| 16 | a_conv_interior | 42.76 | 43.231 | 43.231 | -0.0000 | 0.47 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 71.541] | 30.7105 | 39.66 |
| 17 | a_conv_interior | 66.32 | 67.151 | 67.151 | -0.0000 | 0.83 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 71.495] | 4.7157 | 45.43 |
| 18 | a_conv_interior | 93.30 | 93.836 | 93.836 | -0.0000 | 0.54 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [76.131, 123.154] | 18.1428 | 50.96 |
| 19 | a_conv_interior | 109.73 | 110.651 | 110.651 | -0.0000 | 0.92 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [75.052, 132.586] | 22.9342 | 61.01 |
| 20 | c_interface | 146.52 | 144.444 | 144.443 | +0.0011 | -2.08 | flex_p_down_marginal | 23 | 9 | 0 | 0 | 144.68..147.25 | .. | 144.440 | +0.0040 | [144.443, 144.443] | 0.0000 | 69.64 |
| 21 | c_interface | 183.24 | 144.100 | 144.100 | +0.0006 | -39.14 | flex_p_down_marginal | 23 | 9 | 0 | 0 | 144.38..146.96 | .. | 144.098 | +0.0021 | [144.100, 144.100] | 0.0000 | 69.35 |
| 22 | c_interface | 165.71 | 136.879 | 136.879 | -0.0001 | -28.83 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 137.14..141.24 | .. | 136.876 | +0.0027 | [136.879, 136.879] | 0.0000 | 62.11 |
| 23 | c_interface | 158.90 | 133.651 | 133.650 | +0.0013 | -25.25 | flex_p_down_marginal | 23 | 10 | 0 | 0 | 133.84..136.95 | .. | 133.629 | +0.0220 | [133.650, 133.650] | 0.0000 | 58.81 |
| 24 | c_interface | 139.14 | 131.624 | 131.623 | +0.0005 | -7.52 | flex_p_down_marginal | 30 | 2 | 0 | 0 | 131.83..135.92 | .. | 131.610 | +0.0138 | [131.623, 131.623] | 0.0000 | 56.79 |

mu_l (daily shadow value of shifted energy, EUR/MWh): min 75.03, max 79.13, mean 76.93. cost_flex relation: flex_p_up marginal in 6 load-hours (hours [10, 11]); there y_i - mu_l max abs 7.87e-02 EUR/MWh; mu_l - day-min y_i in [30.71, 33.40]; flex_p_down marginal in hours [1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23, 24].

### DN 5 (case33_1) 2030 Autumn: hour by hour

| h | TSO setter | pi | LMP TSO bus | y0 DN ref | gap | lambda_E/B | mechanism | #P-down marginal | #P-down at ub (primal) | #P-up marginal | #P-up at ub (primal) | mu+c of marginal P-down (min..max) | mu of marginal P-up (min..max) | marginal rows referred to interface (mean) | LMP - referred | bracket [lo, hi] referred | nearest row distance | c_flex mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | c_interface | 101.83 | 87.961 | 87.961 | -0.0000 | -13.87 | flex_p_up_marginal | 0 | 0 | 15 | 0 | .. | 89.84..91.74 | 87.947 | +0.0145 | [87.961, 139.722] | 0.0001 | 54.80 |
| 2 | a_conv_interior | 99.12 | 99.414 | 99.414 | -0.0000 | 0.30 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.968, 140.116] | 11.7812 | 55.23 |
| 3 | c_interface | 98.48 | 87.880 | 87.880 | -0.0000 | -10.60 | flex_p_up_marginal | 0 | 0 | 15 | 0 | .. | 89.84..91.74 | 87.870 | +0.0104 | [87.880, 137.563] | 0.0000 | 52.37 |
| 4 | a_conv_interior | 92.15 | 92.384 | 92.384 | -0.0000 | 0.23 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [88.196, 136.200] | 4.3260 | 50.95 |
| 5 | c_interface | 93.28 | 87.872 | 87.872 | -0.0000 | -5.41 | flex_p_up_marginal | 0 | 0 | 15 | 0 | .. | 89.84..91.74 | 87.860 | +0.0122 | [87.872, 134.091] | 0.0000 | 48.62 |
| 6 | c_interface | 98.40 | 87.836 | 87.836 | +0.0000 | -10.57 | flex_p_up_marginal | 0 | 0 | 15 | 0 | .. | 89.84..91.74 | 87.826 | +0.0100 | [87.836, 134.158] | 0.0000 | 48.73 |
| 7 | a_conv_interior | 113.26 | 113.970 | 113.970 | -0.0000 | 0.71 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.435, 139.224] | 26.6918 | 55.56 |
| 8 | a_conv_interior | 131.95 | 132.610 | 132.610 | -0.0000 | 0.65 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.458, 146.462] | 14.6142 | 62.93 |
| 9 | a_conv_interior | 133.52 | 134.115 | 134.115 | -0.0000 | 0.59 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.421, 150.715] | 17.4639 | 66.97 |
| 10 | c_interface | 127.85 | 87.381 | 87.381 | +0.0000 | -40.47 | flex_p_up_marginal | 0 | 0 | 21 | 0 | .. | 87.92..91.74 | 87.355 | +0.0258 | [87.381, 147.448] | 0.0000 | 63.13 |
| 11 | c_interface | 109.70 | 87.300 | 87.300 | -0.0000 | -22.40 | flex_p_up_marginal | 0 | 0 | 24 | 0 | .. | 87.92..91.74 | 87.286 | +0.0141 | [87.300, 87.300] | 0.0000 | 57.52 |
| 12 | c_interface | 101.19 | 87.387 | 87.387 | -0.0000 | -13.80 | flex_p_up_marginal | 0 | 0 | 16 | 0 | .. | 87.92..91.74 | 87.359 | +0.0282 | [87.387, 87.387] | 0.0000 | 50.47 |
| 13 | c_interface | 100.22 | 87.384 | 87.384 | -0.0000 | -12.84 | flex_p_up_marginal | 0 | 0 | 16 | 0 | .. | 87.92..91.74 | 87.357 | +0.0272 | [87.384, 87.384] | 0.0000 | 54.26 |
| 14 | a_conv_interior | 98.48 | 98.900 | 98.900 | -0.0000 | 0.42 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.846, 136.619] | 11.4214 | 51.79 |
| 15 | a_conv_interior | 101.26 | 101.784 | 101.784 | -0.0000 | 0.53 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.577, 135.880] | 14.5836 | 50.90 |
| 16 | a_conv_interior | 119.14 | 119.981 | 119.981 | -0.0000 | 0.84 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.417, 139.594] | 20.6146 | 55.13 |
| 17 | a_conv_interior | 133.04 | 134.168 | 134.168 | -0.0000 | 1.13 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.436, 146.153] | 12.6698 | 62.90 |
| 18 | a_conv_interior | 150.16 | 151.722 | 151.722 | -0.0000 | 1.56 | flex_p_down_marginal | 1 | 0 | 0 | 0 | 161.86..161.86 | .. | 151.724 | -0.0022 | [87.281, 151.724] | 0.0024 | 70.26 |
| 19 | c_interface | 175.07 | 170.699 | 170.699 | -0.0000 | -4.37 | flex_p_down_marginal | 22 | 10 | 0 | 0 | 171.23..175.09 | .. | 170.714 | -0.0148 | [170.699, 170.699] | 0.0000 | 83.98 |
| 20 | a_conv_interior | 166.66 | 167.893 | 167.893 | -0.0000 | 1.23 | flex_p_down_marginal | 5 | 2 | 0 | 0 | 175.42..176.07 | .. | 167.895 | -0.0018 | [167.857, 167.894] | 0.0000 | 84.33 |
| 21 | a_conv_interior | 143.07 | 144.274 | 144.274 | -0.0000 | 1.21 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.412, 159.578] | 16.2164 | 77.50 |
| 22 | a_conv_interior | 135.89 | 136.812 | 136.812 | -0.0000 | 0.93 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.571, 151.744] | 15.7472 | 68.43 |
| 23 | a_conv_interior | 132.16 | 133.264 | 133.264 | -0.0000 | 1.10 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.451, 146.737] | 14.2406 | 63.51 |
| 24 | a_conv_interior | 121.88 | 122.848 | 122.848 | -0.0000 | 0.97 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.501, 144.341] | 22.6796 | 60.72 |

mu_l (daily shadow value of shifted energy, EUR/MWh): min 87.24, max 91.74, mean 89.80. cost_flex relation: flex_p_up marginal in 137 load-hours (hours [1, 3, 5, 6, 10, 11, 12, 13]); there y_i - mu_l max abs 9.97e-02 EUR/MWh; mu_l - day-min y_i in [-0.36, 0.00]; flex_p_down marginal in hours [18, 19, 20].

### DN 5 (case33_1) 2030 Winter: hour by hour

| h | TSO setter | pi | LMP TSO bus | y0 DN ref | gap | lambda_E/B | mechanism | #P-down marginal | #P-down at ub (primal) | #P-up marginal | #P-up at ub (primal) | mu+c of marginal P-down (min..max) | mu of marginal P-up (min..max) | marginal rows referred to interface (mean) | LMP - referred | bracket [lo, hi] referred | nearest row distance | c_flex mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a_conv_interior | 108.72 | 109.302 | 109.302 | -0.0000 | 0.58 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [95.420, 133.288] | 14.4035 | 40.19 |
| 2 | a_conv_interior | 101.09 | 101.576 | 101.576 | -0.0000 | 0.48 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [95.488, 131.963] | 6.3075 | 38.86 |
| 3 | a_conv_interior | 96.26 | 96.589 | 96.589 | -0.0000 | 0.33 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [95.818, 131.258] | 0.7962 | 37.79 |
| 4 | a_conv_interior | 93.37 | 94.180 | 94.180 | +0.0004 | 0.81 | flex_p_up_marginal | 0 | 0 | 31 | 0 | .. | 94.60..99.84 | 94.178 | +0.0027 | [94.180, 94.180] | 0.0000 | 39.15 |
| 5 | a_conv_interior | 93.17 | 94.102 | 94.101 | +0.0008 | 0.93 | flex_p_up_marginal | 0 | 0 | 31 | 0 | .. | 94.60..99.84 | 94.101 | +0.0005 | [94.101, 94.101] | 0.0000 | 37.44 |
| 6 | a_conv_interior | 95.12 | 95.465 | 95.465 | +0.0002 | 0.35 | flex_p_up_marginal | 0 | 0 | 10 | 0 | .. | 98.48..99.84 | 95.448 | +0.0167 | [95.465, 132.033] | 0.0000 | 38.46 |
| 7 | a_conv_interior | 98.94 | 99.661 | 99.661 | -0.0000 | 0.73 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.556, 133.821] | 5.3416 | 41.90 |
| 8 | a_conv_interior | 120.55 | 121.474 | 121.474 | -0.0000 | 0.93 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.493, 134.678] | 13.9988 | 43.01 |
| 9 | a_conv_interior | 130.75 | 131.815 | 131.815 | +0.0001 | 1.07 | flex_p_down_marginal | 7 | 0 | 0 | 0 | 138.39..139.29 | .. | 131.843 | -0.0287 | [131.814, 131.815] | 0.0000 | 39.45 |
| 10 | a_conv_interior | 125.15 | 125.920 | 125.920 | -0.0000 | 0.77 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [95.130, 128.343] | 2.5406 | 35.65 |
| 11 | a_conv_interior | 113.40 | 113.929 | 113.929 | -0.0000 | 0.53 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [95.914, 124.064] | 10.6014 | 30.84 |
| 12 | a_conv_interior | 96.71 | 97.294 | 97.294 | -0.0000 | 0.58 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [96.079, 122.364] | 1.2163 | 29.42 |
| 13 | a_conv_interior | 94.24 | 95.048 | 95.048 | +0.0002 | 0.81 | flex_p_up_marginal | 0 | 0 | 3 | 0 | .. | 96.16..99.84 | 95.030 | +0.0181 | [95.048, 95.048] | 0.0000 | 25.76 |
| 14 | a_conv_interior | 92.37 | 93.547 | 93.547 | +0.0004 | 1.18 | flex_p_up_marginal | 0 | 0 | 12 | 7 | .. | 95.56..99.77 | 93.545 | +0.0017 | [93.546, 93.547] | 0.0000 | 25.53 |
| 15 | a_conv_interior | 94.97 | 95.616 | 95.616 | +0.0001 | 0.65 | flex_p_up_marginal | 0 | 0 | 3 | 0 | .. | 96.16..99.84 | 95.632 | -0.0153 | [95.616, 95.662] | 0.0000 | 23.90 |
| 16 | a_conv_interior | 98.28 | 98.960 | 98.960 | -0.0000 | 0.68 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [95.681, 118.200] | 3.3443 | 24.57 |
| 17 | a_conv_interior | 116.32 | 117.337 | 117.337 | -0.0000 | 1.01 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.569, 121.646] | 4.5436 | 28.64 |
| 18 | c_interface | 146.55 | 128.427 | 128.429 | -0.0019 | -18.12 | flex_p_down_marginal | 26 | 6 | 0 | 0 | 129.26..133.54 | .. | 128.424 | +0.0035 | [128.429, 128.429] | 0.0000 | 33.71 |
| 19 | c_interface | 161.38 | 136.123 | 136.123 | -0.0002 | -25.26 | flex_p_down_marginal | 25 | 7 | 0 | 0 | 137.16..141.31 | .. | 136.116 | +0.0065 | [136.123, 136.123] | 0.0000 | 41.47 |
| 20 | c_interface | 160.53 | 140.811 | 140.811 | +0.0000 | -19.72 | flex_p_down_marginal | 30 | 3 | 0 | 0 | 141.58..146.19 | .. | 140.813 | -0.0019 | [140.811, 140.811] | 0.0000 | 46.36 |
| 21 | a_conv_interior | 138.34 | 139.484 | 139.484 | -0.0000 | 1.14 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.584, 141.653] | 2.2927 | 49.93 |
| 22 | a_conv_interior | 128.45 | 129.436 | 129.436 | -0.0000 | 0.98 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.874, 135.031] | 5.8941 | 42.47 |
| 23 | a_conv_interior | 118.37 | 119.454 | 119.454 | -0.0000 | 1.09 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.502, 135.595] | 16.9996 | 43.56 |
| 24 | a_conv_interior | 111.97 | 112.974 | 112.974 | -0.0000 | 1.01 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.591, 135.155] | 18.7672 | 42.95 |

mu_l (daily shadow value of shifted energy, EUR/MWh): min 94.60, max 99.84, mean 97.44. cost_flex relation: flex_p_up marginal in 90 load-hours (hours [4, 5, 6, 13, 14, 15]); there y_i - mu_l max abs 9.42e-02 EUR/MWh; mu_l - day-min y_i in [-0.00, 0.49]; flex_p_down marginal in hours [9, 18, 19, 20].

### DN 7 (case33_2) 2030 Spring: hour by hour

| h | TSO setter | pi | LMP TSO bus | y0 DN ref | gap | lambda_E/B | mechanism | #P-down marginal | #P-down at ub (primal) | #P-up marginal | #P-up at ub (primal) | mu+c of marginal P-down (min..max) | mu of marginal P-up (min..max) | marginal rows referred to interface (mean) | LMP - referred | bracket [lo, hi] referred | nearest row distance | c_flex mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | c_interface | 132.01 | 100.032 | 100.035 | -0.0034 | -31.97 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 99.79..102.90 | .. | 100.022 | +0.0100 | [100.035, 100.035] | 0.0000 | 54.14 |
| 2 | c_interface | 133.20 | 97.859 | 97.860 | -0.0015 | -35.34 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 97.59..100.69 | .. | 97.845 | +0.0136 | [97.860, 97.860] | 0.0000 | 51.94 |
| 3 | c_interface | 135.99 | 98.643 | 98.642 | +0.0013 | -37.35 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 98.38..101.48 | .. | 98.636 | +0.0068 | [98.642, 98.642] | 0.0000 | 52.73 |
| 4 | c_interface | 134.90 | 97.350 | 97.347 | +0.0028 | -37.55 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 97.08..100.18 | .. | 97.342 | +0.0076 | [97.347, 97.347] | 0.0000 | 51.43 |
| 5 | c_interface | 141.12 | 98.998 | 98.995 | +0.0030 | -42.13 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 98.75..101.85 | .. | 98.992 | +0.0057 | [98.995, 98.995] | 0.0000 | 53.10 |
| 6 | c_interface | 144.86 | 98.492 | 98.488 | +0.0034 | -46.37 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 98.24..101.34 | .. | 98.486 | +0.0059 | [98.488, 98.488] | 0.0000 | 52.59 |
| 7 | c_interface | 152.60 | 100.883 | 100.878 | +0.0056 | -51.73 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 100.67..103.77 | .. | 100.874 | +0.0093 | [100.878, 100.878] | 0.0000 | 55.02 |
| 8 | c_interface | 131.21 | 98.177 | 98.169 | +0.0087 | -33.04 | flex_p_down_marginal | 31 | 0 | 0 | 0 | 98.02..101.12 | .. | 98.167 | +0.0108 | [98.169, 98.169] | 0.0000 | 52.37 |
| 9 | a_conv_interior | 88.06 | 87.790 | 87.774 | +0.0158 | -0.29 | flex_p_down_marginal | 27 | 2 | 0 | 0 | 87.87..90.80 | .. | 87.768 | +0.0220 | [87.774, 87.774] | 0.0000 | 42.05 |
| 10 | a_conv_interior | 43.13 | 43.492 | 43.492 | -0.0000 | 0.36 | flex_p_up_marginal | 0 | 0 | 23 | 9 | .. | 45.75..48.75 | 43.502 | -0.0100 | [43.492, 43.492] | 0.0000 | 33.04 |
| 11 | a_conv_interior | 38.03 | 38.341 | 38.341 | -0.0000 | 0.32 | flex_p_up_marginal | 0 | 0 | 23 | 9 | .. | 45.75..48.75 | 38.353 | -0.0116 | [38.341, 38.341] | 0.0000 | 30.10 |
| 12 | a_conv_interior | 34.89 | 35.181 | 35.181 | -0.0000 | 0.29 | flex_p_up_marginal | 0 | 0 | 23 | 10 | .. | 45.75..48.75 | 35.195 | -0.0137 | [35.181, 35.181] | 0.0000 | 29.58 |
| 13 | a_conv_interior | 17.75 | 17.899 | 17.899 | -0.0000 | 0.15 | flex_p_up_marginal | 0 | 0 | 20 | 13 | .. | 45.75..48.75 | 17.906 | -0.0067 | [17.899, 17.899] | 0.0000 | 28.77 |
| 14 | a_conv_interior | 14.37 | 14.492 | 14.492 | -0.0000 | 0.12 | flex_p_up_marginal | 0 | 0 | 15 | 17 | .. | 47.13..48.75 | 14.497 | -0.0051 | [14.492, 14.492] | 0.0000 | 26.59 |
| 15 | a_conv_interior | 3.69 | 3.716 | 3.716 | -0.0000 | 0.03 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 43.357] | 41.8451 | 25.17 |
| 16 | a_conv_interior | 7.90 | 7.956 | 7.956 | -0.0000 | 0.05 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 43.880] | 37.5443 | 25.03 |
| 17 | a_conv_interior | 15.39 | 15.485 | 15.485 | -0.0000 | 0.10 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 44.096] | 29.7970 | 27.06 |
| 18 | a_conv_interior | 46.97 | 46.819 | 46.819 | -0.0000 | -0.15 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [46.299, 76.089] | 0.5477 | 31.37 |
| 19 | c_interface | 90.75 | 85.518 | 85.511 | +0.0068 | -5.24 | flex_p_down_marginal | 22 | 0 | 0 | 0 | 85.65..88.65 | .. | 85.522 | -0.0040 | [85.511, 85.511] | 0.0000 | 39.90 |
| 20 | c_interface | 128.94 | 100.249 | 100.252 | -0.0035 | -28.69 | flex_p_down_marginal | 22 | 0 | 0 | 0 | 100.35..103.35 | .. | 100.258 | -0.0098 | [100.252, 100.252] | 0.0000 | 54.60 |
| 21 | c_interface | 161.27 | 106.202 | 106.215 | -0.0133 | -55.05 | flex_p_down_marginal | 30 | 0 | 0 | 0 | 106.25..109.26 | .. | 106.231 | -0.0293 | [106.215, 106.215] | 0.0000 | 60.50 |
| 22 | c_interface | 152.21 | 100.756 | 100.771 | -0.0152 | -51.44 | flex_p_down_marginal | 31 | 0 | 0 | 0 | 100.74..103.75 | .. | 100.780 | -0.0239 | [100.771, 100.771] | 0.0000 | 54.99 |
| 23 | c_interface | 141.99 | 99.976 | 99.974 | +0.0011 | -42.01 | flex_p_down_marginal | 23 | 10 | 0 | 0 | 99.52..102.62 | .. | 99.960 | +0.0151 | [99.974, 99.974] | 0.0000 | 53.87 |
| 24 | c_interface | 178.86 | 98.357 | 98.369 | -0.0117 | -80.49 | flex_p_down_marginal | 30 | 2 | 0 | 0 | 98.13..101.23 | .. | 98.364 | -0.0066 | [98.369, 98.369] | 0.0000 | 52.48 |

mu_l (daily shadow value of shifted energy, EUR/MWh): min 45.65, max 48.75, mean 47.12. cost_flex relation: flex_p_up marginal in 104 load-hours (hours [10, 11, 12, 13, 14]); there y_i - mu_l max abs 9.95e-02 EUR/MWh; mu_l - day-min y_i in [41.85, 44.58]; flex_p_down marginal in hours [1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 20, 21, 22, 23, 24].

### DN 7 (case33_2) 2030 Summer: hour by hour

| h | TSO setter | pi | LMP TSO bus | y0 DN ref | gap | lambda_E/B | mechanism | #P-down marginal | #P-down at ub (primal) | #P-up marginal | #P-up at ub (primal) | mu+c of marginal P-down (min..max) | mu of marginal P-up (min..max) | marginal rows referred to interface (mean) | LMP - referred | bracket [lo, hi] referred | nearest row distance | c_flex mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | c_interface | 157.23 | 128.360 | 128.360 | +0.0002 | -28.87 | flex_p_down_marginal | 23 | 9 | 0 | 0 | 128.04..132.30 | .. | 128.352 | +0.0072 | [128.359, 128.360] | 0.0000 | 53.91 |
| 2 | c_interface | 146.06 | 127.768 | 127.767 | +0.0002 | -18.29 | flex_p_down_marginal | 24 | 8 | 0 | 0 | 127.37..131.63 | .. | 127.749 | +0.0189 | [127.767, 127.767] | 0.0000 | 53.24 |
| 3 | c_interface | 140.13 | 127.145 | 127.145 | +0.0003 | -12.99 | flex_p_down_marginal | 26 | 7 | 0 | 0 | 126.72..130.99 | .. | 127.132 | +0.0130 | [127.145, 127.145] | 0.0000 | 52.60 |
| 4 | c_interface | 136.48 | 125.098 | 125.098 | +0.0004 | -11.38 | flex_p_down_marginal | 26 | 6 | 0 | 0 | 124.70..128.96 | .. | 125.089 | +0.0095 | [125.098, 125.098] | 0.0000 | 50.57 |
| 5 | c_interface | 137.07 | 125.131 | 125.131 | +0.0004 | -11.94 | flex_p_down_marginal | 26 | 6 | 0 | 0 | 124.77..129.03 | .. | 125.126 | +0.0054 | [125.131, 125.131] | 0.0000 | 50.64 |
| 6 | a_conv_interior | 124.61 | 124.795 | 124.737 | +0.0583 | 0.13 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 124.73..129.29 | .. | 124.739 | +0.0563 | [124.737, 124.737] | 0.0000 | 50.90 |
| 7 | a_conv_interior | 126.80 | 127.189 | 127.193 | -0.0038 | 0.39 | flex_p_down_marginal | 23 | 0 | 0 | 0 | 128.72..132.98 | .. | 127.208 | -0.0186 | [127.193, 127.193] | 0.0000 | 54.59 |
| 8 | c_interface | 137.40 | 130.821 | 130.816 | +0.0053 | -6.59 | flex_p_down_marginal | 30 | 1 | 0 | 0 | 130.97..135.33 | .. | 130.816 | +0.0051 | [130.816, 130.816] | 0.0000 | 56.94 |
| 9 | a_conv_interior | 100.57 | 100.973 | 100.973 | -0.0000 | 0.40 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [74.349, 124.800] | 25.3837 | 54.56 |
| 10 | c_interface | 81.90 | 77.314 | 77.314 | -0.0000 | -4.58 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [74.951, 119.244] | 2.3369 | 48.46 |
| 11 | a_conv_interior | 70.69 | 71.269 | 71.302 | -0.0327 | 0.61 | flex_p_up_marginal | 0 | 0 | 16 | 16 | .. | 75.50..78.39 | 71.313 | -0.0434 | [71.302, 71.302] | 0.0000 | 43.95 |
| 12 | a_conv_interior | 63.31 | 63.843 | 63.843 | -0.0000 | 0.53 | flex_p_up_marginal | 0 | 0 | 14 | 20 | .. | 75.76..78.39 | 63.859 | -0.0157 | [63.843, 63.843] | 0.0000 | 40.03 |
| 13 | a_conv_interior | 54.66 | 55.112 | 55.112 | -0.0000 | 0.46 | flex_p_up_marginal | 0 | 0 | 12 | 20 | .. | 76.07..78.39 | 55.117 | -0.0046 | [55.112, 55.112] | 0.0000 | 41.52 |
| 14 | a_conv_interior | 47.25 | 47.649 | 47.649 | -0.0000 | 0.40 | flex_p_up_marginal | 0 | 0 | 11 | 21 | .. | 76.38..78.39 | 47.655 | -0.0060 | [47.649, 47.649] | 0.0000 | 39.02 |
| 15 | a_conv_interior | 44.44 | 44.810 | 44.810 | -0.0000 | 0.37 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 69.435] | 27.7832 | 38.19 |
| 16 | a_conv_interior | 42.76 | 43.066 | 43.066 | -0.0000 | 0.31 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 70.224] | 29.8613 | 39.66 |
| 17 | a_conv_interior | 66.32 | 66.795 | 66.795 | -0.0000 | 0.47 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 70.383] | 3.9961 | 45.43 |
| 18 | a_conv_interior | 93.30 | 93.498 | 93.498 | -0.0000 | 0.20 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [74.140, 122.095] | 19.7345 | 50.96 |
| 19 | a_conv_interior | 109.73 | 110.119 | 110.119 | -0.0000 | 0.39 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [74.052, 131.690] | 22.7148 | 61.01 |
| 20 | c_interface | 146.52 | 143.408 | 143.417 | -0.0088 | -3.11 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 143.48..148.03 | .. | 143.418 | -0.0100 | [143.417, 143.417] | 0.0000 | 69.64 |
| 21 | c_interface | 183.24 | 143.063 | 143.070 | -0.0071 | -40.17 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 143.18..147.74 | .. | 143.074 | -0.0109 | [143.070, 143.070] | 0.0000 | 69.35 |
| 22 | c_interface | 165.71 | 135.842 | 135.842 | +0.0003 | -29.87 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 135.94..140.50 | .. | 135.845 | -0.0034 | [135.842, 135.842] | 0.0000 | 62.11 |
| 23 | c_interface | 158.90 | 132.593 | 132.603 | -0.0102 | -26.30 | flex_p_down_marginal | 30 | 2 | 0 | 0 | 132.64..137.20 | .. | 132.594 | -0.0007 | [132.603, 132.603] | 0.0000 | 58.81 |
| 24 | c_interface | 139.14 | 130.594 | 130.598 | -0.0044 | -8.54 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 130.63..135.18 | .. | 130.596 | -0.0025 | [130.598, 130.598] | 0.0000 | 56.79 |

mu_l (daily shadow value of shifted energy, EUR/MWh): min 73.83, max 78.39, mean 76.00. cost_flex relation: flex_p_up marginal in 53 load-hours (hours [11, 12, 13, 14]); there y_i - mu_l max abs 9.76e-02 EUR/MWh; mu_l - day-min y_i in [29.86, 30.69]; flex_p_down marginal in hours [1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23, 24].

### DN 7 (case33_2) 2030 Autumn: hour by hour

| h | TSO setter | pi | LMP TSO bus | y0 DN ref | gap | lambda_E/B | mechanism | #P-down marginal | #P-down at ub (primal) | #P-up marginal | #P-up at ub (primal) | mu+c of marginal P-down (min..max) | mu of marginal P-up (min..max) | marginal rows referred to interface (mean) | LMP - referred | bracket [lo, hi] referred | nearest row distance | c_flex mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | c_interface | 101.83 | 86.765 | 86.765 | +0.0000 | -15.06 | flex_p_up_marginal | 0 | 0 | 5 | 0 | .. | 89.39..90.23 | 86.756 | +0.0097 | [86.765, 139.363] | 0.0000 | 54.80 |
| 2 | a_conv_interior | 99.12 | 98.152 | 98.152 | -0.0000 | -0.97 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [86.896, 139.545] | 11.6805 | 55.23 |
| 3 | c_interface | 98.48 | 86.727 | 86.727 | +0.0000 | -11.75 | flex_p_up_marginal | 0 | 0 | 10 | 0 | .. | 87.66..90.23 | 86.689 | +0.0375 | [86.727, 86.727] | 0.0000 | 52.37 |
| 4 | a_conv_interior | 92.15 | 91.209 | 91.209 | -0.0000 | -0.94 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.146, 135.818] | 4.2072 | 50.95 |
| 5 | c_interface | 93.28 | 86.746 | 86.746 | +0.0000 | -6.54 | flex_p_up_marginal | 0 | 0 | 11 | 0 | .. | 87.66..90.23 | 86.719 | +0.0277 | [86.746, 133.486] | 0.0000 | 48.62 |
| 6 | c_interface | 98.40 | 86.710 | 86.710 | -0.0000 | -11.69 | flex_p_up_marginal | 0 | 0 | 11 | 0 | .. | 87.66..90.23 | 86.689 | +0.0202 | [86.710, 133.539] | 0.0000 | 48.73 |
| 7 | a_conv_interior | 113.26 | 112.843 | 112.843 | -0.0000 | -0.42 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [86.365, 138.988] | 26.9234 | 55.56 |
| 8 | a_conv_interior | 131.95 | 131.097 | 131.097 | -0.0000 | -0.86 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [86.340, 146.336] | 15.9491 | 62.93 |
| 9 | a_conv_interior | 133.52 | 132.577 | 132.577 | -0.0000 | -0.95 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [86.319, 149.852] | 18.1215 | 66.97 |
| 10 | c_interface | 127.85 | 86.357 | 86.357 | -0.0000 | -41.49 | flex_p_up_marginal | 0 | 0 | 6 | 0 | .. | 86.29..89.51 | 86.339 | +0.0184 | [86.357, 146.156] | 0.0000 | 63.13 |
| 11 | c_interface | 109.70 | 86.280 | 86.280 | -0.0000 | -23.42 | flex_p_up_marginal | 0 | 0 | 7 | 0 | .. | 86.29..89.51 | 86.271 | +0.0087 | [86.280, 86.280] | 0.0000 | 57.52 |
| 12 | c_interface | 101.19 | 86.386 | 86.386 | -0.0000 | -14.80 | flex_p_up_marginal | 0 | 0 | 7 | 0 | .. | 86.29..89.51 | 86.369 | +0.0166 | [86.386, 133.745] | 0.0000 | 50.47 |
| 13 | c_interface | 100.22 | 86.402 | 86.402 | +0.0000 | -13.82 | flex_p_up_marginal | 0 | 0 | 7 | 0 | .. | 86.29..89.51 | 86.388 | +0.0139 | [86.402, 137.336] | 0.0000 | 54.26 |
| 14 | a_conv_interior | 98.48 | 97.952 | 97.952 | -0.0000 | -0.53 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [86.353, 134.728] | 11.9271 | 51.79 |
| 15 | a_conv_interior | 101.26 | 100.820 | 100.820 | -0.0000 | -0.44 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [86.376, 134.302] | 14.8659 | 50.90 |
| 16 | a_conv_interior | 119.14 | 119.046 | 119.046 | -0.0000 | -0.09 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [86.275, 138.073] | 20.0313 | 55.13 |
| 17 | a_conv_interior | 133.04 | 133.250 | 133.250 | -0.0000 | 0.21 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [86.309, 145.685] | 13.0709 | 62.90 |
| 18 | a_conv_interior | 150.16 | 150.610 | 150.610 | -0.0000 | 0.45 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [86.125, 151.465] | 0.9056 | 70.26 |
| 19 | c_interface | 175.07 | 169.067 | 169.067 | +0.0000 | -6.01 | flex_p_down_marginal | 24 | 1 | 0 | 0 | 170.15..174.21 | .. | 169.080 | -0.0129 | [169.067, 169.067] | 0.0000 | 83.98 |
| 20 | a_conv_interior | 166.66 | 166.635 | 166.635 | -0.0000 | -0.03 | flex_p_down_marginal | 5 | 0 | 0 | 0 | 173.75..174.56 | .. | 166.663 | -0.0281 | [166.620, 166.635] | 0.0000 | 84.33 |
| 21 | a_conv_interior | 143.07 | 143.226 | 143.226 | -0.0000 | 0.16 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [86.321, 159.345] | 16.9673 | 77.50 |
| 22 | a_conv_interior | 135.89 | 135.725 | 135.725 | -0.0000 | -0.16 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [86.533, 151.631] | 16.6434 | 68.43 |
| 23 | a_conv_interior | 132.16 | 132.338 | 132.338 | -0.0000 | 0.17 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [86.419, 146.494] | 14.8562 | 63.51 |
| 24 | a_conv_interior | 121.88 | 122.045 | 122.045 | -0.0000 | 0.17 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [86.486, 143.981] | 22.8740 | 60.72 |

mu_l (daily shadow value of shifted energy, EUR/MWh): min 84.86, max 90.23, mean 88.00. cost_flex relation: flex_p_up marginal in 64 load-hours (hours [1, 3, 5, 6, 10, 11, 12, 13]); there y_i - mu_l max abs 9.63e-02 EUR/MWh; mu_l - day-min y_i in [-1.21, 0.00]; flex_p_down marginal in hours [19, 20].

### DN 7 (case33_2) 2030 Winter: hour by hour

| h | TSO setter | pi | LMP TSO bus | y0 DN ref | gap | lambda_E/B | mechanism | #P-down marginal | #P-down at ub (primal) | #P-up marginal | #P-up at ub (primal) | mu+c of marginal P-down (min..max) | mu of marginal P-up (min..max) | marginal rows referred to interface (mean) | LMP - referred | bracket [lo, hi] referred | nearest row distance | c_flex mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a_conv_interior | 108.72 | 108.342 | 108.342 | -0.0000 | -0.38 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.498, 132.895] | 14.3852 | 40.19 |
| 2 | a_conv_interior | 101.09 | 100.550 | 100.550 | -0.0000 | -0.54 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.711, 131.816] | 6.0531 | 38.86 |
| 3 | a_conv_interior | 96.26 | 95.481 | 95.481 | +0.0000 | -0.78 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [95.195, 131.241] | 0.2952 | 37.79 |
| 4 | a_conv_interior | 93.37 | 93.427 | 93.425 | +0.0022 | 0.06 | flex_p_up_marginal | 0 | 0 | 32 | 0 | .. | 93.23..98.19 | 93.414 | +0.0133 | [93.425, 93.425] | 0.0000 | 39.15 |
| 5 | a_conv_interior | 93.17 | 93.346 | 93.355 | -0.0081 | 0.18 | flex_p_up_marginal | 0 | 0 | 32 | 0 | .. | 93.23..98.19 | 93.352 | -0.0055 | [93.355, 93.355] | 0.0000 | 37.44 |
| 6 | a_conv_interior | 95.12 | 94.327 | 94.326 | +0.0007 | -0.79 | flex_p_up_marginal | 0 | 0 | 14 | 0 | .. | 96.18..98.19 | 94.305 | +0.0217 | [94.326, 94.326] | 0.0000 | 38.46 |
| 7 | a_conv_interior | 98.94 | 98.639 | 98.639 | -0.0000 | -0.30 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [93.937, 134.018] | 4.9148 | 41.90 |
| 8 | a_conv_interior | 120.55 | 120.198 | 120.198 | -0.0000 | -0.35 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [93.965, 135.118] | 15.5910 | 43.01 |
| 9 | a_conv_interior | 130.75 | 130.497 | 130.497 | +0.0000 | -0.25 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [93.963, 131.359] | 0.9028 | 39.45 |
| 10 | a_conv_interior | 125.15 | 124.497 | 124.497 | -0.0000 | -0.65 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.503, 127.610] | 3.2652 | 35.65 |
| 11 | a_conv_interior | 113.40 | 112.706 | 112.706 | -0.0000 | -0.69 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.583, 122.925] | 10.7266 | 30.84 |
| 12 | a_conv_interior | 96.71 | 96.478 | 96.478 | -0.0000 | -0.23 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.562, 121.088] | 1.9157 | 29.42 |
| 13 | a_conv_interior | 94.24 | 94.397 | 94.397 | +0.0003 | 0.16 | flex_p_up_marginal | 0 | 0 | 1 | 0 | .. | 94.54..94.54 | 94.396 | +0.0013 | [94.396, 117.550] | 0.0010 | 25.76 |
| 14 | a_conv_interior | 92.37 | 92.728 | 92.741 | -0.0131 | 0.37 | flex_p_up_marginal | 0 | 0 | 14 | 2 | .. | 93.72..97.28 | 92.718 | +0.0098 | [92.741, 92.741] | 0.0000 | 25.53 |
| 15 | a_conv_interior | 94.97 | 95.164 | 95.164 | -0.0000 | 0.19 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.171, 115.856] | 0.9971 | 23.90 |
| 16 | a_conv_interior | 98.28 | 98.503 | 98.503 | -0.0000 | 0.23 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [93.961, 116.474] | 4.5703 | 24.57 |
| 17 | a_conv_interior | 116.32 | 116.636 | 116.636 | -0.0000 | 0.31 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [93.382, 120.228] | 3.7893 | 28.64 |
| 18 | c_interface | 146.55 | 127.248 | 127.233 | +0.0150 | -19.31 | flex_p_down_marginal | 31 | 2 | 0 | 0 | 126.94..131.90 | .. | 127.228 | +0.0200 | [127.233, 127.233] | 0.0000 | 33.71 |
| 19 | c_interface | 161.38 | 134.877 | 134.875 | +0.0021 | -26.51 | flex_p_down_marginal | 31 | 0 | 0 | 0 | 135.04..139.66 | .. | 134.879 | -0.0028 | [134.875, 134.875] | 0.0000 | 41.47 |
| 20 | c_interface | 160.53 | 139.532 | 139.532 | +0.0006 | -21.00 | flex_p_down_marginal | 21 | 0 | 0 | 0 | 140.62..144.55 | .. | 139.549 | -0.0168 | [139.532, 139.532] | 0.0000 | 46.36 |
| 21 | a_conv_interior | 138.34 | 138.571 | 138.571 | -0.0000 | 0.23 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [93.491, 140.874] | 2.4214 | 49.93 |
| 22 | a_conv_interior | 128.45 | 128.589 | 128.589 | -0.0000 | 0.14 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [93.860, 134.455] | 6.1362 | 42.47 |
| 23 | a_conv_interior | 118.37 | 118.635 | 118.635 | -0.0000 | 0.27 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [93.581, 135.064] | 17.2425 | 43.56 |
| 24 | a_conv_interior | 111.97 | 112.194 | 112.194 | -0.0000 | 0.23 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [93.838, 134.881] | 18.8353 | 42.95 |

mu_l (daily shadow value of shifted energy, EUR/MWh): min 93.23, max 98.19, mean 95.77. cost_flex relation: flex_p_up marginal in 93 load-hours (hours [4, 5, 6, 13, 14]); there y_i - mu_l max abs 9.96e-02 EUR/MWh; mu_l - day-min y_i in [-0.03, 0.41]; flex_p_down marginal in hours [18, 19, 20].

### DN 9 (case33_3) 2030 Spring: hour by hour

| h | TSO setter | pi | LMP TSO bus | y0 DN ref | gap | lambda_E/B | mechanism | #P-down marginal | #P-down at ub (primal) | #P-up marginal | #P-up at ub (primal) | mu+c of marginal P-down (min..max) | mu of marginal P-up (min..max) | marginal rows referred to interface (mean) | LMP - referred | bracket [lo, hi] referred | nearest row distance | c_flex mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | c_interface | 132.01 | 100.616 | 100.615 | +0.0007 | -31.39 | flex_p_down_marginal | 8 | 24 | 0 | 0 | 101.06..102.32 | .. | 100.601 | +0.0150 | [100.615, ] | 0.0000 | 54.14 |
| 2 | c_interface | 133.20 | 98.382 | 98.382 | +0.0005 | -34.82 | flex_p_down_marginal | 14 | 23 | 0 | 0 | 98.85..100.12 | .. | 98.342 | +0.0410 | [98.382, ] | 0.0000 | 51.94 |
| 3 | c_interface | 135.99 | 99.131 | 99.131 | -0.0000 | -36.86 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 98.95..100.91 | .. | 99.119 | +0.0122 | [99.131, 99.131] | 0.0000 | 52.73 |
| 4 | c_interface | 134.90 | 97.815 | 97.816 | -0.0003 | -37.08 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 97.65..99.61 | .. | 97.810 | +0.0058 | [97.816, 97.816] | 0.0000 | 51.43 |
| 5 | c_interface | 141.12 | 99.455 | 99.455 | -0.0004 | -41.67 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 99.32..101.28 | .. | 99.454 | +0.0005 | [99.455, 99.455] | 0.0000 | 53.10 |
| 6 | c_interface | 144.86 | 98.948 | 98.948 | -0.0004 | -45.91 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 98.82..100.77 | .. | 98.947 | +0.0004 | [98.948, 98.948] | 0.0000 | 52.59 |
| 7 | c_interface | 152.60 | 101.355 | 101.356 | -0.0012 | -51.25 | flex_p_down_marginal | 32 | 0 | 0 | 0 | 101.24..103.20 | .. | 101.358 | -0.0026 | [101.356, 101.356] | 0.0000 | 55.02 |
| 8 | c_interface | 131.21 | 98.673 | 98.676 | -0.0029 | -32.53 | flex_p_down_marginal | 25 | 0 | 0 | 0 | 98.78..100.55 | .. | 98.680 | -0.0075 | [98.676, 98.676] | 0.0000 | 52.37 |
| 9 | a_conv_interior | 88.06 | 88.124 | 88.127 | -0.0030 | 0.06 | flex_p_down_marginal | 15 | 0 | 0 | 0 | 88.95..90.23 | .. | 88.131 | -0.0075 | [88.127, 88.127] | 0.0000 | 42.05 |
| 10 | a_conv_interior | 43.13 | 43.646 | 43.647 | -0.0010 | 0.51 | flex_p_up_marginal | 0 | 0 | 5 | 27 | .. | 47.77..48.18 | 43.655 | -0.0092 | [43.647, 43.648] | 0.0001 | 33.04 |
| 11 | a_conv_interior | 38.03 | 38.454 | 38.454 | -0.0000 | 0.43 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 42.861] | 4.9538 | 30.10 |
| 12 | a_conv_interior | 34.89 | 35.264 | 35.264 | -0.0000 | 0.37 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 42.994] | 8.6631 | 29.58 |
| 13 | a_conv_interior | 17.75 | 17.942 | 17.942 | -0.0000 | 0.19 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 42.880] | 27.9009 | 28.77 |
| 14 | a_conv_interior | 14.37 | 14.517 | 14.517 | -0.0000 | 0.15 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 43.156] | 31.5366 | 26.59 |
| 15 | a_conv_interior | 3.69 | 3.717 | 3.717 | -0.0000 | 0.03 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 43.799] | 42.5148 | 25.17 |
| 16 | a_conv_interior | 7.90 | 7.960 | 7.960 | -0.0000 | 0.06 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 44.197] | 38.2750 | 25.03 |
| 17 | a_conv_interior | 15.39 | 15.506 | 15.506 | -0.0000 | 0.12 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 44.213] | 30.5398 | 27.06 |
| 18 | a_conv_interior | 46.97 | 47.059 | 47.060 | -0.0005 | 0.09 | flex_p_up_marginal | 0 | 0 | 2 | 0 | .. | 47.83..47.86 | 47.022 | +0.0371 | [47.059, 76.260] | 0.0002 | 31.37 |
| 19 | c_interface | 90.75 | 86.069 | 86.070 | -0.0009 | -4.68 | flex_p_down_marginal | 27 | 0 | 0 | 0 | 86.13..88.08 | .. | 86.091 | -0.0218 | [86.070, 86.070] | 0.0000 | 39.90 |
| 20 | c_interface | 128.94 | 100.865 | 100.864 | +0.0007 | -28.08 | flex_p_down_marginal | 25 | 6 | 0 | 0 | 100.82..102.78 | .. | 100.867 | -0.0022 | [100.864, 100.864] | 0.0000 | 54.60 |
| 21 | c_interface | 161.27 | 106.792 | 106.789 | +0.0026 | -54.48 | flex_p_down_marginal | 24 | 9 | 0 | 0 | 106.73..108.68 | .. | 106.790 | +0.0014 | [106.789, 106.789] | 0.0000 | 60.50 |
| 22 | c_interface | 152.21 | 101.301 | 101.298 | +0.0038 | -50.91 | flex_p_down_marginal | 32 | 1 | 0 | 0 | 101.22..103.17 | .. | 101.292 | +0.0099 | [101.298, 101.298] | 0.0000 | 54.99 |
| 23 | c_interface | 141.99 | 100.670 | 100.670 | -0.0002 | -41.32 | bracketed by flex rows | 0 | 32 | 0 | 0 | .. | .. |  |  | [100.352, ] | 0.3189 | 53.87 |
| 24 | c_interface | 178.86 | 98.936 | 98.934 | +0.0021 | -79.93 | flex_p_down_marginal | 15 | 17 | 0 | 0 | 99.17..100.66 | .. | 98.919 | +0.0170 | [98.934, 98.941] | 0.0000 | 52.48 |

mu_l (daily shadow value of shifted energy, EUR/MWh): min 46.23, max 48.18, mean 47.27. cost_flex relation: flex_p_up marginal in 7 load-hours (hours [10, 18]); there y_i - mu_l max abs 7.66e-02 EUR/MWh; mu_l - day-min y_i in [42.51, 44.09]; flex_p_down marginal in hours [1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 20, 21, 22, 24].

### DN 9 (case33_3) 2030 Summer: hour by hour

| h | TSO setter | pi | LMP TSO bus | y0 DN ref | gap | lambda_E/B | mechanism | #P-down marginal | #P-down at ub (primal) | #P-up marginal | #P-up at ub (primal) | mu+c of marginal P-down (min..max) | mu of marginal P-up (min..max) | marginal rows referred to interface (mean) | LMP - referred | bracket [lo, hi] referred | nearest row distance | c_flex mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | c_interface | 157.23 | 129.160 | 129.160 | -0.0000 | -28.07 | flex_p_down_marginal | 4 | 28 | 0 | 0 | 130.85..131.16 | .. | 129.140 | +0.0208 | [129.160, ] | 0.0000 | 53.91 |
| 2 | c_interface | 146.06 | 128.470 | 128.470 | +0.0000 | -17.59 | flex_p_down_marginal | 5 | 27 | 0 | 0 | 129.98..130.49 | .. | 128.449 | +0.0220 | [128.470, ] | 0.0000 | 53.24 |
| 3 | c_interface | 140.13 | 127.711 | 127.711 | -0.0000 | -12.42 | flex_p_down_marginal | 20 | 12 | 0 | 0 | 127.28..129.95 | .. | 127.703 | +0.0075 | [127.711, 127.711] | 0.0000 | 52.60 |
| 4 | c_interface | 136.48 | 125.631 | 125.631 | -0.0000 | -10.85 | flex_p_down_marginal | 22 | 10 | 0 | 0 | 125.26..127.92 | .. | 125.624 | +0.0074 | [125.631, 125.631] | 0.0000 | 50.57 |
| 5 | c_interface | 137.07 | 125.635 | 125.635 | -0.0000 | -11.43 | flex_p_down_marginal | 22 | 10 | 0 | 0 | 125.33..127.99 | .. | 125.632 | +0.0031 | [125.635, 125.635] | 0.0000 | 50.64 |
| 6 | a_conv_interior | 124.61 | 125.023 | 125.025 | -0.0018 | 0.41 | flex_p_down_marginal | 27 | 1 | 0 | 0 | 125.59..128.25 | .. | 125.027 | -0.0037 | [125.025, 125.025] | 0.0000 | 50.90 |
| 7 | a_conv_interior | 126.80 | 127.430 | 127.433 | -0.0029 | 0.63 | flex_p_down_marginal | 22 | 0 | 0 | 0 | 129.28..131.94 | .. | 127.433 | -0.0031 | [127.433, 127.433] | 0.0000 | 54.59 |
| 8 | c_interface | 137.40 | 131.498 | 131.500 | -0.0015 | -5.90 | flex_p_down_marginal | 20 | 4 | 0 | 0 | 131.67..134.18 | .. | 131.498 | +0.0004 | [131.500, 131.500] | 0.0000 | 56.94 |
| 9 | a_conv_interior | 100.57 | 100.820 | 100.820 | -0.0000 | 0.25 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [76.922, 125.909] | 23.2633 | 54.56 |
| 10 | c_interface | 81.90 | 77.254 | 77.254 | +0.0000 | -4.64 | flex_p_up_marginal | 0 | 0 | 2 | 2 | .. | 74.69..77.33 | 77.254 | +0.0000 | [76.972, 77.254] | 0.0000 | 48.46 |
| 11 | a_conv_interior | 70.69 | 71.181 | 71.182 | -0.0013 | 0.49 | flex_p_up_marginal | 0 | 0 | 3 | 29 | .. | 77.15..77.25 | 71.180 | +0.0008 | [71.180, 71.182] | 0.0000 | 43.95 |
| 12 | a_conv_interior | 63.31 | 63.759 | 63.759 | -0.0000 | 0.45 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 69.888] | 6.7743 | 40.03 |
| 13 | a_conv_interior | 54.66 | 55.039 | 55.039 | -0.0000 | 0.38 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 69.684] | 16.2342 | 41.52 |
| 14 | a_conv_interior | 47.25 | 47.558 | 47.558 | -0.0000 | 0.30 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 69.970] | 24.7429 | 39.02 |
| 15 | a_conv_interior | 44.44 | 44.683 | 44.683 | -0.0000 | 0.24 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 70.685] | 28.4151 | 38.19 |
| 16 | a_conv_interior | 42.76 | 42.963 | 42.963 | -0.0000 | 0.21 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 71.333] | 30.7180 | 39.66 |
| 17 | a_conv_interior | 66.32 | 66.736 | 66.736 | -0.0000 | 0.41 | bracketed by flex rows | 0 | 0 | 0 | 32 | .. | .. |  |  | [, 71.193] | 4.8365 | 45.43 |
| 18 | a_conv_interior | 93.30 | 93.501 | 93.501 | -0.0000 | 0.20 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [76.661, 123.173] | 16.9755 | 50.96 |
| 19 | a_conv_interior | 109.73 | 110.249 | 110.249 | -0.0000 | 0.51 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [74.481, 132.143] | 22.9076 | 61.01 |
| 20 | c_interface | 146.52 | 144.206 | 144.204 | +0.0018 | -2.32 | flex_p_down_marginal | 26 | 6 | 0 | 0 | 144.33..146.89 | .. | 144.197 | +0.0087 | [144.204, 144.204] | 0.0000 | 69.64 |
| 21 | c_interface | 183.24 | 143.874 | 143.872 | +0.0022 | -39.37 | flex_p_down_marginal | 26 | 7 | 0 | 0 | 144.03..146.59 | .. | 143.862 | +0.0116 | [143.872, 143.872] | 0.0000 | 69.35 |
| 22 | c_interface | 165.71 | 136.646 | 136.646 | +0.0001 | -29.07 | flex_p_down_marginal | 30 | 2 | 0 | 0 | 136.79..139.45 | .. | 136.651 | -0.0045 | [136.646, 136.646] | 0.0000 | 62.11 |
| 23 | c_interface | 158.90 | 133.432 | 133.430 | +0.0023 | -25.47 | flex_p_down_marginal | 20 | 12 | 0 | 0 | 133.54..136.05 | .. | 133.404 | +0.0279 | [133.429, 133.430] | 0.0000 | 58.81 |
| 24 | c_interface | 139.14 | 131.404 | 131.403 | +0.0010 | -7.74 | flex_p_down_marginal | 23 | 9 | 0 | 0 | 131.53..134.04 | .. | 131.389 | +0.0144 | [131.403, 131.403] | 0.0000 | 56.79 |

mu_l (daily shadow value of shifted energy, EUR/MWh): min 74.69, max 77.35, mean 76.12. cost_flex relation: flex_p_up marginal in 5 load-hours (hours [10, 11]); there y_i - mu_l max abs 3.70e-03 EUR/MWh; mu_l - day-min y_i in [30.72, 34.11]; flex_p_down marginal in hours [1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23, 24].

### DN 9 (case33_3) 2030 Autumn: hour by hour

| h | TSO setter | pi | LMP TSO bus | y0 DN ref | gap | lambda_E/B | mechanism | #P-down marginal | #P-down at ub (primal) | #P-up marginal | #P-up at ub (primal) | mu+c of marginal P-down (min..max) | mu of marginal P-up (min..max) | marginal rows referred to interface (mean) | LMP - referred | bracket [lo, hi] referred | nearest row distance | c_flex mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | c_interface | 101.83 | 87.871 | 87.871 | -0.0000 | -13.96 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.421, 138.226] | 0.4671 | 54.80 |
| 2 | a_conv_interior | 99.12 | 99.293 | 99.293 | -0.0000 | 0.17 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.445, 138.489] | 12.0625 | 55.23 |
| 3 | c_interface | 98.48 | 87.743 | 87.743 | -0.0000 | -10.73 | flex_p_up_marginal | 0 | 0 | 4 | 0 | .. | 90.18..90.61 | 87.729 | +0.0143 | [87.743, 136.276] | 0.0000 | 52.37 |
| 4 | a_conv_interior | 92.15 | 92.242 | 92.242 | -0.0000 | 0.09 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.873, 134.966] | 4.5051 | 50.95 |
| 5 | c_interface | 93.28 | 87.715 | 87.715 | -0.0000 | -5.57 | flex_p_up_marginal | 0 | 0 | 5 | 0 | .. | 89.86..90.61 | 87.700 | +0.0149 | [87.715, 132.957] | 0.0000 | 48.62 |
| 6 | c_interface | 98.40 | 87.680 | 87.680 | +0.0000 | -10.72 | flex_p_up_marginal | 0 | 0 | 5 | 0 | .. | 89.86..90.61 | 87.667 | +0.0137 | [87.680, 133.019] | 0.0000 | 48.73 |
| 7 | a_conv_interior | 113.26 | 113.668 | 113.668 | -0.0000 | 0.41 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.210, 138.102] | 25.5503 | 55.56 |
| 8 | a_conv_interior | 131.95 | 132.283 | 132.283 | -0.0000 | 0.33 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.224, 145.867] | 14.1341 | 62.93 |
| 9 | a_conv_interior | 133.52 | 133.786 | 133.786 | -0.0000 | 0.26 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.153, 150.645] | 17.4386 | 66.97 |
| 10 | c_interface | 127.85 | 87.167 | 87.167 | +0.0000 | -40.68 | flex_p_up_marginal | 0 | 0 | 9 | 0 | .. | 88.36..90.61 | 87.142 | +0.0251 | [87.167, 147.892] | 0.0001 | 63.13 |
| 11 | c_interface | 109.70 | 87.059 | 87.059 | +0.0000 | -22.64 | flex_p_up_marginal | 0 | 0 | 20 | 0 | .. | 86.92..90.61 | 87.050 | +0.0092 | [87.059, 142.318] | 0.0000 | 57.52 |
| 12 | c_interface | 101.19 | 87.121 | 87.121 | +0.0000 | -14.06 | flex_p_up_marginal | 0 | 0 | 21 | 0 | .. | 86.92..90.61 | 87.117 | +0.0038 | [87.121, 87.121] | 0.0000 | 50.47 |
| 13 | c_interface | 100.22 | 87.127 | 87.127 | -0.0000 | -13.09 | flex_p_up_marginal | 0 | 0 | 21 | 0 | .. | 86.92..90.61 | 87.104 | +0.0224 | [87.127, 87.127] | 0.0000 | 54.26 |
| 14 | a_conv_interior | 98.48 | 98.595 | 98.595 | -0.0000 | 0.11 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.683, 136.839] | 11.0573 | 51.79 |
| 15 | a_conv_interior | 101.26 | 101.401 | 101.401 | -0.0000 | 0.14 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.473, 136.437] | 14.2218 | 50.90 |
| 16 | a_conv_interior | 119.14 | 119.490 | 119.490 | -0.0000 | 0.35 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.244, 140.165] | 21.4576 | 55.13 |
| 17 | a_conv_interior | 133.04 | 133.622 | 133.622 | -0.0000 | 0.58 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.221, 146.231] | 13.0852 | 62.90 |
| 18 | a_conv_interior | 150.16 | 151.084 | 151.084 | +0.0000 | 0.93 | flex_p_down_marginal | 1 | 0 | 0 | 0 | 159.12..159.12 | .. | 151.084 | -0.0000 | [87.027, 151.084] | 0.0000 | 70.26 |
| 19 | c_interface | 175.07 | 170.424 | 170.424 | +0.0000 | -4.65 | flex_p_down_marginal | 22 | 10 | 0 | 0 | 170.85..174.59 | .. | 170.452 | -0.0279 | [170.424, 170.424] | 0.0000 | 83.98 |
| 20 | a_conv_interior | 166.66 | 167.371 | 167.371 | +0.0000 | 0.71 | flex_p_down_marginal | 2 | 2 | 0 | 0 | 171.25..173.62 | .. | 167.371 | -0.0000 | [167.371, 167.371] | 0.0000 | 84.33 |
| 21 | a_conv_interior | 143.07 | 143.785 | 143.785 | -0.0000 | 0.72 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.188, 158.902] | 15.8262 | 77.50 |
| 22 | a_conv_interior | 135.89 | 136.430 | 136.430 | -0.0000 | 0.55 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.372, 150.850] | 15.0350 | 68.43 |
| 23 | a_conv_interior | 132.16 | 132.865 | 132.865 | -0.0000 | 0.70 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.246, 145.355] | 13.0926 | 63.51 |
| 24 | a_conv_interior | 121.88 | 122.526 | 122.526 | -0.0000 | 0.65 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [87.271, 142.691] | 21.1364 | 60.72 |

mu_l (daily shadow value of shifted energy, EUR/MWh): min 86.87, max 90.61, mean 88.64. cost_flex relation: flex_p_up marginal in 85 load-hours (hours [3, 5, 6, 10, 11, 12, 13]); there y_i - mu_l max abs 9.58e-02 EUR/MWh; mu_l - day-min y_i in [-0.42, 0.00]; flex_p_down marginal in hours [18, 19, 20].

### DN 9 (case33_3) 2030 Winter: hour by hour

| h | TSO setter | pi | LMP TSO bus | y0 DN ref | gap | lambda_E/B | mechanism | #P-down marginal | #P-down at ub (primal) | #P-up marginal | #P-up at ub (primal) | mu+c of marginal P-down (min..max) | mu of marginal P-up (min..max) | marginal rows referred to interface (mean) | LMP - referred | bracket [lo, hi] referred | nearest row distance | c_flex mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a_conv_interior | 108.72 | 109.151 | 109.151 | -0.0000 | 0.43 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.977, 131.460] | 14.7884 | 40.19 |
| 2 | a_conv_interior | 101.09 | 101.439 | 101.439 | -0.0000 | 0.34 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [95.079, 130.192] | 6.6284 | 38.86 |
| 3 | a_conv_interior | 96.26 | 96.477 | 96.477 | -0.0000 | 0.22 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [95.525, 129.505] | 0.9882 | 37.79 |
| 4 | a_conv_interior | 93.37 | 93.900 | 93.899 | +0.0009 | 0.53 | flex_p_up_marginal | 0 | 0 | 27 | 0 | .. | 94.30..99.10 | 93.895 | +0.0053 | [93.899, 93.899] | 0.0000 | 39.15 |
| 5 | a_conv_interior | 93.17 | 93.785 | 93.784 | +0.0012 | 0.61 | flex_p_up_marginal | 0 | 0 | 27 | 0 | .. | 94.30..99.10 | 93.788 | -0.0025 | [93.784, 93.784] | 0.0000 | 37.44 |
| 6 | a_conv_interior | 95.12 | 95.307 | 95.307 | +0.0001 | 0.19 | flex_p_up_marginal | 0 | 0 | 4 | 0 | .. | 98.49..99.10 | 95.285 | +0.0219 | [95.307, 130.388] | 0.0000 | 38.46 |
| 7 | a_conv_interior | 98.94 | 99.391 | 99.391 | -0.0000 | 0.46 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.329, 132.795] | 5.3177 | 41.90 |
| 8 | a_conv_interior | 120.55 | 121.078 | 121.078 | -0.0000 | 0.53 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.458, 134.342] | 13.6675 | 43.01 |
| 9 | a_conv_interior | 130.75 | 131.336 | 131.336 | +0.0001 | 0.59 | flex_p_down_marginal | 1 | 0 | 0 | 0 | 137.40..137.40 | .. | 131.336 | +0.0000 | [94.307, 131.336] | 0.0000 | 39.45 |
| 10 | a_conv_interior | 125.15 | 125.471 | 125.471 | -0.0000 | 0.32 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.881, 128.505] | 3.1814 | 35.65 |
| 11 | a_conv_interior | 113.40 | 113.526 | 113.526 | -0.0000 | 0.13 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [96.137, 124.292] | 11.0289 | 30.84 |
| 12 | a_conv_interior | 96.71 | 96.870 | 96.870 | +0.0000 | 0.16 | flex_p_up_marginal | 0 | 0 | 1 | 0 | .. | 94.84..94.84 | 96.811 | +0.0594 | [96.811, 122.797] | 0.0582 | 29.42 |
| 13 | a_conv_interior | 94.24 | 94.667 | 94.666 | +0.0011 | 0.43 | flex_p_up_marginal | 0 | 0 | 8 | 0 | .. | 94.84..98.36 | 94.661 | +0.0062 | [94.666, 94.666] | 0.0000 | 25.76 |
| 14 | a_conv_interior | 92.37 | 93.084 | 93.083 | +0.0005 | 0.72 | flex_p_up_marginal | 0 | 0 | 25 | 7 | .. | 94.84..99.10 | 93.079 | +0.0054 | [93.083, 93.083] | 0.0000 | 25.53 |
| 15 | a_conv_interior | 94.97 | 95.317 | 95.317 | +0.0003 | 0.35 | flex_p_up_marginal | 0 | 0 | 7 | 0 | .. | 94.84..98.36 | 95.315 | +0.0019 | [95.317, 95.317] | 0.0000 | 23.90 |
| 16 | a_conv_interior | 98.28 | 98.623 | 98.623 | -0.0000 | 0.35 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [96.312, 118.258] | 2.3498 | 24.57 |
| 17 | a_conv_interior | 116.32 | 116.926 | 116.926 | -0.0000 | 0.60 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.618, 121.129] | 4.2847 | 28.64 |
| 18 | c_interface | 146.55 | 128.215 | 128.218 | -0.0033 | -18.33 | flex_p_down_marginal | 24 | 8 | 0 | 0 | 128.93..132.80 | .. | 128.211 | +0.0036 | [128.218, 128.218] | 0.0000 | 33.71 |
| 19 | c_interface | 161.38 | 135.926 | 135.926 | -0.0005 | -25.46 | flex_p_down_marginal | 20 | 12 | 0 | 0 | 136.70..140.57 | .. | 135.919 | +0.0068 | [135.926, 135.926] | 0.0000 | 41.47 |
| 20 | c_interface | 160.53 | 140.602 | 140.602 | -0.0004 | -19.93 | flex_p_down_marginal | 27 | 6 | 0 | 0 | 141.22..145.45 | .. | 140.607 | -0.0047 | [140.602, 140.602] | 0.0000 | 46.36 |
| 21 | a_conv_interior | 138.34 | 139.140 | 139.140 | -0.0000 | 0.80 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.142, 140.725] | 1.6381 | 49.93 |
| 22 | a_conv_interior | 128.45 | 129.118 | 129.118 | -0.0000 | 0.67 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.565, 133.803] | 4.8078 | 42.47 |
| 23 | a_conv_interior | 118.37 | 119.094 | 119.094 | -0.0000 | 0.73 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.157, 134.168] | 15.5495 | 43.56 |
| 24 | a_conv_interior | 111.97 | 112.637 | 112.637 | -0.0000 | 0.67 | bracketed by flex rows | 0 | 0 | 0 | 0 | .. | .. |  |  | [94.246, 133.663] | 18.7392 | 42.95 |

mu_l (daily shadow value of shifted energy, EUR/MWh): min 94.30, max 99.10, mean 96.82. cost_flex relation: flex_p_up marginal in 99 load-hours (hours [4, 5, 6, 12, 13, 14, 15]); there y_i - mu_l max abs 9.61e-02 EUR/MWh; mu_l - day-min y_i in [-0.00, 0.74]; flex_p_down marginal in hours [9, 18, 19, 20].

## Task A -- plateau vs cost_flex (all blocks, c_interface hours)

270 c_interface (TSO-classified) DN-hours over the 36 DN blocks (3 DNs x 12; each TSO hour counted once per DN). Mechanism counts: {'flex_p_down_marginal (price = mu_l + c_flex_l,p)': 199, 'bracketed by flex rows (no marginal row within Z_MARG)': 18, 'flex_p_up_marginal (price = mu_l)': 53}. Hours with a marginal P-down row: 199; with only marginal P-up rows: 53; bracketed only: 18. |LMP_TSO - marginal rows referred to the interface| max/mean (0.07759168226007773, np.float64(0.010185471719459912)). P-down hours: |LMP - (mu + c_flex)| max/mean (np.float64(4.030408221071667), np.float64(1.6852968000633564)) (no DN factor); |LMP - (day-min LMP + c_flex)| max/mean (np.float64(64.48586317288384), np.float64(34.03075873961594)) ('off-peak price + cost_flex'). P-up-only hours: |LMP - mu| max/mean (4.441749162309023, 2.2509464653984232). Bracketed-only hours: bracket width max/mean (np.float64(55.54290245618168), np.float64(42.254183793229664)), nearest row distance at the bus max/mean (np.float64(2.3368664164731303), np.float64(0.8694856161072414)).

## Task B -- throughput-weighted flatness split (W25 T4 baseline)

flatness_tw = -(1/T) sum_b w_b T_b (S_pi,b - S_LMP,b) = -(1/T) sum_b w_b T_b H_b + sum_c (1/T) sum_b w_b T_b D_c,b; T_b = sum_p (eff_ch pch + pdch/eff_dch) dt / 2; T = sum_b w_b T_b

| block | T_b (MWh-cycle) | w_r2 | market spread | LMP7 spread | flatness | H | D_ref_gen_limit (c_interface) | D_losses | D_voltage | D_congestion | c_int hours top4/bottom4 | W25 vs W28 LMP max abs | same hours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025_Spring | 0.721 | 460.00 | 133.372 | 127.243 | 6.129 | 2.259 | -3.601 | -0.258 | -0.007 | -0.000 | 1/0 | 8.1e-07 | True |
| 2025_Summer | 0.704 | 455.00 | 96.408 | 93.652 | 2.756 | 2.160 | -0.000 | -0.561 | -0.023 | -0.000 | 0/0 | 8.0e-07 | True |
| 2025_Autumn | 1.310 | 455.00 | 56.260 | 55.957 | 0.303 | 0.000 | -0.000 | -0.314 | 0.007 | -0.000 | 0/0 | 3.3e-07 | True |
| 2025_Winter | 1.340 | 455.00 | 46.276 | 42.643 | 3.634 | 0.195 | -2.483 | -0.894 | -0.048 | -0.000 | 2/0 | 6.1e-07 | True |
| 2030_Spring | 0.624 | 416.64 | 150.898 | 91.610 | 59.287 | 12.480 | -46.240 | -0.555 | -0.008 | -0.000 | 4/0 | 9.8e-07 | True |
| 2030_Summer | 0.624 | 412.11 | 118.994 | 91.067 | 27.927 | 2.677 | -24.115 | -1.094 | -0.028 | -0.000 | 4/0 | 9.7e-07 | True |
| 2030_Autumn | 1.169 | 412.11 | 63.161 | 71.028 | -7.867 | 14.161 | 21.512 | 0.501 | 0.010 | 0.000 | 1/4 | 1.2e-07 | True |
| 2030_Winter | 1.178 | 412.11 | 58.414 | 42.412 | 16.002 | 4.169 | -11.417 | -0.402 | -0.010 | -0.000 | 2/0 | 2.8e-07 | True |
| 2035_Spring | 0.560 | 377.36 | 180.918 | 110.723 | 70.195 | 13.748 | -55.798 | -0.622 | -0.018 | -0.000 | 3/0 | 1.4e-06 | True |
| 2035_Summer | 0.561 | 373.26 | 125.033 | 111.930 | 13.103 | 2.941 | -9.130 | -1.003 | -0.019 | -0.000 | 2/0 | 1.4e-06 | True |
| 2035_Autumn | 0.560 | 373.26 | 96.430 | 81.246 | 15.184 | 13.588 | -1.851 | 0.256 | -0.003 | 0.000 | 1/3 | 1.5e-06 | True |
| 2035_Winter | 0.869 | 373.26 | 47.846 | 46.839 | 1.007 | 5.691 | 4.662 | 0.021 | 0.001 | 0.000 | 2/1 | 1.7e-07 | True |

**Weighting w_r2**: T = 4296.1900 (W25 4296.1900); W25 committed (b) = -12.326163; reproduced with W25's own series -12.326163 (diff -1.1e-14).

| term (contribution to captured, EUR/MWh-cycle) | value |
|---|---|
| hour_selection_H | -5.248418 |
| D_reference_generator_limit | -6.675616 |
| D_losses | -0.385783 |
| D_congestion | -0.000000 |
| D_voltage_bounds | -0.012219 |
| D_voltage_setpoint | +0.000000 |
| D_angle | +0.000000 |
| D_admm_interface_voltage | -0.004126 |
| D_other | +0.000000 |
| D_residual | +0.000000 |
| **sum** | **-12.326162** |
| residual (sum - W25 committed) | +6.14e-07 |

Per year (contribution to the all-years term): 2025: T share 0.432, flatness -1.236 (H -0.363, D_ref_gen_limit -0.630); 2030: T share 0.345, flatness -6.184 (H -2.975, D_ref_gen_limit -3.118); 2035: T share 0.222, flatness -4.905 (H -1.910, D_ref_gen_limit -2.927)

**Weighting w_undisc**: T = 4659.6154 (W25 4659.6154); W25 committed (b) = -12.948705; reproduced with W25's own series -12.948705 (diff -2.7e-14).

| term (contribution to captured, EUR/MWh-cycle) | value |
|---|---|
| hour_selection_H | -5.510272 |
| D_reference_generator_limit | -7.045164 |
| D_losses | -0.377369 |
| D_congestion | -0.000000 |
| D_voltage_bounds | -0.011856 |
| D_voltage_setpoint | +0.000000 |
| D_angle | +0.000000 |
| D_admm_interface_voltage | -0.004043 |
| D_other | +0.000000 |
| D_residual | +0.000000 |
| **sum** | **-12.948704** |
| residual (sum - W25 committed) | +6.21e-07 |

Per year (contribution to the all-years term): 2025: T share 0.399, flatness -1.140 (H -0.335, D_ref_gen_limit -0.581); 2030: T share 0.352, flatness -6.296 (H -3.028, D_ref_gen_limit -3.175); 2035: T share 0.250, flatness -5.513 (H -2.147, D_ref_gen_limit -3.289)

Relationship: W25 (b) is defined on LMP7 at x = 0 (LMP0), not on the storage LMP at x (W25 kept LMP(x) only for the ideal-at-LMP(x) diagnostic, not in (b)); W28 decomposes the SAME quantity per block from the x = 0 capture models (s48_x0_capture, trajectory bitwise identical to a0_c7 x0). The throughput weights are the BASELINE storage schedule (W25 T4). The split is therefore exact by construction (identity S_pi - S_LMP = H - sum_c D_c holds per block to 1e-13), with the only residual being W25-vs-W28 LMP series differences (<= 1.7e-6 EUR/MWh per hour) -- reported above.

