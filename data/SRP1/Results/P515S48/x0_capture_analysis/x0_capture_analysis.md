# P5.15 Addendum 32 W27 -- Q4: what sets the TSO bus-7 marginal cost at x = 0 (zero solves)

Instance: x = 0, candidate key `8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57`; evaluation `data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0` (eval key `d2c96b1480402a3b61aca4abc188e41c6009eb582d8e6ccdd380e51651f996c7`, 132 cycles, Q gross 653,859,461.2279255); reproduction gate `data/SRP1/Results/P515S48/x0_capture_gate/gate.json` gate_pass = False (strict, not overridden); ruled PASS by `P5_15_S48_X0_CAPTURE_GATE_RULING.md` (8f9d67f2, sha256 `86b63891bb55ca5ec404df94caa6213eff079b840534ba177dbfdbc954a251cc`, verified True). Models: `data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/certified_models.pkl` sha256 `03b62593a23f748c819f18dce52c88c6a9802b8af3c52033ae09a3b88d10afce` (not committed). Mode: real.

Solve profile: armed SolveProfileGuard(permitted=()); counts {'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}; verify(0) failures []. all_checks_pass = True; failing: [].

Units / sign: LMP_b = dual(node_balance_p[b]) / baseMVA [EUR/MWh]; the row is Pg - Pd - Pflow = 0, so the dual is +d(objective)/d(load at b) -- the cost of serving one more MWh at b. pi = the market price the TSO pays its CONV generators (objective gradient / baseMVA). KKT convention: r = df/dv - sum_c lambda_c dc/dv - zL - zU; lambda = model.dual, zL = ipopt_zL_out >= 0, zU = ipopt_zU_out <= 0

## Method

Decomposition: LMP7 - pi = (y_ref/B - pi) + (y_ref/B)(z_7 - 1) + sum_g x_g[P,7]/B + residual, where M lambda_U = sum_g rhs_g is the stationarity of the network variables N in the balance / definitional multipliers U, z spans null(M) with z[P,ref] = 1, and x_g solves M x = rhs_g with x[P,ref] = 0 (least squares on the stacked system; exact when rhs_g is consistent, residuals reported). Reference bus = case9 bus 1 (type 3).

Components: **reference_generator_limit** = y_ref/B - pi: the bus-1 marginal cost minus the market price; = -(zL + zU)/B of pg[gen 1] (plus its capability rows): 0 when gen 1 is interior; negative when gen 1 sits at pmin = 0, i.e. the TN would take less conventional output than 0; **losses** = (y_ref/B)(z_7 - 1): marginal losses of serving bus 7 from bus 1; **congestion** = branch flow limits (branch_flow_limit rows); **voltage_bounds** = voltage-magnitude bounds (rows, slacks and their penalty, e/f/vmag bounds); **voltage_setpoint** = PV-bus voltage set-point rows; **angle** = angle rows (voltage_product_real >= 0, angle-difference rows) and their bounds; **admm_interface_voltage** = ADMM interface-voltage definition rows (expected_interface_vmag_def); **other** = any other row / objective term / bound touching the network variables; **residual** = (LMP7 - pi) minus the sum of the components above (KKT tolerance)

Spread: T, B = the 4 highest / 4 lowest hours of LMP7 (ties by hour); S_pi = mean of the 4 highest pi - mean of the 4 lowest pi (own hours); S_LMP = mean_T(LMP7) - mean_B(LMP7); H = S_pi - [mean_T(pi) - mean_B(pi)] >= 0 (hour selection); for each component c of LMP7 - pi: D_c = mean_T(c) - mean_B(c); identity S_LMP = S_pi - H + sum_c D_c; flatness = S_pi - S_LMP = H - sum_c D_c.

## Checks

| check | value |
|---|---|
| structure_matches_case9_all_blocks | True |
| kkt_residual_below_warn_all_blocks | True |
| no_rows_without_dual | True |
| pi_equal_across_conv_generators | True |
| pi_matches_w25_arrays | True |
| lmp7_matches_w25_identity | True |
| rotational_null_space_one_dimensional | True |
| decomposition_reconstructs_multipliers | True |
| interface_identity_residual_below_tol | True |
| decomposition_residual_below_tol | True |
| spread_identity_exact | True |
| aggregate_2030_matches_w25 | True |
| solve_profile_guard_verified_0 | True |

| block | KKT max abs residual (raw) | max abs grad | LMP7 vs W25 identity (EUR/MWh) | pi vs W25 (EUR/MWh) | max recon err (raw) | null sigma_min/max (max) | null sigma_2nd/max (min) | max |residual comp| (EUR/MWh) |
|---|---|---|---|---|---|---|---|---|
| 2025_Spring | 2.95e-04 | 5.00e+04 | 8.14e-07 | 1.42e-14 | 5.48e-03 | 2.3e-17 | 8.5e-05 | 3.79e-08 |
| 2025_Summer | 2.13e-05 | 5.00e+04 | 8.02e-07 | 1.42e-14 | 1.85e-05 | 1.8e-17 | 8.5e-05 | 2.10e-10 |
| 2025_Autumn | 4.55e-04 | 5.00e+04 | 3.25e-07 | 1.42e-14 | 8.03e-04 | 1.6e-17 | 8.6e-05 | 2.16e-08 |
| 2025_Winter | 9.49e-05 | 5.00e+04 | 6.10e-07 | 1.42e-14 | 8.02e-05 | 1.9e-17 | 8.5e-05 | 2.33e-09 |
| 2030_Spring | 6.62e-05 | 5.00e+04 | 9.79e-07 | 7.11e-15 | 1.18e-03 | 3.1e-17 | 8.4e-05 | 7.18e-09 |
| 2030_Summer | 2.56e-04 | 5.00e+04 | 9.66e-07 | 2.84e-14 | 1.25e-04 | 2.5e-17 | 8.5e-05 | 1.14e-09 |
| 2030_Autumn | 2.17e-06 | 5.00e+04 | 1.16e-07 | 1.42e-14 | 4.29e-07 | 2.5e-17 | 8.6e-05 | 1.22e-12 |
| 2030_Winter | 5.70e-04 | 5.00e+04 | 2.83e-07 | 1.42e-14 | 1.34e-03 | 2.4e-17 | 8.6e-05 | 7.82e-08 |
| 2035_Spring | 4.15e-03 | 5.00e+04 | 1.38e-06 | 5.68e-14 | 8.68e-03 | 1.8e-17 | 8.5e-05 | 7.98e-08 |
| 2035_Summer | 1.58e-03 | 5.00e+04 | 1.44e-06 | 2.84e-14 | 2.63e-07 | 2.0e-17 | 8.5e-05 | 3.03e-12 |
| 2035_Autumn | 4.29e-04 | 5.00e+04 | 1.49e-06 | 2.84e-14 | 2.01e-04 | 1.7e-17 | 8.6e-05 | 2.22e-09 |
| 2035_Winter | 3.36e-05 | 5.00e+04 | 1.67e-07 | 1.42e-14 | 6.12e-04 | 1.8e-17 | 8.6e-05 | 4.02e-09 |

## 2030: top-4 and bottom-4 hours of the bus-7 marginal cost, per day

### 2030_Spring

Market 4 h spread 150.90; LMP7 4 h spread 91.61; flatness 59.29 EUR/MWh. LMP7 top-4 hours [7, 20, 21, 22], bottom-4 [14, 15, 16, 17]; pi top-4 [7, 21, 22, 24], bottom-4 [14, 15, 16, 17]. penalty_gen_curtailment = 0.0.

| set | hour | pi | LMP7 | LMP7-pi | price setter | marginal gens (interior) | gens at a bound | curtailed RES (MW) | active branch limits (loading, dual raw) | active voltage rows (dual raw) | losses MW | interface dP 5/7/9 MW | withdrawal 5/7/9 MW |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top | 7 | 152.60 | 100.88 | -51.72 | c_interface | - | G1@1 at_pmin (+51.29), G2@2 at_pmin (+51.77), G3@3 at_pmin (+51.89), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05), G7@4 at_available (-0.05), G8@6 at_available (-0.05), G9@8 at_available (-0.05) | - | - | bus 1 V=0.9000 lower=3.3 | 0.24 | -36.1 / -40.3 / -41.7 | 15.2 / 8.7 / 9.3 |
| top | 20 | 128.94 | 100.25 | -28.69 | c_interface | - | G1@1 at_pmin (+28.15), G2@2 at_pmin (+28.72), G3@3 at_pmin (+28.97), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05), G7@4 at_available (-0.06), G8@6 at_available (-0.05), G9@8 at_available (-0.06) | - | - | - | 0.26 | -29.4 / -18.0 / -29.2 | 20.4 / 12.8 / 14.3 |
| top | 21 | 161.27 | 106.20 | -55.06 | c_interface | - | G1@1 at_pmin (+54.55), G2@2 at_pmin (+55.10), G3@3 at_pmin (+55.35), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05) | - | - | - | 0.23 | -35.2 / -24.9 / -36.9 | 18.3 / 12.5 / 12.4 |
| top | 22 | 152.21 | 100.76 | -51.45 | c_interface | - | G1@1 at_pmin (+50.97), G2@2 at_pmin (+51.50), G3@3 at_pmin (+51.72), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05) | - | - | bus 1 V=0.9000 lower=1.3 | 0.23 | -31.0 / -21.2 / -33.9 | 17.6 / 12.1 / 11.5 |
| bottom | 14 | 14.37 | 14.49 | 0.12 | a_conv_interior | G1@1 CONV 100.9, G2@2 CONV 50.2, G3@3 CONV 35.1 | G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05), G7@4 at_available (-0.05), G8@6 at_available (-0.05), G9@8 at_available (-0.05) | - | - | bus 9 V=1.1000 upper=-11.3 | 1.67 | 57.9 / 59.3 / 67.4 | 85.1 / 99.7 / 81.0 |
| bottom | 15 | 3.69 | 3.72 | 0.03 | a_conv_interior | G1@1 CONV 77.9, G2@2 CONV 41.7, G3@3 CONV 25.2 | G4@4 at_available (-0.04), G5@6 at_available (-0.04), G6@8 at_available (-0.04), G7@4 at_available (-0.04), G8@6 at_available (-0.04), G9@8 at_available (-0.04) | - | - | bus 9 V=1.1000 upper=-2.0 | 1.29 | 53.1 / 63.8 / 61.3 | 72.6 / 94.0 / 64.7 |
| bottom | 16 | 7.90 | 7.96 | 0.05 | a_conv_interior | G1@1 CONV 66.3, G2@2 CONV 32.2, G3@3 CONV 15.1 | G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05), G7@4 at_available (-0.05), G8@6 at_available (-0.05), G9@8 at_available (-0.05) | - | - | bus 9 V=1.1000 upper=-3.0 | 1.03 | 45.0 / 55.0 / 53.4 | 64.2 / 80.5 / 57.4 |
| bottom | 17 | 15.39 | 15.48 | 0.10 | a_conv_interior | G1@1 CONV 74.0, G2@2 CONV 31.6, G3@3 CONV 14.6 | G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05), G7@4 at_available (-0.05), G8@6 at_available (-0.05), G9@8 at_available (-0.05) | - | - | bus 9 V=1.1000 upper=-6.8 | 1.11 | 47.9 / 55.2 / 54.4 | 70.8 / 75.0 / 60.7 |

Decomposition of LMP7 - pi in these hours (EUR/MWh):

| set | hour | LMP7-pi | reference_generator_limit | losses | congestion | voltage_bounds | voltage_setpoint | angle | admm_interface_voltage | other | residual | loss factor z7 | largest row shares |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top | 7 | -51.72 | -51.288 | -0.432 | -0.000 | -0.001 | 0.000 | 0.000 | -0.000 | 0.000 | 0.000 | 0.99574 | - |
| top | 20 | -28.69 | -28.152 | -0.541 | -0.000 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.99463 | - |
| top | 21 | -55.06 | -54.549 | -0.515 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.99517 | - |
| top | 22 | -51.45 | -50.973 | -0.481 | -0.000 | -0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.99525 | - |
| bottom | 14 | 0.12 | -0.000 | 0.095 | 0.000 | 0.017 | 0.000 | -0.000 | 0.008 | 0.000 | 0.000 | 1.00658 | voltage_magnitude_upper_cons[8,0,0,13] +0.02 |
| bottom | 15 | 0.03 | -0.000 | 0.025 | 0.000 | 0.003 | 0.000 | -0.000 | 0.001 | 0.000 | 0.000 | 1.00675 | - |
| bottom | 16 | 0.05 | -0.000 | 0.047 | 0.000 | 0.004 | 0.000 | -0.000 | 0.002 | 0.000 | 0.000 | 1.00599 | - |
| bottom | 17 | 0.10 | -0.000 | 0.084 | 0.000 | 0.008 | 0.000 | -0.000 | 0.004 | 0.000 | 0.000 | 1.00544 | - |

Interface-side identity at bus 7 (EUR/MWh): stationarity of interface_delta_p at bus 7 (free; enters node_balance_p[7] with -1): LMP7 = -(df/d delta)/B + sum_(c != balance) lambda_c (dc/d delta)/B + (zL + zU)/B, i.e. LMP7 - pi = [-(df/d delta)/B - pi] (objective terms on delta other than the settlement at pi; 0 when the settlement -pi*B*delta is the only one) + [ADMM interface consensus: the expected_interface_pf_p_def row] + [delta bound multipliers] + residual

| set | hour | LMP7-pi | objective terms on delta minus pi | ADMM consensus rows | delta bound | residual | delta_p MW |
|---|---|---|---|---|---|---|---|
| top | 7 | -51.721 | 0.000 | expected_interface_pf_p_def -51.721 | 0.000 | 2.0e-15 | -40.34 |
| top | 20 | -28.693 | 0.000 | expected_interface_pf_p_def -28.693 | 0.000 | -1.6e-14 | -18.02 |
| top | 21 | -55.064 | 0.000 | expected_interface_pf_p_def -55.064 | 0.000 | 1.2e-14 | -24.94 |
| top | 22 | -51.455 | 0.000 | expected_interface_pf_p_def -51.455 | 0.000 | -1.1e-14 | -21.23 |
| bottom | 14 | 0.120 | 0.000 | expected_interface_pf_p_def +0.120 | -0.000 | 4.0e-16 | 59.35 |
| bottom | 15 | 0.029 | 0.000 | expected_interface_pf_p_def +0.029 | -0.000 | 2.6e-16 | 63.84 |
| bottom | 16 | 0.053 | 0.000 | expected_interface_pf_p_def +0.053 | -0.000 | 5.6e-16 | 55.02 |
| bottom | 17 | 0.096 | 0.000 | expected_interface_pf_p_def +0.096 | -0.000 | -1.6e-15 | 55.20 |

Price setter per hour (all 24): h1 c_interface; h2 c_interface; h3 c_interface; h4 c_interface; h5 c_interface; h6 c_interface; h7 c_interface; h8 c_interface; h9 a_conv_interior; h10 a_conv_interior; h11 a_conv_interior; h12 a_conv_interior; h13 a_conv_interior; h14 a_conv_interior; h15 a_conv_interior; h16 a_conv_interior; h17 a_conv_interior; h18 a_conv_interior; h19 c_interface; h20 c_interface; h21 c_interface; h22 c_interface; h23 c_interface; h24 c_interface

Spread decomposition (EUR/MWh): hour_selection_H +12.480, minus_D_reference_generator_limit +46.240, minus_D_losses +0.555, minus_D_congestion +0.000, minus_D_voltage_bounds +0.008, minus_D_voltage_setpoint -0.000, minus_D_angle -0.000, minus_D_admm_interface_voltage +0.004, minus_D_other -0.000, minus_D_residual +0.000; sum = flatness +59.287 (identity error 2.8e-14).

### 2030_Summer

Market 4 h spread 118.99; LMP7 4 h spread 91.07; flatness 27.93 EUR/MWh. LMP7 top-4 hours [20, 21, 22, 23], bottom-4 [13, 14, 15, 16]; pi top-4 [1, 21, 22, 23], bottom-4 [13, 14, 15, 16]. penalty_gen_curtailment = 0.0.

| set | hour | pi | LMP7 | LMP7-pi | price setter | marginal gens (interior) | gens at a bound | curtailed RES (MW) | active branch limits (loading, dual raw) | active voltage rows (dual raw) | losses MW | interface dP 5/7/9 MW | withdrawal 5/7/9 MW |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top | 20 | 146.52 | 143.41 | -3.11 | c_interface | - | G1@1 at_pmin (+2.38), G2@2 at_pmin (+3.31), G3@3 at_pmin (+3.46), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05), G7@4 at_available (-0.13), G8@6 at_available (-0.06), G9@8 at_available (-0.05) | - | - | - | 0.23 | -30.0 / -28.4 / -33.8 | 18.2 / 18.1 / 13.6 |
| top | 21 | 183.24 | 143.06 | -40.18 | c_interface | - | G1@1 at_pmin (+39.43), G2@2 at_pmin (+40.36), G3@3 at_pmin (+40.51), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05) | - | - | - | 0.25 | -30.1 / -30.9 / -35.2 | 20.3 / 19.5 / 15.4 |
| top | 22 | 165.71 | 135.84 | -29.87 | c_interface | - | G1@1 at_pmin (+29.12), G2@2 at_pmin (+30.07), G3@3 at_pmin (+30.19), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05) | - | - | - | 0.26 | -22.9 / -23.6 / -27.1 | 20.1 / 20.0 / 15.1 |
| top | 23 | 158.90 | 132.59 | -26.31 | c_interface | - | G1@1 at_pmin (+25.52), G2@2 at_pmin (+26.50), G3@3 at_pmin (+26.59), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05) | - | - | - | 0.31 | -27.4 / -31.7 / -32.9 | 19.9 / 18.6 / 15.2 |
| bottom | 13 | 54.66 | 55.11 | 0.46 | a_conv_interior | G1@1 CONV 71.0, G2@2 CONV 41.5, G3@3 CONV 27.1 | G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05), G7@4 at_available (-0.05), G8@6 at_available (-0.05), G9@8 at_available (-0.05) | - | - | bus 9 V=1.1000 upper=-33.6 | 1.27 | 61.0 / 56.3 / 70.2 | 74.1 / 99.3 / 56.6 |
| bottom | 14 | 47.25 | 47.65 | 0.40 | a_conv_interior | G1@1 CONV 63.7, G2@2 CONV 39.0, G3@3 CONV 23.6 | G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05), G7@4 at_available (-0.05), G8@6 at_available (-0.05), G9@8 at_available (-0.05) | - | - | bus 9 V=1.1000 upper=-25.8 | 1.18 | 57.8 / 57.7 / 65.7 | 69.9 / 99.4 / 51.8 |
| bottom | 15 | 44.44 | 44.81 | 0.37 | a_conv_interior | G1@1 CONV 50.1, G2@2 CONV 34.7, G3@3 CONV 18.0 | G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05), G7@4 at_available (-0.05), G8@6 at_available (-0.05), G9@8 at_available (-0.05) | - | - | bus 9 V=1.1000 upper=-18.7 | 0.99 | 53.0 / 63.5 / 60.6 | 60.3 / 98.1 / 43.3 |
| bottom | 16 | 42.76 | 43.07 | 0.31 | a_conv_interior | G1@1 CONV 44.8, G2@2 CONV 27.1, G3@3 CONV 13.3 | G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05), G7@4 at_available (-0.05), G8@6 at_available (-0.05), G9@8 at_available (-0.05) | - | - | bus 9 V=1.1000 upper=-13.6 | 0.83 | 46.0 / 55.0 / 52.5 | 55.4 / 86.4 / 38.8 |

Decomposition of LMP7 - pi in these hours (EUR/MWh):

| set | hour | LMP7-pi | reference_generator_limit | losses | congestion | voltage_bounds | voltage_setpoint | angle | admm_interface_voltage | other | residual | loss factor z7 | largest row shares |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top | 20 | -3.11 | -2.384 | -0.730 | -0.000 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.99494 | - |
| top | 21 | -40.18 | -39.433 | -0.747 | -0.000 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.000 | 0.99481 | - |
| top | 22 | -29.87 | -29.122 | -0.748 | -0.000 | -0.000 | 0.000 | 0.000 | -0.000 | 0.000 | 0.000 | 0.99452 | - |
| top | 23 | -26.31 | -25.520 | -0.787 | -0.000 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.99410 | - |
| bottom | 13 | 0.46 | -0.000 | 0.393 | 0.000 | 0.043 | 0.000 | -0.000 | 0.021 | 0.000 | 0.000 | 1.00719 | voltage_magnitude_upper_cons[8,0,0,12] +0.04; expected_interface_vmag_def[2,12] +0.02 |
| bottom | 14 | 0.40 | -0.000 | 0.347 | 0.000 | 0.032 | 0.000 | -0.000 | 0.016 | 0.000 | -0.000 | 1.00735 | voltage_magnitude_upper_cons[8,0,0,13] +0.03; expected_interface_vmag_def[2,13] +0.02 |
| bottom | 15 | 0.37 | -0.000 | 0.334 | 0.000 | 0.021 | 0.000 | -0.000 | 0.011 | 0.000 | 0.000 | 1.00753 | voltage_magnitude_upper_cons[8,0,0,14] +0.02; expected_interface_vmag_def[2,14] +0.01 |
| bottom | 16 | 0.31 | -0.000 | 0.288 | 0.000 | 0.014 | 0.000 | -0.000 | 0.007 | 0.000 | 0.000 | 1.00675 | voltage_magnitude_upper_cons[8,0,0,15] +0.01 |

Interface-side identity at bus 7 (EUR/MWh): stationarity of interface_delta_p at bus 7 (free; enters node_balance_p[7] with -1): LMP7 = -(df/d delta)/B + sum_(c != balance) lambda_c (dc/d delta)/B + (zL + zU)/B, i.e. LMP7 - pi = [-(df/d delta)/B - pi] (objective terms on delta other than the settlement at pi; 0 when the settlement -pi*B*delta is the only one) + [ADMM interface consensus: the expected_interface_pf_p_def row] + [delta bound multipliers] + residual

| set | hour | LMP7-pi | objective terms on delta minus pi | ADMM consensus rows | delta bound | residual | delta_p MW |
|---|---|---|---|---|---|---|---|
| top | 20 | -3.114 | 0.000 | expected_interface_pf_p_def -3.114 | 0.000 | 1.9e-14 | -28.40 |
| top | 21 | -40.179 | 0.000 | expected_interface_pf_p_def -40.179 | 0.000 | -1.9e-15 | -30.91 |
| top | 22 | -29.870 | 0.000 | expected_interface_pf_p_def -29.870 | 0.000 | 5.3e-14 | -23.61 |
| top | 23 | -26.308 | 0.000 | expected_interface_pf_p_def -26.308 | 0.000 | 3.1e-14 | -31.65 |
| bottom | 13 | 0.457 | 0.000 | expected_interface_pf_p_def +0.457 | -0.000 | -4.3e-15 | 56.32 |
| bottom | 14 | 0.395 | 0.000 | expected_interface_pf_p_def +0.395 | -0.000 | -2.2e-15 | 57.72 |
| bottom | 15 | 0.366 | 0.000 | expected_interface_pf_p_def +0.366 | -0.000 | -8.2e-15 | 63.47 |
| bottom | 16 | 0.310 | 0.000 | expected_interface_pf_p_def +0.310 | -0.000 | 8.8e-15 | 55.04 |

Price setter per hour (all 24): h1 c_interface; h2 c_interface; h3 c_interface; h4 c_interface; h5 c_interface; h6 a_conv_interior; h7 a_conv_interior; h8 c_interface; h9 a_conv_interior; h10 c_interface; h11 a_conv_interior; h12 a_conv_interior; h13 a_conv_interior; h14 a_conv_interior; h15 a_conv_interior; h16 a_conv_interior; h17 a_conv_interior; h18 a_conv_interior; h19 a_conv_interior; h20 c_interface; h21 c_interface; h22 c_interface; h23 c_interface; h24 c_interface

Spread decomposition (EUR/MWh): hour_selection_H +2.677, minus_D_reference_generator_limit +24.115, minus_D_losses +1.094, minus_D_congestion +0.000, minus_D_voltage_bounds +0.028, minus_D_voltage_setpoint -0.000, minus_D_angle -0.000, minus_D_admm_interface_voltage +0.014, minus_D_other -0.000, minus_D_residual -0.000; sum = flatness +27.927 (identity error 1.4e-14).

### 2030_Autumn

Market 4 h spread 63.16; LMP7 4 h spread 71.03; flatness -7.87 EUR/MWh. LMP7 top-4 hours [18, 19, 20, 21], bottom-4 [10, 11, 12, 13]; pi top-4 [18, 19, 20, 21], bottom-4 [3, 4, 5, 6]. penalty_gen_curtailment = 0.0.

| set | hour | pi | LMP7 | LMP7-pi | price setter | marginal gens (interior) | gens at a bound | curtailed RES (MW) | active branch limits (loading, dual raw) | active voltage rows (dual raw) | losses MW | interface dP 5/7/9 MW | withdrawal 5/7/9 MW |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top | 18 | 150.16 | 150.61 | 0.45 | a_conv_interior | G1@1 CONV 60.5, G2@2 CONV 15.5 | G3@3 at_pmin (+0.03), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05), G7@4 at_available (-1.73), G8@6 at_available (-0.12), G9@8 at_available (-0.12) | - | - | bus 9 V=1.1000 upper=-26.6 | 0.63 | -0.2 / -0.2 / -0.8 | 52.8 / 37.3 / 48.8 |
| top | 19 | 175.07 | 169.07 | -6.01 | c_interface | - | G1@1 at_pmin (+4.78), G2@2 at_pmin (+6.05), G3@3 at_pmin (+6.53), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05) | - | - | bus 5 V=1.0980 upper=-0.0, bus 7 V=1.1000 upper=-5.2, bus 9 V=1.0996 upper=-0.1 | 0.40 | -19.5 / -12.6 / -21.3 | 28.0 / 17.6 / 21.1 |
| top | 20 | 166.66 | 166.64 | -0.03 | a_conv_interior | G1@1 CONV 38.1 | G2@2 at_pmin (+0.22), G3@3 at_pmin (+0.65), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05) | - | - | bus 9 V=1.1000 upper=-12.6 | 0.49 | -2.6 / -1.6 / -3.1 | 43.4 / 27.0 / 36.5 |
| top | 21 | 143.07 | 143.23 | 0.16 | a_conv_interior | G1@1 CONV 46.2 | G2@2 at_pmin (+0.01), G3@3 at_pmin (+0.41), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05) | - | - | bus 9 V=1.1000 upper=-16.1 | 0.54 | -0.2 / -0.1 / -0.1 | 47.1 / 28.5 / 40.4 |
| bottom | 10 | 127.85 | 86.36 | -41.49 | c_interface | - | G1@1 at_pmin (+40.77), G2@2 at_pmin (+41.55), G3@3 at_pmin (+41.80), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05), G7@4 at_available (-0.05), G8@6 at_available (-0.05), G9@8 at_available (-0.05) | - | - | bus 9 V=1.1000 upper=-8.2 | 0.46 | 1.9 / 0.6 / 0.0 | 35.2 / 22.6 / 26.3 |
| bottom | 11 | 109.70 | 86.28 | -23.42 | c_interface | - | G1@1 at_pmin (+22.73), G2@2 at_pmin (+23.49), G3@3 at_pmin (+23.73), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05), G7@4 at_available (-0.05), G8@6 at_available (-0.05), G9@8 at_available (-0.05) | - | - | bus 9 V=1.1000 upper=-8.4 | 0.47 | 5.0 / 3.1 / 5.4 | 36.4 / 23.4 / 25.8 |
| bottom | 12 | 101.19 | 86.39 | -14.80 | c_interface | - | G1@1 at_pmin (+14.14), G2@2 at_pmin (+14.87), G3@3 at_pmin (+15.12), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05), G7@4 at_available (-0.05), G8@6 at_available (-0.05), G9@8 at_available (-0.05) | - | - | bus 9 V=1.1000 upper=-8.7 | 0.45 | 2.6 / 1.0 / 7.6 | 36.6 / 24.5 / 24.0 |
| bottom | 13 | 100.22 | 86.40 | -13.82 | c_interface | - | G1@1 at_pmin (+13.18), G2@2 at_pmin (+13.89), G3@3 at_pmin (+14.13), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05), G7@4 at_available (-0.05), G8@6 at_available (-0.05), G9@8 at_available (-0.05) | - | - | bus 9 V=1.1000 upper=-8.3 | 0.45 | 4.2 / 0.8 / 7.3 | 36.3 / 24.3 / 24.5 |

Decomposition of LMP7 - pi in these hours (EUR/MWh):

| set | hour | LMP7-pi | reference_generator_limit | losses | congestion | voltage_bounds | voltage_setpoint | angle | admm_interface_voltage | other | residual | loss factor z7 | largest row shares |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top | 18 | 0.45 | -0.000 | 0.417 | 0.000 | 0.024 | 0.000 | -0.000 | 0.012 | 0.000 | 0.000 | 1.00277 | voltage_magnitude_upper_cons[8,0,0,17] +0.02; expected_interface_vmag_def[2,17] +0.01 |
| top | 19 | -6.01 | -4.779 | -1.222 | -0.000 | -0.002 | 0.000 | 0.000 | -0.001 | 0.000 | 0.000 | 0.99282 | - |
| top | 20 | -0.03 | -0.000 | -0.041 | 0.000 | 0.008 | 0.000 | -0.000 | 0.005 | 0.000 | 0.000 | 0.99975 | - |
| top | 21 | 0.16 | -0.000 | 0.143 | 0.000 | 0.012 | 0.000 | -0.000 | 0.006 | 0.000 | -0.000 | 1.00100 | voltage_magnitude_upper_cons[8,0,0,20] +0.01 |
| bottom | 10 | -41.49 | -40.774 | -0.720 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.000 | 0.99173 | - |
| bottom | 11 | -23.42 | -22.732 | -0.690 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.99207 | - |
| bottom | 12 | -14.80 | -14.140 | -0.659 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.000 | 0.99242 | - |
| bottom | 13 | -13.82 | -13.180 | -0.639 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.99265 | - |

Interface-side identity at bus 7 (EUR/MWh): stationarity of interface_delta_p at bus 7 (free; enters node_balance_p[7] with -1): LMP7 = -(df/d delta)/B + sum_(c != balance) lambda_c (dc/d delta)/B + (zL + zU)/B, i.e. LMP7 - pi = [-(df/d delta)/B - pi] (objective terms on delta other than the settlement at pi; 0 when the settlement -pi*B*delta is the only one) + [ADMM interface consensus: the expected_interface_pf_p_def row] + [delta bound multipliers] + residual

| set | hour | LMP7-pi | objective terms on delta minus pi | ADMM consensus rows | delta bound | residual | delta_p MW |
|---|---|---|---|---|---|---|---|
| top | 18 | 0.453 | 0.000 | expected_interface_pf_p_def +0.453 | 0.000 | 1.1e-15 | -0.18 |
| top | 19 | -6.005 | 0.000 | expected_interface_pf_p_def -6.005 | 0.000 | 2.9e-14 | -12.58 |
| top | 20 | -0.029 | 0.000 | expected_interface_pf_p_def -0.029 | 0.000 | 3.4e-14 | -1.56 |
| top | 21 | 0.161 | 0.000 | expected_interface_pf_p_def +0.161 | 0.000 | 2.7e-14 | -0.15 |
| bottom | 10 | -41.494 | 0.000 | expected_interface_pf_p_def -41.494 | -0.000 | -3.4e-15 | 0.55 |
| bottom | 11 | -23.422 | 0.000 | expected_interface_pf_p_def -23.422 | -0.000 | 4.0e-16 | 3.13 |
| bottom | 12 | -14.799 | 0.000 | expected_interface_pf_p_def -14.799 | -0.000 | 8.7e-15 | 1.03 |
| bottom | 13 | -13.819 | 0.000 | expected_interface_pf_p_def -13.819 | -0.000 | 1.0e-14 | 0.83 |

Price setter per hour (all 24): h1 c_interface; h2 a_conv_interior; h3 c_interface; h4 a_conv_interior; h5 c_interface; h6 c_interface; h7 a_conv_interior; h8 a_conv_interior; h9 a_conv_interior; h10 c_interface; h11 c_interface; h12 c_interface; h13 c_interface; h14 a_conv_interior; h15 a_conv_interior; h16 a_conv_interior; h17 a_conv_interior; h18 a_conv_interior; h19 c_interface; h20 a_conv_interior; h21 a_conv_interior; h22 a_conv_interior; h23 a_conv_interior; h24 a_conv_interior

Spread decomposition (EUR/MWh): hour_selection_H +14.161, minus_D_reference_generator_limit -21.512, minus_D_losses -0.501, minus_D_congestion -0.000, minus_D_voltage_bounds -0.010, minus_D_voltage_setpoint -0.000, minus_D_angle +0.000, minus_D_admm_interface_voltage -0.005, minus_D_other -0.000, minus_D_residual +0.000; sum = flatness -7.867 (identity error 1.4e-14).

### 2030_Winter

Market 4 h spread 58.41; LMP7 4 h spread 42.41; flatness 16.00 EUR/MWh. LMP7 top-4 hours [9, 19, 20, 21], bottom-4 [4, 5, 6, 14]; pi top-4 [18, 19, 20, 21], bottom-4 [4, 5, 13, 14]. penalty_gen_curtailment = 0.0.

| set | hour | pi | LMP7 | LMP7-pi | price setter | marginal gens (interior) | gens at a bound | curtailed RES (MW) | active branch limits (loading, dual raw) | active voltage rows (dual raw) | losses MW | interface dP 5/7/9 MW | withdrawal 5/7/9 MW |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top | 9 | 130.75 | 130.50 | -0.25 | a_conv_interior | G1@1 CONV 36.8 | G2@2 at_pmin (+0.44), G3@3 at_pmin (+0.44), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05), G7@4 at_available (-0.05), G8@6 at_available (-0.05), G9@8 at_available (-0.05) | - | - | bus 9 V=1.1000 upper=-14.4 | 0.50 | -1.3 / -0.1 / -0.8 | 46.0 / 17.4 / 42.1 |
| top | 19 | 161.38 | 134.88 | -26.51 | c_interface | - | G1@1 at_pmin (+25.59), G2@2 at_pmin (+26.58), G3@3 at_pmin (+26.77), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05) | - | - | bus 5 V=1.0995 upper=-0.1, bus 7 V=1.1000 upper=-1.4, bus 9 V=1.0985 upper=-0.0 | 0.39 | -23.3 / -16.4 / -27.9 | 26.0 / 14.1 / 22.2 |
| top | 20 | 160.53 | 139.53 | -21.00 | c_interface | - | G1@1 at_pmin (+20.07), G2@2 at_pmin (+21.09), G3@3 at_pmin (+21.31), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05) | - | - | bus 5 V=1.1000 upper=-1.2, bus 7 V=1.1000 upper=-1.0, bus 9 V=1.0992 upper=-0.0 | 0.40 | -17.9 / -9.1 / -23.0 | 26.6 / 16.2 / 22.4 |
| top | 21 | 138.34 | 138.57 | 0.23 | a_conv_interior | G1@1 CONV 48.7, G2@2 CONV 1.7 | G3@3 at_pmin (+0.16), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05) | - | - | bus 5 V=1.1000 upper=-14.0, bus 9 V=1.0988 upper=-0.0 | 0.56 | -0.2 / -0.1 / -0.2 | 44.2 / 26.0 / 45.4 |
| bottom | 4 | 93.37 | 93.43 | 0.06 | a_conv_interior | G1@1 CONV 48.6 | G2@2 at_pmin (+0.07), G3@3 at_pmin (+0.17), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05) | - | - | bus 5 V=1.1000 upper=-11.9 | 0.73 | 16.9 / 13.1 / 13.4 | 47.5 / 22.2 / 45.3 |
| bottom | 5 | 93.17 | 93.35 | 0.17 | a_conv_interior | G1@1 CONV 57.6, G2@2 CONV 2.8 | G3@3 at_pmin (+0.03), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05) | - | - | bus 5 V=1.1000 upper=-16.7 | 0.80 | 22.4 / 19.2 / 21.2 | 51.9 / 24.4 / 50.5 |
| bottom | 6 | 95.12 | 94.33 | -0.79 | a_conv_interior | G1@1 CONV 7.2 | G2@2 at_pmin (+0.83), G3@3 at_pmin (+0.92), G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05) | - | - | bus 5 V=1.1000 upper=-8.2 | 0.61 | 2.0 / 4.8 / 1.3 | 33.9 / 9.7 / 32.0 |
| bottom | 14 | 92.37 | 92.73 | 0.36 | a_conv_interior | G1@1 CONV 73.9, G2@2 CONV 18.5, G3@3 CONV 14.5 | G4@4 at_available (-0.05), G5@6 at_available (-0.05), G6@8 at_available (-0.05), G7@4 at_available (-0.05), G8@6 at_available (-0.05), G9@8 at_available (-0.05) | - | - | bus 9 V=1.1000 upper=-33.4 | 0.90 | 24.9 / 15.7 / 34.5 | 64.4 / 47.3 / 61.5 |

Decomposition of LMP7 - pi in these hours (EUR/MWh):

| set | hour | LMP7-pi | reference_generator_limit | losses | congestion | voltage_bounds | voltage_setpoint | angle | admm_interface_voltage | other | residual | loss factor z7 | largest row shares |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top | 9 | -0.25 | -0.000 | -0.261 | 0.000 | 0.007 | 0.000 | -0.000 | 0.004 | 0.000 | 0.000 | 0.99801 | - |
| top | 19 | -26.51 | -25.595 | -0.912 | -0.000 | -0.001 | 0.000 | 0.000 | -0.000 | 0.000 | 0.000 | 0.99328 | - |
| top | 20 | -21.00 | -20.072 | -0.925 | -0.000 | -0.000 | 0.000 | 0.000 | -0.000 | 0.000 | 0.000 | 0.99341 | - |
| top | 21 | 0.23 | -0.000 | 0.215 | 0.000 | 0.010 | 0.000 | -0.000 | 0.005 | 0.000 | 0.000 | 1.00155 | voltage_magnitude_upper_cons[4,0,0,20] +0.01 |
| bottom | 4 | 0.06 | -0.000 | 0.046 | 0.000 | 0.008 | 0.000 | -0.000 | 0.005 | 0.000 | 0.000 | 1.00049 | - |
| bottom | 5 | 0.17 | -0.000 | 0.154 | 0.000 | 0.013 | 0.000 | -0.000 | 0.007 | 0.000 | 0.000 | 1.00165 | voltage_magnitude_upper_cons[4,0,0,4] +0.01 |
| bottom | 6 | -0.79 | -0.000 | -0.791 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.99169 | - |
| bottom | 14 | 0.36 | -0.000 | 0.314 | 0.000 | 0.035 | 0.000 | -0.000 | 0.012 | 0.000 | 0.000 | 1.00340 | voltage_magnitude_upper_cons[8,0,0,13] +0.04; expected_interface_vmag_def[2,13] +0.01 |

Interface-side identity at bus 7 (EUR/MWh): stationarity of interface_delta_p at bus 7 (free; enters node_balance_p[7] with -1): LMP7 = -(df/d delta)/B + sum_(c != balance) lambda_c (dc/d delta)/B + (zL + zU)/B, i.e. LMP7 - pi = [-(df/d delta)/B - pi] (objective terms on delta other than the settlement at pi; 0 when the settlement -pi*B*delta is the only one) + [ADMM interface consensus: the expected_interface_pf_p_def row] + [delta bound multipliers] + residual

| set | hour | LMP7-pi | objective terms on delta minus pi | ADMM consensus rows | delta bound | residual | delta_p MW |
|---|---|---|---|---|---|---|---|
| top | 9 | -0.249 | 0.000 | expected_interface_pf_p_def -0.249 | 0.000 | 8.6e-15 | -0.14 |
| top | 19 | -26.508 | 0.000 | expected_interface_pf_p_def -26.508 | 0.000 | 3.5e-14 | -16.40 |
| top | 20 | -20.998 | 0.000 | expected_interface_pf_p_def -20.998 | 0.000 | -5.7e-17 | -9.05 |
| top | 21 | 0.231 | 0.000 | expected_interface_pf_p_def +0.231 | 0.000 | 9.7e-15 | -0.14 |
| bottom | 4 | 0.059 | 0.000 | expected_interface_pf_p_def +0.059 | -0.000 | 1.5e-14 | 13.14 |
| bottom | 5 | 0.174 | 0.000 | expected_interface_pf_p_def +0.174 | -0.000 | -6.9e-15 | 19.16 |
| bottom | 6 | -0.791 | 0.000 | expected_interface_pf_p_def -0.791 | -0.000 | 1.5e-14 | 4.85 |
| bottom | 14 | 0.361 | 0.000 | expected_interface_pf_p_def +0.361 | -0.000 | 1.5e-14 | 15.70 |

Price setter per hour (all 24): h1 a_conv_interior; h2 a_conv_interior; h3 a_conv_interior; h4 a_conv_interior; h5 a_conv_interior; h6 a_conv_interior; h7 a_conv_interior; h8 a_conv_interior; h9 a_conv_interior; h10 a_conv_interior; h11 a_conv_interior; h12 a_conv_interior; h13 a_conv_interior; h14 a_conv_interior; h15 a_conv_interior; h16 a_conv_interior; h17 a_conv_interior; h18 c_interface; h19 c_interface; h20 c_interface; h21 a_conv_interior; h22 a_conv_interior; h23 a_conv_interior; h24 a_conv_interior

Spread decomposition (EUR/MWh): hour_selection_H +4.169, minus_D_reference_generator_limit +11.417, minus_D_losses +0.402, minus_D_congestion +0.000, minus_D_voltage_bounds +0.010, minus_D_voltage_setpoint -0.000, minus_D_angle -0.000, minus_D_admm_interface_voltage +0.004, minus_D_other -0.000, minus_D_residual -0.000; sum = flatness +16.002 (identity error 0.0e+00).

## Spread decomposition, all blocks (supplementary for 2025 / 2035)

| block | market spread | LMP7 spread | flatness | H (hour selection) | -D reference_generator_limit | -D losses | -D congestion | -D voltage_bounds | -D voltage_setpoint | -D angle | -D admm_interface_voltage | -D other | -D residual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025_Spring | 133.372 | 127.243 | 6.129 | 2.259 | 3.601 | 0.258 | 0.000 | 0.007 | -0.000 | -0.000 | 0.003 | -0.000 | -0.000 |
| 2025_Summer | 96.408 | 93.652 | 2.756 | 2.160 | 0.000 | 0.561 | 0.000 | 0.023 | -0.000 | -0.000 | 0.012 | -0.000 | 0.000 |
| 2025_Autumn | 56.260 | 55.957 | 0.303 | 0.000 | 0.000 | 0.314 | 0.000 | -0.007 | -0.000 | -0.000 | -0.004 | -0.000 | 0.000 |
| 2025_Winter | 46.276 | 42.643 | 3.634 | 0.195 | 2.483 | 0.894 | 0.000 | 0.048 | -0.000 | -0.000 | 0.013 | -0.000 | -0.000 |
| 2030_Spring | 150.898 | 91.610 | 59.287 | 12.480 | 46.240 | 0.555 | 0.000 | 0.008 | -0.000 | -0.000 | 0.004 | -0.000 | 0.000 |
| 2030_Summer | 118.994 | 91.067 | 27.927 | 2.677 | 24.115 | 1.094 | 0.000 | 0.028 | -0.000 | -0.000 | 0.014 | -0.000 | -0.000 |
| 2030_Autumn | 63.161 | 71.028 | -7.867 | 14.161 | -21.512 | -0.501 | -0.000 | -0.010 | -0.000 | 0.000 | -0.005 | -0.000 | 0.000 |
| 2030_Winter | 58.414 | 42.412 | 16.002 | 4.169 | 11.417 | 0.402 | 0.000 | 0.010 | -0.000 | -0.000 | 0.004 | -0.000 | -0.000 |
| 2035_Spring | 180.918 | 110.723 | 70.195 | 13.748 | 55.798 | 0.622 | 0.000 | 0.018 | -0.000 | -0.000 | 0.009 | -0.000 | -0.000 |
| 2035_Summer | 125.033 | 111.930 | 13.103 | 2.941 | 9.130 | 1.003 | 0.000 | 0.019 | -0.000 | -0.000 | 0.009 | -0.000 | -0.000 |
| 2035_Autumn | 96.430 | 81.246 | 15.184 | 13.588 | 1.851 | -0.256 | -0.000 | 0.003 | -0.000 | 0.000 | -0.002 | -0.000 | -0.000 |
| 2035_Winter | 47.846 | 46.839 | 1.007 | 5.691 | -4.662 | -0.021 | -0.000 | -0.001 | -0.000 | 0.000 | -0.000 | -0.000 | -0.000 |

Aggregates (day-weighted with W25 block weights; flatness = H - sum_c D_c):

| scope | weights | market | LMP7 | ratio | flatness | H | -D reference_generator_limit | -D losses | -D congestion | -D voltage_bounds | -D voltage_setpoint | -D angle | -D admm_interface_voltage | -D other | -D residual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2030, day-weighted (undiscounted) | w_undisc | 98.012 | 74.078 | 0.7558 | 23.934 | 8.383 | 15.150 | 0.388 | 0.000 | 0.009 | -0.000 | -0.000 | 0.004 | -0.000 | -0.000 |
| 2030, model block weight (2 %) | w_r2 | 98.012 | 74.078 | 0.7558 | 23.934 | 8.383 | 15.150 | 0.388 | 0.000 | 0.009 | -0.000 | -0.000 | 0.004 | -0.000 | -0.000 |
| all years, day-weighted (undiscounted) | w_undisc | 97.991 | 80.610 | 0.8226 | 17.381 | 6.182 | 10.772 | 0.411 | 0.000 | 0.012 | -0.000 | -0.000 | 0.005 | -0.000 | -0.000 |
| all years, model block weight (2 %) | w_r2 | 97.018 | 80.365 | 0.8284 | 16.653 | 5.919 | 10.300 | 0.416 | 0.000 | 0.012 | -0.000 | -0.000 | 0.005 | -0.000 | -0.000 |
| 2025, day-weighted (undiscounted) | w_undisc | 83.217 | 80.003 | 0.9614 | 3.213 | 1.157 | 1.527 | 0.506 | 0.000 | 0.018 | -0.000 | -0.000 | 0.006 | -0.000 | -0.000 |
| 2035, day-weighted (undiscounted) | w_undisc | 112.744 | 87.748 | 0.7783 | 24.996 | 9.005 | 15.640 | 0.338 | 0.000 | 0.010 | -0.000 | -0.000 | 0.004 | -0.000 | -0.000 |

