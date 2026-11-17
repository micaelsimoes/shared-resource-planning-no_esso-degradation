# Step 6 tables — FROZEN v1 (`frozen_step6_tables_v1_590088fe.json`)

sha256 of the frozen JSON: `590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4`. Predecessor: `data/SRP1/Results/P515S53/w157_step6_tables_a64/w157_step6_tables.json` (sha256 `e08a01614f6af19b6ca0e590223be270ce84b87157a473f884b244d7e5a2a636`, commit `b913ea94`). Package: `P5_15_STEP6_PACKAGE.md` at `e3437284`. Closing reads: W159 `62749030`. Built by `p515_s53_w160_step6_freeze_export.py`; zero solves, pickle blocked. The JSON keeps full precision; the tables below are rounded as the export README states.

**Objective convention (every table):** Q = gross_operational_cost, settlement EXCLUDED (the frozen primary, Addendum 58 Ruling 3); Q_net = net_operational_recourse = Q - terminal salvage credit (beside); Q_cc = Q + t_sum (first-order consensus-consistent diagnostic, REPORT-ONLY); value = Q(0) - Q(x); F = Q + I; EUR.

## What W160 changed against the predecessor

- T6: benchmark.w160_additions (both NRF arms in full; recorded decomposition; passive arm derived; NRF definition; consistency violations; failing sweep blocks; curtailment table)
- T2: cells_appended_w160 (certificates of T4, T5, T10); columns at_or_above_0.95_tau_counted, at_or_above_0.95_tau_twin_of, at_or_above_0.95_tau_twins_counted_once, at_or_above_0.95_tau_note, flag_beside_0.95_tau on every T2 row and appended row; at_or_above_0.95_tau_counted also on T5 rows and T10 cell records; at_or_above_0_95_tau (the count, scope, twins)
- T2 / T9: d_36686489 cause = "growth test after five TSO recoveries (not the dual dead zone; Addendum 65)" (W157 value kept as cause_uncertified_W157 / cause_W157)
- T8: ageing.column_labels.eps_AE_050_superseded = "superseded (0.50 era: C3, soh_min 0.50, E/P ≤ 10, no tight tail)"
- T11: three_by_three (Addenda 52 / 54, from committed records)

## The ≥ 0.95 τ count (Addendum 65 ruling 2)

Counted: 10 — b_2a0ba8b2, b_4649234b, c_6597a79d, d_3632b0ae, d_9246ed01, d_c52e1670, d_c7fee8be, pb_y2025_n7, pb_y2030_n7, ref:bd504ecf. Scope: every certificate the tables use: the 49 T2 cells (certified ones) plus the certificates of T4 (W118 year ladder), T5 (W118 Phase B) and T10 (A64 cells), appended to T2; superseded certificates excluded; bitwise twins counted once (Addendum 65 ruling 2). Twins: ref:bd504ecf = e_c2_calfade = g070_neutrality (evidence holds: True). Beside: i_5a6a88b4 — monotone, 0.939 (below 0.95 tau; flagged beside the ten, Addendum 65).

## Tables

### T1

Pairwise claims (60): difference, determinacy threshold or uncertified bar, multiple and verdict; net of salvage beside (labelled); Q_cc report-only. Determinacy: max(3 × larger band, 2τ) between certified cells; 3·max(|gap|, |slack|) with an uncertified cell; τ = 4,539.07 €. Objective convention: gross = gross_operational_cost, settlement excluded (primary); net = gross − terminal salvage credit, beside; Q_cc = Q + t_sum, report-only.

| claim | statement | ref cell | ref status | other cell | other status | d gross (k€) | rule | threshold / bar (k€) | × gross | verdict (gross) | d net (k€) | × net | verdict (net) | net label | d Q_cc (k€, report-only) | × Q_cc (report-only) | uncertified form beside: bar (k€) | uncertified form beside: × | uncertified form beside: verdict | notes |
|---|---|---|---|---|---|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---:|---:|---|---|
| B:n5_4h_e1 | x = 0 minimises F: F(n5_4h_e1) - F(0) > 0 (S47 A1a baseline ladder) | ref:7aa017f0 | cert. k*181 r/τ 0.93 | b_2a0ba8b2 | cert. oscillatory k*173 r/τ 0.97 | 58.0 | max(3 x larger bar, 2 tau) | 13.2 | 4.39 | determinate | 58.0 | 4.39 | determinate | validated by form + salvage identity | 58.9 | 4.46 | — | — |  |  |
| B:n7_2h_e1 | x = 0 minimises F: F(n7_2h_e1) - F(0) > 0 (S47 A1a baseline ladder) | ref:7aa017f0 | cert. k*181 r/τ 0.93 | d_d3709599 | cert. oscillatory k*175 r/τ 0.86 | 114.1 | max(3 x larger bar, 2 tau) | 12.6 | 9.03 | determinate | 114.1 | 9.03 | determinate | validated by form + salvage identity | 114.5 | 9.06 | — | — |  |  |
| B:n7_2h_e2 | x = 0 minimises F: F(n7_2h_e2) - F(0) > 0 (S47 A1a baseline ladder) | ref:7aa017f0 | cert. k*181 r/τ 0.93 | d_a12d95a2 | uncert. (lapse reset) end 203 | 231.7 | uncertified form | 24.2 | 9.56 | determinate | 231.7 | 9.56 | determinate | validated by form + salvage identity | 232.0 | 9.57 | — | — |  |  |
| B:n7_2h_e3 | x = 0 minimises F: F(n7_2h_e3) - F(0) > 0 (S47 A1a baseline ladder) | ref:7aa017f0 | cert. k*181 r/τ 0.93 | d_f759dd48 | uncert. (growth test) end 198 | 361.7 | uncertified form | 41.4 | 8.74 | determinate | 361.7 | 8.74 | determinate | validated by form + salvage identity | 362.3 | 8.76 | — | — |  |  |
| B:n7_2h_e4 | x = 0 minimises F: F(n7_2h_e4) - F(0) > 0 (S47 A1a baseline ladder) | ref:7aa017f0 | cert. k*181 r/τ 0.93 | d_c7fee8be | cert. oscillatory k*145 r/τ 0.98 | 494.4 | max(3 x larger bar, 2 tau) | 13.4 | 36.91 | determinate | 494.4 | 36.91 | determinate | validated by form + salvage identity | 496.4 | 37.05 | — | — |  |  |
| B:n7_2h_e5 | x = 0 minimises F: F(n7_2h_e5) - F(0) > 0 (S47 A1a baseline ladder) | ref:7aa017f0 | cert. k*181 r/τ 0.93 | d_9246ed01 | cert. oscillatory k*138 r/τ 0.99 | 616.6 | max(3 x larger bar, 2 tau) | 13.4 | 45.94 | determinate | 616.6 | 45.94 | determinate | validated by form + salvage identity | 617.3 | 46.00 | — | — |  |  |
| B:n7_4h_e1 | x = 0 minimises F: F(n7_4h_e1) - F(0) > 0 (S47 A1a baseline ladder) | ref:7aa017f0 | cert. k*181 r/τ 0.93 | ref:bd504ecf | cert. k*172 r/τ 0.96 | 64.4 | max(3 x larger bar, 2 tau) | 13.1 | 4.93 | determinate | 64.4 | 4.93 | determinate | validated by form + salvage identity | 64.9 | 4.97 | — | — |  |  |
| B:n7_4h_e2 | x = 0 minimises F: F(n7_4h_e2) - F(0) > 0 (S47 A1a baseline ladder) | ref:7aa017f0 | cert. k*181 r/τ 0.93 | d_c52e1670 | cert. oscillatory k*150 r/τ 0.98 | 135.5 | max(3 x larger bar, 2 tau) | 13.4 | 10.13 | determinate | 135.5 | 10.13 | determinate | validated by form + salvage identity | 136.7 | 10.22 | — | — |  |  |
| B:n7_4h_e3 | x = 0 minimises F: F(n7_4h_e3) - F(0) > 0 (S47 A1a baseline ladder) | ref:7aa017f0 | cert. k*181 r/τ 0.93 | d_4a82a64a | cert. oscillatory k*142 r/τ 0.72 | 208.3 | max(3 x larger bar, 2 tau) | 12.6 | 16.49 | determinate | 208.3 | 16.49 | determinate | validated by form + salvage identity | 209.1 | 16.56 | 70.7 | 2.94 | determinate | A64 ruling 4: certificate(s) d_4a82a64a rest on a non-clean turning point; kept and flagged; uncertified form beside: bar 70,743.93, 2.94× (determinate) |
| B:n7_4h_e4 | x = 0 minimises F: F(n7_4h_e4) - F(0) > 0 (S47 A1a baseline ladder) | ref:7aa017f0 | cert. k*181 r/τ 0.93 | d_3632b0ae | cert. oscillatory k*153 r/τ 1.00 | 274.0 | max(3 x larger bar, 2 tau) | 13.6 | 20.19 | determinate | 274.0 | 20.19 | determinate | validated by form + salvage identity | 274.7 | 20.24 | — | — |  |  |
| B:n7_4h_e5 | x = 0 minimises F: F(n7_4h_e5) - F(0) > 0 (S47 A1a baseline ladder) | ref:7aa017f0 | cert. k*181 r/τ 0.93 | d_36686489 | uncert. (growth test after five TSO recoveries (not the dual dead zone; Addendum 65)) end 191 | 347.2 | uncertified form | 23.9 | 14.52 | determinate | 347.2 | 14.52 | determinate | validated by form + salvage identity | 348.2 | 14.56 | — | — |  |  |
| B:n9_4h_e1 | x = 0 minimises F: F(n9_4h_e1) - F(0) > 0 (S47 A1a baseline ladder) | ref:7aa017f0 | cert. k*181 r/τ 0.93 | b_0dd237f0 | cert. oscillatory k*173 r/τ 0.92 | 61.1 | max(3 x larger bar, 2 tau) | 12.6 | 4.84 | determinate | 61.1 | 4.84 | determinate | validated by form + salvage identity | 61.6 | 4.87 | — | — |  |  |
| B:n9_4h_e3 | x = 0 minimises F: F(n9_4h_e3) - F(0) > 0 (S47 A1a baseline ladder) | ref:7aa017f0 | cert. k*181 r/τ 0.93 | b_4649234b | cert. oscillatory k*148 r/τ 0.99 | 199.1 | max(3 x larger bar, 2 tau) | 13.5 | 14.78 | determinate | 199.1 | 14.78 | determinate | validated by form + salvage identity | 201.0 | 14.92 | — | — |  |  |
| C:y2025__n5_p0.25_e0.5 | Phase B certificate at x = 0: F(y2025__n5_p0.25_e0.5) - F(0) > 0 | ref:7aa017f0 | cert. k*181 r/τ 0.93 | pb_y2025_n5_v6 | cert. oscillatory k*199 r/τ 0.16 | 49.7 | max(3 x larger bar, 2 tau) | 12.6 | 3.94 | determinate | 49.7 | 3.94 | determinate | validated by form + salvage identity | 49.8 | 3.94 | — | — |  |  |
| C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5 | Phase B certificate at x = 0: F(y2025__n5_p0.25_e0.5__n9_p0.25_e0.5) - F(0) > 0 | ref:7aa017f0 | cert. k*181 r/τ 0.93 | c_156ce2d1 | cert. oscillatory k*176 r/τ 0.86 | 102.1 | max(3 x larger bar, 2 tau) | 12.6 | 8.09 | determinate | 102.1 | 8.09 | determinate | validated by form + salvage identity | 102.7 | 8.13 | — | — |  |  |
| C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5 | Phase B certificate at x = 0: F(y2030__n5_p0.25_e0.5__n7_p0.25_e0.5) - F(0) > 0 | ref:7aa017f0 | cert. k*181 r/τ 0.93 | c_6597a79d | cert. oscillatory k*193 r/τ 0.98 | 93.9 | max(3 x larger bar, 2 tau) | 13.4 | 7.00 | determinate | 84.0 | 6.27 | determinate | validated by form + salvage identity | 93.7 | 6.99 | — | — |  |  |
| G:n7_2h_e1_y2030:gross | year ladder, x = 0 wins (gross): F(n7_2h_e1_y2030) - F(0) > 0 | ref:7aa017f0 | cert. k*181 r/τ 0.93 | g_37b5c499 | cert. oscillatory k*185 r/τ 0.94 | 99.0 | max(3 x larger bar, 2 tau) | 12.8 | 7.73 | determinate | 88.3 | 6.89 | determinate | recorded | 100.0 | 7.80 | — | — |  |  |
| G:n7_2h_e1_y2030:net | year ladder, x = 0 wins (net of salvage): F(n7_2h_e1_y2030) - F(0) > 0 | ref:7aa017f0 | cert. k*181 r/τ 0.93 | g_37b5c499 | cert. oscillatory k*185 r/τ 0.94 | 99.0 | max(3 x larger bar, 2 tau) | 12.8 | 7.73 | determinate | 88.3 | 6.89 | determinate | recorded | 100.0 | 7.80 | — | — |  |  |
| G:n7_2h_e1_y2035:gross | year ladder, x = 0 wins (gross): F(n7_2h_e1_y2035) - F(0) > 0 | ref:7aa017f0 | cert. k*181 r/τ 0.93 | g_47dce43c | cert. oscillatory k*174 r/τ 0.84 | 139.7 | max(3 x larger bar, 2 tau) | 12.6 | 11.06 | determinate | 85.1 | 6.74 | determinate | recorded | 139.9 | 11.08 | — | — |  |  |
| G:n7_2h_e1_y2035:net | year ladder, x = 0 wins (net of salvage): F(n7_2h_e1_y2035) - F(0) > 0 | ref:7aa017f0 | cert. k*181 r/τ 0.93 | g_47dce43c | cert. oscillatory k*174 r/τ 0.84 | 139.7 | max(3 x larger bar, 2 tau) | 12.6 | 11.06 | determinate | 85.1 | 6.74 | determinate | recorded | 139.9 | 11.08 | — | — |  |  |
| G:n7_4h_e2_y2030:gross | year ladder, x = 0 wins (gross): F(n7_4h_e2_y2030) - F(0) > 0 | ref:7aa017f0 | cert. k*181 r/τ 0.93 | g_48749148 | cert. oscillatory k*170 r/τ 0.93 | 137.4 | max(3 x larger bar, 2 tau) | 12.6 | 10.88 | determinate | 113.1 | 8.95 | determinate | recorded | 138.5 | 10.97 | — | — |  |  |
| G:n7_4h_e2_y2030:net | year ladder, x = 0 wins (net of salvage): F(n7_4h_e2_y2030) - F(0) > 0 | ref:7aa017f0 | cert. k*181 r/τ 0.93 | g_48749148 | cert. oscillatory k*170 r/τ 0.93 | 137.4 | max(3 x larger bar, 2 tau) | 12.6 | 10.88 | determinate | 113.1 | 8.95 | determinate | recorded | 138.5 | 10.97 | — | — |  |  |
| G:n7_4h_e2_y2035:gross | year ladder, x = 0 wins (gross): F(n7_4h_e2_y2035) - F(0) > 0 | ref:7aa017f0 | cert. k*181 r/τ 0.93 | g_9abf31d4 | cert. oscillatory k*172 r/τ 0.85 | 223.6 | max(3 x larger bar, 2 tau) | 12.6 | 17.71 | determinate | 109.1 | 8.64 | determinate | recorded | 224.0 | 17.74 | — | — |  |  |
| G:n7_4h_e2_y2035:net | year ladder, x = 0 wins (net of salvage): F(n7_4h_e2_y2035) - F(0) > 0 | ref:7aa017f0 | cert. k*181 r/τ 0.93 | g_9abf31d4 | cert. oscillatory k*172 r/τ 0.85 | 223.6 | max(3 x larger bar, 2 tau) | 12.6 | 17.71 | determinate | 109.1 | 8.64 | determinate | recorded | 224.0 | 17.74 | — | — |  |  |
| H:m1.5:value_minus_I | flexibility ladder m = 1.5: sign of value - I (node 7, 0.25 MVA / 1 MWh) | h_aa8a76d7 | cert. oscillatory k*153 r/τ 0.91 | h_f9eae48f | uncert. (gap clause) end 194 | −14.5 | uncertified form | 18.8 | 0.77 | within the uncertified bar | −14.5 | 0.77 | within the uncertified bar | validated by form + salvage identity | −12.5 | 0.67 | — | — |  | A63: sign unresolved at m = 1.5 (uncertified form, gap-refused unit cell); crossing at or below m = 1.75; the m = 1.75 row is PENDING W156 |
| H:m2:value_minus_I | flexibility ladder m = 2: sign of value - I (node 7, 0.25 MVA / 1 MWh) | h_50dea31c | cert. oscillatory k*146 r/τ 0.48 | h_74eda68d | cert. oscillatory k*150 r/τ 0.90 | 46.3 | max(3 x larger bar, 2 tau) | 12.3 | 3.77 | determinate | 46.3 | 3.77 | determinate | validated by form + salvage identity | 48.2 | 3.93 | — | — |  |  |
| I:m2:e2_value_minus_I | m = 2: node 7 0.5 MVA / 2 MWh pays: value - I > 0 | h_50dea31c | cert. oscillatory k*146 r/τ 0.48 | i_5a6a88b4 | cert. monotone k*198 r/τ 0.94 | 92.8 | max(3 x larger bar, 2 tau) | 12.8 | 7.26 | determinate | 92.8 | 7.26 | determinate | validated by form + salvage identity | 93.2 | 7.29 | — | — |  |  |
| I:m2:second_MWh | m = 2: the second MWh pays: [Q(e1) - Q(e2)] - [I(e2) - I(e1)] > 0 | h_74eda68d | cert. oscillatory k*150 r/τ 0.90 | i_5a6a88b4 | cert. monotone k*198 r/τ 0.94 | 46.5 | max(3 x larger bar, 2 tau) | 12.8 | 3.64 | determinate | 46.5 | 3.64 | determinate | validated by form + salvage identity | 45.0 | 3.52 | — | — |  |  |
| J:e2_to_e3 | F2 ladder: the marginal MWh e2 -> e3 pays | i_5a6a88b4 | cert. monotone k*198 r/τ 0.94 | j_f3aa335e | uncert. (growth test) end 260 | 41.0 | uncertified form | 27.7 | 1.48 | determinate | 41.0 | 1.48 | determinate | validated by form + salvage identity | 41.1 | 1.48 | — | — |  |  |
| J:e3_to_e4 | F2 ladder: the marginal MWh e3 -> e4 pays | j_f3aa335e | uncert. (growth test) end 260 | j_a11d7966 | cert. oscillatory k*158 r/τ 0.71 | 42.8 | uncertified form | 27.7 | 1.55 | determinate | 42.8 | 1.55 | determinate | validated by form + salvage identity | 42.4 | 1.53 | 27.7 | 1.55 | determinate | A64 ruling 4: certificate(s) j_a11d7966 rest on a non-clean turning point; kept and flagged; uncertified form beside: bar 27,694.48, 1.55× (determinate) |
| J:e3_value_minus_I | F2 ladder: node 7 4 h e3 pays: value - I > 0 | h_50dea31c | cert. oscillatory k*146 r/τ 0.48 | j_f3aa335e | uncert. (growth test) end 260 | 133.8 | uncertified form | 27.7 | 4.83 | determinate | 133.8 | 4.83 | determinate | validated by form + salvage identity | 134.2 | 4.85 | — | — |  |  |
| J:e4_to_e5 | F2 ladder: the marginal MWh e4 -> e5 pays | j_a11d7966 | cert. oscillatory k*158 r/τ 0.71 | j_5f3cccb4 | uncert. (gap clause) end 270 | 51.8 | uncertified form | 26.5 | 1.95 | determinate | 51.8 | 1.95 | determinate | validated by form + salvage identity | 61.3 | 2.31 | 26.5 | 1.95 | determinate | A64 ruling 4: certificate(s) j_a11d7966 rest on a non-clean turning point; kept and flagged; uncertified form beside: bar 26,543.70, 1.95× (determinate) |
| J:e4_value_minus_I | F2 ladder: node 7 4 h e4 pays: value - I > 0 | h_50dea31c | cert. oscillatory k*146 r/τ 0.48 | j_a11d7966 | cert. oscillatory k*158 r/τ 0.71 | 176.6 | max(3 x larger bar, 2 tau) | 9.7 | 18.14 | determinate | 176.6 | 18.14 | determinate | validated by form + salvage identity | 176.7 | 18.15 | 15.9 | 11.12 | determinate | A64 ruling 4: certificate(s) j_a11d7966 rest on a non-clean turning point; kept and flagged; uncertified form beside: bar 15,879.58, 11.12× (determinate) / A64 ruling 4: the MANUSCRIPT uses J 4 MWh at its uncertified bar |
| J:e5_value_minus_I | F2 ladder: node 7 4 h e5 pays: value - I > 0 | h_50dea31c | cert. oscillatory k*146 r/τ 0.48 | j_5f3cccb4 | uncert. (gap clause) end 270 | 228.4 | uncertified form | 26.5 | 8.60 | determinate | 228.4 | 8.60 | determinate | validated by form + salvage identity | 238.0 | 8.97 | — | — |  |  |
| L:df1a5525_not_a_neighbour | F(y2030__n5_p0.25_e0.5__n7_p0.75_e2.5) - F(incumbent): listed by the Addendum 53 inventory under L; NOT in the certificate poll set or its cached box neighbours (zE7 differs by 2 lattice units) | ref:5ca4f86c | uncert. (gap clause) end 281 | l_df1a5525 | cert. monotone k*236 r/τ 0.14 | 47.7 | uncertified form | 27.7 | 1.72 | determinate | 58.5 | 2.11 | determinate | validated by form + salvage identity | 56.8 | 2.05 | — | — |  |  |
| L:y2025__n7_p0.75_e3 | F2 certificate: F(y2025__n7_p0.75_e3) - F(incumbent) (positive = worse than the incumbent; the certificate needs no determinate improvement) | ref:5ca4f86c | uncert. (gap clause) end 281 | j_f3aa335e | uncert. (growth test) end 260 | 94.3 | uncertified form | 27.7 | 3.41 | determinate | 133.4 | 4.81 | determinate | validated by form + salvage identity | 103.8 | 3.75 | — | — |  |  |
| L:y2030__n5_p0.25_e0.5__n7_p0.75_e3 | F2 certificate: F(y2030__n5_p0.25_e0.5__n7_p0.75_e3) - F(incumbent) (positive = worse than the incumbent; the certificate needs no determinate improvement) | ref:5ca4f86c | uncert. (gap clause) end 281 | l_195156fa | cert. monotone k*252 r/τ 0.10 | 28.6 | uncertified form | 27.7 | 1.03 | determinate | 32.4 | 1.17 | determinate | validated by form + salvage identity | 37.8 | 1.36 | — | — |  |  |
| L:y2030__n5_p0.25_e0.5__n7_p0.75_e3__n9_p0.25_e0.5 | F2 certificate: F(y2030__n5_p0.25_e0.5__n7_p0.75_e3__n9_p0.25_e0.5) - F(incumbent) (positive = worse than the incumbent; the certificate needs no determinate improvement) | ref:5ca4f86c | uncert. (gap clause) end 281 | l_b2251bc5 | uncert. (gap clause) end 320 | 9.8 | uncertified form | 28.2 | 0.35 | within the uncertified bar | 11.0 | 0.39 | within the uncertified bar | validated by form + salvage identity | 9.6 | 0.34 | — | — |  |  |
| L:y2030__n5_p0.25_e0.5__n7_p1.25_e3 | F2 certificate: F(y2030__n5_p0.25_e0.5__n7_p1.25_e3) - F(incumbent) (positive = worse than the incumbent; the certificate needs no determinate improvement) | ref:5ca4f86c | uncert. (gap clause) end 281 | l_2ab0ce2d | cert. monotone k*400 r/τ 0.09 | 53.3 | uncertified form | 27.7 | 1.92 | determinate | 62.8 | 2.27 | determinate | validated by form + salvage identity | 62.5 | 2.26 | — | — |  |  |
| L:y2030__n5_p0.25_e0.5__n7_p1_e3 | F2 certificate: F(y2030__n5_p0.25_e0.5__n7_p1_e3) - F(incumbent) (positive = worse than the incumbent; the certificate needs no determinate improvement) | ref:5ca4f86c | uncert. (gap clause) end 281 | l_0ee93aca | uncert. (lapse reset) end 284 | 26.4 | uncertified form | 27.9 | 0.95 | within the uncertified bar | 34.1 | 1.22 | determinate | validated by form + salvage identity | 26.4 | 0.94 | — | — |  | salvage convention changes the verdict: gross within the uncertified bar, net determinate |
| L:y2030__n5_p0.25_e1__n7_p1_e3 | F2 certificate: F(y2030__n5_p0.25_e1__n7_p1_e3) - F(incumbent) (positive = worse than the incumbent; the certificate needs no determinate improvement) | ref:5ca4f86c | uncert. (gap clause) end 281 | ref:e28de4ac | uncert. (gap clause) end 261 | −6.1 | uncertified form | 27.9 | 0.22 | within the uncertified bar | −5.4 | 0.19 | within the uncertified bar | validated by form + salvage identity | −6.2 | 0.22 | — | — |  |  |
| L:y2030__n7_p0.75_e3 | F2 certificate: F(y2030__n7_p0.75_e3) - F(incumbent) (positive = worse than the incumbent; the certificate needs no determinate improvement) | ref:5ca4f86c | uncert. (gap clause) end 281 | l_8e4c220e | cert. monotone k*233 r/τ 0.17 | 61.5 | uncertified form | 27.7 | 2.22 | determinate | 69.2 | 2.50 | determinate | validated by form + salvage identity | 70.6 | 2.55 | — | — |  |  |
| L:y2030__n7_p0.75_e3__n9_p0.25_e0.5 | F2 certificate: F(y2030__n7_p0.75_e3__n9_p0.25_e0.5) - F(incumbent) (positive = worse than the incumbent; the certificate needs no determinate improvement) | ref:5ca4f86c | uncert. (gap clause) end 281 | l_7db09f6c | cert. monotone k*251 r/τ 0.09 | 35.3 | uncertified form | 27.7 | 1.28 | determinate | 38.5 | 1.39 | determinate | validated by form + salvage identity | 44.5 | 1.61 | — | — |  |  |
| L:y2030__n7_p1_e3 | F2 certificate: F(y2030__n7_p1_e3) - F(incumbent) (positive = worse than the incumbent; the certificate needs no determinate improvement) | ref:5ca4f86c | uncert. (gap clause) end 281 | l_76c78064 | cert. monotone k*229 r/τ 0.01 | 50.6 | uncertified form | 27.7 | 1.83 | determinate | 61.1 | 2.21 | determinate | validated by form + salvage identity | 59.8 | 2.16 | — | — |  |  |
| L:y2030__n7_p1_e3.5 | F2 certificate: F(y2030__n7_p1_e3.5) - F(incumbent) (positive = worse than the incumbent; the certificate needs no determinate improvement) | ref:5ca4f86c | uncert. (gap clause) end 281 | l_e1da0984 | cert. monotone k*235 r/τ 0.04 | 23.9 | uncertified form | 27.7 | 0.86 | within the uncertified bar | 26.8 | 0.97 | within the uncertified bar | validated by form + salvage identity | 33.1 | 1.19 | — | — |  |  |
| L:y2030__n7_p1_e3.5__n9_p0.25_e0.5 | F2 certificate: F(y2030__n7_p1_e3.5__n9_p0.25_e0.5) - F(incumbent) (positive = worse than the incumbent; the certificate needs no determinate improvement) | ref:5ca4f86c | uncert. (gap clause) end 281 | l_7c455554 | uncert. (gap clause) end 282 | 5.6 | uncertified form | 27.7 | 0.20 | within the uncertified bar | 5.8 | 0.21 | within the uncertified bar | validated by form + salvage identity | 5.6 | 0.20 | — | — |  |  |
| L:y2030__n7_p1_e3__n9_p0.25_e0.5 | F2 certificate: F(y2030__n7_p1_e3__n9_p0.25_e0.5) - F(incumbent) (positive = worse than the incumbent; the certificate needs no determinate improvement) | ref:5ca4f86c | uncert. (gap clause) end 281 | l_45aa25a6 | uncert. (gap clause) end 291 | 31.3 | uncertified form | 27.7 | 1.13 | determinate | 39.2 | 1.41 | determinate | validated by form + salvage identity | 31.4 | 1.13 | — | — |  |  |
| L:y2030__n7_p1_e4 | F2 certificate: F(y2030__n7_p1_e4) - F(incumbent) (positive = worse than the incumbent; the certificate needs no determinate improvement) | ref:5ca4f86c | uncert. (gap clause) end 281 | l_7b199ef9 | cert. oscillatory k*188 r/τ 0.31 | 6.4 | uncertified form | 27.7 | 0.23 | within the uncertified bar | 1.6 | 0.06 | within the uncertified bar | validated by form + salvage identity | 15.9 | 0.57 | — | — |  |  |
| E:C3_unit_value_minus_I | C3 unit does not pay: value - I < 0 | ref:7aa017f0 | cert. k*181 r/τ 0.93 | e_c3_unit | cert. oscillatory k*160 r/τ 0.94 | −73.7 | max(3 x larger bar, 2 tau) | 12.8 | 5.74 | determinate | −73.7 | 5.74 | determinate | validated by form + salvage identity | −73.8 | 5.75 | — | — |  |  |
| E:n7_4h_e1_C2:value_minus_I | ageing variant n7_4h_e1_C2: sign of value - I | ref:7aa017f0 | cert. k*181 r/τ 0.93 | e_c2 | cert. oscillatory k*172 r/τ 0.90 | −31.6 | max(3 x larger bar, 2 tau) | 12.6 | 2.50 | determinate | −31.6 | 2.50 | determinate | validated by form + salvage identity | −31.9 | 2.53 | — | — |  |  |
| E:n7_4h_e1_C2:vs_C3 | ageing variant n7_4h_e1_C2 vs the C3 unit: sign of value(variant) - value(C3) | e_c3_unit | cert. oscillatory k*160 r/τ 0.94 | e_c2 | cert. oscillatory k*172 r/τ 0.90 | 42.1 | max(3 x larger bar, 2 tau) | 12.8 | 3.28 | determinate | 42.1 | 3.28 | determinate | validated by form + salvage identity | 41.9 | 3.27 | — | — |  |  |
| E:n7_4h_e1_C2_calfade:value_minus_I | ageing variant n7_4h_e1_C2_calfade: sign of value - I | ref:7aa017f0 | cert. k*181 r/τ 0.93 | e_c2_calfade | cert. oscillatory k*172 r/τ 0.96 | −64.4 | max(3 x larger bar, 2 tau) | 13.1 | 4.93 | determinate | −64.4 | 4.93 | determinate | validated by form + salvage identity | −64.9 | 4.97 | — | — |  |  |
| E:n7_4h_e1_C2_calfade:vs_C3 | ageing variant n7_4h_e1_C2_calfade vs the C3 unit: sign of value(variant) - value(C3) | e_c3_unit | cert. oscillatory k*160 r/τ 0.94 | e_c2_calfade | cert. oscillatory k*172 r/τ 0.96 | 9.3 | max(3 x larger bar, 2 tau) | 13.1 | 0.71 | within resolution | 9.3 | 0.71 | within resolution | validated by form + salvage identity | 8.9 | 0.68 | — | — |  | A64 ruling 5: reported WITHIN RESOLUTION (the determinacy floor changed this verdict from the superseded rule); S1 restated "within resolution of C3 at a 0.70 floor"; the A61 prediction "the floor changes no verdict" failed on this claim (recorded, not re-litigated) |
| E:n7_4h_e1_C3_midblock:value_minus_I | ageing variant n7_4h_e1_C3_midblock: sign of value - I | ref:7aa017f0 | cert. k*181 r/τ 0.93 | e_c3_midblock | cert. oscillatory k*174 r/τ 0.94 | −65.9 | max(3 x larger bar, 2 tau) | 12.9 | 5.12 | determinate | −65.9 | 5.12 | determinate | validated by form + salvage identity | −66.1 | 5.13 | — | — |  |  |
| E:n7_4h_e1_C3_midblock:vs_C3 | ageing variant n7_4h_e1_C3_midblock vs the C3 unit: sign of value(variant) - value(C3) | e_c3_unit | cert. oscillatory k*160 r/τ 0.94 | e_c3_midblock | cert. oscillatory k*174 r/τ 0.94 | 7.9 | max(3 x larger bar, 2 tau) | 12.9 | 0.61 | within resolution | 7.9 | 0.61 | within resolution | validated by form + salvage identity | 7.8 | 0.60 | — | — |  |  |
| E:n7_4h_e1_C4:value_minus_I | ageing variant n7_4h_e1_C4: sign of value - I | ref:7aa017f0 | cert. k*181 r/τ 0.93 | e_c4 | cert. oscillatory k*173 r/τ 0.84 | −45.2 | max(3 x larger bar, 2 tau) | 12.6 | 3.58 | determinate | −45.2 | 3.58 | determinate | validated by form + salvage identity | −45.7 | 3.62 | — | — |  |  |
| E:n7_4h_e1_C4:vs_C3 | ageing variant n7_4h_e1_C4 vs the C3 unit: sign of value(variant) - value(C3) | e_c3_unit | cert. oscillatory k*160 r/τ 0.94 | e_c4 | cert. oscillatory k*173 r/τ 0.84 | 28.5 | max(3 x larger bar, 2 tau) | 12.8 | 2.22 | determinate | 28.5 | 2.22 | determinate | validated by form + salvage identity | 28.1 | 2.19 | — | — |  |  |
| E:n7_4h_e1_no_ageing:value_minus_I | ageing variant n7_4h_e1_no_ageing: sign of value - I | ref:7aa017f0 | cert. k*181 r/τ 0.93 | e_no_ageing | cert. oscillatory k*177 r/τ 0.94 | −4.1 | max(3 x larger bar, 2 tau) | 12.9 | 0.32 | within resolution | −4.1 | 0.32 | within resolution | validated by form + salvage identity | −4.9 | 0.38 | — | — |  | A64 verbatim sentence: without ageing the unit is at break-even (within resolution) |
| E:n7_4h_e1_no_ageing:vs_C3 | ageing variant n7_4h_e1_no_ageing vs the C3 unit: sign of value(variant) - value(C3) | e_c3_unit | cert. oscillatory k*160 r/τ 0.94 | e_no_ageing | cert. oscillatory k*177 r/τ 0.94 | 69.6 | max(3 x larger bar, 2 tau) | 12.9 | 5.41 | determinate | 69.6 | 5.41 | determinate | validated by form + salvage identity | 68.9 | 5.36 | — | — |  |  |
| CHECK:headline_V_minus_I_settled | headline: x = 0 optimal at SRP1, V - I < 0 (settled references) | ref:7aa017f0 | cert. k*181 r/τ 0.93 | ref:bd504ecf | cert. k*172 r/τ 0.96 | −64.4 | max(3 x larger bar, 2 tau) | 13.1 | 4.93 | determinate | −64.4 | 4.93 | determinate | validated by form + salvage identity | −64.9 | 4.97 | — | — |  |  |

### T2

Certification status of every cell the tables use: the 49 claim cells and the references, then the certificates of T4, T5 and T10 appended (column "table"). at_or_above_0.95_tau_counted is true on exactly ten certificates: superseded certificates excluded, bitwise twins (the unit 3f084f2f = e_c2_calfade = g070_neutrality) counted once (Addendum 65). Objective convention: gross = gross_operational_cost, settlement excluded (primary); net = gross − terminal salvage credit, beside; Q_cc = Q + t_sum, report-only.

| cell | table | item | status | branch | k0 | k* / end | range/τ | ≥ 0.95 τ (range) | at_or_above_0.95_tau_counted | ≥ 0.95 τ note | flag beside | NCTP cycles | cause (uncertified) | gap (k€) | slack (k€) | band (k€) | non-clean cycles after N | replay bitwise through | certifying spec | eval key | candidate key |
|---|---|---|---|---|---:|---:|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| b_0dd237f0 | T2 | B | certified | oscillatory | 103 | 173 | 0.92 | no | no | below 0.95 tau |  |  |  | — | — | 4.2 | — | 103 | v4 | 38d09af2c857755f | a361ac93 |
| b_2a0ba8b2 | T2 | B | certified | oscillatory | 104 | 173 | 0.97 | yes | yes | counted |  |  |  | — | — | 4.4 | — | 104 | v4 | 33447912dab48fff | 18216659 |
| b_4649234b | T2 | B | certified | oscillatory | 84 | 148 | 0.99 | yes | yes | counted |  |  |  | — | — | 4.5 | — | 84 | v5 | 4de68ced93863cec | ace7403b |
| c_156ce2d1 | T2 | C | certified | oscillatory | 105 | 176 | 0.86 | no | no | below 0.95 tau |  |  |  | — | — | 3.9 | 0 | 105 | v6 | 3a4c10c859114e20 | ba670a65 |
| c_6597a79d | T2 | C | certified | oscillatory | 131 | 193 | 0.98 | yes | yes | counted |  |  |  | — | — | 4.5 | 0 | 131 | v6 | 3a0d03e88a16f174 | a5bb68bf |
| d_3632b0ae | T2 | D | certified | oscillatory | 86 | 153 | 1.00 | yes | yes | counted |  |  |  | — | — | 4.5 | 0 | 86 | v6 | 1fe57699ce791ef6 | 11725e45 |
| d_36686489 | T2 | D | uncertified |  | 82 | 191 | — | no | no | not a certificate |  | 113,164,176 | growth test after five TSO recoveries (not the dual dead zone; Addendum 65) | 1.1 | 8.0 | 6.1 | 5 | 82 |  | 4de61708589e6e26 | cab98853 |
| d_4a82a64a | T2 | D | certified | oscillatory | 88 | 142 | 0.72 | no | no | below 0.95 tau |  | 108 |  | — | — | 3.3 | 1 | 88 | v6 | 14b00a04ffbcfd33 | fb21d822 |
| d_9246ed01 | T2 | D | certified | oscillatory | 83 | 138 | 0.99 | yes | yes | counted |  |  |  | — | — | 4.5 | 0 | 83 | v6 | 579a1370d1e916a7 | 354b3881 |
| d_a12d95a2 | T2 | D | uncertified |  | 94 | 203 | — | no | no | not a certificate |  |  | lapse reset | 0.5 | 8.1 | 5.3 | 2 | 94 |  | 9bc66d9ed4863f39 | 423e5a7f |
| d_c52e1670 | T2 | D | certified | oscillatory | 89 | 150 | 0.98 | yes | yes | counted |  |  |  | — | — | 4.5 | 1 | — | v6 (v6 from records) | af42a163b14f0895 | cc4eb5df |
| d_c7fee8be | T2 | D | certified | oscillatory | 83 | 145 | 0.98 | yes | yes | counted |  |  |  | — | — | 4.5 | 0 | 83 | v6 | aa0b99f57c75682a | 7048a00e |
| d_d3709599 | T2 | D | certified | oscillatory | 104 | 175 | 0.86 | no | no | below 0.95 tau |  |  |  | — | — | 3.9 | 0 | 104 | v6 | 64408baeee0990b9 | 3993aebc |
| d_f759dd48 | T2 | D | uncertified |  | 89 | 198 | — | no | no | not a certificate |  | 110,141,170,179 | growth test | 0.8 | 13.8 | 6.8 | 6 | 89 |  | 5f1b1aa621d3450a | 2cd1de82 |
| e_c2 | T2 | E | certified | oscillatory | 111 | 172 | 0.90 | no | no | below 0.95 tau |  |  |  | — | — | 4.1 | 0 | — | ext v3 (rule v6) | ba969ad21c1831ab | db77e154 |
| e_c2_calfade | T2 | E | certified | oscillatory | 103 | 172 | 0.96 | yes | no | bitwise twin of the unit (counted once, on ref:bd504ecf) |  |  |  | — | — | 4.4 | 0 | — | ext v3 (rule v6) | bbd82994a7ddff94 | db77e154 |
| e_c3_midblock | T2 | E | certified | oscillatory | 104 | 174 | 0.94 | no | no | below 0.95 tau |  |  |  | — | — | 4.3 | 0 | — | ext v3 (rule v6) | 8b1eace91df86cd5 | db77e154 |
| e_c3_unit | T2 | E | certified | oscillatory | 104 | 160 | 0.94 | no | no | below 0.95 tau |  |  |  | — | — | 4.3 | 0 | — | ext v3 (rule v6) | 409740fb51f27cab | db77e154 |
| e_c4 | T2 | E | certified | oscillatory | 110 | 173 | 0.84 | no | no | below 0.95 tau |  |  |  | — | — | 3.8 | 0 | — | ext v3 (rule v6) | c59d7006b4b9c0a2 | db77e154 |
| e_no_ageing | T2 | E | certified | oscillatory | 107 | 177 | 0.94 | no | no | below 0.95 tau |  |  |  | — | — | 4.3 | 0 | — | ext v3 (rule v6) | 1b10e9e161523f46 | db77e154 |
| g_37b5c499 | T2 | G | certified | oscillatory | 114 | 185 | 0.94 | no | no | below 0.95 tau |  |  |  | — | — | 4.3 | 0 | — | v6 | bf2d8348d3a4b730 | e7c94efa |
| g_47dce43c | T2 | G | certified | oscillatory | 112 | 174 | 0.84 | no | no | below 0.95 tau |  |  |  | — | — | 3.8 | 0 | — | v6 | 46ff23a160a27b67 | 803d1a6c |
| g_48749148 | T2 | G | certified | oscillatory | 101 | 170 | 0.93 | no | no | below 0.95 tau |  |  |  | — | — | 4.2 | 0 | — | v6 | 22331aa714b8af80 | 76baf11e |
| g_9abf31d4 | T2 | G | certified | oscillatory | 103 | 172 | 0.85 | no | no | below 0.95 tau |  |  |  | — | — | 3.8 | 0 | — | v6 | 5560d0f015a8d196 | ca1b7bac |
| h_50dea31c | T2 | H | certified | oscillatory | 125 | 146 | 0.48 | no | no | below 0.95 tau |  |  |  | — | — | 2.2 | 0 | 125 | v6 | 354b1bdae2b86bce | 8435c718 |
| h_74eda68d | T2 | H | certified | oscillatory | 123 | 150 | 0.90 | no | no | below 0.95 tau |  |  |  | — | — | 4.1 | 0 | 123 | v6 | dca0862812873604 | db77e154 |
| h_aa8a76d7 | T2 | H | certified | oscillatory | 93 | 153 | 0.91 | no | no | below 0.95 tau |  |  |  | — | — | 4.1 | 0 | 93 | v6 | 22dd5ef8a9223f56 | 8435c718 |
| h_f9eae48f | T2 | H | uncertified |  | 85 | 194 | — | no | no | not a certificate |  |  | gap clause | 2.5 | 6.3 | 1.0 | 0 | 85 |  | aed2d618a19ca483 | db77e154 |
| i_5a6a88b4 | T2 | I | certified | monotone | 135 | 198 | 0.94 | no | no | below 0.95 tau | monotone, 0.939 (below 0.95 tau; flagged beside the ten, Addendum 65) |  |  | — | — | 4.3 | 0 | 135 | v6 | e046a8cfd25f819c | cc4eb5df |
| j_5f3cccb4 | T2 | J | uncertified |  | 159 | 270 | — | no | no | not a certificate |  | 249 | gap clause | 8.8 | 6.2 | 2.8 | 1 | 159 |  | cc7f005909db6741 | cab98853 |
| j_a11d7966 | T2 | J | certified | oscillatory | 127 | 158 | 0.71 | no | no | below 0.95 tau |  | 138 |  | — | — | 3.2 | 1 | 127 | v6 | 87efbc3d23566c29 | 11725e45 |
| j_f3aa335e | T2 | J | uncertified |  | 151 | 260 | — | no | no | not a certificate |  | 178 | growth test | 0.3 | 9.2 | 4.6 | 1 | 151 |  | af066d6bc9fdb74d | fb21d822 |
| l_0ee93aca | T2 | L | uncertified |  | 175 | 284 | — | no | no | not a certificate |  |  | lapse reset | 9.3 | 1.5 | 0.2 | 2 | 175 |  | 86d27f42e98503ca | 3579af95 |
| l_195156fa | T2 | L | certified | monotone | 185 | 252 | 0.10 | no | no | below 0.95 tau |  |  |  | — | — | 0.4 | 0 | 185 | v6 | 1fd23b8db2be1ba0 | a033945d |
| l_2ab0ce2d | T2 | L | certified | monotone | 328 | 400 | 0.09 | no | no | below 0.95 tau |  |  |  | — | — | 0.4 | 0 | 328 | v6 | 75266862b990d2eb | 9f4f66b6 |
| l_45aa25a6 | T2 | L | uncertified |  | 182 | 291 | — | no | no | not a certificate |  |  | gap clause | 9.2 | 2.2 | 0.7 | 0 | 182 |  | 296e09ff97ac2db4 | a33409d1 |
| l_76c78064 | T2 | L | certified | monotone | 153 | 229 | 0.01 | no | no | below 0.95 tau |  |  |  | — | — | 0.1 | 0 | 153 | v6 | 571ed31e4139a567 | 2feadb35 |
| l_7b199ef9 | T2 | L | certified | oscillatory | 166 | 188 | 0.31 | no | no | below 0.95 tau |  |  |  | — | — | 1.4 | 0 | 166 | v6 | 331e76c7c613b64e | 0025f345 |
| l_7c455554 | T2 | L | uncertified |  | 173 | 282 | — | no | no | not a certificate |  |  | gap clause | 9.2 | 1.5 | 0.6 | 0 | 173 |  | 9906389c1cd682a9 | e9e995e1 |
| l_7db09f6c | T2 | L | certified | monotone | 183 | 251 | 0.09 | no | no | below 0.95 tau |  |  |  | — | — | 0.4 | 0 | 183 | v6 | 7f5a857b06a6083a | 7126547f |
| l_8e4c220e | T2 | L | certified | monotone | 171 | 233 | 0.17 | no | no | below 0.95 tau |  |  |  | — | — | 0.8 | 0 | 171 | v6 | 5775a50cf6c84503 | 3b77a5f6 |
| l_b2251bc5 | T2 | L | uncertified |  | 211 | 320 | — | no | no | not a certificate |  |  | gap clause | 9.4 | 2.3 | 0.7 | 0 | 211 |  | abf4ac0af85dcb5e | c29e38c3 |
| l_df1a5525 | T2 | L | certified | monotone | 168 | 236 | 0.14 | no | no | below 0.95 tau |  |  |  | — | — | 0.6 | 0 | 168 | v6 | 6c8f6353e29b204f | 7b85ff44 |
| l_e1da0984 | T2 | L | certified | monotone | 156 | 235 | 0.04 | no | no | below 0.95 tau |  |  |  | — | — | 0.2 | 0 | 156 | v6 | 2587ab4987bc3125 | 407fdee0 |
| pb_y2025_n5_v6 | T2 | C | certified | oscillatory | 110 | 199 | 0.16 | no | no | below 0.95 tau |  |  |  | — | — | 0.7 | 1 | 110 | ext v3 (rule v6) | be95e57603437a06 | 33031c08 |
| ref:5ca4f86c | T2 | reference | uncertified |  | — | 281 | — | no | no | not a certificate |  |  | gap clause | 9.2 | 1.5 | 0.4 | — | 172 |  | 24c5ccb6f285219f | 59757776 |
| ref:7aa017f0 | T2 | reference | certified |  | — | 181 | 0.93 | no | no | below 0.95 tau |  |  |  | — | — | 4.2 | — | — |  | d110bd1a5977df1e | 8435c718 |
| ref:bd504ecf | T2 | reference | certified |  | — | 172 | 0.96 | yes | yes | counted |  |  |  | — | — | 4.4 | — | — |  | 3f084f2ffaeef2b7 | db77e154 |
| ref:e28de4ac | T2 | reference | uncertified |  | — | 261 | — | no | no | not a certificate |  |  | gap clause | 9.3 | 1.0 | 1.2 | — | 152 |  | 1fe91e86f11e76af | 4032a138 |
| e_soh050 | T10 | E | certified | oscillatory | 112 | 184 | 0.92 | no | no | below 0.95 tau |  |  |  | — | — | 4.2 | — | — | A64 v1 (rule v6) | 3f01b9eaaab13c82 | db77e154 |
| g070_neutrality | T10 | E | certified | oscillatory | 103 | 172 | 0.96 | yes | no | bitwise twin of the unit (counted once, on ref:bd504ecf) |  |  |  | — | — | 4.4 | — | — | A64 v1 (rule v6) | 5007232b1c3eaf03 | db77e154 |
| h_unit_m175 | T10 | H | certified | oscillatory | 101 | 121 | 0.65 | no | no | below 0.95 tau |  |  |  | — | — | 2.9 | — | — | A64 v1 (rule v6) | dc3ab9329c395890 | db77e154 |
| h_x0_m175 | T10 | H | certified | oscillatory | 108 | 139 | 0.79 | no | no | below 0.95 tau |  |  |  | — | — | 3.6 | — | — | A64 v1 (rule v6) | b9f9be3528a64a93 | 8435c718 |
| pb_y2025_n5 | T5 | Phase B | certified | oscillatory | 110 | 167 | 0.99 | yes | no | superseded (excluded) |  |  |  | — | — | 4.5 | — | 110 | W118 r2 (rule v2) | ca29c5e818366a6e | 33031c08 |
| pb_y2025_n7 | T5 | Phase B | certified | oscillatory | 112 | 172 | 0.97 | yes | yes | counted |  |  |  | — | — | 4.4 | — | 112 | W118 r2 (rule v2) | b9b8e4be4a6c82b6 | 56b9730b |
| pb_y2025_n9 | T5 | Phase B | certified | oscillatory | 113 | 174 | 0.81 | no | no | below 0.95 tau |  |  |  | — | — | 3.7 | — | 113 | W118 r2 (rule v2) | b7bce5a8bf2365b9 | c50a0e02 |
| pb_y2030_n5 | T5 | Phase B | certified | oscillatory | 122 | 182 | 0.94 | no | no | below 0.95 tau |  |  |  | — | — | 4.2 | — | 122 | W118 r2 (rule v2) | 3846675acea98ea7 | e0e61484 |
| pb_y2030_n7 | T5 | Phase B | certified | oscillatory | 125 | 195 | 0.99 | yes | yes | counted |  |  |  | — | — | 4.5 | — | 125 | W118 r2 (rule v2) | 828f03fa9ac2ca6e | 53c1de5c |
| pb_y2030_n9 | T5 | Phase B | certified | oscillatory | 120 | 192 | 0.94 | no | no | below 0.95 tau |  |  |  | — | — | 4.3 | — | 120 | W118 r2 (rule v2) | 344c936fc23919be | da2f6bc1 |
| yl_y2030 | T4 | year ladder | certified | oscillatory | 107 | 169 | 0.90 | no | no | below 0.95 tau |  |  |  | — | — | 4.1 | — | — | W118 r2 (rule v2) | 6e4f11a95dc3d585 | c408836e |
| yl_y2035 | T4 | year ladder | certified | oscillatory | 106 | 167 | 0.89 | no | no | below 0.95 tau |  |  |  | — | — | 4.1 | — | — | W118 r2 (rule v2) | 3a2387f46448f9db | 7fca7670 |

### T3

Break-even fit, node 7 (slope b + c/4; Addendum 64 rulings 3-4). Energy cost 253,880 €/MWh; power cost 256,320 €/MVA. Manuscript figure: break-even ≤ 192,560 €/MWh; margin ≥ 61,320 €/MWh. Objective convention: gross = gross_operational_cost, settlement excluded (primary); net = gross − terminal salvage credit, beside; Q_cc = Q + t_sum, report-only.

| fit | n certified | intervals | certified-only e* (€/MWh) | banded midpoint e* (€/MWh) | e* min (€/MWh) | e* max (€/MWh) | margin min (€/MWh) | margin max (€/MWh) | slope b + c/4 min (€/MWh) | slope b + c/4 max (€/MWh) | b + c/4 agreement at midpoint (%) | b + c/4 agreement at corners (%) | b agreement at midpoint (%) | b agreement at corners (%) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| committed W145 (3 intervals) | 7 | n7_2h_e2, n7_2h_e3, n7_4h_e5 | 182,700 | 181,930 | 173,490 | 190,380 | 63,500 | 80,390 | 237,570 | 254,460 | 0.31 | 3.73 / 3.11 | 0.59 | 7.10 / 5.92 |
| conservative A64 (4 intervals, + d_4a82a64a) -- manuscript figure | 6 | n7_2h_e2, n7_2h_e3, n7_4h_e3, n7_4h_e5 | 183,080 | 181,930 | 171,310 | 192,560 | 61,320 | 82,570 | 235,390 | 256,640 | 0.47 | 4.76 / 3.83 | 0.91 | 10.15 / 8.33 |

### T4

Year ladder, 2035 − 2030 (M = I + Q − Q181, W118 form); gross primary, net beside. The investment-year comparison in this instance is decided by the salvage convention, not by operation. Objective convention: gross = gross_operational_cost, settlement excluded (primary); net = gross − terminal salvage credit, beside; Q_cc = Q + t_sum, report-only.

| row | M gross (k€) | salvage (k€) | M net (k€) | k* | threshold gross (k€) | × gross | verdict gross | threshold net (k€) | × net | verdict net | net label | eval key |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|
| 2030 | 68.5 | 11.8 | 56.7 | 169 | — | — |  | — | — |  |  | 6e4f11a95dc3d585 |
| 2035 | 111.0 | 56.5 | 54.5 | 167 | — | — |  | — | — |  |  | 3a2387f46448f9db |
| 2035 − 2030 | 42.5 | — | −2.3 | — | 12.3 | 3.46 | determinate | 12.3 | 0.19 | within resolution | validated by form + salvage identity (W154b) |  |

### T5

Phase B certificates against x = 0 (gross; v6 determinacy rule applied by DET.resolve_v6). pb_y2025_n5 is superseded by its v6 re-run pb_y2025_n5_v6. Objective convention: gross = gross_operational_cost, settlement excluded (primary); net = gross − terminal salvage credit, beside; Q_cc = Q + t_sum, report-only.

| cell | status | k0 | k* | range/τ | at_or_above_0.95_tau_counted | M gross (k€) | M Q_cc (k€, report-only) | v6 threshold (k€) | × | v6 verdict | recorded (W118 rule) | superseded | note |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|---|---|
| pb_y2030_n9 | certified | 120 | 192 | 0.94 | no | 46.7 | 47.0 | 12.8 | 3.64 | determinate | determinate | no | v6 rule applied here by DET.resolve_v6 (zero-solve); net not in the W154b validation set |
| pb_y2030_n7 | certified | 125 | 195 | 0.99 | yes | 49.0 | 49.6 | 13.5 | 3.62 | determinate | determinate | no | v6 rule applied here by DET.resolve_v6 (zero-solve); net not in the W154b validation set |
| pb_y2025_n5 | certified | 110 | 167 | 0.99 | no | 51.3 | 50.5 | 13.5 | 3.79 | determinate | determinate | yes | superseded by the v6 re-run pb_y2025_n5_v6 (claim C:y2025__n5_p0.25_e0.5): the W118 certificate sat on a non-Optimal cycle (Addendum 59) |
| pb_y2030_n5 | certified | 122 | 182 | 0.94 | no | 43.3 | 43.4 | 12.7 | 3.40 | determinate | determinate | no | v6 rule applied here by DET.resolve_v6 (zero-solve); net not in the W154b validation set |
| pb_y2025_n9 | certified | 113 | 174 | 0.81 | no | 52.5 | 52.7 | 12.6 | 4.15 | determinate | determinate | no | v6 rule applied here by DET.resolve_v6 (zero-solve); net not in the W154b validation set |
| pb_y2025_n7 | certified | 112 | 172 | 0.97 | yes | 54.9 | 55.0 | 13.3 | 4.14 | determinate | determinate | no | v6 rule applied here by DET.resolve_v6 (zero-solve); net not in the W154b validation set |
| pb_y2025_n5_v6 | certified | 110 | 199 | 0.16 | no | 49.7 | 49.8 | 12.6 | 3.94 | determinate |  | no | the extension v6 re-run (claim C:y2025__n5_p0.25_e0.5); T1 carries its full row |

### T6

Uncoordinated benchmark (Addendum 57; spec v5 bca69f97; report_v3 8d42dfb8), both no-reverse-flow (NRF) arms in full. NRF: pg_adn[s_m, s_o, p] ≥ 0 at every DSO interface, scenario and period (no export from a DN to the TN). Benefit = min(Q_passive, Q_price-taker) − Q181; each arm is the minimum over three starts. The coordinated solution has 4 reverse-flow interface-hours (3,308.3 MWh, block-weighted). Sweep: the unconstrained arm (no interface rule), cold start. Objective convention: gross = gross_operational_cost, settlement excluded (primary); net = gross − terminal salvage credit, beside; Q_cc = Q + t_sum, report-only.

| arrangement | start | Q gross (k€) | band (k€) | Q − Q181 (k€) | % of Q181 | × larger band | figure label | NRF violations (n) | NRF max excess (p.u.) | consistency pass effect (k€) | TN curtailment (GWh, day-weighted) | DN curtailment (GWh, day-weighted) | sweep: blocks TN cannot accept (of 12) | sweep: hours | sweep: failing blocks |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| coordinated (settled x = 0, Q181) |  | 653,873.7 | 71.9 | — | — | — | recorded | — | — | — | — | — | — | — |  |
| passive NRF | best of 3: perturbed | 895,049.9 | 25.8 | 241,176.2 | 36.9 | 3,353.11 | derived (W160) | — | — | — | — | — | 1 | 6 | 2035 Summer |
| passive NRF | cold | 895,075.7 | — | — | — | — | recorded | 29 | 0.00618 | −44.0 | 781.8 | 201.2 | — | — |  |
| passive NRF | perturbed | 895,049.9 | — | — | — | — | recorded | 29 | 0.00618 | −69.8 | 781.7 | 201.2 | — | — |  |
| passive NRF | warm_from_certified | 895,049.9 | — | — | — | — | recorded | 29 | 0.00618 | −69.8 | 781.8 | 201.2 | — | — |  |
| price-taker NRF | best of 3: warm_from_certified | 744,770.3 | 0.0 | 90,896.6 | 13.9 | 1,263.75 | recorded (× computed by W157) | — | — | — | — | — | 8 | 25 | 2025 Spring; 2025 Summer; 2030 Spring; 2030 Summer; 2030 Winter; 2035 Spring; 2035 Summer; 2035 Autumn |
| price-taker NRF | cold | 744,770.3 | — | — | — | — | recorded | 70 | 0.000321 | 0.2 | 806.7 | 0.6 | — | — |  |
| price-taker NRF | perturbed | 744,770.3 | — | — | — | — | recorded | 70 | 0.000321 | 0.2 | 807.3 | 0.6 | — | — |  |
| price-taker NRF | warm_from_certified | 744,770.3 | — | — | — | — | recorded | 70 | 0.000321 | 0.2 | 807.6 | 0.6 | — | — |  |
| passive NRF − price-taker NRF | best − best | — | — | 150,279.6 | — | — | recorded | — | — | — | — | — | — | — |  |

### T7

Discount rate (fixed plan: unit at node 7, 0.25 MVA / 1 MWh); one discount factor per representative year applied to the five years of its block; I paid in 2025. Objective convention: gross = gross_operational_cost, settlement excluded (primary); net = gross − terminal salvage credit, beside; Q_cc = Q + t_sum, report-only.

| rate (%) | V (k€) | I (k€) | value − I (k€) | threshold (k€) | × | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 277.1 | 318.0 | −40.8 | 15.9 | 2.57 | determinate |
| 2 | 253.5 | 318.0 | −64.4 | 13.1 | 4.93 | determinate |
| 5 | 225.2 | 318.0 | −92.7 | 13.1 | 7.10 | determinate |
| 8 | 203.4 | 318.0 | −114.6 | 13.1 | 8.78 | determinate |

### T8

Ageing arms at minimum SoH 0.70, fixed plan (unit at node 7, 0.25 MVA / 1 MWh), gross. The ε_AE (0.50) column is the superseded 0.50-era comparator (Addendum 65 read (a)). Objective convention: gross = gross_operational_cost, settlement excluded (primary); net = gross − terminal salvage credit, beside; Q_cc = Q + t_sum, report-only.

| arm | cell | value (k€) | I (k€) | value − I (k€) | threshold (k€, from T1) | × (from T1) | verdict | floor binds (0.70) | AE (PV-weighted) | EFC/day (PV-weighted) | k | ε_AE (0.70) | ε_AE resolvable (0.70) | ε_AE (0.50) -- superseded (0.50 era: C3, soh_min 0.50, E/P ≤ 10, no tight tail) |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|---:|
| C3_unit | e_c3_unit | 244.2 | 318.0 | −73.7 | 12.8 | 5.74 | determinate | 2035 | 0.786 | 0.76 | 11542 | — | — | — |
| C2 | e_c2 | 286.3 | 318.0 | −31.6 | 12.6 | 2.50 | determinate | never | 0.888 | 1.19 | 35851 | 1.30 | yes | 0.60 |
| C4 | e_c4 | 272.8 | 318.0 | −45.2 | 12.6 | 3.58 | determinate | never | 0.834 | 1.13 | 22429 | 1.85 | yes | 0.41 |
| C2_calfade | e_c2_calfade | 253.5 | 318.0 | −64.4 | 13.1 | 4.93 | determinate | 2035 | 0.793 | 0.86 | 35851 | 4.38 | no | 0.68 |
| C3_midblock | e_c3_midblock | 252.1 | 318.0 | −65.9 | 12.9 | 5.12 | determinate | 2035 | 0.836 | 0.76 | 11542 | 0.52 | no | 0.13 |
| no_ageing | e_no_ageing | 313.8 | 318.0 | −4.1 | 12.9 | 0.32 | within resolution | never | 1.000 | 1.29 | 11542 | 1.04 | yes | 0.62 |

### T9

Dual dead zone (Addendum 58 Ruling 1; Addendum 62; Addendum 64 ruling 7): the gap-refused and related uncertified cells, with the priced consensus gap at the cap and its node split. Objective convention: gross = gross_operational_cost, settlement excluded (primary); net = gross − terminal salvage credit, beside; Q_cc = Q + t_sum, report-only.

| cell | status | cause | t_sum at cap (k€) | share n5 (%) | share n7 (%) | share n9 (%) | pf_primal last | lapse resets at | entry |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| h_f9eae48f | uncertified | gap clause | −2.5 | 54.6 | 14.5 | 30.9 | 0.027 |  | gap-refused (the dead-zone label) |
| j_5f3cccb4 | uncertified | gap clause | −8.8 | 55.2 | 13.7 | 31.0 | 0.682 | 160 | gap-refused (the dead-zone label) |
| l_45aa25a6 | uncertified | gap clause | −9.2 | 55.3 | 13.7 | 31.0 | 0.690 |  | gap-refused (the dead-zone label) |
| l_7c455554 | uncertified | gap clause | −9.2 | 55.3 | 13.7 | 31.0 | 0.690 |  | gap-refused (the dead-zone label) |
| l_b2251bc5 | uncertified | gap clause | −9.4 | 55.3 | 13.7 | 31.0 | 0.700 |  | gap-refused (the dead-zone label) |
| l_0ee93aca | uncertified | lapse reset | −9.3 | 55.3 | 13.7 | 31.1 | 0.692 | 218,222 | by its signature, cause stated: two TSO recoveries (lapse resets at 218, 222) reset the rule before the gap clause was reached (Addendum 64 ruling 7a) |
| d_36686489 | uncertified | growth test after five TSO recoveries (not the dual dead zone; Addendum 65) | 1.1 | 58.3 | 11.3 | 30.3 | 0.012 |  | baseline-price degenerate case with large storage (Addendum 62): uncertified by the growth test after repeated TSO recoveries, not by the gap clause |
| ref:5ca4f86c | uncertified | gap clause | −9.2 | — | — | — | — |  | F2 incumbent (W118 r2 f2_incumbent, uncertified at 281, gap clause) |
| ref:e28de4ac | uncertified | gap clause | −9.3 | — | — | — | — |  | F2 challenger (W118 r2 f2_challenger, uncertified at 261, gap clause) |

### T10

Addendum 64 rows: the m = 1.75 pair and the minimum-SoH 0.50 row (both cells certified in each claim). At soh_min 0.50: EFC/day 1.19 / 1.10 / 0.91 (2025 / 2030 / 2035); 2035 SoH_end 0.677; the floor never binds (|dual| ≤ 5.1e-10). Objective convention: gross = gross_operational_cost, settlement excluded (primary); net = gross − terminal salvage credit, beside; Q_cc = Q + t_sum, report-only.

| claim | statement | ref | ref k0 / k* | ref range/τ | other | other k0 / k* | other range/τ | d gross (k€) | rule | threshold (k€) | × | verdict | d Q_cc (k€, report-only) | × Q_cc (report-only) |
|---|---|---|---|---:|---|---|---:|---:|---|---:|---:|---|---:|---:|
| H:m1.75:value_minus_I | flexibility ladder m = 1.75: sign of value - I (node 7, 0.25 MVA / 1 MWh), value = Q(0)_m1.75 - Q(unit)_m1.75 | h_x0_m175 | 108 / 139 | 0.79 | h_unit_m175 | 101 / 121 | 0.65 | 14.2 | max(3 x larger bar, 2 tau) | 10.8 | 1.32 | determinate | 12.9 | 1.20 |
| E:soh050:delta_value_vs_070 | minimum SoH row: Delta value = value(0.50) - value(0.70) = Q(3f084f2f, 0.70, k* 172) - Q(e_soh050); x = 0 is unaffected by soh_min (Q(0) = the settled x0 Q181 d110bd1a on both sides) | bd504ecf | — / 172 | 0.96 | e_soh050 | 112 / 184 | 0.92 | 4.9 | max(3 x larger bar, 2 tau) | 13.1 | 0.38 | within resolution | 4.8 | 0.37 |
| E:soh050:value_minus_I | the C2_calfade unit at minimum_soh 0.50: sign of value - I | 7aa017f0 | — / 181 | 0.93 | e_soh050 | 112 / 184 | 0.92 | −59.5 | max(3 x larger bar, 2 tau) | 12.6 | 4.71 | determinate | −60.1 | 4.76 |
| E:soh050:delta_value_vs_g070_REPORT_BESIDE | Delta value against the neutrality cell (identical to 3f084f2f through 172 by its gate) | g070_neutrality | 103 / 172 | 0.96 | e_soh050 | 112 / 184 | 0.92 | 4.9 | max(3 x larger bar, 2 tau) | 13.1 | 0.38 | within resolution | 4.8 | 0.37 |

### T11

The 3 × 3 multi-scenario instance (Addenda 52 and 54): x = 0 optimal under the baseline; R ∈ [0.909, 0.934] against 0.933 predicted. Not part of T1-T10; sources per row. Objective convention: gross = gross_operational_cost, settlement excluded (primary); net = gross − terminal salvage credit, beside; Q_cc = Q + t_sum, report-only.

| quantity | value | unit | label | source |
|---|---:|---|---|---|
| 3 x 3 storage value V = Q(0) - Q(unit), at certification (cycle 72) | 236.7 | k€ | recorded | data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_results.json value_and_R.value_eur (47a54c89) |
| 3 x 3 investment I (unit) | 318.0 | k€ | recorded | data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_results.json value_and_R.I_eur (47a54c89) |
| 3 x 3 value - I (x = 0 optimal under the baseline) | −81.3 | k€ | recorded; determinate True | data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_results.json value_and_R.value_minus_I (47a54c89); Addendum 52 (brief 4a80c3e2); P5_15_ADDENDUM51_CONTINUATION_REPORT.md (58ff8d88) line 60 |
| 3 x 3 resolution of the value | 18.0 | k€ | recorded | data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_results.json value_and_R.resolution (47a54c89) |
| 3 x 3 \|value - I\| / resolution | 4.52 | × | derived (W160): \|value_minus_I\| / resolution ("4.5x" in Addendum 52) | data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_results.json value_and_R |
| post-hoc settled descent D of the 3 x 3 x = 0 cell after certification | 6.2 | k€ | recorded (post hoc, damped-cosine fit; W99) | data/SRP1/Results/P515S53/w99_stage1_posthoc/w99_stage1_posthoc_analysis.json item3_POSTHOC_damped_oscillation.R_under_posthoc_limit.posthoc_primary_L.D (3fd1c15b) |
| V_SRP1 settled (the reference of R) | 253.5 | k€ | recorded (T7 row 2 %; = T1 CHECK headline V) | T7 (W153 discount row) |
| R at 3 x 3 certification against the settled V_SRP1 (upper end) | 0.934 | - | derived (W160): V / V_SRP1_settled | Addendum 54 (brief 271e9325); P5_15_ADDENDUM53_SRP1_CONTINUATION_REPORT.md (ebe34021) lines 118-119 ("0.9336") |
| R with the 3 x 3 x = 0 post-hoc settled descent (lower end) | 0.909 | - | derived (W160): (V - D) / V_SRP1_settled (W99 formula with R_ref = V_SRP1 settled) | Addendum 54; P5_15_ADDENDUM53_SRP1_CONTINUATION_REPORT.md lines 118-119 ("0.9090") |
| R predicted from the mean-profile spread | 0.9331 | - | recorded | data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_results.json value_and_R.R_prefix_recorded |
| 3 x 3 x = 0 replay bitwise through certification | 72 / 72 cycles | - | recorded (stated in Addendum 52; not re-read here) | Addendum 52; stage-1 evidence 4689475e; P5_15_ADDENDUM51_CONTINUATION_REPORT.md (58ff8d88) |

## Paragraph figure check (paragraphs.md against this JSON, at the written precision)

| id | written | table value | status | table reference | note |
|---|---|---|---|---|---|
| S1a | +14.2 | 14.2 | match | T10 H:m1.75 d gross |  |
| S1b | +46.3 | 46.3 | match | T1 H:m2 d gross |  |
| S1c | 1.63 | 1.63 | match | derived: 1.5 + 0.25·(−d(1.5)) / (d(1.75) − d(1.5)) from T1 H:m1.5 and T10 H:m1.75 |  |
| S2a | +0.18 | None | no table counterpart | T10 carries EFC/day at 0.50 only; the 0.70 per-year EFC/day (W153) is not in T1-T10 |  |
| S2b | +4.9 | 4.9 | match | T10 E:soh050:delta_value_vs_070 |  |
| S2c | within resolution | within resolution | match | T10 verdict |  |
| S2d | 59.5 | 59.5 | match | T10 E:soh050:value_minus_I |  |
| S2e | determinate | determinate | match | T10 verdict |  |
| V1a | −4.1 | -4.1 | match | T1 E no_ageing value − I |  |
| V1b | within resolution | within resolution | match | T1 verdict |  |
| V1c | 31.6 | 31.6 | match | T8 aged arms: smallest loss |  |
| V1d | 73.7 | 73.7 | match | T8 aged arms: largest loss |  |
| V1e | −4,140.54 | -4140.54 | match | T1 |  |
| V1f | 0.32 | 0.32 | match | T1 × |  |
| V1g | −31,607.24 | -31607.24 | match | T8 C2 |  |
| V1h | −73,746.66 | -73746.66 | match | T8 C3_unit |  |
| C1 | 4,539.07 | 4539.07 | match | constants.TAU |  |
| C2 | 453.91 | 453.91 | match | constants.TAU / 10 |  |
| C3 | 2,269.53 | 2269.53 | match | constants.TAU / 2 |  |
| C4 | 10 | 10 | match | T2 at_or_above_0.95_tau_counted (count of true) |  |
| C5 | 42 | 42 | match | T2 cells with certification_stats_included |  |
| C6 | 32 | 32 | match | T2 |  |
| C7 | 24 | 24 | match | T2 branch |  |
| C8 | 8 | 8 | match | T2 branch |  |
| C9 | 5 | 5 | match | T2 cause (W157 wording) = gap clause |  |
| C10 | 2 | 2 | match | T2 cause (W157 wording) = lapse reset |  |
| C11 | 3 | 3 | match | T2 cause (W157 wording) = growth test |  |
| C12 | 174 | 174 | match | T2 median k* (32 certified) |  |
| C13 | 63.5 | 65.0 | MISMATCH | T2: median of k* − k0, where T2 k0 = k0_run = the first residual pass N | W153 (input, not a table) gives 63.5 for k* − k0 AT DECISION (k0 reset by lapses on e_c3_midblock and e_no_ageing) = 63.5; measured from the first residual pass N, as the sentence says, the median is 65.0 (W153 k_star_minus_N median 65.0) |
| C14 | 21 | 21 | match | T2 min k* − k0 |  |
| C15 | 89 | 89 | match | T2 max k* − k0 |  |
| C16 | 0.86 | 0.86 | match | T2 median range/τ (32 certified) |  |
| C17 | 4 | 4 | match | T10 cells certified |  |
| C18 | 138 | 138 | match | T2 k* quantile 0 (linear) |  |
| C19 | 156.75 | 156.75 | match | T2 k* quantile 0.25 (linear) |  |
| C20 | 174 | 174 | match | T2 k* quantile 0.5 (linear) |  |
| C21 | 198.25 | 198.25 | match | T2 k* quantile 0.75 (linear) |  |
| C22 | 400 | 400 | match | T2 k* quantile 1 (linear) |  |
| C23 | 0.012 | 0.012 | match | T2 range/τ (32 certified) |  |
| C24 | 0.860 | 0.860 | match | T2 range/τ (32 certified) |  |
| C25 | 0.997 | 0.997 | match | T2 range/τ (32 certified) |  |
| C26 | 6 | 6 | match | T2 (42-cell set) |  |
| C27 | 1.72 | 1.72 | match | T2 terminal step / EPS0, certified |  |
| C28 | 10.42 | 10.42 | match | T2, certified |  |
| C29 | 25 | 25 | match | T2, all 42 cells | over all 42 cells; over the 32 certified cells the count is 22 |
| C30 | 46 | 46 | match | T2 n_vetoes |  |
| C31 | 17 | 17 | match | T2 |  |
| C32 | 29 | 29 | match | T2 |  |
| C33 | 21 | 21 | match | T2 non-clean cycles after N |  |
| C34 | 10 | 10 | match | T2 |  |
| C35 | 0.970 | 0.970 | match | T2 b_2a0ba8b2 range/τ |  |
| C36 | 0.989 | 0.989 | match | T2 b_4649234b range/τ |  |
| C37 | 0.959 | 0.959 | match | T2 ref:bd504ecf range/τ |  |
| C38 | 0.994 | 0.994 | match | T2 pb_y2030_n7 range/τ |  |
| C39 | 0.993 | 0.993 | match | T2 pb_y2025_n5 range/τ |  |
| C40 | 0.974 | 0.974 | match | T2 pb_y2025_n7 range/τ |  |
| C41 | 138 | 138 | match | T2 j_a11d7966 NCTP |  |
| C41b | 15,879.58 | 15879.58 | match | T2 j_a11d7966 uncertified form bar |  |
| C43 | 108 | 108 | match | T2 d_4a82a64a NCTP |  |
| C43b | 70,743.93 | 70743.93 | match | T2 d_4a82a64a uncertified form bar |  |
| L1 | −8.8 | -8.8 | match | T9 m = 2 cells and F2 references: max t_sum |  |
| L2 | −9.4 | -9.4 | match | T9: min t_sum |  |
| L3 | 55 | 55 | match | T9 node 5 share of the m = 2 cells with a per-node record (j_5f3cccb4, l_45aa25a6, l_7c455554, l_b2251bc5, l_0ee93aca), integer % |  |
| L4 | 31 | 31 | match | T9 node 9 share of the m = 2 cells with a per-node record (j_5f3cccb4, l_45aa25a6, l_7c455554, l_b2251bc5, l_0ee93aca), integer % |  |
| L5 | 14 | 14 | match | T9 node 7 share of the m = 2 cells with a per-node record (j_5f3cccb4, l_45aa25a6, l_7c455554, l_b2251bc5, l_0ee93aca), integer % |  |
| L5b | 6 | 5 | MISMATCH | T9: m = 2 cells with a per-node record | T9 carries j_5f3cccb4, l_45aa25a6, l_7c455554, l_b2251bc5, l_0ee93aca |
| L6 | 0.69 | 0.6819–0.6999 | match | T9 pf_primal last, six m = 2 cells (within ±0.01 of the written value) |  |
| L7 | +1.1 | 1.1 | match | T9 d_36686489 t_sum |  |
| L8 | 5 | 5 | match | T2 d_36686489 non-clean cycles after N |  |
| L9 | 1.04 | 1.04 | match | T8 ε_AE band over resolvable arms (0.70) |  |
| L10 | 1.85 | 1.85 | match | T8 |  |
| L11 | 0.6 | C2 0.60, C4 0.41, C2_calfade 0.68, C3_midblock 0.13, no_ageing 0.62 | approximate: not every arm rounds to 0.6 | T8 ε_AE (0.50, superseded) column, per arm |  |
| L12.0 | −8,847.90 | -8847.90 | match | T9 j_5f3cccb4 t_sum at cap |  |
| L12.1 | −9,197.61 | -9197.61 | match | T9 l_45aa25a6 t_sum at cap |  |
| L12.2 | −9,247.76 | -9247.76 | match | T9 l_7c455554 t_sum at cap |  |
| L12.3 | −9,401.86 | -9401.86 | match | T9 l_b2251bc5 t_sum at cap |  |
| L12.4 | −9,300.09 | -9300.09 | match | T9 l_0ee93aca t_sum at cap |  |
| L12.5 | −9,234.42 | -9234.42 | match | T9 ref:5ca4f86c t_sum at cap |  |
| L12.6 | −9,300.13 | -9300.13 | match | T9 ref:e28de4ac t_sum at cap |  |
| L13 | 0.6819 | 0.6819 | match | T9 min pf_primal last (m = 2 cells) |  |
| L14 | 0.6999 | 0.6999 | match | T9 max |  |
| L15 | −2,481.75 | -2481.75 | match | T9 h_f9eae48f |  |
| L16 | 55/14/31 | 55/15/31 | MISMATCH | T9 h_f9eae48f shares n5/n7/n9, integer % | the m = 2 split is 55/14/31; h_f9eae48f's node-7 share is 14.500 %, which rounds to 15 |
| L17 | 218,222 | 218,222 | match | T9 l_0ee93aca lapse events |  |
| L18 | +1,082.23 | 1082.23 | match | T9 |  |
| L19 | 58/11/30 | 58/11/30 | match | T9 d_36686489 shares |  |
| L20 | 7 | 7 | match | T2 monotone L certificates |  |
| L21 | 0.012 | 0.012 | match | T2 |  |
| L22 | 0.171 | 0.171 | match | T2 |  |
| L23 | 0.939 | 0.939 | match | T2 |  |
| L24 | 1.04 | 1.04 | match | T8 ε_AE (0.70) no_ageing |  |
| L25 | 1.30 | 1.30 | match | T8 ε_AE (0.70) C2 |  |
| L26 | 1.85 | 1.85 | match | T8 ε_AE (0.70) C4 |  |
| P14 | 1 | 1 | match | T6 sweep passive |  |
| P14b | 8 | 8 | match | T6 sweep price-taker |  |
| P20 | +27,488 | 27488 | match | T2 appended pb_y2030_n9 s (W118) |  |
| P22 | 173 | 173 | match | T2 b_2a0ba8b2 k* |  |
| P24 | +20,913 | 20913 | match | T2 s |  |
| P32a | 0.59 | 0.59 | match | T3 committed W145 b agreement at midpoint |  |
| P32b | 0.31 | 0.31 | match | T3 committed W145 b + c/4 agreement at midpoint |  |
| P32c | 7.1 | 7.1 | match | T3 committed W145: worst corner on b |  |
| P32d | 3.7 | 3.7 | match | T3 committed W145: worst corner on b + c/4 |  |
| P32e | 0.47 | 0.47 | match | T3 conservative b + c/4 midpoint |  |
| P33a | 182,702 | 182702 | match | T3 committed certified-only e* |  |
| P33b | 63.5 | 63.5 | match | T3 committed margin min (k€/MWh) |  |
| P33c | 61.3 | 61.3 | match | T3 conservative margin min (k€/MWh) |  |
| P35 | −14.45 | -14.45 | match | T1 H:m1.5 |  |
| P36a | +46.3 | 46.3 | match | T1 H:m2 |  |
| P36b | +46.5 | 46.5 | match | T1 I second MWh |  |
| P37 | 1.95 | 1.95 | match | T1 J:e4_to_e5 × |  |
| P38 | −8,848 | -8848 | match | T9 j_5f3cccb4 |  |
| P47a | +14,249.16 | 14249.16 | match | T10 |  |
| P47b | 1.32 | 1.32 | match | T10 × |  |
| P47c | 10,775 | 10775 | match | T10 |  |
| P47d | 121 | 121 | match | T10 |  |
| P47e | 108 | 108 | match | T10 |  |
| P47f | 101 | 101 | match | T10 |  |
| P48a | +4,916.67 | 4916.67 | match | T10 |  |
| P48b | 0.38 | 0.38 | match | T10 |  |
| P48c | 13,054.22 | 13054.22 | match | T10 |  |
| P48d | 5.1e-10 | 5.1e-10 | match | T10 scored B floor duals |  |
| P48e | 0.677 | 0.677 | match | T10 |  |
| P48f | 1.192 | 1.192 | match | T10 EFC/day 2025 |  |
| P48g | 1.100 | 1.100 | match | T10 EFC/day 2030 |  |
| P48h | 0.909 | 0.909 | match | T10 EFC/day 2035 |  |
| M1 | +42,458.65 | 42458.65 | match | T4 |  |
| M2 | −2,286.25 | -2286.25 | match | T4 |  |
| M3 | −59,500.72 | -59500.72 | match | T10 |  |
| M4 | −64,417.39 | -64417.39 | match | T8 |  |
| M5.0a | −4,140.54 | -4140.54 | match | T8 no_ageing |  |
| M5.0b | 0.32 | 0.32 | match | T1 E:n7_4h_e1_no_ageing:value_minus_I × |  |
| M5.0c | — | — | match | T8 no_ageing floor binds (no_ageing: no floor year recorded, shown —) |  |
| M5.1a | −31,607.24 | -31607.24 | match | T8 C2 |  |
| M5.1b | 2.50 | 2.50 | match | T1 E:n7_4h_e1_C2:value_minus_I × |  |
| M5.1c | never | never | match | T8 C2 floor binds (no_ageing: no floor year recorded, shown —) |  |
| M5.2a | −64,417.39 | -64417.39 | match | T8 C2_calfade |  |
| M5.2b | 4.93 | 4.93 | match | T1 E:n7_4h_e1_C2_calfade:value_minus_I × |  |
| M5.2c | 2035 | 2035 | match | T8 C2_calfade floor binds (no_ageing: no floor year recorded, shown —) |  |
| M5.3a | −73,746.66 | -73746.66 | match | T8 C3_unit |  |
| M5.3b | 5.74 | 5.74 | match | T1 E:C3_unit_value_minus_I × |  |
| M5.3c | 2035 | 2035 | match | T8 C3_unit floor binds (no_ageing: no floor year recorded, shown —) |  |
| M5.4a | −45,196.70 | -45196.70 | match | T8 C4 |  |
| M5.4b | 3.58 | 3.58 | match | T1 E:n7_4h_e1_C4:value_minus_I × |  |
| M5.4c | never | never | match | T8 C4 floor binds (no_ageing: no floor year recorded, shown —) |  |
| M5.5a | −65,891.88 | -65891.88 | match | T8 C3_midblock |  |
| M5.5b | 5.12 | 5.12 | match | T1 E:n7_4h_e1_C3_midblock:value_minus_I × |  |
| M5.5c | 2035 | 2035 | match | T8 C3_midblock floor binds (no_ageing: no floor year recorded, shown —) |  |
| M6 | +69,606.12 | 69606.12 | match | T1 E no_ageing vs C3 |  |
| M6b | 5.41 | 5.41 | match | T1 |  |
| M7 | 4.71 | 4.71 | match | T10 |  |
| M8 | determinate ×4 | determinate ×4 | match | T7 verdicts and signs |  |
| M9 | [0.909, 0.934] | [0.909, 0.934] | match | T11 |  |
| M10 | +90,896,608.40 | 90896608.40 | match | T6 |  |
| M11 | 13.9 | 13.9 | match | T6 |  |
| M12 | 1,263.75 | 1263.75 | match | T6 |  |
| M13 | 71,926.11 | 71926.11 | match | T6 |  |
| M14 | 4 | 4 | match | T6 |  |
| M15 | −64,417.39 | -64417.39 | match | T1 CHECK headline |  |

## Checks

- predecessor_sha_matches_expected_and_manifest: True
- every_input_committed_clean: True
- every_input_manifest_matches: True
- w157_rebuild_equals_predecessor_all_tables: True
- package_blob_equals_working_copy: True
- count_at_or_above_0_95_tau_equals_10: True
- counted_set_equals_addendum65_ten: True
- column_true_on_exactly_ten_rows: True
- twins_bitwise_evidence_holds: True
- d_36686489_growth_test_with_five_non_clean: True
- t6_passive_derived_equals_recorded_decomposition: True
- t11_R_rounds_to_0909_0934: True
- paragraph_fragments_all_found_verbatim: True
- export T1 (LaTeX ASCII / cell counts / braces; CSV parse): True
- export T2 (LaTeX ASCII / cell counts / braces; CSV parse): True
- export T3 (LaTeX ASCII / cell counts / braces; CSV parse): True
- export T4 (LaTeX ASCII / cell counts / braces; CSV parse): True
- export T5 (LaTeX ASCII / cell counts / braces; CSV parse): True
- export T6 (LaTeX ASCII / cell counts / braces; CSV parse): True
- export T7 (LaTeX ASCII / cell counts / braces; CSV parse): True
- export T8 (LaTeX ASCII / cell counts / braces; CSV parse): True
- export T9 (LaTeX ASCII / cell counts / braces; CSV parse): True
- export T10 (LaTeX ASCII / cell counts / braces; CSV parse): True
- export T11 (LaTeX ASCII / cell counts / braces; CSV parse): True
