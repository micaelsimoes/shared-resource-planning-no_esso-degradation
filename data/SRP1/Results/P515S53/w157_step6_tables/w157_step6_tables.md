# P5.15 W157 — Step 6 tables (DRAFT for the Planner)

Generated 2026-10-05T09:27:43.281130+00:00 by `p515_s53_w157_step6_tables.py` (sha256 `35c5dd144e19`), git HEAD `4c89daf8ee37`; zero solves, no model loads. Every input is committed clean; its sha256 is in `w157_step6_tables.json` → `inputs`. **A64 rows: PENDING W156.**

**Objective convention (every table):** Q = gross_operational_cost, settlement EXCLUDED (the frozen primary, Addendum 58 Ruling 3); Q_net = net_operational_recourse = Q - terminal salvage credit (beside); Q_cc = Q + t_sum (first-order consensus-consistent diagnostic, REPORT-ONLY); value = Q(0) - Q(x); F = Q + I; EUR. τ = 4,539.07 €; certified-pair determinacy threshold max(3 × larger band, 2τ = 9,078.14 €) (Addendum 61); uncertified form bar = 3·max(|gap|, |slack|) in both gross and Q_cc (Addendum 58). Multiple = |d| / threshold-or-bar.

**Net label:** every net figure is "validated by form + salvage identity (W154b)", except G rows: "recorded (claim record net_of_salvage)".

## T1 — claims (gross primary; net beside; Q_cc report-only)

| claim | ref cell [status] | other cell [status] | d gross | rule | threshold / bar | × | verdict (gross) | d net | net × | net verdict | net label | d Q_cc (r-o) | Q_cc × | instance (eval / cand.) | notes |
|---|---|---|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|---|
| `B:n5_4h_e1` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | b_2a0ba8b2 [cert. oscillatory k*173 r/τ 0.970 [≥0.95τ]] | 58,036.03 | max(3 x larger bar, 2 tau) | 13,214.10 | 4.39× | determinate | 58,036.03 | 4.39× | determinate | W154b | 58,909.84 | 4.46× | ref:7aa017f0: d110bd1a/8435c718; b_2a0ba8b2: 33447912/18216659 |  |
| `B:n7_2h_e1` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | d_d3709599 [cert. oscillatory k*175 r/τ 0.863] | 114,078.43 | max(3 x larger bar, 2 tau) | 12,627.51 | 9.03× | determinate | 114,078.43 | 9.03× | determinate | W154b | 114,467.68 | 9.06× | ref:7aa017f0: d110bd1a/8435c718; d_d3709599: 64408bae/3993aebc |  |
| `B:n7_2h_e2` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | d_a12d95a2 [UNCERT. (lapse reset) end 203] | 231,709.80 | uncertified form | 24,235.85 | 9.56× | determinate | 231,709.80 | 9.56× | determinate | W154b | 232,030.17 | 9.57× | ref:7aa017f0: d110bd1a/8435c718; d_a12d95a2: 9bc66d9e/423e5a7f |  |
| `B:n7_2h_e3` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | d_f759dd48 [UNCERT. (growth test) end 198] | 361,683.71 | uncertified form | 41,380.26 | 8.74× | determinate | 361,683.71 | 8.74× | determinate | W154b | 362,307.63 | 8.76× | ref:7aa017f0: d110bd1a/8435c718; d_f759dd48: 5f1b1aa6/2cd1de82 |  |
| `B:n7_2h_e4` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | d_c7fee8be [cert. oscillatory k*145 r/τ 0.984 [≥0.95τ]] | 494,416.01 | max(3 x larger bar, 2 tau) | 13,396.62 | 36.91× | determinate | 494,416.01 | 36.91× | determinate | W154b | 496,374.78 | 37.05× | ref:7aa017f0: d110bd1a/8435c718; d_c7fee8be: aa0b99f5/7048a00e |  |
| `B:n7_2h_e5` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | d_9246ed01 [cert. oscillatory k*138 r/τ 0.985 [≥0.95τ]] | 616,554.29 | max(3 x larger bar, 2 tau) | 13,419.54 | 45.94× | determinate | 616,554.29 | 45.94× | determinate | W154b | 617,259.12 | 46.00× | ref:7aa017f0: d110bd1a/8435c718; d_9246ed01: 579a1370/354b3881 |  |
| `B:n7_4h_e1` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | ref:bd504ecf [cert. k*172 r/τ 0.959 [≥0.95τ]] | 64,417.39 | max(3 x larger bar, 2 tau) | 13,054.22 | 4.93× | determinate | 64,417.39 | 4.93× | determinate | W154b | 64,937.86 | 4.97× | ref:7aa017f0: d110bd1a/8435c718; ref:bd504ecf: 3f084f2f/db77e154 |  |
| `B:n7_4h_e2` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | d_c52e1670 [cert. oscillatory k*150 r/τ 0.982 [≥0.95τ]] | 135,539.68 | max(3 x larger bar, 2 tau) | 13,378.75 | 10.13× | determinate | 135,539.68 | 10.13× | determinate | W154b | 136,713.29 | 10.22× | ref:7aa017f0: d110bd1a/8435c718; d_c52e1670: af42a163/cc4eb5df |  |
| `B:n7_4h_e3` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | d_4a82a64a [cert. oscillatory k*142 r/τ 0.722 [NCTP 108]] | 208,290.70 | max(3 x larger bar, 2 tau) | 12,627.51 | 16.49× | determinate | 208,290.70 | 16.49× | determinate | W154b | 209,148.61 | 16.56× | ref:7aa017f0: d110bd1a/8435c718; d_4a82a64a: 14b00a04/fb21d822 | A64 ruling 4: certificate(s) d_4a82a64a rest on a non-clean turning point; kept and flagged; uncertified form beside: bar 70,743.93, 2.94× (determinate) |
| `B:n7_4h_e4` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | d_3632b0ae [cert. oscillatory k*153 r/τ 0.997 [≥0.95τ]] | 273,951.65 | max(3 x larger bar, 2 tau) | 13,571.15 | 20.19× | determinate | 273,951.65 | 20.19× | determinate | W154b | 274,652.34 | 20.24× | ref:7aa017f0: d110bd1a/8435c718; d_3632b0ae: 1fe57699/11725e45 |  |
| `B:n7_4h_e5` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | d_36686489 [UNCERT. (growth test) end 191] | 347,243.79 | uncertified form | 23,919.66 | 14.52× | determinate | 347,243.79 | 14.52× | determinate | W154b | 348,183.50 | 14.56× | ref:7aa017f0: d110bd1a/8435c718; d_36686489: 4de61708/cab98853 |  |
| `B:n9_4h_e1` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | b_0dd237f0 [cert. oscillatory k*173 r/τ 0.921] | 61,100.55 | max(3 x larger bar, 2 tau) | 12,627.51 | 4.84× | determinate | 61,100.55 | 4.84× | determinate | W154b | 61,550.00 | 4.87× | ref:7aa017f0: d110bd1a/8435c718; b_0dd237f0: 38d09af2/a361ac93 |  |
| `B:n9_4h_e3` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | b_4649234b [cert. oscillatory k*148 r/τ 0.989 [≥0.95τ]] | 199,097.53 | max(3 x larger bar, 2 tau) | 13,471.87 | 14.78× | determinate | 199,097.53 | 14.78× | determinate | W154b | 200,983.73 | 14.92× | ref:7aa017f0: d110bd1a/8435c718; b_4649234b: 4de68ced/ace7403b |  |
| `C:y2025__n5_p0.25_e0.5` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | pb_y2025_n5_v6 [cert. oscillatory k*199 r/τ 0.158] | 49,692.73 | max(3 x larger bar, 2 tau) | 12,627.51 | 3.94× | determinate | 49,692.73 | 3.94× | determinate | W154b | 49,798.49 | 3.94× | ref:7aa017f0: d110bd1a/8435c718; pb_y2025_n5_v6: be95e576/33031c08 |  |
| `C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | c_156ce2d1 [cert. oscillatory k*176 r/τ 0.858] | 102,096.79 | max(3 x larger bar, 2 tau) | 12,627.51 | 8.09× | determinate | 102,096.79 | 8.09× | determinate | W154b | 102,671.97 | 8.13× | ref:7aa017f0: d110bd1a/8435c718; c_156ce2d1: 3a4c10c8/ba670a65 |  |
| `C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | c_6597a79d [cert. oscillatory k*193 r/τ 0.984 [≥0.95τ]] | 93,877.52 | max(3 x larger bar, 2 tau) | 13,405.84 | 7.00× | determinate | 84,022.96 | 6.27× | determinate | W154b | 93,676.40 | 6.99× | ref:7aa017f0: d110bd1a/8435c718; c_6597a79d: 3a0d03e8/a5bb68bf |  |
| `G:n7_2h_e1_y2030:gross` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | g_37b5c499 [cert. oscillatory k*185 r/τ 0.941] | 99,027.04 | max(3 x larger bar, 2 tau) | 12,818.83 | 7.73× | determinate | 88,337.82 | 6.89× | determinate | recorded | 99,990.07 | 7.80× | ref:7aa017f0: d110bd1a/8435c718; g_37b5c499: bf2d8348/e7c94efa |  |
| `G:n7_2h_e1_y2030:net` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | g_37b5c499 [cert. oscillatory k*185 r/τ 0.941] | 99,027.04 | max(3 x larger bar, 2 tau) | 12,818.83 | 7.73× | determinate | 88,337.82 | 6.89× | determinate | recorded | 99,990.07 | 7.80× | ref:7aa017f0: d110bd1a/8435c718; g_37b5c499: bf2d8348/e7c94efa |  |
| `G:n7_2h_e1_y2035:gross` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | g_47dce43c [cert. oscillatory k*174 r/τ 0.840] | 139,659.51 | max(3 x larger bar, 2 tau) | 12,627.51 | 11.06× | determinate | 85,088.88 | 6.74× | determinate | recorded | 139,870.07 | 11.08× | ref:7aa017f0: d110bd1a/8435c718; g_47dce43c: 46ff23a1/803d1a6c |  |
| `G:n7_2h_e1_y2035:net` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | g_47dce43c [cert. oscillatory k*174 r/τ 0.840] | 139,659.51 | max(3 x larger bar, 2 tau) | 12,627.51 | 11.06× | determinate | 85,088.88 | 6.74× | determinate | recorded | 139,870.07 | 11.08× | ref:7aa017f0: d110bd1a/8435c718; g_47dce43c: 46ff23a1/803d1a6c |  |
| `G:n7_4h_e2_y2030:gross` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | g_48749148 [cert. oscillatory k*170 r/τ 0.926] | 137,432.30 | max(3 x larger bar, 2 tau) | 12,627.51 | 10.88× | determinate | 113,065.99 | 8.95× | determinate | recorded | 138,529.20 | 10.97× | ref:7aa017f0: d110bd1a/8435c718; g_48749148: 22331aa7/76baf11e |  |
| `G:n7_4h_e2_y2030:net` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | g_48749148 [cert. oscillatory k*170 r/τ 0.926] | 137,432.30 | max(3 x larger bar, 2 tau) | 12,627.51 | 10.88× | determinate | 113,065.99 | 8.95× | determinate | recorded | 138,529.20 | 10.97× | ref:7aa017f0: d110bd1a/8435c718; g_48749148: 22331aa7/76baf11e |  |
| `G:n7_4h_e2_y2035:gross` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | g_9abf31d4 [cert. oscillatory k*172 r/τ 0.845] | 223,582.70 | max(3 x larger bar, 2 tau) | 12,627.51 | 17.71× | determinate | 109,133.90 | 8.64× | determinate | recorded | 223,998.12 | 17.74× | ref:7aa017f0: d110bd1a/8435c718; g_9abf31d4: 5560d0f0/ca1b7bac |  |
| `G:n7_4h_e2_y2035:net` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | g_9abf31d4 [cert. oscillatory k*172 r/τ 0.845] | 223,582.70 | max(3 x larger bar, 2 tau) | 12,627.51 | 17.71× | determinate | 109,133.90 | 8.64× | determinate | recorded | 223,998.12 | 17.74× | ref:7aa017f0: d110bd1a/8435c718; g_9abf31d4: 5560d0f0/ca1b7bac |  |
| `H:m1.5:value_minus_I` | h_aa8a76d7 [cert. oscillatory k*153 r/τ 0.910] | h_f9eae48f [UNCERT. (gap clause) end 194] | -14,450.28 | uncertified form | 18,751.08 | 0.77× | within the uncertified bar | -14,450.28 | 0.77× | within the uncertified bar | W154b | -12,520.40 | 0.67× | h_aa8a76d7: 22dd5ef8/8435c718; h_f9eae48f: aed2d618/db77e154 | A63: sign unresolved at m = 1.5 (uncertified form, gap-refused unit cell); crossing at or below m = 1.75; the m = 1.75 row is PENDING W156 |
| `H:m2:value_minus_I` | h_50dea31c [cert. oscillatory k*146 r/τ 0.484] | h_74eda68d [cert. oscillatory k*150 r/τ 0.901] | 46,285.46 | max(3 x larger bar, 2 tau) | 12,272.19 | 3.77× | determinate | 46,285.46 | 3.77× | determinate | W154b | 48,182.77 | 3.93× | h_50dea31c: 354b1bda/8435c718; h_74eda68d: dca08628/db77e154 |  |
| `I:m2:e2_value_minus_I` | h_50dea31c [cert. oscillatory k*146 r/τ 0.484] | i_5a6a88b4 [cert. monotone k*198 r/τ 0.939] | 92,800.22 | max(3 x larger bar, 2 tau) | 12,785.28 | 7.26× | determinate | 92,800.22 | 7.26× | determinate | W154b | 93,187.58 | 7.29× | h_50dea31c: 354b1bda/8435c718; i_5a6a88b4: e046a8cf/cc4eb5df |  |
| `I:m2:second_MWh` | h_74eda68d [cert. oscillatory k*150 r/τ 0.901] | i_5a6a88b4 [cert. monotone k*198 r/τ 0.939] | 46,514.76 | max(3 x larger bar, 2 tau) | 12,785.28 | 3.64× | determinate | 46,514.76 | 3.64× | determinate | W154b | 45,004.81 | 3.52× | h_74eda68d: dca08628/db77e154; i_5a6a88b4: e046a8cf/cc4eb5df |  |
| `J:e2_to_e3` | i_5a6a88b4 [cert. monotone k*198 r/τ 0.939] | j_f3aa335e [UNCERT. (growth test) end 260] | 40,972.63 | uncertified form | 27,694.48 | 1.48× | determinate | 40,972.63 | 1.48× | determinate | W154b | 41,055.94 | 1.48× | i_5a6a88b4: e046a8cf/cc4eb5df; j_f3aa335e: af066d6b/fb21d822 |  |
| `J:e3_to_e4` | j_f3aa335e [UNCERT. (growth test) end 260] | j_a11d7966 [cert. oscillatory k*158 r/τ 0.715 [NCTP 138]] | 42,807.46 | uncertified form | 27,694.48 | 1.55× | determinate | 42,807.46 | 1.55× | determinate | W154b | 42,417.40 | 1.53× | j_f3aa335e: af066d6b/fb21d822; j_a11d7966: 87efbc3d/11725e45 | A64 ruling 4: certificate(s) j_a11d7966 rest on a non-clean turning point; kept and flagged; uncertified form beside: bar 27,694.48, 1.55× (determinate) |
| `J:e3_value_minus_I` | h_50dea31c [cert. oscillatory k*146 r/τ 0.484] | j_f3aa335e [UNCERT. (growth test) end 260] | 133,772.85 | uncertified form | 27,694.48 | 4.83× | determinate | 133,772.85 | 4.83× | determinate | W154b | 134,243.52 | 4.85× | h_50dea31c: 354b1bda/8435c718; j_f3aa335e: af066d6b/fb21d822 |  |
| `J:e4_to_e5` | j_a11d7966 [cert. oscillatory k*158 r/τ 0.715 [NCTP 138]] | j_5f3cccb4 [UNCERT. (gap clause) end 270] | 51,820.48 | uncertified form | 26,543.70 | 1.95× | determinate | 51,820.48 | 1.95× | determinate | W154b | 61,309.99 | 2.31× | j_a11d7966: 87efbc3d/11725e45; j_5f3cccb4: cc7f0059/cab98853 | A64 ruling 4: certificate(s) j_a11d7966 rest on a non-clean turning point; kept and flagged; uncertified form beside: bar 26,543.70, 1.95× (determinate) |
| `J:e4_value_minus_I` | h_50dea31c [cert. oscillatory k*146 r/τ 0.484] | j_a11d7966 [cert. oscillatory k*158 r/τ 0.715 [NCTP 138]] | 176,580.30 | max(3 x larger bar, 2 tau) | 9,734.72 | 18.14× | determinate | 176,580.30 | 18.14× | determinate | W154b | 176,660.92 | 18.15× | h_50dea31c: 354b1bda/8435c718; j_a11d7966: 87efbc3d/11725e45 | A64 ruling 4: certificate(s) j_a11d7966 rest on a non-clean turning point; kept and flagged; uncertified form beside: bar 15,879.58, 11.12× (determinate) / A64 ruling 4: the MANUSCRIPT uses J 4 MWh at its uncertified bar |
| `J:e5_value_minus_I` | h_50dea31c [cert. oscillatory k*146 r/τ 0.484] | j_5f3cccb4 [UNCERT. (gap clause) end 270] | 228,400.78 | uncertified form | 26,543.70 | 8.60× | determinate | 228,400.78 | 8.60× | determinate | W154b | 237,970.91 | 8.97× | h_50dea31c: 354b1bda/8435c718; j_5f3cccb4: cc7f0059/cab98853 |  |
| `L:df1a5525_not_a_neighbour` | ref:5ca4f86c [UNCERT. (gap clause) end 281] | l_df1a5525 [cert. monotone k*236 r/τ 0.138] | 47,651.09 | uncertified form | 27,703.27 | 1.72× | determinate | 58,515.77 | 2.11× | determinate | W154b | 56,812.49 | 2.05× | ref:5ca4f86c: 24c5ccb6/59757776; l_df1a5525: 6c8f6353/7b85ff44 |  |
| `L:y2025__n7_p0.75_e3` | ref:5ca4f86c [UNCERT. (gap clause) end 281] | j_f3aa335e [UNCERT. (growth test) end 260] | 94,347.05 | uncertified form | 27,703.27 | 3.41× | determinate | 133,390.26 | 4.81× | determinate | W154b | 103,833.03 | 3.75× | ref:5ca4f86c: 24c5ccb6/59757776; j_f3aa335e: af066d6b/fb21d822 |  |
| `L:y2030__n5_p0.25_e0.5__n7_p0.75_e3` | ref:5ca4f86c [UNCERT. (gap clause) end 281] | l_195156fa [cert. monotone k*252 r/τ 0.098] | 28,621.21 | uncertified form | 27,703.27 | 1.03× | determinate | 32,373.87 | 1.17× | determinate | W154b | 37,774.51 | 1.36× | ref:5ca4f86c: 24c5ccb6/59757776; l_195156fa: 1fd23b8d/a033945d |  |
| `L:y2030__n5_p0.25_e0.5__n7_p0.75_e3__n9_p0.25_e0.5` | ref:5ca4f86c [UNCERT. (gap clause) end 281] | l_b2251bc5 [UNCERT. (gap clause) end 320] | 9,803.71 | uncertified form | 28,205.57 | 0.35× | within the uncertified bar | 10,965.87 | 0.39× | within the uncertified bar | W154b | 9,636.27 | 0.34× | ref:5ca4f86c: 24c5ccb6/59757776; l_b2251bc5: abf4ac0a/c29e38c3 |  |
| `L:y2030__n5_p0.25_e0.5__n7_p1.25_e3` | ref:5ca4f86c [UNCERT. (gap clause) end 281] | l_2ab0ce2d [cert. monotone k*400 r/τ 0.092] | 53,259.29 | uncertified form | 27,703.27 | 1.92× | determinate | 62,774.73 | 2.27× | determinate | W154b | 62,494.61 | 2.26× | ref:5ca4f86c: 24c5ccb6/59757776; l_2ab0ce2d: 75266862/9f4f66b6 |  |
| `L:y2030__n5_p0.25_e0.5__n7_p1_e3` | ref:5ca4f86c [UNCERT. (gap clause) end 281] | l_0ee93aca [UNCERT. (lapse reset) end 284] | 26,427.21 | uncertified form | 27,900.27 | 0.95× | within the uncertified bar | 34,120.26 | 1.22× | determinate | W154b | 26,361.55 | 0.94× | ref:5ca4f86c: 24c5ccb6/59757776; l_0ee93aca: 86d27f42/3579af95 | salvage convention changes the verdict: gross within the uncertified bar, net determinate |
| `L:y2030__n5_p0.25_e1__n7_p1_e3` | ref:5ca4f86c [UNCERT. (gap clause) end 281] | ref:e28de4ac [UNCERT. (gap clause) end 261] | -6,107.64 | uncertified form | 27,900.38 | 0.22× | within the uncertified bar | -5,407.94 | 0.19× | within the uncertified bar | W154b | -6,173.34 | 0.22× | ref:5ca4f86c: 24c5ccb6/59757776; ref:e28de4ac: 1fe91e86/4032a138 |  |
| `L:y2030__n7_p0.75_e3` | ref:5ca4f86c [UNCERT. (gap clause) end 281] | l_8e4c220e [cert. monotone k*233 r/τ 0.171] | 61,485.98 | uncertified form | 27,703.27 | 2.22× | determinate | 69,233.46 | 2.50× | determinate | W154b | 70,642.80 | 2.55× | ref:5ca4f86c: 24c5ccb6/59757776; l_8e4c220e: 5775a50c/3b77a5f6 |  |
| `L:y2030__n7_p0.75_e3__n9_p0.25_e0.5` | ref:5ca4f86c [UNCERT. (gap clause) end 281] | l_7db09f6c [cert. monotone k*251 r/τ 0.092] | 35,335.93 | uncertified form | 27,703.27 | 1.28× | determinate | 38,450.14 | 1.39× | determinate | W154b | 44,506.90 | 1.61× | ref:5ca4f86c: 24c5ccb6/59757776; l_7db09f6c: 7f5a857b/7126547f |  |
| `L:y2030__n7_p1_e3` | ref:5ca4f86c [UNCERT. (gap clause) end 281] | l_76c78064 [cert. monotone k*229 r/τ 0.012] | 50,616.86 | uncertified form | 27,703.27 | 1.83× | determinate | 61,140.88 | 2.21× | determinate | W154b | 59,782.58 | 2.16× | ref:5ca4f86c: 24c5ccb6/59757776; l_76c78064: 571ed31e/2feadb35 |  |
| `L:y2030__n7_p1_e3.5` | ref:5ca4f86c [UNCERT. (gap clause) end 281] | l_e1da0984 [cert. monotone k*235 r/τ 0.037] | 23,868.87 | uncertified form | 27,703.27 | 0.86× | within the uncertified bar | 26,794.78 | 0.97× | within the uncertified bar | W154b | 33,062.27 | 1.19× | ref:5ca4f86c: 24c5ccb6/59757776; l_e1da0984: 2587ab49/407fdee0 |  |
| `L:y2030__n7_p1_e3.5__n9_p0.25_e0.5` | ref:5ca4f86c [UNCERT. (gap clause) end 281] | l_7c455554 [UNCERT. (gap clause) end 282] | 5,576.82 | uncertified form | 27,743.28 | 0.20× | within the uncertified bar | 5,751.85 | 0.21× | within the uncertified bar | W154b | 5,563.48 | 0.20× | ref:5ca4f86c: 24c5ccb6/59757776; l_7c455554: 9906389c/e9e995e1 |  |
| `L:y2030__n7_p1_e3__n9_p0.25_e0.5` | ref:5ca4f86c [UNCERT. (gap clause) end 281] | l_45aa25a6 [UNCERT. (gap clause) end 291] | 31,329.47 | uncertified form | 27,703.27 | 1.13× | determinate | 39,165.52 | 1.41× | determinate | W154b | 31,366.28 | 1.13× | ref:5ca4f86c: 24c5ccb6/59757776; l_45aa25a6: 296e09ff/a33409d1 |  |
| `L:y2030__n7_p1_e4` | ref:5ca4f86c [UNCERT. (gap clause) end 281] | l_7b199ef9 [cert. oscillatory k*188 r/τ 0.312] | 6,384.11 | uncertified form | 27,703.27 | 0.23× | within the uncertified bar | 1,611.81 | 0.06× | within the uncertified bar | W154b | 15,885.36 | 0.57× | ref:5ca4f86c: 24c5ccb6/59757776; l_7b199ef9: 331e76c7/0025f345 |  |
| `E:C3_unit_value_minus_I` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | e_c3_unit [cert. oscillatory k*160 r/τ 0.943] | -73,746.66 | max(3 x larger bar, 2 tau) | 12,840.02 | 5.74× | determinate | -73,746.66 | 5.74× | determinate | W154b | -73,826.86 | 5.75× | ref:7aa017f0: d110bd1a/8435c718; e_c3_unit: 409740fb/db77e154 |  |
| `E:n7_4h_e1_C2:value_minus_I` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | e_c2 [cert. oscillatory k*172 r/τ 0.903] | -31,607.24 | max(3 x larger bar, 2 tau) | 12,627.51 | 2.50× | determinate | -31,607.24 | 2.50× | determinate | W154b | -31,894.34 | 2.53× | ref:7aa017f0: d110bd1a/8435c718; e_c2: ba969ad2/db77e154 |  |
| `E:n7_4h_e1_C2:vs_C3` | e_c3_unit [cert. oscillatory k*160 r/τ 0.943] | e_c2 [cert. oscillatory k*172 r/τ 0.903] | 42,139.42 | max(3 x larger bar, 2 tau) | 12,840.02 | 3.28× | determinate | 42,139.42 | 3.28× | determinate | W154b | 41,932.52 | 3.27× | e_c3_unit: 409740fb/db77e154; e_c2: ba969ad2/db77e154 |  |
| `E:n7_4h_e1_C2_calfade:value_minus_I` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | e_c2_calfade [cert. oscillatory k*172 r/τ 0.959 [≥0.95τ]] | -64,417.39 | max(3 x larger bar, 2 tau) | 13,054.22 | 4.93× | determinate | -64,417.39 | 4.93× | determinate | W154b | -64,937.86 | 4.97× | ref:7aa017f0: d110bd1a/8435c718; e_c2_calfade: bbd82994/db77e154 |  |
| `E:n7_4h_e1_C2_calfade:vs_C3` | e_c3_unit [cert. oscillatory k*160 r/τ 0.943] | e_c2_calfade [cert. oscillatory k*172 r/τ 0.959 [≥0.95τ]] | 9,329.27 | max(3 x larger bar, 2 tau) | 13,054.22 | 0.71× | within resolution | 9,329.27 | 0.71× | within resolution | W154b | 8,889.01 | 0.68× | e_c3_unit: 409740fb/db77e154; e_c2_calfade: bbd82994/db77e154 | A64 ruling 5: reported WITHIN RESOLUTION (the determinacy floor changed this verdict from the superseded rule); S1 restated "within resolution of C3 at a 0.70 floor"; the A61 prediction "the floor changes no verdict" failed on this claim (recorded, not re-litigated) |
| `E:n7_4h_e1_C3_midblock:value_minus_I` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | e_c3_midblock [cert. oscillatory k*174 r/τ 0.945] | -65,891.88 | max(3 x larger bar, 2 tau) | 12,867.48 | 5.12× | determinate | -65,891.88 | 5.12× | determinate | W154b | -66,058.24 | 5.13× | ref:7aa017f0: d110bd1a/8435c718; e_c3_midblock: 8b1eace9/db77e154 |  |
| `E:n7_4h_e1_C3_midblock:vs_C3` | e_c3_unit [cert. oscillatory k*160 r/τ 0.943] | e_c3_midblock [cert. oscillatory k*174 r/τ 0.945] | 7,854.78 | max(3 x larger bar, 2 tau) | 12,867.48 | 0.61× | within resolution | 7,854.78 | 0.61× | within resolution | W154b | 7,768.63 | 0.60× | e_c3_unit: 409740fb/db77e154; e_c3_midblock: 8b1eace9/db77e154 |  |
| `E:n7_4h_e1_C4:value_minus_I` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | e_c4 [cert. oscillatory k*173 r/τ 0.838] | -45,196.70 | max(3 x larger bar, 2 tau) | 12,627.51 | 3.58× | determinate | -45,196.70 | 3.58× | determinate | W154b | -45,721.44 | 3.62× | ref:7aa017f0: d110bd1a/8435c718; e_c4: c59d7006/db77e154 |  |
| `E:n7_4h_e1_C4:vs_C3` | e_c3_unit [cert. oscillatory k*160 r/τ 0.943] | e_c4 [cert. oscillatory k*173 r/τ 0.838] | 28,549.96 | max(3 x larger bar, 2 tau) | 12,840.02 | 2.22× | determinate | 28,549.96 | 2.22× | determinate | W154b | 28,105.42 | 2.19× | e_c3_unit: 409740fb/db77e154; e_c4: c59d7006/db77e154 |  |
| `E:n7_4h_e1_no_ageing:value_minus_I` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | e_no_ageing [cert. oscillatory k*177 r/τ 0.944] | -4,140.54 | max(3 x larger bar, 2 tau) | 12,858.70 | 0.32× | within resolution | -4,140.54 | 0.32× | within resolution | W154b | -4,905.42 | 0.38× | ref:7aa017f0: d110bd1a/8435c718; e_no_ageing: 1b10e9e1/db77e154 | A64 verbatim sentence: without ageing the unit is at break-even (within resolution) |
| `E:n7_4h_e1_no_ageing:vs_C3` | e_c3_unit [cert. oscillatory k*160 r/τ 0.943] | e_no_ageing [cert. oscillatory k*177 r/τ 0.944] | 69,606.12 | max(3 x larger bar, 2 tau) | 12,858.70 | 5.41× | determinate | 69,606.12 | 5.41× | determinate | W154b | 68,921.45 | 5.36× | e_c3_unit: 409740fb/db77e154; e_no_ageing: 1b10e9e1/db77e154 |  |
| `CHECK:headline_V_minus_I_settled` | ref:7aa017f0 [cert. k*181 r/τ 0.927] | ref:bd504ecf [cert. k*172 r/τ 0.959 [≥0.95τ]] | -64,417.39 | max(3 x larger bar, 2 tau) | 13,054.22 | 4.93× | determinate | -64,417.39 | 4.93× | determinate | W154b | -64,937.86 | 4.97× | ref:7aa017f0: d110bd1a/8435c718; ref:bd504ecf: 3f084f2f/db77e154 |  |

Status legend: `cert. <branch> k*N r/τ x` certified at N with range/τ x; `[≥0.95τ]` range ≥ 0.95 τ; `[NCTP c]` a turning point on non-clean cycle c (certificate kept, flagged: Addendum 64 ruling 4); `UNCERT. (cause) end N`.

## T2 — cells (certification status; instance keys)

| cell | item | status | branch | k0 | k* / end | range/τ | ≥0.95τ | NCTP | cause | gap | slack | band | replay bitwise through | certifying spec | eval key | candidate key |
|---|---|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---|---|---|
| b_0dd237f0 | B | certified | oscillatory | 103 | 173 | 0.921 |  | — |  | — | — | 4,180.76 | 103 | v4 | 38d09af2c857755f | a361ac93 |
| b_2a0ba8b2 | B | certified | oscillatory | 104 | 173 | 0.970 | yes | — |  | — | — | 4,404.70 | 104 | v4 | 33447912dab48fff | 18216659 |
| b_4649234b | B | certified | oscillatory | 84 | 148 | 0.989 | yes |  |  | — | — | 4,490.62 | 84 | v5 | 4de68ced93863cec | ace7403b |
| c_156ce2d1 | C | certified | oscillatory | 105 | 176 | 0.858 |  |  |  | — | — | 3,894.25 | 105 | v6 | 3a4c10c859114e20 | ba670a65 |
| c_6597a79d | C | certified | oscillatory | 131 | 193 | 0.984 | yes |  |  | — | — | 4,468.61 | 131 | v6 | 3a0d03e88a16f174 | a5bb68bf |
| d_3632b0ae | D | certified | oscillatory | 86 | 153 | 0.997 | yes |  |  | — | — | 4,523.72 | 86 | v6 | 1fe57699ce791ef6 | 11725e45 |
| d_36686489 | D | uncertified | — | 82 | 191 | — |  | 113,164,176 | growth test | 1,082.23 | 7,973.22 | 6,076.28 | 82 | — | 4de61708589e6e26 | cab98853 |
| d_4a82a64a | D | certified | oscillatory | 88 | 142 | 0.722 |  | 108 |  | — | — | 3,277.30 | 88 | v6 | 14b00a04ffbcfd33 | fb21d822 |
| d_9246ed01 | D | certified | oscillatory | 83 | 138 | 0.985 | yes |  |  | — | — | 4,473.18 | 83 | v6 | 579a1370d1e916a7 | 354b3881 |
| d_a12d95a2 | D | uncertified | — | 94 | 203 | — |  |  | lapse reset | 462.90 | 8,078.62 | 5,316.47 | 94 | — | 9bc66d9ed4863f39 | 423e5a7f |
| d_c52e1670 | D | certified | oscillatory | 89 | 150 | 0.982 | yes |  |  | — | — | 4,459.58 | — | v6 (v6 from records) | af42a163b14f0895 | cc4eb5df |
| d_c7fee8be | D | certified | oscillatory | 83 | 145 | 0.984 | yes |  |  | — | — | 4,465.54 | 83 | v6 | aa0b99f57c75682a | 7048a00e |
| d_d3709599 | D | certified | oscillatory | 104 | 175 | 0.863 |  |  |  | — | — | 3,915.50 | 104 | v6 | 64408baeee0990b9 | 3993aebc |
| d_f759dd48 | D | uncertified | — | 89 | 198 | — |  | 110,141,170,179 | growth test | 766.45 | 13,793.42 | 6,802.54 | 89 | — | 5f1b1aa621d3450a | 2cd1de82 |
| e_c2 | E | certified | oscillatory | 111 | 172 | 0.903 |  |  |  | — | — | 4,099.77 | — | ext v3 (rule v6) | ba969ad21c1831ab | db77e154 |
| e_c2_calfade | E | certified | oscillatory | 103 | 172 | 0.959 | yes |  |  | — | — | 4,351.41 | — | ext v3 (rule v6) | bbd82994a7ddff94 | db77e154 |
| e_c3_midblock | E | certified | oscillatory | 104 | 174 | 0.945 |  |  |  | — | — | 4,289.16 | — | ext v3 (rule v6) | 8b1eace91df86cd5 | db77e154 |
| e_c3_unit | E | certified | oscillatory | 104 | 160 | 0.943 |  |  |  | — | — | 4,280.01 | — | ext v3 (rule v6) | 409740fb51f27cab | db77e154 |
| e_c4 | E | certified | oscillatory | 110 | 173 | 0.838 |  |  |  | — | — | 3,801.50 | — | ext v3 (rule v6) | c59d7006b4b9c0a2 | db77e154 |
| e_no_ageing | E | certified | oscillatory | 107 | 177 | 0.944 |  |  |  | — | — | 4,286.23 | — | ext v3 (rule v6) | 1b10e9e161523f46 | db77e154 |
| g_37b5c499 | G | certified | oscillatory | 114 | 185 | 0.941 |  |  |  | — | — | 4,272.94 | — | v6 | bf2d8348d3a4b730 | e7c94efa |
| g_47dce43c | G | certified | oscillatory | 112 | 174 | 0.840 |  |  |  | — | — | 3,810.79 | — | v6 | 46ff23a160a27b67 | 803d1a6c |
| g_48749148 | G | certified | oscillatory | 101 | 170 | 0.926 |  |  |  | — | — | 4,202.37 | — | v6 | 22331aa714b8af80 | 76baf11e |
| g_9abf31d4 | G | certified | oscillatory | 103 | 172 | 0.845 |  |  |  | — | — | 3,837.72 | — | v6 | 5560d0f015a8d196 | ca1b7bac |
| h_50dea31c | H | certified | oscillatory | 125 | 146 | 0.484 |  |  |  | — | — | 2,194.83 | 125 | v6 | 354b1bdae2b86bce | 8435c718 |
| h_74eda68d | H | certified | oscillatory | 123 | 150 | 0.901 |  |  |  | — | — | 4,090.73 | 123 | v6 | dca0862812873604 | db77e154 |
| h_aa8a76d7 | H | certified | oscillatory | 93 | 153 | 0.910 |  |  |  | — | — | 4,130.84 | 93 | v6 | 22dd5ef8a9223f56 | 8435c718 |
| h_f9eae48f | H | uncertified | — | 85 | 194 | — |  |  | gap clause | 2,481.75 | 6,250.36 | 1,011.37 | 85 | — | aed2d618a19ca483 | db77e154 |
| i_5a6a88b4 | I | certified | monotone | 135 | 198 | 0.939 |  |  |  | — | — | 4,261.76 | 135 | v6 | e046a8cfd25f819c | cc4eb5df |
| j_5f3cccb4 | J | uncertified | — | 159 | 270 | — |  | 249 | gap clause | 8,847.90 | 6,190.52 | 2,763.22 | 159 | — | cc7f005909db6741 | cab98853 |
| j_a11d7966 | J | certified | oscillatory | 127 | 158 | 0.715 |  | 138 |  | — | — | 3,244.91 | 127 | v6 | 87efbc3d23566c29 | 11725e45 |
| j_f3aa335e | J | uncertified | — | 151 | 260 | — |  | 178 | growth test | 251.55 | 9,231.49 | 4,578.21 | 151 | — | af066d6bc9fdb74d | fb21d822 |
| l_0ee93aca | L | uncertified | — | 175 | 284 | — |  |  | lapse reset | 9,300.09 | 1,472.41 | 249.92 | 175 | — | 86d27f42e98503ca | 3579af95 |
| l_195156fa | L | certified | monotone | 185 | 252 | 0.098 |  |  |  | — | — | 446.08 | 185 | v6 | 1fd23b8db2be1ba0 | a033945d |
| l_2ab0ce2d | L | certified | monotone | 328 | 400 | 0.092 |  |  |  | — | — | 418.45 | 328 | v6 | 75266862b990d2eb | 9f4f66b6 |
| l_45aa25a6 | L | uncertified | — | 182 | 291 | — |  |  | gap clause | 9,197.61 | 2,191.62 | 691.70 | 182 | — | 296e09ff97ac2db4 | a33409d1 |
| l_76c78064 | L | certified | monotone | 153 | 229 | 0.012 |  |  |  | — | — | 52.78 | 153 | v6 | 571ed31e4139a567 | 2feadb35 |
| l_7b199ef9 | L | certified | oscillatory | 166 | 188 | 0.312 |  |  |  | — | — | 1,418.39 | 166 | v6 | 331e76c7c613b64e | 0025f345 |
| l_7c455554 | L | uncertified | — | 173 | 282 | — |  |  | gap clause | 9,247.76 | 1,468.81 | 602.28 | 173 | — | 9906389c1cd682a9 | e9e995e1 |
| l_7db09f6c | L | certified | monotone | 183 | 251 | 0.092 |  |  |  | — | — | 416.45 | 183 | v6 | 7f5a857b06a6083a | 7126547f |
| l_8e4c220e | L | certified | monotone | 171 | 233 | 0.171 |  |  |  | — | — | 777.41 | 171 | v6 | 5775a50cf6c84503 | 3b77a5f6 |
| l_b2251bc5 | L | uncertified | — | 211 | 320 | — |  |  | gap clause | 9,401.86 | 2,335.01 | 731.33 | 211 | — | abf4ac0af85dcb5e | c29e38c3 |
| l_df1a5525 | L | certified | monotone | 168 | 236 | 0.138 |  |  |  | — | — | 626.49 | 168 | v6 | 6c8f6353e29b204f | 7b85ff44 |
| l_e1da0984 | L | certified | monotone | 156 | 235 | 0.037 |  |  |  | — | — | 167.87 | 156 | v6 | 2587ab4987bc3125 | 407fdee0 |
| pb_y2025_n5_v6 | C | certified | oscillatory | 110 | 199 | 0.158 |  |  |  | — | — | 716.82 | 110 | ext v3 (rule v6) | be95e57603437a06 | 33031c08 |
| ref:5ca4f86c | reference | uncertified | — | — | 281 | — |  | — | gap clause | 9,234.42 | 1,461.49 | 398.19 | 172 | — | 24c5ccb6f285219f | 59757776 |
| ref:7aa017f0 | reference | certified | — | — | 181 | 0.927 |  | — |  | — | — | 4,209.17 | — | — | d110bd1a5977df1e | 8435c718 |
| ref:bd504ecf | reference | certified | — | — | 172 | 0.959 | yes | — |  | — | — | 4,351.41 | — | — | 3f084f2ffaeef2b7 | db77e154 |
| ref:e28de4ac | reference | uncertified | — | — | 261 | — |  | — | gap clause | 9,300.13 | 997.84 | 1,219.02 | 152 | — | 1fe91e86f11e76af | 4032a138 |

- `j_a11d7966` uncertified form beside (Addendum 64 ruling 4): gap 641.61, slack 5,293.19, bar 15,879.58.
- `d_4a82a64a` uncertified form beside (Addendum 64 ruling 4): gap 1,000.44, slack 23,581.31, bar 70,743.93.

## T3 — break-even fit, node 7 (slope b + c/4; Addendum 64 rulings 3–4)

Convention: Q = gross_operational_cost, settlement EXCLUDED (the frozen primary, Addendum 58 Ruling 3); Q_net = net_operational_recourse = Q - terminal salvage credit (beside); Q_cc = Q + t_sum (first-order consensus-consistent diagnostic, REPORT-ONLY); value = Q(0) - Q(x); F = Q + I; EUR; e* = b + c/4 - p_cost/4 (EUR/MWh). Energy cost e_cost = 253,877.68 €/MWh; p_cost = 256,317.32 €/MVA.

| fit | n cert. | intervals | certified-only e* | banded mid e* | e* range | margin range | b + c/4 range | b + c/4 rel. mid | corners |
|---|---:|---|---:|---:|---|---|---|---:|---|
| committed W145 (3 intervals) | 7 | n7_2h_e2, n7_2h_e3, n7_4h_e5 | 182,701.62 | 181,933.27 | 173,486.04 – 190,380.50 | 63,497.18 – 80,391.63 | 237,565.38 – 254,459.83 | 0.31 % | 3.7 / 3.1 % |
| **conservative A64 (4 intervals, + d_4a82a64a)** | 6 | n7_2h_e2, n7_2h_e3, n7_4h_e3, n7_4h_e5 | 183,083.50 | 181,933.27 | 171,309.08 – 192,557.46 | 61,320.21 – 82,568.60 | 235,388.41 – 256,636.79 | 0.47 % | 4.8 / 3.8 % |

**Manuscript figure (conservative):** break-even ≤ 192,557 €/MWh; margin to energy cost ≥ 61,320 €/MWh. Hand cross-check (labelled, not the table figure): 61,320 €/MWh. The committed W145 result is reproduced bitwise from its points: True.

Interval of d_4a82a64a: {"Q_cap": 653128121.86, "gap": 1000.44, "slack": 23581.31, "bar": 70743.93, "half_width": 79822.07}.

## T4 — year ladder (2035 − 2030), gross primary, net beside

Convention: Q = gross_operational_cost, settlement EXCLUDED (the frozen primary, Addendum 58 Ruling 3); Q_net = net_operational_recourse = Q - terminal salvage credit (beside); Q_cc = Q + t_sum (first-order consensus-consistent diagnostic, REPORT-ONLY); value = Q(0) - Q(x); F = Q + I; EUR; M = I + Q - Q181 (W118 form); net = gross - salvage.

| | M gross | salvage | M net | eval key | k* |
|---|---:|---:|---:|---|---:|
| 2030 | 68,498.92 | 11,759.26 | 56,739.66 | 6e4f11a95dc3d585 | 169 |
| 2035 | 110,957.57 | 56,504.17 | 54,453.40 | 3a2387f46448f9db | 167 |
| **2035 − 2030** | **42,458.65** — determinate 3.46× (thr 12,254.24; W118 rule: determinate) | | **-2,286.25** — within resolution 0.19× (thr 12,254.24) | | |

Net label: validated by form + salvage identity (W154b); the table carries -2,286.25 (the prose -2,286.2 was rounded-component arithmetic). Sentence (A58 Ruling 3): the investment-year comparison in this instance is decided by the salvage convention, not by operation.

## T5 — Phase B certificates against x = 0 (gross; v6 rule)

| cell | M gross | M Q_cc (r-o) | v6 threshold | × | v6 verdict | recorded (W118 rule) | k* | range/τ | note |
|---|---:|---:|---:|---:|---|---|---:|---:|---|
| pb_y2030_n9 | 46,705.97 | 47,016.78 | 12,826.33 | 3.64× | determinate | determinate | 192 | 0.942 | v6 rule applied here by DET.resolve_v6 (zero-solve); net not in the W154b validation set |
| pb_y2030_n7 | 49,021.70 | 49,602.38 | 13,536.05 | 3.62× | determinate | determinate | 195 | 0.994 | v6 rule applied here by DET.resolve_v6 (zero-solve); net not in the W154b validation set |
| pb_y2025_n5 | 51,262.81 | 50,533.68 | 13,525.89 | 3.79× | determinate | determinate | 167 | 0.993 | superseded by the v6 re-run pb_y2025_n5_v6 (claim C:y2025__n5_p0.25_e0.5): the W118 certificate sat on a non-Optimal cycle (Addendum 59) |
| pb_y2030_n5 | 43,325.65 | 43,427.76 | 12,748.14 | 3.40× | determinate | determinate | 182 | 0.936 | v6 rule applied here by DET.resolve_v6 (zero-solve); net not in the W154b validation set |
| pb_y2025_n9 | 52,460.26 | 52,749.21 | 12,627.51 | 4.15× | determinate | determinate | 174 | 0.814 | v6 rule applied here by DET.resolve_v6 (zero-solve); net not in the W154b validation set |
| pb_y2025_n7 | 54,940.74 | 55,021.53 | 13,265.22 | 4.14× | determinate | determinate | 172 | 0.974 | v6 rule applied here by DET.resolve_v6 (zero-solve); net not in the W154b validation set |
| pb_y2025_n5_v6 | 49,692.73 | 49,798.49 | 12,627.51 | 3.94× | determinate | — | — | — | the extension v6 re-run (claim C:y2025__n5_p0.25_e0.5); T1 carries its full row |

## T6 — uncoordinated benchmark (Addendum 57; spec v5 `bca69f97`)

Convention: Q = gross_operational_cost, settlement EXCLUDED (production _get_operational_recourse_components); every block priced for the evaluation with production's ADMM-…

| arrangement | Q (gross) | band |
|---|---:|---:|
| coordinated (settled x = 0) | 653,873,702.19 | 71,926.11 |
| passive NRF (best of 3 starts: perturbed) | 895,049,918.47 | 25,804.990 |
| price_taker NRF (best of 3 starts: warm_from_certified) | 744,770,310.59 | 0.004 |

Claim: **+90,896,608.40 € (13.9 %)**, 1263.75× the larger band 71,926.11 — determinate True. Reverse-flow interface-hours in the coordinated solution: 4 (3,308.30 MWh, block-weighted). Sweep (no interface rule): sweep_passive_cold: 1/12 blocks, 6 h; sweep_price_taker_cold: 8/12 blocks, 25 h.

## T7 — discount rate (fixed plan; unit n7 0.25 MVA / 1 MWh)

| rate | V | I | value − I | A61 threshold (cons.) | × | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 0 % | 277,119.69 | 317,957.01 | -40,837.32 | 15,913.02 | 2.57× | determinate |
| 2 % | 253,539.62 | 317,957.01 | -64,417.39 | 13,054.22 | 4.93× | determinate |
| 5 % | 225,232.11 | 317,957.01 | -92,724.89 | 13,054.22 | 7.10× | determinate |
| 8 % | 203,366.63 | 317,957.01 | -114,590.38 | 13,054.22 | 8.78× | determinate |

## T8 — ageing arms at minimum SoH 0.70 (gross)

| arm | cell | value | value − I | verdict | floor binds | ε_AE (0.70) | resolvable | ε_AE (0.50, superseded) |
|---|---|---:|---:|---|---|---:|---|---:|
| C3_unit | e_c3_unit | 244,210.35 | -73,746.66 | determinate | 2035 | — | None | — |
| C2 | e_c2 | 286,349.77 | -31,607.24 | determinate | never | 1.30 | True | 0.60 |
| C4 | e_c4 | 272,760.31 | -45,196.70 | determinate | never | 1.85 | True | 0.41 |
| C2_calfade | e_c2_calfade | 253,539.62 | -64,417.39 | determinate | 2035 | 4.38 | False | 0.68 |
| C3_midblock | e_c3_midblock | 252,065.13 | -65,891.88 | determinate | 2035 | 0.52 | False | 0.13 |
| no_ageing | e_no_ageing | 313,816.47 | -4,140.54 | within resolution | never | 1.04 | True | 0.62 |

ε_AE band over resolvable arms: 1.04–1.85. scored "consistent in direction, not resolved at this precision"; S5 restated; the late-life-tail reading stays a hypothesis.

## T9 — dual dead zone (Addendum 58 Ruling 1; Addendum 62; Addendum 64 ruling 7)

| cell | status | cause | t_sum at cap | share n5 / n7 / n9 | pf_primal last | entry |
|---|---|---|---:|---|---:|---|
| h_f9eae48f | uncertified | gap clause | -2,481.75 | 55 % / 15 % / 31 % | 0.0267 | gap-refused (the dead-zone label) |
| j_5f3cccb4 | uncertified | gap clause | -8,847.90 | 55 % / 14 % / 31 % | 0.6819 | gap-refused (the dead-zone label) |
| l_45aa25a6 | uncertified | gap clause | -9,197.61 | 55 % / 14 % / 31 % | 0.6904 | gap-refused (the dead-zone label) |
| l_7c455554 | uncertified | gap clause | -9,247.76 | 55 % / 14 % / 31 % | 0.6904 | gap-refused (the dead-zone label) |
| l_b2251bc5 | uncertified | gap clause | -9,401.86 | 55 % / 14 % / 31 % | 0.6999 | gap-refused (the dead-zone label) |
| l_0ee93aca | uncertified | lapse reset | -9,300.09 | 55 % / 14 % / 31 % | 0.6917 | by its signature, cause stated: two TSO recoveries (lapse resets at 218, 222) reset the rule before the gap clause was reached (Addendum 64 ruling 7a) |
| d_36686489 | uncertified | growth test | 1,082.23 | 58 % / 11 % / 30 % | 0.0117 | baseline-price degenerate case with large storage (Addendum 62): uncertified by the growth test after repeated TSO recoveries, not by the gap clause |
| ref:5ca4f86c | uncertified | gap clause | -9,234.42 | — | — | F2 incumbent (W118 r2 f2_incumbent, uncertified at 281, gap clause) |
| ref:e28de4ac | uncertified | gap clause | -9,300.13 | — | — | F2 challenger (W118 r2 f2_challenger, uncertified at 261, gap clause) |

W149 rows (h cells at m = 1.5 and the F2 pair); the TSO-lever column is **report-only, added post hoc**:

| cell | m | storage | TN at bound | DN flex at kink | TSO cheapest lever = shared ESS (post hoc, r-o) | t_sum | verdict |
|---|---:|---|---|---|---|---:|---|
| h_f9eae48f | 1.5 | n7 0.25 MVA / 1.0 MWh (2025) | y | partial (1/24) | y | -2,481.75 | partially reproduced |
| h_aa8a76d7 | 1.5 | none (x = 0) | y | partial (1/24) | n | -551.88 | control: partially reproduced on the subject's entries |
| f2_challenger | 2.0 | n5 0.25/1.0, n7 1.0/3.0 (2030) | y | y | y | -9,300.13 | dual dead zone (W127 verdict, Addendum 57 report) |
| f2_incumbent | 2.0 | n5 0.25/0.5, n7 1.0/3.5 (2030) | not recorded (W127 loaded the challenger only) | not recorded (W127 loaded the challenger only) | not recorded (W127 loaded the challenger only) | -9,234.42 | reported with the challenger as the F2 dead zone (Addendum 57 report); no model read |

## T10 — Addendum 64 rows (**PENDING W156**)

| claim | ref | other | statement | d gross | threshold / bar | × | verdict | status |
|---|---|---|---|---:|---:|---:|---|---|
| `H:m1.75:value_minus_I` | h_x0_m175 | h_unit_m175 | flexibility ladder m = 1.75: sign of value - I (node 7, 0.25 MVA / 1 MWh), value = Q(0)_m1.75 - Q(unit)_m1.75 | — | — | — | PENDING W156 | **PENDING W156** |
| `E:soh050:delta_value_vs_070` | bd504ecf | e_soh050 | minimum SoH row: Delta value = value(0.50) - value(0.70) = Q(3f084f2f, 0.70, k* 172) - Q(e_soh050); x = 0 is unaffected by soh_min (Q(0) = the settled x0 Q181 d110bd1a on both sides) | — | — | — | PENDING W156 | **PENDING W156** |
| `E:soh050:value_minus_I` | 7aa017f0 | e_soh050 | the C2_calfade unit at minimum_soh 0.50: sign of value - I | — | — | — | PENDING W156 | **PENDING W156** |
| `E:soh050:delta_value_vs_g070_REPORT_BESIDE` | g070_neutrality | e_soh050 | Delta value against the neutrality cell (identical to 3f084f2f through 172 by its gate) | — | — | — | PENDING W156 | **PENDING W156** |

Recorded predictions (frozen spec `44a2dce8`, before any run): A (m = 1.75) value − I positive, point +15 k€, [+8, +22] k€; B (soh_min 0.50) Δvalue positive, point +12 k€, [+4, +22] k€, more likely within resolution (threshold 13,054). **Scored: PENDING W156.**

## Checks

- claims_60: True
- net_labels_8_recorded_G_rows_52_validated: True
- cells_49_distinct: True
- every_claim_cell_in_cells_table: True
- W145_committed_result_reproduced_bitwise: True
- conservative_bar_d_4a82a64a_70.7k: True
- conservative_margin_min_61.3k: True
- J4_uncertified_bar_15880_and_11.1x: True
- E_C2_calfade_vs_C3_within_resolution: True
- year_ladder_net_-2286.25: True
- a64_rows_4: True
- a64_rows_pending_unless_included: True

