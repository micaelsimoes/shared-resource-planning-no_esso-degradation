# W169 — closure slack and ESSO P-net slack at the certified point, every T1–T11 evaluation

Script `p515_s53_w169_slack_inventory.py`; zero solves (guard verified 0), pickle blocked (0 calls). Frozen tables `data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json` (sha256 590088fe). Manuscript comments read at Overleaf 42794d4 (main.tex sha256 7effd898) l. 747 and l. 857.

## Definitions

- **closure slack**: s+ / s- = slack_shared_es_soc_final_up / _down (network block models; scenario-free copy)
- **closure level per block**: component_levels_terminal.json blocks[b].unweighted.shared_ess_day_balance_slack / PENALTY_SHARED_ESS_BALANCE = sum over the block's units of (s+ + s-) in MWh
- **closure relative**: (s+ + s-) / E^Av(unit, year); TSO blocks: per-unit bound (sum + (n_active - 1) x 2 x 1e-06 MWh) / smallest active E^Av; E^Av = published e_available
- **sigma**: sigma+ / sigma- = slack_es_pnet_up / _down (ESSO, per node and (y, d, t), MW)
- **sigma relative**: (sigma+ + sigma-) / S(e, y), S = the cohort converter rating s_max (MVA)
- **threshold**: 1e-06 (the prediction's "1e-6 relative")
- **signed reading**: max relative <= 1e-6 (the slack sum may be negative: IPOPT bound relaxation)
- **absolute reading**: max |relative| <= 1e-6
- **point**: certified: k*; uncertified: terminal (frozen end_cycle); T11: certification cycle; w98: continuation terminal; arms: arm evaluation

## Prediction (expert, Addendum 68) against the outcome

Prediction: both slacks are zero to solver tolerance (<= 1e-6 relative) at every certified point (expert, Addendum 68)

| quantity | reading | certified points | scored | source of the score | pass | fail | max relative (argmax cell) |
|---|---|---:|---:|---|---:|---:|---|
| closure (s+ + s-) / E^Av | signed | 51 | 45 | committed component_levels_terminal.json at P | 44 | 1 | 5.039e-06 (pb_y2025_n5) |
| closure (s+ + s-) / E^Av | absolute | 51 | 45 | committed component_levels_terminal.json at P | 10 | 35 | 5.714e-06 (c_156ce2d1) |
| sigma (sigma+ + sigma-) / S | signed | 51 | 46 | UNCOMMITTED esso_capture file at P (per element) | 46 | 0 | -8.000e-09 (d_9246ed01) |
| sigma (sigma+ + sigma-) / S | absolute | 51 | 46 | UNCOMMITTED esso_capture file at P (per element) | 46 | 0 | 8.000e-08 (c_6597a79d) |
| sigma (sigma+ + sigma-) / S | signed, committed bound | 51 | 45 | committed D3 sum at P -> per-element upper bound (relaxed-bound floor) / min active S | 45 | 0 | -7.937e-09 (d_9246ed01) |

Outcome: closure scored at 45 / 51 certified points (committed), sigma per element at 46 / 51 (uncommitted esso_capture), sigma by the committed sum bound at 45 / 51. SIGNED reading: FAILS at closure (s+ + s-) / E^Av: ['pb_y2025_n5 (superseded certificate)']. ABSOLUTE reading: closure (s+ + s-) / E^Av: 10 pass / 35 fail (max 5.714e-06 at c_156ce2d1); sigma (sigma+ + sigma-) / S: 46 pass / 0 fail (max 8.000e-08 at c_6597a79d). Certified points with no active unit (x = 0; relative closure undefined, absolute level reported): ['h_50dea31c', 'h_aa8a76d7', 'ref:7aa017f0', 'h_x0_m175', 't11_3x3_x0']; every block without an active unit is exactly 0.0 in every scored cell: True.

## Coverage — every evaluation

Closure: max over the blocks holding an active unit of s⁺+s⁻ (MWh; a TSO block = the sum over its units); for x = 0 cells (no active unit) the max over all blocks. Relative signed = per-unit bound / E^Av (max over blocks); relative abs = |s⁺+s⁻| / E^Av over the blocks with exactly one active unit. σ: max over (e, y, d, t) of σ⁺+σ⁻ (MW) and σ / S. C = committed record, U = uncommitted file (read-only), — = not carried at P. Every value: the cell's point (k* when certified).

| cell | table | point | eval key / candidate | closure src | max s⁺+s⁻ (MWh) | argmax block | rel. signed | rel. abs | σ src | max σ (MW) | argmax (e, y, d, t) | σ/S signed | σ/S abs | D3 sum (C) | detector max ratio at P |
|---|---|---|---|---|---:|---|---:|---:|---|---:|---|---:|---:|---:|---:|
| b_0dd237f0 | T2 | k* 173 | 38d09af2c857755f / a361ac93 | C | -2.000e-06 | TSO|2035|Winter | -2.272e-06 | 2.857e-06 | U | -2.000e-08 | (9, 2025, Spring, 23) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 0.000e+00 |
| b_2a0ba8b2 | T2 | k* 173 | 33447912dab48fff / 18216659 | C | -2.000e-06 | TSO|2035|Winter | -2.272e-06 | 2.857e-06 | U | -2.000e-08 | (5, 2025, Spring, 0) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 0.000e+00 |
| b_4649234b | T2 | k* 148 | 4de68ced93863cec / ace7403b | C | -2.000e-06 | TSO|2035|Winter | -7.568e-07 | 9.523e-07 | U | -2.000e-08 | (9, 2030, Spring, 7) | -2.667e-08 | 2.667e-08 | -1.728e-05 | 0.000e+00 |
| c_156ce2d1 | T2 | k* 176 | 3a4c10c859114e20 / ba670a65 | C | -2.000e-06 | DSO|9|2035|Spring | -4.538e-06 | 5.714e-06 | U | -2.000e-08 | (5, 2025, Spring, 6) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 0.000e+00 |
| c_6597a79d | T2 | k* 193 | 3a0d03e88a16f174 / a5bb68bf | C | -2.000e-06 | DSO|5|2035|Summer | -4.617e-06 | 5.284e-06 | U | -2.000e-08 | (7, 2030, Spring, 14) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 3.631e-05 |
| d_3632b0ae | T2 | k* 153 | 1fe57699ce791ef6 / 11725e45 | C | -2.000e-06 | TSO|2035|Winter | -5.676e-07 | 7.142e-07 | U | -2.000e-08 | (7, 2025, Spring, 0) | -2.000e-08 | 2.000e-08 | -1.728e-05 | 0.000e+00 |
| d_36686489 | T2 | terminal 191 | 4de61708589e6e26 / cab98853 | C | -2.000e-06 | TSO|2035|Winter | -4.540e-07 | 5.714e-07 | U | -2.000e-08 | (7, 2030, Spring, 19) | -1.600e-08 | 1.600e-08 | -1.728e-05 | 0.000e+00 |
| d_4a82a64a | T2 | k* 142 | 14b00a04ffbcfd33 / fb21d822 | C | -2.000e-06 | TSO|2035|Winter | -7.568e-07 | 9.523e-07 | U | -2.000e-08 | (7, 2025, Spring, 0) | -2.667e-08 | 2.667e-08 | -1.728e-05 | 0.000e+00 |
| d_9246ed01 | T2 | k* 138 | 579a1370d1e916a7 / 354b3881 | C | -2.000e-06 | TSO|2035|Winter | -4.545e-07 | 5.714e-07 | U | -2.000e-08 | (7, 2025, Spring, 0) | -8.000e-09 | 8.000e-09 | -1.728e-05 | 0.000e+00 |
| d_a12d95a2 | T2 | terminal 203 | 9bc66d9ed4863f39 / 423e5a7f | C | -2.000e-06 | TSO|2035|Winter | -1.136e-06 | 1.428e-06 | U | -2.000e-08 | (7, 2030, Spring, 21) | -2.000e-08 | 2.000e-08 | -1.728e-05 | 0.000e+00 |
| d_c52e1670 | T2 | k* 150 | af42a163b14f0895 / cc4eb5df | — | — | — | — | — | U | -2.000e-08 | (7, 2025, Spring, 0) | -4.000e-08 | 4.000e-08 | — | 0.000e+00 |
| d_c7fee8be | T2 | k* 145 | aa0b99f57c75682a / 7048a00e | C | -2.000e-06 | TSO|2035|Winter | -5.681e-07 | 7.142e-07 | U | -2.000e-08 | (7, 2030, Spring, 7) | -1.000e-08 | 1.000e-08 | -1.728e-05 | 0.000e+00 |
| d_d3709599 | T2 | k* 175 | 64408baeee0990b9 / 3993aebc | C | -2.000e-06 | TSO|2035|Winter | -2.271e-06 | 2.857e-06 | U | -2.000e-08 | (7, 2025, Spring, 0) | -4.000e-08 | 4.000e-08 | -1.728e-05 | 0.000e+00 |
| d_f759dd48 | T2 | terminal 198 | 5f1b1aa621d3450a / 2cd1de82 | C | -2.000e-06 | TSO|2035|Winter | -7.573e-07 | 9.523e-07 | U | -2.000e-08 | (7, 2030, Spring, 19) | -1.333e-08 | 1.333e-08 | -1.728e-05 | 0.000e+00 |
| e_c2 | T2 | k* 172 | ba969ad21c1831ab / db77e154 | C | -2.000e-06 | TSO|2035|Summer | -2.132e-06 | 2.394e-06 | U | -2.000e-08 | (7, 2025, Spring, 14) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 3.632e-05 |
| e_c2_calfade | T2 | k* 172 | bbd82994a7ddff94 / db77e154 | C | -2.000e-06 | TSO|2035|Winter | -2.271e-06 | 2.857e-06 | U | -2.000e-08 | (7, 2025, Spring, 23) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 0.000e+00 |
| e_c3_midblock | T2 | k* 174 | 8b1eace91df86cd5 / db77e154 | C | -2.000e-06 | TSO|2035|Winter | -2.142e-06 | 2.723e-06 | U | -2.000e-08 | (7, 2025, Spring, 23) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 0.000e+00 |
| e_c3_unit | T2 | k* 160 | 409740fb51f27cab / db77e154 | C | -2.000e-06 | TSO|2035|Winter | -2.296e-06 | 2.857e-06 | U | -2.000e-08 | (7, 2025, Spring, 23) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 0.000e+00 |
| e_c4 | T2 | k* 173 | c59d7006b4b9c0a2 / db77e154 | C | -2.000e-06 | TSO|2035|Summer | -2.209e-06 | 2.631e-06 | U | -2.000e-08 | (7, 2025, Spring, 0) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 3.632e-05 |
| e_no_ageing | T2 | k* 177 | 1b10e9e161523f46 / db77e154 | C | -2.000e-06 | TSO|2035|Summer | -2.000e-06 | 2.000e-06 | U | -2.000e-08 | (7, 2030, Spring, 14) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 3.632e-05 |
| g_37b5c499 | T2 | k* 185 | bf2d8348d3a4b730 / e7c94efa | C | -2.000e-06 | TSO|2035|Summer | -2.304e-06 | 2.624e-06 | U | -2.000e-08 | (7, 2030, Spring, 14) | -4.000e-08 | 4.000e-08 | -1.728e-05 | 1.816e-05 |
| g_47dce43c | T2 | k* 174 | 46ff23a160a27b67 / 803d1a6c | C | -2.000e-06 | TSO|2035|Summer | -2.295e-06 | 2.295e-06 | U | -2.000e-08 | (7, 2035, Spring, 0) | -4.000e-08 | 4.000e-08 | -1.728e-05 | 1.816e-05 |
| g_48749148 | T2 | k* 170 | 22331aa714b8af80 / 76baf11e | C | -2.000e-06 | TSO|2035|Summer | -1.147e-06 | 1.297e-06 | U | -2.000e-08 | (7, 2030, Spring, 3) | -4.000e-08 | 4.000e-08 | -1.728e-05 | 1.815e-05 |
| g_9abf31d4 | T2 | k* 172 | 5560d0f015a8d196 / ca1b7bac | C | -2.000e-06 | TSO|2035|Summer | -1.136e-06 | 1.136e-06 | U | -2.000e-08 | (7, 2035, Spring, 15) | -4.000e-08 | 4.000e-08 | -1.728e-05 | 1.816e-05 |
| h_50dea31c | T2 | k* 146 | 354b1bdae2b86bce / 8435c718 | C | 0.000e+00 | TSO|2025|Spring | — | — | U | — | — | — | — | -1.728e-05 | 0.000e+00 |
| h_74eda68d | T2 | k* 150 | dca0862812873604 / db77e154 | C | -2.000e-06 | TSO|2035|Spring | -2.247e-06 | 2.857e-06 | U | -2.000e-08 | (7, 2025, Spring, 6) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 0.000e+00 |
| h_aa8a76d7 | T2 | k* 153 | 22dd5ef8a9223f56 / 8435c718 | C | 0.000e+00 | TSO|2025|Spring | — | — | U | — | — | — | — | -1.728e-05 | 0.000e+00 |
| h_f9eae48f | T2 | terminal 194 | aed2d618a19ca483 / db77e154 | C | -2.000e-06 | TSO|2035|Winter | -2.254e-06 | 2.857e-06 | U | -2.000e-08 | (7, 2025, Spring, 23) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 0.000e+00 |
| i_5a6a88b4 | T2 | k* 198 | e046a8cfd25f819c / cc4eb5df | C | -2.000e-06 | TSO|2035|Spring | -1.124e-06 | 1.428e-06 | U | -2.000e-08 | (7, 2030, Spring, 20) | -4.000e-08 | 4.000e-08 | -1.728e-05 | 0.000e+00 |
| j_5f3cccb4 | T2 | terminal 270 | cc7f005909db6741 / cab98853 | C | -2.000e-06 | TSO|2035|Spring | -4.493e-07 | 5.714e-07 | U | -2.000e-08 | (7, 2025, Spring, 6) | -1.600e-08 | 1.600e-08 | -1.728e-05 | 0.000e+00 |
| j_a11d7966 | T2 | k* 158 | 87efbc3d23566c29 / 11725e45 | C | -2.000e-06 | TSO|2035|Spring | -5.626e-07 | 7.142e-07 | U | -2.000e-08 | (7, 2025, Spring, 5) | -2.000e-08 | 2.000e-08 | -1.728e-05 | 0.000e+00 |
| j_f3aa335e | T2 | terminal 260 | af066d6bc9fdb74d / fb21d822 | C | -2.000e-06 | TSO|2035|Spring | -7.502e-07 | 9.523e-07 | U | -2.000e-08 | (7, 2025, Spring, 5) | -2.667e-08 | 2.667e-08 | -1.728e-05 | 0.000e+00 |
| l_0ee93aca | T2 | terminal 284 | 86d27f42e98503ca / 3579af95 | C | -2.000e-06 | DSO|7|2030|Spring | -7.736e-07 | 5.420e-06 | U | -2.000e-08 | (5, 2030, Spring, 0) | -2.000e-08 | 8.000e-08 | -1.728e-05 | 3.631e-05 |
| l_195156fa | T2 | k* 252 | 1fd23b8db2be1ba0 / a033945d | C | -2.000e-06 | DSO|7|2035|Summer | -7.700e-07 | 5.416e-06 | U | -2.000e-08 | (5, 2030, Spring, 0) | -2.667e-08 | 8.000e-08 | -1.728e-05 | 3.631e-05 |
| l_2ab0ce2d | T2 | k* 400 | 75266862b990d2eb / 9f4f66b6 | C | -2.000e-06 | DSO|7|2030|Spring | -7.757e-07 | 5.413e-06 | U | -2.000e-08 | (5, 2030, Spring, 0) | -1.600e-08 | 8.000e-08 | -1.728e-05 | 3.632e-05 |
| l_45aa25a6 | T2 | terminal 291 | 296e09ff97ac2db4 / a33409d1 | C | -2.000e-06 | DSO|7|2030|Spring | -7.736e-07 | 5.436e-06 | U | -2.000e-08 | (7, 2030, Spring, 0) | -2.000e-08 | 8.000e-08 | -1.728e-05 | 3.630e-05 |
| l_76c78064 | T2 | k* 229 | 571ed31e4139a567 / 2feadb35 | C | -2.000e-06 | TSO|2035|Spring | -7.733e-07 | 8.826e-07 | U | -2.000e-08 | (7, 2030, Spring, 2) | -2.000e-08 | 2.000e-08 | -1.728e-05 | 9.081e-06 |
| l_7b199ef9 | T2 | k* 188 | 331e76c7c613b64e / 0025f345 | C | -2.000e-06 | TSO|2035|Spring | -5.770e-07 | 6.546e-07 | U | -2.000e-08 | (7, 2030, Spring, 0) | -2.000e-08 | 2.000e-08 | -1.728e-05 | 9.081e-06 |
| l_7c455554 | T2 | terminal 282 | 9906389c1cd682a9 / e9e995e1 | C | -2.000e-06 | DSO|7|2035|Summer | -6.612e-07 | 5.431e-06 | U | -2.000e-08 | (7, 2030, Spring, 0) | -2.000e-08 | 8.000e-08 | -1.728e-05 | 3.631e-05 |
| l_7db09f6c | T2 | k* 251 | 7f5a857b06a6083a / 7126547f | C | -2.000e-06 | DSO|7|2035|Summer | -7.695e-07 | 5.424e-06 | U | -2.000e-08 | (7, 2030, Spring, 1) | -2.667e-08 | 8.000e-08 | -1.728e-05 | 3.632e-05 |
| l_8e4c220e | T2 | k* 233 | 5775a50cf6c84503 / 3b77a5f6 | C | -2.000e-06 | TSO|2035|Spring | -7.706e-07 | 8.763e-07 | U | -2.000e-08 | (7, 2030, Spring, 14) | -2.667e-08 | 2.667e-08 | -1.728e-05 | 1.211e-05 |
| l_b2251bc5 | T2 | terminal 320 | abf4ac0af85dcb5e / c29e38c3 | C | -2.000e-06 | DSO|7|2035|Summer | -7.705e-07 | 5.432e-06 | U | -2.000e-08 | (5, 2030, Spring, 0) | -2.667e-08 | 8.000e-08 | -1.728e-05 | 3.632e-05 |
| l_df1a5525 | T2 | k* 236 | 6c8f6353e29b204f / 7b85ff44 | C | -2.000e-06 | DSO|7|2035|Summer | -9.265e-07 | 5.415e-06 | U | -2.000e-08 | (5, 2030, Spring, 0) | -2.667e-08 | 8.000e-08 | -1.728e-05 | 3.632e-05 |
| l_e1da0984 | T2 | k* 235 | 2587ab4987bc3125 / 407fdee0 | C | -2.000e-06 | TSO|2035|Spring | -6.610e-07 | 7.518e-07 | U | -2.000e-08 | (7, 2030, Spring, 2) | -2.000e-08 | 2.000e-08 | -1.728e-05 | 9.081e-06 |
| pb_y2025_n5_v6 | T2 | k* 199 | be95e57603437a06 / 33031c08 | C | -2.000e-06 | TSO|2035|Winter | -4.538e-06 | 5.714e-06 | U | -2.000e-08 | (5, 2025, Spring, 6) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 0.000e+00 |
| ref:5ca4f86c | T2 | terminal 281 | 24c5ccb6f285219f / 59757776 | C | -2.000e-06 | DSO|7|2035|Summer | -6.612e-07 | 5.417e-06 | U | -2.000e-08 | (5, 2030, Spring, 0) | -2.000e-08 | 8.000e-08 | -1.728e-05 | 3.632e-05 |
| ref:7aa017f0 | T2, T6 (coordinated row) | k* 181 | d110bd1a5977df1e / 8435c718 | C | 0.000e+00 | TSO|2025|Spring | — | — | U | — | — | — | — | -1.728e-05 | 0.000e+00 |
| ref:bd504ecf | T2 | k* 172 | 3f084f2ffaeef2b7 / db77e154 | C | -2.000e-06 | TSO|2035|Winter | -2.271e-06 | 2.857e-06 | U | -2.000e-08 | (7, 2025, Spring, 23) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 0.000e+00 |
| ref:e28de4ac | T2 | terminal 261 | 1fe91e86f11e76af / 4032a138 | C | -2.000e-06 | DSO|7|2030|Spring | -7.737e-07 | 2.631e-06 | U | -2.000e-08 | (5, 2030, Spring, 14) | -2.000e-08 | 8.000e-08 | -1.728e-05 | 3.631e-05 |
| e_soh050 | T10, T2 (appended by W160) | k* 184 | 3f01b9eaaab13c82 / db77e154 | C | -2.000e-06 | TSO|2035|Summer | -2.292e-06 | 2.953e-06 | U | -2.000e-08 | (7, 2025, Spring, 14) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 3.632e-05 |
| g070_neutrality | T10, T2 (appended by W160) | k* 172 | 5007232b1c3eaf03 / db77e154 | C | -2.000e-06 | TSO|2035|Winter | -2.271e-06 | 2.857e-06 | U | -2.000e-08 | (7, 2025, Spring, 23) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 0.000e+00 |
| h_unit_m175 | T10, T2 (appended by W160) | k* 121 | dc3ab9329c395890 / db77e154 | C | -2.000e-06 | TSO|2035|Spring | -2.252e-06 | 2.857e-06 | U | -2.000e-08 | (7, 2025, Spring, 6) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 0.000e+00 |
| h_x0_m175 | T10, T2 (appended by W160) | k* 139 | b9f9be3528a64a93 / 8435c718 | C | 0.000e+00 | TSO|2025|Spring | — | — | U | — | — | — | — | -1.728e-05 | 0.000e+00 |
| pb_y2025_n5 | T5, T2 (appended by W160) | k* 167 | ca29c5e818366a6e / 33031c08 | C | 1.764e-06 | TSO|2035|Spring | 5.039e-06 | 5.714e-06 | U | -2.000e-08 | (5, 2025, Spring, 6) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 0.000e+00 |
| pb_y2025_n7 | T5, T2 (appended by W160) | k* 172 | b9b8e4be4a6c82b6 / 56b9730b | C | -2.000e-06 | TSO|2035|Winter | -4.540e-06 | 5.714e-06 | U | -2.000e-08 | (7, 2025, Spring, 6) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 0.000e+00 |
| pb_y2025_n9 | T5, T2 (appended by W160) | k* 174 | b7bce5a8bf2365b9 / c50a0e02 | C | -2.000e-06 | TSO|2035|Winter | -4.539e-06 | 5.714e-06 | U | -2.000e-08 | (9, 2025, Spring, 1) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 0.000e+00 |
| pb_y2030_n5 | T5, T2 (appended by W160) | k* 182 | 3846675acea98ea7 / e0e61484 | C | -2.000e-06 | TSO|2035|Summer | -4.619e-06 | 5.292e-06 | U | -2.000e-08 | (5, 2030, Spring, 1) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 3.632e-05 |
| pb_y2030_n7 | T5, T2 (appended by W160) | k* 195 | 828f03fa9ac2ca6e / 53c1de5c | C | -2.000e-06 | TSO|2035|Summer | -4.621e-06 | 5.297e-06 | U | -2.000e-08 | (7, 2030, Spring, 1) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 3.632e-05 |
| pb_y2030_n9 | T5, T2 (appended by W160) | k* 192 | 344c936fc23919be / da2f6bc1 | C | -2.000e-06 | TSO|2035|Summer | -4.620e-06 | 5.296e-06 | U | -2.000e-08 | (9, 2030, Spring, 1) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 3.631e-05 |
| yl_y2030 | T4, T2 (appended by W160) | k* 169 | 6e4f11a95dc3d585 / c408836e | C | -2.000e-06 | TSO|2035|Summer | -2.295e-06 | 2.603e-06 | U | -2.000e-08 | (7, 2030, Spring, 14) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 3.632e-05 |
| yl_y2035 | T4, T2 (appended by W160) | k* 167 | 3a2387f46448f9db / 7fca7670 | C | -2.000e-06 | TSO|2035|Summer | -2.279e-06 | 2.279e-06 | U | -2.000e-08 | (7, 2035, Spring, 1) | -8.000e-08 | 8.000e-08 | -1.728e-05 | 3.631e-05 |
| t11_3x3_x0 | T11 | certification 72 | f6e9cd53fdbb8ee8 / 8435c718 | C | 0.000e+00 | TSO|2025|Spring | — | — | U | — | — | — | — | -2.880e-05 | 0.000e+00 |
| t11_3x3_unit | T11 | certification 69 | c82522f470b35b58 / db77e154 | C | -1.995e-06 | DSO|7|2034|Autumn | -2.155e-06 | 2.857e-06 | U | -2.000e-08 | (7, 2025, Spring, 23) | -8.000e-08 | 8.000e-08 | -2.880e-05 | 0.000e+00 |
| t11_3x3_x0_continuation | T11 | continuation 88 | 25b92ae0f1f2c02e / 8435c718 | C | 0.000e+00 | TSO|2025|Spring | — | — | U | — | — | — | — | -2.880e-05 | 0.000e+00 |
| t6_passive_cold | T6 | arm evaluation | arm / 8435c718 | C (weighted total) | total 0.000e+00 | — | — | — | n/a (no ESSO solved: ['DSO', 'TSO']) | — | — | — | — | — | — |
| t6_passive_perturbed | T6 | arm evaluation | arm / 8435c718 | C (weighted total) | total 0.000e+00 | — | — | — | n/a (no ESSO solved: ['DSO', 'TSO']) | — | — | — | — | — | — |
| t6_passive_warm_from_certified | T6 | arm evaluation | arm / 8435c718 | C (weighted total) | total 0.000e+00 | — | — | — | n/a (no ESSO solved: ['DSO', 'TSO']) | — | — | — | — | — | — |
| t6_price_taker_cold | T6 | arm evaluation | arm / 8435c718 | C (weighted total) | total 0.000e+00 | — | — | — | n/a (no ESSO solved: ['DSO', 'TSO']) | — | — | — | — | — | — |
| t6_price_taker_perturbed | T6 | arm evaluation | arm / 8435c718 | C (weighted total) | total 0.000e+00 | — | — | — | n/a (no ESSO solved: ['DSO', 'TSO']) | — | — | — | — | — | — |
| t6_price_taker_warm_from_certified | T6 | arm evaluation | arm / 8435c718 | C (weighted total) | total 0.000e+00 | — | — | — | n/a (no ESSO solved: ['DSO', 'TSO']) | — | — | — | — | — | — |

## Cells not covered at P, and why

- **d_c52e1670** — closure: no record carries the closure level at cycle 150: component_levels_terminal.json is at cycle 198; per-cycle records carry only the aggregate slack_penalties (recourse_blocks_all.jsonl / creep_diagnostic_per_cycle.jsonl: voltage + flexibility + closure summed); ess_schedule_per_cycle.jsonl (uncommitted, where present) carries the network copies' charge / discharge per cycle, from which only the SIGNED s+ - s- could be reconstructed through the SoC recursion -- not done (a derived value, not the variable); pickles: certified_models.pkl holds cycle 198, not 150 -- no record or pickle holds cycle 150
- **d_c52e1670** — sigma_sum: component_levels_terminal.json is not at cycle 150
  - beside (terminal cycle 198, NOT the point, committed): max s⁺+s⁻ over active blocks -2.000e-06 MWh at TSO|2035|Winter (relative signed -1.135e-06); D3 -1.728e-05 MW
- **t6_passive_cold** — closure_per_block: arm record: block_components carry one scalar per block, not the closure level
- **t6_passive_perturbed** — closure_per_block: arm record: block_components carry one scalar per block, not the closure level
- **t6_passive_warm_from_certified** — closure_per_block: arm record: block_components carry one scalar per block, not the closure level
- **t6_price_taker_cold** — closure_per_block: arm record: block_components carry one scalar per block, not the closure level
- **t6_price_taker_perturbed** — closure_per_block: arm record: block_components carry one scalar per block, not the closure level
- **t6_price_taker_warm_from_certified** — closure_per_block: arm record: block_components carry one scalar per block, not the closure level

## Which records carry the quantities (every ADMM cell)

- **closure at P (exact, s+ + s- per block)**: component_levels_terminal.json (committed) blocks.*.unweighted.shared_ess_day_balance_slack -- only when its cycles_run == P
- **closure (signed s+ - s- per unit, terminal)**: results/*.xlsx "Relaxation Slacks TSO, DSOs" -- only where a workbook exists (uncommitted)
- **closure, weighted total**: evaluation_record.json component_decomposition_totals_weighted.shared_ess_day_balance_slack (committed; total over blocks)
- **closure NOT carried**: per-cycle records (recourse_blocks_all.jsonl, creep_diagnostic_per_cycle.jsonl) carry only slack_penalties = voltage + flexibility + closure summed; ess_schedule_per_cycle.jsonl (uncommitted) carries pch / pdch, not s+-
- **sigma per element at P**: esso_capture/<label>/node<n>_cycle<P>.jsonl (UNCOMMITTED; active cohort-periods only)
- **sigma sum at P**: component_levels_terminal.json esso_feasibility_violation_D3 (committed; sum over every node and (y, d, t)) -- only when cycles_run == P
- **sigma, every node, terminal**: results/*.xlsx "Slacks operation, aggregated" (uncommitted; where present)
- **pickles (never read)**: certified_models.pkl (persisted models at post_certification.json certification_cycle), esso_models_<label>.pkl (terminal ESSO models), results/FrozenSMOPF/*.pkl (fixtures)

## The min(pch, pdch) detector (l. 857)

- **exists in the production path**: yes -- see code_locations: `_get_esso_complementarity_diagnostics` is called in the production ESSO solve routine after every successful ESSO solve (node_id set, solver ipopt) and its result appended to the sink wired by SharedEnergyStorageData.optimize; persistent workers were off in every scored run (g record), so this sequential path is the one that ran
- **what it measures**: complementarity_ratio_max = max over active cohort-periods of min(pch, pdch) / s_max (plus the barrier-identity estimate of spurious throughput); it is LOGGED and RECORDED; no production code compares it with a threshold (scope: configuration_evidence.detector_consumers_search)
- **recorded per evaluation**: g_<label>.json esso_complementarity_diagnostics_by_round (committed) and leak_classification_<label>.jsonl (committed), per ESSO solve and cycle; child_stdout.log lines "[INFO] Shared ESS complementarity diagnostics"
- **outcomes at P**: per cell in cells/<cell>.json detector.per_node_at_point; max over cells in the table above
- **max ratio at P over certified points**: 3.6321744275336843e-05
- **argmax cell**: e_c2

## Code locations (found by exact text at run time)

- `network.py:407` — closure slack s+ declared (e, s_m, s_o)
- `network.py:408` — closure slack s- declared
- `model_construction_helpers.py:983` — closure row
- `model_construction_helpers.py:990` — closure row with the slack pair: SoC_T = 0.5 E + s+ - s- (scenario-free copy)
- `model_construction_helpers.py:987` — closure target 0.5 E (E = the published capacity Param)
- `model_construction_helpers.py:846` — non-first scenario pairs skipped
- `model_construction_helpers.py:410` — closure slack bound fraction
- `model_construction_helpers.py:1166` — closure slack upper bound, re-applied per candidate ([0, 0] when inactive)
- `model_construction_helpers.py:2333` — closure penalty
- `model_construction_helpers.py:2341` — closure penalty baseMVA * 1e3 * (s+ + s-)
- `definitions.py:58` — closure penalty weight (EUR/MWh)
- `definitions.py:63` — ESSO P-net slack penalty weight
- `definitions.py:90` — closure slack bound offset (p.u.)
- `shared_resources_planning.py:771` — per-block closure level (the field component_levels_terminal.json records)
- `shared_energy_storage_data.py:460` — ESSO P-net slack sigma+ declared (y, d, t)
- `shared_energy_storage_data.py:461` — ESSO P-net slack sigma- declared
- `shared_energy_storage_data.py:782` — P-net row: es_pnet = sum_cohorts(pch - pdch) + sigma+ - sigma-
- `shared_energy_storage_data.py:829` — sigma penalty in the ESSO objective
- `shared_energy_storage_data.py:127` — sum of sigma over nodes and (y, d, t) (component_levels_terminal esso_feasibility_violation_D3)
- `shared_energy_storage_data.py:1082` — Addendum 3 remedy (h) tolerance override (tightened by Addendum 8)
- `shared_energy_storage_data.py:1981` — min(pch, pdch) / s_max detector (ratio form)
- `shared_energy_storage_data.py:2095` — per-solve detector + barrier-identity estimate
- `shared_energy_storage_data.py:1378` — detector CALLED after every successful ESSO solve (production ESSO solve path)
- `shared_energy_storage_data.py:1384` — detector outcome appended to the sink the campaign harness records
- `shared_energy_storage_data.py:109` — sink wired by the production ESSO optimize entry point
- `p515_g_g1_g4_admm_gates.py:1804` — writer of component_levels_terminal.json (from the final models)
- `p515_g_g1_g4_admm_gates.py:1854` — esso_feasibility_violation_D3 written
- `p515_g_g1_g4_admm_gates.py:458` — esso_capture per-element sigma written
- `p515_g_g1_g4_admm_gates.py:380` — esso_capture writer (active cohort-periods only)
- `model_construction_helpers.py:1067` — zero-capacity test (used here, in p.u., to decide which units are active)
- `definitions.py:93` — its tolerance (p.u.)

## Sentences for the two CONFIRM comments (scoped to this evidence)

- **l. 747 (closure)**: The strict sentence is NOT supported at 1e-6 of E^Av: s+ + s- exceeds 1e-6 x E^Av at ['pb_y2025_n5'] (largest 5.039e-06 x E^Av = 1.764e-06 MWh, at pb_y2025_n5). A sentence the evidence supports: 'The closure slack was at most 1.8e-06 MWh (5e-06 of the available energy) in every evaluation reported in this paper.' Detail: {"pb_y2025_n5": {"superseded": true, "block": "TSO|2035|Spring", "relative": 5.039052500543942e-06, "s_mwh": 1.7636683499952684e-06, "solve_at_point": {"block": "TSO|2035|Spring", "primary_termination": "maxIterations", "termination": "recovered", "class": "recovered_tier1"}}}. Excluding superseded certificates: no block exceeds 1e-6 x E^Av (signed max -4.493e-07 at j_5f3cccb4), and the strict sentence holds for every evaluation the tables use as a current certificate. Every negative block level is at or above the solver's relaxed-bound floor: True. [Scope: 63 of 64 ADMM evaluations carried at their point by a committed record (component_levels_terminal.json); not carried at the point: ['d_c52e1670']; the six T6 arms are at x = 0, every closure pair bounded [0, 0], recorded weighted total exactly 0.0: True.]
- **l. 857 (sigma)**: The slack pair sigma was inactive at every certified point: sigma+ + sigma- stood at its lower bound to solver tolerance (signed maximum -8.000e-09, absolute maximum 8.000e-08 of the converter rating). The post-solve check of min(P^Ch, P^Dch) / S runs after every ESSO solve in the production path and its value is recorded for every solve. [Scope: per-element sigma from UNCOMMITTED esso_capture files at 46 of 51 certified points (the rest have no active unit: x = 0); the committed record carries only the sum over all elements (esso_feasibility_violation_D3), which bounds every element at the same level under the relaxed-bound floor.]

## Integrity

Integrity failures: 0

