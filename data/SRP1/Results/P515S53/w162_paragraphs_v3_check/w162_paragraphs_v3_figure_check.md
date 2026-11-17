# W162 -- figure check of paragraphs_v3.md (Addendum 66)

Text: `data/SRP1/Results/P515S53/w160_step6_frozen/export/paragraphs_v3.md` sha256 `62d26e275a3c65a7990c2ec8fd0d30050ad5a8fef32513424415eb43c9682673` (commit `cb0a0bc6`). Frozen tables `frozen_step6_tables_v1_590088fe.json` (sha256 `590088fe…`). Script `p515_s53_w162_paragraphs_v3_check.py`. ZERO SOLVES (guards verified 0), pickle blocked. The text is not edited.

## Counts (manuscript scope: lines 10-140 of paragraphs_v3.md)

- numeric tokens in scope: 148 (non-manuscript title/comment: 6)
- figure checks touching v3 prose: 116 -- match 111, MISMATCH 2, approximate 2, no table counterpart 1
- W161 checks carried: 28 verbatim + 15 on a rewritten sentence; 121 of 164 not in the v3 prose (sources tables, map, scorecard, dropped)
- new v3 checks (N): 73; non-manuscript identifier checks (X): 4
- tokens unchecked: 16 (enumerators 9, NO SOURCE FOUND 2, other 5)
- every token assigned: True

## Mismatches

- **N60** line(s) [121]: written `10⁻⁴`, counterpart DSO 1e-04, TSO 0.0005 -- production complementarity tolerance before the tail: DSO = IPOPT default 1e-4 (network.py; no DSO case file sets it); TSO = data/SRP1/case9/case9_params.json compl_inf_tol. holds for the distribution solves only; the transmission solves were at 5 × 10⁻⁴ before the tail (REVISION_CONTEXT Addendum 48 entry says the same)
- **N64** line(s) [123]: written `≈ 7.9`, counterpart 7.8 -- W109 tso + dso priced_neg_part_eur_block_weighted (k€). -7,848.08 € = 7.8481 k€, which is 7.8 at one decimal; 7.9 is the rounding of the rounded 7.85 (TASKS W109 line, Addenda 56)

## Approximate

- **N36**: written `3`, counterpart 3.36. the record states it as an estimate for a half-life above the window, not a measured drift; "measured" is the prose's word
- **N63**: written `≈ 1.1 × 10⁻⁶ relative, with the same sign`, counterpart c_star -1.200e-06, n7_4h_e1 -1.036e-06, x0 -1.116e-06; mean -1.117e-06. range 1.04–1.20 × 10⁻⁶ (one decimal: 1.0 / 1.1 / 1.2), mean 1.12; all negative

## Unchecked numbers

| line | token | category | reason |
|---|---|---|---|
| 1 | 6 | non-manuscript | title "Step 6" (stage label) |
| 22 | 1 | enumerator | list item 1 |
| 24 | 2 | enumerator | list item 2 |
| 26 | 3 | enumerator | list item 3 |
| 28 | 4 | enumerator | list item 4 |
| 29 | 5 | enumerator | list item 5 |
| 54 | one | not a figure | pronoun ("An evaluation without one") |
| 70 | 0 | notation | λ ∈ [0, c_flex]: the lower end of the set-valued interface price; no table carries it |
| 74 | one | not a figure | "they share one fingerprint": the three bullets that follow are each checked (L1-L6) |
| 87 | 10⁻⁵ | NO SOURCE FOUND | "≈ 10⁻⁵ of renewable energy": Addendum 56 states it; no record carries the renewable-energy denominator (searched w109_tso_curtailment_look.json, p515_s53_w109 script, TASKS.md, PLANNER_BRIEF Addenda 55-56) |
| 103 | one | NO SOURCE FOUND | "one machine": no evaluation record carries a host identifier (launch.json keys: command, cwd, thread caps, pids, utc, spec sha, label, key; W159 uname is a read at 2026-10-06) |
| 120 | Two | structural | "Two effects": the count of the two bullets that follow (each checked: N60-N65) |
| 131 | 1 | enumerator | sentence 1 |
| 133 | 2 | enumerator | sentence 2 |
| 136 | 3 | enumerator | sentence 3 |
| 138 | 4 | enumerator | sentence 4 |

## Definition checks

- **D1** settling slack = the objective at the cap minus the objective at the cycle where the earlier run had certified; no earlier certificate -> no slack -> indeterminate: **consistent**. the scorer uses |s| in the bar (the prose's "movement" is signed s; the bar takes its absolute value); Q_N_old is the ORIGINAL record's last cycle, which D2 shows is its certification cycle on every uncertified evaluation of the tables
- **D2** Every uncertified evaluation in the tables has such an earlier certificate: **consistent**. scope: the evaluations of the frozen cell registry (T1/T2/T4/T5/T9/T10). T6 benchmark arms and the T11 3 × 3 evaluations carry no settling-rule status in the frozen JSON and are not covered
- **D3** Every evaluation repeated from an earlier record replayed that record bitwise up to k₀: **consistent**. 43 evaluations; replay through == k0_run on each (the W101 references replay through N > k0, N49)
- **D4** "Every local NLP must return a clean solution (defined below) in the same cycle" (the residual-pass condition): **OBSERVATION: the code requires a successful solve, not the 10× clean classification**. the clean classification (Optimal on any tier, or Acceptable on the primary attempt within 10×) is the certification veto of clause 5 (settling_criterion_v5 all_clean_k), not part of boyd_k; v2 read "must solve successfully". Whether IPOPT "Solved To Acceptable Level" maps to a successful termination here was not confirmed
- **D5** "every cycle the test reads was solved cleanly" (clause 5): **INCONSISTENT with the code: the veto covers the certifying window; 4 certificates read a non-clean cycle outside it**. v2 read "every cycle in the window the test reads was solved cleanly", which matches the code; Addendum 61 ruling 3 accepted the out-of-window reads as enumerated, not vetoed
- **D6** "on the reference evaluations the objective moved by a further 15–21 k€ after the residuals first passed": **figures match s (N4, N5); ANCHOR differs: s is measured from the old certificate N = k0 + 9, not from the first residual pass k0**. measured from k0 the net movement is x0 -16,025.26, n7_4h_e1 -1,440.72; the largest excursion from Q(N) over N..k* is x0 19,940.78, n7_4h_e1 33,604.52. "The reference evaluations" = x0 (d110bd1a) and the unit (3f084f2f); C* is uncertified (s −4,378.88) and outside 15–21

## Every check in v3

| id | lines | written | status | counterpart (at written precision) | source |
|---|---|---|---|---|---|
| S1a | 131 | +14.2 | match | 14.2 | T10 H:m1.75 d gross |
| S1b | 131 | +46.3 | match | 46.3 | T1 H:m2 d gross |
| S1c | 132 | 1.63 | match | 1.63 | derived: 1.5 + 0.25·(−d(1.5)) / (d(1.75) − d(1.5)) from T1 H:m1.5 and T10 H:m1.75 |
| S2a | 133 | +0.18 | no table counterpart | None | T10 carries EFC/day at 0.50 only; the 0.70 per-year EFC/day (W153) is not in T1-T10 |
| S2b | 134 | +4.9 | match | 4.9 | T10 E:soh050:delta_value_vs_070 |
| S2c | - | within resolution | match | within resolution | T10 verdict |
| S2d | 134 | 59.5 | match | 59.5 | T10 E:soh050:value_minus_I |
| S2e | - | determinate | match | determinate | T10 verdict |
| V1a | 136 | −4.1 | match | -4.1 | T1 E no_ageing value − I |
| V1b | - | within resolution | match | within resolution | T1 verdict |
| V1c | 137 | 31.6 | match | 31.6 | T8 aged arms: smallest loss |
| V1d | 137 | 73.7 | match | 73.7 | T8 aged arms: largest loss |
| C1 | 40 | 4,539.07 | match | 4539.07 | constants.TAU |
| C2 | 24 | 453.91 | match | 453.91 | constants.TAU / 10 |
| C3 | 28 | 2,269.53 | match | 2269.53 | constants.TAU / 2 |
| C4 | 42 | 10 | match | 10 | T2 at_or_above_0.95_tau_counted (count of true) |
| C5 | 60 | 42 | match | 42 | T2 cells with certification_stats_included |
| C6 | 60 | 32 | match | 32 | T2 |
| C7 | 60 | 24 | match | 24 | T2 branch |
| C8 | 60 | 8 | match | 8 | T2 branch |
| C9 | 61 | 5 | match | 5 | T2 cause (W157 wording) = gap clause |
| C10 | 61 | 2 | match | 2 | T2 cause (W157 wording) = lapse reset |
| C11 | 61 | 3 | match | 3 | T2 cause (W157 wording) = growth test |
| C12 | 62 | 174 | match | 174 | T2 median k* (32 certified) |
| C13 | 62 | 65 | match | 65 | T2: median of k* − k0, where T2 k0 = k0_run = the first residual pass N |
| C14 | 63 | 21 | match | 21 | T2 min k* − k0 |
| C15 | 63 | 89 | match | 89 | T2 max k* − k0 |
| C16 | 63 | 0.86 | match | 0.86 | T2 median range/τ (32 certified) |
| C17 | 63 | 4 | match | 4 | T10 cells certified |
| C33 | 84 | 21 | match | 21 | T2 non-clean cycles after N |
| C34 | 84 | 10 | match | 10 | T2 |
| L1 | 75 | −8.8 | match | -8.8 | T9 m = 2 cells and F2 references: max t_sum |
| L2 | 75 | −9.4 | match | -9.4 | T9: min t_sum |
| L3 | 76 | 55 | match | 55 | T9 node 5 share of the m = 2 cells with a per-node record (j_5f3cccb4, l_45aa25a6, l_7c455554, l_b2251bc5, l_0ee93aca), integer % |
| L4 | 76 | 31 | match | 31 | T9 node 9 share of the m = 2 cells with a per-node record (j_5f3cccb4, l_45aa25a6, l_7c455554, l_b2251bc5, l_0ee93aca), integer % |
| L5 | 76 | 14 | match | 14 | T9 node 7 share of the m = 2 cells with a per-node record (j_5f3cccb4, l_45aa25a6, l_7c455554, l_b2251bc5, l_0ee93aca), integer % |
| L6 | 77 | 0.69 | match | 0.6819–0.6999 | T9 pf_primal last, six m = 2 cells (within ±0.01 of the written value) |
| L7 | 80 | +1.1 | match | 1.1 | T9 d_36686489 t_sum |
| L9 | 93 | 1.04 | match | 1.04 | T8 ε_AE band over resolvable arms (0.70) |
| L10 | 93 | 1.85 | match | 1.85 | T8 |
| L11a | 94 | 0.41 | match | 0.41 | T8 ε_AE (0.50, superseded), min over the resolvable arms (eps_AE_resolvable_070 true) |
| L11b | 94 | 0.62 | match | 0.62 | T8 ε_AE (0.50, superseded), max over the resolvable arms |
| L11c | 94 | three | match | three | T8 count of resolvable arms (the arms of the 0.70 band 1.04–1.85) |
| N1 | 13 | Boyd et al. 2011, §3.3 | match | 2011, Sec. 3.3.1 (within §3.3) | shared_resources_planning.py comment "P5.15 Step 3.2 (Boyd et al. 2011 Sec. 3.3.1): the stopping test is the Boyd primal/dual residual test" |
| N2 | 13 | 10⁻⁵ | match | 1e-05 | data/SRP1/SRP1_params.json admm.tol.boyd.eps_abs |
| N3 | 14 | 10⁻⁴ | match | 0.0001 | data/SRP1/SRP1_params.json admm.tol.boyd.eps_rel |
| N4 | 17 | 15 | match | 15 | W101 three-reference summary (62bdeafe) reports.x0.s_signed (k€) |
| N5 | 17 | 21 | match | 21 | W101 three-reference summary (62bdeafe) reports.n7_4h_e1.s_signed (k€) |
| N6 | 18 | 10⁻⁶ | match | 1e-06 | v6 spec inputs_in_force_now.configuration_now.convergence_depth_tail.compl_inf_tol |
| N7 | 22 | three | match | 3 | settling_criterion_v2.SettlingRuleV2.evaluate: at_least_3_turning_points = len(T) >= 3 (called by settling_criterion_v6.SettlingRuleV6.evaluate) |
| N8 | 24 | 10 | match | 10 | settling_criterion_v6.SWING_FLOOR = TAU / 10 (= GROWTH_TEST_FLOOR = TURNING_POINT_FLOOR); v6 spec stop_rule.swing_floor.F |
| N9 | 26 | 20 | match | 20 | settling_criterion.W_MIN; v6 spec stop_rule.W.oscillatory |
| N10 | 26 | 1.1 | match | 1.1 | settling_criterion.W_FACTOR |
| N11 | 27 | the three most recent turning points: from the third-last to the last | match | P_hat 28 = T[-1]-T[-3] 28 (T[2]-T[0] 15) | settling_criterion_v2.SettlingRuleV2.evaluate: p_hat = T[-1][0] - T[-3][0] (inherited by v6); v6 spec stop_rule.W.oscillatory; synthetic v6 replay (stdlib, no s |
| N12 | 28 | 2 | match | 2 | settling_criterion_v2.GAP_BOUND = TAU / 2 (v6 GAP_BOUND) |
| N13 | 30 | four | match | 4 | settling_criterion_v5.METRICS (v6) and v6 spec stop_rule.clean_rule.metric_table |
| N14 | 30 | 10 | match | 10 | settling_criterion_v5.CLEAN_FACTOR (v6); v6 spec stop_rule.clean_rule.factor; PRIMARY_ATTEMPT |
| N15 | 32 | 2 P_max = 60 | match | 60 | v6 spec stop_rule.p_max.L and stop_rule.W.monotone ("L = L_MONO = 2 * P_MAX = 60") |
| N16 | 32 | 30 | match | 30 | v6 spec stop_rule.p_max.value (source "W102 x0 P_hat 29, W103 unit P_hat 30"); W101 summary P_hat x0 / unit |
| N17 | 36 | 60 | match | 60 | v6 spec stop_rule.constants.MONOTONE_LAST_STEP_CLAUSE "abs(dQ_k) * L_MONO <= TAU" with L_MONO 60; settling_criterion_v2 last_step_times_L_le_tau |
| N18 | 38 | 4 | match | 4 | settling_criterion.TAU = DELTA_R * R_REF / 4.0 (constants() formula) |
| N19 | 38 | 0.07 | match | 0.07 | settling_criterion.DELTA_R; v6 spec stop_rule.constants.DELTA_R |
| N20 | 38 | 259,375.33 | match | 259375.33 / 259375.33 | settling_criterion.R_REF; W101 summary expert_P2.V_old (the SRP1 value Q(x0, N) − Q(unit, N) at the certificates in force when v39 froze) |
| N21 | 39 | Four | match | 4 | TAU divisor 4 (N18); T11 rows: V = Q(0) − Q(unit) (two evaluations per value) and R = V / V_SRP1 (two values): 2 × 2 = 4 evaluations |
| N22 | 39 | two | match | 2 | T11: R = V / V_SRP1_settled (a ratio of two values) |
| N23 | 42 | 46 | match | 46 | frozen tables.at_or_above_0_95_tau.registry: certified (status in T2 / appended cells) − superseded − bitwise twins |
| N24 | 42 | 5 | match | 5 | frozen constants.FLAG_RANGE_OVER_TAU / registry threshold 0.95: 1 − 0.95 |
| N25 | 44 | 0.9 | match | max 0.882 τ | records: Q(last) − Q(k*) of the runs that continued past a settling-rule certificate of the same evaluation (cell 1 b_2a0ba8b2 v3 run past its v4 k* 173; cell 3 |
| N26 | 49 | three | match | 3 × max(1000, 2000) = 6000 | p515_s53_w132_resettle_v3_campaign.resolve (the scorer): bar = 3.0 × max(gap, slack) over the uncertified cell(s); behaviour on a synthetic view |
| N27 | 49 | two | match | 2 | resolve: bar_components = [gap, slack] per uncertified cell |
| N28 | 58 | 3 | match | 3 | settling_criterion_v6.DETERMINACY_BAR_FACTOR; determinacy_threshold behaviour |
| N29 | 58 | 2 | match | 2 | settling_criterion_v6.DETERMINACY_TAU_MULTIPLE; determinacy_threshold(100, 200) = 2 TAU |
| N73 | 57 | two | match | 2 bars | settling_criterion_v6.determinate_certified(margin, bar_r, bar_o): the difference of two certified cells, one bar each |
| N30 | 60 | 10 | match | 10 | T2 cells with certification_stats_included, status uncertified (W153 totals.n_uncertified) |
| N31 | 79 | 5 | match | 5 | T2 d_36686489 candidate_canonical.nodes.7 = [1.25 MVA, 5.0 MWh]; v6 spec m_flex_price_multiplier 1.0 (baseline price); cause growth test |
| N32 | 84 | 2035 Spring | match | ['TSO|2035|Spring'] (6 of 21) | W153 certification statistics (48e76c9f) cells[].non_clean_after_N_detail blocks, tallied; totals.non_clean_block_events_after_N_by_family_total |
| N33 | 86 | 10⁻⁵ | match | 1e-05 | definitions.py EQUALITY_TOLERANCE = 1e-5; model_construction_helpers.py curtaillable pg upper bound pg_avail + EQUALITY_TOLERANCE; W109 equality_tolerance_pu |
| N34 | 123 | 10⁻⁵ | match | 1e-05 | definitions.py EQUALITY_TOLERANCE = 1e-5; model_construction_helpers.py curtaillable pg upper bound pg_avail + EQUALITY_TOLERANCE; W109 equality_tolerance_pu |
| N35 | 87 | 62 | match | 62 | W109 (f6e3533f) tso + dso neg_part_eur_at_1_block_weighted (MWh-equivalent at 1 €/MWh, block-weighted) |
| N36 | 90 | 3 | approximate | 3.36 | P5_15_ADDENDUM54_56_CONSOLIDATED_NOTE.md (ce96d492): "the monotone branch certifies a drifting cell with ≈ 3 τ left whenever the decay half-life exceeds its 44- |
| N37 | 91 | 44 | match | 44 | frozen spec v41 (fcea4b38, the C* extension W110) report_only_rule.constants.L_MONO and cell.declaration.settling_rule.l_mono |
| N38 | 93 | 0.70 | match | 0.70 | v6 spec inputs_in_force_now ess_ageing_baseline.minimum_soh (the baseline label "soh_min 0.70") |
| N40 | 133 | 0.70 | match | 0.70 | v6 spec inputs_in_force_now ess_ageing_baseline.minimum_soh (the baseline label "soh_min 0.70") |
| N42 | 137 | 0.70 | match | 0.70 | v6 spec inputs_in_force_now ess_ageing_baseline.minimum_soh (the baseline label "soh_min 0.70") |
| N39 | 94 | 0.50 | match | 0.50 | T8 column label eps_AE_050_superseded ("soh_min 0.50") |
| N41 | 133 | 0.50 | match | 0.50/0.50 | A64 e_soh050 r2 campaign spec (28c9478d) candidates[0].settling_resettle.minimum_soh and extra.minimum_soh (the override; configuration.ess_ageing_baseline.mini |
| N43 | 103 | one (solver build) | match | pinned binary, unchanged | W159 (62749030) c3: /usr/local/bin/ipopt sha = the v6 pin, not modified at or after the first v6 start; environment unchanged |
| N44 | 103 | one (evaluation at a time) | match | 1 | v6 spec configuration.concurrency |
| N45 | 104 | single-threaded; one thread | match | thread caps 1 on every table evaluation | W159 c1: harness THREAD_CAP_ENV lines (OMP / MKL / OPENBLAS / VECLIB / NUMEXPR = 1); all_table_evaluations_ran_with_OMP_NUM_THREADS_1 |
| N46 | 105 | MA97 ran serially | match | no OpenMP link or symbol | W159 c4 installed_ipopt_hsl: links_libgomp_or_libomp False, nm_symbols_openmp [], nm_undefined_openmp [] |
| N47 | 107 | 72; 72/72 | match | 72/72 | W98 r2 campaign_results.json (4689475e) replay_gate: bitwise_through_N, N 72, first divergence none; T11 row "72 / 72 cycles" |
| N48 | 107 | 3 × 3; x = 0 | match | 3x3 / x0 | W98 r2 replay reference data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/evals/f6e9cd53fdbb8ee8_x0/per_cycle_record.jsonl (the 3 × 3 pair, x0); candi |
| N49 | 108 | three; 3/3 | match | 3/3 | W101 campaign_results (x0, unit, C*): gates.G19_replay_bitwise_1_N_every_field and G13 replay_bitwise_through_cycle = N (the certification cycle of the replayed |
| N50 | 112 | 3.14.18 | match | 3.14.18 | W159 c3 ipopt --version; c2 banners (ipopt_versions_in_all_banners) |
| N51 | 112 | MA97 (network subproblems) | match | {"ma97": 6887} | W159 c2 banner tally over the 495 hash-matching logs of d_4a82a64a: network |
| N52 | 113 | MA57 (storage-operator subproblem) | match | {"ma57": 429} | W159 c2 banner tally: esso |
| N53 | 113 | 3.11.11 | match | 3.11.11 | W159 c3 sys_version |
| N54 | 113 | 6.9.5 | match | 6.9.5 | W159 c3 pyomo_version |
| N55 | 113 | 27.0 | match | 27.0 | W159 c3 sw_vers ProductVersion; sw_vers now |
| N56 | 113 | Apple M2 Max | match | Apple M2 Max | sysctl machdep.cpu.brand_string, read now (no committed record carries it) |
| N57 | 114 | 32 GB | match | 32 GiB | sysctl hw.memsize now; v6 spec memory_preflight hw_memsize_bytes |
| N58 | 116 | Two distribution blocks (node 7 in 2025 Winter, node 5 in 2035 Winter) | match | DSO|5|2035|Winter 109, DSO|7|2025|Winter 10 | W153 cells[].acceptable_clean_after_N_detail blocks, tallied; totals.acceptable_clean_after_N_by_family_total |
| N59 | 117 | 10 | match | 10 | settling_criterion_v5.CLEAN_FACTOR (W153 acceptable_clean = primary Acceptable within 10×) |
| N60 | 121 | 10⁻⁴ | MISMATCH | DSO 1e-04, TSO 0.0005 | production complementarity tolerance before the tail: DSO = IPOPT default 1e-4 (network.py; no DSO case file sets it); TSO = data/SRP1/case9/case9_params.json c |
| N61 | 121 | 10⁻⁶ | match | 1e-06 | as N6 (tail compl_inf_tol) |
| N62 | 122 | three | match | 3 | W86 tail recert campaign_results.json per_cell (x0, unit, C*) |
| N63 | 122 | ≈ 1.1 × 10⁻⁶ relative, with the same sign | approximate | c_star -1.200e-06, n7_4h_e1 -1.036e-06, x0 -1.116e-06; mean -1.117e-06 | W86 per_cell.*.comparison.dQ_relative |
| N64 | 123 | ≈ 7.9 | MISMATCH | 7.8 | W109 tso + dso priced_neg_part_eur_block_weighted (k€) |
| N65 | 123 | ≈ 1.2 × 10⁻⁵ of it | match | 1.2e-5 | W109 priced negative parts / Q of the settled x0 (T2 ref:7aa017f0, the W109 instance d110bd1a cycle 181) |
| N66 | 126 | 2026-10-06 | match | 2026-10-06 | W159 c3 read_at_utc; environment_unchanged_since_first_v6_start_by_these_reads |
| N67 | 131,132 | 1.75 | match | 1.75 | T10 claim H:m1.75:value_minus_I; A64 h_unit_m175 r2 spec flex_price_multiplier |
| N68 | 131 | 2 | match | 2 | T1 claim H:m2:value_minus_I |
| N69 | 132 | 1.5 | match | 1.5 | T1 claim H:m1.5:value_minus_I |
| N70 | 133 | +0.18 | match | 0.18 | T10 a64.scored.B.efc_per_day[2025] (0.50) − W153 w153_discount_row.json settled_unit_ageing.per_year[2025].efc_per_day (0.70) |
| N71 | 134 | +0.20 | match | 0.20 | T10 a64.scored.B.efc_per_day[2030] (0.50) − W153 w153_discount_row.json settled_unit_ageing.per_year[2030].efc_per_day (0.70) |
| N72 | 134 | +0.27 | match | 0.27 | T10 a64.scored.B.efc_per_day[2035] (0.50) − W153 w153_discount_row.json settled_unit_ageing.per_year[2035].efc_per_day (0.70) |

## W161 checks not in the v3 prose

V1e, V1f, V1g, V1h, C18, C19, C20, C21, C22, C23, C24, C25, C26, C27, C28, C29, C29b, C29c, C29d, C30, C31, C32, C35, C36, C37, C38, C39, C40, C41, C41b, C43, C43b, L5b, L5c, L8, L12.0, L12.1, L12.2, L12.3, L12.4, L12.5, L12.6, L13, L14, L15, L16, L17, L18, L19, L20, L21, L22, L23, L24, L25, L26, P14, P14b, P20, P22, P24, P32a, P32b, P32c, P32d, P32e, P33a, P33b, P33c, P35, P36a, P36b, P37, P38, P47a, P47b, P47c, P47d, P47e, P47f, P48a, P48b, P48c, P48d, P48e, P48f, P48g, P48h, M1, M2, M3, M4, M5.0a, M5.0b, M5.0c, M5.1a, M5.1b, M5.1c, M5.2a, M5.2b, M5.2c, M5.3a, M5.3b, M5.3c, M5.4a, M5.4b, M5.4c, M5.5a, M5.5b, M5.5c, M6, M6b, M7, M8, M9, M10, M11, M12, M13, M14, M15
