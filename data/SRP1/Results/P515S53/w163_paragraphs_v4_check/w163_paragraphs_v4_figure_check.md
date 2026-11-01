# W163 -- figure check of paragraphs_v4.md (Addendum 66)

Text: `data/SRP1/Results/P515S53/w160_step6_frozen/export/paragraphs_v4.md` sha256 `9ed77949b3f86efe62a2b992485074e6116e24b8dc2e979bbfc001613343dea9` (commit `6b5aaadf`); v3 `62d26e27` (`cb0a0bc6`) for the line map. Frozen tables `frozen_step6_tables_v1_590088fe.json` (sha256 `590088fe…`). Script `p515_s53_w163_paragraphs_v4_check.py` (imports `p515_s53_w162_paragraphs_v3_check.py`, not edited). ZERO SOLVES (guards verified 0), pickle blocked. The text is not edited.

## Counts (manuscript scope: lines 11-144 of paragraphs_v4.md)

- numeric tokens in scope: 149 (non-manuscript title/comments: 13)
- figure checks touching v4 prose: 118 -- match 116, MISMATCH 0, approximate 1, no table counterpart 1
- W161 checks carried: 28 verbatim + 15 on a sentence rewritten in v3; 121 of 164 not in the v4 prose
- W162 new checks: 65 kept (tokens remapped), 8 rebuilt on the v4 wording (N4, N5, N25, N36, N57, N60, N61, N64); new in W163: N74, N75, X5, X6, X7; X: {'X1': 'match', 'X2': 'match', 'X3': 'match', 'X4': 'match', 'X5': 'match', 'X6': 'match', 'X7': 'match'}
- tokens unchecked: 15 {'non-manuscript': 1, 'enumerator': 9, 'not a figure': 2, 'notation': 1, 'NO SOURCE FOUND': 1, 'structural': 1}
- every token assigned: True

## W162 non-matches, on v4

| id | W162 on v3 | W163 on v4 | v4 written | counterpart |
|---|---|---|---|---|
| S2a | no table counterpart | no table counterpart | +0.18 | None |
| N36 | approximate | match | 3 | 3 (3.36) |
| N60 | MISMATCH | match | 10⁻⁴ (distribution subproblems) | 0.0001 |
| N63 | approximate | approximate | ≈ 1.1 × 10⁻⁶ relative, with the same sign | c_star -1.200e-06, n7_4h_e1 -1.036e-06, x0 -1.116e-06; mean -1.117e-06 |
| N64 | MISMATCH | match | ≈ 7.8 | 7.8 |
| D4 | OBSERVATION: the code requires a successful solve, not the 10× clean classification | consistent: the code requires a successful solve (status ok, termination optimal / locallyOptimal / globallyOptimal), as v4 now says | None | None |
| D5 | INCONSISTENT with the code: the veto covers the certifying window; 4 certificates read a non-clean cycle outside it | window scope CONSISTENT with the veto; "read only for the turning points and the swing history" is INCOMPLETE: before the window the test also reads oscillatory: window_inside_run (k0: the absence of a Boyd lapse (Q None or not boyd_k) over [k0, k]); monotone: lo_ge_k0_plus_K_EXCL (k0: the absence of a Boyd lapse over [k0, k]), no_sign_change_in_window (the steps over [lo, k] AND the sign of the last non-zero step before lo (cycles j - 1, j, j the last c in [k0 + K_EXCL, lo - 1] with |dQ_c| >= EPS0): a sign change at the first non-zero step of the window is judged against it), steps_decreasing ([lo - 1, k] (dQ_lo = Q_lo - Q_(lo-1))) | None | None |
| D6 | figures match s (N4, N5); ANCHOR differs: s is measured from the old certificate N = k0 + 9, not from the first residual pass k0 | consistent: x0 certified at 132 and the unit at 112, both by the residual rule (10 consecutive cycles passing the residual test with successful local solves, from 123 / 103); s (N4, N5) is measured from those certificates | None | None |

## Mismatches

none

## Approximate

- **N63**: written `≈ 1.1 × 10⁻⁶ relative, with the same sign`, counterpart c_star -1.200e-06, n7_4h_e1 -1.036e-06, x0 -1.116e-06; mean -1.117e-06. range 1.04–1.20 × 10⁻⁶ (one decimal: 1.0 / 1.1 / 1.2), mean 1.12; all negative

## New and rebuilt checks (v4)

- **N4** (match) written `15` -> 15; W101 three-reference summary (62bdeafe) reports.x0.s_signed (k€), measured from the old residual-rule certificate N. s = Q(k*) − Q(N), N = the old certificate (x0 132, unit 112): W86 records 5cfe69a6 / ca8927e7, status certified, stopped_by boyd, 10 consecutive residual passes (123-132 / 103-112), no settling rule in the W86 spec (D6: anchor consistent). "The reference evaluations" = x0 (d110bd1a) and the unit (3f084f2f); C* (uncertified, s = −4,378.88) is outside the range
- **N5** (match) written `21` -> 21; W101 three-reference summary (62bdeafe) reports.n7_4h_e1.s_signed (k€), measured from the old residual-rule certificate N. s = Q(k*) − Q(N), N = the old certificate (x0 132, unit 112): W86 records 5cfe69a6 / ca8927e7, status certified, stopped_by boyd, 10 consecutive residual passes (123-132 / 103-112), no settling rule in the W86 spec (D6: anchor consistent). "The reference evaluations" = x0 (d110bd1a) and the unit (3f084f2f); C* (uncertified, s = −4,378.88) is outside the range
- **N25** (match) written `0.9` -> max 0.882 τ; records: Q(last) − Q(k*) of the runs that continued past a settling-rule certificate of the same evaluation (cell 1 b_2a0ba8b2 v3 run past its v4 k* 173; cell 3 b_4649234b v4 run past its v5 k* 148; d_c52e1670 v5 run past its v6 k* 150, and past 148 on the last-pair reading); W141 d_c52e1670_detail. a bound over the runs continued past a SETTLING-RULE certificate (the W162 computation, unchanged): |movement| / τ = 0.004 / 0.738 / 0.781 / 0.882. v4 now scopes the sentence to "a certificate under this rule", which is the computed scope; runs continued past an old residual-rule certificate are outside it (the SRP1 references, s = 3.30 τ and 4.58 τ; the 3 × 3 x = 0, 0.99 τ per W162's note)
- **N36** (match) written `3` -> 3 (3.36); P5_15_ADDENDUM54_56_CONSOLIDATED_NOTE.md (ce96d492): "≈ 3 τ left whenever the decay half-life exceeds its 44-cycle window (C*'s is ≈ 102)"; TASKS.md W110 line: |dQ| decay 0.9932/cycle (validated out of sample), "half-life 102 > L_MONO 44", "≈ 3 τ gross drift left"; derived: a geometric tail with half-life 102 and |last step| × 44 ≤ τ leaves ≤ τ / (44 (1 − 2^(−1/102))). v4 calls it an estimate, as the record does; the half-life is a fitted (measured) decay: ln 0.5 / ln 0.9932 = 101.6 cycles ≈ 102; the derived bound 3.36 τ rounds to 3
- **N57** (match) written `32 GiB` -> 32 GiB; sysctl hw.memsize now; v6 spec memory_preflight hw_memsize_bytes. 34,359,738,368 bytes = 32 GiB exactly
- **N60** (match) written `10⁻⁴ (distribution subproblems)` -> 0.0001; IPOPT default compl_inf_tol: `/usr/local/bin/ipopt --print-options` (binary sha256 = the W159 / v6 pin); network.py IPOPT_DEFAULT_COMPL_INF_TOL; no DSO case file (case33_1/2/3_params.json) sets compl_inf_tol at a51ad9ba^, at the W86 campaign commit or now; W86 tail-state baselines DSO5/7/9 has_key False (x0, unit, C*). the distribution solves pass no compl_inf_tol, so IPOPT's default 1e-4 is in force before the tail
- **N61** (match) written `10⁻⁶ (both)` -> 1e-06; v6 spec convergence_depth_tail.compl_inf_tol; admm_parameters.py tail default; W86 tail states: every holder (TSO, DSO5/7/9; never the ESSO) at the tail value on every tail-active cycle and at its baseline otherwise. 
- **N64** (match) written `≈ 7.8` -> 7.8; W109 (f6e3533f) tso + dso priced_neg_part_eur_block_weighted (k€). -7,848.08 € = 7.8481 k€
- **N74** (match) written `5 × 10⁻⁴ (transmission subproblem)` -> parent_of_tail_commit 0.0005, w86_campaign_git_head 0.0005, HEAD 0.0005; data/SRP1/case9/case9_params.json solver.options.compl_inf_tol read at 590c298f (the parent of a51ad9ba, the commit that introduced the tight tail: "P5.15 Addendum 46 ruling 7 W83 step 1: convergence-depth tight tail (DEFAULT OFF) + per-so"), at the W86 campaign commit (spec git_head) and at HEAD; W86 tail-state baselines TSO (x0, unit, C*). the case file has carried 5e-4 since before the tail was introduced (the same blob at all three commits); the tail replaces it with 1e-6 on tail cycles and restores it otherwise
- **N75** (match) written `solve successfully (optimal or locally optimal)` -> status ok and optimal/locallyOptimal/globallyOptimal; helper_functions.solver_result_succeeded (helper_functions.py line 74: status ok (line 83) and termination in {optimal, locallyOptimal, globallyOptimal} (line 77)), called for every TSO, DSO and ESSO result by _admm_local_solves_succeeded (shared_resources_planning.py line 7734) -> local_solves_ok (line 3258) -> cycle_convergence = boyd_all_pass and local_solves_ok (line 3385). the code also accepts globallyOptimal, which the prose omits. The terms are Pyomo termination conditions: Pyomo's .sol reader maps IPOPT solve_result_num 0-99 to optimal with status ok (sol.py line 117), and in the records every final attempt ending "Solved To Acceptable Level." counted as successful (7 of 7 block-rounds in the W86 references, local_solves_ok True): "optimal" here includes IPOPT's acceptable-level exit
- **X5** (match) paragraphs_v3.md (sha256 62d26e27..., cb0a0bc6); the W162 figure check (d1f3e186): sha256 of paragraphs_v3.md; its last commit; the commit that added the W162 JSON. 
- **X6** (match) 7.8, 15, 21, 5, 3, 0.9, 32: the corrected manuscript tokens (line, text): 7.8 -> line 126, 15 -> line 18, 21 -> line 18, 5 -> line 30, 3 -> line 92, 0.9 -> line 46, 32 -> line 116. each echoed figure is checked where the manuscript writes it (N64, N4, N5, list item 5, N36, N25, N57)
- **X7** (match) 2026-10-07: the latest addendum header of the brief; paragraphs_v4.md commit date. equals the Addendum 66 date (the latest addendum); the file was committed 2026-10-06

## Unchecked numbers

| line | token | category | reason |
|---|---|---|---|
| 1 | 6 | non-manuscript | title "Step 6" (stage label) |
| 23 | 1 | enumerator | list item 1 |
| 25 | 2 | enumerator | list item 2 |
| 27 | 3 | enumerator | list item 3 |
| 29 | 4 | enumerator | list item 4 |
| 30 | 5 | enumerator | list item 5 |
| 56 | one | not a figure | pronoun ("An evaluation without one") |
| 72 | 0 | notation | λ ∈ [0, c_flex]: the lower end of the set-valued interface price; no table carries it |
| 76 | one | not a figure | "they share one fingerprint": the three bullets that follow are each checked (L1-L6) |
| 105 | one | NO SOURCE FOUND | "one machine": no evaluation record carries a host identifier (launch.json keys: command, cwd, thread caps, pids, utc, spec sha, label, key; W159 uname is a read at 2026-10-06) |
| 122 | Two | structural | "Two effects": the count of the two bullets that follow (each checked: N60-N65, N74) |
| 134 | 1 | enumerator | sentence 1 |
| 136 | 2 | enumerator | sentence 2 |
| 139 | 3 | enumerator | sentence 3 |
| 141 | 4 | enumerator | sentence 4 |

## Definition checks

- **D1** settling slack = the objective at the cap minus the objective at the cycle where the earlier run had certified; no earlier certificate -> no slack -> indeterminate: **consistent**. the scorer uses |s| in the bar (the prose's "movement" is signed s; the bar takes its absolute value); Q_N_old is the ORIGINAL record's last cycle, which D2 shows is its certification cycle on every uncertified evaluation of the tables
- **D2** Every uncertified evaluation in the tables has such an earlier certificate: **consistent**. scope: the evaluations of the frozen cell registry (T1/T2/T4/T5/T9/T10). T6 benchmark arms and the T11 3 × 3 evaluations carry no settling-rule status in the frozen JSON and are not covered
- **D3** Every evaluation repeated from an earlier record replayed that record bitwise up to k₀: **consistent**. 43 evaluations; replay through == k0_run on each (the W101 references replay through N > k0, N49)
- **D4** Every local NLP must also solve successfully (optimal or locally optimal) in the same cycle.: **consistent: the code requires a successful solve (status ok, termination optimal / locallyOptimal / globallyOptimal), as v4 now says**. globallyOptimal is accepted by the code but not named in the prose; IPOPT acceptable-level exits are "optimal" in Pyomo's classification and counted successful (N75)
- **D5** "every cycle in the window the test reads was solved cleanly" ... "Cycles before the window are read only for the turning points and the swing history, and are not vetoed.": **window scope CONSISTENT with the veto; "read only for the turning points and the swing history" is INCOMPLETE: before the window the test also reads oscillatory: window_inside_run (k0: the absence of a Boyd lapse (Q None or not boyd_k) over [k0, k]); monotone: lo_ge_k0_plus_K_EXCL (k0: the absence of a Boyd lapse over [k0, k]), no_sign_change_in_window (the steps over [lo, k] AND the sign of the last non-zero step before lo (cycles j - 1, j, j the last c in [k0 + K_EXCL, lo - 1] with |dQ_c| >= EPS0): a sign change at the first non-zero step of the window is judged against it), steps_decreasing ([lo - 1, k] (dQ_lo = Q_lo - Q_(lo-1)))**. the 4 certificates that read a non-clean cycle before their window are all oscillatory; reads before the window are enumerated, not vetoed (Addendum 61 ruling 3), as v4 says. Classification used: the sub-tests ['P_hat_and_W', 'at_least_3_turning_points', 'swings_non_increasing_floored', 'turning_point_floor'] count as turning-point / swing-history reads; every other sub-test that SUB_TEST_READS marks inside_the_window_as_implemented False is listed in the verdict
- **D6** "the objective moved by a further 15–21 k€ after the residual-based stopping rule had certified them": **consistent: x0 certified at 132 and the unit at 112, both by the residual rule (10 consecutive cycles passing the residual test with successful local solves, from 123 / 103); s (N4, N5) is measured from those certificates**. Q(N) = the W86 certified cost exactly on both; k0 = the first cycle of the certifying streak

## Every check in v4

| id | lines | written | status | counterpart (at written precision) | handling |
|---|---|---|---|---|---|
| S1a | 134 | +14.2 | match | 14.2 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| S1b | 134 | +46.3 | match | 46.3 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| S1c | 135 | 1.63 | match | 1.63 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| S2a | 136 | +0.18 | no table counterpart | None | W161 check carried by W162, re-run on v4 (tokens remapped) |
| S2b | 137 | +4.9 | match | 4.9 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| S2c | - | within resolution | match | within resolution | W161 check carried by W162, re-run on v4 (tokens remapped) |
| S2d | 137 | 59.5 | match | 59.5 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| S2e | - | determinate | match | determinate | W161 check carried by W162, re-run on v4 (tokens remapped) |
| V1a | 139 | −4.1 | match | -4.1 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| V1b | - | within resolution | match | within resolution | W161 check carried by W162, re-run on v4 (tokens remapped) |
| V1c | 140 | 31.6 | match | 31.6 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| V1d | 140 | 73.7 | match | 73.7 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C1 | 42 | 4,539.07 | match | 4539.07 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C2 | 25 | 453.91 | match | 453.91 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C3 | 29 | 2,269.53 | match | 2269.53 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C4 | 44 | 10 | match | 10 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C5 | 62 | 42 | match | 42 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C6 | 62 | 32 | match | 32 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C7 | 62 | 24 | match | 24 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C8 | 62 | 8 | match | 8 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C9 | 63 | 5 | match | 5 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C10 | 63 | 2 | match | 2 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C11 | 63 | 3 | match | 3 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C12 | 64 | 174 | match | 174 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C13 | 64 | 65 | match | 65 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C14 | 65 | 21 | match | 21 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C15 | 65 | 89 | match | 89 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C16 | 65 | 0.86 | match | 0.86 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C17 | 65 | 4 | match | 4 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C33 | 86 | 21 | match | 21 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| C34 | 86 | 10 | match | 10 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| L1 | 77 | −8.8 | match | -8.8 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| L2 | 77 | −9.4 | match | -9.4 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| L3 | 78 | 55 | match | 55 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| L4 | 78 | 31 | match | 31 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| L5 | 78 | 14 | match | 14 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| L6 | 79 | 0.69 | match | 0.6819–0.6999 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| L7 | 82 | +1.1 | match | 1.1 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| L9 | 95 | 1.04 | match | 1.04 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| L10 | 95 | 1.85 | match | 1.85 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| L11a | 96 | 0.41 | match | 0.41 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| L11b | 96 | 0.62 | match | 0.62 | W161 check carried by W162, re-run on v4 (tokens remapped) |
| L11c | 96 | three | match | three | W161 check carried by W162, re-run on v4 (tokens remapped) |
| N1 | 14 | Boyd et al. 2011, §3.3 | match | 2011, Sec. 3.3.1 (within §3.3) | W162 new check, re-run on v4 (tokens remapped) |
| N2 | 14 | 10⁻⁵ | match | 1e-05 | W162 new check, re-run on v4 (tokens remapped) |
| N3 | 15 | 10⁻⁴ | match | 0.0001 | W162 new check, re-run on v4 (tokens remapped) |
| N4 | 18 | 15 | match | 15 | rebuilt by W163 on the v4 wording |
| N5 | 18 | 21 | match | 21 | rebuilt by W163 on the v4 wording |
| N6 | 19 | 10⁻⁶ | match | 1e-06 | W162 new check, re-run on v4 (tokens remapped) |
| N7 | 23 | three | match | 3 | W162 new check, re-run on v4 (tokens remapped) |
| N8 | 25 | 10 | match | 10 | W162 new check, re-run on v4 (tokens remapped) |
| N9 | 27 | 20 | match | 20 | W162 new check, re-run on v4 (tokens remapped) |
| N10 | 27 | 1.1 | match | 1.1 | W162 new check, re-run on v4 (tokens remapped) |
| N11 | 28 | the three most recent turning points: from the third-last to the last | match | P_hat 28 = T[-1]-T[-3] 28 (T[2]-T[0] 15) | W162 new check, re-run on v4 (tokens remapped) |
| N12 | 29 | 2 | match | 2 | W162 new check, re-run on v4 (tokens remapped) |
| N13 | 31 | four | match | 4 | W162 new check, re-run on v4 (tokens remapped) |
| N14 | 31 | 10 | match | 10 | W162 new check, re-run on v4 (tokens remapped) |
| N15 | 34 | 2 P_max = 60 | match | 60 | W162 new check, re-run on v4 (tokens remapped) |
| N16 | 34 | 30 | match | 30 | W162 new check, re-run on v4 (tokens remapped) |
| N17 | 38 | 60 | match | 60 | W162 new check, re-run on v4 (tokens remapped) |
| N18 | 40 | 4 | match | 4 | W162 new check, re-run on v4 (tokens remapped) |
| N19 | 40 | 0.07 | match | 0.07 | W162 new check, re-run on v4 (tokens remapped) |
| N20 | 40 | 259,375.33 | match | 259375.33 / 259375.33 | W162 new check, re-run on v4 (tokens remapped) |
| N21 | 41 | Four | match | 4 | W162 new check, re-run on v4 (tokens remapped) |
| N22 | 41 | two | match | 2 | W162 new check, re-run on v4 (tokens remapped) |
| N23 | 44 | 46 | match | 46 | W162 new check, re-run on v4 (tokens remapped) |
| N24 | 44 | 5 | match | 5 | W162 new check, re-run on v4 (tokens remapped) |
| N25 | 46 | 0.9 | match | max 0.882 τ | rebuilt by W163 on the v4 wording |
| N26 | 51 | three | match | 3 × max(1000, 2000) = 6000 | W162 new check, re-run on v4 (tokens remapped) |
| N27 | 51 | two | match | 2 | W162 new check, re-run on v4 (tokens remapped) |
| N28 | 60 | 3 | match | 3 | W162 new check, re-run on v4 (tokens remapped) |
| N29 | 60 | 2 | match | 2 | W162 new check, re-run on v4 (tokens remapped) |
| N73 | 59 | two | match | 2 bars | W162 new check, re-run on v4 (tokens remapped) |
| N30 | 62 | 10 | match | 10 | W162 new check, re-run on v4 (tokens remapped) |
| N31 | 81 | 5 | match | 5 | W162 new check, re-run on v4 (tokens remapped) |
| N32 | 86 | 2035 Spring | match | ['TSO/2035/Spring'] (6 of 21) | W162 new check, re-run on v4 (tokens remapped) |
| N33 | 88 | 10⁻⁵ | match | 1e-05 | W162 new check, re-run on v4 (tokens remapped) |
| N34 | 126 | 10⁻⁵ | match | 1e-05 | W162 new check, re-run on v4 (tokens remapped) |
| N35 | 89 | 62 | match | 62 | W162 new check, re-run on v4 (tokens remapped) |
| N36 | 92 | 3 | match | 3 (3.36) | rebuilt by W163 on the v4 wording |
| N37 | 93 | 44 | match | 44 | W162 new check, re-run on v4 (tokens remapped) |
| N38 | 95 | 0.70 | match | 0.70 | W162 new check, re-run on v4 (tokens remapped) |
| N40 | 136 | 0.70 | match | 0.70 | W162 new check, re-run on v4 (tokens remapped) |
| N42 | 140 | 0.70 | match | 0.70 | W162 new check, re-run on v4 (tokens remapped) |
| N39 | 96 | 0.50 | match | 0.50 | W162 new check, re-run on v4 (tokens remapped) |
| N41 | 136 | 0.50 | match | 0.50/0.50 | W162 new check, re-run on v4 (tokens remapped) |
| N43 | 105 | one (solver build) | match | pinned binary, unchanged | W162 new check, re-run on v4 (tokens remapped) |
| N44 | 105 | one (evaluation at a time) | match | 1 | W162 new check, re-run on v4 (tokens remapped) |
| N45 | 106 | single-threaded; one thread | match | thread caps 1 on every table evaluation | W162 new check, re-run on v4 (tokens remapped) |
| N46 | 107 | MA97 ran serially | match | no OpenMP link or symbol | W162 new check, re-run on v4 (tokens remapped) |
| N47 | 109 | 72; 72/72 | match | 72/72 | W162 new check, re-run on v4 (tokens remapped) |
| N48 | 109 | 3 × 3; x = 0 | match | 3x3 / x0 | W162 new check, re-run on v4 (tokens remapped) |
| N49 | 110 | three; 3/3 | match | 3/3 | W162 new check, re-run on v4 (tokens remapped) |
| N50 | 114 | 3.14.18 | match | 3.14.18 | W162 new check, re-run on v4 (tokens remapped) |
| N51 | 114 | MA97 (network subproblems) | match | {"ma97": 6887} | W162 new check, re-run on v4 (tokens remapped) |
| N52 | 115 | MA57 (storage-operator subproblem) | match | {"ma57": 429} | W162 new check, re-run on v4 (tokens remapped) |
| N53 | 115 | 3.11.11 | match | 3.11.11 | W162 new check, re-run on v4 (tokens remapped) |
| N54 | 115 | 6.9.5 | match | 6.9.5 | W162 new check, re-run on v4 (tokens remapped) |
| N55 | 115 | 27.0 | match | 27.0 | W162 new check, re-run on v4 (tokens remapped) |
| N56 | 115 | Apple M2 Max | match | Apple M2 Max | W162 new check, re-run on v4 (tokens remapped) |
| N57 | 116 | 32 GiB | match | 32 GiB | rebuilt by W163 on the v4 wording |
| N58 | 118 | Two distribution blocks (node 7 in 2025 Winter, node 5 in 2035 Winter) | match | DSO/5/2035/Winter 109, DSO/7/2025/Winter 10 | W162 new check, re-run on v4 (tokens remapped) |
| N59 | 119 | 10 | match | 10 | W162 new check, re-run on v4 (tokens remapped) |
| N60 | 125 | 10⁻⁴ (distribution subproblems) | match | 0.0001 | rebuilt by W163 on the v4 wording |
| N61 | 125 | 10⁻⁶ (both) | match | 1e-06 | rebuilt by W163 on the v4 wording |
| N62 | 124 | three | match | 3 | W162 new check, re-run on v4 (tokens remapped) |
| N63 | 124 | ≈ 1.1 × 10⁻⁶ relative, with the same sign | approximate | c_star -1.200e-06, n7_4h_e1 -1.036e-06, x0 -1.116e-06; mean -1.117e-06 | W162 new check, re-run on v4 (tokens remapped) |
| N64 | 126 | ≈ 7.8 | match | 7.8 | rebuilt by W163 on the v4 wording |
| N65 | 126 | ≈ 1.2 × 10⁻⁵ of it | match | 1.2e-5 | W162 new check, re-run on v4 (tokens remapped) |
| N66 | 129 | 2026-10-06 | match | 2026-10-06 | W162 new check, re-run on v4 (tokens remapped) |
| N67 | 134,135 | 1.75 | match | 1.75 | W162 new check, re-run on v4 (tokens remapped) |
| N68 | 134 | 2 | match | 2 | W162 new check, re-run on v4 (tokens remapped) |
| N69 | 135 | 1.5 | match | 1.5 | W162 new check, re-run on v4 (tokens remapped) |
| N70 | 136 | +0.18 | match | 0.18 | W162 new check, re-run on v4 (tokens remapped) |
| N71 | 137 | +0.20 | match | 0.20 | W162 new check, re-run on v4 (tokens remapped) |
| N72 | 137 | +0.27 | match | 0.27 | W162 new check, re-run on v4 (tokens remapped) |
| N74 | 125 | 5 × 10⁻⁴ (transmission subproblem) | match | parent_of_tail_commit 0.0005, w86_campaign_git_head 0.0005, HEAD 0.0005 | new in W163 |
| N75 | - | solve successfully (optimal or locally optimal) | match | status ok and optimal/locallyOptimal/globallyOptimal | new in W163 |

## W161 checks not in the v4 prose

V1e, V1f, V1g, V1h, C18, C19, C20, C21, C22, C23, C24, C25, C26, C27, C28, C29, C29b, C29c, C29d, C30, C31, C32, C35, C36, C37, C38, C39, C40, C41, C41b, C43, C43b, L5b, L5c, L8, L12.0, L12.1, L12.2, L12.3, L12.4, L12.5, L12.6, L13, L14, L15, L16, L17, L18, L19, L20, L21, L22, L23, L24, L25, L26, P14, P14b, P20, P22, P24, P32a, P32b, P32c, P32d, P32e, P33a, P33b, P33c, P35, P36a, P36b, P37, P38, P47a, P47b, P47c, P47d, P47e, P47f, P48a, P48b, P48c, P48d, P48e, P48f, P48g, P48h, M1, M2, M3, M4, M5.0a, M5.0b, M5.0c, M5.1a, M5.1b, M5.1c, M5.2a, M5.2b, M5.2c, M5.3a, M5.3b, M5.3c, M5.4a, M5.4b, M5.4c, M5.5a, M5.5b, M5.5c, M6, M6b, M7, M8, M9, M10, M11, M12, M13, M14, M15
